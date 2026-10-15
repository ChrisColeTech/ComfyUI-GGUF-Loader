"""Pure-PyTorch TripoSG DiT transformer (1.5B rectified-flow backbone).

Reproduces the inference behavior of
``triposg.models.transformers.triposg_transformer.TripoSGDiTModel`` without
importing diffusers. State-dict keys are identical to the upstream checkpoint
so a strict load of ``diffusion_pytorch_model.safetensors`` succeeds.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from .blocks import DiTBlock
from .config import TripoSGDiTConfig
from .embeddings import TimestepEmbedding, Timesteps
from .normalization import LayerNorm
from .output import Transformer1DModelOutput


class TripoSGDiTModel(nn.Module):
    """TripoSG diffusion transformer backbone (plain ``nn.Module``)."""

    _supports_gradient_checkpointing = True

    def __init__(self, config: TripoSGDiTConfig) -> None:
        super().__init__()
        self.config = config

        num_attention_heads = config.num_attention_heads
        width = config.width
        in_channels = config.in_channels
        num_layers = config.num_layers
        cross_attention_dim = config.cross_attention_dim

        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )
        self.proj_in = nn.Linear(in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    activation_fn="gelu",
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,
                    qk_norm=True,
                    qkv_bias=False,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: Optional[int],
    ) -> Tuple[int, int]:
        if time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4
            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
            return time_embed_dim, timestep_input_dim
        raise ValueError(
            f"{time_embedding_type} does not exist. Only `positional` is supported."
        )

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer1DModelOutput]:
        _, N, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)

        hidden_states = self.proj_in(hidden_states)

        # N + 1 tokens (time token prepended).
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        num_layers = self.config.num_layers
        skips: list[torch.Tensor] = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= num_layers // 2 else skips.pop()

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                )

            if layer < num_layers // 2:
                skips.append(hidden_states)

        # Final layer: norm, drop the time token, project out.
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -N:]
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)
