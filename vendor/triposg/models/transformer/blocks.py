"""Pure-PyTorch DiT transformer block for TripoSG.

Reproduces the inference behavior of
``triposg.models.transformers.triposg_transformer.DiTBlock``: long skip
connection -> self-attention -> cross-attention -> feed-forward, each with its
own ``FP32LayerNorm``. State-dict keys (``norm1/norm2/norm3``, ``attn1/attn2``,
``ff``, ``skip_linear/skip_norm``) are identical to the upstream checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from .attention import Attention, TripoSGAttnProcessor2_0
from .feedforward import FeedForward
from .normalization import FP32LayerNorm


class DiTBlock(nn.Module):
    """Transformer block with optional U-Net-style long skip connection."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        self_attention_norm_type: Optional[str] = None,
        use_cross_attention: bool = True,
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = None,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = True,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,
        skip_norm_last: bool = False,
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last

        dim_head = dim // num_attention_heads

        # 1. Self-Attention
        if use_self_attention:
            if self_attention_norm_type not in (None, "fp32_layer_norm"):
                raise NotImplementedError(f"Unsupported self_attention_norm_type: {self_attention_norm_type}")
            self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim_head,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2. Cross-Attention
        if use_cross_attention:
            assert cross_attention_dim is not None
            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim_head,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Long Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        cross_attention_scale = attention_kwargs.pop("cross_attention_scale", 1.0)

        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat(
                ([skip, hidden_states] if self.skip_concat_front else [hidden_states, skip]),
                dim=-1,
            )
            if self.skip_norm_last:
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if self.use_cross_attention:
            hidden_states = hidden_states + self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            ) * cross_attention_scale

        # 3. Feed-forward
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states
