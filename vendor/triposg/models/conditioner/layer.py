"""DINOv2 transformer layer (pure PyTorch).

Reproduces ``transformers.models.dinov2.modeling_dinov2.Dinov2Layer`` exactly:
pre-norm attention, LayerScale on the attention and MLP outputs, optional
stochastic depth, and pre-norm MLP. For TripoSG inference
``drop_path_rate = 0.0``, so ``DINOv2DropPath`` collapses to ``nn.Identity``.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from .attention import DINOv2Attention
from .config import DINOv2Config
from .mlp import DINOv2MLP


class DINOv2LayerScale(nn.Module):
    """Per-channel learnable scale applied to attention/MLP outputs."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state * self.lambda1


def _drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()
    return input.div(keep_prob) * random_tensor


class DINOv2DropPath(nn.Module):
    """Stochastic-depth drop path; identity in inference mode."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _drop_path(hidden_states, self.drop_prob or 0.0, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class DINOv2Layer(nn.Module):
    """Pre-norm transformer block with LayerScale and residual connections."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv2Attention(config)
        self.layer_scale1 = DINOv2LayerScale(config)
        self.drop_path = (
            DINOv2DropPath(config.drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv2MLP(config)
        self.layer_scale2 = DINOv2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]

        hidden_states = self.drop_path(attention_output) + hidden_states

        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs
        return outputs
