"""Pure-PyTorch normalization layers for the TripoSG DiT.

Reproduces the inference behavior of:
- ``diffusers.models.normalization.FP32LayerNorm`` (compute in float32, cast back)
- ``diffusers.models.normalization.LayerNorm`` (an alias of ``torch.nn.LayerNorm``
  on torch>=2.1, which is guaranteed by the project's ``torch>=2.2`` pin)
- ``diffusers.models.normalization.RMSNorm`` (used for ``qk_norm="rms_norm"``)
"""

from __future__ import annotations

import numbers
from typing import Sized

import torch
import torch.nn.functional as F
from torch import nn


class FP32LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` that always computes in float32 and casts back."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


# On torch>=2.1 (project requires torch>=2.2) diffusers aliases LayerNorm to
# nn.LayerNorm, so the default elementwise_affine=True and eps=1e-5 semantics
# match exactly.
LayerNorm = nn.LayerNorm


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization used for ``qk_norm``."""

    def __init__(self, dim, eps: float, elementwise_affine: bool = True) -> None:
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states
