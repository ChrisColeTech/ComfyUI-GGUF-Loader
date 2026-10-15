"""Pure-PyTorch feed-forward layer for the TripoSG DiT.

Reproduces the inference behavior of ``diffusers.models.attention.FeedForward``
and ``diffusers.models.activations.GELU`` for the TripoSG activation
``activation_fn="gelu"`` (exact GELU, ``approximate="none"``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GELU(nn.Module):
    """Linear projection followed by exact GELU (``approximate="none"``)."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """Two-layer feed-forward network: ``[act, dropout, out]``.

    For the TripoSG checkpoint ``activation_fn="gelu"`` the activation module
    is :class:`GELU` with ``approximate="none"`` (exact GELU), matching
    ``diffusers`` exactly.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            raise NotImplementedError("GEGLU is not used by the TripoSG checkpoint")
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
