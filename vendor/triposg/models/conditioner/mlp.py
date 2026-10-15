"""DINOv2 feed-forward networks (pure PyTorch).

``DINOv2MLP`` is the standard two-layer MLP used when
``config.use_swiglu_ffn`` is ``False`` (the TripoSG checkpoint case). The
``"gelu"`` activation in Transformers 4.45.2 resolves to
``torch.nn.functional.gelu`` (exact, erf-based), so we keep the same call to
preserve numerical parity.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import DINOv2Config


class DINOv2MLP(nn.Module):
    """Two-layer MLP with the configured activation."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = _resolve_activation(config.hidden_act)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class _GELU(nn.Module):
    """Exact (erf-based) GELU activation."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input)


class _GELUTanh(nn.Module):
    """Tanh approximation of GELU."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(input, approximate="tanh")


def _resolve_activation(name: str) -> nn.Module:
    name_lower = str(name).lower()
    if name_lower == "gelu":
        return _GELU()
    if name_lower in {"gelu_new", "gelu_python", "gelu_pytorch_tanh"}:
        return _GELUTanh()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "silu" or name_lower == "swish":
        return nn.SiLU()
    raise ValueError(f"Unsupported DINOv2 activation: {name!r}")
