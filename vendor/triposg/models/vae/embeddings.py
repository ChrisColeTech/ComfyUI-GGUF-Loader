"""Pure-PyTorch frequency positional embedding for the TripoSG VAE.

Reproduces the inference behavior of
``triposg.models.embeddings.FrequencyPositionalEmbedding`` without importing
diffusers. The buffer ``frequencies`` matches the upstream
``register_buffer(..., persistent=False)`` convention, so it is recomputed
deterministically at construction and is not part of the state dict.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyPositionalEmbedding(nn.Module):
    """Sinusoidal frequency embedding used by the TripoSG VAE.

    Each feature channel of ``x[..., i]`` is lifted to
    ``[x[..., i], sin(f_1 x), ..., sin(f_N x), cos(f_1 x), ..., cos(f_N x)]``
    when ``include_input=True``. With ``logspace=True`` the frequencies are
    ``[2^0, 2^1, ..., 2^(num_freqs-1)]`` and when ``include_pi=True`` they are
    additionally scaled by ``pi``.
    """

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies = frequencies * torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim: int) -> int:
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            return torch.cat((embed.sin(), embed.cos()), dim=-1)
        return x
