"""Pure-PyTorch timestep embeddings for the TripoSG DiT.

Reproduces the inference behavior of ``diffusers.models.embeddings.Timesteps``
and ``diffusers.models.embeddings.TimestepEmbedding``. State-dict keys are
kept identical to the upstream checkpoint (``linear_1``/``linear_2``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (DDPM formulation)."""

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Sinusoidal positional timestep embedding (no learnable parameters)."""

    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """MLP timestep embedding: ``linear_1`` -> activation -> ``linear_2``."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
        sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        act_fn = act_fn.lower()
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "swish":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            post_act_fn = post_act_fn.lower()
            if post_act_fn == "silu":
                self.post_act = nn.SiLU()
            elif post_act_fn == "gelu":
                self.post_act = nn.GELU()
            else:
                raise ValueError(f"Unsupported post activation function: {post_act_fn}")

    def forward(self, sample: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
