"""Pure-PyTorch attention for the TripoSG DiT.

Reproduces the inference behavior of ``diffusers.models.attention_processor.Attention``
together with ``triposg.models.attention_processor.TripoSGAttnProcessor2_0``.

CRITICAL: the upstream checkpoint splits heads FIRST and then splits qkv (or
kv), i.e. ``view(..., heads, head_dim*{3|2})`` rather than the standard
diffusers ``view(..., 3, heads, head_dim)``. This module replicates that
convention exactly so a strict checkpoint load produces identical math.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .normalization import RMSNorm


class TripoSGAttnProcessor2_0:
    """Scaled-dot-product attention with heads-first qkv splitting."""

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "TripoSGAttnProcessor2_0 requires PyTorch 2.0+ "
                "(torch.nn.functional.scaled_dot_product_attention)."
            )

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # NOTE that pre-trained models split heads first then split qkv or kv,
        # like .view(..., attn.heads, 3, dim) instead of .view(..., 3, attn.heads, dim).
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)

        head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed (not used by TripoSG image-to-3D, but kept for parity).
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = _apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Rotary embedding application (pure-PyTorch ``apply_rotary_emb`` parity).

    Not exercised by the TripoSG image-to-3D path (``image_rotary_emb`` is
    always ``None``) but kept for behavioral parity with diffusers.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = hidden_states.reshape(*hidden_states.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = hidden_states.reshape(*hidden_states.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (hidden_states.float() * cos + x_rotated.float() * sin).to(hidden_states.dtype)
        return out

    x_rotated = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(2)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(hidden_states)


class Attention(nn.Module):
    """Projection layers + qk-norm + processor for TripoSG attention."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_norm: Optional[str] = None,
        qk_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        processor: Optional[TripoSGAttnProcessor2_0] = None,
    ) -> None:
        super().__init__()

        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.out_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.group_norm = None
        self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"Unsupported qk_norm: {qk_norm}")

        # NOTE: ``cross_attention_norm="layer_norm"`` is exercised only by the
        # TripoSG VAE cross-attention block (it owns ``attn2.norm_cross.*``
        # checkpoint keys). The TripoSG DiT transformer always passes
        # ``cross_attention_norm=None`` so its inference path is unchanged.
        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        else:
            raise ValueError(f"Unsupported cross_attention_norm: {cross_attention_norm}")

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if processor is None:
            processor = TripoSGAttnProcessor2_0()
        self.set_processor(processor)

    def set_processor(self, processor: TripoSGAttnProcessor2_0) -> None:
        self.processor = processor

    def get_processor(self) -> TripoSGAttnProcessor2_0:
        return self.processor

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply ``norm_cross`` to encoder hidden states (LayerNorm path).

        Matches ``diffusers.models.attention_processor.Attention.norm_encoder_hidden_states``
        for the ``cross_attention_norm="layer_norm"`` case used by the VAE.
        """
        assert self.norm_cross is not None, (
            "self.norm_cross must be defined to call self.norm_encoder_hidden_states"
        )
        return self.norm_cross(encoder_hidden_states)

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        head_size = self.heads
        if attention_mask.dim() == 2:
            current_batch_size, source_length = attention_mask.shape
            padding_shape = (0, target_length - source_length)
            mask = F.pad(attention_mask, padding_shape, value=0)
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, None, :]
        elif attention_mask.dim() == 4:
            mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Unsupported attention_mask rank: {attention_mask.dim()}")

        if out_dim == 3:
            expanded_mask = mask[:, None, :, :].expand(current_batch_size, head_size, -1, -1)
        elif out_dim == 4:
            expanded_mask = mask[:, :, None, :].expand(-1, -1, target_length, -1)
        else:
            raise ValueError(f"Unsupported out_dim: {out_dim}")
        return expanded_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
