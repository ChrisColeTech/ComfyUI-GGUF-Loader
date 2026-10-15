"""
ComfyUI-native attention dispatch for UniRig models.

Replaces flash_attn and manual einsum attention with ComfyUI's
optimized_attention backend, which auto-selects the best available
implementation (flash, xformers, sdpa, etc.).
"""

import torch
from comfy.ldm.modules.attention import optimized_attention_for_device


def comfy_attention(q, k, v, heads):
    """Standard attention using ComfyUI dispatch.

    Args:
        q, k, v: [B, L, H, D] tensors (Michelangelo layout)
        heads: number of attention heads

    Returns:
        [B, L, H, D] output tensor
    """
    # Convert from [B, L, H, D] to ComfyUI's [B, H, L, D]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    attn_fn = optimized_attention_for_device(q.device)
    out = attn_fn(q, k, v, heads=heads, skip_reshape=True, skip_output_reshape=True)

    # Convert back to [B, L, H, D]
    return out.permute(0, 2, 1, 3)


def comfy_attention_flat(q, k, v, heads):
    """Attention for flat [B, L, H*D] tensors.

    Used by the skin model's cross-attention where q/k/v are
    already flattened across heads.

    Args:
        q, k, v: [B, L, H*D] tensors
        heads: number of attention heads

    Returns:
        [B, L, H*D] output tensor
    """
    attn_fn = optimized_attention_for_device(q.device)
    return attn_fn(q, k, v, heads=heads)
