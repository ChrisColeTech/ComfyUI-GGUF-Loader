"""Z-Image reference/ComfyUI key layout -> diffusers key layout.

ComfyUI ships `comfy.utils.z_image_to_diffusers`, but it runs the other way:
it maps a diffusers name to the reference name it loads from, and encodes the
fused-QKV case as `(target, (dim, offset, length))` slice tuples. This is the
inverse — it materialises the split tensors instead of describing where to
read them.

Reference (ComfyUI checkpoint)        diffusers (HF Tongyi-MAI release)
  attention.qkv.weight        ->        attention.to_q/to_k/to_v.weight
  attention.q_norm.weight     ->        attention.norm_q.weight
  attention.k_norm.weight     ->        attention.norm_k.weight
  attention.out.weight        ->        attention.to_out.0.weight
  x_embedder.*                ->        all_x_embedder.2-1.*
  final_layer.*               ->        all_final_layer.2-1.*

The `2-1` suffix is an aspect-ratio bucket name. The diffusers release carries
exactly one bucket, so this is a rename and not a broadcast.

Splitting has to happen while the weights are still dense — a K-quant packs
256-element superblocks that do not align to the QKV boundaries, so a fused
`qkv` cannot be cut apart after quantization.
"""
from __future__ import annotations

import re

import torch

# Blocks that carry an attention module, by key prefix.
_BLOCK_RE = re.compile(r"^(layers|context_refiner|noise_refiner)\.(\d+)\.")

# Within a block, plain renames.
_BLOCK_RENAMES = {
    "attention.q_norm.weight": "attention.norm_q.weight",
    "attention.k_norm.weight": "attention.norm_k.weight",
    "attention.out.weight": "attention.to_out.0.weight",
    "attention.out.bias": "attention.to_out.0.bias",
}

# Top-level renames. Everything not listed passes through unchanged
# (cap_embedder.*, t_embedder.*, x_pad_token, cap_pad_token).
_TOP_RENAMES = {
    "x_embedder.weight": "all_x_embedder.2-1.weight",
    "x_embedder.bias": "all_x_embedder.2-1.bias",
    "final_layer.linear.weight": "all_final_layer.2-1.linear.weight",
    "final_layer.linear.bias": "all_final_layer.2-1.linear.bias",
    "final_layer.adaLN_modulation.1.weight": "all_final_layer.2-1.adaLN_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias": "all_final_layer.2-1.adaLN_modulation.1.bias",
}

# Present in ComfyUI checkpoints, absent from the diffusers model. `norm_final`
# is inert either way — ComfyUI's Z-Image path has `self.norm_final` commented
# out in comfy/ldm/lumina/model.py — and in the uncensored merge it is NaN.
_DROP = {"norm_final.weight", "norm_final.bias"}


def reference_to_diffusers(sd: dict[str, torch.Tensor], log=print) -> dict[str, torch.Tensor]:
    """Return a new state dict in the diffusers layout.

    Raises if a fused `qkv` is not exactly 3x a third of its rows, which is the
    one assumption that would silently corrupt every attention layer if the
    checkpoint ever used grouped-query attention with unequal Q/K/V widths.
    """
    out: dict[str, torch.Tensor] = {}
    n_split = n_renamed = n_dropped = 0

    for key, tensor in sd.items():
        if key in _DROP:
            log(f"  dropping {key} (not present in the diffusers model)")
            n_dropped += 1
            continue

        block = _BLOCK_RE.match(key)
        if block:
            prefix = key[: block.end()]
            rest = key[block.end():]

            if rest in ("attention.qkv.weight", "attention.qkv.bias"):
                suffix = rest.rsplit(".", 1)[1]  # weight | bias
                rows = tensor.shape[0]
                if rows % 3 != 0:
                    raise ValueError(
                        f"{key}: fused qkv has {rows} rows, not divisible by 3 — "
                        f"this checkpoint does not use equal-width Q/K/V and needs "
                        f"per-head sizes to split correctly"
                    )
                width = rows // 3
                for i, name in enumerate(("to_q", "to_k", "to_v")):
                    out[f"{prefix}attention.{name}.{suffix}"] = tensor[
                        i * width : (i + 1) * width
                    ].clone()
                n_split += 1
                continue

            if rest in _BLOCK_RENAMES:
                out[prefix + _BLOCK_RENAMES[rest]] = tensor
                n_renamed += 1
                continue

            out[key] = tensor
            continue

        if key in _TOP_RENAMES:
            out[_TOP_RENAMES[key]] = tensor
            n_renamed += 1
            continue

        out[key] = tensor

    log(f"  remap: split {n_split} fused qkv, renamed {n_renamed}, "
        f"dropped {n_dropped}  ({len(sd)} -> {len(out)} tensors)")
    return out
