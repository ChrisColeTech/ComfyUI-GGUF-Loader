#!/usr/bin/env python3
"""Step 3: verify a quantized GGUF against the FP8 checkpoint it came from.

Reads the GGUF the way `loader.gguf_sd_loader` does — logical shape is
`reversed(gguf ne[])` — dequantizes every tensor, and checks each one against
the dequantized source weight:

  * the arch tag is in loader.py's IMG_ARCH_LIST
  * no NaN/Inf anywhere
  * shapes match the source
  * per-tensor correlation with the source (a transposed or mis-shaped write
    lands near 0 here, which is the failure this catches)

Dequantization goes through `gguf.quants`, not this repo's `dequant.py`, so a
bug shared between the writer and the loader can't hide the problem. It also
keeps the check runnable outside a ComfyUI install.

Usage:
    python 03_verify.py quantized.gguf --src original_fp8.safetensors
    python 03_verify.py quantized.gguf --src original_fp8.safetensors --sample 40
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]

FP8_DTYPES = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    ) if d is not None
)


def img_arch_list() -> set[str]:
    """loader.py's IMG_ARCH_LIST, read without importing it (it needs comfy)."""
    src = (REPO_ROOT / "loader.py").read_text(encoding="utf-8")
    m = re.search(r"IMG_ARCH_LIST\s*=\s*(\{.*?\n\})", src, re.S)
    return set(ast.literal_eval(m.group(1)))


def read_gguf(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    """Tensors as float32, shaped the way loader.gguf_sd_loader shapes them."""
    reader = gguf.GGUFReader(str(path))
    arch_field = reader.get_field("general.architecture")
    arch = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")

    out = {}
    for t in reader.tensors:
        shape = tuple(int(x) for x in reversed(t.shape))
        qt = t.tensor_type
        if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                  gguf.GGMLQuantizationType.BF16):
            data = torch.from_numpy(t.data.astype(np.float32))
        else:
            data = torch.from_numpy(gguf.quants.dequantize(t.data, qt).astype(np.float32))
        out[t.name] = data.reshape(-1)[: int(np.prod(shape))].reshape(shape)
    return arch, out


def source_tensors(src: Path) -> dict[str, torch.Tensor]:
    """Dequantized source weights keyed the way the GGUF names them."""
    out = {}
    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
        keyset = set(keys)
        prefix = "model.diffusion_model." if any(
            k.startswith("model.diffusion_model.") for k in keys
        ) else ""
        for k in keys:
            if not k.startswith(prefix):
                continue
            if k.endswith(".comfy_quant"):
                continue
            if k.endswith("_scale") and k[: -len("_scale")] in keyset:
                continue
            t = f.get_tensor(k)
            if t.dtype in FP8_DTYPES:
                t = t.to(torch.float32)
                sk = f"{k}_scale"
                if sk in keyset:
                    t = t * f.get_tensor(sk).float()
            out[k[len(prefix):]] = t.float()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--src", required=True, type=Path, help="original FP8 safetensors")
    ap.add_argument("--sample", type=int, default=0, help="check only N largest tensors (0 = all)")
    ap.add_argument("--min-corr", type=float, default=0.99)
    args = ap.parse_args()

    print(f"GGUF:   {args.gguf} ({args.gguf.stat().st_size / 1024**3:.2f} GB)")
    arch, sd = read_gguf(args.gguf)
    allowed = img_arch_list()
    print(f"        arch={arch!r} {'(in IMG_ARCH_LIST)' if arch in allowed else '!! NOT IN IMG_ARCH_LIST'}")
    print(f"        {len(sd)} tensors loaded")

    print(f"Source: {args.src}")
    src = source_tensors(args.src)
    print(f"        {len(src)} tensors")

    missing = sorted(set(src) - set(sd))
    extra = sorted(set(sd) - set(src))
    if missing:
        print(f"  MISSING from gguf ({len(missing)}): {missing[:5]}")
    if extra:
        print(f"  EXTRA in gguf ({len(extra)}): {extra[:5]}")

    keys = sorted(set(sd) & set(src))
    if args.sample:
        keys = sorted(keys, key=lambda k: src[k].numel(), reverse=True)[: args.sample]

    bad_shape, bad_finite, low_corr, sanitised = [], [], [], []
    worst = (2.0, None)
    for k in keys:
        ref = src[k]
        got = sd[k].reshape(-1)
        if got.numel() != ref.numel():
            bad_shape.append((k, tuple(ref.shape), tuple(sd[k].shape)))
            continue
        if not torch.isfinite(got).all():
            bad_finite.append(k)
            continue
        a, b = got, ref.reshape(-1)
        # Non-finite values in the SOURCE were zeroed on the way in (see
        # 01_fp8_to_f16_gguf.py); compare only where the source is meaningful.
        keep = torch.isfinite(b)
        if not keep.all():
            n_bad_src = int((~keep).sum())
            sanitised.append((k, n_bad_src))
            if not keep.any():
                continue
            a, b = a[keep], b[keep]
        # Population std throughout — a Bessel-corrected std against an
        # n-normalised covariance reads as (n-1)/n on small tensors.
        sa, sb = a.std(unbiased=False), b.std(unbiased=False)
        if sa == 0 or sb == 0:
            corr = 1.0 if torch.allclose(a, b, atol=1e-6) else 0.0
        else:
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        if corr < worst[0]:
            worst = (corr, k)
        if corr < args.min_corr:
            low_corr.append((k, round(corr, 4)))

    print(f"\nchecked {len(keys)} tensors")
    print(f"  shape mismatches : {len(bad_shape)}")
    print(f"  non-finite       : {len(bad_finite)}")
    print(f"  corr < {args.min_corr}      : {len(low_corr)}")
    if sanitised:
        print(f"  source had non-finite values, compared on the finite part only:")
        for k, n in sanitised:
            print(f"    {k}: {n} value(s) zeroed")
    print(f"  worst corr       : {worst[0]:.5f}  ({worst[1]})")
    for row in (bad_shape[:5] + bad_finite[:5] + low_corr[:10]):
        print(f"    {row}")

    ok = not bad_shape and not bad_finite and not low_corr and arch in allowed and not missing
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
