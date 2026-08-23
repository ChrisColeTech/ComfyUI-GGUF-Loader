#!/usr/bin/env python3
"""Step 1: FP8 safetensors checkpoint -> F16/BF16 GGUF (arch-tagged for this repo's loader).

Streams the source with safe_open (mmap), so the FP8 file is never fully
resident; only the GGUF output buffer lives in RAM.

Handles both FP8 layouts seen in ComfyUI checkpoints:

  A) unscaled  (Z-Image turbo uncensored)
     `<key>` is float8_e4m3fn, no companion scale.  Dequant = cast.

  B) per-tensor scaled (Krea-2 turbo uncensored, ComfyUI "comfy_quant")
     `<key>`              float8_e4m3fn
     `<key>_scale`        f32 scalar
     `<layer>.comfy_quant`  uint8 JSON blob describing the format
     Dequant = fp8.to(f16) * scale.  The `_scale` / `comfy_quant` helper
     tensors are dropped from the GGUF.

Mixed checkpoints (DiT + VAE in one file) are narrowed to the diffusion model
by tools/convert.py's strip_prefix(): keys outside `model.diffusion_model.`
are dropped.

Non-finite values are sanitised to 0 with a loud warning: llama-quantize
rejects the whole file with "tensor '<name>' has invalid data" if a single NaN
survives, and NaNs in a checkpoint are always merge damage, never signal.

Usage:
    python 01_fp8_to_f16_gguf.py --src model_fp8.safetensors --dst out-F16.gguf
    python 01_fp8_to_f16_gguf.py --src model_fp8.safetensors --dst out.gguf --arch zimage
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf  # noqa: E402
import convert as c96  # tools/convert.py  # noqa: E402

FP8_DTYPES = tuple(
    d for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    ) if d is not None
)


F16_MAX = 65504.0


def load_and_dequant(src: Path, cast_f16: bool = False) -> dict[str, torch.Tensor]:
    """Read the checkpoint, returning a dequantised state dict.

    FP8 weights come back as float16 (their dynamic range is far inside fp16;
    the largest value either model reaches is ~6). Everything else keeps its
    source dtype so convert.py's F32-for-1D rule still fires.

    `cast_f16` additionally narrows BF16 tensors of rank >= 2 to F16. 1-D
    tensors are left alone precisely so that rule still applies to them.
    """
    log = logging.getLogger("dequant")
    sd: dict[str, torch.Tensor] = {}

    with safe_open(str(src), framework="pt") as f:
        keys = list(f.keys())
        keyset = set(keys)

        # Helper tensors that describe the quantisation rather than the model.
        helper = set()
        for k in keys:
            if k.endswith(".comfy_quant"):
                helper.add(k)
            elif k.endswith("_scale") and k[: -len("_scale")] in keyset:
                # `<w>.weight_scale` next to `<w>.weight`. Note this must NOT
                # catch real weights like `attn.qknorm.qnorm.scale`, which have
                # no matching base key.
                helper.add(k)
        log.info(f"  {len(keys)} tensors, {len(helper)} quantisation helpers dropped")

        n_fp8 = 0
        n_nan = 0
        n_cast = 0
        for k in keys:
            if k in helper:
                continue
            t = f.get_tensor(k)

            if t.dtype in FP8_DTYPES:
                scale_key = f"{k}_scale"
                deq = t.to(torch.float16)
                if scale_key in keyset:
                    deq = deq * f.get_tensor(scale_key).to(torch.float16)
                t = deq
                n_fp8 += 1

            if t.is_floating_point():
                bad = ~torch.isfinite(t.float())
                n_bad = int(bad.sum())
                if n_bad:
                    log.warning(
                        f"  !! {k}: {n_bad}/{t.numel()} non-finite values in the "
                        f"SOURCE checkpoint -> zeroed"
                    )
                    t = torch.where(bad, torch.zeros_like(t), t)
                    n_nan += 1

            if cast_f16 and t.dtype == torch.bfloat16 and t.dim() >= 2:
                peak = float(t.abs().max())
                if peak <= F16_MAX:
                    t = t.to(torch.float16)
                    n_cast += 1
                else:
                    # BF16 carries f32's exponent range; narrowing would clip.
                    log.warning(f"  !! {k}: peak {peak:.3g} exceeds f16 range, kept BF16")

            sd[k] = t

    log.info(f"  dequantised {n_fp8} FP8 tensors; sanitised {n_nan} tensors; "
             f"cast {n_cast} BF16->F16")
    return sd


def bake_lora(sd, lora_path: Path, strength: float, log) -> dict:
    """Merge a LoRA into the weights: W += strength * (alpha/rank) * (B @ A).

    Keys are `<prefix>.<module>.lora_A/lora_B.weight`; the module maps onto
    `<module>.weight` in the (prefix-stripped) state dict. `.alpha` is optional
    and defaults to the rank, which makes the scale 1.0 — that is the case for
    krea2_identity_edit_v1_2_r128, which carries no alpha tensors.

    The merge is done in float32 and cast back, so a long accumulation of rank-
    128 outer products does not lose precision to the f16 destination.
    """
    import re
    from safetensors.torch import load_file

    raw = load_file(str(lora_path))
    pairs, alphas = {}, {}
    for key, tensor in raw.items():
        m = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
        if m:
            pairs.setdefault(m.group(1), {})[m.group(2)] = tensor
        elif key.endswith(".alpha"):
            alphas[key[: -len(".alpha")]] = float(tensor.item())

    log(f"  LoRA: {len(pairs)} modules from {lora_path.name}")
    applied, missing, peak_ratio = 0, [], 0.0
    for module, pair in sorted(pairs.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete LoRA pair for {module!r}")
        name = module.split("diffusion_model.", 1)[-1]
        target = f"{name}.weight"
        if target not in sd:
            missing.append(target)
            continue

        A, B = pair["A"].float(), pair["B"].float()
        rank = A.shape[0]
        scale = strength * (alphas.get(module, float(rank)) / rank)

        W = sd[target]
        orig_dtype = W.dtype
        if tuple(B.shape[0:1] + A.shape[1:2]) != tuple(W.shape):
            raise ValueError(
                f"{target}: LoRA delta {(B.shape[0], A.shape[1])} does not match "
                f"weight {tuple(W.shape)}"
            )
        merged = W.float() + scale * (B @ A)

        peak = float(merged.abs().max())
        if orig_dtype == torch.float16 and peak > F16_MAX:
            raise ValueError(f"{target}: merged peak {peak:.3g} overflows f16")
        peak_ratio = max(peak_ratio, peak / max(float(W.float().abs().max()), 1e-9))
        sd[target] = merged.to(orig_dtype)
        applied += 1

    if missing:
        raise ValueError(
            f"{len(missing)} LoRA targets absent from the checkpoint, refusing a "
            f"partial merge: {missing[:5]}"
        )
    log(f"  LoRA: merged {applied} modules (scale {strength:g}); "
        f"largest peak growth {peak_ratio:.2f}x")
    return sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="FP8 safetensors checkpoint")
    ap.add_argument("--dst", required=True, type=Path, help="output F16 GGUF")
    ap.add_argument("--arch", default=None, help="override the detected general.architecture")
    ap.add_argument("--name", default=None, help="general.name (default: dst stem)")
    ap.add_argument(
        "--f16", action="store_true",
        help="narrow rank>=2 BF16 tensors to F16. Required for the giga-images "
             "z_image_v4 loader: gguf-py hands BF16 back as uint8 with a doubled "
             "last dimension, and its reinterpret path does not cover bare "
             "nn.Parameters, so pad tokens load at twice their width.",
    )
    ap.add_argument("--lora", type=Path, default=None,
                    help="merge this LoRA into the weights before quantizing")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument(
        "--remap", choices=["zimage-to-diffusers"], default=None,
        help="rewrite keys before conversion. zimage-to-diffusers targets the "
             "giga-images z_image_v4 pipeline, which wants split to_q/to_k/to_v "
             "rather than the checkpoint's fused qkv.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    log = logging.getLogger("fp8->gguf")

    if not args.src.is_file():
        log.error(f"missing source: {args.src}")
        return 1

    log.info("=" * 70)
    log.info("FP8 SAFETENSORS -> F16 GGUF")
    log.info("=" * 70)
    log.info(f"Input:  {args.src} ({args.src.stat().st_size / 1024**3:.2f} GB)")
    log.info(f"Output: {args.dst}")

    sd = load_and_dequant(args.src, cast_f16=args.f16)
    sd = c96.strip_prefix(sd)
    log.info(f"  {len(sd)} diffusion-model tensors after prefix strip")

    if args.lora:
        sd = bake_lora(sd, args.lora, args.lora_strength, log.info)

    if args.remap == "zimage-to-diffusers":
        from remap_zimage import reference_to_diffusers
        # Must run on dense weights: a K-quant's 256-element superblocks do not
        # align to the QKV boundaries, so the split is impossible after
        # quantization.
        sd = reference_to_diffusers(sd, log=log.info)

    model_arch = c96.detect_arch(sd)
    if args.arch and args.arch != model_arch.arch:
        log.warning(f"  arch override: detected {model_arch.arch!r} -> writing {args.arch!r}")
        model_arch.arch = args.arch
    log.info(f"  architecture: {model_arch.arch}")

    dtypes = [t.dtype for t in sd.values()]
    main_dtype = max(set(dtypes), key=dtypes.count)
    ftype_gguf = (
        gguf.LlamaFileType.MOSTLY_BF16 if main_dtype == torch.bfloat16
        else gguf.LlamaFileType.MOSTLY_F16
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_name(args.name or args.dst.stem)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype_gguf)

    # GGUF stores ne[] as reversed(torch shape) and drops trailing 1s, so a
    # torch shape with a LEADING 1 comes back a rank short. llama-quantize
    # applies that truncation even when the F16 input recorded both dims:
    # Krea-2's txtfusion.projector.weight goes (1,12) -> ne (12,1) -> ne (12,),
    # and ComfyUI's model_detection then does .shape[1] on a 1-D tensor and
    # raises IndexError. loader.py special-cases the Z-Image pad tokens for the
    # same reason; recording orig_shape fixes the whole class generically,
    # since get_orig_shape is consulted for every tensor.
    for key, tensor in sd.items():
        shape = tuple(tensor.shape)
        if len(shape) > 1 and shape[0] == 1:
            writer.add_array(f"comfy.gguf.orig_shape.{key}", [int(d) for d in shape])
            log.info(f"  recording orig_shape for {key}: {shape}")

    c96.handle_tensors(writer, sd, model_arch)
    del sd

    writer.write_header_to_file(path=str(args.dst))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    log.info(f"\nDone: {args.dst} ({args.dst.stat().st_size / 1024**3:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
