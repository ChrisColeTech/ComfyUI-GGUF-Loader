#!/usr/bin/env python3
"""Step 2: quantize an F16 GGUF to one or more K-quants via llama-quantize.

The binary must be a build carrying City96's image-model patch (tools/lcpp.patch);
a stock llama.cpp refuses DiT arch tags with "unknown model architecture".
D:/Projects/llama.cpp/bin_cuda120 is such a build and is preferred here — the
`bin/` tree next to it is stock and will fail.

Usage:
    python 02_quantize.py in-F16.gguf out_dir --name z_image_turbo_uncensored
    python 02_quantize.py in-F16.gguf out_dir --types Q4_K_M Q6_K Q8_0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

QUANTIZE_CANDIDATES = (
    Path("D:/Projects/llama.cpp/bin_cuda120/bin/Release/llama-quantize.exe"),
    Path("D:/Projects/llama.cpp/build/bin/Release/llama-quantize.exe"),
    Path("D:/Projects/llama.cpp/bin/bin/Release/llama-quantize.exe"),
)
DEFAULT_TYPES = ("Q4_K_M", "Q6_K", "Q8_0")


def find_quantizer(override: str | None) -> Path | None:
    if override:
        p = Path(override)
        return p if p.is_file() else None
    for cand in QUANTIZE_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="F16 GGUF")
    ap.add_argument("out_dir", type=Path, help="directory for the quantized files")
    ap.add_argument("--name", default=None, help="output basename (default: input stem minus -F16)")
    ap.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES))
    ap.add_argument("--quantizer", default=None, help="explicit llama-quantize.exe path")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"missing input: {args.input}")
        return 1

    exe = find_quantizer(args.quantizer)
    if exe is None:
        print("no llama-quantize.exe found; tried:")
        for c in QUANTIZE_CANDIDATES:
            print(f"  {c}")
        return 1

    base = args.name or args.input.stem
    for suffix in ("-F16", "-BF16", "-f16"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Quantizer: {exe}")
    print(f"Input:     {args.input} ({args.input.stat().st_size / 1024**3:.2f} GB)")
    print(f"Types:     {', '.join(args.types)}")

    failures = []
    for qtype in args.types:
        out = args.out_dir / f"{base}-{qtype}.gguf"
        log_path = out.with_suffix(".log")
        print(f"\n{'=' * 70}\n{qtype} -> {out}\n{'=' * 70}", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                [str(exe), str(args.input), str(out), qtype],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            logf.write(proc.stdout or "")
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-8:])
            print(f"FAILED rc={proc.returncode}\n{tail}")
            failures.append(qtype)
            continue
        print(f"ok in {time.time() - t0:.0f}s -> {out.stat().st_size / 1024**3:.2f} GB")

    if failures:
        print(f"\nFailed: {', '.join(failures)}")
        return 1
    print("\nAll quantizations succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
