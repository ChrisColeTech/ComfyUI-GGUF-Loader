"""Compare SenseNova-turbo GGUF vs turbo safetensors; probe a loaded forward."""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

GGUF = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\checkpoints\SenseNova-turbo-Q4_K_M.gguf")
SRC = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\checkpoints\SenseNova-turbo.safetensors")
FP8 = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\checkpoints\SenseNova-turbo-fp8_scaled.safetensors")

MUST = [
    "fm_modules.fm_head.conv1.weight",
    "fm_modules.fm_head.conv2.weight",
    "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
    "language_model.model.layers.0.self_attn.o_proj_mot_gen.weight",
    "language_model.model.layers.0.mlp_mot_gen.down_proj.weight",
    "language_model.model.layers.0.input_layernorm_mot_gen.weight",
    "language_model.model.layers.0.self_attn.q_norm_mot_gen.weight",
    "language_model.model.embed_tokens.weight",
]


def corr(a, b) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        return float("nan")
    x, y = x[: min(len(x), 2_000_000)], y[: min(len(y), 2_000_000)]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return 1.0 if np.allclose(x, y, atol=1e-5) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def dequant(t):
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
                         gguf.GGMLQuantizationType.BF16):
        return t.data.astype(np.float32)
    return gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)


def main():
    r = gguf.GGUFReader(str(GGUF))
    by = {t.name: t for t in r.tensors}
    print("gguf", GGUF.name, "n", len(by), "arch",
          bytes(r.fields["general.architecture"].parts[-1]).decode("utf-8", "ignore").rstrip("\x00"))
    hashed = [n for n in by if n.startswith("t.")]
    print("leftover t.hash", len(hashed), "names>=64", sum(1 for n in by if len(n) >= 64))

    src_path = SRC if SRC.is_file() else FP8
    print("source", src_path.name)
    with safe_open(str(src_path), framework="pt") as f:
        keys = set(f.keys())
        fp8 = src_path == FP8
        for k in MUST:
            t = by.get(k)
            if t is None:
                print(f"  MISSING {k}")
                continue
            logical = tuple(int(x) for x in reversed(t.shape))
            dq = dequant(t).reshape(-1)[: int(np.prod(logical))].reshape(logical)
            if k not in keys:
                print(f"  {k}: not in source")
                continue
            ref = f.get_tensor(k)
            if fp8 and str(ref.dtype).startswith("torch.float8"):
                ref = ref.float()
                sk = k + "_scale"
                if sk in keys:
                    ref = ref * f.get_tensor(sk).float()
            ref = ref.float().numpy()
            print(f"  {t.tensor_type.name:6} {k}: gguf={logical} src={tuple(ref.shape)} "
                  f"corr={corr(dq, ref):.6f}")

    print("types", collections.Counter(t.tensor_type.name for t in r.tensors))


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
