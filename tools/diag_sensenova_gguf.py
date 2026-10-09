"""Load SenseNova GGUF the way CheckpointLoaderGGUF does; probe conv/linear."""
from __future__ import annotations

import os
import sys

COMFY = r"N:\ComfyUI_windows_portable_nvidia\ComfyUI"
sys.path.insert(0, COMFY)
os.chdir(COMFY)
sys.argv = ["diag_sensenova_gguf.py"]

import importlib
import types
from pathlib import Path

import torch
from safetensors import safe_open

PACK = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\custom_nodes\comfyui-gguf-loader")
pkg = types.ModuleType("comfyui_gguf_loader")
pkg.__path__ = [str(PACK)]
pkg.__package__ = "comfyui_gguf_loader"
sys.modules["comfyui_gguf_loader"] = pkg
gguf_loader = importlib.import_module("comfyui_gguf_loader.loader")
ops_mod = importlib.import_module("comfyui_gguf_loader.ops")
GGMLOps, GGMLTensor = ops_mod.GGMLOps, ops_mod.GGMLTensor
dequantize_tensor = ops_mod.dequant.dequantize_tensor
is_quantized = ops_mod.dequant.is_quantized

import comfy.sd

GGUF = r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\checkpoints\SenseNova-U1.5-8B-MoT-BF16-T8-Q4_K_M.gguf"
BF16 = r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\checkpoints\SenseNova-U1.5-8B-MoT-BF16-T8.safetensors"

KEYS = [
    "fm_modules.fm_head.conv1.weight",
    "fm_modules.fm_head.conv1.bias",
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.input_layernorm.weight",
]


def corr(a, b):
    x, y = a.reshape(-1).float(), b.reshape(-1).float()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).mean() / (x.std() * y.std() + 1e-12))


def main():
    print("cuda", torch.cuda.is_available(), "torch", torch.__version__)
    print("loading gguf via gguf_sd_loader...")
    sd, extra = gguf_loader.gguf_sd_loader(GGUF)
    print("arch", extra.get("arch_str"), "n", len(sd))
    for k in KEYS:
        t = sd[k]
        print(f"  {k}: type={type(t).__name__} dtype={t.dtype} shape={tuple(t.shape)} "
              f"quant={is_quantized(t)} ggml={isinstance(t, GGMLTensor)} "
              f"tt={getattr(t, 'tensor_type', None)}")

    print("load_state_dict_guess_config + GGMLOps...")
    ops = GGMLOps()
    out = comfy.sd.load_state_dict_guess_config(
        sd, output_vae=True, output_clip=True, output_clipvision=False,
        model_options={"custom_operations": ops},
    )
    model = out[0]
    inner = model.model.diffusion_model
    conv = inner.fm_modules["fm_head"].conv1
    lin = inner.language_model.model.layers[0].self_attn.q_proj
    print("conv", type(conv), "w", type(conv.weight), conv.weight.dtype, tuple(conv.weight.shape))
    print("lin ", type(lin), "w", type(lin.weight), getattr(lin.weight, "dtype", None),
          getattr(lin.weight, "tensor_type", None), tuple(lin.weight.shape))

    device = torch.device("cuda")
    x = torch.randn(1, 1024, 8, 8, device=device, dtype=torch.bfloat16)
    conv = conv.to(device)
    with torch.no_grad():
        y = conv(x)
    print("conv out", y.shape, y.dtype, "mean", float(y.float().mean()), "std", float(y.float().std()),
          "finite", bool(torch.isfinite(y).all()))

    w = dequantize_tensor(lin.weight, torch.float32)
    if isinstance(w, GGMLTensor):
        w = w.as_subclass(torch.Tensor)
    print("q_proj dequant", w.dtype, tuple(w.shape), "mean", float(w.mean()), "std", float(w.std()))

    xin = torch.randn(1, 4, 4096, device=device, dtype=torch.bfloat16)
    lin = lin.to(device)
    with torch.no_grad():
        z = lin(xin)
    print("lin out", z.shape, z.dtype, "mean", float(z.float().mean()), "std", float(z.float().std()),
          "finite", bool(torch.isfinite(z).all()))

    emb = inner.language_model.model.embed_tokens
    print("embed", type(emb), type(emb.weight), getattr(emb.weight, "tensor_type", None),
          tuple(emb.weight.shape))
    ids = torch.randint(0, 1000, (1, 8), device=device)
    emb = emb.to(device)
    with torch.no_grad():
        e = emb(ids)
    print("embed out", e.shape, e.dtype, "mean", float(e.float().mean()), "std", float(e.float().std()),
          "finite", bool(torch.isfinite(e).all()))


if __name__ == "__main__":
    main()
