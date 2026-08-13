# ComfyUI-GGUF-Loader

GGUF Quantization support for native ComfyUI models — the CCTech Suite fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is no need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## What this fork adds

- **Open upstream PRs backported**, including newer architectures (LTX-2, Z-Image, Ideogram-4, Qwen3-VL / MiniMax-H3 text encoders). See [PR_BACKPORT.md](PR_BACKPORT.md).
- **MiniMax-H3 fix**: high-precision buffers such as `adaln_t_table` are kept in float32 through both conversion and loading. Stored as F16 they crash the sampler on the first step with `expected dtype struct c10::Half for 'weight' but got dtype float`.
- **Two extra loader nodes**: a dual VAE loader and a text encoder + ClipProj loader (see *Nodes* below).

## Installation

> [!IMPORTANT]
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader ComfyUI/custom_nodes/ComfyUI-GGUF-Loader
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF-Loader\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this upstream issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF UNET loader found under the `🤖 CCTech/GGUF` category. Place the `.gguf` model files in your `ComfyUI/models/unet` (or `diffusion_models`) folder.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models:

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)

See the instructions in the [tools](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader/tree/main/tools) folder for how to create your own quants.

## Nodes

All of them live under `🤖 CCTech/GGUF`.

| Node | Purpose |
|---|---|
| **UNET Loader (GGUF)** | drop-in replacement for `Load Diffusion Model` |
| **UNET Loader (GGUF/Advanced)** | same, with `dequant_dtype` / `patch_dtype` / `patch_on_device` exposed |
| **CLIP Loader (GGUF)** | single text encoder, `.gguf` or `safetensors` |
| **Dual / Triple / Quadruple CLIP Loader (GGUF)** | the same for models taking several encoders |
| **Dual VAE Loader (video + audio)** | two VAEs from one node, with the outputs named for their streams |
| **Text Encoder + ClipProj Loader** | loads a small encoder, GGUF included, and projects it into a large one's space |

### Dual VAE Loader

MiniMax-H3 decodes video and audio through two separate VAEs, so every workflow ends up with a pair of `VAELoader` nodes side by side. This is that pair in one node, with outlets named `video_vae` and `audio_vae`. Loading is delegated to the stock loader, so `taesd` entries stay available and nothing behaves differently.

### Text Encoder + ClipProj Loader

Loads a small text encoder and projects it into a large one's space in a single node — for MiniMax-H3, a Qwen3-VL-4B or -8B standing in for the 32B the model expects, which takes the text encoder from 15.7 GB to about 5.

The projection is **ported into this pack** (`clipproj.py`) from [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) by nicolab28, MIT, with the licence travelling alongside it as [LICENSE-ClipProj](LICENSE-ClipProj). No dependency on that pack: install it or don't, this node works either way. Verified against the original on a real prompt through the learned 8B matrix — identical conditioning, to the bit.

What you do still need is a **matrix**, from [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3), in `ComfyUI/models/clip_projections/`. `mmh3-4b-*` goes with a 4B and `mmh3-8b-*` with an 8B; they are not interchangeable. The `<control:...>` entries are not projections but deliberate baselines — zero ignores your prompt entirely, identity copies the raw dimensions with no learning — and they run on any encoder. Run them first to see that the matrix is doing the work.

Differences from the upstream node:

- **It reads GGUF.** Routing is by extension: `.gguf` through this pack's loader, everything else through ComfyUI's stock one — deliberately not through the GGUF path, which refuses scaled-fp8 checkpoints, and `qwen3vl_*_fp8_scaled` is the encoder most people have.
- **`type` defaults to `auto`** and is checked against the file even when set by hand. Detection reads headers only, so a 10 GB encoder costs nothing to identify — and an mmproj file, which is the vision projector alone with no text model in it, is refused before the load instead of surfacing later as a missing-attribute error.
- **No `device` or `mode` widget.** Those exist upstream for multi-GPU pinning, which its own README calls actively harmful on a single card: a pinned encoder holds 4–9 GB away from the diffusion model at every sampling step. This node uses ComfyUI's normal paging. If you do want an encoder pinned to a chosen card, install the upstream pack and use its `ClipProj Device Loader` → `ClipProj Apply`.

## Credits

This is a fork. The GGUF loader, the quantization tooling, the custom ops and effectively everything that makes this work are **[city96](https://github.com/city96)**'s — see [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), Apache-2.0. Most of what is merged here on top of it comes from open pull requests on that repository, written by their respective authors and listed in [PR_BACKPORT.md](PR_BACKPORT.md). Bugs in this fork are ours, not theirs — report them here rather than upstream.

`clipproj.py` is ported from **[nicolab28](https://github.com/nicolab28)**'s [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj), MIT — the projection method, the reconstruction formula, the MiniMax-H3 tokenisation and the attention-sink substitution are all theirs, and their licence travels with the file as [LICENSE-ClipProj](LICENSE-ClipProj). The projection matrices are theirs too and are not bundled: [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3).

The GGUF format itself comes from [llama.cpp](https://github.com/ggerganov/llama.cpp).

Model weights belong to their authors and are covered by their own licences — MiniMax-H3 in particular ships under a custom licence rather than an open one. Read it before any use.
