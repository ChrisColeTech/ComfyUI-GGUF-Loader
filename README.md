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

The GGUF loaders live under `🤖 CCTech/GGUF`; the Scenema Audio nodes under `🤖 CCTech/Scenema`.

| Node | Purpose |
|---|---|
| **UNET Loader (GGUF)** | drop-in replacement for `Load Diffusion Model` |
| **UNET Loader (GGUF/Advanced)** | same, with `dequant_dtype` / `patch_dtype` / `patch_on_device` exposed |
| **CLIP Loader (GGUF)** | single text encoder, `.gguf` or `safetensors` |
| **Dual / Triple / Quadruple CLIP Loader (GGUF)** | the same for models taking several encoders |
| **Dual VAE Loader (video + audio)** | two VAEs from one node, with the outputs named for their streams |
| **Text Encoder + ClipProj Loader** | loads a small encoder, GGUF included, and projects it into a large one's space |
| **Scenema Models Loader** | the Scenema Audio stack from your own folders — DiT, Gemma-3 text encoder (safetensors **or GGUF**), pipeline checkpoint, VAE encoder |
| **Scenema VAE Encode (voice reference)** | reference clip → audio latent for voice cloning || **Scenema VAE Encode (voice reference)** | reference clip → audio latent for voice cloning |
| **Scenema Audio Generate** | expressive TTS with presets, scene/language, chunking and A2V voice cloning |

### Dual VAE Loader

MiniMax-H3 decodes video and audio through two separate VAEs, so every workflow ends up with a pair of `VAELoader` nodes side by side. This is that pair in one node, with outlets named `video_vae` and `audio_vae`. Loading is delegated to the stock loader, so `taesd` entries stay available and nothing behaves differently.

### Text Encoder + ClipProj Loader

Loads a small text encoder and projects it into a large one's space in a single node — for MiniMax-H3, a Qwen3-VL-4B or -8B standing in for the 32B the model expects, which takes the text encoder from 15.7 GB to about 5.

The projection lives in `clipproj.py`, so there is no dependency on another node pack.[^clipproj] What you do need is a **matrix**, in `ComfyUI/models/clip_projections/`: `mmh3-4b-*` goes with a 4B and `mmh3-8b-*` with an 8B, and they are not interchangeable. The `<control:...>` entries are not projections but deliberate baselines — zero ignores your prompt entirely, identity copies the raw dimensions with no learning — and they run on any encoder. Run them first to see that the matrix is doing the work.

[^clipproj]: Ported from [ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) by nicolab28 (MIT, [LICENSE-ClipProj](LICENSE-ClipProj)), and checked against it: identical conditioning to the bit. The matrices are theirs as well — [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3).

Differences from the upstream node:

- **It reads GGUF.** Routing is by extension: `.gguf` through this pack's loader, everything else through ComfyUI's stock one — deliberately not through the GGUF path, which refuses scaled-fp8 checkpoints, and `qwen3vl_*_fp8_scaled` is the encoder most people have.
- **`type` defaults to `auto`** and is checked against the file even when set by hand. Detection reads headers only, so a 10 GB encoder costs nothing to identify — and an mmproj file, which is the vision projector alone with no text model in it, is refused before the load instead of surfacing later as a missing-attribute error.
- **No `device` or `mode` widget.** Those exist upstream for multi-GPU pinning, which its own README calls actively harmful on a single card: a pinned encoder holds 4–9 GB away from the diffusion model at every sampling step. This node uses ComfyUI's normal paging. If you do want an encoder pinned to a chosen card, install the upstream pack and use its `ClipProj Device Loader` → `ClipProj Apply`.

### Scenema Audio

Ported from [ScenemaAI/ComfyUI-ScenemaAudio](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio) (MIT), rebuilt on ComfyUI's native LTX-AV machinery and this pack's loaders — no vendored model code, no HuggingFace auto-download. Point the nodes at the same components the sidecar uses (from [ScenemaAI/scenema-audio](https://huggingface.co/ScenemaAI/scenema-audio)):

| File | Folder | Dropdown |
|---|---|---|
| `scenema-audio-transformer-int8.safetensors` (or bf16, or a GGUF quant) | `models/diffusion_models` (unet) | transformer |
| `gemma-3-12b-it-*.gguf` (or a safetensors single-file Gemma-3 12B) | `models/text_encoders` (clip) | text encoder |
| `scenema-audio-pipeline.safetensors` | `models/vae` | pipeline |
| `scenema-audio-vae-encoder.safetensors` | `models/vae` | VAE encoder |

The Models Loader outputs plain comfy **MODEL / CLIP / VAE**: the audio DiT loads as a comfy LTX-AV model with the video paths gated off (the comfy-native equivalent of the original nodes' audio-only monkey-patch), the Gemma text encoder runs as comfy's `LTXAVTEModel` — a GGUF stays quantized through this pack's ops, with the tokenizer rebuilt from the GGUF metadata — and the VAE is comfy's `AudioVAE` (encoder + decoder + BigVGAN vocoder with the 16 kHz → 48 kHz bandwidth extension, so output is 48 kHz stereo). The pipeline checkpoint also supplies the text projection and embeddings connectors.

Generate keeps the original node's behaviour: the `<speak>` XML prompt compiler (with `[bracketed cues]` and per-line action tags), the 12 production presets, scene/language dropdowns, pace, seed, automatic sentence-boundary chunking with A2V voice chaining between chunks, and per-chunk trim/normalize before concatenation. Optional `ref_latent` (from Scenema VAE Encode fed by LoadAudio) gives zero-shot voice cloning. Not ported, because they need non-comfy dependencies: Whisper word-match validation, SeedVC polish, and the MelBandRoFormer SFX strip.

A CPU smoke test for the load paths lives at `tools/smoke_scenema.py`.

## Credits

Fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0) — the loader, the quantization tooling and the custom ops are theirs; the backported PRs are their authors', listed in [PR_BACKPORT.md](PR_BACKPORT.md). Report bugs in this fork here, not upstream.

`clipproj.py` is ported from [nicolab28/ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) (MIT, [LICENSE-ClipProj](LICENSE-ClipProj)); the matrices are theirs too, on [Hugging Face](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3). GGUF comes from [llama.cpp](https://github.com/ggerganov/llama.cpp). The Scenema Audio nodes are ported from [ScenemaAI/ComfyUI-ScenemaAudio](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio) (MIT).

Model weights stay under their own licences — MiniMax-H3's is a custom one, worth reading before any use.
