# ComfyUI-GGUF-Loader

GGUF Quantization support for native ComfyUI models — the CCTech Suite fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF).

These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is no need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## What this fork adds

- **Open upstream PRs backported**, including newer architectures (LTX-2, Z-Image, Ideogram-4, Qwen3-VL / MiniMax-H3 text encoders). See [PR_BACKPORT.md](PR_BACKPORT.md).
- **MiniMax-H3 fix**: high-precision buffers such as `adaln_t_table` are kept in float32 through both conversion and loading. Stored as F16 they crash the sampler on the first step with `expected dtype struct c10::Half for 'weight' but got dtype float`.
- **Convenience pipelines**: Scenema Audio and MiniMax Music 3 loaders/generators built on ComfyUI's native models.

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

The GGUF loaders live under `🤖 CCTech/GGUF`; Scenema Audio under `🤖 CCTech/Scenema`; MiniMax Music 3 under `🤖 CCTech/MiniMax Music`; LTX-2.3 A/V under `🤖 CCTech/LTX-2.3`; local LLM/VLM prompting under `🤖 CCTech/LM Studio`; Qwen3-TTS under `🤖 CCTech/Qwen TTS`.

| Node | Purpose |
|---|---|
| **UNET Loader (GGUF)** | drop-in replacement for `Load Diffusion Model` |
| **UNET Loader (GGUF/Advanced)** | same, with `dequant_dtype` / `patch_dtype` / `patch_on_device` exposed |
| **CLIP Loader (GGUF)** | single text encoder, `.gguf` or `safetensors` |
| **Dual / Triple / Quadruple CLIP Loader (GGUF)** | the same for models taking several encoders |
| **Dual VAE Loader (video + audio)** | two VAEs from one node, with the outputs named for their streams |
| **Text Encoder + ClipProj Loader** | loads a small encoder, GGUF included, and projects it into a large one's space |
| **Scenema Models Loader** | the Scenema Audio stack from your own folders — DiT, Gemma-3 text encoder (safetensors **or GGUF**), pipeline checkpoint, VAE encoder |
| **Scenema VAE Encode (voice reference)** | reference clip → audio latent for voice cloning |
| **Scenema Audio Generate** | expressive TTS with presets, scene/language, chunking and A2V voice cloning |
| **Scenema Audio Voice Clone** | standalone offline SeedVC identity transfer |
| **MiniMax Music 3 Models Loader** | loads the pruned AR conditioner, flow DiT, and DAV decoder from explicit dropdowns |
| **MiniMax Music 3 Audio Generate** | caption + structured lyrics → 44.1 kHz stereo music, including AR and DiT controls |

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

Ported from [ScenemaAI/ComfyUI-ScenemaAudio](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio) (MIT), rebuilt on ComfyUI's native LTX-AV machinery and this pack's loaders with no HuggingFace runtime downloads. The SeedVC post-pass reuses Comfy's Whisper and BigVGAN implementations and includes only the checkpoint-specific architecture that Comfy core does not provide. Point the nodes at the same components the sidecar uses (from [ScenemaAI/scenema-audio](https://huggingface.co/ScenemaAI/scenema-audio)):

| File | Folder | Dropdown |
|---|---|---|
| `scenema-audio-transformer-int8.safetensors` (or bf16, or a GGUF quant) | `models/diffusion_models` (unet) | transformer |
| `gemma-3-12b-it-*.gguf` (or a safetensors single-file Gemma-3 12B) | `models/text_encoders` (clip) | text encoder |
| `scenema-audio-pipeline.safetensors` | `models/vae` | pipeline |
| `scenema-audio-vae-encoder.safetensors` | `models/vae` | VAE encoder |

The Models Loader outputs plain comfy **MODEL / CLIP / VAE**: the audio DiT loads as a comfy LTX-AV model with the video paths gated off (the comfy-native equivalent of the original nodes' audio-only monkey-patch), the Gemma text encoder runs as comfy's `LTXAVTEModel` — a GGUF stays quantized through this pack's ops, with the tokenizer rebuilt from the GGUF metadata — and the VAE is comfy's `AudioVAE` (encoder + decoder + BigVGAN vocoder with the 16 kHz → 48 kHz bandwidth extension, so output is 48 kHz stereo). The pipeline checkpoint also supplies the text projection and embeddings connectors.

Generate keeps the original node's behaviour: the `<speak>` XML prompt compiler (with `[bracketed cues]` and per-line action tags), the 12 production presets, scene/language dropdowns, pace, seed, automatic sentence-boundary chunking with A2V voice chaining between chunks, and per-chunk trim/normalize before concatenation. Optional `ref_latent` (from Scenema VAE Encode fed by LoadAudio) gives zero-shot A2V cloning. A final offline SeedVC pass provides fixed identity consistency for multi-chunk output or an explicit `identity_reference`; `Scenema Audio Voice Clone` exposes the same conversion independently. The pass is seeded from the workflow seed, and Generate keeps the unpolished audio if conversion fails or returns anything other than a same-shape, same-length, audible waveform. Place the `seedvc`, `campplus`, `bigvgan`, and `whisper-small` folders from [ChrisColeTech/scenema-audio extras](https://huggingface.co/ChrisColeTech/scenema-audio/tree/main/extras) under `models/scenema-audio/extras`. Whisper word-match validation and the MelBandRoFormer SFX strip remain unported.

A CPU smoke test for the load paths lives at `tools/smoke_scenema.py`.

### LTX-2.3 A/V

For LTX-2.3-family A/V checkpoints split into components — built for the 10Eros v1.4 distilled kit (distill LoRA pre-fused offline), works for any split with the same layout. One loader builds the whole stack on ComfyUI's native LTX-AV machinery:

| File | Folder |
|---|---|
| `10Eros_v1.4_distilled-r72_Q4_K_M/Q6_K/Q8_0.gguf` or `..._fp8mixed.safetensors` | `models/diffusion_models` (unet) |
| `gemma-3-12b-*-Q4_K_M.gguf` (or a Gemma-3 12B safetensors) | `models/text_encoders` |
| `10Eros_v1.4_projections.safetensors` (text_embedding_projection) | `models/text_encoders` |
| `10Eros_v1.4_video_vae.safetensors` | `models/vae` |
| `10Eros_v1.4_audio_vae.safetensors` | `models/vae` |

The GGUF DiTs carry their transformer config as a GGUF KV — no metadata sidecar to lose — so comfy builds the real LTX-2.3 geometry (48 layers, 9-row modulation tables, 4096/2048 embeddings connectors) instead of guessing the older LTX-2 layout. GGUF and fp8 weights stay quantized; the loader outputs plain comfy **MODEL / CLIP / VAE** plus the audio VAE, so everything composes with core nodes.

Sampling: `CLIPTextEncode` → **LTX-2.3 Img/Audio to Video** (one prep node: leave both optionals unconnected for txt2video, connect `image` for i2v, connect `reference_audio` for lip-synced audio-to-video — the official IA2V recipe, with the audio encoded and noise-mask-locked and the video length derived from the clip) → **LTX-2.3 KSampler (distilled)** or **LTX-2.3 Two-Stage Sampler (base + refine)**. Both pass the exact LTX-2 distilled schedules (cfg 1.0, euler) and stamp `frame_rate` onto the conditioning for RoPE; the two-stage sampler additionally runs the official 3-step `refine` pass after a spatial ×2 latent upscale (`LatentUpscaleModelLoader`'s output) in one node, splitting and rejoining the video/audio branches around the upscale model for you. Finish with **LTX-2.3 AV Decode** (video + audio VAE decode + mux, one `fps`) or, by hand, core `LTXVSeparateAVLatent` → `VAE Decode` + `LTXVAudioVAEDecode` → `CreateVideo`.

For an ID-LoRA talking-head pipeline (photo + reference voice → lip-synced video, e.g. the `ltxv23_talking_head` gallery workflow): stack a distilled LoRA (~0.5 strength) and an ID-LoRA (~1.0 strength) onto the model with core `LoraLoaderModelOnly` ×2, and set the reference voice with core `LTXVReferenceAudio` before the sampler — both are already correctly served by stock nodes, no CCTech wrapper needed.

**LTX-2.3 ID-LoRA Prompt Editor** reviews and edits a captioner's generated `[VISUAL]`/`[SPEECH]`/`[SOUNDS]` block before it's used. Wire a captioner (e.g. `LMStudioVisionPrompt`) into its `source` socket; the three multiline boxes fill with the parsed fields after a run and stay directly editable. Type over any of them and the edit survives later runs; generate a genuinely new caption and all three refresh to it. Outputs `tagged_prompt` (the reassembled `[VISUAL]: .../[SPEECH]: .../[SOUNDS]: ...` string for `CLIPTextEncode`) plus `visual_text`/`speech_text`/`sounds_text` individually — wire `speech_text` straight into a TTS node's `text` input so what you typed is exactly what gets spoken.

The keep-the-edit-vs-refresh decision is made in Python by remembering the `source` this node last parsed (keyed by node id), not guessed from whether a box looks empty — and the paired `web/` JS then writes the resolved values back unconditionally. Comfy-core has no widget that is auto-filled *and* editable *and* edit-preserving (its only populate-from-execution widget, `TEXT_PREVIEW` behind `PreviewAny`/`SaveText`, is hard-coded read-only), so this is genuinely custom. Note the parser stops each field at the next `[TAG]:` marker rather than end-of-string, so a middle section like `[SPEECH]` can't swallow `[SOUNDS]` after it.

A 5th output, `speech_text_batch`, splits `[SPEECH]` into one clip per non-blank line (a blank line is a separator, not an empty clip) and carries them as a real comfy list — `OUTPUT_IS_LIST = (False, False, False, False, True)` — rather than a delimited string. Pair it with **LTX-2.3 Speech Batch Selector**, which takes that list plus an `index` (negative counts from the end like Python; out-of-range clamps instead of erroring) and outputs the one clip at that position — `INPUT_IS_LIST = True` so it receives the whole batch in one call instead of comfy fanning out a separate call per clip.

`tools/smoke_ltx23.py` validates every kit file's load path (both DiT formats, all three quants, TE + projections, both VAEs) without sampling. `tools/smoke_id_lora_prompt_editor.py` covers the parser, every state-machine transition (first run empty/filled, edit preserved, source changed, per-node isolation), the batch split, and the selector's indexing/clamping — no GPU required; the JS write-back itself needs a browser to confirm.

### LM Studio

**LM Studio Vision Prompt**, under `🤖 CCTech/LM Studio`, is a local drop-in for a cloud vision-prompt node such as comfy-core's `GeminiNode`: optional image + prompt + system_prompt in, one STRING out, so it slots into an existing workflow (e.g. `ltxv23_talking_head`'s photo-to-prompt step) without touching anything downstream. It talks to [LM Studio](https://lmstudio.ai/)'s local OpenAI-compatible server (`LM Studio > Developer > Start Server`; `base_url` defaults to `http://localhost:1234/v1` and is editable per node). Leave `model` blank to auto-use whatever's currently loaded in LM Studio (a live `/v1/models` call at run time, so it stays correct across model switches and across whichever `base_url` this node points at), or type an exact model id to pin one. A connection failure raises a clear "is the server running?" error rather than a bare traceback. Needs the optional `requests` dependency (`pip install requests`, already in `requirements.txt`).

`tools/smoke_lmstudio.py` mocks the HTTP calls and validates payload assembly, response parsing and error handling — no GPU or running server required.

### Qwen3-TTS

**Qwen3-TTS Models Loader** + **Qwen3-TTS Custom Voice**, under `🤖 CCTech/Qwen TTS`, are a local port of the `flybirdxx/ComfyUI-Qwen-TTS` pack's `FB_Qwen3TTSCustomVoice` node (used for the reference voice in the `ltxv23_talking_head` workflow), wrapping the `qwen-tts` pip package's own `Qwen3TTSModel` directly — the model itself is a `transformers` checkpoint plus a separate codec/vocoder submodel, not this repo's GGUF-quantization territory, so there's nothing to port at the weights level.

The models-folder layout matches `DarioFT/ComfyUI-Qwen3-TTS`'s convention rather than inventing a new one: pick a `repo_id` from the loader's dropdown (CustomVoice/VoiceDesign/Base × 1.7B/0.6B) and it downloads once into `models/Qwen3-TTS/<folder_name>/` (the speech tokenizer/codec lives inside that same repo, no separate download) — an existing HuggingFace/ModelScope cache copy is migrated in place instead of re-downloading if found. Every load after the first is fully offline. To pre-fetch by hand instead:

```
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ComfyUI/models/Qwen3-TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The loader always loads with `local_files_only=True` — no network access once the files are on disk. `speaker` is a built-in named voice baked into the checkpoint (e.g. `"Dylan"`); an unknown name raises an error listing every valid one. `instruct` is real model-native conditioning on the 1.7B checkpoint, but the 0.6B checkpoint silently drops it upstream — this node logs a warning instead of reproducing that silence. Needs the optional `qwen-tts` dependency (already in `requirements.txt`).

`tools/smoke_qwen_tts.py` stubs `folder_paths` and `qwen_tts.Qwen3TTSModel` and validates path discovery, loader kwargs/caching, and the generate node's seeding/AUDIO-dict/unload logic — no GPU or real weights required.

### MiniMax Music 3

The two-node convenience pipeline composes ComfyUI's native MiniMax Music 3 implementation; no model code is vendored. Download `Comfy-Org/MiniMax-Music-3` and place its split files as follows:

| File | Folder |
|---|---|
| `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` (or a prepared GGUF) | `models/text_encoders` |
| `minimax_music3_dit_int8_convrot.safetensors` (or fp16/bf16/GGUF) | `models/diffusion_models` |
| `minimax_music3_dav.safetensors` | `models/vae` |

The loader returns standard comfy **MODEL / CLIP / VAE** objects. Native ConvRot INT8 safetensors use Comfy's optimized mixed-precision path; GGUF remains packed through this pack's operations. Generate exposes caption, section-tagged lyrics, maximum duration (up to 360 seconds), seed, AR CFG/top-k, Euler/simple steps and DiT CFG. Duration is an upper bound: the autoregressive model can end a musically complete song earlier. DAV decoding switches to tiled mode automatically for long outputs and returns 44.1 kHz stereo `AUDIO`.

`tools/smoke_minimax_music3.py` validates the three real checkpoint load paths without generating a song.

## Credits

Fork of [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) (Apache-2.0) — the loader, the quantization tooling and the custom ops are theirs; the backported PRs are their authors', listed in [PR_BACKPORT.md](PR_BACKPORT.md). Report bugs in this fork here, not upstream.

`clipproj.py` is ported from [nicolab28/ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj) (MIT, [LICENSE-ClipProj](LICENSE-ClipProj)); the matrices are theirs too, on [Hugging Face](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3). GGUF comes from [llama.cpp](https://github.com/ggerganov/llama.cpp). The Scenema Audio nodes are ported from [ScenemaAI/ComfyUI-ScenemaAudio](https://github.com/ScenemaAI/ComfyUI-ScenemaAudio) (MIT). SeedVC inference is adapted from [billwuhao/ComfyUI_Seed-VC](https://github.com/billwuhao/ComfyUI_Seed-VC) and [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) (Apache-2.0).

Model weights stay under their own licences — MiniMax-H3's is a custom one, worth reading before any use.
