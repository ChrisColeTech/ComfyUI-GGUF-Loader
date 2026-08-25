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

The Models Loader outputs plain comfy **MODEL / CLIP / VAE**: the audio DiT loads as a comfy LTX-AV model with the video paths gated off (the comfy-native equivalent of the original nodes' audio-only monkey-patch), the Gemma text encoder runs as comfy's `LTXAVTEModel` — a GGUF stays quantized through this pack's ops, with the tokenizer rebuilt from the GGUF metadata — and the VAE is comfy's `AudioVAE` (encoder + decoder + BigVGAN vocoder with the 16 kHz → 48 kHz bandwidth extension, so output is 48 kHz stereo). The pipeline checkpoint also supplies the text projection and embeddings connectors. `keep_loaded` (on by default) caches the built model/clip/vae keyed on the four filename dropdowns, so re-queuing with the same selections skips rebuilding from disk — the DiT alone is 7-10 GB and the text encoder ~24 GB, and unlike `LTXV23ModelsLoader`'s text-encoder cache, this loader previously cached nothing at all. Turn it off to force a genuine reload, e.g. after replacing a file on disk without renaming it.

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

A 5th output, `speech_text_batch`, splits `[SPEECH]` into one clip per non-blank line (a blank line is a separator, not an empty clip) and carries them as a real comfy list — `OUTPUT_IS_LIST = (False, False, False, False, True)` — rather than a delimited string. Pair it with **LTX-2.3 Speech Batch Selector**, which takes that list plus an `index` (negative counts from the end like Python; out-of-range clamps instead of erroring) and outputs the clip at that position plus `count` (the batch's total length, for driving a for-each loop) — `INPUT_IS_LIST = True` so it receives the whole batch in one call instead of comfy fanning out a separate call per clip.

**LTX-2.3 ID-LoRA Assembler** is the Editor's formatting step exposed standalone: three plain `visual`/`speech`/`sounds` `STRING` inputs in, one formatted `[VISUAL]: .../[SPEECH]: .../[SOUNDS]: ...` string out — no `source`, no parsing, no edit-state. For when you already have the three pieces from elsewhere (e.g. a clip picked via `LTXV23SpeechBatchSelector`, or hand-typed values) and just need them combined, rather than parsed apart from a captioner's raw output.

`tools/smoke_ltx23.py` validates every kit file's load path (both DiT formats, all three quants, TE + projections, both VAEs) without sampling. `tools/smoke_id_lora_prompt_editor.py` covers the parser, every state-machine transition (first run empty/filled, edit preserved, source changed, per-node isolation), the batch split, and the selector's indexing/clamping — no GPU required; the JS write-back itself needs a browser to confirm.

### Preprocessors

Under `🤖 CCTech/Preprocessors`: `image` in, structural/identity map `image` out, for `control_image`/`reference_image` inputs across Krea2/Qwen-Image/Flux Klein, or wired anywhere else a photo needs turning into a depth/edge/normal/line/pose map. Each is a from-scratch port of one [`comfyui_controlnet_aux`](https://github.com/Fannovel16/comfyui_controlnet_aux) (Apache-2.0) preprocessor's actual architecture and inference code — not a wrapper around that pack — following this repo's established checkpoint-compatible-port convention (see Depth Anything V2 below). None of the eleven preprocessor families upstream ships got dropped in without checking: this is 11 of the ~15-20 total, picked for having a clean single architecture; the rest (DWPose, Metric3D, UniFormer, Mesh Graphormer, Diffusion Edge, Unimatch, and everything that's really a thin `transformers.from_pretrained(...)` wrapper upstream now) are a deliberately separate, larger follow-up.

Every detector auto-downloads its own weights from HuggingFace on first use into the real ComfyUI install's `models/<family>/` folder — nothing is ever vendored as a weight file in this repo, same pattern this pack's Qwen3-TTS/Depth Anything V2 loaders already use.

| Node | Detects | Architecture | Extra dependency |
|---|---|---|---|
| **Depth Map (Depth Anything V2)** | Depth | DINOv2 ViT + DPT decoder | — |
| **Normal Map (BAE)** | Surface normals | EfficientNet-B5 encoder + uncertainty-aware BN decoder | `timm` |
| **Normal Map (DSINE)** | Surface normals (camera-aware) | EfficientNet-B5 encoder + iterative refinement | `timm` |
| **Soft Edge (HED)** | Soft edges | Small VGG-like multi-scale CNN | — |
| **Soft Edge (PiDiNet)** | Soft edges | Pixel Difference Convolution CNN | — |
| **MLSD Lines** | Straight line segments | MobileNetV2-based line-segment detector | — |
| **Lineart** | Realistic line drawing | ResNet encoder-decoder generator (fine/coarse checkpoints) | — |
| **Lineart (Anime)** | Anime-style line drawing | pix2pix-style U-Net generator | — |
| **Manga Line** | Manga-style clean line extraction | `res_skip` CNN | — |
| **OpenPose** | Body/hand/face keypoints, rendered as a skeleton | Three classic (pre-DWPose) multi-stage CNNs | — (see license note) |
| **Canny** | Edges | Plain `cv2.Canny` — no model, no download | — |

`timm` (`pip install timm`, already in `requirements.txt`) is needed only for the two Normal Map nodes, to build their EfficientNet-B5 backbone the same way the source pack does — everything else needs nothing beyond this repo's existing dependencies.

**License note — Soft Edge (PiDiNet)**: the original PiDiNet authors' LICENSE adds a research-use restriction beyond plain MIT: *"It is just for research purpose, and commercial use should be contacted with authors first."* This is quoted verbatim in `vendor/pidinet.py`'s header and the node's own docstring — read it before using this specific node in a commercial context.

**License note — OpenPose**: the underlying body/hand/face architecture and checkpoints trace back to Carnegie Mellon University's own OpenPose license — **academic or non-profit organization, noncommercial research use only**, quoted in full in `vendor/openpose.py`'s header. Same situation as every other ComfyUI pack that ships this detector (including `comfyui_controlnet_aux` itself, under its own Apache-2.0 wrapper) — the wrapper code's license and the underlying architecture/weights' license are separate things. Read the actual restriction before using this node commercially.

**Consolidation, not triplication**: `Depth Map` and `Canny` used to be duplicated per-pipeline (`Krea2 Depth Map`, `Flux2 Klein Depth Map`, `Qwen-Image Canny`, plus the depth/canny-deriving logic copy-pasted inline in three `Img2Img` nodes' `control_mode="auto_depth"/"auto_canny"` branches). They're now these two shared nodes, used everywhere — the old node names still work in already-saved workflows (registered as aliases resolving to the same shared classes), and `control_mode="auto_depth"`/`"auto_canny"` on Krea2Img2Img/QwenImageImg2Img still work unchanged, now calling the shared implementation under the hood instead of their own copy.

**Not auto-wired into `control_mode` on Krea2Img2Img/QwenImageImg2Img**: those two pipelines' `control_mode` is tied to a specific loaded checkpoint/LoRA (Krea2's Control LoRA is depth/canny-trained only; Qwen-Image's DiffSynth patches are canny/depth/inpaint only) — adding e.g. `auto_normal_bae` there would be mechanically wireable but silently useless, since there's no matching Control LoRA/patch to attach the result to. All eleven preprocessors are still reachable manually on both pipelines: run the standalone node, wire its output into `control_image`. **Flux Klein img2img's `control_mode` DOES include every one of them** — see Flux Klein below, since Klein's mechanism is generic reference-attachment rather than tied to a specific loaded checkpoint.

`tools/smoke_preprocessors.py` covers every node's shape/dtype contract against a faked detector (no download in the offline suite) — 13/13, no GPU. Every node's registration, and a real end-to-end run per architecture family (real HuggingFace download + real inference on a real image, confirming output shape/dtype) were verified separately against the actual portable ComfyUI environment.

### Krea2 Control

Krea2 is natively detected by ComfyUI core (`comfy.sd.load_diffusion_model_state_dict` picks it up via `unet_config.image_model == "krea2"`, `comfy.sd.CLIPType.KREA2` selects its Qwen3-VL-4B text encoder) — there's no bespoke sampling algorithm or conditioning format to reimplement, unlike LTX-2.3. Only two things needed building: a GGUF-aware loader, and the Control LoRA mechanism, which has no comfy-native equivalent at all.

| File | Folder | Dropdown |
|---|---|---|
| Krea2 diffusion model (`.safetensors` or GGUF) | `models/diffusion_models` (unet) | unet_name |
| Krea2 text encoder, Qwen3-VL-4B (`.safetensors` or GGUF) | `models/text_encoders` (clip) | clip_name |
| Krea2 VAE | `models/vae` | vae_name |

**Krea2 Model Loader** is a thin convenience loader — MODEL/CLIP/VAE by name, matching `ZImageLoader`'s shape (no per-checkpoint surgery needed, unlike Scenema, since comfy already knows the architecture).

**Krea2 Control LoRA Loader** loads *any* Krea2 LoRA from `models/loras` and patches it onto a MODEL — it auto-detects which of two unrelated mechanisms the file actually needs, the same auto-detect-and-dispatch approach `Qwen-Image ControlNet Loader` uses for Qwen-Image's own two ControlNet formats, so you don't need to know in advance which loader a given file requires:

- **Widened-projection Control LoRAs** (e.g. `depth-control-lora.safetensors`) ship with the DiT's `first` input-projection layer *widened* — trained to accept image tokens concatenated with control tokens — plus small LoRA-rank patches on the attention blocks. Detected by shape-matching an expanded `first` weight against the live model. The loader patches the block weights through the normal `ModelPatcher` machinery (so offload/low-VRAM handling still applies), and registers a `DIFFUSION_MODEL` wrapper plus a `PatcherInjection` that swap the widened projection in only for the duration of each forward call — image tokens still pass through the model's original `first` layer during that swap (summed with the control contribution), so an ordinary LoRA on the base model keeps working. The projection is restored immediately after each forward call, so removing this node leaves the base model untouched. Ported essentially verbatim from the local `comfyui-krea2-controlnet-main` pack (no LICENSE file shipped; its README credits [Tanmaypatil123/Krea-2-controlnet](https://github.com/Tanmaypatil123/Krea-2-controlnet) for documenting the reference pipeline and [Patil/Krea-2-depth-controlnet](https://huggingface.co/Patil/Krea-2-depth-controlnet) for the public depth LoRA weights) — this is correctness-critical low-level `ModelPatcher` plumbing validated against a working pack. Use `control_mode`/`control_image` on `Krea2 img2img` after this.
- **Ordinary in-context LoRAs** (e.g. [nynxz/NK2E](https://huggingface.co/nynxz/NK2E)'s `krea2_canny-v0.1.safetensors`) have no widened projection at all (confirmed by inspecting its actual tensor keys: plain `lora_down`/`lora_up`/`alpha`, nothing else). Detected by the *absence* of that expanded weight, and applied via `comfy.sd.load_lora_for_models` — the same call stock `LoraLoaderModelOnly` makes internally, no wrapper or injection needed since there's no runtime control-token swap to perform. These work like Qwen-Image-Edit: "structure comes from the edges, content from the text prompt." Use `edit_reference` on `Krea2 img2img` after this, not `control_image`.

**Krea2 Depth Map** turns a source photo into a depth map standalone — for hand-building a graph, or feeding something other than `Krea2 img2img`. Runs Depth Anything V2 (DINOv2 encoder + DPT decoder head), ported from [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) (Apache-2.0) and consolidated into `vendor/depth_anything_v2.py` per this pack's flat-file convention. Weights (`ckpt_name`: vits/vitb/vitl/vitg) auto-download from HuggingFace on first use into `models/depth_anything_v2/`, same pattern as `Qwen3-TTS Models Loader` — nothing extra to install. This is now an alias for the shared `Depth Map (Depth Anything V2)` node under `🤖 CCTech/Preprocessors` (see that section above) — same node, kept registered under its original name so saved workflows keep working. For the common case (depth Control LoRA), you don't need this node at all — see below.

**Krea2 img2img** is the one-node prompt/init-latent/control prep, matching this pack's own `ZImageImg2Img` convention rather than the reference pack's separate Encode+Apply split: `model, clip, vae, prompt, negative_prompt, strength, width, height`, plus optional `image` (your source photo — leave unconnected for txt2img) and optional `control_image` for the loaded Control LoRA.

**Not every "Krea2 canny LoRA" is a Control LoRA** — there are two unrelated families of Krea2 LoRA in the wild ([nynxz/NK2E](https://huggingface.co/nynxz/NK2E)'s `krea2_canny-v0.1.safetensors` is the ordinary in-context kind, the depth one is the widened-projection kind) — but `Krea2 Control LoRA Loader` now handles both from the same node, auto-detecting which one you loaded (see above). Wire `control_mode`/`control_image` after it for a widened-projection LoRA, or `edit_reference` for an in-context one.

`control_mode` (on the widened-projection path) picks how the control signal gets produced, since nothing in a LoRA file says what type it is:
- `auto_depth` (default) — derives a depth map from `image` automatically, using the same Depth Anything V2 model as **Krea2 Depth Map**.
- `auto_canny` — derives a canny edge map from `image` automatically (plain `cv2.Canny`, no model, no download) — for a canny checkpoint that *is* a widened-projection Control LoRA.
- `manual` — no automatic derivation; connect `control_image` yourself. Use this for any widened-projection Control LoRA the two auto modes don't cover (pose/lineart/normal).

Connecting `control_image` explicitly always overrides auto-derivation, in any mode.

**`edit_reference`** is the separate mechanism for in-context/edit-style LoRAs: VAE-encodes the reference image and attaches it to `positive` conditioning as `reference_latents` — confirmed by reading `comfy/ldm/krea2/model.py`'s DiT `_forward()` directly: it has the exact same `ref_latents` parameter Qwen-Image-Edit's DiT does, genuinely separate from both the img2img latent and the widened-projection `control_image`. Not a custom sampler — comfy's generic `extra_conds`/`apply_model` machinery already reads `reference_latents` out of conditioning, same as everywhere else in this pack.

It VAE-encodes the init image and the resolved control image, attaches the control latent to the model, CLIP-encodes your prompt, and attaches `edit_reference` if given — outputs `model, positive, negative, latent, denoise` straight into a stock `KSampler`. If a widened-projection Control LoRA is loaded and neither `control_image` nor a usable `image` is available to derive one from, it raises immediately instead of silently sampling a half-configured model — the same guarantee the original pack's separate `Apply` node existed for. The reverse (`control_image` connected but no Control LoRA loaded) is not an error — there's nothing to attach it to, so it's simply ignored with a warning, so you can leave a preprocessor chain wired in while toggling the LoRA loader on/off.

Minimal graph, either LoRA: `Krea2 Model Loader` → `Krea2 Control LoRA Loader` (same node, any Krea2 LoRA file) → `Krea2 img2img` (prompt typed directly into this node) → `Krea2 KSampler` (or stock `KSampler`) → `VAE Decode`. `Load Image` → `Krea2 img2img`'s `image` (depth derives automatically via `control_mode`) for the depth LoRA, or → `edit_reference` for the canny one — the only thing that changes between the two is which input slot your photo goes into.

**Krea2 KSampler** is a drop-in for stock `KSampler` with one extra option, `denoise_mode`, mirroring `Z-Image KSampler`/`Qwen-Image KSampler`. Krea2 shares the exact same `ModelSamplingFlux`/`shift=1.15` setup as Qwen-Image (literally the same shift value, copy-pasted alongside the Qwen-Image-family config in comfy's own `supported_models.py`), so the same comfy-vs-diffusers denoise-slicing discrepancy applies — verified against a real loaded Krea2 model, not assumed: at 9 steps, denoise 0.9, comfy starts at sigma ≈0.9660 vs ≈0.9619 under the diffusers-style slice. `denoise_mode="comfy"` (default) is unchanged stock behavior; `"diffusers"` matches diffusers-pipeline img2img exactly.

`tools/smoke_krea2.py` covers the tensor-prep helpers (grayscale/normalize/invert/resize), the `Krea2ControlInputProjection` forward math (image-only fallback, and image+control summation), the widened-projection-vs-ordinary-LoRA auto-detection, `Krea2 img2img`'s guard rails, auto_depth/auto_canny derivation, manual-override precedence, `edit_reference` attaching `reference_latents` to positive conditioning only, and `Krea2 KSampler`'s two denoise modes (29/29, no GPU). Loading real GGUF/LoRA weights end-to-end through the *same* `Krea2ControlLoRALoader` for both the widened-projection depth LoRA and the ordinary canny LoRA, the Depth Anything V2 port's `load_state_dict(strict=True)` against the real HuggingFace checkpoint, `ref_latents` existing on the real loaded model's forward signature, and the diffusers-mode sigma math against that same real model, were all verified separately against the actual portable ComfyUI environment.

### Qwen-Image ControlNet

Unlike Krea2's Control LoRA, Qwen-Image ControlNet needed no algorithm ported at all — every format in circulation is already native to ComfyUI core:

| Format | Example file | Comfy mechanism |
|---|---|---|
| InstantX / Union | `*-InstantX-ControlNet-Union.safetensors` | `comfy.controlnet.load_controlnet_state_dict()` → a real `ControlNet` object, attaches to **CONDITIONING** (same as any classic ControlNet) |
| Qwen-Image-Fun ControlNet | — | same dispatcher, also a `ControlNet` object |
| DiffSynth patches (canny/depth/inpaint) | `qwen_image_{canny,depth,inpaint}_diffsynth_controlnet.safetensors` | `comfy_extras.nodes_model_patch.ModelPatchLoader` → a `MODEL_PATCH`, attaches to **MODEL** via `DiffSynthCnetPatch` — the same mechanism this pack's `nodes_zimage.py` already uses for Z-Image's ControlNet |

**Qwen-Image Model Loader** is the same thin GGUF-aware convenience loader as `Krea2ModelLoader`/`ZImageLoader` — MODEL/CLIP/VAE by name (`type="qwen_image"` for CLIP).

**Qwen-Image ControlNet Loader** loads a checkpoint from `models/model_patches` or `models/controlnet` and auto-detects which of the two mechanisms above it needs, checking only the DiffSynth signature (`controlnet_blocks.0.y_rms.weight`, the only real MODEL_PATCH format) and falling through to `comfy.controlnet.load_controlnet_state_dict()` for everything else — InstantX/Union **and** Qwen-Image-Fun both land there, since Fun is a real `ControlNet` architecture despite its name, not a model patch (comfy's own dispatcher already checks the Fun signature — `control_blocks.0.after_proj.weight` + `control_img_in.weight` — internally). It's a dispatcher, not a reimplementation. Outputs a `QWEN_IMAGE_CONTROL` wrapper tagging which attachment point the loaded checkpoint needs.

**Qwen-Image Canny** is the standalone version of `control_mode="auto_canny"` (below) — plain `cv2.Canny` edge detection, no model, no download. Same role `Krea2 Depth Map` plays for depth: wire it in explicitly to preview the edge map before it goes into `control_image`, or reuse it elsewhere, instead of it happening invisibly inside `Qwen-Image img2img`. This is now an alias for the shared `Canny` node under `🤖 CCTech/Preprocessors`.

**Qwen-Image img2img** — same one-node shape as `Krea2Img2Img`: `model, clip, vae, prompt, negative_prompt, strength, width, height`, plus optional `image`, `qwen_control` + `control_image`, and `mask`. It routes to whichever attachment the loaded checkpoint needs automatically — `DiffSynthCnetPatch` on a cloned `MODEL` for DiffSynth patches, or the same `.set_cond_hint()`/`.set_previous_controlnet()` calls stock `ControlNetApplyAdvanced` makes, applied to **CONDITIONING**, for InstantX/Union/Fun — so you never need to know which mechanism a given checkpoint uses.

Unlike Krea2 (one control type: depth), Qwen-Image checkpoints span several different preprocessing needs this pack can't detect from the file, so `control_mode` picks how `control_image` gets produced:
- `manual` (default) — connect `control_image` yourself, matching whichever checkpoint you loaded.
- `auto_canny` — derives a canny edge map from `image` automatically (plain `cv2.Canny`, no model, no download).
- `auto_depth` — derives a depth map from `image` automatically, using the same Depth Anything V2 model as `Krea2 Depth Map`.

Connecting `control_image` explicitly always overrides auto-derivation. `control_image` is required even for an inpaint checkpoint — `mask` only refines the region, it doesn't replace it; there's no photo-only way to auto-derive a mask. Same guard rails as `Krea2Img2Img`: `qwen_control` with nothing usable to attach raises; `control_image` with no `qwen_control` is ignored with a warning.

Graph: `Qwen-Image Model Loader` → `Qwen-Image img2img` (prompt typed directly into this node), alongside `Qwen-Image ControlNet Loader` → `qwen_control`, with `image` doing double duty as the `control_mode` derivation source (or your own control map via `control_image`) → stock `KSampler` → `VAE Decode`.

Not ported: Lotus depth estimation (a diffusion-based depth model some Qwen-Image InstantX workflows use as their depth preprocessor) — it's a different architecture from Depth Anything V2, and a separate job (preprocessor, not ControlNet). `Krea2 Depth Map`'s Depth Anything V2 covers the same role for now.

**`edit_reference`** is a separate, optional input on `Qwen-Image img2img` for Qwen-Image-**Edit** checkpoints, and it is not ControlNet and not a custom sampler. Confirmed by reading `comfy/ldm/qwen_image/model.py`'s DiT forward pass directly: it takes a distinct `ref_latents` parameter, genuinely separate from both the img2img latent and ControlNet's `control` parameter — an Edit checkpoint run without it behaves like a plain generator using weights fine-tuned for editing, not an actual edit. `edit_reference` VAE-encodes the photo you want edited and attaches it to `positive` conditioning as `reference_latents`, the exact mechanism stock comfy's own `ReferenceLatent` node uses (`node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent]}, append=True)`) — comfy's generic `extra_conds`/`apply_model` machinery already reads it out of conditioning and threads it into the real forward parameter, so `KSampler` stays completely architecture-agnostic. Keep it separate from `image` (the img2img starting latent) and `control_image` (structural guidance) — all three answer different questions: what to start denoising from, what structure to follow, and what photo to actually edit.

**Qwen-Image KSampler** is a drop-in for stock `KSampler` with one extra option, `denoise_mode`. Qwen-Image has no bespoke sampling code in comfy at all — it shares `ModelSamplingFlux` (`shift=1.15`) with the rest of the Flux family — but comfy's own denoise-slicing convention (`KSampler.set_steps`: re-expand to `int(steps/denoise)` steps, take the tail) provably diverges from the diffusers img2img convention (compute the schedule at `steps`, slice from `t_start = steps - round(steps*denoise)`), the same class of discrepancy `Z-Image KSampler` was built for. Verified against Qwen-Image's actual `shift` value on a real loaded model, not assumed: at 9 steps, denoise 0.9, comfy starts at sigma ≈0.9660 vs ≈0.9619 under the diffusers-style slice — smaller than Z-Image's measured gap (0.9643 vs 0.9567) but the identical mechanism. `denoise_mode="comfy"` (default) is unchanged stock behavior; `"diffusers"` matches the original pipeline's img2img exactly. This is a compatibility switch, not a quality fix — small in magnitude, worth having for exact parity.

`tools/smoke_qwen_image.py` covers the control-dispatch helper (conditioning stamping, hint/strength passthrough), `Qwen-Image img2img`'s guard rails, latent shapes, both control attachment paths (model_patch → cloned `MODEL`; controlnet → `CONDITIONING`, `MODEL` left untouched), `Qwen-Image Canny`'s output shape, `edit_reference` attaching `reference_latents` to positive conditioning only, and `Qwen-Image KSampler`'s two denoise modes — 13/13, no GPU. The format auto-detection, both control attachment mechanisms, and the diffusers-mode sigma math were all verified separately against the real files/real loaded model in the actual portable ComfyUI environment.

### Flux Klein

FLUX.2 Klein is natively detected by comfy core (`unet_config.image_model == "flux2"`, matched by `comfy/supported_models.py`) — like Krea2 and Qwen-Image, this needed only a thin GGUF-aware convenience loader plus a real port of the multi-reference/identity-transfer tooling from [`ComfyUI-Flux2Klein-Enhancer`](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer) (MIT License, capitan01R) — none of it is vendored or reimplemented from scratch, it's a faithful port of that pack's own mechanisms onto stock comfy `MODEL`/`CONDITIONING` objects.

**Flux Klein Model Loader** is the same thin GGUF-aware convenience loader as `Krea2ModelLoader`/`QwenImageModelLoader` — MODEL/CLIP/VAE by name (`clip_type=CLIPType.FLUX2`, matching core's `CLIPLoader` type dropdown).

**Flux Klein img2img** is the one-node prep step: `model, clip, vae, prompt, negative_prompt, strength, batch_size, width, height`, plus optional `image` and `reference_image`/`control_mode`/`depth_ckpt_name`. The empty txt2img latent uses Flux.2's real shape (`[batch_size, 128, height // 16, width // 16]`, confirmed via comfy core's own `EmptyFlux2LatentImage`) — not the generic 4-channel/8-downscale placeholder used elsewhere in this pack, since comfy's `fix_empty_latent_channels` only auto-corrects channel count, not the spatial downscale ratio, unless `downscale_ratio_spacial` is explicitly passed.

`reference_image` is Klein's real editing mechanism, confirmed by tracing the actual node graph inside the shipped example workflow (`Klein Controlnet.json`, subgraph "Image Edit (Flux.2 Klein 9B Distilled)") rather than assumed: it starts from a *pure-noise* `EmptyFlux2LatentImage` — **not** an img2img partial denoise of the edited photo — and drives the whole edit off reference images VAE-encoded and attached to positive **and** negative conditioning as `reference_latents`, plus a text instruction (its actual saved prompt: *"change the pose of the subject in the image2 to the pose in the image1"*). `image` (the img2img partial-denoise starting point) and `reference_image` (conditioning-only reference) are independent — they answer different questions, same convention as `edit_reference` on `Krea2Img2Img`/`Qwen-Image img2img`.

`control_mode` picks what `reference_image` actually becomes: `manual` (default) attaches it raw; `auto_depth` runs it through this pack's Depth Anything V2 first — reproducing the example workflow's own trick of feeding one reference through `AIO_Preprocessor` (set to `MiDaS-DepthMapPreprocessor`) before encoding it, so the model gets *structure* from one image and *identity/content* from the prompt or another raw reference, rather than two competing raw photos. `depth_ckpt_name` picks the Depth Anything V2 model size (downloads on first use, same as `Krea2 Depth Map`/`Qwen-Image`'s `auto_depth`).

**Flux Klein Depth Map** is that same Depth Anything V2 detector exposed as its own standalone node — image in, depth map out — for wiring a depth reference into `Flux2 Klein Multi Reference Latent` (3+ references) or anywhere else a Klein reference image is wanted, mirroring `Krea2 Depth Map`'s existing explicit-node convention rather than hiding the preprocessing. This is now an alias for the shared `Depth Map (Depth Anything V2)` node under `🤖 CCTech/Preprocessors`, alongside nine other preprocessor families (normal maps, soft edges, lines) any of which can feed `reference_image` manually the same way.

**Flux2 Klein Multi Reference Latent** is a direct port of the source pack's `Multi ReferenceLatent`: one required + up to seven optional `LATENT` inputs, splits each input's batch into individual references, and stamps `reference_latents` (overwrite, not append) + `reference_latents_method="index"` onto both positive **and** negative conditioning — matching the real Klein Controlnet.json example workflow's own reference-conditioning subgraph. `"index"` is a real branch inside comfy's own `Flux._forward` (`comfy/ldm/flux/model.py`, shared Flux/Kontext/Klein code): simple sequential RoPE-index offsets, as opposed to `"uxo"` (spatial tiling) or the unset default (auto-packing).

**Flux Klein Identity Feature Transfer** is a near-verbatim port of the source pack's flagship `IdentityFeatureTransferFinal` — multi-reference identity-preserving feature transfer, using only stock `ModelPatcher` hooks (`set_model_attn1_output_patch`, always; `set_model_attn1_patch`, only when `mask_behavior="zero_unmasked_tokens"` and a mask is wired). Both hooks fire generically from comfy's own `comfy/ldm/flux/layers.py` `DoubleStreamBlock`/`SingleStreamBlock` forward passes — shared Flux/Kontext/Klein code, nothing Klein-repo-specific — and read four `extra_options` keys (`reference_image_num_tokens`, `block_index`, `block_type`, `img_slice`) that comfy's own model code already populates every forward call. The transfer does per-image centering of generated vs. reference features, normalized similarity matching with a configurable floor, temperature-controlled reference pooling, and confidence-gated pull at scheduled double/single blocks, all ported as-is: `preset` (`HARD_LOCK`/`MID_LOCK`/`SOFT_LOCK`/`custom`), `reference_index`/`reference_indices`, `similarity_floor`, `softmax_temperature`, `mask_threshold`, `double_blocks`/`single_blocks` schedule strings, optional `sigmas` (per-step strength decay), `debug`, `mask_behavior`, and up to eight `subject_mask_1..8` inputs.

**Known caveat, carried over from the source rather than fixed speculatively**: the default schedules and presets hardcode block counts (8 double / 24 single blocks) as magic numbers tuned for the Klein 9B layout — they're never read from the live model. Out-of-range indices clamp harmlessly on a different-sized checkpoint (e.g. the 4B `klein-base` variant), but a preset may end up applying strength to the wrong semantic blocks — use `preset="custom"` with your own schedule strings for non-9B checkpoints.

**Explicitly not ported**: `Flux2KleinKSamplerExperimental` — confirmed to reimplement Euler sampling from scratch (manual forward-pass loop, hand-rolled CFG, local shift/schedule math) instead of going through `comfy.samplers`/`CFGGuider`, which means it bypasses comfy's `sampler_post_cfg_function` pipeline entirely. Within the source pack itself this makes it strictly *less* compatible than stock `KSampler` — pairing it with the source's own Color Anchor or Identity Guidance nodes would silently do nothing. Use stock `KSampler` for Klein. Also not ported: the source pack's own superseded `IdentityFeatureTransfer`/`Advanced`/`V3` nodes, kept there only for that pack's backward compatibility.

The remaining nodes round out full parity with the source pack, all ported as faithful, direct translations — no architecture decisions needed since each already reduces to conditioning-dict mutation, `set_model_attn1_patch`, or `sampler_post_cfg_function`:

- **Flux Klein Enhancer** — scalar/whitening ops on the active-token conditioning region, plus a Klein-specific per-Qwen3-layer scale (Klein conditioning stacks 3 hidden-layer slices along the embed dim; `early_layer_scale`/`mid_layer_scale`/`late_layer_scale` target them individually).
- **Flux Klein Detail Controller** — per-section (front/mid/end) conditioning multiplier. Reads real section boundaries from `meta["klein_sections"]` when `Flux Klein Sectioned Encoder` is upstream; otherwise falls back to a fixed, honestly-labeled-as-arbitrary 25/50/25 split.
- **Flux Klein Sectioned Encoder** — tokenizes a front/mid/end sectioned prompt (three text boxes, or one `combined_prompt` with `[FRONT]`/`[MID]`/`[END]` markers) and stamps real per-section token ranges as conditioning metadata by reaching into the CLIP's own HF tokenizer (`clip.tokenizer.qwen3_4b.tokenizer` / `.qwen3_8b.tokenizer` depending on which Klein text-encoder variant is loaded — confirmed against a real loaded Klein CLIP object, not assumed). The one node in this port with a genuine Klein-CLIP-internals dependency; every other node reaches only stock `MODEL`/`CONDITIONING` objects.
- **Flux Klein Text Enhancer** — normalize/contrast/magnitude adjustments on the active text tokens, skipping the BOS token by default (its embedding norm is much larger and skews the stats). Negative contrast never inverts sign (`exp(contrast)` rather than `1+contrast`).
- **Flux Klein Mask Ref Controller** — spatially attenuates one reference latent using a painted `MASK`, replacing it in `reference_latents`. Feather radius for soft edges, `invert_mask` to flip which side gets attenuated.
- **Flux Klein Ref Latent Controller** — scales one reference's K/V attention contribution at every block via `set_model_attn1_patch`, with an optional spatial fade (center-out/edges-out/top-down/left-right) computed over that reference's own token grid.
- **Flux Klein Text/Ref Balance** — single `balance` dial (0.0 = text-only, 1.0 = reference-only, 0.5 = both unscaled) trading text-prompt strength against reference-image strength via the same K/V-scaling mechanism.
- **Flux Klein Ref Latent Weight** — the simple case of Ref Latent Controller: a flat K/V multiplier for one reference, no conditioning input or spatial fade.
- **Flux Klein Color Anchor** and **Flux Klein Identity Guidance** both register via `model_options["sampler_post_cfg_function"]`, which only fires through comfy's own `CFGGuider`/`sampling_function` — **both require sampling through stock `KSampler`** (or anything using comfy's normal sampling pipeline) to have any effect at all. Color Anchor pulls each step's x0 prediction toward a reference latent's per-channel color mean, ramping in over the run (sigma-progress and step-count signals, whichever is further along). Identity Guidance pulls each step toward a VAE-encoded identity reference over a configurable sigma window, in `adaptive` (cosine-similarity-weighted), `direct`, or `channel_match` mode.

Graph (simple, 2-reference edit — matches the shipped example workflow): `Flux Klein Model Loader` → `Flux Klein img2img` (prompt typed directly into this node, one raw `reference_image`, another via `Flux Klein Depth Map` → `control_mode=auto_depth` if you want a second, structural reference) → stock `KSampler` → `VAE Decode`. For 3+ references, chain `Flux2 Klein Multi Reference Latent` after `Flux Klein img2img` instead — it **overwrites** `reference_latents` (matching the source pack's own behavior), so re-supply `reference_image` there too rather than mixing both mechanisms. Optionally `Flux Klein Identity Feature Transfer` on `model`. Any of the conditioning-mutation nodes (`Enhancer`, `Detail Controller`, `Text Enhancer`, `Mask Ref Controller`) slot in between and the sampler; any of the model-hook nodes (`Ref Latent Controller`, `Text/Ref Balance`, `Ref Latent Weight`, `Color Anchor`, `Identity Guidance`) slot onto `model` alongside or after Identity Feature Transfer.

`control_mode` on `Flux Klein img2img` includes an `auto_*` option for every `🤖 CCTech/Preprocessors` node (`auto_canny`, `auto_normal_bae`, `auto_normal_dsine`, `auto_soft_edge_hed`, `auto_soft_edge_pidinet`, `auto_mlsd`, `auto_lineart`, `auto_lineart_anime`, `auto_manga_line`, `auto_openpose`) alongside `manual`/`auto_depth` — a dispatch table maps each mode to its preprocessor node class and calls that node's own tested method, so there's no duplicated detector logic here. Each `auto_*` mode uses that preprocessor's own default settings; for custom settings, run the standalone node yourself with `control_mode=manual`.

`tools/smoke_flux_klein.py` covers all sixteen nodes — `Flux Klein img2img`'s latent shapes (real Flux.2 128-channel shape, denoise=strength for img2img, batch repeat), its `reference_image`/`control_mode` attachment (manual attaches raw to both positive+negative, auto_depth routes through the depth helper first, auto_canny dispatches to the shared Canny node for real, a completeness check that every preprocessor node has a control_mode entry, no `reference_image` leaves conditioning untouched), `Flux Klein Depth Map`'s delegation to the shared depth helper, `Flux2 Klein Multi Reference Latent`'s batch-splitting/ordering/overwrite behavior and dual positive+negative attachment, `Flux Klein Identity Feature Transfer`'s hook registration and similarity-pull math on a synthetic attention tensor, and every Part E node's core behavior (Color Anchor's inactive-without-reference guard and hook registration on a clone; Enhancer's no-op passthrough and active-region scaling; Detail Controller's real-vs-fallback section ranges; Text Enhancer's BOS-skipping; Mask Ref Controller's black-mask attenuation and no-reference-latents guard; Ref Latent Controller/Text-Ref Balance/Ref Latent Weight's attn1_patch K/V scaling on synthetic q/k/v tensors; Identity Guidance's direct-mode pull math and sigma-window gating; Sectioned Encoder's klein_sections emission with a fake HF tokenizer and its no-tokenizer fallback) — 36/36, no GPU. Native Flux2 detection, `CLIPType.FLUX2`, all 16 nodes' registration, the real `clip.tokenizer.qwen3_4b.tokenizer` attribute path, every model-hook node's `set_model_attn1_patch`/`sampler_post_cfg_function` registration, `reference_image`'s actual `reference_latents` attachment, and `control_mode=auto_openpose` end-to-end (real OpenPose detector + real Klein VAE/CLIP) were all verified separately against the actual portable ComfyUI environment, the user's real `flux-2-klein-9b` GGUF checkpoint, a real loaded Klein CLIP object, and the real Flux.2 VAE.

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
