# AGENTS.md — Comfy-GGUF (CCTech fork) + Scenema Audio nodes

Working notes for any agent (or human) continuing work in this repo. Everything
below was verified against real checkpoints and real ComfyUI code, not inferred.

## Repo layout

The pack is a Python package (`__init__.py` at repo root doing `from .nodes import
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS`) with three subpackages:

- `nodes/` — every ComfyUI node class, one module per feature area.
  - `nodes/gguf.py` — the base GGUF loader nodes (UNET/CLIP loaders: `GGUFModelPatcher`,
    `UnetLoaderGGUF(Advanced)`, `CLIPLoaderGGUF`, `Dual/Triple/QuadrupleCLIPLoaderGGUF`).
    Registers `unet_gguf` / `clip_gguf` folder keys. Node category: `🤖 CCTech/GGUF`.
  - `nodes/__init__.py` — aggregates every other `nodes/*.py` module's
    `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS` into one dict (order matters —
    see the comments inline for why each import comes where it does).
  - `nodes/scenema.py` — the Scenema Audio nodes (category `🤖 CCTech/Scenema`).
  - `nodes/extra.py`, `nodes/minimax_h3_prompt.py` — DualVAELoader/ClipProjLoader, and
    the MiniMax-H3 prompt-schema helper (CCTech's own, not vendored).
  - `nodes/preprocessors.py` — the MINIMUM shared preprocessing: `DepthMap`
    (Depth Anything V2) + `Canny` (plain cv2), used internally by
    `control_mode="auto_depth"/"auto_canny"` on Krea2Img2Img/QwenImageImg2Img/
    FluxKleinImg2Img. Does NOT register any node of its own - registering
    generic "DepthMap"/"Canny" here would collide with the separate
    ComfyUI-ControlNet-Nodes package's own registration of those names.
    `Krea2DepthMap`/`Flux2KleinDepthMap`/`QwenImageCanny` are registered in
    their own pipeline's `NODE_CLASS_MAPPINGS`, pointing at this file's
    `DepthMap`/`Canny` classes. The full 11-node ControlNet-aux preprocessor
    set (normal maps, soft edges, MLSD, lineart variants, OpenPose) lives in
    the standalone https://github.com/ChrisColeTech/ComfyUI-ControlNet-Nodes
    package instead - see the 2026-08-25 dated entries below for why.
  - `nodes/{flux_klein,krea2,lmstudio,ltx23,ltx25,minimax_h3,minimax_music,qwen_image,
    qwen_tts,stems,zimage}.py` — the rest of the node families, one file each.
    `flux_klein.py` only holds the 5 genuinely Klein-specific nodes - 9
    architecturally-generic Flux-family reference-conditioning nodes moved to
    the standalone https://github.com/ChrisColeTech/ComfyUI-Flux-Reference-Tools
    package (see the 2026-08-25 dated entry below).
- `ops/` — GGMLTensor/GGMLOps: weights stay quantized, dequant per-layer at forward
  time. **This is the repo's core identity.** `ops/__init__.py` (was `ops.py`) is the
  package's public surface; `ops/dequant.py` holds the low-level dequant kernels.
- `vendor/` — ported/adapted third-party model code, each file's header says which
  project it was ported from: `clipproj.py`, `depth_anything_v2.py` (the only
  preprocessor architecture kept here - see 2026-08-25 dated entries),
  `melband_arch.py`, `seedvc.py`, `seedvc_arch.py`, `seedvc_utils.py`,
  `LICENSE-ClipProj`.
- `loader.py` — GGUF → fake-quantized state dict (`gguf_sd_loader`), text-encoder
  post-processing (`gguf_clip_loader`: key remaps, Gemma-3 norm `+1` un-bake,
  sentencepiece tokenizer rebuild from GGUF metadata — now cached next to the
  GGUF as `<name>.spiece_cache.bin`). Stays at repo root — distinct from `ops/`'s
  low-level tensor ops.
- `tools/smoke_scenema.py` — CPU dry-run of the Scenema load paths against real
  checkpoints (`--skip-te` skips the slow Gemma GGUF part).

Cross-file relative imports: within `nodes/`, siblings import each other as
`from .gguf import X`; things one level up (`loader.py`, `ops/`, `vendor/`) are
`from ..loader import X` / `from ..ops import X` / `from ..vendor import X`. `import
nodes` (unqualified, no dot) anywhere in the codebase means ComfyUI core's own global
`nodes` module (e.g. `nodes.MAX_RESOLUTION`) — unrelated to this repo's `nodes/`
package despite the name collision.

Reference ComfyUI checkout used for verification: `D:\Projects\ComfyUI\upstream\ComfyUI`
(rev 155 / v0.32.0 era). Weights: `D:\models\image-models\scenema-audio` (the
sidecar layout, `split/` subdirs). The original node pack lives at
`D:\Projects\ComfyUI\ComfyUI-ScenemaAudio` (MIT, ScenemaAI); the pure-torch
sidecar port at `D:\Projects\giga-videos\src\sidecar\vendor\pipelines\scenema_audio`.

## The Scenema weight set (what the nodes must load)

| File | Role |
|---|---|
| `scenema-audio-transformer-int8.safetensors` (4.9 GB) | audio DiT. INT8 per-layer: keys `…weight.int8` (int8, `[out,in]`) + `…weight.scale` (fp32, per output row). Metadata `config` holds the full transformer/VAE/vocoder config. **Video-path weights are stripped** — audio-only checkpoint. |
| `scenema-audio-transformer.safetensors` (9.8 GB) | same DiT in bf16 |
| `scenema-audio-pipeline.safetensors` (7 GB) | text_embedding_projection (dual: 3840·49 → 4096 video / 2048 audio), `model.diffusion_model.{audio,video}_embeddings_connector` (8× transformer-1d blocks + 128 learnable registers each), audio VAE encoder+decoder, BigVGAN vocoder + BWE stage (16 kHz → 48 kHz stereo). Same `config` metadata. |
| `scenema-audio-pipeline-audio.safetensors` | decoder+vocoder only subset (optional; not on disk here) |
| `scenema-audio-vae-encoder.safetensors` (43 MB) | standalone VAE encoder (44 tensors, bare keys `encoder.*`, `per_channel_statistics.*`) |
| `gemma-3-12b-it-Q4_K_M.gguf` | the text encoder. arch `gemma3` — already supported by this repo's `gguf_clip_loader`. |

DiT keys are `velocity_model.*` → comfy layout is `model.diffusion_model.*`.
Key names then match comfy's `LTXAVModel` **1:1** (verified against
`comfy/ldm/lightricks/av_model.py`: `transformer_blocks.N.audio_attn1/2`,
`audio_ff`, `audio_to_video_attn`, `video_to_audio_attn`, adaln singles, etc.).
There is also `mmproj-F16.gguf` in the TE folder — that is Gemma's SigLIP
vision projector, **not** usable as the text encoder.

## The comfy-native mapping (why this port works at all)

ComfyUI core ships the entire LTX-2 A/V stack; nothing needs vendoring:

- **DiT** → `comfy.ldm.lightricks.av_model.LTXAVModel`. Detection
  (`model_detection.py:398-408`) reads `transformer_blocks.0.attn2.to_k.weight`
  (a *video* tensor, missing in the audio-only ckpt) — so the loader pads one
  zero tensor of the right shape; the embedded `config` metadata then overrides
  every detection value. `image_model: "ltxav"` requires `audio_adaln_single.linear.weight`
  present (it is). Embeddings connectors are real comfy modules on LTXAVModel
  (`preprocess_text_embeds`) — merge the pipeline ckpt's
  `model.diffusion_model.*` tensors into the DiT sd before load.
- **Audio-only forward**: comfy's block forward has native gates —
  `transformer_options["run_vx"]=False`, `a2v_cross_attn=False`,
  `v2a_cross_attn=False` — which reproduce the original pack's monkey-patched
  `audio_only_forward` exactly. Bake them into `model.model_options["transformer_options"]`;
  samplers merge patcher-level transformer_options into every step (`samplers.py:312`).
- **Text encoder** → `comfy.sd.load_text_encoder_state_dicts(clip_type=CLIPType.LTXV,
  state_dicts=[gemma_sd, pipeline_sd])`. Comfy detects the dual projection from
  `text_embedding_projection.audio_aggregate_embed.bias` (`sd_detect` in
  `comfy/text_encoders/lt.py`) and builds `LTXAVTEModel` = `Gemma3_12BModel`
  (layer="all" → all 49 hidden states via `Llama2_.forward(intermediate_output="all")`)
  + `DualLinearProjection`. The GGUF path rebuilds a spiece tokenizer from
  metadata; Gemma-3 norm `+1` un-bake and key remap are already in `loader.py`.
  `CLIPType.LTXV` branch lives at `sd.py:1962`. A plain `CLIP Loader (GGUF)`
  (e.g. type `stable_diffusion`) emits raw Gemma hidden states in the wrong
  layout — the DiT's connector crashes on them (`cat 4-D vs 3-D`). Generate
  validates for `text_embedding_projection` and errors early.
- **VAE** → `comfy.sd.VAE(sd={audio_vae.*, vocoder.*}, metadata=pipeline meta)`.
  The `vocoder.resblocks…` detection branch (`sd.py:931`) builds
  `comfy.ldm.lightricks.vae.audio_vae.AudioVAE` (encoder + decoder + BigVGAN
  vocoder with BWE → 48 kHz stereo out). The VAE loader node in comfy core
  (`nodes_lt_audio.py LTXVAudioVAELoader`) does exactly this prefix replace.
  `AudioVAE.encode(waveform[B,C,T], sample_rate=…)`, `.decode(latent)` returns
  [B, T, C]; `fsm.num_of_latents_from_frames(frames, fps)`, `latent_channels=8`,
  `latent_frequency_bins=16`, `sample_rate=16000`, `output_sample_rate=48000`.
  10 s @ 24 fps → 251 latents (verified).
- **Sampling** → comfy's normal stack. The distilled schedule is
  `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`
  (8 steps, `ltx_pipelines/utils/constants.py DISTILLED_SIGMAS`).
  `ModelSamplingFlux.timestep(sigma)=sigma` and the DiT applies ×1000 itself —
  identical semantics to ltx_core, so passing these sigmas directly to
  `comfy.sample.sample(..., cfg=1.0, sampler="euler", scheduler="simple")`
  matches `SimpleDenoiser` + `EulerDiffusionStep`. Latents are comfy
  NestedTensors: build `(video_zeros[1,128,1,1,1], audio_zeros[1,8,T,16])`,
  `comfy.sample.prepare_noise(latent, seed)`, output `.unbind()[-1]` is the
  audio stream.
- **A2V voice reference** → cond `ref_audio={"tokens": [B, T, C*F]}` on the
  conditioning (comfy's `LTXVReferenceAudio` node is the reference impl); the
  model prepends them at timestep 0 and strips them on output
  (`av_model.py:708-731, 1029-1033`). Chunk chaining: encode 3 s tail of chunk
  N as the reference for chunk N+1 (original: `REF_TAIL_SECONDS=3.0`).

## LTX-2.5 split support (D:\models\image-models\ltxv25\split)

Verified headless on the portable (ComfyUI 0.33.0, RTX 5090): full E2E
generation (Gemma-4 TE GGUF + ltx-v2-projections → 22B distilled DiT GGUF →
video+audio VAE decode) works. Three sidecar bridges make the GGUFs loadable:

1. **Gemma-4 tokenizer sidecar** — `gemma4`-arch TEs need the full HF
   `tokenizer.json` (comfy's `Gemma4SDTokenizer` is `tokenizers`-lib based,
   NOT sentencepiece-rebuildable like gemma3). Without it comfy falls back to
   a path string and dies on `'str' object has no attribute 'decode'`. The
   loader reads `<gguf>.tokenizer.json` (extract once from the comfy
   safetensors' `tokenizer_json` U8 tensor) and vocab-checks it against the
   GGUF token count.
2. **Gemma-4 fixup sidecar** — `<gguf>.fixup.safetensors` carries the 48
   per-layer `layer_scalar` constants (learned values ~0.005–0.93, multiplied
   into every block output; comfy holds them in `torch.empty` buffers, so a
   missing tensor = silent garbage, and the loader now hard-errors instead).
   Extract from the source checkpoint's `model.layers.N.layer_scalar` (BF16
   [1]).
3. **DiT config sidecar** — `UnetLoaderGGUF` merges `<gguf>-metadata.json` /
   a *claiming* `*metadata*.json` sibling into the metadata, translates LTX's key
   names to comfy kwargs (`cross_attn_mod`→`cross_attention_adaln`,
   `gated_attn`→`apply_gated_attention`, `rope_theta`→
   `positional_embedding_theta`, `cross_attn_timestep_scale_multiplier`→
   `av_ca_timestep_scale_multiplier`, `pos_embed_max_pos`+base_*→
   `positional_embedding_max_pos`, connector heads from main heads), and
   forces flags from weight shapes (9-row `scale_shift_table`→adaln,
   connector `to_gate_logits`→`connector_apply_gated_attention`, no
   `caption_projection.*`+connectors→`caption_proj_before_connector=True` +
   identity projections for the 6144-dim pre-projected context). GGUFs also
   store `keyframes_abs_pos_embedding` 1-D; the loader reshapes to `[1, dim]`.

Correct workflow wiring: `Dual CLIP Loader (GGUF)` clip1=`gemma4-12b-ltx-2.5-
Q6_K.gguf`, clip2=`ltx-v2-projections.safetensors`, type=`ltxv`; VAEs via the
stock VAE Loader (`ltx-2.5-video-vae-bf16`, `ltx-2.5-audio-vae-bf16`). The
portable copies of all sidecars are already in place.

`models/diffusion_models` is a shared drawer, so a folder-level sidecar only
applies to a GGUF that it *claims*: either the sidecar lists matching filename
globs in `applies_to` (our shipped one carries `["ltx-2.5-*"]`), or its own
name shares a distinctive token with the GGUF's (quant tags and words like
`transformer`/`dit`/`model` do not count). A sidecar named exactly after the
GGUF always wins; a nameless `metadata.json` is still trusted since there is
nothing to check it against. Before this rule the sole LTX sidecar was merged
onto every other GGUF in the folder — MiniMax H3 built with 48 blocks instead
of 50 and the wrong attention geometry, loading fine and then failing in
`token_refiner` with `too many values to unpack (expected 3)`.

The latent comes from `LTXV25EmptyLatentAVBatch` (`nodes_ltx25.py`, category
`🤖 CCTech/LTX-2.5`). It exists because the MiniMax H3 empty-AV node
produces the same nested-tensor *type* and so wires up cleanly, then dies in
`patchify_proj` — H3 is video `[B,24,T,H/16,W/16]` + audio `[B,32,2,T*40]`,
LTX-2.5 is video `[B,128,(len-1)//8+1,H/32,W/32]` + audio
`[B,z,n_latents,bins]`. The audio side is read off the audio VAE
(`first_stage_model.num_of_latents_from_frames` / `.latent_frequency_bins`,
25 latents/s on the shipped VAE) rather than hardcoded, and a video VAE in the
audio slot is rejected with a named error. Unlike H3 the LTX-2.5 DiT is
batch-aware, so `batch_size > 1` needs no patch node.

## What the original ScenemaAudio nodes did that we ported

XML `<speak>` prompt compiler (voice/gender/scene/language/shot attrs, action/
sound/text blocks, `[bracketed cues]` inline), 12 production presets,
scene presets + CLEAN_SPEECH_SCENES, sentence-boundary chunking at 15 s max
with per-sentence action re-attachment, pace multiplier (default 1.5), word-count
duration fallback (no Kokoro), per-chunk trim/normalize/LUFS then concat +
silence shortening, 20 s reference cap. Dropped (non-comfy deps): Whisper
validation and MelBandRoFormer SFX strip. **SeedVC must not be treated as
optional for multi-chunk voice consistency**; the current node omitted it and
therefore does not reproduce the complete Scenema pipeline.

## SeedVC voice-consistency stage

The Hugging Face model card explicitly documents both the ~15-second segment
window and the long-form continuity path:

- Limitations: "15-second generation window: Each segment capped at ~15s."
- Architecture: long text splits at sentence boundaries, A2V conditions the
  next segment, then SeedVC performs voice identity transfer for multi-chunk
  output (or an explicit reference).

`nodes_scenema.py` implements A2V latent chaining and a final SeedVC pass over
the combined waveform using one fixed identity. These are separate stages:
using only each generated tail as the next A2V reference can accumulate small
voice changes, while SeedVC anchors the final output to one identity.

### Verified reference implementations

- Standalone Comfy SeedVC node used as the architecture port source:
  `D:\Projects\ComfyUI\ComfyUI_Seed-VC-main`
- Original Comfy pack wrapper:
  `D:\Projects\ComfyUI\ComfyUI-ScenemaAudio\nodes\seedvc.py`
  - public helper: `convert_voice(source_audio, reference_audio, steps=25,
    cfg_rate=0.5)`
  - standalone node: `ScenemaAudioVoiceClone`
  - automatic Generate call:
    `D:\Projects\ComfyUI\ComfyUI-ScenemaAudio\nodes\generate.py:417-428,557-571`
- Pure-torch reference (preferred behavior, no transformers/diffusers/HF
  runtime download):
  `D:\Projects\giga-videos\src\sidecar\vendor\pipelines\scenema_audio\ops\seed_vc.py`
  - `load_seed_vc(device=None)`
  - `unload_seed_vc()`
  - `convert_voice(source, source_sr, reference, reference_sr, *, steps=25,
    cfg_rate=0.5, output_sr=None, unload_after=True, device=None)`
  - `apply_seed_vc_to_result(...)`
- Pure Whisper-small encoder used by that path:
  `D:\Projects\giga-videos\src\sidecar\vendor\pipelines\scenema_audio\models\whisper_encoder.py`
- SeedVC architecture reference:
  `D:\Projects\giga-videos\src\sidecar\vendor\pipelines\scenema_audio\vendor\seedvc\modules`

The Comfy custom node should use Comfy's own model/device/offload facilities
and existing installed model classes where available. Do not import the
`giga-videos` checkout at runtime and do not make the node depend on a
machine-specific source path.

### Required components and model layout

Canonical download source:
`https://huggingface.co/ChrisColeTech/scenema-audio/tree/main/extras`.
Install beneath the target Comfy models tree's Scenema extras directory (the
loader must register/resolve this location rather than hard-code a development
checkout). Components:

| Component | Required files | Role |
|---|---|---|
| SeedVC DiT | `seedvc/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`, `seedvc/config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | conditional flow/diffusion voice conversion model |
| CAMPPlus | `campplus/campplus_cn_common.bin` | 192-d speaker identity encoder |
| BigVGAN | `bigvgan/config.json`, `bigvgan/bigvgan_generator.pt` | 22.05 kHz waveform vocoder |
| Whisper-small | `whisper-small/config.json`, `whisper-small/model.safetensors`, `whisper-small/preprocessor_config.json` | semantic speech encoder |

Known development copy with matching sizes/checksums:
`D:\models\image-models\scenema-audio\extras\{seedvc,campplus,bigvgan,whisper-small}`.
The target portable Comfy tree is:
`P:\Projects\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models`.

Principal weight sizes are approximately 440 MB SeedVC DiT, 28 MB CAMPPlus,
449 MB BigVGAN, and 967 MB Whisper-small (about 1.76 GiB total).

### Runtime contract

- Internal SeedVC waveform rate: 22,050 Hz.
- Whisper/CAMPPlus feature rate: 16,000 Hz.
- Defaults: 25 diffusion steps, CFG rate 0.5, length adjustment 1.0,
  three quantizers.
- Reference is capped at 25 seconds.
- Sources over 30 seconds use overlapping semantic windows; generated SeedVC
  chunks overlap by 16 mel frames (~186 ms) and are crossfaded.
- Input/output Comfy contract: `AUDIO` waveform `[B,C,T]`; internal conversion
  is mono; resample final output back to 48 kHz and restore the original channel
  count so Generate still returns `[1,C,T]` stereo-compatible audio.

Correct automatic trigger and identity precedence (sidecar reference):

1. Run when `not skip_vc and (len(chunks) > 1 or reference_audio is not None)`.
2. Explicit reference waveform wins when supplied.
3. Otherwise use the first post-trim/normalized generated chunk as the fixed
   identity for the entire combined output.
4. `ref_latent` remains the separate LTX A2V conditioning input; do not confuse
   it with SeedVC waveform identity. Add an optional `AUDIO` reference input if
   explicit SeedVC identity is required.
5. Automatic Generate integration should catch SeedVC failures, log them, and
   return the already-generated unpolished waveform rather than losing a long
   generation. A standalone explicitly requested voice-conversion node may
   raise.
6. An exception is not the only failure mode. Validate the returned waveform
   (`seedvc_output_is_usable`) before it replaces good speech: same shape, a
   comparable length, finite, and audible.
7. The flow-matching noise is seeded from the workflow seed, so a conversion is
   reproducible and A/B comparisons against the reference implementation use
   identical initial noise.
8. `convert_voice` either owns its bundle (loads and unloads it) or borrows a
   caller-supplied `bundle=`. There is no flag that loads a bundle and then
   drops the only reference to it.

### Resource lifecycle

- Finish all DiT chunks and VAE decoding first.
- Release/offload Gemma, Scenema DiT, and VAE through Comfy model management
  before staging SeedVC; do not use unmanaged permanent CUDA singletons.
- Load SeedVC components only for the final post-pass and unload them in
  `finally`, including partial-load failures.
- Keep final conversion out of the per-chunk loop.
- The original wrapper estimates roughly 3.5 GB SeedVC VRAM. The portable
  environment must not overlap that with the 7.3 GB Scenema DiT or 28 GB text
  encoder staging.

## Resource-correct implementation

The broken loader from commit 8c2cd9f called `_expand_int8`, multiplying every
INT8 matrix by its scale into a persistent float32 state dict. That path has
been removed. The current implementation follows these rules:

1. **Scenema safetensors INT8 stays packed.** `_convert_int8_keys` only maps
   `<layer>.weight.int8` to `<layer>.weight`, maps the scale to
   `<layer>.weight_scale` (a `[out, 1]` view), and adds
   `<layer>.comfy_quant = {"format":"int8_tensorwise"}`. Comfy selects
   `mixed_precision_ops`, stores a `QuantizedTensor`, accounts the actual INT8
   bytes, and dequantizes only the active layer during forward. There is no
   dense fallback; old Comfy versions fail with an upgrade message.
2. **Missing video weights are never created for INT8 or bf16.** Native mixed
   ops construct Linear modules with `weight=None`. The INT8 metadata selects
   them automatically; bf16 explicitly uses `comfy.ops.mixed_precision_ops({})`.
   Present bf16 weights are assigned normally while missing video weights stay
   `None`.
3. **GGUF missing video weights also stay unallocated.** `_ScenemaGGMLOps`
   overrides the generic GGMLOps missing-Linear behavior, which otherwise
   creates a large dense zero tensor.
4. **All gated block modules are removed before staging.** `_nuke_video_paths`
   replaces `attn1`, `attn2`, `ff`, `audio_to_video_attn`, and
   `video_to_audio_attn` on all 48 blocks, then resets ModelPatcher size
   accounting. The permanent gates are `run_vx=False`,
   `a2v_cross_attn=False`, and `v2a_cross_attn=False`.
5. **Chunk generation has two phases.** All prompts are encoded in one TE
   session; all chunks then diffuse in one DiT session. There is no per-chunk
   `soft_empty_cache()` that would swap TE/DiT/VAE repeatedly.
6. **Tokenizer rebuilding is cached.** SentencePiece reconstruction is cached
   at `<gguf>.spiece_cache.bin`; verified 22.4 s first build to 0.5 s cached,
   byte-identical.

Real-checkpoint CPU smoke verification (`tools/smoke_scenema.py --skip-te`):

- all mapped transformer storage bytes remain INT8;
- loaded audio linears are Comfy `QuantizedTensor` objects;
- model detects as LTXAV with 48 layers;
- 240 unused block modules are removed;
- final ModelPatcher size is 10.96 GiB, down from the reported 40.05 GB staged;
- AudioVAE remains 16 kHz input / 48 kHz output with 251 latents per 10 s.

GPU sampling still needs validation on the target RTX 5090; the CPU test proves
the persistent representation and model-size accounting, not CUDA peak memory.

Other verified gotchas:

- **`GGMLTensor` must never alias an activation.** The subclass leaks out of
  any module that uses a bare `nn.Parameter` for a GGUF weight instead of a
  `GGMLLayer` — comfy's `text_encoders/llama.py` `RMSNorm` does, so with a GGUF
  Gemma every hidden state downstream of the first norm is a `GGMLTensor`.
  `GGMLTensor.clone()` returns `self` for packed quantized weights, so any
  comfy code that snapshots an activation with `.clone()` and then writes the
  buffer in place gets an alias. That is exactly what LTX-AV's text encoder
  does: it asks for `layer="all"` and comfy captures the 49 per-layer hidden
  states as `x.unsqueeze(1).clone()`. Measured with a GGUF Gemma-3 12B and
  ComfyUI `af3d2153`, all 48 post-embedding states collapsed to the final one
  (per-state std 1840.30 for every layer, max abs error 6.66e5 vs correct),
  which is heard as fluent speech with the wrong phonemes. `clone()` now copies
  for real unless the tensor is actually quantized.
- `AudioVAE.encode` wants `[B, C, T]` (mono auto-expands to stereo); decode
  returns `[B, T, C]`. The VAE wrapper `vae.decode` is safe (no latent-format
  scaling on audio); direct `fsm.encode` calls need
  `comfy.model_management.load_models_gpu([vae.patcher], force_full_load=…)`
  first (helper `_audio_vae_loaded`).
- Conditioning must carry `frame_rate=24` (FPS constant) or RoPE positions are
  wrong.
- `ref_latent` from comfy-core `EmptyLatentAudio` is `[B, C, samples]` — not a
  Scenema VAE latent; Generate validates shape (4-D, 8 ch, 16 freq bins).
- The workflow-side fix for the `cat 4-D vs 3-D` crash: Generate's `clip` input
  must come from Scenema Models Loader's `clip` output (Gemma + pipeline
  projection + connectors), **not** a plain CLIP Loader (GGUF); VAE from the
  Models Loader too (merges the reference encoder).

## Test infrastructure

`tools/smoke_scenema.py` — CPU dry-run (`--cpu` via comfy cli args, needs
`comfy.options.args_parsing = True` before importing comfy). Checks: DiT
detects as ltxav/48 layers; VAE is AudioVAE 16k→48k with 251 latents/10s;
optional Gemma GGUF → LTXAVTEModel with DualLinearProjection + tokenizer.
`tools/smoke_seedvc.py` — opt-in real-checkpoint SeedVC test. Loads every
component from `--extras` and runs a full end-to-end conversion (`--device
cuda|cpu`, `--load-only` to skip the conversion), asserting a finite,
same-length, audible result that is bit-identical across two seeded runs.
Verified on the RTX 5090 (fp16 autocast, peak 0.309) and on CPU float32
(peak 0.327), both reproducible with `seed=1234`.

A node-level test (registering folder_paths manually) lives at
`C:\Users\RISKYB~1\AppData\Local\Temp\opencode\node_test.py` — temp, not in repo.
Note: the dev box's Python env intermittently access-violates (0xC0000005) in
native code during model builds; rerun before assuming code is broken.

## LM Studio + LTX-2.3 talking-head port (2026-08-22)

`nodes_lmstudio.py` (new module) replaces the one cloud call in the
`ltxv23_talking_head` gallery workflow (a `GeminiNode` vision-prompt call)
with `LMStudioVisionPrompt`, hitting a local LM Studio server's
OpenAI-compatible `/v1/chat/completions`. Every other node in that reference
workflow's video-generation subgraph — `CheckpointLoaderSimple`,
`LoraLoaderModelOnly` ×2, `LTXVReferenceAudio`, `LTXVImgToVideoInplace`,
`LTXVConcatAVLatent`/`LTXVSeparateAVLatent`, `LTXVLatentUpsampler`,
`VAEDecodeTiled`, `LTXVAudioVAEDecode`, `CreateVideo`, samplers — is stock
comfy-core (`cnr_id: comfy-core` in the workflow JSON), not something to port.
Only two genuinely missing pieces got new nodes in `nodes_ltx23.py`:
`LTXV23RefineSampler` (base pass → split → `LTXVLatentUpsampler` on the video
branch only → rejoin → refine pass, mirroring core's exact wiring since
calling the upscale model on the joint AV tensor directly does not error, it
silently corrupts the audio branch) and `LTXV23AVDecode` (video+audio VAE
decode + `CreateVideo` mux in one node, one `fps` instead of two unwired
widgets that can silently drift).

**Not yet GPU-verified**: whether stock `LoraLoaderModelOnly` patches
correctly onto a GGUF-loaded, `GGMLTensor`-backed LTX-2.3 model from
`LTXV23ModelsLoader`. If it does not, that becomes a follow-up
`LTXV23LoraLoader` node rather than an assumption. `LTXV23RefineSampler`
and `LTXV23AVDecode` themselves also still need a real-checkpoint GPU run
(no CPU-mockable path, same as the rest of the LTX/Scenema stack) —
`nodes_lmstudio.py` is the only piece with a real test today
(`tools/smoke_lmstudio.py`, HTTP-mocked, no GPU needed).

**Correction, and the TTS stage got ported too**: the user explicitly
wants every pipeline stage ported into this repo, including ones already
covered by a working third-party node pack — do not treat "leave X as-is"
as settled just because an earlier multi-choice answer said so; confirm
if in doubt (see the `feedback_port_everything` memory in the user's
Claude memory store, not in this repo). `nodes_qwen_tts.py` ports the
`ltxv23_talking_head` workflow's `FB_Qwen3TTSCustomVoice` node
(`flybirdxx/ComfyUI-Qwen-TTS`) by wrapping the underlying `qwen-tts` pip
package's own `Qwen3TTSModel` directly — the model is a `transformers`
`PreTrainedModel` plus a separate codec/vocoder submodel, not this repo's
GGUF-quantization territory (verified via a deep read of
`qwen_tts/inference/qwen3_tts_model.py`), so there's nothing to port at
the weights level, only a comfy face. Real, verified-against-source
gotchas baked into the node:

  * the speech tokenizer/codec lives inside the same repo folder
    (`speech_tokenizer/` subdir), it is not a second download — confirmed
    on disk after a real pull (see below), not just from source reading;
  * `instruct` is real model-native conditioning on the 1.7B checkpoint,
    but the package silently drops it for 0.6B
    (`tts_model_size in "0b6"` — a **substring** check, not a whitelist;
    `_instruct_is_silently_dropped` in `nodes_qwen_tts.py` mirrors it
    exactly rather than guessing at the real `tts_model_size` values,
    which are unconfirmed);
  * the package's own sampling defaults (`top_k=50, top_p=1.0,
    temperature=0.9`) differ from the reference workflow's node
    (`top_k=20, top_p=0.8, temperature=1.0`) — this node's widget
    defaults match the workflow, not the package.

**Models-folder convention was rewritten mid-session** to match
`DarioFT/ComfyUI-Qwen3-TTS` (a full comfy node pack for the same
`qwen-tts` package, found at
`D:\Projects\ComfyUI\ComfyUI-Qwen3-TTS-main` after the first pass had
already invented its own `models/qwen_tts/` scheme) rather than diverging:
`models/Qwen3-TTS/<folder_name>/`, registered via
`folder_paths.add_model_folder_path`, a fixed 5-entry `repo_id` dropdown
(CustomVoice/VoiceDesign/Base × 1.7B/0.6B), auto-download via
`huggingface_hub.snapshot_download` (or ModelScope) with HF/ModelScope
cache migration checked first. The fixed 9-name `CUSTOM_VOICE_SPEAKERS`
dropdown and the `does not support generate_custom_voice` → plainer
error remap are also lifted from that pack; the sampling-parameter
surface (`top_p`/`top_k`/`temperature`/`repetition_penalty`) and the
0.6B `instruct`-drop warning are this repo's own additions, since
DarioFT's `Qwen3CustomVoice` node exposes neither. **Not ported**: that
pack's "FORCE SPEAKER MAPPING FIX" deep-injection hack (`nodes.py`
`Qwen3Loader`/`Qwen3CustomVoice`, config-object attribute surgery to
patch in a custom `spk_id` mapping) — clearly a defensive patch for some
specific checkpoint/config mismatch that isn't understood yet; if a
speaker-mapping error surfaces during real testing, look there before
inventing a fix from scratch.

Verified: `tools/smoke_qwen_tts.py` (mocked `folder_paths`/
`comfy.model_management`/`Qwen3TTSModel`, 14 checks — download-vs-reuse
decision, from_pretrained kwargs, local_model_path's speech_tokenizer/
validation, cache eviction, wrong-model-type remap, the instruct-drop
check). A real `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` pull (via
`huggingface-cli download`, `PYTHONUTF8=1` needed on Windows — the CLI's
own deprecation-warning emoji crashes on the default cp1252 console)
landed cleanly at
`N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice\`
(4.3 GB, `config.json` + `speech_tokenizer/` both present at the repo
root as expected) — confirms the folder layout end-to-end. **Update: the
user confirmed real audio generation works end-to-end** through
`QwenTTSModelsLoader` → `QwenTTSCustomVoiceGenerate` against these
weights inside ComfyUI itself — the loader's dropdown-list-installed-only
fix (below) landed after that first successful run.

**The `repo_id` dropdown went through one more fix**: it started as the
full fixed 5-entry `QWEN3_TTS_MODELS` list (matching DarioFT's pack
literally), but the user correctly pushed back — a fixed list of names
you can't tell apart from what's actually downloaded isn't "the
convention," it's confusing. `_installed_repo_ids()` now scans
`models/Qwen3-TTS/` and the dropdown shows only what's really on disk
(falling back to the full list only when nothing is installed yet, so a
fresh setup isn't stuck with an empty dropdown). Also rejected mid-fix: a
`✓`-prefix decoration on the full list — the user explicitly wanted a
plain list of installed models, not a fixed list with indicators. Lesson:
"follow this pack's convention" doesn't mean copy its UI choices
uncritically when they conflict with what the user is actually asking
for — confirm the specific complaint, don't just port harder.

## ID-LoRA prompt editing: four designs before the right one (2026-08-23)

Went through four designs for "let the user review/edit a captioner's
generated `[VISUAL]`/`[SPEECH]`/`[SOUNDS]` block before it's used" before
landing on the right one. Worth reading in order, because each version's
failure is what motivated the next:

1. **One node, `source` input, three `*_override` widgets auto-filled by
   a `web/*.js` extension "only if the widget is still empty."** This
   repo's first (and now removed) JS frontend code. Real bug, caught by
   the user testing it: updated on the first run, never again. Root cause
   wasn't the JS guard — it was structural: once a widget is auto-filled
   it's non-empty, and the Python side has no way to tell "leftover
   auto-fill from last run" apart from "a deliberate override" from
   widget content alone. No JS-only fix exists for that; the ambiguity is
   server-side.
2. **Same node, but split into 3 real override widgets + 3 SEPARATE
   read-only display widgets**, modeled on reading `ComfyUI-Custom-Scripts`'
   `ShowText` node (`web/js/showText.js` + `py/show_text.py`) at the
   user's request — `ShowText` is a pure display node that destroys and
   rebuilds its widget(s) from scratch every execution, no ambiguity
   because it has no override concept at all. Fixed the bug, but now 6
   text areas on one node. The user's reaction was the right call: "then
   you need to rethink this entire design" — doubling the widget count to
   patch around an ambiguity is a sign the ambiguity shouldn't exist in
   the first place.
3. **Tried an explicit `lock_edits` toggle instead of the empty-widget
   guess** (still one node, 3 widgets + 1 checkbox) — correct in theory
   (explicit state beats inferring intent from content), but confusing in
   practice: it read as "you need to flip a switch to make a text box
   editable," which was never true and wasn't what the toggle actually
   did (it controlled auto-refresh, not editability) — a framing failure,
   not a logic failure, but real feedback that the design still needed a
   concept the user had to be taught.

4. **Dropped the node entirely for 3 stock `RegexExtract` nodes + a dumb
   `LTXV23IDLoraAssembler` formatter**, on the theory that ComfyUI's
   native wire-vs-widget duality (disconnect a wire → the box becomes
   editable) already gives "auto-filled but overridable" for free. It
   does — but that's a multi-node setup, and the ask was explicitly for
   **one node** whose boxes you type into without wire-juggling. Rejected.

**Final design, `LTXV23IDLoraPromptEditor` — one node, three editable
boxes, decision made in Python.** Preceded by a real read of the frontend
source (`D:\Projects\ComfyUI\ComfyUI_frontend-main`, since
`upstream/ComfyUI` ships only the built `comfyui-frontend-package` and has
no `web/` source). What that established, and which any future
widget-behavior work should start from rather than re-deriving:

  * **Widget catalog** is `src/scripts/widgets.ts:222-242`; a STRING
    input's *entire* option surface is `multiline`, `dynamicPrompts`,
    `default`, `placeholder`, `forceInput`, `socketless`, `hidden`,
    `tooltip` (`packages/object-info-parser/src/schemas/nodeDefSchema.ts:28-88`,
    mirrored in `comfy_api/latest/_io.py:322-342` for V3). There is no
    custom widget type to author and V3 `io` adds nothing here.
  * **Socket+widget duality is automatic and always on**
    (`src/services/litegraphService.ts:271-327`); the old
    convert-widget-to-input is deprecated
    (`src/extensions/core/widgetInputs.ts:423-439`). A *wired* widget
    input greys out, blanks its drawn text and stops taking clicks
    (`LGraphNode.ts:3956-3961`, `BaseWidget.ts:250-252`), and the link
    always beats the widget value at serialization
    (`src/utils/executionUtil.ts:93-136`). Hence `source` is
    `forceInput: True` here — a visible box on a wired input is dead
    weight.
  * **Core has no auto-filled-and-editable widget.** Its only
    populate-from-own-execution widget is `TEXT_PREVIEW`
    (`src/extensions/core/textPreviewWidgets.ts:20-67`, behind
    `PreviewAny`/`SaveText`), and it is hard-coded `readonly` +
    `serialize: false` (`WidgetTextPreview.vue:42-45`). Confirmed absent,
    not overlooked — custom logic is mandatory for this behavior.
  * **Write-back is safe in `onExecuted`**: a multiline STRING is a DOM
    widget whose `.value` setter routes to `options.setValue`
    (`src/scripts/domWidget.ts:143-150,381-389`), updating both the real
    `<textarea>` and the reactive Pinia store
    (`useStringWidget.ts:26-43`). The "writes don't render" trap applies
    to writes made *before* the widget registers into that store
    (`BaseWidget.ts:146-157`) — i.e. too early in `onNodeCreated`.

The state that resolves attempt 1's ambiguity lives in Python:
`_LAST_SOURCE[unique_id]` (via the standard `UNIQUE_ID` hidden input)
remembers the `source` last parsed for that node. Source changed →
re-parse and overwrite the boxes; same source → keep them (this is where
an edit survives); nothing remembered yet (fresh ComfyUI start) → fill
only *empty* boxes, so edits saved in the workflow survive a restart. The
JS then writes the resolved values back **unconditionally** — correct
precisely because Python already decided, so it either echoes the user's
own edit or shows the fresh parse. Every earlier JS-side guess was wrong
for the same reason: widget content alone cannot distinguish "the user
typed this" from "we typed this last run."

Verified: `tools/smoke_id_lora_prompt_editor.py` (9 checks — parser
correctness incl. SPEECH-not-swallowing-SOUNDS, plus every state-machine
transition and per-node isolation) and a real-environment `INPUT_TYPES`
check against the portable's comfy. **The JS write-back has no offline
test** — it needs a browser (and a Ctrl+F5 hard refresh, since `web/` files
are cached).

### speech_text_batch + LTXV23SpeechBatchSelector (2026-08-24)

User reported `[SPEECH]` "losing all the line breaks" — traced both
`_parse_id_lora_prompt` and `assemble()` with an embedded-`\n` test and
found no stripping in either; turned out the actual ask was a 5th editor
output, `speech_text_batch` (one clip per non-blank line of `[SPEECH]`,
a blank line being a separator not an empty clip), plus a new node to
index into it. Once that shipped, the "losing line breaks" complaint was
resolved by definition — there was no separate bug to chase.

Real comfy list mechanics, not a delimited-string workaround: verified
against `upstream/ComfyUI/execution.py` directly rather than assumed.
`OUTPUT_IS_LIST` (`merge_result_data`, `:326-345`) is a **per-output**
tuple — a `True` slot's returned list is `extend`-ed straight into the
graph output, so `assemble()` just returns a Python list for that slot.
`INPUT_IS_LIST` (`get_input_data` `:159-190` +
`_async_map_node_over_list` `:241-265`) is a **class-level** bool on the
*consumer*: a linked input's cached upstream value is handed over
completely unwrapped (so `LTXV23SpeechBatchSelector`'s `batch` param
receives the real list, no extra layer), while every OTHER input
(including plain widgets like `index`) is still wrapped in a length-1
list regardless — hence `index[0]` to unwrap it. Getting this backwards
(assuming `batch` arrives double-wrapped, or that `index` doesn't need
unwrapping) is an easy, silent mistake; there is no error, just a
selector that always returns the same clip regardless of index.

`LTXV23SpeechBatchSelector.select()` clamps out-of-range `index`
(including negative, Python-style) to the nearest valid position with a
logged warning rather than raising — this is a UI convenience node for
iterating through clips, not a strict data-integrity boundary, so a
momentarily-out-of-range index while editing shouldn't hard-fail a queue.

## ScenemaModelLoader gained a keep_loaded cache (2026-08-24)

Had zero caching until now — every `load()` call rebuilt MODEL/CLIP/VAE
from raw state dicts regardless of whether the four filename dropdowns
were unchanged, unlike `nodes_ltx23.py`'s `_load_ltxv_clip`/
`_ENCODER_CACHE`, which exists for exactly this reason ("comfy re-runs
loader nodes on every prompt edit"). Same fix, same shape: a module-level
`_MODEL_CACHE` dict keyed on `(transformer_name, text_encoder_name,
pipeline_name, vae_encoder_name)`, single-slot (`.clear()` before storing
— holding more than one full DiT+TE+VAE stack isn't worth the RAM/VRAM),
gated by a new `keep_loaded` BOOLEAN widget (default `True`) rather than
always-on, since the user specifically asked for a toggle — an explicit
off switch for the case a file on disk was replaced without renaming it,
where the cache would otherwise serve stale weights under the same key.

Verified: real-environment `INPUT_TYPES()` check via the synthetic-package
import pattern (`cctech_gguf_pkg` rooted at the repo, importing this
repo's own `nodes.py` first so `unet_gguf`/`clip_gguf` folder keys are
registered before `ScenemaModelLoader.INPUT_TYPES()` calls
`folder_paths.get_filename_list("unet_gguf")` — needed because
`nodes_scenema.py` does `from .loader import ...` at module level, not
lazily inside a function like `nodes_ltx23.py` does, so it can't be
loaded via a bare `spec_from_file_location` the way the LTX-2.3/Qwen-TTS
checks in this file were). Not yet GPU-verified that the cached objects
survive a real generate → SeedVC's `unload_all_models()` → generate-again
cycle correctly (expected to, since that only pages weights off GPU, it
doesn't touch the Python objects this cache holds — but not confirmed
with real weights).

## SEARCH_ALIASES added to every node (2026-08-24)

User's real repro: searching "latent" in ComfyUI found none of our nodes.
Traced this to actual source, not guessed: comfy's plain-text node search
(`ComfyUI_frontend-main/src/services/nodeSearchService.ts:16-24`) only
indexes `['name', 'display_name', 'search_aliases']` via Fuse.js — it does
**not** search input/output type strings unless you invoke the separate
`o:`/`i:` type filter. `search_aliases` comes from a plain class attribute,
`SEARCH_ALIASES`, read generically off any node class regardless of V1/V3
API (`N:\...\ComfyUI\server.py:793`:
`info['search_aliases'] = getattr(obj_class, 'SEARCH_ALIASES', [])`) —
core ships this generously (`nodes.py` has ~35 `SEARCH_ALIASES` entries,
e.g. `KSampler`: `["sampler", "sample", "generate", "denoise", "diffuse",
"txt2img", "img2img"]`). **None of this repo's 33 nodes had it at all.**
Only 2 (`LTXV25EmptyLatentAVBatch`, `MiniMaxH3EmptyLatentAVBatch`) happen
to have "Latent" in their title, so a "latent" search only ever found
those two — every other LATENT-touching node (samplers, `LTXV23AVDecode`,
`ScenemaVAEEncode`, `ZImageImg2Img`, `LTXV23ImgToVideo`) was invisible to
it despite having perfectly correct `RETURN_TYPES`/`INPUT_TYPES`.

Added `SEARCH_ALIASES` to all 33 nodes, one line each right after `TITLE`,
matching core's own list style/length (3-8 short phrases, lowercase,
covering the node's actual verbs and the types it touches — not just
"latent" but "generate", "load model", "decode", etc. per node).
Verified via a real-environment check (not just compiling): 0/33 missing
the attribute after the fix, and a simulated "latent" query against
`name + display_name + search_aliases` (matching the frontend's exact
`nodeFuseSearch` key list) now returns 7 nodes instead of 2.

**Same gap existed in the sibling repos this session already touched**
(`D:\Projects\ComfyUI-Line-counter`, `D:\Projects\ComfyUI-Get-Random-File`)
— fixed there too, same convention, no `TITLE` attribute to anchor on in
those repos so `SEARCH_ALIASES` was inserted as the first class-level
line instead.

Before assuming a "why doesn't X show up" report is a code bug (or isn't
one), check `nodeSearchService.ts` for what the search UI actually
indexes — don't assume type strings are searchable text, and don't assume
metadata attributes require the V3 `io.ComfyNode` API when a V1 dict-style
class can carry them just as well via plain `getattr`.

## LTXV23IDLoraAssembler: the deleted node comes back, deliberately narrower (2026-08-24)

`LTXV23IDLoraAssembler` existed once already this session (attempt 4 of
the prompt-editor redesign - see "ID-LoRA prompt editing" above) and was
deleted when the design converged on `LTXV23IDLoraPromptEditor` doing
everything (parse + edit + reassemble) in one node. Brought back on
explicit request, but scoped correctly this time: `LTXV23IDLoraPromptEditor`
handles the "one raw source in, edited fields out" case; this node handles
"three already-separate strings in, one formatted string out" — e.g.
combining a clip picked via `LTXV23SpeechBatchSelector` with hand-typed
`visual`/`sounds` values, no source/parsing/state involved at all. They
are not redundant; the first parses, the second only formats. Verified
identical formatting output between the two (`assemble()` on both given
the same three fields produces byte-identical strings) via
`tools/smoke_id_lora_prompt_editor.py`, and a real-environment check
confirms 34 nodes total register cleanly, `SEARCH_ALIASES` present.

## Dev-box memory corruption (expanded 2026-08-19)

The instability above is broader than a plain access violation and can surface
*inside Python objects*, not just as a process crash. Observed from repeated runs
of one unchanged script in the portable env:

- `0xC0000409` STATUS_STACK_BUFFER_OVERRUN (fail-fast)
- `0xC0000005` access violation / bare segfault before any stdout
- `TypeError: 'cell' object is not subscriptable` raised inside torch's own
  `import` (`torch/_decomp/__init__.py`, a module-level dict had become a cell)
- `AttributeError: 'BasicAVTransformerBlock' object has no attribute '__dict__'`
  from `nn.Module.__getattr__` — i.e. an instance whose dict pointer was gone

That last one was reported as a loader bug (`_load_list` →
`check_module_offload_mem` → `get_key_weight` → `comfy.utils.get_attr`). It is
not: `BasicAVTransformerBlock` has no `__slots__`, and a freshly built LTX-2.5
model scans clean (0 modules missing `__dict__`, 0 dotted submodule names across
all 2382 loadable modules), with `_load_list` and a full `load_models_gpu` both
passing under the live server's exact config (no CLI flags). Bare `import torch`
is 6/6 clean and RAM is not exhausted (~108 GB free).

Rule of thumb: an error that is *structurally impossible* for the object
involved (missing `__dict__`, a dict that became a cell, a type that changed
identity) is corruption, not logic. Rerun; if it does not reproduce with a
deterministic replication of the same code path, stop hunting in the pack.

## Krea2 Control (2026-08-24)

Deep-dived `D:\Projects\ComfyUI\comfyui-krea2-controlnet-main` (no LICENSE
file; nodes.py + README.md only) before writing anything. Its own convention:
it ships **zero loader nodes** and assumes the base MODEL/VAE come from stock
comfy loaders — its 3 nodes (`Krea2ControlLoRALoader`, `Krea2ControlApply`,
`Krea2ControlImageEncode`) only patch a control-LoRA onto whatever MODEL you
hand them, via comfy's native `ModelPatcher.add_patches` +
`comfy.patcher_extension.PatcherInjection` + a `WrappersMP.DIFFUSION_MODEL`
wrapper — no monkey-patching of the diffusion model class itself.

Checked whether Krea2 needs the same from-scratch treatment `nodes_ltx23.py`
got and confirmed it does not: `comfy/supported_models.py` has a native
`Krea2(supported_models_base.BASE)` class (`unet_config.image_model ==
"krea2"`), `comfy/ldm/krea2/model.py` has the real DiT, `comfy/sd.py` has
`CLIPType.KREA2` wired to `comfy.text_encoders.krea2` (Qwen3-VL-4B), and
`nodes.py`'s core `CLIPLoader` already lists `"krea2"` in its type dropdown.
So unlike LTX-2.3 (comfy supports 0% of that pipeline: model, conditioning,
sampler, and AV decode/mux all had to be built from scratch), Krea2's model /
conditioning / sampler / decoder are ALL already correctly handled by stock
`comfy.sd.load_diffusion_model_state_dict`, `CLIPTextEncode`, `KSampler`, and
`VAEDecode`. Confirmed live: `UnetLoaderGGUF` in `nodes.py` already loads a
Krea2 GGUF checkpoint with zero changes (`comfy.sd.load_diffusion_model_state_dict`
auto-detects it); the only genuinely new thing is the Control LoRA mechanism,
which has no comfy-native or LTX-2.3 equivalent at all.

Landed on 3 nodes in `nodes_krea2.py`, after several rejected shapes mid-
conversation:

1. A combined loader that VAE-encoded the img2img source photo AND applied
   the control-LoRA data in one call — rejected: it silently assumed
   `control_image == source photo`, which is wrong (the control image is
   usually a *depth map generated from* the photo by a separate preprocessor,
   e.g. Depth Anything from `comfyui_controlnet_aux`), and it swallowed the
   reference pack's deliberate "Apply is a required, loud-failing step"
   safety design by merging it away.
2. Reverting to the reference pack's exact 4-node shape (Loader + LoRALoader +
   Encode + Apply, one-to-one) — technically correct but not this repo's
   convention: asked "is this what we did for zimage?" and re-read
   `nodes_zimage.py`. `ZImageImg2Img` already answers the "should
   encode+attach live inside the prep node?" question for this repo: yes —
   only the *loading* of control weights stays a separate node (comparable to
   Z-Image's `ModelPatchLoader`); encode-and-attach lives inside the img2img
   prep node itself, alongside the prompt and the init-image encode.

Final shape mirrors `ZImageImg2Img` directly:

- `Krea2ModelLoader` — MODEL/CLIP/VAE by name (mirrors `ZImageLoader`; no
  `keep_loaded` cache, since `ZImageLoader` has none and nothing here needs
  per-checkpoint surgery the way `ScenemaModelLoader` does).
- `Krea2ControlLoRALoader` — `model` in, LoRA-patched `model` out. Stays
  separate (unlike Z-Image's generic `ModelPatchLoader`) because the LoRA
  patching reads the *specific* model's live weight shapes to find the
  expanded `first` projection — it cannot be pre-loaded independent of a
  model. Ported essentially verbatim from the reference pack (widened
  `Krea2ControlInputProjection`, `LoRAAdapter` block patches via
  `add_patches`, `PatcherInjection` inject/eject swapping `diffusion_model.first`
  only for the duration of each forward call, restored in a `finally` so
  removing the node leaves the base model untouched) — this is
  correctness-critical low-level `ModelPatcher` plumbing, not something to
  rederive.
- `Krea2Img2Img` — `model, clip, vae, prompt, negative_prompt, strength,
  width, height` + optional `image` (source photo, img2img) + optional
  `control_image` (the depth/canny/etc. map) + prep knobs. VAE-encodes both,
  attaches the control latent to the model, CLIP-encodes the prompt, all in
  one call → `model, positive, negative, latent, denoise`, straight into a
  stock `KSampler`. Raises immediately if `control_image` is given with no
  Control LoRA loaded, or vice versa — same "never silently run a
  half-configured model" guarantee the reference pack's separate `Apply`
  node existed for, kept without a second required node. The empty-latent
  (txt2img) path deliberately does NOT hardcode Krea2's channel count or
  downscale ratio — it builds a plain 4-channel placeholder the same shape
  stock `EmptyLatentImage` always produces, and relies on
  `comfy.sample.fix_empty_latent_channels` (called from `common_ksampler`)
  to correct channels/temporal-dim for whatever model is attached, exactly
  like every other architecture already does through that same node.

No `Krea2KSampler`: unlike Z-Image (which needed one for a Turbo-specific
diffusers-vs-comfy denoise-schedule compatibility switch), nothing about
Krea2 sampling needed reimplementing, so stock `KSampler` is enough.

Verified against real weights on disk (not just import-checked): GGUF unet
`D:\models\image-models\krea2\split\diffusion_models\Krea2_turbo_uncensored_edit-Q6_K.gguf`
auto-detects as `Krea2` via `comfy.sd.load_diffusion_model_state_dict`; TE
`qwen3vl_4b_fp8_scaled.safetensors` loads as `Krea2TEModel_` under
`CLIPType.KREA2`; VAE `qwen_image_vae.safetensors` loads clean
(`latent_dim=3`); the real `depth-control-lora.safetensors` shape-matches an
expanded `first` projection (out=6144, image=64, control=64) and yields 224
compatible block LoRA patches, all accepted by `add_patches`. `tools/smoke_krea2.py`
covers the pure tensor-prep helpers, the `Krea2ControlInputProjection` forward
math (both the image-only-fallback and image+control-summation paths — the
latter is what proves an ordinary LoRA on the base `first` layer keeps
working under the control patch), and `Krea2Img2Img`'s guard rails/latent
shapes offline (17/17 passing, no GPU).

## Krea2 Depth Map: porting a full model architecture, not just a wrapper node (2026-08-24)

User asked "is that what the other repo does" re: auto-downloading a depth
model - checked `comfyui-krea2-controlnet-main`'s own README, which says
outright "`Krea2 Control Image Encode` is generic and does not run depth,
canny, pose, or other preprocessors" and points at
`Fannovel16/comfyui_controlnet_aux` instead. So a depth node is not a port
of anything in that pack - it's new code, using whichever depth library.
Per this repo's standing "port everything, no exceptions" rule, deep-dived
`comfyui_controlnet_aux-main` (`node_wrappers/depth_anything_v2.py` +
`src/custom_controlnet_aux/depth_anything_v2/`) to see what porting its
actual Depth Anything V2 node requires, rather than defaulting to
`transformers.AutoModelForDepthEstimation` (also already installed, and
functionally equivalent) as a shortcut.

Finding: the node wrapper is thin, but the detector underneath is a full
from-scratch DINOv2 ViT encoder + DPT decoder reimplementation - NOT a call
into `transformers` or any other library - spread across 9 files
(`dinov2.py`, `dinov2_layers/{attention,block,drop_path,layer_scale,mlp,
patch_embed,swiglu_ffn}.py`, `dpt.py`, `util/{blocks,transform}.py`),
~1,200-1,500 lines. Both its deps (`cv2`, `einops`) are already installed
in the ComfyUI env, so either approach needs zero new pip installs - the
real trade-off was code volume/maintenance, not dependencies. Presented
both options; user chose the full port ("it will be an easy port. go ahead
and port it").

Ported into one flat file, `depth_anything_v2.py`, matching this repo's
existing flat-architecture-file convention (`melband_arch.py`,
`seedvc_arch.py`) instead of the source's nested `dinov2_layers/`
subpackage. Apache-2.0 in both repos, so no license conflict (checked
`comfyui_controlnet_aux-main/LICENSE.txt` before porting anything).

Deliberately trimmed dead-for-inference code paths, none of which change
the module hierarchy or state_dict keys a checkpoint loads into:
- `NestedTensorBlock` dropped in favor of plain `Block` - for a single
  Tensor input (never a list, since this only ever runs one image through
  eval-mode inference) `NestedTensorBlock.forward` dispatches straight
  through to `Block.forward`, so they're behavior-identical here. This
  also drops the entire xformers nested-tensor batch-grouping machinery
  (`drop_add_residual_stochastic_depth_list`, `get_attn_bias_and_cat`,
  `attn_bias_cache`, `add_residual`, `get_branges_scales`) - all only
  reachable from list-of-tensors input.
- `BlockChunk`/`_get_intermediate_layers_chunked`/`forward_features_list`
  dropped - the source's own `DINOv2()` factory always calls with
  `block_chunks=0`, so the chunked path is dead code for every checkpoint
  this loads, confirmed by reading the factory before cutting it.
- Stochastic-depth training branches in `Block.forward` dropped - only
  reachable when `self.training` is True; this pack only ever calls
  `.eval()` models.
- `ConvBlock` in `dpt.py` dropped - defined in the source but never
  referenced anywhere else in that file.

Weight download reuses the exact convention `nodes_qwen_tts.py` already
established: `os.path.join(folder_paths.models_dir, "depth_anything_v2")`,
plain local folder (no symlink cache tricks, unlike the source pack's own
`custom_hf_download` which supports a legacy `AUX_ANNOTATOR_CKPTS_PATH`
config-file system this repo has no equivalent of and doesn't need).

Verified by download+load, not just import-check: `_download_checkpoint`
pulled the real `depth_anything_v2_vits.pth` from
`depth-anything/Depth-Anything-V2-Small` on HuggingFace, and
`model.load_state_dict(sd, strict=True)` succeeded with zero missing and
zero unexpected keys - the strongest signal the consolidated module
hierarchy exactly matches the original 9-file version's, since any
renamed/reordered/dropped-by-mistake submodule would show up as a key
mismatch here. End-to-end inference on a synthetic 256x320 image produced
a correctly-shaped, full-range (0-255) depth map with no shape errors.

New node `Krea2DepthMap` (`nodes_krea2.py`): `image` in, `ckpt_name` +
`resolution` widgets, `IMAGE` (the depth map) out. Slots directly into
`Krea2Img2Img`'s `control_image` input, closing the "I don't have any
preprocessor installed" gap without requiring
`comfyui_controlnet_aux` or any other external pack.

## Krea2Img2Img's two guard rails were wrongly made symmetric (2026-08-24)

User hit `ValueError: control_image was given, but model has no Krea2
Control LoRA loaded` from a real workflow and asked why it wasn't just
ignored. Right call - the two directions of the original guard rail were
NOT actually equal-risk, and treating them as a symmetric pair was a
mistake:

- Control LoRA loaded, no `control_image` -> genuinely dangerous. The
  model has a wrapper installed (from `Krea2ControlLoRALoader`) that
  expects a control latent in `transformer_options` at sample time; if it's
  missing, the forward pass fails (or worse) deep inside sampling instead
  of at graph-build time. Correctly raises immediately - kept as-is.
- `control_image` given, no Control LoRA loaded -> NOT dangerous. There is
  no widened input projection installed at all, so there's nothing to
  attach the control latent to and no half-configured model state to worry
  about. Raising here just forces users to physically disconnect
  `control_image` every time they want to toggle the LoRA loader off,
  which is exactly the "not easy" friction this whole feature was built to
  avoid.

Fixed: this direction now logs a warning and sets `control_image = None`
instead of raising, so a preprocessor chain (e.g. `Krea2DepthMap`) can stay
wired into the graph permanently while the Control LoRA loader is toggled
on/off. `tools/smoke_krea2.py`'s
`test_img2img_rejects_control_image_without_loaded_lora` renamed to
`test_img2img_ignores_control_image_without_loaded_lora` and updated to
assert the call succeeds instead of raising (17/17 still passing).

## Krea2Img2Img: control_mode, folding depth derivation into the img2img node (2026-08-24)

User's own summary of the confusion, verbatim: "that wasnt exactly my
quatsion, i didnt sat 2 nodes, i said 2 slots" - i.e. even with the
mechanics explained, wiring the same source photo into two different input
slots (`Krea2DepthMap.image` for structure, `Krea2Img2Img.image` for the
img2img starting latent) to do one thing (depth-guided generation from one
photo) was genuinely bad UX, not just under-explained.

Then, a direct and correct challenge: "because you didnt display some type
of selector or something, how do you know what type of lora is loaded? or
what type of control image is selected?" Answer is honest, not a design
flaw to patch around: nothing in a `.safetensors` LoRA file says whether
it's a depth/canny/pose/etc. LoRA - there is no metadata field for it, so
the pack genuinely cannot detect it. The fix isn't detection (impossible);
it's an explicit selector for the one decision this pack *can* make on its
own (derive a depth map automatically) vs. everything else (which needs a
human to say what it is).

Added `control_mode` (`["auto_depth", "manual"]`, default `auto_depth`) to
`Krea2Img2Img`:
- `auto_depth` - if a Control LoRA is loaded and `control_image` isn't
  explicitly connected, runs the same `depth_anything_v2.DepthAnythingV2Detector`
  `Krea2DepthMap` uses, internally, on `image` - so the depth-LoRA case
  (the one this pack can fully support in software) collapses to one photo,
  one slot, no second node.
- `manual` - no automatic derivation at all; `control_image` must be
  supplied by hand, for canny/pose/lineart/normal or any other Control LoRA
  type this pack has no automatic preprocessor for.
- An explicitly-connected `control_image` always overrides `auto_depth`
  in either mode - a hand-picked depth map, or `Krea2DepthMap`'s own
  output routed in manually, still works exactly as before.

`Krea2DepthMap` stays as a standalone node (not removed) for anyone who
wants to inspect the depth map directly, feed it into something else, or
build the graph by hand instead of relying on auto_depth.

`depth_ckpt_name` (same combo as `Krea2DepthMap.ckpt_name`) added
alongside `control_mode` so auto_depth's model size is configurable without
needing the standalone node.

Added 3 tests to `tools/smoke_krea2.py`
(`test_img2img_manual_mode_requires_control_image`,
`test_img2img_auto_depth_derives_control_image_from_image`,
`test_img2img_explicit_control_image_overrides_auto_depth`), the latter two
monkeypatching `krea2.depth_anything_v2.DepthAnythingV2Detector` with a
fake (real weights/inference not appropriate for an offline smoke test -
covered separately against real weights, same as the rest of this feature).
`_FakeModelPatcher` gained a `.model` attribute and a `get_model_object`
that raises, since these tests now reach `_process_control_latent_for_model`
instead of stopping at the earlier guard-rail checks. 21/21 passing.

## Qwen-Image ControlNet: a dispatcher, not a port (2026-08-24)

User referenced a blog post (stablediffusiontutorials.com, Qwen-Image
ControlNets) and asked "is this a clone of krea2 or is more involved."
Answer turned out to be neither - it's LESS involved than Krea2, because
unlike Krea2's control-LoRA (a genuinely missing mechanism, nothing in
comfy or any pack implements the widened-input-projection trick), every
Qwen-Image ControlNet format already has full native comfy support:

- InstantX/Union and Qwen-Image-Fun checkpoints: real diffusers-style
  ControlNet weights, auto-detected and built by
  `comfy.controlnet.load_controlnet_state_dict()` (checked
  `comfy/controlnet.py:661-722`) into an actual `ControlNet` object - the
  same one stock `ControlNetLoader`/`ControlNetApplyAdvanced` produce and
  consume. Attaches to CONDITIONING.
- DiffSynth canny/depth/inpaint patches and the Fun ControlNet variant
  loaded as a raw patch: auto-detected and built by
  `comfy_extras/nodes_model_patch.py`'s `ModelPatchLoader` (key signature
  `'controlnet_blocks.0.y_rms.weight' in sd` for
  `QwenImageBlockWiseControlNet`) into a `MODEL_PATCH`, applied via
  `DiffSynthCnetPatch` - the exact mechanism `nodes_zimage.py`'s
  `ZImageFunControlnet` already reuses in this repo (it literally
  subclasses core's `QwenImageDiffsynthControlnet` apply node).

So `QwenImageControlNetLoader` doesn't reimplement either detection or
either model class - it calls `comfy.controlnet.load_controlnet_state_dict()`
directly for the InstantX/Union branch, and instantiates core's own
`ModelPatchLoader` node class directly for the DiffSynth/Fun branch (same
"reuse the stock class, don't reimplement its dispatch logic" instinct as
`nodes_zimage.py`'s `_te_name`/`ZImageLoader`). The only new code is the
`QwenImageControl` wrapper tagging which of the two attachment points
(`"controlnet"` vs `"model_patch"`) a loaded checkpoint needs, and
`QwenImageImg2Img` routing to the right one - `_apply_controlnet_to_conditioning`
replicates core's `ControlNetApplyAdvanced.apply_controlnet()` inline
(`control_net.copy().set_cond_hint(...).set_previous_controlnet(...)`,
stamped into each conditioning item's `dict['control']`) for the
conditioning-attachment path, and `DiffSynthCnetPatch` (imported directly
from `comfy_extras.nodes_model_patch`, not reimplemented) for the
model-attachment path.

User provided 5 real files, all placed under
`N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\model_patches\`:
`qwen_image_{canny,depth,inpaint}_diffsynth_controlnet.safetensors`,
`qwen_image-InstantX-ControlNet-Union.safetensors`, and
`qwen_image_lotus-depth-d-v1-1.safetensors`. Verified format detection
against all 4 controlnet/patch files (not the 5th) end-to-end through the
real node class, not just key-sniffing: the 3 DiffSynth files each loaded
as `MODEL_PATCH` wrapping a real `QwenImageBlockWiseControlNet`, the
InstantX file loaded as a `ControlNet` wrapping a real
`QwenImageControlNetModel`.

`qwen_image_lotus-depth-d-v1-1.safetensors` is NOT a ControlNet at all -
Lotus is a diffusion-based depth *estimator* (a preprocessor, like Depth
Anything V2), architecturally unrelated to either mechanism above. Left
unported for now; `Krea2 Depth Map`'s Depth Anything V2 already covers the
"turn a photo into a depth map" role generically (nothing Krea2-specific
about it - it just estimates depth), so Qwen-Image workflows can reuse it
rather than needing a second, different depth-estimator port. Flagged in
README as explicitly out of scope, not silently dropped.

New file `nodes_qwen_image.py`: `QwenImageModelLoader`,
`QwenImageControlNetLoader`, `QwenImageImg2Img` - same 3-node shape as
Krea2's (loader / control loader / one-node img2img+control prep).
`tools/smoke_qwen_image.py` (8 tests, no GPU) covers the dispatch helper
and both attachment paths via fakes (`_FakeControlNet`,
`_FakeDiffSynthCnetPatch` standing in for the real `comfy_extras` class,
which needs real model weights to construct) - real-weight verification
done separately via a real-environment check script against the actual
portable install.

## Fix: Qwen-Image-Fun ControlNet mis-routed to ModelPatchLoader (2026-08-25)

User tried a real file, `Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors`,
and hit `UnboundLocalError: cannot access local variable 'model'` inside
core's own `ModelPatchLoader.load_model_patch()`. Root cause: my own
`is_diffsynth_or_fun` detection in `QwenImageControlNetLoader.load()` was
wrong, not a missing-format gap. I'd copied the key signature
`comfy/controlnet.py:801` checks for Qwen-Image-Fun
(`'control_blocks.0.after_proj.weight' in sd and 'control_img_in.weight' in sd`)
into the *MODEL_PATCH* branch, but that signature is actually the trigger
for `comfy.controlnet.py`'s own `load_controlnet_qwen_fun()`, which builds a
real `QwenFunControlNet` (a `ControlNet` subclass) - a CONDITIONING
attachment, same family as InstantX/Union, not a MODEL_PATCH at all. "Fun"
sounds like it should be a DiffSynth-style patch (Z-Image's Fun ControlNet
IS one), but Qwen-Image's own Fun ControlNet isn't - the name is misleading
across the two families.

Fixed: `is_diffsynth_or_fun` renamed `is_diffsynth_patch`, checking ONLY
`'controlnet_blocks.0.y_rms.weight' in sd` (the real DiffSynth block-wise
signature). Everything else - including the Fun signature - falls through
to `comfy.controlnet.load_controlnet_state_dict()`, which already has its
own correct internal dispatch for InstantX/Union/Fun. Updated every
docstring/comment/README passage that claimed Fun attaches to MODEL.

Verified against the real file after the fix: loads as
`kind="controlnet"`, payload `QwenFunControlNet` wrapping a real
`QwenImageFunControlNetModel` - confirmed via the actual node class, not
just re-reading the key signature. Re-ran the earlier 4-file real-weights
check too (3 DiffSynth patches + InstantX/Union) to confirm no regression
- all still classify the same as before.

## QwenImageImg2Img: control_mode, matching Krea2Img2Img's auto-derive convention (2026-08-25)

User: "its not auto using my image as control image" - fair, since
Krea2Img2Img auto-derives depth by default. Difference: Krea2 has exactly
one control type, so a single `auto_depth` default was safe. Qwen-Image
checkpoints span several distinct preprocessing needs (canny, depth,
inpaint, and Union which can be any of those) that this pack has no way to
detect from the loaded file - so `QwenImageImg2Img` now offers the same
convenience Krea2 has, but defaults to `manual` rather than guessing:

- `control_mode="manual"` (default) - unchanged, connect `control_image`.
- `control_mode="auto_canny"` - new `_auto_canny_control_image()` helper,
  plain `cv2.Canny` on `image` - no model, no download, matches
  `comfyui_controlnet_aux`'s own Canny preprocessor defaults (100/200
  thresholds). Zero new deps (cv2 already used by depth_anything_v2.py).
- `control_mode="auto_depth"` - reuses `depth_anything_v2.DepthAnythingV2Detector`
  (the same Depth Anything V2 port `Krea2DepthMap` uses), added
  `depth_ckpt_name` widget alongside it.

Also fixed a real gap surfaced while reading `QwenImageDiffsynthControlnet`'s
signature again for this: `mask` was never exposed on `QwenImageImg2Img` at
all, silently making inpaint checkpoints unusable for masked inpainting (a
supplied `image` was the only thing reaching `DiffSynthCnetPatch`). Added
`mask` (MASK, optional), replicating stock `QwenImageDiffsynthControlnet`'s
own mask preprocessing exactly (`ndim==3 -> unsqueeze(1)`,
`ndim==4 -> unsqueeze(2)`, then `1.0 - mask`) before passing to
`DiffSynthCnetPatch`.

Almost added a `mask`-without-`control_image` shortcut for inpaint-only
use (assuming mask alone could substitute for the image), but re-checked
`comfy_extras/nodes_model_patch.py`'s `QwenImageDiffsynthControlnet.INPUT_TYPES()`
before shipping it: `image` is `"required"` even there, `mask` is only
`"optional"` - it refines the inpaint region, it doesn't replace the base
image. Removed that branch before it became a real bug (would have crashed
inside `DiffSynthCnetPatch.encode_latent_cond()`'s unconditional
`self.vae.encode(image)` call with `image=None`).

`tools/smoke_qwen_image.py`'s stub `folder_paths` gained `models_dir` and
`comfy.model_management` gained `get_torch_device` (needed by
`depth_anything_v2.py`'s module-level constant and the auto_depth path
respectively); the plain `sys.path.insert` + `import nodes_qwen_image`
import was replaced with the same synthetic-package trick already used in
`tools/smoke_krea2.py`, since `from . import depth_anything_v2` is a
relative import that only resolves inside a real package context.

## Qwen-Image: standalone Canny node + edit_reference (reference_latents) (2026-08-25)

Two real gaps, both surfaced by the user while live-testing a real
workflow (`Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors` +
`qwen-image-edit-Lightning-4steps` LoRA +
`Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors`, prompt "make her
dress blue and give her a halo") - neither a guess, both confirmed against
real comfy source before shipping. Went through plan mode for this one
since the scope grew mid-investigation (started as "add a canny node,"
became "also wire up Edit").

**Gap 1 - depth had a standalone node, canny didn't.** User caught this
directly: `Krea2DepthMap` is a standalone, previewable node reusable for
any model (already documented as such), but canny preprocessing
(`_auto_canny_control_image`) only existed inline inside
`QwenImageImg2Img`'s `control_mode="auto_canny"` branch - no way to see or
reuse it standalone. Fixed: added `QwenImageCanny`, mirroring
`Krea2DepthMap`'s exact shape, calling the same existing helper (no
duplication - `QwenImageImg2Img`'s `auto_canny` branch still calls
`_auto_canny_control_image` directly too, same "independent calls to
shared logic" pattern `Krea2DepthMap`/`Krea2Img2Img.auto_depth` already
both use).

**Gap 2 - Qwen-Image-Edit's real edit mechanism was never wired up.** User
asked directly: "did you say you did not wire up edit in the diffusion
loop? do we need a custom sampler again?" Answered by actually reading
`comfy/ldm/qwen_image/model.py`'s DiT `_forward()` (not guessing): it takes
a distinct `ref_latents` parameter (line 442), genuinely separate from
both the img2img latent (`x`) and ControlNet's `control` parameter - when
present, reference-image tokens get concatenated onto the sequence before
the block loop runs. `QwenImageImg2Img` never supplied this at all, so an
Edit checkpoint (fine-tuned specifically for editing) was running as a
plain generator - explaining "not garbage, but not working like a proper
canny model" even after the canny-wiring fix from the previous session
entry.

Confirmed the fix is conditioning, not a sampler change - read
`comfy_extras/nodes_edit_model.py`'s stock `ReferenceLatent` node in full:
it just VAE-encodes a photo into a `LATENT` and calls
`node_helpers.conditioning_set_values(conditioning, {"reference_latents":
[latent["samples"]]}, append=True)`. Comfy's generic `extra_conds`/
`apply_model` machinery already reads `reference_latents` out of
conditioning and threads it into the real `ref_latents` forward parameter -
`KSampler` stays completely architecture-agnostic, same as everywhere else
in this pack. Added `edit_reference` (optional IMAGE) to
`QwenImageImg2Img`, doing exactly that, applied to `positive` conditioning
only (matches `ReferenceLatent`'s typical single-conditioning-stream
usage - edit intent is a positive-conditioning concept). Added
`import node_helpers` - same module, same call pattern `nodes_ltx23.py`
already uses for `frame_rate`.

Kept `edit_reference` a separate input from `image` (img2img starting
latent) and `control_image` (structural guidance) rather than overloading
one of them - the three answer genuinely different questions and a real
Qwen-Image-Edit + ControlNet workflow could plausibly want all three
connected to different things (or the same photo) simultaneously.

`tools/smoke_qwen_image.py`: added `test_canny_node_produces_edge_map_matching_input_shape`
and `test_img2img_edit_reference_attaches_reference_latents_to_positive_only`.
The latter needed stubbing `node_helpers` in the offline harness - checked
its real source first (`node_helpers.py` imports `from comfy.cli_args
import args` at module level, too heavy to import directly offline) and
stubbed a faithful 1:1 copy of the real 15-line `conditioning_set_values`
rather than a simplified fake, so the test exercises the actual logic, not
a stand-in for it. 10/10 passing. Real-environment check confirmed
`QwenImageCanny` registers correctly and `QwenImageImg2Img`'s optional
list now includes `edit_reference`; total registered node count went
41 -> 42 (one new node class).

Left for the user to confirm since it needs real GPU inference (not
possible in this dev environment): whether Edit checkpoint output now
actually preserves the source photo's identity while following the
prompt's edit instruction, with the canny ControlNet's structural guidance
applied on top.

## Krea2: auto_canny + edit_reference, and two unrelated LoRA families (2026-08-25)

Following the Qwen-Image auto_canny/edit_reference work, extended
`Krea2Img2Img` symmetrically since the user has a real canny Krea2 LoRA
(`krea2_canny-v0.1.safetensors`) alongside the depth one.

Added `auto_canny` to `control_mode` (mirrors `nodes_qwen_image.py`'s
helper exactly - `_auto_canny_control_image` duplicated into
`nodes_krea2.py` rather than shared, matching this repo's established
"independent per-file helpers" convention). Straightforward - no surprises.

**Real surprise**, caught by testing against the actual file instead of
trusting the mirrored pattern: loading `krea2_canny-v0.1.safetensors`
through `Krea2ControlLoRALoader` failed with `Could not find expanded
Krea2 first projection weight`. Inspected its actual tensor keys before
assuming a bug: 588 keys, all plain `lora_down.weight`/`lora_up.weight`/
`alpha` triples on `attn.wq/wk/wv/wo` and `mlp.gate/up/down`, rank 32 - NO
`first`/`img_in` key at all. This is architecturally a completely
different, ordinary LoRA, not a widened-projection Control LoRA.

User pointed at its actual HuggingFace page (nynxz/NK2E) to settle it: "a
separate control LoRA (ControlNet-style, **in-context**)... Feed a canny
edge map as the reference instead of a source image: structure comes from
the edges, content from the text prompt... uses the same node setup as
editing." That's the Qwen-Image-Edit pattern, not the widened-projection
one. Confirmed against real source, not just the model card: read
`comfy/ldm/krea2/model.py`'s DiT `_forward()` (line 295) - it has the
exact same `ref_latents` parameter Qwen-Image's DiT does. Krea2 has the
same two-mechanism split Qwen-Image does; this repo just hadn't hit the
second one for Krea2 yet.

Added `edit_reference` to `Krea2Img2Img`, identical implementation to
`QwenImageImg2Img`'s (VAE-encode, `node_helpers.conditioning_set_values(
positive, {"reference_latents": [latent]}, append=True)`, positive only).
Added `import node_helpers` to `nodes_krea2.py` too.

Corrected usage for `krea2_canny-v0.1.safetensors`: load it with stock
`LoraLoaderModelOnly` (NOT `Krea2ControlLoRALoader` - that node correctly
rejects it, this isn't a bug to route around), feed the canny map into
`Krea2Img2Img`'s new `edit_reference` input instead of `control_image`.

Verified against real weights, not just structural code review:
- `krea2_canny-v0.1.safetensors` applies cleanly via
  `comfy.sd.load_lora_for_models(model, None, lora_sd, 1.0, 0.0)` (the
  same call `LoraLoaderModelOnly` makes internally) - confirms it's a
  normal, compatible LoRA once routed to the right loader.
- `ref_latents` confirmed present via
  `inspect.signature(model.model.diffusion_model._forward)` on the
  actual real, loaded Krea2 model instance, not just by reading the
  source file - rules out the parameter being dead/unused on this
  checkpoint's actual code path.

`tools/smoke_krea2.py` gained `test_img2img_auto_canny_derives_control_image_from_image`,
`test_auto_canny_control_image_produces_edge_map_matching_input_shape`
(cv2.Canny is real/deterministic, no fake needed unlike depth), and
`test_img2img_edit_reference_attaches_reference_latents_to_positive_only`
(needed the same `node_helpers` stub added to `smoke_qwen_image.py` -
copied verbatim, same reasoning). `_clip()`-shaped fixture for this test
needed a real conditioning-list return (`[[tensor, {}]]`), unlike the
`"cond"`-string fixture most of this file's other tests use, since
`node_helpers.conditioning_set_values` actually iterates the list. 24/24
passing. Real-environment check confirms `edit_reference` registers;
total node count unchanged at 42 (both additions were on existing node
classes, no new node this time).

## Krea2ControlLoRALoader: auto-dispatch instead of two loader nodes (2026-08-25)

User pushed back on the "use a different loader node for the canny LoRA"
guidance from the previous entry: real error report
(`RuntimeError: Could not find expanded Krea2 first projection weight`),
followed by "so lets backup, youre telling me our controlnet lora loader
doesnt work with this type of controlnet? that actually dos sound like you
need to patch the loader properly." Fair, and it's exactly the same shape
of problem `QwenImageControlNetLoader` already solved for Qwen-Image's two
ControlNet formats (auto-detect from file content, one node, no manual
"which loader do I need" decision) - `Krea2ControlLoRALoader` should do the
same instead of making the user pick between it and stock
`LoraLoaderModelOnly` by hand.

Extracted the existing shape-matching logic (previously buried inline
inside `_make_control_projection`, which raised immediately if not found)
into a standalone, non-raising detector, `_lora_expanded_first_weight_key`.
`Krea2ControlLoRALoader.load_lora` now calls it first: found -> existing
widened-projection path unchanged (still calls `_make_control_projection`,
which still raises on failure - that raise is now unreachable in practice
since `load_lora` never reaches it without a positive detection first, but
left in place since `_make_control_projection` is still called elsewhere
implicitly). Not found -> applies the file as an ordinary LoRA via
`comfy.sd.load_lora_for_models(model, None, state_dict, strength, 0.0)` -
the exact call stock `LoraLoaderModelOnly` makes internally (verified this
call works against the real canny file two sessions-turns ago) - no
wrapper, no injection, since a plain LoRA patch needs neither.

This means `Krea2Img2Img`'s existing `has_control_lora` check
(`model.get_attachment(WRAPPER_KEY) is not None`) now does exactly the
right thing automatically for both cases with zero changes to that node:
the plain-LoRA path never sets `WRAPPER_KEY`, so `control_image`/
`control_mode` correctly stay inert (ignored-with-warning, per the earlier
guard rail) and `edit_reference` (which never depended on `WRAPPER_KEY` at
all) keeps working exactly as it already did.

Verified against both real files through the SAME loader instance/class,
not just re-checking each separately: depth LoRA -> `WRAPPER_KEY` attached
(confirmed True), canny LoRA -> `WRAPPER_KEY` absent (confirmed False), no
error either way. `tools/smoke_krea2.py` gained
`test_lora_expanded_first_weight_key_detects_widened_projection` and
`test_lora_expanded_first_weight_key_returns_none_for_ordinary_lora`
(26/26 total). Docs updated to drop the "load canny with stock
LoraLoaderModelOnly instead" guidance from the previous entry - it's no
longer necessary, `Krea2ControlLoRALoader` handles it now.

## QwenImageKSampler + Krea2KSampler: measured, not assumed (2026-08-25)

User asked directly why Z-Image has `ZImageKSampler` but Qwen-Image/Krea2
don't, and pushed to build them regardless of whether I could "measure"
anything first ("i dont think you need to measure anything... thats
something youll have to find out in the code"). Measured anyway, via two
parallel research agents, because building a sampler without knowing
whether the underlying premise (a real schedule divergence) holds would
risk shipping something that doesn't address anything real - same
standard as every other feature this session.

**Confirmed real for both, same mechanism, smaller than Z-Image's but
measurable - and critically, NOT a Qwen-Image-specific or Krea2-specific
bug.** It's generic to `comfy.samplers.KSampler.set_steps()`
(`comfy/samplers.py:1431-1440`) for ANY `denoise<1.0` sampling of ANY
flow-matching model: comfy re-expands to `new_steps=int(steps/denoise)`
and takes the tail, instead of computing the schedule at `steps` directly
and slicing from `t_start = steps - round(steps*denoise)` (the diffusers
img2img convention `ZImageKSampler`'s `diffusers_mode` already
reproduces). Both Qwen-Image and Krea2 share `ModelSamplingFlux` with
`shift=1.15` (literally the same value, copy-pasted alongside the
Qwen-Image-family config in `comfy/supported_models.py` - not
independently calibrated for either). Recomputed at 9 steps/denoise=0.9,
first via research-agent hand-calculation, then verified against a REAL
loaded Krea2 model's actual `model_sampling` object
(`model.get_model_object("model_sampling")` → real
`comfy.samplers.calculate_sigmas` calls, not simulated): comfy convention
0.9660138..., diffusers-style 0.9619314..., delta 0.00408 - matches the
hand-computed research numbers exactly. Smaller than Z-Image's documented
gap (0.9643 vs 0.9567, Δ≈0.0076) because `ModelSamplingFlux`'s shift curve
is less steep near σ≈1 than Z-Image's `ModelSamplingDiscreteFlow`, same
underlying cause either way.

Honest calibration, not hedging on whether to ship: Z-Image's own
docstring calls this "a compatibility switch for matching another
pipeline, not a quality setting" - a ~0.4% sigma difference is real but
small, unlikely to be the full explanation for any "not working properly"
report on its own (more likely cause: the wiring issues already found and
fixed - raw photo as control_image, missing edit_reference). Built anyway
since the gap is confirmed real and it's zero-risk (default
`denoise_mode="comfy"` is byte-identical to current behavior).

Added `QwenImageKSampler` (`nodes_qwen_image.py`) and `Krea2KSampler`
(`nodes_krea2.py`) as near-verbatim clones of `ZImageKSampler`
(`nodes_zimage.py:238-321`) - the `sample()` body is copied unchanged
since it was already fully generic (only calls
`model.get_model_object("model_sampling")` + `comfy.samplers`/
`comfy.sample` primitives, confirmed to need zero Z-Image-specific math).
Only `TITLE`/`CATEGORY`/`SEARCH_ALIASES`/docstring differ per file, plus
Qwen-Image/Krea2-appropriate `steps`/`cfg` defaults (20/2.5, vs Z-Image
Turbo's 9/1.0 - these aren't Turbo-distilled checkpoints).

`tools/smoke_qwen_image.py` and `tools/smoke_krea2.py` each gained 3 tests
(comfy-mode delegates to `common_ksampler` unchanged, diffusers-mode
rejects `denoise<=0`, diffusers-mode slices sigmas and returns a correctly
shaped latent) - needed new fakes for `comfy.samplers`/`comfy.sample`/
`latent_preview`/`nodes.common_ksampler` in both offline harnesses.
Krea2's existing `_FakeModelPatcher.get_model_object` raises by design (to
test other guard rails) - added a separate minimal `_FakeSamplerModel`
fake for the sampler tests rather than changing the shared fixture. 44/44
combined smoke tests across the two files pass. Real-environment check
confirmed both nodes register with the real `comfy.samplers.KSampler.SAMPLERS`/
`SCHEDULERS` lists, and the sigma math itself was verified against a real
loaded Krea2 GGUF model (see numbers above) - not just fakes.

## Flux Klein: foundation + Identity Feature Transfer port (2026-08-25)

New file `nodes_flux_klein.py`. Category `🤖 CCTech/Flux Klein`. FLUX.2
Klein needed zero bespoke loading logic - `comfy/supported_models.py`
already detects `unet_config.image_model == "flux2"` natively, same
situation as Krea2/Qwen-Image, so `FluxKleinModelLoader` is the same thin
GGUF-aware MODEL/CLIP/VAE loader shape as `Krea2ModelLoader`/
`QwenImageModelLoader` (`clip_type=comfy.sd.CLIPType.FLUX2`, confirmed
present in core's `CLIPLoader` dropdown).

`FluxKleinImg2Img`'s empty txt2img latent uses Flux.2's REAL shape -
`[batch_size, 128, height // 16, width // 16]` (confirmed via
`comfy_extras/nodes_flux.py`'s `EmptyFlux2LatentImage`) - not this repo's
usual generic 4-channel/8-downscale placeholder, since
`comfy.sample.fix_empty_latent_channels` only auto-corrects channel
count, not spatial downscale ratio, unless `downscale_ratio_spacial` is
explicitly passed (it isn't, in this repo's plain `{"samples": latent}`
convention). Deliberately does NOT take multi-reference images itself -
kept in its own node to match the real example workflow
(`positive conditioning -> Multi ReferenceLatent -> sampler`) and avoid
an 8-image-slot node.

`Flux2KleinMultiReferenceLatent` is a direct, faithful port of the source
pack's `multi_reference_latent.py` (MIT, capitan01R,
https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer): one required
+ 7 optional `LATENT` inputs, splits each input's batch into individual
references, stamps `reference_latents` (overwrite, not append) +
`reference_latents_method="index"` onto BOTH positive and negative
conditioning (matches the real `Klein Controlnet.json` example
workflow's own reference-conditioning subgraph, which chains onto both).
`"index"` is a real branch inside comfy's own `Flux._forward`
(`comfy/ldm/flux/model.py:365-387`, shared Flux/Kontext/Klein code) - not
something this pack invented.

`Flux2KleinIdentityFeatureTransfer` is a near-verbatim port of the source
pack's flagship `IdentityFeatureTransferFinal`
(`identity_feature_transfer.py:789-1291`). Uses ONLY stock `ModelPatcher`
hooks - `model.clone()` then always
`m.set_model_attn1_output_patch(output_patch)`, and
`m.set_model_attn1_patch(reference_source_mask_patch)` only when
`mask_behavior="zero_unmasked_tokens"` and at least one `subject_mask_N`
is wired. Both fire generically from comfy's own
`comfy/ldm/flux/layers.py` `DoubleStreamBlock.forward`/
`SingleStreamBlock.forward` - shared Flux/Kontext/Klein code, confirmed
real methods on `comfy.model_patcher.ModelPatcher` before relying on
them. Reads four `extra_options` keys comfy's own model forward pass
already populates every call - `reference_image_num_tokens`,
`block_index`, `block_type`, `img_slice` - zero independent Klein-
internals computation needed, direct port of the read logic. All
`INPUT_TYPES` ported as-is: `preset` (HARD_LOCK/MID_LOCK/SOFT_LOCK/
custom), `enabled`, `reference_index`/`reference_indices`,
`similarity_floor`, `softmax_temperature`, `mask_threshold`,
`double_blocks`/`single_blocks` schedule strings, optional `sigmas`
(per-step strength decay via an equal-energy schedule ratio),
`debug`, `mask_behavior`, `subject_mask_1..8`.

KNOWN CAVEAT ported as-is, not fixed speculatively: default schedules/
presets hardcode block counts (8 double / 24 single) as magic numbers
tuned for the Klein 9B layout - never read from the live model. Harmless
clamp on a different-sized variant (e.g. 4B `klein-base`), but a preset
could apply strength to the wrong semantic blocks there - documented in
the node's docstring/README to use `preset="custom"` for non-9B
checkpoints, same honesty standard as everything else this session.

Explicitly NOT ported: `Flux2KleinKSamplerExperimental` (confirmed to
reimplement Euler sampling by hand instead of going through
`comfy.samplers`/`CFGGuider`, which means it silently bypasses comfy's
`sampler_post_cfg_function` pipeline - within the source pack itself this
makes it strictly less compatible than stock `KSampler`, since it'd break
that pack's own Color Anchor/Identity Guidance nodes) and the source
pack's own superseded `IdentityFeatureTransfer`/`Advanced`/`V3` (kept
there only for that pack's backward compatibility). Klein uses stock
`KSampler` - no custom sampler needed here, unlike Qwen-Image/Krea2.

`tools/smoke_flux_klein.py`: 14/14, no GPU. `FluxKleinImg2Img` latent
shape/denoise/batch-repeat tests, `Flux2KleinMultiReferenceLatent`
batch-splitting/ordering/overwrite/dual-conditioning tests, and 7 new
`Flux2KleinIdentityFeatureTransfer` tests - hook registration under
enabled/disabled and focus_only/zero_unmasked_tokens+mask, schedule
parsing (`_parse_schedule`, `_parse_ref_indices`), and the actual
similarity-pull math verified end-to-end on a synthetic attention tensor
(text token + generated tokens + reference tokens along a shared axis,
confirming only the generated-token range moves and only toward whichever
reference token shares its centered direction).

Real-environment check (synthetic-package trick against
`N:\ComfyUI_windows_portable_nvidia\`): all 4 new nodes register with
correct `INPUT_TYPES`/`RETURN_TYPES` (48 total registered nodes, up from
44); `CLIPType.FLUX2` and native `Flux2` in `comfy.supported_models`
confirmed; `set_model_attn1_patch`/`set_model_attn1_output_patch`
confirmed as real `ModelPatcher` methods; loaded the user's real
`D:\models\image-models\flux2-klein\split\flux-2-klein-9b-Q5_K_M.gguf`
end-to-end through `UnetLoaderGGUF`'s own loading path, confirmed
auto-detection as `Flux2` model class, and ran
`Flux2KleinIdentityFeatureTransfer.apply()` against the real loaded
`GGUFModelPatcher` - clones correctly, registers `attn1_output_patch` in
`model_options["transformer_options"]["patches"]`.

Not yet ported (tracked as remaining Klein work): `Flux2KleinColorAnchor`,
`Flux2KleinEnhancer`+`Flux2KleinDetailController`, `Flux2KleinTextEnhancer`,
`Flux2KleinSectionedEncoder` (needs verifying
`clip.tokenizer.qwen3_8b.tokenizer`/`.qwen3_4b.tokenizer` against this
repo's actual loaded Klein CLIP before porting), `Flux2KleinMaskRefController`,
`Flux2KleinRefLatentController`+`Flux2KleinTextRefBalance`+
`Flux2KleinRefLatentWeight`, `IdentityGuidance`.

## Flux Klein: Part E, the rest of the pack (2026-08-25)

Same file, `nodes_flux_klein.py`. Ported the remaining 10 nodes from
ComfyUI-Flux2Klein-Enhancer - all straightforward, faithful translations,
no architecture decisions needed since each reduces to one of the three
mechanism families already established in Parts C/D:

- `Flux2KleinColorAnchor` + `Flux2KleinIdentityGuidance` (renamed from
  source's `IdentityGuidance` for naming consistency with the rest of
  this file) both register via
  `model.model_options["sampler_post_cfg_function"]` - confirmed this
  only fires through comfy's own `CFGGuider`/`sampling_function`, so
  BOTH require sampling through stock `KSampler` to do anything at all.
  Ported as a list-append on a fresh copy (`list(...) + [fn]`) rather
  than the source's `setdefault(...).append(...)` in-place mutation, to
  avoid the clone potentially sharing the same list object as the
  original model's `model_options` dict depending on how deep
  `ModelPatcher.clone()`'s copy actually goes.
- `Flux2KleinEnhancer`, `Flux2KleinDetailController`,
  `Flux2KleinTextEnhancer`, `Flux2KleinMaskRefController` are pure
  conditioning-dict mutation, no model hooks. `DetailController` reads
  `meta["klein_sections"]` (from `SectionedEncoder`) for real section
  boundaries, falls back to fixed 25/50/25 slicing without it - same
  "honest about the fallback being arbitrary" framing as the source.
- `Flux2KleinRefLatentController` + `Flux2KleinTextRefBalance` +
  `Flux2KleinRefLatentWeight` are `set_model_attn1_patch` variants -
  same hook family as Identity Feature Transfer's output patch, but
  scaling K/V directly instead of pulling values toward a reference
  bank.
- `Flux2KleinSectionedEncoder` reaches into
  `clip.tokenizer.qwen3_8b.tokenizer`/`.qwen3_4b.tokenizer` directly -
  the ONE node in this whole port with a real Klein-CLIP-internals
  dependency. Verified live against the user's real
  `qwen3-4b-fp8_mixed.safetensors` Klein CLIP (loaded via
  `comfy.sd.load_clip(..., clip_type=CLIPType.FLUX2)`):
  `clip.tokenizer` is a real `KleinTokenizer`, `.qwen3_4b` is a real
  `Qwen3Tokenizer` with `.tokenizer` being a genuine HF `Qwen2Tokenizer`
  (`.qwen3_8b` is `None` on this 4B-text-encoder checkpoint, as
  expected - the source's own dual-attribute check handles this by
  design). Ports cleanly, confirmed rather than assumed.

`tools/smoke_flux_klein.py` gained 14 more tests (now 28/28 total, no
GPU): Color Anchor's inactive-without-reference guard and hook
registration on a clone (not the original model); Enhancer's no-op
passthrough (`out is cond`, no tensor copy) and active-region scaling;
Detail Controller's real-vs-fallback section range selection; Text
Enhancer's BOS-token-skipped magnitude scaling; Mask Ref Controller's
black-mask full-attenuation and no-reference-latents passthrough; Ref
Latent Controller/Text-Ref Balance/Ref Latent Weight's `attn1_patch`
K/V-range scaling verified on synthetic q/k/v tensors with a fake
`extra_options`; Identity Guidance's direct-mode pull math (denoised
moves exactly halfway to the reference at strength=0.5) and sigma-window
gating (no-op outside `[start_percent, end_percent]`); Sectioned
Encoder's `klein_sections` emission with a fake HF tokenizer plus its
graceful no-tokenizer fallback (still encodes, just no section metadata).

Real-environment check: all 10 new nodes register with correct
`INPUT_TYPES`/`RETURN_TYPES` (58 total registered nodes, up from 48);
`Flux2KleinRefLatentController`, `Flux2KleinRefLatentWeight`,
`Flux2KleinColorAnchor`, and `Flux2KleinIdentityGuidance` were each
applied against the real GGUF-loaded `flux-2-klein-9b` `GGUFModelPatcher`
and confirmed their respective hook (`attn1_patch` or
`sampler_post_cfg_function`) actually lands in
`model_options`/`transformer_options["patches"]` on the cloned model,
not the original.

This completes the Klein port plan (Parts A-E). Full parity with
ComfyUI-Flux2Klein-Enhancer achieved except the two deliberate
exclusions (`Flux2KleinKSamplerExperimental`, the source's own
superseded Identity Feature Transfer V1/Advanced/V3) - both documented
in the module docstring and README with the reasoning.

## Fix: FluxKleinImg2Img was missing Klein's real reference mechanism (2026-08-25)

User correction, and it was right: I'd read the shipped example workflow
(`N:\...\Klein Controlnet.json`) for Parts B/C but mis-scoped what to
port from it - I only ported the multi-reference chaining pattern
(`Flux2KleinMultiReferenceLatent`) and left `FluxKleinImg2Img` with just
a single plain `image` (img2img partial-denoise) input, no second
reference/control slot at all. User: "i dont even have a second slot
here to attach the control image or second ref image."

Traced the actual node graph inside the workflow JSON (not just its node
type list, which I'd looked at before and mis-summarized) to get this
right: the "Image Edit (Flux.2 Klein 9B Distilled)" subgraph starts from
a PURE-NOISE `EmptyFlux2LatentImage` - not an img2img denoise of the
edited photo at all - and drives the whole edit off two VAE-encoded
reference images attached to positive AND negative conditioning as
`reference_latents`, plus a text instruction (real saved prompt: "change
the pose of the subject in the image2 to the pose in the image1"). Of
the two references, one is the RAW photo; the other is fed through
`AIO_Preprocessor` (set to `MiDaS-DepthMapPreprocessor`) BEFORE being
VAE-encoded - i.e. depth-as-reference, not depth-as-ControlNet. Confirmed
via `json.load` + walking `data["nodes"]`/`data["links"]` and the
subgraph `data["definitions"]["subgraphs"]` (this workflow uses
ComfyUI's subgraph feature - the top-level node list alone doesn't show
what's inside "Image Edit"/"Reference Conditioning").

Fix, in `nodes_flux_klein.py`: added `reference_image` (IMAGE, optional)
+ `control_mode` (`["manual", "auto_depth"]`, default `manual`) +
`depth_ckpt_name` to `FluxKleinImg2Img.INPUT_TYPES`/`.prepare()`. `manual`
attaches `reference_image` raw; `auto_depth` runs it through Depth
Anything V2 first (new `_depth_anything_batch()` helper, same detector
class Krea2Img2Img's `control_mode="auto_depth"` already uses) before
VAE-encoding, reproducing the AIO_Preprocessor step. Both attach via
`node_helpers.conditioning_set_values(..., {"reference_latents": [...]},
append=True)` on BOTH positive and negative - same mechanism as Krea2/
Qwen-Image's `edit_reference`, and matches what the real example workflow
does with stock `ReferenceLatent` nodes (append, no explicit
`reference_latents_method`, unlike `Flux2KleinMultiReferenceLatent`'s
own `"index"` overwrite convention - deliberately different, since this
single-reference path is meant to compose with plain chained
`ReferenceLatent`/stock nodes, not just this pack's own multi-reference
node).

Also added `Flux2KleinDepthMap` (new node): the same Depth Anything V2
detector as a standalone IMAGE->IMAGE node, mirroring `Krea2 Depth Map`'s
existing convention (explicit, user-wired preprocessing nodes rather than
hidden auto-derivation) - user's explicit choice via AskUserQuestion over
redesigning `Flux2KleinMultiReferenceLatent`'s LATENT inputs into
IMAGE+VAE+per-reference control_mode (a breaking signature change to an
already-shipped node), so the LATENT-based multi-reference node is
unchanged; `Flux2KleinDepthMap` composes with it via
`Flux2KleinDepthMap -> VAEEncode -> Flux2KleinMultiReferenceLatent`.

`tools/smoke_flux_klein.py` gained 5 tests (32/32 total): reference_image
manual/auto_depth/absent behavior on `FluxKleinImg2Img` (monkeypatching
`fk._depth_anything_batch` to avoid a real model download in the offline
suite), and `Flux2KleinDepthMap` delegating to the shared helper with the
right ckpt_name/resolution. Needed one new smoke-harness fake:
`folder_paths.models_dir` (depth_anything_v2.py reads it at import time
for its model-cache directory constant) - same fix `smoke_krea2.py`
already had.

Real-environment check: `FluxKleinImg2Img`'s new optional inputs and
`Flux2KleinDepthMap` both register correctly (59 total nodes, up from
58); ran `FluxKleinImg2Img.prepare()` against a REAL loaded Klein VAE
(`flux2-vae.safetensors`) and REAL loaded Klein CLIP
(`qwen3-4b-fp8_mixed.safetensors`, `CLIPType.FLUX2`) with
`control_mode="manual"` and confirmed `reference_latents` actually lands
in both positive and negative conditioning with the correct encoded
shape (`[1, 128, 16, 16]` for a 256x256 reference at Flux.2's real VAE
downscale ratio) - not just registration, the real encode+attach path.

## Preprocessor expansion: 8 new ControlNet-aux ports + dedup (2026-08-25)

User asked whether the full `comfyui_controlnet_aux` package had been
ported (it hadn't - only Depth Anything V2 and a plain cv2 Canny
existed), requested a full inventory, then set two requirements before
any more porting: (1) no model weight files ever get added to the repo -
every preprocessor downloads its own weights on first use into the real
ComfyUI install's `models/<family>/` folder, same as `depth_anything_v2.py`
already did; (2) a concrete integration plan before implementation.
Planned via EnterPlanMode/ExitPlanMode (approved plan preserved the
research: full architecture/dependency/license inventory of all ~15-20
comfyui_controlnet_aux preprocessor families, tiered by port effort).

**Consolidation first, then new ports.** Confirmed real duplication
before adding anything: `_auto_canny_control_image()` was copy-pasted
verbatim in `nodes/krea2.py` and `nodes/qwen_image.py`; the Depth
Anything V2 detector loop was duplicated three times (inline in both
`Img2Img.prepare()` methods, plus its own helper in `nodes/flux_klein.py`);
three near-identical standalone "Depth Map" nodes existed. New file
`nodes/preprocessors.py` (category `🤖 CCTech/Preprocessors`) is now the
one place this logic lives - `DepthMap`/`Canny` classes, `_estimate_batch()`
(shared per-image-in-batch loop, matches every ported detector's
`.estimate(image_hwc_uint8, resolution=512, **kwargs)` contract) and
`_depth_anything_batch()`. `Krea2DepthMap`, `Flux2KleinDepthMap`,
`QwenImageCanny` stay registered in their own pipeline's
`NODE_CLASS_MAPPINGS` as aliases pointing at the shared classes (with
their own display-name strings preserved, e.g. `"Krea2 Depth Map ⚡"`) -
zero graph-breaking change for saved workflows. `control_mode="auto_depth"`/
`"auto_canny"` on Krea2Img2Img/QwenImageImg2Img now call the shared
helpers instead of their own inline copies, unchanged behavior.

**Real bug found and fixed during dedup verification, not part of the
plan**: `depth_anything_v2.py`'s `_image_to_tensor` hardcoded its own
`"cuda" if torch.cuda.is_available() else "mps"/"cpu"` device detection
instead of respecting whatever device `DepthAnythingV2Detector.to(device)`
actually moved the model to - so an explicit `.to("cpu")` call (e.g. a
`--cpu` ComfyUI launch on a CUDA-capable box) was silently ignored
whenever CUDA was available, causing a real
`RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
(torch.FloatTensor) should be the same` the first time anyone actually
exercised the real detector on such a machine (never caught before
because every prior real-environment check of Krea2/Qwen-Image's
`auto_depth` used the offline-mocked smoke suite or `control_mode="manual"`
to skip a real download). Fixed: `device = next(self.parameters()).device`
inside `DepthAnythingV2Detector`'s underlying model. Every subsequent
port's agent was explicitly briefed on this exact bug and told to derive
device from the model's own parameters, never re-detect independently -
confirmed via grep/review in every one of the 9 new ports below.

**8 new preprocessors ported and shipped** (parallel `general-purpose`
agents, 2 batches, each agent given the exact `depth_anything_v2.py`/
`hed.py` template convention to mirror, told to verify real HF repo
ids/filenames against the actual source rather than guess, and to flag
any new dependency rather than silently add one):

- `vendor/normal_bae.py` (`NormalBAEDetector`) + `vendor/dsine.py`
  (`DSINEDetector`) - both need the optional `timm` dependency (added to
  `requirements.txt`) to build a `tf_efficientnet_b5.ap_in1k` encoder,
  lazily imported inside `Encoder.__init__` (not required just to import
  the module or register the node).
- `vendor/hed.py` (`HEDDetector`), `vendor/pidinet.py` (`PiDiNetDetector`,
  MIT but with an added "commercial use should be contacted with authors
  first" note - flagged in the file header, node docstring, and README),
  `vendor/mlsd.py` (`MLSDDetector`) - no new dependencies.
- `vendor/lineart.py` (`LineartDetector`, fine/coarse checkpoints),
  `vendor/lineart_anime.py` (`LineartAnimeDetector`), `vendor/manga_line.py`
  (`MangaLineDetector`) - no new dependencies; each replaced the source's
  `einops`/PIL usage with plain `.permute()`/numpy since this repo's
  convention is HWC uint8 numpy in/out, not PIL images.

Each got a matching `nodes/preprocessors.py` node class using
`_estimate_batch()`, registered in `NODE_CLASS_MAPPINGS`/
`NODE_DISPLAY_NAME_MAPPINGS`. `tools/smoke_preprocessors.py` grew from
3 to 12 tests (a shared `_check_detector_backed_node()` helper fakes each
architecture's Detector class to confirm the node wires it up and returns
the right IMAGE batch shape/dtype without downloading real weights) - all
pass, no GPU. Real-environment check against the actual portable ComfyUI:
all 10 preprocessor nodes register correctly (69 total registered nodes,
up from 59), and one real end-to-end run per architecture family (real
HuggingFace download + real inference on a real image) confirmed output
shape/dtype for all 8 newly-ported detectors, not just Depth Anything
V2/Canny.

**OpenPose (classic, 3-CNN body/hand/face)**: `vendor/openpose.py`
compiles clean, and its port agent verified the same device-derivation
discipline across all three sub-networks (`_Body`/`_Hand`/`_Face`).
`open_pose/LICENSE` in the source pack is CMU's actual OpenPose license -
"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY,"
forbidding sublicensing/transferring/"provid[ing] third parties access
to" the software - a materially different, much more restrictive
situation than every other license in this batch (Apache-2.0/MIT, one
soft PiDiNet research-use note). Initially held back from registration
pending a legal question I didn't think was mine to decide unilaterally;
user's call: this pack shipping the code is no different from
`comfyui_controlnet_aux` itself shipping the identical code under its
own Apache-2.0 wrapper (the wrapper's license and the underlying
architecture/weights' license are separate things, same situation every
OpenPose-shipping ComfyUI pack is in) - registered as the `OpenPose`
node, license text quoted in full in `vendor/openpose.py`'s header, the
node's own docstring, and README with the same "read this before
commercial use" framing as PiDiNet's softer note.

**Klein's `control_mode` now covers every non-depth preprocessor too**
(user caught this was still missing after the batch above shipped):
`FluxKleinImg2Img`'s `control_mode` grew from `["manual", "auto_depth"]`
to include one `auto_*` entry per `nodes/preprocessors.py` node
(`auto_canny`, `auto_normal_bae`, `auto_normal_dsine`,
`auto_soft_edge_hed`, `auto_soft_edge_pidinet`, `auto_mlsd`,
`auto_lineart`, `auto_lineart_anime`, `auto_manga_line`,
`auto_openpose`) via a `_CONTROL_MODE_NODES` dispatch dict mapping mode
string -> preprocessor node CLASS (not a duplicated detector call) -
`prepare()` does `getattr(node_cls(), node_cls.FUNCTION)(reference_image)`
generically, reusing each node's own tested `estimate()`/`detect()`
method (including OpenPose's, which returns a `(canvas, pose_dict)`
tuple - the dispatch takes only the canvas via the same unpacking
`OpenPose.estimate()` itself already does). Krea2Img2Img/QwenImageImg2Img
deliberately did NOT get this same treatment - their `control_mode` is
tied to a specific loaded Control LoRA/DiffSynth patch (depth/canny
only), so e.g. `auto_normal_bae` there would be mechanically wireable
but silently useless. Klein's reference_latents mechanism has no such
constraint. `tools/smoke_flux_klein.py` gained 2 tests (36/36 total): a
real `auto_canny` end-to-end run (no model needed) confirming the
dispatch actually attaches `reference_latents`, and a completeness check
that `_CONTROL_MODE_NODES` covers every preprocessor node except the two
handled outside the dict (`manual`=raw, `auto_depth`=its own
`depth_ckpt_name`-parameterized path). Real-environment check:
confirmed `control_mode=auto_openpose` against a REAL loaded Klein VAE +
CLIP + the real OpenPose detector - `reference_latents` attached with
the correct encoded shape (70 total registered nodes, up from 69).

Not ported (per the approved plan, explicitly deferred): DWPose (the
modern pose preprocessor - real extra scope, `onnxruntime`, two-stage
detector+pose pipeline; worth its own dedicated follow-up), everything
in the research pass's "heavy" tier (Metric3D, UniFormer, Mesh
Graphormer, Diffusion Edge, Unimatch - each drags in a vendored
framework or has an ill-fitting I/O shape), everything that's really a
thin `transformers.from_pretrained(...)` wrapper upstream now (MiDaS,
Zoe, Depth Anything v1, OneFormer, SAM, DensePose - porting these means
adding a `transformers` dependency, a separate decision), and the ~15
"no real model" utility nodes (Scribble, Binary, Tile, Color, Recolor,
Shuffle, Inpaint, pose-keypoint-drawing helpers).

## Split into three packages; trimmed back to a real minimum (2026-08-25)

User's reaction to landing at 70 registered nodes: "that's a bit much."
Rather than just trimming, asked for a proper split: two new standalone
repos published via `gh` CLI, and a real definition of "minimum" for
what stays in this repo.

**Two new standalone packages** (both `gh repo create --public`,
`.github/workflows/publish_action.yml` copied from this repo's own,
`REGISTRY_ACCESS_TOKEN` secret added by the user, both manually triggered
via `gh workflow run` and confirmed successful, both `main`-branch
default matching this repo's convention):

- **[ComfyUI-ControlNet-Nodes](https://github.com/ChrisColeTech/ComfyUI-ControlNet-Nodes)**
  - the 11-node preprocessor set (Depth Anything V2, Normal BAE/DSINE,
  Soft Edge HED/PiDiNet, MLSD, Lineart/Anime/Manga, OpenPose, Canny) -
  a straight copy-and-adapt of what had just been built here, done by a
  general-purpose agent given explicit instructions to leave this repo
  completely untouched (verified: `git status` in this repo before/after
  showed only pre-existing changes from other work, nothing from that
  agent).
- **[ComfyUI-Flux-Reference-Tools](https://github.com/ChrisColeTech/ComfyUI-Flux-Reference-Tools)**
  - 9 nodes extracted from `nodes/flux_klein.py` and renamed to drop
  "Klein" branding (`Flux2KleinMultiReferenceLatent` ->
  `FluxMultiReferenceLatent`, etc.) - a second general-purpose agent
  confirmed via grep that none of the 9 actually import anything from
  this repo's `ops`/`loader`/`vendor` modules, i.e. they were genuinely
  Flux-family-generic all along, just built and named as part of a
  Klein-specific port. `FluxDetailController` keeps reading the literal
  `meta["klein_sections"]` metadata key (optional, still works standalone
  without it) for cross-package compatibility with this repo's own
  `Flux2KleinSectionedEncoder`.

**Defining "minimum" for what stays here**: not "whatever seems useful,"
but literally the two preprocessors this repo's own img2img nodes have
always auto-derived internally - Depth Anything V2 and plain `cv2.Canny`
- for `control_mode="auto_depth"/"auto_canny"` on `Krea2Img2Img`/
`QwenImageImg2Img`/`FluxKleinImg2Img`. Everything else the earlier
preprocessor batch added (9 vendor files, 9 node classes) got removed
from `vendor/` and `nodes/preprocessors.py` via `git rm` + a full
rewrite of that file down to just `DepthMap`+`Canny` (implementation
only - no `NODE_CLASS_MAPPINGS` of its own anymore, see the file's own
docstring for why: registering generic "DepthMap"/"Canny" names here
would collide with ComfyUI-ControlNet-Nodes' own registration of those
same names if both packs are installed). `nodes/__init__.py`'s merge of
`nodes.preprocessors`'s own mappings was removed accordingly.

Same logic for `nodes/flux_klein.py`: reverted `FluxKleinImg2Img`'s
`control_mode` from the 10-option set (added earlier the same day) back
down to `["manual", "auto_depth", "auto_canny", "none"]` - the exact
same set Krea2/Qwen-Image use, since the other 7 `auto_*` modes were
scope creep past "minimum," not something Krea2/Qwen-Image ever had
either. The `_CONTROL_MODE_NODES` dispatch dict and its `from . import
preprocessors as pp` import were removed; `auto_canny` is now handled
the same inline way `auto_depth` already was, calling
`_auto_canny_control_image` directly (imported from `.preprocessors`
alongside `_depth_anything_batch`, both still needed).

**New: `control_mode="none"` on all three img2img nodes** (Krea2Img2Img,
QwenImageImg2Img, FluxKleinImg2Img). User caught a real design gap while
discussing the trim: `manual` was never "off," it meant "supply
`control_image`/`reference_image` yourself" - Krea2Img2Img's own guard
rail (`if has_control_lora and control_image is None:`) fired
unconditionally whenever a Control LoRA was loaded upstream, regardless
of `control_mode`, so there was no way to skip control for one
generation without physically disconnecting or removing the loader.
Same shape in QwenImageImg2Img (`if qwen_control is not None and
control_image is None:`). Fixed by adding `and control_mode != "none"`
to both guard conditions - minimal, surgical, doesn't touch the
"explicit `control_image` always wins" behavior since that check only
gates the *auto-derivation-or-raise* branch, not an explicitly-wired
`control_image`.

**Real bug caught while adding this to QwenImageImg2Img**: the
downstream attachment block (`if qwen_control is not None:` ->
`control_image.movedim(...)`) had no `control_image is not None` guard
of its own - it never needed one before, since the upstream guard always
either set `control_image` to something or raised. Once `none` could
leave `control_image` as `None` while `qwen_control` stayed connected,
this would have crashed with `AttributeError: 'NoneType' object has no
attribute 'movedim'`. Fixed: `if qwen_control is not None and
control_image is not None:`.

For Klein, `reference_image is not None and control_mode == "none"` is
handled as its own branch (logs and does nothing) ahead of the normal
`elif reference_image is not None:` derivation branch - Klein never had
Krea2/Qwen-Image's "forced" problem in the first place (its
`reference_image` was always genuinely optional, no Control-LoRA-style
gate), so `none` here is purely a graph-toggle convenience, not a bug
fix - but added for UI consistency across all three nodes.

**Breaking change accepted, not silently avoided**: the 9 nodes moved to
ComfyUI-Flux-Reference-Tools could NOT be kept as backward-compat
aliases the way `Krea2DepthMap`/`Flux2KleinDepthMap`/`QwenImageCanny`
were during the preprocessor consolidation - those aliases work because
the implementation still lives in this same repo/package; a node that
now lives in an entirely separate installed package has no path back.
Any saved workflow using e.g. `Flux2KleinMultiReferenceLatent` directly
will show a missing-node error unless the user also installs
ComfyUI-Flux-Reference-Tools AND manually re-adds the (differently-
named) replacement node. Documented plainly in `nodes/flux_klein.py`'s
own `NODE_CLASS_MAPPINGS` comment, README, and here - not something to
paper over.

Net result, verified against the real portable ComfyUI environment:
**70 -> 50 registered nodes.** `tools/smoke_preprocessors.py` shrank
from 13 to 4 tests (just `DepthMap`/`Canny` + a check that the module
self-registers nothing). `tools/smoke_flux_klein.py` shrank from 36 to
20 tests (removed everything covering the 9 relocated nodes, kept/added
coverage for the minimum `control_mode` set including a real
`control_mode=none` skip-attachment test and a `_CONTROL_MODES` exact-
match test). `smoke_krea2.py`/`smoke_qwen_image.py` each gained one
`control_mode=none` test. Real-environment check: all 20 removed/moved
node-type names confirmed absent from `NODE_CLASS_MAPPINGS`, all 8
kept/alias names confirmed present, `FluxKleinImg2Img` with
`control_mode="none"` and `control_mode="auto_canny"` both verified
against a real loaded Klein VAE/CLIP - node count exactly 50.

## Fix: FluxKleinImg2Img canvas sizing + dual reference_image_2 (2026-08-25)

User traced a real broken img2img result back to a specific real shipped
example workflow, `image_flux2_klein_image_edit_9b_base.json`, and asked
us to confirm `FluxKleinImg2Img` does the same thing. It didn't, in two
ways:

1. **Canvas sizing was disconnected from the actual reference photo.**
   `width`/`height` were plain user-typed widget defaults (1024x1024)
   trusted as exact - used to center-crop-resize (`comfy.utils.common_
   upscale(..., "center")`) any connected reference image to that exact
   aspect ratio, and to build the empty-latent canvas independently of
   the photo's own shape. The real workflow never does this: every
   reference photo goes through `ImageScaleToTotalPixels` (aspect-
   preserving resize to a fixed megapixel budget) then `GetImageSize`,
   and THAT derived size feeds `EmptyFlux2LatentImage`/`Flux2Scheduler`
   - the canvas always matches the reference photo's own aspect ratio,
   never an independent literal size. Confirmed by direct JSON graph
   inspection of the shipped workflow (two subgraphs sharing a name but
   different UUIDs - a single-reference 9B one and a dual-reference
   9B-base one), not assumed.

2. **Only one reference image slot existed.** The real workflow's
   dual-reference subgraph (node 92: "apply the logo from image 2 onto
   the car in image 1") uses TWO `LoadImage` nodes, each independently
   scaled and VAE-encoded, then chained via two sequential
   `ReferenceLatent` calls per conditioning branch (positive and
   negative) - the second image's reference attached on top of the
   first's, both raw, no preprocessing on either.

Fixed both: new `_scale_to_megapixels(image, megapixels, resolution_
steps=16)` helper is an aspect-preserving port of comfy core's own
`ImageScaleToTotalPixels` math (`comfy_extras/nodes_post_processing.py`)
- `resolution_steps=16` is a deliberate deviation from the real
workflow's own `resolution_steps=1`, chosen to stay aligned with
Flux.2's real `/16` latent downscale stride and avoid a fractional-
latent-pixel edge case; not an unverified guess. `width`/`height` are
now a pixel budget: whenever `image`, `reference_image`, or the new
`reference_image_2` is connected, the real canvas re-derives from that
photo's own aspect ratio at the same budget, with `image` taking
priority over `reference_image` over `reference_image_2` for sizing.
Pure txt2img (nothing connected) still uses `width`/`height` literally.
`reference_image_2` is raw-only (no `control_mode` dispatch - that
still only applies to `reference_image`), attached after
`reference_image` on both positive and negative conditioning, matching
the real dual-reference chain exactly.

User's explicit note: `image`/`reference_image`/`reference_image_2`'s
naming is confusing (`reference_image` reads like "the ControlNet
image" but is really Klein's general edit-reference mechanism) - a
separate, later, repo-wide renaming pass was explicitly deferred, not
folded into this fix.

`tools/smoke_flux_klein.py` gained 6 tests (26 total): `_scale_to_
megapixels` aspect-ratio/megapixel-budget/rounding correctness on its
own, the actual bug scenario (non-square `reference_image` no longer
forced into the square widget default - asserted via real latent
shape), `image`-vs-`reference_image` sizing priority (asserted via a
custom fake VAE that records the shape actually passed into `encode()`,
since the shared `_vae()` fixture's fixed-shape return value would have
made a naive output-shape assertion pass trivially regardless of
whether the fix worked), and `reference_image_2` attaching raw
alongside `reference_image` both combined and standalone. Real-
environment check against the actual portable ComfyUI install, the real
Flux.2 VAE, and a real loaded Klein CLIP: a 2:1 landscape
`reference_image` produced a 91x45 (non-square, correctly-proportioned)
latent instead of the old square default, and `reference_image` +
`reference_image_2` together produced exactly 2 `reference_latents`
entries on both positive and negative conditioning.

## Fix: control_mode=none was silently dropping the raw reference too (2026-08-25)

User caught this against a REAL edit attempt, not a review: wired a photo
into the previous `reference_image` slot, used the exact real Flux2Scheduler
+ CFGGuider + SamplerCustomAdvanced sampler chain (matching the shipped
example workflow precisely, cfg=5, correct resolution), and still got "a
coherent image, but not the edit" - same symptom reported the first time,
with `image` wired instead of `reference_image`. Traced it by re-reading
the user's saved workflow JSONs directly: `control_mode` was left at
`'none'` in both graphs (a leftover default), and the previous
`FluxKleinImg2Img.prepare()` had `if reference_image is not None and
control_mode == "none": skip` - `none` was implemented to mean "skip
reference attachment entirely," not "skip only the depth/canny
preprocessing," so a genuinely-connected `reference_image` got silently
dropped with zero error/warning surfaced to the user, and the model just
fell back to prompt-only generation - explaining the "coherent but wrong"
symptom exactly.

Root cause: `reference_image` was doing double duty as both Klein's raw
identity/content reference AND the thing `control_mode` decided how to
preprocess, so there was no way to say "attach raw, skip only
preprocessing" separately from "skip attaching it at all" - `manual`
already meant "attach raw," leaving `none` nothing to mean except "don't
attach." Checked whether `Krea2Img2Img`/`QwenImageImg2Img` have the same
bug (`nodes/krea2.py:989-994`, `nodes/qwen_image.py:387-395`): they don't
- both keep `edit_reference` (their raw-reference equivalent) in its own
independent code block, never touched by `control_mode`, which only ever
gates `control_image`/`qwen_control`. On those two, `control_mode="none"`
in the worst case only skips *auto-deriving* a control image when one
wasn't explicitly supplied (`control_image is None` guards the whole
branch) - an explicitly-connected `control_image` is never dropped by
`none`. Klein was the outlier because it never had that two-slot
separation to begin with.

User pushed on the actual design ("that is not correct... you still need
a slot for the controlnet image, how else would you attach it" /
"imagine we took control net out of the picture, which slot would i use
to reproduce the actual proper workflow" / "so then it should be one
slot that says images, and accept one or more image nodes") - landed on
splitting `FluxKleinImg2Img` into the same two-slot shape Krea2/Qwen-
Image already use, closing the bug by construction rather than just
fixing the one conditional:

- `images` (single IMAGE socket, batch-aware) - one or more RAW reference
  photos, always attached to `reference_latents` on positive+negative
  when connected, with ZERO dependency on `control_mode`. Researched
  whether comfy has a native "one socket, many wires" mechanism before
  picking this shape (`io.Autogrow`, `comfy_api/latest/_io.py:1053-1166`)
  - real, but V3-only, and every node in this pack is still V1-style
    (`INPUT_TYPES`); migrating just Klein would make it inconsistent with
    Krea2/Qwen-Image. Went with a single `IMAGE` socket that treats its
    batch dimension as N separate images internally (`for i in
    range(images.shape[0]): ...` - each its own `VAEEncode` +
    `reference_latents` append, never one batched encode) - combine
    multiple photos upstream via stock ComfyUI's "Batch Images" node.
    This exactly reproduces the real dual-reference example subgraph's
    own per-image `ReferenceLatent` chain, confirmed by re-reading its
    node graph (not assumed).
- `control_source_image` (single IMAGE) + `control_mode` - a photo to
  turn INTO a controlnet-style map (Klein has no real ControlNet/
  Control-LoRA of its own) via this pack's Depth Anything V2/`cv2.Canny`,
  then attached through the SAME `reference_latents` mechanism as
  `images`, appended after them. `control_mode` now applies ONLY to this
  slot - `none` skips `control_source_image` attachment and can no
  longer reach `images` at all, closing the bug by construction rather
  than by careful conditional ordering.

Also dropped the `image`/`strength` partial-denoise img2img slot
entirely, confirmed with the user this was intentional and fine to lose
("i never said drop denoise dial - but it is redundant because its in
the ksampler, so thats ok"): no real Klein edit example ever partially
denoises the source photo, `denoise` is now a fixed `1.0` output. This
also makes `QwenImageKSampler`/`Krea2KSampler`'s diffusers-vs-comfy
sigma-slicing fix a non-issue for Klein specifically - that bug only
applies to `denoise<1.0`, which Klein's node no longer produces.

Investigated and ruled out two other hypotheses before finding the real
bug, worth recording since they're real, separate, measured facts even
though they weren't the cause here: (1) `Flux2Scheduler`'s resolution-
adaptive `compute_empirical_mu(seq_len, steps)` shift
(`comfy_extras/nodes_flux.py:206-233`) differs from stock `KSampler`'s
`ModelSamplingFlux` fixed `shift=2.02` (`comfy/supported_models.py:801`)
- real, but the user's test graph had `Flux2Scheduler`'s width/height
manually set to match the actual reference photo's size, so this wasn't
in play; (2) confirmed `clip.tokenize()`+`clip.encode_from_tokens_
scheduled()` is byte-identical between our node and stock
`CLIPTextEncode` (`nodes.py:73-77`) - ruled out an encoding difference.

`tools/smoke_flux_klein.py`: same 26 test count, but the `images`/
`control_source_image` set was rewritten for the new signature - added
a test asserting a 2-image batch in `images` produces exactly 2
`reference_latents` (not 1 batched encode), a test asserting `images`+
`control_source_image` combine correctly with `control_source_image`
appended after `images`, and critically a test asserting
`control_mode="none"` skips ONLY `control_source_image` and leaves
`images` completely untouched - the actual regression test for this bug.
Real-environment check against the real Flux.2 VAE/CLIP: `INPUT_TYPES`
confirmed `image`/`reference_image`/`reference_image_2`/`strength` are
gone and `images`/`control_source_image` are present; a 2-image batch
produced 2 `reference_latents`; `images`+`control_source_image`
(auto_canny) combined into 2 `reference_latents` total.

**Follow-up in the same pass**: user asked why the node still had a
`denoise` output slot at all, given it was now a hardcoded constant
`1.0` ("im confused, what is it passing out? why would you hardcode the
outlet"). Checked stock `KSampler`'s own `denoise` widget default
(`nodes.py:1610`) - it's already `1.0`. A `FLOAT` output that never
varies and matches the downstream node's own default carries zero
information - it was leftover from when this output used to pass the
real `strength` value, and that reasoning stopped applying once the
partial-denoise path was removed. Dropped the `denoise` output
entirely: `RETURN_TYPES`/`RETURN_NAMES` shrink from 5 to 4 (`MODEL,
CONDITIONING, CONDITIONING, LATENT` / `model, positive, negative,
latent`), `prepare()` returns a 4-tuple. Updated all 11 `node.prepare()`
call sites across `tools/smoke_flux_klein.py` to unpack 4 values instead
of 5, README, and this file's own prior entry. Real-environment check
confirmed the new `RETURN_TYPES`/`RETURN_NAMES` and that `prepare()`
returns exactly a 4-tuple against the real VAE/CLIP.

## Krea2Img2Img/QwenImageImg2Img: dead-simple images/control_image, edit_reference removed (2026-08-26)

Same investigation thread as the Klein `control_mode=none` bug, but the
user reported a fresh symptom on a totally different node
(`krea2_img2img.json`, Krea2's canny Control LoRA, "change her shirt to
a plain black top" not applying). Traced the actual saved workflow JSON,
same discipline as the Klein debugging: found `Krea2Img2Img.denoise`
output wasn't wired to `KSampler.denoise` at all (KSampler used its own
literal `1.0` widget, silently ignoring the configured `strength=0.33`),
and separately that `auto_canny`/`auto_depth` derivation was hardcoded to
only ever read `image`, with no fallback to `edit_reference` - so moving
the photo to `edit_reference` (the fix I first suggested) broke the
Control LoRA auto-derivation entirely (`raise ValueError(...)`, "no image
was given to derive a control image from").

Confirmed the identical narrow gap exists in `QwenImageImg2Img`
(`nodes/qwen_image.py:345,349`, hardcoded to `image` only). Started
patching both with an `image`-else-`edit_reference` fallback, but the
user stopped mid-edit and pushed for the real fix instead: "edit
reference goes away that doesn't make sense and isnt working properly.
klein doesn't have an edit reference slot, it has images and a control
net image slot. dead simple" / "you need one or more images for img2img
and a controlnet image for control net, i dont understand what else you
need."

Had my own understanding of img2img directly challenged and corrected
mid-conversation: initially conflated "does this node take a photo" with
"is this img2img" and proposed dropping partial-denoise img2img entirely
to match Klein's shape - wrong, since (unlike Klein, which never uses
partial-denoise in any real workflow) Krea2/Qwen-Image's `image`+
`strength` IS real, legitimate, working img2img (VAE-encode, add noise
proportional to `1 - strength`, denoise from there) and removing it would
be a real capability loss, not a naming cleanup. The user's actual
correction: `images` (renamed from `image`, batch-aware) STAYS real
img2img - `edit_reference`'s separate reference_latents mechanism is what
goes away, since it's redundant with `images` doing the actual editing
job and it wasn't working right anyway.

Applied to both `Krea2Img2Img` and `QwenImageImg2Img`:
- `image` -> `images` (identical code - `vae.encode()`/`KSampler` already
  process a batched tensor as N independent parallel generations, no
  per-image loop needed unlike Klein's reference_latents case, since
  this is genuine partial-denoise img2img, not reference conditioning).
- `edit_reference` and its whole `reference_latents`-attachment code
  block removed entirely from both nodes.
- `auto_canny`/`auto_depth` derivation now reads `images` (the renamed
  slot) - the `derive_source`-fallback patch became unnecessary once
  `edit_reference` was gone, so it was reverted rather than kept as dead
  code.
- `Krea2ControlLoRALoader`'s docstring/log line pointed at
  `edit_reference` for ordinary in-context LoRAs (e.g.
  `krea2_canny-v0.1.safetensors`) - updated to point at
  ComfyUI-Flux-Reference-Tools instead, since there's no in-repo
  reference-conditioning mechanism left on these two nodes for that kind
  of LoRA.

`tools/smoke_krea2.py`/`tools/smoke_qwen_image.py`: renamed every
`image=`/`image is not None` test call site to `images=`, dropped both
`edit_reference` tests, added a batch test per file (a 3-image batch in
`images` produces a batch-3 latent - `vae.encode()`'s own batch handling
does the work, confirmed with a batch-aware fake VAE since the shared
`_vae()` fixture in both files returns a fixed batch-1 shape regardless
of input). 29/29 (Krea2) and 14/14 (Qwen-Image), no GPU. Real-environment
check against the real portable ComfyUI: confirmed `INPUT_TYPES` on both
nodes have `images` and no `image`/`edit_reference`, and exercised the
`images`-connected/no-Control-LoRA path end to end against real object
shapes.

## Fix: GGUF text-encoder mmproj merge skipped for "qwen3"-tagged VL conversions (2026-08-26)

User reported the redesigned Krea2 nodes broke, but the actual error
("Krea2 expects conditioning with 12x2560=30720 features... but got
2560. Load the text encoder with CLIPLoader type 'krea2'.") traced back
to a real bug in `loader.py`, unrelated to any node redesign this
session - confirmed by checking `Krea2ModelLoader.load()`
(`nodes/krea2.py:654-673`) is completely unchanged and already correctly
passes `clip_type=CLIPType.KREA2`.

Root cause, confirmed by reading the user's actual GGUF file's header
metadata directly (`loader.read_gguf_arch()`), not guessed: their
`Qwen3-VL-4B-Q4_K_M.gguf` reports `general.architecture = "qwen3"` (the
base LLM arch), not `"qwen3vl"`, even though it's genuinely a Qwen3-VL-4B
text-encoder conversion shipped beside a matching mmproj vision-tower
file (`Qwen3-VL-4B-Instruct-mmproj-BF16.gguf`) - some quantizers tag the
text-only conversion with the base LLM arch string. `loader.py`'s
`gguf_clip_loader()` only attempted the sibling-mmproj auto-merge for
`arch in {"qwen2vl", "qwen3vl"}` (`loader.py:857`, pre-fix) - so a
"qwen3"-tagged VL conversion never got its vision weights merged, and
comfy's own `detect_te_model()` (which requires
`model.visual.deepstack_merger_list.0.norm.weight` to recognize
`TEModel.QWEN3VL_4B`) fell through to misdetecting it as plain
`TEModel.QWEN3_4B` - producing a single-layer 2560-feature embedding
instead of Krea2's required 12-layer 30720-feature stack, with no error
until the model's own forward-pass validation in
`comfy/ldm/krea2/model.py`'s `_unpack_context()`.

Verified the mmproj-matching logic itself was never broken - called
`gguf_mmproj_loader()` directly against the real file pair and it found
and loaded all 315 vision-tower keys correctly by filename match; the
bug was purely the arch-string gate deciding whether to even attempt
that lookup. Fixed by widening `loader.py:857`'s condition to
`arch in {"qwen2", "qwen2vl", "qwen3", "qwen3vl"}` - safe by
construction since `gguf_mmproj_loader()` already no-ops (empty dict,
just a warning) when no matching mmproj file exists nearby, so this
can't affect a genuinely text-only Qwen2/Qwen3 checkpoint.

Verified end to end against the real files: `gguf_clip_loader()` on the
real `Qwen3-VL-4B-Q4_K_M.gguf` now returns 713 keys including 19
visual/deepstack keys (was 0), `comfy.sd.detect_te_model()` now returns
`TEModel.QWEN3VL_4B` (was misdetecting `QWEN3_4B`), and
`Krea2ModelLoader.load()` end-to-end through the real portable ComfyUI
now produces a CLIP whose `encode_from_tokens_scheduled()` output has
exactly `(1, 8, 30720)` shape - the real number Krea2's DiT expects.
Added `test_gguf_clip_loader_merges_mmproj_even_when_te_arch_is_the_base_llm_tag`
to `tests/test_loader_metadata.py` (84/84 full pytest suite passes) as
the regression test, reproducing the real arch/filename pair with
synthetic GGUF fixtures.

Separately investigated (but NOT fixed, and not fixable in this repo) a
second error the user hit with the non-GGUF safetensors CLIP path
(`qwen3vl_4b_fp8_scaled.safetensors`): `ValueError: Expected trailing
dimension of mat1 to be divisible by 16 but got mat1 shape:
(25600x12)`. Traced to `comfy/ldm/krea2/model.py`'s `txtfusion` module,
which fuses Krea2's 12 tapped Qwen3-VL layers per-token via a linear
layer with 12 input features - inherent to Krea2's architecture, not a
bug. PyTorch's FP8 scaled-matmul kernel requires the contracted
dimension to be a multiple of 16, which 12 can never satisfy - a hard
incompatibility between Krea2's 12-layer-fusion design and running the
DiT as an `fp8_scaled` checkpoint (`krea2_turbo_uncensored_refined-
fp8_scaled.safetensors`), regardless of which CLIP is used. Not
something this repo's loader code can fix - the real remedy is to use a
non-fp8-scaled (bf16/fp16 safetensors, or GGUF) Krea2 diffusion-model
checkpoint instead, since GGUF quantization uses a different ops path
(`GGMLOps`, not `_scaled_mm`) that doesn't hit this constraint.

## Port: Krea2EditModelPatch + Krea2EditGroundedEncode from comfyui-krea2edit (2026-08-26)

Continuation of the same debugging thread: after the GGUF/fp8 fixes above,
user reported the actual instructed edit still wasn't taking effect even
with "the edit lora baked in." Traced it back to the wrong LoRA entirely
at first (I assumed `krea2_canny-v0.1.safetensors`, the ordinary in-
context Control-adjacent LoRA already documented in this file) - user
corrected: "i didnt say the controlnet lora i said the edit lora." The
real LoRA is `krea2_identity_edit_v1_2.safetensors` ("Krea 2 Identity
Edit," licensed separately under Krea AI's own Community License
Agreement), and user pointed at a local reference install for how it's
actually driven: "should we follow their convention
'D:\Projects\ComfyUI\comfyui-krea2edit-main'."

Read that pack directly rather than guessing. Confirmed it does NOT use
`reference_latents`/`ReferenceLatent` at all - comfy's native Krea2
forward (`comfy/ldm/krea2/model.py`) only ever builds `[text | target]`,
with reference_latents (`ref_method`/`ref_latents` kwarg) concatenating
into the IMAGE token stream at whatever frame index the CALLER chooses,
but with no built-in "this is the clean appearance-preservation source"
semantics. The Identity Edit LoRA was trained (via `ai-toolkit`'s
`predict_velocity_edit`) against a SPECIFIC sequence shape -
`[text | source(frame=1) | target(frame=0)]` - that nothing in comfy or
this pack could reproduce without wrapping the diffusion model's forward
directly. This explains why neither the old `edit_reference` mechanism
(removed earlier this session) nor anything reference_latents-based
could ever have worked correctly for this LoRA - wrong mechanism, not a
config mistake.

Before porting, spawned a research fork to verify every comfy API the
source pack calls against the REAL installed comfy at
`N:\ComfyUI_windows_portable_nvidia\ComfyUI`, since comfy APIs drift
across versions and this session's own established discipline is
"verify against real source, don't trust another pack's assumptions."
All five verified clean, no drift:
`comfy.patcher_extension.add_wrapper_with_key`/`WrappersMP.DIFFUSION_MODEL`
exist as used; `SingleStreamDiT` (`comfy/ldm/krea2/model.py:232`) has
every attribute the port reads (`patch`, `channels`, `tdim`,
`pe_embedder`, `first`, `blocks`, `tmlp`, `txtfusion`, `txtmlp`, `last`,
`tproj`, `_unpack_context`) 1:1; `process_latent_in` is inherited from
`BaseModel` unchanged; `Krea2Tokenizer.tokenize_with_weights`
(`comfy/text_encoders/krea2.py:28-30`) genuinely accepts
`llama_template=`/`images=` and `CLIP.tokenize` forwards `**kwargs`
straight through; `pad_to_patch_size` accepts `padding_mode` as called.

Found and resolved one real ambiguity mid-port: the source pack's own
production code calls the MODULE-LEVEL
`comfy.patcher_extension.add_wrapper_with_key(type, key, wrapper,
transformer_options_dict)`, writing into
`model.model_options["transformer_options"]["wrappers"]` - but this
repo's OWN existing `Krea2ControlLoRALoader` wrapper registration (real-
environment verified working in an earlier session) uses the
ModelPatcher INSTANCE METHOD `model.add_wrapper_with_key(type, key,
wrapper)` (3 args, no options dict), writing into `self.wrappers`
instead - a genuinely different storage location. Traced the bridge by
hand rather than assuming either was wrong: `comfy/ldm/krea2/model.py:280`
reads wrappers via `get_all_wrappers(DIFFUSION_MODEL, transformer_options)`
off the plain `transformer_options` dict threaded through the forward
call, and `comfy/sampler_helpers.py:224`'s `prepare_model_patcher` (run
on every real sample) explicitly merges `model.wrappers` into
`model_options["transformer_options"]["wrappers"]` before sampling
starts - so BOTH registration styles converge to the same place at
sampling time. Kept the method-based style to match this file's existing
proven convention rather than introducing a second style. Confirmed the
bridge for real (not just by reading code): built a real
`comfy.model_patcher.ModelPatcher`, called
`Krea2EditModelPatch.patch()`, then called the real
`comfy.sampler_helpers.prepare_model_patcher()` and confirmed the
wrapper landed in `model_options["transformer_options"]["wrappers"]`.

Ported `krea2_edit_forward` + its helpers (`_imgids`, `_imgids_offset`,
`_to_4d`, `_fit_src`, `_fit_encode_image`, `_ref_attn_bias`) and both
node classes near-verbatim into `nodes/krea2.py`. Originally registered
under the source pack's own `NODE_CLASS_MAPPINGS` keys (`Krea2EditModelPatch`,
`Krea2EditGroundedEncode`) so a workflow built against comfyui-krea2edit
would keep resolving if that pack were swapped out for this one -
superseded 2026-08-26 below once that turned out to actively collide
with the original pack when BOTH are installed at once (the more common
case in practice). `print(..., flush=True)` debug statements replaced with this
repo's own `logger.info`/`logger.warning` convention; two dead imports
(`apply_rope`, `optimized_attention_masked` - confirmed unused via grep,
vestigial from a larger file this distribution was trimmed from) were
dropped rather than carried over.

`tools/smoke_krea2.py` gained 11 new tests (39/39 total, no GPU):
wrapper registration on a fake ModelPatcher, the wrapper calling
`krea2_edit_forward` with the `process_latent_in`-scaled source, the
`target_latent` pre-encode timing (before sampling vs. on-first-step,
mirroring the source pack's own `test_pre_encode.py` regression tests
almost exactly), dual-reference pre-encoding, `krea2_edit_forward`'s
real math exercised against a shape-accurate synthetic DiT (a real
`[text | source | target]` concatenation round-tripping to the original
shape, not a stub), and `Krea2EditGroundedEncode`'s text-only fallback,
image-grounded `images=`/`llama_template=` kwargs, two-image vision-
block count, and `grounding_px` downscaling. Needed two new module
stubs in the offline harness (`comfy.ldm.flux.layers.timestep_embedding`,
plus explicitly wiring `comfy_ldm.common_dit`/`comfy_ldm.flux` as
attributes - discovered because `Krea2ControlLoRALoader`'s own
`pad_to_patch_size` call had never been offline-tested before, only
verified in real-environment checks). Real-environment check against the
actual portable ComfyUI: both nodes register with the documented
`INPUT_TYPES`, and the wrapper-registration-to-`transformer_options`
bridge was confirmed against real `comfy.model_patcher.ModelPatcher`/
`comfy.sampler_helpers` objects, not fakes.

## Split img2img and ControlNet img2img into separate nodes: Krea2, Qwen-Image, Klein (2026-08-26)

User's directive after the Krea2 GGUF/fp8 debugging session and the
Krea2 Identity Edit port: two real bugs this session traced back to the
exact same root cause - one node's `IMAGE` inputs doing two structurally
different jobs at once ("what to partially denoise/reference" and "what
control map to derive/attach") - `FluxKleinImg2Img`'s `control_mode=
none` silently skipping the entire reference attachment, and
`Krea2Img2Img`'s `auto_canny` derivation breaking when a photo moved off
`image` onto a reference slot. Rather than keep patching one blurred
node per model, the fix: split *every* image model's img2img node into
two separate, independent, standalone nodes - a plain img2img node with
zero control-related inputs, and a `*ControlNetImg2Img` node with
everything the current combined node already has (img2img inputs *and*
control inputs together - full superset, not control-only). Each node
is complete on its own; there is no dependency between them, and no
chained "Apply" step - a misread of the instruction, corrected by the
user ("i didnt say anything about an apply node, where did that come
from").

Scope: Krea2, Qwen-Image, Klein this pass. Z-Image (`nodes/zimage.py`,
`ZImageImg2Img`) has the exact same blurred pattern (`image` mixed with
`control_patch`/`control_image`) but is explicitly deferred to a
follow-up alongside the LTX models, per the user's own sequencing.

Same mechanical pattern in all three files - duplicate the existing
combined class, trim one copy down to `images` only (drop every
control-related input and the control-guard logic entirely - a Control
LoRA loaded upstream with the plain node in use now surfaces the
model's own wrapper-level error at sample time instead of a friendly
node-level guard, since the plain node has no way to attach a control
latent at all), keep the other copy exactly as the current node already
is, renamed `*ControlNetImg2Img`:

- `nodes/krea2.py`: `Krea2Img2Img` (trimmed) + `Krea2ControlNetImg2Img`
  (new, has `control_mode`/`control_image`/`depth_ckpt_name`/
  `control_channel_mode`/`control_normalize`/`control_invert`/
  `control_batch_mode` and the `has_control_lora` guard).
- `nodes/qwen_image.py`: `QwenImageImg2Img` (trimmed) +
  `QwenImageControlNetImg2Img` (new, has `qwen_control`/`control_mode`/
  `depth_ckpt_name`/`control_image`/`mask`/`control_strength`, both
  attachment paths - DiffSynth model-patch vs. real ControlNet-on-
  conditioning - unchanged).
- `nodes/flux_klein.py`: `FluxKleinImg2Img` (trimmed, `images` only,
  canvas-sizing now derives from `images` alone) +
  `FluxKleinControlNetImg2Img` (new, has `control_source_image`/
  `control_mode`/`depth_ckpt_name`, canvas-sizing priority `images` then
  `control_source_image` unchanged).

This is a real breaking change for any saved workflow using a control
input by name on the old combined node type (the socket becomes
orphaned once that type name resolves to the trimmed class) - consistent
with this session's established practice of taking breaking changes
when they genuinely remove confusion, always documented rather than
silently avoided.

`_auto_canny_control_image`/`_depth_anything_batch` stay the only
genuinely shared helpers (already imported from `nodes/preprocessors.py`)
- the control-plumbing code itself is copy-pasted into each
`*ControlNetImg2Img` class, not refactored into a shared function,
matching this repo's existing convention of per-pipeline duplication
over premature abstraction.

`tools/smoke_krea2.py`/`tools/smoke_qwen_image.py`/
`tools/smoke_flux_klein.py`: no new test *cases* invented - each file's
existing img2img coverage already split cleanly along this exact seam,
so tests that only ever exercised `images`/txt2img moved (unchanged) to
instantiate the trimmed class, and tests exercising `control_mode`/
`control_image`/`qwen_control`/`control_source_image` moved to
instantiate the new `*ControlNetImg2Img` class instead - same
assertions, same fakes, just pointed at the right class per the split.
39/39 (Krea2), 14/14 (Qwen-Image), 26/26 (Klein) unchanged, 84/84 full
pytest suite. Real-environment check against the actual portable
ComfyUI: all six classes register, plain nodes have exactly zero of the
control-related keys, and each `*ControlNetImg2Img`'s `INPUT_TYPES` is
confirmed a strict superset of its plain sibling's.

## Fix: Krea2EditModelPatch/Krea2EditGroundedEncode invisible with comfyui-krea2edit installed (2026-08-26)

User reported not being able to find these two nodes in Add Node at
all, despite the server log showing the package importing cleanly and
both classes present in `nodes/krea2.py`'s `NODE_CLASS_MAPPINGS`.
Traced it to the deliberate choice made when porting them (see the
entry above): they were registered under the exact same
`NODE_CLASS_MAPPINGS` string keys (`"Krea2EditModelPatch"`,
`"Krea2EditGroundedEncode"`) as the original `comfyui-krea2edit`
package they were ported from. ComfyUI merges every installed custom
node package's `NODE_CLASS_MAPPINGS` into one process-wide dict keyed
by that string - not namespaced per package - so with BOTH packs
installed (the user's actual setup, confirmed via their server log:
`comfyui-gguf-loader` imports first, `[krea2edit] nodes v1.2.5 loaded`
right after it), whichever pack's `__init__.py` runs later silently
overwrites the earlier pack's dict entry for that key. Ours lost:
`comfyui-krea2edit` loads second, so Add Node only ever surfaced its
version, even though our code was fully imported and registered too -
no error, no warning, just a quiet dict overwrite.

Fix: renamed only the two colliding registration keys to
`"CCTechKrea2EditModelPatch"`/`"CCTechKrea2EditGroundedEncode"` in
`nodes/krea2.py`'s `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`.
The Python class names (`Krea2EditModelPatch`, `Krea2EditGroundedEncode`)
are untouched - they live in this repo's own module namespace and were
never actually part of the collision; only the registration key is a
process-wide shared namespace across every installed pack. Every other
node in this file already uses a `Krea2`-prefixed (not bare) id and
was never at risk - these two were the only ones a straight verbatim
port left un-namespaced.

## Fix: rename the CLASSES too, not just the registration ids (2026-08-26)

The id-only fix above was scoped correctly for what it fixed (the
collision is entirely about the registration key, never the Python
class name), but it left something the user flagged as a real problem
on its own: `Krea2EditModelPatch`/`Krea2EditGroundedEncode` are the
exact same class names as `comfyui-krea2edit`'s own nodes (expected -
they're a verbatim port) - and every other node in this pack (`Krea2Img2Img`,
`Krea2KSampler`, `Krea2ModelLoader`, ...) is an ORIGINAL name, not a
copy of some other installed pack's class name. Leaving these two as
the one pair that's literally indistinguishable from a different
author's code - by name alone, with no way to tell which is which
in conversation or in a stack trace - was never actually resolved by
namespacing the id; the id is invisible (nobody ever sees
`CCTechKrea2EditModelPatch` printed anywhere), while the class name is
what shows up in every docstring, every smoke-test assertion, every
AGENTS.md entry, and every conversation about this code.

Renamed the classes themselves:
- `Krea2EditModelPatch` -> `Krea2IdentityEditSourcePatch`
- `Krea2EditGroundedEncode` -> `Krea2IdentityEditGroundedEncode`

`IdentityEdit` names the actual LoRA/feature these two nodes exist for
(the Krea 2 Identity Edit LoRA) rather than the generic word "Edit" the
source pack used; `SourcePatch`/`GroundedEncode` describe each node's
actual job (source-image injection vs. image-grounded prompt encode) -
words already used in this file's own `TITLE` strings
(`"Krea2 Identity Edit (source patch) ⚡"` /
`"Krea2 Identity Edit (grounded encode) ⚡"`), so the class name and the
display title now share the same vocabulary by construction instead of
being two independently-invented strings that happened to both refer
to the same node. Registration ids renamed to match
(`CCTechKrea2IdentityEditSourcePatch`/`CCTechKrea2IdentityEditGroundedEncode`),
keeping the `CCTech`-prefix convention `nodes/extra.py`'s
`CCTechDualVAELoader`/`CCTechClipProjLoader` already established for
collision-prone names in this pack.

`tools/smoke_krea2.py` updated to instantiate the renamed classes
(`krea2.Krea2IdentityEditSourcePatch()`/
`krea2.Krea2IdentityEditGroundedEncode()`); assertions/print-message
text updated to match. All 39 Krea2 smoke tests and the full 84-test
suite pass unchanged otherwise (pure rename, no logic touched).
`README.md`'s node-name references updated to
`Krea2 Identity Edit (source patch)`/`Krea2 Identity Edit (grounded
encode)` to match the new `TITLE`. Two already-saved real workflows on
the portable ComfyUI install (`Krea 2 Edit (Full Context).json`,
`krea2_img2img.json`) that reference the old `CCTechKrea2Edit*` type
ids from the previous fix were updated in place to the new ids -
otherwise both would fail to resolve their node type on next load.
