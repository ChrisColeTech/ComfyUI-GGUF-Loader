# AGENTS.md — Comfy-GGUF (CCTech fork) + Scenema Audio nodes

Working notes for any agent (or human) continuing work in this repo. Everything
below was verified against real checkpoints and real ComfyUI code, not inferred.

## Repo layout

- `nodes.py` — GGUF loader nodes (UNET/CLIP loaders, dual-VAE, ClipProj). Registers
  `unet_gguf` / `clip_gguf` folder keys. Node category: `🤖 CCTech/GGUF`.
- `nodes_scenema.py` — the Scenema Audio nodes (category `🤖 CCTech/Scenema`).
- `loader.py` — GGUF → fake-quantized state dict (`gguf_sd_loader`), text-encoder
  post-processing (`gguf_clip_loader`: key remaps, Gemma-3 norm `+1` un-bake,
  sentencepiece tokenizer rebuild from GGUF metadata — now cached next to the
  GGUF as `<name>.spiece_cache.bin`).
- `ops.py` / `dequant.py` — GGMLTensor/GGMLOps: weights stay quantized, dequant
  per-layer at forward time. **This is the repo's core identity.**
- `nodes_extra.py`, `clipproj.py` — DualVAELoader, ClipProjLoader.
- `tools/smoke_scenema.py` — CPU dry-run of the Scenema load paths against real
  checkpoints (`--skip-te` skips the slow Gemma GGUF part).

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
