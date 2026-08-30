# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""LTX-2.5 A/V nodes: kit loader, T2V/I2V prep, distilled sampler, latent
upscale, AV decode - plus the empty joint AV latent.

  LTXV25ModelsLoader   DiT (safetensors or GGUF) + Gemma-4 TE + video/audio VAE
  LTXV25ImgToVideo     prompts, init latent + noise mask, multi-LoRA chaining
  LTXV25KSampler       euler_ancestral on the LTX-2.5 distilled/refine sigmas
                       through core's dual-CFG (video/audio) guider
  LTXV25LatentUpscale  spatial x2 on the video half, optional i2v re-hold
  LTXV25AVDecode       joint AV latent -> muxed VIDEO

The recipe mirrored here is the official ``video_ltx2_5_i2v.json`` workflow's
"Image to Video (LTX-2.5)" subgraph (read directly, not guessed):

  * LTXVPreprocess(img_compression=18) on the input image;
  * stage 1 at HALF the target resolution (the workflow's ``a/2`` math nodes):
    EmptyLTXVLatentVideo + LTXVImgToVideoInplace(strength 0.7) +
    LTXVEmptyLatentAudio -> LTXVConcatAVLatent;
  * SamplerCustomAdvanced with LTXVDualCFGGuider(1,1), euler_ancestral and
    ManualSigmas ``1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725,
    0.421875, 0.0`` (LTX25_DISTILLED_SIGMAS below, verbatim);
  * LTXVSeparateAVLatent -> LTXVLatentUpsampler (x2 spatial) ->
    LTXVImgToVideoInplace(strength 1.0, same preprocessed image) ->
    LTXVConcatAVLatent -> a second SamplerCustomAdvanced with ManualSigmas
    ``0.85, 0.7250, 0.4219, 0.0`` (LTX25_REFINE_SIGMAS, verbatim);
  * VAEDecodeTiled(512, 64, 64, 16) + LTXVAudioVAEDecode -> CreateVideo(fps);
  * one or more chained LoraLoaderModelOnly on the model.

The conditioning/latent path mirrors comfy_extras/nodes_lt.py rather than
reinterpreting it (same discipline as nodes/ltx23.py):

  * the init latent is core EmptyLTXVLatentVideo's exact geometry
    [B, 128, (L-1)//8+1, H//32, W//32];
  * the i2v hold is core LTXVImgToVideoInplace's math verbatim (nodes_lt.py:
    151-176): the encoded image OVERWRITES the first latent frames of the
    existing latent (nothing is appended - which is why this family needs no
    CropVideoGuide node, see LTXV25ImgToVideo's docstring), and the hold is a
    per-frame mask [B, 1, T, 1, 1] carrying 1 - strength on the held frames;
  * the joint AV latent and its mask are NestedTensor((video, audio)) - core
    LTXVConcatAVLatent's own form;
  * frame_rate rides on the conditioning via node_helpers
    .conditioning_set_values, the same call LTXVConditioning makes;
  * sampling composes core's real Guider_LTXAVDualCFG + sampler_object +
    CFGGuider.sample - the exact objects SamplerCustomAdvanced wires up.

There is also ``LTXV25EmptyLatentAVBatch``: comfy core ships
``EmptyLTXVLatentVideo`` (video stream only) and the ``LTXVConcatAVLatent`` /
``LTXVSeparateAVLatent`` pair, but nothing that hands the LTX-2.5 AV DiT a
ready-made video+audio latent. The gap is easy to fill with the wrong thing:
the MiniMax H3 empty-AV node produces a nested latent of the same *type*, so it
wires up cleanly and then dies inside ``patchify_proj`` - H3 video is 24
channels on a /16 spatial grid, LTX-2.5 wants 128 on /32, and the audio streams
disagree on both rank and layout.

Comfy core ships ``EmptyLTXVLatentVideo`` (video stream only) and the
``LTXVConcatAVLatent`` / ``LTXVSeparateAVLatent`` pair, but nothing that hands
the LTX-2.5 AV DiT a ready-made video+audio latent. The gap is easy to fill with
the wrong thing: the MiniMax H3 empty-AV node produces a nested latent of the
same *type*, so it wires up cleanly and then dies inside ``patchify_proj`` — H3
video is 24 channels on a /16 spatial grid, LTX-2.5 wants 128 on /32, and the
audio streams disagree on both rank and layout.

This node builds the LTX-2.5 geometry directly:

    video  [B, 128, (length - 1) // 8 + 1, height // 32, width // 32]
    audio  [B, z_channels, n_latents, frequency_bins]

The audio side is read off the audio VAE rather than hardcoded, because
``n_latents`` depends on the clip duration and on the VAE's mel hop
(``sample_rate / mel_hop_length / 8`` latents per second) — the same derivation
``LTXVReferenceAudio`` relies on when it encodes real audio.
"""

import logging
import math
from fractions import Fraction

import folder_paths
import node_helpers
import nodes
import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils

try:
    from comfy_api.latest import InputImpl as _VideoInputImpl
    from comfy_api.latest import Types as _VideoTypes
except ImportError:  # pragma: no cover - older comfy without the video API
    _VideoInputImpl = _VideoTypes = None

logger = logging.getLogger(__name__)

LTX25_CATEGORY = "🤖 CCTech/LTX-2.5"

# The exact ManualSigmas strings from the official video_ltx2_5_i2v.json
# workflow, stage 1 and stage 2. Passed through verbatim - no resampling: the
# distilled bake was trained on this curve at cfg 1.0 and any deviation looks
# like a broken model, not a slightly different one.
LTX25_DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                          0.909375, 0.725, 0.421875, 0.0]
LTX25_REFINE_SIGMAS = [0.85, 0.7250, 0.4219, 0.0]

# Order matters: this is the dropdown order, and the default sits first.
LTX25_SIGMA_SETS = {
    "distilled (8 steps)": LTX25_DISTILLED_SIGMAS,
    "refine (3 steps)": LTX25_REFINE_SIGMAS,
}

# The reference workflow's values, link-traced (node 357 / node 349 in
# video_ltx2_5_i2v.json's subgraph): LTXVImgToVideoInplace strength 0.7 on the
# stage-1 half-res EMPTY latent, 1.0 on the UPSCALED latent before the refine
# pass, LTXVPreprocess img_compression 18 on the input image.
LTX25_I2V_STRENGTH = 0.7
LTX25_REFINE_STRENGTH = 1.0
LTX25_IMG_COMPRESSION = 18
FPS = 24.0

# The video VAE's compression: /32 in space, /8 in time with a causal first frame.
VIDEO_LATENT_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8


def video_latent_t(length):
    """Latent frame count for ``length`` pixel frames (first frame is causal)."""
    return ((max(1, int(length)) - 1) // VIDEO_TEMPORAL_RATIO) + 1


def audio_latent_shape(audio_vae, length, frame_rate, batch_size):
    """Read the audio stream's geometry off the LTX-2.5 audio VAE.

    Returns ``(shape, latents_per_second)``. Raises if the VAE is not an LTX
    audio VAE, which is the common miswiring (the *video* VAE plugged into the
    audio slot).
    """
    model = getattr(audio_vae, "first_stage_model", None)
    required = ("num_of_latents_from_frames", "latent_frequency_bins")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "audio_vae is not an LTX-2.5 audio VAE (missing {}); load "
            "ltx-2.5-audio-vae-bf16.safetensors with the stock VAE Loader, not "
            "the video VAE.".format(", ".join(missing))
        )

    n_latents = int(model.num_of_latents_from_frames(int(length), float(frame_rate)))
    if n_latents < 1:
        raise ValueError(
            "length {} at {} fps is too short to produce an audio latent "
            "(need at least {:.3f}s).".format(
                length, frame_rate, 1.0 / float(model.latents_per_second))
        )

    channels = int(getattr(audio_vae, "latent_channels", model.latent_channels))
    bins = int(model.latent_frequency_bins)
    return ([int(batch_size), channels, n_latents, bins],
            float(model.latents_per_second))


def empty_av_latent(audio_vae, width, height, length, frame_rate, batch_size=1):
    """Build the nested (video, audio) latent the LTX-2.5 AV DiT samples."""
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [int(batch_size), VIDEO_LATENT_CHANNELS, video_latent_t(length),
         int(height) // VIDEO_SPATIAL_RATIO, int(width) // VIDEO_SPATIAL_RATIO],
        device=device)
    shape, _ = audio_latent_shape(audio_vae, length, frame_rate, batch_size)
    audio = torch.zeros(shape, device=device)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
    }


class LTXV25EmptyLatentAVBatch:
    """Empty video+audio latent shaped for the LTX-2.5 AV transformer.

    Feeding this to the sampler instead of the MiniMax H3 empty-AV node is the
    difference between a clip and a channel-mismatch traceback: the two models
    share the nested-latent container but not a single dimension of it.
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "Empty LTX-2.5 AV Latent (Batch) ⚡"
    SEARCH_ALIASES = ['empty latent', 'new latent', 'create latent', 'blank latent', 'audio latent', 'video latent', 'batch latent']

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE", {
                    "tooltip": "ltx-2.5-audio-vae-bf16.safetensors. Only its "
                               "geometry is read here (latent channels, "
                               "frequency bins, latents per second) — no "
                               "encoding happens, so it costs nothing."}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 97, "min": 1, "max": nodes.MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Frame count. The video VAE compresses 8:1 in "
                               "time with a causal first frame, so 8k+1 values "
                               "(9, 97, 121...) tile exactly."}),
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Sets the clip duration, which sets the audio "
                               "latent length. Use the same value on LTXV "
                               "Conditioning or the two streams drift apart."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Clips per job, each with its own noise from the "
                               "sampler seed. The LTX-2.5 DiT is batch-aware, "
                               "so unlike MiniMax H3 no patch node is needed — "
                               "but VRAM scales with the batch."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    DESCRIPTION = ("Joint video+audio latent in LTX-2.5 geometry: video "
                   "[B,128,(len-1)//8+1,H/32,W/32] and audio "
                   "[B,z,n_latents,bins] read from the audio VAE. Use this, not "
                   "the MiniMax H3 empty-AV node, to drive an LTX-2.5 sampler.")

    def generate(self, audio_vae, width, height, length, frame_rate, batch_size):
        latent = empty_av_latent(audio_vae, width, height, length, frame_rate,
                                 batch_size)
        video, audio = latent["samples"].unbind()
        logger.info("LTX-2.5 empty AV latent: video %s, audio %s (%d frames @ "
                    "%.2f fps = %.2fs)", tuple(video.shape), tuple(audio.shape),
                    length, frame_rate, length / float(frame_rate))
        return (latent,)


# ── loading ─────────────────────────────────────────────────────────────────

def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def _align_length(length):
    """Round a frame count up to the 8k+1 grid the video VAE tiles on."""
    length = max(9, int(length))
    while (length - 1) % VIDEO_TEMPORAL_RATIO != 0:
        length += 1
    return length


# Gemma-4 12B costs many GB and seconds to build, and comfy re-runs loader
# nodes on every prompt edit. Cache the most recent one; the CLIP owns its
# ModelPatcher so comfy still manages its VRAM. Same single-entry-cache
# pattern as ltx23's _ENCODER_CACHE (the 2.3 loader has no keep_loaded
# widget either - the cache IS its keep-loaded mechanism).
_ENCODER_CACHE = {}


def _load_ltx25_clip(clip_name):
    if clip_name in _ENCODER_CACHE:
        return _ENCODER_CACHE[clip_name]

    path = folder_paths.get_full_path("clip", clip_name) \
        or folder_paths.get_full_path_or_raise("text_encoders", clip_name)

    if path.lower().endswith(".gguf"):
        from ..loader import gguf_clip_loader
        from ..ops import GGMLOps
        sd = gguf_clip_loader(path)
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.LTXV,
            state_dicts=[sd],
            model_options={
                "custom_operations": GGMLOps(),
                "initial_device": comfy.model_management.text_encoder_offload_device()},
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
    else:
        # Same call comfy-core's CLIPLoader makes for type "ltxv" - the 2.5
        # kit's gemma4-12b-with-proj file carries encoder + projection in one
        # file, unlike the 2.3 kit's separate *_projections.safetensors.
        clip = comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.LTXV,
        )

    _ENCODER_CACHE.clear()
    _ENCODER_CACHE[clip_name] = clip
    return clip


def _load_vae(vae_name, want_audio):
    path = folder_paths.get_full_path("vae", vae_name) \
        or folder_paths.get_full_path_or_raise("vae", vae_name)
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    is_audio = hasattr(vae.first_stage_model, "num_of_latents_from_frames")
    if want_audio and not is_audio:
        raise ValueError(f"{vae_name} is not an LTX audio VAE (looks like the "
                         f"video VAE); audio_vae wants ltx-2.5-audio-vae-*.safetensors.")
    if not want_audio and is_audio:
        raise ValueError(f"{vae_name} is the audio VAE; video_vae wants "
                         f"ltx-2.5-video-vae-*.safetensors.")
    return vae


class LTXV25ModelsLoader:
    """Load the whole LTX-2.5 A/V kit in one node.

    DiT GGUFs (Q6_K/Q8_0) stay quantized (dequantized per layer at forward
    time) via this pack's own UnetLoaderGGUF, same dispatch-by-extension
    LTXV23ModelsLoader uses; the comfy int8 safetensors goes through
    comfy.sd.load_diffusion_model. Outputs are plain comfy MODEL / CLIP / VAE
    objects, so they compose with comfy's own LTXV nodes as well as these.
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "LTX-2.5 Models Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'audio vae',
                      'video vae', 'load clip', 'text encoder']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the LTX-2.5 A/V components (DiT, Gemma-4 text encoder "
                   "with projection, video VAE, audio VAE).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "LTX-2.5 A/V DiT from models/diffusion_models - "
                               "the comfy int8 safetensors or a Q6_K/Q8_0 GGUF. "
                               "GGUF stays quantized."}),
                "clip_name": (_clip_filename_list(), {
                    "tooltip": "gemma4-12b-with-proj-ltx-2.5-*.safetensors from "
                               "models/text_encoders - encoder and projection in "
                               "one file (loaded as CLIPLoader type ltxv)."}),
                "video_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "ltx-2.5-video-vae-bf16.safetensors."}),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "ltx-2.5-audio-vae-bf16.safetensors."}),
            },
        }

    def load(self, unet_name, clip_name, video_vae_name, audio_vae_name):
        if unet_name.lower().endswith(".gguf"):
            from .gguf import UnetLoaderGGUF
            model, = UnetLoaderGGUF().load_unet(unet_name)
        else:
            model = comfy.sd.load_diffusion_model(
                folder_paths.get_full_path("unet", unet_name)
                or folder_paths.get_full_path_or_raise("unet", unet_name))

        cfg = model.model.model_config.unet_config
        if cfg.get("image_model") != "ltxav":
            raise ValueError(
                f"{unet_name} built as image_model={cfg.get('image_model')}, "
                "expected ltxav. Not an LTX-2.5 A/V checkpoint."
            )
        logger.info("LTX-2.5: %s -> ltxav/%s layers (%.2f GiB stored)",
                    unet_name, cfg.get("num_layers"), model.model_size() / 1024 ** 3)

        return (model,
                _load_ltx25_clip(clip_name),
                _load_vae(video_vae_name, want_audio=False),
                _load_vae(audio_vae_name, want_audio=True))


# ── conditioning / latent prep ──────────────────────────────────────────────

def _match_batch(t, n):
    """Bring ``t`` to exactly ``n`` along dim 0 by tiling, then truncating.

    Deliberately idempotent: calling it twice is the same as calling it once
    (``Tensor.repeat`` alone is not - it multiplies).
    """
    have = t.shape[0]
    if have == n:
        return t
    if have > n:
        return t.narrow(0, 0, n)
    reps = -(-n // have)                      # ceil
    return t.repeat(reps, *([1] * (t.dim() - 1))).narrow(0, 0, n)


def _preprocess_images(images, img_compression):
    """Core LTXVPreprocess's per-frame H.264 round-trip, delegated to
    comfy_extras.nodes_lt.preprocess (not reimplemented). crf 0 is core's own
    documented passthrough."""
    if int(img_compression) == 0:
        return images
    import comfy_extras.nodes_lt as nodes_lt
    return torch.stack([nodes_lt.preprocess(images[i], int(img_compression))
                        for i in range(images.shape[0])])


def _apply_i2v_hold(vae, images, video_samples, video_mask, strength, batch_size):
    """Core LTXVImgToVideoInplace's math on an already-built latent tensor
    (nodes_lt.py:151-176, mirrored not reinterpreted): resize the image to the
    latent grid's pixel size, encode, OVERWRITE the first latent frames, and
    carry 1 - strength on those frames of the per-frame noise mask. Nothing is
    appended - which is why the 2.5 family needs no CropVideoGuide node."""
    latent_h, latent_w = video_samples.shape[-2:]
    width = latent_w * VIDEO_SPATIAL_RATIO
    height = latent_h * VIDEO_SPATIAL_RATIO
    if images.shape[1] != height or images.shape[2] != width:
        pixels = comfy.utils.common_upscale(
            images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    else:
        pixels = images
    t = _match_batch(vae.encode(pixels[:, :, :, :3]), batch_size)
    video_samples[:, :, :t.shape[2]] = t.to(video_samples.device, video_samples.dtype)
    video_mask[:, :, :t.shape[2]] = 1.0 - strength
    return t.shape[2]


class LTXV25ImgToVideo:
    """Prompts + stage-1 init latent for LTX-2.5: T2V and I2V, with multi-LoRA
    chaining built in - no external LoraLoaderModelOnly wiring.

    ``mode`` (required, LTXV23VidToVideo's selector philosophy) declares which
    base behavior this call is: ``t2v`` (images must be disconnected) or
    ``i2v`` (images required - first-frame hold via core
    LTXVImgToVideoInplace's exact semantics, after core LTXVPreprocess's
    ``img_compression`` round-trip, both mirrored from the official workflow).

    ``width``/``height`` are the FINAL output resolution. The official
    LTX-2.5 recipe samples stage 1 at HALF that resolution (the workflow's
    ``a/2`` math nodes) and doubles it back with the spatial upscaler, so the
    latent built here is the half-res stage-1 latent - feed it to
    LTXV25KSampler on "distilled (8 steps)", then LTXV25LatentUpscale (which
    re-holds the same image at 1.0, the workflow's second
    LTXVImgToVideoInplace), then LTXV25KSampler again on "refine (3 steps)",
    then LTXV25AVDecode.

    ``lora_1``/``lora_2``/``lora_3`` ("none" = skip) chain through comfy-core's
    real ``LoraLoaderModelOnly`` in order, each at its own strength - the same
    multi-LoRA-by-chaining the official workflow wires by hand. Always take
    ``model`` from THIS node's output when any selector is set.

    No CropVideoGuide counterpart exists for this family on purpose:
    LTXVImgToVideoInplace overwrites latent frames in place (nodes_lt.py:171)
    rather than appending guide frames, so there is nothing to crop - the
    sampled latent is already exactly the output timeline.

    No keep_original_audio either: that option is tied to a source VIDEO input
    (keep ITS audio), which this i2v-only prep node does not have; the audio
    stream starts empty (mask 1) and the model generates it, matching the
    workflow's LTXVEmptyLatentAudio.
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "LTX-2.5 Img to Video ⚡"
    SEARCH_ALIASES = ['image to video', 'img2vid', 'i2v', 'text to video',
                      't2v', 'empty latent', 'video conditioning', 'lora']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "frame_rate")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, half-res stage-1 init latent and noise masks for "
                   "LTX-2.5 T2V/I2V, with up to three LoRAs chained in-node. "
                   "Feed into LTXV25KSampler (distilled), then "
                   "LTXV25LatentUpscale, LTXV25KSampler (refine), "
                   "LTXV25AVDecode.")

    @classmethod
    def INPUT_TYPES(s):
        lora_choices = ["none"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "mode": (["i2v", "t2v"], {"default": "i2v",
                          "tooltip": "Which base behavior this call is. i2v: images "
                                     "required (first-frame hold). t2v: images must "
                                     "be disconnected."}),
                "vae": ("VAE", {"tooltip": "The loader's vae (video VAE) output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the scene and its motion. "
                                                 "A caption, not an instruction."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1280, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 2,
                                  "tooltip": "FINAL output width. Stage 1 samples at "
                                             "half this (the official recipe) and the "
                                             "latent upscaler doubles it back."}),
                "height": ("INT", {"default": 720, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 2,
                                   "tooltip": "FINAL output height - stage 1 runs at "
                                              "half, like width."}),
                "length": ("INT", {"default": 121, "min": 9,
                                   "max": nodes.MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Frames; 8k+1 tiles exactly (9, 97, "
                                              "121...). 121 @ 24 fps = the workflow's "
                                              "5 s default."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0,
                                         "step": 0.01,
                                         "tooltip": "24 is the LTX-2 convention. The "
                                                    "frame_rate output carries it to "
                                                    "LTXV25AVDecode."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "First frame. Resized and CENTER-"
                                     "CROPPED to the stage-1 grid here - do not "
                                     "scale it upstream. Wire the SAME image into "
                                     "LTXV25LatentUpscale for the refine re-hold."}),
                "image_strength": ("FLOAT", {
                    "default": LTX25_I2V_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "i2v only. How hard the first frame is held in stage 1. "
                               "0.7 is the official stage-1 value; the refine "
                               "re-hold in LTXV25LatentUpscale uses 1.0 (locked)."}),
                "img_compression": ("INT", {
                    "default": LTX25_IMG_COMPRESSION, "min": 0, "max": 100,
                    "tooltip": "Core LTXVPreprocess's H.264 crf round-trip on the "
                               "input image (official value 18; 0 = off). Matches "
                               "the compression statistics the model was trained "
                               "on so the first frame doesn't pop."}),
                "lora_1": (lora_choices, {"default": "none",
                           "tooltip": "First LoRA, loaded onto model HERE via comfy-"
                                      "core's LoraLoaderModelOnly (no external wiring). "
                                      "\"none\" = skip."}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                    "step": 0.01}),
                "lora_2": (lora_choices, {"default": "none",
                           "tooltip": "Second LoRA, chained after lora_1."}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                    "step": 0.01}),
                "lora_3": (lora_choices, {"default": "none",
                           "tooltip": "Third LoRA, chained after lora_2."}),
                "lora_3_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                    "step": 0.01}),
            },
        }

    @torch.inference_mode()
    def prepare(self, model, clip, mode, vae, audio_vae, prompt, negative_prompt,
                width, height, length, frame_rate, batch_size, images=None,
                image_strength=LTX25_I2V_STRENGTH,
                img_compression=LTX25_IMG_COMPRESSION,
                lora_1="none", lora_1_strength=1.0,
                lora_2="none", lora_2_strength=1.0,
                lora_3="none", lora_3_strength=1.0):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use "
                             "ltx-2.5-audio-vae-bf16.safetensors in the audio_vae slot.")

        if mode == "t2v" and images is not None:
            raise ValueError("LTX-2.5 i2v: mode=t2v but images is connected - "
                             "disconnect it or pick i2v.")
        if mode == "i2v" and images is None:
            raise ValueError("LTX-2.5 i2v: mode=i2v needs images connected.")

        for lora_name, strength in ((lora_1, lora_1_strength),
                                    (lora_2, lora_2_strength),
                                    (lora_3, lora_3_strength)):
            if lora_name not in (None, "none", ""):
                model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                    model, lora_name, strength)[0]
                logger.info("LTX-2.5 prep: LoRA %s chained @ %.2f", lora_name, strength)

        # ── stage-1 video stream at HALF the target resolution (the official
        # workflow's a/2 math nodes), core EmptyLTXVLatentVideo's geometry ──
        length = _align_length(length)
        stage_w, stage_h = width // 2, height // 2
        t_latent = video_latent_t(length)
        device = comfy.model_management.intermediate_device()
        video_samples = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             stage_h // VIDEO_SPATIAL_RATIO, stage_w // VIDEO_SPATIAL_RATIO],
            device=device)
        # Core shape: per-frame, broadcast over channels and space. Sampling
        # resizes masks, so this is NOT interchangeable with a full-size one.
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)

        held_t = 0
        if images is not None:
            images = _preprocess_images(images[:, :, :, :3], img_compression)
            held_t = _apply_i2v_hold(vae, images, video_samples, video_mask,
                                     image_strength, batch_size)

        # ── audio stream: empty, mask 1 (the model generates it) - the
        # workflow's LTXVEmptyLatentAudio + LTXVConcatAVLatent ──
        audio_shape, _ = audio_latent_shape(audio_vae, length, frame_rate, batch_size)
        audio = torch.zeros(audio_shape, device=video_samples.device)
        audio_mask = torch.ones_like(audio)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_samples, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        # The DiT's RoPE reads frame_rate off the conditioning; same call
        # core's LTXVConditioning makes.
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        logger.info("LTX-2.5 prep: stage 1 %dx%d (target %dx%d), video %s mask %s, "
                    "audio %s%s",
                    stage_w, stage_h, width, height,
                    tuple(video_samples.shape), tuple(video_mask.shape),
                    tuple(audio.shape),
                    ", image held @ %.2f over %d latent frame(s)" % (image_strength, held_t)
                    if images is not None else "")
        return (model, positive, negative, latent, frame_rate)


# ── sampling ────────────────────────────────────────────────────────────────

class LTXV25KSampler:
    """The official LTX-2.5 schedules through core's dual-CFG guider.

    The stock schedulers (simple/karras/...) do not reproduce the trained
    distilled schedule, and a distilled model on the wrong schedule looks like
    a broken model. The two presets are the workflow's two ManualSigmas
    strings verbatim, passed straight through - no resampling, no step count
    widget to get wrong.

    Sampling composes exactly what the workflow's SamplerCustomAdvanced +
    LTXVDualCFGGuider pair does: comfy-core's real Guider_LTXAVDualCFG
    (separate video/audio CFG on the packed AV latent - both 1.0 in the
    official workflow, the distilled bake is trained without CFG) driving
    CFGGuider.sample with the euler_ancestral sampler object.
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "LTX-2.5 KSampler (distilled) ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse',
                      'txt2img', 'img2img', 'dual cfg']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Sample a joint AV latent on the official LTX-2.5 schedules "
                   "(distilled 8-step or refine 3-step, euler_ancestral, dual "
                   "video/audio CFG 1.0).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "schedule": (list(LTX25_SIGMA_SETS), {"default": "distilled (8 steps)",
                             "tooltip": "distilled (8 steps) for the stage-1 pass, "
                                        "refine (3 steps) after LTXV25LatentUpscale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"default": "euler_ancestral"}),
                "video_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                              "step": 0.1,
                              "tooltip": "CFG on the video stream (core "
                                         "LTXVDualCFGGuider). 1.0 - the distilled "
                                         "bake is trained without CFG."}),
                "audio_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                              "step": 0.1,
                              "tooltip": "CFG on the audio stream. 1.0 official."}),
            },
        }

    def sample(self, model, positive, negative, latent_image, seed, schedule,
               sampler_name, video_cfg, audio_cfg):
        import comfy_extras.nodes_lt as nodes_lt

        sigmas = torch.tensor(LTX25_SIGMA_SETS[schedule], dtype=torch.float32)

        # Exactly what core SamplerCustomAdvanced.execute does around
        # guider.sample (nodes_custom_sampler.py:1040-1061), with the
        # workflow's LTXVDualCFGGuider composed in.
        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"],
            latent_image.get("downscale_ratio_spacial", None),
            latent_image.get("downscale_ratio_temporal", None))
        noise_mask = latent_image.get("noise_mask", None)

        guider = nodes_lt.Guider_LTXAVDualCFG(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(video_cfg, audio_cfg)

        noise = comfy.sample.prepare_noise(samples, seed,
                                           latent_image.get("batch_index", None))

        logger.info("LTX-2.5 sample: %s, %d steps from sigma %.4f, %s, "
                    "cfg v%.1f/a%.1f, mask %s",
                    schedule, len(sigmas) - 1, float(sigmas[0]), sampler_name,
                    video_cfg, audio_cfg,
                    "yes" if noise_mask is not None else "NONE")

        out = guider.sample(
            noise, samples, comfy.samplers.sampler_object(sampler_name), sigmas,
            denoise_mask=noise_mask, seed=seed,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED)
        out = out.to(comfy.model_management.intermediate_device())

        latent["samples"] = out
        latent.pop("noise_mask", None)
        return (latent,)


# ── latent upscale + decode ─────────────────────────────────────────────────

def _upsample_video_latent(latent, upscale_model, vae):
    """Port of core LTXVLatentUpsampler, restricted to the video branch.

    The upsample model only understands a single video-shaped tensor, not a
    joint AV one, so the latent must be split before calling it and rejoined
    after - calling it directly on the concatenated AV tensor does not error,
    it silently treats part of the audio latent as video channels. (Same split
    the official workflow wires by hand as LTXVSeparateAVLatent ->
    LTXVLatentUpsampler -> LTXVConcatAVLatent; same helper LTXV23LatentUpscale
    uses.)
    """
    samples = latent["samples"]
    video, audio = samples.unbind()

    device = upscale_model.load_device
    model = upscale_model.model
    model_dtype = upscale_model.model_dtype()
    input_dtype = video.dtype

    memory_required = math.prod(video.shape) * 3000.0  # matches core's estimate
    comfy.model_management.load_models_gpu([upscale_model], memory_required=memory_required)

    video = video.to(dtype=model_dtype, device=device)
    video = vae.first_stage_model.per_channel_statistics.un_normalize(video)
    video = model(video)
    video = vae.first_stage_model.per_channel_statistics.normalize(video)
    video = video.to(dtype=input_dtype, device=comfy.model_management.intermediate_device())
    audio = audio.to(video.device)

    out = latent.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out.pop("noise_mask", None)  # the upsampled latent has no held frames left
    return out


class LTXV25LatentUpscale:
    """Spatial x2 latent upscale between the distilled and refine passes -
    the official workflow's LTXVSeparateAVLatent -> LTXVLatentUpsampler ->
    LTXVImgToVideoInplace(1.0) -> LTXVConcatAVLatent chain in one node.

    The video half goes through the official spatial upscaler; the audio half
    passes through untouched. Wire the SAME first-frame image that went into
    LTXV25ImgToVideo into ``images`` here to re-hold it on the upscaled latent
    at ``image_strength`` 1.0 (the workflow's second LTXVImgToVideoInplace) -
    leave it disconnected for T2V, where the upscaled latent goes to the
    refine pass with no mask (core LTXVLatentUpsampler's own behavior).
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "LTX-2.5 Latent Upscale x2 ⚡"
    SEARCH_ALIASES = ['latent upscale', 'upscale', 'spatial upscaler', 'x2',
                      'two stage', 'refine']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    DESCRIPTION = ("x2 spatial latent upscale via the official LTX-2.5 spatial "
                   "upscaler, on the video half of the joint AV latent, with "
                   "the optional refine-pass first-frame re-hold @ 1.0. Follow "
                   "with LTXV25KSampler on 'refine (3 steps)'.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL", {
                    "tooltip": "ltx-2.5-latent-spatial-upscaler-x2-bf16 via core "
                               "LatentUpscaleModelLoader."}),
                "vae": ("VAE", {"tooltip": "Video VAE (for the latent statistics, "
                                "and the re-hold encode when images is wired)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The SAME first frame given to "
                           "LTXV25ImgToVideo - re-held on the upscaled latent at "
                           "image_strength before the refine pass (the official "
                           "recipe). Leave disconnected for T2V."}),
                "image_strength": ("FLOAT", {
                    "default": LTX25_REFINE_STRENGTH, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "images only. 1.0 is the official refine-pass value "
                               "(stage 1 held at 0.7)."}),
                "img_compression": ("INT", {
                    "default": LTX25_IMG_COMPRESSION, "min": 0, "max": 100,
                    "tooltip": "images only. Same LTXVPreprocess round-trip as the "
                               "prep node (official 18; 0 = off) - the workflow "
                               "feeds one preprocessed image to both holds."}),
            },
        }

    @torch.inference_mode()
    def upscale(self, latent, upscale_model, vae, images=None,
                image_strength=LTX25_REFINE_STRENGTH,
                img_compression=LTX25_IMG_COMPRESSION):
        upscaled = _upsample_video_latent(latent, upscale_model, vae)
        video, audio = upscaled["samples"].unbind()
        logger.info("LTX-2.5 latent upscale: %s -> %s",
                    tuple(latent["samples"].unbind()[0].shape), tuple(video.shape))

        if images is not None:
            batch_size = video.shape[0]
            video = video.clone()
            video_mask = torch.ones((batch_size, 1, video.shape[2], 1, 1),
                                    dtype=torch.float32, device=video.device)
            pixels = _preprocess_images(images[:, :, :, :3], img_compression)
            held_t = _apply_i2v_hold(vae, pixels, video, video_mask,
                                     image_strength, batch_size)
            upscaled["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
            upscaled["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (video_mask, torch.ones_like(audio)))
            logger.info("LTX-2.5 latent upscale: first frame re-held @ %.2f over "
                        "%d latent frame(s)", image_strength, held_t)
        return (upscaled,)


class LTXV25AVDecode:
    """Joint AV latent -> muxed VIDEO, in one node.

    Wraps core VAEDecodeTiled (video) + LTXVAudioVAEDecode (audio) +
    CreateVideo (mux), threading one ``fps`` through all - the plain-node
    version needs the same value typed into two different widgets that have no
    wire between them, and a mismatch there is a silent audio/video drift, not
    an error. Tile defaults are the official workflow's VAEDecodeTiled values
    (512, 64, 64, 16).
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "LTX-2.5 AV Decode ⚡"
    SEARCH_ALIASES = ['decode', 'decode latent', 'latent to video',
                      'latent to audio', 'video decode', 'audio decode']
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "decode"
    DESCRIPTION = "Decode a joint AV latent to a muxed VIDEO output."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE", {"tooltip": "Video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE."}),
                "fps": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0, "step": 0.01,
                        "tooltip": "Wire LTXV25ImgToVideo's frame_rate output here."}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 16, "min": 4, "max": 4096, "step": 4}),
            },
        }

    def decode(self, latent, vae, audio_vae, fps, tile_size=512, overlap=64,
               temporal_size=64, temporal_overlap=16):
        if _VideoInputImpl is None:
            raise RuntimeError(
                "This ComfyUI is too old to expose comfy_api.latest's VIDEO "
                "type; update ComfyUI or wire VAEDecodeTiled + "
                "LTXVAudioVAEDecode + CreateVideo by hand.")

        images, = nodes.VAEDecodeTiled().decode(
            vae, latent, tile_size, overlap, temporal_size, temporal_overlap)

        samples = latent["samples"]
        audio_latent = samples.unbind()[-1] if samples.is_nested else samples
        waveform = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
        sample_rate = int(audio_vae.first_stage_model.output_sample_rate)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        video = _VideoInputImpl.VideoFromComponents(
            _VideoTypes.VideoComponents(images=images, audio=audio,
                                        frame_rate=Fraction(fps)))
        logger.info("LTX-2.5 AV decode: %s frames @ %.2f fps, audio %s @ %d Hz",
                    tuple(images.shape), fps, tuple(waveform.shape), sample_rate)
        return (video,)


NODE_CLASS_MAPPINGS = {
    "LTXV25EmptyLatentAVBatch": LTXV25EmptyLatentAVBatch,
    "LTXV25ModelsLoader": LTXV25ModelsLoader,
    "LTXV25ImgToVideo": LTXV25ImgToVideo,
    "LTXV25KSampler": LTXV25KSampler,
    "LTXV25LatentUpscale": LTXV25LatentUpscale,
    "LTXV25AVDecode": LTXV25AVDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV25EmptyLatentAVBatch": LTXV25EmptyLatentAVBatch.TITLE,
    "LTXV25ModelsLoader": LTXV25ModelsLoader.TITLE,
    "LTXV25ImgToVideo": LTXV25ImgToVideo.TITLE,
    "LTXV25KSampler": LTXV25KSampler.TITLE,
    "LTXV25LatentUpscale": LTXV25LatentUpscale.TITLE,
    "LTXV25AVDecode": LTXV25AVDecode.TITLE,
}
