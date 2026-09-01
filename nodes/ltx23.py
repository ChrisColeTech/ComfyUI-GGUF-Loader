"""LTX-2.3 A/V nodes: kit loader, T2V/I2V/A2V/IA2V prep, distilled sampler.

  LTXV23ModelsLoader   DiT + Gemma-3 TE (+ dual projection) + video VAE + audio VAE
  LTXV23ImgToVideo     prompts, init latent and noise masks for every mode
  LTXV23KSampler       euler on the DMD / distilled / refine sigma schedules
  LTXV23RefineSampler  base pass -> spatial x2 latent upscale -> refine pass
  LTXV23AVDecode       joint AV latent -> muxed VIDEO (video + audio VAE decode)

ID-LoRA talking-head pipelines (e.g. the ``ltxv23_talking_head`` gallery
workflow) layer a distilled LoRA + an ID-LoRA onto the model and a reference
audio clip onto the conditioning before sampling. Both of those are already
correctly served by stock core nodes -- ``LoraLoaderModelOnly`` (chain it
twice: distilled strength ~0.5, then the ID-LoRA at strength ~1.0) and
``LTXVReferenceAudio`` -- so this module does not wrap them; only the pieces
core leaves as loose multi-node wiring (the two-stage refine sampler, and the
final decode+mux) get a home here.

The conditioning path mirrors comfy_extras/nodes_lt.py rather than
reinterpreting it, because every deviation is a way to get a plausible-looking
video that is subtly wrong:

  * the init latent is [B, 128, (L-1)//8+1, H//32, W//32] and the encoded image
    is written into its FIRST latent frames (core LTXVImgToVideo);
  * the i2v hold is a PER-FRAME mask of shape [B, 1, T, 1, 1] carrying
    1 - strength on the held frames. Core's shape, not a full-size mask -
    sampling resizes masks, and a mask that already matches the latent skips
    that path entirely, so the two are not interchangeable by inspection;
  * the joint AV latent and its mask are NestedTensor((video, audio)), which is
    the only form core's samplers and LTXVSeparateAVLatent understand;
  * frame_rate rides on the conditioning (the DiT's RoPE reads it) via
    node_helpers.conditioning_set_values, the same call LTXVConditioning makes.

Audio-to-video follows the IA2V recipe: the encoded reference becomes the audio
stream with a ZERO mask, so it is held while the DiT generates matching video.
A clip shorter than the video is zero-padded with mask 1 on the tail, so the
model generates the remainder.
"""
import logging
import math
import re
from fractions import Fraction

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes
import torch
import torch.nn.functional as F

try:
    from comfy_api.latest import InputImpl as _VideoInputImpl
    from comfy_api.latest import Types as _VideoTypes
except ImportError:  # pragma: no cover - older comfy without the video API
    _VideoInputImpl = _VideoTypes = None

logger = logging.getLogger(__name__)

LTX23_CATEGORY = "\U0001F916 CCTech/LTX-2.3"

# LTX-2 distilled denoising schedule (ltx_pipelines.utils.constants
# .DISTILLED_SIGMAS). The kit has the distill LoRA pre-fused, so this schedule
# at cfg 1.0 is the intended way to run it.
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                    0.909375, 0.725, 0.421875, 0.0]
# Stage-2 refine schedule from the official workflows (after a latent x2
# spatial upscale): 3 steps, cfg 1.0, euler.
REFINE_SIGMAS = [0.85, 0.725, 0.4219, 0.0]

# Schedules for a DMD-distilled bake (TenStrip LTX2.3_DMD_reshaped_r256 fused
# at 1.0). DMD replaces the distill LoRA rather than stacking with it, so these
# are alternatives to DISTILLED_SIGMAS, not refinements of it.
#
# DMD_SIGMAS is what 10Eros_10SNodes_I2V_DMD_v1.json actually runs: the
# distilled curve with 0.98125 dropped and 0.78 inserted in the 0.909->0.725
# gap. That workflow feeds it through EchoDMDSigmaRemap(interpolate), which is
# an identity function -- interpolating at the input value returns the input --
# so the list below is the effective schedule, no remap needed.
DMD_SIGMAS = [1.0, 0.99375, 0.9875, 0.975,
              0.909375, 0.78, 0.725, 0.421875, 0.0]
# The model card's own recommendation, a smooth descent that shares no interior
# anchors with the distilled curve.
DMD_CARD_SIGMAS = [1.0, 0.955, 0.893, 0.812, 0.715,
                   0.603, 0.482, 0.241, 0.121, 0.0]
# Stage-2 upscale pass for DMD, from the same workflow.
DMD_UPSCALE_SIGMAS = [0.92, 0.909375, 0.725, 0.421875, 0.0]

# Order matters: this is the dropdown order, and the default sits first.
SIGMA_SETS = {
    "dmd (8 steps)": DMD_SIGMAS,
    "dmd card (9 steps)": DMD_CARD_SIGMAS,
    "dmd upscale (4 steps)": DMD_UPSCALE_SIGMAS,
    "distilled (8 steps)": DISTILLED_SIGMAS,
    "refine (3 steps)": REFINE_SIGMAS,
}

# LTXVImgToVideoInplace's value in the official I2V/IA2V first pass.
I2V_STRENGTH = 0.7
FPS = 24.0
_EXPECTED_LAYERS = 48

# Same geometry family as LTX-2.5; core builds exactly these numbers.
VIDEO_LATENT_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8


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


def _snap_target_keep_aspect(width, height, unit, label, factor):
    """Snap a typed target size to the pixel grid ``unit`` (32 x the guide's
    latent_downscale_factor), choosing the candidate whose ASPECT is closest
    to what was typed - never by flooring each side independently, which at
    coarse units (128px at factor 2) can shear the aspect and center-crop
    the inputs hard.

    For each of floor/ceil of the width, the aspect-matching height is
    rounded to the grid; ties on aspect error break toward the pixel count
    closest to the request. Raises only when the target can't fit one guide
    cell per side at all.
    """
    if width < unit or height < unit:
        raise ValueError(
            f"{label}: target {width}x{height} is too small for "
            f"latent_downscale_factor {factor} - needs at least {unit}px "
            f"per side.")
    if width % unit == 0 and height % unit == 0:
        return width, height
    aspect = width / height
    candidates = set()
    for w in {width // unit * unit, -(-width // unit) * unit}:
        if w < unit:
            continue
        h_ideal = w / aspect
        for h in {int(h_ideal // unit) * unit, int(-(-h_ideal // unit)) * unit,
                  height // unit * unit, -(-height // unit) * unit}:
            if h >= unit:
                candidates.add((w, h))
    def _score(wh):
        w, h = wh
        # aspect error first, then closest pixel count, then the smaller
        # size (deterministic on exact ties; cheaper render, never a
        # surprise VRAM jump above the request).
        return (abs((w / h) - aspect) / aspect,
                abs(w * h - width * height),
                w * h)
    return min(candidates, key=_score)


def distilled_sigma_schedule(steps, denoise=1.0, sigmas=None):
    """The LTX-2 distilled sigmas, resampled to ``steps``.

    At the trained 8 steps this is exactly the reference curve; other counts
    are a linear resample of it. ``denoise`` slices the head off, matching how
    KSampler treats denoise.
    """
    base = torch.tensor(sigmas if sigmas is not None else DISTILLED_SIGMAS,
                        dtype=torch.float32)
    steps = max(1, int(steps))
    if steps + 1 == len(base):
        out = base
    else:
        idx = torch.linspace(0.0, len(base) - 1.0, steps + 1)
        lo = idx.floor().long().clamp(0, len(base) - 2)
        frac = idx - lo
        out = base[lo] * (1.0 - frac) + base[lo + 1] * frac
    denoise = min(max(float(denoise), 0.0), 1.0)
    if denoise < 1.0:
        out = out[int(round(steps * (1.0 - denoise))):]
    return out


# ── loading ─────────────────────────────────────────────────────────────────

# Gemma-3 12B + projections costs ~24 GB and many seconds to build, and comfy
# re-runs loader nodes on every prompt edit. Cache the most recent pair; the
# CLIP owns its ModelPatcher so comfy still manages its VRAM.
_ENCODER_CACHE = {}


def _load_ltxv_clip(te_name, projections_name):
    key = (te_name, projections_name)
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]

    from ..loader import gguf_clip_loader
    from ..ops import GGMLOps

    te_path = folder_paths.get_full_path("clip", te_name) \
        or folder_paths.get_full_path_or_raise("text_encoders", te_name)
    proj_path = folder_paths.get_full_path("clip", projections_name) \
        or folder_paths.get_full_path_or_raise("text_encoders", projections_name)

    proj_sd, _ = comfy.utils.load_torch_file(proj_path, return_metadata=True)
    if "text_embedding_projection.audio_aggregate_embed.bias" not in proj_sd:
        raise ValueError(
            f"{projections_name} carries no text_embedding_projection tensors - "
            "it is not an LTX-2 projections file. Use the *_projections.safetensors "
            "that ships with the kit."
        )

    if te_path.lower().endswith(".gguf"):
        te_sd = gguf_clip_loader(te_path)
        te_options = {"custom_operations": GGMLOps(),
                      "initial_device": comfy.model_management.text_encoder_offload_device()}
    else:
        te_sd, _ = comfy.utils.load_torch_file(te_path, return_metadata=True)
        te_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=comfy.sd.CLIPType.LTXV,
        state_dicts=[te_sd, proj_sd],
        model_options=te_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    del te_sd

    projection = type(getattr(clip.cond_stage_model,
                              "text_embedding_projection", None)).__name__
    if projection != "DualLinearProjection":
        raise ValueError(
            f"{te_name} + {projections_name} built a {projection}, not "
            "DualLinearProjection - the DiT's embeddings connectors cannot "
            "consume it."
        )

    _ENCODER_CACHE.clear()
    _ENCODER_CACHE[key] = clip
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
                         f"video VAE); audio_vae wants *_audio_vae.safetensors.")
    if not want_audio and is_audio:
        raise ValueError(f"{vae_name} is the audio VAE; video_vae wants "
                         f"*_video_vae.safetensors.")
    return vae


class LTXV23ModelsLoader:
    """Load the whole LTX-2.3 A/V kit in one node.

    DiT GGUFs stay quantized (dequantized per layer at forward time). Outputs
    are plain comfy MODEL / CLIP / VAE objects, so they compose with comfy's
    own LTXV nodes as well as these.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Models Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'audio vae', 'video vae', 'load clip', 'text encoder']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the LTX-2.3 A/V components (DiT, Gemma-3 text encoder "
                   "with dual projection, video VAE, audio VAE).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "LTX-2.3 A/V DiT from models/diffusion_models. "
                               "GGUF stays quantized."}),
                "text_encoder_name": (_clip_filename_list(), {
                    "tooltip": "Gemma-3 12B text encoder from models/text_encoders."}),
                "projections_name": (_clip_filename_list(), {
                    "tooltip": "The kit's *_projections.safetensors — the dual "
                               "4096/2048 projection paired with the encoder."}),
                "video_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 video VAE."}),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 audio VAE + vocoder."}),
            },
        }

    def load(self, unet_name, text_encoder_name, projections_name,
             video_vae_name, audio_vae_name):
        if unet_name.lower().endswith(".gguf"):
            from .gguf import UnetLoaderGGUF
            model, = UnetLoaderGGUF().load_unet(unet_name)
        else:
            model = comfy.sd.load_diffusion_model(
                folder_paths.get_full_path("unet", unet_name)
                or folder_paths.get_full_path_or_raise("unet", unet_name))

        cfg = model.model.model_config.unet_config
        if cfg.get("image_model") != "ltxav" or cfg.get("num_layers") != _EXPECTED_LAYERS:
            raise ValueError(
                f"{unet_name} built as image_model={cfg.get('image_model')} "
                f"num_layers={cfg.get('num_layers')}, expected "
                f"ltxav/{_EXPECTED_LAYERS}. Not an LTX-2.3 A/V checkpoint."
            )
        logger.info("LTX-2.3: %s -> ltxav/%d layers (%.2f GiB stored)",
                    unet_name, cfg.get("num_layers"), model.model_size() / 1024 ** 3)

        return (model,
                _load_ltxv_clip(text_encoder_name, projections_name),
                _load_vae(video_vae_name, want_audio=False),
                _load_vae(audio_vae_name, want_audio=True))


# ── conditioning / latent prep ──────────────────────────────────────────────

def _encode_reference_audio(audio_vae, audio, duration_s):
    """Encode an AUDIO reference into a held audio latent [B, C, T, bins]."""
    fsm = getattr(audio_vae, "first_stage_model", None)
    if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
        raise ValueError("audio_vae is not an LTX audio VAE; reference_audio "
                         "needs the kit's *_audio_vae.safetensors.")

    waveform = audio["waveform"][0]          # [C, T] - one reference
    sr = audio["sample_rate"]
    max_samples = int(duration_s * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    comfy.model_management.load_models_gpu(
        [audio_vae.patcher],
        force_full_load=getattr(audio_vae, "disable_offload", False))
    # The VAE is on the compute device now; the waveform must follow it or
    # torch raises "Input type (CPU) and weight type (CUDA) should be the same".
    waveform = waveform.to(device=audio_vae.device,
                           dtype=getattr(audio_vae, "vae_dtype", torch.float32))

    latent = fsm.encode(waveform.unsqueeze(0), sample_rate=sr)   # [1,C,T,bins]
    latent = latent.to(comfy.model_management.intermediate_device()).float()
    return latent, torch.zeros_like(latent)  # 0 = held: audio drives the video


def _match_batch(t, n):
    """Bring ``t`` to exactly ``n`` along dim 0 by tiling, then truncating.

    Deliberately idempotent: calling it twice is the same as calling it once.
    ``Tensor.repeat`` is not -- it multiplies -- so a value that has already
    been expanded once must never be repeated again.
    """
    have = t.shape[0]
    if have == n:
        return t
    if have > n:
        return t.narrow(0, 0, n)
    reps = -(-n // have)                      # ceil
    return t.repeat(reps, *([1] * (t.dim() - 1))).narrow(0, 0, n)


def _fit_audio_latent(audio, mask, target_shape):
    """Trim or zero-pad the audio latent along time to ``target_shape``.

    Port of core LTXVConcatAVLatent.fit_audio: a padded tail keeps mask 1 so
    the model generates it, which is what a clip shorter than the video means.

    Time only. The pad is allocated at the audio's own batch rather than
    ``target_shape[0]`` so this cannot silently expand the batch on the pad
    path while leaving it alone on the trim path -- that asymmetry is what
    made a following .repeat() produce batch_size**2 for short clips only.
    """
    dim, length = 2, target_shape[2]
    if audio.shape[dim] > length:
        audio = audio.narrow(dim, 0, length)
        if mask is not None:
            mask = mask.narrow(dim, 0, length)
    elif audio.shape[dim] < length:
        pad_shape = [audio.shape[0]] + list(target_shape[1:])
        pad = torch.zeros(pad_shape, device=audio.device, dtype=audio.dtype)
        pad[:, :, :audio.shape[dim]] = audio
        if mask is not None:
            pmask = torch.ones_like(pad)
            pmask[:, :, :mask.shape[dim]] = mask
            mask = pmask
        audio = pad
    return audio, mask


class LTXV23ImgToVideo:
    """Prompts + init latent for LTX-2.3: T2V, I2V, A2V and IA2V in one node.

      no image, no reference_audio -> text-to-video
      image                        -> image-to-video (first frames held at
                                      ``image_strength``)
      reference_audio              -> audio-to-video (audio stream held, video
                                      generated to match: lip sync, Foley)
      image + reference_audio      -> image-audio-to-video

    Feed the outputs into the LTX-2.3 KSampler; split its result with core
    LTXVSeparateAVLatent, decode video with VAE Decode and audio with
    LTXVAudioVAEDecode.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Img/Audio to Video ⚡"
    SEARCH_ALIASES = ['image to video', 'audio to video', 'img2vid', 'empty latent', 'video conditioning', 'text to video']
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and noise masks for LTX-2.3 "
                   "T2V/I2V/A2V/IA2V. Feed into the LTX-2.3 KSampler.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the scene and its motion. "
                                                 "A caption, not an instruction."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 121, "min": 9,
                                   "max": nodes.MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Frames; 8k+1 tiles exactly (9, 97, 121...). "
                                              "Ignored when length_from_audio is on."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0,
                                         "step": 0.01,
                                         "tooltip": "24 is the LTX-2 convention. Match "
                                                    "this in CreateVideo or playback drifts."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First frame. Resized and CENTER-CROPPED "
                                               "to width x height here - do not scale it "
                                               "upstream."}),
                "reference_audio": ("AUDIO",),
                "image_strength": ("FLOAT", {
                    "default": I2V_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "i2v only. How much of the init image to keep. 0.7 is "
                               "the official value; 1.0 locks the first frames hard."}),
                "length_from_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "With reference_audio: size the video to the clip."}),
            },
        }

    @torch.inference_mode()
    def prepare(self, clip, vae, audio_vae, prompt, negative_prompt, width,
                height, length, frame_rate, batch_size, image=None,
                reference_audio=None, image_strength=I2V_STRENGTH,
                length_from_audio=True):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use the kit's "
                             "*_audio_vae.safetensors in the audio_vae slot.")

        # ── audio stream ──
        audio = audio_mask = None
        if reference_audio is not None:
            seconds = reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
            if length_from_audio:
                length = _align_length(seconds * frame_rate + 1)
                logger.info("LTX-2.3 a2v: %.2fs of audio -> %d frames @ %.2f fps",
                            seconds, length, frame_rate)
            audio, audio_mask = _encode_reference_audio(
                audio_vae, reference_audio, length / frame_rate)

        # ── video stream, core LTXVImgToVideo's geometry ──
        length = _align_length(length)
        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        device = comfy.model_management.intermediate_device()
        video = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO],
            device=device)

        # Core shape: per-frame, broadcast over channels and space. Sampling
        # resizes masks, so this is NOT interchangeable with a full-size one.
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)
        if image is not None:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            t = _match_batch(vae.encode(pixels[:, :, :, :3]), batch_size)
            video[:, :, :t.shape[2]] = t.to(video.device, video.dtype)
            video_mask[:, :, :t.shape[2]] = 1.0 - image_strength

        # ── audio geometry: empty, or the reference fitted to the video ──
        n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
        channels = int(getattr(audio_vae, "latent_channels", fsm.latent_channels))
        target = [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)]
        if audio is None:
            audio = torch.zeros(target, device=video.device)
            audio_mask = torch.ones_like(audio)
        else:
            audio, audio_mask = _fit_audio_latent(audio, audio_mask, target)
            audio = _match_batch(audio, batch_size).to(video.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video.device)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        # The DiT's RoPE reads frame_rate off the conditioning; same call
        # core's LTXVConditioning makes.
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        logger.info("LTX-2.3 prep: video %s mask %s, audio %s mask %s%s%s",
                    tuple(video.shape), tuple(video_mask.shape),
                    tuple(audio.shape), tuple(audio_mask.shape),
                    ", image held @ %.2f" % image_strength if image is not None else "",
                    ", audio locked" if reference_audio is not None else "")
        return (positive, negative, latent)


class LTXV23VidToVideo:
    """Prompts + init latent for LTX-2.3: T2V/I2V/V2V, with the IC-LoRA
    task adapter loaded right in this node - no external
    LoraLoaderModelOnly wiring, no separate "is X attached" boolean toggles.
    The mechanism is driven purely by whether its selector is set
    (a real filename) or left at "none" - matching this node's `model`
    input/output design (LoRA-loaded model out, or the original untouched
    model straight through, if nothing was selected).

    `mode` (required) declares which of the three base behaviors this call
    is: `t2v` (images/video must be disconnected - pure text-to-video),
    `i2v` (`images` required - ordinary first-frame hold, VAE-encoded and
    held at `image_strength` via the noise mask), or `v2v` (`video`
    required - IC-LoRA guide injection, see `ic_lora` below). `i2v`/`v2v`
    can still layer on top of each other exactly as before (image hold +
    video guide in the same call) - `mode` only says which one is the
    PRIMARY, required source; the other stays optional and combinable.

    `ic_lora` (optional selector, "none" = off) loads an IC-LoRA (in-context
    LoRA) task adapter directly onto `model` inside this node (comfy-core's
    real `LoraLoaderModelOnly.load_lora_model_only`, not reimplemented) at
    `ic_lora_strength`, then injects `video`'s frames as extra "guide"
    tokens the model attends to at the SAME timeline position as what it's
    generating (comfy-core's keyframe_idxs RoPE mechanism,
    comfy_extras/nodes_lt.py's LTXVAddGuide) - a genuinely different thing
    from `images`'s first-frame hold, and from Krea2Img2Img's partial-denoise
    img2img. `ic_lora="none"` (default): `video`, if connected, is only
    used for length/frame_rate/original audio, ignored for guidance. When
    `video` is connected, `length`/`frame_rate` are taken FROM it
    (overriding the widgets, logged); `width`/`height` always come from the
    widgets regardless of source, same as `images`. `keep_original_audio`
    (default on, only relevant with `video`) keeps the source clip's own
    audio unchanged in the output using this file's own already-proven
    `_encode_reference_audio`/`_fit_audio_latent` helpers - mechanically
    identical to `reference_audio` below, just fed the source video's own
    audio track instead of a separate clip. Off = the model generates new
    audio from scratch instead.

    `reference_audio` (optional, mutually exclusive in effect with
    `video`+`keep_original_audio` - the LAST one prepared wins if both are
    wired, though wiring both makes little sense) - LTXV23ImgToVideo's own
    A2V path, unchanged, for driving generation from a voice/sound clip
    with no source video at all.

    The returned `model` is the LoRA-loaded clone (or the original,
    untouched, if `ic_lora` is "none"); always take `model` from THIS
    node's output, not the original upstream model, whenever the selector
    is used. `ic_lora` lists comfy's real `loras` folder_paths category -
    the same place a plain `LoraLoaderModelOnly` looks.

    This node calls comfy-core's real LTXVAddGuide.get_latent_index()/
    append_keyframe() directly (comfy_extras.nodes_lt, always present
    regardless of any custom_nodes pack's state) rather than re-deriving the
    RoPE coordinate math by hand - the same call the official Lightricks
    IC-LoRA nodes make internally (ComfyUI-LTXVideo/iclora.py), just wired
    into this pack's own joint-AV-latent convention
    (comfy.nested_tensor.NestedTensor((video, audio)), matching
    LTXV23ImgToVideo) instead of the ~10-node chain the official example
    workflows wire by hand. `frame_idx` is always 0 (guide spans the whole
    clip from the start) - the only sensible mode for a plain vid2vid node;
    an offset guide needs the raw core/Lightricks nodes directly.

    Feed the outputs into LTXV23KSampler UNCHANGED - it already samples any
    joint AV latent + noise mask generically, so no custom sampler is
    needed here. Then LTXV23CropVideoGuide before LTXV23AVDecode to strip
    the reference frames back out (a no-op if `video`/`ic_lora` were never
    used).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Video to Video (IC-LoRA) ⚡"
    SEARCH_ALIASES = ['video to video', 'vid2vid', 'v2v', 'ic-lora', 'ic lora',
                       'video guide', 'video conditioning', 'image to video', 'img2vid']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "frame_rate")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts and init latent for LTX-2.3 T2V/I2V/V2V - the "
                   "IC-LoRA task adapter loads right in the node, no "
                   "external LoraLoaderModelOnly wiring needed. Feed the "
                   "outputs into LTXV23KSampler, then LTXV23CropVideoGuide "
                   "before LTXV23AVDecode.")

    @classmethod
    def INPUT_TYPES(s):
        lora_choices = ["none"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "mode": (["t2v", "i2v", "v2v"], {"default": "t2v",
                          "tooltip": "Which base behavior this call is. t2v: images/video "
                                     "must be disconnected. i2v: images required (video may "
                                     "still layer on top as an IC-LoRA guide). v2v: video "
                                     "required (images may still layer on top as a first-"
                                     "frame hold)."}),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "With ic_lora set: describe the OUTPUT "
                                                 "you want - most IC-LoRAs are trained on an "
                                                 "instruction-style caption describing the "
                                                 "transformed result. Otherwise: describe "
                                                 "the scene and its motion, a caption not an "
                                                 "instruction."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 121, "min": 9,
                                   "max": nodes.MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Frames; 8k+1 tiles exactly. Ignored (taken "
                                              "from the clip instead) when video is "
                                              "connected."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0, "step": 0.01,
                                         "tooltip": "Ignored (taken from the clip instead) "
                                                    "when video is connected."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "First frame(s) for image-to-video (ordinary "
                                    "i2v hold, independent of video/ic_lora below) - "
                                    "resized and CENTER-CROPPED to width x height."}),
                "image_strength": ("FLOAT", {
                    "default": I2V_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "images only. How much of the init image(s) to keep. 0.7 is "
                               "the official value; 1.0 locks the first frames hard."}),
                "video": ("VIDEO", {"tooltip": "Source clip for IC-LoRA video-to-video "
                                    "(see ic_lora) and/or its original audio (see "
                                    "keep_original_audio). Sets length/frame_rate from itself."}),
                "ic_lora": (lora_choices, {"default": "none",
                            "tooltip": "video only. The IC-LoRA task adapter (beard removal, "
                                       "HDR grading, motion tracking, ...) - loaded onto model "
                                       "HERE (no external LoraLoaderModelOnly needed) at "
                                       "ic_lora_strength, then drives the actual vid2vid guide-"
                                       "injection mechanism. \"none\" = video is used only for "
                                       "length/frame_rate/original audio, ignored for guidance "
                                       "- useful for A/B-ing whether the IC-LoRA is doing "
                                       "anything."}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                     "step": 0.01,
                                     "tooltip": "ic_lora only. Same as LoraLoaderModelOnly's "
                                                "strength_model."}),
                "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "ic_lora only. How strongly the reference is "
                                              "held. 1.0 = fully held (official default)."}),
                "keep_original_audio": ("BOOLEAN", {"default": True,
                               "tooltip": "video only. On = output keeps the source clip's "
                                          "own audio unchanged. Off = the model generates "
                                          "new audio from scratch instead."}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0,
                                            "step": 1.0,
                                            "tooltip": "ic_lora only. Only for IC-LoRAs "
                                                       "trained on a downscaled reference "
                                                       "grid (rare - check the LoRA's model "
                                                       "card / reference_downscale_factor "
                                                       "metadata; most, including every "
                                                       "official example, use 1.0)."}),
                "reference_audio": ("AUDIO", {"tooltip": "Drive generation from a voice/sound "
                                              "clip with no source video (LTXV23ImgToVideo's "
                                              "A2V path). Not meant to be combined with "
                                              "video+keep_original_audio."}),
                "length_from_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "With reference_audio and no video: size the video to the clip."}),
            },
        }

    @torch.inference_mode()
    def prepare(self, model, clip, vae, audio_vae, mode, prompt, negative_prompt, width, height,
                length, frame_rate, batch_size, images=None, image_strength=I2V_STRENGTH,
                video=None, ic_lora="none", ic_lora_strength=1.0, guide_strength=1.0,
                keep_original_audio=True, latent_downscale_factor=1.0, reference_audio=None,
                length_from_audio=True):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use the kit's "
                             "*_audio_vae.safetensors in the audio_vae slot.")

        ic_lora_attached = ic_lora not in (None, "none", "")

        if mode == "t2v" and video is not None:
            raise ValueError("LTX-2.3 v2v: mode=t2v but video is connected - disconnect it "
                             "or pick v2v.")
        if mode == "t2v" and images is not None:
            raise ValueError("LTX-2.3 v2v: mode=t2v but images is connected - disconnect it "
                             "or pick i2v.")
        if mode == "i2v" and images is None:
            raise ValueError("LTX-2.3 v2v: mode=i2v needs images connected.")
        if mode == "v2v" and video is None:
            raise ValueError("LTX-2.3 v2v: mode=v2v needs video connected.")

        if ic_lora_attached:
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, ic_lora, ic_lora_strength)[0]

        video_frames = video_audio = None
        if video is not None:
            components = video.get_components()
            video_frames = components.images
            video_audio = components.audio
            src_length = _align_length(video_frames.shape[0])
            src_fps = float(components.frame_rate)
            if src_length != _align_length(length) or src_fps != frame_rate:
                logger.info("LTX-2.3 v2v: video connected - length %d -> %d, "
                           "frame_rate %.2f -> %.2f (taken from the clip)",
                           length, src_length, frame_rate, src_fps)
            length, frame_rate = src_length, src_fps
        elif reference_audio is not None and length_from_audio:
            seconds = reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
            length = _align_length(seconds * frame_rate + 1)
            logger.info("LTX-2.3 a2v: %.2fs of audio -> %d frames @ %.2f fps",
                        seconds, length, frame_rate)
        else:
            length = _align_length(length)

        # The latent grid needs whole 32px cells (core's own arithmetic); with
        # latent_downscale_factor > 1 the cell count must also divide by the
        # factor. Snap to the grid size whose SHAPE is closest to what was
        # typed - flooring each side independently distorts the aspect (at
        # factor 2 the steps are coarse; 728x1296 floored to 640x1280 turned
        # 0.56 into 0.50 and center-cropped everything hard). ASPECT first,
        # then closest pixel count; log the effective size.
        guide_factor = (max(1, int(latent_downscale_factor))
                        if video is not None and ic_lora_attached else 1)
        eff_w, eff_h = _snap_target_keep_aspect(
            width, height, VIDEO_SPATIAL_RATIO * guide_factor, "LTX-2.3 v2v",
            latent_downscale_factor)
        if (eff_w, eff_h) != (width, height):
            logger.info(
                "LTX-2.3 v2v: target %dx%d -> effective %dx%d (closest "
                "aspect-preserving latent grid%s; the typed size is not "
                "producible on this grid)",
                width, height, eff_w, eff_h,
                " aligned for downscale_factor %d" % guide_factor
                if guide_factor > 1 else "")
            width, height = eff_w, eff_h

        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        device = comfy.model_management.intermediate_device()
        video_samples = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO], device=device)
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)
        if images is not None:
            pixels = comfy.utils.common_upscale(
                images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            t = _match_batch(vae.encode(pixels[:, :, :, :3]), batch_size)
            video_samples[:, :, :t.shape[2]] = t.to(video_samples.device, video_samples.dtype)
            video_mask[:, :, :t.shape[2]] = 1.0 - image_strength

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # Source-video guide append: the task IC-LoRA's conditioning - the
        # clip's frames become clean guide tokens the model attends to at
        # matching timeline positions.
        if video is not None and ic_lora_attached:
            import comfy_extras.nodes_lt as nodes_lt

            scale_factors = vae.downscale_index_formula
            time_scale = scale_factors[0]
            n = video_frames.shape[0]
            guide_frames = video_frames[:((n - 1) // time_scale) * time_scale + 1]
            target_w = int(width / latent_downscale_factor)
            target_h = int(height / latent_downscale_factor)
            guide_pixels = comfy.utils.common_upscale(
                guide_frames.movedim(-1, 1), target_w, target_h, "bilinear", "center"
            ).movedim(1, -1)[:, :, :, :3]
            guide_latent = _match_batch(vae.encode(guide_pixels), batch_size)

            guide_mask = None
            if latent_downscale_factor > 1:
                # width/height were snapped to a factor-compatible latent grid
                # above, so the guide always lands on whole 32px cells here.
                guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(
                    guide_latent, latent_downscale_factor)

            frame_idx, _ = nodes_lt.LTXVAddGuide.get_latent_index(
                positive, t_latent, guide_latent.shape[2], 0, scale_factors,
                latent_shape=video_samples.shape)
            positive, negative, video_samples, video_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, video_samples, video_mask, guide_latent,
                guide_strength, scale_factors, guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor, causal_fix=True)
            logger.info("LTX-2.3 v2v: guide %s appended @ strength %.2f (frame_idx=%d)",
                        tuple(guide_latent.shape), guide_strength, frame_idx)
        elif video is not None:
            logger.info("LTX-2.3 v2v: ic_lora=none - video used for length/frame_rate/"
                        "held audio only, ignored for guidance")

        n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
        channels = int(getattr(audio_vae, "latent_channels", fsm.latent_channels))
        target = [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)]
        if video is not None and keep_original_audio and video_audio is not None:
            audio_latent, audio_mask = _encode_reference_audio(
                audio_vae, video_audio, length / frame_rate)
            audio_latent, audio_mask = _fit_audio_latent(audio_latent, audio_mask, target)
            audio_latent = _match_batch(audio_latent, batch_size).to(video_samples.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video_samples.device)
        elif video is None and reference_audio is not None:
            audio_latent, audio_mask = _encode_reference_audio(
                audio_vae, reference_audio, length / frame_rate)
            audio_latent, audio_mask = _fit_audio_latent(audio_latent, audio_mask, target)
            audio_latent = _match_batch(audio_latent, batch_size).to(video_samples.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video_samples.device)
        else:
            audio_latent = torch.zeros(target, device=video_samples.device)
            audio_mask = torch.ones_like(audio_latent)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_samples, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }
        logger.info("LTX-2.3 v2v prep: video %s mask %s, audio %s mask %s%s%s",
                    tuple(video_samples.shape), tuple(video_mask.shape),
                    tuple(audio_latent.shape), tuple(audio_mask.shape),
                    ", image held @ %.2f" % image_strength if images is not None else "",
                    ", audio held" if (video is not None and keep_original_audio and video_audio is not None)
                    or (video is None and reference_audio is not None) else "")
        return (model, positive, negative, latent, frame_rate)


# ── Stage 1: masked removal (LTX-2.3 in/outpainting IC-LoRA) ──
#
# Ported from D:\Projects\Wan2GP-main\models\ltx2\inpainting.py +
# ltx2.py/ltx2_handler.py orchestration (read directly, full trace this
# session - not guessed). The trained recipe, exactly:
#   1. dilate the per-frame mask by 5px (max-pool),
#   2. paint the masked region chroma green #66FF00 (102,255,0) - the
#      in/outpainting IC-LoRA learned "green = regenerate this",
#   3. green-pad the tail to the full frame count,
#   4. append the green-filled video as CLEAN guide tokens at frame_idx 0,
#      strength exactly 1.0 (Wan2GP validates this in its UI). The user
#      mask is NOT a denoise mask during sampling (masking_strength is
#      force-zeroed in Wan2GP) - the LoRA does the regeneration,
#   5. after decode, composite the generated result back over the original
#      pixels with a multi-level Laplacian pyramid blend (7 levels, mask
#      softened by a low-res dilation of 6 at a 64px long side, source
#      sanitized inside the mask) so the seam doesn't show.
# Steps 1-4 are LTXV23RemovePerson (a conditioning-prep node shaped like
# LTXV23VidToVideo); step 5 is LTXV23MaskBlend (post-decode, IMAGE-space).

LTX2_MASKED_CONTROL_VIDEO_PAD_RGB = (102, 255, 0)   # ltx2_runtime.py, "#66FF00"
LTX2_INPAINT_MASK_DILATION = 5                       # LTX2_INPAINTING_PREPROCESS_MASK_DILATION
LTX2_INPAINT_LAPLACIAN_MASK_LOW_RES_DILATION = 6     # LTX2_INPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION
LTX2_LAPLACIAN_MASK_LOW_RES_LONG_SIDE = 64           # LTX2_LAPLACIAN_BLEND_MASK_LOW_RES_LONG_SIDE


def _ltx2_dilate_mask(mask_t1hw, radius):
    """Wan2GP's _apply_ltx2_inpaint_preprocess_dilation, mask half: max-pool
    dilation with kernel 2r+1. mask is [T,1,H,W] float 0..1."""
    if radius <= 0:
        return mask_t1hw
    return F.max_pool2d(mask_t1hw.float().clamp(0.0, 1.0), kernel_size=radius * 2 + 1,
                        stride=1, padding=radius).clamp(0.0, 1.0)


def _ltx2_greenfill(frames_thwc, mask_t1hw):
    """Paint masked pixels chroma green. frames are comfy IMAGE [T,H,W,C]
    float 0..1 (Wan2GP paints the same color in its own -1..1/uint8
    conventions - (102,255,0)/255 here is the identical color)."""
    color = torch.tensor(LTX2_MASKED_CONTROL_VIDEO_PAD_RGB, device=frames_thwc.device,
                         dtype=frames_thwc.dtype) / 255.0
    mask_thw1 = mask_t1hw.permute(0, 2, 3, 1).to(device=frames_thwc.device)
    return torch.where(mask_thw1 > 0, color.view(1, 1, 1, 3), frames_thwc[..., :3])


def _ltx2_green_tail_pad(frames_thwc, mask_t1hw, target_frames):
    """Wan2GP's _pad_ltx2_masked_control_video_tail: pad the control video
    with solid green frames (and the mask with ones) up to target_frames."""
    t = frames_thwc.shape[0]
    if t < target_frames:
        color = torch.tensor(LTX2_MASKED_CONTROL_VIDEO_PAD_RGB, device=frames_thwc.device,
                             dtype=frames_thwc.dtype) / 255.0
        pad = color.view(1, 1, 1, 3).expand(target_frames - t, frames_thwc.shape[1],
                                            frames_thwc.shape[2], 3)
        frames_thwc = torch.cat([frames_thwc, pad], dim=0)
    if mask_t1hw.shape[0] < frames_thwc.shape[0]:
        pad = torch.ones((frames_thwc.shape[0] - mask_t1hw.shape[0], *mask_t1hw.shape[1:]),
                         device=mask_t1hw.device, dtype=mask_t1hw.dtype)
        mask_t1hw = torch.cat([mask_t1hw, pad], dim=0)
    return frames_thwc, mask_t1hw


def _ltx2_resize_long_side(tensor, long_side, mode):
    height, width = tensor.shape[-2:]
    current = max(height, width)
    if current == long_side:
        return tensor
    scale = long_side / current
    size = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


def _ltx2_low_res_mask_dilation(mask_c_thw, radius, long_side=LTX2_LAPLACIAN_MASK_LOW_RES_LONG_SIDE):
    """Wan2GP's _apply_low_res_mask_dilation: soften/expand the blend mask by
    dilating a low-res copy and upsampling back - gives the pyramid blend a
    wide soft skirt around the mask instead of a hard edge."""
    if radius <= 0:
        return mask_c_thw
    original_size = mask_c_thw.shape[-2:]
    low = _ltx2_resize_long_side(mask_c_thw.float(), long_side, "bilinear")
    low = F.max_pool2d(low, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return F.interpolate(low, size=original_size, mode="bilinear", align_corners=False)


def _ltx2_laplacian_pyramid_blend(generated_tchw, source_tchw, mask_t1hw, levels=7,
                                  mask_low_res_dilation=0):
    """Faithful port of Wan2GP's _laplacian_pyramid_blend (inpainting.py:121-150),
    operating on [T,C,H,W] float 0..1 (comfy's unit range - Wan2GP's own
    _to_unit_video conversion from its -1..1 convention is already done by
    virtue of comfy IMAGE being unit-range; the math from that point on is
    identical). Blends generated over source inside the mask, per pyramid
    level, so low frequencies transition over a wide area and high
    frequencies stay crisp - no visible seam at the mask boundary."""
    generated = generated_tchw.float().clamp(0.0, 1.0).cpu()
    source = source_tchw.float().clamp(0.0, 1.0).cpu()
    mask = mask_t1hw.float().clamp(0.0, 1.0).cpu()
    mask = _ltx2_low_res_mask_dilation(mask, mask_low_res_dilation).clamp(0.0, 1.0)
    levels = max(1, min(int(levels), int(math.log2(max(2, min(generated.shape[-2:])))) - 2))

    def gaussian_pyramid(x):
        pyramid = [x]
        for _ in range(1, levels):
            if min(pyramid[-1].shape[-2:]) <= 8:
                break
            pyramid.append(F.interpolate(pyramid[-1], scale_factor=0.5, mode="bilinear",
                                         align_corners=False, recompute_scale_factor=False))
        return pyramid

    def laplacian_pyramid(x):
        gaussian = gaussian_pyramid(x)
        laps = [cur - F.interpolate(nxt, size=cur.shape[-2:], mode="bilinear", align_corners=False)
                for cur, nxt in zip(gaussian[:-1], gaussian[1:])]
        laps.append(gaussian[-1])
        return laps

    generated_pyr = laplacian_pyramid(generated)
    source_pyr = laplacian_pyramid(source)
    mask_pyr = gaussian_pyramid(mask)
    blended = [gen * m + src * (1.0 - m) for gen, src, m in zip(generated_pyr, source_pyr, mask_pyr)]
    result = blended[-1]
    for level in reversed(blended[:-1]):
        result = F.interpolate(result, size=level.shape[-2:], mode="bilinear",
                               align_corners=False) + level
    return result.clamp(0.0, 1.0)


def _comfy_mask_to_t1hw(mask, frames, height, width):
    """Normalize a comfy MASK ([T,H,W] or [H,W]) to [T,1,H,W] at the target
    frame count/size: resized bilinear, tiled/truncated along T (a 1-frame
    mask becomes a static mask for the whole clip)."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    m = mask.unsqueeze(1).float()  # [T,1,H,W]
    if m.shape[-2:] != (height, width):
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
    t = m.shape[0]
    if t < frames:
        reps = -(-frames // t)
        m = m.repeat(reps, 1, 1, 1)
    return m[:frames].clamp(0.0, 1.0)


class LTXV23RemovePerson:
    """Stage 1 of the two-stage person-replacement recipe: regenerate a
    masked region of a video (typically the person to remove) using the
    official LTX-2.3 in/outpainting IC-LoRA - the trained Wan2GP recipe
    (see the section comment above), shaped like LTXV23VidToVideo: `model`
    in/out, LoRA loaded in-node via the selector, feed the outputs into
    LTXV23KSampler → LTXV23CropVideoGuide → LTXV23AVDecode. Then composite
    with LTXV23MaskBlend (using this node's `source_frames`/`blend_mask`
    outputs) so everything outside the mask stays pixel-original.

    `mask`: per-frame comfy MASK batch (white = remove/regenerate). Get one
    from comfy-core's SAM3 nodes (SAM3_Detect on the frames, or
    SAM3_VideoTrack → SAM3_TrackToMask) with a text prompt like "woman".
    A single-frame mask is tiled across the whole clip (static mask).

    `prompt`: describe the scene WITHOUT the removed subject (what the
    filled-in region should show), e.g. "An empty room with a wooden floor
    and beige walls, doorway in the background, static camera."
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Remove Person (inpaint) ⚡"
    SEARCH_ALIASES = ['remove person', 'inpaint', 'inpainting', 'erase', 'masked removal',
                      'object removal', 'clean plate']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT", "IMAGE", "MASK")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "frame_rate",
                    "source_frames", "blend_mask")
    FUNCTION = "prepare"
    DESCRIPTION = ("Masked video inpainting via the official LTX-2.3 in/outpainting "
                   "IC-LoRA (green-fill recipe). Feed into LTXV23KSampler, then "
                   "LTXV23CropVideoGuide, LTXV23AVDecode, and finally LTXV23MaskBlend "
                   "with this node's source_frames/blend_mask outputs.")

    @classmethod
    def INPUT_TYPES(s):
        lora_choices = ["none"] + folder_paths.get_filename_list("loras")
        default_lora = next((c for c in lora_choices if "in-outpainting" in c), "none")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "video": ("VIDEO", {"tooltip": "Source clip. Sets length/frame_rate from "
                                    "itself, same as LTXV23VidToVideo."}),
                "mask": ("MASK", {"tooltip": "Per-frame mask, white = regenerate (the person/"
                                  "object to remove). SAM3_Detect / SAM3_VideoTrack + "
                                  "SAM3_TrackToMask produce this. A 1-frame mask is tiled "
                                  "across the clip."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                           "tooltip": "Describe the scene WITHOUT the removed subject - what "
                                      "the regenerated region should contain."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "inpaint_lora": (lora_choices, {"default": default_lora,
                                 "tooltip": "The official ltx-2.3-22b-ic-lora-in-outpainting "
                                            "LoRA - loaded onto model HERE at strength 1.0 "
                                            "(the trained value). \"none\" disables the whole "
                                            "mechanism (for A/B only - the green fill would "
                                            "just be reproduced verbatim)."}),
            },
            "optional": {
                "keep_original_audio": ("BOOLEAN", {"default": True,
                               "tooltip": "On = output keeps the source clip's own audio "
                                          "unchanged. Off = the model generates new audio."}),
                "mask_dilation": ("INT", {"default": LTX2_INPAINT_MASK_DILATION, "min": 0,
                                  "max": 128,
                                  "tooltip": "Pre-fill mask dilation radius in pixels "
                                             "(trained default 5)."}),
                "start_image": ("IMAGE", {"tooltip": "Optional identity anchor: becomes the "
                                "control video's FIRST frame verbatim (no green fill on frame "
                                "0). For person replacement: an image of the NEW person in the "
                                "same scene and starting pose - the green silhouette in later "
                                "frames carries the motion, this frame carries who fills it. "
                                "The prompt should then describe the scene WITH the new "
                                "subject."}),
            },
        }

    @torch.inference_mode()
    def prepare(self, model, clip, vae, audio_vae, video, mask, prompt, negative_prompt,
                width, height, batch_size, inpaint_lora, keep_original_audio=True,
                mask_dilation=LTX2_INPAINT_MASK_DILATION, start_image=None):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use the kit's "
                             "*_audio_vae.safetensors in the audio_vae slot.")

        if inpaint_lora not in (None, "none", ""):
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(model, inpaint_lora, 1.0)[0]
        else:
            logger.warning("LTX-2.3 remove: inpaint_lora=none - the green-filled region will "
                           "NOT be regenerated (A/B mode only)")

        components = video.get_components()
        video_frames = components.images
        video_audio = components.audio
        length = _align_length(video_frames.shape[0])
        frame_rate = float(components.frame_rate)

        # Source frames at target size - kept pristine for the post-decode blend.
        source_frames = comfy.utils.common_upscale(
            video_frames.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)[:, :, :, :3]

        mask_t1hw = _comfy_mask_to_t1hw(mask, source_frames.shape[0], height, width)
        mask_t1hw = _ltx2_dilate_mask(mask_t1hw, int(mask_dilation))
        control_frames = _ltx2_greenfill(source_frames, mask_t1hw)
        if start_image is not None:
            # Identity anchor: the control video opens on the provided frame
            # verbatim (no green fill), so the regenerated green region in
            # later frames inherits this subject's appearance via temporal
            # propagation from t=0. Frame 0 is excluded from the blend mask.
            anchor = comfy.utils.common_upscale(
                start_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)[:, :, :, :3].to(control_frames)
            control_frames = torch.cat([anchor, control_frames[1:]], dim=0)
            mask_t1hw = mask_t1hw.clone()
            mask_t1hw[0].zero_()
        control_frames, mask_t1hw = _ltx2_green_tail_pad(control_frames, mask_t1hw, length)

        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        device = comfy.model_management.intermediate_device()
        video_samples = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO], device=device)
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        # Green-filled control video appended as clean guide tokens at
        # frame_idx 0, strength exactly 1.0 - identical delegation to
        # LTXV23VidToVideo's guide path. The user mask deliberately does
        # NOT touch the denoise mask (Wan2GP force-zeroes masking_strength
        # for this mode - the LoRA does the regeneration).
        import comfy_extras.nodes_lt as nodes_lt

        scale_factors = vae.downscale_index_formula
        time_scale = scale_factors[0]
        n = control_frames.shape[0]
        guide_frames = control_frames[:((n - 1) // time_scale) * time_scale + 1]
        guide_latent = _match_batch(vae.encode(guide_frames), batch_size)
        frame_idx, _ = nodes_lt.LTXVAddGuide.get_latent_index(
            positive, t_latent, guide_latent.shape[2], 0, scale_factors,
            latent_shape=video_samples.shape)
        positive, negative, video_samples, video_mask = nodes_lt.LTXVAddGuide.append_keyframe(
            positive, negative, frame_idx, video_samples, video_mask, guide_latent,
            1.0, scale_factors, causal_fix=True)
        logger.info("LTX-2.3 remove: green-filled guide %s appended @ strength 1.0 "
                    "(mask dilation %d)", tuple(guide_latent.shape), int(mask_dilation))

        n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
        channels = int(getattr(audio_vae, "latent_channels", fsm.latent_channels))
        target = [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)]
        if keep_original_audio and video_audio is not None:
            audio_latent, audio_mask = _encode_reference_audio(
                audio_vae, video_audio, length / frame_rate)
            audio_latent, audio_mask = _fit_audio_latent(audio_latent, audio_mask, target)
            audio_latent = _match_batch(audio_latent, batch_size).to(video_samples.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video_samples.device)
        else:
            audio_latent = torch.zeros(target, device=video_samples.device)
            audio_mask = torch.ones_like(audio_latent)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_samples, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }
        blend_mask = mask_t1hw[:source_frames.shape[0], 0]  # comfy MASK [T,H,W]
        return (model, positive, negative, latent, frame_rate, source_frames, blend_mask)


class LTXV23MaskBlend:
    """Stage 1's post-decode composite: Laplacian-pyramid blend of the
    generated (inpainted) frames back over the ORIGINAL source frames,
    inside the (softened) mask only - Wan2GP's _apply_ltx2_mask_blend,
    ported faithfully (7-level pyramid, low-res mask dilation 6 at a 64px
    long side, sanitize-source inside the mask). Everything outside the
    mask ends up pixel-identical to the source; inside, only the
    regenerated content shows, with no visible seam.

    Wire: LTXV23AVDecode's video → core GetVideoComponents → `images` here;
    LTXV23RemovePerson's `source_frames`/`blend_mask` → the matching
    inputs. Output frames go to core CreateVideo (with the held audio).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Mask Blend ⚡"
    SEARCH_ALIASES = ['mask blend', 'laplacian blend', 'composite', 'inpaint blend',
                      'seamless blend']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    DESCRIPTION = ("Laplacian-pyramid composite of inpainted frames over the original "
                   "source inside the mask - the post-decode half of the LTX-2.3 "
                   "removal recipe.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Generated (inpainted) frames, decoded."}),
                "source_frames": ("IMAGE", {"tooltip": "LTXV23RemovePerson's source_frames "
                                            "output (the pristine originals)."}),
                "mask": ("MASK", {"tooltip": "LTXV23RemovePerson's blend_mask output."}),
            },
            "optional": {
                "mask_low_res_dilation": ("INT", {
                    "default": LTX2_INPAINT_LAPLACIAN_MASK_LOW_RES_DILATION, "min": 0, "max": 64,
                    "tooltip": "Soft-skirt width around the mask (trained default 6)."}),
                "mask_b": ("MASK", {"tooltip": "Optional second mask, unioned with mask "
                           "after both are fitted to the common frame count. For "
                           "replacement composites: mask = the ORIGINAL subject's "
                           "silhouette, mask_b = the GENERATED subject's - the union is "
                           "the region taken from the generated frames, so the original "
                           "subject can never peek out and no inpainting is needed. "
                           "Differing frame counts between the two masks (the 8k+1 "
                           "grid round-up) are reconciled here, which core "
                           "MaskComposite cannot do."}),
            },
        }

    @torch.inference_mode()
    def blend(self, images, source_frames, mask,
              mask_low_res_dilation=LTX2_INPAINT_LAPLACIAN_MASK_LOW_RES_DILATION,
              mask_b=None):
        height, width = source_frames.shape[1], source_frames.shape[2]
        frames = min(images.shape[0], source_frames.shape[0])
        generated = images[:frames, :height, :width, :3]
        source = source_frames[:frames, :, :, :3]
        if generated.shape[1:3] != source.shape[1:3]:
            generated = comfy.utils.common_upscale(
                generated.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        mask_t1hw = _comfy_mask_to_t1hw(mask, frames, height, width)
        if mask_b is not None:
            mask_b_t1hw = _comfy_mask_to_t1hw(mask_b, frames, height, width)
            mask_t1hw = torch.maximum(mask_t1hw, mask_b_t1hw)

        # sanitize_masked_source (Wan2GP inpainting.py:162-170): inside the
        # mask the "source" is replaced by the generated pixels BEFORE the
        # blend, so leftover original content (the removed person) can't
        # bleed back through the soft pyramid skirt.
        mask_thw1 = mask_t1hw.permute(0, 2, 3, 1)
        source = torch.where(mask_thw1 > 0, generated.to(source.dtype), source)

        blended = _ltx2_laplacian_pyramid_blend(
            generated.movedim(-1, 1), source.movedim(-1, 1), mask_t1hw,
            mask_low_res_dilation=int(mask_low_res_dilation))
        out = blended.movedim(1, -1).to(images.dtype)
        if frames < images.shape[0]:
            out = torch.cat([out, images[frames:]], dim=0)
        return (out,)


class LTXV23KSampler:
    """Euler on the official LTX-2 distilled schedules.

    The stock schedulers (simple/karras/...) do not reproduce the trained
    distilled schedule, and a distilled model on the wrong schedule looks like
    a broken model. This passes the exact sigmas through; 8 steps is the
    trained configuration, other counts resample the same curve.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 KSampler (distilled) ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'txt2img', 'img2img']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Sample a joint AV latent on the official LTX-2 schedules "
                   "(distilled 8-step or refine 3-step, cfg 1.0, euler).")

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
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000,
                                  "tooltip": "8 for dmd/distilled, 4 for dmd upscale, "
                                             "3 for refine."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "1.0 - both the DMD and distilled bakes "
                                             "are trained without CFG."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "schedule": (list(SIGMA_SETS), {"default": "dmd (8 steps)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, schedule, denoise):
        samples = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask", None)
        sigmas = distilled_sigma_schedule(
            steps, denoise, sigmas=SIGMA_SETS[schedule]).to(
            device=comfy.model_management.intermediate_device(), dtype=torch.float32)
        noise = comfy.sample.prepare_noise(samples, seed,
                                           latent_image.get("batch_index", None))

        logger.info("LTX-2.3 sample: %s, %d steps from sigma %.4f, mask %s",
                    schedule, len(sigmas) - 1, float(sigmas[0]),
                    "yes" if noise_mask is not None else "NONE")

        out = comfy.sample.sample(
            model, noise, steps=len(sigmas) - 1, cfg=cfg,
            sampler_name=sampler_name, scheduler="simple",
            positive=positive, negative=negative, latent_image=samples,
            sigmas=sigmas, seed=seed, noise_mask=noise_mask,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        latent = latent_image.copy()
        latent["samples"] = out
        # Keep noise_mask in the output (mirrors LTXV25KSampler): the AUDIO
        # half's hold mask is what lets a downstream refine pass re-hold the
        # original audio instead of re-noising it. Popping it here was the
        # root of the two-stage original-audio loss (found live twice: the
        # upscale-level fix was necessary but not sufficient - the mask died
        # one node earlier, right here). Held content is baked into the
        # samples, so keeping the mask is a no-op for single-pass graphs.
        return (latent,)


# ── two-stage refine + decode ───────────────────────────────────────────────

def _upsample_video_latent(latent, upscale_model, vae):
    """Port of core LTXVLatentUpsampler, restricted to the video branch.

    The upsample model only understands a single video-shaped tensor, not a
    joint AV one, so the latent must be split before calling it and rejoined
    after - calling it directly on the concatenated AV tensor does not error,
    it silently treats part of the audio latent as video channels.
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
    # The VIDEO hold is void after a spatial upscale (guide frames are
    # cropped, and the refine pass resamples every frame at the new grid) -
    # but the AUDIO half is untouched by a spatial x2, and its hold mask is
    # what keeps keep_original_audio's encoded waveform from being re-noised
    # by the refine sampler. Popping the whole mask here silently destroyed
    # the original audio in every two-stage graph (found live).
    mask = out.pop("noise_mask", None)
    if mask is not None and getattr(mask, "is_nested", False):
        _vm, audio_mask = mask.unbind()
        # Per-frame broadcast shape (B,1,T,1,1) - the mask convention every
        # prep node in this family emits (a full-size mask is documented
        # there as NOT interchangeable with it).
        video_mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1),
                                dtype=torch.float32, device=video.device)
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (video_mask, audio_mask.to(video.device)))
    return out


class LTXV23RefineSampler:
    """Base pass -> spatial x2 latent upscale -> refine pass, in one node.

    Reuses LTXV23KSampler for both passes (same schedule/noise semantics,
    verified separately) so there is exactly one place that owns the sigma
    math. The only genuinely new logic is the upscale hop between them: see
    ``_upsample_video_latent`` for why the joint latent has to be split and
    rejoined around the upscale model rather than fed to it directly.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Two-Stage Sampler (base + refine) ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'refine latent', 'upscale latent', 'enlarge latent']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Base sampling pass, spatial x2 latent upscale, then a "
                   "refine pass - the official LTX-2.3 two-stage recipe.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE", {"tooltip": "Video VAE - normalizes the latent "
                                          "around the upscale model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "base_schedule": (list(SIGMA_SETS), {"default": "dmd (8 steps)"}),
                "base_steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "refine_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                        "control_after_generate": True}),
                "refine_schedule": (list(SIGMA_SETS), {"default": "refine (3 steps)"}),
                "refine_steps": ("INT", {"default": 3, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            },
        }

    def sample(self, model, positive, negative, latent_image, upscale_model, vae,
               seed, base_schedule, base_steps, refine_seed, refine_schedule,
               refine_steps, cfg, sampler_name):
        sampler = LTXV23KSampler()
        base_latent, = sampler.sample(
            model, positive, negative, latent_image, seed, base_steps, cfg,
            sampler_name, base_schedule, denoise=1.0)

        # Crop appended guide frames + keyframe conditioning BEFORE the
        # upscale - the refine pass must see neither: stale keyframe coords
        # trip the model's grid check after the x2 ("keyframe_idxs holds N
        # tokens, which is not a whole number of ...-token latent frames",
        # hit live on a manual two-stage chain missing its crop), and the
        # guide frames themselves would be upscaled and re-sampled as
        # content. Identical to wiring LTXV23CropVideoGuide between the
        # stages by hand; a no-op when nothing was appended (t2v/i2v).
        positive, negative, base_latent = LTXV23CropVideoGuide().crop(
            positive, negative, base_latent)

        upscaled = _upsample_video_latent(base_latent, upscale_model, vae)
        logger.info("LTX-2.3 refine: base %s -> upscaled %s",
                    tuple(base_latent["samples"].unbind()[0].shape),
                    tuple(upscaled["samples"].unbind()[0].shape))

        refined, = sampler.sample(
            model, positive, negative, upscaled, refine_seed, refine_steps, cfg,
            sampler_name, refine_schedule, denoise=1.0)
        return (refined,)


class LTXV23LatentUpscale:
    """Standalone x2 spatial latent upscale (the same _upsample_video_latent
    the refine sampler uses internally, exposed as its own step) - for
    composing a custom two-stage flow: sample at half resolution, crop the
    guides, upscale x2 here, then run LTXV23KSampler with the "refine
    (3 steps)" schedule on the result. Feed the CROPPED conditioning to the
    refine pass - the keyframe coords would misalign after upscaling."""

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Latent Upscale x2 ⚡"
    SEARCH_ALIASES = ['latent upscale', 'upscale', 'spatial upscaler', 'x2', 'two stage']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    DESCRIPTION = ("x2 spatial latent upscale via the official LTX-2.3 spatial "
                   "upscaler, on the video half of the joint AV latent. Crop "
                   "guides first; refine after with the 'refine (3 steps)' "
                   "schedule.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE", {"tooltip": "Video VAE (for the latent statistics)."}),
            },
        }

    @torch.inference_mode()
    def upscale(self, latent, upscale_model, vae):
        upscaled = _upsample_video_latent(latent, upscale_model, vae)
        logger.info("LTX-2.3 latent upscale: %s -> %s",
                    tuple(latent["samples"].unbind()[0].shape),
                    tuple(upscaled["samples"].unbind()[0].shape))
        return (upscaled,)


class LTXV23CropVideoGuide:
    """Strip the video-guide reference frames LTXV23VidToVideo appended,
    after sampling and before LTXV23AVDecode.

    A thin wrapper around comfy-core's own LTXVCropGuides (comfy_extras/
    nodes_lt.py's get_keyframe_idxs) applied to just the video half of the
    joint AV latent - core's version only understands a plain video latent,
    not this pack's NestedTensor((video, audio)) convention, so the two
    streams get split before the crop and rejoined after (same reason
    _upsample_video_latent above has to do this for the refine sampler's
    upscale model). A no-op (returns the latent unchanged) when no guide was
    ever appended (ic_lora=none upstream, or nothing to crop) -
    matches core's own early-return behavior.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Crop Video Guide ⚡"
    SEARCH_ALIASES = ['crop guide', 'remove guide', 'strip reference', 'ic-lora',
                      'vid2vid', 'v2v']
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "crop"
    DESCRIPTION = ("Strip the video-guide reference frames LTXV23VidToVideo "
                   "appended, after sampling. No-op if none were appended.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
        }

    def crop(self, positive, negative, latent):
        import comfy_extras.nodes_lt as nodes_lt

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("LTX-2.3 Crop Video Guide: latent isn't a joint AV "
                             "latent - expected LTXV23KSampler's output.")
        video, audio = samples.unbind()

        video_mask = audio_mask = None
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None:
            video_mask, audio_mask = noise_mask.unbind()

        _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, video.shape)
        if num_keyframes == 0:
            return (positive, negative, latent)

        cropped_video = video[:, :, :-num_keyframes]
        cropped_mask = video_mask[:, :, :-num_keyframes] if video_mask is not None else None

        positive = node_helpers.conditioning_set_values(
            positive, {"keyframe_idxs": None, "guide_attention_entries": None})
        negative = node_helpers.conditioning_set_values(
            negative, {"keyframe_idxs": None, "guide_attention_entries": None})

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((cropped_video, audio))
        if cropped_mask is not None and audio_mask is not None:
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((cropped_mask, audio_mask))
        else:
            out.pop("noise_mask", None)

        logger.info("LTX-2.3 crop guide: removed %d guide frame(s), video %s -> %s",
                    num_keyframes, tuple(video.shape), tuple(cropped_video.shape))
        return (positive, negative, out)


class LTXV23AVDecode:
    """Joint AV latent -> muxed VIDEO, in one node.

    Wraps core VAEDecodeTiled (video) + LTXVAudioVAEDecode (audio) +
    CreateVideo (mux), threading one ``fps`` through all three - the
    plain-node version needs the same value typed into two different widgets
    that have no wire between them, and a mismatch there is a silent
    audio/video drift, not an error.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 AV Decode ⚡"
    SEARCH_ALIASES = ['decode', 'decode latent', 'latent to video', 'latent to audio', 'video decode', 'audio decode']
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
                "fps": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 96, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 16, "min": 4, "max": 4096, "step": 4}),
            },
        }

    def decode(self, latent, vae, audio_vae, fps, tile_size=768, overlap=64,
               temporal_size=96, temporal_overlap=16):
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
        logger.info("LTX-2.3 AV decode: %s frames @ %.2f fps, audio %s @ %d Hz",
                    tuple(images.shape), fps, tuple(waveform.shape), sample_rate)
        return (video,)


# ── ID-LoRA prompt editing ──────────────────────────────────────────────────

# Lazily matches each [TAG]: body up to the next [TAG]: or end of string, so
# it works whether SOUNDS (the last tag) or a middle tag like SPEECH is being
# pulled out - a greedy `(.*)$` makes SPEECH swallow SOUNDS too. Tolerant of
# a Part-1 spoken-script preamble before the tags (the captioner's raw
# two-part output) since it anchors on the tags themselves, not on position.
_ID_LORA_TAG_RE = re.compile(
    r"\[(VISUAL|SPEECH|SOUNDS)\]:\s*(.*?)\s*(?=\[(?:VISUAL|SPEECH|SOUNDS)\]:|$)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_id_lora_prompt(source):
    fields = {"VISUAL": "", "SPEECH": "", "SOUNDS": ""}
    for tag, body in _ID_LORA_TAG_RE.findall(source or ""):
        fields[tag.upper()] = body.strip()
    return fields["VISUAL"], fields["SPEECH"], fields["SOUNDS"]


# node unique_id -> the source text that was last parsed for it. Lets
# assemble() tell "the boxes hold the user's edit" from "the boxes hold
# stale auto-fill" without guessing from widget content (undecidable - see
# the class docstring) or making the user flip a switch.
_LAST_SOURCE = {}


class LTXV23IDLoraPromptEditor:
    """Show a captioner's [VISUAL]/[SPEECH]/[SOUNDS] in three editable
    boxes, and reassemble them into an ID-LoRA prompt string.

    ``source`` is whatever the upstream captioner (e.g. LMStudioVisionPrompt)
    produced - its full two-part output (spoken script, ``---``, tagged
    block) or just the tagged block; the parser only looks for the three
    ``[TAG]:`` markers, ignoring a Part-1 preamble. It is ``forceInput`` so
    it renders as a socket only: a widget-backed input that is wired greys
    out, blanks its own text and stops accepting clicks
    (comfy frontend ``LGraphNode.updateComputedDisabled``), so a visible box
    there would be dead weight.

    The three boxes auto-fill with the parsed values after a run and stay
    directly editable; an edit survives later runs, and a genuinely new
    ``source`` refreshes them. That combination does not exist as a widget
    flag in comfy-core - its only populate-from-own-execution widget,
    TEXT_PREVIEW (PreviewAny/SaveText), is hard-coded read-only and
    ``serialize: False`` - so the state lives here instead:

      * this run's ``source`` differs from the one remembered for this node
        -> the boxes are stale, re-parse and overwrite them;
      * same ``source`` as last run -> the boxes are the truth, keep them
        (this is where a manual edit survives);
      * nothing remembered yet (fresh ComfyUI start) -> fill only empty
        boxes, so edits saved into the workflow survive a restart.

    The paired ``web/`` JS then writes the resolved values back into the
    widgets unconditionally. That is safe precisely because the decision
    already happened here: it either echoes the user's own edit back or
    shows the fresh parse. Earlier versions put that decision in the JS
    ("fill only if the widget is empty"), which cannot work - once
    auto-filled a widget is non-empty forever, so it filled once and then
    appeared frozen.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 ID-LoRA Prompt Editor ⚡"
    SEARCH_ALIASES = ['edit prompt', 'prompt editor', 'visual speech sounds', 'id lora prompt', 'prompt writer']
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("tagged_prompt", "visual_text", "speech_text", "sounds_text",
                    "speech_text_batch")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    FUNCTION = "assemble"
    OUTPUT_NODE = True          # required for the ui payload to reach the frontend
    DESCRIPTION = ("Edit a captioner's [VISUAL]/[SPEECH]/[SOUNDS] fields and "
                   "reassemble them into an ID-LoRA prompt.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("STRING", {"forceInput": True,
                    "tooltip": "The captioner's raw output - its full "
                               "two-part text, or just the tagged block."}),
                "visual": ("STRING", {"multiline": True, "default": ""}),
                "speech": ("STRING", {"multiline": True, "default": ""}),
                "sounds": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def assemble(self, source, visual, speech, sounds, unique_id=None):
        previous = _LAST_SOURCE.get(unique_id)
        if previous is None:
            if not (visual.strip() or speech.strip() or sounds.strip()):
                visual, speech, sounds = _parse_id_lora_prompt(source)
        elif previous != source:
            visual, speech, sounds = _parse_id_lora_prompt(source)
        _LAST_SOURCE[unique_id] = source

        tagged_prompt = f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}"
        # One clip per non-blank line. A blank line is treated as a separator,
        # not an empty clip - "line one\n\nline two" is 2 clips, not 3.
        speech_batch = [line.strip() for line in speech.splitlines() if line.strip()]
        if not speech_batch:
            speech_batch = [""]  # never return an empty list - nothing to index

        return {
            "ui": {"visual": [visual], "speech": [speech], "sounds": [sounds]},
            "result": (tagged_prompt, visual, speech, sounds, speech_batch),
        }


class LTXV23SpeechBatchSelector:
    """Pick one clip out of a speech_text_batch list by index.

    ``batch`` must arrive as the whole list in one call, not fanned out one
    call per item - INPUT_IS_LIST=True (comfy's execution.py:245,
    _async_map_node_over_list) does exactly that, at the cost of every input
    (including ``index``) arriving wrapped in a length-1 list, unwrapped
    below.

    ``count`` is the batch's total length - wire it into a loop node's
    iteration-count input to drive a for-each over the batch (e.g. bump
    ``index`` from 0 to ``count - 1`` across iterations).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Speech Batch Selector ⚡"
    SEARCH_ALIASES = ['select from batch', 'pick clip', 'batch subset', 'index selector', 'choose from list']
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_text", "count")
    FUNCTION = "select"
    INPUT_IS_LIST = True
    DESCRIPTION = "Pick one clip from a speech_text_batch list by index."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch": ("STRING", {"forceInput": True,
                    "tooltip": "A speech_text_batch list, e.g. from the "
                               "ID-LoRA Prompt Editor."}),
                "index": ("INT", {"default": 0, "min": -0xffffffff, "max": 0xffffffff,
                    "tooltip": "0-based; negative counts from the end like "
                               "Python (-1 = last clip). Out-of-range clamps "
                               "to the nearest valid index."}),
            },
        }

    def select(self, batch, index):
        index = int(index[0])
        n = len(batch)
        clamped = max(-n, min(index, n - 1))
        if clamped != index:
            logger.warning("LTX-2.3 Speech Batch Selector: index %d out of "
                           "range for %d clip(s), clamped to %d", index, n, clamped)
        return (batch[clamped], n)


class LTXV23IDLoraAssembler:
    """Merge three separate visual/speech/sounds strings into the canonical
    ID-LoRA prompt string.

    The counterpart to ``LTXV23IDLoraPromptEditor``: that node parses ONE
    combined source string apart into three editable fields; this node goes
    the other direction, combining three ALREADY-SEPARATE strings (e.g. one
    edited by hand, one picked from ``LTXV23SpeechBatchSelector``, one from
    somewhere else entirely) into one formatted string. No source input, no
    parsing, no state - a pure formatter, same job
    ``LTXV23IDLoraPromptEditor.assemble()`` does internally, exposed on its
    own for pipelines that already have the three pieces from elsewhere.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 ID-LoRA Assembler ⚡"
    SEARCH_ALIASES = ["merge prompt", "combine prompt", "format prompt",
                      "assemble prompt", "id lora prompt", "visual speech sounds"]
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tagged_prompt",)
    FUNCTION = "assemble"
    DESCRIPTION = "Combine separate visual/speech/sounds strings into the ID-LoRA prompt format."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "visual": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [VISUAL] section text."}),
                "speech": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [SPEECH] section text."}),
                "sounds": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [SOUNDS] section text."}),
            },
        }

    def assemble(self, visual, speech, sounds):
        return (f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}",)


NODE_CLASS_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader,
    "LTXV23ImgToVideo": LTXV23ImgToVideo,
    "LTXV23VidToVideo": LTXV23VidToVideo,
    "LTXV23RemovePerson": LTXV23RemovePerson,
    "LTXV23MaskBlend": LTXV23MaskBlend,
    "LTXV23KSampler": LTXV23KSampler,
    "LTXV23RefineSampler": LTXV23RefineSampler,
    "LTXV23LatentUpscale": LTXV23LatentUpscale,
    "LTXV23CropVideoGuide": LTXV23CropVideoGuide,
    "LTXV23AVDecode": LTXV23AVDecode,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor,
    "LTXV23SpeechBatchSelector": LTXV23SpeechBatchSelector,
    "LTXV23IDLoraAssembler": LTXV23IDLoraAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader.TITLE,
    "LTXV23ImgToVideo": LTXV23ImgToVideo.TITLE,
    "LTXV23VidToVideo": LTXV23VidToVideo.TITLE,
    "LTXV23RemovePerson": LTXV23RemovePerson.TITLE,
    "LTXV23MaskBlend": LTXV23MaskBlend.TITLE,
    "LTXV23KSampler": LTXV23KSampler.TITLE,
    "LTXV23RefineSampler": LTXV23RefineSampler.TITLE,
    "LTXV23LatentUpscale": LTXV23LatentUpscale.TITLE,
    "LTXV23CropVideoGuide": LTXV23CropVideoGuide.TITLE,
    "LTXV23AVDecode": LTXV23AVDecode.TITLE,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor.TITLE,
    "LTXV23SpeechBatchSelector": LTXV23SpeechBatchSelector.TITLE,
    "LTXV23IDLoraAssembler": LTXV23IDLoraAssembler.TITLE,
}
