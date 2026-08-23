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

    from .loader import gguf_clip_loader
    from .ops import GGMLOps

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
            from .nodes import UnetLoaderGGUF
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


class LTXV23KSampler:
    """Euler on the official LTX-2 distilled schedules.

    The stock schedulers (simple/karras/...) do not reproduce the trained
    distilled schedule, and a distilled model on the wrong schedule looks like
    a broken model. This passes the exact sigmas through; 8 steps is the
    trained configuration, other counts resample the same curve.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 KSampler (distilled) ⚡"
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
        latent.pop("noise_mask", None)
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
    out.pop("noise_mask", None)  # the upsampled latent has no held frames left
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

        upscaled = _upsample_video_latent(base_latent, upscale_model, vae)
        logger.info("LTX-2.3 refine: base %s -> upscaled %s",
                    tuple(base_latent["samples"].unbind()[0].shape),
                    tuple(upscaled["samples"].unbind()[0].shape))

        refined, = sampler.sample(
            model, positive, negative, upscaled, refine_seed, refine_steps, cfg,
            sampler_name, refine_schedule, denoise=1.0)
        return (refined,)


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
# pulled out - a greedy `(.*)$` reproduces the exact "SPEECH swallows SOUNDS
# too" bug this pattern exists to avoid. Tolerant of a Part-1 spoken-script
# preamble before the tags (the LM Studio/captioner node's raw two-part
# output) since it only anchors on the tags themselves, not on position.
_ID_LORA_TAG_RE = re.compile(
    r"\[(VISUAL|SPEECH|SOUNDS)\]:\s*(.*?)\s*(?=\[(?:VISUAL|SPEECH|SOUNDS)\]:|$)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_id_lora_prompt(source):
    fields = {"VISUAL": "", "SPEECH": "", "SOUNDS": ""}
    for tag, body in _ID_LORA_TAG_RE.findall(source or ""):
        fields[tag.upper()] = body.strip()
    return fields["VISUAL"], fields["SPEECH"], fields["SOUNDS"]


class LTXV23IDLoraPromptEditor:
    """Parse a generated [VISUAL]/[SPEECH]/[SOUNDS] block into three
    editable fields, and reassemble a canonical ID-LoRA prompt string.

    ``source`` is whatever the upstream captioner (e.g. LMStudioVisionPrompt)
    produced - either its full two-part output (spoken script + tagged
    block) or just the tagged block on its own; the parser only looks for
    the three ``[TAG]:`` markers and ignores everything else, including a
    Part-1 preamble.

    Each ``*_override`` widget wins over the parsed value when non-empty, so
    editing one field does not require retyping the other two, and does not
    require re-running the captioner. The paired ``web/`` JS extension
    auto-fills the three widgets with the parsed values right after the node
    runs, but ONLY when a widget is still empty - it never clobbers an edit.

    Each field is also available as its own (post-override) output, not
    just baked into ``tagged_prompt`` - ``speech_text`` in particular so it
    can feed a TTS node's ``text`` input directly instead of being extracted
    a second time from a separate Part-1 split, which would let the two
    extractions drift (e.g. one gets edited here, the other doesn't).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 ID-LoRA Prompt Editor ⚡"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("tagged_prompt", "visual_text", "speech_text", "sounds_text")
    FUNCTION = "assemble"
    OUTPUT_NODE = True
    DESCRIPTION = ("Parse a generated [VISUAL]/[SPEECH]/[SOUNDS] block into "
                   "three editable fields and reassemble it.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The captioner's raw output - the full "
                               "two-part text, or just the tagged block."}),
            },
            "optional": {
                "visual_override": ("STRING", {"multiline": True, "default": ""}),
                "speech_override": ("STRING", {"multiline": True, "default": ""}),
                "sounds_override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def assemble(self, source, visual_override="", speech_override="",
                sounds_override=""):
        visual, speech, sounds = _parse_id_lora_prompt(source)
        visual = visual_override.strip() or visual
        speech = speech_override.strip() or speech
        sounds = sounds_override.strip() or sounds

        tagged_prompt = f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}"
        return {
            "ui": {"visual": [visual], "speech": [speech], "sounds": [sounds]},
            "result": (tagged_prompt, visual, speech, sounds),
        }


NODE_CLASS_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader,
    "LTXV23ImgToVideo": LTXV23ImgToVideo,
    "LTXV23KSampler": LTXV23KSampler,
    "LTXV23RefineSampler": LTXV23RefineSampler,
    "LTXV23AVDecode": LTXV23AVDecode,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader.TITLE,
    "LTXV23ImgToVideo": LTXV23ImgToVideo.TITLE,
    "LTXV23KSampler": LTXV23KSampler.TITLE,
    "LTXV23RefineSampler": LTXV23RefineSampler.TITLE,
    "LTXV23AVDecode": LTXV23AVDecode.TITLE,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor.TITLE,
}
