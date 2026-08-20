# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""LTX-2.3 A/V nodes: kit loader, joint AV latent, distilled sampler.

Built for the 10Eros LTX-2.3 22B kit (distill-LoRA pre-fused), but works for
any LTX-2.3-family checkpoint split the same way:

  LTXV23ModelsLoader  DiT (GGUF stays quantized; fp8/bf16 safetensors works
                      too) + Gemma-3 text encoder + text-embedding projections
                      + video VAE + audio VAE -> MODEL, CLIP, VAE, VAE(audio)
  LTXV23EmptyLatentAV nested video+audio latent in LTX-2.x geometry
  LTXV23KSampler      euler + the LTX-2 distilled schedule (8 steps, cfg 1.0)

Kit layout on disk (produced by the giga-images ltxv23_10eros_turbo pipeline):

  models/diffusion_models  10Eros_v1.4_distilled-r72_{Q4_K_M,Q6_K,Q8_0}.gguf
                           (config embedded as a GGUF KV — no sidecar needed)
                           or 10Eros_v1.4_distilled-r72_fp8mixed.safetensors
  models/text_encoders     gemma-3-12b-it-ablit-norms-biproj-Q4_K_M.gguf
                           10Eros_v1.4_projections.safetensors
  models/vae               10Eros_v1.4_video_vae.safetensors
                           10Eros_v1.4_audio_vae.safetensors

Decode wiring after sampling: core ``LTXVSeparateAVLatent`` splits the nested
latent; video stream -> stock VAE Decode, audio stream ->
``LTXVAudioVAEDecode``.
"""

import logging

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths
import nodes

logger = logging.getLogger(__name__)

LTX23_CATEGORY = "🤖 CCTech/LTX-2.3"

# LTX-2 distilled denoising schedule (ltx_pipelines.utils.constants
# .DISTILLED_SIGMAS). The 10Eros kit has the distill LoRA pre-fused, so this
# schedule at cfg 1.0 is the intended way to run it.
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                    0.909375, 0.725, 0.421875, 0.0]

# Stage-2 refine schedule from the official LTX-2.3 workflows (the pass after
# the latent x2 spatial upscale): 3 steps, cfg 1.0, euler.
REFINE_SIGMAS = [0.85, 0.725, 0.4219, 0.0]

SIGMA_SETS = {
    "distilled (8 steps)": DISTILLED_SIGMAS,
    "refine (3 steps)": REFINE_SIGMAS,
}

# Image conditioning strength the official I2V/IA2V workflows use for the
# first pass (LTXVImgToVideoInplace strength 0.7).
I2V_STRENGTH = 0.7

FPS = 24.0

_EXPECTED_LAYERS = 48


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def distilled_sigma_schedule(steps, denoise=1.0, sigmas=None):
    """The LTX-2 distilled sigmas, interpolated to ``steps`` steps.

    At the trained 8 steps this is exactly the reference schedule; other step
    counts are a linear resample of it (not a re-derived shift schedule).
    ``denoise`` slices the tail, matching how KSampler treats denoise.
    """
    base = torch.tensor(sigmas if sigmas is not None else DISTILLED_SIGMAS,
                        dtype=torch.float32)
    steps = max(1, int(steps))
    if steps + 1 == len(base):
        sigmas = base
    else:
        # linear resample of the reference curve (torch has no interp)
        idx = torch.linspace(0.0, len(base) - 1.0, steps + 1)
        lo = idx.floor().long().clamp(0, len(base) - 2)
        frac = idx - lo
        sigmas = base[lo] * (1.0 - frac) + base[lo + 1] * frac
    denoise = min(max(float(denoise), 0.0), 1.0)
    if denoise < 1.0:
        start = int(round((steps) * (1.0 - denoise)))
        sigmas = sigmas[start:]
    return sigmas


def _set_cond_frame_rate(cond, frame_rate):
    return [[c[0], {**c[1], "frame_rate": frame_rate}] for c in cond]


def _fit_audio_latent(audio, mask, target_shape):
    """Trim or zero-pad the audio latent to ``target_shape`` along time.

    Port of core ``LTXVConcatAVLatent.fit_audio``: a padded tail keeps mask 1
    so the model generates it, which is what a clip shorter than the video
    should do.
    """
    dim = 2  # [B, C, T, bins] -> time
    length = target_shape[dim]
    if audio.shape[dim] > length:
        audio = audio.narrow(dim, 0, length)
        if mask is not None:
            mask = mask.narrow(dim, 0, length)
    elif audio.shape[dim] < length:
        pad = torch.zeros(target_shape, device=audio.device, dtype=audio.dtype)
        pad[:, :, :audio.shape[dim]] = audio
        audio = pad
        if mask is not None:
            pmask = torch.ones_like(pad)
            pmask[:, :, :mask.shape[dim]] = mask
            mask = pmask
    return audio, mask


def _encode_reference_audio(audio_vae, audio, duration_s):
    """Encode an AUDIO reference into a locked audio latent [B, C, T, bins].

    Follows the official IA2V recipe: the encoded audio becomes the audio
    stream with a ZERO noise mask, so it stays fixed while the DiT generates
    video (and, on a shorter-than-video clip, a generated tail).
    """
    fsm = getattr(audio_vae, "first_stage_model", None)
    if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
        raise ValueError(
            "audio_vae is not an LTX audio VAE; the reference_audio input "
            "needs the kit's *_audio_vae.safetensors."
        )
    waveform = audio["waveform"][0]  # [C, T] - one reference, batched later
    sr = audio["sample_rate"]
    max_samples = int(duration_s * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    comfy.model_management.load_models_gpu(
        [audio_vae.patcher],
        force_full_load=getattr(audio_vae, "disable_offload", False))
    latent = fsm.encode(waveform.unsqueeze(0), sample_rate=sr)  # [1,C,T,bins]
    latent = latent.to(comfy.model_management.intermediate_device()).float()
    mask = torch.zeros_like(latent)  # 0 = locked: audio drives the video
    return latent, mask


def _align_length(length):
    """Round a frame count up to the 8k+1 grid the video VAE tiles on."""
    length = max(9, int(length))
    while (length - 1) % 8 != 0:
        length += 1
    return length


# The Gemma-3 12B TE plus projections costs ~24 GB and many seconds to build,
# and ComfyUI re-runs loader nodes on every prompt edit. Cache the most recent
# (te, projections) pair; the CLIP owns its ModelPatcher, so comfy still
# manages VRAM/offload for it.
_ENCODER_CACHE = {}


def _load_ltxv_clip(te_name, projections_name):
    key = (te_name, projections_name)
    cached = _ENCODER_CACHE.get(key)
    if cached is not None:
        return cached

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
            "that ships with the kit (or ltx-v2-projections for LTX-2.5)."
        )

    if te_path.lower().endswith(".gguf"):
        te_sd = gguf_clip_loader(te_path)
        te_options = {
            "custom_operations": GGMLOps(),
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        }
    else:
        te_sd, _ = comfy.utils.load_torch_file(te_path, return_metadata=True)
        te_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        }

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=comfy.sd.CLIPType.LTXV,
        state_dicts=[te_sd, proj_sd],
        model_options=te_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    del te_sd

    projection = type(getattr(clip.cond_stage_model, "text_embedding_projection",
                              None)).__name__
    if projection != "DualLinearProjection":
        raise ValueError(
            f"{te_name} + {projections_name} built a {projection}, not "
            "DualLinearProjection - the DiT's embeddings connectors cannot "
            "consume it. Use the kit's Gemma-3 GGUF with its projections file."
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
    fsm = vae.first_stage_model
    is_audio = hasattr(fsm, "num_of_latents_from_frames")
    if want_audio and not is_audio:
        raise ValueError(
            f"{vae_name} is not an LTX audio VAE (that looks like the video "
            "VAE). The audio_vae input wants the kit's *_audio_vae.safetensors."
        )
    if not want_audio and is_audio:
        raise ValueError(
            f"{vae_name} is the audio VAE. The video_vae input wants the kit's "
            "*_video_vae.safetensors."
        )
    return vae


class LTXV23ModelsLoader:
    """Load the whole LTX-2.3 A/V kit in one node.

    DiT GGUFs stay quantized (dequantized per layer at forward time); the
    fp8mixed safetensors rebuild also works. Outputs are plain comfy
    MODEL / CLIP / VAE objects, so they compose with comfy's own LTXV nodes.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Models Loader ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "LTX-2.3 A/V DiT from models/diffusion_models "
                               "(unet). GGUF quants stay quantized; the "
                               "fp8mixed safetensors also works. The 10Eros "
                               "kit has the distill LoRA pre-fused."}),
                "text_encoder_name": (_clip_filename_list(), {
                    "tooltip": "Gemma-3 12B text encoder, .gguf (stays "
                               "quantized) or safetensors, from "
                               "models/text_encoders (clip)."}),
                "projections_name": (_clip_filename_list(), {
                    "tooltip": "text_embedding_projection safetensors from "
                               "models/text_encoders — the dual 4096/2048 "
                               "output projection that pairs with the Gemma "
                               "encoder (10Eros_v1.4_projections.safetensors)."}),
                "video_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 video VAE (10Eros_v1.4_video_vae.safetensors "
                               "or ltx-2.x equivalent)."}),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 audio VAE + vocoder "
                               "(10Eros_v1.4_audio_vae.safetensors)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the LTX-2.3 A/V components (DiT, Gemma-3 text encoder "
                   "with dual projection, video VAE, audio VAE) as native "
                   "comfy objects.")

    def load(self, unet_name, text_encoder_name, projections_name,
             video_vae_name, audio_vae_name):
        # ── DiT → comfy LTXAV MODEL ──
        if unet_name.lower().endswith(".gguf"):
            from .nodes import UnetLoaderGGUF
            model, = UnetLoaderGGUF().load_unet(unet_name)
        else:
            path = folder_paths.get_full_path("unet", unet_name) \
                or folder_paths.get_full_path_or_raise("unet", unet_name)
            model = comfy.sd.load_diffusion_model(path)

        unet_config = model.model.model_config.unet_config
        image_model = unet_config.get("image_model")
        num_layers = unet_config.get("num_layers")
        if image_model != "ltxav" or num_layers != _EXPECTED_LAYERS:
            raise ValueError(
                f"{unet_name} built as image_model={image_model} "
                f"num_layers={num_layers}, expected ltxav/{_EXPECTED_LAYERS}. "
                "This is not an LTX-2.3 A/V checkpoint (or its config is "
                "missing - GGUFs from this kit carry the config inside)."
            )
        logger.info("LTX-2.3: %s -> ltxav/%d layers (%.2f GiB stored)",
                    unet_name, num_layers, model.model_size() / 1024 ** 3)

        # ── text encoder → LTXAV CLIP (Gemma + dual projection) ──
        clip = _load_ltxv_clip(text_encoder_name, projections_name)

        # ── VAEs ──
        vae = _load_vae(video_vae_name, want_audio=False)
        audio_vae = _load_vae(audio_vae_name, want_audio=True)

        return (model, clip, vae, audio_vae)


class LTXV23EmptyLatentAV:
    """Empty video+audio latent shaped for the LTX-2.3 A/V transformer.

    Same geometry family as LTX-2.5 (video 128ch / /32 spatial / /8 temporal
    with a causal first frame; audio channels, frequency bins and latents per
    second read off the audio VAE), so this is a thin wrapper over the
    LTX-2.5 builder with LTX-2.3 naming and defaults.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "Empty LTX-2.3 AV Latent ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE", {
                    "tooltip": "The loader's audio_vae output. Only its "
                               "geometry is read (latent channels, frequency "
                               "bins, latents per second) - no encoding "
                               "happens, so it costs nothing."}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 121, "min": 1, "max": nodes.MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Frame count. The video VAE compresses 8:1 in "
                               "time with a causal first frame, so 8k+1 values "
                               "(9, 97, 121...) tile exactly."}),
                "frame_rate": ("FLOAT", {
                    "default": FPS, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "24 for LTX-2.3. Sets the audio latent length; "
                               "use the same value on the sampler."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    DESCRIPTION = ("Joint video+audio latent in LTX-2.x geometry: video "
                   "[B,128,(len-1)//8+1,H/32,W/32] and audio "
                   "[B,z,n_latents,bins] read from the audio VAE.")

    def generate(self, audio_vae, width, height, length, frame_rate, batch_size):
        from .nodes_ltx25 import empty_av_latent
        latent = empty_av_latent(audio_vae, width, height, length, frame_rate,
                                 batch_size)
        video, audio = latent["samples"].unbind()
        logger.info("LTX-2.3 empty AV latent: video %s, audio %s (%d frames @ "
                    "%.2f fps = %.2fs)", tuple(video.shape), tuple(audio.shape),
                    length, frame_rate, length / float(frame_rate))
        return (latent,)


class LTXV23ImgToVideo:
    """Prompts + init latent for LTX-2.3: T2V, I2V, A2V and IA2V in one node.

    Everything except the prompt path is optional, mirroring the official
    workflows' wiring but collapsed into one node:

      * no image, no reference_audio           -> text-to-video
      * image connected                        -> image-to-video (the encoded
        first frame is injected with a per-frame noise mask at
        ``image_strength``, exactly core ``LTXVImgToVideo``'s mechanism)
      * reference_audio connected              -> audio-to-video: the encoded
        audio becomes the audio stream with a ZERO noise mask so it stays
        fixed while the DiT generates matching video (lip sync, Foley). This
        is the official IA2V recipe.
      * image + reference_audio                -> image-AUDIO-to-video

    ``length_from_audio`` (default on when audio is connected) sizes the video
    to the reference clip instead of the ``length`` widget.

    Feed the outputs into the LTX-2.3 KSampler (distilled); split its result
    with core ``LTXVSeparateAVLatent`` and decode video with ``VAE Decode``,
    audio with ``LTXVAudioVAEDecode``.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Img/Audio to Video ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 121, "min": 9, "max": nodes.MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Frame count (8k+1 tiles exactly: 9, 97, "
                               "121...). Ignored when reference_audio is "
                               "connected and length_from_audio is on."}),
                "frame_rate": ("FLOAT", {
                    "default": FPS, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "24 is the LTX-2 convention. Use the same "
                               "value on the sampler."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Init image for i2v. Leave unconnected for "
                               "text-to-video."}),
                "reference_audio": ("AUDIO", {
                    "tooltip": "Reference audio for a2v / ia2v (lip sync). "
                               "Encoded and locked as the audio stream; the "
                               "video is generated to match it. Leave "
                               "unconnected to generate audio from scratch."}),
                "image_strength": ("FLOAT", {
                    "default": I2V_STRENGTH, "min": 0.0, "max": 1.0,
                    "step": 0.01,
                    "tooltip": "i2v only. How much of the init image to keep "
                               "(noise mask on the first frames). 0.7 is the "
                               "official workflows' value."}),
                "length_from_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "With reference_audio: size the video to the "
                               "clip's duration instead of the length widget."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts plus the init AV latent for LTX-2.3: txt2video, "
                   "img2video, audio2video or image+audio2video, all through "
                   "the same node. Outputs feed the LTX-2.3 KSampler "
                   "(distilled).")

    @torch.inference_mode()
    def prepare(self, clip, vae, audio_vae, prompt, negative_prompt, width,
                height, length, frame_rate, batch_size, image=None,
                reference_audio=None, image_strength=I2V_STRENGTH,
                length_from_audio=True):
        from .nodes_ltx25 import VIDEO_LATENT_CHANNELS, VIDEO_SPATIAL_RATIO, \
            VIDEO_TEMPORAL_RATIO

        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError(
                "audio_vae is not an LTX audio VAE (missing "
                "num_of_latents_from_frames); use the kit's "
                "*_audio_vae.safetensors in the audio_vae slot."
            )

        # ── audio stream ──
        if reference_audio is not None:
            seconds = reference_audio["waveform"].shape[-1] / \
                reference_audio["sample_rate"]
            if length_from_audio:
                length = _align_length(seconds * frame_rate + 1)
                logger.info("LTX-2.3 a2v: %.2fs of audio -> %d frames @ %.2f fps",
                            seconds, length, frame_rate)
            audio, audio_mask = _encode_reference_audio(
                audio_vae, reference_audio, length / frame_rate)
        else:
            audio = None
            audio_mask = None

        # ── video stream (+ first-frame conditioning) ──
        length = _align_length(length)
        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        video = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO],
            device=comfy.model_management.intermediate_device())
        video_mask = None
        if image is not None:
            # core LTXVImgToVideo's mechanism: encode the (resized) image,
            # write its latent into the first frames, and hold those frames
            # via a per-frame noise mask of 1 - strength.
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
            t = vae.encode(pixels[:, :, :, :3])
            if t.shape[0] < batch_size:
                t = t.repeat(batch_size, *([1] * (t.dim() - 1)))
            video[:, :, :t.shape[2]] = t.to(video.device, video.dtype)
            video_mask = torch.ones_like(video)
            video_mask[:, :, :t.shape[2]] = 1.0 - image_strength

        # ── audio geometry (empty or fitted reference) ──
        if audio is None:
            channels = int(getattr(audio_vae, "latent_channels",
                                   fsm.latent_channels))
            n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
            audio = torch.zeros(
                [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)],
                device=video.device)
        else:
            target = [batch_size] + list(audio.shape[1:])
            audio, audio_mask = _fit_audio_latent(audio, audio_mask, target)
            if batch_size > 1:
                audio = audio.repeat(batch_size, *([1] * (audio.dim() - 1)))
                audio_mask = (audio_mask.repeat(batch_size, *([1] * (audio_mask.dim() - 1)))
                              if audio_mask is not None else None)
            audio = audio.to(video.device)

        if video_mask is None:
            video_mask = torch.ones_like(video)
        if audio_mask is None:
            audio_mask = torch.ones_like(audio)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor(
                (video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt))
        positive = _set_cond_frame_rate(positive, frame_rate)
        negative = _set_cond_frame_rate(negative, frame_rate)

        logger.info("LTX-2.3 prep: video %s, audio %s%s%s",
                    tuple(video.shape), tuple(audio.shape),
                    ", image held @ %.2f" % image_strength if image is not None else "",
                    ", audio locked" if reference_audio is not None else "")
        return (positive, negative, latent, 1.0)


class LTXV23KSampler:
    """KSampler for LTX-2 distilled checkpoints (euler + official schedules).

    The stock schedulers (simple/karras/...) do not reproduce the trained
    LTX-2 distilled schedule, and a distilled model run on the wrong schedule
    looks like a broken model. This node passes the exact sigmas through -
    8 steps is the trained configuration; other step counts linearly resample
    the same curve. ``schedule = refine`` switches to the official 3-step
    stage-2 pass used after a latent x2 spatial upscale.

    It also stamps ``frame_rate`` onto the conditioning (the DiT's RoPE needs
    it) and honors the noise mask the prep node attaches (i2v held frames,
    a2v locked audio).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 KSampler (distilled) ⚡"

    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000,
                                  "tooltip": "8 for the distilled schedule, "
                                             "3 for refine. Other counts "
                                             "resample the chosen curve."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                  "step": 0.1,
                                  "tooltip": "1.0 - the distilled model is "
                                             "trained without CFG."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"default": "euler"}),
                "schedule": (list(SIGMA_SETS), {"default": "distilled (8 steps)",
                                                "tooltip": "distilled: the "
                                                           "trained 8-step "
                                                           "generation pass. "
                                                           "refine: the "
                                                           "official 3-step "
                                                           "pass after a "
                                                           "latent x2 upscale."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0,
                                         "step": 0.01,
                                         "tooltip": "Stamped onto the "
                                                    "conditioning for RoPE. "
                                                    "Keep it 24 and match the "
                                                    "prep node."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                      "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Sample a joint AV latent with the official LTX-2 schedules "
                   "(distilled 8-step or refine 3-step, cfg 1.0, euler). Split "
                   "the output with core LTXVSeparateAVLatent, then decode "
                   "video with VAE Decode and audio with LTXVAudioVAEDecode.")

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, schedule, frame_rate, denoise):
        positive = _set_cond_frame_rate(positive, frame_rate)
        negative = _set_cond_frame_rate(negative, frame_rate)

        samples = latent_image["samples"]
        sigmas = distilled_sigma_schedule(
            steps, denoise, sigmas=SIGMA_SETS[schedule]).to(
            device=comfy.model_management.intermediate_device(),
            dtype=torch.float32)
        noise = comfy.sample.prepare_noise(
            samples, seed, latent_image.get("batch_index", None))

        out = comfy.sample.sample(
            model, noise, steps=len(sigmas) - 1, cfg=cfg,
            sampler_name=sampler_name, scheduler="simple",
            positive=positive, negative=negative, latent_image=samples,
            sigmas=sigmas, seed=seed,
            noise_mask=latent_image.get("noise_mask", None),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        latent = latent_image.copy()
        latent["samples"] = out
        latent.pop("noise_mask", None)
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader,
    "LTXV23EmptyLatentAV": LTXV23EmptyLatentAV,
    "LTXV23ImgToVideo": LTXV23ImgToVideo,
    "LTXV23KSampler": LTXV23KSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader.TITLE,
    "LTXV23EmptyLatentAV": LTXV23EmptyLatentAV.TITLE,
    "LTXV23ImgToVideo": LTXV23ImgToVideo.TITLE,
    "LTXV23KSampler": LTXV23KSampler.TITLE,
}
