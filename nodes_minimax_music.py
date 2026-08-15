# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Convenience nodes for ComfyUI's native MiniMax Music 3 pipeline."""

import inspect
import logging

import torch

import comfy.model_management
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

from .loader import gguf_clip_loader, gguf_sd_loader
from .ops import GGMLOps


logger = logging.getLogger(__name__)

MUSIC_CATEGORY = "🤖 CCTech/MiniMax Music"
AUDIO_FRAMES_PER_SECOND = 25
MAX_AUDIO_FRAMES = 9000
AUTO_TILE_LATENT_FRAMES = 2600


def _file_list(*folder_keys):
    files = []
    for key in folder_keys:
        try:
            for name in folder_paths.get_filename_list(key):
                if name not in files:
                    files.append(name)
        except KeyError:
            pass
    return sorted(files)


def _full_path(name, *folder_keys):
    for key in folder_keys:
        try:
            path = folder_paths.get_full_path(key, name)
        except KeyError:
            continue
        if path:
            return path
    raise FileNotFoundError(f"Could not find {name} in any of: {', '.join(folder_keys)}")


def _load_diffusion(path):
    if not path.lower().endswith(".gguf"):
        return comfy.sd.load_diffusion_model(path)

    sd, extra = gguf_sd_loader(path)
    kwargs = {}
    valid = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    metadata = extra.get("metadata", {})
    if "metadata" in valid and metadata:
        kwargs["metadata"] = metadata
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": GGMLOps()}, **kwargs)
    if model is not None:
        from .nodes import GGUFModelPatcher
        model = GGUFModelPatcher.clone(model)
    return model


def _load_text_encoder(path):
    clip_type = getattr(comfy.sd.CLIPType, "MINIMAX", None)
    if clip_type is None:
        raise RuntimeError("MiniMax Music 3 requires a newer ComfyUI with CLIPType.MINIMAX support.")

    if not path.lower().endswith(".gguf"):
        return comfy.sd.load_clip(
            ckpt_paths=[path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )

    sd = gguf_clip_loader(path)
    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=clip_type,
        state_dicts=[sd],
        model_options={
            "custom_operations": GGMLOps(),
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        },
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    from .nodes import GGUFModelPatcher
    clip.patcher = GGUFModelPatcher.clone(clip.patcher)
    return clip


def _load_dav(path):
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _zero_conditioning(conditioning):
    negative = []
    for hidden, values in conditioning:
        values = values.copy()
        for key in ("pooled_output", "conditioning_lyrics", "conditioning_scale"):
            value = values.get(key)
            if value is not None:
                values[key] = torch.zeros_like(value)
        negative.append([torch.zeros_like(hidden), values])
    return negative


def _decode_audio(vae, samples, tiled, tile_size, overlap):
    if tiled:
        audio = vae.decode_tiled(
            samples, tile_x=tile_size, tile_y=tile_size, overlap=overlap)
    else:
        audio = vae.decode(samples)
    audio = audio.movedim(-1, 1)

    # Match Comfy's native VAEDecodeAudio normalization.
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    sample_rate = getattr(
        vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": int(sample_rate)}


class MiniMaxMusic3ModelsLoader:
    CATEGORY = MUSIC_CATEGORY
    TITLE = "MiniMax Music 3 Models Loader ⚡"

    @classmethod
    def INPUT_TYPES(cls):
        transformers = [name for name in _file_list(
            "diffusion_models", "unet", "unet_gguf")
            if "minimax_music3_dit" in name.lower()]
        text_encoders = [name for name in _file_list(
            "text_encoders", "clip", "clip_gguf")
            if "minimax_music3_text_encoder" in name.lower()]
        davs = [name for name in _file_list("vae")
                if "minimax_music3_dav" in name.lower()]
        return {
            "required": {
                "transformer_name": (transformers, {
                    "tooltip": "MiniMax Music 3 DiT from models/diffusion_models; native "
                               "ConvRot INT8, fp16/bf16, and GGUF are supported."}),
                "text_encoder_name": (text_encoders, {
                    "tooltip": "Pruned MiniMax Music 3 AR text/audio encoder with embedded "
                               "tokenizer_json, from models/text_encoders."}),
                "dav_name": (davs, {
                    "tooltip": "minimax_music3_dav.safetensors from models/vae."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the MiniMax Music 3 AR conditioner, flow DiT, and DAV "
                   "decoder through ComfyUI's native model implementations.")

    def load(self, transformer_name, text_encoder_name, dav_name):
        model = _load_diffusion(_full_path(
            transformer_name, "diffusion_models", "unet", "unet_gguf"))
        if model is None:
            raise RuntimeError(f"Could not detect {transformer_name} as MiniMax Music 3.")
        model_config = getattr(getattr(model.model, "model_config", None), "unet_config", {})
        if model_config.get("audio_model") != "minimax_music3":
            raise RuntimeError(f"{transformer_name} is not a MiniMax Music 3 DiT checkpoint.")

        clip = _load_text_encoder(_full_path(
            text_encoder_name, "text_encoders", "clip", "clip_gguf"))
        if type(clip.cond_stage_model).__name__ != "MiniMaxMusic3TEModel":
            raise RuntimeError(
                f"{text_encoder_name} is not a MiniMax Music 3 pruned AR checkpoint.")

        vae = _load_dav(_full_path(dav_name, "vae"))
        if (getattr(vae, "latent_channels", None) != 128
                or getattr(vae, "audio_sample_rate", None) != 44100):
            raise RuntimeError(f"{dav_name} is not a MiniMax Music 3 DAV checkpoint.")
        return model, clip, vae


class MiniMaxMusic3AudioGenerate:
    CATEGORY = MUSIC_CATEGORY
    TITLE = "MiniMax Music 3 Audio Generate ⚡"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "caption": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "Global Metadata: Dreamy synth-pop, 112 BPM, warm analog production.\n\n"
                               "Vocal Details: Expressive lead vocal with layered harmonies.\n\n"
                               "Arrangement: Atmospheric intro, intimate verses, wide anthemic choruses.",
                    "tooltip": "Describe genre, BPM, key, mood, production, vocal character, "
                               "instruments, and section arrangement."}),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "[intro]\n\n[verse]\nWrite your first verse here\n\n"
                               "[chorus]\nWrite your chorus here\n\n[outro]",
                    "tooltip": "Use tags such as [intro], [verse], [pre-chorus], "
                               "[chorus], [bridge], [instrumental], [drop], and [outro]."}),
                "max_duration": ("FLOAT", {
                    "default": 120.0, "min": 0.04, "max": 360.0, "step": 0.04,
                    "tooltip": "Upper bound in seconds; the model may finish earlier."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {
                    "default": 1.7, "min": 0.0, "max": 100.0, "step": 0.1,
                    "round": 0.01}),
                "decode_mode": (["auto", "tiled", "dense"], {"default": "auto"}),
                "ar_cfg_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1,
                    "round": 0.01, "advanced": True}),
                "ar_top_k": ("INT", {
                    "default": 50, "min": 1, "max": 16384, "advanced": True}),
                "tile_size": ("INT", {
                    "default": 1536, "min": 32, "max": 8192, "step": 8,
                    "advanced": True}),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 1024, "step": 8,
                    "advanced": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    DESCRIPTION = ("Generate caption- and lyrics-conditioned stereo music with the "
                   "native MiniMax Music 3 AR, flow DiT, and DAV pipeline.")

    @torch.inference_mode()
    def generate(self, model, clip, vae, caption, lyrics, max_duration, seed,
                 steps, guidance_scale, decode_mode, ar_cfg_scale=1.5,
                 ar_top_k=50, tile_size=1536, tile_overlap=64):
        if not caption or not caption.strip():
            raise ValueError("caption must not be empty")
        if not lyrics or not lyrics.strip():
            raise ValueError("lyrics must not be empty")
        if type(clip.cond_stage_model).__name__ != "MiniMaxMusic3TEModel":
            raise RuntimeError("Connect the CLIP output of MiniMax Music 3 Models Loader.")
        model_config = getattr(getattr(model.model, "model_config", None), "unet_config", {})
        if model_config.get("audio_model") != "minimax_music3":
            raise RuntimeError("Connect the MODEL output of MiniMax Music 3 Models Loader.")
        if (getattr(vae, "latent_channels", None) != 128
                or getattr(vae, "audio_sample_rate", None) != 44100
                or getattr(vae, "upscale_ratio", None) != 512):
            raise RuntimeError("Connect the VAE output of MiniMax Music 3 Models Loader.")
        if tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")

        max_frames = min(
            MAX_AUDIO_FRAMES,
            max(1, round(max_duration * AUDIO_FRAMES_PER_SECOND)),
        )
        tokens = clip.tokenize(
            caption,
            lyrics=lyrics,
            seed=seed,
            max_audio_frames=max_frames,
            cfg_scale=ar_cfg_scale,
            top_k=ar_top_k,
        )
        positive = clip.encode_from_tokens_scheduled(tokens)
        for hidden, values in positive:
            values["conditioning_scale"] = torch.ones(
                (hidden.shape[0], 1, 1), device=hidden.device, dtype=hidden.dtype)
        negative = _zero_conditioning(positive)

        audio_frames = positive[0][0].shape[1]
        from comfy.ldm.minimax_music.dit import latent_length
        latent_frames = latent_length(audio_frames)
        latent_image = torch.zeros(
            (1, 128, latent_frames),
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        noise = comfy.sample.prepare_noise(latent_image, seed)
        samples = comfy.sample.sample(
            model,
            noise,
            steps=steps,
            cfg=guidance_scale,
            sampler_name="euler",
            scheduler="simple",
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            seed=seed,
        )

        tiled = decode_mode == "tiled" or (
            decode_mode == "auto" and latent_frames > AUTO_TILE_LATENT_FRAMES)
        logger.info(
            "MiniMax Music 3: generated %d AR frames (%.2fs); decoding %s",
            audio_frames,
            audio_frames / AUDIO_FRAMES_PER_SECOND,
            "tiled" if tiled else "dense",
        )
        return (_decode_audio(vae, samples, tiled, tile_size, tile_overlap),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3ModelsLoader": MiniMaxMusic3ModelsLoader,
    "MiniMaxMusic3AudioGenerate": MiniMaxMusic3AudioGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3ModelsLoader": MiniMaxMusic3ModelsLoader.TITLE,
    "MiniMaxMusic3AudioGenerate": MiniMaxMusic3AudioGenerate.TITLE,
}
