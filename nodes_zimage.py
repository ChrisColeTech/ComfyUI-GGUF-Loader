"""Z-Image nodes: component loader, prompt/latent prep, and an optional sampler.

  ZImageLoader     unet / clip / vae by name -> MODEL, CLIP, VAE
  ZImageImg2Img    model, clip, vae, prompts (+ optional init image, +
                   optional ControlNet) -> model, positive, negative, latent,
                   denoise -> stock KSampler
  ZImageKSampler   optional drop-in for KSampler

Why the ControlNet lives inside the prep node rather than beside it: on
Z-Image, plain img2img cannot recolour or restyle a specific object while
keeping the scene. Measured on the same init in both ComfyUI and the diffusers
stack, at 9 steps: at strength 0.75 the composition holds (corr ~0.89 to the
init) and the requested colours never appear; by the strength where they do
appear (0.95) correlation has collapsed to 0.08-0.24, i.e. the init is gone.
Steps (9 -> 24) and cfg (1.0 -> 3.5) move it by less than 3%. The init latent
carries the scene AND its low-frequency colour, so the two cannot be separated
by any sampler setting. A ControlNet moves structure onto its own channel and
frees the latent, which is the only configuration that produces both.

Everything is optional except the prompt path, so one node covers txt2img
(no image), img2img (image), and either of those with a ControlNet attached.
"""
import logging

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes
import torch

logger = logging.getLogger(__name__)

ZIMAGE_CATEGORY = "\U0001F916 CCTech/Z-Image"

# comfy/text_encoders/z_image.py builds this for a text-only Qwen3-4B.
_EXPECTED_TE = "ZImageTEModel"
# Z-Image latents are 16-channel at an 8x spatial downscale.
_LATENT_CHANNELS = 16
_LATENT_DOWNSCALE = 8


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def _te_name(clip):
    return type(getattr(clip, "cond_stage_model", clip)).__name__


def _attach_controlnet(model, model_patch, vae, image, strength):
    """Attach a Z-Image ControlNet the way comfy_extras' ZImageFunControlnet does.

    Z-Image ControlNets are model patches, not conditioning: residuals are
    injected inside the Lumina forward, so the patch goes on a cloned
    ModelPatcher and the conditioning is left alone.
    """
    try:
        from comfy_extras.nodes_model_patch import ZImageControlPatch
        import comfy.ldm.lumina.controlnet
    except ImportError as e:
        raise RuntimeError(
            "This ComfyUI has no Z-Image ControlNet support "
            "(comfy_extras/nodes_model_patch.py); update ComfyUI."
        ) from e

    inner = getattr(model_patch, "model", None)
    if not isinstance(inner, comfy.ldm.lumina.controlnet.ZImage_Control):
        raise ValueError(
            f"control_patch is a {type(inner).__name__}, not a Z-Image ControlNet. "
            f"Load a Z-Image Fun ControlNet with ModelPatchLoader."
        )

    patched = model.clone()
    patch = ZImageControlPatch(model_patch, vae,
                               None if image is None else image[:, :, :, :3],
                               strength)
    patched.set_model_noise_refiner_patch(patch)
    patched.set_model_double_block_patch(patch)
    return patched


class ZImageLoader:
    TITLE = "Z-Image Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'load clip']
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the three Z-Image components by name. GGUF and "
                   "safetensors both work for the transformer and the text "
                   "encoder; GGUF stays quantized.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "Z-Image transformer from models/diffusion_models. "
                               ".gguf stays quantized."}),
                "clip_name": (_clip_filename_list(), {
                    "tooltip": "TEXT-ONLY Qwen3-4B from models/text_encoders "
                               "(qwen_3_4b.safetensors or a Qwen3-4B GGUF). NOT a "
                               "Qwen3-VL encoder - that one belongs to Qwen-Image."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Z-Image VAE (ae.safetensors)."}),
            },
        }

    def load(self, unet_name, clip_name, vae_name):
        from .nodes import UnetLoaderGGUF, CLIPLoaderGGUF

        if unet_name.lower().endswith(".gguf"):
            model, = UnetLoaderGGUF().load_unet(unet_name)
        else:
            model = comfy.sd.load_diffusion_model(
                folder_paths.get_full_path_or_raise("unet", unet_name))

        if clip_name.lower().endswith(".gguf"):
            # Z-Image's encoder is chosen from the DETECTED model, not from the
            # type dropdown, so any non-FLUX type reaches the right branch.
            clip, = CLIPLoaderGGUF().load_clip(clip_name, type="stable_diffusion")
        else:
            clip = comfy.sd.load_clip(
                ckpt_paths=[folder_paths.get_full_path_or_raise("clip", clip_name)],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        te = _te_name(clip)
        if _EXPECTED_TE not in te:
            logger.warning(
                "Z-Image Loader: '%s' loaded as %s, not %s_. Z-Image expects a "
                "text-only Qwen3-4B; a Qwen3-VL encoder loads cleanly but is read "
                "at the wrong layer and quietly degrades output.",
                clip_name, te, _EXPECTED_TE)
        return (model, clip, vae)


class ZImageImg2Img:
    TITLE = "Z-Image img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'encode image', 'image to latent']
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and optional ControlNet for Z-Image. "
                   "Leave image unconnected for txt2img. Feed the outputs "
                   "straight into a stock KSampler.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "img2img only. How much of the init image "
                                                  "to discard. Ignored without an image."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8,
                                  "tooltip": "Output size. With an init image this resizes it."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Init image for img2img. Leave unconnected "
                                               "for txt2img."}),
                # Z-Image cannot recolour or restyle one object through img2img
                # alone at any strength - see the module docstring. This is the
                # path that works.
                "control_patch": ("MODEL_PATCH", {"tooltip": "ModelPatchLoader with a Z-Image "
                                                             "Fun ControlNet."}),
                "control_image": ("IMAGE", {"tooltip": "Control map, usually Canny."}),
                "control_strength": ("FLOAT", {"default": 0.7, "min": -10.0, "max": 10.0, "step": 0.01,
                                               "tooltip": "1.0 over-conditions and softens detail; "
                                                          "0.7 keeps structure with detail."}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, image=None, control_patch=None, control_image=None,
                control_strength=0.7):
        te = _te_name(clip)
        if _EXPECTED_TE not in te:
            raise ValueError(
                f"This CLIP is a {te}, but Z-Image needs {_EXPECTED_TE}_. Load a "
                f"TEXT-ONLY Qwen3-4B encoder (qwen_3_4b.safetensors or a Qwen3-4B "
                f"GGUF), not a Qwen3-VL one such as qwen3vl_4b_fp8_scaled."
            )
        if control_image is not None and control_patch is None:
            raise ValueError("control_image needs a control_patch; connect a "
                             "ModelPatchLoader to control_patch.")

        if image is None:
            # txt2img: empty latent, and denoise must be 1.0 - there is nothing
            # to preserve, so any lower value would just start from noise the
            # sampler then partially keeps.
            latent = torch.zeros(
                [batch_size, _LATENT_CHANNELS,
                 height // _LATENT_DOWNSCALE, width // _LATENT_DOWNSCALE],
                device=comfy.model_management.intermediate_device())
            denoise = 1.0
            logger.info("Z-Image: txt2img, empty latent %s", tuple(latent.shape))
        else:
            # comfy IMAGE is BHWC; common_upscale wants BCHW.
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(image[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Z-Image: img2img, latent %s, strength %.2f",
                        tuple(latent.shape), strength)

        if control_patch is not None:
            model = _attach_controlnet(model, control_patch, vae, control_image,
                                       control_strength)
            logger.info("Z-Image: ControlNet attached @ %.2f", control_strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        return (model, positive, negative, {"samples": latent}, denoise)


class ZImageKSampler:
    """KSampler for Z-Image, with the one option stock KSampler lacks.

    `denoise_mode`:
      comfy      - delegates to common_ksampler, so it stays identical to
                   KSampler as ComfyUI changes.
      diffusers  - slices a steps-long schedule at
                   t_start = steps - round(steps * denoise), reproducing a
                   diffusers img2img pipeline step for step.

    Measured on Z-Image at 9 steps, denoise 0.9 starts at sigma 0.9643 under
    comfy and 0.9567 under diffusers, so this is a compatibility switch for
    matching another pipeline, not a quality setting.
    """

    TITLE = "Z-Image KSampler ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'txt2img', 'img2img']
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("KSampler with Z-Image Turbo defaults and an optional "
                   "diffusers-style denoise convention.")

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "1.0 disables CFG - correct for Turbo. "
                                             "Z-Image Base wants 3-5."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_mode": (["comfy", "diffusers"], {"default": "comfy"}),
            },
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, denoise_mode):
        if denoise_mode == "comfy":
            return nodes.common_ksampler(model, seed, steps, cfg, sampler_name,
                                         scheduler, positive, negative,
                                         latent_image, denoise=denoise)

        import comfy.sample
        import comfy.samplers
        import latent_preview

        if denoise <= 0.0:
            raise ValueError("denoise must be > 0")

        samples = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"],
            latent_image.get("downscale_ratio_spacial", None),
            latent_image.get("downscale_ratio_temporal", None))

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, steps)
        sigmas = sigmas[int(round(max(steps - steps * denoise, 0))):]

        noise = comfy.sample.prepare_noise(samples, seed,
                                           latent_image.get("batch_index", None))
        out_samples = comfy.sample.sample_custom(
            model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sigmas,
            positive, negative, samples,
            noise_mask=latent_image.get("noise_mask", None),
            callback=latent_preview.prepare_callback(model, len(sigmas) - 1),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)

        logger.info("Z-Image KSampler (diffusers): denoise %.2f -> %d of %d steps, "
                    "start sigma %.4f", denoise, len(sigmas) - 1, steps, float(sigmas[0]))
        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = out_samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ZImageLoader": ZImageLoader,
    "ZImageImg2Img": ZImageImg2Img,
    "ZImageKSampler": ZImageKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoader": ZImageLoader.TITLE,
    "ZImageImg2Img": ZImageImg2Img.TITLE,
    "ZImageKSampler": ZImageKSampler.TITLE,
}
