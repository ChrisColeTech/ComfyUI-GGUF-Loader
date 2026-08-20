"""Z-Image nodes: component loader, img2img prep, and an optional sampler.

Three nodes, each doing one thing:

  ZImageLoader     unet / clip / vae by name  ->  MODEL, CLIP, VAE
  ZImageImg2Img    clip, vae, image, prompts, strength
                       ->  positive, negative, latent, denoise
  ZImageKSampler   optional drop-in for KSampler, with Z-Image defaults and a
                   diffusers-compatible denoise convention

ZImageImg2Img samples nothing - feed it to stock KSampler, or to ZImageKSampler
if you want the diffusers denoise mode. Everything else — LoRA, ControlNet,
Canny, image loading — is a stock node wired in around these.

The one thing worth encoding here is the trap that cost real debugging time: a
Qwen-Image text encoder (`qwen3vl_4b_fp8_scaled`) loads without error and emits
2560-wide conditioning exactly like Z-Image's own, but read at layer_idx=-1
instead of -2. Nothing fails; output just degrades. ZImageImg2Img refuses it by
name instead.
"""
import logging

import comfy.sd
import comfy.utils
import folder_paths
import nodes

logger = logging.getLogger(__name__)

ZIMAGE_CATEGORY = "\U0001F916 CCTech/Z-Image"

# comfy/text_encoders/z_image.py builds this for a text-only Qwen3-4B.
_EXPECTED_TE = "ZImageTEModel"


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


class ZImageLoader:
    TITLE = "Z-Image Loader ⚡"
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
            # A warning rather than an error: the loader is also a convenient way
            # to pull components for something else. ZImageImg2Img is where it
            # actually matters, and that one raises.
            logger.warning(
                "Z-Image Loader: '%s' loaded as %s, not %s_. Z-Image expects a "
                "text-only Qwen3-4B; a Qwen3-VL encoder loads cleanly but is read "
                "at the wrong layer and quietly degrades output.",
                clip_name, te, _EXPECTED_TE)
        return (model, clip, vae)


class ZImageImg2Img:
    TITLE = "Z-Image img2img ⚡"
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Encode the prompts and the init image for Z-Image img2img. "
                   "Feed positive/negative/latent into a stock KSampler and "
                   "connect denoise to its denoise input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "How much of the init image to discard. "
                                                  "0.6 keeps the composition; 1.0 ignores it."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096,
                                       "tooltip": "Variations of the same init image. The "
                                                  "encoded latent is tiled; KSampler draws "
                                                  "different noise per batch item."}),
            },
            "optional": {
                # 0 keeps the input image's size. Z-Image is 1024-native.
                "width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
        }

    def prepare(self, clip, vae, image, prompt, negative_prompt, strength,
                batch_size=1, width=0, height=0):
        te = _te_name(clip)
        if _EXPECTED_TE not in te:
            raise ValueError(
                f"This CLIP is a {te}, but Z-Image needs {_EXPECTED_TE}_. Load a "
                f"TEXT-ONLY Qwen3-4B encoder (qwen_3_4b.safetensors or a Qwen3-4B "
                f"GGUF), not a Qwen3-VL one such as qwen3vl_4b_fp8_scaled."
            )

        if width and height:
            # comfy IMAGE is BHWC; common_upscale wants BCHW.
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)

        latent = vae.encode(image[:, :, :, :3])
        if batch_size > 1:
            # Tile, matching RepeatLatentBatch. The latent is identical across
            # the batch; the variation comes from KSampler drawing separate
            # noise per item.
            latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        logger.info("Z-Image img2img: latent %s, strength %.2f",
                    tuple(latent.shape), strength)
        return (positive, negative, {"samples": latent}, strength)


NODE_CLASS_MAPPINGS = {
    "ZImageLoader": ZImageLoader,
    "ZImageImg2Img": ZImageImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageLoader": ZImageLoader.TITLE,
    "ZImageImg2Img": ZImageImg2Img.TITLE,
}


class ZImageKSampler:
    """KSampler for Z-Image, with the option stock KSampler does not offer.

    `denoise_mode` is the only real reason this exists rather than using
    KSampler with different defaults:

      comfy      - identical to KSampler. Builds a LONGER schedule
                   (steps / denoise) and keeps its tail.
      diffusers  - builds the `steps`-long schedule and starts partway in
                   (t_start = steps - round(steps * denoise)), which is what the
                   diffusers img2img pipelines do. Use this to reproduce a
                   result from a diffusers stack step for step.

    The two land close together in practice - measured on Z-Image at 9 steps,
    denoise 0.9 starts at sigma 0.9643 under comfy and 0.9567 under diffusers -
    so reach for `diffusers` when matching another pipeline, not as a quality
    setting.

    Defaults are Z-Image Turbo: 9 steps, cfg 1.0, euler/simple. Z-Image Base
    wants roughly 30-50 steps at cfg 3-5.
    """

    TITLE = "Z-Image KSampler ⚡"
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
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Connect the Z-Image img2img node's "
                                                 "denoise output here for img2img."}),
                "denoise_mode": (["comfy", "diffusers"], {"default": "comfy"}),
            },
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, denoise_mode):
        if denoise_mode == "comfy":
            # Delegate rather than reimplement, so this stays bit-identical to
            # KSampler as ComfyUI changes.
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
        t_start = int(round(max(steps - steps * denoise, 0)))
        sigmas = sigmas[t_start:]

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


NODE_CLASS_MAPPINGS["ZImageKSampler"] = ZImageKSampler
NODE_DISPLAY_NAME_MAPPINGS["ZImageKSampler"] = ZImageKSampler.TITLE
