"""Z-Image nodes: img2img with diffusers-parity strength, and the sigma helper.

Z-Image is easy to wire wrongly in ComfyUI, and every way of getting it wrong
fails quietly rather than loudly:

  * A Qwen-Image text encoder (`qwen3vl_4b_fp8_scaled`) loads without error and
    produces 2560-wide conditioning, same as Z-Image's own encoder, but read at
    the wrong layer. Output is noise or mush and nothing in the log says why.
  * Swapping the `positive` and `negative` links costs nothing at runtime.
  * `denoise` on KSampler and `strength` in the diffusers pipelines are defined
    differently, so settings do not port across unchanged.

`ZImageImg2Img` collapses the whole graph into one node so none of those are
reachable, and it validates the encoder up front with an actionable message.
`ZImageStrengthSigmas` exposes just the schedule maths for anyone who would
rather build the graph by hand.

Turbo defaults: 9 steps, cfg 1.0, euler/simple. Z-Image *Base* wants ~30-50
steps at cfg 3-5 — override the widgets for that.
"""
import logging

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import nodes
import torch

logger = logging.getLogger(__name__)

# Sibling of the other suite roots (see nodes_ltx25.py): the emoji prefix is
# what groups them together in the node menu.
ZIMAGE_CATEGORY = "🤖 CCTech/Z-Image"

# The encoder Z-Image was trained against: a text-only Qwen3-4B, read at
# layer_idx=-2 (comfy/text_encoders/z_image.py). Qwen-Image's Qwen3-VL encoder
# is also 2560-wide and loads happily, which is what makes the mistake silent.
_EXPECTED_TE = "ZImageTEModel"


def _check_text_encoder(clip):
    name = type(getattr(clip, "cond_stage_model", clip)).__name__
    if _EXPECTED_TE not in name:
        raise ValueError(
            f"This CLIP is a {name}, but Z-Image needs {_EXPECTED_TE}_. "
            f"Load a TEXT-ONLY Qwen3-4B encoder (e.g. qwen_3_4b.safetensors, or "
            f"a Qwen3-4B GGUF) rather than a Qwen3-VL one such as "
            f"qwen3vl_4b_fp8_scaled.safetensors, which belongs to Qwen-Image."
        )


def strength_sigmas(model, scheduler: str, steps: int, strength: float):
    """Sigmas for `strength`, using the diffusers img2img convention.

    diffusers builds the full `steps`-long schedule and starts partway in:

        t_start = steps - round(steps * strength)
        sigmas  = sigmas[t_start:]

    ComfyUI's KSampler instead builds a LONGER schedule (`steps / denoise`) and
    keeps its tail. The two land in much the same place, but only this one
    matches a diffusers pipeline step for step, which matters when porting
    settings or reproducing a result from another stack.
    """
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps)
    t_start = int(round(max(steps - steps * strength, 0)))
    return sigmas[t_start:]


def _apply_control_patch(model, model_patch, vae, image, strength):
    """Attach a Z-Image ControlNet the way ZImageFunControlnet does.

    Z-Image ControlNets are model patches, not conditioning: the residuals are
    injected inside the Lumina forward, so the patch is set on a cloned
    ModelPatcher rather than applied to `positive`. Mirrors
    comfy_extras/nodes_model_patch.py so behaviour stays identical to the
    stock node.
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
    if image is not None:
        image = image[:, :, :, :3]

    patched = model.clone()
    patch = ZImageControlPatch(model_patch, vae, image, strength)
    patched.set_model_noise_refiner_patch(patch)
    patched.set_model_double_block_patch(patch)
    return patched


class ZImageStrengthSigmas:
    TITLE = "Z-Image Strength → Sigmas ⚡"
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    DESCRIPTION = ("Sigma schedule for a diffusers-style img2img `strength`. "
                   "Feed into SamplerCustom together with a VAEEncode latent.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            "steps": ("INT", {"default": 9, "min": 1, "max": 10000}),
            "strength": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01}),
        }}

    def get_sigmas(self, model, scheduler, steps, strength):
        return (strength_sigmas(model, scheduler, steps, strength),)


class ZImageImg2Img:
    TITLE = "Z-Image img2img ⚡"
    CATEGORY = ZIMAGE_CATEGORY
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "generate"
    DESCRIPTION = ("Z-Image img2img in one node: prompt encode, VAE encode, "
                   "diffusers-style strength, sample, decode. Validates that the "
                   "CLIP is actually Z-Image's encoder.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01,
                                       "tooltip": "1.0 discards the input image entirely."}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "1.0 for Turbo (no CFG); 3-5 for Z-Image Base."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                # 0 keeps the input image's dimensions. Z-Image is a 1024-native
                # model, so anything much smaller composes poorly.
                "width": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                # Picked here so a LoRA needs no extra node. Chaining a
                # LoraLoader into `model`/`clip` still works and stacks with it.
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                # Attach ModelPatchLoader here to skip the ZImageFunControlnet
                # node. Feeding an already-patched MODEL into `model` also works;
                # do one or the other, not both.
                "control_patch": ("MODEL_PATCH",),
                "control_image": ("IMAGE",),
                "control_strength": ("FLOAT", {"default": 0.7, "min": -10.0, "max": 10.0, "step": 0.01,
                                               "tooltip": "1.0 over-conditions and softens detail; ~0.7 keeps structure with detail."}),
            },
        }

    def generate(self, model, clip, vae, image, prompt, negative_prompt, strength,
                 steps, cfg, seed, sampler_name, scheduler, width=0, height=0,
                 lora_name="None", lora_strength=1.0, control_patch=None,
                 control_image=None, control_strength=0.7):
        _check_text_encoder(clip)

        if lora_name and lora_name != "None" and lora_strength != 0.0:
            lora = comfy.utils.load_torch_file(
                folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(
                model, clip, lora, lora_strength, lora_strength)
            logger.info("Z-Image img2img: merged LoRA %s @ %.2f", lora_name, lora_strength)

        if control_patch is not None:
            model = _apply_control_patch(model, control_patch, vae,
                                         control_image, control_strength)
            logger.info("Z-Image img2img: ControlNet attached @ %.2f", control_strength)
        elif control_image is not None:
            raise ValueError("control_image was given without a control_patch; "
                             "connect a ModelPatchLoader to control_patch.")

        if width and height:
            # movedim: comfy IMAGE is BHWC, common_upscale wants BCHW.
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)

        latent = vae.encode(image[:, :, :, :3])
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        sigmas = strength_sigmas(model, scheduler, steps, strength)
        sampler = comfy.samplers.sampler_object(sampler_name)
        noise = comfy.sample.prepare_noise(latent, seed)

        callback = latent_preview.prepare_callback(model, len(sigmas) - 1)
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas, positive, negative, latent,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )

        logger.info("Z-Image img2img: strength %.2f -> %d of %d steps, "
                    "start sigma %.4f", strength, len(sigmas) - 1, steps,
                    float(sigmas[0]))
        return (vae.decode(samples), {"samples": samples})


NODE_CLASS_MAPPINGS = {
    "ZImageImg2Img": ZImageImg2Img,
    "ZImageStrengthSigmas": ZImageStrengthSigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageImg2Img": ZImageImg2Img.TITLE,
    "ZImageStrengthSigmas": ZImageStrengthSigmas.TITLE,
}
