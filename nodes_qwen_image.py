"""Qwen-Image nodes: component loader, control loader, one-node img2img/control prep.

  QwenImageModelLoader      unet / clip / vae by name -> MODEL, CLIP, VAE
  QwenImageControlNetLoader control checkpoint by name -> QWEN_IMAGE_CONTROL
  QwenImageImg2Img          model, clip, vae, prompts, qwen_control (+ optional
                            init image, + optional control image) -> model,
                            positive, negative, latent, denoise -> stock KSampler

Unlike Krea2's Control LoRA (which needed a from-scratch widened-input-
projection port - nothing in comfy or any other pack implements that
mechanism), Qwen-Image ControlNet is fully native to ComfyUI core across
every format in circulation:

  - InstantX/Union AND Qwen-Image-Fun checkpoints (real diffusers-style
    ControlNet weights - both are actual ControlNet architectures, not model
    patches, despite "Fun" sounding like the DiffSynth-style patches below)
    are auto-detected and built by comfy.controlnet.load_controlnet_state_dict()
    into a real ControlNet object - the exact same object stock
    ControlNetLoader + ControlNetApplyAdvanced produce and consume. This
    attaches to CONDITIONING, like any classic ControlNet.
  - DiffSynth block-wise patches (canny/depth/inpaint) are auto-detected and
    built by comfy_extras.nodes_model_patch's ModelPatchLoader into a
    MODEL_PATCH object, applied via comfy_extras.nodes_model_patch.
    DiffSynthCnetPatch - the same mechanism already used for Z-Image's
    ControlNet in nodes_zimage.py. This attaches to MODEL, not conditioning.

So there is no algorithm to port here - QwenImageControlNetLoader is a thin
dispatcher (reusing core's own detection + core's own model classes, not
reimplementing either), and QwenImageImg2Img (mirroring this repo's
Krea2Img2Img/ZImageImg2Img "one node" convention) routes to whichever
attachment point the loaded control checkpoint actually needs.
"""
import logging

import torch

import comfy.controlnet
import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import nodes

logger = logging.getLogger(__name__)

QWEN_IMAGE_CATEGORY = "\U0001F916 CCTech/Qwen-Image"


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def _control_filename_list():
    """DiffSynth/Fun patches and InstantX/Union checkpoints both work here,
    regardless of which of the two folders they're actually sitting in -
    core's own ModelPatchLoader only scans model_patches, and stock
    ControlNetLoader only scans controlnet, but this loader dispatches on
    file content, not folder, so it needs both lists merged.
    """
    files = folder_paths.get_filename_list("model_patches")
    files += [f for f in folder_paths.get_filename_list("controlnet") if f not in files]
    return sorted(files)


def _resolve_control_path(name):
    path = folder_paths.get_full_path("model_patches", name)
    if path is None:
        path = folder_paths.get_full_path("controlnet", name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}' in models/model_patches or models/controlnet.")
    return path


class QwenImageControl:
    """Wraps whichever of the two attachment mechanisms a control checkpoint needs.

    kind == "controlnet": payload is a real comfy.controlnet.ControlNet,
        attaches to CONDITIONING (positive/negative), like any classic
        ControlNet (InstantX/Union AND Qwen-Image-Fun checkpoints - despite
        the name, Fun is a real ControlNet architecture, not a model patch).
    kind == "model_patch": payload is a MODEL_PATCH (ModelPatcher wrapping a
        QwenImageBlockWiseControlNet), attaches to MODEL via
        DiffSynthCnetPatch (DiffSynth canny/depth/inpaint patches only).
    """

    def __init__(self, kind, payload):
        self.kind = kind
        self.payload = payload


def _apply_controlnet_to_conditioning(conditioning, control_net, control_hint, strength, vae):
    c_net = control_net.copy().set_cond_hint(control_hint, strength, (0.0, 1.0), vae=vae)
    out = []
    for t in conditioning:
        d = t[1].copy()
        c_net.set_previous_controlnet(d.get("control", None))
        d["control"] = c_net
        d["control_apply_to_uncond"] = False
        out.append([t[0], d])
    return out


# ── Nodes ─────────────────────────────────────────────────────────────────

class QwenImageModelLoader:
    """Load the three Qwen-Image components by name.

    Qwen-Image is natively supported by ComfyUI core, so this is a thin
    convenience loader (like Krea2ModelLoader/ZImageLoader) rather than
    anything bespoke - GGUF and safetensors both work for the transformer
    and the text encoder.
    """

    CATEGORY = QWEN_IMAGE_CATEGORY
    TITLE = "Qwen-Image Model Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'load clip', 'qwen image loader',
                       'checkpoint loader']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load Qwen-Image's UNET, CLIP, and VAE as native comfy "
                   "objects. GGUF quants stay quantized.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "Qwen-Image diffusion model - .safetensors or GGUF quant, "
                               "from models/diffusion_models (unet)."}),
                "clip_name": (_clip_filename_list(), {
                    "tooltip": "Qwen-Image text encoder - .safetensors or GGUF quant, "
                               "from models/text_encoders (clip)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Qwen-Image VAE, from models/vae."}),
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
            clip, = CLIPLoaderGGUF().load_clip(clip_name, type="qwen_image")
        else:
            clip = comfy.sd.load_clip(
                ckpt_paths=[folder_paths.get_full_path_or_raise("clip", clip_name)],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.QWEN_IMAGE)

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (model, clip, vae)


class QwenImageControlNetLoader:
    """Load a Qwen-Image control checkpoint - InstantX/Union, Fun, or a
    DiffSynth canny/depth/inpaint patch - auto-detecting which of the two
    comfy-native mechanisms it needs (see the module docstring). Reuses
    core's own detection and model classes; this node is a dispatcher, not
    a reimplementation.
    """

    CATEGORY = QWEN_IMAGE_CATEGORY
    TITLE = "Qwen-Image ControlNet Loader ⚡"
    SEARCH_ALIASES = ['load controlnet', 'controlnet loader', 'control net', 'diffsynth',
                       'instantx', 'qwen controlnet', 'load model patch']
    RETURN_TYPES = ("QWEN_IMAGE_CONTROL",)
    RETURN_NAMES = ("qwen_control",)
    FUNCTION = "load"
    DESCRIPTION = ("Load a Qwen-Image ControlNet or DiffSynth control patch "
                   "from models/model_patches or models/controlnet.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_name": (_control_filename_list(), {
                    "tooltip": "An InstantX/Union or Fun ControlNet checkpoint, or a "
                               "DiffSynth canny/depth/inpaint patch, from "
                               "models/model_patches or models/controlnet."}),
            },
        }

    def load(self, control_name):
        path = _resolve_control_path(control_name)
        sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)

        # Only the block-wise DiffSynth patch format is a MODEL_PATCH. The
        # Qwen-Image-Fun ControlNet ('control_blocks.0.after_proj.weight' +
        # 'control_img_in.weight') is a real ControlNet too - comfy.controlnet's
        # own load_controlnet_state_dict() already dispatches that signature to
        # load_controlnet_qwen_fun() internally, so it's handled by the same
        # branch as InstantX/Union below, not by ModelPatchLoader.
        is_diffsynth_patch = "controlnet_blocks.0.y_rms.weight" in sd

        if is_diffsynth_patch:
            from comfy_extras.nodes_model_patch import ModelPatchLoader
            model_patch, = ModelPatchLoader().load_model_patch(control_name)
            logger.info("Qwen-Image ControlNet: '%s' loaded as a MODEL_PATCH "
                       "(DiffSynth - attaches to MODEL).", control_name)
            return (QwenImageControl("model_patch", model_patch),)

        control = comfy.controlnet.load_controlnet_state_dict(dict(sd))
        if control is None:
            raise RuntimeError(
                f"'{control_name}' was not recognized as a supported Qwen-Image "
                "ControlNet or DiffSynth control patch format.")
        logger.info("Qwen-Image ControlNet: '%s' loaded as a ControlNet "
                   "(InstantX/Union/Fun - attaches to CONDITIONING).", control_name)
        return (QwenImageControl("controlnet", control),)


class QwenImageImg2Img:
    """Prompts, init latent, and optional control attach for Qwen-Image - one node.

    Leave image unconnected for txt2img. Leave qwen_control/control_image
    unconnected for plain img2img/txt2img with no ControlNet. Connect
    qwen_control (from QwenImageControlNetLoader) and control_image together
    to apply control - whichever attachment point the loaded checkpoint
    needs (CONDITIONING for InstantX/Union, MODEL for DiffSynth/Fun) is
    handled automatically based on what QwenImageControlNetLoader detected.

    Feed the outputs straight into a stock KSampler.
    """

    CATEGORY = QWEN_IMAGE_CATEGORY
    TITLE = "Qwen-Image img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img',
                       'encode image', 'image to latent', 'controlnet apply']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and ControlNet attach for Qwen-Image. "
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
                                  "tooltip": "Output size. With an init or control image this resizes it."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Init image for img2img. Leave unconnected "
                                               "for txt2img."}),
                "qwen_control": ("QWEN_IMAGE_CONTROL", {
                    "tooltip": "From QwenImageControlNetLoader. Requires control_image."}),
                "control_image": ("IMAGE", {
                    "tooltip": "Control map matching qwen_control - a canny/depth/pose map "
                               "etc. Required when qwen_control is connected."}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, image=None, qwen_control=None, control_image=None,
                control_strength=1.0):
        if qwen_control is not None and control_image is None:
            raise ValueError(
                "qwen_control was given, but no control_image was connected. Connect a "
                "control image, or remove QwenImageControlNetLoader.")
        if control_image is not None and qwen_control is None:
            logger.warning(
                "QwenImageImg2Img: control_image was given, but no qwen_control was "
                "connected - ignoring control_image.")
            control_image = None

        if image is None:
            # txt2img: a plain, architecture-agnostic empty latent - comfy's own
            # sampling path (comfy.sample.fix_empty_latent_channels, called from
            # common_ksampler) corrects channel count and adds the time dimension
            # for whatever model is attached, the same way stock EmptyLatentImage
            # works across every architecture.
            latent = torch.zeros(
                [batch_size, 4, height // 8, width // 8],
                device=comfy.model_management.intermediate_device())
            denoise = 1.0
            logger.info("Qwen-Image: txt2img, empty latent %s", tuple(latent.shape))
        else:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Qwen-Image: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if qwen_control is not None:
            control_pixels = comfy.utils.common_upscale(
                control_image.movedim(-1, 1), width, height, "lanczos", "disabled"
            ).movedim(1, -1)[:, :, :, :3]

            if qwen_control.kind == "model_patch":
                from comfy_extras.nodes_model_patch import DiffSynthCnetPatch
                model = model.clone()
                model.set_model_double_block_patch(
                    DiffSynthCnetPatch(qwen_control.payload, vae, control_pixels, control_strength))
                logger.info("Qwen-Image: DiffSynth/Fun control patch attached to MODEL")
            else:
                control_hint = control_pixels.movedim(-1, 1)
                positive = _apply_controlnet_to_conditioning(
                    positive, qwen_control.payload, control_hint, control_strength, vae)
                negative = _apply_controlnet_to_conditioning(
                    negative, qwen_control.payload, control_hint, control_strength, vae)
                logger.info("Qwen-Image: ControlNet attached to CONDITIONING")

        return (model, positive, negative, {"samples": latent}, denoise)


NODE_CLASS_MAPPINGS = {
    "QwenImageModelLoader": QwenImageModelLoader,
    "QwenImageControlNetLoader": QwenImageControlNetLoader,
    "QwenImageImg2Img": QwenImageImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageModelLoader": QwenImageModelLoader.TITLE,
    "QwenImageControlNetLoader": QwenImageControlNetLoader.TITLE,
    "QwenImageImg2Img": QwenImageImg2Img.TITLE,
}
