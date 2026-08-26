"""Qwen-Image nodes: component loader, control loader, one-node img2img/control prep.

  QwenImageModelLoader      unet / clip / vae by name -> MODEL, CLIP, VAE
  QwenImageControlNetLoader control checkpoint by name -> QWEN_IMAGE_CONTROL
  QwenImageImg2Img          model, clip, vae, prompts (+ optional init
                            image) -> model, positive, negative, latent,
                            denoise -> stock KSampler. No ControlNet inputs.
  QwenImageControlNetImg2Img  same as QwenImageImg2Img, plus qwen_control/
                            control_image/control_mode for ControlNet-
                            guided generation - kept as a separate node so
                            the plain one above never shows control inputs.

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
import node_helpers
import nodes

from ..vendor import depth_anything_v2
from .preprocessors import Canny, _auto_canny_control_image, _depth_anything_batch

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
        from .gguf import UnetLoaderGGUF, CLIPLoaderGGUF

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
    """Prompts and init latent for Qwen-Image - one node, no ControlNet.

    One image-shaped slot: `images`. Leave unconnected for txt2img. For
    ControlNet-guided generation use `QwenImageControlNetImg2Img` instead -
    kept as a separate node on purpose, so this one never shows control-
    related inputs at all.

    `images` is real img2img: VAE-encoded, then partially denoised at
    `strength`. It's batch-aware - a batch of N photos naturally becomes N
    independent img2img generations, since `vae.encode()`/`KSampler` already
    process a batched latent as N parallel runs, no special-casing needed.
    This is also the only mechanism this node has for "edit this photo" -
    for a genuine Qwen-Image-Edit checkpoint, connect the photo here and use
    a real `strength` (not just a control map) so the model is actually
    denoising from it.

    Feed the outputs straight into a stock KSampler.
    """

    CATEGORY = QWEN_IMAGE_CATEGORY
    TITLE = "Qwen-Image img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img',
                       'encode image', 'image to latent']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts and init latent for Qwen-Image, no ControlNet. "
                   "Leave images unconnected for txt2img. For ControlNet-"
                   "guided generation use QwenImageControlNetImg2Img instead. "
                   "Feed the outputs straight into a stock KSampler.")

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
                                       "tooltip": "img2img only. How much of the init image(s) "
                                                  "to discard. Ignored without images."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8,
                                  "tooltip": "Output size. With init image(s) this resizes them."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "One or more init images for img2img "
                                               "(batch-aware - a batch of N becomes N "
                                               "independent img2img generations). Leave "
                                               "unconnected for txt2img."}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, images=None):
        if images is None:
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
                images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Qwen-Image: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        return (model, positive, negative, {"samples": latent}, denoise)


class QwenImageControlNetImg2Img:
    """Prompts, init latent, and control attach for Qwen-Image - ControlNet img2img.

    Same as `QwenImageImg2Img` plus everything needed for ControlNet-guided
    generation, kept in its own node so the plain img2img node never has to
    show control-related inputs. Two image-shaped slots, dead simple:
    `images` for img2img, `control_image` for ControlNet. Leave `images`
    unconnected for txt2img. Leave qwen_control/control_image unconnected
    for plain img2img/txt2img with no ControlNet. Connect qwen_control
    (from QwenImageControlNetLoader) to apply control - whichever
    attachment point the loaded checkpoint needs (CONDITIONING for
    InstantX/Union/Fun, MODEL for DiffSynth) is handled automatically
    based on what QwenImageControlNetLoader detected.

    `images` is real img2img: VAE-encoded, then partially denoised at
    `strength`. It's batch-aware - a batch of N photos naturally becomes N
    independent img2img generations, since `vae.encode()`/`KSampler` already
    process a batched latent as N parallel runs, no special-casing needed.

    control_mode picks how the control image is produced, since (unlike
    Krea2, which only has one control type) Qwen-Image checkpoints span
    several different preprocessing needs this pack can't detect from the
    loaded file:
      "manual" (default) - control_image must be supplied by hand (a canny
        map, depth map, etc. matching whichever checkpoint you loaded).
      "auto_canny" - derives a canny edge map from `images` automatically
        (plain cv2.Canny, no model, no extra download) - for a canny
        DiffSynth/Union checkpoint.
      "auto_depth" - derives a depth map from `images` automatically, using
        the same Depth Anything V2 model as Krea2DepthMap - for a depth
        DiffSynth/Union checkpoint.
    Connecting control_image explicitly always overrides auto-derivation,
    in any mode. control_image is required even for an inpaint checkpoint;
    connect `mask` alongside it to refine which region gets inpainted -
    mask alone isn't enough, there's no photo-only way to auto-derive one.

    Feed the outputs straight into a stock KSampler.
    """

    CATEGORY = QWEN_IMAGE_CATEGORY
    TITLE = "Qwen-Image ControlNet img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img',
                       'encode image', 'image to latent', 'controlnet apply',
                       'controlnet img2img']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and ControlNet attach for Qwen-Image. "
                   "Leave images unconnected for txt2img. Feed the outputs "
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
                                       "tooltip": "img2img only. How much of the init image(s) "
                                                  "to discard. Ignored without images."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8,
                                  "tooltip": "Output size. With init or control image(s) this resizes them."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "One or more init images for img2img "
                                               "(batch-aware - a batch of N becomes N "
                                               "independent img2img generations), and (in "
                                               "auto_canny/auto_depth modes) the source photo "
                                               "the control image is derived from. Leave "
                                               "unconnected for txt2img."}),
                "qwen_control": ("QWEN_IMAGE_CONTROL", {
                    "tooltip": "From QwenImageControlNetLoader."}),
                "control_mode": (["manual", "auto_canny", "auto_depth", "none"], {"default": "manual",
                    "tooltip": "manual: connect control_image yourself. auto_canny/auto_depth: "
                               "derive it from `images` automatically - pick whichever matches "
                               "the loaded checkpoint. none: skip control attachment entirely "
                               "even if qwen_control is connected - for toggling control off "
                               "without rewiring. Ignored without qwen_control."}),
                "depth_ckpt_name": (list(depth_anything_v2.MODEL_CONFIGS.keys()), {
                    "default": "depth_anything_v2_vitb.pth",
                    "tooltip": "auto_depth mode only. Downloads on first use if not already "
                               "in models/depth_anything_v2."}),
                "control_image": ("IMAGE", {
                    "tooltip": "Control map matching qwen_control - a canny/depth/etc map. "
                               "Overrides auto_canny/auto_depth when connected. Required in "
                               "manual mode."}),
                "mask": ("MASK", {"tooltip": "For an inpaint checkpoint - the region to "
                                            "inpaint. Not derivable automatically."}),
                "control_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, images=None, qwen_control=None, control_mode="manual",
                depth_ckpt_name="depth_anything_v2_vitb.pth", control_image=None,
                mask=None, control_strength=1.0):
        if control_image is not None and qwen_control is None:
            logger.warning(
                "QwenImageControlNetImg2Img: control_image was given, but no qwen_control was "
                "connected - ignoring control_image.")
            control_image = None

        if qwen_control is not None and control_image is None and control_mode != "none":
            if control_mode == "auto_canny" and images is not None:
                logger.info("Qwen-Image: auto-deriving a canny edge map from images "
                            "(control_mode=auto_canny)")
                control_image = _auto_canny_control_image(images)
            elif control_mode == "auto_depth" and images is not None:
                logger.info("Qwen-Image: auto-deriving a depth map from images "
                            "(control_mode=auto_depth)")
                control_image = _depth_anything_batch(images, depth_ckpt_name)
            else:
                raise ValueError(
                    "qwen_control was given, but no usable control_image is available. "
                    "Connect control_image directly, or set control_mode to auto_canny/"
                    "auto_depth with images connected. control_image is required even "
                    "for an inpaint checkpoint - mask (optional) only refines the region, "
                    "it doesn't replace it.")
        elif qwen_control is not None and control_mode == "none":
            logger.info("Qwen-Image: control_mode=none - skipping control attachment even "
                        "though qwen_control is connected.")

        if images is None:
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
                images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Qwen-Image: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if qwen_control is not None and control_image is not None:
            control_pixels = comfy.utils.common_upscale(
                control_image.movedim(-1, 1), width, height, "lanczos", "disabled"
            ).movedim(1, -1)[:, :, :, :3]

            if qwen_control.kind == "model_patch":
                from comfy_extras.nodes_model_patch import DiffSynthCnetPatch
                patch_mask = mask
                if patch_mask is not None:
                    if patch_mask.ndim == 3:
                        patch_mask = patch_mask.unsqueeze(1)
                    if patch_mask.ndim == 4:
                        patch_mask = patch_mask.unsqueeze(2)
                    patch_mask = 1.0 - patch_mask
                model = model.clone()
                model.set_model_double_block_patch(
                    DiffSynthCnetPatch(qwen_control.payload, vae, control_pixels,
                                       control_strength, patch_mask))
                logger.info("Qwen-Image: DiffSynth control patch attached to MODEL%s",
                          " (with inpaint mask)" if patch_mask is not None else "")
            else:
                control_hint = control_pixels.movedim(-1, 1)
                positive = _apply_controlnet_to_conditioning(
                    positive, qwen_control.payload, control_hint, control_strength, vae)
                negative = _apply_controlnet_to_conditioning(
                    negative, qwen_control.payload, control_hint, control_strength, vae)
                logger.info("Qwen-Image: ControlNet attached to CONDITIONING")

        return (model, positive, negative, {"samples": latent}, denoise)


class QwenImageKSampler:
    """KSampler for Qwen-Image, with the one option stock KSampler lacks.

    `denoise_mode`:
      comfy      - delegates to common_ksampler, so it stays identical to
                   KSampler as ComfyUI changes.
      diffusers  - slices a steps-long schedule at
                   t_start = steps - round(steps * denoise), reproducing a
                   diffusers img2img pipeline step for step.

    Qwen-Image has no bespoke sampling code in comfy at all - it shares
    ModelSamplingFlux (shift=1.15) with the rest of the Flux family, and
    comfy's own denoise-slicing convention (KSampler.set_steps:
    new_steps=int(steps/denoise), take the tail) genuinely diverges from
    the diffusers img2img convention used above - confirmed by recomputing
    both schedules for Qwen-Image's actual shift value: at 9 steps,
    denoise 0.9 starts at sigma ~0.9660 under comfy vs ~0.9619 under
    diffusers. Smaller than Z-Image's measured gap (0.9643 vs 0.9567) but
    the same mechanism, so this is a compatibility switch for matching
    another pipeline, not a quality setting - "comfy" (default) is
    unchanged stock behavior.
    """

    TITLE = "Qwen-Image KSampler ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'txt2img', 'img2img']
    CATEGORY = QWEN_IMAGE_CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("KSampler for Qwen-Image with an optional diffusers-style "
                   "denoise convention for exact img2img pipeline matching.")

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
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 100.0, "step": 0.1}),
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

        logger.info("Qwen-Image KSampler (diffusers): denoise %.2f -> %d of %d steps, "
                    "start sigma %.4f", denoise, len(sigmas) - 1, steps, float(sigmas[0]))
        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = out_samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "QwenImageModelLoader": QwenImageModelLoader,
    "QwenImageControlNetLoader": QwenImageControlNetLoader,
    # Backward-compat alias: this node's logic moved to the shared
    # nodes/preprocessors.py Canny class - old saved workflows using the
    # "QwenImageCanny" type id keep resolving to a working node.
    "QwenImageCanny": Canny,
    "QwenImageImg2Img": QwenImageImg2Img,
    "QwenImageControlNetImg2Img": QwenImageControlNetImg2Img,
    "QwenImageKSampler": QwenImageKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageModelLoader": QwenImageModelLoader.TITLE,
    "QwenImageControlNetLoader": QwenImageControlNetLoader.TITLE,
    "QwenImageCanny": "Qwen-Image Canny ⚡",
    "QwenImageKSampler": QwenImageKSampler.TITLE,
    "QwenImageImg2Img": QwenImageImg2Img.TITLE,
    "QwenImageControlNetImg2Img": QwenImageControlNetImg2Img.TITLE,
}
