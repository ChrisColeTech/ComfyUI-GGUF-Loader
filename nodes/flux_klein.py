"""FLUX.2 Klein nodes: component loader, one-node img2img prep, and a full
port of ComfyUI-Flux2Klein-Enhancer's multi-reference/identity-transfer/
conditioning tools.

  FluxKleinModelLoader        unet / clip / vae by name -> MODEL, CLIP, VAE
  FluxKleinImg2Img            model, clip, vae, prompts (+ optional
                               reference images / controlnet-style source
                               image) -> model, positive, negative, latent,
                               denoise -> stock KSampler or FluxKleinKSampler
  Flux2KleinMultiReferenceLatent  up to 8 reference latents -> CONDITIONING

Flux.2 Klein is natively supported by ComfyUI core (unet_config.image_model
== "flux2", comfy/supported_models.py:795) - the same situation as Krea2/
Qwen-Image: no bespoke sampling algorithm or conditioning format to
reimplement for the base loading/generation path.

Everything below Part C is ported from ComfyUI-Flux2Klein-Enhancer by
capitan01R (MIT License,
https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer). That pack uses
ONLY stock ComfyUI ModelPatcher/conditioning APIs - set_model_attn1_patch/
set_model_attn1_output_patch (fired generically from comfy's own
comfy/ldm/flux/layers.py DoubleStreamBlock/SingleStreamBlock, shared Flux/
Kontext/Klein code, not Klein-repo-specific), model_options
["sampler_post_cfg_function"] (fired only through comfy's own CFGGuider/
sampling_function, i.e. only when sampled via a real comfy sampling path),
and plain conditioning-dict mutation. The four extra_options keys the
identity-transfer/reference-control nodes depend on
(reference_image_num_tokens, block_index, block_type, img_slice) are all
populated by comfy's own generic Flux model code - the source pack does
zero independent computation of Klein internals, which is what makes this
a clean, faithful port rather than a reimplementation.

Not ported: Flux2KleinKSamplerExperimental (confirmed strictly LESS
compatible than stock KSampler within the source pack itself - it bypasses
comfy's sampler_post_cfg_function pipeline entirely, silently breaking
that same pack's own Color Anchor and Identity Guidance nodes when paired
with it) and the source's own explicitly-superseded older Identity
Feature Transfer / Advanced / V3 nodes (kept in the source only for ITS
OWN backward compatibility).
"""
import gc
import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes

from ..vendor import depth_anything_v2
from .preprocessors import DepthMap, _auto_canny_control_image, _depth_anything_batch

# Minimum control_mode set, matching Krea2Img2Img/QwenImageImg2Img exactly -
# depth + canny are the only preprocessors any img2img node in this repo
# auto-derives internally. Advanced preprocessors (normal maps, soft edges,
# MLSD, lineart variants, OpenPose) live in the separate ComfyUI-ControlNet-
# Nodes package - wire one of those nodes into `reference_image` yourself
# for anything beyond depth/canny, same as Krea2/Qwen-Image already require
# for control types they don't auto-derive.
_CONTROL_MODES = ["manual", "auto_depth", "auto_canny", "none"]

logger = logging.getLogger(__name__)

FLUX_KLEIN_CATEGORY = "\U0001F916 CCTech/Flux Klein"


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def _resize_image(image, width, height, upscale_method="lanczos", crop="center"):
    samples = image[..., :3].clamp(0.0, 1.0).movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _scale_to_megapixels(image, megapixels, resolution_steps=16):
    """Aspect-preserving resize to hit `megapixels` million pixels total,
    rounded to the nearest multiple of resolution_steps. Exact port of
    comfy_extras/nodes_post_processing.py's ImageScaleToTotalPixels math
    (scale_by = sqrt(total / (w*h)), each dim rounded to resolution_steps) -
    confirmed against the real shipped Klein example workflow
    (image_flux2_klein_image_edit_9b_base.json), which chains
    ImageScaleToTotalPixels -> GetImageSize -> EmptyFlux2LatentImage/
    Flux2Scheduler so the canvas ALWAYS matches the reference photo's own
    aspect ratio - never an independent user-typed width/height. Default
    resolution_steps=16 (not comfy's default of 1) to stay aligned with
    Flux.2's real /16 latent downscale, avoiding a fractional-pixel latent
    edge; the shipped workflow itself used resolution_steps=1, so this is a
    deliberate, safer deviation, not an unverified guess."""
    h, w = image.shape[1], image.shape[2]
    total = megapixels * 1024 * 1024
    scale_by = (total / (w * h)) ** 0.5
    new_w = max(resolution_steps, round(w * scale_by / resolution_steps) * resolution_steps)
    new_h = max(resolution_steps, round(h * scale_by / resolution_steps) * resolution_steps)
    return int(new_w), int(new_h)


# ── Nodes ─────────────────────────────────────────────────────────────────

class FluxKleinModelLoader:
    """Load the three FLUX.2 Klein components by name.

    Flux.2/Klein is natively supported by ComfyUI core, so this is a thin
    convenience loader (like Krea2ModelLoader/QwenImageModelLoader) rather
    than anything bespoke - GGUF and safetensors both work for the
    transformer and the text encoder.
    """

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Model Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'load clip', 'klein loader',
                       'flux2 loader', 'checkpoint loader']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load FLUX.2 Klein's UNET, CLIP, and VAE as native comfy "
                   "objects. GGUF quants stay quantized.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "Flux.2 Klein diffusion model - .safetensors or GGUF quant, "
                               "from models/diffusion_models (unet)."}),
                "clip_name": (_clip_filename_list(), {
                    "tooltip": "Flux.2 Klein text encoder (Qwen3) - .safetensors or GGUF "
                               "quant, from models/text_encoders (clip)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Flux.2 VAE, from models/vae."}),
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
            clip, = CLIPLoaderGGUF().load_clip(clip_name, type="flux2")
        else:
            clip = comfy.sd.load_clip(
                ckpt_paths=[folder_paths.get_full_path_or_raise("clip", clip_name)],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.FLUX2)

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (model, clip, vae)


class FluxKleinImg2Img:
    """Prompts, init latent, and reference images for FLUX.2 Klein.

    Klein's real editing mechanism - confirmed against the actual shipped
    example workflows (single-reference "Image Edit (Flux.2 Klein 9B)" and
    the dual-reference "9B-base" variant, image_flux2_klein_image_edit_9b_
    base.json) - ALWAYS starts from a pure-noise EmptyFlux2LatentImage
    (never an img2img partial denoise of an existing photo) and drives the
    edit entirely off VAE-encoded reference images attached to positive
    AND negative conditioning as reference_latents, plus a text
    instruction. This node has no partial-denoise img2img path at all - it
    only builds pure-noise latents, because that's the only mechanism any
    real Klein edit example ever uses; `denoise` is always 1.0. (Generic
    partial-denoise img2img is a real, separate diffusion technique - it's
    just not how Klein edits images, and mixing the two into one node
    input was actively misleading.)

    Two separate inputs cover the two things a reference image can be:

    `images` - one or more RAW reference photos, always attached untouched
    as reference_latents to both positive and negative conditioning. This
    is a single IMAGE socket but batch-aware: if the incoming tensor has N
    images stacked in the batch dimension, each one gets its own VAEEncode
    + reference_latents append, exactly matching the real dual-reference
    subgraph's own per-image ReferenceLatent chain (confirmed by tracing
    its actual node graph, not assumed) - combine multiple photos upstream
    with a stock "Batch Images" node before wiring the result in here.

    `control_source_image` - a photo to turn INTO a controlnet-style map
    before attaching it. Klein has no real ControlNet/Control-LoRA of its
    own; "using it as a controlnet image" here just means running it
    through this pack's own preprocessor and feeding the RESULT through
    the same reference_latents mechanism as `images`, appended after them.
    `control_mode` picks the preprocessor: `manual` attaches
    control_source_image raw (e.g. if you already computed your own map
    elsewhere and just want it attached without reprocessing), `auto_depth`
    runs it through Depth Anything V2 first - reproducing the real example
    workflow's own structural-reference trick (AIO_Preprocessor -> MiDaS
    depth -> reference_latents), the one mode CONFIRMED meaningful to
    Klein's own training - `auto_canny` runs plain cv2 edge detection
    first, mechanically valid but unverified for Klein specifically, and
    `none` skips control_source_image attachment entirely even if it's
    connected, for toggling it off without rewiring. For anything beyond
    depth/canny (normal maps, soft edges, MLSD, lineart variants,
    OpenPose), install the separate ComfyUI-ControlNet-Nodes package and
    wire its output into `control_source_image` yourself with
    control_mode=manual - same pattern Krea2Img2Img/QwenImageImg2Img
    already use for control types they don't auto-derive.

    For 3+ references or advanced reference-conditioning tools (color
    anchoring, per-reference weighting, identity guidance), install the
    separate ComfyUI-Flux-Reference-Tools package - those nodes work on any
    Flux-family model including Klein, not just this one.

    `width`/`height` are the pixel BUDGET (width*height), not necessarily
    the exact output size: whenever `images` or `control_source_image` is
    connected, the canvas is re-derived from that photo's own aspect ratio
    at the same total pixel count (aspect-preserving, `_scale_to_megapixels`,
    matching the real example workflow's own
    ImageScaleToTotalPixels -> GetImageSize -> EmptyFlux2LatentImage chain
    exactly) - `width`/`height` are used as-given only for pure txt2img
    (nothing connected). This was a real bug before it was fixed:
    `width`/`height` used to be trusted as exact independent of any
    connected photo, silently distorting the reference image via a center-
    crop resize to a mismatched aspect ratio, and generating a canvas that
    didn't match it.

    The empty-latent path uses Flux.2's REAL shape - confirmed via
    comfy_extras/nodes_flux.py's EmptyFlux2LatentImage:
    [batch_size, 128, height//16, width//16] - NOT the generic 4-channel/
    8-downscale placeholder used for Krea2/Qwen-Image elsewhere in this
    pack, since comfy.sample.fix_empty_latent_channels only auto-corrects
    channel count, not spatial downscale ratio, unless
    downscale_ratio_spacial is explicitly passed (it isn't, in the plain
    {"samples": latent} dict form used throughout this repo) - Flux.2's
    real /16 downscale would otherwise silently produce a latent twice
    the correct spatial size.

    `denoise` is a fixed 1.0 output, kept only for graph-wiring convenience
    (feed it straight into a stock KSampler's `denoise` input) - it's
    always 1.0 because every real Klein edit example starts from pure
    noise; KSampler's own `denoise` widget already exists for anyone who
    genuinely wants a different value for some other reason.

    Feed the outputs straight into a stock KSampler or FluxKleinKSampler.
    """

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img', 'encode image',
                       'reference image', 'edit reference', 'pose transfer']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent, and reference images for FLUX.2 "
                   "Klein's real edit mechanism (always a pure-noise start). "
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
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16,
                                  "tooltip": "Pixel budget (width*height), not necessarily the exact "
                                             "output size - if images/control_source_image is "
                                             "connected, the canvas is re-derived from that photo's "
                                             "own aspect ratio at this same total pixel count. Used "
                                             "as-given only for pure txt2img."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "One or more RAW reference photos (batch-aware - "
                                               "combine multiple with a stock Batch Images node "
                                               "upstream). Each is independently encoded and "
                                               "attached to positive+negative conditioning as "
                                               "reference_latents - Klein's real edit mechanism."}),
                "control_source_image": ("IMAGE", {"tooltip": "A photo to turn into a controlnet-"
                                               "style map (via control_mode) before attaching it "
                                               "the same way as `images`, appended after them."}),
                "control_mode": (_CONTROL_MODES, {"default": "manual",
                    "tooltip": "manual: attach control_source_image raw. auto_depth: run it "
                               "through Depth Anything V2 first - reproduces the real example "
                               "workflow's structural-reference trick (AIO_Preprocessor -> MiDaS "
                               "depth -> reference_latents), the one mode confirmed meaningful to "
                               "Klein's own training. auto_canny: plain cv2 edge detection first - "
                               "mechanically valid, unverified for Klein specifically. none: "
                               "skip control_source_image attachment entirely even if it's "
                               "connected. For normal/soft-edge/lineart/pose maps, install "
                               "ComfyUI-ControlNet-Nodes and wire its output in with "
                               "control_mode=manual instead."}),
                "depth_ckpt_name": (list(depth_anything_v2.MODEL_CONFIGS.keys()), {
                    "default": "depth_anything_v2_vitb.pth",
                    "tooltip": "auto_depth mode only. Model size for the automatic depth "
                               "estimation. Downloads on first use if not already in "
                               "models/depth_anything_v2."}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, batch_size, width, height,
                images=None, control_source_image=None, control_mode="manual",
                depth_ckpt_name="depth_anything_v2_vitb.pth"):
        # width/height are a pixel BUDGET, not necessarily the exact output
        # size - re-derive the real canvas from whichever photo defines it
        # (the first `images` frame takes priority; else
        # control_source_image), matching the real example workflow's own
        # ImageScaleToTotalPixels -> GetImageSize -> EmptyFlux2LatentImage
        # chain exactly, instead of trusting a user-typed value that may
        # not match the photo's aspect ratio.
        if images is not None:
            size_source = images[0:1]
        elif control_source_image is not None:
            size_source = control_source_image
        else:
            size_source = None
        if size_source is not None:
            megapixels = (width * height) / (1024.0 * 1024.0)
            width, height = _scale_to_megapixels(size_source, megapixels)

        # Klein's real edit mechanism always starts from pure noise - see
        # class docstring for why this can't use the generic /8-downscale
        # placeholder.
        latent = torch.zeros(
            [batch_size, 128, height // 16, width // 16],
            device=comfy.model_management.intermediate_device())
        denoise = 1.0
        logger.info("Flux Klein: empty latent %s (pure noise - Klein's real edit "
                    "mechanism never partially denoises an existing photo)", tuple(latent.shape))

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if images is not None:
            for i in range(images.shape[0]):
                pixels = _resize_image(images[i:i + 1], width, height)
                ref_latent = vae.encode(pixels[:, :, :, :3])
                values = {"reference_latents": [ref_latent]}
                positive = node_helpers.conditioning_set_values(positive, values, append=True)
                negative = node_helpers.conditioning_set_values(negative, values, append=True)
            logger.info("Flux Klein: %d image(s) from `images` attached raw to positive+negative "
                        "conditioning as reference_latents", images.shape[0])

        if control_source_image is not None and control_mode == "none":
            logger.info("Flux Klein: control_mode=none - skipping control_source_image "
                        "attachment even though it's connected.")
        elif control_source_image is not None:
            ctrl_pixels = control_source_image
            if control_mode == "auto_depth":
                logger.info("Flux Klein: auto-deriving depth map from control_source_image "
                            "(control_mode=auto_depth)")
                ctrl_pixels = _depth_anything_batch(control_source_image, depth_ckpt_name)
            elif control_mode == "auto_canny":
                logger.info("Flux Klein: auto-deriving canny edge map from control_source_image "
                            "(control_mode=auto_canny)")
                ctrl_pixels = _auto_canny_control_image(control_source_image)
            ctrl_pixels = _resize_image(ctrl_pixels, width, height)
            ctrl_latent = vae.encode(ctrl_pixels[:, :, :, :3])
            values = {"reference_latents": [ctrl_latent]}
            positive = node_helpers.conditioning_set_values(positive, values, append=True)
            negative = node_helpers.conditioning_set_values(negative, values, append=True)
            logger.info("Flux Klein: control_source_image (%s) attached to positive+negative "
                        "conditioning as reference_latents", control_mode)

        return (model, positive, negative, {"samples": latent}, denoise)


# ── Part D: Flux2KleinIdentityFeatureTransfer (ported from
# identity_feature_transfer.py's IdentityFeatureTransferFinal, MIT,
# capitan01R) ───────────────────────────────────────────────────────────

# Default schedules (tuned for the Klein 9B layout - 8 double blocks, 24
# single blocks - ported verbatim; see class docstring for the caveat on
# other variants like the 4B klein-base).
_HARD_DOUBLE = "0-7:mid_img=0.55"
_HARD_SINGLE = (
    "0:mid_img=0.22; "
    "1:mid_img=0.24; "
    "3:mid_img=0.28; "
    "4:mid_img=0.22; "
    "6:mid_img=0.26; "
    "7:mid_img=0.27; "
    "8:mid_img=0.25; "
    "10:mid_img=0.27; "
    "13:mid_img=0.27"
)


class Flux2KleinIdentityFeatureTransfer:
    """Multi-reference identity-preserving feature transfer for FLUX.2 Klein.

    Ported near-verbatim from ComfyUI-Flux2Klein-Enhancer's
    identity_feature_transfer.py IdentityFeatureTransferFinal (MIT,
    capitan01R) - the pack's own current/best identity-transfer
    implementation (the older IdentityFeatureTransfer/Advanced/V3
    variants are explicitly superseded by this one in the source and are
    not ported here).

    Uses ONLY stock ModelPatcher hooks: model.set_model_attn1_output_patch
    (always) and model.set_model_attn1_patch (only when
    mask_behavior="zero_unmasked_tokens" and a mask is wired). These fire
    generically from comfy's own comfy/ldm/flux/layers.py
    DoubleStreamBlock/SingleStreamBlock.forward - shared Flux/Kontext/
    Klein code, not something requiring Klein-repo-specific plumbing.

    Reads four extra_options keys comfy's own model code already
    populates every forward call - reference_image_num_tokens,
    block_index, block_type, img_slice - zero independent computation of
    Klein internals needed.

    The transfer performs: per-image centering of generated and reference
    features, normalized similarity matching, similarity-floor filtering,
    temperature-controlled reference pooling, and confidence-gated
    transfer at the scheduled double/single blocks.

    KNOWN CAVEAT, ported as-is rather than "fixed" speculatively: the
    default schedules (double_blocks/single_blocks, and the HARD/MID/
    SOFT_LOCK presets) hardcode block counts as magic numbers (8 double /
    24 single) tuned for the Klein 9B layout - they are never read from
    the live model. On a different-sized variant (e.g. the 4B
    klein-base), out-of-range indices clamp harmlessly but presets may
    apply strengths to the wrong semantic blocks - use "custom" and your
    own schedule strings for non-9B checkpoints.

    Requires multi-reference conditioning built with a method that lays
    references out predictably at the tail of the token sequence (e.g.
    Flux2KleinMultiReferenceLatent's reference_latents_method="index") -
    reference_image_num_tokens/img_slice come from comfy's own model code
    at sample time regardless of how references were attached, but the
    similarity-matching math assumes a stable per-reference token range.
    """

    PRESETS = {
        "HARD_LOCK": {
            "double_blocks": _HARD_DOUBLE,
            "single_blocks": _HARD_SINGLE,
            "similarity_floor": 0.040,
            "softmax_temperature": 0.0250,
            "mask_threshold": 1.0,
        },
        "MID_LOCK": {
            "double_blocks": _HARD_DOUBLE,
            "single_blocks": _HARD_SINGLE,
            "similarity_floor": 0.200,
            "softmax_temperature": 0.0700,
            "mask_threshold": 1.0,
        },
        "SOFT_LOCK": {
            "double_blocks": _HARD_DOUBLE,
            "single_blocks": _HARD_SINGLE,
            "similarity_floor": 0.500,
            "softmax_temperature": 0.0700,
            "mask_threshold": 1.0,
        },
    }

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Identity Feature Transfer ⚡"
    SEARCH_ALIASES = ['identity transfer', 'identity feature transfer final', 'face transfer',
                       'identity preserving', 'identity lock', 'reference identity']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = ("Multi-reference identity-preserving feature transfer for "
                   "FLUX.2 Klein - schedules, presets, per-reference masks, "
                   "optional sigma-aware strength scaling.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (["HARD_LOCK", "MID_LOCK", "SOFT_LOCK", "custom"], {"default": "HARD_LOCK"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False,
                    "tooltip": "Zero-based references used by the transfer. 'all', "
                               "comma-separated indices like '0,2,3', or ranges like '0-3'."}),
                "similarity_floor": ("FLOAT", {"default": 0.040, "min": 0.0, "max": 0.95, "step": 0.001}),
                "softmax_temperature": ("FLOAT", {"default": 0.0250, "min": 0.0001, "max": 0.25, "step": 0.0001}),
                "mask_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks": ("STRING", {"default": _HARD_DOUBLE, "multiline": False}),
                "single_blocks": ("STRING", {"default": _HARD_SINGLE, "multiline": False}),
                "debug": ("BOOLEAN", {"default": False}),
                "mask_behavior": (["focus_only", "zero_unmasked_tokens"], {
                    "default": "focus_only",
                    "tooltip": "focus_only: the mask limits this node's reference bank "
                               "while Klein still sees the complete reference. "
                               "zero_unmasked_tokens: also blocks each wired reference's "
                               "unmasked tokens as attention sources in every block. "
                               "References without a wired mask remain complete either way."}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"forceInput": True,
                    "tooltip": "Optional sampler sigma schedule. When connected, block "
                               "strengths decay per sampling step by "
                               "delta_sigma_0 / delta_sigma_step."}),
                "subject_mask_1": ("MASK",),
                "subject_mask_2": ("MASK",),
                "subject_mask_3": ("MASK",),
                "subject_mask_4": ("MASK",),
                "subject_mask_5": ("MASK",),
                "subject_mask_6": ("MASK",),
                "subject_mask_7": ("MASK",),
                "subject_mask_8": ("MASK",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    @staticmethod
    def _parse_ref_indices(text: str, count: int) -> List[int]:
        if count <= 0:
            return []
        value = str(text or "all").strip().lower()
        if value in ("", "all", "*"):
            return list(range(count))
        out = set()
        for part in re.split(r"[;, ]+", value):
            if not part:
                continue
            try:
                if "-" in part:
                    a, b = part.split("-", 1)
                    lo, hi = int(a), int(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    for i in range(lo, hi + 1):
                        if 0 <= i < count:
                            out.add(i)
                else:
                    i = int(part)
                    if 0 <= i < count:
                        out.add(i)
            except ValueError:
                continue
        return sorted(out)

    @staticmethod
    def _parse_schedule(text: str, max_block: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for row in str(text or "").split(";"):
            row = row.strip()
            if not row or ":" not in row:
                continue
            block_part, value_part = row.split(":", 1)
            value_part = value_part.strip()
            if "=" in value_part:
                key, value_part = value_part.split("=", 1)
                if key.strip().lower() not in ("mid", "mid_img"):
                    continue
            try:
                strength = float(value_part.strip())
            except ValueError:
                continue
            try:
                if "-" in block_part:
                    lo_s, hi_s = block_part.split("-", 1)
                    lo, hi = int(lo_s.strip()), int(hi_s.strip())
                else:
                    lo = hi = int(block_part.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(max_block, hi)
            for idx in range(lo, hi + 1):
                out[idx] = strength
        return out

    @staticmethod
    def _prep_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None or not torch.is_tensor(mask):
            return None
        x = mask.detach().float().cpu()
        if x.dim() == 4:
            if x.shape[-1] in (1, 3, 4):
                x = x[0].mean(dim=-1)
            else:
                x = x[0, 0]
        elif x.dim() == 3:
            if x.shape[-1] in (1, 3, 4) and x.shape[0] != 1:
                x = x.mean(dim=-1)
            else:
                x = x[0]
        elif x.dim() != 2:
            return None
        return x.contiguous()

    @staticmethod
    def _grid_for_tokens(count: int, mask: torch.Tensor) -> Tuple[int, int]:
        count = max(1, int(count))
        mh, mw = mask.shape[-2:]
        target = mh / max(mw, 1)
        best = (1, count)
        best_err = float("inf")
        for h in range(1, int(count ** 0.5) + 3):
            if count % h != 0:
                continue
            w = count // h
            for hh, ww in ((h, w), (w, h)):
                err = abs((hh / max(ww, 1)) - target)
                if err < best_err:
                    best = (hh, ww)
                    best_err = err
        return best

    @staticmethod
    def _scalar_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return None
                return float(value.detach().flatten()[0].double().cpu().item())
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _sigma_equal_energy_schedule(sigmas):
        if sigmas is None:
            return None
        if not torch.is_tensor(sigmas):
            raise ValueError("Flux2KleinIdentityFeatureTransfer: sigmas must be a SIGMAS tensor.")

        values = sigmas.detach().flatten().double().cpu()
        if values.numel() < 2:
            raise ValueError("Flux2KleinIdentityFeatureTransfer: at least two sigma values are required.")
        if not torch.isfinite(values).all().item():
            raise ValueError("Flux2KleinIdentityFeatureTransfer: sigma schedule contains a non-finite value.")

        deltas = (values[:-1] - values[1:]).abs()
        if deltas[0].item() <= 0.0:
            raise ValueError("Flux2KleinIdentityFeatureTransfer: Step 0 sigma interval must be greater than zero.")
        zero_steps = torch.nonzero(deltas <= 0.0, as_tuple=False).flatten()
        if zero_steps.numel() > 0:
            step = int(zero_steps[0].item())
            raise ValueError(
                f"Flux2KleinIdentityFeatureTransfer: sigma interval for step {step} is zero."
            )

        ratios = deltas[0] / deltas
        return values, ratios

    @staticmethod
    def _sigma_step_index(values: torch.Tensor, current_sigma: float) -> int:
        step_values = values[:-1]
        differences = torch.abs(step_values - float(current_sigma))
        tolerance = max(1e-7, abs(float(current_sigma)) * 1e-6)
        exact = torch.nonzero(differences <= tolerance, as_tuple=False).flatten()
        if exact.numel() > 0:
            return int(exact[0].item())

        for step_idx in range(values.numel() - 1):
            start = float(values[step_idx].item())
            end = float(values[step_idx + 1].item())
            lo, hi = min(start, end), max(start, end)
            if lo <= float(current_sigma) <= hi:
                return step_idx

        return int(torch.argmin(differences).item())

    def apply(
        self,
        model,
        preset="HARD_LOCK",
        enabled=True,
        reference_index=0,
        reference_indices="all",
        similarity_floor=0.040,
        softmax_temperature=0.0250,
        mask_threshold=1.0,
        double_blocks=_HARD_DOUBLE,
        single_blocks=_HARD_SINGLE,
        debug=False,
        mask_behavior="focus_only",
        subject_mask_1=None,
        subject_mask_2=None,
        subject_mask_3=None,
        subject_mask_4=None,
        subject_mask_5=None,
        subject_mask_6=None,
        subject_mask_7=None,
        subject_mask_8=None,
        sigmas=None,
    ):
        m = model.clone()
        if not bool(enabled):
            return (m,)

        if preset in self.PRESETS:
            cfg = self.PRESETS[preset]
            double_blocks = cfg["double_blocks"]
            single_blocks = cfg["single_blocks"]
            similarity_floor = cfg["similarity_floor"]
            softmax_temperature = cfg["softmax_temperature"]
            mask_threshold = cfg["mask_threshold"]

        double_map = self._parse_schedule(double_blocks, 7)
        single_map = self._parse_schedule(single_blocks, 23)
        sim_floor = float(max(0.0, min(0.95, similarity_floor)))
        temperature = float(max(1e-6, softmax_temperature))
        mask_threshold = float(max(0.0, min(1.0, mask_threshold)))
        ref_idx = int(reference_index)
        ref_indices_text = str(reference_indices)
        mask_behavior = str(mask_behavior)
        if mask_behavior not in ("focus_only", "zero_unmasked_tokens"):
            mask_behavior = "focus_only"
        sigma_schedule = self._sigma_equal_energy_schedule(sigmas)
        sigma_debug_seen = set()
        sigma_missing_warned = False

        def sigma_strength_multiplier(extra_options):
            nonlocal sigma_missing_warned
            if sigma_schedule is None:
                return 1.0, None, None

            current_sigma = self._scalar_float(extra_options.get("sigmas"))
            if current_sigma is None:
                if debug and not sigma_missing_warned:
                    sigma_missing_warned = True
                    logger.info(
                        "Flux2KleinIdentityFeatureTransfer: SIGMAS connected, but "
                        "the current model sigma is unavailable; using unscaled strengths.")
                return 1.0, None, None

            values, ratios = sigma_schedule
            step_idx = self._sigma_step_index(values, current_sigma)
            return float(ratios[step_idx].item()), step_idx, current_sigma

        masks = [
            self._prep_mask(subject_mask_1),
            self._prep_mask(subject_mask_2),
            self._prep_mask(subject_mask_3),
            self._prep_mask(subject_mask_4),
            self._prep_mask(subject_mask_5),
            self._prep_mask(subject_mask_6),
            self._prep_mask(subject_mask_7),
            self._prep_mask(subject_mask_8),
        ]
        mask_cache: Dict[Tuple[int, int, float], Optional[torch.Tensor]] = {}

        def mask_indices(ref_id: int, count: int, device):
            if ref_id < 0 or ref_id >= len(masks):
                return None
            src = masks[ref_id]
            if src is None:
                return None
            key = (ref_id, int(count), mask_threshold)
            if key in mask_cache:
                cached = mask_cache[key]
                return cached.to(device) if cached is not None else None
            grid = self._grid_for_tokens(int(count), src)
            pooled = F.adaptive_avg_pool2d(src[None, None], grid).view(-1)
            keep = pooled >= mask_threshold
            if keep.sum().item() == 0:
                mask_cache[key] = torch.empty((0,), dtype=torch.long)
            else:
                mask_cache[key] = torch.nonzero(keep, as_tuple=False).squeeze(-1).to(torch.long).cpu()
            return mask_cache[key].to(device)

        def selected_ref_slices(ref_tokens: Sequence[int], base: int):
            selected = self._parse_ref_indices(ref_indices_text, len(ref_tokens))
            if not selected:
                selected = [min(max(ref_idx, 0), len(ref_tokens) - 1)]
            selected_set = set(selected)
            slices = []
            offset = 0
            for i, count in enumerate(ref_tokens):
                start = base + offset
                end = start + int(count)
                if i in selected_set and end > start:
                    slices.append((i, start, end))
                offset += int(count)
            return slices

        def reference_bank(tokens: torch.Tensor, slices):
            parts = []
            for ref_id, start, end in slices:
                ref = tokens[:, start:end]
                idx = mask_indices(ref_id, end - start, tokens.device)
                if idx is not None:
                    if idx.numel() == 0:
                        continue
                    ref = ref.index_select(1, idx.to(torch.long))
                if ref.shape[1] > 0:
                    parts.append(ref)
            if not parts:
                return None
            return torch.cat(parts, dim=1)

        def reference_source_mask_patch(q, k, v, pe, attn_mask, extra_options):
            ref_tokens = extra_options.get("reference_image_num_tokens", []) or []
            img_slice = extra_options.get("img_slice")
            if not ref_tokens or img_slice is None:
                return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

            total_seq = int(k.shape[2])
            total_ref = int(sum(ref_tokens))
            slices = selected_ref_slices(ref_tokens, total_seq - total_ref)
            allow = torch.ones((k.shape[0], total_seq), dtype=torch.bool, device=k.device)
            changed = False
            for ref_id, start, end in slices:
                idx = mask_indices(ref_id, end - start, k.device)
                if idx is None:
                    continue
                allow[:, start:end] = False
                if idx.numel() > 0:
                    allow[:, start + idx.to(torch.long)] = True
                changed = True

            if not changed:
                return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

            key_allow = allow[:, None, None, :]
            if attn_mask is None:
                combined_mask = key_allow
            elif attn_mask.dtype == torch.bool:
                existing = attn_mask
                if existing.ndim == 2:
                    if existing.shape[0] == k.shape[0] and existing.shape[1] == total_seq:
                        existing = existing[:, None, None, :]
                    else:
                        existing = existing[None, None, :, :]
                elif existing.ndim == 3:
                    existing = existing[:, None, :, :]
                combined_mask = existing & key_allow
            else:
                existing = attn_mask
                if existing.ndim == 2:
                    if existing.shape[0] == k.shape[0] and existing.shape[1] == total_seq:
                        existing = existing[:, None, None, :]
                    else:
                        existing = existing[None, None, :, :]
                elif existing.ndim == 3:
                    existing = existing[:, None, :, :]
                source_bias = torch.zeros(
                    (k.shape[0], 1, 1, total_seq),
                    dtype=existing.dtype,
                    device=k.device,
                )
                source_bias.masked_fill_(~key_allow, torch.finfo(existing.dtype).min)
                combined_mask = existing + source_bias

            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": combined_mask}

        def pull_delta(gen: torch.Tensor, ref: torch.Tensor, strength: float):
            if strength <= 0.0 or ref is None or ref.shape[1] <= 0:
                return None
            gen_f = gen.float()
            ref_f = ref.float()
            gen_norm = F.normalize(gen_f - gen_f.mean(dim=1, keepdim=True), dim=-1)
            ref_norm = F.normalize(ref_f - ref_f.mean(dim=1, keepdim=True), dim=-1)
            sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
            neg = torch.finfo(sim.dtype).min
            sim = torch.where(sim >= sim_floor, sim, torch.full_like(sim, neg))
            weights = torch.softmax(sim / temperature, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            pooled = torch.bmm(weights, ref_f)
            best = sim.max(dim=-1).values
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
            confidence = ((best - sim_floor) / max(1.0 - sim_floor, 1e-6)).clamp(0.0, 1.0)
            weight = (confidence * float(strength)).unsqueeze(-1).to(gen.dtype)
            return (pooled.to(gen.dtype) - gen) * weight

        def output_patch(attn, extra_options):
            ref_tokens = extra_options.get("reference_image_num_tokens", []) or []
            img_slice = extra_options.get("img_slice")
            if not ref_tokens or img_slice is None:
                return attn
            block_type = extra_options.get("block_type", "double")
            block_idx = int(extra_options.get("block_index", 0))
            if block_type == "double":
                strength = double_map.get(block_idx, 0.0)
            elif block_type == "single":
                strength = single_map.get(block_idx, 0.0)
            else:
                return attn
            sigma_multiplier, sigma_step, current_sigma = sigma_strength_multiplier(extra_options)
            strength *= sigma_multiplier
            if strength == 0.0:
                return attn
            if debug and sigma_step is not None and sigma_step not in sigma_debug_seen:
                sigma_debug_seen.add(sigma_step)
                logger.info(
                    "Flux2KleinIdentityFeatureTransfer sigma: step=%d sigma=%.7f strength_multiplier=%.7f",
                    sigma_step, current_sigma, sigma_multiplier)
            txt_end, total_seq = int(img_slice[0]), int(img_slice[1])
            total_ref = int(sum(ref_tokens))
            gen_start = txt_end
            gen_end = total_seq - total_ref
            if gen_end <= gen_start:
                return attn
            ref = reference_bank(attn, selected_ref_slices(ref_tokens, total_seq - total_ref))
            if ref is None:
                return attn
            gen = attn[:, gen_start:gen_end]
            delta = pull_delta(gen, ref, strength)
            if delta is None:
                return attn
            out = attn.clone()
            out[:, gen_start:gen_end] = gen + delta
            return out

        if bool(debug):
            logger.info(
                "Flux2KleinIdentityFeatureTransfer: preset=%s sim=%.4f temp=%.4f "
                "mask_behavior=%s mask=%.2f double=%s single=%s sigma_aware=%s",
                preset, sim_floor, temperature, mask_behavior, mask_threshold,
                double_map, single_map, sigma_schedule is not None)

        m.set_model_attn1_output_patch(output_patch)
        if mask_behavior == "zero_unmasked_tokens" and any(mask is not None for mask in masks):
            m.set_model_attn1_patch(reference_source_mask_patch)
        return (m,)


# ── Part E: remaining simple Klein nodes (ported from
# ComfyUI-Flux2Klein-Enhancer, MIT, capitan01R) ─────────────────────────

def _detect_active_end(meta: dict, seq_len: int, override: int) -> int:
    """Active-region detection: honest attention_mask read, falls back to
    the FULL sequence length (not a hardcoded 77) when no mask is present."""
    if override > 0:
        return min(override, seq_len)
    attn_mask = meta.get("attention_mask", None)
    if attn_mask is not None and attn_mask.dim() >= 2:
        nonzero = attn_mask[0].nonzero()
        if len(nonzero) > 0:
            return int(nonzero[-1].item()) + 1
    return seq_len


def _klein_layer_slice_size(embed_dim: int) -> int:
    """Klein conditioning stacks 3 Qwen3 hidden-layer slices along the embed
    dim (12288 = 3*4096 for the 8B text encoder, 7680 = 3*2560 for 4B).
    Returns the per-layer slice width, or embed_dim unchanged (disabling
    the per-layer ops) for an unrecognized architecture."""
    if embed_dim % 3 == 0:
        return embed_dim // 3
    return embed_dim


class Flux2KleinEnhancer:
    """Scalar/whitening operations on the active-token region of Klein
    conditioning, plus a Klein-specific per-Qwen3-layer scale (Klein
    conditioning stacks 3 hidden-layer slices along the embed dim)."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Enhancer ⚡"
    SEARCH_ALIASES = ['conditioning enhancer', 'prompt strength', 'contrast', 'whiten']
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    DESCRIPTION = "Scalar/whitening/per-layer scaling on Klein's active-token conditioning region."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "active_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "per_token_whiten": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "norm_equalize": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "early_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "mid_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "late_layer_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "active_end_override": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def enhance(self, conditioning, active_scale=1.0, per_token_whiten=0.0,
                norm_equalize=0.0, early_layer_scale=1.0, mid_layer_scale=1.0,
                late_layer_scale=1.0, preserve_original=0.0,
                active_end_override=0, debug=False):
        if not conditioning:
            return (conditioning,)

        no_op = (active_scale == 1.0 and per_token_whiten == 0.0 and norm_equalize == 0.0
                  and early_layer_scale == 1.0 and mid_layer_scale == 1.0
                  and late_layer_scale == 1.0 and preserve_original == 0.0)
        if no_op:
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.float()
            if cond.dim() != 3:
                output.append((cond_tensor, meta))
                continue

            seq_len, embed_dim = cond.shape[1], cond.shape[2]
            active_end = _detect_active_end(meta, seq_len, active_end_override)
            slice_w = _klein_layer_slice_size(embed_dim)

            active = cond[:, :active_end, :].clone()
            original_active = active.clone()

            if per_token_whiten != 0.0:
                seq_mean = active.mean(dim=1, keepdim=True)
                active = seq_mean + (active - seq_mean) * (1.0 + per_token_whiten)

            if norm_equalize > 0.0:
                token_norms = active.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                target_norm = token_norms.mean()
                normalized = active / token_norms * target_norm
                active = active * (1.0 - norm_equalize) + normalized * norm_equalize

            if active_scale != 1.0:
                active = active * active_scale

            if slice_w * 3 == embed_dim and (
                early_layer_scale != 1.0 or mid_layer_scale != 1.0 or late_layer_scale != 1.0
            ):
                if early_layer_scale != 1.0:
                    active[:, :, :slice_w] = active[:, :, :slice_w] * early_layer_scale
                if mid_layer_scale != 1.0:
                    active[:, :, slice_w:2 * slice_w] = active[:, :, slice_w:2 * slice_w] * mid_layer_scale
                if late_layer_scale != 1.0:
                    active[:, :, 2 * slice_w:] = active[:, :, 2 * slice_w:] * late_layer_scale

            if preserve_original > 0.0:
                active = active * (1.0 - preserve_original) + original_active * preserve_original

            result = cond.clone()
            result[:, :active_end, :] = active
            if debug:
                diff = (result - cond).abs()
                logger.info("Flux2KleinEnhancer: item %d diff mean=%.6f max=%.6f",
                            idx, diff.mean().item(), diff.max().item())
            output.append((result.to(original_dtype), meta))

        gc.collect()
        return (output,)


_KLEIN_CHAT_TEMPLATE = (
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
_KLEIN_SECTION_SEPARATORS = {"comma": ", ", "period": ". ", "space": " ", "newline": "\n"}


def _klein_hf_tokenizer(clip):
    """Return the underlying HF tokenizer (Qwen3-8B or Qwen3-4B Klein
    variant) - confirmed against a real loaded Klein CLIP: clip.tokenizer
    is a KleinTokenizer exposing .qwen3_4b (present) / .qwen3_8b (present
    only on the 8B-text-encoder variant), each with a real HF .tokenizer."""
    tok = getattr(clip, "tokenizer", None)
    if tok is None:
        return None
    for attr in ("qwen3_8b", "qwen3_4b"):
        sub = getattr(tok, attr, None)
        if sub is not None and hasattr(sub, "tokenizer"):
            return sub.tokenizer
    return None


def _klein_count_tokens(hf_tok, text: str) -> int:
    if not text:
        return 0
    out = hf_tok(text, add_special_tokens=False, return_tensors=None)
    return len(out["input_ids"])


def _klein_wrapper_lengths(hf_tok) -> Tuple[int, int]:
    prefix, suffix = _KLEIN_CHAT_TEMPLATE.split("{}")
    return _klein_count_tokens(hf_tok, prefix), _klein_count_tokens(hf_tok, suffix)


def _klein_section_ranges(hf_tok, sections: Dict[str, str], separator: str) -> Optional[Dict[str, Tuple[int, int]]]:
    if hf_tok is None:
        return None
    prefix_len, _ = _klein_wrapper_lengths(hf_tok)
    sep_len = _klein_count_tokens(hf_tok, separator) if separator else 0
    front_n = _klein_count_tokens(hf_tok, sections.get("front", ""))
    mid_n = _klein_count_tokens(hf_tok, sections.get("mid", ""))
    end_n = _klein_count_tokens(hf_tok, sections.get("end", ""))

    pos = prefix_len
    ranges: Dict[str, Tuple[int, int]] = {}
    ranges["front"] = (pos, pos + front_n)
    pos += front_n
    if front_n > 0 and mid_n > 0:
        pos += sep_len
    ranges["mid"] = (pos, pos + mid_n)
    pos += mid_n
    if (mid_n > 0 and end_n > 0) or (front_n > 0 and mid_n == 0 and end_n > 0):
        pos += sep_len
    ranges["end"] = (pos, pos + end_n)
    return ranges


def _klein_parse_marker_sections(text: str) -> Optional[Dict[str, str]]:
    if not text:
        return None
    pattern = r"\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    sections = {"front": "", "mid": "", "end": ""}
    for name, content in matches:
        sections[name.lower()] = content.strip()
    return sections


class Flux2KleinSectionedEncoder:
    """Tokenize a 3-section prompt (front/mid/end) and emit per-section
    token ranges as conditioning metadata (meta["klein_sections"]) that
    Flux Klein Detail Controller reads for real (not fixed-25/50/25)
    section boundaries. The only node in this port with a real
    Klein-CLIP-internals dependency - confirmed against a real loaded
    Klein CLIP object that clip.tokenizer.qwen3_4b.tokenizer (or
    .qwen3_8b.tokenizer on the 8B-text-encoder variant) is a genuine HF
    tokenizer, not something this pack needs to reimplement."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Sectioned Encoder ⚡"
    SEARCH_ALIASES = ['sectioned encoder', 'section prompt', 'front mid end']
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "front_section", "mid_section", "end_section", "full_prompt")
    FUNCTION = "encode_sectioned"
    OUTPUT_NODE = True
    DESCRIPTION = "Tokenizes a front/mid/end sectioned prompt and stamps real per-section token ranges onto conditioning."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"clip": ("CLIP",)},
            "optional": {
                "front_text": ("STRING", {"multiline": True, "default": ""}),
                "mid_text": ("STRING", {"multiline": True, "default": ""}),
                "end_text": ("STRING", {"multiline": True, "default": ""}),
                "combined_prompt": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional single prompt with [FRONT]/[MID]/[END] markers - "
                               "overrides the three text boxes when non-empty and contains markers."}),
                "separator": (list(_KLEIN_SECTION_SEPARATORS.keys()), {"default": "comma"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def encode_sectioned(self, clip, front_text="", mid_text="", end_text="",
                         combined_prompt="", separator="comma", debug=False):
        marker_sections = _klein_parse_marker_sections(combined_prompt)
        sections = marker_sections or {"front": front_text or "", "mid": mid_text or "", "end": end_text or ""}
        sep_str = _KLEIN_SECTION_SEPARATORS.get(separator, ", ")
        parts = [sections[k] for k in ("front", "mid", "end") if sections[k]]
        full_prompt = sep_str.join(parts)

        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

        hf_tok = _klein_hf_tokenizer(clip)
        ranges = _klein_section_ranges(hf_tok, sections, sep_str)
        meta: Dict = {"pooled_output": pooled}
        if ranges is not None:
            meta["klein_sections"] = ranges
            if debug:
                logger.info("Flux2KleinSectionedEncoder: klein_sections=%s", ranges)
        else:
            logger.warning("Flux2KleinSectionedEncoder: HF tokenizer not accessible on "
                           "CLIP - no klein_sections metadata emitted; Detail Controller "
                           "will fall back to fixed 25/50/25 slicing.")

        return ([[cond, meta]], sections["front"], sections["mid"], sections["end"], full_prompt)



# NOTE on scope: this file used to also register 9 architecturally-generic
# Flux-family reference-conditioning nodes (Flux2KleinMultiReferenceLatent,
# Flux2KleinColorAnchor, Flux2KleinDetailController, Flux2KleinTextEnhancer,
# Flux2KleinMaskRefController, Flux2KleinRefLatentController,
# Flux2KleinTextRefBalance, Flux2KleinRefLatentWeight,
# Flux2KleinIdentityGuidance) - none of them actually read anything
# Klein-specific (no Klein tokenizer internals, no Klein-only conditioning
# shape assumptions), they just operate on reference_latents/attn1_patch/
# sampler_post_cfg_function mechanisms comfy's own Flux-family code shares
# across Flux.1/Kontext/Klein. They were extracted, renamed (dropping the
# "Klein" branding), and generalized into the standalone
# ComfyUI-Flux-Reference-Tools package - install that separately for this
# functionality on Klein or any other Flux-family model. This IS a breaking
# change for any saved workflow using those 9 old node-type names directly
# (no in-repo alias is possible for a node that now lives in a different
# package) - see AGENTS.md for the full reasoning.

NODE_CLASS_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader,
    "FluxKleinImg2Img": FluxKleinImg2Img,
    # Backward-compat alias: this node's logic moved to the shared
    # nodes/preprocessors.py DepthMap class - old saved workflows using the
    # "Flux2KleinDepthMap" type id keep resolving to a working node.
    "Flux2KleinDepthMap": DepthMap,
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer,
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader.TITLE,
    "FluxKleinImg2Img": FluxKleinImg2Img.TITLE,
    "Flux2KleinDepthMap": "Flux Klein Depth Map ⚡",
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer.TITLE,
    "Flux2KleinEnhancer": Flux2KleinEnhancer.TITLE,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder.TITLE,
}
