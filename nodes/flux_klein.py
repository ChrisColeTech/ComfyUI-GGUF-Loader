"""FLUX.2 Klein nodes: component loader, one-node img2img prep, and a full
port of ComfyUI-Flux2Klein-Enhancer's multi-reference/identity-transfer/
conditioning tools.

  FluxKleinModelLoader        unet / clip / vae by name -> MODEL, CLIP, VAE
  FluxKleinImg2Img            model, clip, vae, prompts (+ optional init
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
from .preprocessors import DepthMap, _depth_anything_batch

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
    """Prompts, init latent, and one reference image for FLUX.2 Klein.

    Leave image unconnected for txt2img. `reference_image` is Klein's real
    editing mechanism - confirmed against the actual shipped example
    workflow ("Image Edit (Flux.2 Klein 9B Distilled)"): it starts from a
    pure-noise EmptyFlux2LatentImage (NOT an img2img partial denoise of the
    edited photo) and drives the edit entirely off two VAE-encoded
    reference images attached to positive AND negative conditioning as
    reference_latents, plus a text instruction (e.g. "change the pose of
    the subject in image2 to the pose in image1"). One of those two
    references in the example is the RAW photo; the other is that photo's
    DEPTH MAP (via the source workflow's AIO_Preprocessor set to
    MiDaS-DepthMapPreprocessor) - control_mode picks which `reference_image`
    is: `manual` attaches it raw, `auto_depth` runs it through this pack's
    already-ported Depth Anything V2 first (same detector as Flux2 Klein
    Depth Map / Krea2Img2Img's auto_depth) to reproduce that exact
    structural-reference trick.

    `image` (img2img partial-denoise starting point) and `reference_image`
    (conditioning-only reference) are independent and answer different
    questions - what to start denoising from vs. what identity/structure to
    reference - matching this pack's edit_reference convention on Krea2Img2Img/
    QwenImageImg2Img. For 3+ references, use Flux2KleinMultiReferenceLatent
    downstream instead (it OVERWRITES reference_latents, so re-supply this
    node's reference_image there too rather than mixing both mechanisms).

    The empty-latent (txt2img) path uses Flux.2's REAL shape - confirmed
    via comfy_extras/nodes_flux.py's EmptyFlux2LatentImage:
    [batch_size, 128, height//16, width//16] - NOT the generic 4-channel/
    8-downscale placeholder used for Krea2/Qwen-Image elsewhere in this
    pack, since comfy.sample.fix_empty_latent_channels only auto-corrects
    channel count, not spatial downscale ratio, unless
    downscale_ratio_spacial is explicitly passed (it isn't, in the plain
    {"samples": latent} dict form used throughout this repo) - Flux.2's
    real /16 downscale would otherwise silently produce a latent twice
    the correct spatial size.

    Feed the outputs straight into a stock KSampler or FluxKleinKSampler.
    """

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img', 'encode image',
                       'reference image', 'edit reference', 'pose transfer']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent, and one reference image for FLUX.2 "
                   "Klein. Leave image unconnected for txt2img. Feed the "
                   "outputs straight into a stock KSampler.")

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
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Init image for img2img. Leave unconnected "
                                               "for txt2img."}),
                "reference_image": ("IMAGE", {"tooltip": "Klein's real edit mechanism: encoded "
                                               "and attached to positive+negative conditioning "
                                               "as reference_latents. Independent of `image` - "
                                               "for a pure reference-driven edit, leave `image` "
                                               "unconnected (txt2img latent) and connect only "
                                               "this."}),
                "control_mode": (["manual", "auto_depth"], {"default": "manual",
                    "tooltip": "manual: attach reference_image raw. auto_depth: run it through "
                               "Depth Anything V2 first and attach the depth map instead - "
                               "reproduces the real example workflow's structural-reference "
                               "trick (AIO_Preprocessor -> MiDaS depth -> reference_latents)."}),
                "depth_ckpt_name": (list(depth_anything_v2.MODEL_CONFIGS.keys()), {
                    "default": "depth_anything_v2_vitb.pth",
                    "tooltip": "auto_depth mode only. Model size for the automatic depth "
                               "estimation. Downloads on first use if not already in "
                               "models/depth_anything_v2."}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, image=None, reference_image=None, control_mode="manual",
                depth_ckpt_name="depth_anything_v2_vitb.pth"):
        if image is None:
            # Flux.2's real empty-latent shape - see class docstring for why
            # this can't use the generic /8-downscale placeholder.
            latent = torch.zeros(
                [batch_size, 128, height // 16, width // 16],
                device=comfy.model_management.intermediate_device())
            denoise = 1.0
            logger.info("Flux Klein: txt2img, empty latent %s", tuple(latent.shape))
        else:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Flux Klein: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        if reference_image is not None:
            ref_pixels = reference_image
            if control_mode == "auto_depth":
                logger.info("Flux Klein: auto-deriving depth map from reference_image "
                            "(control_mode=auto_depth)")
                ref_pixels = _depth_anything_batch(reference_image, depth_ckpt_name)
            ref_pixels = _resize_image(ref_pixels, width, height)
            ref_latent = vae.encode(ref_pixels[:, :, :, :3])
            values = {"reference_latents": [ref_latent]}
            positive = node_helpers.conditioning_set_values(positive, values, append=True)
            negative = node_helpers.conditioning_set_values(negative, values, append=True)
            logger.info("Flux Klein: reference_image (%s) attached to positive+negative "
                        "conditioning as reference_latents", control_mode)

        return (model, positive, negative, {"samples": latent}, denoise)


class Flux2KleinMultiReferenceLatent:
    """Place up to 8 encoded reference latents into conditioning at once,
    using Klein's indexed reference method.

    Ported from ComfyUI-Flux2Klein-Enhancer's multi_reference_latent.py
    (68 lines, MIT). One required + up to 7 optional LATENT inputs. Each
    connected latent's batch is split into individual references (a
    batch of 3 becomes 3 separate references, not one 3-image reference),
    then meta["reference_latents"] is OVERWRITTEN (not appended - unlike
    stock ReferenceLatent's chaining pattern) with the full list, and
    meta["reference_latents_method"] = "index" is set.

    Confirmed against comfy's real Flux._forward (comfy/ldm/flux/model.py)
    that this method string is genuinely read and branched on: "index"
    lays references out via simple sequential RoPE-index offsets (as
    opposed to "uxo"'s spatial tiling, or the default auto-packing mode) -
    this predictable layout is what the token-slice math in
    IdentityFeatureTransferFinal and the other reference-control nodes
    depends on.

    Apply to both positive AND negative conditioning (matches the real
    Klein Controlnet.json example workflow's own "Reference Conditioning"
    subgraph, which chains this onto both).
    """

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Multi ReferenceLatent ⚡"
    SEARCH_ALIASES = ['reference latent', 'multi reference', 'identity reference',
                       'reference image', 'klein reference']
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    DESCRIPTION = ("Attach up to 8 reference latents to conditioning using "
                   "Klein's indexed reference method.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_1": ("LATENT",),
            },
            "optional": {
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
                "latent_6": ("LATENT",),
                "latent_7": ("LATENT",),
                "latent_8": ("LATENT",),
            },
        }

    @staticmethod
    def _samples(latent):
        if latent is None:
            return None
        return latent["samples"] if isinstance(latent, dict) else latent

    def apply(self, positive, negative, latent_1, latent_2=None, latent_3=None,
              latent_4=None, latent_5=None, latent_6=None, latent_7=None, latent_8=None):
        refs = []
        for latent in (latent_1, latent_2, latent_3, latent_4, latent_5, latent_6, latent_7, latent_8):
            samples = self._samples(latent)
            if samples is None:
                continue
            for b in range(samples.shape[0]):
                refs.append(samples[b:b + 1].detach())

        values = {"reference_latents": refs, "reference_latents_method": "index"}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        logger.info("Flux Klein: %d reference latent(s) attached (method=index)", len(refs))
        return (positive, negative)


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


class Flux2KleinColorAnchor:
    """Nudge each denoising step's x0 prediction toward the reference
    latent's per-channel spatial-mean color, ramping in over the course of
    sampling. Registers via model_options["sampler_post_cfg_function"],
    which only fires through comfy's own CFGGuider/sampling_function - a
    stock KSampler (or anything using comfy's normal sampling pipeline) is
    required for this to have any effect; it does nothing paired with a
    hand-rolled sampling loop that bypasses CFGGuider."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Color Anchor ⚡"
    SEARCH_ALIASES = ['color anchor', 'color correction', 'color drift', 'color match']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = ("Pulls each sampling step's color back toward a reference "
                   "latent's channel means, ramping in over the run. Requires "
                   "sampling through stock KSampler.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "ramp_curve": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1}),
                "ref_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "channel_weights": (["uniform", "by_variance"], {"default": "uniform"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def apply(self, model, conditioning, strength=0.5, ramp_curve=1.5,
              ref_index=0, channel_weights="uniform", debug=False):
        if strength == 0.0:
            return (model,)

        ref_means = None
        ch_trust = None
        for _, meta in conditioning:
            rl = meta.get("reference_latents", None)
            if rl is not None and ref_index < len(rl):
                ref = rl[ref_index].float()
                ref_means = ref.mean(dim=(-2, -1), keepdim=True)
                if channel_weights == "by_variance":
                    spatial_var = ref.var(dim=(-2, -1), keepdim=True)
                    ch_trust = 1.0 / (1.0 + spatial_var)
                    ch_trust = ch_trust / ch_trust.max().clamp(min=1e-8)
                break

        if ref_means is None:
            logger.info("Flux2KleinColorAnchor: no reference latent found in "
                        "conditioning - node inactive.")
            return (model,)

        _ref_means, _ch_trust = ref_means, ch_trust
        _strength = strength
        _curve = max(ramp_curve, 1e-3)
        _state = {"sigma_max": None, "step": 0}

        def _color_anchor_fn(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            try:
                s = sigma.max().item()
            except (AttributeError, TypeError):
                s = float(sigma)

            if _state["sigma_max"] is None or s > _state["sigma_max"]:
                _state["sigma_max"] = s
                _state["step"] = 0

            sigma_max = _state["sigma_max"]
            sigma_progress = max(0.0, min(1.0, (sigma_max - s) / sigma_max if sigma_max > 1e-6 else 0.0))
            _state["step"] += 1
            step_progress = 1.0 - 0.5 ** _state["step"]
            progress = max(sigma_progress, step_progress)
            curved = progress ** (1.0 / _curve)
            effective = _strength * curved
            if effective < 1e-5:
                return denoised

            ref = _ref_means.to(denoised.device, dtype=denoised.dtype)
            cur = denoised.mean(dim=(-2, -1), keepdim=True)
            correction = ref - cur
            if _ch_trust is not None:
                correction = correction * _ch_trust.to(denoised.device, dtype=denoised.dtype)
            corrected = denoised + correction * effective

            if debug:
                logger.info("Flux2KleinColorAnchor: step=%d sigma=%.4f progress=%.3f effective=%.3f",
                            _state["step"], s, progress, effective)
            return corrected

        m = model.clone()
        m.model_options["sampler_post_cfg_function"] = list(
            m.model_options.get("sampler_post_cfg_function", [])) + [_color_anchor_fn]
        return (m,)


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


class Flux2KleinDetailController:
    """Per-section conditioning multiplier. Reads meta["klein_sections"]
    (emitted by Flux Klein Sectioned Encoder) for real section boundaries;
    falls back to a fixed 25/50/25 split of the active region when that
    metadata is absent (arbitrary boundaries - Qwen3 has no positional
    semantic role for tokens; pair with the Sectioned Encoder for a
    meaningful effect)."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Detail Controller ⚡"
    SEARCH_ALIASES = ['detail controller', 'section multiplier', 'prompt section']
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    DESCRIPTION = "Per-section (front/mid/end) conditioning multiplier, honest with Sectioned Encoder metadata."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",)},
            "optional": {
                "front_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "mid_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "end_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "emphasis_start": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "emphasis_end": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "emphasis_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def control(self, conditioning, front_mult=1.0, mid_mult=1.0, end_mult=1.0,
                emphasis_start=0, emphasis_end=0, emphasis_mult=1.0,
                preserve_original=0.0, debug=False):
        if not conditioning:
            return (conditioning,)

        no_op = (front_mult == 1.0 and mid_mult == 1.0 and end_mult == 1.0
                  and (emphasis_end == 0 or emphasis_mult == 1.0)
                  and preserve_original == 0.0)
        if no_op:
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            original_dtype = cond_tensor.dtype
            cond = cond_tensor.float()
            if cond.dim() != 3:
                output.append((cond_tensor, meta))
                continue

            seq_len = cond.shape[1]
            active_end = _detect_active_end(meta, seq_len, 0)

            sections = meta.get("klein_sections")
            if sections and all(k in sections for k in ("front", "mid", "end")):
                front_range, mid_range, end_range = sections["front"], sections["mid"], sections["end"]
                source = "klein_sections (Sectioned Encoder, real boundaries)"
            else:
                num = active_end
                f_end, m_end = int(num * 0.25), int(num * 0.75)
                front_range, mid_range, end_range = (0, f_end), (f_end, m_end), (m_end, num)
                source = "fixed 25/50/25 fallback (pair with Sectioned Encoder for real ranges)"

            if debug:
                logger.info("Flux2KleinDetailController: item %d active=[0:%d] source=%s front=%s mid=%s end=%s",
                            idx, active_end, source, front_range, mid_range, end_range)

            active = cond[:, :active_end, :].clone()
            original_active = active.clone()

            def _scale(rng, mult):
                s, e = rng
                s = max(0, min(s, active_end))
                e = max(s, min(e, active_end))
                if mult == 1.0 or e <= s:
                    return
                active[:, s:e, :] = active[:, s:e, :] * mult

            _scale(front_range, front_mult)
            _scale(mid_range, mid_mult)
            _scale(end_range, end_mult)
            if emphasis_end > 0 and emphasis_mult != 1.0:
                _scale((emphasis_start, emphasis_end), emphasis_mult)
            if preserve_original > 0.0:
                active = active * (1.0 - preserve_original) + original_active * preserve_original

            result = cond.clone()
            result[:, :active_end, :] = active
            output.append((result.to(original_dtype), meta))

        gc.collect()
        return (output,)


class Flux2KleinTextEnhancer:
    """Normalize / contrast / magnitude adjustments on the active text
    conditioning region, with a safe (never sign-inverting) negative-
    contrast formula."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Text Enhancer ⚡"
    SEARCH_ALIASES = ['text enhancer', 'prompt magnitude', 'token contrast']
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    DESCRIPTION = "Normalize/contrast/magnitude adjustments on the active text conditioning tokens."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "normalize_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "skip_bos": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def enhance(self, conditioning, magnitude=1.0, contrast=0.0,
                normalize_strength=0.0, skip_bos=True, debug=False):
        if not conditioning:
            return (conditioning,)
        if magnitude == 1.0 and contrast == 0.0 and normalize_strength == 0.0:
            return (conditioning,)

        import math
        output = []
        for cond_tensor, meta in conditioning:
            cond = cond_tensor.float().clone()
            seq_len = cond.shape[1]
            active_end = _detect_active_end(meta, seq_len, 0)
            start_idx = 1 if skip_bos else 0
            active = cond[:, start_idx:active_end, :]

            if normalize_strength > 0.0:
                norms = active.norm(dim=-1, keepdim=True)
                mean_norm = norms.mean()
                normalized = active / (norms + 1e-8) * mean_norm
                active = active * (1 - normalize_strength) + normalized * normalize_strength

            if contrast != 0.0:
                seq_mean = active.mean(dim=1, keepdim=True)
                deviation = active - seq_mean
                scale = (1.0 + contrast) if contrast >= 0 else math.exp(contrast)
                active = seq_mean + deviation * scale

            if magnitude != 1.0:
                active = active * magnitude

            cond[:, start_idx:active_end, :] = active
            output.append((cond.to(cond_tensor.dtype), meta))
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


class Flux2KleinMaskRefController:
    """Spatially attenuate a reference latent using a painted mask -
    multiplies it by a per-pixel scalar derived from the mask and replaces
    meta["reference_latents"][reference_index] with the result."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Mask Ref Controller ⚡"
    SEARCH_ALIASES = ['mask ref controller', 'reference mask', 'attenuate reference']
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_mask"
    DESCRIPTION = "Spatially attenuates a reference latent by a painted mask, replacing it in conditioning."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",), "mask": ("MASK",)},
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    @staticmethod
    def _resize_mask_to_latent(mask: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
        if mask.dim() == 2:
            m = mask.unsqueeze(0).unsqueeze(0).float()
        elif mask.dim() == 3:
            m = mask[0:1].unsqueeze(1).float()
        elif mask.dim() == 4:
            m = mask[0:1, 0:1].float()
        else:
            raise ValueError(f"unexpected mask shape {tuple(mask.shape)}")
        return F.interpolate(m, size=(lat_h, lat_w), mode="bilinear", align_corners=False)

    @staticmethod
    def _feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return mask
        sigma = max(radius / 3.0, 1e-6)
        ax = torch.arange(radius * 2 + 1, dtype=torch.float32, device=mask.device) - radius
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return F.conv2d(mask, kernel, padding=radius).clamp(0.0, 1.0)

    def apply_mask(self, conditioning, mask, strength=1.0, invert_mask=False,
                   feather=0, reference_index=0, debug=False):
        if not conditioning or strength == 0.0:
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            new_meta = meta.copy()
            ref_latents = meta.get("reference_latents", None)
            if not ref_latents or reference_index >= len(ref_latents):
                if debug:
                    logger.info("Flux2KleinMaskRefController: item %d has no ref latent "
                                "at index %d, skipping", idx, reference_index)
                output.append((cond_tensor, new_meta))
                continue

            ref = ref_latents[reference_index].float().clone()
            original_dtype = ref_latents[reference_index].dtype
            _, num_ch, lat_h, lat_w = ref.shape

            spatial_mask = self._resize_mask_to_latent(mask, lat_h, lat_w)
            if invert_mask:
                spatial_mask = 1.0 - spatial_mask
            if feather > 0:
                spatial_mask = self._feather_mask(spatial_mask, feather)

            multiplier = (1.0 - strength * (1.0 - spatial_mask)).to(ref.device)
            modified = ref * multiplier

            new_refs = list(ref_latents)
            new_refs[reference_index] = modified.to(original_dtype)
            new_meta["reference_latents"] = new_refs
            output.append((cond_tensor, new_meta))

        gc.collect()
        return (output,)


def _klein_spatial_token_weights(num_tokens, ref_latent, mode, fade_strength, device):
    if mode == "none" or ref_latent is None:
        return None
    _, _, H, W = ref_latent.shape
    patch_size = 2
    h_p = (H + patch_size // 2) // patch_size
    w_p = (W + patch_size // 2) // patch_size
    y = torch.linspace(0.0, 1.0, h_p, device=device)
    x = torch.linspace(0.0, 1.0, w_p, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    if mode == "center_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = 1.0 - dist * fade_strength
    elif mode == "edges_out":
        dist = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
        dist = dist / dist.max().clamp(min=1e-8)
        weights = (1.0 - fade_strength) + dist * fade_strength
    elif mode == "top_down":
        weights = 1.0 - yy * fade_strength
    elif mode == "left_right":
        weights = 1.0 - xx * fade_strength
    else:
        return None

    weights = weights.clamp(0.0, 5.0).flatten()
    n = weights.shape[0]
    if n > num_tokens:
        weights = weights[:num_tokens]
    elif n < num_tokens:
        weights = torch.cat([weights, torch.ones(num_tokens - n, device=device)])
    return weights


class Flux2KleinRefLatentController:
    """Scale a specific reference's K/V contribution at every attention
    block, with an optional spatial fade (center-out/edges-out/top-down/
    left-right) over that reference's own token grid."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Ref Latent Controller ⚡"
    SEARCH_ALIASES = ['ref latent controller', 'reference strength', 'spatial fade']
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "control"
    DESCRIPTION = "Scales one reference's K/V attention contribution, with an optional spatial fade."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.05}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 7}),
            },
            "optional": {
                "spatial_fade": (["none", "center_out", "edges_out", "top_down", "left_right"], {"default": "none"}),
                "spatial_fade_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def control(self, model, conditioning, strength=1.0, reference_index=0,
                spatial_fade="none", spatial_fade_strength=0.5, debug=False):
        m = model.clone()

        ref_latent = None
        if conditioning and spatial_fade != "none":
            for _, meta in conditioning:
                rl = meta.get("reference_latents", None)
                if rl and reference_index < len(rl):
                    ref_latent = rl[reference_index]
                    break

        _strength, _ref_idx, _fade, _fade_s, _ref_latent = (
            strength, reference_index, spatial_fade, spatial_fade_strength, ref_latent)

        def ref_weight_patch(q, k, v, extra_options=None, **kwargs):
            extra_options = extra_options or {}
            ref_tokens = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens or _ref_idx >= len(ref_tokens):
                return {}

            total_ref = sum(ref_tokens)
            tok_start = sum(ref_tokens[:_ref_idx])
            tok_end = tok_start + ref_tokens[_ref_idx]
            num_ref_tok = ref_tokens[_ref_idx]
            seq_start = -total_ref + tok_start
            seq_end = -total_ref + tok_end

            if _fade != "none" and _ref_latent is not None:
                token_w = _klein_spatial_token_weights(num_ref_tok, _ref_latent, _fade, _fade_s, k.device)
                scale = (_strength * token_w).view(1, 1, -1, 1).to(k.dtype) if token_w is not None else _strength
            else:
                scale = _strength

            seq_end_idx = None if seq_end == 0 else seq_end
            k = k.clone()
            v = v.clone()
            k[:, :, seq_start:seq_end_idx, :] = k[:, :, seq_start:seq_end_idx, :] * scale
            v[:, :, seq_start:seq_end_idx, :] = v[:, :, seq_start:seq_end_idx, :] * scale
            if debug:
                logger.info("Flux2KleinRefLatentController: block=%s ref_index=%d tokens=[%d:%d] strength=%.3f",
                            extra_options.get("block_index", "?"), _ref_idx, seq_start, seq_end, _strength)
            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(ref_weight_patch)
        return (m, conditioning)


class Flux2KleinTextRefBalance:
    """Scale text-token vs. reference-token K/V contributions in opposite
    directions from a single balance dial: 0.0 = text only, 1.0 = reference
    only, 0.5 = both unscaled."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Text/Ref Balance ⚡"
    SEARCH_ALIASES = ['text ref balance', 'prompt vs reference']
    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "balance_streams"
    DESCRIPTION = "Single dial trading text-prompt strength against reference-image strength."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {"debug": ("BOOLEAN", {"default": False})},
        }

    def balance_streams(self, model, conditioning, balance=0.5, debug=False):
        m = model.clone()
        if balance <= 0.5:
            text_scale, ref_scale = balance * 2.0, 1.0
        else:
            text_scale, ref_scale = 1.0, (1.0 - balance) * 2.0

        _text_s, _ref_s = text_scale, ref_scale

        def balance_patch(q, k, v, extra_options=None, **kwargs):
            extra_options = extra_options or {}
            img_slice = extra_options.get("img_slice", None)
            ref_tokens = extra_options.get("reference_image_num_tokens", [])
            if img_slice is None and not ref_tokens:
                return {}

            k = k.clone()
            v = v.clone()
            if img_slice is not None and _text_s != 1.0:
                txt_end = img_slice[0]
                k[:, :, :txt_end, :] *= _text_s
                v[:, :, :txt_end, :] *= _text_s
            if ref_tokens and _ref_s != 1.0:
                total_ref = sum(ref_tokens)
                k[:, :, -total_ref:, :] *= _ref_s
                v[:, :, -total_ref:, :] *= _ref_s
            if debug:
                logger.info("Flux2KleinTextRefBalance: block=%s txt_scale=%.3f ref_scale=%.3f",
                            extra_options.get("block_index", "?"), _text_s, _ref_s)
            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(balance_patch)
        return (m, conditioning)


class Flux2KleinRefLatentWeight:
    """Flat K/V weight multiplier for a single reference's tokens - the
    simple case of Ref Latent Controller with no conditioning input or
    spatial fade needed."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Ref Latent Weight ⚡"
    SEARCH_ALIASES = ['ref latent weight', 'reference weight']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    DESCRIPTION = "Flat attention K/V weight multiplier for one reference's tokens."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 7}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
        }

    def execute(self, model, reference_index, weight):
        m = model.clone()
        _ref_idx, _weight = reference_index, weight

        def ref_weight_patch(q, k, v, extra_options=None, **kwargs):
            extra_options = extra_options or {}
            ref_tokens = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens or _ref_idx >= len(ref_tokens):
                return {}
            total_ref = sum(ref_tokens)
            tok_start = sum(ref_tokens[:_ref_idx])
            tok_end = tok_start + ref_tokens[_ref_idx]
            seq_start = -total_ref + tok_start
            seq_end = -total_ref + tok_end
            seq_end_idx = None if seq_end == 0 else seq_end
            k = k.clone()
            v = v.clone()
            k[:, :, seq_start:seq_end_idx, :] *= _weight
            v[:, :, seq_start:seq_end_idx, :] *= _weight
            return {"q": q, "k": k, "v": v}

        m.set_model_attn1_patch(ref_weight_patch)
        return (m,)


class Flux2KleinIdentityGuidance:
    """Sampling-loop correction: pulls each step's x0 prediction toward a
    VAE-encoded identity reference latent, over a configurable sigma
    window, via one of three blend modes. Registers via
    model_options["sampler_post_cfg_function"] - same stock-KSampler
    requirement as Flux2KleinColorAnchor."""

    CATEGORY = FLUX_KLEIN_CATEGORY
    TITLE = "Flux Klein Identity Guidance ⚡"
    SEARCH_ALIASES = ['identity guidance', 'identity correction', 'sampling loop correction']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    DESCRIPTION = "Pulls each sampling step toward a reference latent over a sigma window. Requires sampling through stock KSampler."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "identity_latent": ("LATENT", {"tooltip": "VAE-encoded reference image at full resolution."}),
                "strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_percent": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mode": (["adaptive", "direct", "channel_match"], {"default": "adaptive"}),
            },
        }

    def apply(self, model, identity_latent, strength=0.3,
              start_percent=0.0, end_percent=0.8, mode="adaptive"):
        m = model.clone()
        ref = identity_latent["samples"]
        _ref, _strength, _start, _end, _mode = ref, strength, start_percent, end_percent, mode

        def post_cfg_fn(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            s_now = float(sigma.flatten()[0])
            progress = max(0.0, min(1.0, 1.0 - s_now))
            if progress < _start or progress > _end:
                return denoised

            ref_resized = _ref.to(device=denoised.device, dtype=denoised.dtype)
            if ref_resized.shape[0] != denoised.shape[0]:
                ref_resized = ref_resized[:1].expand(denoised.shape[0], -1, -1, -1)
            if ref_resized.shape[2:] != denoised.shape[2:]:
                ref_resized = F.interpolate(ref_resized, size=denoised.shape[2:], mode="bilinear", align_corners=False)
            if ref_resized.shape[1] != denoised.shape[1]:
                if ref_resized.shape[1] > denoised.shape[1]:
                    ref_resized = ref_resized[:, :denoised.shape[1]]
                else:
                    ref_resized = F.pad(ref_resized, (0, 0, 0, 0, 0, denoised.shape[1] - ref_resized.shape[1]))

            if _mode == "direct":
                denoised = denoised + (ref_resized - denoised) * _strength
            elif _mode == "adaptive":
                d_flat = denoised.flatten(2)
                r_flat = ref_resized.flatten(2)
                cos_sim = F.cosine_similarity(d_flat, r_flat, dim=1)
                weight = cos_sim.clamp(0.0, 1.0).unsqueeze(1).view(
                    denoised.shape[0], 1, denoised.shape[2], denoised.shape[3])
                denoised = denoised + (ref_resized - denoised) * weight * _strength
            elif _mode == "channel_match":
                ref_mean = ref_resized.mean(dim=(2, 3), keepdim=True)
                ref_std = ref_resized.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
                den_mean = denoised.mean(dim=(2, 3), keepdim=True)
                den_std = denoised.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
                matched = (denoised - den_mean) / den_std * ref_std + ref_mean
                denoised = denoised + (matched - denoised) * _strength
            return denoised

        m.model_options["sampler_post_cfg_function"] = list(
            m.model_options.get("sampler_post_cfg_function", [])) + [post_cfg_fn]
        return (m,)


NODE_CLASS_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader,
    "FluxKleinImg2Img": FluxKleinImg2Img,
    # Backward-compat alias: this node's logic moved to the shared
    # nodes/preprocessors.py DepthMap class - old saved workflows using the
    # "Flux2KleinDepthMap" type id keep resolving to a working node.
    "Flux2KleinDepthMap": DepthMap,
    "Flux2KleinMultiReferenceLatent": Flux2KleinMultiReferenceLatent,
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer,
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor,
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinTextEnhancer": Flux2KleinTextEnhancer,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
    "Flux2KleinRefLatentController": Flux2KleinRefLatentController,
    "Flux2KleinTextRefBalance": Flux2KleinTextRefBalance,
    "Flux2KleinRefLatentWeight": Flux2KleinRefLatentWeight,
    "Flux2KleinIdentityGuidance": Flux2KleinIdentityGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader.TITLE,
    "FluxKleinImg2Img": FluxKleinImg2Img.TITLE,
    "Flux2KleinDepthMap": "Flux Klein Depth Map ⚡",
    "Flux2KleinMultiReferenceLatent": Flux2KleinMultiReferenceLatent.TITLE,
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer.TITLE,
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor.TITLE,
    "Flux2KleinEnhancer": Flux2KleinEnhancer.TITLE,
    "Flux2KleinDetailController": Flux2KleinDetailController.TITLE,
    "Flux2KleinTextEnhancer": Flux2KleinTextEnhancer.TITLE,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder.TITLE,
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController.TITLE,
    "Flux2KleinRefLatentController": Flux2KleinRefLatentController.TITLE,
    "Flux2KleinTextRefBalance": Flux2KleinTextRefBalance.TITLE,
    "Flux2KleinRefLatentWeight": Flux2KleinRefLatentWeight.TITLE,
    "Flux2KleinIdentityGuidance": Flux2KleinIdentityGuidance.TITLE,
}
