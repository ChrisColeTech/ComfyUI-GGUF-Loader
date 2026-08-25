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
        from .nodes import UnetLoaderGGUF, CLIPLoaderGGUF

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
    """Prompts and init latent for FLUX.2 Klein - one node.

    Leave image unconnected for txt2img. Does NOT handle multi-reference
    images itself (see Flux2KleinMultiReferenceLatent) - Klein's real
    multi-reference workflow is `positive conditioning -> Multi
    ReferenceLatent -> sampler`, matching the reference example workflow's
    own structure, so this node stays a plain prompt/img2img prep step and
    reference attachment lives in its own node instead of needing 8 image
    input slots here.

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
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img', 'encode image']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts and init latent for FLUX.2 Klein. Leave image "
                   "unconnected for txt2img. Feed the outputs straight into "
                   "a stock KSampler.")

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
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, image=None):
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


NODE_CLASS_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader,
    "FluxKleinImg2Img": FluxKleinImg2Img,
    "Flux2KleinMultiReferenceLatent": Flux2KleinMultiReferenceLatent,
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKleinModelLoader": FluxKleinModelLoader.TITLE,
    "FluxKleinImg2Img": FluxKleinImg2Img.TITLE,
    "Flux2KleinMultiReferenceLatent": Flux2KleinMultiReferenceLatent.TITLE,
    "Flux2KleinIdentityFeatureTransfer": Flux2KleinIdentityFeatureTransfer.TITLE,
}
