"""Krea2 Control nodes: component loader + one-node img2img/prompt/control prep.

  Krea2ModelLoader     unet / clip / vae by name -> MODEL, CLIP, VAE
  Krea2ControlLoRALoader  model + control-LoRA file -> LoRA-patched model
  Krea2Img2Img          model, clip, vae, prompts (+ optional init image)
                         -> model, positive, negative, latent, denoise
                         -> stock KSampler. No ControlNet inputs.
  Krea2ControlNetImg2Img  same as Krea2Img2Img, plus control_image/
                         control_mode for Control-LoRA-guided generation -
                         kept as a separate node so the plain one above
                         never shows control-related inputs.

Krea2 is natively detected by ComfyUI core (comfy.sd.load_diffusion_model_state_dict
picks it up via unet_config.image_model == "krea2"; comfy.sd.CLIPType.KREA2 selects
its Qwen3-VL-4B text encoder) - there is no bespoke sampling algorithm or
conditioning format to reimplement here, unlike LTX-2.3. Krea2ModelLoader is a
thin GGUF-aware convenience loader, matching nodes_zimage.py's ZImageLoader.

The Control LoRA mechanism has no comfy-native or LTX-2.3 equivalent at all: a
Krea2 Control LoRA ships with the DiT's first input-projection layer WIDENED
(trained to accept image tokens concatenated with control tokens), plus small
LoRA-rank patches on the attention blocks. Krea2ControlLoRALoader's load/patch/
inject/wrapper mechanics (widened input projection, LoRAAdapter block patches,
a PatcherInjection that swaps `diffusion_model.first` in and back out per
forward call, a DIFFUSION_MODEL wrapper that turns the encoded control latent
into control tokens at sample time) are ported essentially verbatim from the
local comfyui-krea2-controlnet-main pack (no LICENSE file shipped; its own
README credits Tanmaypatil123/Krea-2-controlnet for documenting the reference
pipeline and Patil/Krea-2-depth-controlnet for the public depth LoRA weights) -
this is correctness-critical low-level ModelPatcher plumbing validated against
a working pack, not something to rederive from scratch.

The img2img/prompt/control-image prep is NOT a port of that pack's separate
Krea2ControlImageEncode + Krea2ControlApply nodes. It follows this repo's own
convention instead (see nodes_zimage.py's ZImageImg2Img, which bundles model +
clip + vae + prompts + optional init image + optional ControlNet into one
node): Krea2Img2Img does the VAE-encode-and-attach step inline, and raises
loudly if a Control LoRA is loaded with no control_image (or vice versa) -
the same "never silently run a half-configured model" guarantee the original
pack's separate Apply node existed for, kept without a second required node.

Krea2EditModelPatch + Krea2EditGroundedEncode are a faithful port of
comfyui-krea2edit (Apache-2.0, lbouaraba, https://github.com/lbouaraba/
comfyui-krea2edit) - the node pack behind the separately-licensed "Krea 2
Identity Edit" LoRA (krea2_identity_edit_v1_2.safetensors, Krea 2 Community
License Agreement - weights are NOT bundled or vendored here, download
separately). This is a genuinely different mechanism from Krea2Img2Img's
control_image path and from the removed edit_reference/reference_latents
attempt: comfy's native Krea2 forward only ever builds `[text | target]` -
no built-in way to prepend a clean source-image block - so this LoRA needs
the diffusion model's forward wrapped directly to rebuild the sequence as
`[text | source(frame=1) | target(frame=0)]`, exactly matching how it was
trained (ai-toolkit's predict_velocity_edit). Ported via a real comfy
ModelPatcher extension point (comfy.patcher_extension.WrappersMP.
DIFFUSION_MODEL - the same mechanism this file's own Krea2ControlLoRALoader
wrapper already uses, and the Identity Feature Transfer hooks in
nodes_flux_klein.py use elsewhere in this pack) - not a from-scratch
sampler, CFGGuider/sampler_post_cfg_function stays fully intact.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.ldm.common_dit
import comfy.model_management
import comfy.patcher_extension
import comfy.sd
import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
import numpy as np
import folder_paths
import node_helpers
import nodes

from ..vendor import depth_anything_v2
from .preprocessors import DepthMap, _auto_canny_control_image, _depth_anything_batch

logger = logging.getLogger(__name__)

KREA2_CATEGORY = "\U0001F916 CCTech/Krea2"

CONTROL_LATENT_KEY = "krea2_control_latent"
WRAPPER_KEY = "krea2_control"
EPS = 1e-6


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


# ── Control LoRA mechanics (ported from comfyui-krea2-controlnet-main) ──────

class Krea2ControlInputProjection(nn.Module):
    def __init__(self, weight, bias=None, image_features=None, original_first=None):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("Krea2 control input projection weight must be a 2D tensor.")

        total_features = weight.shape[1]
        if image_features is None:
            if total_features % 2 != 0:
                raise ValueError("Cannot infer Krea2 image/control feature split from odd input width.")
            image_features = total_features // 2
        if image_features <= 0 or image_features >= total_features:
            raise ValueError("Invalid Krea2 image/control feature split.")

        self.image_features = int(image_features)
        self.control_features = int(total_features - image_features)
        self.out_features = int(weight.shape[0])
        self.in_features = int(total_features)
        self.weight = nn.Parameter(weight.detach().cpu().clone(), requires_grad=False)
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().cpu().clone(), requires_grad=False)
        self.control_tokens = None
        object.__setattr__(self, "_original_first", original_first)

    @property
    def original_first(self):
        return object.__getattribute__(self, "_original_first")

    def set_original_first(self, original_first):
        object.__setattr__(self, "_original_first", original_first)

    def forward(self, image_tokens):
        if image_tokens.shape[-1] != self.image_features:
            raise RuntimeError(
                f"Krea2 control projection expected {self.image_features} image features, "
                f"got {image_tokens.shape[-1]}."
            )

        control_tokens = self.control_tokens
        if control_tokens is None:
            original_first = self.original_first
            if original_first is not None:
                return original_first(image_tokens)
            raise RuntimeError("Krea2 control projection was called without control tokens.")
        if control_tokens.shape[1] != image_tokens.shape[1]:
            raise RuntimeError(
                f"Krea2 control token count mismatch: image={image_tokens.shape[1]}, "
                f"control={control_tokens.shape[1]}."
            )
        control_tokens = comfy.utils.repeat_to_batch_size(control_tokens, image_tokens.shape[0])
        control_tokens = control_tokens.to(device=image_tokens.device, dtype=image_tokens.dtype)

        original_first = self.original_first
        if original_first is not None:
            image_out = original_first(image_tokens)
            control_weight = self.weight[:, self.image_features:]
            control_weight = comfy.model_management.cast_to_device(control_weight, image_tokens.device, image_tokens.dtype)
            return image_out + F.linear(control_tokens, control_weight, None)

        x = torch.cat((image_tokens, control_tokens), dim=-1)
        weight = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        bias = None
        if self.bias is not None:
            bias = comfy.model_management.cast_to_device(self.bias, x.device, x.dtype)
        return F.linear(x, weight, bias)


def _tensor_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _resize_image(image, width, height, upscale_method="lanczos", crop="center"):
    samples = image[..., :3].clamp(0.0, 1.0).movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def _prepare_control_image(image, channel_mode, normalize, invert):
    if image.ndim != 4:
        raise RuntimeError(f"Krea2 control IMAGE must be 4D [B,H,W,C], got shape {tuple(image.shape)}.")
    if image.shape[-1] < 1:
        raise RuntimeError("Krea2 control IMAGE must have at least one channel.")

    image = image.clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    else:
        image = image[..., :3]

    if channel_mode == "grayscale":
        weights = torch.tensor((0.299, 0.587, 0.114), device=image.device, dtype=image.dtype)
        image = (image * weights).sum(dim=-1, keepdim=True).repeat(1, 1, 1, 3)

    if normalize == "per_image_minmax":
        reduce_dims = tuple(range(1, image.ndim))
        image_min = image.amin(dim=reduce_dims, keepdim=True)
        image_max = image.amax(dim=reduce_dims, keepdim=True)
        image = (image - image_min) / (image_max - image_min).clamp_min(EPS)

    if invert:
        image = 1.0 - image

    return image.clamp(0.0, 1.0)


def _encode_control_image(vae, image, batch_mode):
    latent_dim = getattr(vae, "latent_dim", None)
    treats_batch_as_video = latent_dim == 3 and not getattr(vae, "not_video", False)
    if batch_mode == "independent_images" and treats_batch_as_video and image.shape[0] > 1:
        encoded = []
        for i in range(image.shape[0]):
            encoded.append(vae.encode(image[i : i + 1]))
        return torch.cat(encoded, dim=0)
    return vae.encode(image)


def _find_first_weight_key(state_dict, out_features, in_features):
    preferred = (
        "first.weight",
        "diffusion_model.first.weight",
        "model.diffusion_model.first.weight",
        "transformer.first.weight",
    )
    for key in preferred:
        value = state_dict.get(key)
        if torch.is_tensor(value) and tuple(value.shape) == (out_features, in_features):
            return key

    for key, value in state_dict.items():
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        if tuple(value.shape) != (out_features, in_features):
            continue
        if key.endswith("first.weight") or key.endswith("img_in.weight"):
            return key
    return None


def _find_matching_bias(state_dict, weight_key, out_features):
    candidates = []
    if weight_key.endswith(".weight"):
        candidates.append(weight_key[:-7] + ".bias")
    candidates.extend(
        (
            "first.bias",
            "diffusion_model.first.bias",
            "model.diffusion_model.first.bias",
            "transformer.first.bias",
        )
    )
    for key in candidates:
        value = state_dict.get(key)
        if torch.is_tensor(value) and tuple(value.shape) == (out_features,):
            return value
    return None


def _strip_known_prefixes(base):
    changed = True
    while changed:
        changed = False
        for prefix in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
    return base


def _target_key_from_lora_base(base):
    base = _strip_known_prefixes(base)
    if base.startswith("blocks."):
        return f"diffusion_model.{base}.weight"
    return None


def _lora_pairs(state_dict):
    pair_specs = (
        (".A", ".B"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora_A", ".lora_B"),
        (".lora_down.weight", ".lora_up.weight"),
        (".lora_down", ".lora_up"),
        ("_lora.down.weight", "_lora.up.weight"),
    )

    seen = set()
    for down_suffix, up_suffix in pair_specs:
        for down_key in state_dict.keys():
            if not down_key.endswith(down_suffix):
                continue
            base = down_key[: -len(down_suffix)]
            up_key = base + up_suffix
            if up_key not in state_dict:
                continue
            pair_id = (down_key, up_key)
            if pair_id in seen:
                continue
            seen.add(pair_id)
            yield base, down_key, up_key


def _get_nested_model_attr(obj, key):
    for part in key.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            if part.isdigit() and hasattr(obj, "__getitem__"):
                obj = obj[int(part)]
            else:
                raise
    return obj


def _shape_from_weight(weight):
    tensor_shape = getattr(weight, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(tensor_shape)
    data = getattr(weight, "data", None)
    tensor_shape = getattr(data, "tensor_shape", None)
    if tensor_shape is not None:
        return tuple(tensor_shape)
    shape = getattr(weight, "shape", None)
    if shape is not None:
        return tuple(shape)
    return None


def _shape_from_model_key(model_patcher, key):
    try:
        weight = _get_nested_model_attr(model_patcher.model, key)
    except Exception:
        return None
    return _shape_from_weight(weight)


def _build_lora_patches(state_dict, model_patcher):
    from comfy.weight_adapter.lora import LoRAAdapter

    patches = {}
    loaded_keys = set()
    skipped = []
    model_state_dict = model_patcher.model.state_dict()

    for base, down_key, up_key in _lora_pairs(state_dict):
        target_key = _target_key_from_lora_base(base)
        if target_key is None:
            continue

        down = state_dict[down_key]
        up = state_dict[up_key]
        target_shape = _shape_from_model_key(model_patcher, target_key)
        if target_shape is None:
            value = model_state_dict.get(target_key)
            if torch.is_tensor(value):
                target_shape = tuple(value.shape)
        if target_shape is None:
            continue
        if len(target_shape) < 2:
            skipped.append((down_key, up_key, f"target shape is {target_shape}"))
            continue

        if not (torch.is_tensor(down) and torch.is_tensor(up) and down.ndim == 2 and up.ndim == 2):
            skipped.append((down_key, up_key, "not 2D tensors"))
            continue

        out_features, in_features = target_shape[0], target_shape[1]
        if up.shape[0] == out_features and down.shape[1] == in_features and up.shape[1] == down.shape[0]:
            rank = down.shape[0]
        elif down.shape[0] == in_features and up.shape[1] == out_features and down.shape[1] == up.shape[0]:
            down = down.t().contiguous()
            up = up.t().contiguous()
            rank = down.shape[0]
        else:
            skipped.append((down_key, up_key, f"shape does not match {target_key}"))
            continue

        alpha_key = None
        alpha = rank
        for suffix in (".alpha", ".network_alpha", ".scale"):
            candidate = base + suffix
            if candidate in state_dict:
                alpha_key = candidate
                alpha = _tensor_scalar(state_dict[candidate])
                break

        keys = {down_key, up_key}
        if alpha_key is not None:
            keys.add(alpha_key)
        patches[target_key] = LoRAAdapter(keys, (up, down, alpha, None, None, None))
        loaded_keys.update(keys)

    if skipped:
        logger.info("Krea2 control skipped %d LoRA tensor pairs with incompatible shapes.", len(skipped))
    return patches, loaded_keys


def _get_first_module(model_patcher):
    try:
        return model_patcher.get_model_object("diffusion_model.first")
    except Exception as exc:
        raise RuntimeError("The supplied MODEL does not look like a native ComfyUI Krea2 model.") from exc


def _first_shape(first):
    if isinstance(first, Krea2ControlInputProjection):
        return first.out_features, first.image_features, first.control_features
    weight = getattr(first, "weight", None)
    weight_shape = _shape_from_weight(weight)
    if weight_shape is None or len(weight_shape) != 2:
        raise RuntimeError("Krea2 first projection does not expose a 2D weight tensor.")
    return int(weight_shape[0]), int(weight_shape[1]), int(weight_shape[1])


def _lora_expanded_first_weight_key(model_patcher, state_dict):
    """Detection only, never raises: the expanded first-layer weight key if
    state_dict shape-matches the live model's widened first projection, else
    None. Used to auto-detect a widened-projection Control LoRA vs an
    ordinary in-context LoRA (no expanded projection at all) before deciding
    which mechanism to apply the file with.
    """
    try:
        first = _get_first_module(model_patcher)
        out_features, image_features, control_features = _first_shape(first)
    except Exception:
        return None
    expected_in = image_features + control_features
    return _find_first_weight_key(state_dict, out_features, expected_in)


def _make_control_projection(model_patcher, state_dict):
    first = _get_first_module(model_patcher)
    out_features, image_features, control_features = _first_shape(first)
    expected_in = image_features + control_features
    weight_key = _lora_expanded_first_weight_key(model_patcher, state_dict)
    if weight_key is None:
        raise RuntimeError(
            f"Could not find expanded Krea2 first projection weight with shape "
            f"({out_features}, {expected_in}) in the selected LoRA file."
        )

    bias = _find_matching_bias(state_dict, weight_key, out_features)
    if bias is None and hasattr(first, "bias") and torch.is_tensor(first.bias):
        bias = first.bias.detach()

    if isinstance(first, Krea2ControlInputProjection):
        original_first = first.original_first
    else:
        original_first = first

    return Krea2ControlInputProjection(
        state_dict[weight_key],
        bias=bias,
        image_features=image_features,
        original_first=original_first,
    )


def _clean_original_first(first):
    if isinstance(first, Krea2ControlInputProjection) and first.original_first is not None:
        return first.original_first
    return first


def _flatten_temporal_if_needed(control_latent):
    if control_latent.ndim == 4:
        return control_latent
    if control_latent.ndim == 5:
        b, c, t, h, w = control_latent.shape
        return control_latent.reshape(b * t, c, h, w)
    raise RuntimeError(f"Krea2 control latent must be 4D or 5D, got shape {tuple(control_latent.shape)}.")


def _expected_latent_channels(model_patcher):
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception:
        return None
    return getattr(latent_format, "latent_channels", None)


def _process_control_latent_for_model(model_patcher, control_latent):
    if control_latent.ndim not in (4, 5):
        raise RuntimeError(f"Krea2 control latent must be 4D or 5D, got shape {tuple(control_latent.shape)}.")

    expected_channels = _expected_latent_channels(model_patcher)
    if expected_channels is not None and control_latent.shape[1] != expected_channels:
        raise RuntimeError(
            f"Krea2 control latent has {control_latent.shape[1]} channels, "
            f"but the selected model expects {expected_channels}. Use the Krea2 VAE."
        )

    processed = control_latent
    try:
        latent_format = model_patcher.get_model_object("latent_format")
    except Exception:
        latent_format = None

    added_time_dim = False
    if latent_format is not None and getattr(latent_format, "latent_dimensions", 2) == 3 and processed.ndim == 4:
        processed = processed.unsqueeze(2)
        added_time_dim = True

    if hasattr(model_patcher.model, "process_latent_in"):
        processed = model_patcher.model.process_latent_in(processed)

    if added_time_dim and processed.ndim == 5 and processed.shape[2] == 1:
        processed = processed[:, :, 0]
    return processed


def _control_tokens_from_latent(control_latent, x, patch, expected_features):
    if x.ndim == 5:
        target_batch = x.shape[0] * x.shape[2]
    elif x.ndim == 4:
        target_batch = x.shape[0]
    else:
        raise RuntimeError(f"Krea2 input latent must be 4D or 5D, got shape {tuple(x.shape)}.")

    control = _flatten_temporal_if_needed(control_latent)
    control = comfy.utils.repeat_to_batch_size(control, target_batch)
    control = comfy.model_management.cast_to_device(control, x.device, x.dtype)

    target_h, target_w = x.shape[-2], x.shape[-1]
    if control.shape[-2:] != (target_h, target_w):
        control = comfy.utils.common_upscale(control, target_w, target_h, "bilinear", "disabled")

    control = comfy.ldm.common_dit.pad_to_patch_size(control, (patch, patch))
    b, c, h, w = control.shape
    if h % patch != 0 or w % patch != 0:
        raise RuntimeError("Krea2 control latent padding failed to align to patch size.")

    features = c * patch * patch
    if features != expected_features:
        raise RuntimeError(
            f"Krea2 control latent produces {features} token features, "
            f"but the projection expects {expected_features}. Check that you encoded with the Krea2 VAE."
        )

    control = control.reshape(b, c, h // patch, patch, w // patch, patch)
    control = control.permute(0, 2, 4, 1, 3, 5).reshape(b, (h // patch) * (w // patch), features)
    return control


def _get_transformer_options_from_forward(args, kwargs):
    transformer_options = kwargs.get("transformer_options", None)
    if transformer_options is None and len(args) >= 5 and isinstance(args[4], dict):
        transformer_options = args[4]
    if transformer_options is None and len(args) > 0 and isinstance(args[-1], dict):
        transformer_options = args[-1]
    return transformer_options


def _restore_control_projection(diffusion_model, control_projection):
    control_projection.control_tokens = None
    original_first = control_projection.original_first
    if original_first is not None and getattr(diffusion_model, "first", None) is control_projection:
        diffusion_model.first = original_first


def _make_control_projection_injection(control_projection):
    def inject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is None:
            return
        current_first = _clean_original_first(getattr(diffusion_model, "first", None))
        if current_first is not None and current_first is not control_projection:
            control_projection.set_original_first(current_first)
            diffusion_model.first = current_first
        control_projection.control_tokens = None

    def eject(model_patcher):
        diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
        if diffusion_model is not None:
            _restore_control_projection(diffusion_model, control_projection)

    return [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)]


def _restore_control_projection_callback(model_patcher, *args):
    attachment = model_patcher.get_attachment(WRAPPER_KEY)
    if not isinstance(attachment, dict):
        return
    control_projection = attachment.get("control_projection")
    if not isinstance(control_projection, Krea2ControlInputProjection):
        return
    diffusion_model = getattr(model_patcher.model, "diffusion_model", None)
    if diffusion_model is None:
        return
    _restore_control_projection(diffusion_model, control_projection)


def _krea2_control_wrapper(control_projection):
    def wrapper(executor, *args, **kwargs):
        return krea2_control_wrapper(executor, control_projection, *args, **kwargs)

    return wrapper


def krea2_control_wrapper(executor, control_projection, *args, **kwargs):
    transformer_options = _get_transformer_options_from_forward(args, kwargs)
    if not isinstance(transformer_options, dict):
        raise RuntimeError("Krea2 Control LoRA could not find transformer_options during sampling.")

    diffusion_model = executor.class_obj
    control_latent = transformer_options.get(CONTROL_LATENT_KEY)
    if control_latent is None:
        _restore_control_projection(diffusion_model, control_projection)
        raise RuntimeError(
            "Krea2 Control LoRA is loaded, but no control latent is attached. "
            "Connect control_image on Krea2Img2Img, or remove Krea2ControlLoRALoader."
        )

    if not isinstance(control_projection, Krea2ControlInputProjection):
        raise RuntimeError("Krea2 Control LoRA input projection is not installed. Reload the base model and loader.")

    x = args[0]
    previous_first = getattr(diffusion_model, "first", None)
    previous_tokens = control_projection.control_tokens
    try:
        control_tokens = _control_tokens_from_latent(
            control_latent,
            x,
            diffusion_model.patch,
            control_projection.control_features,
        )
        control_projection.control_tokens = control_tokens
        if getattr(diffusion_model, "first", None) is not control_projection:
            diffusion_model.first = control_projection
        return executor(*args, **kwargs)
    finally:
        control_projection.control_tokens = previous_tokens
        if getattr(diffusion_model, "first", None) is control_projection:
            original_first = control_projection.original_first
            diffusion_model.first = original_first if original_first is not None else previous_first


# ── Nodes ─────────────────────────────────────────────────────────────────

class Krea2ModelLoader:
    """Load the three Krea2 components by name.

    Krea2 is natively supported by ComfyUI core, so this is a thin
    convenience loader (like ZImageLoader) rather than anything bespoke -
    GGUF and safetensors both work for the transformer and the text encoder.
    """

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 Model Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'load clip', 'krea2 loader', 'checkpoint loader']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load Krea2's UNET, CLIP (Qwen3-VL-4B), and VAE as native "
                   "comfy objects. GGUF quants stay quantized.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "Krea2 diffusion model - .safetensors or GGUF quant, "
                               "from models/diffusion_models (unet)."}),
                "clip_name": (_clip_filename_list(), {
                    "tooltip": "Krea2 text encoder (Qwen3-VL-4B) - .safetensors or "
                               "GGUF quant, from models/text_encoders (clip)."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "Krea2 VAE, from models/vae."}),
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
            clip, = CLIPLoaderGGUF().load_clip(clip_name, type="krea2")
        else:
            clip = comfy.sd.load_clip(
                ckpt_paths=[folder_paths.get_full_path_or_raise("clip", clip_name)],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=comfy.sd.CLIPType.KREA2)

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        return (model, clip, vae)


class Krea2ControlLoRALoader:
    """Load a Krea2 LoRA (e.g. depth-control-lora.safetensors,
    krea2_canny-v0.1.safetensors) onto a MODEL - auto-detects which of two
    unrelated mechanisms the file actually needs, so you don't have to know
    in advance (same auto-detect-and-dispatch approach as
    QwenImageControlNetLoader, which does the same thing for Qwen-Image's
    own two ControlNet formats):

      Widened-projection Control LoRA (e.g. the depth one) - detected by
      shape-matching an expanded `first` layer weight against the live
      model. Ported from comfyui-krea2-controlnet-main's
      Krea2ControlLoRALoader: widens the DiT's first input projection,
      patches the attention-block LoRA weights through the normal
      ModelPatcher machinery, and registers the sampling wrapper +
      injection that swap the widened projection in only for the duration
      of each forward call. Connect Krea2Img2Img's control_image after
      this - sampling raises if loaded with no control latent attached.

      Ordinary in-context LoRA (e.g. krea2_canny-v0.1.safetensors - no
      expanded projection at all, confirmed by inspecting its tensor
      keys) - applied as a normal LoRA via comfy.sd.load_lora_for_models,
      the same call stock LoraLoaderModelOnly makes. No wrapper, no
      injection needed - it's a plain weight patch. Krea2Img2Img has no
      reference_latents-style attachment for this kind anymore - install
      ComfyUI-Flux-Reference-Tools and wire its reference-conditioning
      node onto `model` instead (works on any Flux-family model, not
      just Krea2).
    """

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 Control LoRA Loader ⚡"
    SEARCH_ALIASES = ['load lora', 'controlnet loader', 'control lora', 'depth control',
                       'krea2 controlnet', 'apply controlnet', 'in-context lora', 'edit lora']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    DESCRIPTION = ("Load a Krea2 LoRA (depth/canny/pose/edit/etc.) from models/loras "
                   "and patch it onto the model - auto-detects whether it's a "
                   "widened-projection Control LoRA (control_image) or an ordinary "
                   "in-context LoRA (see ComfyUI-Flux-Reference-Tools for that kind).")

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "Any Krea2 LoRA file, from models/loras. Auto-detects "
                               "whether it's a widened-projection Control LoRA (use "
                               "Krea2Img2Img's control_image after this) or an ordinary "
                               "in-context LoRA (install ComfyUI-Flux-Reference-Tools for "
                               "that kind's reference-conditioning attachment)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    def load_lora(self, model, lora_name, strength):
        if strength == 0:
            return (model,)
        if model.get_attachment(WRAPPER_KEY) is not None:
            raise RuntimeError("A Krea2 Control LoRA is already loaded on this MODEL. Use only one loader per model path.")

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state_dict = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                state_dict = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if state_dict is None:
            state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, state_dict)

        if _lora_expanded_first_weight_key(model, state_dict) is None:
            # No widened first-layer projection at all - this is an ordinary
            # in-context LoRA (e.g. krea2_canny-v0.1.safetensors), not a
            # widened-projection Control LoRA. Apply it the same way stock
            # LoraLoaderModelOnly would - no wrapper/injection needed, since
            # there's no runtime control-token swap to perform.
            new_model, _ = comfy.sd.load_lora_for_models(model, None, state_dict, strength, 0.0)
            logger.info(
                "Krea2ControlLoRALoader: '%s' has no widened first-layer projection - "
                "applied as an ordinary in-context LoRA. Krea2Img2Img has no "
                "reference_latents attachment for this kind - install ComfyUI-Flux-"
                "Reference-Tools and wire its reference-conditioning node onto model "
                "instead.", lora_name)
            return (new_model,)

        new_model = model.clone()
        control_projection = _make_control_projection(new_model, state_dict)
        lora_patches, loaded_keys = _build_lora_patches(state_dict, new_model)
        if not lora_patches:
            raise RuntimeError("No compatible Krea2 control LoRA block weights were found in the selected file.")

        patched_keys = new_model.add_patches(lora_patches, strength_patch=strength, strength_model=1.0)
        if not patched_keys:
            raise RuntimeError("The selected MODEL did not accept any Krea2 control LoRA patches.")

        new_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            _krea2_control_wrapper(control_projection),
        )
        new_model.set_injections(WRAPPER_KEY, _make_control_projection_injection(control_projection))
        new_model.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_DETACH,
            WRAPPER_KEY,
            _restore_control_projection_callback,
        )
        new_model.add_callback_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
            WRAPPER_KEY,
            _restore_control_projection_callback,
        )
        new_model.set_attachments(
            WRAPPER_KEY,
            {
                "lora_name": lora_name,
                "strength": strength,
                "loaded_lora_keys": len(loaded_keys),
                "patched_model_keys": len(patched_keys),
                "control_projection": control_projection,
            },
        )
        return (new_model,)


class Krea2Img2Img:
    """Prompts and init latent for Krea2 - one node, no ControlNet.

    One image-shaped slot: `images`. Leave unconnected for txt2img. For
    ControlNet-guided generation (Krea2 Control LoRA), use `Krea2ControlNetImg2Img`
    instead - kept as a separate node on purpose, so this one never shows
    control-related inputs at all. If a Krea2 Control LoRA happens to be
    loaded upstream anyway and this node is used, sampling raises the
    model's own clear error ("no control latent is attached") rather than
    silently half-configuring anything - but this node has no way to
    attach one itself.

    `images` is real img2img: VAE-encoded, then partially denoised at
    `strength` (comfy's own `KSampler`-style img2img - noise added onto the
    encoded latent proportional to `1 - strength`, sampled from there). It's
    batch-aware - a batch of N photos naturally becomes N independent img2img
    generations, since `vae.encode()`/`KSampler` already process a batched
    latent as N parallel runs, no special-casing needed.

    Feed the outputs straight into a stock KSampler.
    """

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img',
                       'encode image', 'image to latent']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts and init latent for Krea2, no ControlNet. Leave "
                   "images unconnected for txt2img. For ControlNet-guided "
                   "generation use Krea2ControlNetImg2Img instead. Feed the "
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
            logger.info("Krea2: txt2img, empty latent %s", tuple(latent.shape))
        else:
            pixels = comfy.utils.common_upscale(
                images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Krea2: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        return (model, positive, negative, {"samples": latent}, denoise)


class Krea2ControlNetImg2Img:
    """Prompts, init latent, and control image for Krea2 - ControlNet img2img.

    Same as `Krea2Img2Img` plus everything needed for ControlNet-guided
    generation, kept in its own node so the plain img2img node never has to
    show control-related inputs. Two image-shaped slots, dead simple:
    `images` for img2img, `control_image` for ControlNet. Leave `images`
    unconnected for txt2img. Leave `control_image` unconnected for plain
    img2img/txt2img with no LoRA-guided control - if no Control LoRA is
    loaded, a connected control_image is simply ignored (with a warning),
    since there's nothing to attach it to. If a Krea2 Control LoRA IS
    loaded (via Krea2ControlLoRALoader) and neither control_image nor a
    usable images is available, this raises instead of silently sampling a
    half-configured model, the same guarantee the original pack's separate
    Apply node existed for.

    `images` is real img2img: VAE-encoded, then partially denoised at
    `strength`. It's batch-aware - a batch of N photos naturally becomes N
    independent img2img generations, since `vae.encode()`/`KSampler` already
    process a batched latent as N parallel runs, no special-casing needed.

    control_mode/control_image apply ONLY to widened-input-projection
    Control LoRAs (loaded via Krea2ControlLoRALoader - the depth LoRA is
    one of these). Nothing in the LoRA file says what type of control image
    it expects, so control_mode picks how it gets produced:
      "auto_depth" (default) - derive a depth map from `images` automatically
        (Depth Anything V2, same as Krea2DepthMap) - correct for the depth
        Control LoRA, so one photo plugged into `images` is enough.
      "auto_canny" - derive a canny edge map from `images` automatically
        (plain cv2.Canny, no model, no download) - for a canny checkpoint
        that IS a widened-projection Control LoRA (if one exists - most
        "canny Krea2 LoRA" files are an ordinary in-context LoRA instead,
        which this node does not have a separate mechanism for anymore;
        install ComfyUI-Flux-Reference-Tools for that kind of edit-style
        reference conditioning).
      "manual" - do no automatic derivation; control_image must be supplied
        by hand - use this for any widened-projection Control LoRA the two
        auto modes don't cover (pose/lineart/normal).
    Connecting control_image explicitly always overrides auto-derivation,
    in any mode.

    Feed the outputs straight into a stock KSampler.
    """

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 ControlNet img2img ⚡"
    SEARCH_ALIASES = ['image to image', 'img2img', 'text to image', 'txt2img',
                       'encode image', 'image to latent', 'controlnet apply',
                       'controlnet img2img']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "denoise")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and control-image attach for Krea2. "
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
                                               "auto_depth/auto_canny modes) the source photo "
                                               "the control image is derived from. Leave "
                                               "unconnected for txt2img."}),
                "control_mode": (["auto_depth", "auto_canny", "manual", "none"], {"default": "auto_depth",
                    "tooltip": "auto_depth/auto_canny: derive the control image from `images` "
                               "automatically - pick whichever matches the loaded Control LoRA. "
                               "manual: no automatic derivation, connect control_image yourself "
                               "- use this for any Control LoRA the two auto modes don't cover "
                               "(pose/lineart/normal). none: skip control attachment entirely "
                               "even if a Control LoRA is loaded - for toggling control off "
                               "without rewiring or removing the loader."}),
                "depth_ckpt_name": (list(depth_anything_v2.MODEL_CONFIGS.keys()), {
                    "default": "depth_anything_v2_vitb.pth",
                    "tooltip": "auto_depth mode only. Model size for the automatic depth "
                               "estimation. Downloads on first use if not already in "
                               "models/depth_anything_v2."}),
                "control_image": ("IMAGE", {
                    "tooltip": "Manual control map - a depth/canny/pose/etc. map. Overrides "
                               "auto_depth when connected. Required in manual mode."}),
                "control_channel_mode": (["grayscale", "rgb"], {"default": "grayscale",
                    "tooltip": "grayscale for depth; rgb for canny/pose/lineart/normal."}),
                "control_normalize": (["per_image_minmax", "none"], {"default": "per_image_minmax",
                    "tooltip": "per_image_minmax for depth; none for canny/pose/lineart/normal."}),
                "control_invert": ("BOOLEAN", {"default": False,
                    "tooltip": "Flip if the preprocessor's convention is reversed from "
                               "the LoRA's training convention (e.g. depth preview shows "
                               "near objects dark instead of white)."}),
                "control_batch_mode": (["independent_images", "video_frames"], {"default": "independent_images"}),
            },
        }

    def prepare(self, model, clip, vae, prompt, negative_prompt, strength, batch_size,
                width, height, images=None, control_mode="auto_depth",
                depth_ckpt_name="depth_anything_v2_vitb.pth", control_image=None,
                control_channel_mode="grayscale", control_normalize="per_image_minmax",
                control_invert=False, control_batch_mode="independent_images"):
        has_control_lora = model.get_attachment(WRAPPER_KEY) is not None
        if control_image is not None and not has_control_lora:
            # Nothing to attach it to - no widened input projection is
            # installed - so there's no half-configured state to worry
            # about. Ignore it rather than force control_image to be
            # disconnected just to toggle the LoRA loader on/off.
            logger.warning(
                "Krea2ControlNetImg2Img: control_image was given, but model has no Krea2 "
                "Control LoRA loaded - ignoring control_image.")
            control_image = None

        if has_control_lora and control_image is None and control_mode != "none":
            if control_mode == "auto_depth" and images is not None:
                logger.info("Krea2: auto-deriving depth map from images (control_mode=auto_depth)")
                control_image = _depth_anything_batch(images, depth_ckpt_name)
            elif control_mode == "auto_canny" and images is not None:
                logger.info("Krea2: auto-deriving canny edge map from images (control_mode=auto_canny)")
                control_image = _auto_canny_control_image(images)
            elif control_mode in ("auto_depth", "auto_canny"):
                raise ValueError(
                    f"model has a Krea2 Control LoRA loaded and control_mode is {control_mode}, "
                    "but no images were given to derive a control image from. Connect images, or "
                    "connect control_image directly, or remove Krea2ControlLoRALoader.")
            else:
                raise ValueError(
                    "model has a Krea2 Control LoRA loaded and control_mode is manual, but "
                    "no control_image was given. Sampling would fail with a missing control "
                    "latent - connect control_image, switch control_mode to auto_depth/"
                    "auto_canny, or remove Krea2ControlLoRALoader.")
        elif has_control_lora and control_mode == "none":
            logger.info("Krea2: control_mode=none - skipping control attachment even though "
                        "a Control LoRA is loaded.")

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
            logger.info("Krea2: txt2img, empty latent %s", tuple(latent.shape))
        else:
            pixels = comfy.utils.common_upscale(
                images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            latent = vae.encode(pixels[:, :, :, :3])
            if batch_size > 1:
                latent = latent.repeat(batch_size, *([1] * (latent.dim() - 1)))
            denoise = strength
            logger.info("Krea2: img2img, latent %s, strength %.2f", tuple(latent.shape), strength)

        if control_image is not None:
            prepped = _prepare_control_image(control_image, "rgb", "none", False)
            prepped = _resize_image(prepped, width, height)
            prepped = _prepare_control_image(prepped, control_channel_mode, control_normalize, control_invert)
            control_samples = _encode_control_image(vae, prepped, control_batch_mode)

            model = model.clone()
            control_samples = _process_control_latent_for_model(model, control_samples)
            topts = model.model_options.setdefault("transformer_options", {})
            topts[CONTROL_LATENT_KEY] = control_samples
            logger.info("Krea2: control latent attached %s", tuple(control_samples.shape))

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))

        return (model, positive, negative, {"samples": latent}, denoise)


class Krea2KSampler:
    """KSampler for Krea2, with the one option stock KSampler lacks.

    `denoise_mode`:
      comfy      - delegates to common_ksampler, so it stays identical to
                   KSampler as ComfyUI changes.
      diffusers  - slices a steps-long schedule at
                   t_start = steps - round(steps * denoise), reproducing a
                   diffusers img2img pipeline step for step.

    Krea2 has no bespoke sampling code in comfy at all - it shares
    ModelSamplingFlux (shift=1.15, the same value copy-pasted alongside
    the Qwen-Image-family config) with the rest of the Flux family, and
    comfy's own denoise-slicing convention (KSampler.set_steps:
    new_steps=int(steps/denoise), take the tail) genuinely diverges from
    the diffusers img2img convention used above - confirmed by recomputing
    both schedules for Krea2's actual shift value: at 9 steps, denoise 0.9
    starts at sigma ~0.9660 under comfy vs ~0.9619 under diffusers.
    Smaller than Z-Image's measured gap (0.9643 vs 0.9567) but the same
    mechanism, so this is a compatibility switch for matching another
    pipeline, not a quality setting - "comfy" (default) is unchanged
    stock behavior.
    """

    TITLE = "Krea2 KSampler ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'txt2img', 'img2img']
    CATEGORY = KREA2_CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("KSampler for Krea2 with an optional diffusers-style "
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

        logger.info("Krea2 KSampler (diffusers): denoise %.2f -> %d of %d steps, "
                    "start sigma %.4f", denoise, len(sigmas) - 1, steps, float(sigmas[0]))
        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = out_samples
        return (out,)


# ── Krea2 Identity Edit (ported from comfyui-krea2edit, Apache-2.0,
# lbouaraba) ─────────────────────────────────────────────────────────────

def _imgids(bs, frame, h_, w_, device):
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _imgids_offset(bs, frame, gh, gw, th, tw, device):
    """Stride-1 integer positions at a centered integer offset. For `fit` refs the
    pixels are already resampled to target grid density, so the position grid is
    stride-1 BY CONSTRUCTION - scaling it again only manufactures skip/collision
    artifacts. Requires gh<=th, gw<=tw (guaranteed by the floor+cap in fit)."""
    off_h, off_w = max(0.0, (th - gh) / 2), max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def _to_4d(v):
    """(B,C,T,H,W) -> (B*T,C,H,W); pass 4D through. Images use T=1."""
    if v.ndim == 5:
        b, c, t, h, w = v.shape
        return v.reshape(b * t, c, h, w)
    return v


def _fit_src(src, H, W):
    """Fit a source latent to the target grid the way TRAINING did: center-crop to
    the target aspect ratio, then resize. A plain interpolate stretches mixed-AR
    sources - visible as stretched people whenever input AR differs from output."""
    sh, sw = src.shape[-2:]
    if (sh, sw) == (H, W):
        return src
    s = max(H / sh, W / sw)
    ch, cw = min(sh, int(round(H / s))), min(sw, int(round(W / s)))
    y0, x0 = (sh - ch) // 2, (sw - cw) // 2
    src = src[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(src.float(), size=(H, W), mode="bilinear")


def _fit_encode_image(image, vae, H, W, cache, key, fit_mode="crop"):
    """Pixel-space source prep (blur-proof path): center-crop the IMAGE to the
    target AR, resize to the exact target pixel grid, VAE-encode. Latent-space
    resizing (the old fallback) softens VAE latents - this path never resizes
    latents at all. Cached per target resolution (encode once, not per step)."""
    key = key + (fit_mode,)
    if key in cache:
        return cache[key]
    logger.info("Krea2Edit: _fit_encode_image mode=%s in=%s target_latent=%dx%d",
                fit_mode, tuple(image.shape), H, W)
    px_h, px_w = H * 8, W * 8
    img = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
    ih, iw = img.shape[-2:]
    if fit_mode == "fit":
        # "bilinear" answer to scale mismatch: resample CONTENT (pixel space, bicubic)
        # to the target's grid density instead of moving positions. AR-preserving
        # fit-inside, no crop, no grey canvas - the forward places it at an integer
        # centered offset (scaled-pos with s=1 -> stride 1, no rounding artifacts).
        sc = min(px_h / ih, px_w / iw)
        # NEAR-MATCHED AR: fill the target grid EXACTLY via a minimal center-crop.
        # Fit-inside margins of 1-2 tokens are not harmless: target edge columns
        # with no ref correspondence get filled by repeating adjacent ref content.
        # This also restores the design promise fit == crop at matched AR.
        CROP_TOL = 0.08
        if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):
            s = max(px_h / ih, px_w / iw)
            ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            img = img[..., y0:y0 + ch, x0:x0 + cw]
            nh, nw = px_h, px_w
        else:
            # genuine AR mismatch: MUST match the trainer's _fit_prep EXACTLY
            # (/16 floor snap capped at the target's /16 floor) - the model is
            # trained on this geometry; a different node grid would produce a
            # different ref latent size -> different centered offset -> a
            # visible margin-boundary seam even from a well-trained model.
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            # Center-crop the source so the fitted axis lands on the /16 grid at
            # scale sc EXACTLY: zero squash, stride-1 stays true.
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            img = img[..., y0:y0 + ch2, x0:x0 + cw2]
        img = F.interpolate(img.float(), size=(nh, nw), mode="bicubic", antialias=True)
        lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
        cache[key] = lat
        return lat
    # crop (default / "v1 legacy"): center-crop to the target AR, then resize.
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    img = img[..., y0:y0 + ch, x0:x0 + cw]
    img = F.interpolate(img.float(), size=(px_h, px_w), mode="bicubic", antialias=True)
    lat = vae.encode(img.movedim(1, -1)[..., :3].clamp(0, 1))
    cache[key] = lat
    return lat


def _ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
    """Additive attention-logit bias on the [text | refs... | target] sequence.

    boosts: per-ref factor on target->ref attention, aligned with the source blocks
    (last entry = last ref = the subject by workflow convention). Equivalent to
    multiplying those keys' post-softmax attention weight before renormalization.
    boost_mask (ComfyUI MASK, ref-image pixel space) restricts the LAST ref's boost
    to a region (e.g. the face).
    """
    nsrc = len(slens)
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    L = rows0 + tgtlen
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b == 1.0:
            continue
        off, sl = offs[i], slens[i]
        if boost_mask is not None and i == nsrc - 1 and mask_hw is not None:
            mask = boost_mask[:1]
            if mask.ndim == 2:
                mask = mask[None]
            mask = F.interpolate(mask[None].float(), size=mask_hw[i], mode="area")[0, 0]
            cols = off + torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=True)[0].to(device)
        else:
            cols = torch.arange(off, off + sl, device=device)
        bias[:, :, rows0:, cols] = math.log(max(b, 1e-4))
    return bias


def krea2_edit_forward(m, x, timesteps, context, src_latent, transformer_options,
                       ref_boost=1.0, ref_boost_a=1.0, ref_boost_mask=None,
                       ref_native=False, pos_mode="anchor"):
    """Krea2 SingleStreamDiT._forward, but with source block(s) prepended.

    m           : the SingleStreamDiT (LoRA-patched at sample time)
    x           : (B,C,H,W) or (B,C,T,H,W) noisy TARGET latent
    src_latent  : clean SOURCE latent (VAE-encoded), 4D/5D - or a LIST of them
                  (multi-ref: [scene, subject], frames 1..N, training-matched)
    context     : (B, seq, txtlayers*txtdim) - the 12-layer Qwen3-VL stack
    """
    patch = m.patch

    # Mirror ComfyUI _forward: latents may arrive 5D (B,C,T,H,W) for this model.
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
    x = _to_4d(x)
    bs, c, H_orig, W_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch), padding_mode="replicate")
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    # source(s) -> (bs, C, H, W): flatten temporal, match batch, fit to the target grid
    # (center-crop to target AR then resize - training-matched; never stretch).
    src_list = src_latent if isinstance(src_latent, (list, tuple)) else [src_latent]
    srcs = []
    for sl in src_list:
        src = _to_4d(sl).to(x.device, x.dtype)
        if src.shape[0] != bs:
            src = src[:1].expand(bs, *src.shape[1:])
        if not ref_native and src.shape[-2:] != (H, W):
            logger.info("Krea2Edit: LATENT-PATH fit_src (crop): src=%s -> %dx%d",
                        tuple(src.shape[-2:]), H, W)
            src = _fit_src(src, H, W).to(x.dtype)
        srcs.append(comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch), padding_mode="replicate"))
    src_grids = [(s_.shape[-2] // patch, s_.shape[-1] // patch) for s_ in srcs]

    context = m._unpack_context(context)                       # (B, seq, 12, 2560)

    tgt_img = m.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    src_imgs = [m.first(rearrange(s_, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
                for s_ in srcs]

    t = m.tmlp(timestep_embedding(timesteps, m.tdim).unsqueeze(1).to(tgt_img.dtype))
    tvec = m.tproj(t)

    context = m.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = m.txtmlp(context)

    txtlen, tgtlen = context.shape[1], tgt_img.shape[1]
    srclen = sum(si.shape[1] for si in src_imgs)
    combined = torch.cat([context] + src_imgs + [tgt_img], dim=1)  # [text | refs... | target]

    device = combined.device
    if pos_mode == "stride1" and ref_native:
        logger.info("Krea2Edit: STRIDE1-POS fit: ref grids %s centered in (%d,%d)", src_grids, h_, w_)
        ref_ids = [_imgids_offset(bs, i + 1, gh, gw, h_, w_, device)
                   for i, (gh, gw) in enumerate(src_grids)]
    else:
        ref_ids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(src_grids)]
    pos = torch.cat([
        torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)]   # text @ 0
        + ref_ids
        + [_imgids(bs, 0, h_, w_, device)],                                    # target frame=0
        dim=1)
    freqs = m.pe_embedder(pos)

    attn_bias = None
    if ref_boost != 1.0 or ref_boost_a != 1.0:
        # last ref = subject (single-ref: the only ref); earlier refs (scene) get ref_boost_a
        boosts = [ref_boost_a] * (len(src_imgs) - 1) + [ref_boost]
        attn_bias = _ref_attn_bias(boosts, ref_boost_mask, txtlen,
                                   [si.shape[1] for si in src_imgs], tgtlen,
                                   src_grids, combined.device, combined.dtype)

    for block in m.blocks:
        combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

    final = m.last(combined, t)
    out = final[:, txtlen + srclen: txtlen + srclen + tgtlen, :]         # target tokens only
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=m.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, m.channels, H_orig, W_orig).movedim(1, 2)
    return out


class Krea2EditModelPatch:
    """Adds krea2_edit's in-context source-preservation path to a Krea2 model.

    Ported from comfyui-krea2edit's Krea2EditModelPatch (Apache-2.0, lbouaraba).
    Wraps the diffusion model's forward (comfy.patcher_extension.WrappersMP.
    DIFFUSION_MODEL - a real comfy ModelPatcher extension point, not a custom
    sampler) so the VAE-encoded source image is prepended as clean in-context
    tokens at RoPE frame 1 (the target stays frame 0) - reproducing exactly how
    the Krea 2 Identity Edit LoRA (krea2_identity_edit_v1_2.safetensors,
    licensed separately under Krea AI's own Community License Agreement - not
    bundled here) was trained. Comfy's native Krea2 forward only ever builds
    `[text | target]`; this pack has no other mechanism (not Krea2Img2Img's
    control_image path, not reference_latents) that can drive this LoRA
    correctly.

    Wiring: LoadImage -> VAEEncode -> source_latent (or vae+source_image for
    the blur-proof pixel-space path) -> this node's model output -> KSampler.
    Pair with Krea2EditGroundedEncode for positive/negative conditioning -
    both are required for correct results (see that class's docstring).
    """

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 Edit (source patch) ⚡"
    SEARCH_ALIASES = ['krea2 edit', 'identity edit', 'image edit', 'instruction edit',
                       'in-context edit', 'source preservation']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Adds the krea2_edit in-context source-preservation path "
                   "(source latent as frame=1 tokens) to a Krea2 model.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "source_latent": ("LATENT",),
            },
            "optional": {
                "source_latent_b": ("LATENT", {"tooltip": "2nd reference (subject photo) for "
                                               "multi-ref LoRAs -> RoPE frame=2, training-"
                                               "matched order: scene first, subject second."}),
                "ref_boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                                        "round": 0.001,
                                        "tooltip": "Reference-fidelity dial: multiplies target->"
                                                   "reference attention. Applies to the LAST ref "
                                                   "(= the subject in two-ref workflows, the only "
                                                   "ref in single-ref). 1.0 = off, >1 pulls harder "
                                                   "toward the reference's appearance, <1 loosens. "
                                                   "Optimal value is model-specific."}),
                "ref_boost_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                                          "round": 0.001,
                                          "tooltip": "Same dial for the FIRST ref (= the scene in "
                                                     "two-ref workflows). No effect in single-ref "
                                                     "workflows. 1.0 = off."}),
                "fit_mode": (["fit", "crop (legacy)"], {"default": "fit",
                             "tooltip": "How a source fits a mismatched output aspect ratio "
                                        "(needs vae + source_image connected): fit = resample "
                                        "the source to the target grid at a centered offset - "
                                        "matches how this model was trained (default, use this). "
                                        "crop (legacy) = center-crop to the target AR then "
                                        "resize (v1/v1.1 geometry, only for older weights)."}),
                "ref_boost_mask": ("MASK", {"tooltip": "Optional region on the (last) reference "
                                            "to boost, e.g. the face; empty = whole reference."}),
                "vae": ("VAE", {"tooltip": "RECOMMENDED with source_image: enables the blur-"
                                "proof pixel-space path (crop+resize in pixels, encode "
                                "internally) - immune to input/output resolution mismatches."}),
                "source_image": ("IMAGE", {"tooltip": "Source as IMAGE (with vae connected): "
                                 "overrides source_latent with exact pixel-space fitting - "
                                 "fixes blurry results from mismatched resolutions."}),
                "source_image_b": ("IMAGE", {"tooltip": "2nd reference as IMAGE (with vae)."}),
                "target_latent": ("LATENT", {"tooltip": "RECOMMENDED with vae + source_image: "
                                  "wire the SAME latent you feed KSampler.latent_image. Lets "
                                  "the node VAE-encode the source here, before sampling "
                                  "starts, instead of on the first step - otherwise the VAE "
                                  "is pulled onto the GPU mid-sampling and can evict part of "
                                  "the diffusion model, slowing every remaining step on "
                                  "VRAM-tight setups."}),
            },
        }

    def patch(self, model, source_latent, source_latent_b=None, ref_boost=1.0, ref_boost_a=1.0,
              ref_boost_mask=None, vae=None, source_image=None,
              source_image_b=None, fit_mode="fit", target_latent=None):
        m = model.clone()
        # The target latent reaches the diffusion model already scaled (process_latent_in);
        # scale the source(s) the same way so all share one latent space.
        src_samples = model.model.process_latent_in(source_latent["samples"])
        if source_latent_b is not None:
            src_samples = [src_samples, model.model.process_latent_in(source_latent_b["samples"])]

        px_cache = {}   # pixel-path encoded sources, keyed per target resolution
        mm = model.model  # for process_latent_in on the pixel path
        state = {"announced": False}
        fit_mode_key = "fit" if fit_mode == "fit" else "crop"

        if fit_mode_key == "fit" and (vae is None or source_image is None):
            logger.warning("Krea2Edit: fit_mode='fit' has NO EFFECT - it needs both "
                           "'vae' and 'source_image' connected (the pixel path). "
                           "Falling back to the latent crop path.")

        # Pre-encode OUTSIDE the sampling window. vae.encode -> load_models_gpu ->
        # free_memory(keep_loaded=[]), which partially unloads whatever is resident -
        # including the diffusion model, if the first call lands inside the sampler.
        # Nothing re-expands it (sampler_helpers loads once, before the loop), so the
        # rest of the run streams weights from CPU every step. Running the encode here,
        # at node-execution time, restores the ordinary VAEEncode -> KSampler order
        # where the sampler evicts the VAE instead of the reverse.
        primed = None
        if vae is not None and source_image is not None:
            if target_latent is not None:
                Hh, Ww = target_latent["samples"].shape[-2], target_latent["samples"].shape[-1]
                logger.info("Krea2Edit: pre-encoding sources at target %dx%dpx "
                           "(before sampling, fit_mode=%s)", Hh * 8, Ww * 8, fit_mode_key)
                _fit_encode_image(source_image, vae, Hh, Ww, px_cache, ("a", Hh, Ww), fit_mode_key)
                if source_image_b is not None:
                    _fit_encode_image(source_image_b, vae, Hh, Ww, px_cache, ("b", Hh, Ww), fit_mode_key)
                primed = (Hh, Ww)
            else:
                logger.info("Krea2Edit: connect 'target_latent' (the same latent that feeds "
                           "KSampler.latent_image) to pre-encode the source here instead of "
                           "on the first sampling step. Without it the VAE is loaded mid-"
                           "sampling and can evict part of the diffusion model, slowing "
                           "every remaining step.")

        def wrapper(executor, x, timesteps, context, *wargs, **kwargs):
            # ComfyUI's SingleStreamDiT.forward signature (comfy/ldm/krea2/model.py):
            #   execute(x, t, ctx, attention_mask, ref_latents, transformer_options, **kwargs)
            # Accept extra trailing positionals defensively in case that signature
            # drifts again - transformer_options is always the trailing dict.
            transformer_options = kwargs.pop("transformer_options", None)
            if transformer_options is None:
                transformer_options = {}
                for a in reversed(wargs):
                    if isinstance(a, dict):
                        transformer_options = a
                        break
            dm = executor.class_obj  # the SingleStreamDiT instance
            src = src_samples
            if vae is not None and source_image is not None:
                xx = _to_4d(x)
                Hh, Ww = xx.shape[-2], xx.shape[-1]
                # not `if not px_cache` - the cache is already primed when target_latent
                # is wired, so an explicit one-shot flag is needed to announce once.
                if not state["announced"]:
                    state["announced"] = True
                    logger.info("Krea2Edit: pixel path ACTIVE (fit_mode=%s)", fit_mode_key)
                    if primed is not None and primed != (Hh, Ww):
                        logger.warning("Krea2Edit: 'target_latent' is %dx%dpx but sampling "
                                       "is at %dx%dpx - the pre-encode is unused and the VAE "
                                       "will run mid-sampling. Wire the SAME latent that "
                                       "feeds KSampler.", primed[0] * 8, primed[1] * 8, Hh * 8, Ww * 8)
                lat = mm.process_latent_in(_fit_encode_image(source_image, vae, Hh, Ww, px_cache, ("a", Hh, Ww), fit_mode_key))
                if source_image_b is not None:
                    lat = [lat, mm.process_latent_in(_fit_encode_image(source_image_b, vae, Hh, Ww, px_cache, ("b", Hh, Ww), fit_mode_key))]
                src = lat
            v = krea2_edit_forward(dm, x, timesteps, context, src, transformer_options,
                                   ref_boost=ref_boost, ref_boost_a=ref_boost_a,
                                   ref_boost_mask=ref_boost_mask,
                                   ref_native=(fit_mode_key == "fit" and vae is not None
                                               and source_image is not None),
                                   pos_mode=("stride1" if fit_mode_key == "fit" else "anchor"))
            return v

        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "krea2_edit", wrapper)
        return (m,)


class Krea2EditGroundedEncode:
    """Image-grounded instruction encode - the SEMANTIC path of krea2_edit.

    Ported from comfyui-krea2edit's Krea2EditGroundedEncode (Apache-2.0,
    lbouaraba). Training always encodes the instruction TOGETHER with the
    source image through Qwen3-VL (user turn = <vision tokens: source> +
    instruction) and taps 12 layers. Stock CLIPTextEncode is text-only, so
    encoding without this node runs with the grounding half of the recipe
    missing (Krea2EditModelPatch's VAE source tokens carry appearance; THIS
    carries scene semantics: "the man on the left", "the sign in the back").
    Both nodes are required for correct results with the Krea 2 Identity
    Edit LoRA.

    Requires a krea2-type CLIP checkpoint WITH the vision tower (confirmed
    via clip.tokenize(text, images=[...], llama_template=...) - real kwargs
    on comfy's own Krea2Tokenizer, comfy/text_encoders/krea2.py).
    grounding_px caps the longest side fed to the VLM - the v1.2 LoRA was
    trained with 384-768px jitter, so that range is in-distribution; 0 =
    native resolution. For CFG > 1, ground the NEGATIVE too: a second
    instance of this node, empty prompt, the same image (matches training's
    unconditional).
    """

    DEFAULT_SYSTEM = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )

    CATEGORY = KREA2_CATEGORY
    TITLE = "Krea2 Edit (grounded encode) ⚡"
    SEARCH_ALIASES = ['krea2 edit', 'identity edit', 'grounded encode', 'image grounded prompt',
                       'text prompt', 'positive prompt', 'negative prompt']
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    DESCRIPTION = ("Encodes the edit instruction grounded on the source image "
                   "(training-matched semantic path for the Krea 2 Identity Edit LoRA).")

    @classmethod
    def _template(cls, nimg, system_prompt=""):
        sp = system_prompt.strip() or cls.DEFAULT_SYSTEM
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * nimg
        return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
                + vis + "{}<|im_end|>\n<|im_start|>assistant\n")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "The source image being edited - encoded "
                                    "together with the prompt. Leave unconnected to fall "
                                    "back to plain text-only encoding (stock CLIPTextEncode "
                                    "behavior)."}),
                "image_b": ("IMAGE", {"tooltip": "2nd reference (subject) for multi-ref "
                                      "LoRAs; vision blocks in training order: scene, "
                                      "subject."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                 "tooltip": "Cap longest side fed to Qwen3-VL; 0 = native. "
                                            "Quality dial: lower = stronger edit adherence, "
                                            "higher = stronger identity/likeness."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "",
                                  "tooltip": "Advanced (optional): override the grounding "
                                             "system prompt (empty = training default). "
                                             "Steers what the vision encoder attends to, "
                                             "e.g. facial identity detail."}),
            },
        }

    def _prep(self, image, grounding_px):
        samples = image.movedim(-1, 1)  # B,H,W,C -> B,C,H,W
        h, w = samples.shape[2], samples.shape[3]
        if grounding_px and max(h, w) > grounding_px:
            s = grounding_px / max(h, w)
            samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        return samples.movedim(1, -1)[:, :, :, :3]

    def encode(self, clip, prompt, image=None, image_b=None, grounding_px=768, system_prompt=""):
        if image is None:  # text-only fallback = old behavior
            tokens = clip.tokenize(prompt)
            return (clip.encode_from_tokens_scheduled(tokens),)
        imgs = [self._prep(image, grounding_px)]
        if image_b is not None:
            imgs.append(self._prep(image_b, grounding_px))
        template = self._template(len(imgs), system_prompt)
        tokens = clip.tokenize(prompt, images=imgs, llama_template=template)
        return (clip.encode_from_tokens_scheduled(tokens),)


NODE_CLASS_MAPPINGS = {
    "Krea2ModelLoader": Krea2ModelLoader,
    "Krea2ControlLoRALoader": Krea2ControlLoRALoader,
    # Backward-compat alias: this node's logic moved to the shared
    # nodes/preprocessors.py DepthMap class - old saved workflows using the
    # "Krea2DepthMap" type id keep resolving to a working node.
    "Krea2DepthMap": DepthMap,
    "Krea2Img2Img": Krea2Img2Img,
    "Krea2ControlNetImg2Img": Krea2ControlNetImg2Img,
    "Krea2KSampler": Krea2KSampler,
    "Krea2EditModelPatch": Krea2EditModelPatch,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2ModelLoader": Krea2ModelLoader.TITLE,
    "Krea2ControlLoRALoader": Krea2ControlLoRALoader.TITLE,
    "Krea2DepthMap": "Krea2 Depth Map ⚡",
    "Krea2Img2Img": Krea2Img2Img.TITLE,
    "Krea2ControlNetImg2Img": Krea2ControlNetImg2Img.TITLE,
    "Krea2EditModelPatch": Krea2EditModelPatch.TITLE,
    "Krea2EditGroundedEncode": Krea2EditGroundedEncode.TITLE,
    "Krea2KSampler": Krea2KSampler.TITLE,
}
