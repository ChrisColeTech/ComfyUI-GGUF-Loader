# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import os
import re
import torch
import logging
import inspect
import collections

import nodes
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths

from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader, validate_te_type, is_vision_projector
from .dequant import is_quantized, is_torch_compatible

def update_folder_names_and_paths(key, targets=[]):
    # check for existing key
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    # find base key & add w/ fallback, sanity check + warning
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")

# Add a custom keys for files ending in .gguf
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = comfy.lora.calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)


    def pin_weight_to_device(self, key):
        op_key = key.rsplit('.', 1)[0]
        if not self.mmap_released and op_key in self.named_modules_to_munmap:
            # TODO: possible to OOM, find better way to detach
            self.named_modules_to_munmap[op_key].to(self.load_device).to(self.offload_device)
            del self.named_modules_to_munmap[op_key]
        super().pin_weight_to_device(key)

    mmap_released = False
    named_modules_to_munmap = {}

    # PR #469: partial load/unload must also force weight patching so quantized
    # mmap tensors are not left half-applied.
    def partially_unload(self, *args, force_patch_weights=False, **kwargs):
        return super().partially_unload(*args, force_patch_weights=True, **kwargs)

    def partially_load(self, *args, force_patch_weights=False, **kwargs):
        return super().partially_load(*args, force_patch_weights=True, **kwargs)

    def load(self, *args, force_patch_weights=False, **kwargs):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(self.model.named_modules())

        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.named_modules_to_munmap.items():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked and self.load_device != self.offload_device:
                logging.info(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        n.mmap_released = getattr(self, "mmap_released", False)
        if src_cls != GGUFModelPatcher:
            n.size = 0 # force recalc
        return n

_LTX_TRANSFORMER_KEY_MAP = {
    # LTX's own transformer-config names -> comfy LTXV/LTXAV kwargs. The raw
    # names land in **kwargs and are silently ignored, which builds the wrong
    # architecture variant (6-param adaln instead of LTX-2.5's 9).
    "cross_attn_mod": "cross_attention_adaln",
    "gated_attn": "apply_gated_attention",
    "rope_theta": "positional_embedding_theta",
    "cross_attn_timestep_scale_multiplier": "av_ca_timestep_scale_multiplier",
}

def _translate_ltx_transformer_cfg(t):
    out = dict(t)
    for src, dst in _LTX_TRANSFORMER_KEY_MAP.items():
        if src in out:
            out[dst] = out.pop(src)
    out.pop("audio_cross_attn_mod", None)  # one flag drives both streams
    if "pos_embed_max_pos" in out:
        v = out.pop("pos_embed_max_pos")
        out["positional_embedding_max_pos"] = [
            v, out.pop("base_height", 2048), out.pop("base_width", 2048)]
    if "audio_pos_embed_max_pos" in out:
        out["audio_positional_embedding_max_pos"] = [out.pop("audio_pos_embed_max_pos")]
    # LTX derives the connector width from the main attention geometry
    # (defaults 30 heads x 128 = 3840; LTX-2.5 is 32 x 128 = 4096).
    if "num_attention_heads" in out and "connector_num_attention_heads" not in out:
        out["connector_num_attention_heads"] = out["num_attention_heads"]
    if "audio_num_attention_heads" in out and "audio_connector_num_attention_heads" not in out:
        out["audio_connector_num_attention_heads"] = out["audio_num_attention_heads"]
    if "audio_attention_head_dim" in out and "audio_connector_attention_head_dim" not in out:
        out["audio_connector_attention_head_dim"] = out["audio_attention_head_dim"]
    return out

# Tokens that say nothing about which checkpoint a sidecar belongs to, so
# they must not be what makes a folder-level sidecar look like a match.
_SIDECAR_GENERIC_TOKENS = frozenset((
    "metadata", "config", "model", "models", "transformer", "dit", "unet",
    "diffusion", "weights", "fp8", "fp16", "bf16", "f16", "f32", "int8",
    "gguf", "safetensors", "k", "m", "s", "l", "xl", "0", "1",
))


def _name_tokens(stem):
    """Distinctive lowercase tokens of a filename stem, quant tags dropped."""
    parts = re.split(r"[^0-9a-zA-Z]+", stem.lower())
    tokens = set()
    for part in parts:
        if not part or part in _SIDECAR_GENERIC_TOKENS:
            continue
        if re.fullmatch(r"q\d(_[km])?", part):  # q6, q4_k, q8 ...
            continue
        if len(part) < 3 and part.isdigit():  # version fragments: "2", "5"
            continue
        tokens.add(part)
    return tokens


def _sidecar_claims(sidecar_path, unet_path):
    """Does a folder-level sidecar actually belong to this checkpoint?

    A sidecar may say so outright with an ``applies_to`` list of filename
    globs. Failing that, fall back to sharing a distinctive filename token —
    ``ltx-2.5-transformer-metadata.json`` claims
    ``ltx-2.5-22b-distilled-transformer-Q6_K.gguf`` but not
    ``minimax_h3_ref2va_turbo_Q6_K.gguf``. Sidecars whose own name carries no
    distinctive token (a bare ``metadata.json``) are taken at face value,
    since there is nothing to check them against.
    """
    import fnmatch
    import json
    unet_name = os.path.basename(unet_path)
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            patterns = json.load(f).get("applies_to")
    except (OSError, ValueError):
        patterns = None
    if isinstance(patterns, str):
        patterns = [patterns]
    if isinstance(patterns, (list, tuple)) and patterns:
        return any(fnmatch.fnmatch(unet_name.lower(), str(p).lower()) for p in patterns)

    sidecar_tokens = _name_tokens(os.path.splitext(os.path.basename(sidecar_path))[0])
    if not sidecar_tokens:
        return True
    return bool(sidecar_tokens & _name_tokens(os.path.splitext(unet_name)[0]))


def _unet_metadata_sidecar(unet_path, extra_metadata, sd=None):
    """Merge a transformer-config JSON sidecar into the GGUF metadata.

    Split checkpoints (e.g. LTX-2.5) ship the model config as a JSON sidecar
    next to the weights — safetensors carries it in file metadata, GGUF
    cannot. Without it, detection builds the wrong architecture variant
    (LTX-2 tables instead of LTX-2.5's 9-parameter adaln) and every
    scale_shift_table copy fails with a size mismatch.

    A sidecar named after the checkpoint is taken as-is. A folder-level one
    has to earn it: ``models/diffusion_models`` is a shared drawer, and a
    config merged onto the wrong GGUF builds a plausible-looking model with
    the wrong block count and head geometry, which then dies deep in the
    forward rather than at load.
    """
    import json
    candidates = [unet_path + s for s in ("-metadata.json", ".metadata.json")
                  if os.path.isfile(unet_path + s)]
    if not candidates:
        folder = os.path.dirname(unet_path)
        folder_hits = [os.path.join(folder, f) for f in os.listdir(folder)
                       if "metadata" in f.lower() and f.lower().endswith(".json")]
        claimed = [f for f in folder_hits if _sidecar_claims(f, unet_path)]
        # Use a folder-level sidecar only when exactly one claims this file.
        if len(claimed) == 1:
            candidates = claimed
        elif len(claimed) > 1:
            logging.warning(
                "Found %d metadata sidecars claiming %s; none named after it, "
                "skipping them all.", len(claimed), os.path.basename(unet_path))
        elif folder_hits:
            logging.debug(
                "Ignoring %d metadata sidecar(s) next to %s: none name it or "
                "list it in applies_to.", len(folder_hits), os.path.basename(unet_path))
    metadata = dict(extra_metadata)
    for cand in candidates:
        with open(cand, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar.pop("applies_to", None)
        config = sidecar.get("config")
        if isinstance(config, str):
            config = json.loads(config)
        if isinstance(config, dict):
            config = dict(config)
            if "transformer" in config and isinstance(config["transformer"], dict):
                config["transformer"] = _translate_ltx_transformer_cfg(config["transformer"])
            sidecar["config"] = json.dumps(config)
        sidecar.pop("notes", None)
        metadata.update(sidecar)
        logging.info("Using UNet metadata sidecar: %s", cand)

    # Ground truth from the weights themselves: a 9-row scale_shift_table is
    # the LTX-2/2.5 cross-attn-adaln layout, whatever the config called it.
    if sd is not None and "config" in metadata:
        table = sd.get("transformer_blocks.0.scale_shift_table")
        force = {}
        if table is not None and getattr(table, "shape", None) is not None:
            if len(table.shape) >= 1 and table.shape[0] == 9:
                force["cross_attention_adaln"] = True
        # connectors with to_gate_logits tensors are gated-attention connectors;
        # built without the flag, the gate weights load as "unexpected" and are
        # silently dropped.
        gate = "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight"
        if sd.get(gate) is not None:
            force["connector_apply_gated_attention"] = True
        # A checkpoint with embeddings connectors but no caption_projection
        # weights consumes pre-projected context (the caption projection lives
        # in the text encoder, e.g. ltx-v2-projections): split by the connector
        # dims and make the absent projections identities instead of missing
        # weights. Otherwise the forward splits by caption_channels and dies
        # on a 6144-dim context.
        if (not force.get("caption_proj_before_connector")
                and "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight" in sd
                and not any(k.startswith("caption_projection.") for k in sd)):
            force["caption_proj_before_connector"] = True
            force["caption_projection_first_linear"] = False
        if force:
            config = json.loads(metadata["config"])
            transformer = config.setdefault("transformer", {})
            changed = [k for k, v in force.items() if not transformer.get(k)]
            transformer.update({k: v for k, v in force.items() if not transformer.get(k)})
            metadata["config"] = json.dumps(config)
            if changed:
                logging.info("LTX: forced %s from the weights", ", ".join(changed))
    return metadata

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "UNET Loader (GGUF) ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'gguf', 'unet loader', 'diffusion model loader', 'quantized model']

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd, extra = gguf_sd_loader(unet_path)

        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = _unet_metadata_sidecar(
                unet_path, extra.get("metadata", {}), sd)

        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}, **kwargs,
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)

class UnetLoaderGGUFAdvanced(UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }
    TITLE = "UNET Loader (GGUF/Advanced) ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'gguf', 'unet loader', 'diffusion model loader', 'quantized model', 'advanced']

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": _merged_type_options(),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "CLIP Loader (GGUF) ⚡"
    SEARCH_ALIASES = ['load clip', 'text encoder', 'clip loader', 'gguf', 'quantized clip']

    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list("clip")
        files += folder_paths.get_filename_list("clip_gguf")
        return sorted(files)

    @staticmethod
    def _resolve_clip_path(clip_name):
        """Resolve a file listed by get_filename_list to its full path.

        The listing merges `clip` and `clip_gguf`, so resolution has to try
        both — files living only in clip_gguf would otherwise 404 on load.
        """
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if clip_path is None and clip_name.endswith(".gguf"):
            clip_path = folder_paths.get_full_path("clip_gguf", clip_name)
        return clip_path

    def load_data(self, ckpt_paths):
        clip_data = []
        vision_data = []
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                # An mmproj is the vision tower of a VL text encoder, not a text
                # encoder in its own right. Handing it to comfy as a second
                # state dict makes TE detection fall through to SDXLClipModel,
                # whose clip_g then never gets loaded at all. Merge it into the
                # TE instead - the same thing the sibling auto-detection in
                # gguf_clip_loader() does when the mmproj isn't picked by hand.
                if is_vision_projector(p):
                    vision_data.append((p, gguf_clip_loader(p)))
                    continue
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)

        for p, vsd in vision_data:
            if not clip_data:
                raise ValueError(
                    f"'{os.path.basename(p)}' is an mmproj vision tower, not a text"
                    f" encoder - it can only accompany the VL text encoder it belongs"
                    f" to. Select that text encoder GGUF as well (it is picked up"
                    f" automatically when both live in the same folder)."
                )
            # Prefer the VL text encoder if several are loaded; explicit picks
            # override the keys auto-detection already merged in.
            target = next((sd for sd in clip_data if any(k.startswith("visual.") for k in sd)), clip_data[0])
            logging.info(f"Merging mmproj '{os.path.basename(p)}' into the text encoder it belongs to.")
            target.update(vsd)
        return clip_data

    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type = clip_type,
            state_dicts = clip_data,
            model_options = {
                "custom_operations": GGMLOps,
                "initial_device": comfy.model_management.text_encoder_offload_device()
            },
            embedding_directory = folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip

    def load_clip(self, clip_name, type="stable_diffusion"):
        clip_path = self._resolve_clip_path(clip_name)
        validate_te_type(clip_path, type)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher([clip_path], clip_type, self.load_data([clip_path])),)

class DualCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        # Same canonical type list as every other GGUF CLIP loader: core's
        # dual-only list (SD3/Flux pairs) omits single-TE types like
        # qwen_image, but this node happily loads one TE in a dual-template
        # workflow and must offer those types too.
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": _merged_type_options(),
            }
        }

    TITLE = "Dual CLIP Loader (GGUF) ⚡"
    SEARCH_ALIASES = ['load clip', 'dual clip', 'text encoder', 'clip loader', 'gguf']

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = self._resolve_clip_path(clip_name1)
        clip_path2 = self._resolve_clip_path(clip_name2)
        validate_te_type(clip_path1, type)
        validate_te_type(clip_path2, type)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

def _type_values(widget):
    """Combo widget value -> flat list of option strings.

    Combos arrive as ([...],) or [...]; accept either shape.
    """
    if isinstance(widget, (tuple, list)) and len(widget) == 1 \
            and isinstance(widget[0], (tuple, list)):
        return list(widget[0])
    return list(widget)

def _merged_type_options():
    """One canonical `type` list shared by EVERY GGUF CLIP loader.

    Core splits its type vocabulary across loader classes — the single list
    carries qwen_image/minimax/etc., the dual list carries sdxl/flux, and the
    triple/quadruple classes may only exist as comfy_extras nodes on newer
    cores. The GGUF loaders accept arbitrary TE files regardless of slot
    count, so all four expose the same union: the single list first (native
    order), then any values only present in the multi-loader lists appended.
    A class that is missing or broken is skipped; the union still builds
    from whatever remains.

    Raises (at node registration) if NO core loader yields a list at all —
    never returns an empty widget: a silently empty type dropdown would pin
    the Python-default CLIPType and every GGUF choice would validate against
    the wrong type.
    """
    merged, seen, errors = [], set(), []
    for name in ("CLIPLoader", "DualCLIPLoader",
                 "TripleCLIPLoader", "QuadrupleCLIPLoader"):
        base = getattr(nodes, name, None)
        if base is None:
            errors.append(f"{name!r} not in core 'nodes'")
            continue
        try:
            widget = base.INPUT_TYPES()["required"]["type"]
        except Exception as e:
            errors.append(f"{name!r} INPUT_TYPES failed ({e!r})")
            continue
        for t in _type_values(widget):
            if t not in seen:
                merged.append(t)
                seen.add(t)
    if not merged:
        raise RuntimeError(
            "Comfy-GGUF: cannot build the 'type' dropdown for the GGUF "
            f"CLIP loaders ({'; '.join(errors)}). ComfyUI core changed "
            "incompatibly - update ComfyUI or this node pack."
        )
    return (merged,)

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        required = {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "type": _merged_type_options(),
        }
        return {"required": required}

    TITLE = "Triple CLIP Loader (GGUF) ⚡"
    SEARCH_ALIASES = ['load clip', 'triple clip', 'text encoder', 'clip loader', 'gguf']

    def load_clip(self, clip_name1, clip_name2, clip_name3, type="sd3"):
        clip_path1 = self._resolve_clip_path(clip_name1)
        clip_path2 = self._resolve_clip_path(clip_name2)
        clip_path3 = self._resolve_clip_path(clip_name3)
        for p in (clip_path1, clip_path2, clip_path3):
            validate_te_type(p, type)
        clip_paths = (clip_path1, clip_path2, clip_path3)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

class QuadrupleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        required = {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
            "clip_name4": file_options,
            "type": _merged_type_options(),
        }
        return {"required": required}

    TITLE = "Quadruple CLIP Loader (GGUF) ⚡"
    SEARCH_ALIASES = ['load clip', 'quadruple clip', 'text encoder', 'clip loader', 'gguf']

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type="stable_diffusion"):
        clip_path1 = self._resolve_clip_path(clip_name1)
        clip_path2 = self._resolve_clip_path(clip_name2)
        clip_path3 = self._resolve_clip_path(clip_name3)
        clip_path4 = self._resolve_clip_path(clip_name4)
        for p in (clip_path1, clip_path2, clip_path3, clip_path4):
            validate_te_type(p, type)
        clip_paths = (clip_path1, clip_path2, clip_path3, clip_path4)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

# Stable class keys (workflow-safe). Display titles come from each node's TITLE.
NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced,
    "CLIPLoaderGGUF": CLIPLoaderGGUF,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF.TITLE,
    "UnetLoaderGGUFAdvanced": UnetLoaderGGUFAdvanced.TITLE,
    "CLIPLoaderGGUF": CLIPLoaderGGUF.TITLE,
    "DualCLIPLoaderGGUF": DualCLIPLoaderGGUF.TITLE,
    "TripleCLIPLoaderGGUF": TripleCLIPLoaderGGUF.TITLE,
    "QuadrupleCLIPLoaderGGUF": QuadrupleCLIPLoaderGGUF.TITLE,
}

# Imported last: nodes_extra subclasses CLIPLoaderGGUF, so it has to come after
# the class definitions above rather than at the top of the file.
from .nodes_extra import (NODE_CLASS_MAPPINGS as _EXTRA_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NAMES)

NODE_CLASS_MAPPINGS.update(_EXTRA_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_EXTRA_NAMES)

# Imported last because the Scenema nodes reuse GGUF loader helpers above.
from .nodes_scenema import (NODE_CLASS_MAPPINGS as _SCENEMA_CLASSES,
                            NODE_DISPLAY_NAME_MAPPINGS as _SCENEMA_NAMES)

NODE_CLASS_MAPPINGS.update(_SCENEMA_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_SCENEMA_NAMES)

from .nodes_minimax_music import (NODE_CLASS_MAPPINGS as _MUSIC_CLASSES,
                                  NODE_DISPLAY_NAME_MAPPINGS as _MUSIC_NAMES)

NODE_CLASS_MAPPINGS.update(_MUSIC_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_MUSIC_NAMES)

from .nodes_stems import (NODE_CLASS_MAPPINGS as _STEM_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _STEM_NAMES)

NODE_CLASS_MAPPINGS.update(_STEM_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_STEM_NAMES)

from .nodes_minimax_h3 import (NODE_CLASS_MAPPINGS as _H3_CLASSES,
                               NODE_DISPLAY_NAME_MAPPINGS as _H3_NAMES)

NODE_CLASS_MAPPINGS.update(_H3_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_H3_NAMES)


from .nodes_ltx25 import (NODE_CLASS_MAPPINGS as _LTX25_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _LTX25_NAMES)

NODE_CLASS_MAPPINGS.update(_LTX25_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LTX25_NAMES)

from .nodes_zimage import (NODE_CLASS_MAPPINGS as _ZIMAGE_CLASSES,
                           NODE_DISPLAY_NAME_MAPPINGS as _ZIMAGE_NAMES)

NODE_CLASS_MAPPINGS.update(_ZIMAGE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_ZIMAGE_NAMES)

from .nodes_ltx23 import (NODE_CLASS_MAPPINGS as _LTX23_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _LTX23_NAMES)

NODE_CLASS_MAPPINGS.update(_LTX23_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LTX23_NAMES)

from .nodes_lmstudio import (NODE_CLASS_MAPPINGS as _LMSTUDIO_CLASSES,
                             NODE_DISPLAY_NAME_MAPPINGS as _LMSTUDIO_NAMES)

NODE_CLASS_MAPPINGS.update(_LMSTUDIO_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_LMSTUDIO_NAMES)

from .nodes_qwen_tts import (NODE_CLASS_MAPPINGS as _QWEN_TTS_CLASSES,
                             NODE_DISPLAY_NAME_MAPPINGS as _QWEN_TTS_NAMES)

NODE_CLASS_MAPPINGS.update(_QWEN_TTS_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_QWEN_TTS_NAMES)

from .nodes_krea2 import (NODE_CLASS_MAPPINGS as _KREA2_CLASSES,
                          NODE_DISPLAY_NAME_MAPPINGS as _KREA2_NAMES)

NODE_CLASS_MAPPINGS.update(_KREA2_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_KREA2_NAMES)

from .nodes_qwen_image import (NODE_CLASS_MAPPINGS as _QWEN_IMAGE_CLASSES,
                               NODE_DISPLAY_NAME_MAPPINGS as _QWEN_IMAGE_NAMES)

NODE_CLASS_MAPPINGS.update(_QWEN_IMAGE_CLASSES)
NODE_DISPLAY_NAME_MAPPINGS.update(_QWEN_IMAGE_NAMES)
