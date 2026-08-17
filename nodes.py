# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
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
from .loader import gguf_sd_loader, gguf_clip_loader, validate_te_type
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
            kwargs["metadata"] = extra.get("metadata", {})

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

class CLIPLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {
            "required": {
                "clip_name": (s.get_filename_list(),),
                "type": base["required"]["type"],
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "CLIP Loader (GGUF) ⚡"

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
        for p in ckpt_paths:
            if p.endswith(".gguf"):
                sd = gguf_clip_loader(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
                if "scaled_fp8" in sd: # NOTE: Scaled FP8 would require different custom ops, but only one can be active
                    raise NotImplementedError(f"Mixing scaled FP8 with GGUF is not supported! Use regular CLIP loader or switch model(s)\n({p})")
            clip_data.append(sd)
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
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = (s.get_filename_list(), )
        return {
            "required": {
                "clip_name1": file_options,
                "clip_name2": file_options,
                "type": base["required"]["type"],
            }
        }

    TITLE = "Dual CLIP Loader (GGUF) ⚡"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = self._resolve_clip_path(clip_name1)
        clip_path2 = self._resolve_clip_path(clip_name2)
        validate_te_type(clip_path1, type)
        validate_te_type(clip_path2, type)
        clip_paths = (clip_path1, clip_path2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        return (self.load_patcher(clip_paths, clip_type, self.load_data(clip_paths)),)

def _core_type_widget(core_class_name):
    """Core's `type` options for a CLIP loader class, with a safe fallback.

    Newer ComfyUI moved Triple/Quadruple loaders to comfy_extras as
    io.ComfyNode (no legacy INPUT_TYPES), so the named class can vanish
    between releases — in that case fall back to the single CLIPLoader's
    list, which every core exposes and which carries the full type set
    (qwen_image etc.). Returning None would render NO type widget, leaving
    the loader silently stuck on its Python default.
    """
    base = getattr(nodes, core_class_name, None)
    if base is not None:
        try:
            widget = base.INPUT_TYPES()["required"].get("type")
            if widget is not None:
                return widget
        except Exception:
            pass
    try:
        return nodes.CLIPLoader.INPUT_TYPES()["required"]["type"]
    except Exception:
        return None

class TripleCLIPLoaderGGUF(CLIPLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = (s.get_filename_list(), )
        required = {
            "clip_name1": file_options,
            "clip_name2": file_options,
            "clip_name3": file_options,
        }
        type_widget = _core_type_widget("TripleCLIPLoader")
        if type_widget is not None:
            required["type"] = type_widget
        return {"required": required}

    TITLE = "Triple CLIP Loader (GGUF) ⚡"

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
        }
        type_widget = _core_type_widget("QuadrupleCLIPLoader")
        if type_widget is not None:
            required["type"] = type_widget
        return {"required": required}

    TITLE = "Quadruple CLIP Loader (GGUF) ⚡"

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

