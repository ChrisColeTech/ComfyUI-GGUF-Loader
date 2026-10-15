# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""TRELLIS.2 + Pixal3D (image -> textured 3D mesh) from per-stage files (``🤖 CCTech/TRELLIS.2``).

ComfyUI core runs both models natively (pure PyTorch: ``Trellis2Conditioning`` /
``Pixal3DConditioning``, ``Trellis2ShapeStage``, ``Trellis2UpsampleStage``,
``Trellis2TextureStage``, the ``VaeDecode*Trellis`` decoders and the 3d/mesh
texturing nodes) but only loads Comfy-Org's merged single files. The owner's
weights are split per stage (HF ``ChrisColeTech/TRELLIS.2-4B-FP16`` ``fp16/``,
``ChrisColeTech/Pixal3D`` ``ckpts/``, ``ChrisColeTech/Pixal3D-GGUF`` ``split/``)
and their extras are upstream-format folders. These loaders put those files
into core's own objects, so core's stage nodes do the rest:

  CCTechTrellis2ModelLoader     4 flow DiTs (safetensors or GGUF) -> MODEL
  CCTechTrellis2VAELoader       structure/shape/texture decoders -> shape VAE, texture VAE
  CCTechTrellis2DINOv3Loader    dinov3-vitl16 (+ NAF for Pixal3D) -> CLIP_VISION
  CCTechTrellis2BiRefNetLoader  extras/birefnet -> BACKGROUND_REMOVAL
  CCTechTrellis2MoGeLoader      extras/moge-2-vitl/model.pt -> MOGE_MODEL (Pixal3D camera)

Two stages are the owner's sidecar behaviour that core does not have (vendored,
``vendor/trellis2``):

  CCTechTrellis2PrepareImage       the sidecar's preprocess_image crop (pad 1.0 / 1.1)
  CCTechTrellis2StructureCleanup   the sidecar's sparse-structure hole fill + shell

GGUF DiTs stay quantized through this pack's GGMLOps, like UnetLoaderGGUF.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

CATEGORY = "\U0001F916 CCTech/TRELLIS.2"
FOLDERS = ("trellis2", "pixal3d")

for _key in FOLDERS:
    _dir = os.path.join(folder_paths.models_dir, _key)
    try:
        os.makedirs(_dir, exist_ok=True)
    except OSError:
        pass
    folder_paths.add_model_folder_path(_key, _dir)

NONE = "none"


def _roots():
    out = []
    for key in FOLDERS:
        try:
            paths = folder_paths.get_folder_paths(key)
        except KeyError:
            paths = [os.path.join(folder_paths.models_dir, key)]
        out += [(key, Path(p)) for p in paths]
    return out


def list_files(suffixes, test=None):
    """``<folder key>/<relative path>`` for weight files under models/trellis2 and models/pixal3d."""
    names = []
    for key, root in _roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in suffixes or not path.is_file():
                continue
            name = f"{key}/{path.relative_to(root).as_posix()}"
            if (test is None or test(name.lower())) and name not in names:
                names.append(name)
    return names


def resolve(name):
    key, _, rel = name.partition("/")
    for root_key, root in _roots():
        if root_key == key and (root / rel).is_file():
            return root / rel
    raise FileNotFoundError(f"'{name}' not found under models/{key} "
                            f"(searched {', '.join(str(r) for k, r in _roots() if k == key)})")


def load_weights(name):
    """(state_dict, is_gguf) for a .safetensors / .gguf / .pt / .pth file."""
    path = resolve(name)
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        from ..loader import gguf_sd_loader
        sd, _extra = gguf_sd_loader(str(path))
        return sd, True
    if suffix in (".pt", ".pth"):
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        for key in ("model", "state_dict"):
            if isinstance(sd, dict) and isinstance(sd.get(key), dict):
                sd = sd[key]
        return sd, False
    return comfy.utils.load_torch_file(str(path), safe_load=True), False


def _stage_choices(table, stage, suffixes, optional=False):
    from ..vendor.trellis2.stage_files import matches
    names = list_files(suffixes, lambda n: matches(table, stage, n)) or list_files(suffixes)
    if optional:
        return [NONE] + names
    return names or [NONE]


class CCTechTrellis2ModelLoader:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 / Pixal3D Model Loader (per-stage) ⚡"
    SEARCH_ALIASES = ["trellis2", "pixal3d", "image to 3d", "trellis gguf", "3d model loader"]

    @classmethod
    def INPUT_TYPES(s):
        from ..vendor.trellis2.stage_files import DIT_STAGES, WEIGHT_EXTENSIONS as ext
        return {"required": {
            "sparse_structure": (_stage_choices(DIT_STAGES, "sparse_structure", ext), {
                "tooltip": "ss_flow_img_dit_1_3B_64 (.safetensors or .gguf)."}),
            "shape_512": (_stage_choices(DIT_STAGES, "shape_512", ext), {
                "tooltip": "slat_flow_img2shape_dit_1_3B_512 - the first cascade shape pass."}),
            "shape_1024": (_stage_choices(DIT_STAGES, "shape_1024", ext), {
                "tooltip": "slat_flow_img2shape_dit_1_3B_1024 - the cascade's high-res pass."}),
            "texture": (_stage_choices(DIT_STAGES, "texture", ext, optional=True), {
                "tooltip": "slat_flow_imgshape2tex_dit_1_3B_1024 (none = shape only)."}),
            "weight_dtype": (["default", "bf16", "fp32"], {
                "default": "default",
                "tooltip": "Safetensors compute dtype; GGUF stays quantized either way."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    DESCRIPTION = ("Merge the per-stage TRELLIS.2 or Pixal3D flow DiTs into ComfyUI core's "
                   "Trellis2 model (Pixal3D's projection heads are detected from the weights).")

    def load(self, sparse_structure, shape_512, shape_1024, texture=NONE, weight_dtype="default"):
        from ..vendor.trellis2.stage_files import DIT_STAGES, is_pixal3d, merge
        chosen = {"sparse_structure": sparse_structure, "shape_512": shape_512,
                  "shape_1024": shape_1024}
        if texture != NONE:
            chosen["texture"] = texture
        parts, gguf_any, kinds = {}, False, {}
        for stage, name in chosen.items():
            if name == NONE:
                raise ValueError(f"TRELLIS.2 loader: no file chosen for {stage}")
            sd, is_gguf = load_weights(name)
            gguf_any |= is_gguf
            kinds[stage] = "Pixal3D" if is_pixal3d(sd) else "TRELLIS.2"
            parts[stage] = sd
        if len(set(kinds.values())) > 1:
            raise ValueError("TRELLIS.2 loader: TRELLIS.2 and Pixal3D stages mixed: "
                             + ", ".join(f"{k}={v}" for k, v in kinds.items()))
        sd = merge(parts, DIT_STAGES)
        model_options = {}
        if weight_dtype != "default":
            model_options["dtype"] = {"bf16": torch.bfloat16, "fp32": torch.float32}[weight_dtype]
        if gguf_any:
            from ..ops import GGMLOps
            ops = GGMLOps()
            ops.Linear.dequant_dtype = None
            ops.Linear.patch_dtype = None
            model_options["custom_operations"] = ops
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=model_options)
        if model is None:
            raise RuntimeError("TRELLIS.2 loader: ComfyUI did not recognise the merged model "
                               "(needs ComfyUI with native TRELLIS.2 support, 0.39+)")
        if gguf_any:
            from .gguf import GGUFModelPatcher
            model = GGUFModelPatcher.clone(model)
        logger.info("TRELLIS.2 loader: %s model from %s", next(iter(kinds.values())),
                    ", ".join(chosen.values()))
        return (model,)


class CCTechTrellis2VAELoader:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 / Pixal3D VAE Loader (per-stage) ⚡"
    SEARCH_ALIASES = ["trellis2 vae", "pixal3d vae", "shape decoder", "texture decoder"]

    @classmethod
    def INPUT_TYPES(s):
        from ..vendor.trellis2.stage_files import DECODER_STAGES as T
        ext = (".safetensors",)
        return {"required": {
            "structure_decoder": (_stage_choices(T, "structure_decoder", ext), {
                "tooltip": "ss_dec_conv3d_16l8 (TRELLIS-image-large's sparse-structure decoder)."}),
            "shape_decoder": (_stage_choices(T, "shape_decoder", ext), {
                "tooltip": "shape_dec_next_dc_f16c32."}),
            "texture_decoder": (_stage_choices(T, "texture_decoder", ext, optional=True), {
                "tooltip": "tex_dec_next_dc_f16c32 (none = no texture VAE)."}),
        }}

    RETURN_TYPES = ("VAE", "VAE")
    RETURN_NAMES = ("shape_vae", "texture_vae")
    FUNCTION = "load"
    DESCRIPTION = ("Build core's TRELLIS.2 shape VAE (structure + shape decoders) and texture "
                   "VAE from the per-stage decoder files. Same files serve Pixal3D.")

    def load(self, structure_decoder, shape_decoder, texture_decoder=NONE):
        from ..vendor.trellis2.stage_files import DECODER_STAGES, merge
        shape_sd = merge({"structure_decoder": load_weights(structure_decoder)[0],
                          "shape_decoder": load_weights(shape_decoder)[0]}, DECODER_STAGES)
        shape_vae = comfy.sd.VAE(sd=shape_sd)
        shape_vae.throw_exception_if_invalid()
        texture_vae = None
        if texture_decoder != NONE:
            tex_sd = merge({"texture_decoder": load_weights(texture_decoder)[0]}, DECODER_STAGES)
            texture_vae = comfy.sd.VAE(sd=tex_sd)
            texture_vae.throw_exception_if_invalid()
        return (shape_vae, texture_vae)


class CCTechTrellis2DINOv3Loader:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 / Pixal3D DINOv3 Loader ⚡"
    SEARCH_ALIASES = ["dinov3", "clip vision trellis", "naf", "pixal3d dino"]

    @classmethod
    def INPUT_TYPES(s):
        weights = (".safetensors",)
        return {"required": {
            "dinov3": (list_files(weights, lambda n: "dino" in n) or [NONE], {
                "tooltip": "facebook/dinov3-vitl16-pretrain-lvd1689m model.safetensors (upstream "
                           "Hugging Face format)."}),
            "naf": ([NONE] + list_files((".pth", ".pt", ".safetensors"), lambda n: "naf" in n), {
                "tooltip": "valeoai/NAF upsampler weights (naf_release.pth) - required by Pixal3D "
                           "Conditioning, unused by TRELLIS.2. none = no upsampler."}),
        }}

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("clip_vision",)
    FUNCTION = "load"
    DESCRIPTION = "DINOv3 ViT-L/16 image encoder for Trellis2/Pixal3D Conditioning (+ NAF)."

    def load(self, dinov3, naf=NONE):
        import comfy.clip_vision
        sd = dict(load_weights(dinov3)[0])
        if naf != NONE:
            naf_sd = load_weights(naf)[0]
            sd.update({k if k.startswith("naf.") else f"naf.{k}": v for k, v in naf_sd.items()})
        clip_vision = comfy.clip_vision.load_clipvision_from_sd(sd)
        if clip_vision is None:
            raise RuntimeError(f"'{dinov3}' is not a DINOv3 image encoder ComfyUI recognises")
        if naf != NONE and getattr(clip_vision, "naf", None) is None:
            raise RuntimeError(f"NAF weights in '{naf}' were not loaded")
        return (clip_vision,)


class CCTechTrellis2BiRefNetLoader:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 / Pixal3D BiRefNet Loader ⚡"
    SEARCH_ALIASES = ["birefnet", "background removal model", "rmbg trellis"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "birefnet": (list_files((".safetensors",), lambda n: "birefnet" in n) or [NONE], {
                "tooltip": "extras/birefnet/model.safetensors (ZhengPeng7 BiRefNet, the sidecar's "
                           "background remover)."}),
        }}

    RETURN_TYPES = ("BACKGROUND_REMOVAL",)
    RETURN_NAMES = ("bg_model",)
    FUNCTION = "load"
    DESCRIPTION = "BiRefNet for core Remove Background, from the owner's extras folder."

    def load(self, birefnet):
        import comfy.bg_removal_model
        model = comfy.bg_removal_model.load_background_removal_model(load_weights(birefnet)[0])
        if model is None:
            raise RuntimeError(f"'{birefnet}' is not a BiRefNet checkpoint ComfyUI recognises")
        return (model,)


class CCTechTrellis2MoGeLoader:
    CATEGORY = CATEGORY
    TITLE = "Pixal3D MoGe-2 Loader ⚡"
    SEARCH_ALIASES = ["moge", "moge2", "camera fov", "pixal3d camera"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "moge": (list_files((".pt", ".pth", ".safetensors"), lambda n: "moge" in n) or [NONE], {
                "tooltip": "extras/moge-2-vitl/model.pt (Ruicheng/moge-2-vitl checkpoint)."}),
        }}

    RETURN_TYPES = ("MOGE_MODEL",)
    RETURN_NAMES = ("moge_model",)
    FUNCTION = "load"
    DESCRIPTION = ("MoGe-2 for core MoGe Inference -> MoGe Geometry To FOV -> Pixal3D "
                   "Conditioning (the sidecar's camera estimate).")

    def load(self, moge):
        from comfy.ldm.moge.model import MoGeModel
        return (MoGeModel(load_weights(moge)[0]),)


class CCTechTrellis2PrepareImage:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 / Pixal3D Prepare Image ⚡"
    SEARCH_ALIASES = ["trellis2 preprocess", "pixal3d preprocess", "crop to subject 3d"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK", {"tooltip": "Foreground mask, 1 = subject (core Remove Background). "
                                         "For LoadImage's alpha mask, InvertMask it first."}),
            "pipeline": (["trellis2", "pixal3d"], {
                "default": "trellis2",
                "tooltip": "Crop margin as the sidecar: TRELLIS.2 1.0x, Pixal3D 1.1x the subject."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "prepare"
    DESCRIPTION = ("The sidecar's preprocess_image: 1024 cap, alpha > 0.8 bounding box, centred "
                   "square crop, subject premultiplied on black. Feeds Trellis2/Pixal3D "
                   "Conditioning (and MoGe for Pixal3D).")

    def prepare(self, image, mask, pipeline="trellis2"):
        from PIL import Image
        from ..vendor.trellis2.preprocess import preprocess_image, to_rgba
        if mask.ndim == 2:
            mask = mask[None]
        if mask.shape[-2:] != image.shape[1:3]:
            mask = comfy.utils.common_upscale(mask[:, None].float(), image.shape[2], image.shape[1],
                                              "bilinear", "disabled")[:, 0]
        outs = []
        for i in range(image.shape[0]):
            rgba = to_rgba(image[i, ..., :3].cpu().float().numpy(),
                           mask[min(i, mask.shape[0] - 1)].cpu().float().numpy())
            outs.append(preprocess_image(rgba, pipeline))
        first = outs[0].size
        if any(o.size != first for o in outs):
            logger.warning("TRELLIS.2 Prepare Image: batch crops differ in size; resizing to %s", first)
            outs = [o if o.size == first else o.resize(first, Image.Resampling.LANCZOS) for o in outs]
        batch = np.stack([np.asarray(o.convert("RGB"), dtype=np.float32) / 255.0 for o in outs])
        return (torch.from_numpy(batch),)


class CCTechTrellis2StructureCleanup:
    CATEGORY = CATEGORY
    TITLE = "TRELLIS.2 Sparse Structure Cleanup ⚡"
    SEARCH_ALIASES = ["fill holes voxel", "trellis2 holes", "keep shell", "sparse structure"]

    @classmethod
    def INPUT_TYPES(s):
        from ..vendor.trellis2.structure import ALGORITHMS
        return {"required": {
            "voxel": ("VOXEL", {"tooltip": "Trellis2 VAE Decode Structure output."}),
            "fill_holes": ("BOOLEAN", {"default": True,
                                       "tooltip": "The sidecar's TRELLIS.2 runs: on. Off passes "
                                                  "the voxel through unchanged."}),
            "hole_fill_algorithm": (list(ALGORITHMS), {"default": "remove_small_holes"}),
            "hole_structure": ("INT", {"default": 1, "min": 1, "max": 9,
                                       "tooltip": "Hole area (remove_small_holes: structure^2) / "
                                                  "closing kernel size."}),
            "hole_iterations": ("INT", {"default": 1, "min": 1, "max": 8}),
            "keep_only_shell": ("BOOLEAN", {"default": True,
                                            "tooltip": "Drop voxels >= 2 layers inside the surface "
                                                       "(applies with fill_holes, as in the sidecar)."}),
        }}

    RETURN_TYPES = ("VOXEL",)
    RETURN_NAMES = ("voxel",)
    FUNCTION = "clean"
    DESCRIPTION = ("The sidecar's sparse-structure post-pass (TRELLIS.2 engine defaults): fill "
                   "enclosed holes, keep the shell. Between VAE Decode Structure and Shape Stage.")

    def clean(self, voxel, fill_holes=True, hole_fill_algorithm="remove_small_holes",
              hole_structure=1, hole_iterations=1, keep_only_shell=True):
        if not fill_holes:
            return (voxel,)
        from comfy_api.latest import Types
        from ..vendor.trellis2.structure import fill_holes as run
        data = voxel.data
        arr = (data > 0.5).cpu().numpy()
        out = run(arr, hole_fill_algorithm, hole_structure, hole_iterations, keep_only_shell)
        return (Types.VOXEL(torch.from_numpy(out).to(device=data.device, dtype=data.dtype)),)


NODE_CLASS_MAPPINGS = {
    "CCTechTrellis2ModelLoader": CCTechTrellis2ModelLoader,
    "CCTechTrellis2VAELoader": CCTechTrellis2VAELoader,
    "CCTechTrellis2DINOv3Loader": CCTechTrellis2DINOv3Loader,
    "CCTechTrellis2BiRefNetLoader": CCTechTrellis2BiRefNetLoader,
    "CCTechTrellis2MoGeLoader": CCTechTrellis2MoGeLoader,
    "CCTechTrellis2PrepareImage": CCTechTrellis2PrepareImage,
    "CCTechTrellis2StructureCleanup": CCTechTrellis2StructureCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
