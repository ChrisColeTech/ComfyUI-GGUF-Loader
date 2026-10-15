# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""TripoSG image -> 3D mesh, as ComfyUI nodes (``🤖 CCTech/TripoSG``).

Port of the owner's verified pure-PyTorch pipeline (``D:\\Projects\\poc\\triposg-poc``,
``pipelines/triposg_v1``; the giga-models sidecar engine ``engines/triposg.py``
drives the same code). The POC's one call is split at its own stage seams so
each can be inspected or swapped:

  CCTechTripoSGModelLoader        split checkpoint -> CCTECH_TRIPOSG_MODEL (DiT, VAE, DINOv2)
  CCTechTripoSGRMBGLoader         RMBG-1.4 folder -> CCTECH_TRIPOSG_RMBG
  CCTechTripoSGPrepareImage       IMAGE (+ alpha MASK) -> square foreground crop
  CCTechTripoSGSampler            prepared IMAGE -> CCTECH_TRIPOSG_LATENT (DINOv2 + DiT)
  CCTechTripoSGDecodeMesh         CCTECH_TRIPOSG_LATENT -> core MESH (SDF VAE + marching cubes)

The MESH output is ComfyUI core's own type, so ``SaveGLB``, ``MeshToFile3D``,
``Preview3D`` and the ``3d/mesh`` post-processing nodes (``DecimateMesh``,
``FillHoles``, ``UnwrapMesh`` ...) take it directly. Vertices are written as the
POC exported them - no axis flip or rescale (glTF is +Y up, and so is TripoSG).

Model files load from ``models/triposg`` only; nothing is downloaded at run time.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

CATEGORY = "\U0001F916 CCTech/TripoSG"
MODEL_FOLDER = "triposg"

_DIR = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
try:
    os.makedirs(_DIR, exist_ok=True)
except OSError:
    pass
folder_paths.add_model_folder_path(MODEL_FOLDER, _DIR)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _roots():
    try:
        return [Path(p) for p in folder_paths.get_folder_paths(MODEL_FOLDER)]
    except KeyError:
        return [Path(_DIR)]


def _scan(predicate, depth):
    """Folders under models/triposg (up to ``depth`` levels) passing ``predicate``."""
    found = []
    for root in _roots():
        if not root.is_dir():
            continue
        level = [root]
        for _ in range(depth):
            nxt = []
            for parent in level:
                for child in sorted(p for p in parent.iterdir() if p.is_dir()):
                    nxt.append(child)
                    if predicate(child):
                        rel = child.relative_to(root).as_posix()
                        if rel not in found:
                            found.append(rel)
            level = nxt
    return found


def _resolve(name, predicate, what):
    for root in _roots():
        candidate = root / name
        if candidate.is_dir() and predicate(candidate):
            return candidate
    raise FileNotFoundError(
        f"{what} '{name}' not found under models/{MODEL_FOLDER} "
        f"(searched: {', '.join(str(r) for r in _roots())}). See the README's TripoSG table.")


def available_models():
    from ..vendor.triposg.loaders import is_triposg_dir
    return _scan(is_triposg_dir, 1) or ["none"]


def available_rmbg():
    from ..vendor.triposg.loaders import is_rmbg_dir
    return _scan(is_rmbg_dir, 2) or ["none"]


def _patcher(model):
    return comfy.model_patcher.CoreModelPatcher(
        model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())


def _resolve_dtype(name):
    if name != "auto":
        return DTYPES[name]
    # The POC's default: float16 on CUDA, float32 where half is unsupported.
    device = comfy.model_management.get_torch_device()
    return torch.float16 if device.type == "cuda" else torch.float32


def _stage(patcher):
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    return patcher.load_device


class CCTechTripoSGModelLoader:
    CATEGORY = CATEGORY
    TITLE = "TripoSG Model Loader ⚡"
    SEARCH_ALIASES = ["triposg", "image to 3d", "load triposg", "3d mesh model"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (available_models(), {
                "tooltip": "A TripoSG split checkpoint folder in models/triposg (the HF "
                           "ChrisColeTech/TripoSG split/TripoSG tree: transformer/, vae/, "
                           "image_encoder_dinov2/, feature_extractor_dinov2/, scheduler/)."}),
            "dtype": (["auto", "float16", "bfloat16", "float32"], {
                "default": "auto",
                "tooltip": "auto = float16 on CUDA (the POC default), float32 on CPU."}),
        }}

    RETURN_TYPES = ("CCTECH_TRIPOSG_MODEL",)
    RETURN_NAMES = ("triposg_model",)
    FUNCTION = "load"
    DESCRIPTION = "Load the TripoSG DiT, SDF VAE and DINOv2 image encoder from local files."

    def load(self, model, dtype="auto"):
        from ..vendor.triposg import loaders
        root = _resolve(model, loaders.is_triposg_dir, "TripoSG model")
        torch_dtype = _resolve_dtype(dtype)
        if torch_dtype == torch.float16 and comfy.model_management.get_torch_device().type == "cpu":
            raise ValueError("float16 TripoSG inference on CPU is unsupported; use float32")
        logger.info("TripoSG: loading %s (%s)", root, torch_dtype)
        transformer = loaders.load_transformer(root, torch_dtype)
        vae = loaders.load_vae(root, torch_dtype)
        encoder = loaders.load_image_encoder(root, torch_dtype)
        return ({
            "name": model,
            "dtype": torch_dtype,
            "transformer": _patcher(transformer),
            "vae": _patcher(vae),
            "image_encoder": _patcher(encoder),
            "feature_extractor": loaders.load_feature_extractor(root),
            "scheduler_config": loaders.scheduler_config(root),
        },)


class CCTechTripoSGRMBGLoader:
    CATEGORY = CATEGORY
    TITLE = "TripoSG RMBG Loader ⚡"
    SEARCH_ALIASES = ["rmbg", "background removal", "remove background", "bria rmbg"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (available_rmbg(), {
                "tooltip": "An RMBG-1.4 folder (config.json + model.safetensors) in "
                           "models/triposg, e.g. RMBG-1.4 or TripoSG/rmbg-1.4."}),
        }}

    RETURN_TYPES = ("CCTECH_TRIPOSG_RMBG",)
    RETURN_NAMES = ("rmbg",)
    FUNCTION = "load"
    DESCRIPTION = "Load the BriaRMBG-1.4 background remover TripoSG preprocessing uses."

    def load(self, model):
        from ..vendor.triposg import loaders
        folder = _resolve(model, loaders.is_rmbg_dir, "RMBG model")
        return ({"name": model, "patcher": _patcher(loaders.load_rmbg(folder))},)


def _to_pil(image, mask=None):
    """One IMAGE item [H,W,3] (+ ComfyUI MASK, 1 = transparent) -> PIL RGB/RGBA."""
    rgb = np.clip(np.rint(image.detach().cpu().float().numpy() * 255.0), 0, 255).astype(np.uint8)
    if mask is None or tuple(mask.shape) != tuple(rgb.shape[:2]):
        return Image.fromarray(rgb, mode="RGB")
    alpha = 1.0 - mask.detach().cpu().float().numpy()
    alpha = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def _pad_square(source, background, padding_ratio):
    """remove_background off: the whole frame is the subject, padded like the POC."""
    source = source.convert("RGB")
    if max(source.size) > 2000:
        scale = 2000.0 / max(source.size)
        source = source.resize((max(1, round(source.width * scale)),
                                max(1, round(source.height * scale))), Image.Resampling.LANCZOS)
    width, height = source.size
    side = max(width, height)
    padding = int(side * float(padding_ratio))
    out_side = side + 2 * padding
    fill = tuple(int(round(c * 255)) for c in background)
    result = Image.new("RGB", (out_side, out_side), fill)
    result.paste(source, ((out_side - width) // 2, (out_side - height) // 2))
    return result


def _pil_to_image(pil):
    return torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0)


class CCTechTripoSGPrepareImage:
    CATEGORY = CATEGORY
    TITLE = "TripoSG Prepare Image ⚡"
    SEARCH_ALIASES = ["triposg preprocess", "foreground crop", "remove background 3d"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "remove_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run RMBG-1.4 when the image has no meaningful alpha (the POC "
                               "always does). Off: the whole frame is treated as the subject."}),
                "padding_ratio": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Border added around the subject's square crop (POC: 0.1)."}),
                "background": (["white", "black", "gray"], {
                    "default": "white",
                    "tooltip": "Fill behind the subject (POC: white)."}),
            },
            "optional": {
                "rmbg": ("CCTECH_TRIPOSG_RMBG",),
                "mask": ("MASK", {
                    "tooltip": "LoadImage's alpha mask (1 = transparent). Used instead of RMBG "
                               "when the alpha is meaningful (>=1% fully transparent and >=1% "
                               "opaque), as the POC does. A foreground mask (1 = subject, e.g. "
                               "core Remove Background) needs InvertMask first."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "prepare"
    DESCRIPTION = ("The POC's RMBG foreground crop: alpha (or RMBG-1.4 + Otsu + small-blob "
                   "cleanup) -> composite on the background -> centred square crop + padding.")

    def prepare(self, image, remove_background=True, padding_ratio=0.1, background="white",
                rmbg=None, mask=None):
        from ..vendor.triposg.preprocessing import prepare_image
        bg = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "gray": (0.5, 0.5, 0.5)}[background]
        model, device = None, None
        if remove_background and rmbg is not None:
            device = _stage(rmbg["patcher"])
            model = rmbg["patcher"].model
        outputs = []
        for i in range(image.shape[0]):
            item_mask = None
            if mask is not None:
                item_mask = mask[min(i, mask.shape[0] - 1)] if mask.ndim == 3 else mask
            pil = _to_pil(image[i], item_mask)
            if remove_background:
                if model is None and pil.mode != "RGBA":
                    raise ValueError("TripoSG Prepare Image: connect a TripoSG RMBG Loader, or an "
                                     "alpha mask, or turn remove_background off.")
                try:
                    prepared = prepare_image(pil, rmbg=model, device=device or "cpu",
                                             background=bg, padding_ratio=padding_ratio)
                except ValueError as exc:
                    if model is None and "RMBG model is required" in str(exc):
                        raise ValueError("TripoSG Prepare Image: the mask has no meaningful alpha; "
                                         "connect a TripoSG RMBG Loader.") from exc
                    raise
            else:
                prepared = _pad_square(pil, bg, padding_ratio)
            outputs.append(prepared)
        first = outputs[0].size
        if any(p.size != first for p in outputs):
            logger.warning("TripoSG Prepare Image: batch crops differ in size; resizing to %s", first)
            outputs = [p if p.size == first else p.resize(first, Image.Resampling.LANCZOS)
                       for p in outputs]
        return (torch.stack([_pil_to_image(p) for p in outputs]),)


class CCTechTripoSGSampler:
    CATEGORY = CATEGORY
    TITLE = "TripoSG Sampler ⚡"
    SEARCH_ALIASES = ["triposg generate", "image to 3d", "triposg shape"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "triposg_model": ("CCTECH_TRIPOSG_MODEL",),
            "image": ("IMAGE", {"tooltip": "Output of TripoSG Prepare Image."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 500,
                              "tooltip": "Rectified-flow steps (POC: 50)."}),
            "guidance_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                         "tooltip": "CFG scale; 1 or less disables CFG (POC: 7.0)."}),
            "num_tokens": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256,
                                   "tooltip": "Shape latent tokens (POC: 2048)."}),
        }}

    RETURN_TYPES = ("CCTECH_TRIPOSG_LATENT",)
    RETURN_NAMES = ("shape_latent",)
    FUNCTION = "sample"
    DESCRIPTION = "DINOv2 image tokens -> TripoSG rectified-flow DiT -> shape latents."

    def sample(self, triposg_model, image, seed, steps=50, guidance_scale=7.0, num_tokens=2048):
        from ..vendor.triposg import loaders, pipeline
        pils = [Image.fromarray(np.clip(np.rint(img.cpu().float().numpy() * 255.0), 0, 255)
                                .astype(np.uint8), mode="RGB") for img in image]
        device = _stage(triposg_model["image_encoder"])
        embeds, negative = pipeline.encode_image(
            triposg_model["image_encoder"].model, triposg_model["feature_extractor"],
            pils if len(pils) > 1 else pils[0], device)

        device = _stage(triposg_model["transformer"])
        pbar = comfy.utils.ProgressBar(steps)

        def on_step(done, total):
            comfy.model_management.throw_exception_if_processing_interrupted()
            pbar.update_absolute(done, total)

        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents = pipeline.sample_latents(
            triposg_model["transformer"].model,
            loaders.make_scheduler(triposg_model["scheduler_config"]),
            embeds.to(device), negative.to(device),
            num_inference_steps=steps, guidance_scale=guidance_scale, num_tokens=num_tokens,
            generator=generator, device=device, step_callback=on_step)
        return ({"samples": latents.cpu(), "seed": int(seed)},)


def make_mesh(items):
    """[(vertices [N,3], faces [M,3]) numpy] -> core MESH (padded batch if B > 1)."""
    from comfy_api.latest import Types
    verts = [torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for v, _ in items]
    faces = [torch.from_numpy(np.ascontiguousarray(f, dtype=np.int64)) for _, f in items]
    if len(items) == 1:
        return Types.MESH(vertices=verts[0].unsqueeze(0), faces=faces[0].unsqueeze(0))
    max_v = max(v.shape[0] for v in verts)
    max_f = max(f.shape[0] for f in faces)
    pv = verts[0].new_zeros((len(items), max_v, 3))
    pf = faces[0].new_zeros((len(items), max_f, 3))
    for i, (v, f) in enumerate(zip(verts, faces)):
        pv[i, :v.shape[0]] = v
        pf[i, :f.shape[0]] = f
    return Types.MESH(vertices=pv, faces=pf,
                      vertex_counts=torch.tensor([v.shape[0] for v in verts], dtype=torch.int64),
                      face_counts=torch.tensor([f.shape[0] for f in faces], dtype=torch.int64))


class CCTechTripoSGDecodeMesh:
    CATEGORY = CATEGORY
    TITLE = "TripoSG Decode Mesh ⚡"
    SEARCH_ALIASES = ["triposg decode", "marching cubes", "sdf to mesh", "latent to mesh"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "triposg_model": ("CCTECH_TRIPOSG_MODEL",),
            "shape_latent": ("CCTECH_TRIPOSG_LATENT",),
            "dense_octree_depth": ("INT", {
                "default": 8, "min": 5, "max": 10,
                "tooltip": "Dense occupancy grid = 2^depth per axis (POC: 8 -> 256^3)."}),
            "hierarchical_octree_depth": ("INT", {
                "default": 9, "min": 5, "max": 11,
                "tooltip": "Final surface resolution = 2^depth per axis, refined only near "
                           "the surface (POC: 9 -> 512^3). Must be >= dense depth."}),
            "bounds": ("FLOAT", {
                "default": 1.005, "min": 0.5, "max": 2.0, "step": 0.005,
                "tooltip": "Half-extent of the sampled cube (POC: 1.005)."}),
        }}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "decode"
    DESCRIPTION = ("TripoSG SDF VAE queried on a hierarchical octree, then Lewiner marching "
                   "cubes -> a watertight, untextured core MESH.")

    def decode(self, triposg_model, shape_latent, dense_octree_depth=8,
               hierarchical_octree_depth=9, bounds=1.005):
        from ..vendor.triposg import pipeline
        if hierarchical_octree_depth < dense_octree_depth:
            raise ValueError("hierarchical_octree_depth must be at least dense_octree_depth")
        try:
            import skimage.measure  # noqa: F401  (Lewiner marching cubes)
        except ImportError as exc:
            raise RuntimeError("TripoSG Decode Mesh needs scikit-image "
                               "(pip install scikit-image; listed in requirements.txt).") from exc
        device = _stage(triposg_model["vae"])
        latents = shape_latent["samples"]
        b = float(bounds)
        items = []
        for i in range(latents.shape[0]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            z = latents[i:i + 1].to(device=device, dtype=triposg_model["dtype"])
            items.append(pipeline.extract_mesh(
                triposg_model["vae"].model, z, device, bounds=(-b, -b, -b, b, b, b),
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth, index=i))
            logger.info("TripoSG: mesh %d -> %d vertices, %d faces",
                        i, len(items[-1][0]), len(items[-1][1]))
        return (make_mesh(items),)


NODE_CLASS_MAPPINGS = {
    "CCTechTripoSGModelLoader": CCTechTripoSGModelLoader,
    "CCTechTripoSGRMBGLoader": CCTechTripoSGRMBGLoader,
    "CCTechTripoSGPrepareImage": CCTechTripoSGPrepareImage,
    "CCTechTripoSGSampler": CCTechTripoSGSampler,
    "CCTechTripoSGDecodeMesh": CCTechTripoSGDecodeMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
