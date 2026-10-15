# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# Model code under vendor/unirig is GPL-3.0 (from ComfyUI-UniRig) - see its LICENSE.
"""UniRig auto-rigging as ComfyUI nodes (``🤖 CCTech/UniRig``).

Port of the owner's verified pipeline (``D:\\Projects\\poc\\hymotion-unirig-poc``,
``pipelines.unirig_v1``; giga-models sidecar ``engines/unirig.py``), which in
turn runs PozzettiAndrea/ComfyUI-UniRig's inference code. Stage per node, the
same split ComfyUI-UniRig itself uses (separate skeleton / skinning models and
nodes). Ids and types carry a ``CCTech`` prefix so this pack and ComfyUI-UniRig
can be installed side by side without overwriting each other's registrations:

  CCTechUniRigSkeletonModelLoader  models/unirig/skeleton.safetensors -> CCTECH_UNIRIG_SKELETON_MODEL
  CCTechUniRigSkinModelLoader      models/unirig/skin.safetensors -> CCTECH_UNIRIG_SKINNING_MODEL
  CCTechUniRigLoadMesh             input/ or output/ GLB/OBJ/STL -> core MESH
  CCTechUniRigExtractSkeleton      core MESH -> CCTECH_UNIRIG_SKELETON (+ the decimated MESH it binds)
  CCTechUniRigApplySkinning        CCTECH_UNIRIG_SKELETON -> CCTECH_UNIRIG_RIG (per-vertex skin weights)
  CCTechUniRigRigToGLB             CCTECH_UNIRIG_RIG -> FILE_3D_GLB (skinned bind-pose GLB)

The input is ComfyUI core's MESH - TripoSG's output directly, or any GLB/OBJ/STL
via core ``Load3D`` -> ``Get3DComponents``. The output GLB is a core FILE_3D, so
``SaveGLB``, ``Save3DAdvanced`` and ``Preview3D`` take it. Everything runs in pure
PyTorch: no Blender, no spconv / torch_scatter / torch_cluster / flash-attn, no
transformers (skeleton decode is the POC's pure OPT decoder). Needs ``trimesh``
for the skinning stage's voxelization (as the POC does).
"""

import logging
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

CATEGORY = "\U0001F916 CCTech/UniRig"
MODEL_FOLDER = "unirig"   # same folder ComfyUI-UniRig downloads into

_DIR = os.path.join(folder_paths.models_dir, MODEL_FOLDER)
try:
    os.makedirs(_DIR, exist_ok=True)
except OSError:
    pass
folder_paths.add_model_folder_path(MODEL_FOLDER, _DIR)

PRECISIONS = ["fp32", "bf16", "fp16", "auto"]
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
TEMPLATES = ["mixamo", "vroid", "articulationxl"]


def _checkpoints(prefer):
    try:
        names = [n for n in folder_paths.get_filename_list(MODEL_FOLDER)
                 if n.lower().endswith(".safetensors")]
    except Exception:
        names = []
    names.sort(key=lambda n: (prefer not in n.lower(), n))
    return names or ["none"]


def _checkpoint_path(name):
    path = folder_paths.get_full_path(MODEL_FOLDER, name) if name != "none" else None
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"UniRig checkpoint '{name}' not found in models/{MODEL_FOLDER}. Place "
            "skeleton.safetensors and skin.safetensors from ChrisColeTech/unirig there.")
    return path


def _dtype(precision):
    if precision != "auto":
        return _DTYPES[precision]
    device = comfy.model_management.get_torch_device()
    if comfy.model_management.should_use_bf16(device):
        return torch.bfloat16
    if comfy.model_management.should_use_fp16(device):
        return torch.float16
    return torch.float32


def _patcher(model):
    return comfy.model_patcher.CoreModelPatcher(
        model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())


def _stage(patcher, memory_required=0):
    comfy.model_management.load_models_gpu([patcher], memory_required=memory_required,
                                           force_full_load=True)
    return patcher.load_device


class CCTechUniRigSkeletonModelLoader:
    CATEGORY = CATEGORY
    TITLE = "UniRig Skeleton Model Loader ⚡"
    SEARCH_ALIASES = ["unirig", "auto rig", "rigging model", "skeleton model"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "checkpoint": (_checkpoints("skeleton"), {
                "tooltip": "skeleton.safetensors (UniRig articulation-xl AR model) in "
                           "models/unirig."}),
            "precision": (PRECISIONS, {
                "default": "fp32",
                "tooltip": "fp32 is the POC default; auto = bf16/fp16 where the GPU "
                           "supports it (ComfyUI-UniRig's default)."}),
        }}

    RETURN_TYPES = ("CCTECH_UNIRIG_SKELETON_MODEL",)
    RETURN_NAMES = ("skeleton_model",)
    FUNCTION = "load"
    DESCRIPTION = "Load UniRig's skeleton generator (Michelangelo encoder + pure-torch OPT-350m)."

    def load(self, checkpoint, precision="fp32"):
        from ..vendor.unirig.rig import load_skeleton_model
        dtype = _dtype(precision)
        model = load_skeleton_model(_checkpoint_path(checkpoint), dtype=dtype)
        return ({"name": checkpoint, "dtype": dtype, "patcher": _patcher(model)},)


class CCTechUniRigSkinModelLoader:
    CATEGORY = CATEGORY
    TITLE = "UniRig Skin Model Loader ⚡"
    SEARCH_ALIASES = ["unirig skin", "skinning model", "skin weights model"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "checkpoint": (_checkpoints("skin"), {
                "tooltip": "skin.safetensors (UniRig PTv3 skinning model) in models/unirig."}),
            "precision": (PRECISIONS, {"default": "fp32"}),
        }}

    RETURN_TYPES = ("CCTECH_UNIRIG_SKINNING_MODEL",)
    RETURN_NAMES = ("skinning_model",)
    FUNCTION = "load"
    DESCRIPTION = "Load UniRig's skinning-weight model (PTv3 + Michelangelo + bone attention)."

    def load(self, checkpoint, precision="fp32"):
        from ..vendor.unirig.rig import load_skin_model
        dtype = _dtype(precision)
        model = load_skin_model(_checkpoint_path(checkpoint), dtype=dtype)
        return ({"name": checkpoint, "dtype": dtype, "patcher": _patcher(model)},)


def mesh_item(mesh, index=0):
    """(vertices, faces, normals|None) numpy for one core MESH batch item."""
    n_v = n_f = None
    if getattr(mesh, "vertex_counts", None) is not None:
        n_v = int(mesh.vertex_counts[index])
        n_f = int(mesh.face_counts[index])
    verts = mesh.vertices[index, :n_v].detach().cpu().float().numpy()
    faces = mesh.faces[index, :n_f].detach().cpu().long().numpy()
    normals = getattr(mesh, "normals", None)
    if normals is not None:
        normals = normals[index, :n_v].detach().cpu().float().numpy()
    return verts, faces, normals


def _core_decimate(vertices, faces, target):
    """ComfyUI core's QEM decimator (the DecimateMesh node), for when the POC's
    fast_simplification is not installed."""
    from comfy_extras.mesh3d.postprocess.qem_decimate import QEMConfig, qem_decimate_simplify
    device = comfy.model_management.get_torch_device()
    rv, rf, *_ = qem_decimate_simplify(
        torch.from_numpy(vertices).float().to(device), torch.from_numpy(faces).long().to(device),
        int(target), colors=None, config=QEMConfig())
    return rv.cpu().numpy(), rf.cpu().numpy()


def _mesh_out(data):
    from comfy_api.latest import Types
    return Types.MESH(vertices=torch.from_numpy(data.vertices).unsqueeze(0),
                      faces=torch.from_numpy(data.faces).unsqueeze(0),
                      normals=torch.from_numpy(data.normals).unsqueeze(0))


def resolve_mesh_file(source_folder, file_path):
    """A mesh file inside ComfyUI's input/ or output/ folder; never outside it."""
    base = Path(folder_paths.get_input_directory() if source_folder == "input"
                else folder_paths.get_output_directory()).resolve()
    rel = str(file_path or "").strip().strip('"')
    if not rel:
        raise ValueError("UniRig Load Mesh: file_path is empty")
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"UniRig Load Mesh: {rel!r} is outside the {source_folder} folder")
    if not target.is_file():
        raise FileNotFoundError(f"UniRig Load Mesh: no such file {target}")
    return target


class CCTechUniRigLoadMesh:
    CATEGORY = CATEGORY
    TITLE = "UniRig Load Mesh ⚡"
    SEARCH_ALIASES = ["load glb", "load mesh file", "mesh from output", "unirig load mesh"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "source_folder": (["input", "output"], {"default": "output"}),
            "file_path": ("STRING", {
                "default": "3d/triposg_00001_.glb",
                "tooltip": "Path relative to the folder, e.g. a SaveGLB result "
                           "(3d/<prefix>_00001_.glb). GLB, glTF, OBJ or STL."}),
        }}

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    DESCRIPTION = ("Read a GLB/glTF/OBJ/STL from ComfyUI's input or output folder into a core "
                   "MESH (ComfyUI-UniRig's Load Mesh, via core Get3DComponents' parser).")

    def load(self, source_folder="output", file_path=""):
        from comfy_api.latest import Types
        from comfy_extras.nodes_mesh_io import Get3DComponents
        path = resolve_mesh_file(source_folder, file_path)
        return (Get3DComponents.execute(Types.File3D(str(path))).result[0],)

    @classmethod
    def IS_CHANGED(s, source_folder="output", file_path=""):
        try:
            return resolve_mesh_file(source_folder, file_path).stat().st_mtime_ns
        except (OSError, ValueError):
            return float("nan")


class CCTechUniRigExtractSkeleton:
    CATEGORY = CATEGORY
    TITLE = "UniRig Extract Skeleton ⚡"
    SEARCH_ALIASES = ["auto rig", "predict skeleton", "unirig skeleton", "mixamo skeleton"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mesh": ("MESH",),
            "skeleton_model": ("CCTECH_UNIRIG_SKELETON_MODEL",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,
                             "control_after_generate": True,
                             "tooltip": "Seeds surface sampling and beam sampling."}),
            "skeleton_template": (TEMPLATES, {
                "default": "mixamo",
                "tooltip": "mixamo: 52-bone humanoid named mixamorig:* (vroid generation + "
                           "rename); vroid: same bones, J_Bip_* names; articulationxl: "
                           "generic skeleton for any creature/object."}),
            "max_faces": ("INT", {
                "default": 50000, "min": 0, "max": 1000000, "step": 1000,
                "tooltip": "Decimate above this face count first (POC/upstream: 50000; "
                           "0 = never). The rig binds to the decimated mesh."}),
        }, "optional": {
            "num_beams": ("INT", {"default": 15, "min": 1, "max": 32,
                                  "tooltip": "Beam-sampling width (POC: 15)."}),
            "temperature": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 3.0, "step": 0.05,
                                      "tooltip": "Sampling temperature (POC: 1.5)."}),
            "num_samples": ("INT", {"default": 2048, "min": 512, "max": 16384, "step": 256,
                                    "tooltip": "Surface points fed to the encoder (POC: 2048)."}),
        }}

    RETURN_TYPES = ("MESH", "CCTECH_UNIRIG_SKELETON", "STRING")
    RETURN_NAMES = ("mesh", "skeleton", "skeleton_info")
    FUNCTION = "extract"
    DESCRIPTION = ("Predict a skeleton for a mesh with UniRig: normalize, sample the surface, "
                   "autoregressive bone tokens, named to the chosen template.")

    def extract(self, mesh, skeleton_model, seed, skeleton_template="mixamo", max_faces=50000,
                num_beams=15, temperature=1.5, num_samples=2048):
        from ..vendor.unirig import rig
        if mesh.vertices.shape[0] > 1:
            logger.warning("UniRig rigs one mesh per run; using batch item 0 of %d",
                           mesh.vertices.shape[0])
        verts, faces, normals = mesh_item(mesh, 0)
        data = rig.prepare_mesh(verts, faces, normals, max_faces=max_faces,
                                decimate_fallback=_core_decimate)
        comfy.model_management.throw_exception_if_processing_interrupted()
        # Beam search over 15 x 2048 tokens wants ~2 GB beside the weights.
        device = _stage(skeleton_model["patcher"], memory_required=2 * 1024 ** 3)
        skeleton = rig.predict_skeleton(
            skeleton_model["patcher"].model, data, device, template=skeleton_template,
            seed=seed, num_samples=num_samples, num_beams=num_beams, temperature=temperature)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        info = rig.summary(skeleton)
        logger.info(info.splitlines()[0])
        return (_mesh_out(data), skeleton, info)


class CCTechUniRigApplySkinning:
    CATEGORY = CATEGORY
    TITLE = "UniRig Apply Skinning ⚡"
    SEARCH_ALIASES = ["skin weights", "unirig skin", "bind mesh to skeleton"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "skeleton": ("CCTECH_UNIRIG_SKELETON",),
            "skinning_model": ("CCTECH_UNIRIG_SKINNING_MODEL",),
            "voxel_grid_size": ("INT", {
                "default": 196, "min": 64, "max": 512, "step": 4,
                "tooltip": "Voxel grid for the geodesic skin prior (POC/upstream: 196)."}),
            "smooth_iters": ("INT", {
                "default": 1, "min": 0, "max": 10,
                "tooltip": "Edge-diffusion passes of upstream's reskin cleanup (POC: 1)."}),
        }}

    RETURN_TYPES = ("CCTECH_UNIRIG_RIG", "STRING")
    RETURN_NAMES = ("rig", "rig_info")
    FUNCTION = "skin"
    DESCRIPTION = "Predict per-vertex skin weights for a UniRig skeleton (PTv3, pure torch)."

    def skin(self, skeleton, skinning_model, voxel_grid_size=196, smooth_iters=1):
        from ..vendor.unirig import rig
        try:
            import trimesh  # noqa: F401  (voxelization backend)
        except ImportError as exc:
            raise RuntimeError("UniRig skinning needs the 'trimesh' package "
                               "(pip install trimesh; listed in requirements.txt).") from exc
        device = _stage(skinning_model["patcher"], memory_required=1024 ** 3)
        weights = rig.predict_skin(skinning_model["patcher"].model, skeleton, device,
                                   voxel_grid_size=voxel_grid_size, smooth_iters=smooth_iters)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ({"skeleton": skeleton, "skin_weights": weights}, rig.summary(skeleton, weights))


class CCTechUniRigRigToGLB:
    CATEGORY = CATEGORY
    TITLE = "UniRig Rig to GLB ⚡"
    SEARCH_ALIASES = ["export rig", "rigged glb", "skinned glb", "save rig"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"rig": ("CCTECH_UNIRIG_RIG",)}}

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("model_3d",)
    FUNCTION = "export"
    DESCRIPTION = ("Write the rig as a skinned bind-pose GLB (joint node tree, inverse bind "
                   "matrices, top-4 weights) for SaveGLB / Save3DAdvanced / Preview3D.")

    def export(self, rig):
        from comfy_api.latest import Types
        from ..vendor.unirig.glb import skinned_glb_bytes
        skeleton = rig["skeleton"]
        data = skeleton["mesh"]
        glb = skinned_glb_bytes(data.vertices, data.faces, skeleton["joints"],
                                skeleton["parents"], skeleton["names"], rig["skin_weights"],
                                normals=data.normals)
        return (Types.File3D(BytesIO(glb), file_format="glb"),)


NODE_CLASS_MAPPINGS = {
    "CCTechUniRigSkeletonModelLoader": CCTechUniRigSkeletonModelLoader,
    "CCTechUniRigSkinModelLoader": CCTechUniRigSkinModelLoader,
    "CCTechUniRigLoadMesh": CCTechUniRigLoadMesh,
    "CCTechUniRigExtractSkeleton": CCTechUniRigExtractSkeleton,
    "CCTechUniRigApplySkinning": CCTechUniRigApplySkinning,
    "CCTechUniRigRigToGLB": CCTechUniRigRigToGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
