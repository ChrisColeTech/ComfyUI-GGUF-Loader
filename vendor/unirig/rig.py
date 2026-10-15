# GPL-3.0 (see LICENSE). Stage split of the owner's unirig_v1 ``wrapper.py``
# (UniRigWrapper.rig / _predict_skin), ``preprocessing.py`` and ``loaders.py``,
# plus ComfyUI-UniRig's ``direct.py`` inference helpers.
"""UniRig stages, with the POC's preprocessing and defaults unchanged.

``UniRigWrapper.rig(mesh_path)`` in the POC runs, in order:

1. load the mesh (trimesh, ``process=False``) with its vertex normals;
   quadric-decimate above ``max_faces`` (50k) and recompute normals;
2. predict the skeleton: normalize to [-1, 1] (centre of the bbox, half the
   longest extent), sample 2048 surface points + face normals (seeded), encode
   with Michelangelo, beam-sample the OPT decoder (15 beams, temperature 1.5,
   top-k 5, top-p 0.95, repetition penalty 3.0, 2048 new tokens) constrained by
   the tokenizer, detokenize; remap names to the template (vroid -> mixamorig:*)
   and denormalize the joints back to mesh space;
3. predict skin weights on the *normalized* mesh: trimesh voxelization +
   geodesic ``voxel_skin`` prior (grid 196), PTv3 + Michelangelo + bone
   cross-attention, then upstream's reskin edge diffusion + 0.03 threshold.

:func:`prepare_mesh` is 1, :func:`predict_skeleton` 2, :func:`predict_skin` 3.
Two additions, both for ComfyUI's reproducible seeds: the beam sampler gets a
generator seeded from ``seed`` (the POC seeded only the surface sampling), and
PTv3's serialization ``randperm`` runs under a forked RNG seeded the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .bone_names import cls_for_template, remap_names

log = logging.getLogger("unirig")

# ar_inference_articulationxl.yaml / ComfyUI-UniRig direct.predict_skeleton.
GENERATE_DEFAULTS = {
    "max_new_tokens": 2048,
    "num_beams": 15,
    "do_sample": True,
    "top_k": 5,
    "top_p": 0.95,
    "repetition_penalty": 3.0,
    "temperature": 1.5,
}


@dataclass
class MeshData:
    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray     # (F, 3) int64
    normals: np.ndarray   # (V, 3) float32

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """POC ``preprocessing._vertex_normals``: area-weighted face-normal average."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)
    vn = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vn, faces[:, i], face_normals)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _input_normals(vertices, faces, normals):
    """POC ``load_mesh``: the file's normals when present, else trimesh's."""
    if normals is not None and normals.shape == vertices.shape and np.any(normals):
        return np.ascontiguousarray(normals, dtype=np.float32)
    try:
        import trimesh
        vn = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals
        if vn.shape[0] == vertices.shape[0] and np.any(vn):
            return np.ascontiguousarray(vn, dtype=np.float32)
    except ImportError:
        pass
    return vertex_normals(vertices, faces)


def decimate(mesh: MeshData, max_faces: int, fallback=None) -> MeshData:
    """POC ``decimate_mesh`` (fast_simplification quadric decimation).

    When fast_simplification is not installed, ``fallback(vertices, faces,
    target)`` is used instead - the node passes ComfyUI core's QEM decimator."""
    if max_faces <= 0 or mesh.num_faces <= max_faces:
        return mesh
    try:
        import fast_simplification
        ratio = 1.0 - (max_faces / mesh.num_faces)
        v, f = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), target_reduction=ratio)
    except ImportError:
        if fallback is None:
            raise
        log.info("fast_simplification not installed; using ComfyUI core QEM decimation")
        v, f = fallback(mesh.vertices, mesh.faces, max_faces)
    v = np.ascontiguousarray(v, dtype=np.float32)
    f = np.ascontiguousarray(f, dtype=np.int64)
    log.info("decimated mesh %d -> %d faces (max_faces=%d)", mesh.num_faces, f.shape[0], max_faces)
    return MeshData(vertices=v, faces=f, normals=vertex_normals(v, f))


def prepare_mesh(vertices, faces, normals=None, max_faces=50000, decimate_fallback=None):
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("UniRig: mesh has no vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("UniRig: mesh must be a non-empty triangle mesh")
    mesh = MeshData(vertices, faces, _input_normals(vertices, faces, normals))
    return decimate(mesh, max_faces, decimate_fallback)


def normalize_to_unit_cube(vertices: np.ndarray):
    """POC ``preprocessing.normalize_to_unit_cube``."""
    min_c, max_c = vertices.min(axis=0), vertices.max(axis=0)
    center = (min_c + max_c) / 2
    scale = (max_c - min_c).max() / 2
    scale = scale if scale > 0 else 1.0
    return ((vertices - center) / scale).astype(np.float32), {
        "center": center, "scale": float(scale), "min": min_c, "max": max_c}


def denormalize(points: np.ndarray, params: dict) -> np.ndarray:
    return points * params["scale"] + params["center"]


def sample_mesh_surface(vertices, faces, num_samples=2048, seed=42):
    """ComfyUI-UniRig ``direct.sample_mesh_surface`` (area-weighted, seeded)."""
    rng = np.random.RandomState(seed)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) / 2
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)
    probs = face_areas / face_areas.sum()
    face_indices = rng.choice(len(faces), size=num_samples, p=probs)
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)
    sqrt_r1 = np.sqrt(r1)
    u, v, w = 1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2
    points = (u[:, None] * vertices[faces[face_indices, 0]]
              + v[:, None] * vertices[faces[face_indices, 1]]
              + w[:, None] * vertices[faces[face_indices, 2]])
    return points.astype(np.float32), face_normals[face_indices].astype(np.float32)


@torch.no_grad()
def predict_skeleton(model, mesh: MeshData, device, *, template="mixamo", seed=42,
                     num_samples=2048, **generate_kwargs) -> dict:
    """Stage 2. ``model`` is the pure skeleton spec (``pure_loader``)."""
    norm_vertices, norm_params = normalize_to_unit_cube(mesh.vertices)
    points, normals = sample_mesh_surface(norm_vertices, mesh.faces, num_samples, seed)
    kwargs = dict(GENERATE_DEFAULTS)
    kwargs.update({k: v for k, v in generate_kwargs.items() if v is not None})
    dtype = model.get_dtype() or torch.float32
    result = model.generate(
        vertices=torch.from_numpy(points).to(dtype=dtype, device=device),
        normals=torch.from_numpy(normals).to(dtype=dtype, device=device),
        cls=cls_for_template(template), seed=int(seed), **kwargs)
    if result.joints is None or len(result.joints) == 0:
        raise RuntimeError("UniRig skeleton prediction produced no joints")
    joints_norm = np.asarray(result.joints, dtype=np.float32)
    parents = [None if p is None else int(p) for p in list(result.parents)]
    validate_hierarchy(parents)
    names = list(result.names) if getattr(result, "names", None) else \
        [f"bone_{i}" for i in range(len(joints_norm))]
    tails = getattr(result, "tails", None)
    return {
        "joints": denormalize(joints_norm, norm_params).astype(np.float32),
        "joints_normalized": joints_norm,
        "tails_normalized": None if tails is None else np.asarray(tails, dtype=np.float32),
        "parents": parents,
        "names": remap_names(names, template),
        "template": template,
        "seed": int(seed),
        "normalization": norm_params,
        "mesh": mesh,
    }


def validate_hierarchy(parents) -> None:
    """POC ``cli.validate_hierarchy``: fail on degenerate skeletons (no root,
    self-parents, out-of-range parents, cycles) instead of exporting them."""
    num_joints = len(parents)
    par = [-1 if p is None else int(p) for p in list(parents)]
    if -1 not in par:
        raise ValueError("UniRig skeleton has no root bone (no None/-1 parent)")
    for j, p in enumerate(par):
        if p == j:
            raise ValueError(f"UniRig bone {j} is its own parent (degenerate skeleton)")
        if p >= num_joints:
            raise ValueError(f"UniRig bone {j} parent {p} out of range")
    for j in range(num_joints):
        steps, cur = 0, j
        while par[cur] != -1:
            cur = par[cur]
            steps += 1
            if steps > num_joints:
                raise ValueError(f"cycle in UniRig skeleton hierarchy at bone {j}")


@torch.no_grad()
def predict_skin(model, skeleton: dict, device, *, voxel_grid_size=196, smooth_iters=1):
    """Stage 3: per-vertex weights ``(V, J)`` on the skeleton's mesh."""
    from .data_vertex_group import voxel_skin, voxelization
    from .skin_post import diffuse_skin_weights

    mesh: MeshData = skeleton["mesh"]
    norm_vertices, _ = normalize_to_unit_cube(mesh.vertices)
    joints = skeleton["joints_normalized"]
    parents = np.asarray([-1 if p is None else p for p in skeleton["parents"]], dtype=np.int64)
    tails = skeleton["tails_normalized"]
    tails = joints.copy() if tails is None else tails

    grid_coords = voxelization(vertices=norm_vertices, faces=mesh.faces,
                               grid=voxel_grid_size, backend="trimesh")
    prior = voxel_skin(grid=voxel_grid_size, grid_coords=grid_coords, joints=joints,
                       vertices=norm_vertices, faces=mesh.faces, alpha=0.5, link_dis=0.00001,
                       grid_query=7, vertex_query=1, grid_weight=3.0)
    prior = np.nan_to_num(prior, nan=0., posinf=0., neginf=0.)

    dtype = model.get_dtype() or torch.float32

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dtype=dtype).unsqueeze(0).to(device)

    batch = {
        "vertices": t(norm_vertices), "normals": t(mesh.normals), "joints": t(joints),
        "tails": t(tails), "voxel_skin": t(prior),
        "parents": torch.from_numpy(parents).long().unsqueeze(0).to(device),
        "num_bones": torch.tensor([len(joints)], dtype=torch.long, device=device),
        "offset": torch.tensor([len(norm_vertices)], dtype=torch.long, device=device),
        "path": ["comfyui"],
    }
    devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(skeleton.get("seed", 0)))
        skin = model.predict_step(batch)[0]
    skin = skin.cpu().float().numpy()
    skin = diffuse_skin_weights(skin, norm_vertices, mesh.faces, list(skeleton["parents"]),
                                iter_steps=smooth_iters)
    return np.asarray(skin, dtype=np.float32)


def load_skin_model(checkpoint_path: str, dtype=None):
    """ComfyUI-UniRig ``direct._load_skin_model`` without ``model_parse`` (which
    would import the transformers-based ``unirig_ar``)."""
    import comfy.ops
    import comfy.utils

    from .configs import SKIN_MODEL_CONFIG
    from .unirig_skin import UniRigSkin

    operations = (comfy.ops.pick_operations(dtype, dtype) if dtype is not None
                  else comfy.ops.disable_weight_init)
    config = {k: v for k, v in SKIN_MODEL_CONFIG.items() if k != "__target__"}
    config.update(dtype=dtype, device="meta", operations=operations)
    with torch.device("meta"):
        model = UniRigSkin(**config)
    state = comfy.utils.load_torch_file(str(checkpoint_path))
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys:
        raise RuntimeError(f"UniRig skin: missing checkpoint keys {result.missing_keys[:8]}")
    if result.unexpected_keys:
        log.debug("UniRig skin: unused checkpoint keys %s", result.unexpected_keys[:8])
    left = [n for n, b in list(model.named_parameters()) + list(model.named_buffers())
            if b.device.type == "meta"]
    if left:
        raise RuntimeError(f"UniRig skin: tensors left on meta device {left[:8]}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.eval()


def load_skeleton_model(checkpoint_path: str, dtype=None):
    """The POC's transformers-free skeleton load (``pure_loader``)."""
    from .pure_loader import load_skeleton_pure
    model, _tokenizer = load_skeleton_pure(str(checkpoint_path), dtype=dtype, attn_backend="sdpa")
    return model


def summary(skeleton: dict, skin: Optional[np.ndarray] = None) -> str:
    mesh = skeleton["mesh"]
    lines = [f"UniRig rig: {len(skeleton['joints'])} bones, template={skeleton['template']}, "
             f"seed={skeleton['seed']}, mesh {len(mesh.vertices)} verts / {mesh.num_faces} faces"]
    if skin is not None:
        lines.append(f"skin weights {skin.shape[0]} x {skin.shape[1]}")
    for i, (name, parent) in enumerate(zip(skeleton["names"], skeleton["parents"])):
        lines.append(f"{i:3d} {name}  <- {'-' if parent is None else skeleton['names'][parent]}")
    return "\n".join(lines)
