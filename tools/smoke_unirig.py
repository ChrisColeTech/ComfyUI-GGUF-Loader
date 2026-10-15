"""Real-environment smoke test for the UniRig nodes (nodes/unirig.py).

Imports the real ComfyUI modules of the portable install (CPU only, nothing
launched). Without weights it checks:

* registration, widgets and POC defaults (seed 42, mixamo, 50k faces, 15 beams,
  temperature 1.5, 2048 samples, voxel grid 196, 1 smoothing pass);
* core MESH in: padded-batch slicing (``vertex_counts``) and normals pass-through;
* the pure-torch SubMConv3d against a dense ``F.conv3d`` on a sparse grid (the
  definition it must equal) - it was also checked against a real spconv
  install (max abs 1e-5, see AGENTS.md);
* the skinned GLB: parsed back, JOINTS_0/WEIGHTS_0/IBM/node tree verified, and
  read by ComfyUI core's own ``Get3DComponents`` (proves FILE_3D interop).

With the checkpoints in ``--models`` (skeleton.safetensors / skin.safetensors)
both loader nodes build the real models on CPU (strict key checks), and
``--rig MESH`` runs the whole chain on CPU through the nodes: core
``Get3DComponents`` -> Extract Skeleton -> (Apply Skinning -> Rig to GLB when
``trimesh`` is importable). CPU beam search takes minutes; it is opt-in.

``models_dir`` is redirected to a temp folder so importing never creates folders
inside the ComfyUI install.

Usage: python -B tools/smoke_unirig.py [--models D:\\models\\image-models\\unirig] [--rig mesh.obj]
"""
import argparse
import importlib
import json
import struct
import sys
import tempfile
import types
from pathlib import Path

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--models", default=r"D:\models\image-models\unirig")
ARGS.add_argument("--comfy", default=r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
ARGS.add_argument("--rig", default=None, help="mesh file (glb/obj/stl) to rig on CPU")
OPTS = ARGS.parse_args()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, OPTS.comfy)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.args_parsing = True
import folder_paths  # noqa: E402

folder_paths.models_dir = tempfile.mkdtemp(prefix="cctech_unirig_")
folder_paths.add_model_folder_path("unirig", OPTS.models)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
ur = importlib.import_module("cctech_gguf_pkg.nodes.unirig")  # noqa: E402
sc = importlib.import_module("cctech_gguf_pkg.vendor.unirig.sparse_conv")  # noqa: E402
glb = importlib.import_module("cctech_gguf_pkg.vendor.unirig.glb")  # noqa: E402

from comfy_api.latest import Types  # noqa: E402


def test_registration_and_defaults():
    assert set(ur.NODE_CLASS_MAPPINGS) == {
        "CCTechUniRigSkeletonModelLoader", "CCTechUniRigSkinModelLoader", "CCTechUniRigLoadMesh",
        "CCTechUniRigExtractSkeleton", "CCTechUniRigApplySkinning", "CCTechUniRigRigToGLB"}
    spec = ur.CCTechUniRigExtractSkeleton.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert req["mesh"][0] == "MESH"
    assert req["seed"][1]["default"] == 42
    assert req["skeleton_template"][1]["default"] == "mixamo"
    assert req["max_faces"][1]["default"] == 50000
    assert opt["num_beams"][1]["default"] == 15 and opt["temperature"][1]["default"] == 1.5
    assert opt["num_samples"][1]["default"] == 2048
    skin = ur.CCTechUniRigApplySkinning.INPUT_TYPES()["required"]
    assert skin["voxel_grid_size"][1]["default"] == 196 and skin["smooth_iters"][1]["default"] == 1
    assert ur.CCTechUniRigRigToGLB.RETURN_TYPES == ("FILE_3D_GLB",)
    print("  checkpoints:", ur._checkpoints("skeleton"))


def test_mesh_item_slices_padded_batches():
    v = torch.rand(2, 6, 3)
    f = torch.tensor([[[0, 1, 2], [2, 3, 4]], [[0, 1, 2], [0, 0, 0]]])
    n = torch.rand(2, 6, 3)
    mesh = Types.MESH(vertices=v, faces=f, normals=n,
                      vertex_counts=torch.tensor([6, 3]), face_counts=torch.tensor([2, 1]))
    verts, faces, normals = ur.mesh_item(mesh, 1)
    assert verts.shape == (3, 3) and faces.shape == (1, 3) and normals.shape == (3, 3)
    verts, faces, normals = ur.mesh_item(Types.MESH(vertices=v[:1], faces=f[:1]), 0)
    assert verts.shape == (6, 3) and faces.dtype == np.int64 and normals is None


def test_submanifold_conv_equals_dense_conv3d():
    torch.manual_seed(0)
    grid, ci, co = 12, 5, 7
    active = torch.rand(grid, grid, grid) < 0.25
    coords = active.nonzero()
    indices = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int()
    feats = torch.randn(len(coords), ci)
    for k in (3, 5):
        conv = sc.SubMConv3d(ci, co, kernel_size=k, bias=True, indice_key="t")
        with torch.no_grad():
            conv.weight.normal_()
            conv.bias.normal_()
        out = conv(sc.SparseConvTensor(feats, indices, [grid] * 3, 1)).features
        dense = torch.zeros(1, ci, grid, grid, grid)
        dense[0, :, coords[:, 0], coords[:, 1], coords[:, 2]] = feats.t()
        w = conv.weight.detach().permute(0, 4, 1, 2, 3)  # [out, in, k0, k1, k2]
        ref = F.conv3d(dense, w, conv.bias.detach(), padding=k // 2)
        ref = ref[0, :, coords[:, 0], coords[:, 1], coords[:, 2]].t()
        assert torch.allclose(out, ref, atol=1e-4), (k, (out - ref).abs().max())


def _parse_glb(data):
    magic, version, total = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    n, kind = struct.unpack("<II", data[12:20])
    assert kind == 0x4E4F534A
    gltf = json.loads(data[20:20 + n])
    off = 20 + n
    bl, kind = struct.unpack("<II", data[off:off + 8])
    assert kind == 0x004E4942
    return gltf, data[off + 8:off + 8 + bl]


def _accessor(gltf, blob, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    return np.frombuffer(blob, dtype, count=acc["count"] * width,
                         offset=view["byteOffset"]).reshape(acc["count"], width)


def test_skinned_glb_round_trip_and_core_reader():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 2, 0]], np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    joints = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], np.float32)
    parents = [None, 0, 1]
    names = ["mixamorig:Hips", "mixamorig:Spine", "mixamorig:Head"]
    weights = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.7, 0.1], [0, 0, 1]], np.float32)
    data = glb.skinned_glb_bytes(vertices, faces, joints, parents, names, weights,
                                 normals=np.tile([[0, 0, 1]], (4, 1)).astype(np.float32))
    gltf, blob = _parse_glb(data)
    attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
    assert set(attrs) == {"POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0"}
    w = _accessor(gltf, blob, attrs["WEIGHTS_0"])
    j = _accessor(gltf, blob, attrs["JOINTS_0"])
    assert np.allclose(w.sum(1), 1.0) and j[2, 0] == 1 and j[3, 0] == 2
    ibm = _accessor(gltf, blob, gltf["skins"][0]["inverseBindMatrices"]).reshape(3, 4, 4)
    assert np.allclose(ibm[:, 3, :3], -joints)
    assert [n["name"] for n in gltf["nodes"][1:]] == names
    assert gltf["nodes"][2]["translation"] == [0.0, 1.0, 0.0]
    assert gltf["nodes"][1]["children"] == [2] and gltf["scenes"][0]["nodes"] == [0, 1]
    from io import BytesIO
    from comfy_extras.nodes_mesh_io import Get3DComponents
    mesh = Get3DComponents.execute(Types.File3D(BytesIO(data), file_format="glb")).result[0]
    assert mesh.vertices.shape[1] == 4 and mesh.faces.shape[1] == 2
    # ... and through Load Mesh, from a temp "output" folder, with traversal refused.
    out_dir = Path(tempfile.mkdtemp(prefix="cctech_unirig_out_"))
    folder_paths.set_output_directory(str(out_dir))
    (out_dir / "3d").mkdir()
    (out_dir / "3d" / "rig_00001_.glb").write_bytes(data)
    loaded, = ur.CCTechUniRigLoadMesh().load("output", "3d/rig_00001_.glb")
    assert loaded.vertices.shape[1] == 4
    for bad in ("../outside.glb", str(out_dir.parent / "x.glb")):
        try:
            ur.CCTechUniRigLoadMesh().load("output", bad)
        except ValueError as exc:
            assert "outside" in str(exc)
        else:
            raise AssertionError(f"{bad} must be refused")


def test_real_loaders_and_optional_rig():
    have = [n for n in ur._checkpoints("skeleton") if n != "none"]
    if not have:
        print(f"  SKIP: no checkpoints under {OPTS.models}")
        return
    skel_name = next((n for n in have if "skeleton" in n.lower()), None)
    skin_name = next((n for n in have if "skin" in n.lower()), None)
    skeleton_model, = ur.CCTechUniRigSkeletonModelLoader().load(skel_name, "fp32")
    skinning_model, = ur.CCTechUniRigSkinModelLoader().load(skin_name, "fp32")
    n_s = sum(p.numel() for p in skeleton_model["patcher"].model.parameters()) / 1e6
    n_k = sum(p.numel() for p in skinning_model["patcher"].model.parameters()) / 1e6
    print(f"  loaded {skel_name} ({n_s:.0f}M params), {skin_name} ({n_k:.0f}M params)")
    if not OPTS.rig:
        return
    from comfy_extras.nodes_mesh_io import Get3DComponents
    mesh = Get3DComponents.execute(Types.File3D(OPTS.rig)).result[0]
    out_mesh, skeleton, info = ur.CCTechUniRigExtractSkeleton().extract(
        mesh, skeleton_model, 42, "mixamo", 50000)
    print("  " + info.splitlines()[0])
    assert skeleton["names"][0].startswith("mixamorig:"), skeleton["names"][:3]
    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("  skinning SKIPPED: trimesh not installed in this environment")
        return
    rig, rig_info = ur.CCTechUniRigApplySkinning().skin(skeleton, skinning_model)
    model_3d, = ur.CCTechUniRigRigToGLB().export(rig)
    out = Path(tempfile.gettempdir()) / "cctech_unirig_smoke.glb"
    model_3d.save_to(str(out))
    print("  " + rig_info.splitlines()[1], "->", out)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK - {len(tests)} UniRig smoke checks passed")
