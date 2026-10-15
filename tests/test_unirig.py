"""CPU-light checks of the vendored UniRig pieces (no weights, no ComfyUI)."""
import json
import struct

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vendor.unirig import glb, rig, sparse_conv
from vendor.unirig.bone_names import cls_for_template, remap_names
from vendor.unirig.skin_post import diffuse_skin_weights


def test_submanifold_conv_matches_dense_conv3d_at_active_voxels():
    torch.manual_seed(0)
    grid, ci, co = 10, 4, 6
    coords = (torch.rand(grid, grid, grid) < 0.3).nonzero()
    indices = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int()
    feats = torch.randn(len(coords), ci)
    for k in (3, 5):
        conv = sparse_conv.SubMConv3d(ci, co, kernel_size=k, bias=True, indice_key="a")
        with torch.no_grad():
            conv.weight.normal_()
            conv.bias.normal_()
        x = sparse_conv.SparseConvTensor(feats, indices, [grid] * 3, 1)
        out = conv(x).features
        dense = torch.zeros(1, ci, grid, grid, grid)
        dense[0, :, coords[:, 0], coords[:, 1], coords[:, 2]] = feats.t()
        ref = F.conv3d(dense, conv.weight.detach().permute(0, 4, 1, 2, 3), conv.bias.detach(),
                       padding=k // 2)[0, :, coords[:, 0], coords[:, 1], coords[:, 2]].t()
        assert torch.allclose(out, ref, atol=1e-4)
        assert ("a", (k, k, k), (1, 1, 1)) in x.indice_dict  # neighbour map cached


def test_neighbor_map_respects_batches_and_bounds():
    idx = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.int32)
    nbr = sparse_conv.neighbor_map(idx, [2, 1, 1], 3)
    centre = sparse_conv.kernel_offsets(3, 1).index((0, 0, 0))
    plus_x = sparse_conv.kernel_offsets(3, 1).index((1, 0, 0))
    assert nbr[:, centre].tolist() == [0, 1, 2]
    assert nbr[0, plus_x] == 1 and nbr[1, plus_x] == -1  # out of bounds
    minus_x = sparse_conv.kernel_offsets(3, 1).index((-1, 0, 0))
    assert nbr[2, minus_x] == -1  # batch 1 never sees batch 0
    assert sparse_conv.modules.is_spconv_module(sparse_conv.SubMConv3d(1, 1))


def _parse(data):
    n = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    off = 20 + n
    return gltf, data[off + 8:off + 8 + struct.unpack("<I", data[off:off + 4])[0]]


def test_skinned_glb_layout_matches_the_poc_exporter():
    v = np.random.RandomState(0).rand(5, 3).astype(np.float32)
    f = np.array([[0, 1, 2], [2, 3, 4]])
    joints = np.array([[0, 0, 0], [0, 1, 0], [0.5, 1, 0]], np.float32)
    w = np.random.RandomState(1).rand(5, 3).astype(np.float32)
    data = glb.skinned_glb_bytes(v, f, joints, [None, 0, 1], ["a", "b", "c"], w)
    assert data[:4] == b"glTF" and len(data) % 4 == 0
    assert struct.unpack("<I", data[8:12])[0] == len(data)
    gltf, blob = _parse(data)
    assert gltf["nodes"][0] == {"name": "BodyMesh", "mesh": 0, "skin": 0}
    assert gltf["nodes"][3]["translation"] == pytest.approx([0.5, 0.0, 0.0])
    assert gltf["skins"][0]["joints"] == [1, 2, 3] and gltf["skins"][0]["skeleton"] == 1
    ibm = gltf["accessors"][gltf["skins"][0]["inverseBindMatrices"]]
    view = gltf["bufferViews"][ibm["bufferView"]]
    mats = np.frombuffer(blob, np.float32, 48, view["byteOffset"]).reshape(3, 4, 4)
    assert np.allclose(mats[:, 3, :3], -joints)
    with pytest.raises(ValueError, match="do not match"):
        glb.skinned_glb_bytes(v, f, joints, [None, 0, 1], ["a", "b", "c"], w[:, :2])


def test_top4_weights_renormalize_and_pad():
    w = np.array([[0.1, 0.2, 0.3, 0.4, 1.0], [0, 0, 0, 0, 1]], np.float64)
    j, wt = glb.top4_weights(w)
    assert j[0].tolist() == [4, 3, 2, 1] and np.allclose(wt.sum(1), 1)
    assert j[1].tolist() == [4, 0, 0, 0]
    j2, w2 = glb.top4_weights(np.array([[0.25, 0.75]]))
    assert j2.shape == (1, 4) and j2[0].tolist() == [1, 0, 0, 0] and w2[0, 1] == 0.25


def test_hierarchy_validation_matches_the_poc():
    rig.validate_hierarchy([None, 0, 1, 1])
    rig.validate_hierarchy([None, 0, None])  # several roots are allowed, as in the POC
    with pytest.raises(ValueError, match="no root"):
        rig.validate_hierarchy([1, 0])
    with pytest.raises(ValueError, match="own parent"):
        rig.validate_hierarchy([None, 1])
    with pytest.raises(ValueError, match="out of range"):
        rig.validate_hierarchy([None, 5])
    with pytest.raises(ValueError, match="cycle"):
        rig.validate_hierarchy([None, 2, 1])


def test_normalization_sampling_and_mesh_preparation():
    rs = np.random.RandomState(0)
    v = (rs.rand(50, 3) * [2, 4, 1] + 3).astype(np.float32)
    n, params = rig.normalize_to_unit_cube(v)
    assert n.min() >= -1 - 1e-6 and n.max() <= 1 + 1e-6 and np.isclose(np.abs(n[:, 1]).max(), 1)
    assert np.allclose(rig.denormalize(n, params), v, atol=1e-5)
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    p1, _ = rig.sample_mesh_surface(n, f, 64, seed=7)
    p2, _ = rig.sample_mesh_surface(n, f, 64, seed=7)
    assert np.array_equal(p1, p2)
    mesh = rig.prepare_mesh(v, f, normals=None, max_faces=0)
    assert mesh.normals.shape == v.shape and np.allclose(np.linalg.norm(mesh.normals[:9], axis=1), 1)
    calls = []

    def fake_decimate(vertices, faces, target):
        calls.append(target)
        return vertices, faces[:target]

    try:
        import fast_simplification  # noqa: F401
    except ImportError:
        small = rig.prepare_mesh(v, f, max_faces=2, decimate_fallback=fake_decimate)
        assert calls == [2] and small.num_faces == 2
    with pytest.raises(ValueError):
        rig.prepare_mesh(v, np.zeros((0, 3)), max_faces=0)


def test_template_names_and_skin_cleanup():
    assert cls_for_template("mixamo") == "vroid" and cls_for_template("articulationxl") == "articulationxl"
    assert remap_names(["J_Bip_C_Hips", "bone_3"], "mixamo") == ["mixamorig:Hips", "bone_3"]
    assert remap_names(["J_Bip_C_Hips"], "vroid") == ["J_Bip_C_Hips"]
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    w = np.array([[0.9, 0.1], [0.02, 0.98], [0.5, 0.5]], np.float32)
    out = diffuse_skin_weights(w, v, np.array([[0, 1, 2]]), [None, 0])
    assert np.allclose(out.sum(1), 1) and (out >= 0).all()
