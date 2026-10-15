# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Skinned bind-pose GLB writer, no dependencies beyond numpy.

The owner's ``compose_v1/gltf_export.export_skinned_glb`` (rig-only branch),
with pygltflib replaced by the ~40 lines of glTF 2.0 container it was used for.
Same scene layout, so rigs from both are interchangeable:

* node 0 = ``BodyMesh`` (mesh 0, skin 0); nodes 1..J = joints, named after the
  bones (``mixamorig:*`` for the mixamo template), each translated by its rest
  offset from its parent; bind pose = identity rotations;
* inverse bind matrices = pure translations to ``-joint_rest``;
* JOINTS_0 / WEIGHTS_0 = the top 4 influences per vertex, renormalized
  (glTF's limit), zero-weight slots pointing at joint 0;
* glTF is +Y-up right-handed, as are the input meshes - no axis change.

One addition: a ``NORMAL`` attribute (the normals the skinning saw) so viewers
shade smoothly; the POC wrote positions only.
"""

from __future__ import annotations

import json
import struct

import numpy as np

_F32, _U16, _U32 = 5126, 5123, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def top4_weights(skin_weights: np.ndarray):
    """POC ``_top4_weights``."""
    order = np.argsort(skin_weights, axis=1)[:, ::-1][:, :4]
    w = np.take_along_axis(skin_weights, order, axis=1)
    if w.shape[1] < 4:  # fewer than four bones: pad with empty slots
        pad = 4 - w.shape[1]
        order = np.pad(order, ((0, 0), (0, pad)))
        w = np.pad(w, ((0, 0), (0, pad)))
    total = w.sum(axis=1, keepdims=True)
    w = np.divide(w, np.where(total > 1e-8, total, 1.0))
    joints = order.astype(np.uint16)
    joints[w <= 0.0] = 0
    return joints, w.astype(np.float32)


class _Builder:
    def __init__(self):
        self.blob = bytearray()
        self.views, self.accessors = [], []

    def add(self, data, component, kind, target=None, minmax=False):
        while len(self.blob) % 4:
            self.blob.append(0)
        raw = np.ascontiguousarray(data).tobytes()
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.blob.extend(raw)
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": component,
               "count": int(data.shape[0]), "type": kind}
        if minmax:
            flat = data.reshape(data.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def skinned_glb_bytes(vertices, faces, joints_rest, parents, names, skin_weights,
                      normals=None, generator="ComfyUI-GGUF-Loader UniRig") -> bytes:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    joints_rest = np.asarray(joints_rest, dtype=np.float64)
    par = np.asarray([-1 if p is None else int(p) for p in list(parents)], dtype=np.int64)
    num_joints = joints_rest.shape[0]
    if np.asarray(skin_weights).shape != (vertices.shape[0], num_joints):
        raise ValueError(f"skin weights {np.asarray(skin_weights).shape} do not match "
                         f"{vertices.shape[0]} vertices x {num_joints} joints")
    if num_joints > 65535:
        raise ValueError("glTF JOINTS_0 is uint16: at most 65535 joints")

    b = _Builder()
    attributes = {"POSITION": b.add(vertices, _F32, "VEC3", _ARRAY_BUFFER, minmax=True)}
    if normals is not None:
        attributes["NORMAL"] = b.add(np.asarray(normals, dtype=np.float32), _F32, "VEC3",
                                     _ARRAY_BUFFER)
    indices = b.add(faces.reshape(-1, 1), _U32, "SCALAR", _ELEMENT_ARRAY_BUFFER)
    jw_joints, jw_weights = top4_weights(np.asarray(skin_weights, dtype=np.float64))
    attributes["JOINTS_0"] = b.add(jw_joints, _U16, "VEC4", _ARRAY_BUFFER)
    attributes["WEIGHTS_0"] = b.add(jw_weights, _F32, "VEC4", _ARRAY_BUFFER)

    nodes = [{"name": "BodyMesh", "mesh": 0, "skin": 0}]
    roots = [0]
    for j in range(num_joints):
        p = int(par[j])
        local = joints_rest[j] - (joints_rest[p] if p >= 0 else 0.0)
        nodes.append({"name": str(names[j]), "translation": [float(x) for x in local]})
    for j in range(num_joints):
        p = int(par[j])
        if p >= 0:
            nodes[1 + p].setdefault("children", []).append(1 + j)
        else:
            roots.append(1 + j)

    ibm = np.tile(np.eye(4, dtype=np.float32), (num_joints, 1, 1))
    ibm[:, 3, :3] = -joints_rest.astype(np.float32)  # column-major translation
    ibm_acc = b.add(ibm.reshape(num_joints, 16), _F32, "MAT4")

    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": roots}],
        "nodes": nodes,
        "meshes": [{"name": "Body", "primitives": [{"attributes": attributes,
                                                     "indices": indices}]}],
        "skins": [{"name": "Rig", "inverseBindMatrices": ibm_acc,
                   "joints": [1 + j for j in range(num_joints)],
                   "skeleton": 1 + int(np.nonzero(par == -1)[0][0])}],
        "buffers": [{"byteLength": len(b.blob)}],
        "bufferViews": b.views,
        "accessors": b.accessors,
    }
    payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    payload += b" " * (-len(payload) % 4)
    blob = bytes(b.blob) + b"\x00" * (-len(b.blob) % 4)
    total = 12 + 8 + len(payload) + 8 + len(blob)
    return b"".join([
        struct.pack("<III", 0x46546C67, 2, total),
        struct.pack("<II", len(payload), 0x4E4F534A), payload,
        struct.pack("<II", len(blob), 0x004E4942), blob,
    ])
