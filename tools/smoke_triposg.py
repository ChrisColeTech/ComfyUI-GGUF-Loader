"""Real-environment smoke test for the TripoSG nodes (nodes/triposg.py).

Imports the real ComfyUI modules of the portable install (CPU only, nothing
launched) and checks, without a GPU:

* registration: all five nodes, their widgets/defaults, and the folder scan
  (TripoSG roots and RMBG folders found under ``--models``);
* Prepare Image: the alpha path (no RMBG), the remove_background=off path,
  and the "connect an RMBG loader" refusal;
* the decode stage end to end on an analytic sphere field: hierarchical
  extraction -> validated mesh -> core MESH -> core SaveGLB bytes and
  MeshToFile3D (proves the MESH contract, not just the shape);
* the sampler stage on a tiny random DiT: seeded determinism and CFG batching
  (unconditional tokens first, zeros);
* with real weights (``--models`` holding the HF split tree): every component
  built on the meta device and its keys + shapes checked strictly against the
  checkpoint headers - no tensor data is read. ``--full`` additionally loads the
  real weights on CPU (float32, ~9 GB RAM) and runs RMBG on a synthetic image.

``models_dir`` is redirected to a temp folder so importing the nodes never
creates folders inside the ComfyUI install.

Usage: python -B tools/smoke_triposg.py [--models D:\\models\\image-models\\triposg] [--full]
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
ARGS.add_argument("--models", default=r"D:\models\image-models\triposg")
ARGS.add_argument("--comfy", default=r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
ARGS.add_argument("--full", action="store_true")
OPTS = ARGS.parse_args()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, OPTS.comfy)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.args_parsing = True
import folder_paths  # noqa: E402

folder_paths.models_dir = tempfile.mkdtemp(prefix="cctech_triposg_")
folder_paths.add_model_folder_path("triposg", OPTS.models)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
tsg = importlib.import_module("cctech_gguf_pkg.nodes.triposg")  # noqa: E402
pipeline = importlib.import_module("cctech_gguf_pkg.vendor.triposg.pipeline")  # noqa: E402
loaders = importlib.import_module("cctech_gguf_pkg.vendor.triposg.loaders")  # noqa: E402

from comfy_api.latest import Types  # noqa: E402


def test_registration_and_widgets():
    names = set(tsg.NODE_CLASS_MAPPINGS)
    assert names == {"CCTechTripoSGModelLoader", "CCTechTripoSGRMBGLoader",
                     "CCTechTripoSGPrepareImage", "CCTechTripoSGSampler",
                     "CCTechTripoSGDecodeMesh"}, names
    req = tsg.CCTechTripoSGSampler.INPUT_TYPES()["required"]
    assert req["steps"][1]["default"] == 50
    assert req["guidance_scale"][1]["default"] == 7.0
    assert req["num_tokens"][1]["default"] == 2048
    assert req["seed"][1]["default"] == 42
    dec = tsg.CCTechTripoSGDecodeMesh.INPUT_TYPES()["required"]
    assert dec["dense_octree_depth"][1]["default"] == 8
    assert dec["hierarchical_octree_depth"][1]["default"] == 9
    assert dec["bounds"][1]["default"] == 1.005
    assert tsg.CCTechTripoSGDecodeMesh.RETURN_TYPES == ("MESH",)
    prep = tsg.CCTechTripoSGPrepareImage.INPUT_TYPES()["required"]
    assert prep["padding_ratio"][1]["default"] == 0.1
    print("  models:", tsg.available_models(), " rmbg:", tsg.available_rmbg())


def _rgba_subject(size=160):
    """A red disc on a fully transparent background, as LoadImage would deliver it."""
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - 70) ** 2 + (xx - 90) ** 2 < 40 ** 2
    rgb = np.zeros((size, size, 3), np.float32)
    rgb[inside] = (1.0, 0.0, 0.0)
    mask = (~inside).astype(np.float32)  # ComfyUI MASK: 1 = transparent
    return torch.from_numpy(rgb)[None], torch.from_numpy(mask)[None]


def test_prepare_alpha_path_and_refusals():
    image, mask = _rgba_subject()
    node = tsg.CCTechTripoSGPrepareImage()
    out, = node.prepare(image, True, 0.1, "white", rmbg=None, mask=mask)
    # The disc spans 79 px; padding int(79 * 0.1) = 7 on each side -> 93.
    side = out.shape[1]
    assert out.shape[0] == 1 and out.shape[1] == out.shape[2] == 93, tuple(out.shape)
    corner = out[0, 0, 0]
    assert torch.allclose(corner, torch.ones(3)), corner      # white fill
    centre = out[0, side // 2, side // 2]
    assert centre[0] > 0.99 and centre[1] < 0.01, centre      # the subject
    try:
        node.prepare(image, True, 0.1, "white", rmbg=None, mask=None)
    except ValueError as exc:
        assert "RMBG" in str(exc)
    else:
        raise AssertionError("no RMBG and no alpha must be refused")
    full, = node.prepare(image, False, 0.1, "white")
    assert full.shape[1] == full.shape[2] == 160 + 2 * 16


class _SphereVAE(torch.nn.Module):
    """decode() -> occupancy logits of a sphere of radius 0.6 (positive inside)."""

    def decode(self, z, sampled_points):
        r = sampled_points.float().norm(dim=-1, keepdim=True)
        return types.SimpleNamespace(sample=(0.6 - r) * 20.0)


def _glb_json(data: bytes):
    magic, _, _ = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67
    length, kind = struct.unpack("<II", data[12:20])
    assert kind == 0x4E4F534A
    return json.loads(data[20:20 + length])


def test_decode_sphere_to_core_mesh_and_glb():
    try:
        import skimage  # noqa: F401
    except ImportError:
        print("  SKIP: scikit-image not installed")
        return
    v, f = pipeline.extract_mesh(_SphereVAE(), torch.zeros(1, 4, 4), torch.device("cpu"),
                                 dense_octree_depth=5, hierarchical_octree_depth=6)
    radius = np.linalg.norm(v, axis=1)
    assert abs(radius.mean() - 0.6) < 0.02, radius.mean()
    mesh = tsg.make_mesh([(v, f)])
    assert isinstance(mesh, Types.MESH)
    assert mesh.vertices.shape == (1, len(v), 3) and mesh.faces.dtype == torch.int64
    from comfy_extras.nodes_save_3d import MeshToFile3D, mesh_item_to_glb_bytes
    glb = mesh_item_to_glb_bytes(mesh, 0)
    gltf = _glb_json(glb)
    assert gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]][
        "count"] == len(v)
    file3d = MeshToFile3D.execute(mesh).result[0]
    assert file3d.format == "glb" and len(file3d.get_bytes()) == len(glb)
    batch = tsg.make_mesh([(v, f), (v[:10], f[:4])])
    assert batch.vertex_counts.tolist() == [len(v), 10]


def test_sampler_tiny_dit_seeded_and_cfg_order():
    from cctech_gguf_pkg.vendor.triposg.models.transformer import (TripoSGDiTConfig,
                                                                   TripoSGDiTModel)
    torch.manual_seed(0)
    dit = TripoSGDiTModel(TripoSGDiTConfig(num_attention_heads=2, width=32, in_channels=8,
                                           num_layers=2, cross_attention_dim=16)).eval()
    seen = []
    forward = dit.forward

    def spy(x, t, encoder_hidden_states=None, **kw):
        seen.append(encoder_hidden_states.detach().clone())
        return forward(x, t, encoder_hidden_states=encoder_hidden_states, **kw)

    dit.forward = spy
    cond = torch.randn(1, 5, 16)
    neg = torch.zeros_like(cond)
    sched = {"num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False}

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return pipeline.sample_latents(dit, loaders.make_scheduler(sched), cond, neg,
                                       num_inference_steps=3, guidance_scale=7.0, num_tokens=12,
                                       generator=g, device=torch.device("cpu"))

    a, b, c = run(1), run(1), run(2)
    assert a.shape == (1, 12, 8) and torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(seen[0][0], neg[0]) and torch.equal(seen[0][1], cond[0])
    seen.clear()
    g = torch.Generator().manual_seed(1)
    pipeline.sample_latents(dit, loaders.make_scheduler(sched), cond, neg, num_inference_steps=2,
                            guidance_scale=1.0, num_tokens=12, generator=g,
                            device=torch.device("cpu"))
    assert seen[0].shape[0] == 1  # guidance <= 1 disables CFG, as in the POC


def _header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in head.items()}


def _meta_model_vs_header(build, header, drop=()):
    with torch.device("meta"):
        model = build()
    ours = {k: tuple(t.shape) for k, t in model.state_dict().items()}
    theirs = {k: s for k, s in header.items() if not k.startswith(drop)}
    assert set(ours) == set(theirs), (sorted(set(ours) ^ set(theirs))[:8])
    bad = [k for k in ours if ours[k] != theirs[k]]
    assert not bad, bad[:8]
    return len(ours)


def test_real_checkpoint_layout():
    names = [n for n in tsg.available_models() if n != "none"]
    if not names:
        print(f"  SKIP: no TripoSG root under {OPTS.models}")
        return
    root = Path(OPTS.models) / names[0]
    from cctech_gguf_pkg.vendor.triposg.models.conditioner import DINOv2Config, DINOv2Model
    from cctech_gguf_pkg.vendor.triposg.models.transformer import (TripoSGDiTConfig,
                                                                   TripoSGDiTModel)
    from cctech_gguf_pkg.vendor.triposg.models.vae import TripoSGVAEConfig, TripoSGVAEModel
    n_dit = _meta_model_vs_header(
        lambda: TripoSGDiTModel(TripoSGDiTConfig.from_pretrained(root / "transformer")),
        _header(root / "transformer" / loaders.WEIGHTS))
    n_vae = _meta_model_vs_header(
        lambda: TripoSGVAEModel(TripoSGVAEConfig.from_pretrained(root / "vae")),
        _header(root / "vae" / loaders.WEIGHTS), drop=("encoder.", "quant."))
    n_enc = _meta_model_vs_header(
        lambda: DINOv2Model(DINOv2Config.from_pretrained(root / "image_encoder_dinov2")),
        _header(root / "image_encoder_dinov2" / "model.safetensors"))
    assert loaders.load_feature_extractor(root) is not None
    assert loaders.scheduler_config(root)["num_train_timesteps"] == 1000
    print(f"  {names[0]}: DiT {n_dit} / VAE {n_vae} / DINOv2 {n_enc} tensors match strictly")
    rmbg = [n for n in tsg.available_rmbg() if n != "none"]
    if rmbg:
        from cctech_gguf_pkg.vendor.triposg.briarmbg import BriaRMBG
        n_rmbg = _meta_model_vs_header(lambda: BriaRMBG(),
                                       _header(Path(OPTS.models) / rmbg[0] / "model.safetensors"))
        print(f"  {rmbg[0]}: RMBG {n_rmbg} tensors match strictly")
    if not OPTS.full:
        return
    model, = tsg.CCTechTripoSGModelLoader().load(names[0], "float32")
    assert model["transformer"].model.config.num_layers == 21
    r, = tsg.CCTechTripoSGRMBGLoader().load(rmbg[0])
    image = torch.full((1, 256, 256, 3), 0.92)
    image[0, 64:200, 80:180] = torch.tensor([0.2, 0.3, 0.8])
    out, = tsg.CCTechTripoSGPrepareImage().prepare(image, True, 0.1, "white", rmbg=r)
    print("  full load OK; RMBG crop ->", tuple(out.shape))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK - {len(tests)} TripoSG smoke checks passed")
