"""Real-environment smoke test for the TRELLIS.2 / Pixal3D nodes (nodes/trellis2.py).

Imports the real ComfyUI modules of the portable install (CPU, nothing
launched). Without weights it checks registration and widgets, the two
vendored sidecar stages (Prepare Image crop arithmetic, Structure Cleanup hole
fill + shell) and the per-stage -> core key merge on synthetic tensors.

With the owner's folders (``--trellis2`` / ``--pixal3d``, defaults are the
D:\\models copies) it checks every per-stage file against ComfyUI core's own
modules key by key and shape by shape, reading headers only (no tensor data):
TRELLIS.2 safetensors DiTs, Pixal3D GGUF DiTs (through this pack's GGUF
reader), the three decoders. ``--full`` then loads everything for real on CPU
through the nodes (~25 GB RAM): both MODELs (and that no GGUF tensor sits in
a module GGMLOps does not own), both VAEs, DINOv3 + NAF, BiRefNet (and runs
it on a synthetic image), MoGe-2. ``--forward`` adds one CPU forward of the
GGUF sparse-structure DiT (minutes).

``models_dir`` is redirected to a temp folder so nothing is created inside the
ComfyUI install.

Usage: python -B tools/smoke_trellis2.py [--full] [--forward] [--naf PATH]
"""
import argparse
import importlib
import sys
import tempfile
import types
from pathlib import Path

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--trellis2", default=r"D:\models\image-models\trellis2")
ARGS.add_argument("--pixal3d", default=r"D:\models\image-models\pixal3d")
ARGS.add_argument("--naf", default=str(Path.home() / ".cache/torch/hub/checkpoints/naf_release.pth"))
ARGS.add_argument("--comfy", default=r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
ARGS.add_argument("--full", action="store_true")
ARGS.add_argument("--forward", action="store_true")
OPTS = ARGS.parse_args()
sys.argv = [sys.argv[0], "--cpu"]
sys.path.insert(0, OPTS.comfy)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import comfy.options  # noqa: E402
comfy.options.args_parsing = True
import folder_paths  # noqa: E402

folder_paths.models_dir = tempfile.mkdtemp(prefix="cctech_trellis2_")
folder_paths.add_model_folder_path("trellis2", OPTS.trellis2)
folder_paths.add_model_folder_path("pixal3d", OPTS.pixal3d)
if Path(OPTS.naf).is_file():  # NAF lives in the torch hub cache on the dev box
    folder_paths.add_model_folder_path("pixal3d", str(Path(OPTS.naf).parent))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
t2 = importlib.import_module("cctech_gguf_pkg.nodes.trellis2")  # noqa: E402
sf = importlib.import_module("cctech_gguf_pkg.vendor.trellis2.stage_files")  # noqa: E402

from comfy_api.latest import Types  # noqa: E402


def test_registration_and_defaults():
    assert set(t2.NODE_CLASS_MAPPINGS) == {
        "CCTechTrellis2ModelLoader", "CCTechTrellis2VAELoader", "CCTechTrellis2DINOv3Loader",
        "CCTechTrellis2BiRefNetLoader", "CCTechTrellis2MoGeLoader", "CCTechTrellis2PrepareImage",
        "CCTechTrellis2StructureCleanup"}
    req = t2.CCTechTrellis2ModelLoader.INPUT_TYPES()["required"]
    for stage in ("sparse_structure", "shape_512", "shape_1024", "texture"):
        print(f"  {stage}: {req[stage][0][:4]}")
    clean = t2.CCTechTrellis2StructureCleanup.INPUT_TYPES()["required"]
    assert clean["fill_holes"][1]["default"] is True and clean["keep_only_shell"][1]["default"]
    assert clean["hole_fill_algorithm"][1]["default"] == "remove_small_holes"
    for node in ("CCTechTrellis2DINOv3Loader", "CCTechTrellis2BiRefNetLoader",
                 "CCTechTrellis2MoGeLoader", "CCTechTrellis2VAELoader"):
        spec = t2.NODE_CLASS_MAPPINGS[node].INPUT_TYPES()["required"]
        print(f"  {node}: " + "; ".join(f"{k}={v[0][:3]}" for k, v in spec.items()))


def test_prepare_image_matches_sidecar_crop():
    img = torch.zeros(1, 300, 400, 3)
    img[0, 100:200, 150:230] = torch.tensor([0.2, 0.6, 1.0])
    mask = torch.zeros(1, 300, 400)
    mask[0, 100:200, 150:230] = 1.0
    node = t2.CCTechTrellis2PrepareImage()
    a, = node.prepare(img, mask, "trellis2")
    b, = node.prepare(img, mask, "pixal3d")
    # bbox 80x100 -> size 99 (max(229-150, 199-100)); crop size//2 each side -> 98
    assert a.shape == (1, 98, 98, 3), a.shape
    assert b.shape == (1, 108, 108, 3), b.shape   # int(99 * 1.1) = 108
    assert torch.allclose(a[0, 49, 49], torch.tensor([0.2, 0.6, 1.0]), atol=1 / 255)
    assert a[0, 0, 0].abs().sum() == 0 and b[0, 0, 0].abs().sum() == 0  # black background


def test_structure_cleanup_fills_and_shells():
    try:
        import skimage  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print("  SKIP: scipy/scikit-image missing")
        return
    r = 32
    zz, yy, xx = np.mgrid[:r, :r, :r]
    solid = ((zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2) < 12 ** 2
    data = torch.from_numpy(solid.astype(np.float32))[None]
    data[0, 16, 16, 16] = 0.0  # a one-voxel hole
    out, = t2.CCTechTrellis2StructureCleanup().clean(Types.VOXEL(data))
    assert isinstance(out, Types.VOXEL) and out.data.shape == data.shape
    shell = out.data[0].bool()
    assert shell.sum() < data.sum() and not shell[16, 16, 16]  # interior removed
    assert shell[16, 16, 5] and not shell[16, 16, 4]  # surface (radius 12) kept, outside empty
    same, = t2.CCTechTrellis2StructureCleanup().clean(Types.VOXEL(data), fill_holes=False)
    assert same.data is data


def _header(path):
    path = str(path)
    if path.endswith(".gguf"):
        from cctech_gguf_pkg.loader import gguf_sd_loader
        sd, _ = gguf_sd_loader(path)
        return {k: tuple(v.shape) for k, v in sd.items()}
    import json
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    head.pop("__metadata__", None)
    return {k: tuple(v["shape"]) for k, v in head.items()}


def _core_keys(merged):
    import comfy.model_detection
    import comfy.ops
    from comfy.ldm.trellis2.model import Trellis2
    cfg = comfy.model_detection.detect_unet_config(merged, "")
    cfg = {k: v for k, v in cfg.items() if k != "image_model"}
    with torch.device("meta"):
        model = Trellis2(operations=comfy.ops.disable_weight_init, **cfg)
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}, cfg


def _check_dits(label, files):
    parts = {}
    for stage, path in files.items():
        if not Path(path).is_file():
            print(f"  SKIP {label}: {path} missing")
            return
        parts[stage] = {k: torch.empty(s, device="meta") for k, s in _header(path).items()}
    merged = sf.merge(parts, sf.DIT_STAGES)
    ours, cfg = _core_keys(merged)
    assert set(ours) == set(merged), sorted(set(ours) ^ set(merged))[:8]
    bad = [k for k in ours if ours[k] != tuple(merged[k].shape)]
    assert not bad, bad[:8]
    print(f"  {label}: {len(ours)} tensors match core Trellis2 "
          f"(proj: {[k for k in cfg if k.startswith('image_attn_mode')]})")


def test_real_files_match_core_modules():
    T, P = Path(OPTS.trellis2), Path(OPTS.pixal3d)
    _check_dits("TRELLIS.2 safetensors", {
        "sparse_structure": T / "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "shape_512": T / "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "shape_1024": T / "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "texture": T / "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors"})
    _check_dits("Pixal3D GGUF", {
        "sparse_structure": P / "split/Sparse/ss_flow_img_dit_1_3B_64_bf16.gguf",
        "shape_512": P / "split/shape/slat_flow_img2shape_dit_1_3B_512_bf16.gguf",
        "shape_1024": P / "split/shape/slat_flow_img2shape_dit_1_3B_1024_bf16.gguf",
        "texture": P / "split/texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.gguf"})
    import comfy.ldm.trellis2.vae as V
    with torch.device("meta"):
        shape_vae, tex_vae = V.ShapeVae(), V.TextureVae()
    decs = {"structure_decoder": T / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            "shape_decoder": T / "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "texture_decoder": T / "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"}
    if all(p.is_file() for p in decs.values()):
        merged = sf.merge({k: _header(p) for k, p in decs.items()}, sf.DECODER_STAGES)
        core = {k: tuple(v.shape) for k, v in list(shape_vae.state_dict().items())
                + list(tex_vae.state_dict().items())}
        assert set(core) == set(merged) and all(core[k] == merged[k] for k in core)
        print(f"  decoders: {len(core)} tensors match core ShapeVae + TextureVae")


def _no_stray_ggml(model):
    from cctech_gguf_pkg.ops import GGMLTensor
    from cctech_gguf_pkg.ops import GGMLLayer
    stray = []
    for mname, module in model.named_modules():
        for pname, p in list(module.named_parameters(recurse=False)):
            if isinstance(p, GGMLTensor) and getattr(p, "tensor_type", None) is not None \
                    and not isinstance(module, GGMLLayer):
                stray.append(f"{mname}.{pname}")
    return stray


def test_full_loads():
    if not OPTS.full:
        return
    L = t2.CCTechTrellis2ModelLoader()
    t2m, = L.load("trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
                  "trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
                  "trellis2/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
                  "trellis2/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")
    dm = t2m.model.diffusion_model
    assert dm.shape2txt is not None and dm.image_attn_mode_shape == "global"
    print("  TRELLIS.2 MODEL:", type(t2m.model).__name__, t2m.model_dtype())
    del t2m, dm
    pm, = L.load("pixal3d/split/Sparse/ss_flow_img_dit_1_3B_64_bf16.gguf",
                 "pixal3d/split/shape/slat_flow_img2shape_dit_1_3B_512_bf16.gguf",
                 "pixal3d/split/shape/slat_flow_img2shape_dit_1_3B_1024_bf16.gguf",
                 "pixal3d/split/texture/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.gguf")
    pdm = pm.model.diffusion_model
    assert pdm.image_attn_mode_shape == "proj" and pdm.image_attn_mode_structure == "proj"
    assert not _no_stray_ggml(pdm), _no_stray_ggml(pdm)[:5]
    print("  Pixal3D GGUF MODEL:", type(pm).__name__, "projection heads on all stages")
    if OPTS.forward:
        ss = pdm.structure_model
        x = torch.randn(1, 8, 16, 16, 16)
        # Pixal3D's structure stage takes the global DINOv3 tokens plus per-voxel
        # back-projected features (16^3 voxels x 1024), as core's _shape_proj_cond builds them.
        cond = {"global": torch.randn(1, 1029, 1024), "proj": torch.randn(1, 4096, 1024)}
        with torch.no_grad():
            out = ss(x, torch.tensor([500.0]), cond)
        assert torch.isfinite(out).all() and out.shape == x.shape
        print("  GGUF sparse-structure forward OK", tuple(out.shape))
    del pm, pdm
    shape_vae, tex_vae = t2.CCTechTrellis2VAELoader().load(
        "trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "trellis2/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "trellis2/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors")
    print("  VAEs:", type(shape_vae.first_stage_model).__name__, type(tex_vae.first_stage_model).__name__)
    naf = [n for n in t2.list_files((".pth",), lambda n: "naf" in n)]
    cv, = t2.CCTechTrellis2DINOv3Loader().load(
        "pixal3d/extras/dinov3-vitl16/model.safetensors", naf[0] if naf else t2.NONE)
    print("  DINOv3:", type(cv).__name__, "NAF:", cv.naf is not None)
    bg, = t2.CCTechTrellis2BiRefNetLoader().load("pixal3d/extras/birefnet/model.safetensors")
    img = torch.full((1, 256, 256, 3), 0.95)
    img[0, 64:192, 96:160] = torch.tensor([0.8, 0.2, 0.1])
    mask = bg.encode_image(img)
    assert mask.shape == (1, 256, 256) and mask[0, 128, 128] > 0.5 > mask[0, 5, 5]
    print("  BiRefNet mask ok; centre", float(mask[0, 128, 128]), "corner", float(mask[0, 5, 5]))
    moge, = t2.CCTechTrellis2MoGeLoader().load("pixal3d/extras/moge-2-vitl/model.pt")
    print("  MoGe:", moge.version)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print(f"OK - {len(tests)} TRELLIS.2/Pixal3D smoke checks passed")
