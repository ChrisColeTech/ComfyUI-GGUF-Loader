"""CPU-only smoke test for nodes/preprocessors.py.

This module now only backs two historically-named nodes (Krea2DepthMap,
Flux2KleinDepthMap, QwenImageCanny) - the advanced preprocessor set
(normal maps, soft edges, MLSD, lineart variants, OpenPose) moved to the
standalone ComfyUI-ControlNet-Nodes package. See that package's own
tools/smoke_preprocessors.py for coverage of those.

Stubs the comfy-internal modules preprocessors.py imports at the top level
so DepthMap/Canny's shape/dtype contracts can be verified without a running
ComfyUI or GPU. Depth Anything V2's real architecture (vendor/depth_anything_v2.py)
is exercised separately (real weights, real image) against the actual
portable ComfyUI environment - this suite fakes the detector to avoid a
model download.

Usage:  python tools/smoke_preprocessors.py
"""
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

comfy = types.ModuleType("comfy")
comfy_model_management = types.ModuleType("comfy.model_management")
comfy_model_management.get_torch_device = lambda: torch.device("cpu")
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = str(REPO_ROOT / "models")

sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = comfy_model_management
sys.modules["folder_paths"] = folder_paths
comfy.model_management = comfy_model_management

sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
# Fake the `nodes` subpackage too (pointed at the real nodes/ dir) so importing
# nodes.preprocessors doesn't execute the real nodes/__init__.py aggregation.
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
import importlib
pp = importlib.import_module("cctech_gguf_pkg.nodes.preprocessors")  # noqa: E402


# ── DepthMap ──────────────────────────────────────────────────────────────

def test_depth_map_delegates_to_detector_per_image():
    calls = []

    class _FakeDetector:
        def __init__(self, ckpt_name):
            calls.append(("init", ckpt_name))

        def to(self, device):
            calls.append(("to", device))
            return self

        def estimate(self, np_image, resolution=512):
            calls.append(("estimate", np_image.shape, resolution))
            return np_image  # already HWC uint8, echo it back

    original = pp.depth_anything_v2.DepthAnythingV2Detector
    pp.depth_anything_v2.DepthAnythingV2Detector = _FakeDetector
    try:
        node = pp.DepthMap()
        out, = node.estimate(torch.rand(2, 16, 16, 3), ckpt_name="depth_anything_v2_vits.pth",
                             resolution=256)
    finally:
        pp.depth_anything_v2.DepthAnythingV2Detector = original

    assert out.shape == (2, 16, 16, 3)
    assert out.dtype == torch.float32
    inits = [c for c in calls if c[0] == "init"]
    estimates = [c for c in calls if c[0] == "estimate"]
    assert inits == [("init", "depth_anything_v2_vits.pth")]  # detector built once, reused
    assert len(estimates) == 2  # once per image in the batch
    assert all(r == 256 for (_n, _shape, r) in estimates)
    print("[ok] DepthMap: builds the detector once, calls estimate() per image, "
          "returns a float32 IMAGE batch of the same shape")


# ── Canny ─────────────────────────────────────────────────────────────────

def test_canny_output_shape_matches_input_and_is_grayscale_replicated():
    node = pp.Canny()
    out, = node.detect(torch.rand(1, 32, 32, 3), low_threshold=100, high_threshold=200)
    assert out.shape == (1, 32, 32, 3)
    assert out.dtype == torch.float32
    # Canny output is a grayscale edge map replicated across all 3 channels.
    assert torch.equal(out[0, :, :, 0], out[0, :, :, 1])
    assert torch.equal(out[0, :, :, 1], out[0, :, :, 2])
    print("[ok] Canny: output shape matches input, float32, grayscale replicated to 3 channels")


def test_canny_solid_color_image_has_no_edges():
    node = pp.Canny()
    solid = torch.full((1, 16, 16, 3), 0.5)
    out, = node.detect(solid)
    assert torch.all(out == 0.0)
    print("[ok] Canny: a solid-color image (no edges) produces an all-zero edge map")


def test_module_does_not_self_register_nodes():
    # DepthMap/Canny are implementation-only here - only krea2.py/qwen_image.py/
    # flux_klein.py register them, under their own historical names, to avoid
    # colliding with ComfyUI-ControlNet-Nodes' own generic "DepthMap"/"Canny".
    assert not hasattr(pp, "NODE_CLASS_MAPPINGS")
    assert not hasattr(pp, "NODE_DISPLAY_NAME_MAPPINGS")
    print("[ok] nodes/preprocessors.py does not self-register any nodes")


if __name__ == "__main__":
    test_depth_map_delegates_to_detector_per_image()
    test_canny_output_shape_matches_input_and_is_grayscale_replicated()
    test_canny_solid_color_image_has_no_edges()
    test_module_does_not_self_register_nodes()
    print("[ok] all nodes/preprocessors smoke tests passed")
