"""CPU-only smoke test for nodes/preprocessors.py.

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


# ── Batch 1/2 detector-backed nodes (NormalBAE, DSINE, HED, PiDiNet, MLSD,
# Lineart, Lineart Anime, Manga Line) - each follows the identical shape:
# node.estimate() builds <vendor_module>.<X>Detector(...), moves it to the
# comfy device, and calls the shared _estimate_batch() helper. Fake the
# Detector class per-test to confirm the node wires it up and returns the
# right IMAGE batch shape/dtype, without downloading real weights.

class _FakeSimpleDetector:
    """Matches every ported detector's .to(device)/.estimate(...) contract,
    echoing the input straight back so shape/dtype can be asserted."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def estimate(self, np_image, resolution=512, **kwargs):
        return np_image


def _check_detector_backed_node(node, vendor_module, detector_attr, node_kwargs=None):
    original = getattr(vendor_module, detector_attr)
    setattr(vendor_module, detector_attr, _FakeSimpleDetector)
    try:
        out, = node.estimate(torch.rand(2, 16, 16, 3), **(node_kwargs or {}))
    finally:
        setattr(vendor_module, detector_attr, original)
    assert out.shape == (2, 16, 16, 3)
    assert out.dtype == torch.float32
    return out


def test_normal_map_bae_delegates_to_detector():
    _check_detector_backed_node(pp.NormalMapBAE(), pp.normal_bae, "NormalBAEDetector")
    print("[ok] NormalMapBAE: builds NormalBAEDetector, returns a float32 IMAGE batch")


def test_normal_map_dsine_delegates_to_detector():
    _check_detector_backed_node(pp.NormalMapDSINE(), pp.dsine, "DSINEDetector")
    print("[ok] NormalMapDSINE: builds DSINEDetector, returns a float32 IMAGE batch")


def test_soft_edge_hed_delegates_to_detector():
    _check_detector_backed_node(pp.SoftEdgeHED(), pp.hed, "HEDDetector")
    print("[ok] SoftEdgeHED: builds HEDDetector, returns a float32 IMAGE batch")


def test_soft_edge_pidinet_delegates_to_detector():
    _check_detector_backed_node(pp.SoftEdgePiDiNet(), pp.pidinet, "PiDiNetDetector")
    print("[ok] SoftEdgePiDiNet: builds PiDiNetDetector, returns a float32 IMAGE batch")


def test_mlsd_lines_delegates_to_detector():
    _check_detector_backed_node(pp.MLSDLines(), pp.mlsd, "MLSDDetector")
    print("[ok] MLSDLines: builds MLSDDetector, returns a float32 IMAGE batch")


def test_lineart_delegates_to_detector():
    _check_detector_backed_node(pp.Lineart(), pp.lineart, "LineartDetector")
    print("[ok] Lineart: builds LineartDetector, returns a float32 IMAGE batch")


def test_lineart_anime_delegates_to_detector():
    _check_detector_backed_node(pp.LineartAnime(), pp.lineart_anime, "LineartAnimeDetector")
    print("[ok] LineartAnime: builds LineartAnimeDetector, returns a float32 IMAGE batch")


def test_manga_line_delegates_to_detector():
    _check_detector_backed_node(pp.MangaLine(), pp.manga_line, "MangaLineDetector")
    print("[ok] MangaLine: builds MangaLineDetector, returns a float32 IMAGE batch")


def test_openpose_delegates_to_detector_and_unpacks_tuple():
    # OpenPoseDetector.estimate() returns (canvas, pose_dict) - unlike every
    # other ported detector's single-ndarray return - confirm the node
    # unpacks it correctly and discards the keypoint dict for the IMAGE-only
    # output contract this pack's preprocessor nodes share.
    calls = []

    class _FakeOpenPoseDetector:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

        def estimate(self, np_image, resolution=512, include_body=True,
                     include_hand=True, include_face=True):
            calls.append((np_image.shape, include_body, include_hand, include_face))
            return np_image, {"people": []}

    original = pp.openpose.OpenPoseDetector
    pp.openpose.OpenPoseDetector = _FakeOpenPoseDetector
    try:
        node = pp.OpenPose()
        out, = node.estimate(torch.rand(2, 16, 16, 3), detect_body=True,
                             detect_hand=False, detect_face=True)
    finally:
        pp.openpose.OpenPoseDetector = original

    assert out.shape == (2, 16, 16, 3)
    assert out.dtype == torch.float32
    assert len(calls) == 2  # once per image in the batch
    assert all(c[1:] == (True, False, True) for c in calls)  # detect_* flags threaded through
    print("[ok] OpenPose: builds OpenPoseDetector, unpacks the (canvas, pose_dict) tuple, "
          "returns a float32 IMAGE batch, threads detect_body/hand/face through")


def test_all_preprocessor_nodes_registered():
    expected = {
        "DepthMap", "NormalMapBAE", "NormalMapDSINE", "SoftEdgeHED",
        "SoftEdgePiDiNet", "MLSDLines", "Lineart", "LineartAnime",
        "MangaLine", "OpenPose", "Canny",
    }
    assert expected.issubset(pp.NODE_CLASS_MAPPINGS.keys()), (
        expected - pp.NODE_CLASS_MAPPINGS.keys())
    assert expected.issubset(pp.NODE_DISPLAY_NAME_MAPPINGS.keys())
    print("[ok] all preprocessor nodes are registered in NODE_CLASS_MAPPINGS/"
          "NODE_DISPLAY_NAME_MAPPINGS")


if __name__ == "__main__":
    test_depth_map_delegates_to_detector_per_image()
    test_canny_output_shape_matches_input_and_is_grayscale_replicated()
    test_canny_solid_color_image_has_no_edges()
    test_normal_map_bae_delegates_to_detector()
    test_normal_map_dsine_delegates_to_detector()
    test_soft_edge_hed_delegates_to_detector()
    test_soft_edge_pidinet_delegates_to_detector()
    test_mlsd_lines_delegates_to_detector()
    test_lineart_delegates_to_detector()
    test_lineart_anime_delegates_to_detector()
    test_manga_line_delegates_to_detector()
    test_openpose_delegates_to_detector_and_unpacks_tuple()
    test_all_preprocessor_nodes_registered()
    print("[ok] all nodes/preprocessors smoke tests passed")
