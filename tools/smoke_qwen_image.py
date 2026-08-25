"""CPU-only smoke test for nodes_qwen_image.py.

Stubs the comfy-internal modules nodes_qwen_image.py imports at the top
level so QwenImageImg2Img's guard rails, latent shapes, and control
dispatch (model_patch vs controlnet attachment) can be verified without a
running ComfyUI or GPU. Loading real DiffSynth/InstantX weights end-to-end
is covered separately against the actual portable ComfyUI environment, not
part of this offline suite.

Usage:  python tools/smoke_qwen_image.py
"""
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

comfy = types.ModuleType("comfy")
comfy_controlnet = types.ModuleType("comfy.controlnet")
comfy_controlnet.load_controlnet_state_dict = lambda sd: object()
comfy_model_management = types.ModuleType("comfy.model_management")
comfy_model_management.intermediate_device = lambda: torch.device("cpu")
comfy_model_management.get_torch_device = lambda: torch.device("cpu")
comfy_sd = types.ModuleType("comfy.sd")
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.common_upscale = lambda samples, w, h, method, crop: torch.nn.functional.interpolate(
    samples, size=(h, w), mode="bilinear", align_corners=False)
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_filename_list = lambda key: []
folder_paths.get_full_path = lambda key, name: None
folder_paths.get_full_path_or_raise = lambda key, name: name
folder_paths.get_folder_paths = lambda key: []
folder_paths.models_dir = str(REPO_ROOT / "models")
nodes = types.ModuleType("nodes")
nodes.MAX_RESOLUTION = 16384

class _FakeDiffSynthCnetPatch:
    """Stand-in for comfy_extras.nodes_model_patch.DiffSynthCnetPatch - the
    real one calls model_patch.model.process_input_latent_image(...) in
    __init__, which needs a real QwenImageBlockWiseControlNet. Just record
    the args; this offline suite only verifies the attach wiring, not the
    patch's own internal math (covered separately against real weights).
    """

    def __init__(self, model_patch, vae, image, strength, mask=None):
        self.model_patch = model_patch
        self.vae = vae
        self.image = image
        self.strength = strength
        self.mask = mask


comfy_extras = types.ModuleType("comfy_extras")
comfy_extras_nodes_model_patch = types.ModuleType("comfy_extras.nodes_model_patch")
comfy_extras_nodes_model_patch.DiffSynthCnetPatch = _FakeDiffSynthCnetPatch

sys.modules["comfy"] = comfy
sys.modules["comfy.controlnet"] = comfy_controlnet
sys.modules["comfy.model_management"] = comfy_model_management
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils
sys.modules["folder_paths"] = folder_paths
sys.modules["nodes"] = nodes
sys.modules["comfy_extras"] = comfy_extras
sys.modules["comfy_extras.nodes_model_patch"] = comfy_extras_nodes_model_patch
comfy.controlnet = comfy_controlnet
comfy.model_management = comfy_model_management
comfy.sd = comfy_sd
comfy.utils = comfy_utils

sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
import importlib
qi = importlib.import_module("cctech_gguf_pkg.nodes_qwen_image")  # noqa: E402


class _FakeModelPatcher:
    def __init__(self):
        self.model_options = {}
        self.double_block_patches = []

    def clone(self):
        c = _FakeModelPatcher()
        c.double_block_patches = list(self.double_block_patches)
        return c

    def set_model_double_block_patch(self, patch):
        self.double_block_patches.append(patch)


class _FakeControlNet:
    def __init__(self):
        self.hint = None
        self.strength = None

    def copy(self):
        return self

    def set_cond_hint(self, cond_hint, strength, timestep_range, vae=None, extra_concat=[]):
        self.hint = cond_hint
        self.strength = strength
        return self

    def set_previous_controlnet(self, prev):
        self.prev = prev


def _clip():
    return types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: [[torch.zeros(1, 1, 4), {}]],
        tokenize=lambda s: s)


def _vae():
    return types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 8, 8))


# ── QwenImageControl / dispatch helper ──────────────────────────────────

def test_apply_controlnet_stamps_control_key_on_every_conditioning_item():
    control_net = _FakeControlNet()
    conditioning = [[torch.zeros(1, 1, 4), {}], [torch.zeros(1, 1, 4), {"foo": "bar"}]]
    out = qi._apply_controlnet_to_conditioning(
        conditioning, control_net, torch.zeros(1, 3, 8, 8), 0.8, vae=None)
    assert len(out) == 2
    for t in out:
        assert t[1]["control"] is control_net
        assert t[1]["control_apply_to_uncond"] is False
    assert out[1][1]["foo"] == "bar"  # existing conditioning dict keys survive
    print("[ok] _apply_controlnet_to_conditioning: stamps control onto every conditioning item, preserves existing keys")


def test_apply_controlnet_sets_hint_and_strength():
    control_net = _FakeControlNet()
    hint = torch.rand(1, 3, 8, 8)
    qi._apply_controlnet_to_conditioning([[torch.zeros(1, 1, 4), {}]], control_net, hint, 0.42, vae=None)
    assert control_net.strength == 0.42
    assert torch.equal(control_net.hint, hint)
    print("[ok] _apply_controlnet_to_conditioning: passes hint/strength through to set_cond_hint")


# ── QwenImageImg2Img guard rails ─────────────────────────────────────────

def test_img2img_rejects_qwen_control_without_control_image():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    try:
        node.prepare(model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
                     qwen_control=qi.QwenImageControl("controlnet", _FakeControlNet()))
        raised = False
    except ValueError as e:
        raised = "control_image" in str(e)
    assert raised
    print("[ok] QwenImageImg2Img: qwen_control with no control_image raises, not a silent partial run")


def test_img2img_ignores_control_image_without_qwen_control():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    result = node.prepare(model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
                          control_image=torch.rand(1, 8, 8, 3))
    assert result is not None
    print("[ok] QwenImageImg2Img: control_image with no qwen_control is ignored, not an error")


def test_img2img_txt2img_empty_latent_shape():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    _, _, _, latent, denoise = node.prepare(model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64)
    assert latent["samples"].shape == (1, 4, 8, 8)
    assert denoise == 1.0
    print("[ok] QwenImageImg2Img: txt2img (no image) -> empty latent sized off width/height, denoise=1.0")


def test_img2img_with_image_uses_strength_as_denoise():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    _, _, _, latent, denoise = node.prepare(model, _clip(), _vae(), "prompt", "", 0.37, 1, 64, 64,
                                            image=torch.rand(1, 64, 64, 3))
    assert denoise == 0.37
    print("[ok] QwenImageImg2Img: img2img (image given) -> denoise = strength")


def test_img2img_model_patch_control_attaches_to_model():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    fake_patch = object()
    result_model, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        image=torch.rand(1, 64, 64, 3),
        qwen_control=qi.QwenImageControl("model_patch", types.SimpleNamespace(model=fake_patch)),
        control_image=torch.rand(1, 64, 64, 3))
    assert result_model is not model  # cloned
    assert len(result_model.double_block_patches) == 1
    assert "control" not in positive[0][1]  # conditioning untouched for this attachment kind
    print("[ok] QwenImageImg2Img: kind='model_patch' attaches via set_model_double_block_patch on a cloned MODEL")


def test_img2img_controlnet_control_attaches_to_conditioning():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    control_net = _FakeControlNet()
    result_model, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        image=torch.rand(1, 64, 64, 3),
        qwen_control=qi.QwenImageControl("controlnet", control_net),
        control_image=torch.rand(1, 64, 64, 3))
    assert result_model is model  # not cloned - conditioning-based attachment, model untouched
    assert positive[0][1]["control"] is control_net
    assert negative[0][1]["control"] is control_net
    assert not result_model.double_block_patches
    print("[ok] QwenImageImg2Img: kind='controlnet' attaches via CONDITIONING, MODEL left untouched")


if __name__ == "__main__":
    test_apply_controlnet_stamps_control_key_on_every_conditioning_item()
    test_apply_controlnet_sets_hint_and_strength()
    test_img2img_rejects_qwen_control_without_control_image()
    test_img2img_ignores_control_image_without_qwen_control()
    test_img2img_txt2img_empty_latent_shape()
    test_img2img_with_image_uses_strength_as_denoise()
    test_img2img_model_patch_control_attaches_to_model()
    test_img2img_controlnet_control_attaches_to_conditioning()
    print("[ok] all nodes_qwen_image smoke tests passed")
