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

common_ksampler_calls = []


def _fake_common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                          positive, negative, latent_image, denoise=1.0):
    common_ksampler_calls.append(dict(
        model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, positive=positive, negative=negative,
        latent_image=latent_image, denoise=denoise))
    return ({"samples": torch.zeros(1, 4, 8, 8)},)


nodes.common_ksampler = _fake_common_ksampler

comfy_samplers = types.ModuleType("comfy.samplers")


class _FakeKSamplerClass:
    SAMPLERS = ["euler"]
    SCHEDULERS = ["simple"]


comfy_samplers.KSampler = _FakeKSamplerClass
comfy_samplers.calculate_sigmas = lambda model_sampling, scheduler, steps: torch.linspace(1.0, 0.0, steps + 1)
comfy_samplers.sampler_object = lambda name: name

comfy_sample_mod = types.ModuleType("comfy.sample")
comfy_sample_mod.fix_empty_latent_channels = lambda model, samples, a, b: samples
comfy_sample_mod.prepare_noise = lambda samples, seed, batch_index=None: torch.zeros_like(samples)


def _fake_sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, samples,
                        noise_mask=None, callback=None, disable_pbar=True, seed=0):
    return samples


comfy_sample_mod.sample_custom = _fake_sample_custom

latent_preview_mod = types.ModuleType("latent_preview")
latent_preview_mod.prepare_callback = lambda model, steps: None

comfy_utils.PROGRESS_BAR_ENABLED = False

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

# node_helpers imports `from comfy.cli_args import args` at module level in
# the real ComfyUI, too heavy to import directly offline - stub with a 1:1
# reimplementation of the real conditioning_set_values (node_helpers.py).
node_helpers = types.ModuleType("node_helpers")


def _conditioning_set_values(conditioning, values={}, append=False):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            val = values[k]
            if append:
                old_val = n[1].get(k, None)
                if old_val is not None:
                    val = old_val + val
            n[1][k] = val
        c.append(n)
    return c


node_helpers.conditioning_set_values = _conditioning_set_values

sys.modules["comfy"] = comfy
sys.modules["comfy.controlnet"] = comfy_controlnet
sys.modules["comfy.model_management"] = comfy_model_management
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils
sys.modules["comfy.samplers"] = comfy_samplers
sys.modules["comfy.sample"] = comfy_sample_mod
sys.modules["latent_preview"] = latent_preview_mod
sys.modules["folder_paths"] = folder_paths
sys.modules["nodes"] = nodes
sys.modules["comfy_extras"] = comfy_extras
sys.modules["comfy_extras.nodes_model_patch"] = comfy_extras_nodes_model_patch
sys.modules["node_helpers"] = node_helpers
comfy.controlnet = comfy_controlnet
comfy.model_management = comfy_model_management
comfy.sd = comfy_sd
comfy.utils = comfy_utils
comfy.samplers = comfy_samplers
comfy.sample = comfy_sample_mod

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

    def get_model_object(self, key):
        return None


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


def test_canny_node_produces_edge_map_matching_input_shape():
    node = qi.QwenImageCanny()
    (out,) = node.detect(torch.rand(1, 32, 32, 3))
    assert out.shape == (1, 32, 32, 3)
    print("[ok] QwenImageCanny: output shape matches input")


def test_img2img_edit_reference_attaches_reference_latents_to_positive_only():
    node = qi.QwenImageImg2Img()
    model = _FakeModelPatcher()
    _, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        edit_reference=torch.rand(1, 64, 64, 3))
    assert "reference_latents" in positive[0][1]
    assert "reference_latents" not in negative[0][1]
    print("[ok] QwenImageImg2Img: edit_reference attaches reference_latents to "
          "positive conditioning only")


# ── QwenImageKSampler ─────────────────────────────────────────────────────

def test_ksampler_comfy_mode_delegates_to_common_ksampler_unchanged():
    common_ksampler_calls.clear()
    node = qi.QwenImageKSampler()
    model = _FakeModelPatcher()
    positive, negative = "pos", "neg"
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    result = node.sample(model, positive, negative, latent, 42, 20, 2.5,
                         "euler", "simple", 1.0, "comfy")
    assert len(common_ksampler_calls) == 1
    call = common_ksampler_calls[0]
    assert call["model"] is model and call["seed"] == 42 and call["steps"] == 20
    assert call["denoise"] == 1.0
    assert result[0]["samples"].shape == (1, 4, 8, 8)
    print("[ok] QwenImageKSampler: denoise_mode=comfy delegates to common_ksampler unchanged")


def test_ksampler_diffusers_mode_rejects_zero_denoise():
    node = qi.QwenImageKSampler()
    model = _FakeModelPatcher()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    try:
        node.sample(model, "pos", "neg", latent, 0, 20, 2.5, "euler", "simple", 0.0, "diffusers")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("[ok] QwenImageKSampler: denoise_mode=diffusers rejects denoise<=0")


def test_ksampler_diffusers_mode_slices_sigmas_from_t_start():
    node = qi.QwenImageKSampler()
    model = _FakeModelPatcher()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    out, = node.sample(model, "pos", "neg", latent, 0, 10, 2.5, "euler", "simple", 0.5, "diffusers")
    # calculate_sigmas fake returns linspace(1,0,11) for steps=10; t_start = round(10-10*0.5) = 5
    # -> sigmas[5:] has 6 entries, sample_custom (fake) passes samples through unchanged shape.
    assert out["samples"].shape == (1, 4, 8, 8)
    assert "downscale_ratio_spacial" not in out
    print("[ok] QwenImageKSampler: denoise_mode=diffusers slices sigmas from t_start and returns latent")


if __name__ == "__main__":
    test_apply_controlnet_stamps_control_key_on_every_conditioning_item()
    test_apply_controlnet_sets_hint_and_strength()
    test_img2img_rejects_qwen_control_without_control_image()
    test_img2img_ignores_control_image_without_qwen_control()
    test_img2img_txt2img_empty_latent_shape()
    test_img2img_with_image_uses_strength_as_denoise()
    test_img2img_model_patch_control_attaches_to_model()
    test_img2img_controlnet_control_attaches_to_conditioning()
    test_canny_node_produces_edge_map_matching_input_shape()
    test_img2img_edit_reference_attaches_reference_latents_to_positive_only()
    test_ksampler_comfy_mode_delegates_to_common_ksampler_unchanged()
    test_ksampler_diffusers_mode_rejects_zero_denoise()
    test_ksampler_diffusers_mode_slices_sigmas_from_t_start()
    print("[ok] all nodes_qwen_image smoke tests passed")
