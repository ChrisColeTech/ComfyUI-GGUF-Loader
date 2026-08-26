"""CPU-only smoke test for nodes_krea2.py.

Stubs the comfy-internal modules nodes_krea2.py imports at the top level so
the pure tensor-prep helpers and the Krea2Img2Img control-LoRA guard rails
can be verified without a running ComfyUI or GPU. Loading real GGUF/LoRA
weights is covered separately (see check_krea2_real.py, run against the
actual portable ComfyUI env - not part of this offline suite).

Usage:  python tools/smoke_krea2.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakePatcherInjection:
    def __init__(self, inject=None, eject=None):
        self.inject = inject
        self.eject = eject


class _FakeWrappersMP:
    DIFFUSION_MODEL = "diffusion_model"


class _FakeCallbacksMP:
    ON_DETACH = "on_detach"
    ON_CLEANUP = "on_cleanup"


comfy = types.ModuleType("comfy")
comfy_ldm = types.ModuleType("comfy.ldm")
comfy_ldm_common_dit = types.ModuleType("comfy.ldm.common_dit")
comfy_ldm_common_dit.pad_to_patch_size = lambda x, patch, **kwargs: x
comfy_ldm_flux = types.ModuleType("comfy.ldm.flux")
comfy_ldm_flux_layers = types.ModuleType("comfy.ldm.flux.layers")
comfy_ldm_flux_layers.timestep_embedding = lambda t, dim: torch.zeros(t.shape[0], dim)
comfy_ldm_flux.layers = comfy_ldm_flux_layers
comfy_model_management = types.ModuleType("comfy.model_management")
comfy_model_management.cast_to_device = lambda t, device, dtype: t.to(device=device, dtype=dtype)
comfy_model_management.intermediate_device = lambda: torch.device("cpu")
comfy_model_management.get_torch_device = lambda: torch.device("cpu")
comfy_patcher_extension = types.ModuleType("comfy.patcher_extension")
comfy_patcher_extension.PatcherInjection = _FakePatcherInjection
comfy_patcher_extension.WrappersMP = _FakeWrappersMP
comfy_patcher_extension.CallbacksMP = _FakeCallbacksMP
comfy_sd = types.ModuleType("comfy.sd")
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.common_upscale = lambda samples, w, h, method, crop: torch.nn.functional.interpolate(
    samples, size=(h, w), mode="bilinear", align_corners=False)
comfy_utils.repeat_to_batch_size = lambda t, b: t.repeat(b, *([1] * (t.dim() - 1))) if t.shape[0] != b else t
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_filename_list = lambda key: []
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
sys.modules["comfy.ldm"] = comfy_ldm
sys.modules["comfy.ldm.common_dit"] = comfy_ldm_common_dit
sys.modules["comfy.ldm.flux"] = comfy_ldm_flux
sys.modules["comfy.ldm.flux.layers"] = comfy_ldm_flux_layers
sys.modules["comfy.model_management"] = comfy_model_management
sys.modules["comfy.patcher_extension"] = comfy_patcher_extension
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils
sys.modules["comfy.samplers"] = comfy_samplers
sys.modules["comfy.sample"] = comfy_sample_mod
sys.modules["latent_preview"] = latent_preview_mod
sys.modules["folder_paths"] = folder_paths
sys.modules["nodes"] = nodes
sys.modules["node_helpers"] = node_helpers
comfy.ldm = comfy_ldm
comfy_ldm.common_dit = comfy_ldm_common_dit
comfy_ldm.flux = comfy_ldm_flux
comfy.model_management = comfy_model_management
comfy.patcher_extension = comfy_patcher_extension
comfy.sd = comfy_sd
comfy.utils = comfy_utils
comfy.samplers = comfy_samplers
comfy.sample = comfy_sample_mod

sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
# Fake the `nodes` subpackage too (pointed at the real nodes/ dir) so importing
# nodes.krea2 doesn't execute the real nodes/__init__.py aggregation, which
# would import every other node module and need a much bigger stub surface.
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
import importlib
krea2 = importlib.import_module("cctech_gguf_pkg.nodes.krea2")  # noqa: E402


class _FakeModelPatcher:
    """Just enough of ModelPatcher for Krea2Img2Img's guard rails."""

    def __init__(self, attachments=None):
        self._attachments = dict(attachments or {})
        self.model_options = {}
        self.model = types.SimpleNamespace()  # no get_model_object/process_latent_in

    def get_attachment(self, key):
        return self._attachments.get(key)

    def get_model_object(self, key):
        raise AttributeError(key)

    def clone(self):
        return _FakeModelPatcher(self._attachments)


# ── pure tensor-prep helpers ─────────────────────────────────────────────

def test_prepare_control_image_grayscale_repeats_to_three_channels():
    img = torch.rand(1, 8, 8, 3)
    out = krea2._prepare_control_image(img, "grayscale", "none", False)
    assert out.shape == (1, 8, 8, 3)
    assert torch.allclose(out[..., 0], out[..., 1]) and torch.allclose(out[..., 1], out[..., 2])
    print("[ok] _prepare_control_image: grayscale repeats to 3 identical channels")


def test_prepare_control_image_minmax_normalizes_to_0_1():
    img = torch.rand(1, 8, 8, 3) * 0.4 + 0.3  # narrow range, not touching 0 or 1
    out = krea2._prepare_control_image(img, "rgb", "per_image_minmax", False)
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6
    assert out.max() > 0.9  # actually stretched to fill the range
    print("[ok] _prepare_control_image: per_image_minmax stretches to [0, 1]")


def test_prepare_control_image_invert_flips_values():
    img = torch.zeros(1, 4, 4, 3)
    out = krea2._prepare_control_image(img, "rgb", "none", True)
    assert torch.allclose(out, torch.ones_like(out))
    print("[ok] _prepare_control_image: invert flips 0 -> 1")


def test_prepare_control_image_single_channel_expands_to_rgb():
    img = torch.rand(1, 4, 4, 1)
    out = krea2._prepare_control_image(img, "rgb", "none", False)
    assert out.shape == (1, 4, 4, 3)
    print("[ok] _prepare_control_image: 1-channel input expands to 3")


def test_resize_image_changes_spatial_dims():
    img = torch.rand(1, 16, 16, 3)
    out = krea2._resize_image(img, 32, 24)
    assert out.shape == (1, 24, 32, 3)
    print("[ok] _resize_image: resizes to the requested width/height")


def test_flatten_temporal_if_needed_merges_batch_and_time():
    x = torch.rand(2, 16, 3, 8, 8)
    out = krea2._flatten_temporal_if_needed(x)
    assert out.shape == (6, 16, 8, 8)
    print("[ok] _flatten_temporal_if_needed: 5D [B,C,T,H,W] -> 4D [B*T,C,H,W]")


def test_flatten_temporal_if_needed_passes_through_4d():
    x = torch.rand(2, 16, 8, 8)
    out = krea2._flatten_temporal_if_needed(x)
    assert out is x
    print("[ok] _flatten_temporal_if_needed: already-4D input passes through unchanged")


def test_strip_known_prefixes():
    assert krea2._strip_known_prefixes("model.diffusion_model.blocks.3.attn1.to_q") == "blocks.3.attn1.to_q"
    assert krea2._strip_known_prefixes("blocks.3.attn1.to_q") == "blocks.3.attn1.to_q"
    print("[ok] _strip_known_prefixes: strips every known checkpoint prefix layer")


def test_target_key_from_lora_base():
    assert krea2._target_key_from_lora_base("model.diffusion_model.blocks.3.attn1.to_q") \
        == "diffusion_model.blocks.3.attn1.to_q.weight"
    assert krea2._target_key_from_lora_base("first") is None
    print("[ok] _target_key_from_lora_base: maps a block LoRA base to its ModelPatcher key")


def test_lora_pairs_finds_matching_up_down():
    sd = {
        "blocks.0.attn1.to_q.lora_down.weight": torch.zeros(4, 8),
        "blocks.0.attn1.to_q.lora_up.weight": torch.zeros(8, 4),
        "blocks.0.attn1.to_q.lora_down.bias": torch.zeros(4),  # not a real pair, no .up match
    }
    pairs = list(krea2._lora_pairs(sd))
    assert len(pairs) == 1
    base, down_key, up_key = pairs[0]
    assert base == "blocks.0.attn1.to_q"
    print("[ok] _lora_pairs: finds exactly the one real down/up pair, ignores the rest")


def test_first_shape_reads_expanded_projection():
    proj = krea2.Krea2ControlInputProjection(torch.zeros(16, 8), image_features=4)
    out_f, img_f, ctrl_f = krea2._first_shape(proj)
    assert (out_f, img_f, ctrl_f) == (16, 4, 4)
    print("[ok] _first_shape: reads out/image/control feature counts off an expanded projection")


def test_lora_expanded_first_weight_key_detects_widened_projection():
    class _FirstModule:
        weight = torch.zeros(16, 8)  # base first layer: out=16, in=8 -> expects a (16, 16) widened weight

    class _ModelWithFirst(_FakeModelPatcher):
        def get_model_object(self, key):
            return _FirstModule()

    model = _ModelWithFirst()
    sd = {"first.weight": torch.zeros(16, 16)}
    key = krea2._lora_expanded_first_weight_key(model, sd)
    assert key == "first.weight"
    print("[ok] _lora_expanded_first_weight_key: detects a real widened-projection weight "
          "(the depth LoRA's case)")


def test_lora_expanded_first_weight_key_returns_none_for_ordinary_lora():
    # get_model_object always raises on the plain fake - detection returns
    # None gracefully, same as an ordinary in-context LoRA with no first key
    # at all (the canny LoRA's case).
    model = _FakeModelPatcher()
    sd = {"blocks.0.attn.wq.lora_down.weight": torch.zeros(4, 4)}
    key = krea2._lora_expanded_first_weight_key(model, sd)
    assert key is None
    print("[ok] _lora_expanded_first_weight_key: returns None for an ordinary LoRA with no "
          "expanded projection (the canny LoRA's case) - lets Krea2ControlLoRALoader "
          "auto-dispatch to the plain-LoRA path instead of raising")


# ── Krea2ControlInputProjection forward ─────────────────────────────────────

def test_control_projection_falls_back_to_original_first_with_no_control_tokens():
    original = torch.nn.Linear(4, 16, bias=False)
    with torch.no_grad():
        original.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(16, 4))
    weight = torch.cat([original.weight, torch.zeros(16, 4)], dim=1)
    proj = krea2.Krea2ControlInputProjection(weight, image_features=4, original_first=original)

    x = torch.rand(1, 5, 4)
    out = proj(x)
    assert torch.allclose(out, original(x))
    print("[ok] Krea2ControlInputProjection: no control tokens -> behaves as the original layer")


def test_control_projection_adds_control_contribution():
    original = torch.nn.Linear(4, 16, bias=False)
    control_weight = torch.rand(16, 4)
    with torch.no_grad():
        original.weight.copy_(torch.rand(16, 4))
    weight = torch.cat([original.weight, control_weight], dim=1)
    proj = krea2.Krea2ControlInputProjection(weight, image_features=4, original_first=original)

    x = torch.rand(1, 5, 4)
    proj.control_tokens = torch.rand(1, 5, 4)
    out = proj(x)
    expected = original(x) + torch.nn.functional.linear(proj.control_tokens, control_weight, None)
    assert torch.allclose(out, expected, atol=1e-5)
    print("[ok] Krea2ControlInputProjection: with control tokens, image half + control half sum "
          "(so an ordinary LoRA on the base 'first' layer still applies)")


# ── Krea2Img2Img guard rails (the "never silently half-configured" checks) ──

def test_img2img_ignores_control_image_without_loaded_lora():
    # No widened input projection is installed, so there's nothing to attach
    # control_image to and no half-configured state - it's safe to ignore
    # (with a warning), not an error. Unlike the reverse case (LoRA loaded,
    # no control_image), which IS a real half-configured-model risk.
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher()
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    result = node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 64, 64,
                          control_image=torch.rand(1, 8, 8, 3))
    assert result is not None
    print("[ok] Krea2Img2Img: control_image with no Control LoRA loaded is ignored, not an error")


def test_img2img_rejects_loaded_lora_without_control_image():
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    try:
        node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 64, 64)
        raised = False
    except ValueError as e:
        raised = "control_image" in str(e)
    assert raised
    print("[ok] Krea2Img2Img: Control LoRA loaded, auto_depth mode, no images or control_image -> raises")


def test_img2img_manual_mode_requires_control_image():
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    try:
        node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 64, 64,
                     images=torch.rand(1, 32, 32, 3), control_mode="manual")
        raised = False
    except ValueError as e:
        raised = "manual" in str(e) and "control_image" in str(e)
    assert raised
    print("[ok] Krea2Img2Img: Control LoRA loaded, manual mode, images given but no "
          "control_image -> raises (manual mode never auto-derives)")


def test_img2img_auto_depth_derives_control_image_from_images():
    # Auto-derivation goes through depth_anything_v2.DepthAnythingV2Detector,
    # which needs real downloaded weights - not appropriate for an offline
    # smoke test. Swap in a fake detector to verify the wiring/control flow
    # (default control_mode="auto_depth", no control_image needed) without
    # touching the real model.
    class _FakeDetector:
        def __init__(self, ckpt_name):
            self.ckpt_name = ckpt_name

        def to(self, device):
            return self

        def estimate(self, np_image, resolution=512):
            return np.zeros_like(np_image)

    original = krea2.depth_anything_v2.DepthAnythingV2Detector
    krea2.depth_anything_v2.DepthAnythingV2Detector = _FakeDetector
    try:
        node = krea2.Krea2Img2Img()
        model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
        clip = types.SimpleNamespace(
            encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
        vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
        result = node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 32, 32,
                              images=torch.rand(1, 32, 32, 3))
        assert result is not None
    finally:
        krea2.depth_anything_v2.DepthAnythingV2Detector = original
    print("[ok] Krea2Img2Img: auto_depth mode (default) derives control_image from "
          "images automatically - one photo, one slot")


def test_img2img_auto_canny_derives_control_image_from_images():
    # cv2.Canny is real, deterministic, no model - no fake needed, unlike auto_depth.
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    result = node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 32, 32,
                          images=torch.rand(1, 32, 32, 3), control_mode="auto_canny")
    assert result is not None
    print("[ok] Krea2Img2Img: auto_canny mode derives control_image from images "
          "automatically via cv2.Canny")


def test_auto_canny_control_image_produces_edge_map_matching_input_shape():
    out = krea2._auto_canny_control_image(torch.rand(1, 32, 32, 3))
    assert out.shape == (1, 32, 32, 3)
    print("[ok] _auto_canny_control_image: output shape matches input")


def test_img2img_control_mode_none_skips_control_even_with_lora_loaded():
    # Before "none" existed, a Control LoRA loaded upstream forced every
    # generation to either auto-derive or manually supply a control_image -
    # there was no way to just skip control for one call. Confirm "none"
    # bypasses that entirely: no raise, no control attached, even though
    # has_control_lora is True.
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    result = node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 32, 32,
                          images=torch.rand(1, 32, 32, 3), control_mode="none")
    assert result is not None
    print("[ok] Krea2Img2Img: control_mode=none skips control attachment even "
          "though a Control LoRA is loaded, instead of raising")


def test_img2img_explicit_control_image_overrides_auto_depth():
    # Even in the default auto_depth mode, an explicitly-connected
    # control_image must win over auto-derivation (e.g. a hand-picked depth
    # map). Prove the detector is never touched when control_image is given.
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("DepthAnythingV2Detector must not be constructed "
                             "when control_image is explicitly connected")

    original = krea2.depth_anything_v2.DepthAnythingV2Detector
    krea2.depth_anything_v2.DepthAnythingV2Detector = _fail_if_called
    try:
        node = krea2.Krea2Img2Img()
        model = _FakeModelPatcher(attachments={krea2.WRAPPER_KEY: {}})
        clip = types.SimpleNamespace(
            encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
        vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
        result = node.prepare(model, clip, vae, "prompt", "", 0.6, 1, 32, 32,
                              images=torch.rand(1, 32, 32, 3),
                              control_image=torch.rand(1, 32, 32, 3))
        assert result is not None
    finally:
        krea2.depth_anything_v2.DepthAnythingV2Detector = original
    print("[ok] Krea2Img2Img: an explicitly-connected control_image overrides "
          "auto_depth derivation")


def test_img2img_txt2img_empty_latent_shape():
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher()
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    _, _, _, latent, denoise = node.prepare(
        model, clip, vae, "prompt", "", 0.6, 1, 64, 64)
    assert latent["samples"].shape == (1, 4, 8, 8)
    assert denoise == 1.0
    print("[ok] Krea2Img2Img: txt2img (no images) -> empty latent sized off width/height, denoise=1.0")


def test_img2img_with_images_uses_strength_as_denoise():
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher()
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(1, 16, 1, 8, 8))
    _, _, _, latent, denoise = node.prepare(
        model, clip, vae, "prompt", "", 0.42, 1, 64, 64, images=torch.rand(1, 64, 64, 3))
    assert denoise == 0.42
    print("[ok] Krea2Img2Img: img2img (images given) -> denoise = strength")


def test_img2img_images_batch_produces_batched_latent():
    # A batch of N images naturally becomes N independent img2img
    # generations via vae.encode()'s own batch handling - no per-image
    # loop needed here, unlike Klein's reference_latents attachment.
    node = krea2.Krea2Img2Img()
    model = _FakeModelPatcher()
    clip = types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: "cond", tokenize=lambda s: s)
    vae = types.SimpleNamespace(encode=lambda img: torch.zeros(img.shape[0], 16, 1, 8, 8))
    _, _, _, latent, denoise = node.prepare(
        model, clip, vae, "prompt", "", 0.5, 1, 64, 64, images=torch.rand(3, 64, 64, 3))
    assert latent["samples"].shape[0] == 3
    print("[ok] Krea2Img2Img: a 3-image batch in `images` produces a batch-3 latent "
          "(3 independent img2img generations)")


# ── Krea2KSampler ─────────────────────────────────────────────────────────

class _FakeSamplerModel:
    """Minimal fake for Krea2KSampler - unlike _FakeModelPatcher, needs a
    working get_model_object("model_sampling") since the diffusers-mode
    path calls it directly (no try/except - the real ModelPatcher always
    has one)."""

    def get_model_object(self, key):
        return object()


def test_ksampler_comfy_mode_delegates_to_common_ksampler_unchanged():
    common_ksampler_calls.clear()
    node = krea2.Krea2KSampler()
    model = _FakeSamplerModel()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    result = node.sample(model, "pos", "neg", latent, 42, 20, 2.5,
                         "euler", "simple", 1.0, "comfy")
    assert len(common_ksampler_calls) == 1
    call = common_ksampler_calls[0]
    assert call["model"] is model and call["seed"] == 42 and call["steps"] == 20
    assert call["denoise"] == 1.0
    assert result[0]["samples"].shape == (1, 4, 8, 8)
    print("[ok] Krea2KSampler: denoise_mode=comfy delegates to common_ksampler unchanged")


def test_ksampler_diffusers_mode_rejects_zero_denoise():
    node = krea2.Krea2KSampler()
    model = _FakeSamplerModel()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    try:
        node.sample(model, "pos", "neg", latent, 0, 20, 2.5, "euler", "simple", 0.0, "diffusers")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("[ok] Krea2KSampler: denoise_mode=diffusers rejects denoise<=0")


def test_ksampler_diffusers_mode_slices_sigmas_from_t_start():
    node = krea2.Krea2KSampler()
    model = _FakeSamplerModel()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    out, = node.sample(model, "pos", "neg", latent, 0, 10, 2.5, "euler", "simple", 0.5, "diffusers")
    assert out["samples"].shape == (1, 4, 8, 8)
    assert "downscale_ratio_spacial" not in out
    print("[ok] Krea2KSampler: denoise_mode=diffusers slices sigmas from t_start and returns latent")


# ── Krea2EditModelPatch / Krea2EditGroundedEncode ────────────────────────

class _FakeKreaEditModel:
    def process_latent_in(self, x):
        return x


class _FakeKreaEditModelPatcher:
    """Minimal stand-in for ModelPatcher - only what Krea2EditModelPatch.patch()
    touches. wrappers/add_wrapper_with_key mirror the real ModelPatcher method
    (comfy/model_patcher.py) that comfy/sampler_helpers.py merges into
    transformer_options at real sampling time."""

    def __init__(self):
        self.model = _FakeKreaEditModel()
        self.model_options = {}
        self.wrappers = {}

    def clone(self):
        c = _FakeKreaEditModelPatcher()
        c.model = self.model
        return c

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


class _FakeKreaEditVAE:
    """Records every encode call so tests can assert on timing/count."""

    def __init__(self):
        self.calls = []

    def encode(self, pixels):  # pixels: (B, H, W, C)
        self.calls.append(tuple(pixels.shape))
        b, h, w, _ = pixels.shape
        return torch.zeros(b, 4, h // 8, w // 8)


def _krea2edit_latent(h=64, w=64):
    return {"samples": torch.zeros(1, 4, h, w)}


def _krea2edit_image(px=512):
    return torch.zeros(1, px, px, 3)


def _krea2edit_get_wrapper(m):
    return m.wrappers["diffusion_model"]["krea2_edit"][0]


def _krea2edit_run_wrapper(wrapper, n=1, h=64, w=64):
    """Drive the wrapper the way a sampler would: same latent shape, n times."""
    seen = []
    for _ in range(n):
        class _Executor:
            class_obj = object()  # stands in for the SingleStreamDiT instance

        seen.append(wrapper(_Executor(), torch.zeros(1, 4, h, w), None, None, None, {}))
    return seen


def test_krea2edit_patch_clones_and_registers_diffusion_model_wrapper():
    node = krea2.Krea2EditModelPatch()
    model = _FakeKreaEditModelPatcher()
    (m,) = node.patch(model, _krea2edit_latent())
    assert m is not model  # cloned
    assert len(m.wrappers["diffusion_model"]["krea2_edit"]) == 1
    print("[ok] Krea2EditModelPatch: patch() clones the model and registers a "
          "DIFFUSION_MODEL wrapper under key 'krea2_edit'")


def test_krea2edit_wrapper_calls_krea2_edit_forward_with_scaled_source():
    node = krea2.Krea2EditModelPatch()
    model = _FakeKreaEditModelPatcher()
    calls = []
    original = krea2.krea2_edit_forward
    krea2.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: calls.append(src) or torch.zeros_like(x)
    try:
        (m,) = node.patch(model, _krea2edit_latent(h=32, w=32))
        wrapper = _krea2edit_get_wrapper(m)
        _krea2edit_run_wrapper(wrapper, h=32, w=32)
    finally:
        krea2.krea2_edit_forward = original
    assert len(calls) == 1
    assert calls[0].shape == (1, 4, 32, 32)  # source_latent scaled via process_latent_in (identity fake)
    print("[ok] Krea2EditModelPatch: wrapper calls krea2_edit_forward with the "
          "process_latent_in-scaled source latent")


def test_krea2edit_target_latent_encodes_before_sampling():
    """target_latent wired -> encode happens during patch(), not in the wrapper -
    the whole point of target_latent (see class docstring: avoids evicting the
    resident diffusion model mid-sampling)."""
    node = krea2.Krea2EditModelPatch()
    model = _FakeKreaEditModelPatcher()
    calls = []
    original = krea2.krea2_edit_forward
    krea2.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: calls.append(src) or torch.zeros_like(x)
    try:
        vae = _FakeKreaEditVAE()
        (m,) = node.patch(model, _krea2edit_latent(), vae=vae, source_image=_krea2edit_image(),
                          target_latent=_krea2edit_latent())
        assert len(vae.calls) == 1, "source should be encoded during patch(), not later"
        wrapper = _krea2edit_get_wrapper(m)
        _krea2edit_run_wrapper(wrapper, n=3)
        assert len(vae.calls) == 1, "sampling must reuse the pre-encode, never re-encode"
    finally:
        krea2.krea2_edit_forward = original
    print("[ok] Krea2EditModelPatch: target_latent pre-encodes the pixel-path "
          "source during patch(), before any sampling step")


def test_krea2edit_without_target_latent_encode_lands_in_wrapper():
    node = krea2.Krea2EditModelPatch()
    model = _FakeKreaEditModelPatcher()
    original = krea2.krea2_edit_forward
    krea2.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: torch.zeros_like(x)
    try:
        vae = _FakeKreaEditVAE()
        (m,) = node.patch(model, _krea2edit_latent(), vae=vae, source_image=_krea2edit_image())
        assert vae.calls == [], "nothing to encode yet without a known target resolution"
        wrapper = _krea2edit_get_wrapper(m)
        _krea2edit_run_wrapper(wrapper)
        assert len(vae.calls) == 1, "encode happens on the first sampling step, then caches"
    finally:
        krea2.krea2_edit_forward = original
    print("[ok] Krea2EditModelPatch: legacy path (no target_latent) - pixel-path "
          "encode happens on the first wrapper call, then is cached")


def test_krea2edit_both_references_are_pre_encoded():
    node = krea2.Krea2EditModelPatch()
    model = _FakeKreaEditModelPatcher()
    calls = []
    original = krea2.krea2_edit_forward
    krea2.krea2_edit_forward = lambda dm, x, t, ctx, src, to, **k: calls.append(src) or torch.zeros_like(x)
    try:
        vae = _FakeKreaEditVAE()
        (m,) = node.patch(model, _krea2edit_latent(), vae=vae, source_image=_krea2edit_image(),
                          source_image_b=_krea2edit_image(), target_latent=_krea2edit_latent())
        assert len(vae.calls) == 2, "scene and subject refs both pre-encoded"
        wrapper = _krea2edit_get_wrapper(m)
        _krea2edit_run_wrapper(wrapper)
        assert len(vae.calls) == 2
        assert isinstance(calls[0], list) and len(calls[0]) == 2
    finally:
        krea2.krea2_edit_forward = original
    print("[ok] Krea2EditModelPatch: source_latent_b/source_image_b -> both "
          "references pre-encoded, passed as a 2-element list to krea2_edit_forward")


def test_krea2edit_forward_pure_noise_shape_roundtrips():
    # No fakes here - exercise the real krea2_edit_forward math against a tiny
    # synthetic SingleStreamDiT-shaped model, confirming the [text | source |
    # target] concatenation and final crop produce the expected output shape.
    patch, channels, tdim, txtlayers, txtdim = 2, 4, 8, 12, 2560
    features = 8  # shared hidden dim `first`/`txtmlp` both project into
    h_tok, w_tok = 4, 4
    H, W = h_tok * patch, w_tok * patch

    class _FakeBlock:
        def __call__(self, combined, tvec, freqs, attn_bias, transformer_options=None):
            return combined

    class _FakeDiT:
        def __init__(self):
            self.patch = patch
            self.channels = channels
            self.tdim = tdim
            self.blocks = [_FakeBlock()]
            self.first = torch.nn.Linear(channels * patch * patch, features)
            self.last = lambda combined, t: torch.zeros(
                combined.shape[0], combined.shape[1], channels * patch * patch)
            self.tmlp = torch.nn.Identity()
            self.tproj = torch.nn.Identity()
            self.txtfusion = lambda ctx, mask=None, transformer_options=None: ctx.mean(dim=2)
            self.txtmlp = torch.nn.Linear(txtdim, features)
            self.pe_embedder = lambda pos: pos

        def _unpack_context(self, context):
            b, seq, fused = context.shape
            return context.reshape(b, seq, txtlayers, txtdim)

    dm = _FakeDiT()
    x = torch.zeros(1, channels, H, W)
    context = torch.zeros(1, 3, txtlayers * txtdim)
    src = torch.zeros(1, channels, H, W)
    out = krea2.krea2_edit_forward(dm, x, torch.zeros(1), context, src, {})
    assert out.shape == (1, channels, H, W)
    print("[ok] krea2_edit_forward: [text | source(frame=1) | target(frame=0)] "
          "concatenation round-trips to the original (B,C,H,W) shape")


def test_krea2edit_grounded_encode_text_only_fallback_without_image():
    node = krea2.Krea2EditGroundedEncode()
    calls = []
    clip = types.SimpleNamespace(
        tokenize=lambda text, **kw: calls.append((text, kw)) or text,
        encode_from_tokens_scheduled=lambda t: "cond")
    out, = node.encode(clip, "a test prompt")
    assert out == "cond"
    assert calls == [("a test prompt", {})]
    print("[ok] Krea2EditGroundedEncode: no image -> plain text-only tokenize, "
          "same as stock CLIPTextEncode")


def test_krea2edit_grounded_encode_with_image_passes_images_and_template():
    node = krea2.Krea2EditGroundedEncode()
    calls = []
    clip = types.SimpleNamespace(
        tokenize=lambda text, **kw: calls.append((text, kw)) or text,
        encode_from_tokens_scheduled=lambda t: "cond")
    node.encode(clip, "recolor the car", image=_krea2edit_image(64))
    assert len(calls) == 1
    text, kw = calls[0]
    assert text == "recolor the car"
    assert len(kw["images"]) == 1
    assert kw["images"][0].shape[0] == 1 and kw["images"][0].shape[-1] == 3
    assert kw["llama_template"].count("<|vision_start|>") == 1
    print("[ok] Krea2EditGroundedEncode: image connected -> clip.tokenize gets "
          "images= and a llama_template with one vision block")


def test_krea2edit_grounded_encode_two_images_two_vision_blocks():
    node = krea2.Krea2EditGroundedEncode()
    calls = []
    clip = types.SimpleNamespace(
        tokenize=lambda text, **kw: calls.append((text, kw)) or text,
        encode_from_tokens_scheduled=lambda t: "cond")
    node.encode(clip, "place the person in the scene",
               image=_krea2edit_image(64), image_b=_krea2edit_image(64))
    text, kw = calls[0]
    assert len(kw["images"]) == 2
    assert kw["llama_template"].count("<|vision_start|>") == 2
    print("[ok] Krea2EditGroundedEncode: image + image_b -> two vision blocks, "
          "training order (scene, subject)")


def test_krea2edit_grounded_encode_downscales_above_grounding_px():
    node = krea2.Krea2EditGroundedEncode()
    calls = []
    clip = types.SimpleNamespace(
        tokenize=lambda text, **kw: calls.append((text, kw)) or text,
        encode_from_tokens_scheduled=lambda t: "cond")
    node.encode(clip, "prompt", image=_krea2edit_image(2048), grounding_px=768)
    _, kw = calls[0]
    assert max(kw["images"][0].shape[1], kw["images"][0].shape[2]) <= 768
    print("[ok] Krea2EditGroundedEncode: grounding_px caps the longest side fed to Qwen3-VL")


if __name__ == "__main__":
    test_prepare_control_image_grayscale_repeats_to_three_channels()
    test_prepare_control_image_minmax_normalizes_to_0_1()
    test_prepare_control_image_invert_flips_values()
    test_prepare_control_image_single_channel_expands_to_rgb()
    test_resize_image_changes_spatial_dims()
    test_flatten_temporal_if_needed_merges_batch_and_time()
    test_flatten_temporal_if_needed_passes_through_4d()
    test_strip_known_prefixes()
    test_target_key_from_lora_base()
    test_lora_pairs_finds_matching_up_down()
    test_first_shape_reads_expanded_projection()
    test_lora_expanded_first_weight_key_detects_widened_projection()
    test_lora_expanded_first_weight_key_returns_none_for_ordinary_lora()
    test_control_projection_falls_back_to_original_first_with_no_control_tokens()
    test_control_projection_adds_control_contribution()
    test_img2img_ignores_control_image_without_loaded_lora()
    test_img2img_rejects_loaded_lora_without_control_image()
    test_img2img_manual_mode_requires_control_image()
    test_img2img_auto_depth_derives_control_image_from_images()
    test_img2img_auto_canny_derives_control_image_from_images()
    test_auto_canny_control_image_produces_edge_map_matching_input_shape()
    test_img2img_control_mode_none_skips_control_even_with_lora_loaded()
    test_img2img_explicit_control_image_overrides_auto_depth()
    test_img2img_txt2img_empty_latent_shape()
    test_img2img_with_images_uses_strength_as_denoise()
    test_img2img_images_batch_produces_batched_latent()
    test_ksampler_comfy_mode_delegates_to_common_ksampler_unchanged()
    test_ksampler_diffusers_mode_rejects_zero_denoise()
    test_ksampler_diffusers_mode_slices_sigmas_from_t_start()
    test_krea2edit_patch_clones_and_registers_diffusion_model_wrapper()
    test_krea2edit_wrapper_calls_krea2_edit_forward_with_scaled_source()
    test_krea2edit_target_latent_encodes_before_sampling()
    test_krea2edit_without_target_latent_encode_lands_in_wrapper()
    test_krea2edit_both_references_are_pre_encoded()
    test_krea2edit_forward_pure_noise_shape_roundtrips()
    test_krea2edit_grounded_encode_text_only_fallback_without_image()
    test_krea2edit_grounded_encode_with_image_passes_images_and_template()
    test_krea2edit_grounded_encode_two_images_two_vision_blocks()
    test_krea2edit_grounded_encode_downscales_above_grounding_px()
    print("[ok] all nodes_krea2 smoke tests passed")
