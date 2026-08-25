"""CPU-only smoke test for nodes/flux_klein.py.

This file now only covers the 5 genuinely Klein-specific nodes
(FluxKleinModelLoader, FluxKleinImg2Img, Flux2KleinIdentityFeatureTransfer,
Flux2KleinEnhancer, Flux2KleinSectionedEncoder) plus the Flux2KleinDepthMap
backward-compat alias. The 9 architecturally-generic Flux-family reference-
conditioning nodes this file used to also register (Flux2KleinMultiReferenceLatent,
Flux2KleinColorAnchor, Flux2KleinDetailController, Flux2KleinTextEnhancer,
Flux2KleinMaskRefController, Flux2KleinRefLatentController,
Flux2KleinTextRefBalance, Flux2KleinRefLatentWeight, Flux2KleinIdentityGuidance)
moved to the standalone ComfyUI-Flux-Reference-Tools package - see that
package's own tools/smoke_flux_reference.py for coverage of those.

Stubs the comfy-internal modules nodes/flux_klein.py imports at the top
level so FluxKleinImg2Img's latent shapes can be verified without a running
ComfyUI or GPU. Loading real GGUF weights end-to-end, and verifying the
real ModelPatcher attn1_patch hooks Flux2KleinIdentityFeatureTransfer uses,
are covered separately against the actual portable ComfyUI environment -
not part of this offline suite.

Usage:  python tools/smoke_flux_klein.py
"""
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

comfy = types.ModuleType("comfy")
comfy_model_management = types.ModuleType("comfy.model_management")
comfy_model_management.intermediate_device = lambda: torch.device("cpu")
comfy_model_management.get_torch_device = lambda: torch.device("cpu")
comfy_sd = types.ModuleType("comfy.sd")
comfy_utils = types.ModuleType("comfy.utils")
comfy_utils.common_upscale = lambda samples, w, h, method, crop: torch.nn.functional.interpolate(
    samples, size=(h, w), mode="bilinear", align_corners=False)
folder_paths = types.ModuleType("folder_paths")
folder_paths.get_filename_list = lambda key: []
folder_paths.get_full_path_or_raise = lambda key, name: name
folder_paths.get_folder_paths = lambda key: []
folder_paths.models_dir = str(REPO_ROOT / "models")
nodes = types.ModuleType("nodes")
nodes.MAX_RESOLUTION = 16384

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
sys.modules["comfy.model_management"] = comfy_model_management
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils
sys.modules["folder_paths"] = folder_paths
sys.modules["nodes"] = nodes
sys.modules["node_helpers"] = node_helpers
comfy.model_management = comfy_model_management
comfy.sd = comfy_sd
comfy.utils = comfy_utils

sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
# Fake the `nodes` subpackage too (pointed at the real nodes/ dir) so importing
# nodes.flux_klein doesn't execute the real nodes/__init__.py aggregation,
# which would import every other node module and need a much bigger stub
# surface.
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
import importlib
fk = importlib.import_module("cctech_gguf_pkg.nodes.flux_klein")  # noqa: E402


def _clip():
    return types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: [[torch.zeros(1, 1, 4), {}]],
        tokenize=lambda s: s)


def _vae():
    return types.SimpleNamespace(encode=lambda img: torch.zeros(1, 128, 8, 8))


# ── FluxKleinImg2Img ─────────────────────────────────────────────────────

def test_img2img_txt2img_uses_flux2_real_empty_latent_shape():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, _, _, latent, denoise = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 128, 128)
    # width=height=128 -> //16 = 8; 128 channels is Flux.2's real channel
    # count (comfy_extras/nodes_flux.py's EmptyFlux2LatentImage), NOT the
    # generic 4-channel/8-downscale placeholder used for Krea2/Qwen-Image.
    assert latent["samples"].shape == (1, 128, 8, 8)
    assert denoise == 1.0
    print("[ok] FluxKleinImg2Img: txt2img -> empty latent uses Flux.2's real "
          "128-channel/16-downscale shape, denoise=1.0")


def test_img2img_with_image_uses_strength_as_denoise():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, _, _, latent, denoise = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.42, 1, 64, 64, image=torch.rand(1, 64, 64, 3))
    assert denoise == 0.42
    print("[ok] FluxKleinImg2Img: img2img (image given) -> denoise = strength")


def test_img2img_batch_size_repeats_txt2img_latent():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, _, _, latent, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 3, 128, 128)
    assert latent["samples"].shape == (3, 128, 8, 8)
    print("[ok] FluxKleinImg2Img: batch_size repeats the empty latent correctly")


def test_img2img_reference_image_manual_attaches_to_both_conditionings():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        reference_image=torch.rand(1, 64, 64, 3), control_mode="manual")
    assert "reference_latents" in positive[0][1]
    assert "reference_latents" in negative[0][1]
    print("[ok] FluxKleinImg2Img: reference_image (manual) attaches reference_latents "
          "to positive AND negative conditioning")


def test_img2img_reference_image_auto_depth_uses_depth_helper():
    node = fk.FluxKleinImg2Img()
    model = object()
    calls = []
    original = fk._depth_anything_batch
    fk._depth_anything_batch = lambda image, ckpt_name, resolution=512: (
        calls.append((tuple(image.shape), ckpt_name)) or torch.rand(1, 64, 64, 3))
    try:
        _, positive, _, _, _ = node.prepare(
            model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
            reference_image=torch.rand(1, 64, 64, 3), control_mode="auto_depth")
    finally:
        fk._depth_anything_batch = original
    assert len(calls) == 1
    assert "reference_latents" in positive[0][1]
    print("[ok] FluxKleinImg2Img: control_mode=auto_depth runs reference_image through "
          "the depth helper before attaching reference_latents")


def test_img2img_reference_image_auto_canny_derives_edge_map():
    # auto_canny needs no model download (plain cv2.Canny) - exercise it
    # for real, confirming FluxKleinImg2Img actually attaches reference_latents.
    node = fk.FluxKleinImg2Img()
    model = object()
    _, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        reference_image=torch.rand(1, 64, 64, 3), control_mode="auto_canny")
    assert "reference_latents" in positive[0][1]
    assert "reference_latents" in negative[0][1]
    print("[ok] FluxKleinImg2Img: control_mode=auto_canny derives an edge map and "
          "attaches reference_latents")


def test_img2img_control_mode_none_skips_reference_attachment_even_if_connected():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64,
        reference_image=torch.rand(1, 64, 64, 3), control_mode="none")
    assert "reference_latents" not in positive[0][1]
    assert "reference_latents" not in negative[0][1]
    print("[ok] FluxKleinImg2Img: control_mode=none skips reference attachment even "
          "though reference_image is connected")


def test_control_modes_are_the_minimum_set():
    assert fk._CONTROL_MODES == ["manual", "auto_depth", "auto_canny", "none"]
    print("[ok] FluxKleinImg2Img: control_mode is the minimum set (manual/auto_depth/"
          "auto_canny/none), matching Krea2Img2Img/QwenImageImg2Img exactly")


def test_img2img_no_reference_image_leaves_conditioning_unchanged():
    node = fk.FluxKleinImg2Img()
    model = object()
    _, positive, negative, _, _ = node.prepare(
        model, _clip(), _vae(), "prompt", "", 0.6, 1, 64, 64)
    assert "reference_latents" not in positive[0][1]
    assert "reference_latents" not in negative[0][1]
    print("[ok] FluxKleinImg2Img: no reference_image -> no reference_latents attached")


# ── Flux2KleinDepthMap (backward-compat alias -> shared DepthMap) ───────

def test_depth_map_node_delegates_to_depth_helper():
    # Flux2KleinDepthMap's own logic moved to the shared preprocessors.
    # DepthMap - confirm the "Flux2KleinDepthMap" NODE_CLASS_MAPPINGS alias
    # resolves to it, and that it delegates to the shared depth helper.
    from cctech_gguf_pkg.nodes import preprocessors as pp
    assert fk.NODE_CLASS_MAPPINGS["Flux2KleinDepthMap"] is pp.DepthMap
    node = pp.DepthMap()
    calls = []
    original = pp._depth_anything_batch
    pp._depth_anything_batch = lambda image, ckpt_name, resolution=512: (
        calls.append((tuple(image.shape), ckpt_name, resolution)) or torch.zeros_like(image))
    try:
        out, = node.estimate(torch.rand(2, 32, 32, 3), ckpt_name="depth_anything_v2_vits.pth",
                             resolution=256)
    finally:
        pp._depth_anything_batch = original
    assert calls == [((2, 32, 32, 3), "depth_anything_v2_vits.pth", 256)]
    assert out.shape == (2, 32, 32, 3)
    print("[ok] Flux2KleinDepthMap alias -> shared DepthMap node: delegates to the shared "
          "depth helper with the given ckpt_name/resolution")


# ── Flux2KleinIdentityFeatureTransfer ────────────────────────────────────

class _FakeIdentityModel:
    def __init__(self):
        self.attn1_patch = None
        self.attn1_output_patch = None

    def clone(self):
        return self

    def set_model_attn1_patch(self, fn):
        self.attn1_patch = fn

    def set_model_attn1_output_patch(self, fn):
        self.attn1_output_patch = fn


def test_identity_transfer_disabled_is_a_noop():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    model = _FakeIdentityModel()
    out, = node.apply(model, enabled=False)
    assert out.attn1_output_patch is None
    assert out.attn1_patch is None
    print("[ok] Flux2KleinIdentityFeatureTransfer: enabled=False registers no hooks")


def test_identity_transfer_enabled_registers_output_patch_only_by_default():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    model = _FakeIdentityModel()
    out, = node.apply(model, enabled=True, mask_behavior="focus_only")
    assert out.attn1_output_patch is not None
    assert out.attn1_patch is None  # no masks wired, focus_only behavior
    print("[ok] Flux2KleinIdentityFeatureTransfer: enabled registers "
          "attn1_output_patch; attn1_patch only wired for zero_unmasked_tokens+mask")


def test_identity_transfer_zero_unmasked_tokens_with_mask_registers_both_hooks():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    model = _FakeIdentityModel()
    mask = torch.ones(32, 32)
    out, = node.apply(model, enabled=True, mask_behavior="zero_unmasked_tokens",
                       subject_mask_1=mask)
    assert out.attn1_output_patch is not None
    assert out.attn1_patch is not None
    print("[ok] Flux2KleinIdentityFeatureTransfer: zero_unmasked_tokens + wired "
          "mask registers both attn1_patch and attn1_output_patch")


def test_identity_transfer_preset_overrides_schedule_fields():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    double_map = node._parse_schedule(node.PRESETS["HARD_LOCK"]["double_blocks"], 7)
    assert double_map == {i: 0.55 for i in range(8)}
    print("[ok] Flux2KleinIdentityFeatureTransfer: HARD_LOCK double_blocks schedule "
          "parses to strength 0.55 across blocks 0-7")


def test_identity_transfer_parse_ref_indices_all_and_ranges():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    assert node._parse_ref_indices("all", 4) == [0, 1, 2, 3]
    assert node._parse_ref_indices("0,2", 4) == [0, 2]
    assert node._parse_ref_indices("1-3", 5) == [1, 2, 3]
    print("[ok] Flux2KleinIdentityFeatureTransfer: reference_indices parses "
          "'all', comma lists, and ranges")


def test_identity_transfer_output_patch_pulls_generated_toward_reference():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    model = _FakeIdentityModel()
    out, = node.apply(model, enabled=True, preset="HARD_LOCK")
    # 1 text token, 2 generated tokens along +/-axis0, 2 reference tokens
    # along the same +/-axis0 but larger magnitude - after per-set
    # centering, gen token 1 matches ref token 1's direction (cosine ~1)
    # and gen token 2 matches ref token 2's direction, so the transfer
    # should pull each generated token toward its matching reference.
    attn = torch.zeros(1, 5, 8)
    attn[0, 1] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])   # gen token 1
    attn[0, 2] = torch.tensor([-1.0, 0, 0, 0, 0, 0, 0, 0])  # gen token 2
    attn[0, 3] = torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 0])  # ref token 1
    attn[0, 4] = torch.tensor([-10.0, 0, 0, 0, 0, 0, 0, 0])  # ref token 2
    extra_options = {
        "reference_image_num_tokens": [2],
        "img_slice": (1, 5),
        "block_type": "double",
        "block_index": 0,
    }
    result = out.attn1_output_patch(attn, extra_options)
    assert result.shape == attn.shape
    assert not torch.equal(result[0, 1:3], attn[0, 1:3])  # generated tokens moved
    assert torch.equal(result[0, 3:5], attn[0, 3:5])  # reference tokens untouched
    assert torch.equal(result[0, 0:1], attn[0, 0:1])  # text token untouched
    print("[ok] Flux2KleinIdentityFeatureTransfer: output_patch pulls only the "
          "generated-image token range toward the reference bank")


def test_identity_transfer_output_patch_noop_without_reference_tokens():
    node = fk.Flux2KleinIdentityFeatureTransfer()
    model = _FakeIdentityModel()
    out, = node.apply(model, enabled=True)
    attn = torch.rand(1, 4, 8)
    result = out.attn1_output_patch(attn, {"reference_image_num_tokens": [], "img_slice": None})
    assert torch.equal(result, attn)
    print("[ok] Flux2KleinIdentityFeatureTransfer: output_patch is a no-op when "
          "extra_options carries no reference tokens (e.g. non-Klein/non-Flux2 model)")


# ── Flux2KleinEnhancer ───────────────────────────────────────────────────

def test_enhancer_noop_returns_conditioning_unchanged():
    node = fk.Flux2KleinEnhancer()
    cond = [[torch.rand(1, 4, 12), {}]]
    out, = node.enhance(cond)
    assert out is cond
    print("[ok] Flux2KleinEnhancer: all-neutral params -> passthrough, no tensor copy")


def test_enhancer_active_scale_multiplies_active_region():
    node = fk.Flux2KleinEnhancer()
    tensor = torch.ones(1, 4, 12)
    cond = [[tensor, {}]]
    out, = node.enhance(cond, active_scale=2.0)
    assert torch.allclose(out[0][0], tensor * 2.0)
    print("[ok] Flux2KleinEnhancer: active_scale=2.0 doubles the (fully-active) conditioning")


# ── Flux2KleinSectionedEncoder ───────────────────────────────────────────

def test_sectioned_encoder_emits_klein_sections_with_real_tokenizer():
    node = fk.Flux2KleinSectionedEncoder()

    class _FakeHFTokenizer:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": text.split() if text else []}

    class _FakeSub:
        tokenizer = _FakeHFTokenizer()

    class _FakeTokenizerHolder:
        qwen3_4b = _FakeSub()

    class _FakeClip:
        tokenizer = _FakeTokenizerHolder()

        def tokenize(self, text):
            return text

        def encode_from_tokens(self, tokens, return_pooled=True):
            return torch.zeros(1, 4, 8), torch.zeros(1, 8)

    cond, front, mid, end, full = node.encode_sectioned(
        _FakeClip(), front_text="a b", mid_text="c", end_text="d e f", separator="comma")
    ranges = cond[0][1]["klein_sections"]
    assert set(ranges.keys()) == {"front", "mid", "end"}
    assert front == "a b" and mid == "c" and end == "d e f"
    print("[ok] Flux2KleinSectionedEncoder: emits klein_sections when the HF tokenizer path resolves")


def test_sectioned_encoder_warns_without_tokenizer_but_still_encodes():
    node = fk.Flux2KleinSectionedEncoder()

    class _FakeClipNoTokenizer:
        tokenizer = None

        def tokenize(self, text):
            return text

        def encode_from_tokens(self, tokens, return_pooled=True):
            return torch.zeros(1, 4, 8), torch.zeros(1, 8)

    cond, *_ = node.encode_sectioned(_FakeClipNoTokenizer(), front_text="a")
    assert "klein_sections" not in cond[0][1]
    print("[ok] Flux2KleinSectionedEncoder: no HF tokenizer path -> still encodes, just no klein_sections metadata")


if __name__ == "__main__":
    test_img2img_txt2img_uses_flux2_real_empty_latent_shape()
    test_img2img_with_image_uses_strength_as_denoise()
    test_img2img_batch_size_repeats_txt2img_latent()
    test_img2img_reference_image_manual_attaches_to_both_conditionings()
    test_img2img_reference_image_auto_depth_uses_depth_helper()
    test_img2img_reference_image_auto_canny_derives_edge_map()
    test_img2img_control_mode_none_skips_reference_attachment_even_if_connected()
    test_control_modes_are_the_minimum_set()
    test_img2img_no_reference_image_leaves_conditioning_unchanged()
    test_depth_map_node_delegates_to_depth_helper()
    test_identity_transfer_disabled_is_a_noop()
    test_identity_transfer_enabled_registers_output_patch_only_by_default()
    test_identity_transfer_zero_unmasked_tokens_with_mask_registers_both_hooks()
    test_identity_transfer_preset_overrides_schedule_fields()
    test_identity_transfer_parse_ref_indices_all_and_ranges()
    test_identity_transfer_output_patch_pulls_generated_toward_reference()
    test_identity_transfer_output_patch_noop_without_reference_tokens()
    test_enhancer_noop_returns_conditioning_unchanged()
    test_enhancer_active_scale_multiplies_active_region()
    test_sectioned_encoder_emits_klein_sections_with_real_tokenizer()
    test_sectioned_encoder_warns_without_tokenizer_but_still_encodes()
    print("[ok] all nodes_flux_klein smoke tests passed")
