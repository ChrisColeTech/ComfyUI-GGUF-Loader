"""CPU-only smoke test for nodes_flux_klein.py.

Stubs the comfy-internal modules nodes_flux_klein.py imports at the top
level so FluxKleinImg2Img's latent shapes and Flux2KleinMultiReferenceLatent's
reference-list building can be verified without a running ComfyUI or GPU.
Loading real GGUF weights end-to-end, and verifying the real ModelPatcher
attn1_patch hooks used by later parts of this file, are covered separately
against the actual portable ComfyUI environment - not part of this offline
suite.

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


# ── Flux2KleinMultiReferenceLatent ──────────────────────────────────────

def test_multi_reference_latent_splits_batch_into_individual_refs():
    node = fk.Flux2KleinMultiReferenceLatent()
    positive = [[torch.zeros(1, 1, 4), {}]]
    negative = [[torch.zeros(1, 1, 4), {}]]
    batched_latent = {"samples": torch.rand(3, 128, 8, 8)}  # a batch of 3
    pos_out, neg_out = node.apply(positive, negative, batched_latent)
    refs = pos_out[0][1]["reference_latents"]
    assert len(refs) == 3  # split into 3 individual references, not 1
    assert all(r.shape == (1, 128, 8, 8) for r in refs)
    assert pos_out[0][1]["reference_latents_method"] == "index"
    print("[ok] Flux2KleinMultiReferenceLatent: splits a batched latent into "
          "individual references")


def test_multi_reference_latent_combines_multiple_inputs_in_order():
    node = fk.Flux2KleinMultiReferenceLatent()
    positive = [[torch.zeros(1, 1, 4), {}]]
    negative = [[torch.zeros(1, 1, 4), {}]]
    latent_1 = {"samples": torch.zeros(1, 128, 8, 8)}
    latent_2 = {"samples": torch.ones(1, 128, 8, 8)}
    pos_out, _ = node.apply(positive, negative, latent_1, latent_2=latent_2)
    refs = pos_out[0][1]["reference_latents"]
    assert len(refs) == 2
    assert torch.all(refs[0] == 0.0) and torch.all(refs[1] == 1.0)
    print("[ok] Flux2KleinMultiReferenceLatent: combines multiple reference "
          "inputs in connection order")


def test_multi_reference_latent_applies_to_both_positive_and_negative():
    node = fk.Flux2KleinMultiReferenceLatent()
    positive = [[torch.zeros(1, 1, 4), {}]]
    negative = [[torch.zeros(1, 1, 4), {}]]
    latent_1 = {"samples": torch.zeros(1, 128, 8, 8)}
    pos_out, neg_out = node.apply(positive, negative, latent_1)
    assert "reference_latents" in pos_out[0][1]
    assert "reference_latents" in neg_out[0][1]
    print("[ok] Flux2KleinMultiReferenceLatent: attaches to both positive AND "
          "negative conditioning (matches the real example workflow)")


def test_multi_reference_latent_overwrites_not_appends():
    node = fk.Flux2KleinMultiReferenceLatent()
    positive = [[torch.zeros(1, 1, 4), {"reference_latents": ["stale"]}]]
    negative = [[torch.zeros(1, 1, 4), {}]]
    latent_1 = {"samples": torch.zeros(1, 128, 8, 8)}
    pos_out, _ = node.apply(positive, negative, latent_1)
    refs = pos_out[0][1]["reference_latents"]
    assert "stale" not in refs
    assert len(refs) == 1
    print("[ok] Flux2KleinMultiReferenceLatent: overwrites existing "
          "reference_latents rather than appending (matches source behavior)")


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


# ── Part E: remaining simple Klein nodes ────────────────────────────────

class _FakeSimpleModel:
    def __init__(self):
        self.model_options = {}
        self.attn1_patch = None

    def clone(self):
        clone = _FakeSimpleModel()
        clone.model_options = dict(self.model_options)
        return clone

    def set_model_attn1_patch(self, fn):
        self.attn1_patch = fn


def test_color_anchor_inactive_without_reference_latents():
    node = fk.Flux2KleinColorAnchor()
    model = _FakeSimpleModel()
    positive = [[torch.zeros(1, 1, 4), {}]]
    out, = node.apply(model, positive, strength=0.5)
    assert "sampler_post_cfg_function" not in out.model_options
    print("[ok] Flux2KleinColorAnchor: no reference_latents in conditioning -> inactive, no hook registered")


def test_color_anchor_registers_post_cfg_hook_with_reference():
    node = fk.Flux2KleinColorAnchor()
    model = _FakeSimpleModel()
    ref = torch.rand(1, 128, 8, 8)
    positive = [[torch.zeros(1, 1, 4), {"reference_latents": [ref]}]]
    out, = node.apply(model, positive, strength=0.5)
    assert len(out.model_options["sampler_post_cfg_function"]) == 1
    assert model.model_options.get("sampler_post_cfg_function") is None  # clone(), not mutate original
    print("[ok] Flux2KleinColorAnchor: reference present -> registers sampler_post_cfg_function on a clone")


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


def test_detail_controller_uses_klein_sections_when_present():
    node = fk.Flux2KleinDetailController()
    tensor = torch.ones(1, 6, 4)
    cond = [[tensor, {"klein_sections": {"front": (0, 2), "mid": (2, 4), "end": (4, 6)}}]]
    out, = node.control(cond, front_mult=2.0)
    result = out[0][0]
    assert torch.allclose(result[:, 0:2], tensor[:, 0:2] * 2.0)
    assert torch.allclose(result[:, 2:6], tensor[:, 2:6])
    print("[ok] Flux2KleinDetailController: klein_sections metadata drives real front/mid/end ranges")


def test_detail_controller_falls_back_to_fixed_split_without_metadata():
    node = fk.Flux2KleinDetailController()
    tensor = torch.ones(1, 8, 4)
    cond = [[tensor, {}]]
    out, = node.control(cond, front_mult=3.0)
    result = out[0][0]
    # active_end=8 (no attention_mask) -> front = [0:2) (25% of 8)
    assert torch.allclose(result[:, 0:2], tensor[:, 0:2] * 3.0)
    assert torch.allclose(result[:, 2:8], tensor[:, 2:8])
    print("[ok] Flux2KleinDetailController: no klein_sections -> fixed 25/50/25 fallback")


def test_text_enhancer_magnitude_scales_active_region_skipping_bos():
    node = fk.Flux2KleinTextEnhancer()
    tensor = torch.ones(1, 4, 4)
    cond = [[tensor, {}]]
    out, = node.enhance(cond, magnitude=2.0)
    result = out[0][0]
    assert torch.allclose(result[:, 0], tensor[:, 0])  # BOS token untouched
    assert torch.allclose(result[:, 1:], tensor[:, 1:] * 2.0)
    print("[ok] Flux2KleinTextEnhancer: magnitude scales all but the skipped BOS token")


def test_mask_ref_controller_attenuates_black_regions():
    node = fk.Flux2KleinMaskRefController()
    ref = torch.ones(1, 4, 4, 4)
    cond = [[torch.zeros(1, 1, 4), {"reference_latents": [ref]}]]
    mask = torch.zeros(4, 4)  # fully black -> full attenuation at strength=1.0
    out, = node.apply_mask(cond, mask, strength=1.0)
    new_ref = out[0][1]["reference_latents"][0]
    assert torch.allclose(new_ref, torch.zeros_like(new_ref), atol=1e-5)
    print("[ok] Flux2KleinMaskRefController: black mask + strength=1.0 zeroes the reference latent")


def test_mask_ref_controller_noop_without_reference_latents():
    node = fk.Flux2KleinMaskRefController()
    cond = [[torch.zeros(1, 1, 4), {}]]
    mask = torch.ones(4, 4)
    out, = node.apply_mask(cond, mask, strength=1.0)
    assert "reference_latents" not in out[0][1]
    print("[ok] Flux2KleinMaskRefController: no reference_latents -> passthrough, no crash")


def test_ref_latent_controller_registers_attn1_patch():
    node = fk.Flux2KleinRefLatentController()
    model = _FakeSimpleModel()
    positive = [[torch.zeros(1, 1, 4), {}]]
    out_model, out_cond = node.control(model, positive, strength=2.0, reference_index=0)
    assert out_model.attn1_patch is not None
    assert out_cond is positive
    result = out_model.attn1_patch(
        torch.zeros(1, 1, 3, 4), torch.ones(1, 1, 5, 4), torch.ones(1, 1, 5, 4),
        extra_options={"reference_image_num_tokens": [2]})
    assert torch.allclose(result["k"][:, :, -2:, :], torch.full((1, 1, 2, 4), 2.0))
    assert torch.allclose(result["k"][:, :, :-2, :], torch.ones(1, 1, 3, 4))
    print("[ok] Flux2KleinRefLatentController: attn1_patch scales only the addressed reference's K/V range")


def test_text_ref_balance_scales_text_and_reference_oppositely():
    node = fk.Flux2KleinTextRefBalance()
    model = _FakeSimpleModel()
    positive = [[torch.zeros(1, 1, 4), {}]]
    out_model, _ = node.balance_streams(model, positive, balance=0.0)  # text_scale=0, ref_scale=1
    k = torch.ones(1, 1, 6, 4)
    result = out_model.attn1_patch(
        torch.zeros(1, 1, 3, 4), k, k.clone(),
        extra_options={"img_slice": (2, 6), "reference_image_num_tokens": [2]})
    assert torch.allclose(result["k"][:, :, :2, :], torch.zeros(1, 1, 2, 4))  # text zeroed at balance=0
    assert torch.allclose(result["k"][:, :, -2:, :], torch.ones(1, 1, 2, 4))  # ref untouched (ref_scale=1)
    print("[ok] Flux2KleinTextRefBalance: balance=0.0 zeroes text tokens, leaves reference tokens at scale 1")


def test_ref_latent_weight_registers_flat_multiplier():
    node = fk.Flux2KleinRefLatentWeight()
    model = _FakeSimpleModel()
    out_model, = node.execute(model, reference_index=0, weight=3.0)
    assert out_model.attn1_patch is not None
    result = out_model.attn1_patch(
        torch.zeros(1, 1, 2, 4), torch.ones(1, 1, 4, 4), torch.ones(1, 1, 4, 4),
        extra_options={"reference_image_num_tokens": [2]})
    assert torch.allclose(result["k"][:, :, -2:, :], torch.full((1, 1, 2, 4), 3.0))
    print("[ok] Flux2KleinRefLatentWeight: flat weight multiplies only the addressed reference's K/V")


def test_identity_guidance_direct_mode_pulls_toward_reference():
    node = fk.Flux2KleinIdentityGuidance()
    model = _FakeSimpleModel()
    ref_latent = {"samples": torch.ones(1, 4, 8, 8) * 5.0}
    out_model, = node.apply(model, ref_latent, strength=0.5, start_percent=0.0,
                            end_percent=1.0, mode="direct")
    fn = out_model.model_options["sampler_post_cfg_function"][0]
    denoised = torch.zeros(1, 4, 8, 8)
    result = fn({"denoised": denoised, "sigma": torch.tensor([0.5])})
    assert torch.allclose(result, torch.full_like(denoised, 2.5))  # halfway to 5.0
    print("[ok] Flux2KleinIdentityGuidance: direct mode pulls denoised halfway toward the reference at strength=0.5")


def test_identity_guidance_outside_window_is_a_noop():
    node = fk.Flux2KleinIdentityGuidance()
    model = _FakeSimpleModel()
    ref_latent = {"samples": torch.ones(1, 4, 8, 8) * 5.0}
    out_model, = node.apply(model, ref_latent, strength=0.5, start_percent=0.9,
                            end_percent=1.0, mode="direct")
    fn = out_model.model_options["sampler_post_cfg_function"][0]
    denoised = torch.zeros(1, 4, 8, 8)
    result = fn({"denoised": denoised, "sigma": torch.tensor([0.5])})  # progress=0.5, outside [0.9,1.0]
    assert torch.equal(result, denoised)
    print("[ok] Flux2KleinIdentityGuidance: sigma progress outside [start,end] window -> no-op")


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
    test_img2img_no_reference_image_leaves_conditioning_unchanged()
    test_depth_map_node_delegates_to_depth_helper()
    test_multi_reference_latent_splits_batch_into_individual_refs()
    test_multi_reference_latent_combines_multiple_inputs_in_order()
    test_multi_reference_latent_applies_to_both_positive_and_negative()
    test_multi_reference_latent_overwrites_not_appends()
    test_identity_transfer_disabled_is_a_noop()
    test_identity_transfer_enabled_registers_output_patch_only_by_default()
    test_identity_transfer_zero_unmasked_tokens_with_mask_registers_both_hooks()
    test_identity_transfer_preset_overrides_schedule_fields()
    test_identity_transfer_parse_ref_indices_all_and_ranges()
    test_identity_transfer_output_patch_pulls_generated_toward_reference()
    test_identity_transfer_output_patch_noop_without_reference_tokens()
    test_color_anchor_inactive_without_reference_latents()
    test_color_anchor_registers_post_cfg_hook_with_reference()
    test_enhancer_noop_returns_conditioning_unchanged()
    test_enhancer_active_scale_multiplies_active_region()
    test_detail_controller_uses_klein_sections_when_present()
    test_detail_controller_falls_back_to_fixed_split_without_metadata()
    test_text_enhancer_magnitude_scales_active_region_skipping_bos()
    test_mask_ref_controller_attenuates_black_regions()
    test_mask_ref_controller_noop_without_reference_latents()
    test_ref_latent_controller_registers_attn1_patch()
    test_text_ref_balance_scales_text_and_reference_oppositely()
    test_ref_latent_weight_registers_flat_multiplier()
    test_identity_guidance_direct_mode_pulls_toward_reference()
    test_identity_guidance_outside_window_is_a_noop()
    test_sectioned_encoder_emits_klein_sections_with_real_tokenizer()
    test_sectioned_encoder_warns_without_tokenizer_but_still_encodes()
    print("[ok] all nodes_flux_klein smoke tests passed")
