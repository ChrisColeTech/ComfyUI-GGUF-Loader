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
import importlib
fk = importlib.import_module("cctech_gguf_pkg.nodes_flux_klein")  # noqa: E402


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


if __name__ == "__main__":
    test_img2img_txt2img_uses_flux2_real_empty_latent_shape()
    test_img2img_with_image_uses_strength_as_denoise()
    test_img2img_batch_size_repeats_txt2img_latent()
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
    print("[ok] all nodes_flux_klein smoke tests passed")
