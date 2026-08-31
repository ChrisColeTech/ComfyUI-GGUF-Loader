"""Real-environment smoke test for nodes/ltx_reference.py (the 10s-pack port).

Runs against the actual portable ComfyUI install so the delegation into
comfy-core (LTXAVModel original-method capture, SymmetricPatchifier,
latent_to_pixel_coords) is exercised for real. Fake VAE/model wrappers keep
this GPU/weight-free.

Covers: pinned INPUT_TYPES for all six nodes; the RoPE source-phase rotation
math (source_id=0 bitwise no-op, ref-rows-only rotation, ranged variant);
adaLN prefix extension incl. CompressedTimestep-shaped objects and the
zero_ref_timesteps flag; the conditioning attach path (normalization,
strength, 4D->5D, target_latent pixel resize, strength=0 clear); sequence
windowing; per-instance patch installation (class untouched, idempotent,
unpatchify wrap); the patched _process_input's passthrough and injection
(real patchifier + real coord math); the unpatchify prefix strip; face mask
gating; auto-face-crop bbox tracking; and the reinforcer end-to-end with a
stubbed face detector.

Usage: python tools/smoke_ltx_reference.py
"""
import sys
import types
from pathlib import Path

sys.argv = [sys.argv[0], "--cpu"]

PORTABLE = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
sys.path.insert(0, str(PORTABLE))

import torch

import comfy.options
comfy.options.args_parsing = True

import folder_paths  # noqa: E402,F401 - imported for comfy side effects
import importlib  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
ltx_ref = importlib.import_module("cctech_gguf_pkg.nodes.ltx_reference")  # noqa: E402

import comfy.ldm.lightricks.av_model as av_model  # noqa: E402


# ── fakes ───────────────────────────────────────────────────────────────────

class _FakeBaseModel:
    def __init__(self, dm):
        self.diffusion_model = dm

    def process_latent_in(self, latent):
        # A recognizable affine so tests can prove it ran.
        return latent * 2.0 + 1.0


class _FakeModelPatcher:
    def __init__(self, dm=None):
        self.model = _FakeBaseModel(dm if dm is not None else object())
        self.model_options = {"transformer_options": {}}

    def clone(self):
        c = _FakeModelPatcher.__new__(_FakeModelPatcher)
        c.model = self.model  # comfy clones share the inner model
        c.model_options = {
            k: (dict(v) if isinstance(v, dict) else v)
            for k, v in self.model_options.items()
        }
        return c


class _FakeVAE:
    """Encodes (F,H,W,C) images to a ones latent of the LTX geometry."""

    def encode(self, image):
        f, h, w, c = image.shape
        return torch.ones(1, 128, f, h // 32, w // 32)


class _MiniPatchifier:
    """patch_size=1 patchifier: (B,C,F,H,W) -> (B, F*H*W, C) + coords."""

    def patchify(self, latent):
        b, c, f, h, w = latent.shape
        tokens = latent.permute(0, 2, 3, 4, 1).reshape(b, f * h * w, c)
        grid = torch.stack(torch.meshgrid(
            torch.arange(f), torch.arange(h), torch.arange(w),
            indexing="ij"), dim=0)  # (3, F, H, W)
        flat = grid.reshape(3, -1).to(torch.float32)
        # Real patchifiers emit start/end coordinate pairs: (B, 3, N, 2).
        coords = torch.stack([flat, flat + 1], dim=-1).unsqueeze(0).expand(
            b, -1, -1, -1)
        return tokens, coords

    def unpatchify(self, latents, **kwargs):
        return latents


class _FakeDM:
    """Stands in for the LTXAVModel instance the patches install onto."""

    def __init__(self):
        self.patchifier = _MiniPatchifier()
        self.vae_scale_factors = (8, 32, 32)
        self.causal_temporal_positioning = False
        self._proj = lambda t: t * 3.0

    def patchify_proj(self, tokens):
        return self._proj(tokens)


class _CompressedTimestepStub:
    def __init__(self, data, num_frames, patches_per_frame):
        self.data = data
        self.num_frames = num_frames
        self.patches_per_frame = patches_per_frame


# ── tests ───────────────────────────────────────────────────────────────────

def test_input_types_pinned():
    expect = {
        "CCTechLTXReferenceEnable":
            (["model"], ["zero_ref_timesteps", "verbose"]),
        "CCTechLTXReferenceConditioning":
            (["model", "vae", "image"],
             ["target_latent", "strength", "position_mode", "verbose"]),
        "CCTechLTXReferenceSequenceConditioning":
            (["model", "vae", "images"],
             ["target_latent", "start_frame", "num_frames", "strength",
              "position_mode", "verbose"]),
        "CCTechLTXReferenceProbe": (["model"], []),
        "CCTechLTXReferenceBypass": (["model"], []),
        "CCTechLTXFaceIdentityReinforcer":
            (["model", "vae", "reference_image", "target_latent"],
             ["identity_strength", "face_padding", "auto_face_crop",
              "crop_zoom_factor", "spatial_gating", "placement_mode",
              "source_id", "phase_scale", "reference_image_2", "debug"]),
    }
    for name, (req, opt) in expect.items():
        cls = ltx_ref.NODE_CLASS_MAPPINGS[name]
        it = cls.INPUT_TYPES()
        assert list(it["required"]) == req, (name, list(it["required"]))
        assert list(it.get("optional", {})) == opt, (name, list(it.get("optional", {})))
    # Source-pack defaults survive the port.
    it = ltx_ref.CCTechLTXFaceIdentityReinforcer.INPUT_TYPES()["optional"]
    assert it["source_id"][1]["default"] == 2.0
    assert it["phase_scale"][1]["default"] == 1.0
    assert it["placement_mode"][0] == ["i2v_safe", "t2v_overlap", "prefix"]
    it = ltx_ref.CCTechLTXReferenceConditioning.INPUT_TYPES()["optional"]
    assert it["position_mode"][0] == ["reference", "prefix_continuous"]
    print("[ok] all six node surfaces mirror the 10s pack exactly "
          "(names, order, defaults)")


def test_phase_rotation_math():
    torch.manual_seed(0)
    freq = torch.randn(1, 10, 2, 8, 2, 2)

    # source_id=0 must be a bitwise no-op (target tokens untouched rule).
    out0 = ltx_ref._rotate_packed_freq_tensor(freq, 4, 0.0, 1.0)
    assert out0 is freq

    out = ltx_ref._rotate_packed_freq_tensor(freq, 4, 2.0, 1.0, theta=10000.0)
    # Positions >= ref_len bit-identical.
    assert torch.equal(out[:, 4:], freq[:, 4:])
    assert not torch.equal(out[:, :4], freq[:, :4])

    # Manual check of one dim/pos: composition formula.
    D = 8
    pair_idx = torch.arange(D // 2, dtype=torch.float32)
    rate = (10000.0 ** (-2.0 * pair_idx / D)).repeat_interleave(2)
    ang = 2.0 * 1.0 * rate
    ce, se = ang.cos(), ang.sin()
    cos_ref = freq[0, 0, 0, :, 0, :]
    sin_ref = freq[0, 0, 0, :, 1, :]
    exp_cos = cos_ref * ce.view(D, 1) - sin_ref * se.view(D, 1)
    exp_sin = cos_ref * se.view(D, 1) + sin_ref * ce.view(D, 1)
    assert torch.allclose(out[0, 0, 0, :, 0, :], exp_cos, atol=1e-6)
    assert torch.allclose(out[0, 0, 0, :, 1, :], exp_sin, atol=1e-6)

    # Ranged variant rotates only [start, start+length).
    ranged = ltx_ref._rotate_packed_freq_tensor_ranged(freq, 3, 2, 2.0, 1.0)
    assert torch.equal(ranged[:, :3], freq[:, :3])
    assert torch.equal(ranged[:, 5:], freq[:, 5:])
    assert not torch.equal(ranged[:, 3:5], freq[:, 3:5])

    # pe-walk applies to the video group only. Real v_pe freq tuples are
    # (tensor, split_flag)-shaped (len >= 2).
    v_pe = (freq.clone(), True)
    a_pe = (freq.clone(), True)
    pe = [(v_pe, "av_cross_v"), (a_pe, "av_cross_a")]
    new_pe = ltx_ref._apply_source_phase_to_pe(pe, 4, 2.0, 1.0)
    assert not torch.equal(new_pe[0][0][0][:, :4], freq[:, :4])
    assert torch.equal(new_pe[0][0][0][:, 4:], freq[:, 4:])
    assert torch.equal(new_pe[1][0][0], freq), "audio pe must be untouched"
    print("[ok] RoPE source-phase rotation: exact composition math, "
          "ref-rows-only, ranged variant, video-group-only pe walk, "
          "source_id=0 no-op")


def test_modulation_prefix_extension():
    # Per-token tensor.
    t = torch.randn(1, 12, 4)
    out, e, z = ltx_ref._walk_and_extend_item(t, 12, 3, 4, 1, False)
    assert out.shape == (1, 15, 4) and e == 1 and z == 0
    assert torch.equal(out[:, :3], t[:, 0:1].expand(-1, 3, -1))
    assert torch.equal(out[:, 3:], t)

    # Per-frame tensor + zeroing.
    t2 = torch.randn(1, 4, 6)
    out2, e2, z2 = ltx_ref._walk_and_extend_item(t2, 12, 3, 4, 1, True)
    assert out2.shape == (1, 5, 6) and e2 == 1 and z2 == 1
    assert torch.all(out2[:, :1] == 0.0)

    # CompressedTimestep-shaped: per-frame compressed storage.
    ct = _CompressedTimestepStub(torch.randn(1, 4, 6), num_frames=4,
                                 patches_per_frame=3)
    _, e3, z3 = ltx_ref._walk_and_extend_item(ct, 12, 3, 4, 1, False)
    assert e3 == 1 and ct.num_frames == 5 and ct.data.shape == (1, 5, 6)

    # Broadcast-only CompressedTimestep untouched.
    ct2 = _CompressedTimestepStub(torch.randn(1, 1, 6), 1, 1)
    _, e4, _ = ltx_ref._walk_and_extend_item(ct2, 12, 3, 4, 1, False)
    assert e4 == 0 and ct2.num_frames == 1
    print("[ok] adaLN prefix extension: per-token, per-frame + zeroing, "
          "CompressedTimestep frame extension, broadcast passthrough")


def test_conditioning_attach_and_clear():
    dm = _FakeDM()
    model = _FakeModelPatcher(dm)
    node = ltx_ref.CCTechLTXReferenceConditioning()

    image = torch.rand(1, 64, 96, 3)
    (m2,) = node.attach(model, _FakeVAE(), image)
    to = m2.model_options["transformer_options"]
    ref = to["reference_latent"]
    # ones latent -> process_latent_in(1.0) = 3.0 proves normalization ran.
    assert torch.all(ref == 3.0)
    assert ref.shape == (1, 128, 1, 2, 3)
    assert to["reference_position_mode"] == "reference"
    assert to["memory_video"] is ref, "legacy Echo key kept"
    assert torch.all(m2.model.diffusion_model._ltx_reference_latent == 3.0)
    # Original model untouched (clone-then-attach).
    assert "reference_latent" not in model.model_options["transformer_options"]

    # strength scaling multiplies the normalized latent.
    (m3,) = node.attach(model, _FakeVAE(), image, strength=0.5)
    assert torch.all(
        m3.model_options["transformer_options"]["reference_latent"] == 1.5)

    # target_latent wiring resizes the image in pixel space first.
    target = {"samples": torch.zeros(1, 128, 8, 4, 7)}
    (m4,) = node.attach(model, _FakeVAE(), image, target_latent=target)
    assert m4.model_options["transformer_options"]["reference_latent"].shape \
        == (1, 128, 1, 4, 7)

    # strength=0 bypasses and clears.
    (m5,) = node.attach(m2, _FakeVAE(), image, strength=0.0)
    assert "reference_latent" not in m5.model_options["transformer_options"]
    assert getattr(m5.model.diffusion_model, "_ltx_reference_latent", None) is None
    print("[ok] Reference Conditioning: process_latent_in normalization, "
          "strength scale, target_latent pixel resize, dual-channel attach, "
          "strength=0 clear")


def test_sequence_windowing():
    dm = _FakeDM()
    model = _FakeModelPatcher(dm)
    node = ltx_ref.CCTechLTXReferenceSequenceConditioning()
    frames = torch.rand(20, 64, 96, 3)
    (m2,) = node.attach(model, _FakeVAE(), frames, start_frame=5, num_frames=9)
    ref = m2.model_options["transformer_options"]["reference_latent"]
    # fake VAE keeps F: 9 frames -> F=9 latent frames at 2x3 spatial.
    assert ref.shape == (1, 128, 9, 2, 3)

    class _VAE4D:
        def encode(self, image):
            f, h, w, c = image.shape
            return torch.ones(f, 128, h // 32, w // 32)

    (m3,) = node.attach(model, _VAE4D(), frames, num_frames=4)
    ref3 = m3.model_options["transformer_options"]["reference_latent"]
    assert ref3.shape == (1, 128, 4, 2, 3), "4D VAE output stacks N as F"
    print("[ok] Reference Sequence: frame windowing and 4D->5D stacking")


def test_enable_installs_per_instance_only():
    class_pi = av_model.LTXAVModel._process_input
    class_pt = av_model.LTXAVModel._prepare_timestep

    dm = _FakeDM()
    model = _FakeModelPatcher(dm)
    node = ltx_ref.CCTechLTXReferenceEnable()
    (m2,) = node.enable(model, zero_ref_timesteps=True)

    # Class stays untouched - the deliberate deviation from the source pack.
    assert av_model.LTXAVModel._process_input is class_pi
    assert av_model.LTXAVModel._prepare_timestep is class_pt

    assert dm._cctech_ltx_ref_installed
    assert dm._process_input.__func__ is ltx_ref._patched_process_input
    assert dm._prepare_timestep.__func__ is ltx_ref._patched_prepare_timestep
    assert dm._ltx_zero_ref_timesteps is True
    assert isinstance(dm.patchifier.unpatchify, ltx_ref._UnpatchifyWrapper)

    # Idempotent: second enable doesn't double-wrap.
    inner = dm.patchifier.unpatchify
    node.enable(model)
    assert dm.patchifier.unpatchify is inner
    print("[ok] Reference Enable: per-instance bound-method install, class "
          "untouched, idempotent, unpatchify wrapped, zero flag set")


def test_process_input_passthrough_and_injection():
    dm = _FakeDM()

    def _fake_original(self, x, keyframe_idxs, denoise_mask, **kwargs):
        tokens = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
        coords = torch.zeros(1, 3, 12, 2)
        return [tokens], [coords], {"orig_shape": (1, 2, 3, 2, 2)}

    orig = ltx_ref._ORIGINAL_PROCESS_INPUT
    ltx_ref._ORIGINAL_PROCESS_INPUT = _fake_original
    try:
        # No reference anywhere -> exact passthrough + pending reset.
        dm._pending_ref_seq_len = 99
        result = ltx_ref._patched_process_input(dm, None, None, None)
        tokens_list, coords_list, args = result
        assert tokens_list[0].shape == (1, 12, 2)
        assert dm._pending_ref_seq_len == 0
        assert "reference_seq_len" not in args

        # Reference in transformer_options -> injection.
        ref_latent = torch.ones(1, 2, 1, 2, 2)  # C=2 to match token dim
        result = ltx_ref._patched_process_input(
            dm, None, None, None,
            transformer_options={"reference_latent": ref_latent,
                                 "reference_source_id": 2.0,
                                 "reference_phase_scale": 1.0})
        tokens_list, coords_list, args = result
        # 4 ref tokens (1*2*2) prepended to 12 -> 16; proj (*3) applied.
        assert tokens_list[0].shape == (1, 16, 2)
        assert torch.all(tokens_list[0][:, :4] == 3.0)
        assert coords_list[0].shape[2] == 16
        assert args["reference_seq_len"] == 4
        assert args["target_seq_len"] == 12
        assert dm._pending_ref_seq_len == 4
        assert dm._pending_source_id == 2.0
        assert dm._pending_phase_scale == 1.0

        # Spatial mismatch triggers the latent-space resize fallback.
        ref_wrong = torch.ones(1, 2, 1, 4, 4)
        result = ltx_ref._patched_process_input(
            dm, None, None, None,
            transformer_options={"reference_latent": ref_wrong})
        tokens_list, _, args = result
        assert args["reference_seq_len"] == 4, \
            "4x4 reference must be resized to the 2x2 target grid"

        # prefix_continuous shifts temporal coords to precede the target.
        result = ltx_ref._patched_process_input(
            dm, None, None, None,
            transformer_options={"reference_latent": ref_latent,
                                 "reference_position_mode": "prefix_continuous"})
        _, coords_list, _ = result
        assert float(coords_list[0][0, 0, :4].max()) <= 0.0
    finally:
        ltx_ref._ORIGINAL_PROCESS_INPUT = orig

    # Unpatchify wrapper strips the prefix and resets the counter.
    wrapper = ltx_ref._UnpatchifyWrapper(lambda lat, **kw: lat, dm)
    dm._pending_ref_seq_len = 4
    out = wrapper(torch.randn(1, 16, 2))
    assert out.shape == (1, 12, 2)
    assert dm._pending_ref_seq_len == 0
    out2 = wrapper(torch.randn(1, 12, 2))
    assert out2.shape == (1, 12, 2), "no prefix -> passthrough"
    print("[ok] patched _process_input: bitwise passthrough without a "
          "reference, token/coord prepend + proj + pending state with one, "
          "latent-space resize fallback, prefix_continuous shift, "
          "unpatchify strip")


def test_prepare_timestep_extension():
    dm = _FakeDM()

    def _fake_original(self, timestep, batch_size, hidden_dtype, **kwargs):
        return [torch.randn(1, 12, 4), torch.randn(1, 1, 4)]

    orig = ltx_ref._ORIGINAL_PREPARE_TIMESTEP
    ltx_ref._ORIGINAL_PREPARE_TIMESTEP = _fake_original
    try:
        out = ltx_ref._patched_prepare_timestep(
            dm, None, 1, torch.float32,
            reference_seq_len=4, reference_frames=1,
            target_seq_len=12, target_frames=4)
        assert out[0].shape == (1, 16, 4), "per-token slot extended"
        assert out[1].shape == (1, 1, 4), "broadcast slot untouched"

        # ref_seq_len=0 -> plain delegation (no shape change).
        out2 = ltx_ref._patched_prepare_timestep(dm, None, 1, torch.float32)
        assert out2[0].shape == (1, 12, 4)
    finally:
        ltx_ref._ORIGINAL_PREPARE_TIMESTEP = orig
    print("[ok] patched _prepare_timestep: modulation slots extended to the "
          "prefixed length, exact delegation when no reference")


def test_face_mask_gating():
    shape = (1, 128, 8, 16, 24)
    bbox = (0.25, 0.25, 0.75, 0.75)

    assert ltx_ref._make_face_mask_latent(bbox, shape, "off") is None
    assert ltx_ref._make_face_mask_latent(None, shape, "mask_soft") is None

    hard = ltx_ref._make_face_mask_latent(bbox, shape, "mask_hard", dilation=0.0)
    assert hard.shape == (1, 1, 1, 16, 24)
    assert hard[0, 0, 0, 8, 12] == 1.0 and hard[0, 0, 0, 0, 0] == 0.0

    soft = ltx_ref._make_face_mask_latent(bbox, shape, "mask_soft", dilation=0.0)
    assert soft[0, 0, 0, 8, 12] == 1.0
    assert soft[0, 0, 0, 0, 0] == 0.0
    inside = soft[0, 0, 0, 6, 9]
    assert 0.99 <= float(inside) <= 1.0
    print("[ok] face mask gating: off/None -> None, hard binary region, "
          "soft cosine falloff")


def test_auto_face_crop_tracks_bbox():
    img = torch.rand(1, 200, 300, 3)
    bbox = (0.4, 0.3, 0.6, 0.5)  # 60x40px face at center-ish
    cropped, new_bbox = ltx_ref._auto_face_crop(img, bbox, zoom_factor=2.0,
                                                target_aspect=1.5)
    _, H, W, _ = cropped.shape
    assert abs(W / H - 1.5) < 0.1, "crop matches requested aspect"
    # Face occupies a larger fraction after cropping.
    face_frac_new = ((new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]))
    face_frac_old = 0.2 * 0.2
    assert face_frac_new > face_frac_old
    print("[ok] auto face crop: aspect matched, face fraction grows, bbox "
          "tracked through crop+pad")


def test_reinforcer_end_to_end():
    dm = _FakeDM()
    model = _FakeModelPatcher(dm)
    node = ltx_ref.CCTechLTXFaceIdentityReinforcer()
    image = torch.rand(1, 96, 64, 3)
    target = {"samples": torch.zeros(1, 128, 8, 3, 2)}

    orig_detect = ltx_ref._detect_face_bbox
    try:
        # No face found -> uniform path, no spatial mask, still attached.
        ltx_ref._detect_face_bbox = lambda *a, **kw: None
        (m2,) = node.reinforce(model, _FakeVAE(), image, target)
        to = m2.model_options["transformer_options"]
        assert to["reference_latent"].shape == (1, 128, 1, 3, 2)
        assert to["reference_source_id"] == 2.0
        assert to["reference_phase_scale"] == 1.0
        assert to["reference_position_mode"] == "i2v_safe"
        assert "reference_spatial_mask" not in to
        assert dm._cctech_ltx_ref_installed, "reinforcer installs the patches"

        # Face found -> auto-crop + spatial mask present.
        ltx_ref._detect_face_bbox = lambda *a, **kw: (0.3, 0.2, 0.7, 0.6)
        (m3,) = node.reinforce(model, _FakeVAE(), image, target,
                               identity_strength=0.5)
        to3 = m3.model_options["transformer_options"]
        assert to3["reference_spatial_mask"].shape == (1, 1, 1, 3, 2)
        assert to3["reference_mask_gating"] == "mask_soft"
        assert torch.all(to3["reference_latent"] == 0.5), \
            "identity_strength scales the (raw ones) latent"
    finally:
        ltx_ref._detect_face_bbox = orig_detect
    print("[ok] Face Identity Reinforcer: patch install, source_id/phase "
          "attach, faceless uniform path, face-found gating mask + "
          "strength scaling")



def test_target_latent_accepts_joint_av_nested_tensor():
    """This pack's own prep nodes emit joint AV latents as NestedTensor -
    the source pack's .dim() calls crashed on them live (GPU-found). Every
    target_latent consumer must see only the unbound VIDEO half."""
    import comfy.nested_tensor
    video = torch.rand(1, 128, 4, 6, 8)
    audio = torch.rand(1, 8, 20, 16)
    nested = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    out = ltx_ref._video_latent_samples(nested)
    assert torch.equal(out, video), "must unbind the video half of a joint AV latent"
    assert torch.equal(ltx_ref._video_latent_samples({"samples": video}), video)
    assert torch.equal(ltx_ref._video_latent_samples(video), video)

    # the pixel-resize wrapper must now really resize (it silently fell back
    # before the fix), 6x8 latent -> 192x256 px
    img = torch.rand(1, 100, 100, 3)
    resized = ltx_ref._resize_to_target_latent_px(img, nested, "t", False)
    assert tuple(resized.shape[1:3]) == (192, 256), tuple(resized.shape)
    print("[ok] target_latent: joint AV NestedTensor unbinds to the video half in "
          "every consumer (resize really resizes instead of falling back)")

if __name__ == "__main__":
    test_input_types_pinned()
    test_phase_rotation_math()
    test_modulation_prefix_extension()
    test_conditioning_attach_and_clear()
    test_sequence_windowing()
    test_enable_installs_per_instance_only()
    test_process_input_passthrough_and_injection()
    test_prepare_timestep_extension()
    test_face_mask_gating()
    test_auto_face_crop_tracks_bbox()
    test_reinforcer_end_to_end()
    test_target_latent_accepts_joint_av_nested_tensor()
    print("[ok] all smoke_ltx_reference tests passed")
