"""Real-environment smoke test for LTXV23VidToVideo / LTXV23CropVideoGuide.

Runs against the actual portable ComfyUI install so the new code's delegation
into comfy-core's real comfy_extras.nodes_lt.LTXVAddGuide is exercised for
real (not stubbed) - that's the one piece genuinely new here, everything
else (audio-hold helpers, joint-latent convention, the image-hold path) is
already-proven code reused from LTXV23ImgToVideo. Fake VAE/CLIP objects
keep this GPU/weight-free (CPU only, tiny tensors) while still running the
real RoPE/keyframe math.

Usage: python tools/smoke_ltx23_vid2vid.py
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--cpu"]

PORTABLE = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
sys.path.insert(0, str(PORTABLE))

import types

import torch

import comfy.options
comfy.options.args_parsing = True

import folder_paths  # noqa: E402
folder_paths.folder_names_and_paths.setdefault("unet", ([], set()))

import importlib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
# Only the ltx23 submodule, not the package's own __init__.py aggregation -
# that would import every other node file (gguf, krea2, ...) for no reason.
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
ltx23 = importlib.import_module("cctech_gguf_pkg.nodes.ltx23")  # noqa: E402

import comfy_extras.nodes_lt as nodes_lt  # noqa: E402 - real core module, used directly below


class _FakeFSM:
    latent_channels = 8
    latent_frequency_bins = 16

    def num_of_latents_from_frames(self, length, frame_rate):
        return max(1, int(length / frame_rate * 10))


class _FakeAudioVAE:
    def __init__(self):
        self.first_stage_model = _FakeFSM()
        self.latent_channels = 8


class _FakeVideoVAE:
    downscale_index_formula = (8, 32, 32)

    def encode(self, pixels):
        f, h, w, c = pixels.shape
        t = (f - 1) // 8 + 1
        return torch.zeros(1, 128, t, h // 32, w // 32)


class _FakeComponents:
    def __init__(self, images, audio, frame_rate):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate


class _FakeVideo:
    def __init__(self, n_frames=25, h=256, w=448, fps=24.0):
        self.images = torch.rand(n_frames, h, w, 3)
        self.fps = fps

    def get_components(self):
        return _FakeComponents(self.images, None, self.fps)


def _fake_clip():
    # Real comfy CONDITIONING shape: a list of [tensor, dict] pairs -
    # node_helpers.conditioning_set_values() indexes t[0]/t[1].copy() on it.
    return types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: [[torch.zeros(1, 1, 8), {}]],
        tokenize=lambda s, **kw: s)


def _target_t(n_frames):
    return ((ltx23._align_length(n_frames) - 1) // 8) + 1


class _FakeLoraLoaderModelOnly:
    """Stands in for comfy-core's real nodes.LoraLoaderModelOnly - records
    calls and returns a distinguishable sentinel per lora_name so tests can
    prove chaining/order without needing real lora files on disk."""
    calls = []

    def load_lora_model_only(self, model, lora_name, strength_model):
        _FakeLoraLoaderModelOnly.calls.append((model, lora_name, strength_model))
        return (("lora-patched", lora_name, model),)


def test_ic_lora_appends_and_crops_cleanly():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    _FakeLoraLoaderModelOnly.calls = []
    try:
        out_model, positive, negative, latent, frame_rate = node.prepare(
            "base-model", clip, vae, audio_vae, "v2v", "prompt", "", 448, 256, 121, 24.0, 1,
            video=video, ic_lora="beard_removal.safetensors", ic_lora_strength=0.8,
            keep_original_audio=False)
    finally:
        ltx23.nodes.LoraLoaderModelOnly = orig_loader

    assert out_model == ("lora-patched", "beard_removal.safetensors", "base-model"), (
        "ic_lora set -> model must come back through the real LoraLoaderModelOnly delegation")
    assert _FakeLoraLoaderModelOnly.calls == [("base-model", "beard_removal.safetensors", 0.8)]
    assert frame_rate == 24.0  # taken from the (fake) video, overriding the 24.0 widget too
    samples = latent["samples"]
    assert samples.is_nested
    video_latent, audio_latent = samples.unbind()

    target_t = _target_t(25)
    assert video_latent.shape[0] == 1 and video_latent.shape[1] == 128
    assert video_latent.shape[2] > target_t  # guide frames appended on top

    _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, video_latent.shape)
    assert num_keyframes > 0
    assert video_latent.shape[2] - num_keyframes == target_t

    crop_node = ltx23.LTXV23CropVideoGuide()
    cpos, cneg, cropped = crop_node.crop(positive, negative, latent)
    cropped_video, cropped_audio = cropped["samples"].unbind()
    assert cropped_video.shape[2] == target_t
    assert torch.equal(cropped_audio, audio_latent)

    _, num_keyframes_after = nodes_lt.get_keyframe_idxs(cpos, cropped_video.shape)
    assert num_keyframes_after == 0
    print("[ok] LTXV23VidToVideo: ic_lora set loads the LoRA via comfy-core's real "
          "LoraLoaderModelOnly (not reimplemented) and appends real IC-LoRA guide frames "
          "(real comfy_extras.nodes_lt.LTXVAddGuide), taking length/frame_rate from the "
          "clip; LTXV23CropVideoGuide removes exactly them back off")


def test_video_with_ic_lora_none_is_plain_shape():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    out_model, positive, negative, latent, _ = node.prepare(
        None, clip, vae, audio_vae, "v2v", "prompt", "", 448, 256, 121, 24.0, 1,
        video=video, ic_lora="none", keep_original_audio=False)

    assert out_model is None  # ic_lora=none -> model passed through unchanged, no lora load
    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    assert video_latent.shape[2] == _target_t(25)  # no guide appended

    crop_node = ltx23.LTXV23CropVideoGuide()
    _, _, cropped = crop_node.crop(positive, negative, latent)
    assert cropped is latent  # no-op passthrough, nothing to crop
    print("[ok] LTXV23VidToVideo: ic_lora=none -> plain shape (video only used for "
          "length/frame_rate), no LoraLoaderModelOnly call, LTXV23CropVideoGuide is a "
          "no-op passthrough")


def test_image_only_is_ordinary_i2v_hold_no_video_needed():
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)

    out_model, positive, negative, latent, frame_rate = node.prepare(
        None, clip, vae, audio_vae, "i2v", "prompt", "", 448, 256, 121, 30.0, 1,
        images=image, image_strength=0.7)

    assert out_model is None
    assert frame_rate == 30.0  # widget value used - no video connected to override it
    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    assert video_latent.shape[2] == _target_t(121)  # widget length, not any clip's

    noise_mask = latent["noise_mask"].unbind()[0]
    assert torch.allclose(noise_mask[:, :, 0], torch.tensor(1.0 - 0.7))
    assert torch.allclose(noise_mask[:, :, 1], torch.tensor(1.0))  # rest of the clip untouched

    crop_node = ltx23.LTXV23CropVideoGuide()
    _, _, cropped = crop_node.crop(positive, negative, latent)
    assert cropped is latent  # no ic_lora path ever ran -> nothing to crop
    print("[ok] LTXV23VidToVideo: mode=i2v, no video connected -> ordinary i2v "
          "first-frame hold, width/height/length/frame_rate all from the widgets")


def test_image_and_ic_lora_combine_independently():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)

    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    try:
        _, positive, negative, latent, _ = node.prepare(
            "base-model", clip, vae, audio_vae, "i2v", "prompt", "", 448, 256, 121, 24.0, 1,
            images=image, image_strength=0.7, video=video, ic_lora="task.safetensors",
            keep_original_audio=False)
    finally:
        ltx23.nodes.LoraLoaderModelOnly = orig_loader

    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    target_t = _target_t(25)  # video's own length wins over the length widget
    assert video_latent.shape[2] > target_t  # guide frames still appended on top

    noise_mask = latent["noise_mask"].unbind()[0]
    assert torch.allclose(noise_mask[:, :, 0], torch.tensor(1.0 - 0.7))  # image hold still applied

    _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, video_latent.shape)
    assert num_keyframes > 0
    print("[ok] LTXV23VidToVideo: mode=i2v with video also connected - image (i2v hold) "
          "and video (IC-LoRA guide) combine in the same call without interfering with "
          "each other")


def test_mode_validates_required_socket_is_connected():
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)
    video = _FakeVideo()

    try:
        node.prepare(None, clip, vae, audio_vae, "i2v", "prompt", "", 448, 256, 121, 24.0, 1)
        assert False, "expected ValueError for mode=i2v with no image connected"
    except ValueError as e:
        assert "image" in str(e)

    try:
        node.prepare(None, clip, vae, audio_vae, "v2v", "prompt", "", 448, 256, 121, 24.0, 1)
        assert False, "expected ValueError for mode=v2v with no video connected"
    except ValueError as e:
        assert "video" in str(e)

    try:
        node.prepare(None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
                     images=image)
        assert False, "expected ValueError for mode=t2v with image connected"
    except ValueError as e:
        assert "t2v" in str(e)

    try:
        node.prepare(None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
                     video=video)
        assert False, "expected ValueError for mode=t2v with video connected"
    except ValueError as e:
        assert "t2v" in str(e)
    print("[ok] LTXV23VidToVideo: mode is validated against what's actually connected - "
          "i2v/v2v demand their required socket, t2v rejects either being connected")


def test_mode_t2v_allows_image_when_editanything_active():
    # image is dual-purpose: an i2v hold under i2v/v2v, but ALSO the
    # EditAnything reference photo when editanything_module_path is set -
    # under mode=t2v specifically, image connected must be rejected UNLESS
    # EditAnything is active, in which case it's legitimate (a pure
    # "inject this identity, no first-frame hold" recipe).
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)

    orig_apply = ltx23._apply_editanything_patch
    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23._apply_editanything_patch = lambda *a, **kw: "patched"
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    try:
        out_model, *_ = node.prepare(
            None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
            images=image, image_strength=0.0,
            editanything_lora="standard.safetensors", editanything_module_path="module.safetensors")
    finally:
        ltx23._apply_editanything_patch = orig_apply
        ltx23.nodes.LoraLoaderModelOnly = orig_loader

    assert out_model == "patched"
    print("[ok] LTXV23VidToVideo: mode=t2v + image connected is accepted when "
          "editanything_module_path is set (image doubling as the EditAnything reference, "
          "not an i2v hold) - still rejected under t2v without EditAnything active")


def test_editanything_lora_and_module_path_must_be_set_together():
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    try:
        node.prepare(None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
                     editanything_lora="standard.safetensors")
        assert False, "expected ValueError: editanything_lora set but editanything_module_path isn't"
    except ValueError as e:
        assert "together" in str(e)

    try:
        node.prepare(None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
                     editanything_module_path="module.safetensors")
        assert False, "expected ValueError: editanything_module_path set but editanything_lora isn't"
    except ValueError as e:
        assert "together" in str(e)

    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    try:
        node.prepare(None, clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
                     editanything_lora="standard.safetensors",
                     editanything_module_path="module.safetensors")
        assert False, "expected ValueError for missing image (doubles as the reference photo)"
    except ValueError as e:
        assert "image" in str(e)
    finally:
        ltx23.nodes.LoraLoaderModelOnly = orig_loader
    print("[ok] LTXV23VidToVideo: editanything_lora/editanything_module_path must both be "
          "\"none\" or both set - either alone raises a clear error instead of doing "
          "something inert or half-configured")


def test_editanything_selectors_chain_lora_then_patch_helper():
    # The patch mechanism itself (block residual, ref_attn wiring, per-image
    # batching) is already covered by smoke_ltx23_editanything.py - this only
    # proves LTXV23VidToVideo's selectors route into comfy-core's real
    # LoraLoaderModelOnly for the LoRA half, THEN the same shared
    # _apply_editanything_patch helper for the module half (not
    # reimplementations), in the right order, with the right args - and
    # that `image` (not a separate reference_image slot) is what's passed
    # through as the reference photo.
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(2, 256, 448, 3)
    sentinel_patched_model = object()
    patch_calls = []

    def fake_apply(model, vae_arg, reference_image, module_path, reference_mode):
        patch_calls.append((model, reference_image.shape[0], module_path, reference_mode))
        return sentinel_patched_model

    orig_apply = ltx23._apply_editanything_patch
    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23._apply_editanything_patch = fake_apply
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    _FakeLoraLoaderModelOnly.calls = []
    try:
        out_model, *_ = node.prepare(
            "base-model", clip, vae, audio_vae, "t2v", "prompt", "", 448, 256, 121, 24.0, 1,
            images=image, image_strength=0.0,
            editanything_lora="standard.safetensors", editanything_lora_strength=0.9,
            editanything_module_path="module.safetensors", reference_mode="per_batch_item")
    finally:
        ltx23._apply_editanything_patch = orig_apply
        ltx23.nodes.LoraLoaderModelOnly = orig_loader

    assert out_model is sentinel_patched_model, (
        "editanything_module_path set -> must return the PATCHED model, not the original")
    assert _FakeLoraLoaderModelOnly.calls == [("base-model", "standard.safetensors", 0.9)]
    assert len(patch_calls) == 1
    called_model, ref_count, module_path, mode = patch_calls[0]
    assert called_model == ("lora-patched", "standard.safetensors", "base-model"), (
        "the module patch must apply to the LoRA-loaded model, not the original")
    assert ref_count == 2
    assert module_path == "module.safetensors"
    assert mode == "per_batch_item"
    print("[ok] LTXV23VidToVideo: editanything_lora/editanything_module_path route into "
          "comfy-core's real LoraLoaderModelOnly THEN the shared _apply_editanything_patch "
          "helper, in that order, with the LoRA-loaded model (not the original) passed "
          "through to the patch")


if __name__ == "__main__":
    test_ic_lora_appends_and_crops_cleanly()
    test_video_with_ic_lora_none_is_plain_shape()
    test_image_only_is_ordinary_i2v_hold_no_video_needed()
    test_image_and_ic_lora_combine_independently()
    test_mode_validates_required_socket_is_connected()
    test_mode_t2v_allows_image_when_editanything_active()
    test_editanything_lora_and_module_path_must_be_set_together()
    test_editanything_selectors_chain_lora_then_patch_helper()
    print("[ok] all smoke_ltx23_vid2vid tests passed")
