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


def test_video_guide_appends_and_crops_cleanly():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    positive, negative, latent, frame_rate = node.prepare(
        clip, vae, audio_vae, "prompt", "", 448, 256, 121, 24.0, 1,
        video=video, video_guide=True, hold_audio=False)

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
    print("[ok] LTXV23VidToVideo: video_guide=True appends real IC-LoRA guide frames "
          "(real comfy_extras.nodes_lt.LTXVAddGuide) and takes length/frame_rate from "
          "the clip; LTXV23CropVideoGuide removes exactly them back off")


def test_video_without_guide_is_plain_shape():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    positive, negative, latent, _ = node.prepare(
        clip, vae, audio_vae, "prompt", "", 448, 256, 121, 24.0, 1,
        video=video, video_guide=False, hold_audio=False)

    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    assert video_latent.shape[2] == _target_t(25)  # no guide appended

    crop_node = ltx23.LTXV23CropVideoGuide()
    _, _, cropped = crop_node.crop(positive, negative, latent)
    assert cropped is latent  # no-op passthrough, nothing to crop
    print("[ok] LTXV23VidToVideo: video_guide=False -> plain shape (video only used for "
          "length/frame_rate), LTXV23CropVideoGuide is a no-op passthrough")


def test_image_only_is_ordinary_i2v_hold_no_video_needed():
    node = ltx23.LTXV23VidToVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)

    positive, negative, latent, frame_rate = node.prepare(
        clip, vae, audio_vae, "prompt", "", 448, 256, 121, 30.0, 1,
        image=image, image_strength=0.7)

    assert frame_rate == 30.0  # widget value used - no video connected to override it
    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    assert video_latent.shape[2] == _target_t(121)  # widget length, not any clip's

    noise_mask = latent["noise_mask"].unbind()[0]
    assert torch.allclose(noise_mask[:, :, 0], torch.tensor(1.0 - 0.7))
    assert torch.allclose(noise_mask[:, :, 1], torch.tensor(1.0))  # rest of the clip untouched

    crop_node = ltx23.LTXV23CropVideoGuide()
    _, _, cropped = crop_node.crop(positive, negative, latent)
    assert cropped is latent  # no video_guide ever ran -> nothing to crop
    print("[ok] LTXV23VidToVideo: image only (no video connected) -> ordinary i2v "
          "first-frame hold, width/height/length/frame_rate all from the widgets")


def test_image_and_video_guide_combine_independently():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    image = torch.rand(1, 256, 448, 3)

    positive, negative, latent, _ = node.prepare(
        clip, vae, audio_vae, "prompt", "", 448, 256, 121, 24.0, 1,
        image=image, image_strength=0.7, video=video, video_guide=True, hold_audio=False)

    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    target_t = _target_t(25)  # video's own length wins over the length widget
    assert video_latent.shape[2] > target_t  # guide frames still appended on top

    noise_mask = latent["noise_mask"].unbind()[0]
    assert torch.allclose(noise_mask[:, :, 0], torch.tensor(1.0 - 0.7))  # image hold still applied

    _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, video_latent.shape)
    assert num_keyframes > 0
    print("[ok] LTXV23VidToVideo: image (i2v hold) and video (IC-LoRA guide) combine "
          "in the same call without interfering with each other")


if __name__ == "__main__":
    test_video_guide_appends_and_crops_cleanly()
    test_video_without_guide_is_plain_shape()
    test_image_only_is_ordinary_i2v_hold_no_video_needed()
    test_image_and_video_guide_combine_independently()
    print("[ok] all smoke_ltx23_vid2vid tests passed")
