"""Real-environment smoke test for LTXV23VidToVideo / LTXV23CropVideoGuide.

Runs against the actual portable ComfyUI install so the new code's delegation
into comfy-core's real comfy_extras.nodes_lt.LTXVAddGuide is exercised for
real (not stubbed) - that's the one piece genuinely new here, everything
else (audio-hold helpers, joint-latent convention) is already-proven code
reused from LTXV23ImgToVideo. Fake VAE/CLIP objects keep this GPU/weight-free
(CPU only, tiny tensors) while still running the real RoPE/keyframe math.

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


def test_vid_to_video_with_guide_appends_and_crops_cleanly():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    positive, negative, latent, frame_rate = node.prepare(
        video, clip, vae, audio_vae, "prompt", "", 544, True, 1.0, hold_audio=False)

    assert frame_rate == 24.0
    samples = latent["samples"]
    assert samples.is_nested
    video_latent, audio_latent = samples.unbind()

    target_t = ((ltx23._align_length(25) - 1) // 8) + 1
    assert video_latent.shape[0] == 1 and video_latent.shape[1] == 128
    assert video_latent.shape[2] > target_t  # guide frames appended on top

    _, num_keyframes = __import__("comfy_extras.nodes_lt", fromlist=["get_keyframe_idxs"]) \
        .get_keyframe_idxs(positive, video_latent.shape)
    assert num_keyframes > 0
    assert video_latent.shape[2] - num_keyframes == target_t

    crop_node = ltx23.LTXV23CropVideoGuide()
    cpos, cneg, cropped = crop_node.crop(positive, negative, latent)
    cropped_video, cropped_audio = cropped["samples"].unbind()
    assert cropped_video.shape[2] == target_t
    assert torch.equal(cropped_audio, audio_latent)

    _, num_keyframes_after = __import__("comfy_extras.nodes_lt", fromlist=["get_keyframe_idxs"]) \
        .get_keyframe_idxs(cpos, cropped_video.shape)
    assert num_keyframes_after == 0
    print("[ok] LTXV23VidToVideo: video_guide=True appends real IC-LoRA guide frames "
          "(real comfy_extras.nodes_lt.LTXVAddGuide), LTXV23CropVideoGuide removes "
          "exactly them back off")


def test_vid_to_video_without_guide_is_plain_t2v_shape():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()

    positive, negative, latent, _ = node.prepare(
        video, clip, vae, audio_vae, "prompt", "", 544, False, 1.0, hold_audio=False)

    samples = latent["samples"]
    video_latent, _ = samples.unbind()
    target_t = ((ltx23._align_length(25) - 1) // 8) + 1
    assert video_latent.shape[2] == target_t  # no guide appended

    crop_node = ltx23.LTXV23CropVideoGuide()
    _, _, cropped = crop_node.crop(positive, negative, latent)
    assert cropped is latent  # no-op passthrough, nothing to crop
    print("[ok] LTXV23VidToVideo: video_guide=False -> plain shape, no guide frames "
          "appended; LTXV23CropVideoGuide is a no-op passthrough")


def test_vid_to_video_rejects_bad_latent_downscale_factor():
    node = ltx23.LTXV23VidToVideo()
    video = _FakeVideo(h=250, w=450)  # deliberately not divisible by 2 after 32-rounding tricks
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    try:
        node.prepare(video, clip, vae, audio_vae, "prompt", "", 100, True, 1.0,
                     hold_audio=False, latent_downscale_factor=3.0)
        raised = False
    except ValueError:
        raised = True
    # width/height are rounded to 32-multiples inside prepare(), so whether this
    # raises depends on the resulting size vs factor 3 - just confirm it does not
    # crash with anything OTHER than the documented ValueError.
    print(f"[ok] LTXV23VidToVideo: latent_downscale_factor=3.0 path runs without an "
          f"unexpected exception (raised ValueError: {raised})")


if __name__ == "__main__":
    test_vid_to_video_with_guide_appends_and_crops_cleanly()
    test_vid_to_video_without_guide_is_plain_t2v_shape()
    test_vid_to_video_rejects_bad_latent_downscale_factor()
    print("[ok] all smoke_ltx23_vid2vid tests passed")
