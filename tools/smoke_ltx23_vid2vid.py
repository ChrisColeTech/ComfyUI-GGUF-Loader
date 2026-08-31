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


def test_remove_person_greenfill_and_guide():
    # Stage 1 prep: mask dilation grows the region, masked pixels become
    # exactly #66FF00, unmasked pixels stay untouched, and the green-filled
    # control video is appended as held guide tokens at strength 1.0.
    node = ltx23.LTXV23RemovePerson()
    video = _FakeVideo()
    vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    clip = _fake_clip()
    mask = torch.zeros(25, 256, 448)
    mask[:, 100:150, 200:300] = 1.0

    orig_loader = ltx23.nodes.LoraLoaderModelOnly
    ltx23.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    _FakeLoraLoaderModelOnly.calls = []
    try:
        out = node.prepare("base-model", clip, vae, audio_vae, video, mask,
                           "empty room", "", 448, 256, 1, "inpaint.safetensors",
                           keep_original_audio=False)
    finally:
        ltx23.nodes.LoraLoaderModelOnly = orig_loader

    out_model, positive, negative, latent, frame_rate, source_frames, blend_mask = out
    assert _FakeLoraLoaderModelOnly.calls == [("base-model", "inpaint.safetensors", 1.0)], \
        "inpaint LoRA must load at the trained strength 1.0"

    # green-fill check: rebuild the control frames the node built internally
    green = torch.tensor(ltx23.LTX2_MASKED_CONTROL_VIDEO_PAD_RGB).float() / 255.0
    m = ltx23._comfy_mask_to_t1hw(mask, 25, 256, 448)
    m_dil = ltx23._ltx2_dilate_mask(m, 5)
    assert m_dil.sum() > m.sum(), "dilation must grow the mask"
    filled = ltx23._ltx2_greenfill(source_frames, m_dil)
    inside = filled[0, 120, 250]  # deep inside the mask
    assert torch.allclose(inside, green), f"masked pixel must be exact #66FF00, got {inside}"
    outside = filled[0, 10, 10]
    assert torch.allclose(outside, source_frames[0, 10, 10]), "unmasked pixels must be untouched"

    video_latent, _ = latent["samples"].unbind()
    noise_video, _ = latent["noise_mask"].unbind()
    target_t = _target_t(25)
    assert video_latent.shape[2] > target_t, "green control video must be appended as guide"
    assert torch.all(noise_video[:, :, target_t:] == 0.0), "guide must be held @ strength 1.0"
    # the user mask must NOT touch the denoise mask of the generated region
    assert torch.all(noise_video[:, :, :target_t] == 1.0), \
        "generated region must be fully noised - the mask is not a denoise mask in this recipe"
    assert blend_mask.shape == (25, 256, 448)
    assert source_frames.shape == (25, 256, 448, 3)
    print("[ok] LTXV23RemovePerson: mask dilated, masked region painted exact #66FF00, "
          "green control video appended as held guide @ 1.0, user mask kept OUT of the "
          "denoise mask (LoRA does the regeneration), source/blend passthroughs correct")


def test_mask_blend_identity_outside_mask_and_generated_inside():
    node = ltx23.LTXV23MaskBlend()
    torch.manual_seed(0)
    source = torch.rand(3, 128, 128, 3)
    generated = torch.rand(3, 128, 128, 3)

    # zero mask -> output == source everywhere (pyramid collapse is exact)
    zero_mask = torch.zeros(3, 128, 128)
    out, = node.blend(generated, source, zero_mask, mask_low_res_dilation=0)
    assert torch.allclose(out, source, atol=1e-5), "zero mask must reproduce source exactly"

    # full mask -> output == generated everywhere
    ones_mask = torch.ones(3, 128, 128)
    out, = node.blend(generated, source, ones_mask, mask_low_res_dilation=0)
    assert torch.allclose(out, generated, atol=1e-5), "full mask must reproduce generated exactly"

    # partial mask (with the trained soft skirt): deep inside -> generated,
    # far outside -> source
    part_mask = torch.zeros(3, 128, 128)
    part_mask[:, 40:88, 40:88] = 1.0
    out, = node.blend(generated, source, part_mask)
    assert torch.allclose(out[:, 64, 64], generated[:, 64, 64], atol=0.05), \
        "deep inside the mask must be (close to) the generated pixels"
    assert torch.allclose(out[:, 4, 4], source[:, 4, 4], atol=0.05), \
        "far outside the mask must be (close to) the source pixels"
    print("[ok] LTXV23MaskBlend: Laplacian pyramid blend - exact source outside, exact "
          "generated inside, soft trained skirt in between")


def test_mask_blend_union_mask_b_and_frame_count_reconciliation():
    node = ltx23.LTXV23MaskBlend()
    torch.manual_seed(1)
    # generated longer than source (the 8k+1 round-up case: 105 gen vs 99 src)
    source = torch.rand(4, 64, 64, 3)
    generated = torch.rand(6, 64, 64, 3)

    # two disjoint masks with MISMATCHED frame counts - mask covers the left
    # block, mask_b (gen count) the right block; union must take generated in
    # BOTH blocks and source elsewhere, over the common 4 frames.
    mask = torch.zeros(4, 64, 64)
    mask[:, 16:32, 8:24] = 1.0
    mask_b = torch.zeros(6, 64, 64)
    mask_b[:, 16:32, 40:56] = 1.0
    out, = node.blend(generated, source, mask, mask_low_res_dilation=0, mask_b=mask_b)
    assert out.shape[0] == 6, ("output keeps the FULL generated length - frames past the "
                               "source are appended un-composited (no source to blend onto)")
    assert torch.allclose(out[4:], generated[4:], atol=1e-5), \
        "tail frames beyond the source must be the generated frames verbatim"
    assert torch.allclose(out[:4, 24, 16], generated[:4, 24, 16], atol=1e-5), \
        "inside mask (left block) must be generated"
    assert torch.allclose(out[:4, 24, 48], generated[:4, 24, 48], atol=1e-5), \
        "inside mask_b (right block) must be generated - the union must include it"
    assert torch.allclose(out[:4, 4, 32], source[:, 4, 32], atol=1e-5), \
        "outside both masks must be source"

    # mask_b=None keeps the original single-mask behaviour
    out2, = node.blend(generated, source, mask, mask_low_res_dilation=0)
    assert torch.allclose(out2[:4, 24, 48], source[:, 24, 48], atol=1e-5), \
        "without mask_b the right block must stay source"
    print("[ok] LTXV23MaskBlend: mask_b union with mismatched frame counts - trimmed to "
          "common length, union region from generated, elsewhere source, None = old behaviour")


if __name__ == "__main__":
    test_ic_lora_appends_and_crops_cleanly()
    test_video_with_ic_lora_none_is_plain_shape()
    test_image_only_is_ordinary_i2v_hold_no_video_needed()
    test_image_and_ic_lora_combine_independently()
    test_mode_validates_required_socket_is_connected()
    test_remove_person_greenfill_and_guide()
    test_mask_blend_identity_outside_mask_and_generated_inside()
    test_mask_blend_union_mask_b_and_frame_count_reconciliation()
    print("[ok] all smoke_ltx23_vid2vid tests passed")
