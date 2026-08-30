"""Real-environment smoke test for the LTX-2.5 node family (nodes/ltx25.py).

Runs against the actual portable ComfyUI install so the new code's delegation
into comfy-core (nodes_lt.preprocess, Guider_LTXAVDualCFG, sampler_object,
prepare_noise/fix_empty_latent_channels on nested latents, VAEDecodeTiled)
is exercised for real (not stubbed). Fake VAE/CLIP/upscale objects keep this
GPU/weight-free (CPU only, tiny tensors).

Covers: stage-1 latent geometry (half the target resolution), the i2v hold
strength math (LTXVImgToVideoInplace semantics), the two sigma presets
verbatim, LoRA chaining order through the (spied) LoraLoaderModelOnly
delegation, the x2 upscale + refine re-hold, and the AV decode wiring.

Usage: python tools/smoke_ltx25.py
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
# Only the ltx25 submodule, not the package's own __init__.py aggregation -
# that would import every other node file (gguf, krea2, ...) for no reason.
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
ltx25 = importlib.import_module("cctech_gguf_pkg.nodes.ltx25")  # noqa: E402

import comfy_extras.nodes_lt as nodes_lt  # noqa: E402 - real core module
import comfy.model_management  # noqa: E402


class _FakeFSM:
    latent_channels = 8
    latent_frequency_bins = 16
    latents_per_second = 10.0

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
        return torch.ones(1, 128, t, h // 32, w // 32)


def _fake_clip():
    # Real comfy CONDITIONING shape: a list of [tensor, dict] pairs -
    # node_helpers.conditioning_set_values() indexes t[0]/t[1].copy() on it.
    return types.SimpleNamespace(
        encode_from_tokens_scheduled=lambda t: [[torch.zeros(1, 1, 8), {}]],
        tokenize=lambda s, **kw: s)


class _FakeLoraLoaderModelOnly:
    """Stands in for comfy-core's real nodes.LoraLoaderModelOnly - records
    calls and returns a distinguishable sentinel per lora_name so tests can
    prove chaining/order without needing real lora files on disk."""
    calls = []

    def load_lora_model_only(self, model, lora_name, strength_model):
        _FakeLoraLoaderModelOnly.calls.append((model, lora_name, strength_model))
        return (("lora-patched", lora_name, model),)


def _prep(node=None, **kw):
    node = node or ltx25.LTXV25ImgToVideo()
    args = dict(model="base-model", clip=_fake_clip(), mode="t2v",
                vae=_FakeVideoVAE(), audio_vae=_FakeAudioVAE(),
                prompt="prompt", negative_prompt="", width=448, height=256,
                length=121, frame_rate=24.0, batch_size=1)
    args.update(kw)
    return node.prepare(**args)


def test_sigma_presets_are_the_workflow_lists_verbatim():
    # The two ManualSigmas strings from video_ltx2_5_i2v.json, exactly.
    assert ltx25.LTX25_SIGMA_SETS["distilled (8 steps)"] == [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    assert ltx25.LTX25_SIGMA_SETS["refine (3 steps)"] == [0.85, 0.7250, 0.4219, 0.0]
    assert list(ltx25.LTX25_SIGMA_SETS) == ["distilled (8 steps)", "refine (3 steps)"]
    print("[ok] LTXV25KSampler: both sigma presets match the official workflow's "
          "ManualSigmas strings verbatim, distilled first (the default)")


def test_t2v_stage1_latent_geometry_is_half_resolution():
    _, positive, negative, latent, frame_rate = _prep()
    assert frame_rate == 24.0
    samples = latent["samples"]
    assert samples.is_nested
    video, audio = samples.unbind()
    # target 448x256 -> stage 1 at 224x128 -> latent 224//32 x 128//32 = 7x4;
    # 121 frames -> (121-1)//8+1 = 16 latent frames.
    assert tuple(video.shape) == (1, 128, 16, 4, 7), tuple(video.shape)
    assert torch.count_nonzero(video) == 0
    # audio geometry read off the (fake) VAE: 121/24*10 = 50 latents.
    assert tuple(audio.shape) == (1, 8, 50, 16), tuple(audio.shape)
    mask_v, mask_a = latent["noise_mask"].unbind()
    assert tuple(mask_v.shape) == (1, 1, 16, 1, 1)  # per-frame, core's shape
    assert torch.all(mask_v == 1.0) and torch.all(mask_a == 1.0)
    assert latent["downscale_ratio_spacial"] == 32
    # frame_rate rides on the conditioning (the LTXVConditioning call).
    assert positive[0][1]["frame_rate"] == 24.0
    assert negative[0][1]["frame_rate"] == 24.0
    print("[ok] LTXV25ImgToVideo: t2v stage-1 latent is built at HALF the target "
          "resolution (the workflow's a/2 math), audio geometry read off the VAE, "
          "frame_rate set on both conditionings")


def test_i2v_hold_strength_math_matches_inplace_semantics():
    image = torch.rand(1, 256, 448, 3)
    _, _, _, latent, _ = _prep(mode="i2v", images=image, image_strength=1.0,
                               img_compression=0)
    video, _ = latent["samples"].unbind()
    mask_v, _ = latent["noise_mask"].unbind()
    # fake encode returns ones -> the held first latent frame is overwritten
    # IN PLACE (LTXVImgToVideoInplace's samples[:, :, :t] = t), not appended.
    assert video.shape[2] == 16, "nothing may be appended - inplace hold only"
    assert torch.all(video[:, :, 0] == 1.0), "first latent frame must hold the image"
    assert torch.all(video[:, :, 1:] == 0.0), "later frames must stay empty"
    # strength 1.0 -> mask 1 - 1.0 = 0 on the held frame, 1 elsewhere.
    assert torch.all(mask_v[:, :, 0] == 0.0)
    assert torch.all(mask_v[:, :, 1:] == 1.0)

    # 0.7 is the official STAGE-1 value (workflow node 357, link-traced:
    # the EmptyLTXVLatentVideo feeds the 0.7 hold; the 1.0 hold sits on the
    # upsampled latent) - and the node's default.
    assert ltx25.LTX25_I2V_STRENGTH == 0.7
    _, _, _, latent, _ = _prep(mode="i2v", images=image, image_strength=0.7,
                               img_compression=0)
    mask_v, _ = latent["noise_mask"].unbind()
    assert torch.allclose(mask_v[:, :, 0], torch.tensor(1.0 - 0.7))
    print("[ok] LTXV25ImgToVideo: i2v hold is LTXVImgToVideoInplace's exact math - "
          "image overwrites the first latent frame in place, mask carries "
          "1 - strength on it (0.3 @ the official stage-1 0.7, 0.0 @ 1.0)")


def test_img_compression_delegates_to_core_preprocess():
    # crf 18 goes through comfy-core's real nodes_lt.preprocess H.264
    # round-trip; crf 0 is core's documented passthrough.
    image = torch.rand(1, 64, 64, 3)
    out = ltx25._preprocess_images(image, 18)
    assert out.shape == image.shape
    assert not torch.equal(out, image), "crf 18 must actually alter the pixels"
    assert torch.equal(ltx25._preprocess_images(image, 0), image)
    print("[ok] LTXV25ImgToVideo: img_compression delegates to comfy-core's real "
          "LTXVPreprocess round-trip (18 alters pixels, 0 is a passthrough)")


def test_lora_chaining_order_and_none_skip():
    orig_loader = ltx25.nodes.LoraLoaderModelOnly
    ltx25.nodes.LoraLoaderModelOnly = _FakeLoraLoaderModelOnly
    _FakeLoraLoaderModelOnly.calls = []
    try:
        out_model, *_ = _prep(lora_1="a.safetensors", lora_1_strength=0.6,
                              lora_2="none",
                              lora_3="c.safetensors", lora_3_strength=0.9)
    finally:
        ltx25.nodes.LoraLoaderModelOnly = orig_loader

    # lora_2="none" skipped; order preserved; each load receives the PREVIOUS
    # load's output (real chaining, like the workflow's two chained
    # LoraLoaderModelOnly nodes).
    assert _FakeLoraLoaderModelOnly.calls == [
        ("base-model", "a.safetensors", 0.6),
        (("lora-patched", "a.safetensors", "base-model"), "c.safetensors", 0.9),
    ]
    assert out_model == ("lora-patched", "c.safetensors",
                         ("lora-patched", "a.safetensors", "base-model"))

    out_model, *_ = _prep()
    assert out_model == "base-model", "all-none -> model passes through untouched"
    print("[ok] LTXV25ImgToVideo: lora_1/2/3 chain through comfy-core's real "
          "LoraLoaderModelOnly in order, \"none\" entries skipped, all-none is a "
          "pure passthrough")


def test_mode_validates_required_socket_is_connected():
    image = torch.rand(1, 256, 448, 3)
    try:
        _prep(mode="i2v")
        assert False, "expected ValueError for mode=i2v with no images connected"
    except ValueError as e:
        assert "images" in str(e)
    try:
        _prep(mode="t2v", images=image)
        assert False, "expected ValueError for mode=t2v with images connected"
    except ValueError as e:
        assert "t2v" in str(e)
    print("[ok] LTXV25ImgToVideo: mode is validated against what's connected - "
          "i2v demands images, t2v rejects it")


class _FakeUpscaleModel:
    load_device = torch.device("cpu")

    @staticmethod
    def model(latents):
        # the official spatial upscaler: x2 in H and W, everything else kept
        return torch.nn.functional.interpolate(
            latents, scale_factor=(1, 2, 2), mode="nearest")

    def model_dtype(self):
        return torch.float32


class _FakeStats:
    def un_normalize(self, x):
        return x

    def normalize(self, x):
        return x


def _upscale_vae():
    vae = _FakeVideoVAE()
    vae.first_stage_model = types.SimpleNamespace(per_channel_statistics=_FakeStats())
    return vae


def test_latent_upscale_doubles_video_and_passes_audio_through():
    video = torch.rand(1, 128, 16, 4, 7)
    audio = torch.rand(1, 8, 50, 16)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio)),
              "noise_mask": comfy.nested_tensor.NestedTensor(
                  (torch.ones(1, 1, 16, 1, 1), torch.ones_like(audio))),
              "downscale_ratio_spacial": 32}

    orig_load = comfy.model_management.load_models_gpu
    comfy.model_management.load_models_gpu = lambda *a, **kw: None
    try:
        node = ltx25.LTXV25LatentUpscale()
        upscaled, = node.upscale(latent, _FakeUpscaleModel(), _upscale_vae())
    finally:
        comfy.model_management.load_models_gpu = orig_load

    up_video, up_audio = upscaled["samples"].unbind()
    assert tuple(up_video.shape) == (1, 128, 16, 8, 14), "video must double in H/W only"
    assert torch.equal(up_audio, audio), "audio must pass through untouched"
    assert "noise_mask" not in upscaled, "no images -> mask dropped (core behavior)"
    print("[ok] LTXV25LatentUpscale: x2 on the video half only, audio passthrough, "
          "stale noise_mask dropped when no re-hold is requested")


def test_latent_upscale_rehold_applies_refine_strength():
    video = torch.rand(1, 128, 16, 4, 7)
    audio = torch.rand(1, 8, 50, 16)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    image = torch.rand(1, 256, 448, 3)

    orig_load = comfy.model_management.load_models_gpu
    comfy.model_management.load_models_gpu = lambda *a, **kw: None
    try:
        node = ltx25.LTXV25LatentUpscale()
        upscaled, = node.upscale(latent, _FakeUpscaleModel(), _upscale_vae(),
                                 images=image, img_compression=0)
    finally:
        comfy.model_management.load_models_gpu = orig_load

    up_video, _ = upscaled["samples"].unbind()
    mask_v, mask_a = upscaled["noise_mask"].unbind()
    # fake encode returns ones -> re-held first frame overwritten on the
    # UPSCALED grid (the workflow's second LTXVImgToVideoInplace).
    assert torch.all(up_video[:, :, 0] == 1.0)
    # 1.0 is the official REFINE value (workflow node 349, link-traced: the
    # LTXVLatentUpsampler feeds the 1.0 hold) - and the node's default.
    assert ltx25.LTX25_REFINE_STRENGTH == 1.0
    assert torch.all(mask_v[:, :, 0] == 0.0), \
        "refine re-hold must default to the official 1.0 (mask 1 - 1.0 = 0)"
    assert torch.all(mask_v[:, :, 1:] == 1.0)
    assert torch.all(mask_a == 1.0)
    print("[ok] LTXV25LatentUpscale: wiring images re-holds the first frame on the "
          "upscaled latent @ the official 1.0 and rebuilds the joint noise mask")


class _FakeDualCFGGuider:
    """Spy standing in for comfy-core's real Guider_LTXAVDualCFG - records
    the composition LTXV25KSampler builds (conds, dual cfg, sigmas,
    denoise_mask) and returns a recognizable output."""
    instances = []

    def __init__(self, model):
        self.model = model
        self.conds = self.cfg = self.sample_args = None
        _FakeDualCFGGuider.instances.append(self)

    def set_conds(self, positive, negative):
        self.conds = (positive, negative)

    def set_cfg(self, video_cfg, audio_cfg):
        self.cfg = (video_cfg, audio_cfg)

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               seed=None, disable_pbar=False):
        self.sample_args = dict(noise=noise, latent_image=latent_image,
                                sampler=sampler, sigmas=sigmas,
                                denoise_mask=denoise_mask, seed=seed)
        return comfy.nested_tensor.NestedTensor(
            tuple(torch.zeros_like(t) for t in latent_image.unbind()))


def test_ksampler_composes_dual_cfg_guider_with_exact_sigmas():
    _, positive, negative, latent, _ = _prep()

    orig_guider = nodes_lt.Guider_LTXAVDualCFG
    nodes_lt.Guider_LTXAVDualCFG = _FakeDualCFGGuider
    _FakeDualCFGGuider.instances = []
    try:
        node = ltx25.LTXV25KSampler()
        out, = node.sample("model", positive, negative, latent, seed=7,
                           schedule="distilled (8 steps)",
                           sampler_name="euler_ancestral",
                           video_cfg=1.0, audio_cfg=1.0)
        out2, = node.sample("model", positive, negative, latent, seed=7,
                            schedule="refine (3 steps)",
                            sampler_name="euler_ancestral",
                            video_cfg=1.0, audio_cfg=1.0)
    finally:
        nodes_lt.Guider_LTXAVDualCFG = orig_guider

    g1, g2 = _FakeDualCFGGuider.instances
    assert g1.model == "model"
    assert g1.conds == (positive, negative)
    assert g1.cfg == (1.0, 1.0), "the workflow's LTXVDualCFGGuider(1, 1)"
    # exact pass-through: the sigma tensors ARE the preset lists, unresampled
    assert torch.equal(g1.sample_args["sigmas"], torch.tensor(
        [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]))
    assert torch.equal(g2.sample_args["sigmas"],
                       torch.tensor([0.85, 0.7250, 0.4219, 0.0]))
    # a real euler_ancestral sampler object from comfy-core's sampler_object
    assert g1.sample_args["sampler"] is not None
    assert g1.sample_args["seed"] == 7
    # the joint noise mask travels through as the denoise mask
    assert g1.sample_args["denoise_mask"] is latent["noise_mask"]
    # real prepare_noise ran on the nested latent: per-stream noise shapes
    n_video, n_audio = g1.sample_args["noise"].unbind()
    v, a = latent["samples"].unbind()
    assert n_video.shape == v.shape and n_audio.shape == a.shape
    assert "noise_mask" not in out and out["samples"].is_nested
    assert "noise_mask" not in out2
    print("[ok] LTXV25KSampler: composes core's Guider_LTXAVDualCFG(video/audio "
          "cfg) + sampler_object(euler_ancestral) + prepare_noise on the nested "
          "latent, passing the preset sigmas through verbatim and the joint "
          "noise mask as denoise_mask")


class _FakeDecodeVideoVAE:
    def spacial_compression_decode(self):
        return 32

    def temporal_compression_decode(self):
        return 8

    def decode_tiled(self, latent, **kw):
        b, c, t, h, w = latent.shape
        return torch.rand(b, (t - 1) * 8 + 1, h * 32, w * 32, 3)


class _FakeDecodeAudioVAE:
    def __init__(self):
        self.first_stage_model = types.SimpleNamespace(output_sample_rate=24000)

    def decode(self, audio_latent):
        return torch.rand(audio_latent.shape[0], audio_latent.shape[2] * 100, 2)


def test_av_decode_wires_video_audio_and_fps():
    video = torch.rand(1, 128, 16, 8, 14)
    audio = torch.rand(1, 8, 50, 16)
    latent = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

    node = ltx25.LTXV25AVDecode()
    out_video, = node.decode(latent, _FakeDecodeVideoVAE(), _FakeDecodeAudioVAE(),
                             fps=24.0)
    components = out_video.get_components()
    assert components.images.shape == (121, 256, 448, 3), \
        "video half decoded through core VAEDecodeTiled (nested unbind inside)"
    assert components.audio["sample_rate"] == 24000
    assert components.audio["waveform"].shape == (1, 2, 5000)
    assert float(components.frame_rate) == 24.0
    print("[ok] LTXV25AVDecode: joint latent -> muxed VIDEO - video via core "
          "VAEDecodeTiled, audio via the audio VAE, one fps threaded through")


if __name__ == "__main__":
    test_sigma_presets_are_the_workflow_lists_verbatim()
    test_t2v_stage1_latent_geometry_is_half_resolution()
    test_i2v_hold_strength_math_matches_inplace_semantics()
    test_img_compression_delegates_to_core_preprocess()
    test_lora_chaining_order_and_none_skip()
    test_mode_validates_required_socket_is_connected()
    test_latent_upscale_doubles_video_and_passes_audio_through()
    test_latent_upscale_rehold_applies_refine_strength()
    test_ksampler_composes_dual_cfg_guider_with_exact_sigmas()
    test_av_decode_wires_video_audio_and_fps()
    print("[ok] all smoke_ltx25 tests passed")
