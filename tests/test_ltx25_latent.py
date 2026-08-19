"""LTX-2.5 empty AV latent geometry.

The failure this node exists to prevent is silent at wiring time: a MiniMax H3
AV latent is the same nested-tensor *type* as an LTX-2.5 one, so it connects and
then explodes inside patchify_proj. These tests pin both streams' shapes and the
guard that rejects the wrong VAE.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_PKG = "comfy_gguf_ltx25_under_test"


def _load_module():
    """Import nodes_ltx25.py against the handful of Comfy bits it touches."""
    full = f"{_PKG}.nodes_ltx25"
    if full in sys.modules:
        return sys.modules[full]

    comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    mm = sys.modules.setdefault("comfy.model_management",
                                types.ModuleType("comfy.model_management"))
    mm.intermediate_device = lambda: torch.device("cpu")
    comfy.model_management = mm

    nested = sys.modules.setdefault("comfy.nested_tensor",
                                    types.ModuleType("comfy.nested_tensor"))
    # The real NestedTensor packs streams for the sampler; only unbind() matters here.
    nested.NestedTensor = lambda streams: types.SimpleNamespace(
        unbind=lambda streams=tuple(streams): streams)
    comfy.nested_tensor = nested

    core = sys.modules.setdefault("nodes", types.ModuleType("nodes"))
    core.MAX_RESOLUTION = 16384

    package = sys.modules.setdefault(_PKG, types.ModuleType(_PKG))
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(full, ROOT / "nodes_ltx25.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


ltx25 = _load_module()


class _FakeAudioVAEModel:
    """Stand-in for comfy.ldm.lightricks.vae.audio_vae, matching its geometry API.

    The shipped ltx-2.5 audio VAE runs at 25 latents/second with 16 frequency
    bins and 8 latent channels; these are the values the E2E run reported.
    """

    latents_per_second = 25.0
    latent_frequency_bins = 16
    latent_channels = 8

    def num_of_latents_from_frames(self, frames_number, frame_rate):
        return round((float(frames_number) / frame_rate) * self.latents_per_second)


class _FakeAudioVAE:
    latent_channels = 8

    def __init__(self):
        self.first_stage_model = _FakeAudioVAEModel()


class _FakeVideoVAE:
    """A video VAE has no audio geometry — the miswiring the guard catches."""
    latent_channels = 128

    def __init__(self):
        self.first_stage_model = types.SimpleNamespace()


def _shapes(**kwargs):
    params = dict(width=768, height=512, length=97, frame_rate=24.0, batch_size=1)
    params.update(kwargs)
    latent = ltx25.empty_av_latent(_FakeAudioVAE(), **params)
    video, audio = latent["samples"].unbind()
    return tuple(video.shape), tuple(audio.shape), latent


def test_video_stream_matches_ltx25_compression():
    # 97 frames -> 13 latent frames (8:1 with a causal first frame), 768x512 -> /32.
    video, _, latent = _shapes()
    assert video == (1, 128, 13, 16, 24)
    assert latent["downscale_ratio_spacial"] == 32


def test_audio_stream_length_follows_duration():
    # 97 frames @ 24 fps = 4.04s * 25 latents/s = 101.
    _, audio, _ = _shapes()
    assert audio == (1, 8, 101, 16)


def test_frame_rate_changes_only_the_audio_length():
    video_24, audio_24, _ = _shapes(length=121, frame_rate=24.0)
    video_25, audio_25, _ = _shapes(length=121, frame_rate=25.0)
    assert video_24 == video_25 == (1, 128, 16, 16, 24)
    assert audio_24 == (1, 8, 126, 16)
    assert audio_25 == (1, 8, 121, 16)


def test_batch_size_applies_to_both_streams():
    video, audio, _ = _shapes(length=9, batch_size=4)
    assert video[0] == audio[0] == 4


def test_causal_first_frame_in_temporal_compression():
    # 8k+1 frame counts tile exactly; the +1 is the causal frame.
    assert ltx25.video_latent_t(1) == 1
    assert ltx25.video_latent_t(9) == 2
    assert ltx25.video_latent_t(97) == 13


def test_does_not_match_minimax_h3_geometry():
    """The whole point: H3 latents must not be interchangeable with these."""
    video, audio, _ = _shapes(width=1344, height=768, length=124)
    assert video[1] == 128 and video[3:] == (768 // 32, 1344 // 32)  # H3: 24ch, /16
    assert len(audio) == 4  # H3 audio is [B, 32, 2, T]; here [B, z, n, bins]
    assert audio[1] == 8 and audio[3] == 16


def test_video_vae_in_the_audio_slot_is_rejected():
    with pytest.raises(ValueError, match="not an LTX-2.5 audio VAE"):
        ltx25.empty_av_latent(_FakeVideoVAE(), 768, 512, 97, 24.0, 1)


def test_clip_too_short_for_one_audio_latent_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        ltx25.empty_av_latent(_FakeAudioVAE(), 768, 512, 1, 120.0, 1)


def test_node_registers_with_a_latent_output():
    node_class = ltx25.NODE_CLASS_MAPPINGS["LTXV25EmptyLatentAVBatch"]
    assert node_class.RETURN_TYPES == ("LATENT",)
    required = node_class.INPUT_TYPES()["required"]
    assert required["audio_vae"][0] == "VAE"
    assert list(required) == ["audio_vae", "width", "height", "length",
                              "frame_rate", "batch_size"]
