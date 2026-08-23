"""CPU-only smoke test for nodes_qwen_tts.py - no GPU, no real weights.

Stubs out ``folder_paths`` (a fake models/qwen_tts/<name>/config.json on
disk under a temp dir) and ``qwen_tts.Qwen3TTSModel`` (a fake with a canned
``generate_custom_voice``), then checks: model-path discovery, the loader's
from_pretrained kwargs and caching, and the generate node's seed/AUDIO-dict/
unload/instruct-drop-warning logic.

Usage:  python tools/smoke_qwen_tts.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

_tmp = tempfile.mkdtemp(prefix="qwen_tts_smoke_")
_MODEL_DIR = os.path.join(_tmp, "qwen_tts", "fake-1.7b-customvoice")
os.makedirs(_MODEL_DIR, exist_ok=True)
with open(os.path.join(_MODEL_DIR, "config.json"), "w") as f:
    f.write("{}")

# ── stub folder_paths before importing the node module ──────────────────────
_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.models_dir = _tmp
_folder_paths_stub.folder_names_and_paths = {}


def _get_folder_paths(key):
    return _folder_paths_stub.folder_names_and_paths.get(key, ([],))[0]


_folder_paths_stub.get_folder_paths = _get_folder_paths
sys.modules["folder_paths"] = _folder_paths_stub

# ── stub qwen_tts before importing the node module ───────────────────────────
_fake_model_instance = MagicMock()
_fake_model_instance.model.tts_model_size = "1_7b"  # not a 0.6b size


def _fake_generate_custom_voice(**kwargs):
    wav = np.zeros(4000, dtype=np.float32)
    return [wav], 24000


_fake_model_instance.generate_custom_voice.side_effect = _fake_generate_custom_voice

_qwen_tts_module = types.ModuleType("qwen_tts")
_qwen_tts_module.Qwen3TTSModel = MagicMock()
_qwen_tts_module.Qwen3TTSModel.from_pretrained.return_value = _fake_model_instance
sys.modules["qwen_tts"] = _qwen_tts_module

sys.path.insert(0, str(REPO_ROOT))
import nodes_qwen_tts as qt  # noqa: E402


def test_model_paths_finds_local_config():
    paths = qt._model_paths()
    assert "fake-1.7b-customvoice" in paths
    print("[ok] _model_paths: finds the fake local checkpoint folder")


def test_loader_calls_from_pretrained_with_expected_kwargs_and_caches():
    qt._MODEL_CACHE.clear()
    node = qt.QwenTTSModelsLoader()
    (bundle,) = node.load("fake-1.7b-customvoice", "cuda", "bf16", "auto")
    assert bundle["model"] is _fake_model_instance

    call_kwargs = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
    assert call_kwargs["dtype"] is torch.bfloat16
    assert call_kwargs["device_map"] == "cuda"
    assert call_kwargs["local_files_only"] is True
    assert "attn_implementation" not in call_kwargs  # "auto" omits it
    print("[ok] loader: from_pretrained called offline with correct dtype/device")

    calls_before = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count
    (bundle2,) = node.load("fake-1.7b-customvoice", "cuda", "bf16", "auto")
    assert bundle2["model"] is _fake_model_instance
    assert _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count == calls_before
    print("[ok] loader: identical widgets hit the cache, no reload")


def test_loader_explicit_attention_is_forwarded():
    qt._MODEL_CACHE.clear()
    qt.QwenTTSModelsLoader().load("fake-1.7b-customvoice", "cpu", "fp32", "sdpa")
    call_kwargs = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
    assert call_kwargs["attn_implementation"] == "sdpa"
    print("[ok] loader: non-'auto' attention is forwarded explicitly")


def test_generate_builds_correct_audio_dict_and_seeds():
    qt._MODEL_CACHE.clear()
    (bundle,) = qt.QwenTTSModelsLoader().load("fake-1.7b-customvoice", "cuda", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    with patch.object(torch, "manual_seed") as seed_fn:
        (audio,) = node.generate(
            bundle, "hello world", "Dylan", "auto", 1234,
            2048, 0.8, 20, 1.0, 1.05)
        seed_fn.assert_called_once_with(1234)
    assert audio["sample_rate"] == 24000
    assert audio["waveform"].shape == (1, 1, 4000)
    print("[ok] generate: seeds torch and builds a [1,1,T] AUDIO dict at the right rate")


def test_generate_passes_none_language_for_auto():
    qt._MODEL_CACHE.clear()
    (bundle,) = qt.QwenTTSModelsLoader().load("fake-1.7b-customvoice", "cuda", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    node.generate(bundle, "hi", "Dylan", "Auto", 0, 2048, 0.8, 20, 1.0, 1.05)
    call_kwargs = _fake_model_instance.generate_custom_voice.call_args.kwargs
    assert call_kwargs["language"] is None
    assert call_kwargs["speaker"] == "Dylan"
    print("[ok] generate: 'Auto'/'auto' language resolves to None, speaker passed through")


def test_generate_unload_clears_cache():
    qt._MODEL_CACHE.clear()
    (bundle,) = qt.QwenTTSModelsLoader().load("fake-1.7b-customvoice", "cuda", "bf16", "auto")
    assert bundle["key"] in qt._MODEL_CACHE
    node = qt.QwenTTSCustomVoiceGenerate()
    node.generate(bundle, "hi", "Dylan", "auto", 0, 2048, 0.8, 20, 1.0, 1.05,
                  instruct="", unload_model_after_generate=True)
    assert bundle["key"] not in qt._MODEL_CACHE
    print("[ok] generate: unload_model_after_generate evicts the cache entry")


def test_instruct_drop_warning_fires_for_06b_size_string():
    # Upstream's own check is `tts_model_size in "0b6"` - a substring test,
    # not a whitelist - so only values that are themselves substrings of
    # the literal "0b6" (e.g. "0b6" itself, "b6", "0", "") match.
    assert qt._instruct_is_silently_dropped(types.SimpleNamespace(
        model=types.SimpleNamespace(tts_model_size="0b6"))) is True
    assert qt._instruct_is_silently_dropped(types.SimpleNamespace(
        model=types.SimpleNamespace(tts_model_size="1_7b"))) is False
    assert qt._instruct_is_silently_dropped(types.SimpleNamespace()) is False
    print("[ok] _instruct_is_silently_dropped: matches upstream's substring "
          "check, never raises on a missing attribute")


if __name__ == "__main__":
    assert qt.Qwen3TTSModel is not None, "qwen_tts stub failed to install"
    test_model_paths_finds_local_config()
    test_loader_calls_from_pretrained_with_expected_kwargs_and_caches()
    test_loader_explicit_attention_is_forwarded()
    test_generate_builds_correct_audio_dict_and_seeds()
    test_generate_passes_none_language_for_auto()
    test_generate_unload_clears_cache()
    test_instruct_drop_warning_fires_for_06b_size_string()
    print("[ok] all nodes_qwen_tts smoke tests passed")
