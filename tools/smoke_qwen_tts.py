"""CPU-only smoke test for nodes_qwen_tts.py - no GPU, no real weights.

Stubs out ``folder_paths``, ``comfy.model_management`` and
``qwen_tts.Qwen3TTSModel`` (a fake with a canned ``generate_custom_voice``),
then checks: the models/Qwen3-TTS/<folder_name> layout, the loader's
download-vs-reuse decision and from_pretrained kwargs, the local_model_path
override's speech_tokenizer/ validation, and the generate node's seed/
AUDIO-dict/unload/instruct-drop-warning/wrong-model-type logic.

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

# ── stub folder_paths before importing the node module ──────────────────────
_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.models_dir = _tmp
_folder_paths_stub.folder_names_and_paths = {}


def _add_model_folder_path(key, path):
    entry = _folder_paths_stub.folder_names_and_paths.setdefault(key, ([], set()))
    entry[0].append(path)


def _get_folder_paths(key):
    return _folder_paths_stub.folder_names_and_paths.get(key, ([],))[0]


_folder_paths_stub.add_model_folder_path = _add_model_folder_path
_folder_paths_stub.get_folder_paths = _get_folder_paths
sys.modules["folder_paths"] = _folder_paths_stub

# ── stub comfy.model_management before importing the node module ────────────
_comfy_module = types.ModuleType("comfy")
_comfy_mm_module = types.ModuleType("comfy.model_management")
_comfy_mm_module.get_torch_device = lambda: torch.device("cpu")
_comfy_module.model_management = _comfy_mm_module
sys.modules["comfy"] = _comfy_module
sys.modules["comfy.model_management"] = _comfy_mm_module

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

sys.path.insert(0, str(REPO_ROOT.parent))
import types
# Fake the `cctech_gguf_pkg` and `cctech_gguf_pkg.nodes` packages (pointed at
# the repo root / its nodes/ dir) so importing nodes.qwen_tts doesn't run
# either the real root __init__.py (needs comfy.utils) or the real
# nodes/__init__.py (aggregates every other node module).
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
import importlib
qt = importlib.import_module("cctech_gguf_pkg.nodes.qwen_tts")  # noqa: E402

_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _populate_local_folder():
    """Pretend the model was already downloaded: create the target folder
    with a dummy file, matching what a real snapshot_download would leave.
    """
    path = qt._get_local_model_path(_REPO_ID)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        f.write("{}")
    return path


def _clear_local_folder():
    path = qt._get_local_model_path(_REPO_ID)
    if os.path.isdir(path):
        for name in os.listdir(path):
            os.remove(os.path.join(path, name))
        os.rmdir(path)


def test_folder_registered_under_models_dir():
    assert qt.QWEN3_TTS_MODELS_DIR == os.path.join(_tmp, "Qwen3-TTS")
    assert "Qwen3-TTS" in _folder_paths_stub.folder_names_and_paths
    print("[ok] models/Qwen3-TTS/ registered via folder_paths.add_model_folder_path")


def test_get_local_model_path_uses_known_folder_name():
    path = qt._get_local_model_path(_REPO_ID)
    assert path == os.path.join(qt.QWEN3_TTS_MODELS_DIR, "Qwen3-TTS-12Hz-1.7B-CustomVoice")
    print("[ok] _get_local_model_path: repo_id maps to its known folder name")


def test_loader_reuses_existing_local_folder_without_download():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    with patch.object(qt, "_download_model") as download:
        node = qt.QwenTTSModelsLoader()
        (bundle,) = node.load(_REPO_ID, "HuggingFace", "bf16", "auto")
        download.assert_not_called()
    assert bundle["model"] is _fake_model_instance

    call_kwargs = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
    assert call_kwargs["dtype"] is torch.bfloat16
    assert call_kwargs["device_map"] == torch.device("cpu")
    assert call_kwargs["attn_implementation"] == qt._resolve_attention("auto")
    print("[ok] loader: existing local folder is reused, no download call")
    _clear_local_folder()


def test_loader_downloads_when_folder_missing():
    qt._MODEL_CACHE.clear()
    _clear_local_folder()
    fake_path = _populate_local_folder()  # what the mocked download "produces"
    with patch.object(qt, "_download_model", return_value=fake_path) as download:
        # Folder starts empty from the loader's point of view: simulate by
        # clearing again right before, then let the mock "download" refill it.
        _clear_local_folder()
        node = qt.QwenTTSModelsLoader()
        node.load(_REPO_ID, "HuggingFace", "bf16", "auto")
        download.assert_called_once_with(_REPO_ID, "HuggingFace")
    print("[ok] loader: missing local folder triggers exactly one download call")
    _clear_local_folder()


def test_loader_caches_identical_widgets():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    node = qt.QwenTTSModelsLoader()
    node.load(_REPO_ID, "HuggingFace", "bf16", "auto")
    calls_before = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count
    node.load(_REPO_ID, "HuggingFace", "bf16", "auto")
    assert _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count == calls_before
    print("[ok] loader: identical widgets hit the cache, no reload")
    _clear_local_folder()


def test_loader_explicit_attention_is_forwarded():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "fp32", "sdpa")
    call_kwargs = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_args.kwargs
    assert call_kwargs["attn_implementation"] == "sdpa"
    assert call_kwargs["dtype"] is torch.float32
    print("[ok] loader: non-'auto' attention/precision forwarded explicitly")
    _clear_local_folder()


def test_loader_local_model_path_requires_speech_tokenizer():
    qt._MODEL_CACHE.clear()
    bad_dir = tempfile.mkdtemp(prefix="qwen_tts_bad_")
    try:
        qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto",
                                      local_model_path=bad_dir)
    except ValueError as e:
        assert "speech_tokenizer" in str(e)
        print("[ok] loader: local_model_path without speech_tokenizer/ raises ValueError")
        return
    raise AssertionError("expected ValueError")


def test_loader_local_model_path_override_skips_download():
    qt._MODEL_CACHE.clear()
    good_dir = tempfile.mkdtemp(prefix="qwen_tts_good_")
    os.makedirs(os.path.join(good_dir, "speech_tokenizer"), exist_ok=True)
    with patch.object(qt, "_download_model") as download:
        qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto",
                                      local_model_path=good_dir)
        download.assert_not_called()
    call_kwargs = _qwen_tts_module.Qwen3TTSModel.from_pretrained
    assert call_kwargs.call_args.args[0] == good_dir
    print("[ok] loader: valid local_model_path override is used directly, no download")


def test_generate_builds_correct_audio_dict_and_seeds():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    (bundle,) = qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    with patch.object(torch, "manual_seed") as seed_fn:
        (audio,) = node.generate(
            bundle, "hello world", "Dylan", "Auto", 1234,
            2048, 0.8, 20, 1.0, 1.05)
        seed_fn.assert_called_once_with(1234)
    assert audio["sample_rate"] == 24000
    assert audio["waveform"].shape == (1, 1, 4000)
    print("[ok] generate: seeds torch and builds a [1,1,T] AUDIO dict at the right rate")
    _clear_local_folder()


def test_generate_passes_none_language_for_auto():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    (bundle,) = qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    node.generate(bundle, "hi", "Dylan", "Auto", 0, 2048, 0.8, 20, 1.0, 1.05)
    call_kwargs = _fake_model_instance.generate_custom_voice.call_args.kwargs
    assert call_kwargs["language"] is None
    assert call_kwargs["speaker"] == "Dylan"
    print("[ok] generate: 'Auto' language resolves to None, speaker passed through")
    _clear_local_folder()


def test_generate_custom_speaker_name_overrides_dropdown():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    (bundle,) = qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    node.generate(bundle, "hi", "Dylan", "Auto", 0, 2048, 0.8, 20, 1.0, 1.05,
                  custom_speaker_name="MyCloneVoice")
    call_kwargs = _fake_model_instance.generate_custom_voice.call_args.kwargs
    assert call_kwargs["speaker"] == "MyCloneVoice"
    print("[ok] generate: custom_speaker_name overrides the speaker dropdown")
    _clear_local_folder()


def test_generate_unload_clears_cache():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    (bundle,) = qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto")
    assert bundle["key"] in qt._MODEL_CACHE
    node = qt.QwenTTSCustomVoiceGenerate()
    node.generate(bundle, "hi", "Dylan", "Auto", 0, 2048, 0.8, 20, 1.0, 1.05,
                  instruct="", unload_model_after_generate=True)
    assert bundle["key"] not in qt._MODEL_CACHE
    print("[ok] generate: unload_model_after_generate evicts the cache entry")
    _clear_local_folder()


def test_generate_wrong_model_type_error_is_remapped():
    qt._MODEL_CACHE.clear()
    _populate_local_folder()
    (bundle,) = qt.QwenTTSModelsLoader().load(_REPO_ID, "HuggingFace", "bf16", "auto")
    node = qt.QwenTTSCustomVoiceGenerate()
    _fake_model_instance.generate_custom_voice.side_effect = ValueError(
        "Qwen3TTSBaseForConditionalGeneration does not support generate_custom_voice")
    try:
        node.generate(bundle, "hi", "Dylan", "Auto", 0, 2048, 0.8, 20, 1.0, 1.05)
    except ValueError as e:
        assert "CustomVoice" in str(e)
        print("[ok] generate: wrong-model-type ValueError is remapped to a plainer message")
        return
    finally:
        _fake_model_instance.generate_custom_voice.side_effect = _fake_generate_custom_voice
        _clear_local_folder()
    raise AssertionError("expected ValueError")


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
    test_folder_registered_under_models_dir()
    test_get_local_model_path_uses_known_folder_name()
    test_loader_reuses_existing_local_folder_without_download()
    test_loader_downloads_when_folder_missing()
    test_loader_caches_identical_widgets()
    test_loader_explicit_attention_is_forwarded()
    test_loader_local_model_path_requires_speech_tokenizer()
    test_loader_local_model_path_override_skips_download()
    test_generate_builds_correct_audio_dict_and_seeds()
    test_generate_passes_none_language_for_auto()
    test_generate_custom_speaker_name_overrides_dropdown()
    test_generate_unload_clears_cache()
    test_generate_wrong_model_type_error_is_remapped()
    test_instruct_drop_warning_fires_for_06b_size_string()
    print("[ok] all nodes_qwen_tts smoke tests passed")
