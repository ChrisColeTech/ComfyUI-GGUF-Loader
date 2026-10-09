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
_fake_model_instance.generate_voice_clone.side_effect = _fake_generate_custom_voice
_fake_model_instance.generate_voice_design.side_effect = _fake_generate_custom_voice
_fake_model_instance.create_voice_clone_prompt.return_value = ["PROMPT_ITEMS"]

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


def test_rope_default_init_resolves_theta_from_all_config_shapes():
    # The restored ROPE_INIT_FUNCTIONS["default"] shim must resolve the base
    # frequency from a flat rope_theta (qwen-tts's own configs), a
    # rope_parameters dict (transformers' Mimi codec config - the speech
    # tokenizer's AutoModel pulls it in and its weight init routes through
    # this entry; a bare config.rope_theta read crashed from_pretrained with
    # AttributeError: 'MimiConfig' object has no attribute 'rope_theta'), and
    # neither (transformers 4.x's 10000.0 default).
    import math

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    fn = ROPE_INIT_FUNCTIONS.get("default")
    assert fn is not None

    def _invfreq_for_base(base, head_dim=8):
        dim = head_dim
        expo = torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float32) / dim
        return 1.0 / (base ** expo)

    # 1. flat rope_theta (qwen-tts's own config style)
    cfg = types.SimpleNamespace(rope_theta=200.0, head_dim=8)
    inv, scale = fn(config=cfg, device=None)
    assert torch.allclose(inv, _invfreq_for_base(200.0)) and scale == 1.0

    # 2. Mimi style: NO flat attribute, dict instead
    cfg = types.SimpleNamespace(rope_parameters={"rope_theta": 500.0}, head_dim=8)
    inv, _ = fn(config=cfg, device=None)
    assert torch.allclose(inv, _invfreq_for_base(500.0))

    # 3. neither -> transformers 4.x default
    cfg = types.SimpleNamespace(head_dim=8)
    inv, _ = fn(config=cfg, device=None)
    assert torch.allclose(inv, _invfreq_for_base(10000.0))

    # 4. head_dim fallback via hidden_size // num_attention_heads
    cfg = types.SimpleNamespace(rope_theta=100.0, hidden_size=16, num_attention_heads=2)
    inv, _ = fn(config=cfg, device=None)
    assert torch.allclose(inv, _invfreq_for_base(100.0, head_dim=8))
    print("[ok] rope default-init: theta resolved from flat attr, "
          "rope_parameters dict, and the 4.x default")


def _populate_local_model(family, size="0.6B"):
    """Pretend the flybirdxx-tree model folder exists (models/qwen-tts/...) so
    `_load_qwen_model`'s local search finds it and never downloads."""
    path = os.path.join(qt._QTTS_MODELS_DIR, f"Qwen3-TTS-12Hz-{size}-{family}")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        f.write("{}")
    return path


def _clear_local_models():
    import shutil
    qt._MODEL_CACHE.clear()
    if os.path.isdir(qt._QTTS_MODELS_DIR):
        for d in os.listdir(qt._QTTS_MODELS_DIR):
            if d != "voices":
                shutil.rmtree(os.path.join(qt._QTTS_MODELS_DIR, d), ignore_errors=True)


def _fake_audio_dict(stereo=False, samples=4410, sr=24000):
    c = 2 if stereo else 1
    return {"waveform": torch.randn(1, c, samples), "sample_rate": sr}


def test_model_loader_local_search_and_cache():
    qt._MODEL_CACHE.clear()
    _populate_local_model("Base")
    node = qt.QwenTTSVoiceCloneGenerate()
    node.generate("hi", "0.6B", "cpu", "fp32", "Auto",
                  ref_audio=_fake_audio_dict(samples=1200))
    call = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_args
    assert call.args[0].endswith("Qwen3-TTS-12Hz-0.6B-Base")
    assert call.kwargs["attn_implementation"] == "sdpa"  # auto on a backend-less box
    assert call.kwargs["dtype"] is torch.float32

    calls_before = _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count
    node.generate("hi again", "0.6B", "cpu", "fp32", "Auto",
                  ref_audio=_fake_audio_dict(samples=1200))
    assert _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count == calls_before
    print("[ok] loader: local qwen-tts tree found, kwargs correct, cache reused")

    # a different config evicts the single-slot cache (their unload semantics)
    _populate_local_model("Base", "1.7B")
    node.generate("hi", "1.7B", "cpu", "fp32", "Auto",
                  ref_audio=_fake_audio_dict(samples=1200))
    assert _qwen_tts_module.Qwen3TTSModel.from_pretrained.call_count == calls_before + 1
    print("[ok] loader: changed model config reloads (single-slot cache)")
    _clear_local_models()


def test_voice_design_06b_is_rejected():
    try:
        qt.QwenTTSVoiceDesignGenerate().generate(
            "hi", "a voice", "0.6B", "cpu", "fp32", "Auto")
    except RuntimeError as e:
        assert "1.7B" in str(e)
        print("[ok] voice design: 0.6B rejected (upstream: VoiceDesign is 1.7B only)")
        return
    raise AssertionError("expected RuntimeError")


def test_voice_clone_requires_ref_audio_or_prompt():
    _populate_local_model("Base")
    try:
        qt.QwenTTSVoiceCloneGenerate().generate(
            "hi", "0.6B", "cpu", "fp32", "Auto")
    except RuntimeError as e:
        assert "reference audio or voice clone prompt" in str(e)
        print("[ok] voice clone: neither ref_audio nor prompt raises a clear error")
        _clear_local_models()
        return
    raise AssertionError("expected RuntimeError")


def test_voice_clone_converts_comfyui_audio_and_kwargs():
    qt._MODEL_CACHE.clear()
    _populate_local_model("Base")
    node = qt.QwenTTSVoiceCloneGenerate()
    with patch.object(torch, "manual_seed") as seed_fn:
        (audio,) = node.generate("hello", "0.6B", "cpu", "fp32", "English",
                                 seed=7, max_new_tokens=512,
                                 ref_audio=_fake_audio_dict(stereo=True),
                                 ref_text="  ", x_vector_only=True,
                                 instruct=" calm ")
        seed_fn.assert_called_once_with(7)
    kw = _fake_model_instance.generate_voice_clone.call_args.kwargs
    wav, sr = kw["ref_audio"]
    assert isinstance(wav, np.ndarray) and wav.ndim == 1 and wav.dtype == np.float32
    assert sr == 24000
    assert kw["ref_text"] is None          # blank -> None (upstream contract)
    assert kw["instruct"] == " calm "      # kept verbatim, blank would be None
    assert kw["x_vector_only_mode"] is True
    assert kw["language"] == "english"     # flybirdxx's map lowercases
    assert audio["waveform"].shape == (1, 1, 4000) and audio["sample_rate"] == 24000
    print("[ok] voice clone: stereo AUDIO dict -> mono 1-D tuple, kwargs match upstream")
    _clear_local_models()


def test_voice_clone_prompt_wins_over_ref_audio():
    qt._MODEL_CACHE.clear()
    _populate_local_model("Base")
    node = qt.QwenTTSVoiceCloneGenerate()
    node.generate("hi", "0.6B", "cpu", "fp32", "Auto",
                  ref_audio=_fake_audio_dict(),
                  voice_clone_prompt=["PROMPT_ITEMS"])
    kw = _fake_model_instance.generate_voice_clone.call_args.kwargs
    assert kw["voice_clone_prompt"] == ["PROMPT_ITEMS"] and kw["ref_audio"] is None
    print("[ok] voice clone: a connected prompt takes precedence over ref_audio")
    _clear_local_models()


def test_voice_clone_short_reference_is_padded():
    qt._MODEL_CACHE.clear()
    _populate_local_model("Base")
    node = qt.QwenTTSVoiceCloneGenerate()
    node.generate("hi", "0.6B", "cpu", "fp32", "Auto",
                  ref_audio=_fake_audio_dict(samples=100))
    wav, _sr = _fake_model_instance.generate_voice_clone.call_args.kwargs["ref_audio"]
    assert wav.shape[0] == 1024  # upstream's own minimum
    print("[ok] voice clone: sub-1024-sample references are zero-padded")
    _clear_local_models()


def test_voice_clone_prompt_node_passes_through():
    qt._MODEL_CACHE.clear()
    _populate_local_model("Base")
    node = qt.QwenTTSVoiceClonePrompt()
    (prompt,) = node.create_prompt(_fake_audio_dict(), "the transcript",
                                   "0.6B", "cpu", "fp32", "auto")
    kw = _fake_model_instance.create_voice_clone_prompt.call_args.kwargs
    assert kw["ref_text"] == "the transcript" and kw["x_vector_only_mode"] is False
    assert prompt == ["PROMPT_ITEMS"]
    print("[ok] voice clone prompt: extraction kwargs + passthrough return")
    _clear_local_models()


def test_voice_design_requires_instruct_and_generates():
    qt._MODEL_CACHE.clear()
    _populate_local_model("VoiceDesign", "1.7B")
    node = qt.QwenTTSVoiceDesignGenerate()
    try:
        node.generate("hi", "", "1.7B", "cpu", "fp32", "Auto")
    except RuntimeError as e:
        assert "required" in str(e)
        print("[ok] voice design: blank instruct rejected")
    else:
        raise AssertionError("expected RuntimeError")

    (audio,) = node.generate("hello there", "a deep calm male voice",
                             "1.7B", "cpu", "fp32", "English", 3, 512,
                             1.0, 50, 0.9, 1.05)
    kw = _fake_model_instance.generate_voice_design.call_args.kwargs
    assert kw["instruct"] == "a deep calm male voice"
    assert kw["language"] == "english"      # mapped lowercase; Auto would be None
    assert audio["waveform"].shape == (1, 1, 4000)
    print("[ok] voice design: valid call forwards instruct + mapped language")
    _clear_local_models()


def test_role_bank_collects_named_prompts_and_skips_empty():
    node = qt.QwenTTSRoleBank()
    (bank,) = node.create_bank(
        role_name_1="Narrator", prompt_1=["p1"],
        role_name_2="  ",           # blank name -> skipped even with a prompt
        role_name_3="Role3",        # comfy fills the widget default when unwired
        prompt_3=["p3"],
        role_name_5="Ghost",        # name, no prompt -> skipped
    )
    assert bank == {"Narrator": ["p1"], "Role3": ["p3"]}
    print("[ok] role bank: named prompts collected, blank/empty slots skipped")


def test_save_and_load_speaker_roundtrip():
    import json as _json
    # a real picklable prompt (MagicMock is not torch.save-able)
    prompt = [{"emb": torch.zeros(4)}, "meta"]
    audio = _fake_audio_dict(samples=2000)
    save = qt.QwenTTSSaveVoice()
    save.save(prompt, "..\\evil name.qvp", audio=audio, ref_text="the transcript")
    voices = qt._voices_dir()
    stem = "evil name"  # basename + extension strip - no traversal
    assert os.path.isfile(os.path.join(voices, stem + ".qvp"))
    assert os.path.isfile(os.path.join(voices, stem + ".json"))
    assert os.path.isfile(os.path.join(voices, stem + ".wav"))  # stdlib fallback wrote it
    with open(os.path.join(voices, stem + ".json"), encoding="utf-8") as f:
        assert _json.load(f)["ref_text"] == "the transcript"
    assert stem + ".wav" in qt.QwenTTSLoadSpeaker.INPUT_TYPES()["required"]["filename"][0]

    (loaded_prompt, loaded_audio, ref_text) = qt.QwenTTSLoadSpeaker().load_speaker(stem + ".wav")
    assert loaded_prompt[1] == "meta"                          # .qvp fast-load round-trips
    assert torch.equal(loaded_prompt[0]["emb"], torch.zeros(4))
    assert ref_text == "the transcript"       # .json transcript restored
    assert loaded_audio["waveform"].shape[0] == 1 and loaded_audio["waveform"].shape[1] == 1
    assert loaded_audio["waveform"].shape[2] == 2000 and loaded_audio["sample_rate"] == 24000
    print("[ok] save/load speaker: qvp+json+wav written (no audio backend needed), "
          "prompt/transcript/audio round-trip")

    # missing .qvp -> prompt None, audio still usable via ref_audio
    os.remove(os.path.join(voices, stem + ".qvp"))
    (p2, a2, t2) = qt.QwenTTSLoadSpeaker().load_speaker(stem + ".wav")
    assert p2 is None and t2 == "the transcript" and a2["waveform"].shape[-1] == 2000
    print("[ok] load speaker: no cached .qvp -> prompt None, audio+transcript intact")
    import contextlib
    for ext in (".qvp", ".json", ".wav"):  # .qvp may already be gone (sub-test above)
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(voices, stem + ext))


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
    test_rope_default_init_resolves_theta_from_all_config_shapes()
    test_model_loader_local_search_and_cache()
    test_voice_design_06b_is_rejected()
    test_voice_clone_requires_ref_audio_or_prompt()
    test_voice_clone_converts_comfyui_audio_and_kwargs()
    test_voice_clone_prompt_wins_over_ref_audio()
    test_voice_clone_short_reference_is_padded()
    test_voice_clone_prompt_node_passes_through()
    test_voice_design_requires_instruct_and_generates()
    test_role_bank_collects_named_prompts_and_skips_empty()
    test_save_and_load_speaker_roundtrip()
    print("[ok] all nodes_qwen_tts smoke tests passed")
