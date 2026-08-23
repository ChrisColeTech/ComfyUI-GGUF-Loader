# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Qwen3-TTS CustomVoice: local loader + generate, in one node pair.

  QwenTTSModelsLoader        local HF-layout folder -> a Qwen3TTSModel + processor
  QwenTTSCustomVoiceGenerate text + built-in speaker (+ optional instruct) -> AUDIO

Replaces the ``flybirdxx/ComfyUI-Qwen-TTS`` pack's ``FB_Qwen3TTSCustomVoice``
node used by the ``ltxv23_talking_head`` gallery workflow, wrapping the
``qwen-tts`` pip package's own ``Qwen3TTSModel`` directly rather than
reimplementing it - the model itself (a transformers ``PreTrainedModel`` +
a separate codec/vocoder submodel) is not this repo's GGUF-quantization
territory, so there is nothing to port at the weights level, only a comfy
face for it.

Gotchas worth knowing before touching this file (from reading
``qwen_tts/inference/qwen3_tts_model.py`` directly, not the package's docs):

  * ``Qwen3TTSModel.from_pretrained(path, **kwargs)`` only network-fetches
    when ``path`` is NOT a local directory and ``local_files_only`` is not
    set - passing a local folder path with ``local_files_only=True`` (what
    this loader always does) makes the load fully offline, matching this
    repo's "no HuggingFace runtime downloads" convention (see AGENTS.md's
    Scenema Audio section);
  * the speech tokenizer/codec is NOT a separate download - it lives in a
    ``speech_tokenizer/`` subfolder inside the CustomVoice repo itself, so
    one local folder is the whole model, not two;
  * ``instruct`` is real, model-native conditioning (tokenized as a chat
    turn and passed as ``instruct_ids``) for the 1.7B checkpoint, but the
    package SILENTLY drops it for the 0.6B checkpoint
    (``if tts_model_size in "0b6": instruct = None`` - a substring check,
    not a whitelist). This node logs a warning instead of reproducing that
    silence;
  * the package's own sampling defaults (``top_k=50, top_p=1.0,
    temperature=0.9``) differ from what the reference workflow's node
    actually ships (``top_k=20, top_p=0.8, temperature=1.0``) - this node's
    defaults match the reference workflow, not the package;
  * bad speaker/language names raise ``ValueError`` listing exactly what IS
    supported - let those propagate unwrapped, they are already the most
    useful message available.
"""
import logging
import os

import folder_paths
import torch

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:  # pragma: no cover - optional dependency, see requirements.txt
    Qwen3TTSModel = None

logger = logging.getLogger(__name__)

QWEN_TTS_CATEGORY = "\U0001F916 CCTech/Qwen TTS"

_QWEN_TTS_DIR = os.path.join(folder_paths.models_dir, "qwen_tts")
if "qwen_tts" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["qwen_tts"] = ([_QWEN_TTS_DIR], set())

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_ATTENTION_CHOICES = ["auto", "sdpa", "eager", "flash_attention_2"]

# Keeps the most recently loaded model resident so re-queuing a graph with
# unchanged loader widgets does not reload multi-GB weights every prompt -
# same rationale as nodes_ltx23.py's _ENCODER_CACHE.
_MODEL_CACHE = {}


def _model_paths():
    """Subdirectories of models/qwen_tts that look like a Qwen3-TTS repo
    (own config.json at the root - mirrors comfy-core's DiffusersLoader
    scanning for model_index.json, same idea for a plain transformers repo).
    """
    paths = []
    for search_path in folder_paths.get_folder_paths("qwen_tts"):
        if not os.path.isdir(search_path):
            continue
        for root, _dirs, files in os.walk(search_path, followlinks=True):
            if "config.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
    return paths


class QwenTTSModelsLoader:
    """Load a local Qwen3-TTS CustomVoice checkpoint folder.

    Point ``model_path`` at a folder under ``models/qwen_tts`` holding a
    downloaded ``Qwen/Qwen3-TTS-12Hz-<size>-CustomVoice`` repo (e.g. via
    ``huggingface-cli download <repo> --local-dir models/qwen_tts/<name>``).
    Always loads with ``local_files_only=True`` - no network access at
    generate time.
    """

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Models Loader ⚡"
    RETURN_TYPES = ("QWEN_TTS_MODEL",)
    FUNCTION = "load"
    DESCRIPTION = "Load a local Qwen3-TTS CustomVoice checkpoint folder."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (_model_paths(), {
                    "tooltip": "A folder under models/qwen_tts containing a "
                               "downloaded Qwen3-TTS-*-CustomVoice repo."}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                "precision": (list(_DTYPES), {"default": "bf16"}),
                "attention": (_ATTENTION_CHOICES, {"default": "auto"}),
            },
        }

    def load(self, model_path, device, precision, attention):
        if Qwen3TTSModel is None:
            raise RuntimeError(
                "The 'qwen-tts' package is required for QwenTTSModelsLoader "
                "but is not installed (pip install qwen-tts).")

        resolved = None
        for search_path in folder_paths.get_folder_paths("qwen_tts"):
            candidate = os.path.join(search_path, model_path)
            if os.path.isfile(os.path.join(candidate, "config.json")):
                resolved = candidate
                break
        if resolved is None:
            raise FileNotFoundError(
                f"Qwen3-TTS model {model_path!r} not found under models/qwen_tts.")

        key = (resolved, device, precision, attention)
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)

        kwargs = {"dtype": _DTYPES[precision], "device_map": device,
                  "local_files_only": True}
        if attention != "auto":
            kwargs["attn_implementation"] = attention

        model = Qwen3TTSModel.from_pretrained(resolved, **kwargs)
        logger.info("Qwen3-TTS: loaded %s on %s (%s, attn=%s)",
                    model_path, device, precision, attention)

        _MODEL_CACHE.clear()
        bundle = {"model": model, "key": key}
        _MODEL_CACHE[key] = bundle
        return (bundle,)


def _instruct_is_silently_dropped(model):
    """Best-effort mirror of the package's own ``tts_model_size in "0b6"``
    check, so the warning fires under the same condition the drop does.
    Never raises - if the attribute path does not exist, stay silent
    rather than block generation over a diagnostics-only check.
    """
    try:
        return model.model.tts_model_size in "0b6"
    except AttributeError:
        return False


class QwenTTSCustomVoiceGenerate:
    """Text + a built-in named speaker (+ optional instruct) -> AUDIO.

    Bad ``speaker``/``language`` values raise ComfyUI's normal execution
    error with the package's own message, which already lists every
    supported name - there is nothing this node could add.
    """

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Custom Voice ⚡"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    DESCRIPTION = ("Generate speech with a Qwen3-TTS CustomVoice model's "
                   "built-in named speakers.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_tts_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello world"}),
                "speaker": ("STRING", {"default": "",
                    "tooltip": "A built-in speaker name from the loaded "
                               "checkpoint (e.g. 'Dylan'). Wrong name raises "
                               "an error listing every valid one."}),
                "language": ("STRING", {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Voice-design instruction. Ignored by 0.6B "
                               "CustomVoice checkpoints (upstream behavior)."}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    def generate(self, qwen_tts_model, text, speaker, language, seed,
                max_new_tokens, top_p, top_k, temperature, repetition_penalty,
                instruct="", unload_model_after_generate=False):
        model = qwen_tts_model["model"]

        if instruct and _instruct_is_silently_dropped(model):
            logger.warning("Qwen3-TTS: instruct is set but this checkpoint "
                           "is a 0.6B CustomVoice model, which silently "
                           "ignores it (upstream behavior). Use a 1.7B "
                           "checkpoint if the instruction matters.")

        torch.manual_seed(seed)
        wavs, sample_rate = model.generate_custom_voice(
            text=text, speaker=speaker,
            language=None if language.strip().lower() == "auto" else language,
            instruct=instruct or None,
            do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        )
        waveform = torch.from_numpy(wavs[0]).float()[None, None, :]
        audio = {"waveform": waveform, "sample_rate": int(sample_rate)}
        logger.info("Qwen3-TTS: generated %s @ %d Hz (speaker=%s)",
                    tuple(waveform.shape), sample_rate, speaker)

        if unload_model_after_generate:
            _MODEL_CACHE.pop(qwen_tts_model["key"], None)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (audio,)


NODE_CLASS_MAPPINGS = {
    "QwenTTSModelsLoader": QwenTTSModelsLoader,
    "QwenTTSCustomVoiceGenerate": QwenTTSCustomVoiceGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelsLoader": QwenTTSModelsLoader.TITLE,
    "QwenTTSCustomVoiceGenerate": QwenTTSCustomVoiceGenerate.TITLE,
}
