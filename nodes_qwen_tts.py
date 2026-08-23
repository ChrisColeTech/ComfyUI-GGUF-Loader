# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Qwen3-TTS CustomVoice: local loader + generate, in one node pair.

  QwenTTSModelsLoader        repo_id -> a Qwen3TTSModel (downloads once, then local)
  QwenTTSCustomVoiceGenerate text + speaker (+ optional instruct) -> AUDIO

Replaces the ``flybirdxx/ComfyUI-Qwen-TTS`` pack's ``FB_Qwen3TTSCustomVoice``
node used by the ``ltxv23_talking_head`` gallery workflow, wrapping the
``qwen-tts`` pip package's own ``Qwen3TTSModel`` directly - the model itself
(a transformers ``PreTrainedModel`` plus a separate codec/vocoder submodel)
is not this repo's GGUF-quantization territory, so there is nothing to port
at the weights level, only a comfy face for it.

The models-folder layout and download mechanics here deliberately mirror
``DarioFT/ComfyUI-Qwen3-TTS`` (a full comfy node pack for this same
``qwen-tts`` package, checked at ``D:\\Projects\\ComfyUI\\ComfyUI-Qwen3-TTS-main``)
rather than inventing a new convention: ``models/Qwen3-TTS/<folder_name>/``,
registered via ``folder_paths.add_model_folder_path``, a fixed ``repo_id``
dropdown of the known CustomVoice/VoiceDesign/Base x 1.7B/0.6B repos, and an
auto-download-on-first-use path (via ``huggingface_hub.snapshot_download``,
or ``modelscope.snapshot_download`` if ``source="ModelScope"``) that first
checks for an existing HF/ModelScope cache to migrate in place rather than
re-downloading. Once a repo's files are on disk under that folder, loading
never touches the network again.

Gotchas worth knowing before touching this file (from reading
``qwen_tts/inference/qwen3_tts_model.py`` and DarioFT's pack directly, not
either one's docs):

  * the speech tokenizer/codec is NOT a separate download - it lives in a
    ``speech_tokenizer/`` subfolder inside the CustomVoice repo itself, so
    one repo id is the whole model, not two (the package's own
    ``get_speaker_names``-adjacent code reads ``speech_tokenizer/config.json``
    via ``cached_file(pretrained_model_name_or_path, ...)`` - i.e. relative
    to whatever single path/repo you pass);
  * ``instruct`` is real, model-native conditioning (tokenized as a chat
    turn and passed as ``instruct_ids``) for the 1.7B checkpoint, but the
    package SILENTLY drops it for the 0.6B checkpoint
    (``if tts_model_size in "0b6": instruct = None`` - a substring check,
    not a whitelist). This node logs a warning instead of reproducing that
    silence - DarioFT's pack does not check for this at all;
  * calling ``generate_custom_voice`` against a non-CustomVoice checkpoint
    (Base/VoiceDesign) raises a ``ValueError`` whose message is a raw
    "does not support generate_custom_voice" - re-raised here with a
    plainer "load a CustomVoice checkpoint" message, following the same
    catch DarioFT's pack does;
  * the package's own sampling defaults (``top_k=50, top_p=1.0,
    temperature=0.9``) differ from what the reference workflow's node
    actually ships (``top_k=20, top_p=0.8, temperature=1.0``) - this node's
    defaults match the reference workflow, not the package. DarioFT's own
    ``Qwen3CustomVoice`` node exposes none of these (only ``max_new_tokens``)
    - this node keeps the full sampling surface because the reference
    workflow's node needs it.
"""
import logging
import os
import shutil

import comfy.model_management
import folder_paths
import torch

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:  # pragma: no cover - optional dependency, see requirements.txt
    Qwen3TTSModel = None

logger = logging.getLogger(__name__)

QWEN_TTS_CATEGORY = "\U0001F916 CCTech/Qwen TTS"

QWEN3_TTS_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-TTS")
os.makedirs(QWEN3_TTS_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-TTS", QWEN3_TTS_MODELS_DIR)

# Known repo_id -> local folder name. Matches DarioFT/ComfyUI-Qwen3-TTS's
# mapping so a folder downloaded by either pack is interchangeable.
QWEN3_TTS_MODELS = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": "Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": "Qwen3-TTS-12Hz-0.6B-Base",
}

# The 9 built-in CustomVoice speakers, for a discoverable dropdown. A
# fine-tuned/custom checkpoint may ship different names - use
# custom_speaker_name to override with any string.
CUSTOM_VOICE_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
                        "Ryan", "Aiden", "Ono_Anna", "Sohee"]

CUSTOM_VOICE_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean",
                          "German", "French", "Russian", "Portuguese",
                          "Spanish", "Italian"]

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_ATTENTION_CHOICES = ["auto", "flash_attention_2", "sdpa", "eager"]

# Keeps the most recently loaded model resident so re-queuing a graph with
# unchanged loader widgets does not reload multi-GB weights every prompt -
# same rationale as nodes_ltx23.py's _ENCODER_CACHE.
_MODEL_CACHE = {}


def _get_local_model_path(repo_id):
    folder_name = QWEN3_TTS_MODELS.get(repo_id, repo_id.replace("/", "_"))
    return os.path.join(QWEN3_TTS_MODELS_DIR, folder_name)


def _migrate_cached_model(repo_id, target_path):
    """Adopt an existing HF/ModelScope cache copy instead of re-downloading."""
    if os.path.exists(target_path) and os.listdir(target_path):
        return True

    hf_model_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                                "hub", f"models--{repo_id.replace('/', '--')}")
    snapshots_dir = os.path.join(hf_model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            logger.info("Qwen3-TTS: migrating %s from the HF cache to %s",
                        repo_id, target_path)
            shutil.copytree(os.path.join(snapshots_dir, snapshots[0]),
                            target_path, dirs_exist_ok=True)
            return True

    ms_model_dir = os.path.join(os.path.expanduser("~"), ".cache", "modelscope",
                                "hub", *repo_id.split("/"))
    if os.path.isdir(ms_model_dir):
        logger.info("Qwen3-TTS: migrating %s from the ModelScope cache to %s",
                    repo_id, target_path)
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True

    return False


def _download_model(repo_id, source):
    target_path = _get_local_model_path(repo_id)
    if _migrate_cached_model(repo_id, target_path):
        return target_path

    os.makedirs(target_path, exist_ok=True)
    if source == "ModelScope":
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    logger.info("Qwen3-TTS: downloading %s from %s to %s", repo_id, source, target_path)
    snapshot_download(repo_id, local_dir=target_path)
    return target_path


def _installed_repo_ids():
    """repo_ids actually present under models/Qwen3-TTS/ right now - what's
    installed, not what's theoretically downloadable. Falls back to the
    full known list only when nothing is installed yet, so the dropdown is
    never empty on a fresh setup.
    """
    installed = [repo_id for repo_id, folder_name in QWEN3_TTS_MODELS.items()
                if os.path.isdir(os.path.join(QWEN3_TTS_MODELS_DIR, folder_name))
                and os.listdir(os.path.join(QWEN3_TTS_MODELS_DIR, folder_name))]
    return installed or list(QWEN3_TTS_MODELS)


def _resolve_attention(attention):
    if attention != "auto":
        return attention
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


class QwenTTSModelsLoader:
    """Load a Qwen3-TTS checkpoint, downloading it into ``models/Qwen3-TTS/``
    on first use (or migrating an existing HF/ModelScope cache copy) and
    reusing that local folder on every load after.
    """

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Models Loader ⚡"
    RETURN_TYPES = ("QWEN_TTS_MODEL",)
    FUNCTION = "load"
    DESCRIPTION = "Load a Qwen3-TTS checkpoint (downloads on first use)."

    @classmethod
    def INPUT_TYPES(s):
        installed = _installed_repo_ids()
        default = ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
                  if "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in installed
                  else installed[0])
        return {
            "required": {
                "repo_id": (installed, {"default": default,
                    "tooltip": "Models found under models/Qwen3-TTS/. To add "
                               "another variant, download it with the CLI "
                               "into models/Qwen3-TTS/<name>/ and re-add this "
                               "node (see README)."}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (list(_DTYPES), {"default": "bf16"}),
                "attention": (_ATTENTION_CHOICES, {"default": "auto"}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default": "",
                    "tooltip": "Override: an existing local folder containing "
                               "a full model (must have a speech_tokenizer/ "
                               "subfolder). Leave blank to use repo_id."}),
            },
        }

    def load(self, repo_id, source, precision, attention, local_model_path=""):
        if Qwen3TTSModel is None:
            raise RuntimeError(
                "The 'qwen-tts' package is required for QwenTTSModelsLoader "
                "but is not installed (pip install qwen-tts).")

        local_model_path = local_model_path.strip()
        if local_model_path:
            if not os.path.isdir(os.path.join(local_model_path, "speech_tokenizer")):
                raise ValueError(
                    f"local_model_path {local_model_path!r} has no "
                    f"speech_tokenizer/ subfolder - it is not a full Qwen3-TTS "
                    f"model folder. Leave local_model_path blank to use repo_id "
                    f"instead.")
            model_path = local_model_path
        else:
            model_path = _get_local_model_path(repo_id)
            if not (os.path.isdir(model_path) and os.listdir(model_path)):
                model_path = _download_model(repo_id, source)

        device = comfy.model_management.get_torch_device()
        dtype = _DTYPES[precision]
        if precision == "bf16" and device.type == "mps":
            dtype = torch.float16  # bf16 has limited support on MPS
        attn_impl = _resolve_attention(attention)

        key = (model_path, str(device), precision, attn_impl)
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)

        model = Qwen3TTSModel.from_pretrained(
            model_path, device_map=device, dtype=dtype, attn_implementation=attn_impl)
        logger.info("Qwen3-TTS: loaded %s on %s (%s, attn=%s)",
                    model_path, device, precision, attn_impl)

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
    """Text + a built-in named speaker (+ optional instruct) -> AUDIO."""

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
                "speaker": (CUSTOM_VOICE_SPEAKERS, {"default": "Dylan"}),
                "language": (CUSTOM_VOICE_LANGUAGES, {"default": "Auto"}),
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
                "custom_speaker_name": ("STRING", {"default": "",
                    "tooltip": "Overrides speaker with any string - for a "
                               "fine-tuned checkpoint's own speaker names."}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    def generate(self, qwen_tts_model, text, speaker, language, seed,
                max_new_tokens, top_p, top_k, temperature, repetition_penalty,
                instruct="", custom_speaker_name="", unload_model_after_generate=False):
        model = qwen_tts_model["model"]
        target_speaker = custom_speaker_name.strip() or speaker

        if instruct and _instruct_is_silently_dropped(model):
            logger.warning("Qwen3-TTS: instruct is set but this checkpoint "
                           "is a 0.6B CustomVoice model, which silently "
                           "ignores it (upstream behavior). Use a 1.7B "
                           "checkpoint if the instruction matters.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            wavs, sample_rate = model.generate_custom_voice(
                text=text, speaker=target_speaker,
                language=None if language == "Auto" else language,
                instruct=instruct or None,
                do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
            )
        except ValueError as e:
            if "does not support generate_custom_voice" in str(e):
                raise ValueError(
                    "This checkpoint is not a CustomVoice model (Base/"
                    "VoiceDesign checkpoints don't support generate_custom_"
                    "voice). Load a *-CustomVoice repo in the models loader."
                ) from e
            raise

        waveform = torch.from_numpy(wavs[0]).float()[None, None, :]
        audio = {"waveform": waveform, "sample_rate": int(sample_rate)}
        logger.info("Qwen3-TTS: generated %s @ %d Hz (speaker=%s)",
                    tuple(waveform.shape), sample_rate, target_speaker)

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
