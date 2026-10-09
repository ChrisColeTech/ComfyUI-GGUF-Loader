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
import sys

import comfy.model_management
import folder_paths
import torch

def _patch_check_model_inputs():
    """qwen-tts 0.1.1 (latest on PyPI as of 2026-09) decorates with
    ``@check_model_inputs()`` - the parenthesized factory form - while
    transformers 5.x ships ``check_model_inputs(func)`` bare-only, so
    importing qwen_tts under a current transformers dies with
    ``TypeError: check_model_inputs() missing 1 required positional
    argument: 'func'`` (which used to take this whole pack down with it,
    since the error is a TypeError, not an ImportError). Teach the bare
    form to also accept the no-argument call. Only patches when the
    installed transformers actually needs it, so it is inert on any
    version that already supports the factory form - and once qwen-tts
    ships a release fixed for transformers 5.x.
    """
    import inspect

    import transformers.utils.generic as _generic

    original = _generic.check_model_inputs
    try:
        inspect.signature(original).bind()
        return
    except TypeError:
        pass

    def _compat(func=None, **_kwargs):
        if func is None:
            def _decorator(f):
                return original(f)
            return _decorator
        return original(func)

    _generic.check_model_inputs = _compat


def _patch_rope_default_init():
    """transformers 5.x removed both the ``"default"`` entry and the
    ``_compute_default_rope_parameters`` function from
    ``transformers.modeling_rope_utils`` - qwen-tts's four rotary embedding
    classes look up ``ROPE_INIT_FUNCTIONS[self.rope_type]`` with
    ``rope_type == "default"`` whenever a config has no rope_scaling, which
    made model construction die with ``KeyError: 'default'``. Restore the
    standard theta-based default (ported from transformers 4.x's
    modeling_rope_utils, Apache-2.0) under the same registry key.

    Later 5.x point releases restored their own ``"default"`` - but with a
    bare ``config.rope_theta`` read that crashes on configs which keep the
    base frequency elsewhere (transformers' own Mimi codec config stores it
    in the ``rope_parameters`` dict; its weight init routes through this
    entry during ``from_pretrained`` -> ``AttributeError: 'MimiConfig'
    object has no attribute 'rope_theta'``). So a present-but-fragile entry
    is probed and replaced too; inert once transformers ships a "default"
    that handles those config shapes.
    """
    import types as _types

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def _compute_default_rope_parameters(config=None, device=None, **kwargs):
        # Resolve the base frequency wherever the config family keeps it:
        # flat ``rope_theta`` (qwen-tts's own configs), the
        # ``rope_parameters`` dict (Mimi), or the 4.x default.
        base = getattr(config, "rope_theta", None)
        if base is None:
            params = getattr(config, "rope_parameters", None)
            if isinstance(params, dict):
                base = params.get("rope_theta")
        if base is None:
            base = 10000.0
        partial_rotary_factor = (getattr(config, "partial_rotary_factor", None)
                                 or 1.0)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                    .to(device=device, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    existing = ROPE_INIT_FUNCTIONS.get("default", None)
    if existing is not None:
        try:
            # probe with a Mimi-shaped config: no flat rope_theta, no
            # rope_parameters -> must fall back, never raise
            existing(config=_types.SimpleNamespace(head_dim=8), device=None)
            return  # handles every shape we know about - leave it alone
        except Exception:
            pass  # fragile (e.g. 5.17's bare config.rope_theta read)

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_masking_kwarg():
    """transformers 5.x restructured ``create_causal_mask``/``create_sliding_
    window_causal_mask``: the embedding kwarg was renamed ``input_embeds`` ->
    ``inputs_embeds`` and ``cache_position`` was removed outright. qwen-tts
    still calls the 4.x shape, which died mid-generation (first model
    forward) with ``TypeError: ... unexpected keyword argument``. Wrap both
    to rename the embedding kwarg and drop kwargs the installed signature no
    longer accepts. Applied BEFORE importing qwen_tts because the package
    binds these names with a module-level ``from ... import``. Inert on any
    transformers that already accepts the 4.x call shape.
    """
    import inspect

    import transformers.masking_utils as masking

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(masking, name, None)
        if original is None:
            continue
        try:
            params = inspect.signature(original).parameters
        except (TypeError, ValueError):
            continue
        accepts_old = "input_embeds" in params
        if accepts_old and any(p.kind == inspect.Parameter.VAR_KEYWORD
                               for p in params.values()):
            continue
        accepted = {n for n, p in params.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)}
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in params.values())

        def _make_wrapper(_original, _accepted, _has_var_keyword, _accepts_old):
            def _wrapper(*args, **kwargs):
                if not _accepts_old and "input_embeds" in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                if not _has_var_keyword:
                    kwargs = {k: v for k, v in kwargs.items()
                              if k in _accepted}
                return _original(*args, **kwargs)
            return _wrapper

        setattr(masking, name, _make_wrapper(original, accepted,
                                             has_var_keyword, accepts_old))


_patch_check_model_inputs()
_patch_rope_default_init()
_patch_masking_kwarg()


def _patch_legacy_config_defaults():
    """transformers 5.x stopped defaulting the legacy token-id attributes
    (``pad_token_id`` & friends) to None on ``PretrainedConfig`` - they only
    exist now when the checkpoint's config.json actually sets them. qwen-tts
    reads e.g. ``config.pad_token_id`` unconditionally (written against the
    transformers 4.x behavior), which made model construction die with
    ``AttributeError: 'Qwen3TTSTalkerConfig' object has no attribute
    'pad_token_id'``. Give every PretrainedConfig subclass the package
    defines the old None defaults at CLASS level: instances that DO carry a
    value from config.json still win (instance dict beats class attr), and
    nothing outside qwen_tts is touched.
    """
    from transformers import PretrainedConfig

    legacy = ("bos_token_id", "eos_token_id", "pad_token_id",
              "decoder_start_token_id", "forced_bos_token_id",
              "forced_eos_token_id")
    patched = 0
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("qwen_tts"):
            continue
        for cls in list(vars(module).values()):
            if (isinstance(cls, type) and issubclass(cls, PretrainedConfig)
                    and cls.__module__.startswith("qwen_tts")):
                for key in legacy:
                    if not hasattr(cls, key):
                        setattr(cls, key, None)
                        patched += 1
    return patched


try:
    from qwen_tts import Qwen3TTSModel
    _patched = _patch_legacy_config_defaults()
    if _patched:
        logging.getLogger(__name__).info(
            "Qwen3-TTS compat: restored %d legacy token-id config default(s) "
            "that transformers 5.x no longer sets", _patched)
except Exception:  # optional dependency - must never take the pack down
    Qwen3TTSModel = None
    logging.getLogger(__name__).error(
        "Qwen3-TTS nodes disabled: importing the 'qwen-tts' package failed "
        "(it is either not installed, or version-incompatible with the "
        "installed transformers). Every other node in this pack still "
        "works. See the traceback below.", exc_info=True)

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
    SEARCH_ALIASES = ['load model', 'tts model', 'model loader', 'text to speech loader']
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
                "The 'qwen-tts' package is required for QwenTTSModelsLoader but "
                "could not be imported (not installed, or version-incompatible "
                "with the installed transformers - the reason is in the ComfyUI "
                "startup log). Try: pip install qwen-tts")

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
        # "name" carries the actually-loaded folder so the generate nodes can
        # family-check (CustomVoice/VoiceDesign/Base) from it; local_model_path
        # loads carry their own folder name instead of the repo_id widget.
        bundle = {"model": model, "key": key, "name": model_path}
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
    SEARCH_ALIASES = ['generate audio', 'text to speech', 'tts', 'voice generation', 'speak text']
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


def _model_family(qwen_tts_model):
    """Family token in the loaded model's name (CustomVoice/VoiceDesign/Base).

    None for names carrying no known family token - a local_model_path or
    fine-tuned checkpoint we cannot classify by name.
    """
    name = (qwen_tts_model.get("name") or "").lower()
    for family in ("customvoice", "voicedesign", "base"):
        if family in name:
            return family
    return None


def _require_family(qwen_tts_model, wanted, action):
    """The upstream package deliberately binds each generation mode to one
    checkpoint family (clone -> Base, design -> VoiceDesign, built-in
    speakers -> CustomVoice). Enforce the same pairing up front with a clear
    error instead of failing somewhere inside the model. Unrecognizable names
    (local_model_path/fine-tunes) pass through unchecked.
    """
    family = _model_family(qwen_tts_model)
    if family is not None and family != wanted:
        raise ValueError(
            f"{action} needs a *-{wanted.capitalize()} checkpoint, but the "
            f"models loader has {qwen_tts_model.get('name')!r} (family "
            f"{family!r}). Point Qwen3-TTS Models Loader at a *-"
            f"{wanted.capitalize()} repo.")


def _ref_audio_to_tuple(audio):
    """ComfyUI AUDIO dict -> the (1-D float32 numpy, sample_rate) tuple the
    qwen-tts package's reference-audio argument expects: mono downmix, float
    1-D, padded to at least 1024 samples (upstream's own minimum).
    """
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError(
            "ref_audio must be a ComfyUI AUDIO connection "
            "({'waveform': [B,C,T] tensor, 'sample_rate': int}).")
    wave = audio["waveform"]
    sr = int(audio.get("sample_rate") or 0)
    if sr <= 0:
        raise ValueError("ref_audio carries no sample_rate.")
    if isinstance(wave, torch.Tensor):
        if wave.dim() > 1:  # [B, C, T] -> mono
            wave = wave.squeeze()
            if wave.dim() > 1:
                wave = wave.mean(dim=0)
        wave = wave.detach().cpu().numpy()
    import numpy as np
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        raise ValueError("ref_audio waveform is empty.")
    if wave.size < 1024:
        wave = np.concatenate([wave, np.zeros(1024 - wave.size, np.float32)])
    return wave, sr


def _wavs_to_audio(wavs, sample_rate):
    """Package output (list of 1-D numpy wavs, sr) -> ComfyUI AUDIO dict."""
    if not wavs:
        raise RuntimeError("Qwen3-TTS generated no audio.")
    waveform = torch.from_numpy(wavs[0]).float()[None, None, :]
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


class QwenTTSVoiceCloneGenerate:
    """Clone a reference voice (AUDIO or a cached prompt) onto target text."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Voice Clone ⚡"
    SEARCH_ALIASES = ['voice clone', 'clone voice', 'reference audio', 'tts', 'speech']
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    DESCRIPTION = ("Clone a voice from reference audio (or a reusable "
                   "voice-clone prompt) and speak the target text. Needs a "
                   "*-Base checkpoint in the models loader.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_tts_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello world"}),
                "language": (CUSTOM_VOICE_LANGUAGES, {"default": "Auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio to clone "
                    "from. Ignored when voice_clone_prompt is connected."}),
                "ref_text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Transcript of the reference audio (optional, "
                               "improves clone quality)."}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT", {"tooltip":
                    "Reusable prompt from Qwen3-TTS Voice Clone Prompt (or "
                    "Qwen3-TTS Load Speaker). Takes precedence over ref_audio."}),
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip":
                    "Only use the speaker embedding (ref_text not needed)."}),
                "instruct": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional style/emotion instruction."}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    def generate(self, qwen_tts_model, text, language, seed, max_new_tokens,
                 top_p, top_k, temperature, repetition_penalty,
                 ref_audio=None, ref_text="", voice_clone_prompt=None,
                 x_vector_only=False, instruct="", unload_model_after_generate=False):
        _require_family(qwen_tts_model, "base", "Voice cloning")
        model = qwen_tts_model["model"]

        prompt_param = None
        ref_tuple = None
        if voice_clone_prompt is not None:
            prompt_param = voice_clone_prompt
        elif ref_audio is not None:
            ref_tuple = _ref_audio_to_tuple(ref_audio)
        else:
            raise ValueError(
                "Connect either ref_audio or voice_clone_prompt - voice "
                "cloning needs a voice to clone from.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=None if language == "Auto" else language,
            ref_audio=ref_tuple,
            ref_text=ref_text.strip() or None,
            voice_clone_prompt=prompt_param,
            x_vector_only_mode=x_vector_only,
            instruct=instruct.strip() or None,
            max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k, temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        audio = _wavs_to_audio(wavs, sample_rate)
        logger.info("Qwen3-TTS: voice-clone generated %s @ %d Hz",
                    tuple(audio["waveform"].shape), sample_rate)

        if unload_model_after_generate:
            _MODEL_CACHE.pop(qwen_tts_model["key"], None)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (audio,)


class QwenTTSVoiceDesignGenerate:
    """Describe a voice in text (instruct) and speak the target text with it."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Voice Design ⚡"
    SEARCH_ALIASES = ['voice design', 'describe voice', 'tts', 'speech', 'generate audio']
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    DESCRIPTION = ("Design a voice from a text description (instruct, e.g. "
                   "'a deep male voice with a calm tone') and speak the text "
                   "with it. Needs a *-VoiceDesign checkpoint.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_tts_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello world"}),
                "instruct": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Voice description, e.g. 'a warm female voice "
                               "with an energetic tone'. Required for voice "
                               "design."}),
                "language": (CUSTOM_VOICE_LANGUAGES, {"default": "Auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 32768}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 500}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 5.0, "step": 0.01}),
            },
            "optional": {
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    def generate(self, qwen_tts_model, text, instruct, language, seed,
                 max_new_tokens, top_p, top_k, temperature, repetition_penalty,
                 unload_model_after_generate=False):
        _require_family(qwen_tts_model, "voicedesign", "Voice designing")
        model = qwen_tts_model["model"]

        if not text.strip() or not instruct.strip():
            raise ValueError("Voice design needs both text and instruct.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language=None if language == "Auto" else language,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k, temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        audio = _wavs_to_audio(wavs, sample_rate)
        logger.info("Qwen3-TTS: voice-design generated %s @ %d Hz",
                    tuple(audio["waveform"].shape), sample_rate)

        if unload_model_after_generate:
            _MODEL_CACHE.pop(qwen_tts_model["key"], None)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (audio,)


class QwenTTSVoiceClonePrompt:
    """Extract reusable voice features from reference audio."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Voice Clone Prompt ⚡"
    SEARCH_ALIASES = ['voice prompt', 'speaker embedding', 'clone prompt', 'tts']
    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    DESCRIPTION = ("Extract voice features from reference audio once and "
                   "reuse them in Voice Clone / Role Bank / Save Voice. Needs "
                   "a *-Base checkpoint (same family Voice Clone uses).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_tts_model": ("QWEN_TTS_MODEL",),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Transcript of the reference audio (optional, "
                               "improves quality)."}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip":
                    "Only extract the speaker embedding (ref_text unused)."}),
            },
        }

    def create_prompt(self, qwen_tts_model, ref_audio, ref_text,
                      x_vector_only=False):
        _require_family(qwen_tts_model, "base", "Voice-clone prompt extraction")
        model = qwen_tts_model["model"]
        ref_tuple = _ref_audio_to_tuple(ref_audio)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_tuple,
            ref_text=ref_text.strip() or None,
            x_vector_only_mode=x_vector_only,
        )
        logger.info("Qwen3-TTS: extracted voice-clone prompt (%s)",
                    type(prompt_items).__name__)
        return (prompt_items,)


class QwenTTSRoleBank:
    """Collect up to 8 named voice prompts for multi-role workflows."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Role Bank ⚡"
    SEARCH_ALIASES = ['role bank', 'multi character', 'dialogue', 'voices', 'tts']
    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create_bank"
    DESCRIPTION = ("Collect up to 8 named voice-clone prompts into one bank "
                   "for multi-role dialogue workflows. Roles with no prompt "
                   "connected are skipped.")

    @classmethod
    def INPUT_TYPES(s):
        inputs = {"required": {}, "optional": {}}
        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            inputs["optional"][f"prompt_{i}"] = ("VOICE_CLONE_PROMPT",)
        return inputs

    def create_bank(self, **kwargs):
        bank = {}
        for i in range(1, 9):
            name = (kwargs.get(f"role_name_{i}") or "").strip()
            prompt = kwargs.get(f"prompt_{i}")
            if name and prompt is not None:
                bank[name] = prompt
        return (bank,)


def _voices_dir():
    path = os.path.join(QWEN3_TTS_MODELS_DIR, "voices")
    os.makedirs(path, exist_ok=True)
    return path


def _write_wav_preview(path, audio):
    """Best-effort .wav preview write - never fails the save. Prefers
    torchaudio/soundfile when installed, falls back to a stdlib `wave`
    16-bit PCM write so the node works on minimal installs too."""
    try:
        import numpy as np
        import wave as wavmod
        wave2d = audio["waveform"].detach().cpu()
        sr = int(audio["sample_rate"])
        if wave2d.dim() == 3:  # [B, C, T] -> [C, T]
            wave2d = wave2d[0]
        if wave2d.dim() == 1:
            wave2d = wave2d[None]
        try:
            import torchaudio
            torchaudio.save(path, wave2d.T.contiguous(), sr)
        except Exception:  # no audio backend installed -> next fallback
            try:
                import soundfile as sf
                sf.write(path, wave2d.T.numpy(), sr)
            except Exception:  # not installed either -> stdlib 16-bit PCM
                data = np.clip(wave2d.numpy(), -1.0, 1.0)
                pcm = (data * 32767.0).astype("<i2")
                with wavmod.open(path, "wb") as w:
                    w.setnchannels(pcm.shape[0])
                    w.setsampwidth(2)
                    w.setframerate(sr)
                    w.writeframes(pcm.T.tobytes())
    except Exception as e:
        logger.warning("Qwen3-TTS: could not write voice preview wav: %s", e)


def _load_audio_file(path):
    """Read a saved voice file into ([C, T] float tensor, sr). torchaudio /
    soundfile cover every format; the stdlib fallback covers the .wav
    previews this pack writes itself on backend-less installs."""
    try:
        import torchaudio
        wave2d, sr = torchaudio.load(path)
        return wave2d.float(), int(sr)
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        return torch.from_numpy(data.T).float(), int(sr)
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        import numpy as np
        import wave as wavmod
        with wavmod.open(path, "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())
        pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, ch)
        return torch.from_numpy(pcm.T.astype(np.float32) / 32768.0), int(sr)
    raise RuntimeError(
        f"Reading {os.path.basename(path)} needs torchaudio or soundfile "
        f"installed (only .wav previews are readable without them).")


class QwenTTSSaveVoice:
    """Persist a voice-clone prompt (+ optional preview audio) to disk."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Save Voice ⚡"
    SEARCH_ALIASES = ['save voice', 'persist voice', 'voice bank', 'tts']
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    DESCRIPTION = ("Save a voice-clone prompt under models/Qwen3-TTS/voices/ "
                   "as <name>.qvp (+ .json metadata and an optional .wav "
                   "preview), for reuse with Qwen3-TTS Load Speaker.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
                "filename": ("STRING", {"default": "my_voice"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Reference audio to save as a "
                    ".wav preview alongside the prompt."}),
                "ref_text": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Transcript stored in the .json metadata and "
                               "restored by Load Speaker."}),
            },
        }

    def save(self, voice_clone_prompt, filename, audio=None, ref_text=""):
        import json
        stem = os.path.splitext(os.path.basename(filename.strip()))[0]
        if not stem:
            raise ValueError("filename must not be empty.")
        out_dir = _voices_dir()

        qvp_path = os.path.join(out_dir, stem + ".qvp")
        torch.save(voice_clone_prompt, qvp_path)

        with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"ref_text": ref_text.strip(), "source": "QwenTTSSaveVoice"},
                      f, ensure_ascii=False, indent=2)

        if audio is not None:
            _write_wav_preview(os.path.join(out_dir, stem + ".wav"), audio)

        logger.info("Qwen3-TTS: saved voice %r to %s", stem, out_dir)
        return {}


class QwenTTSLoadSpeaker:
    """Load a saved voice (.wav preview + cached .qvp prompt) back."""

    CATEGORY = QWEN_TTS_CATEGORY
    TITLE = "Qwen3-TTS Load Speaker ⚡"
    SEARCH_ALIASES = ['load voice', 'load speaker', 'voice bank', 'tts']
    RETURN_TYPES = ("VOICE_CLONE_PROMPT", "AUDIO", "STRING")
    RETURN_NAMES = ("voice_clone_prompt", "audio", "ref_text")
    FUNCTION = "load_speaker"
    DESCRIPTION = ("Load a saved voice from models/Qwen3-TTS/voices/. "
                   "Fast-loads the cached .qvp prompt and .json transcript "
                   "when they exist; otherwise wire audio -> Voice Clone's "
                   "ref_audio (the prompt output is then None).")

    @classmethod
    def INPUT_TYPES(s):
        files = sorted(f for f in os.listdir(_voices_dir())
                       if f.lower().endswith((".wav", ".mp3", ".flac")))
        return {"required": {"filename": (files or ["<no saved voices>"],)}}

    def load_speaker(self, filename):
        import json
        if filename.startswith("<"):
            raise ValueError("No saved voices found under models/Qwen3-TTS/voices/.")

        voices = _voices_dir()
        stem = os.path.splitext(filename)[0]
        wav_path = os.path.join(voices, filename)
        if not os.path.isfile(wav_path):
            raise ValueError(f"Saved voice {filename!r} not found.")

        wave2d, sr = _load_audio_file(wav_path)
        audio = {"waveform": wave2d[None], "sample_rate": sr}

        ref_text = ""
        json_path = os.path.join(voices, stem + ".json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    ref_text = (json.load(f) or {}).get("ref_text", "")
            except Exception as e:
                logger.warning("Qwen3-TTS: could not read %s.json: %s", stem, e)

        prompt_items = None
        qvp_path = os.path.join(voices, stem + ".qvp")
        if os.path.isfile(qvp_path):
            try:
                # weights_only=False is required: the .qvp holds
                # qwen_tts.VoiceClonePromptItem objects, not plain tensors.
                # Only ever load .qvp files you saved yourself.
                prompt_items = torch.load(qvp_path, map_location="cpu",
                                          weights_only=False)
            except Exception as e:
                logger.warning("Qwen3-TTS: could not fast-load %s.qvp (%s) - "
                               "the prompt output stays None", stem, e)

        logger.info("Qwen3-TTS: loaded saved voice %r (prompt=%s)",
                    filename, "yes" if prompt_items is not None else "no")
        return (prompt_items, audio, ref_text)


NODE_CLASS_MAPPINGS = {
    "QwenTTSModelsLoader": QwenTTSModelsLoader,
    "QwenTTSCustomVoiceGenerate": QwenTTSCustomVoiceGenerate,
    "QwenTTSVoiceCloneGenerate": QwenTTSVoiceCloneGenerate,
    "QwenTTSVoiceDesignGenerate": QwenTTSVoiceDesignGenerate,
    "QwenTTSVoiceClonePrompt": QwenTTSVoiceClonePrompt,
    "QwenTTSRoleBank": QwenTTSRoleBank,
    "QwenTTSSaveVoice": QwenTTSSaveVoice,
    "QwenTTSLoadSpeaker": QwenTTSLoadSpeaker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelsLoader": QwenTTSModelsLoader.TITLE,
    "QwenTTSCustomVoiceGenerate": QwenTTSCustomVoiceGenerate.TITLE,
    "QwenTTSVoiceCloneGenerate": QwenTTSVoiceCloneGenerate.TITLE,
    "QwenTTSVoiceDesignGenerate": QwenTTSVoiceDesignGenerate.TITLE,
    "QwenTTSVoiceClonePrompt": QwenTTSVoiceClonePrompt.TITLE,
    "QwenTTSRoleBank": QwenTTSRoleBank.TITLE,
    "QwenTTSSaveVoice": QwenTTSSaveVoice.TITLE,
    "QwenTTSLoadSpeaker": QwenTTSLoadSpeaker.TITLE,
}
