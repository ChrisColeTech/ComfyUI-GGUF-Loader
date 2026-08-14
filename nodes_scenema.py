# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Scenema Audio nodes — expressive TTS on the LTX 2.3 audio diffusion transformer.

Ported from ScenemaAI/ComfyUI-ScenemaAudio (MIT). Instead of vendoring that
pack's ltx_core/ltx_pipelines model code and auto-downloading weights from
HuggingFace, these nodes load the same components the sidecar uses —
the Scenema safetensors (INT8 or bf16), the Gemma-3 12B text encoder
(safetensors **or** GGUF), the pipeline checkpoint carrying the text
projection + embeddings connectors + audio VAE + vocoder, and the standalone
VAE encoder — through ComfyUI's native LTX-AV machinery and this pack's
GGUF loaders:

  * the audio DiT is a comfy `LTXAVModel` minus its (never-run) video paths:
    the checkpoint keys map 1:1 onto comfy's, the audio-only forward the
    original nodes monkey-patched in is reproduced with comfy's own
    `transformer_options` gates (`run_vx=False`, cross-modal attention off);
  * the text encoder is ComfyUI's `LTXAVTEModel` (Gemma-3 12B + the pipeline
    checkpoint's dual `text_embedding_projection` and embeddings connectors),
    so a Q4_K_M GGUF of Gemma stays quantized through GGMLOps exactly like
    any other GGUF text encoder in this pack;
  * the audio VAE is ComfyUI's `AudioVAE` (encoder + decoder + BigVGAN
    vocoder with the 16k→48k bandwidth extension), built from the pipeline
    checkpoint's config metadata;
  * sampling uses the distilled 8-step sigma schedule with cfg=1 (the
    original's SimpleDenoiser) through comfy's regular sampler stack.

Dropped from the original pack because they carry heavy non-comfy deps:
Whisper word-match validation (faster-whisper), SeedVC polish, and the
MelBandRoFormer SFX strip. Everything else — the XML prompt compiler,
Kokoro-free chunk planning, per-chunk trim/normalize/concat, A2V reference
chaining, and the 12 production presets — is ported.
"""

import inspect
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.ops
import comfy.sample
import comfy.sd
import comfy.utils
import folder_paths

from .loader import gguf_sd_loader, gguf_clip_loader
from .ops import GGMLOps

logger = logging.getLogger(__name__)

# ── Constants (ported from ComfyUI-ScenemaAudio / ltx_pipelines) ──────────

FPS = 24
MAX_REF_SECONDS = 20.0        # hard cap on voice-clone reference length
REF_TAIL_SECONDS = 3.0        # A2V chaining: tail of chunk N conditions N+1
MAX_CHUNK_DURATION_S = 15.0   # model repeats itself beyond ~15s
LTX_MULTIPLIER = 1.5          # LTX speaks slower than the estimator below
FALLBACK_WORDS_PER_SEC = 2.2
ACTION_DURATION_S = 1.5

# Distilled 8-step schedule (ltx_pipelines.utils.constants.DISTILLED_SIGMAS)
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

SCENE_SENTINEL = "Choose a scene..."
SCENE_PRESETS = [
    SCENE_SENTINEL,
    "Absolute silence",
    "Quiet indoor room",
    "Reverberant hall",
    "Broadcast studio",
    "Outdoor, open air",
    "Café or restaurant",
    "Windy outdoors",
    "Rainy outdoors",
]
CLEAN_SPEECH_SCENES = {"Absolute silence", "Broadcast studio"}

LANGUAGE_OPTIONS = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese",
    "Japanese", "Korean", "Chinese", "Hindi", "Arabic", "Swahili",
]
LANGUAGE_CODES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
    "Chinese": "zh", "Hindi": "hi", "Arabic": "ar", "Swahili": "sw",
}

# ── Prompt compilation (ported from ScenemaAudio audio_core/compiler.py) ──

DEFAULT_SCENE = "a person speaking to camera"
SHOT_PREFIXES = {"closeup": "Close-up in", "wide": "Wide shot of", "scene": ""}


@dataclass
class CompiledPrompt:
    prompt: str
    speech_text: str
    voice: str
    scene: str | None
    language: str
    gender: str
    shot: str


def _ensure_trailing_punctuation(text):
    if text and text[-1] not in ".!?\"'":
        return text + "."
    return text


def _extract_blocks(root):
    """Walk <speak> children in document order -> text/action/sound blocks."""
    blocks = []
    if root.text and root.text.strip():
        blocks.append(("text", root.text.strip()))
    for child in root:
        if child.tag in ("action", "sound") and child.text and child.text.strip():
            blocks.append((child.tag, child.text.strip()))
        if child.tail and child.tail.strip():
            blocks.append(("text", child.tail.strip()))
    return blocks


def _compile_blocks(blocks, voice, scene, gender="male", shot="closeup"):
    parts = []
    is_scene_mode = shot in ("scene", "wide")
    pronoun = "She" if gender == "female" else "He"

    scene_text = scene if scene else DEFAULT_SCENE
    prefix = SHOT_PREFIXES.get(shot, SHOT_PREFIXES["closeup"])
    parts.append(f"{prefix} {scene_text}." if prefix else f"{scene_text}.")

    first_speech = True
    for kind, text in blocks:
        if kind == "sound":
            parts.append(_ensure_trailing_punctuation(text))
        elif kind == "action":
            parts.append(text + ":" if is_scene_mode else _ensure_trailing_punctuation(text))
        else:
            clean = _ensure_trailing_punctuation(text)
            if (is_scene_mode and first_speech
                    and not any(k == "action" for k, _ in blocks)):
                parts.append(f'{pronoun} speaks: "{clean}"')
            else:
                parts.append(f'"{clean}"')
            first_speech = False

    parts.append(_ensure_trailing_punctuation(voice))
    if is_scene_mode and scene:
        parts.append(_ensure_trailing_punctuation(scene))
    return " ".join(parts)


def compile_prompt(xml_string):
    root = ET.fromstring(xml_string)
    voice = root.get("voice", "").strip()
    scene = root.get("scene")
    scene = scene.strip() if scene else scene
    language = root.get("language", "en").strip()
    gender = root.get("gender", "male").strip()
    shot = root.get("shot", "closeup").strip()

    blocks = _extract_blocks(root)
    prompt = _compile_blocks(blocks, voice, scene, gender, shot)
    speech_text = " ".join(t for k, t in blocks if k == "text")
    return CompiledPrompt(prompt, speech_text, voice, scene, language, gender, shot)


def compile_chunk_prompt(speech_text, voice, scene=None, actions_before=None,
                         gender="male", shot="closeup"):
    blocks = []
    for a in (actions_before or []):
        blocks.append(("action", a))
    blocks.append(("text", speech_text))
    return _compile_blocks(blocks, voice, scene, gender, shot)


def extract_sentence_actions(xml_string):
    """Map sentence index -> action blocks that precede it (for chunking)."""
    blocks = _extract_blocks(ET.fromstring(xml_string))
    sentence_actions = {}
    pending = []
    sentence_idx = 0
    for kind, text in blocks:
        if kind == "action":
            pending.append(text)
        elif kind == "text":
            sentences = _split_into_sentences(text)
            if pending and sentences:
                sentence_actions[sentence_idx] = pending.copy()
                pending.clear()
            sentence_idx += len(sentences)
    return sentence_actions


def _xml_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _speech_body_parts(speech_text):
    """Split `[bracketed cues]` into interleaved text / <action> fragments."""
    parts = re.split(r"\[([^\]]+)\]", speech_text)
    for i, part in enumerate(parts):
        if i % 2 == 0:
            text = part.strip()
            if text:
                yield f"  {_xml_escape(text)}"
        else:
            cue = part.strip()
            if cue:
                yield f"  <action>{_xml_escape(cue)}</action>"


def build_speak_xml(voice_description, gender, speech_text, scene,
                    custom_scene="", action_tags="", language="English"):
    """Construct the <speak> XML from form fields (ported from generate.py)."""
    scene_text = custom_scene.strip() if custom_scene and custom_scene.strip() else scene
    lang_code = LANGUAGE_CODES.get(language, "en")
    shot = "closeup" if scene_text in CLEAN_SPEECH_SCENES else "wide"

    voice_attr = _xml_escape(voice_description).replace('"', "&quot;")
    attrs = f'voice="{voice_attr}" gender="{gender}"'
    if scene_text:
        attrs += f' scene="{_xml_escape(scene_text)}"'
    if lang_code != "en":
        attrs += f' language="{lang_code}"'
    if shot != "closeup":
        attrs += f' shot="{shot}"'

    body_parts = []
    if action_tags and action_tags.strip():
        for line in action_tags.strip().split("\n"):
            line = line.strip()
            if line:
                body_parts.append(f"  <action>{_xml_escape(line)}</action>")
    body_parts.extend(_speech_body_parts(speech_text))
    return f"<speak {attrs}>\n" + "\n".join(body_parts) + "\n</speak>"


# ── Chunk planning (ported from audio_core/chunker.py, word-count path) ───

@dataclass
class ChunkSpec:
    compiled_prompt: str
    duration_s: float
    seed: int
    expected_text: str
    language: str = "en"


def _split_into_sentences(text):
    sentences, current = [], ""
    for char in text:
        current += char
        if char in ".!?":
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _estimate_duration(text):
    """Word-count duration estimate (the chunker's estimator without Kokoro)."""
    words = len(text.split())
    return words / FALLBACK_WORDS_PER_SEC + 0.5


def split_text_by_duration(text, multiplier=LTX_MULTIPLIER,
                           max_duration=MAX_CHUNK_DURATION_S):
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    expanded = []
    for sent in sentences:
        dur = _estimate_duration(sent)
        if dur * multiplier > max_duration and "," in sent:
            clauses = [c.strip() for c in sent.split(",") if c.strip()]
            sub_texts, sub_dur = [], 0.0
            for clause in clauses:
                cdur = _estimate_duration(clause)
                if sub_texts and (sub_dur + cdur) * multiplier > max_duration:
                    expanded.append(", ".join(sub_texts))
                    sub_texts, sub_dur = [], 0.0
                sub_texts.append(clause)
                sub_dur += cdur
            if sub_texts:
                expanded.append(", ".join(sub_texts))
        else:
            expanded.append(sent)

    chunks, current_texts, current_dur = [], [], 0.0
    for sent in expanded:
        dur = _estimate_duration(sent)
        if current_texts and (current_dur + dur) * multiplier > max_duration:
            chunks.append((" ".join(current_texts),
                           min(current_dur * multiplier, max_duration)))
            current_texts, current_dur = [], 0.0
        current_texts.append(sent)
        current_dur += dur
    if current_texts:
        chunks.append((" ".join(current_texts),
                       min(current_dur * multiplier, max_duration)))
    return chunks


def plan_chunks(xml_string, compiled, base_seed=0, pace=LTX_MULTIPLIER):
    """Plan generation chunks from a compiled prompt (no Kokoro dependency)."""
    total_dur = _estimate_duration(compiled.speech_text) * pace
    if total_dur <= MAX_CHUNK_DURATION_S:
        return [ChunkSpec(compiled.prompt, min(total_dur, MAX_CHUNK_DURATION_S),
                          base_seed, compiled.speech_text, compiled.language)]

    sentence_action_map = extract_sentence_actions(xml_string)
    text_chunks = split_text_by_duration(compiled.speech_text, multiplier=pace)

    specs, global_sentence_idx = [], 0
    for chunk_text, chunk_dur in text_chunks:
        actions_before = sentence_action_map.get(global_sentence_idx)
        chunk_prompt = compile_chunk_prompt(
            speech_text=chunk_text, voice=compiled.voice, scene=compiled.scene,
            actions_before=actions_before, gender=compiled.gender, shot=compiled.shot,
        )
        specs.append(ChunkSpec(chunk_prompt, chunk_dur, base_seed,
                               chunk_text, compiled.language))
        global_sentence_idx += len(_split_into_sentences(chunk_text))

    logger.info("Scenema: planned %d chunk(s), %.1fs total estimated",
                len(specs), sum(s.duration_s for s in specs))
    return specs


# ── Audio post-processing (ported from audio_core/audio_utils.py) ─────────

def trim_silence(audio_np, sr, max_silence=0.5, threshold_db=-40):
    threshold = 10 ** (threshold_db / 20.0)
    max_silent = int(max_silence * sr)
    window = int(0.02 * sr)
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    if len(mono) < window:
        return audio_np
    energy = np.array([np.abs(mono[i:i + window]).max()
                       for i in range(0, len(mono) - window, window)])
    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        return audio_np
    first = max(0, voiced[0] * window - max_silent)
    last = min(len(audio_np), (voiced[-1] + 1) * window + max_silent)
    return audio_np[first:last]


def normalize_volume(audio_np, sr, target_lufs=-23.0):
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    rms = np.sqrt(np.mean(mono ** 2))
    if rms < 1e-8:
        return audio_np
    current_lufs = 20 * math.log10(rms) - 0.691
    gain = 10 ** ((target_lufs - current_lufs) / 20.0)
    gain = max(0.1, min(gain, 10.0))
    result = audio_np * gain
    peak = np.abs(result).max()
    if peak > 0.99:
        result = result * (0.99 / peak)
    return result


def shorten_long_silence(audio_np, sr, max_duration=1.0, target_duration=0.3,
                         threshold_db=-35):
    threshold = 10 ** (threshold_db / 20.0)
    window = int(0.02 * sr)
    max_samples = int(max_duration * sr)
    target_samples = int(target_duration * sr)
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    if len(mono) < window:
        return audio_np
    energy = np.array([np.abs(mono[i:i + window]).max()
                       for i in range(0, len(mono) - window, window)])
    is_silent = energy < threshold

    regions, in_silence, start = [], False, 0
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start, in_silence = i * window, True
        elif not silent and in_silence:
            end = i * window
            if end - start > max_samples:
                regions.append((start, end))
            in_silence = False
    if in_silence and len(mono) - start > max_samples:
        regions.append((start, len(mono)))
    if not regions:
        return audio_np

    parts, prev_end = [], 0
    for s_start, s_end in regions:
        parts.append(audio_np[prev_end:s_start])
        parts.append(audio_np[s_start:s_start + target_samples])
        prev_end = s_end
    parts.append(audio_np[prev_end:])
    return np.concatenate(parts, axis=0)


# ── Presets (ported from nodes/presets.py, scenema.ai demos) ──────────────

CUSTOM = "Custom"

PRESETS = {
    "Old Male Storyteller (fireside)": {
        "voice_description": "Male, mid 60s. Deep baritone with gravel. Slight Southern American inflection. Worn but warm. The voice of someone who has seen too much and chosen kindness anyway. Nostalgic, firelight cadence.",
        "gender": "male",
        "scene": "Quiet indoor room",
        "custom_scene": "Fireside, night, crickets in the distance",
        "action_tags": "He settles into his chair and stares at the fire",
        "speech_text": "There was a summer, back when the river still ran clear, when my father took me out past the property line and pointed at the stars. He said, boy, every one of those is a story somebody forgot to write down. [He smiles to himself] I have been writing them down ever since.",
    },
    "Young Woman (breathless discovery)": {
        "voice_description": "Female, early 20s. Bright soprano. Slightly breathy. American West Coast. The kind of voice that smiles while speaking. Breathless awe, tumbling over words.",
        "gender": "female",
        "scene": "Outdoor, open air",
        "custom_scene": "An open field, something glowing in front of her",
        "action_tags": "She freezes, eyes wide",
        "speech_text": "Oh my god. Oh my god, it is real. I thought they were lying, I thought it was just some internet thing but it is actually here and it is glowing and I do not know what to do with my hands right now.",
    },
    "Terrified Whisper": {
        "voice_description": "Male, mid 30s. Whisper. Terrified. Shaking. A man hiding, trying not to be found. Every word is a risk. Breath catching between words.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He presses against the wall, barely breathing",
        "speech_text": "Listen to me. Do not turn around. The man in the grey coat has been following us since the bridge. I need you to walk to the cafe on the corner, order something, and leave through the back. I will find you. Do you understand? Nod if you understand.",
    },
    "Irish Woman, Dry Wit": {
        "voice_description": "Woman, mid 40s. Strong Irish accent, Dublin. Dry, sardonic, cutting. Bone-dry wit. She sounds like she has seen it all and finds most of it beneath her.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She speaks flatly, unimpressed",
        "speech_text": "Apparently the committee has decided that what this building really needs is another meeting room. Because the problem with this organization was never the decisions. It was that we did not have enough places to avoid making them.",
    },
    "Rage to Vulnerability": {
        "voice_description": "A man on the edge. Explosive rage building with every sentence. Gravelly, intimidating. Italian-American inflection. Controlled fury that could snap at any moment. The kind of anger that comes from deep disrespect.",
        "gender": "male",
        "scene": "Quiet indoor room",
        "custom_scene": "A dimly lit office, late at night",
        "action_tags": "He stands up slowly, voice dangerously low",
        "speech_text": "You come into my house, you eat my food, and then you got the nerve to tell me how to run my business. You know what your problem is? You got no respect. None. Zero. [Voice rising, finger pointing] I built this thing from nothing, nothing, while you were sitting on your ass doing God knows what. So don't come in here with that attitude. You understand me?",
    },
    "Eulogy (grief, extremely slow)": {
        "voice_description": "Woman, mid 60s. Deep. Extremely slow. Heavy with grief. Each word lands like a stone dropped into still water. Long pauses between phrases. Barely above a whisper.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She speaks so slowly that each phrase feels like its own sentence. Heavy pauses. The weight of loss in every breath.",
        "speech_text": "Even in our sleep. Pain which cannot forget. Falls drop by drop upon the heart. Until in our own despair. Against our will. Comes wisdom. Through the awful grace of God.",
    },
    "Terror (sobbing, hyperventilating)": {
        "voice_description": "Woman, late 20s. Voice shaking violently. Hyperventilating. Sobbing. Choking on tears. Words barely coming out between gasps for air. Throat tight with panic. Speaking through crying.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "She gasps for air between sobs, voice breaking on every word, barely able to speak through the tears",
        "speech_text": "Please. Please help me. I can hear them downstairs. They broke the window. My baby is with me. Please send help. Please hurry. Please.",
    },
    "Villain (laughing menace)": {
        "voice_description": "Male. Deep, resonant, theatrical voice dripping with contempt and dark amusement. Dramatic pauses. Shifting between sinister whispers and booming declarations.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He laughs, quiet at first, then louder, then speaks with cold precision",
        "speech_text": "Heheheh. Hahahaha! Oh I have waited so long for this. They told me you were clever. They said be careful. And here you are, on your knees, with nothing left. Tell me. Was it worth it? All that running?",
    },
    "Rain and Thunder (SFX)": {
        "voice_description": "Male, mid 40s. Baritone. Weathered. Urgent, projecting over wind and rain.",
        "gender": "male",
        "scene": "Rainy outdoors",
        "custom_scene": "Open dock in a thunderstorm, heavy rain, waves crashing against the pier",
        "action_tags": "Heavy rain and wind howling. He cups his hands and shouts over the wind and rain",
        "speech_text": "Get the lines! Get the lines now! She is pulling loose! If we lose this boat we lose everything! [Thunder cracks overhead] [He screams louder] Move! I said move!",
    },
    "Italian Cooking Show (SFX)": {
        "voice_description": "Female, mid 30s. Warm, enthusiastic. Italian accent. A home cook who treats every meal like a celebration.",
        "gender": "female",
        "scene": "Café or restaurant",
        "custom_scene": "Busy home kitchen, oil sizzling in a hot pan, pots bubbling on the stove",
        "action_tags": "Oil sizzling loudly in a hot pan, a pot bubbling on the stove. She talks over the sizzling, gesturing with a wooden spoon, energetic and happy",
        "speech_text": "Okay now this is the important part. You wait until the oil is really hot, you see the smoke? That is when you drop the garlic in. [Garlic hits the hot oil with a loud sizzle and crackle] [She stirs quickly, laughing] Beautiful! You smell that? Now we add the tomatoes and let it all come together.",
    },
    "Kid Explaining Dinosaurs": {
        "voice_description": "Boy, 8 years old. Small clear voice. Speaking carefully like he is the authority on this subject. A child explaining something important to someone younger.",
        "gender": "male",
        "scene": "Absolute silence",
        "custom_scene": "",
        "action_tags": "He speaks seriously, like a tiny professor",
        "speech_text": "Okay so dinosaurs. They were really really big, like bigger than this whole house. And they lived a million billion years ago. And you know what happened? A giant rock came from space and hit the earth and then it got really cold and they all had to go away. But birds are actually dinosaurs. So technically we have dinosaurs right now.",
    },
    "British Woman, East London Rage": {
        "voice_description": "Shrill angry British female voice, East London accent. Screaming and furious.",
        "gender": "female",
        "scene": "Absolute silence",
        "custom_scene": "A messy flat, pointing at the camera",
        "action_tags": "She points at the camera, face twisted with rage",
        "speech_text": "Are you having a bloody laugh? You absolute muppet! I told you THREE times to sort the bins out and what do I come home to? This! This absolute disaster! I swear to God if you don't get your shit together by tomorrow I am DONE. Finished! Pack your bags and piss off back to your mum's! I am NOT joking!",
    },
}

PRESET_NAMES = [CUSTOM] + list(PRESETS.keys())


# ── Checkpoint loading helpers ────────────────────────────────────────────

def _convert_int8_keys(sd):
    """Expose Scenema INT8 tensors through comfy's native quantized ops.

    This only renames tensors and views the per-row scale as ``[out, 1]``.
    The packed weight is never dequantized here; comfy expands only the active
    layer at forward time and accounts for the packed storage during offload.
    """
    int8_keys = [k for k in sd if k.endswith(".weight.int8")]
    if not int8_keys:
        return sd

    try:
        from comfy.quant_ops import QUANT_ALGOS
        supported = "int8_tensorwise" in QUANT_ALGOS
    except ImportError:
        supported = False
    if not supported or not hasattr(comfy.ops, "mixed_precision_ops"):
        raise RuntimeError(
            "This Scenema INT8 checkpoint requires a ComfyUI version with "
            "native int8_tensorwise mixed-precision operations. Update ComfyUI; "
            "the checkpoint will not be expanded to a dense fallback."
        )

    marker = torch.tensor(
        list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8)
    for k8 in int8_keys:
        weight_key = k8[: -len(".int8")]
        scale_key = weight_key + ".scale"
        if scale_key not in sd:
            raise RuntimeError(f"Scenema INT8 layer has no scale tensor: {k8}")
        weight = sd.pop(k8)
        scale = sd.pop(scale_key)
        if weight.dtype != torch.int8:
            raise RuntimeError(f"Scenema INT8 layer is {weight.dtype}, expected int8: {k8}")
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        module_prefix = weight_key[: -len("weight")]
        sd[weight_key] = weight
        sd[module_prefix + "weight_scale"] = scale
        sd[module_prefix + "comfy_quant"] = marker
    logger.info("Scenema: retained %d linear layers as native packed INT8", len(int8_keys))
    return sd


class _ScenemaGGMLOps(GGMLOps):
    """GGUF ops which leave absent audio-only video weights unallocated."""

    class Linear(GGMLOps.Linear):
        def ggml_load_from_state_dict(
                self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
            prefix_len = len(prefix)
            for key, value in state_dict.items():
                name = key[prefix_len:]
                if name == "weight":
                    self.weight = torch.nn.Parameter(value, requires_grad=False)
                elif name == "bias" and value is not None:
                    self.bias = torch.nn.Parameter(value, requires_grad=False)
                else:
                    unexpected_keys.append(key)
            if self.weight is None:
                missing_keys.append(prefix + "weight")
            elif getattr(self.weight, "is_largest_weight", False):
                self.largest_layer = True


def _normalize_transformer_keys(sd):
    """Map Scenema/ltx_core DiT keys onto comfy's `model.diffusion_model.` layout."""
    if any(k.startswith("model.diffusion_model.") for k in sd):
        return sd
    if any(k.startswith("velocity_model.") for k in sd):
        return {"model.diffusion_model." + k[len("velocity_model."):]: v
                for k, v in sd.items()}
    # bare keys (e.g. a GGUF quantized with the diffusion_model prefix stripped)
    return {"model.diffusion_model." + k: v for k, v in sd.items()}


def _nuke_video_paths(model):
    """Remove every per-block module skipped by the permanent audio gates."""
    blocks = model.model.diffusion_model.transformer_blocks
    names = ("attn1", "attn2", "ff", "audio_to_video_attn", "video_to_audio_attn")
    for block in blocks:
        for name in names:
            setattr(block, name, torch.nn.Identity())
    model.size = 0
    logger.info("Scenema: removed %d unused video/cross-modal modules", len(blocks) * len(names))


def _pad_detection_key(sd, metadata):
    """Zero-fill the one video tensor model detection reads the shape of.

    The audio-only Scenema checkpoint ships no video-path weights (they are
    never run — `run_vx=False` is baked into the loaded model), but comfy's
    ltxv detection unconditionally reads
    `transformer_blocks.0.attn2.to_k.weight`. The value read off it is
    overridden by the checkpoint's embedded config metadata right after, so a
    correctly-shaped zero tensor is enough.
    """
    key = "model.diffusion_model.transformer_blocks.0.attn2.to_k.weight"
    if key in sd:
        return sd
    cfg = {}
    try:
        if metadata and "config" in metadata:
            cfg = json.loads(metadata["config"]).get("transformer", {})
    except Exception:
        cfg = {}
    head_dim = cfg.get("attention_head_dim", 128)
    heads = cfg.get("num_attention_heads", 32)
    cad = cfg.get("cross_attention_dim", 4096)
    sd[key] = torch.zeros(heads * head_dim, cad, dtype=torch.bfloat16)
    return sd


def _load_file_sd(path):
    """safetensors/bin load with metadata, or GGUF via this pack's loader."""
    if path.lower().endswith(".gguf"):
        sd, extra = gguf_sd_loader(path, handle_prefix=None)
        return sd, extra.get("metadata", {}), True
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    return sd, metadata, False


def _audio_vae_loaded(vae):
    """Load the audio VAE onto the compute device, mirroring comfy's VAE.decode.

    Encoding has to call ``first_stage_model.encode(waveform, sample_rate=...)``
    directly — the VAE.encode wrapper can't pass the source sample rate, and a
    wrong one would silently mis-resample (Scenema's VAE runs at 16 kHz).
    """
    comfy.model_management.load_models_gpu(
        [vae.patcher], force_full_load=getattr(vae, "disable_offload", False))
    return vae.first_stage_model


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


# ── Nodes ─────────────────────────────────────────────────────────────────

SCENEMA_CATEGORY = "🤖 CCTech/Scenema"


class ScenemaModelLoader:
    """Load the Scenema Audio stack from user-selectable checkpoint files.

    Outputs plain comfy MODEL / CLIP / VAE objects built on ComfyUI's native
    LTX-AV machinery, so they also compose with comfy's own LTXV nodes.
    The MODEL has the audio-only forward baked in (video paths gated off),
    matching the original ScenemaAudio nodes' monkey-patched transformer.
    """

    CATEGORY = SCENEMA_CATEGORY
    TITLE = "Scenema Models Loader ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer_name": (_unet_filename_list(), {
                    "tooltip": "scenema-audio-transformer-int8.safetensors (or the bf16 "
                               "checkpoint, or a GGUF quant of either) from "
                               "models/diffusion_models (unet). INT8 and GGUF weights "
                               "remain quantized and dequantize one layer at a time."}),
                "text_encoder_name": (_clip_filename_list(), {
                    "tooltip": "Gemma-3 12B text encoder, .gguf (stays quantized) or "
                               "safetensors, from models/text_encoders (clip)."}),
                "pipeline_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "scenema-audio-pipeline.safetensors (or -pipeline-audio) "
                               "from models/vae. Carries the text projection, embeddings "
                               "connectors, audio VAE decoder and vocoder."}),
                "vae_encoder_name": (["none"] + folder_paths.get_filename_list("vae"), {
                    "tooltip": "scenema-audio-vae-encoder.safetensors from models/vae. "
                               "Optional: needed to encode voice references only when "
                               "the pipeline file above ships without its encoder "
                               "(the full pipeline checkpoint already includes one)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the Scenema Audio components (audio DiT, Gemma-3 text "
                   "encoder with GGUF support, audio VAE) as native comfy objects.")

    def load(self, transformer_name, text_encoder_name, pipeline_name,
             vae_encoder_name="none"):
        # ── transformer → comfy LTXAV MODEL (audio-only forward baked in) ──
        t_path = folder_paths.get_full_path("unet", transformer_name) \
            or folder_paths.get_full_path_or_raise("unet", transformer_name)
        t_sd, t_meta, is_gguf = _load_file_sd(t_path)

        pipeline_path = folder_paths.get_full_path("vae", pipeline_name) \
            or folder_paths.get_full_path_or_raise("vae", pipeline_name)
        p_sd, p_meta, _ = _load_file_sd(pipeline_path)

        has_native_int8 = any(k.endswith(".weight.int8") for k in t_sd)
        t_sd = _convert_int8_keys(t_sd)
        t_sd = _normalize_transformer_keys(t_sd)

        # Merge the embeddings connectors from the pipeline checkpoint so the
        # DiT's own preprocess_text_embeds path (used when the TE emits
        # unprocessed embeds) has real weights.
        merged = 0
        for k, v in p_sd.items():
            if k.startswith("model.diffusion_model.") and k not in t_sd:
                t_sd[k] = v
                merged += 1
        if merged:
            logger.info("Scenema: merged %d connector tensors from pipeline ckpt", merged)

        metadata = t_meta if t_meta and "config" in t_meta else p_meta
        t_sd = _pad_detection_key(t_sd, metadata)

        model_options = {}
        kwargs = {}
        valid = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid and metadata:
            kwargs["metadata"] = metadata
        if is_gguf:
            model_options["custom_operations"] = _ScenemaGGMLOps()
        elif not has_native_int8:
            # MixedPrecisionOps creates linears without dense weight storage.
            # Present BF16 weights are assigned; absent video weights stay None.
            model_options["custom_operations"] = comfy.ops.mixed_precision_ops({})

        model = comfy.sd.load_diffusion_model_state_dict(
            t_sd, model_options=model_options, **kwargs,
        )
        if model is None:
            raise RuntimeError(
                f"Could not detect {transformer_name} as an LTX-AV diffusion model. "
                "Check that it is a Scenema audio transformer checkpoint.")
        if is_gguf:
            from .nodes import GGUFModelPatcher
            model = GGUFModelPatcher.clone(model)

        _nuke_video_paths(model)

        # Audio-only forward — the comfy-native equivalent of the original
        # nodes' monkey-patched BasicAVTransformerBlock.forward.
        topts = model.model_options.setdefault("transformer_options", {})
        topts.update({"run_vx": False, "a2v_cross_attn": False, "v2a_cross_attn": False})
        del t_sd

        # ── text encoder → comfy LTXAV CLIP (Gemma + projection + connectors) ──
        te_path = folder_paths.get_full_path("clip", text_encoder_name) \
            or folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)
        if te_path.lower().endswith(".gguf"):
            te_sd = gguf_clip_loader(te_path)
            te_options = {
                "custom_operations": GGMLOps(),
                "initial_device": comfy.model_management.text_encoder_offload_device(),
            }
        else:
            te_sd, _ = comfy.utils.load_torch_file(te_path, return_metadata=True)
            te_options = {
                "initial_device": comfy.model_management.text_encoder_offload_device(),
            }

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.LTXV,
            state_dicts=[te_sd, p_sd],
            model_options=te_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        del te_sd

        # ── audio VAE → comfy VAE (AudioVAE: encoder + decoder + vocoder/BWE) ──
        vae_sd = {k: v for k, v in p_sd.items()
                  if k.startswith(("audio_vae.", "vocoder."))}
        if not vae_sd:
            raise RuntimeError(
                f"{pipeline_name} carries no audio_vae./vocoder. tensors — it is not "
                "a Scenema pipeline checkpoint.")
        if "autoencoder.encoder.conv_in.conv.weight" not in \
                {k.replace("audio_vae.", "autoencoder.") for k in vae_sd} \
                and vae_encoder_name != "none":
            enc_path = folder_paths.get_full_path("vae", vae_encoder_name) \
                or folder_paths.get_full_path_or_raise("vae", vae_encoder_name)
            enc_sd, _, _ = _load_file_sd(enc_path)
            n = 0
            for k, v in enc_sd.items():
                if k.startswith("per_channel_statistics."):
                    vae_sd["audio_vae." + k] = v
                elif not k.startswith(("decoder.", "vocoder.")):
                    vae_sd["audio_vae.encoder." + k] = v
                    n += 1
            logger.info("Scenema: merged %d encoder tensors from %s", n, vae_encoder_name)
            del enc_sd

        vae = comfy.sd.VAE(sd=vae_sd, metadata=p_meta)
        vae.throw_exception_if_invalid()
        if "autoencoder.encoder.conv_in.conv.weight" not in \
                {k.replace("audio_vae.", "autoencoder.") for k in vae_sd}:
            logger.warning(
                "Scenema: pipeline checkpoint ships no VAE encoder — decode-only. "
                "Set vae_encoder_name to scenema-audio-vae-encoder.safetensors "
                "to use voice references.")
        del vae_sd, p_sd, p_meta

        return (model, clip, vae)

class ScenemaVAEEncode:
    """Encode reference audio to an audio latent for voice cloning (A2V).

    Port of the original Scenema Audio VAE Encode node: caps the reference at
    20 seconds (longer does not improve cloning), resamples to the VAE rate,
    and outputs an audio LATENT for the Generate node's `ref_latent` input.
    """

    CATEGORY = SCENEMA_CATEGORY
    TITLE = "Scenema VAE Encode (voice reference) ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "max_seconds": ("FLOAT", {
                    "default": MAX_REF_SECONDS, "min": 1.0, "max": MAX_REF_SECONDS,
                    "step": 0.5,
                    "tooltip": "Seconds of reference audio to encode. Hard-capped at 20s."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("ref_latent",)
    FUNCTION = "encode"
    DESCRIPTION = "Encode a voice reference clip to an audio latent for Scenema generation."

    @torch.inference_mode()
    def encode(self, vae, audio, max_seconds=MAX_REF_SECONDS):
        max_seconds = min(max_seconds, MAX_REF_SECONDS)
        fsm = _audio_vae_loaded(vae)
        if not hasattr(fsm, "encode"):
            raise RuntimeError("The connected VAE is not a Scenema/LTX audio VAE.")

        waveform = audio["waveform"][0].to(
            device=vae.device, dtype=vae.vae_dtype)  # [C, T]
        sr = audio["sample_rate"]

        max_samples = int(max_seconds * sr)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]

        # AudioVAE.encode takes [B, C, T]; mono is expanded to stereo inside
        audio_vae = fsm.encode(waveform.unsqueeze(0), sample_rate=sr).to(
            vae.output_device)
        return ({"samples": audio_vae, "type": "audio"},)


class ScenemaAudioGenerate:
    """Generate expressive speech — the main Scenema Audio node.

    Same fields and behaviour as the original: voice description, action tags
    (one per line), speech text with inline ``[bracketed cues]``, scene and
    language, a preset dropdown with the 12 production voices, pace, seed and
    an optional voice-clone reference latent. Long text is auto-split into
    ~15s chunks chained through A2V voice conditioning.
    """

    CATEGORY = SCENEMA_CATEGORY
    TITLE = "Scenema Audio Generate ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "preset": (PRESET_NAMES, {
                    "default": CUSTOM,
                    "tooltip": "Pick a preset to auto-fill voice, gender, scene, action "
                               "tags and speech text. Choose Custom to write your own."}),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "Male, late 60s. Deep, gravelly. Slow and deliberate. "
                               "The weight of the cosmos in every word.",
                    "tooltip": "Describe the voice: age, gender presentation, timbre, "
                               "accent, delivery style."}),
                "gender": (["male", "female"], {
                    "tooltip": "Grammatical gender used for pronouns in the compiled prompt."}),
                "speech_text": ("STRING", {
                    "multiline": True,
                    "default": "Look again at that dot. That's here. That's home. That's us.",
                    "tooltip": "The text to speak. Use [bracketed cues] inline for "
                               "mid-speech performance direction: [He laughs], [She "
                               "whispers]. Long text is auto-split at sentence boundaries."}),
                "scene": (SCENE_PRESETS, {
                    "tooltip": "Acoustic environment injected into the prompt."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "custom_scene": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Freeform scene description. When non-empty, overrides "
                               "the scene dropdown."}),
                "action_tags": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Delivery cues, one per line. Each becomes a stage "
                               "direction the model performs."}),
                "language": (LANGUAGE_OPTIONS, {
                    "default": "English",
                    "tooltip": "Target language. Write speech_text in that language."}),
                "pace": ("FLOAT", {
                    "default": LTX_MULTIPLIER, "min": 0.5, "max": 3.0, "step": 0.1,
                    "tooltip": "Duration budget multiplier. Higher = slower speech. "
                               "1.5 is the validated default."}),
                "ref_latent": ("LATENT", {
                    "tooltip": "Optional voice reference (Scenema VAE Encode output) "
                               "for zero-shot cloning."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    DESCRIPTION = ("Expressive text-to-speech with zero-shot voice cloning on the "
                   "Scenema audio diffusion model.")

    @torch.inference_mode()
    def generate(self, model, clip, vae, preset, voice_description, gender,
                 speech_text, scene, seed, custom_scene="", action_tags="",
                 language="English", pace=LTX_MULTIPLIER, ref_latent=None):
        if preset != CUSTOM and preset in PRESETS:
            p = PRESETS[preset]
            voice_description = p["voice_description"]
            gender = p["gender"]
            speech_text = p["speech_text"]
            scene = p["scene"]
            custom_scene = p.get("custom_scene", "")
            action_tags = p["action_tags"]
            logger.info("Scenema: preset applied: %s", preset)

        if scene == SCENE_SENTINEL and not (custom_scene and custom_scene.strip()):
            raise ValueError(
                "scene: required field. Pick a scene preset or provide custom_scene.")

        fsm = vae.first_stage_model
        if not hasattr(fsm, "num_of_latents_from_frames"):
            raise RuntimeError("The connected VAE is not a Scenema/LTX audio VAE.")
        if not hasattr(clip.cond_stage_model, "text_embedding_projection"):
            raise RuntimeError(
                "The connected CLIP is not a Scenema/LTX-AV text encoder — it "
                "emits raw Gemma hidden states the audio model cannot consume. "
                "Connect the 'clip' output of Scenema Models Loader (it pairs "
                "the Gemma encoder with the pipeline checkpoint's text "
                "projection), not a plain CLIP Loader (GGUF).")

        xml_prompt = build_speak_xml(voice_description, gender, speech_text,
                                     scene, custom_scene, action_tags, language)
        logger.info("Scenema XML prompt:\n%s", xml_prompt)
        compiled = compile_prompt(xml_prompt)
        logger.info("Scenema compiled prompt: %s", compiled.prompt)
        chunks = plan_chunks(xml_prompt, compiled, base_seed=seed, pace=pace)

        device = comfy.model_management.get_torch_device()
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device,
                              dtype=torch.float32)

        current_ref = None
        if ref_latent is not None:
            ref = ref_latent["samples"]
            if getattr(ref, "is_nested", False):
                ref = ref.unbind()[-1]
            if (ref.dim() != 4 or ref.shape[-3] != fsm.latent_channels
                    or ref.shape[-1] != fsm.latent_frequency_bins):
                raise ValueError(
                    f"ref_latent shape {tuple(ref.shape)} is not a Scenema audio "
                    f"VAE latent (expected [B, {fsm.latent_channels}, T, "
                    f"{fsm.latent_frequency_bins}]). Connect the output of "
                    "'Scenema VAE Encode (voice reference)' fed by LoadAudio — "
                    "a plain audio latent (e.g. EmptyLatentAudio) carries no "
                    "voice identity and cannot be used for cloning.")
            b, c, t, f = ref.shape
            current_ref = ref.permute(0, 2, 1, 3).reshape(b, t, c * f).to(device)

        # ── Phase 1: encode ALL chunk prompts in one text-encoder session ──
        # (matches the original node; keeps the TE from being swapped in and
        # out per chunk). ref_audio differs per chunk, so it is attached in
        # phase 2 to a shallow copy of the conditioning.
        chunk_conds = []
        for i, chunk in enumerate(chunks):
            logger.info("Scenema: encoding chunk %d/%d (%.1fs)",
                        i + 1, len(chunks), chunk.duration_s)
            cond = clip.encode_from_tokens_scheduled(
                clip.tokenize(chunk.compiled_prompt))
            cond[0][0] = cond[0][0].cpu()
            cond[0][1]["frame_rate"] = FPS
            chunk_conds.append(cond)

        # ── Phase 2: diffuse + decode all chunks in one transformer session ──
        waveforms = []
        sr = None
        for i, (chunk, cond) in enumerate(zip(chunks, chunk_conds)):
            logger.info("Scenema: chunk %d/%d (%.1fs) — diffusing",
                        i + 1, len(chunks), chunk.duration_s)
            if current_ref is not None:
                cond = [[cond[0][0], {**cond[0][1],
                                      "ref_audio": {"tokens": current_ref}}]]

            num_latents = fsm.num_of_latents_from_frames(
                int(chunk.duration_s * FPS) + 1, FPS)
            video = torch.zeros(1, 128, 1, 1, 1,
                                device=comfy.model_management.intermediate_device())
            audio = torch.zeros(1, fsm.latent_channels, num_latents,
                                fsm.latent_frequency_bins,
                                device=comfy.model_management.intermediate_device())
            latent_image = comfy.nested_tensor.NestedTensor((video, audio))
            noise = comfy.sample.prepare_noise(latent_image, chunk.seed)

            out = comfy.sample.sample(
                model, noise, steps=len(DISTILLED_SIGMA_VALUES) - 1, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=cond, negative=cond, latent_image=latent_image,
                sigmas=sigmas, seed=chunk.seed,
            )
            audio_latent = out.unbind()[-1] if out.is_nested else out
            waveform = vae.decode(audio_latent)[0].float().cpu()  # [T, C]
            sr = fsm.output_sample_rate
            waveforms.append(waveform)
            del cond, latent_image, noise, out, audio_latent

            if i < len(chunks) - 1:
                current_ref = self._encode_reference(
                    vae, waveform, sr).to(device)

        # Per-chunk trim + normalize before concat, then cap internal silences
        # (ported from the original generate loop).
        processed = []
        for w in waveforms:
            w_np = w.numpy()
            w_np = trim_silence(w_np, sr, max_silence=0.5)
            w_np = normalize_volume(w_np, sr)
            processed.append(w_np)
        combined_np = np.concatenate(processed, axis=0)
        combined_np = shorten_long_silence(combined_np, sr,
                                           max_duration=min(0.5 * pace, 1.5))
        if combined_np.ndim == 1:
            combined = torch.from_numpy(combined_np).float().unsqueeze(0)
        else:
            combined = torch.from_numpy(combined_np.T).float()
        combined = combined.unsqueeze(0)  # [1, C, T]

        total = combined.shape[-1] / sr
        logger.info("Scenema: done — %.1fs of audio from %d chunk(s)", total, len(chunks))
        return ({"waveform": combined, "sample_rate": int(sr)},)

    @torch.inference_mode()
    def _encode_reference(self, vae, waveform, sr):
        """Encode the tail of a chunk as the A2V reference for the next one.

        ``waveform`` is [T, C] (decode layout); encode wants [B, C, T].
        """
        fsm = _audio_vae_loaded(vae)
        tail = waveform[-int(REF_TAIL_SECONDS * sr):, :].to(
            device=vae.device, dtype=vae.vae_dtype)
        latents = fsm.encode(tail.T.unsqueeze(0), sample_rate=sr)
        b, c, t, f = latents.shape
        return latents.permute(0, 2, 1, 3).reshape(b, t, c * f)


NODE_CLASS_MAPPINGS = {
    "ScenemaModelLoader": ScenemaModelLoader,
    "ScenemaVAEEncode": ScenemaVAEEncode,
    "ScenemaAudioGenerate": ScenemaAudioGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScenemaModelLoader": ScenemaModelLoader.TITLE,
    "ScenemaVAEEncode": ScenemaVAEEncode.TITLE,
    "ScenemaAudioGenerate": ScenemaAudioGenerate.TITLE,
}
