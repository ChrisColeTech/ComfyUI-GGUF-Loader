# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Write MiniMax H3 prompts with a local text encoder.

H3's base schema is three labelled fields (plus a fixed instruction line for the
keyframe modes), and quality drops sharply when the two audio fields are missing
— the host wrap does not invent them. Writing that by hand for every clip is
tedious, so this node hands the schema to a local LLM and assembles the envelope
itself: the model supplies only the three field bodies, and the exact MiniMax
instruction wording, the labels, and the blank-line layout come from
``minimax_h3_prompt.py``. A model that ignores the format cannot corrupt it.
"""

import logging

import comfy.model_management
import comfy.sd
import folder_paths

from .minimax_h3_prompt import (build_prompt, duration_seconds, last_shot_index,
                                parse_fields, reference_header, schema_problems)

logger = logging.getLogger(__name__)

H3_CATEGORY = "🤖 CCTech/MiniMax H3"

# Product envelope from the H3 guide: 24 fps on a n % 17 == 5 frame grid.
FRAME_PRESETS = {
    "124 (~5.17s)": 124,
    "243 (~10.13s)": 243,
    "362 (~15.08s)": 362,
    "481 (~20.04s)": 481,
}

SYSTEM_PROMPT = """You write prompts for the MiniMax H3 audiovisual video model. Given a user's short idea, you output exactly three labelled fields and nothing else.

Output format — these three labels, each on its own line, separated by a blank line:

integrated_multimodal_description: <the timeline>

overall_soundscape: <ambient and physical sound>

non_diegetic_music: <the score>

Never output anything else: no preamble, no explanation, no markdown, no code fences, no headings, no bullet points. Never write the instruction line about reference pictures — that is added for you.

FIELD 1 — integrated_multimodal_description
The full timeline of visuals, camera, action, speakers, dialogue, on-screen text, and diegetic sound. Every detail must correspond to something visible or audible.
- Open [Shot 1] with the overall visual style and the initial composition, e.g. "[Shot 1] Live-action, cinematic, a medium-wide shot frames...". Styles: cinematic, live-action, 2D-animated, 3D CG, claymation, watercolour, vintage film.
- [Shot 1] NEVER carries a timestamp. Do not write "at 00:00.000" or any other time on Shot 1 — it starts the clip by definition. Only shots after the first carry a cut time, numbered in order and strictly increasing inside the clip duration: "[Shot 2] At 00:03.500, the camera cuts to...". Prefer the phrases "the camera cuts to", "the shot cuts to", "the shot transitions to". A cut must introduce new information; if only the distance or angle changes slightly, use camera motion instead of a cut.
- Use few shots. One continuous shot is best, and a short clip rarely needs more than two or three; a cut every couple of seconds wastes the duration and looks frantic. Let the camera move within a shot instead of cutting.
- Write camera motion as natural English inside the sentence, never as trailing tags. Combine type + amplitude + speed, omitting amplitude and speed when they are ordinary. Types: Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down, Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly, POV, Roll Clockwise/Counterclockwise. Amplitude: "with small amplitude", "with large amplitude". Speed: "at slow speed", "at fast speed". Example: "The camera pushes in with small amplitude at slow speed toward the folded letter in her hands."
- Give every character who vocalises a stable ID: (S1), (S2), joint speech (S1,S2). The same person keeps the same ID across shots. Characters who never vocalise get no ID. On a speaker's first appearance, establish identity outside the dialogue tag: age, gender, on-screen status, pitch, timbre, rate, accent as needed.
- Dialogue goes inside <d> tags containing only the language marker and the exact spoken words: The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
- Never translate or reword dialogue the user supplied, and never put dialogue outside <d>.
- A character speaking on screen simply says their line — write nothing about their lips. Only a VOICEOVER, where the audience hears a character who is not speaking on screen, uses the exact phrase "says in an off-screen voiceover", and only then, immediately after that <d> block, do you add that the character's lips remain completely closed. Never attach "lips remain closed" to ordinary on-screen dialogue: it would tell the model to animate a closed mouth over spoken words.
- On-screen text goes in double quotes with its original spelling: A red neon sign reading "OPEN" glows above the doorway.
- Diegetic sound — sound the characters can hear, such as footsteps timed to a step, a door slam, a ringtone, a radio playing in the scene — belongs here on the timeline, not in the soundscape field. Singing the characters perform also belongs here, with speaker IDs and <d> for the lyrics.

FIELD 2 — overall_soundscape
One continuous paragraph, one to four English sentences, covering ambient sound, physical action sound, and non-verbal human sound across the whole clip: wind, rain, traffic, footsteps, fabric, impacts, breathing, laughter, room tone, water, machinery. Never put dialogue, singing, or already-described diegetic music here, and never write an abstract mood essay. Use "N/A" only when the user explicitly asked for complete silence.
Example: Steady rain taps against the cafe windows while low room ambience continues underneath. The entrance bell rings once, followed by wet footsteps and the soft scrape of a chair.

FIELD 3 — non_diegetic_music
One to three English sentences describing score that only the audience hears, never the characters. State instrumentation, tempo or rhythm, and dynamic change. Never use mood words alone such as "emotional" or "epic", and never put character-audible music here. Use "N/A" when the clip has no score.
Example: Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out.

Follow every element of the user's idea. Where the idea is vague, invent concrete, observable detail. Describe only what can be seen or heard — never smell, taste, or touch, and never a character's inner feelings."""

MODE_GUIDANCE = {
    "t2va": (
        "Mode: text-to-video. There is no reference image. Build the whole scene "
        "from the idea alone."
    ),
    "i2va": (
        "Mode: image-to-video from a first frame. <Picture 1> IS the first frame at "
        "0.00 seconds and belongs to [Shot 1]. Open by establishing the style, "
        "subjects, wardrobe, props, and composition of <Picture 1>, then describe "
        "the action that develops forward from it, keeping identity, clothing, "
        "colours, key objects, and spatial relationships consistent. Refer to the "
        "reference as <Picture 1> in your text. Structure: first-frame anchor, "
        "action onset, continuous development, result or reaction."
    ),
    "fl2va": (
        "Mode: first-and-last-frame video. Picture 1 opens the clip and Picture 2 "
        "closes it. Describe the motion path between them — subject motion, pose "
        "change, object manipulation, evolving composition, lighting transition — "
        "not two static stills. Strongly prefer a single shot so the model can "
        "interpolate continuously. The final shot must arrive at the state, pose, "
        "spacing, and composition established by Picture 2. Structure: first-frame "
        "state, observable intermediate change, narrowing differences, last-frame state."
    ),
    "l2va": (
        "Mode: last-frame video. <Picture 1> is the FINAL frame and belongs to the "
        "last shot, not Shot 1. Infer a plausible earlier state from the idea and "
        "the final frame, then converge on it, landing exactly on the arrangement, "
        "camera angle, lighting, and composition of <Picture 1> at the end. "
        "Structure: plausible preceding state, explicit action, gradual convergence, "
        "last-frame landing."
    ),
}


def _text_encoder_list():
    # models/text_encoders collects side-car files too (tokenizer json, spiece
    # caches); only offer things that could actually be a model.
    files = list(folder_paths.get_filename_list("clip"))
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    files = [f for f in files if f.lower().endswith((".gguf", ".safetensors", ".sft"))]
    return sorted(files) or ["none"]


# comfy identifies the architecture from the state dict itself; a CLIP type only
# selects which downstream wrapper gets built around it. Most generation-capable
# encoders ignore it entirely (the Gemma-3 12B branch has no type check at all),
# and where it does branch — Qwen3-VL — the variants differ in chat templating,
# which this node supplies itself. So pick a type that is known to yield a full
# LLM and let detection do the real work.
_TYPE_FOR_ARCH = {
    "GEMMA_3_12B": "ltxv",
    "GEMMA_3_4B": "lumina2",
    "GEMMA_3_4B_VISION": "lumina2",
    "QWEN3VL_4B": "qwen_image",
    "QWEN3VL_8B": "qwen_image",
    "QWEN25_7B": "qwen_image",
    "QWEN25_3B": "omnigen2",
}
_DEFAULT_TYPE = "stable_diffusion"


def _detect_encoder_type(state_dict):
    try:
        detected = comfy.sd.detect_te_model(state_dict)
    except Exception:
        detected = None
    name = getattr(detected, "name", None)
    return name, _TYPE_FOR_ARCH.get(name, _DEFAULT_TYPE)


# Loading a text encoder costs several GB and many seconds, and ComfyUI re-runs
# this node on every prompt edit. Cache the most recent one; the CLIP owns a
# ModelPatcher, so comfy still manages its VRAM and can offload it.
_ENCODER_CACHE = {}


def _load_text_encoder(name):
    cached = _ENCODER_CACHE.get(name)
    if cached is not None:
        return cached

    from .nodes import CLIPLoaderGGUF

    loader = CLIPLoaderGGUF()
    path = folder_paths.get_full_path("clip", name) \
        or folder_paths.get_full_path_or_raise("text_encoders", name)
    state_dict = loader.load_data([path])[0]
    arch, type_name = _detect_encoder_type(state_dict)
    logger.info("MiniMax H3: loading %s (detected %s, building as %s)",
                name, arch or "unknown architecture", type_name)

    clip_type = getattr(comfy.sd.CLIPType, type_name.upper(),
                        comfy.sd.CLIPType.STABLE_DIFFUSION)
    _ENCODER_CACHE.clear()
    clip = loader.load_patcher([path], clip_type, [state_dict])
    _ENCODER_CACHE[name] = clip
    return clip


def _format_chat(clip, system, user):
    """Wrap system/user text in the encoder's chat markers.

    Comfy's Gemma tokenizers for the video encoders do not apply a chat template
    themselves, so the turn markers are written explicitly, exactly as comfy's
    own LTX2 prompt node does.
    """
    name = getattr(getattr(clip, "tokenizer", None), "clip_name", "") or ""
    if "gemma4" in name:
        return (f"<|turn>system\n{system}<turn|>\n"
                f"<|turn>user\n{user}<turn|>\n"
                f"<|turn>model\n<|channel>final\n")
    if "gemma" in name:
        return (f"<start_of_turn>system\n{system}<end_of_turn>\n"
                f"<start_of_turn>user\n{user}<end_of_turn>\n"
                f"<start_of_turn>model\n")
    # Qwen and friends use the ChatML markers this matches.
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class MiniMaxH3PromptWriter:
    """Turn a short idea into a schema-correct MiniMax H3 prompt."""

    CATEGORY = H3_CATEGORY
    TITLE = "MiniMax H3 Prompt Writer ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_encoder": (_text_encoder_list(), {
                    "tooltip": "Local LLM that writes the prompt. Needs a "
                               "generation-capable encoder such as Gemma-3 12B "
                               "or a Qwen3-VL. The architecture is detected from "
                               "the file; ignored when a clip is connected."}),
                "idea": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "A lighthouse keeper climbs the stairs at dawn and "
                               "looks out over a calm sea.",
                    "tooltip": "Your raw idea. Everything you state here is kept; "
                               "anything vague is filled in with concrete detail."}),
                "mode": (list(MODE_GUIDANCE), {
                    "default": "t2va",
                    "tooltip": "t2va: text only. i2va: from a first frame. "
                               "fl2va: between a first and last frame. "
                               "l2va: landing on a last frame."}),
                "length": (list(FRAME_PRESETS), {
                    "default": "243 (~10.13s)",
                    "tooltip": "Clip length. Sets the reference-alignment time and "
                               "the latest valid shot cut."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "max_length": ("INT", {"default": 768, "min": 64, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0,
                                          "step": 0.01,
                                          "tooltip": "0 generates deterministically."}),
                "include_music": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off writes non_diegetic_music: N/A, for clips that "
                               "should carry no score."}),
            },
            "optional": {
                "clip": ("CLIP", {
                    "tooltip": "Optional. Overrides the dropdown and avoids "
                               "reloading, e.g. the CLIP already loaded for the "
                               "video model."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "raw_output")
    FUNCTION = "write"
    DESCRIPTION = ("Write a MiniMax H3 prompt with a local text encoder. The model "
                   "supplies the three core fields; the node assembles the exact "
                   "MiniMax envelope around them.")

    def write(self, text_encoder, idea, mode, length, seed, max_length,
              temperature, include_music, clip=None):
        if not idea or not idea.strip():
            raise ValueError("idea must not be empty")

        seconds = duration_seconds(FRAME_PRESETS[length])

        if clip is None:
            if text_encoder == "none":
                raise ValueError(
                    "No text encoders found. Put one in models/text_encoders, or "
                    "connect a CLIP.")
            clip = _load_text_encoder(text_encoder)
        if not hasattr(clip.cond_stage_model, "generate"):
            raise ValueError(
                f"{type(clip.cond_stage_model).__name__} cannot generate text. Use a "
                "full LLM text encoder (Gemma-3 12B, Qwen3-VL); conditioning-only "
                "encoders such as the MiniMax H3 one are truncated and have no head.")

        request = [MODE_GUIDANCE[mode]]
        request.append(
            f"The clip is {seconds:.2f} seconds long, so every shot cut must be "
            f"strictly increasing and earlier than 00:{seconds:06.3f}.")
        if not include_music:
            request.append("This clip has no score: write exactly 'N/A' for "
                           "non_diegetic_music.")
        request.append(f"Idea: {idea.strip()}")

        prompt = _format_chat(clip, SYSTEM_PROMPT, "\n\n".join(request))
        tokens = clip.tokenize(prompt, skip_template=True, min_length=1)
        generated = clip.generate(
            tokens,
            do_sample=temperature > 0,
            max_length=int(max_length),
            temperature=max(temperature, 0.01),
            top_k=64,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.05,
            seed=int(seed),
        )
        raw = clip.decode(generated)

        visual, soundscape, music = parse_fields(raw)
        if not visual:
            # No usable field labels. Rather than emit a broken prompt, fall back
            # to the whole reply as the timeline so the run still produces video.
            logger.warning(
                "MiniMax H3: the model did not emit the field labels; using its "
                "whole reply as integrated_multimodal_description")
            visual = " ".join((raw or "").split()) or idea.strip()
        if not include_music:
            music = "N/A"

        for problem in schema_problems(visual, seconds):
            logger.warning("MiniMax H3: %s", problem)
        for name, value in (("overall_soundscape", soundscape),
                            ("non_diegetic_music", music)):
            if not value:
                logger.warning(
                    "MiniMax H3: the model omitted %s; H3 quality drops without it", name)

        # N in the keyframe instruction is the shot owning the last frame,
        # which the timeline just written already states.
        final = build_prompt(visual, soundscape, music, mode=mode,
                             duration_s=seconds,
                             last_shot_index=last_shot_index(visual))
        logger.info("MiniMax H3: wrote a %d-character %s prompt for %.2fs",
                    len(final), mode, seconds)
        return (final, raw)


class MiniMaxH3PromptFormat:
    """Assemble a MiniMax H3 prompt from fields you have already written."""

    CATEGORY = H3_CATEGORY
    TITLE = "MiniMax H3 Prompt Format ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "integrated_multimodal_description": ("STRING", {
                    "multiline": True, "dynamicPrompts": True, "default": "",
                    "tooltip": "The timeline. [Shot 1] is added if you omit it."}),
                "overall_soundscape": ("STRING", {
                    "multiline": True, "dynamicPrompts": True, "default": "",
                    "tooltip": "Ambient, physical, and non-verbal sound. "
                               "N/A only for deliberate silence."}),
                "non_diegetic_music": ("STRING", {
                    "multiline": True, "dynamicPrompts": True, "default": "",
                    "tooltip": "Score the characters cannot hear. N/A if none."}),
                "mode": (list(MODE_GUIDANCE), {"default": "t2va"}),
                "length": (list(FRAME_PRESETS), {"default": "243 (~10.13s)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "format"
    DESCRIPTION = ("Wrap hand-written H3 fields in the exact MiniMax envelope, "
                   "including the keyframe instruction line.")

    def format(self, integrated_multimodal_description, overall_soundscape,
               non_diegetic_music, mode, length):
        if not integrated_multimodal_description.strip():
            raise ValueError("integrated_multimodal_description must not be empty")
        seconds = duration_seconds(FRAME_PRESETS[length])
        for problem in schema_problems(integrated_multimodal_description, seconds):
            logger.warning("MiniMax H3: %s", problem)
        return (build_prompt(
            integrated_multimodal_description, overall_soundscape, non_diegetic_music,
            mode=mode, duration_s=seconds,
            last_shot_index=last_shot_index(integrated_multimodal_description)),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptWriter": MiniMaxH3PromptWriter,
    "MiniMaxH3PromptFormat": MiniMaxH3PromptFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptWriter": MiniMaxH3PromptWriter.TITLE,
    "MiniMaxH3PromptFormat": MiniMaxH3PromptFormat.TITLE,
}
