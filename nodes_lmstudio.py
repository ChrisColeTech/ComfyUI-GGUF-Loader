# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Local LLM/VLM prompting via an LM Studio server.

LM Studio serves an OpenAI-compatible REST API (default
``http://localhost:1234/v1``) for whatever model is currently loaded, vision
or text-only. This module gives that server a comfy face with the same
image-in/string-out contract as the cloud vision nodes it is meant to replace
(e.g. comfy-core's ``GeminiNode``: optional image, prompt, system_prompt in,
one STRING out) so it drops into an existing workflow's slot without touching
anything downstream.
"""

import base64
import io
import json
import logging

import numpy as np
from PIL import Image

try:
    import requests
except ImportError:  # pragma: no cover - ComfyUI's own env ships requests
    requests = None

logger = logging.getLogger(__name__)

LMSTUDIO_CATEGORY = "\U0001F916 CCTech/LM Studio"
DEFAULT_BASE_URL = "http://localhost:1234/v1"

# The ltxv23_talking_head gallery workflow's GeminiNode system prompt,
# verbatim - the two-part (spoken script + [VISUAL]/[SPEECH]/[SOUNDS]
# ID-LoRA prompt) contract the workflow's downstream RegexExtract nodes are
# already built to split on. Default it here so this node drops into that
# workflow's slot with zero setup; override the widget for anything else.
DEFAULT_SYSTEM_PROMPT = """Role: You are a professional prompt writer for ID-LoRA prompt generation.

Task: Generate a two-part output from the user’s input.

Strict Output Constraint (IMPORTANT):

Return the output as RAW PLAIN TEXT only.
DO NOT use Markdown code blocks.
DO NOT use any headers, labels, or bold text outside of the generated two-part output.
Use ONLY the separator --- to divide the two parts.
Part 1: spoken dialogue/performance text only, optimized to sound natural when spoken aloud.
Part 2: the same concept rewritten in the exact ID-LoRA tagged format using these three sections only:
[VISUAL]: ...
[SPEECH]: ...
[SOUNDS]: ...
Do NOT output anything before Part 1 or after Part 2.

Global Length Rule:

Keep the spoken content short enough to produce about 10 seconds of speech unless the user explicitly asks otherwise.
Aim for roughly 20 to 35 spoken words in Part 1.
Condense long inputs aggressively while preserving the core meaning, tone, and key message.
Prioritize brevity, clarity, speakability, and a strong opening hook over completeness.
Remove repetition, side details, filler, and nonessential context.
If the input is too long, compress it into a concise spoken version rather than preserving everything.

Part 1 Rules:

Part 1 must contain ONLY the words meant to be spoken aloud.
Do NOT begin Part 1 with visual description, scene-setting narration, character description, or action description.
Start immediately with the most attention-grabbing, high-impact spoken hook.
The hook should appear as early as possible, ideally in the first sentence or phrase.
Do NOT describe what the character looks like, where they are, what they are wearing, what they are holding, or what is happening visually unless that information is spoken by the character as part of the dialogue itself.
Part 1 should read like a real spoken performance, monologue, ad read, or direct address.
Use punctuation, pauses, ellipses, and selective capitalization to improve spoken delivery when useful.
Do NOT use SSML.
Do NOT use non-voice tags such as music, ambience, sound effects, camera notes, or technical markup.
Prefer one clear hook, one core message or benefit, and one short closing beat.

Part 2 Rules:

Part 2 must contain exactly three tagged sections in this exact order:
[VISUAL]:
[SPEECH]:
[SOUNDS]:

[VISUAL] Rules:
Describe the shot type, subject appearance, clothing, setting, lighting, framing, and visible actions.
Be descriptive enough to guide generation clearly, but keep it compact and production-useful.
Explicitly indicate that the person is speaking or talking to camera so the line is generated as on-screen speech rather than voice-over.
Default toward believable UGC-style behavior unless the user asks otherwise:
direct-to-camera delivery
handheld or phone-like framing
selfie or testimonial feel
natural gestures
authentic facial expression
slight body movement
casual presenting or showing when relevant
Avoid cinematic, polished, theatrical, or overly staged action unless the user explicitly asks for that.

[SPEECH] Rules:
Write the exact words the person should say.
This must be the literal transcript, not a summary.
Keep it closely aligned with Part 1, ideally verbatim except for minor punctuation cleanup if needed.
Do NOT add scene description, camera notes, or sound cues inside [SPEECH].

[SOUNDS] Rules:
Describe both the vocal delivery and the ambient/background audio.
Include speaker qualities such as tone, volume, pace, energy, and mic proximity or distance.
Include relevant environmental sounds, room tone, music, nature sounds, or other ambience when appropriate.
Keep the audio grounded, coherent with the visual scene, and not overly busy unless requested.

Consistency Rules:

Part 1 and Part 2 must describe the same idea, message, tone, and scene.
[SPEECH] in Part 2 should match Part 1 as closely as possible.
[VISUAL], [SPEECH], and [SOUNDS] must feel like one unified prompt, not separate concepts.
Do not introduce unrelated ideas, props, settings, or actions that were not implied by the user’s request.
If the user provides exact wording, preserve it in Part 1 and [SPEECH] unless the user asks for rewriting or shortening.

Safety and Quality Rules:

Do not introduce sensitive, explicit, hateful, political, or unsafe content that was not already in the user input.
Ensure the final output always contains exactly two parts separated by ---.
Ensure Part 2 always uses the exact three ID-LoRA tags and includes all three of them once."""


def _list_models(base_url):
    """Live /v1/models call. Raises on failure - callers decide how to
    surface that, this has no silent fallback to hide behind.
    """
    resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def resolve_model(base_url, model):
    """An explicit ``model`` id wins outright. Blank means "whatever LM
    Studio currently has loaded" - a live call, not a value cached from
    whenever the node happened to be added to the graph, so it stays correct
    across LM Studio model switches and across whatever ``base_url`` this
    node instance is pointed at right now.
    """
    if model:
        return model
    try:
        ids = _list_models(base_url)
    except Exception as e:
        raise RuntimeError(
            f"model was left blank (auto-detect) but LM Studio at {base_url} "
            f"could not be reached to look up the loaded model: {e}"
        ) from e
    if not ids:
        raise RuntimeError(
            f"model was left blank (auto-detect) but LM Studio at {base_url} "
            f"reports no loaded model. Load one in LM Studio or type its id "
            f"into the model field."
        )
    return ids[0]


def _image_to_data_uri(image):
    """First frame of a comfy IMAGE batch -> a base64 PNG data URI."""
    frame = (image[0].clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_chat_payload(prompt, model, temperature, max_tokens, seed,
                       image=None, system_prompt=""):
    """Pure payload assembly, split out so it can be unit-tested without a
    running server or a torch/PIL image tensor.
    """
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": _image_to_data_uri(image)}})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }


def extract_reply(data):
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected LM Studio response shape: {json.dumps(data)[:500]}"
        ) from e


class LMStudioVisionPrompt:
    """Chat/vision call against a local LM Studio server.

    Drop-in replacement for a cloud vision-prompt node (image + prompt +
    system_prompt in, one STRING out) - e.g. the reference talking-head
    workflow's ``GeminiNode`` slot, unchanged downstream regex extraction.
    """

    CATEGORY = LMSTUDIO_CATEGORY
    TITLE = "LM Studio Vision Prompt ⚡"
    SEARCH_ALIASES = ['llm', 'vision prompt', 'caption image', 'local llm', 'chat completion', 'image to text', 'image captioning']
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    DESCRIPTION = ("Send an optional image + prompt to a local LM Studio "
                   "server's OpenAI-compatible /chat/completions endpoint.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL,
                                        "tooltip": "LM Studio > Developer > Start Server."}),
                "model": ("STRING", {"default": "",
                    "tooltip": "Leave blank to auto-use whatever model is "
                               "currently loaded in LM Studio (queried from "
                               "base_url at run time). Type an exact model "
                               "id (see LM Studio > Developer's model list, "
                               "or GET {base_url}/models) to pin one."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Sent as a vision content block. "
                                               "Needs a vision-capable model loaded."}),
                "system_prompt": ("STRING", {"multiline": True,
                                             "default": DEFAULT_SYSTEM_PROMPT}),
            },
        }

    def generate(self, prompt, base_url, model, temperature, max_tokens, seed,
                image=None, system_prompt=""):
        if requests is None:
            raise RuntimeError("The 'requests' package is required for "
                               "LMStudioVisionPrompt but is not installed.")

        model = resolve_model(base_url, model)
        payload = build_chat_payload(prompt, model, temperature, max_tokens,
                                     seed, image=image, system_prompt=system_prompt)
        url = f"{base_url.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Could not reach LM Studio at {base_url}. Is the local "
                f"server running (LM Studio > Developer > Start Server)?"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"LM Studio returned {resp.status_code}: {resp.text[:500]}"
            ) from e

        text = extract_reply(resp.json())
        logger.info("LM Studio (%s): %d chars back", model, len(text))
        return (text,)


NODE_CLASS_MAPPINGS = {
    "LMStudioVisionPrompt": LMStudioVisionPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMStudioVisionPrompt": LMStudioVisionPrompt.TITLE,
}
