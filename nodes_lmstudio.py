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
NO_SERVER_MODEL = "(no server reached - type a model id)"


def _list_models(base_url):
    """Best-effort model list for the COMBO widget. Never raises: LM Studio
    not being up yet is the expected state at graph-build time, not an error.
    """
    if requests is None:
        return [NO_SERVER_MODEL]
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=2)
        resp.raise_for_status()
        ids = [m["id"] for m in resp.json().get("data", [])]
        return ids or [NO_SERVER_MODEL]
    except Exception:
        return [NO_SERVER_MODEL]


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
    TITLE = "LM Studio Vision Prompt \U0001F5A5️"
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
                "model": (_list_models(DEFAULT_BASE_URL), {
                    "tooltip": "Populated from a live /v1/models call at graph "
                               "build time. Re-add the node after loading a "
                               "different model in LM Studio to refresh it."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Sent as a vision content block. "
                                               "Needs a vision-capable model loaded."}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def generate(self, prompt, base_url, model, temperature, max_tokens, seed,
                image=None, system_prompt=""):
        if requests is None:
            raise RuntimeError("The 'requests' package is required for "
                               "LMStudioVisionPrompt but is not installed.")

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
