"""Minimum shared preprocessing helpers: Depth Anything V2 + plain cv2
Canny - the only two preprocessors any img2img node in this repo derives
internally (Krea2Img2Img/QwenImageImg2Img's control_mode="auto_depth"/
"auto_canny", FluxKleinImg2Img's same two modes).

This module does NOT register any standalone nodes of its own. `DepthMap`/
`Canny` exist here purely as the shared implementation backing three
historically-named nodes, each registered under its OWN pipeline's
NODE_CLASS_MAPPINGS (not here) so existing saved workflows keep resolving:
`Krea2DepthMap` (nodes/krea2.py), `Flux2KleinDepthMap` (nodes/flux_klein.py),
`QwenImageCanny` (nodes/qwen_image.py). Registering generic "DepthMap"/
"Canny" node names here would collide with the separate, standalone
ComfyUI-ControlNet-Nodes package if both are installed - that package owns
those names and the full 11-preprocessor set (normal maps, soft edges,
MLSD, lineart variants, OpenPose). Install it separately for anything
beyond depth/canny.

Depth Anything V2 downloads its own weights on first use into the real
ComfyUI install's models/depth_anything_v2/ folder - nothing is ever
vendored as a weight file in this repo.
"""
import logging

import cv2
import numpy as np
import torch

import comfy.model_management

from ..vendor import depth_anything_v2

logger = logging.getLogger(__name__)

PREPROCESSORS_CATEGORY = "\U0001F916 CCTech/Preprocessors"


def _estimate_batch(detector, image, resolution=512, **kwargs):
    """Run any detector matching DepthAnythingV2Detector's I/O convention
    (.estimate(image_hwc_uint8, resolution=512, **kwargs) -> image_hwc_uint8)
    over an IMAGE batch, returning an IMAGE batch of the same shape."""
    out = None
    for i in range(image.shape[0]):
        np_image = (image[i].cpu().numpy() * 255.0).astype(np.uint8)
        result_rgb = detector.estimate(np_image, resolution=resolution, **kwargs)
        result_tensor = torch.from_numpy(result_rgb.astype(np.float32) / 255.0)
        if out is None:
            out = torch.zeros(image.shape[0], *result_tensor.shape, dtype=torch.float32)
        out[i] = result_tensor
    return out


def _depth_anything_batch(image, ckpt_name, resolution=512):
    """Run Depth Anything V2 over an IMAGE batch, returning a depth-map
    IMAGE batch of the same shape."""
    detector = depth_anything_v2.DepthAnythingV2Detector(ckpt_name).to(
        comfy.model_management.get_torch_device())
    out = _estimate_batch(detector, image, resolution)
    del detector
    return out


def _auto_canny_control_image(image, low_threshold=100, high_threshold=200):
    """Plain cv2.Canny edge detection - no model, no download, deterministic.
    Same defaults comfyui_controlnet_aux's own Canny preprocessor uses."""
    out = []
    for i in range(image.shape[0]):
        np_image = (image[i].cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = np.repeat(edges[:, :, None], 3, axis=2)
        out.append(torch.from_numpy(edges_rgb.astype(np.float32) / 255.0))
    return torch.stack(out, dim=0)


class DepthMap:
    """Estimate a depth map from an IMAGE - Depth Anything V2 (DINOv2 + DPT).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s Depth
    Anything V2 preprocessor node (see vendor/depth_anything_v2.py for the
    full architecture port). Feed a source photo in, get a depth map IMAGE
    out - the input Krea2's depth Control LoRA, a DiffSynth depth patch, or
    Flux Klein's control_mode="auto_depth" expect. Weights auto-download
    from HuggingFace on first use into models/depth_anything_v2/, same
    pattern as this pack's Qwen3-TTS loader - no extra install needed.

    Only reachable in this repo under its historical pipeline-specific
    names (Krea2DepthMap, Flux2KleinDepthMap) - see module docstring for
    why this class isn't registered under a generic "DepthMap" name here.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Depth Map (Depth Anything V2) ⚡"
    SEARCH_ALIASES = ['depth anything', 'depth estimation', 'depth map', 'preprocessor',
                       'controlnet preprocessor', 'image to depth']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a depth map from a photo (Depth Anything V2), for "
                   "any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ckpt_name": (list(depth_anything_v2.MODEL_CONFIGS.keys()), {
                    "default": "depth_anything_v2_vitb.pth",
                    "tooltip": "Model size. vits (smallest/fastest) to vitg "
                               "(largest/slowest). Downloads on first use if "
                               "not already in models/depth_anything_v2."}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    def estimate(self, image, ckpt_name="depth_anything_v2_vitb.pth", resolution=512):
        return (_depth_anything_batch(image, ckpt_name, resolution),)


class Canny:
    """Canny edge detection - plain cv2.Canny, no model, no download.

    Feed a source photo in, get an edge-map IMAGE out - the control_image
    a canny DiffSynth/Union/Fun Qwen-Image checkpoint, a Krea2 in-context
    canny LoRA, or Flux Klein's control_mode="auto_canny" expect.

    Only reachable in this repo under its historical pipeline-specific
    name (QwenImageCanny) - see module docstring for why this class isn't
    registered under a generic "Canny" name here.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Canny ⚡"
    SEARCH_ALIASES = ['canny', 'edge detection', 'preprocessor',
                       'controlnet preprocessor', 'image to canny', 'edge map']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect"
    DESCRIPTION = ("Canny edge detection for any pipeline's control_image/"
                   "reference_image input. No model, no download.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255}),
            },
        }

    def detect(self, image, low_threshold=100, high_threshold=200):
        return (_auto_canny_control_image(image, low_threshold, high_threshold),)
