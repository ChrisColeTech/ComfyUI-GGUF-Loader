"""Shared ControlNet-aux-style preprocessor nodes: image in, structural/
identity map image out. Used standalone (feed into VAEEncode, a reference-
latent node, PreviewImage, or any ControlNet-consuming node) and by the
`control_mode="auto_*"` convenience options on Krea2Img2Img/QwenImageImg2Img/
FluxKleinImg2Img.

Consolidates what used to be duplicated per-pipeline: `_auto_canny_control_
image` was copy-pasted verbatim in nodes/krea2.py and nodes/qwen_image.py;
the Depth Anything V2 detector loop was duplicated three times (inline in
Krea2Img2Img.prepare(), inline in QwenImageImg2Img.prepare(), and as its own
helper in nodes/flux_klein.py); three near-identical standalone depth-map
nodes existed (Krea2DepthMap, Flux2KleinDepthMap, and no equivalent for
Qwen-Image at all). This file is now the one place that logic lives.

Each pipeline's own NODE_CLASS_MAPPINGS keeps its OLD node names
(Krea2DepthMap, Flux2KleinDepthMap, QwenImageCanny) registered as aliases
pointing at the shared classes here, so existing saved workflows keep
resolving to a working node - zero graph-breaking change.

Every detector downloads its own weights on first use into the real
ComfyUI install's models/<family>/ folder (folder_paths.models_dir-
relative) - nothing is ever vendored as a weight file in this repo.
"""
import logging

import cv2
import numpy as np
import torch

import comfy.model_management

from ..vendor import (depth_anything_v2, dsine, hed, lineart, lineart_anime,
                      manga_line, mlsd, normal_bae, openpose, pidinet)

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


class NormalMapBAE:
    """Estimate a surface normal map from an IMAGE - Bae et al.'s NNET
    (EfficientNet-B5 encoder + uncertainty-aware BatchNorm decoder).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s NormalBAE
    preprocessor node (see vendor/normal_bae.py). Needs the optional `timm`
    dependency (`pip install timm`, already in requirements.txt) to build
    the encoder backbone. Weights auto-download from HuggingFace on first
    use into models/normal_bae/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Normal Map (BAE) ⚡"
    SEARCH_ALIASES = ['normal map', 'normal bae', 'surface normal', 'preprocessor',
                       'controlnet preprocessor', 'image to normal']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a surface normal map from a photo (NNET/BAE), for "
                   "any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    def estimate(self, image, resolution=512):
        detector = normal_bae.NormalBAEDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution)
        del detector
        return (out,)


class NormalMapDSINE:
    """Estimate a surface normal map from an IMAGE - DSINE (camera-intrinsics-
    aware iterative refinement, EfficientNet-B5 encoder).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s DSINE
    preprocessor node (see vendor/dsine.py). Needs the optional `timm`
    dependency (`pip install timm`, already in requirements.txt) to build
    the encoder backbone - same dependency Normal Map (BAE) needs. `fov`
    synthesizes a camera intrinsics matrix (no real camera metadata is ever
    available for a plain photo); `iterations` controls the refinement
    depth. Weights auto-download from HuggingFace on first use into
    models/dsine/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Normal Map (DSINE) ⚡"
    SEARCH_ALIASES = ['normal map', 'dsine', 'surface normal', 'preprocessor',
                       'controlnet preprocessor', 'image to normal']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a surface normal map from a photo (DSINE), for "
                   "any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "fov": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 365.0, "step": 1.0,
                        "tooltip": "Synthetic camera field-of-view in degrees, used to build "
                                   "an assumed intrinsics matrix (no real camera metadata is "
                                   "available for a plain photo)."}),
                "iterations": ("INT", {"default": 5, "min": 1, "max": 20}),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    def estimate(self, image, fov=60.0, iterations=5, resolution=512):
        detector = dsine.DSINEDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution, fov=fov, iterations=iterations)
        del detector
        return (out,)


class SoftEdgeHED:
    """Estimate a soft-edge map from an IMAGE - HED (ControlNetHED_Apache2,
    a small VGG-like multi-scale CNN).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s HED
    preprocessor node (see vendor/hed.py). Weights auto-download from
    HuggingFace on first use into models/hed/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Soft Edge (HED) ⚡"
    SEARCH_ALIASES = ['soft edge', 'hed', 'edge detection', 'preprocessor',
                       'controlnet preprocessor', 'image to edge']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a soft-edge map from a photo (HED), for any "
                   "pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "safe": ("BOOLEAN", {"default": False, "tooltip": "Quantize the edge map "
                                     "into discrete levels - reduces noise, matches the "
                                     "source pack's 'safe' toggle."}),
            },
        }

    def estimate(self, image, resolution=512, safe=False):
        detector = hed.HEDDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution, safe=safe)
        del detector
        return (out,)


class SoftEdgePiDiNet:
    """Estimate a soft-edge map from an IMAGE - PiDiNet (Pixel Difference
    Convolution CNN).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s PiDiNet
    preprocessor node (see vendor/pidinet.py). NOTE: the original PiDiNet
    authors' LICENSE adds a research-use restriction beyond plain MIT -
    "commercial use should be contacted with authors first" - see
    vendor/pidinet.py's header for the exact text. Weights auto-download
    from HuggingFace on first use into models/pidinet/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Soft Edge (PiDiNet) ⚡"
    SEARCH_ALIASES = ['soft edge', 'pidinet', 'edge detection', 'preprocessor',
                       'controlnet preprocessor', 'image to edge']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a soft-edge map from a photo (PiDiNet), for any "
                   "pipeline's control_image/reference_image input. Research-use "
                   "license restriction - see node docstring.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "safe": ("BOOLEAN", {"default": False, "tooltip": "Quantize the edge map "
                                     "into discrete levels - reduces noise, matches the "
                                     "source pack's 'safe' toggle."}),
            },
        }

    def estimate(self, image, resolution=512, safe=False):
        detector = pidinet.PiDiNetDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution, safe=safe)
        del detector
        return (out,)


class MLSDLines:
    """Estimate straight line segments from an IMAGE - M-LSD (a MobileNetV2-
    based line-segment detector), rendered onto a black canvas.

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s M-LSD
    preprocessor node (see vendor/mlsd.py). Weights auto-download from
    HuggingFace on first use into models/mlsd/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "MLSD Lines ⚡"
    SEARCH_ALIASES = ['mlsd', 'line segment detection', 'straight lines', 'preprocessor',
                       'controlnet preprocessor', 'image to lines']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Detect straight line segments from a photo (M-LSD), for "
                   "any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "score_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01}),
                "dist_threshold": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 20.0, "step": 0.01}),
            },
        }

    def estimate(self, image, resolution=512, score_threshold=0.1, dist_threshold=0.1):
        detector = mlsd.MLSDDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution, score_threshold=score_threshold,
                              dist_threshold=dist_threshold)
        del detector
        return (out,)


class Lineart:
    """Estimate a realistic line drawing from an IMAGE - a ResNet
    encoder-decoder generator, fine or coarse checkpoint.

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s Lineart
    preprocessor node (see vendor/lineart.py). Weights auto-download from
    HuggingFace on first use into models/lineart/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Lineart ⚡"
    SEARCH_ALIASES = ['lineart', 'line drawing', 'preprocessor',
                       'controlnet preprocessor', 'image to lineart']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a realistic line drawing from a photo, for any "
                   "pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "coarse": ("BOOLEAN", {"default": False, "tooltip": "Use the coarser of the "
                                       "two checkpoints - fewer, bolder lines."}),
            },
        }

    def estimate(self, image, resolution=512, coarse=False):
        detector = lineart.LineartDetector(coarse=coarse).to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution)
        del detector
        return (out,)


class LineartAnime:
    """Estimate an anime-style line drawing from an IMAGE - a pix2pix-style
    U-Net generator.

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s Lineart
    Anime preprocessor node (see vendor/lineart_anime.py). Weights
    auto-download from HuggingFace on first use into models/lineart_anime/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Lineart (Anime) ⚡"
    SEARCH_ALIASES = ['lineart anime', 'anime line drawing', 'preprocessor',
                       'controlnet preprocessor', 'image to lineart']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate an anime-style line drawing from a photo, for "
                   "any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    def estimate(self, image, resolution=512):
        detector = lineart_anime.LineartAnimeDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution)
        del detector
        return (out,)


class MangaLine:
    """Estimate a manga-style clean line extraction from an IMAGE - the
    `res_skip` CNN (MangaLineExtraction port).

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s Manga Line
    preprocessor node (see vendor/manga_line.py). Weights auto-download from
    HuggingFace on first use into models/manga_line/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "Manga Line ⚡"
    SEARCH_ALIASES = ['manga line', 'line extraction', 'preprocessor',
                       'controlnet preprocessor', 'image to lineart']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate a manga-style clean line extraction from a photo, "
                   "for any pipeline's control_image/reference_image input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            },
        }

    def estimate(self, image, resolution=512):
        detector = manga_line.MangaLineDetector().to(comfy.model_management.get_torch_device())
        out = _estimate_batch(detector, image, resolution)
        del detector
        return (out,)


class OpenPose:
    """Estimate body/hand/face keypoints from an IMAGE, rendered as an
    OpenPose-style skeleton - the classic (pre-DWPose) three-CNN detector.

    Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s OpenPose
    preprocessor node (see vendor/openpose.py). NOTE: the underlying body/
    hand/face architecture and checkpoints trace back to Carnegie Mellon
    University's OpenPose license - ACADEMIC OR NON-PROFIT ORGANIZATION
    NONCOMMERCIAL RESEARCH USE ONLY. See vendor/openpose.py's header for
    the full text. Weights auto-download from HuggingFace on first use
    into models/openpose/.
    """

    CATEGORY = PREPROCESSORS_CATEGORY
    TITLE = "OpenPose ⚡"
    SEARCH_ALIASES = ['openpose', 'pose estimation', 'body pose', 'hand pose',
                       'face pose', 'preprocessor', 'controlnet preprocessor',
                       'image to pose']
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate"
    DESCRIPTION = ("Estimate body/hand/face keypoints from a photo, rendered as "
                   "an OpenPose skeleton, for any pipeline's control_image/"
                   "reference_image input. Non-commercial research-use license - "
                   "see node docstring.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "detect_body": ("BOOLEAN", {"default": True}),
                "detect_hand": ("BOOLEAN", {"default": True}),
                "detect_face": ("BOOLEAN", {"default": True}),
            },
        }

    def estimate(self, image, resolution=512, detect_body=True, detect_hand=True, detect_face=True):
        detector = openpose.OpenPoseDetector().to(comfy.model_management.get_torch_device())
        out = None
        for i in range(image.shape[0]):
            np_image = (image[i].cpu().numpy() * 255.0).astype(np.uint8)
            canvas, _pose_dict = detector.estimate(
                np_image, resolution=resolution, include_body=detect_body,
                include_hand=detect_hand, include_face=detect_face)
            canvas_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)
            if out is None:
                out = torch.zeros(image.shape[0], *canvas_tensor.shape, dtype=torch.float32)
            out[i] = canvas_tensor
        del detector
        return (out,)


class Canny:
    """Canny edge detection - plain cv2.Canny, no model, no download.

    Feed a source photo in, get an edge-map IMAGE out - the control_image
    a canny DiffSynth/Union/Fun Qwen-Image checkpoint, a Krea2 in-context
    canny LoRA, or Flux Klein's control_mode="auto_canny" expect.
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


NODE_CLASS_MAPPINGS = {
    "DepthMap": DepthMap,
    "NormalMapBAE": NormalMapBAE,
    "NormalMapDSINE": NormalMapDSINE,
    "SoftEdgeHED": SoftEdgeHED,
    "SoftEdgePiDiNet": SoftEdgePiDiNet,
    "MLSDLines": MLSDLines,
    "Lineart": Lineart,
    "LineartAnime": LineartAnime,
    "MangaLine": MangaLine,
    "OpenPose": OpenPose,
    "Canny": Canny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthMap": DepthMap.TITLE,
    "NormalMapBAE": NormalMapBAE.TITLE,
    "NormalMapDSINE": NormalMapDSINE.TITLE,
    "SoftEdgeHED": SoftEdgeHED.TITLE,
    "SoftEdgePiDiNet": SoftEdgePiDiNet.TITLE,
    "MLSDLines": MLSDLines.TITLE,
    "Lineart": Lineart.TITLE,
    "LineartAnime": LineartAnime.TITLE,
    "MangaLine": MangaLine.TITLE,
    "OpenPose": OpenPose.TITLE,
    "Canny": Canny.TITLE,
}
