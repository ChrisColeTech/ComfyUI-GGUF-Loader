# MIT. Vendored from the owner's giga-models sidecar:
#   pipelines/trellis2_v1/trellis2/pipelines/trellis2_image_to_3d.py  preprocess_image
#   pipelines/pixal3d_v1/pixal3d/pipelines/pixal3d_image_to_3d.py     preprocess_image
"""The sidecar's image preprocessing, for a ComfyUI IMAGE + foreground MASK.

Both pipelines run the same steps and differ only in the crop margin and the
final quantization:

1. cap the longest side at 1024 (LANCZOS);
2. the foreground alpha is the image's own when it is not fully opaque,
   otherwise BiRefNet's mask (in ComfyUI: core Remove Background, run before
   this - the sidecar ran it after the resize; BiRefNet resizes to 1024
   internally either way);
3. bounding box of alpha > 0.8 * 255, centred square of ``size * pad``
   (TRELLIS.2 1.0, Pixal3D 1.1) cropped with PIL (out-of-frame = transparent);
4. premultiplied composite on black (TRELLIS.2 ``(x * 255).astype(uint8)``;
   Pixal3D ``rgb * a + bg * (1 - a)``, clipped first).

The functions take/return PIL images so the arithmetic is the sidecar's own.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PAD = {"trellis2": 1.0, "pixal3d": 1.1}


def to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """float [H,W,3] in 0..1 + float alpha [H,W] in 0..1 -> RGBA PIL (uint8)."""
    rgb8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    a8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb8, a8]), mode="RGBA")


def preprocess_image(input: Image.Image, mode: str = "trellis2",
                     bg_color=(0, 0, 0)) -> Image.Image:
    """RGBA in (alpha = foreground) -> the square composite the DINOv3 stage sees."""
    if mode not in PAD:
        raise ValueError(f"mode must be one of {sorted(PAD)}")
    max_size = max(input.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        input = input.resize((int(input.width * scale), int(input.height * scale)),
                             Image.Resampling.LANCZOS)
    output = input
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if bbox.size == 0:
        raise ValueError("no foreground: the mask has no pixel above 0.8")
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * PAD[mode])
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = np.array(output).astype(np.float32) / 255
    if mode == "trellis2":
        output = output[:, :, :3] * output[:, :, 3:4]
        return Image.fromarray((output * 255).astype(np.uint8))
    rgb = output[:, :, :3]
    a = output[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    output = rgb * a + bg * (1.0 - a)
    return Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
