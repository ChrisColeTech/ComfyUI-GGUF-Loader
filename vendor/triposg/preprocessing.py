from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError


def _open_image(image: str | os.PathLike[str] | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        try:
            return ImageOps.exif_transpose(image).copy()
        except Exception as exc:
            raise ValueError("Could not read the supplied PIL image") from exc

    if not isinstance(image, (str, os.PathLike)):
        raise TypeError("image must be a path or PIL.Image.Image")

    path = os.fspath(image)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")
    try:
        with Image.open(path) as opened:
            return ImageOps.exif_transpose(opened).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {path}") from exc


def _has_meaningful_alpha(alpha: np.ndarray, min_ratio: float = 0.01) -> bool:
    # Match the reference's first and last bins of a 20-bin alpha histogram.
    required = alpha.size * min_ratio
    return bool(
        np.count_nonzero(alpha < 256 / 20) >= required
        and np.count_nonzero(alpha >= 256 * 19 / 20) >= required
    )


def _otsu_mask(values: np.ndarray) -> np.ndarray:
    pixels = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    histogram = np.bincount(pixels.ravel(), minlength=256).astype(np.float64)
    total = pixels.size
    weighted_total = float(np.dot(np.arange(256), histogram))
    background_count = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 0

    for level, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        background_sum += level * count
        background_mean = background_sum / background_count
        foreground_mean = (weighted_total - background_sum) / foreground_count
        variance = background_count * foreground_count * (
            background_mean - foreground_mean
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = level

    return pixels > threshold


def _remove_small_components(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Remove small 4-connected components, retaining one tiny component if needed."""
    runs: list[tuple[int, int, int, int]] = []
    parents: list[int] = []
    sizes: list[int] = []
    previous: list[tuple[int, int, int]] = []

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, end in zip(starts, ends):
            label = len(parents)
            parents.append(label)
            sizes.append(int(end - start))
            while previous_index < len(previous) and previous[previous_index][1] <= start:
                previous_index += 1
            index = previous_index
            while index < len(previous) and previous[index][0] < end:
                union(label, previous[index][2])
                index += 1
            current.append((int(start), int(end), label))
            runs.append((y, int(start), int(end), label))
        previous = current

    if not runs:
        return np.zeros_like(mask, dtype=bool)

    component_sizes: dict[int, int] = {}
    for _, start, end, label in runs:
        root = find(label)
        component_sizes[root] = component_sizes.get(root, 0) + end - start
    kept = {root for root, size in component_sizes.items() if size >= min_size}
    if not kept:
        kept.add(max(component_sizes, key=component_sizes.get))

    cleaned = np.zeros_like(mask, dtype=bool)
    for y, start, end, label in runs:
        if find(label) in kept:
            cleaned[y, start:end] = True
    return cleaned


def _predict_alpha(
    rgb: np.ndarray, *, rmbg: object, device: str | torch.device
) -> np.ndarray:
    if rmbg is None or not callable(rmbg):
        raise ValueError("A callable RMBG model is required when no meaningful alpha exists")
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid torch device: {device!r}") from exc

    tensor = torch.from_numpy(rgb.copy()).to(
        device=target_device, dtype=torch.float32
    )
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    if float(tensor.max()) < 1e-3:
        raise ValueError("Invalid image: image is pure black")
    tensor = F.interpolate(
        tensor, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True
    )
    tensor = tensor - 0.5

    try:
        with torch.inference_mode():
            prediction = rmbg(tensor)
    except Exception as exc:
        raise RuntimeError(f"RMBG inference failed on device {target_device}") from exc

    while isinstance(prediction, (list, tuple)):
        if not prediction:
            raise ValueError("RMBG model returned no outputs")
        prediction = prediction[0]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("The first RMBG output must be a torch.Tensor")
    while prediction.ndim > 2:
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.numel() == 0:
        raise ValueError("The first RMBG output must contain a 2D mask")

    prediction = F.interpolate(
        prediction.float()[None, None],
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("RMBG model returned non-finite mask values")
    minimum, maximum = prediction.min(), prediction.max()
    if float(maximum - minimum) <= torch.finfo(prediction.dtype).eps:
        raise ValueError("RMBG model returned a constant mask")
    normalized = ((prediction - minimum) / (maximum - minimum)).cpu().numpy()
    return _remove_small_components(_otsu_mask(normalized)).astype(np.float32)


def _validate_background(background: Sequence[float]) -> np.ndarray:
    try:
        values = np.asarray(background, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("background must contain three numeric values") from exc
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("background must contain three finite RGB values")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("background RGB values must be between 0 and 1")
    return values


def prepare_image(
    image: str | os.PathLike[str] | Image.Image,
    *,
    rmbg: object,
    device: str | torch.device,
    background: Sequence[float] = (1, 1, 1),
    padding_ratio: float = 0.1,
) -> Image.Image:
    """Prepare an image for TripoSG and return a square RGB PIL image."""
    background_rgb = _validate_background(background)
    if not isinstance(padding_ratio, (int, float)) or not math.isfinite(padding_ratio):
        raise ValueError("padding_ratio must be a finite non-negative number")
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be a finite non-negative number")

    source = _open_image(image)
    if source.width <= 0 or source.height <= 0:
        raise ValueError("Image dimensions must be positive")
    if max(source.size) > 2000:
        scale = 2000.0 / max(source.size)
        size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
        source = source.resize(size, Image.Resampling.LANCZOS)

    if source.mode == "P" and "transparency" in source.info:
        source = source.convert("RGBA")
    bands = source.getbands()
    alpha_image = source.getchannel("A") if "A" in bands else None
    rgb_image = source.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)

    alpha: np.ndarray | None = None
    if alpha_image is not None:
        candidate = np.asarray(alpha_image, dtype=np.uint8)
        if _has_meaningful_alpha(candidate):
            alpha = candidate.astype(np.float32) / 255.0
    if alpha is None:
        alpha = _predict_alpha(rgb, rmbg=rmbg, device=device)

    foreground = alpha > 0
    rows, columns = np.where(foreground)
    if rows.size == 0:
        raise ValueError("Input image has no foreground after background removal")
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1

    alpha_3d = alpha[..., None]
    composite = rgb.astype(np.float32) / 255.0 * alpha_3d
    composite += background_rgb[None, None, :] * (1.0 - alpha_3d)
    crop = np.clip(composite[y0:y1, x0:x1] * 255.0, 0, 255).astype(np.uint8)

    width, height = x1 - x0, y1 - y0
    content_side = max(width, height)
    padding = int(content_side * float(padding_ratio))
    output_side = content_side + 2 * padding
    background_u8 = tuple(np.rint(background_rgb * 255).astype(np.uint8).tolist())
    result = Image.new("RGB", (output_side, output_side), background_u8)
    result.paste(
        Image.fromarray(crop, mode="RGB"),
        ((output_side - width) // 2, (output_side - height) // 2),
    )
    return result
