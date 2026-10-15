"""Local image preprocessing compatible with the TripoSG BiT configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


@dataclass(slots=True)
class ImageProcessorOutput:
    pixel_values: torch.Tensor | np.ndarray


class BitImageProcessor:
    """Small local replacement for the Transformers BiT image processor."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        *,
        do_resize: bool = True,
        size: dict[str, int] | int = 224,
        resample: int = Image.Resampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: dict[str, int] | int = 224,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: list[float] | float = (0.48145466, 0.4578275, 0.40821073),
        image_std: list[float] | float = (0.26862954, 0.26130258, 0.27577711),
        do_convert_rgb: bool = True,
    ) -> None:
        self.do_resize = bool(do_resize)
        self.size = self._size_dict(size, shortest_edge=True)
        self.resample = int(resample)
        self.do_center_crop = bool(do_center_crop)
        self.crop_size = self._size_dict(crop_size, shortest_edge=False)
        self.do_rescale = bool(do_rescale)
        self.rescale_factor = float(rescale_factor)
        self.do_normalize = bool(do_normalize)
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = bool(do_convert_rgb)
        self._validate_config()

    @staticmethod
    def _size_dict(value: dict[str, int] | int, *, shortest_edge: bool) -> dict[str, int]:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("image sizes must be positive")
            return {"shortest_edge": value} if shortest_edge else {"height": value, "width": value}
        if not isinstance(value, dict):
            raise TypeError("image size must be an integer or dictionary")
        result = {str(key): int(item) for key, item in value.items()}
        expected = {"shortest_edge"} if shortest_edge else {"height", "width"}
        if set(result) != expected or any(item <= 0 for item in result.values()):
            raise ValueError(f"image size must contain positive {sorted(expected)} values")
        return result

    def _validate_config(self) -> None:
        if self.resample not in set(Image.Resampling):
            raise ValueError(f"unsupported Pillow resampling filter: {self.resample}")
        if not np.isfinite(self.rescale_factor):
            raise ValueError("rescale_factor must be finite")
        for name, values in (("image_mean", self.image_mean), ("image_std", self.image_std)):
            array = np.asarray(values, dtype=np.float32)
            if array.ndim > 1 or array.size not in (1, 3) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain one or three finite values")
            if name == "image_std" and np.any(array == 0):
                raise ValueError("image_std values must be nonzero")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        local_files_only: bool = True,
        subfolder: str | None = None,
        **overrides: Any,
    ) -> "BitImageProcessor":
        if local_files_only is not True:
            raise ValueError("BitImageProcessor supports local_files_only=True only")
        directory = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            candidate = Path(subfolder)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("subfolder must stay within the local model directory")
            directory /= candidate
        config_path = directory / "preprocessor_config.json"
        if not directory.is_dir() or not config_path.is_file():
            raise FileNotFoundError(f"Local image processor config does not exist: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read image processor config: {config_path}") from exc
        config.pop("image_processor_type", None)
        config.update(overrides)
        return cls(**config)

    @staticmethod
    def _as_list(images: Any) -> list[Any]:
        if isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("images must not be empty")
            return list(images)
        if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
            return list(images)
        return [images]

    @staticmethod
    def _to_numpy(image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image)
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        if isinstance(image, np.ndarray):
            return image
        raise TypeError("images must be PIL images, NumPy arrays, Torch tensors, or batches of them")

    @staticmethod
    def _channels_last(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image[..., None]
        if image.ndim != 3:
            raise ValueError(f"each image must have two or three dimensions, got {image.ndim}")
        first_is_channel = image.shape[0] in (1, 3, 4)
        last_is_channel = image.shape[-1] in (1, 3, 4)
        if first_is_channel:
            return np.transpose(image, (1, 2, 0))
        if last_is_channel:
            return image
        raise ValueError(f"could not infer the channel dimension from shape {image.shape}")

    @staticmethod
    def _rgb(image: np.ndarray) -> np.ndarray:
        channels = image.shape[-1]
        if channels == 3:
            return image
        if channels == 1:
            return np.repeat(image, 3, axis=-1)
        if channels == 4:
            return image[..., :3]
        raise ValueError(f"RGB conversion requires one, three, or four channels, got {channels}")

    @staticmethod
    def _pil_resize(image: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
        restore_scale = False
        if image.dtype != np.uint8:
            if np.allclose(image, image.astype(int)) and np.all((image >= 0) & (image <= 255)):
                pass
            elif np.all((image >= 0) & (image <= 1)):
                image = image * 255
                restore_scale = True
            else:
                raise ValueError("non-uint8 image values must be in [0, 1] or integral values in [0, 255]")
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized = np.asarray(pil_image.resize((size[1], size[0]), resample=resample))
        if resized.ndim == 2:
            resized = resized[..., None]
        return resized.astype(np.float32) / 255 if restore_scale else resized

    def _preprocess_one(self, value: Any) -> np.ndarray:
        if isinstance(value, Image.Image) and self.do_convert_rgb:
            value = value.convert("RGB")
        image = self._channels_last(self._to_numpy(value))
        if self.do_convert_rgb:
            image = self._rgb(image)

        if self.do_resize:
            height, width = image.shape[:2]
            short = min(height, width)
            target = self.size["shortest_edge"]
            output_size = (target, int(target * width / height)) if height <= width else (int(target * height / width), target)
            image = self._pil_resize(image, output_size, self.resample)

        if self.do_center_crop:
            crop_height = self.crop_size["height"]
            crop_width = self.crop_size["width"]
            height, width = image.shape[:2]
            if height < crop_height or width < crop_width:
                padded = np.zeros((max(height, crop_height), max(width, crop_width), image.shape[-1]), dtype=image.dtype)
                top_pad = (padded.shape[0] - height + 1) // 2
                left_pad = (padded.shape[1] - width + 1) // 2
                padded[top_pad : top_pad + height, left_pad : left_pad + width] = image
                image = padded
                height, width = image.shape[:2]
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
            image = image[top : top + crop_height, left : left + crop_width]

        if self.do_rescale:
            image = (image * self.rescale_factor).astype(np.float32)
        elif not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float32)
        if self.do_normalize:
            mean = np.asarray(self.image_mean, dtype=image.dtype)
            std = np.asarray(self.image_std, dtype=image.dtype)
            image = (image - mean) / std
        return np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)

    def __call__(self, images: Any, *, return_tensors: str | None = None) -> ImageProcessorOutput:
        arrays = [self._preprocess_one(image) for image in self._as_list(images)]
        try:
            batch = np.stack(arrays).astype(np.float32, copy=False)
        except ValueError as exc:
            raise ValueError("all processed images in a batch must have the same shape") from exc
        if return_tensors in (None, "pt"):
            return ImageProcessorOutput(torch.from_numpy(np.ascontiguousarray(batch)))
        if return_tensors == "np":
            return ImageProcessorOutput(batch)
        raise ValueError("return_tensors must be None, 'pt', or 'np'")
