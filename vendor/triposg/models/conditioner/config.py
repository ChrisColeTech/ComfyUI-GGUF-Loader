"""Pure-PyTorch DINOv2 configuration.

This is a small dataclass mirror of the subset of
``transformers.models.dinov2.configuration_dinov2.Dinov2Config`` fields that
affect inference-time mathematics for the TripoSG image encoder. It is loaded
from a local ``config.json`` and never imports Transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping


def _as_int_iterable(value: Any, *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (str, float)):
        raise TypeError(f"{name} must be an int or iterable of ints, got {type(value).__name__}")
    if isinstance(value, Mapping):
        items = tuple(value.values())
    elif isinstance(value, Iterable):
        items = tuple(value)
    else:
        raise TypeError(f"{name} must be an int or iterable of ints, got {type(value).__name__}")
    if len(items) != 2 or any(not isinstance(item, int) for item in items):
        raise ValueError(f"{name} must contain exactly two integers, got {value!r}")
    return int(items[0]), int(items[1])


@dataclass
class DINOv2Config:
    """Inference hyperparameters for the local DINOv2 image encoder."""

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False

    def __post_init__(self) -> None:
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if any(side <= 0 for side in self.patch_size_tuple):
            raise ValueError("patch_size must be positive")
        if any(side <= 0 for side in self.image_size_tuple):
            raise ValueError("image_size must be positive")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.use_swiglu_ffn:
            raise ValueError(
                "use_swiglu_ffn=True is not supported by the local DINOv2 "
                "encoder (the TripoSG checkpoint uses the standard gelu MLP)"
            )

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def image_size_tuple(self) -> tuple[int, int]:
        return _as_int_iterable(self.image_size, name="image_size")

    @property
    def patch_size_tuple(self) -> tuple[int, int]:
        return _as_int_iterable(self.patch_size, name="patch_size")

    @property
    def num_patches(self) -> int:
        height, width = self.image_size_tuple
        patch_h, patch_w = self.patch_size_tuple
        return (height // patch_h) * (width // patch_w)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "DINOv2Config":
        names = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in mapping.items() if key in names}
        return cls(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        local_files_only: bool = True,
        subfolder: str | None = None,
        **overrides: Any,
    ) -> "DINOv2Config":
        if local_files_only is not True:
            raise ValueError("DINOv2Config supports local_files_only=True only")
        directory = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            candidate = Path(subfolder)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("subfolder must stay within the local model directory")
            directory /= candidate
        config_path = directory / "config.json"
        if not directory.is_dir() or not config_path.is_file():
            raise FileNotFoundError(f"Local DINOv2 config does not exist: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read DINOv2 config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"DINOv2 config must be a JSON object: {config_path}")
        ignored = {
            "model_type",
            "architectures",
            "_name_or_path",
            "torch_dtype",
            "transformers_version",
            "out_features",
            "out_indices",
            "stage_names",
            "apply_layernorm",
            "reshape_hidden_states",
        }
        for key in ignored:
            payload.pop(key, None)
        payload.update(overrides)
        return cls.from_dict(payload)
