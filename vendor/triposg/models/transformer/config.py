"""Pure-PyTorch TripoSG DiT configuration.

A small dataclass mirror of the subset of
``diffusers`` ``TripoSGDiTModel`` config fields that affect inference-time
mathematics. It is loaded from a local ``config.json`` and never imports
diffusers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TripoSGDiTConfig:
    """Inference hyperparameters for the local TripoSG DiT transformer."""

    num_attention_heads: int = 16
    width: int = 2048
    in_channels: int = 64
    num_layers: int = 21
    cross_attention_dim: int = 1024
    use_cross_attention_2: bool = False
    cross_attention_2_dim: int | None = None

    def __post_init__(self) -> None:
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.width % self.num_attention_heads != 0:
            raise ValueError(
                f"width ({self.width}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.cross_attention_dim <= 0:
            raise ValueError("cross_attention_dim must be positive")

    @property
    def head_dim(self) -> int:
        return self.width // self.num_attention_heads

    @property
    def ff_inner_dim(self) -> int:
        return int(self.width * 4.0)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "TripoSGDiTConfig":
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
    ) -> "TripoSGDiTConfig":
        if local_files_only is not True:
            raise ValueError("TripoSGDiTConfig supports local_files_only=True only")
        directory = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            candidate = Path(subfolder)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("subfolder must stay within the local model directory")
            directory /= candidate
        config_path = directory / "config.json"
        if not directory.is_dir() or not config_path.is_file():
            raise FileNotFoundError(f"Local DiT config does not exist: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read DiT config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"DiT config must be a JSON object: {config_path}")
        ignored = {"_class_name", "_diffusers_version", "model_type", "architectures"}
        for key in ignored:
            payload.pop(key, None)
        payload.update(overrides)
        return cls.from_dict(payload)
