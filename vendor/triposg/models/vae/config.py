"""Pure-PyTorch TripoSG VAE configuration.

A small dataclass mirror of the subset of the upstream
``TripoSGVAEModel`` config fields that affect inference-time mathematics on
the decoder path. It is loaded from a local ``config.json`` and never
imports diffusers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TripoSGVAEConfig:
    """Inference hyperparameters for the local TripoSG VAE (decoder path)."""

    in_channels: int = 3
    latent_channels: int = 64
    num_attention_heads: int = 8
    width_encoder: int = 512
    width_decoder: int = 1024
    num_layers_encoder: int = 8
    num_layers_decoder: int = 16
    embedding_type: str = "frequency"
    embed_frequency: int = 8
    embed_include_pi: bool = False

    def __post_init__(self) -> None:
        if self.embedding_type != "frequency":
            raise NotImplementedError(
                f"Embedding type {self.embedding_type!r} is not supported "
                "(only 'frequency')"
            )
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.width_decoder % self.num_attention_heads != 0:
            raise ValueError(
                f"width_decoder ({self.width_decoder}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_layers_decoder < 1:
            raise ValueError("num_layers_decoder must be positive")
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")

    @property
    def embedder_out_dim(self) -> int:
        """Output dim of FrequencyPositionalEmbedding for the configured VAE."""
        return self.in_channels * (self.embed_frequency * 2 + 1)

    @property
    def decoder_head_dim(self) -> int:
        return self.width_decoder // self.num_attention_heads

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "TripoSGVAEConfig":
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
    ) -> "TripoSGVAEConfig":
        if local_files_only is not True:
            raise ValueError("TripoSGVAEConfig supports local_files_only=True only")
        directory = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            candidate = Path(subfolder)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("subfolder must stay within the local model directory")
            directory /= candidate
        config_path = directory / "config.json"
        if not directory.is_dir() or not config_path.is_file():
            raise FileNotFoundError(f"Local VAE config does not exist: {config_path}")
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read VAE config: {config_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"VAE config must be a JSON object: {config_path}")
        ignored = {"_class_name", "_diffusers_version"}
        for key in ignored:
            payload.pop(key, None)
        payload.update(overrides)
        return cls.from_dict(payload)
