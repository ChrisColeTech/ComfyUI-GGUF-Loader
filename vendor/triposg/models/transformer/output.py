"""Output dataclass for the TripoSG DiT transformer.

A lightweight stand-in for
``diffusers.models.transformers.modeling_outputs.Transformer1DModelOutput``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Transformer1DModelOutput:
    """Output of :class:`TripoSGDiTModel` with a single ``sample`` field."""

    sample: torch.Tensor
