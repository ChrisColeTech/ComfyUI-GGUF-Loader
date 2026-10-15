"""Output dataclass for the TripoSG VAE decoder.

A lightweight stand-in for
``diffusers.models.autoencoders.vae.DecoderOutput`` carrying only the
``sample`` field used by the decoder-only inference path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DecoderOutput:
    """Output of :class:`TripoSGVAEModel.decode` with a ``sample`` field."""

    sample: torch.Tensor
