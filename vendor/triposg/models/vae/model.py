"""Pure-PyTorch TripoSG VAE model (decoder-only inference path).

Reproduces the inference behavior of
``triposg.models.autoencoders.autoencoder_kl_triposg.TripoSGVAEModel.decode``
without importing diffusers. The encoder, point-cloud FPS sampler, and
``quant`` projection are intentionally omitted because production inference
only ever consumes already-denoised latents and turns them into occupancy
logits.

State-dict keys on the retained modules (``embedder.frequencies`` is a
non-persistent buffer, ``decoder.*``, ``post_quant.*``) match the upstream
checkpoint so a strict load of the decoder subset succeeds. Encoder and
``quant`` keys are filtered out by :func:`load_vae` and reported as
intentionally unused.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import nn

from .config import TripoSGVAEConfig
from .decoder import TripoSGDecoder, build_vae_embedder
from .output import DecoderOutput


class TripoSGVAEModel(nn.Module):
    """Decoder-only TripoSG VAE used for occupancy-logit inference."""

    def __init__(self, config: TripoSGVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.out_channels = 1

        self.embedder = build_vae_embedder(config)

        self.decoder = TripoSGDecoder(
            in_channels=self.embedder.out_dim,
            out_channels=self.out_channels,
            dim=config.width_decoder,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers_decoder,
        )

        # The encoder side (encoder.*, quant.*) is intentionally absent on the
        # pure path: production inference never re-encodes point clouds.
        self.post_quant = nn.Linear(config.latent_channels, config.width_decoder, bias=True)

    def set_flash_decoder(self) -> None:
        """Flash decoding is not supported by the pure-PyTorch VAE.

        The upstream flash path (``FlashTripoSGAttnProcessor2_0`` + ``diso``)
        is intentionally out of scope. The wrapper blocks flash unless a real
        ``diso`` install is present, so reaching this method indicates a
        misconfiguration.
        """
        raise NotImplementedError(
            "TripoSGVAEModel (pure) does not implement flash decoding; "
            "use the hierarchical decoder instead"
        )

    def _decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        num_chunks: int = 50000,
        to_cpu: bool = False,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """Decode latents into occupancy logits at ``sampled_points``.

        Args:
            z: Latent tensor of shape ``(B, num_latents, latent_channels)``.
            sampled_points: Query points of shape ``(B, num_queries, in_channels)``.
            num_chunks: Maximum number of queries processed by the
                cross-attention head in a single forward pass.
            to_cpu: When ``True``, materialize each chunk on CPU (mirrors
                upstream's optional memory optimization).
            return_dict: When ``True``, return :class:`DecoderOutput`.
        """
        xyz_samples = sampled_points

        z = self.post_quant(z)

        num_points = xyz_samples.shape[1]
        kv_cache = None
        dec: list[torch.Tensor] = []

        for start in range(0, num_points, num_chunks):
            queries = xyz_samples[:, start : start + num_chunks, :].to(
                z.device, dtype=z.dtype
            )
            queries = self.embedder(queries)

            chunk_logits, kv_cache = self.decoder(z, queries, kv_cache)
            dec.append(chunk_logits.cpu() if to_cpu else chunk_logits)

        sample = torch.cat(dec, dim=1)

        if not return_dict:
            return (sample,)
        return DecoderOutput(sample=sample)

    def decode(
        self,
        z: torch.Tensor,
        sampled_points: torch.Tensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """Public decoder entry point matching upstream's call signature."""
        decoded = self._decode(z, sampled_points, return_dict=return_dict, **kwargs)
        if isinstance(decoded, DecoderOutput):
            return decoded
        return (decoded[0],) if not return_dict else DecoderOutput(sample=decoded[0])

    def forward(self, x: torch.Tensor):  # pragma: no cover - parity stub
        raise NotImplementedError(
            "TripoSGVAEModel (pure) is decoder-only; call .decode(z, sampled_points)"
        )
