"""Pure-PyTorch TripoSG VAE decoder.

Reproduces the inference behavior of
``triposg.models.autoencoders.autoencoder_kl_triposg.TripoSGDecoder`` without
importing diffusers. State-dict keys are identical to the upstream checkpoint
so a strict load of the VAE ``diffusion_pytorch_model.safetensors`` succeeds
on the decoder subset.

The decoder is composed of ``num_layers_decoder`` self-attention
:class:`DiTBlock` instances followed by a single cross-attention
:class:`DiTBlock` that reads query positions and produces occupancy logits.
Self-attention blocks run once over the (post-quantized) latent tokens to
produce a key/value cache, which is then reused while chunking arbitrarily
many query points through the cross-attention head.

The shared :class:`DiTBlock` is imported from the transformer port (no
duplication). VAE-specific differences are pure configuration: ``qk_norm``
off, ``qkv_bias`` off, no long skip connections, and the cross-attention
block uses ``cross_attention_norm="layer_norm"`` (matching the
``attn2.norm_cross.*`` checkpoint keys).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ..transformer.blocks import DiTBlock
from ..transformer.normalization import LayerNorm
from .embeddings import FrequencyPositionalEmbedding


class TripoSGDecoder(nn.Module):
    """Self-attention stack + cross-attention head for occupancy logits."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dim: int = 1024,
        num_attention_heads: int = 8,
        num_layers: int = 16,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    activation_fn="gelu",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
            ]
        )

        # Query projection lifts frequency-embedded xyz (dim = embedder.out_dim)
        # into the model width.
        self.proj_query = nn.Linear(in_channels, dim, bias=True)

        self.norm_out = LayerNorm(dim)
        self.proj_out = nn.Linear(dim, out_channels, bias=True)

    def forward(
        self,
        sample: torch.Tensor,
        queries: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the self-attention stack once and the cross-attention head per chunk.

        Args:
            sample: Post-quantized latent tokens of shape ``(B, num_latents, dim)``.
            queries: Frequency-embedded query positions of shape
                ``(B, num_queries, embedder.out_dim)``.
            kv_cache: Optional key/value cache produced by a prior call. When
                ``None``, the self-attention stack runs once to compute it.

        Returns:
            ``(logits, kv_cache)`` where ``logits`` has shape
            ``(B, num_queries, out_channels)`` and ``kv_cache`` can be passed
                back in for subsequent chunks.
        """
        if kv_cache is None:
            hidden_states = sample
            for block in self.blocks[:-1]:
                hidden_states = block(hidden_states)
            kv_cache = hidden_states

        q = self.proj_query(queries)
        latent = self.blocks[-1](q, encoder_hidden_states=kv_cache)
        logits = self.proj_out(self.norm_out(latent))

        # Upstream negates logits at the output of the decoder; this matches
        # the sign convention used by ``hierarchical_extract_geometry`` which
        # treats ``grid_logits == 0`` as the surface isosurface.
        logits = logits * -1

        return logits, kv_cache


def build_vae_embedder(config: "TripoSGVAEConfig") -> FrequencyPositionalEmbedding:
    """Construct the upstream's ``FrequencyPositionalEmbedding`` from config.

    Mirrors ``TripoSGVAEModel.__init__``: ``include_pi`` comes from the config
    and ``include_input`` defaults to ``True`` upstream.
    """
    if config.embedding_type != "frequency":
        raise NotImplementedError(
            f"Embedding type {config.embedding_type!r} is not supported "
            "(only 'frequency')"
        )
    return FrequencyPositionalEmbedding(
        num_freqs=config.embed_frequency,
        logspace=True,
        input_dim=config.in_channels,
        include_input=True,
        include_pi=config.embed_include_pi,
    )
