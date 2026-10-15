"""Pure-PyTorch TripoSG VAE package (decoder-only inference path).

Reproduces the inference behavior of
``triposg.models.autoencoders.autoencoder_kl_triposg.TripoSGVAEModel.decode``
without importing diffusers. State-dict keys on the retained modules match
the upstream checkpoint so a strict load of the decoder subset of
``diffusion_pytorch_model.safetensors`` succeeds.
"""

from .config import TripoSGVAEConfig
from .decoder import TripoSGDecoder, build_vae_embedder
from .embeddings import FrequencyPositionalEmbedding
from .model import TripoSGVAEModel
from .output import DecoderOutput

__all__ = [
    "DecoderOutput",
    "FrequencyPositionalEmbedding",
    "TripoSGDecoder",
    "TripoSGVAEConfig",
    "TripoSGVAEModel",
    "build_vae_embedder",
]
