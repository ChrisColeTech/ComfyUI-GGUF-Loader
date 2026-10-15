"""Pure-PyTorch TripoSG DiT transformer package.

Reproduces the inference behavior of
``triposg.models.transformers.triposg_transformer.TripoSGDiTModel`` without
importing diffusers. State-dict keys are identical to the upstream checkpoint
so a strict load of ``diffusion_pytorch_model.safetensors`` succeeds.
"""

from .attention import Attention, TripoSGAttnProcessor2_0
from .blocks import DiTBlock
from .config import TripoSGDiTConfig
from .embeddings import TimestepEmbedding, Timesteps
from .feedforward import FeedForward, GELU
from .model import TripoSGDiTModel
from .normalization import FP32LayerNorm, LayerNorm, RMSNorm
from .output import Transformer1DModelOutput

__all__ = [
    "Attention",
    "DiTBlock",
    "FeedForward",
    "FP32LayerNorm",
    "GELU",
    "LayerNorm",
    "RMSNorm",
    "TimestepEmbedding",
    "Timesteps",
    "Transformer1DModelOutput",
    "TripoSGAttnProcessor2_0",
    "TripoSGDiTConfig",
    "TripoSGDiTModel",
]
