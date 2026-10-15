"""Image conditioning components.

The pure-PyTorch DINOv2 image encoder lives under this package so it can share
the ``core.safetensors`` loader with the rest of the TripoSG stack without
pulling Transformers into production inference.
"""

from .config import DINOv2Config
from .embeddings import DINOv2Embeddings, DINOv2PatchEmbeddings
from .encoder import DINOv2Encoder, DINOv2EncoderOutput
from .image_processor import BitImageProcessor, ImageProcessorOutput
from .layer import DINOv2DropPath, DINOv2Layer, DINOv2LayerScale
from .mlp import DINOv2MLP
from .model import DINOv2Model, DINOv2ModelOutput

__all__ = [
    "BitImageProcessor",
    "DINOv2Config",
    "DINOv2DropPath",
    "DINOv2Embeddings",
    "DINOv2Encoder",
    "DINOv2EncoderOutput",
    "DINOv2Layer",
    "DINOv2LayerScale",
    "DINOv2MLP",
    "DINOv2Model",
    "DINOv2ModelOutput",
    "DINOv2PatchEmbeddings",
    "ImageProcessorOutput",
]
