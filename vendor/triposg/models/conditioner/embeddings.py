"""DINOv2 CLS, mask, patch, and position embeddings (pure PyTorch).

Mathematically equivalent to ``transformers.models.dinov2.modeling_dinov2.Dinov2Embeddings``
for TripoSG inference, including bicubic position-encoding interpolation when
the input resolution differs from the trained grid.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .config import DINOv2Config


class DINOv2PatchEmbeddings(nn.Module):
    """Conv2d patch projection that produces ``(B, num_patches, hidden)`` tokens."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        image_height, image_width = config.image_size_tuple
        patch_height, patch_width = config.patch_size_tuple
        self.image_size = (image_height, image_width)
        self.patch_size = (patch_height, patch_width)
        self.num_channels = config.num_channels
        self.num_patches = (image_width // patch_width) * (image_height // patch_height)
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the "
                f"configuration. Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class DINOv2Embeddings(nn.Module):
    """Construct the CLS token, mask token, position and patch embeddings."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.patch_embeddings = DINOv2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size_tuple[0]
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Bicubic interpolation of the trained position embedding grid.

        Mirrors the upstream implementation exactly: when ``num_patches`` and
        the spatial size match (and we are not tracing), the cached parameter
        is returned unchanged. Otherwise the patch-position grid is reshaped
        to its trained square, up/downsampled with bicubic interpolation in
        float32, and concatenated back with the CLS position.
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(torch.float32),
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1),
                self.mask_token.to(embeddings.dtype).unsqueeze(0),
                embeddings,
            )

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        embeddings = self.dropout(embeddings)
        return embeddings
