"""DINOv2 image encoder for TripoSG conditioning (pure PyTorch).

This module reproduces the inference behavior of
``transformers.models.dinov2.modeling_dinov2.Dinov2Model`` for the TripoSG
image-conditioning path: it accepts a ``pixel_values`` tensor of shape
``(B, C, H, W)``, runs the embeddings + encoder + final LayerNorm, and
returns an object exposing ``last_hidden_state`` of shape
``(B, 1 + num_patches, hidden_size)``. State-dict keys are kept identical to
the upstream checkpoint so a strict load against
``image_encoder_dinov2/model.safetensors`` succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import DINOv2Config
from .embeddings import DINOv2Embeddings
from .encoder import DINOv2Encoder


@dataclass
class DINOv2ModelOutput:
    """Lightweight stand-in for ``BaseModelOutputWithPooling``."""

    last_hidden_state: torch.Tensor
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None

    def __getitem__(self, index: int) -> Optional[torch.Tensor]:
        sequence = (
            self.last_hidden_state,
            self.pooler_output,
            self.hidden_states,
            self.attentions,
        )
        return sequence[index]


class DINOv2Model(nn.Module):
    """Pure-PyTorch DINOv2 base transformer."""

    main_input_name = "pixel_values"

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = DINOv2Embeddings(config)
        self.encoder = DINOv2Encoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Reset all parameters using the same scheme as the upstream initializer so
        # random-init parity tests build a faithful reference state dict.
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        config = self.config
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, DINOv2Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=config.initializer_range,
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=config.initializer_range,
            ).to(module.cls_token.dtype)

    def get_input_embeddings(self) -> DINOv2Embeddings:
        return self.embeddings.patch_embeddings

    def get_head_mask(
        self,
        head_mask: Optional[torch.Tensor],
        num_hidden_layers: int,
    ) -> list[None]:
        # TripoSG never supplies a head mask; the upstream behavior with a
        # ``None`` input is to return ``[None] * num_hidden_layers``.
        if head_mask is not None:
            raise NotImplementedError("DINOv2Model head masks are not supported")
        return [None] * num_hidden_layers

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | DINOv2ModelOutput:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask_list = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask_list,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            sequence_output = encoder_outputs.last_hidden_state
            hidden_states = encoder_outputs.hidden_states
            attentions = encoder_outputs.attentions
        else:
            sequence_output = encoder_outputs[0]
            hidden_states = encoder_outputs[1] if len(encoder_outputs) > 1 else None
            attentions = encoder_outputs[2] if len(encoder_outputs) > 2 else None

        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            tail = tuple(
                value
                for value in (hidden_states, attentions)
                if value is not None
            )
            return head_outputs + tail

        return DINOv2ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=hidden_states,
            attentions=attentions,
        )
