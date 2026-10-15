"""DINOv2 encoder (pure PyTorch).

Stack of ``DINOv2Layer`` blocks producing the final hidden state of the
sequence. Optional ``output_attentions`` and ``output_hidden_states`` are
supported for parity diagnostics; TripoSG only consumes the final hidden
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .config import DINOv2Config
from .layer import DINOv2Layer


@dataclass
class DINOv2EncoderOutput:
    """Plain-PyTorch equivalent of ``transformers.modeling_outputs.BaseModelOutput``."""

    last_hidden_state: torch.Tensor
    hidden_states: Optional[tuple[torch.Tensor, ...]] = None
    attentions: Optional[tuple[torch.Tensor, ...]] = None

    def __getitem__(self, index: int) -> torch.Tensor:
        if index == 0:
            return self.last_hidden_state
        if index == 1:
            return self.hidden_states
        if index == 2:
            return self.attentions
        raise IndexError(f"DINOv2EncoderOutput has no element at index {index}")


class DINOv2Encoder(nn.Module):
    """Stack of pre-norm DINOv2 transformer layers."""

    def __init__(self, config: DINOv2Config) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [DINOv2Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[list[Optional[torch.Tensor]]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | DINOv2EncoderOutput:
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        all_self_attentions: tuple[torch.Tensor, ...] | None = () if output_attentions else None

        for index, layer_module in enumerate(self.layer):
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[index] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]
            if output_attentions:
                assert all_self_attentions is not None
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, all_hidden_states, all_self_attentions)
                if value is not None
            )
        return DINOv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _gradient_checkpointing_func(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Gradient checkpointing is disabled for inference; the field exists for parity only"
        )
