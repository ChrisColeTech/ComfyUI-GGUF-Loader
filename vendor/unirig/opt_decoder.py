"""Pure-torch OPT decoder (transformers OPTForCausalLM replacement).

Faithful reimplementation of the OPT-350m-config decoder UniRig's skeleton AR
model uses, consuming the SAME state dict (``model.transformer.*`` keys in
skeleton.safetensors). Matches transformers' OPT exactly:

- learned positional embeddings with the historical +2 index offset
  (embedding table is max_position_embeddings + 2 rows);
- pre-LN blocks (do_layer_norm_before=True): x += attn(LN(x)); x += ffn(LN(x));
- query scaled by head_dim**-0.5; ReLU FFN; final_layer_norm before lm_head;
- KV cache for incremental decoding.

No dropout paths (inference only). Parity: see tests/test_opt_decoder.py —
logits match transformers to float tolerance and greedy decode is identical.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

_POS_OFFSET = 2  # OPTLearnedPositionalEmbedding historical offset


class KVCache:
    """Preallocated per-layer KV cache.

    One (B, H, max_len, D) buffer pair per layer, written in place — no
    per-step torch.cat (which reallocates the whole cache every token and,
    with 15 beams x a ~1k-token mesh prompt, transiently doubles ~13 GB and
    spills VRAM into WDDM paging: the decode-speed cliff of PhaseB.16).
    """

    def __init__(self, num_layers: int, batch: int, heads: int, max_len: int,
                 head_dim: int, device, dtype):
        self.k = [torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(num_layers)]
        self.v = [torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
                  for _ in range(num_layers)]
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l = k.shape[0], k.shape[2]  # b may be < cache batch during prefill
        self.k[layer][:b, :, self.length : self.length + l] = k
        self.v[layer][:b, :, self.length : self.length + l] = v
        end = self.length + l
        return self.k[layer][:b, :, :end].to(k.dtype), self.v[layer][:b, :, :end].to(v.dtype)

    def advance(self, l: int) -> None:
        self.length += l

    def reorder(self, beam_idx: torch.Tensor) -> None:
        """In-place beam reorder of the filled prefix."""
        end = self.length
        for layer in range(len(self.k)):
            self.k[layer][:, :, :end] = self.k[layer][beam_idx, :, :end]
            self.v[layer][:, :, :end] = self.v[layer][beam_idx, :, :end]

    def expand_beams(self, k_beams: int) -> None:
        """Broadcast beam 0's prefix into all beam rows (post-prefill)."""
        end = self.length
        for layer in range(len(self.k)):
            self.k[layer][1:k_beams, :, :end] = self.k[layer][0:1, :, :end]
            self.v[layer][1:k_beams, :, :end] = self.v[layer][0:1, :, :end]


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, x, cache: Optional[KVCache] = None, layer: int = 0):
        b, l, _ = x.shape
        q = (self.q_proj(x) * self.scaling).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.append(layer, k, v)

        # q is pre-scaled (matching HF), so scale=1.0 here. is_causal only
        # applies to the square prefill case; with a cache the single query
        # attends to everything.
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=(l > 1), scale=1.0)
        attn = attn.transpose(1, 2).reshape(b, l, -1)
        return self.out_proj(attn)


class _Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        self.self_attn = _Attention(hidden, heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, ffn)
        self.fc2 = nn.Linear(ffn, hidden)
        self.final_layer_norm = nn.LayerNorm(hidden)

    def forward(self, x, cache: Optional[KVCache] = None, layer: int = 0):
        x = x + self.self_attn(self.self_attn_layer_norm(x), cache, layer)
        x = x + self.fc2(F.relu(self.fc1(self.final_layer_norm(x))))
        return x


class OPTDecoderLM(nn.Module):
    """OPT causal LM head model; state-dict-compatible with transformers."""

    def __init__(self, vocab_size: int, hidden: int = 1024, num_layers: int = 24,
                 heads: int = 16, ffn: int = 4096, max_positions: int = 3076):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.embed_positions = nn.Embedding(max_positions + _POS_OFFSET, hidden)
        self.layers = nn.ModuleList(_Layer(hidden, heads, ffn) for _ in range(num_layers))
        self.final_layer_norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    # -- weights -----------------------------------------------------------

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor], prefix: str = "model.transformer.") -> "OPTDecoderLM":
        """Build from a transformers-layout state dict (e.g. the skeleton ckpt)."""
        def t(name):
            return state[prefix + name]

        vocab, hidden = t("model.decoder.embed_tokens.weight").shape
        max_pos = t("model.decoder.embed_positions.weight").shape[0] - _POS_OFFSET
        num_layers = 1 + max(
            int(k[len(prefix) + len("model.decoder.layers."):].split(".")[0])
            for k in state
            if k.startswith(prefix + "model.decoder.layers.")
        )
        ffn = t("model.decoder.layers.0.fc1.weight").shape[0]
        # heads count is not in the weights; OPT-350m uses 16 and head_dim=64.
        model = cls(vocab, hidden, num_layers, heads=16, ffn=ffn, max_positions=max_pos)

        remap = {"embed_tokens": "model.decoder.embed_tokens",
                 "embed_positions": "model.decoder.embed_positions",
                 "final_layer_norm": "model.decoder.final_layer_norm",
                 "lm_head": "lm_head"}
        own = {}
        for name, _ in model.named_parameters():
            top = name.split(".")[0]
            if top == "layers":
                src = prefix + "model.decoder." + name
            else:
                src = prefix + remap[top] + name[len(top):]
            own[name] = state[src]
        model.load_state_dict(own)
        model.eval()
        return model

    # -- forward -----------------------------------------------------------

    def make_cache(self, batch: int, max_len: int, device, dtype=torch.float32) -> KVCache:
        heads = self.layers[0].self_attn.heads
        head_dim = self.layers[0].self_attn.head_dim
        return KVCache(len(self.layers), batch, heads, max_len, head_dim, device, dtype)

    @torch.no_grad()
    def forward(self, inputs_embeds: torch.Tensor, cache: Optional[KVCache] = None):
        """Returns logits (B, L, V). ``cache`` (see :meth:`make_cache`) is
        appended to in place and its length advanced.

        Args:
            inputs_embeds: (B, L, H) embedded inputs (mesh condition + token
                embeddings — UniRig always drives OPT via inputs_embeds).
        """
        b, l, _ = inputs_embeds.shape
        past_length = cache.length if cache is not None else 0
        positions = torch.arange(past_length, past_length + l, device=inputs_embeds.device)
        x = inputs_embeds + self.embed_positions(positions + _POS_OFFSET)[None]

        for i, layer in enumerate(self.layers):
            x = layer(x, cache, i)
        if cache is not None:
            cache.advance(l)
        x = self.final_layer_norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_ids)
