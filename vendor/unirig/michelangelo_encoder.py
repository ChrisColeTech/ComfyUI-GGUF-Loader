"""
Michelangelo Perceiver encoder — ComfyUI-native.

Consolidates all Michelangelo model classes from the original
michelangelo/ subdirectory into a single flat file with
operations= support and ComfyUI attention dispatch.

Classes:
    FourierEmbedder, MLP, MultiheadAttention, QKVMultiheadAttention,
    MultiheadCrossAttention, QKVMultiheadCrossAttention,
    ResidualAttentionBlock, ResidualCrossAttentionBlock, Transformer,
    CrossAttentionEncoder, CrossAttentionDecoder,
    ShapeAsLatentModule, ShapeAsLatentPerceiver,
    AlignedShapeLatentPerceiver, ShapeAsLatentPerceiverEncoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from einops import repeat
import numpy as np

# CCTech: pure-torch farthest-point sampling instead of the torch_cluster extension.
from .native_fallbacks import fps
import logging

import comfy.ops
import comfy.utils

from .attention import comfy_attention

log = logging.getLogger("unirig")

# ============================================================================
# Embedders
# ============================================================================

class FourierEmbedder(nn.Module):
    """Sin/cosine positional embedding. No learnable weights."""

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device="cpu")
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32, device="cpu")
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device, dtype=x.dtype)).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

# ============================================================================
# Transformer building blocks
# ============================================================================

class MLP(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 width: int,
                 hidden_width_scale: int = 4,
                 init_scale: float = 1.0,
                 operations=None):
        super().__init__()
        self.width = width
        self.c_fc = operations.Linear(width, width * hidden_width_scale, device=device, dtype=dtype)
        self.c_proj = operations.Linear(width * hidden_width_scale, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    """Compute-only attention (no learnable weights). Uses ComfyUI dispatch."""
    def __init__(self, *, device=None, dtype=None, heads: int, n_ctx: int = 0, flash: bool = False):
        super().__init__()
        self.heads = heads

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        # q, k, v: [B, L, H, D] — use ComfyUI attention dispatch
        out = comfy_attention(q, k, v, self.heads)
        return out.reshape(bs, n_ctx, -1)


class MultiheadAttention(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 n_ctx: int,
                 width: int,
                 heads: int,
                 init_scale: float = 1.0,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 operations=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = operations.Linear(width, width * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = operations.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    """Compute-only cross-attention (no learnable weights). Uses ComfyUI dispatch."""
    def __init__(self, *, device=None, dtype=None, heads: int, flash: bool = False, n_data: Optional[int] = None):
        super().__init__()
        self.heads = heads

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)
        # q: [B, Lq, H, D], k,v: [B, Lkv, H, D]
        out = comfy_attention(q, k, v, self.heads)
        return out.reshape(bs, n_ctx, -1)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 width: int,
                 heads: int,
                 init_scale: float = 1.0,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 n_data: Optional[int] = None,
                 data_width: Optional[int] = None,
                 operations=None):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = operations.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.c_kv = operations.Linear(self.data_width, width * 2, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = operations.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadCrossAttention(
            device=device, dtype=dtype, heads=heads, n_data=n_data
        )

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 n_ctx: int,
                 width: int,
                 heads: int,
                 init_scale: float = 1.0,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False,
                 operations=None):
        super().__init__()
        self.attn = MultiheadAttention(
            device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, operations=operations,
        )
        self.ln_1 = operations.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale, operations=operations)
        self.ln_2 = operations.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 n_data: Optional[int] = None,
                 width: int,
                 heads: int,
                 data_width: Optional[int] = None,
                 mlp_width_scale: int = 4,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 operations=None):
        super().__init__()
        if data_width is None:
            data_width = width
        self.attn = MultiheadCrossAttention(
            device=device, dtype=dtype, n_data=n_data, width=width, heads=heads,
            data_width=data_width, init_scale=init_scale, qkv_bias=qkv_bias,
            flash=flash, operations=operations,
        )
        self.ln_1 = operations.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = operations.LayerNorm(data_width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width,
                       hidden_width_scale=mlp_width_scale, init_scale=init_scale,
                       operations=operations)
        self.ln_3 = operations.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class Transformer(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 n_ctx: int,
                 width: int,
                 layers: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False,
                 operations=None):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
                use_checkpoint=use_checkpoint, operations=operations,
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

# ============================================================================
# Perceiver encoder / decoder
# ============================================================================

class ShapeAsLatentModule(nn.Module):
    """Base class for shape-as-latent models."""
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()


class CrossAttentionEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 query_method: bool = False,
                 use_full_input: bool = True,
                 token_num: int = 256,
                 no_query: bool = False,
                 operations=None):
        super().__init__()
        self.query_method = query_method
        self.token_num = token_num
        self.use_full_input = use_full_input
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        if no_query:
            self.query = None
        else:
            self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = operations.Linear(self.fourier_embedder.out_dim + point_feats, width,
                                            device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, operations=operations,
        )
        self.self_attn = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width, layers=layers,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_checkpoint=False, operations=operations,
        )
        if use_ln_post:
            self.ln_post = operations.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        if self.query_method:
            token_num = self.num_latents
            bs = pc.shape[0]
            data = self.fourier_embedder(pc)
            if feats is not None:
                data = torch.cat([data, feats], dim=-1)
            data = self.input_proj(data)
            query = repeat(self.query.to(device=pc.device, dtype=pc.dtype), "m c -> b m c", b=bs)
            latents = self.cross_attn(query, data)
            latents = self.self_attn(latents)
            if self.ln_post is not None:
                latents = self.ln_post(latents)
            pre_pc = None
        else:
            if isinstance(self.token_num, int):
                token_num = self.token_num
            else:
                import random
                token_num = random.choice(self.token_num)

            rng = np.random.default_rng(seed=0)
            ind = rng.choice(pc.shape[1], token_num * 4, replace=token_num * 4 > pc.shape[1])
            pre_pc = pc[:, ind, :]
            pre_feats = feats[:, ind, :]

            B, N, D = pre_pc.shape
            C = pre_feats.shape[-1]
            pos = pre_pc.view(B * N, D)
            pos_feats = pre_feats.view(B * N, C)
            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)
            idx = fps(pos, batch, ratio=1. / 4, random_start=False)
            sampled_pc = pos[idx].view(B, -1, 3)
            sampled_feats = pos_feats[idx].view(B, -1, C)

            if self.use_full_input:
                data = self.fourier_embedder(pc)
            else:
                data = self.fourier_embedder(pre_pc)
            if feats is not None:
                if not self.use_full_input:
                    feats = pre_feats
                data = torch.cat([data, feats], dim=-1)
            data = self.input_proj(data)

            sampled_data = self.fourier_embedder(sampled_pc)
            if feats is not None:
                sampled_data = torch.cat([sampled_data, sampled_feats], dim=-1)
            sampled_data = self.input_proj(sampled_data)

            latents = self.cross_attn(sampled_data, data)
            latents = self.self_attn(latents)
            if self.ln_post is not None:
                latents = self.ln_post(latents)
            pre_pc = torch.cat([pre_pc, pre_feats], dim=-1)

        return latents, pc, token_num, pre_pc


class CrossAttentionDecoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False,
                 mlp_width_scale: int = 4,
                 supervision_type: str = 'occupancy',
                 operations=None):
        super().__init__()
        self.fourier_embedder = fourier_embedder
        self.supervision_type = supervision_type
        self.query_proj = operations.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, n_data=num_latents, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            mlp_width_scale=mlp_width_scale, operations=operations,
        )
        self.ln_post = operations.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = operations.Linear(width, out_channels, device=device, dtype=dtype)
        if self.supervision_type == 'occupancy-sdf':
            self.output_proj_sdf = operations.Linear(width, out_channels, device=device, dtype=dtype)

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x_1 = self.output_proj(x)
        if self.supervision_type == 'occupancy-sdf':
            x_2 = self.output_proj_sdf(x)
            return x_1, x_2
        return x_1

# ============================================================================
# Top-level perceiver models
# ============================================================================

class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 decoder_width: Optional[int] = None,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 query_method: bool = False,
                 token_num: int = 256,
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 use_full_input: bool = True,
                 freeze_encoder: bool = False,
                 decoder_mlp_width_scale: int = 4,
                 residual_kl: bool = False,
                 operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.supervision_type = supervision_type
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            num_latents=num_latents, point_feats=point_feats, width=width, heads=heads,
            layers=num_encoder_layers, init_scale=init_scale, qkv_bias=qkv_bias,
            flash=flash, use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            query_method=query_method, use_full_input=use_full_input, token_num=token_num,
            operations=operations,
        )

        self.embed_dim = embed_dim
        self.residual_kl = residual_kl
        if decoder_width is None:
            decoder_width = width
        if embed_dim > 0:
            self.pre_kl = operations.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = operations.Linear(embed_dim, decoder_width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
            if self.residual_kl:
                assert self.post_kl.out_features % self.post_kl.in_features == 0
                assert self.pre_kl.in_features % self.pre_kl.out_features == 0
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=decoder_width, layers=num_decoder_layers,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash,
            use_checkpoint=use_checkpoint, operations=operations,
        )

        self.geo_decoder = CrossAttentionDecoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder, out_channels=1,
            num_latents=num_latents, width=decoder_width, heads=heads, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint,
            supervision_type=supervision_type, mlp_width_scale=decoder_mlp_width_scale,
            operations=operations,
        )


class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[Union[torch.dtype, str]] = None,
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 decoder_width: Optional[int] = None,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 query_method: bool = False,
                 use_full_input: bool = True,
                 token_num: int = 256,
                 freeze_encoder: bool = False,
                 decoder_mlp_width_scale: int = 4,
                 residual_kl: bool = False,
                 operations=None):
        MAP_DTYPE = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        if dtype is not None and isinstance(dtype, str):
            dtype = MAP_DTYPE[dtype]
        super().__init__(
            device=device, dtype=dtype, num_latents=1 + num_latents,
            point_feats=point_feats, embed_dim=embed_dim, num_freqs=num_freqs,
            include_pi=include_pi, width=width, decoder_width=decoder_width,
            heads=heads, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, init_scale=init_scale,
            qkv_bias=qkv_bias, flash=flash, use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint, supervision_type=supervision_type,
            grad_type=grad_type, grad_interval=grad_interval,
            query_method=query_method, token_num=token_num,
            use_full_input=use_full_input, freeze_encoder=freeze_encoder,
            decoder_mlp_width_scale=decoder_mlp_width_scale,
            residual_kl=residual_kl, operations=operations,
        )
        self.width = width

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):
        x, _, token_num, pre_pc = self.encoder(pc, feats)
        shape_embed = x[:, 0]
        latents = x
        return shape_embed, latents, token_num, pre_pc

    def forward(self, pc, feats, volume_queries, sample_posterior=True):
        raise NotImplementedError()


class ShapeAsLatentPerceiverEncoder(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device] = None,
                 dtype: Optional[Union[torch.dtype, str]] = None,
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 query_method: bool = False,
                 token_num: int = 256,
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 use_full_input: bool = True,
                 freeze_encoder: bool = False,
                 residual_kl: bool = False,
                 operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        MAP_DTYPE = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        if dtype is not None and isinstance(dtype, str):
            dtype = MAP_DTYPE[dtype]

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.supervision_type = supervision_type
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            num_latents=num_latents, point_feats=point_feats, width=width, heads=heads,
            layers=num_encoder_layers, init_scale=init_scale, qkv_bias=qkv_bias,
            flash=flash, use_ln_post=use_ln_post, use_checkpoint=use_checkpoint,
            query_method=query_method, use_full_input=use_full_input, token_num=token_num,
            no_query=True, operations=operations,
        )
        self.embed_dim = embed_dim
        self.residual_kl = residual_kl
        self.width = width

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):
        x, _, token_num, pre_pc = self.encoder(pc, feats)
        shape_embed = x[:, 0]
        latents = x
        return shape_embed, latents, token_num, pre_pc

    def forward(self):
        raise NotImplementedError()

# ============================================================================
# Factory functions
# ============================================================================

def get_encoder(pretrained_path: str = None, freeze_decoder: bool = False,
                **kwargs) -> AlignedShapeLatentPerceiver:
    model = AlignedShapeLatentPerceiver(**kwargs)
    if pretrained_path is not None:
        state_dict = comfy.utils.load_torch_file(pretrained_path)
        model.load_state_dict(state_dict)
    if freeze_decoder:
        model.geo_decoder.requires_grad_(False)
        model.encoder.query.requires_grad_(False)
        model.pre_kl.requires_grad_(False)
        model.post_kl.requires_grad_(False)
        model.transformer.requires_grad_(False)
    return model


def get_encoder_simplified(pretrained_path: str = None,
                           **kwargs) -> ShapeAsLatentPerceiverEncoder:
    model = ShapeAsLatentPerceiverEncoder(**kwargs)
    if pretrained_path is not None:
        state_dict = comfy.utils.load_torch_file(pretrained_path)
        model.load_state_dict(state_dict)
    return model
