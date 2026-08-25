# Copyright (c) Meta Platforms, Inc. and affiliates. (DINOv2 encoder)
# Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Depth Anything V2 (DINOv2 encoder + DPT decoder head), inference only.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/depth_anything_v2/ - a from-scratch DINOv2 ViT +
DPT reimplementation (not using the `transformers` library), consolidated
here into one flat file per this repo's own convention (melband_arch.py,
seedvc_arch.py) instead of that pack's nested dinov2_layers/ subpackage.

Simplified for eval-only, single-image inference (never training, never a
list-of-tensors batch): dropped BlockChunk/_get_intermediate_layers_chunked
(the source's DINOv2() factory always calls with block_chunks=0, so the
chunked path is dead code for every checkpoint this loads), stochastic-depth
training branches, forward_features_list, and the xformers nested-tensor
batch-grouping helpers (drop_add_residual_stochastic_depth_list,
get_attn_bias_and_cat, etc.) - all only reachable when NestedTensorBlock is
called on a list of tensors, which never happens here. NestedTensorBlock
itself is dropped too: for a plain Tensor input its forward() just calls
straight through to the plain Block.forward() it subclasses, so using Block
directly is behavior-identical for this use case. None of this affects the
module hierarchy or state_dict keys a checkpoint loads into - only unreached
control flow was trimmed. ConvBlock (defined but never referenced in the
source dpt.py) was dropped for the same reason.

Weights auto-download from HuggingFace on first use (depth-anything/
Depth-Anything-V2-{Small,Base,Large,Giant}), same pattern as this repo's
Qwen3-TTS loader: no config file, no symlink cache tricks, just a plain
local folder under models/depth_anything_v2/.
"""
import logging
import math
import os
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torchvision.transforms import Compose

import folder_paths

logger = logging.getLogger(__name__)

try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

DEPTH_MODELS_DIR = os.path.join(folder_paths.models_dir, "depth_anything_v2")

MODEL_REPO_IDS = {
    "depth_anything_v2_vits.pth": "depth-anything/Depth-Anything-V2-Small",
    "depth_anything_v2_vitb.pth": "depth-anything/Depth-Anything-V2-Base",
    "depth_anything_v2_vitl.pth": "depth-anything/Depth-Anything-V2-Large",
    "depth_anything_v2_vitg.pth": "depth-anything/Depth-Anything-V2-Giant",
}

MODEL_CONFIGS = {
    "depth_anything_v2_vits.pth": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "depth_anything_v2_vitb.pth": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "depth_anything_v2_vitl.pth": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "depth_anything_v2_vitg.pth": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


# ── DINOv2 layers ────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


try:
    from xformers.ops import SwiGLU as _XSwiGLU
    _SwiGLUBase = _XSwiGLU
except ImportError:
    _SwiGLUBase = SwiGLUFFN


class SwiGLUFFNFused(_SwiGLUBase):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=None, drop=0.0, bias=True):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(in_features=in_features, hidden_features=hidden_features,
                         out_features=out_features, bias=bias)


class PatchEmbed(nn.Module):
    """2D image to patch embedding: (B,C,H,W) -> (B,N,D)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        image_HW = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_HW = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])
        self.patch_size = patch_HW
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0 and W % patch_W == 0, \
            f"Input {H}x{W} is not a multiple of patch size {patch_H}x{patch_W}"
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = unbind(qkv, 2)
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        try:
            x_out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        except NotImplementedError:
            Bh, Dh = q.shape[0] * q.shape[1], q.shape[-1]
            x_sdpa = F.scaled_dot_product_attention(
                q.reshape(Bh, N, Dh), k.reshape(Bh, N, Dh), v.reshape(Bh, N, Dh),
                attn_mask=None, dropout_p=0.0, is_causal=False)
            x_out = x_sdpa.reshape(B, self.num_heads, N, Dh)
        x_out = x_out.permute(0, 2, 1, 3).reshape(B, N, C)
        x_out = self.proj(x_out)
        return self.proj_drop(x_out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, proj_bias=True,
                 ffn_bias=True, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_class=Attention, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, proj_drop=0.0)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ── DINOv2 vision transformer ────────────────────────────────────────────

def _init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        _named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, qkv_bias=True, ffn_bias=True, proj_bias=True,
                 init_values=None, embed_layer=PatchEmbed, act_layer=nn.GELU, block_fn=Block,
                 ffn_layer="mlp", num_register_tokens=0, interpolate_antialias=False,
                 interpolate_offset=0.1):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None)

        if ffn_layer == "mlp":
            ffn_layer_cls = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer_cls = SwiGLUFFNFused
        else:
            raise NotImplementedError(ffn_layer)

        self.blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     proj_bias=proj_bias, ffn_bias=ffn_bias, norm_layer=norm_layer, act_layer=act_layer,
                     ffn_layer=ffn_layer_cls, init_values=init_values)
            for _ in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        _named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy), mode="bicubic", antialias=self.interpolate_antialias)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def get_intermediate_layers(self, x: Tensor, n: Union[int, Sequence] = 1,
                                return_class_token: bool = False) -> Tuple:
        x = self.prepare_tokens_with_masks(x)
        total_block_len = len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                outputs.append(x)
        outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens:] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


def _vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
                                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def _vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
                                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def _vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
                                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def _vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(patch_size=patch_size, embed_dim=1536, depth=40, num_heads=24,
                                 mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                                 num_register_tokens=num_register_tokens, **kwargs)


def _build_dinov2(encoder_name):
    model_zoo = {"vits": _vit_small, "vitb": _vit_base, "vitl": _vit_large, "vitg": _vit_giant2}
    return model_zoo[encoder_name](
        img_size=518, patch_size=14, init_values=1.0,
        ffn_layer="mlp" if encoder_name != "vitg" else "swiglufused",
        num_register_tokens=0, interpolate_antialias=False, interpolate_offset=0.1)


# ── DPT decoder head ─────────────────────────────────────────────────────

def _make_scratch(in_shape, out_shape, groups=1):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, 3, 1, 1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        if bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, bn=False, align_corners=True):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)
        modifier = {"scale_factor": 2} if size is None else {"size": size}
        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(output)


class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=(256, 512, 1024, 1024)):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, 1, 1, 0) for out_channel in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, 4, 0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, 2, 0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], 3, 2, 1),
        ])
        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn)
        self.scratch.refinenet4 = FeatureFusionBlock(features, nn.ReLU(False), bn=use_bn)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, 3, 1, 1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 1, 1, 1, 0), nn.ReLU(True), nn.Identity())

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x, cls_token = x[0], x[1]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        return self.scratch.output_conv2(out)


class DepthAnythingV2(nn.Module):
    _INTERMEDIATE_LAYER_IDX = {
        "vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23], "vitg": [9, 19, 29, 39],
    }

    def __init__(self, encoder="vitl", features=256, out_channels=(256, 512, 1024, 1024)):
        super().__init__()
        self.encoder = encoder
        self.pretrained = _build_dinov2(encoder)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, out_channels=out_channels)

    def forward(self, x, max_depth):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self._INTERMEDIATE_LAYER_IDX[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w) * max_depth
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_image_bgr, input_size=518, max_depth=1.0):
        image, (h, w) = self._image_to_tensor(raw_image_bgr, input_size)
        depth = self.forward(image, max_depth)
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu().numpy()

    def _image_to_tensor(self, raw_image_bgr, input_size):
        transform = Compose([
            _Resize(input_size, input_size, resize_target=False, keep_aspect_ratio=True,
                   ensure_multiple_of=14, resize_method="lower_bound",
                   image_interpolation_method=cv2.INTER_CUBIC),
            _NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            _PrepareForNet(),
        ])
        h, w = raw_image_bgr.shape[:2]
        image = cv2.cvtColor(raw_image_bgr, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        return image.to(device), (h, w)


# ── preprocessing transforms (util/transform.py) ─────────────────────────

class _Resize:
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False,
                 ensure_multiple_of=1, resize_method="lower_bound",
                 image_interpolation_method=cv2.INTER_AREA):
        self._width = width
        self._height = height
        self._resize_target = resize_target
        self._keep_aspect_ratio = keep_aspect_ratio
        self._multiple_of = ensure_multiple_of
        self._resize_method = resize_method
        self._image_interpolation_method = image_interpolation_method

    def _constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self._multiple_of) * self._multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self._multiple_of) * self._multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self._multiple_of) * self._multiple_of).astype(int)
        return y

    def _get_size(self, width, height):
        scale_height = self._height / height
        scale_width = self._width / width
        if self._keep_aspect_ratio:
            if self._resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self._resize_method} not implemented")
        new_height = self._constrain_to_multiple_of(scale_height * height, min_val=self._height)
        new_width = self._constrain_to_multiple_of(scale_width * width, min_val=self._width)
        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self._get_size(sample["image"].shape[1], sample["image"].shape[0])
        sample["image"] = cv2.resize(sample["image"], (width, height),
                                     interpolation=self._image_interpolation_method)
        return sample


class _NormalizeImage:
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self._mean) / self._std
        return sample


class _PrepareForNet:
    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)
        return sample


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(DEPTH_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("Depth Anything V2: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(DEPTH_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=DEPTH_MODELS_DIR)


def installed_checkpoints():
    if not os.path.isdir(DEPTH_MODELS_DIR):
        return []
    return sorted(f for f in os.listdir(DEPTH_MODELS_DIR) if f in MODEL_CONFIGS)


class DepthAnythingV2Detector:
    """Loads a checkpoint (downloading if needed) and estimates depth for one IMAGE tensor."""

    def __init__(self, filename="depth_anything_v2_vitb.pth"):
        model_path = _download_checkpoint(filename)
        self.model = DepthAnythingV2(**MODEL_CONFIGS[filename])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def estimate(self, image_hwc_uint8, resolution=512):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3]."""
        h, w = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        depth = self.model.infer_image(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), input_size=518, max_depth=1.0)
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-5) * 255.0
        depth = depth.astype(np.uint8)
        depth_rgb = np.repeat(depth[:, :, None], 3, axis=2)

        if (target_h, target_w) != (h, w):
            depth_rgb = cv2.resize(depth_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_rgb
