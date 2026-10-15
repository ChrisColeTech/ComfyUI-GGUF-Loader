"""
UniRig skinning weight predictor — ComfyUI-native.

Predicts per-vertex skinning weights given a mesh and skeleton.
Uses PTv3 for local mesh features, Michelangelo for global features,
and cross-attention to combine bone and mesh representations.

Inference-only: training_step, forward, process_fn, and MHA wrapper are removed.
FlashMHA is replaced with explicit Wq/Wkv/out_proj projections
using ComfyUI attention dispatch.
"""

import logging
import math

import torch
from torch import nn, FloatTensor, LongTensor, Tensor
import torch.nn.functional as F

import comfy.ops

# CCTech: pure-torch segment_csr instead of the torch_scatter extension.
from . import native_fallbacks as torch_scatter

from .attention import comfy_attention_flat
from .model_spec import ModelSpec
from .parse_encoder import MAP_MESH_ENCODER, get_mesh_encoder

from typing import Dict, List

log = logging.getLogger("unirig")


# ============================================================================
# Positional embedding (buffer only, no operations= needed)
# ============================================================================

class FrequencyPositionalEmbedding(nn.Module):
    """Sin/cosine positional embedding."""

    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device="cpu")
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32, device="cpu"
            )
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self._get_dims(input_dim)

    def _get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        return input_dim * (self.num_freqs * 2 + temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies.to(device=x.device)).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


# ============================================================================
# Attention projections (replaces FlashMHA)
# ============================================================================

class CrossAttnProjections(nn.Module):
    """Explicit cross-attention projections matching FlashMHA checkpoint keys.

    Checkpoint keys: Wq.weight, Wq.bias, Wkv.weight, Wkv.bias, out_proj.weight, out_proj.bias
    """
    def __init__(self, feat_dim, num_heads, dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.Wq = operations.Linear(feat_dim, feat_dim, device=device, dtype=dtype)
        self.Wkv = operations.Linear(feat_dim, feat_dim * 2, device=device, dtype=dtype)
        self.out_proj = operations.Linear(feat_dim, feat_dim, device=device, dtype=dtype)

    def forward(self, q, kv):
        q_proj = self.Wq(q)
        kv_proj = self.Wkv(kv)
        k_proj, v_proj = kv_proj.chunk(2, dim=-1)
        attn_out = comfy_attention_flat(q_proj, k_proj, v_proj, heads=self.num_heads)
        return self.out_proj(attn_out)


class ResidualCrossAttn(nn.Module):
    def __init__(self, feat_dim: int, num_heads: int, dtype=None, device=None, operations=None):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm1 = operations.LayerNorm(feat_dim, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(feat_dim, device=device, dtype=dtype)
        # Namespace 'attention' matches raw checkpoint key prefix
        self.attention = CrossAttnProjections(feat_dim, num_heads, dtype=dtype, device=device, operations=operations)
        self.ffn = nn.Sequential(
            operations.Linear(feat_dim, feat_dim * 4, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(feat_dim * 4, feat_dim, device=device, dtype=dtype),
        )

    def forward(self, q, kv):
        residual = q
        attn_output = self.attention(q, kv)
        x = self.norm1(residual + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x


# ============================================================================
# Sub-models
# ============================================================================

class BoneEncoder(nn.Module):
    def __init__(
        self,
        feat_bone_dim: int,
        feat_dim: int,
        embed_dim: int,
        num_heads: int,
        num_attn: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.feat_bone_dim = feat_bone_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.num_attn = num_attn

        self.position_embed = FrequencyPositionalEmbedding(input_dim=self.feat_bone_dim)

        self.bone_encoder = nn.Sequential(
            self.position_embed,
            operations.Linear(self.position_embed.out_dim, embed_dim, device=device, dtype=dtype),
            operations.LayerNorm(embed_dim, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(embed_dim, embed_dim * 4, device=device, dtype=dtype),
            operations.LayerNorm(embed_dim * 4, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(embed_dim * 4, feat_dim, device=device, dtype=dtype),
            operations.LayerNorm(feat_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.attn = nn.ModuleList()
        for _ in range(self.num_attn):
            self.attn.append(ResidualCrossAttn(feat_dim, self.num_heads, dtype=dtype, device=device, operations=operations))

    def forward(
        self,
        base_bone: FloatTensor,
        num_bones: LongTensor,
        parents: LongTensor,
        min_coord: FloatTensor,
        global_latents: FloatTensor,
    ):
        B = base_bone.shape[0]
        J = base_bone.shape[1]
        x = self.bone_encoder((base_bone - min_coord[:, None, :]).reshape(-1, base_bone.shape[-1])).reshape(B, J, -1)
        latents = torch.cat([x, global_latents], dim=1)
        for attn in self.attn:
            x = attn(x, latents)
        return x


class SkinweightPred(nn.Module):
    def __init__(self, in_dim, mlp_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.net = nn.Sequential(
            operations.Linear(in_dim, mlp_dim, device=device, dtype=dtype),
            operations.LayerNorm(mlp_dim, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(mlp_dim, mlp_dim, device=device, dtype=dtype),
            operations.LayerNorm(mlp_dim, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(mlp_dim, mlp_dim, device=device, dtype=dtype),
            operations.LayerNorm(mlp_dim, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(mlp_dim, mlp_dim, device=device, dtype=dtype),
            operations.LayerNorm(mlp_dim, device=device, dtype=dtype),
            nn.GELU(),
            operations.Linear(mlp_dim, 1, device=device, dtype=dtype),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Main model
# ============================================================================

class UniRigSkin(ModelSpec):

    def __init__(self, mesh_encoder, global_encoder, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.dtype = dtype

        self.num_train_vertex       = kwargs['num_train_vertex']
        self.feat_dim               = kwargs['feat_dim']
        self.num_heads              = kwargs['num_heads']
        self.grid_size              = kwargs['grid_size']
        self.mlp_dim                = kwargs['mlp_dim']
        self.num_bone_attn          = kwargs['num_bone_attn']
        self.num_mesh_bone_attn     = kwargs['num_mesh_bone_attn']
        self.bone_embed_dim         = kwargs['bone_embed_dim']
        self.voxel_mask             = kwargs.get('voxel_mask', 2)

        # Thread dtype/device/operations through to encoders
        mesh_encoder = dict(mesh_encoder)
        mesh_encoder.setdefault('dtype', dtype)
        mesh_encoder.setdefault('device', device)
        mesh_encoder.setdefault('operations', operations)
        global_encoder = dict(global_encoder)
        global_encoder.setdefault('dtype', dtype)
        global_encoder.setdefault('device', device)
        global_encoder.setdefault('operations', operations)
        self.mesh_encoder = get_mesh_encoder(**mesh_encoder)
        self.global_encoder = get_mesh_encoder(**global_encoder)
        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):
            self.feat_map = nn.Sequential(
                operations.Linear(mesh_encoder['enc_channels'][-1], self.feat_dim, device=device, dtype=dtype),
                operations.LayerNorm(self.feat_dim, device=device, dtype=dtype),
                nn.GELU(),
            )
        else:
            raise NotImplementedError()
        if isinstance(self.global_encoder, MAP_MESH_ENCODER.michelangelo_encoder):
            self.out_proj = nn.Sequential(
                operations.Linear(self.global_encoder.width, self.feat_dim, device=device, dtype=dtype),
                operations.LayerNorm(self.feat_dim, device=device, dtype=dtype),
                nn.GELU(),
            )
        else:
            raise NotImplementedError()

        self.bone_encoder = BoneEncoder(
            feat_bone_dim=3,
            feat_dim=self.feat_dim,
            embed_dim=self.bone_embed_dim,
            num_heads=self.num_heads,
            num_attn=self.num_bone_attn,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.downscale = nn.Sequential(
            operations.Linear(2 * self.num_heads, self.num_heads, device=device, dtype=dtype),
            operations.LayerNorm(self.num_heads, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.skinweight_pred = SkinweightPred(
            self.num_heads,
            self.mlp_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.mesh_bone_attn = nn.ModuleList()
        self.mesh_bone_attn.extend([
            ResidualCrossAttn(self.feat_dim, self.num_heads, dtype=dtype, device=device, operations=operations)
            for _ in range(self.num_mesh_bone_attn)
        ])

        self.qmesh = operations.Linear(self.feat_dim, self.feat_dim * self.num_heads, device=device, dtype=dtype)
        self.kmesh = operations.Linear(self.feat_dim, self.feat_dim * self.num_heads, device=device, dtype=dtype)

        self.voxel_skin_embed = operations.Linear(1, self.num_heads, device=device, dtype=dtype)
        self.voxel_skin_norm = operations.LayerNorm(self.num_heads, device=device, dtype=dtype)
        self.attn_skin_norm = operations.LayerNorm(self.num_heads, device=device, dtype=dtype)

    def encode_mesh_cond(self, vertices: FloatTensor, normals: FloatTensor) -> FloatTensor:
        assert not torch.isnan(vertices).any()
        assert not torch.isnan(normals).any()
        if isinstance(self.global_encoder, MAP_MESH_ENCODER.michelangelo_encoder):
            if (len(vertices.shape) == 3):
                shape_embed, latents, token_num, pre_pc = self.global_encoder.encode_latents(pc=vertices, feats=normals)
            else:
                shape_embed, latents, token_num, pre_pc = self.global_encoder.encode_latents(pc=vertices.unsqueeze(0), feats=normals.unsqueeze(0))
            latents = self.out_proj(latents)
            return latents
        else:
            raise NotImplementedError()

    def _get_predict(self, batch: Dict) -> FloatTensor:
        """Return predicted skin weights."""
        num_bones: Tensor = batch['num_bones']
        vertices: FloatTensor = batch['vertices']
        normals: FloatTensor = batch['normals']
        joints: FloatTensor = batch['joints']
        tails: FloatTensor = batch['tails']
        voxel_skin: FloatTensor = batch['voxel_skin']
        parents: LongTensor = batch['parents']

        dtype = next(self.parameters()).dtype
        vertices = vertices.type(dtype)
        normals = normals.type(dtype)
        joints = joints.type(dtype)
        tails = tails.type(dtype)
        voxel_skin = voxel_skin.type(dtype)

        B = vertices.shape[0]
        N = vertices.shape[1]
        J = joints.shape[1]

        assert vertices.dim() == 3
        assert normals.dim() == 3

        part_offset = torch.tensor([(i + 1) * N for i in range(B)], dtype=torch.int64, device=vertices.device)
        idx_ptr = torch.nn.functional.pad(part_offset, (1, 0), value=0)
        min_coord = torch_scatter.segment_csr(vertices.reshape(-1, 3), idx_ptr, reduce="min")

        # Inference: process in chunks
        pack = []
        for i in range((N + self.num_train_vertex - 1) // self.num_train_vertex):
            pack.append(torch.arange(i * self.num_train_vertex, min((i + 1) * self.num_train_vertex, N)))

        global_latents = self.encode_mesh_cond(vertices, normals)
        bone_feat = self.bone_encoder(
            base_bone=joints,
            num_bones=num_bones,
            parents=parents,
            min_coord=min_coord,
            global_latents=global_latents,
        )

        if isinstance(self.mesh_encoder, MAP_MESH_ENCODER.ptv3obj):
            feat = torch.cat([vertices, normals, torch.zeros_like(vertices)], dim=-1)
            ptv3_input = {
                'coord': vertices.reshape(-1, 3),
                'feat': feat.reshape(-1, 9),
                'offset': torch.tensor(batch['offset']),
                'grid_size': self.grid_size,
            }
            mesh_feat = self.mesh_encoder(ptv3_input).feat
            mesh_feat = self.feat_map(mesh_feat).view(B, N, self.feat_dim)
            mesh_feat = mesh_feat.type(dtype)
        else:
            raise NotImplementedError()

        latents = torch.cat([bone_feat, global_latents], dim=1)
        for block in self.mesh_bone_attn:
            mesh_feat = block(q=mesh_feat, kv=latents)

        bone_feat = self.kmesh(bone_feat).view(B, J, self.num_heads, self.feat_dim).transpose(1, 2)

        # Compute voxel skin mask
        skin_mask = voxel_skin.clone()
        for b in range(B):
            num = num_bones[b]
            for i in range(num):
                p = parents[b, i]
                if p < 0:
                    continue
                skin_mask[b, :, p] += skin_mask[b, :, i]

        skin_pred_list = []
        for indices in pack:
            cur_N = len(indices)
            cur_mesh_feat = self.qmesh(mesh_feat[:, indices]).view(B, cur_N, self.num_heads, self.feat_dim).transpose(1, 2)

            attn_weight = F.softmax(torch.bmm(
                cur_mesh_feat.reshape(B * self.num_heads, cur_N, -1),
                bone_feat.transpose(-2, -1).reshape(B * self.num_heads, -1, J)
            ) / math.sqrt(self.feat_dim), dim=-1, dtype=dtype)
            attn_weight = attn_weight.reshape(B, self.num_heads, cur_N, J).permute(0, 2, 3, 1)
            attn_weight = self.attn_skin_norm(attn_weight)

            embed_voxel_skin = self.voxel_skin_embed(voxel_skin[:, indices].reshape(B, cur_N, J, 1))
            embed_voxel_skin = self.voxel_skin_norm(embed_voxel_skin)

            attn_weight = torch.cat([attn_weight, embed_voxel_skin], dim=-1)
            attn_weight = self.downscale(attn_weight)

            skin_pred = torch.zeros(B, cur_N, J).to(attn_weight.device, dtype)
            for i in range(B):
                input_features = attn_weight[i, :, :num_bones[i], :].reshape(-1, attn_weight.shape[-1])
                pred = self.skinweight_pred(input_features).reshape(cur_N, num_bones[i])
                skin_pred[i, :, :num_bones[i]] = F.softmax(pred, dim=-1)
            skin_pred_list.append(skin_pred)

        skin_pred_list = torch.cat(skin_pred_list, dim=1)
        # Apply voxel mask
        for i in range(B):
            n = num_bones[i]
            skin_pred_list[i, :, :n] = skin_pred_list[i, :, :n] * torch.pow(skin_mask[i, :, :n], self.voxel_mask)
            skin_pred_list[i, :, :n] = skin_pred_list[i, :, :n] / skin_pred_list[i, :, :n].sum(dim=-1, keepdim=True)
        return skin_pred_list, torch.cat(pack, dim=0)

    def predict_step(self, batch: Dict):
        with torch.no_grad():
            num_bones: Tensor = batch['num_bones']
            skin_pred, _ = self._get_predict(batch=batch)
            outputs = []
            for i in range(skin_pred.shape[0]):
                outputs.append(skin_pred[i, :, :num_bones[i]])
            return outputs
