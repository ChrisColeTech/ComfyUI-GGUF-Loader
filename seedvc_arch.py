# Apache-2.0. Adapted from Plachtaa/seed-vc; CAMPPlus from 3D-Speaker.
"""Checkpoint-compatible architecture pieces missing from ComfyUI core."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


def sequence_mask(lengths, max_length=None):
    max_length = int(lengths.max()) if max_length is None else max_length
    return torch.arange(max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def _pad1d(x, padding, mode="reflect"):
    left, right = padding
    extra = max(0, max(left, right) - x.shape[-1] + 1) if mode == "reflect" else 0
    if extra:
        x = F.pad(x, (0, extra))
    x = F.pad(x, (left, right), mode)
    return x[..., :x.shape[-1] - extra] if extra else x


class _NormConv1d(nn.Module):
    def __init__(self, *args, norm="none", **kwargs):
        super().__init__()
        conv = nn.Conv1d(*args, **kwargs)
        self.conv = weight_norm(conv) if norm == "weight_norm" else conv

    def forward(self, x):
        return self.conv(x)


class SConv1d(nn.Module):
    """The small EnCodec padding wrapper used by SeedVC's WaveNet."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True, causal=False, norm="none", pad_mode="reflect"):
        super().__init__()
        self.conv = _NormConv1d(in_channels, out_channels, kernel_size, stride,
                                dilation=dilation, groups=groups, bias=bias, norm=norm)
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        conv = self.conv.conv
        kernel = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1
        total = kernel - conv.stride[0]
        frames = (x.shape[-1] - kernel + total) / conv.stride[0] + 1
        ideal = (math.ceil(frames) - 1) * conv.stride[0] + kernel - total
        extra = ideal - x.shape[-1]
        right = extra if self.causal else total // 2 + extra
        left = total if self.causal else total - total // 2
        return self.conv(_pad1d(x, (left, right), self.pad_mode))


class WN(nn.Module):
    def __init__(self, hidden_channels=512, kernel_size=5, n_layers=8, gin_channels=512):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.cond_layer = SConv1d(gin_channels, 2 * hidden_channels * n_layers, 1,
                                  norm="weight_norm")
        for i in range(n_layers):
            self.in_layers.append(SConv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                                          norm="weight_norm"))
            out = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(SConv1d(hidden_channels, out, 1, norm="weight_norm"))

    def forward(self, x, mask, g):
        output = torch.zeros_like(x)
        g = self.cond_layer(g)
        for i, (in_layer, res_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            offset = i * 2 * self.hidden_channels
            acts = in_layer(x) + g[:, offset:offset + 2 * self.hidden_channels]
            a, b = acts.chunk(2, dim=1)
            acts = torch.tanh(a) * torch.sigmoid(b)
            res = res_layer(acts)
            if i < self.n_layers - 1:
                x = (x + res[:, :self.hidden_channels]) * mask
                output = output + res[:, self.hidden_channels:]
            else:
                output = output + res
        return output * mask


class InterpolateRegulator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend([nn.Conv1d(512, 512, 3, padding=1), nn.GroupNorm(1, 512), nn.Mish()])
        layers.append(nn.Conv1d(512, 512, 1))
        self.model = nn.Sequential(*layers)
        self.embedding = nn.Embedding(2048, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 512))
        self.content_in_proj = nn.Linear(768, 512)

    def forward(self, x, ylens, n_quantizers=3, f0=None):
        del n_quantizers, f0
        x = self.content_in_proj(x).transpose(1, 2)
        x = F.interpolate(x, size=int(ylens.max()), mode="nearest")
        out = self.model(x).transpose(1, 2)
        mask = sequence_mask(ylens, out.shape[1]).unsqueeze(-1)
        return out * mask, ylens, None, None, None


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.project_layer = nn.Linear(dim, 2 * dim)
        self.norm = RMSNorm(dim)

    def forward(self, x, embedding):
        if embedding is None:
            return self.norm(x)
        weight, bias = self.project_layer(embedding).chunk(2, dim=-1)
        return weight * self.norm(x) + bias


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-5

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope, mask):
        b, t, d = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.head_dim)
        k = k.view(b, t, self.heads, self.head_dim)
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, rope), _apply_rope(k, rope)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v,
                                            attn_mask=mask, dropout_p=0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, dim=512, intermediate=1536):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate, bias=False)
        self.w3 = nn.Linear(dim, intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.ffn_norm = AdaptiveLayerNorm(512)
        self.attention_norm = AdaptiveLayerNorm(512)
        self.skip_in_linear = nn.Linear(1024, 512)

    def forward(self, x, c, rope, mask, skip=None):
        if skip is not None:
            x = self.skip_in_linear(torch.cat([x, skip], -1))
        x = x + self.attention(self.attention_norm(x, c), rope, mask)
        return x + self.feed_forward(self.ffn_norm(x, c))


def _rope_cache(length, dim, device, dtype):
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = torch.outer(torch.arange(length, device=device).float(), inv)
    return torch.stack([angles.cos(), angles.sin()], -1).to(dtype)


def _apply_rope(x, rope):
    pair = x.float().reshape(*x.shape[:-1], -1, 2)
    r = rope.view(1, x.shape[1], 1, pair.shape[-2], 2).float()
    out = torch.stack([pair[..., 0] * r[..., 0] - pair[..., 1] * r[..., 1],
                       pair[..., 1] * r[..., 0] + pair[..., 0] * r[..., 1]], -1)
    return out.flatten(3).to(x.dtype)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(13)])
        self.norm = AdaptiveLayerNorm(512)

    def forward(self, x, c, mask):
        rope = _rope_cache(x.shape[1], 64, x.device, x.dtype)
        skips = []
        for i, layer in enumerate(self.layers):
            skip = skips.pop() if i > 6 else None
            x = layer(x, c, rope, mask, skip)
            if i < 6:
                skips.append(x)
        return self.norm(x, c)


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 512))
        self.register_buffer("freqs", torch.exp(-math.log(10000) * torch.arange(128).float() / 128),
                             persistent=False)

    def forward(self, t):
        args = 1000 * t[:, None].float() * self.freqs[None]
        return self.mlp(torch.cat([args.cos(), args.sin()], -1))


class FinalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_final = nn.LayerNorm(512, elementwise_affine=False, eps=1e-6)
        self.linear = weight_norm(nn.Linear(512, 512))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(512, 1024))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, 1)
        return self.linear(self.norm_final(x) * (1 + scale[:, None]) + shift[:, None])


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()
        self.x_embedder = weight_norm(nn.Linear(80, 512))
        self.cond_embedder = nn.Embedding(1024, 512)
        self.cond_projection = nn.Linear(512, 512)
        self.t_embedder = TimestepEmbedder()
        self.register_buffer("input_pos", torch.arange(8192))
        self.t_embedder2 = TimestepEmbedder()
        self.conv1 = nn.Linear(512, 512)
        self.conv2 = nn.Conv1d(512, 80, 1)
        self.wavenet = WN()
        self.final_layer = FinalLayer()
        self.content_mask_embedder = nn.Embedding(1, 512)
        self.f0_embedder = nn.Embedding(512, 512)
        self.res_projection = nn.Linear(512, 512)
        self.skip_linear = nn.Linear(592, 512)
        self.cond_x_merge_linear = nn.Linear(864, 512)

    def setup_caches(self, max_batch_size=1, max_seq_length=8192):
        del max_batch_size, max_seq_length

    def forward(self, x, prompt_x, x_lens, t, style, cond):
        t1 = self.t_embedder(t)
        original = x.transpose(1, 2)
        merged = torch.cat([original, prompt_x.transpose(1, 2), self.cond_projection(cond),
                            style[:, None].expand(-1, x.shape[-1], -1)], -1)
        merged = self.cond_x_merge_linear(merged)
        valid = sequence_mask(x_lens, merged.shape[1])
        mask = valid[:, None, None, :].expand(-1, 1, merged.shape[1], -1)
        result = self.transformer(merged, t1[:, None], mask)
        result = self.skip_linear(torch.cat([result, original], -1))
        wave = self.conv1(result).transpose(1, 2)
        wave = self.wavenet(wave, valid[:, None], self.t_embedder2(t)[:, :, None]).transpose(1, 2)
        wave = wave + self.res_projection(result)
        return self.conv2(self.final_layer(wave, t1).transpose(1, 2))


class CFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = DiT()
        self.in_channels = 80

    def inference(self, mu, lengths, prompt, style, steps=25, cfg_rate=0.5):
        x = torch.randn(mu.shape[0], 80, mu.shape[1], device=mu.device, dtype=mu.dtype)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt.shape[-1]] = prompt
        x[..., :prompt.shape[-1]] = 0
        for t in torch.linspace(0, 1, steps + 1, device=x.device)[:-1]:
            dt = 1.0 / steps
            if cfg_rate > 0:
                stacked = self.estimator(torch.cat([x, x]), torch.cat([prompt_x, torch.zeros_like(prompt_x)]),
                                         lengths, t.expand(2), torch.cat([style, torch.zeros_like(style)]),
                                         torch.cat([mu, torch.zeros_like(mu)]))
                conditioned, null = stacked.chunk(2)
                velocity = (1 + cfg_rate) * conditioned - cfg_rate * null
            else:
                velocity = self.estimator(x, prompt_x, lengths, t[None], style, mu)
            x = x + dt * velocity
            x[..., :prompt.shape[-1]] = 0
        return x


class SeedVCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfm = CFM()
        self.length_regulator = InterpolateRegulator()


# CAMPPlus, adapted from 3D-Speaker (Apache-2.0).
def _nonlinear(channels, affine=True):
    from collections import OrderedDict
    return nn.Sequential(OrderedDict([
        ("batchnorm", nn.BatchNorm1d(channels, affine=affine)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, (stride, 1), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential() if stride == 1 and in_planes == planes else nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, (stride, 1), bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class FCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32)
        self.layer1 = nn.Sequential(BasicResBlock(32, 32, 2), BasicResBlock(32, 32))
        self.layer2 = nn.Sequential(BasicResBlock(32, 32, 2), BasicResBlock(32, 32))
        self.conv2, self.bn2 = nn.Conv2d(32, 32, 3, (2, 1), 1, bias=False), nn.BatchNorm2d(32)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        x = F.relu(self.bn2(self.conv2(self.layer2(self.layer1(x)))))
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])


class TDNNLayer(nn.Module):
    def __init__(self, inc, outc, kernel=1, stride=1, dilation=1, bias=False):
        super().__init__()
        self.linear = nn.Conv1d(inc, outc, kernel, stride, (kernel - 1) // 2 * dilation,
                                dilation, bias=bias)
        self.nonlinear = _nonlinear(outc)
    def forward(self, x): return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, inc, outc, dilation):
        super().__init__()
        self.linear_local = nn.Conv1d(inc, outc, 3, padding=dilation, dilation=dilation, bias=False)
        self.linear1, self.relu = nn.Conv1d(inc, inc // 2, 1), nn.ReLU(inplace=True)
        self.linear2, self.sigmoid = nn.Conv1d(inc // 2, outc, 1), nn.Sigmoid()
    def forward(self, x):
        seg = F.avg_pool1d(x, 100, 100, ceil_mode=True)
        seg = seg.unsqueeze(-1).expand(*seg.shape, 100).reshape(*seg.shape[:-1], -1)[..., :x.shape[-1]]
        gate = self.sigmoid(self.linear2(self.relu(self.linear1(x.mean(-1, keepdim=True) + seg))))
        return self.linear_local(x) * gate


class CAMDenseLayer(nn.Module):
    def __init__(self, inc, dilation):
        super().__init__()
        self.nonlinear1, self.linear1 = _nonlinear(inc), nn.Conv1d(inc, 128, 1, bias=False)
        self.nonlinear2, self.cam_layer = _nonlinear(128), CAMLayer(128, 32, dilation)
    def forward(self, x): return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(x))))


class CAMDenseBlock(nn.ModuleList):
    def __init__(self, count, inc, dilation):
        super().__init__()
        for i in range(count):
            self.add_module(f"tdnnd{i + 1}", CAMDenseLayer(inc + i * 32, dilation))
    def forward(self, x):
        for layer in self: x = torch.cat([x, layer(x)], 1)
        return x


class TransitLayer(nn.Module):
    def __init__(self, inc, outc):
        super().__init__(); self.nonlinear, self.linear = _nonlinear(inc), nn.Conv1d(inc, outc, 1, bias=False)
    def forward(self, x): return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
    def __init__(self, inc, outc):
        super().__init__(); self.linear, self.nonlinear = nn.Conv1d(inc, outc, 1, bias=False), _nonlinear(outc, False)
    def forward(self, x): return self.nonlinear(self.linear(x.unsqueeze(-1))).squeeze(-1)


class CAMPPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = FCM()
        modules = [("tdnn", TDNNLayer(320, 128, 5, stride=2))]
        channels = 128
        for i, (count, dilation) in enumerate(((12, 1), (24, 2), (16, 2)), 1):
            modules.append((f"block{i}", CAMDenseBlock(count, channels, dilation)))
            channels += count * 32
            modules.append((f"transit{i}", TransitLayer(channels, channels // 2)))
            channels //= 2
        modules.append(("out_nonlinear", _nonlinear(channels)))
        from collections import OrderedDict
        self.xvector = nn.Sequential(OrderedDict(modules))
        self.stats = nn.Identity()
        self.dense = DenseLayer(channels * 2, 192)

    def forward(self, x):
        x = self.xvector(self.head(x.transpose(1, 2)))
        x = torch.cat([x.mean(-1), x.std(-1)], -1)
        return self.dense(x)
