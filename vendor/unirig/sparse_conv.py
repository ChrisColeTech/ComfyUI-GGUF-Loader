# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Pure-torch stand-in for the slice of ``spconv.pytorch`` PTv3 uses.

UniRig's skinning encoder (Point Transformer V3) needs exactly two things from
spconv: ``SparseConvTensor`` as a feature/coordinate carrier and
``SubMConv3d`` (submanifold 3-D convolution: outputs only at the input's own
voxels). spconv/cumm ship as CUDA-version-specific wheels with no build for
every torch, so this module implements the same operation with a sorted-key
hash lookup (``torch.searchsorted``) and one gather-matmul per kernel offset.

Weight layout is spconv 2.x's ``[out, k0, k1, k2, in]`` (the checkpoint's own
layout - ``stem.conv.weight`` is ``[32, 5, 5, 5, 9]``), with kernel axis ``ki``
moving along index column ``i + 1`` and the output at voxel ``p`` being
``sum_k W[:, k, :] @ x[p + k - k // 2]`` (cross-correlation, as torch's dense
conv). ``tools/smoke_unirig.py --spconv-parity`` checks it against a real
spconv install. Same idea as ComfyUI core's ``comfy/ldm/trellis2/flexgemm.py``,
kept local so the pack does not depend on that internal path.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import torch
from torch import nn


class SparseConvTensor:
    """Features ``[N, C]`` at integer voxels ``indices [N, 4]`` (batch, i, j, k)."""

    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        # Neighbour maps keyed by (indice_key, kernel, dilation), shared by every
        # tensor derived from this one via replace_feature (spconv's indice_dict).
        self.indice_dict = {} if indice_dict is None else indice_dict

    def replace_feature(self, features):
        return SparseConvTensor(features, self.indices, self.spatial_shape,
                                self.batch_size, self.indice_dict)


def _triple(v):
    return tuple(int(x) for x in v) if isinstance(v, (list, tuple)) else (int(v),) * 3


def kernel_offsets(kernel_size, dilation):
    """Offsets in weight order (k0 outermost), centred on ``k // 2``."""
    ks, dil = _triple(kernel_size), _triple(dilation)
    return [tuple((a - k // 2) * d for a, k, d in zip(idx, ks, dil))
            for idx in itertools.product(*(range(k) for k in ks))]


@torch.no_grad()
def neighbor_map(indices, spatial_shape, kernel_size, dilation=1):
    """``[N, K]`` row index of the voxel at ``indices[n] + offset[k]``, or -1."""
    coords = indices.long()
    dims = [int(s) for s in spatial_shape]
    b, x, y, z = coords.unbind(1)

    def flat(bb, xx, yy, zz):
        return ((bb * dims[0] + xx) * dims[1] + yy) * dims[2] + zz

    keys = flat(b, x, y, z)
    sorted_keys, order = torch.sort(keys)
    n = keys.shape[0]
    offsets = kernel_offsets(kernel_size, dilation)
    out = torch.full((n, len(offsets)), -1, dtype=torch.long, device=coords.device)
    for col, (ox, oy, oz) in enumerate(offsets):
        nx, ny, nz = x + ox, y + oy, z + oz
        inside = ((nx >= 0) & (nx < dims[0]) & (ny >= 0) & (ny < dims[1])
                  & (nz >= 0) & (nz < dims[2]))
        query = flat(b, nx, ny, nz)
        pos = torch.searchsorted(sorted_keys, query).clamp_(max=max(n - 1, 0))
        hit = inside & (sorted_keys[pos] == query)
        out[:, col] = torch.where(hit, order[pos], out[:, col])
    return out


def submanifold_conv3d(features, neighbors, weight, bias=None):
    """Gather-matmul per kernel offset; ``weight`` is ``[out, k0, k1, k2, in]``."""
    co, ci = weight.shape[0], weight.shape[-1]
    w = weight.reshape(co, -1, ci).to(device=features.device, dtype=features.dtype)
    out = features.new_zeros((features.shape[0], co))
    for col in range(neighbors.shape[1]):
        idx = neighbors[:, col]
        hit = idx >= 0
        if not bool(hit.any()):
            continue
        rows = hit.nonzero(as_tuple=True)[0]
        out.index_add_(0, rows, features[idx[rows]] @ w[:, col, :].t())
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out


class SubMConv3d(nn.Module):
    """``spconv.pytorch.SubMConv3d`` (inference). ``padding``/``stride`` are moot:
    a submanifold conv keeps the input's active set by definition."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, **_):
        super().__init__()
        if _triple(stride) != (1, 1, 1) or groups != 1:
            raise NotImplementedError("SubMConv3d: only stride 1 / groups 1 are used by PTv3")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.dilation = _triple(dilation)
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        key = (self.indice_key, self.kernel_size, self.dilation)
        nbr = x.indice_dict.get(key)
        if nbr is None:
            nbr = neighbor_map(x.indices, x.spatial_shape, self.kernel_size, self.dilation)
            x.indice_dict[key] = nbr
        return x.replace_feature(submanifold_conv3d(x.features, nbr, self.weight, self.bias))


def is_spconv_module(module) -> bool:
    return isinstance(module, SubMConv3d)


# ``spconv.modules.is_spconv_module`` is how PointSequential dispatches.
modules = SimpleNamespace(is_spconv_module=is_spconv_module)
