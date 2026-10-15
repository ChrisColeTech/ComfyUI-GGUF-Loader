"""Pure-PyTorch fallbacks for UniRig native-extension dependencies.

These are NOT diffusers/transformers ports. They are drop-in replacements for
two PyG native extensions that have no cu130 wheel on Windows:

* ``torch_cluster.fps``        -> :func:`fps`  (farthest-point sampling)
* ``torch_scatter.segment_csr`` -> :func:`segment_csr` (packed-segment reductions)

and a packed-sequence attention used by PTv3:

* ``comfy_sparse_attn.dispatch_varlen_attention`` -> :func:`dispatch_varlen_attention`

They are written for correctness over speed; the real extensions are preferred
whenever importable (see ``source.py``).
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# torch_cluster.fps
# ---------------------------------------------------------------------------

@torch.no_grad()
def fps(pos: torch.Tensor, batch: torch.Tensor, ratio: float, random_start: bool = False) -> torch.Tensor:
    """Farthest-point sampling mirroring ``torch_cluster.fps``.

    Args:
        pos:    (N, D) point coordinates.
        batch:  (N,)   long tensor of batch id per point.
        ratio:  per-batch sampling ratio (float in (0, 1]).
        random_start: if True the seed point is random, else index 0.

    Returns:
        (M,) long tensor of selected indices into the *packed* ``pos``.
        M = sum_b round(N_b * ratio), clamped to >= 1 per batch.
    """
    device = pos.device
    out = []
    for b in torch.unique(batch):
        global_idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
        n = int(global_idx.numel())
        m = max(round(n * float(ratio)), 1)
        if m >= n:
            out.append(global_idx)
            continue
        sel = _fps_single(pos[global_idx], m, random_start=random_start)
        out.append(global_idx[sel])
    return torch.cat(out, dim=0).to(device)


@torch.no_grad()
def _fps_single(points: torch.Tensor, m: int, random_start: bool = False) -> torch.Tensor:
    """Iterative farthest-point sampling for a single point cloud.

    Returns (m,) local indices. O(n*m) which is fine for the few-thousand-point
    counts UniRig feeds the Michelangelo perceiver.
    """
    n = points.shape[0]
    device = points.device
    if m >= n:
        return torch.arange(n, device=device)

    # d2 holds the *minimum* squared distance from each point to the selected
    # set. The next farthest point is the one maximizing that minimum.
    if random_start:
        start = int(torch.randint(0, n, (1,), device=device).item())
    else:
        start = 0
    selected = [start]
    d2 = _point_dist2(points, points[start])

    for _ in range(1, m):
        nxt = int(torch.argmax(d2).item())
        selected.append(nxt)
        new_d2 = _point_dist2(points, points[nxt])
        d2 = torch.minimum(d2, new_d2)

    return torch.tensor(selected, device=device, dtype=torch.long)


def _point_dist2(points: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    diff = points - query
    return torch.einsum("ij,ij->i", diff, diff)


# ---------------------------------------------------------------------------
# torch_scatter.segment_csr
# ---------------------------------------------------------------------------

def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    """Segmented reduction mirroring ``torch_scatter.segment_csr``.

    Args:
        src:     (N, ...) source values along dim 0.
        indptr:  (S+1,) long CSR pointer; segment i is src[indptr[i]:indptr[i+1]].
        reduce:  one of {"min", "max", "sum", "mean"}.

    Returns:
        (S, ...) reduced tensor.
    """
    reduce = reduce.lower()
    if indptr.numel() == 0:
        return src.new_empty((0,) + tuple(src.shape[1:]))
    seg_ids = torch.repeat_interleave(
        torch.arange(indptr.shape[0] - 1, device=indptr.device, dtype=torch.long),
        (indptr[1:] - indptr[:-1]).to(torch.long),
    )
    num_seg = int(indptr.shape[0] - 1)
    out_shape = (num_seg,) + tuple(src.shape[1:])
    if src.dim() == 1:
        idx = seg_ids
    else:
        idx = seg_ids.view(-1, *([1] * (src.dim() - 1))).expand_as(src)

    if reduce in ("min", "max"):
        fill = float("inf") if reduce == "min" else float("-inf")
        out = src.new_full(out_shape, fill)
        if src.numel():
            out.scatter_reduce_(0, idx, src, reduce="amin" if reduce == "min" else "amax", include_self=False)
        return out
    if reduce == "sum":
        out = src.new_zeros(out_shape)
        if src.numel():
            out.scatter_add_(0, idx, src)
        return out
    if reduce == "mean":
        total = src.new_zeros(out_shape)
        if src.numel():
            total.scatter_add_(0, idx, src)
        counts = src.new_zeros(num_seg).scatter_add_(
            0, seg_ids, torch.ones_like(seg_ids, dtype=src.dtype)
        )
        counts = counts.clamp(min=1.0)
        if src.dim() == 1:
            return total / counts
        return total / counts.view(-1, *([1] * (src.dim() - 1)))
    raise ValueError(f"unsupported reduce '{reduce}' for segment_csr")


# ---------------------------------------------------------------------------
# comfy_sparse_attn.dispatch_varlen_attention  (PTv3 packed attention)
# ---------------------------------------------------------------------------

@torch.no_grad()
def dispatch_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor:
    """Variable-length (packed) attention used by PTv3.

    q, k, v are packed (T, H, D) tensors. ``cu_seqlens_*`` are (S+1,) cumulative
    segment boundaries. Each segment is a self-contained block attending to
    itself (q/kv share the same packing in PTv3). Falls back to per-segment
    ``F.scaled_dot_product_attention`` which is mathematically equivalent.

    ``max_seqlen_*`` are accepted for signature parity and otherwise unused.
    """
    import torch.nn.functional as F

    cu_q = cu_seqlens_q.detach().cpu().tolist()
    outs = []
    for s, e in zip(cu_q[:-1], cu_q[1:]):
        qs, ks, vs = q[s:e], k[s:e], v[s:e]  # each (L, H, D)
        # sdpa wants (B, H, L, D)
        qs = qs.transpose(0, 1).unsqueeze(0)
        ks = ks.transpose(0, 1).unsqueeze(0)
        vs = vs.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0, is_causal=False)
        outs.append(o.squeeze(0).transpose(0, 1))  # back to (L, H, D)
    return torch.cat(outs, dim=0)
