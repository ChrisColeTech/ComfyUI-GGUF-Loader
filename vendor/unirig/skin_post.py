"""Skin-weight post-processing, ported from upstream UniRig ``reskin``.

Upstream (UniRig-main src/system/skin.py) cleans predicted skin weights with a
hierarchical edge diffusion followed by a small-weight threshold. The k-NN
median resample half of ``reskin`` does not apply here — our wrapper feeds the
actual mesh vertices to the skinning network, so weights are already
per-vertex — but the diffusion pass is what tames stray diffuse-weight
vertices (which otherwise blend distant bone transforms and fly off under
LBS).

Faithful port, defaults matching upstream's inference call:
``alpha=2.0, iter_steps=1, threshold=0.03`` (threshold 0.03 from
``sample_method='median'`` call site; generic default was 0.01).
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def diffuse_skin_weights(
    skin: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    parents: List[Union[None, int]],
    alpha: float = 2.0,
    iter_steps: int = 1,
    threshold: float = 0.03,
) -> np.ndarray:
    """Upstream's edge-diffusion + threshold cleanup for skin weights.

    Args:
        skin:     (N, J) predicted weights (rows need not be normalized).
        vertices: (N, 3) rest vertices the weights live on (mesh space; only
                  edge lengths matter, so normalized or raw both work — use
                  the same space the model saw for parity with upstream).
        faces:    (F, 3) triangles.
        parents:  (J,) parent indices, None/-1 for root.
        alpha:    distance falloff for edge transfer coefficients.
        iter_steps: diffusion iterations (upstream inference uses 1).
        threshold: post-diffusion small-weight cutoff.

    Returns:
        (N, J) cleaned, row-normalized weights (float32).
    """
    skin = np.asarray(skin, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    n, j = skin.shape
    par = [None if (p is None or int(p) < 0) else int(p) for p in list(parents)]

    # Directed edges both ways, as upstream.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edges = np.concatenate([edges, edges[:, [1, 0]]], axis=0)  # (6F, 2)

    skin = skin / np.clip(skin.sum(axis=-1, keepdims=True), 1e-12, None)

    for _ in range(iter_steps):
        sum_skin = skin.copy()
        # Accumulate child weights into parents (leaf -> root).
        for i in reversed(range(j)):
            p = par[i]
            if p is None:
                continue
            sum_skin[:, p] += sum_skin[:, i]

        # Transfer only from hotter to cooler endpoint, weighted by edge length.
        mask = sum_skin[edges[:, 1]] < sum_skin[edges[:, 0]]
        dis = np.sqrt(((vertices[edges[:, 1]] - vertices[edges[:, 0]]) ** 2).sum(axis=1, keepdims=True))
        co = np.exp(-dis * alpha)

        neighbor_skin = np.zeros_like(sum_skin)
        neighbor_co = np.zeros((n, j), dtype=np.float64)
        np.add.at(neighbor_skin, edges[:, 1], sum_skin[edges[:, 0]] * co * mask)
        np.add.at(neighbor_co, edges[:, 1], co * mask)

        sum_skin = (sum_skin + neighbor_skin) / (1.0 + neighbor_co)

        # Un-accumulate (root -> leaf) and renormalize.
        for i in range(j):
            p = par[i]
            if p is None:
                continue
            sum_skin[:, p] -= sum_skin[:, i]
        skin = sum_skin / np.clip(sum_skin.sum(axis=-1, keepdims=True), 1e-12, None)

    # Threshold small weights (skip rows that would go all-zero).
    keep = (skin >= threshold).any(axis=-1, keepdims=True)
    skin[(skin < threshold) & keep] = 0.0
    skin = skin / np.clip(skin.sum(axis=-1, keepdims=True), 1e-12, None)
    return skin.astype(np.float32)
