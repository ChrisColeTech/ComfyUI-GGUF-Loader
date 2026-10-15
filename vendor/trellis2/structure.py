# MIT. Vendored from the owner's giga-models sidecar:
#   pipelines/trellis2_v1/trellis2/pipelines/trellis2_image_to_3d.py
#   Trellis2ImageTo3DPipeline.sample_sparse_structure (the fill_holes branch)
"""Sparse-structure hole filling + shell extraction, as the sidecar runs it.

The sidecar's TRELLIS.2 engine samples the structure with ``fill_holes=True,
hole_structure=1, hole_iterations=1, hole_fill_algorithm="remove_small_holes",
keep_only_shell=True``; ComfyUI core's ``Trellis2ShapeStage`` takes the decoded
occupancy as is. This is that branch, unchanged apart from taking and
returning a boolean ``[B, R, R, R]`` array instead of living inside the
sampler: count enclosed holes, fill them with the chosen algorithm, then keep
only voxels within two layers of the surface (two 3x3x3 erosions).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

ALGORITHMS = ("remove_small_holes", "morphological_closing", "flood_fill")


def _count_holes(filled, border_mask):
    from scipy.ndimage import label
    inv = ~filled
    labeled, _ = label(inv)
    border_labels = np.unique(labeled[border_mask == 1])
    holes = np.isin(labeled, border_labels, invert=True) & (labeled > 0)
    return np.unique(labeled[holes]).size


def fill_holes(arr: np.ndarray, hole_fill_algorithm: str = "remove_small_holes",
               hole_structure: int = 1, hole_iterations: int = 1,
               keep_only_shell: bool = True) -> np.ndarray:
    """``arr`` bool ``[B, R, R, R]`` -> same shape, holes filled (+ shell only)."""
    from scipy.ndimage import binary_closing, binary_erosion, binary_fill_holes, label

    closed = np.zeros_like(arr)
    for b in range(arr.shape[0]):
        border_mask = np.zeros_like(arr[b])
        border_mask[0, :, :] = border_mask[-1, :, :] = 1
        border_mask[:, 0, :] = border_mask[:, -1, :] = 1
        border_mask[:, :, 0] = border_mask[:, :, -1] = 1
        n_holes = _count_holes(arr[b].astype(np.bool_), border_mask)
        if hole_fill_algorithm == "morphological_closing":
            closed[b] = binary_closing(arr[b], structure=np.ones((hole_structure,) * 3),
                                       iterations=hole_iterations)
        elif hole_fill_algorithm == "flood_fill":
            # closing -> fill internal holes -> keep the largest connected component
            closed1 = binary_closing(arr[b], structure=np.ones((hole_structure,) * 3),
                                     iterations=hole_iterations)
            filled = binary_fill_holes(closed1)
            labeled, num = label(filled)
            if num > 0:
                sizes = np.bincount(labeled.ravel())
                sizes[0] = 0
                closed[b] = (labeled == sizes.argmax())
            else:
                closed[b] = filled
        elif hole_fill_algorithm == "remove_small_holes":
            # by area, per slice along the first axis
            from skimage.morphology import remove_small_holes
            temp = np.copy(arr[b])
            for z in range(temp.shape[0]):
                temp[z] = remove_small_holes(temp[z].astype(bool), area_threshold=hole_structure ** 2)
            closed[b] = temp
        else:
            raise ValueError(f"unknown hole_fill_algorithm {hole_fill_algorithm!r}")
        n_holes2 = _count_holes(closed[b].astype(np.bool_), border_mask)
        log.info("[Sparse HoleFill] batch %d: %d holes filled, %d remain", b, n_holes - n_holes2,
                 n_holes2)

        if keep_only_shell:
            filled = closed[b].astype(np.bool_)
            struct = np.ones((3, 3, 3), dtype=bool)
            # survivors of two erosions are >= 2 layers deep; thin parts are kept
            eroded = binary_erosion(filled, structure=struct, border_value=0)
            eroded = binary_erosion(eroded, structure=struct, border_value=0)
            closed[b] = filled & ~eroded
    if not closed.any(axis=(1, 2, 3)).all():
        raise RuntimeError("No voxels remain after hole filling/shell extraction. The mask is "
                           "empty. Adjust your input, mask, or hole filling parameters.")
    return closed
