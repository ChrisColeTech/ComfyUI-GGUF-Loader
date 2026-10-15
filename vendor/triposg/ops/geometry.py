"""Device-neutral hierarchical geometry extraction for TripoSG.

Pure-PyTorch port of ``triposg.inference_utils.hierarchical_extract_geometry``
plus the helpers it composes (``generate_dense_grid_points_gpu``,
``find_candidates_band``, ``expand_edge_region_fast``, ``parallel_zoom``).

The algorithm is unchanged from upstream:

1. Dense query a ``2**dense_octree_depth`` occupancy grid.
2. Upsample it by ``2`` to ``2**hierarchical_octree_depth`` and replace the
   band of near-surface voxels with refined logits from the geometry
   function. The band is found on the low-resolution grid, dilated by a
   max-pool kernel, and upsampled by a factor of two.
3. Run ``skimage.measure.marching_cubes`` (Lewiner) on the refined grid at
   isosurface ``0``.

Two upstream device-neutrality bugs are fixed:

- ``expand_edge_region_fast`` no longer hardcodes ``device='cuda'``; it uses
  ``edge_coords.device``.
- The fancy-index assignment no longer casts indices to ``torch.IntTensor``
  (which forces CPU) - it keeps them on the active device and only calls
  ``torch.cuda.empty_cache()`` when CUDA is actually available.
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def generate_dense_grid_points(
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    octree_depth: int,
    indexing: str = "ij",
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
    """Linspace a cubic grid at resolution ``2**octree_depth``.

    Args:
        bbox_min: Per-axis lower bound, shape ``(3,)``.
        bbox_max: Per-axis upper bound, shape ``(3,)``.
        octree_depth: Base-2 exponent of the per-axis resolution.
        indexing: ``meshgrid`` indexing convention (matches upstream's ``"ij"``).
        dtype: Grid dtype. Defaults to ``torch.float16`` to match upstream.

    Returns:
        ``xyz`` of shape ``(num_cells**3, 3)``, ``grid_size`` list (3x
        ``num_cells``), and ``length`` per-axis extent.
    """
    length = bbox_max - bbox_min
    num_cells = 2 ** octree_depth
    device = bbox_min.device

    x = torch.linspace(
        float(bbox_min[0]), float(bbox_max[0]), int(num_cells), dtype=dtype, device=device
    )
    y = torch.linspace(
        float(bbox_min[1]), float(bbox_max[1]), int(num_cells), dtype=dtype, device=device
    )
    z = torch.linspace(
        float(bbox_min[2]), float(bbox_max[2]), int(num_cells), dtype=dtype, device=device
    )

    xs, ys, zs = torch.meshgrid(x, y, z, indexing=indexing)
    xyz = torch.stack((xs, ys, zs), dim=-1)
    xyz = xyz.view(-1, 3)
    grid_size = [int(num_cells), int(num_cells), int(num_cells)]
    return xyz, grid_size, length


def find_candidates_band(
    occupancy_grid: torch.Tensor,
    band_threshold: float,
    n_limits: int = -1,
) -> torch.Tensor:
    """Return voxel coords whose ``|sigmoid(logit)*2 - 1|`` is below threshold.

    Args:
        occupancy_grid: 3D tensor of logits covering the full padded grid.
        band_threshold: Maximum absolute SDF magnitude for inclusion.
        n_limits: Optional cap on returned voxels (-1 disables the cap).

    Returns:
        ``(N, 3)`` integer coordinates within the band, offset by ``+1`` so
        they index back into the padded grid.
    """
    core_grid = occupancy_grid[1:-1, 1:-1, 1:-1]
    # logits -> sdf, matching upstream
    core_grid = torch.sigmoid(core_grid) * 2 - 1
    in_band = torch.abs(core_grid) < band_threshold
    core_mesh_coords = torch.nonzero(in_band, as_tuple=False) + 1

    if n_limits != -1 and core_mesh_coords.shape[0] > n_limits:
        ind = np.random.choice(core_mesh_coords.shape[0], n_limits, True)
        core_mesh_coords = core_mesh_coords[ind]

    return core_mesh_coords


def expand_edge_region_fast(
    edge_coords: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Dilate band voxels and upsample to the next octree depth.

    Matches ``triposg.inference_utils.expand_edge_region_fast`` while being
    device-neutral: the intermediate volume inherits ``edge_coords.device``
    instead of a hardcoded ``cuda`` device.

    Args:
        edge_coords: ``(N, 3)`` integer coordinates on the low-resolution grid.
        grid_size: Per-axis size of the low-resolution grid.

    Returns:
        ``(M, 3)`` integer coordinates on the high-resolution (2x) grid.
    """
    if edge_coords.numel() == 0:
        # No band: keep the device/dtype contract stable for callers.
        return edge_coords.new_zeros((0, 3), dtype=torch.int16)

    device = edge_coords.device
    expanded_tensor = torch.zeros(
        grid_size, grid_size, grid_size,
        device=device, dtype=torch.float16, requires_grad=False,
    )
    expanded_tensor[edge_coords[:, 0], edge_coords[:, 1], edge_coords[:, 2]] = 1

    if grid_size < 512:
        kernel_size, padding = 5, 2
    else:
        kernel_size, padding = 3, 1
    pooled_tensor = F.max_pool3d(
        expanded_tensor.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size, stride=1, padding=padding,
    ).squeeze()
    expanded_coords_low_res = torch.nonzero(pooled_tensor, as_tuple=False).to(torch.int16)

    # Each low-res voxel fans out into 8 high-res voxels (2 per axis).
    base = expanded_coords_low_res * 2
    ones = base + 1

    expanded_coords_high_res = torch.stack(
        [
            torch.cat(
                (base[:, 0], base[:, 0], base[:, 0], base[:, 0],
                 ones[:, 0], ones[:, 0], ones[:, 0], ones[:, 0])
            ),
            torch.cat(
                (base[:, 1], base[:, 1], ones[:, 1], ones[:, 1],
                 base[:, 1], base[:, 1], ones[:, 1], ones[:, 1])
            ),
            torch.cat(
                (base[:, 2], ones[:, 2], base[:, 2], ones[:, 2],
                 base[:, 2], ones[:, 2], base[:, 2], ones[:, 2])
            ),
        ],
        dim=1,
    )
    return expanded_coords_high_res


def parallel_zoom(
    occupancy_grid: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """Trilinear upsample via ``F.interpolate``.

    Matches ``triposg.inference_utils.parallel_zoom`` exactly: input is
    ``(D, D, D)``, output is ``(D*scale, D*scale, D*scale)``.
    """
    result = F.interpolate(
        occupancy_grid.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor
    )
    return result.squeeze(0).squeeze(0)


def _empty_cache_if_cuda() -> None:
    """Mirror upstream's ``torch.cuda.empty_cache()`` calls when CUDA is usable."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def hierarchical_extract_geometry(
    geometric_func: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    bounds: Union[Tuple[float, ...], List[float], float] = (
        -1.25, -1.25, -1.25, 1.25, 1.25, 1.25,
    ),
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9,
) -> List[Tuple]:
    """Dense + banded hierarchical refinement followed by marching cubes.

    Args:
        geometric_func: Maps ``(1, N, 3)`` points to ``(1, N, 1)`` logits.
        device: Device on which to evaluate ``geometric_func``.
        bounds: Either a scalar broadcast to ``(-b, -b, -b, b, b, b)`` or the
            6-element ``(xmin, ymin, zmin, xmax, ymax, zmax)`` bbox.
        dense_octree_depth: Base-2 exponent of the dense evaluation grid.
        hierarchical_octree_depth: Base-2 exponent of the refined output grid
            (must be ``>= dense_octree_depth``).

    Returns:
        A single-element list of ``(vertices, faces)`` tuples as expected by
        :class:`TripoSGPipeline` consumers. ``vertices`` and ``faces`` are
        ``None`` if marching cubes fails.
    """
    if isinstance(bounds, (int, float)):
        b = float(bounds)
        bounds = [-b, -b, -b, b, b, b]
    if hierarchical_octree_depth < dense_octree_depth:
        raise ValueError(
            "hierarchical_octree_depth must be >= dense_octree_depth"
        )

    bbox_min = torch.tensor(bounds[0:3], device=device)
    bbox_max = torch.tensor(bounds[3:6], device=device)
    bbox_size = bbox_max - bbox_min

    xyz_samples, grid_size, _ = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=dense_octree_depth,
        indexing="ij",
    )

    grid_logits = (
        geometric_func(xyz_samples.unsqueeze(0))
        .to(torch.float16)
        .view(grid_size[0], grid_size[1], grid_size[2])
    )

    for i in range(hierarchical_octree_depth - dense_octree_depth):
        curr_octree_depth = dense_octree_depth + i + 1
        # ``grid_size`` and ``normalize_offset`` refer to the NEXT depth so
        # that normalized coordinates in ``[-bounds, bounds]`` map onto the
        # upsampled grid [0, 2*grid_size).
        grid_size = 2 ** curr_octree_depth
        normalize_offset = grid_size / 2
        high_res_occupancy = parallel_zoom(grid_logits, 2)

        band_threshold = 1.0
        edge_coords = find_candidates_band(grid_logits, band_threshold)
        expanded_coords = expand_edge_region_fast(
            edge_coords, grid_size=int(grid_size / 2)
        ).to(torch.float16)
        expanded_coords_norm = (
            (expanded_coords - normalize_offset)
            * (abs(float(bounds[0])) / normalize_offset)
        )

        all_logits = geometric_func(expanded_coords_norm.unsqueeze(0)).to(torch.float16)
        all_logits = torch.cat([expanded_coords_norm, all_logits[0]], dim=1)

        indices = all_logits[..., :3]
        indices = indices * (normalize_offset / abs(float(bounds[0]))) + normalize_offset
        # Device-neutral integer cast (upstream's torch.IntTensor forces CPU).
        indices = indices.to(torch.int32)
        values = all_logits[:, 3]
        high_res_occupancy[indices[:, 0], indices[:, 1], indices[:, 2]] = values
        grid_logits = high_res_occupancy
        _empty_cache_if_cuda()

    mesh_v_f: Tuple = (None, None)
    try:
        from skimage import measure

        grid_np = grid_logits.float().cpu().numpy()
        vertices, faces, _, _ = measure.marching_cubes(grid_np, 0, method="lewiner")
        bbox_min_np = bbox_min.cpu().numpy()
        bbox_size_np = bbox_size.cpu().numpy()
        vertices = vertices / (2 ** hierarchical_octree_depth) * bbox_size_np + bbox_min_np
        mesh_v_f = (vertices.astype(np.float32), np.ascontiguousarray(faces))
    except Exception as exc:  # pragma: no cover - mirrors upstream error handling
        print(f"hierarchical_extract_geometry marching cubes failed: {exc}")
        _empty_cache_if_cuda()
        mesh_v_f = (None, None)

    return [mesh_v_f]
