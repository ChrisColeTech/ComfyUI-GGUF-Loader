# MIT (VAST-AI-Research/TripoSG). Stage split of the owner's POC
# ``pipelines/triposg_v1/pipelines/img2mesh.py`` (TripoSGPipeline.__call__).
"""TripoSG inference stages: image tokens -> shape latents -> mesh.

``TripoSGPipeline.__call__`` in the POC runs three stages back to back; they
are kept here as three functions with the POC's code and defaults unchanged so
ComfyUI can run (and cache) them as separate nodes:

* :func:`encode_image` - BitImageProcessor + DINOv2, complete last hidden state,
  zero tokens as the unconditional branch.
* :func:`sample_latents` - seeded Gaussian ``[B, num_tokens, 64]`` tokens, the
  RectifiedFlowScheduler, CFG batched unconditional-first and combined as
  ``uncond + scale * (cond - uncond)``.
* :func:`extract_mesh` - the VAE occupancy field sampled hierarchically
  (dense ``2**dense`` grid, banded refinement to ``2**hierarchical``) and
  Lewiner marching cubes, validated exactly like ``_validate_meshes``.

The one difference: no ``trimesh``. The POC wraps the result in a
``trimesh.Trimesh`` to validate and export it; this returns the numpy
``(vertices, faces)`` pair for ComfyUI's MESH type, with the same validation and
the same implicit vertex merge (:func:`merge_vertices`), so the arrays equal
the POC's (checked bit-for-bit on CPU against the POC pipeline, AGENTS.md).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .ops.geometry import hierarchical_extract_geometry

# The POC pipeline's default extraction bounds.
DEFAULT_BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)


@torch.no_grad()
def encode_image(image_encoder, feature_extractor, images, device, num_images_per_prompt=1):
    """Conditional DINOv2 tokens and the matching zero (unconditional) tokens."""
    pixel_values = feature_extractor(images, return_tensors="pt").pixel_values
    try:
        dtype = next(image_encoder.parameters()).dtype
    except StopIteration:
        dtype = torch.float32
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    image_embeds = image_encoder(pixel_values).last_hidden_state
    image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    return image_embeds, torch.zeros_like(image_embeds)


@torch.no_grad()
def sample_latents(
    transformer,
    scheduler,
    image_embeds: torch.Tensor,
    negative_image_embeds: torch.Tensor,
    *,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    num_tokens: int = 2048,
    generator: Optional[torch.Generator] = None,
    device=None,
    step_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """The POC's denoising loop, unchanged; returns the final shape latents."""
    do_cfg = guidance_scale > 1
    batch_size = image_embeds.shape[0]
    if do_cfg:
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

    scheduler.set_timesteps(num_inference_steps, device=device)
    schedule = scheduler.timesteps

    shape = (batch_size, num_tokens, transformer.config.in_channels)
    latents = torch.randn(shape, generator=generator, device=device, dtype=image_embeds.dtype)

    total = len(schedule)
    for step_index, timestep_value in enumerate(schedule):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep = timestep_value.expand(latent_model_input.shape[0])
        noise_pred = transformer(
            latent_model_input,
            timestep,
            encoder_hidden_states=image_embeds,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_image - noise_pred_uncond
            )
        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype and torch.backends.mps.is_available():
            latents = latents.to(latents_dtype)
        if step_callback is not None:
            step_callback(step_index + 1, total)
    return latents


def validate_geometry(vertices, faces, index: int = 0):
    """``TripoSGPipeline._validate_meshes`` for one sample, minus trimesh."""
    if vertices is None or faces is None:
        raise RuntimeError(f"TripoSG geometry extraction failed for sample {index}")
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"TripoSG sample {index} has invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError(f"TripoSG sample {index} has invalid faces")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"TripoSG sample {index} has non-finite vertices")
    if not np.issubdtype(faces.dtype, np.integer):
        raise RuntimeError(f"TripoSG sample {index} has non-integer faces")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError(f"TripoSG sample {index} has out-of-range faces")
    vertices, faces = merge_vertices(vertices.astype(np.float32), faces)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    if extents.shape != (3,) or np.any(extents <= 0):
        raise RuntimeError(f"TripoSG sample {index} has non-positive extents")
    return vertices, np.ascontiguousarray(faces)


def merge_vertices(vertices: np.ndarray, faces: np.ndarray, digits: int = 8):
    """What ``trimesh.Trimesh(vertices, faces)`` (``process=True``) does to the POC's
    mesh: drop unreferenced vertices and merge positions equal to ``digits``
    decimals (``tol.merge = 1e-8``), keeping first-occurrence order - so the
    output is the POC's vertex/face arrays exactly, without trimesh."""
    referenced = np.zeros(len(vertices), dtype=bool)
    referenced[faces] = True
    keys = (vertices.astype(np.float64) * (10 ** digits)).round().astype(np.int64)[referenced]
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    remap = np.zeros(len(vertices), dtype=np.int64)
    remap[referenced] = rank[inverse.reshape(-1)]
    mask = np.nonzero(referenced)[0][first[order]]
    return vertices[mask], remap[faces]


@torch.no_grad()
def extract_mesh(
    vae,
    latents: torch.Tensor,
    device,
    *,
    bounds: Sequence[float] = DEFAULT_BOUNDS,
    dense_octree_depth: int = 8,
    hierarchical_octree_depth: int = 9,
    index: int = 0,
):
    """One latent ``[1, T, 64]`` -> validated ``(vertices [N,3], faces [M,3])``."""

    def geometric_func(points: torch.Tensor) -> torch.Tensor:
        return vae.decode(latents, sampled_points=points).sample

    samples = hierarchical_extract_geometry(
        geometric_func,
        device,
        bounds=bounds,
        dense_octree_depth=dense_octree_depth,
        hierarchical_octree_depth=hierarchical_octree_depth,
    )
    if not samples:
        raise RuntimeError("TripoSG extraction did not produce geometry")
    vertices, faces = samples[0]
    return validate_geometry(vertices, faces, index)
