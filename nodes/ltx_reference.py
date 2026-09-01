"""LTX reference-token prefix injection + Best-Face-ID identity reinforcer.

Ported 2026-08-31, at the user's request, from their own dev pack
D:\\Projects\\ComfyUI\\10s-comfy-nodes (files ltx_reference_enable.py,
ltx_reference_conditioning.py, ltx_face_identity_reinforcer.py, plus the
face-detection chain from latent_likeness_guide.py). That pack carries no
LICENSE file and is the user's own work (its CLAUDE.md documents it as their
private node lab), so the port into this public repo is by-owner. The RoPE
source-phase convention implemented here follows the published spec of the
Alissonerdx/LTX-Best-Face-ID LoRA's creator (credited in the source pack).

THE MECHANISM (genuinely different from this pack's ic_lora guide append):

comfy-core's ``LTXVAddGuide`` (what LTXV23VidToVideo/LTXV25VidToVideo
delegate to) appends guide LATENT FRAMES to the latent before sampling -
the sampler sees a longer latent, keyframe_idxs mark the appends, and a
crop node strips them after. This module instead prepends REFERENCE TOKENS
inside the transformer's own forward pass:

  1. ``_process_input`` (patched, per-instance): the attached reference
     latent is patchified through the model's own patchifier and its tokens
     are concatenated in FRONT of the video token sequence. Coordinates are
     the TARGET'S OWN coords ("overlap" layout - reference tokens sit at
     identical T/H/W RoPE positions as frame 0) or shifted to precede it
     (``prefix_continuous``).
  2. ``_prepare_timestep`` (patched): the adaLN modulation tensors are
     extended so the prefixed sequence length matches - reference tokens
     inherit frame 0's modulation rows (optionally zeroed =
     "clean reference" sigma).
  3. ``_prepare_positional_embeddings`` (patched): optional per-rotary-dim
     PHASE ROTATION on the first ref_len video self-attn positions -
     ``extra_angle(d) = source_id * phase_scale * theta**(-2d/L)`` - the
     Best-Face-ID LoRA's trained source tag (source_id=2, phase_scale=1).
     source_id=0 is a bitwise no-op.
  4. ``patchifier.unpatchify`` (wrapped, per-instance): the prefix is
     stripped from the output tokens, so the sampler never sees a shape
     change. No crop node exists or is needed - the latent the sampler
     holds is untouched.

Because the injection happens inside the forward, it composes with ANY
sampler and with this pack's v2v guide path (the appended guide frames are
part of the target sequence the reference is prefixed to) - though that
combination is untested on GPU; see AGENTS.md.

Deviation from the source pack, on purpose: the source patched
``LTXAVModel`` at CLASS level (every LTX-AV model in the process). This
port installs the same three method overrides as BOUND METHODS on the one
diffusion_model instance being wired (this repo's clone-then-patch
discipline; comfy model clones share that instance, which is exactly the
scope the source's activation-gating assumed). All patched paths are exact
passthroughs when no reference latent is attached.

The mechanism is version-agnostic within the LTX-AV family: it reads every
shape at runtime, targets ``LTXAVModel`` (which both the 2.3 and 2.5
checkpoints load as - see LTXV23/25ModelsLoader's image_model=="ltxav"
checks), and assumes only the 128-channel /32-spatial VAE geometry both
share. Trained-LoRA compatibility is a separate question: Best-Face-ID was
trained on 2.3.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The mechanism is version-agnostic (any ltxav model), but the pack's menus
# are per-version and every title there is version-prefixed - so each node
# registers twice, once under each family menu, sharing one implementation.
# (A node id carries exactly one CATEGORY in comfy; the second listing is a
# thin subclass with its own id, same precedent as the pack's other aliases.)
LTX23_REF_CATEGORY = "\U0001F916 CCTech/LTX-2.3"
LTX25_REF_CATEGORY = "\U0001F916 CCTech/LTX-2.5"

# ── module state (mirrors the source's, minus the class-patch flags) ────────
_ORIGINAL_PROCESS_INPUT = None
_ORIGINAL_PREPARE_TIMESTEP = None
_ORIGINAL_PREPARE_PE = None
_ORIGINALS_ERROR: Optional[str] = None
_CALL_COUNTER = 0
_VERBOSE = False

_V2_DEBUG_DONE = False
_V2_STRATA_DONE = False
_V2_MULTIREF_DONE = False


def _log(msg: str):
    if _VERBOSE:
        logger.info("[LTX Ref] %s", msg)


def _import_comfy():
    """Lazy import so this module loads even outside Comfy (CPU tests)."""
    import comfy.ldm.lightricks.av_model as av_module
    import comfy.ldm.lightricks.model as model_module
    from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
    return av_module, model_module, latent_to_pixel_coords


def _capture_originals():
    """Capture the unpatched LTXAVModel methods once per process.

    The source pack captured these at class-patch time; we capture the same
    functions but never write to the class. Resolution for
    _prepare_positional_embeddings walks the MRO exactly as the source did
    (LTXAVModel may or may not override the base class version).
    """
    global _ORIGINAL_PROCESS_INPUT, _ORIGINAL_PREPARE_TIMESTEP
    global _ORIGINAL_PREPARE_PE, _ORIGINALS_ERROR

    if _ORIGINAL_PROCESS_INPUT is not None:
        return True
    try:
        av_module, _mm, _fn = _import_comfy()
        LTXAVModel = av_module.LTXAVModel
        _ORIGINAL_PROCESS_INPUT = LTXAVModel._process_input
        _ORIGINAL_PREPARE_TIMESTEP = LTXAVModel._prepare_timestep

        if "_prepare_positional_embeddings" in LTXAVModel.__dict__:
            _ORIGINAL_PREPARE_PE = LTXAVModel.__dict__["_prepare_positional_embeddings"]
            pe_source = LTXAVModel.__name__
        else:
            pe_source = None
            for cls in LTXAVModel.__mro__[1:]:
                if "_prepare_positional_embeddings" in cls.__dict__:
                    _ORIGINAL_PREPARE_PE = cls.__dict__["_prepare_positional_embeddings"]
                    pe_source = cls.__name__
                    break
        if _ORIGINAL_PREPARE_PE is None:
            logger.warning("[LTX Ref] could not find _prepare_positional_embeddings "
                           "anywhere in LTXAVModel's MRO - phase rotation disabled")
        else:
            logger.info("[LTX Ref] originals captured (PE from %s)", pe_source)
        _ORIGINALS_ERROR = None
        return True
    except Exception as e:  # noqa: BLE001 - mirrors source's broad guard
        _ORIGINALS_ERROR = f"{type(e).__name__}: {e}"
        logger.error("[LTX Ref] failed to capture LTXAVModel originals: %s",
                     _ORIGINALS_ERROR)
        return False


# ── RoPE source-phase rotation (verbatim math from the source pack) ─────────

def _compose_source_phase(cos_orig, sin_orig, ref_len, source_id, phase_scale,
                          theta=10000.0):
    """Legacy 4-dim (B,H,T,D_head) cos/sin path."""
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return cos_orig, sin_orig

    B, H, T, D = cos_orig.shape
    device = cos_orig.device
    dtype = cos_orig.dtype
    if ref_len > T:
        ref_len = T

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    cos_extra = extra_angle.cos().to(dtype=dtype).view(1, 1, 1, D)
    sin_extra = extra_angle.sin().to(dtype=dtype).view(1, 1, 1, D)

    cos_ref = cos_orig[:, :, :ref_len, :]
    sin_ref = sin_orig[:, :, :ref_len, :]

    cos_new_ref = cos_ref * cos_extra - sin_ref * sin_extra
    sin_new_ref = cos_ref * sin_extra + sin_ref * cos_extra

    cos_out = cos_orig.clone()
    sin_out = sin_orig.clone()
    cos_out[:, :, :ref_len, :] = cos_new_ref
    sin_out[:, :, :ref_len, :] = sin_new_ref
    return cos_out, sin_out


def _rotate_packed_freq_tensor(freq_tensor, ref_len, source_id, phase_scale,
                               theta=10000.0):
    """Rotate a packed (B, T, H, D_head, 2, 2) frequency tensor's first
    ref_len positions. source_id=0 leaves the tensor bit-identical."""
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return freq_tensor
    if freq_tensor.dim() != 6:
        return freq_tensor

    B, T, H, D, two_a, two_b = freq_tensor.shape
    if two_a != 2 or two_b != 2:
        return freq_tensor
    if ref_len > T:
        ref_len = T

    device = freq_tensor.device
    dtype = freq_tensor.dtype

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    ce = extra_angle.cos().to(dtype=dtype)
    se = extra_angle.sin().to(dtype=dtype)

    ce_b = ce.view(1, 1, 1, D, 1)
    se_b = se.view(1, 1, 1, D, 1)

    ref_slice = freq_tensor[:, :ref_len]
    cos_ref = ref_slice[..., 0, :]
    sin_ref = ref_slice[..., 1, :]

    cos_new = cos_ref * ce_b - sin_ref * se_b
    sin_new = cos_ref * se_b + sin_ref * ce_b

    ref_rotated = torch.stack([cos_new, sin_new], dim=-2)

    result = freq_tensor.clone()
    result[:, :ref_len] = ref_rotated
    return result


def _rotate_freq_tuple(freq_tuple, ref_len, source_id, phase_scale, theta):
    """Dispatch to the correct rotation based on tensor layout."""
    if not isinstance(freq_tuple, tuple) or len(freq_tuple) < 2:
        return freq_tuple
    first = freq_tuple[0]
    if not hasattr(first, "shape"):
        return freq_tuple

    if first.dim() == 6 and first.shape[-1] == 2 and first.shape[-2] == 2:
        rotated = _rotate_packed_freq_tensor(
            first, ref_len, source_id, phase_scale, theta)
        extras = freq_tuple[1:] if len(freq_tuple) > 1 else ()
        return (rotated, *extras)

    if first.dim() == 4 and len(freq_tuple) >= 2 and hasattr(freq_tuple[1], "shape"):
        cos_new, sin_new = _compose_source_phase(
            first, freq_tuple[1], ref_len, source_id, phase_scale, theta)
        extras = freq_tuple[2:] if len(freq_tuple) > 2 else ()
        return (cos_new, sin_new, *extras)

    return freq_tuple


def _apply_multi_source_phase_to_pe(pe, segments, phase_scale, theta=10000.0):
    """Per-segment phase rotation for stacked references."""
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe
    result_pe = pe
    for seg in segments:
        start_pos = seg[0]
        length = seg[1]
        seg_source_id = seg[2]
        if length <= 0 or seg_source_id == 0.0:
            continue
        result_pe = _apply_source_phase_to_pe_ranged(
            result_pe, start_pos, length, seg_source_id, phase_scale, theta)
    return result_pe


def _apply_source_phase_to_pe_ranged(pe, start, length, source_id, phase_scale, theta):
    """Apply rotation to a specific position range [start, start+length]."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return pe
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe

    def rotate_video_selfattn(v_pe):
        if not isinstance(v_pe, tuple) or len(v_pe) < 2:
            return v_pe
        first = v_pe[0]
        if not hasattr(first, "shape"):
            new_halves = []
            for half in v_pe:
                new_halves.append(_rotate_freq_tuple_ranged(
                    half, start, length, source_id, phase_scale, theta))
            return tuple(new_halves)
        return _rotate_freq_tuple_ranged(v_pe, start, length, source_id,
                                         phase_scale, theta)

    pe_video_group = pe[0]
    if (isinstance(pe_video_group, tuple) and len(pe_video_group) == 2
            and isinstance(pe_video_group[0], tuple)):
        v_pe_orig, av_cross_v = pe_video_group[0], pe_video_group[1]
        v_pe_new = rotate_video_selfattn(v_pe_orig)
        new_pe0 = (v_pe_new, av_cross_v)
    else:
        new_pe0 = rotate_video_selfattn(pe_video_group)

    if isinstance(pe, list):
        return [new_pe0] + list(pe[1:])
    return (new_pe0,) + tuple(pe[1:])


def _rotate_freq_tuple_ranged(freq_tuple, start, length, source_id, phase_scale, theta):
    """Range-aware version of _rotate_freq_tuple."""
    if not isinstance(freq_tuple, tuple) or len(freq_tuple) < 2:
        return freq_tuple
    first = freq_tuple[0]
    if not hasattr(first, "shape"):
        return freq_tuple

    if first.dim() == 6 and first.shape[-1] == 2 and first.shape[-2] == 2:
        rotated = _rotate_packed_freq_tensor_ranged(
            first, start, length, source_id, phase_scale, theta)
        extras = freq_tuple[1:] if len(freq_tuple) > 1 else ()
        return (rotated, *extras)

    if first.dim() == 4 and len(freq_tuple) >= 2 and hasattr(freq_tuple[1], "shape"):
        cos_new, sin_new = _compose_source_phase_ranged(
            first, freq_tuple[1], start, length, source_id, phase_scale, theta)
        extras = freq_tuple[2:] if len(freq_tuple) > 2 else ()
        return (cos_new, sin_new, *extras)
    return freq_tuple


def _rotate_packed_freq_tensor_ranged(freq_tensor, start, length, source_id,
                                      phase_scale, theta=10000.0):
    """Range-aware rotation of a packed 6-dim freq tensor."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return freq_tensor
    if freq_tensor.dim() != 6:
        return freq_tensor

    B, T, H, D, two_a, two_b = freq_tensor.shape
    if two_a != 2 or two_b != 2:
        return freq_tensor
    end = min(T, start + length)
    if end <= start:
        return freq_tensor

    device = freq_tensor.device
    dtype = freq_tensor.dtype

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    ce = extra_angle.cos().to(dtype=dtype)
    se = extra_angle.sin().to(dtype=dtype)

    ce_b = ce.view(1, 1, 1, D, 1)
    se_b = se.view(1, 1, 1, D, 1)

    ref_slice = freq_tensor[:, start:end]
    cos_ref = ref_slice[..., 0, :]
    sin_ref = ref_slice[..., 1, :]
    cos_new = cos_ref * ce_b - sin_ref * se_b
    sin_new = cos_ref * se_b + sin_ref * ce_b
    ref_rotated = torch.stack([cos_new, sin_new], dim=-2)

    result = freq_tensor.clone()
    result[:, start:end] = ref_rotated
    return result


def _compose_source_phase_ranged(cos_orig, sin_orig, start, length,
                                 source_id, phase_scale, theta=10000.0):
    """Legacy 4-dim path: range-aware rotation."""
    if length <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return cos_orig, sin_orig
    B, H, T, D = cos_orig.shape
    end = min(T, start + length)
    if end <= start:
        return cos_orig, sin_orig
    device = cos_orig.device
    dtype = cos_orig.dtype

    n_pairs = D // 2
    pair_idx = torch.arange(n_pairs, device=device, dtype=torch.float32)
    rate_per_pair = theta ** (-2.0 * pair_idx / D)
    rate = rate_per_pair.repeat_interleave(2)
    if D % 2 == 1:
        rate = torch.cat([rate, rate_per_pair[-1:].expand(1)], dim=0)

    extra_angle = source_id * phase_scale * rate
    cos_extra = extra_angle.cos().to(dtype=dtype).view(1, 1, 1, D)
    sin_extra = extra_angle.sin().to(dtype=dtype).view(1, 1, 1, D)

    cos_ref = cos_orig[:, :, start:end, :]
    sin_ref = sin_orig[:, :, start:end, :]
    cos_new_ref = cos_ref * cos_extra - sin_ref * sin_extra
    sin_new_ref = cos_ref * sin_extra + sin_ref * cos_extra

    cos_out = cos_orig.clone()
    sin_out = sin_orig.clone()
    cos_out[:, :, start:end, :] = cos_new_ref
    sin_out[:, :, start:end, :] = sin_new_ref
    return cos_out, sin_out


def _apply_source_phase_to_pe(pe, ref_len, source_id, phase_scale, theta=10000.0):
    """Walk the pe structure and rotate video self-attn frequencies only.
    Audio self-attn and AV cross-attn tensors are untouched (reference
    tokens are video-only)."""
    if ref_len <= 0 or source_id == 0.0 or phase_scale == 0.0:
        return pe
    if not (isinstance(pe, (list, tuple)) and len(pe) >= 1):
        return pe

    def rotate_video_selfattn(v_pe):
        if not isinstance(v_pe, tuple) or len(v_pe) < 2:
            return v_pe
        first = v_pe[0]
        if not hasattr(first, "shape"):
            new_halves = []
            for half in v_pe:
                new_halves.append(
                    _rotate_freq_tuple(half, ref_len, source_id, phase_scale, theta))
            return tuple(new_halves)
        return _rotate_freq_tuple(v_pe, ref_len, source_id, phase_scale, theta)

    pe_video_group = pe[0]
    if (isinstance(pe_video_group, tuple) and len(pe_video_group) == 2
            and isinstance(pe_video_group[0], tuple)):
        v_pe_orig, av_cross_v = pe_video_group[0], pe_video_group[1]
        v_pe_new = rotate_video_selfattn(v_pe_orig)
        new_pe0 = (v_pe_new, av_cross_v)
    else:
        new_pe0 = rotate_video_selfattn(pe_video_group)

    if isinstance(pe, list):
        return [new_pe0] + list(pe[1:])
    return (new_pe0,) + tuple(pe[1:])


# ── the three patched methods (bodies mirror the source pack) ───────────────

def _patched_prepare_positional_embeddings(self, pixel_coords, frame_rate, x_dtype):
    """Wrap pe construction with reference-token phase rotation."""
    ref_len = int(getattr(self, "_pending_ref_seq_len", 0) or 0)
    source_id = float(getattr(self, "_pending_source_id", 0.0) or 0.0)
    phase_scale = float(getattr(self, "_pending_phase_scale", 0.0) or 0.0)
    theta = float(getattr(self, "positional_embedding_theta", 10000.0) or 10000.0)
    segments = getattr(self, "_pending_ref_segments", None)

    pe = _ORIGINAL_PREPARE_PE(self, pixel_coords, frame_rate, x_dtype)

    global _V2_DEBUG_DONE, _V2_MULTIREF_DONE
    _first_time = not _V2_DEBUG_DONE

    if ref_len > 0 and phase_scale != 0.0:
        if segments is not None and len(segments) > 1:
            unique_source_ids = set(s[2] for s in segments)
            if len(unique_source_ids) == 1:
                uniform_sid = next(iter(unique_source_ids))
                pe = _apply_source_phase_to_pe(pe, ref_len, uniform_sid,
                                               phase_scale, theta)
                if not _V2_MULTIREF_DONE:
                    _V2_MULTIREF_DONE = True
                    _V2_DEBUG_DONE = True
                    logger.info("[LTX Ref] phase active (multi-ref, uniform sid): "
                                "%d refs, source_id=%s, ref_len=%d, phase_scale=%s",
                                len(segments), uniform_sid, ref_len, phase_scale)
            else:
                pe = _apply_multi_source_phase_to_pe(pe, segments, phase_scale, theta)
                if not _V2_MULTIREF_DONE:
                    _V2_MULTIREF_DONE = True
                    _V2_DEBUG_DONE = True
                    logger.info("[LTX Ref] multi-source phase active: %d refs, "
                                "source_ids=%s, phase_scale=%s",
                                len(segments), [s[2] for s in segments], phase_scale)
        elif source_id != 0.0:
            pe = _apply_source_phase_to_pe(pe, ref_len, source_id, phase_scale, theta)
            if _first_time:
                _V2_DEBUG_DONE = True
                logger.info("[LTX Ref] source phase active: ref_len=%d source_id=%s "
                            "phase_scale=%s theta=%s",
                            ref_len, source_id, phase_scale, theta)
    elif _first_time:
        _V2_DEBUG_DONE = True
        _log(f"phase idle (ref_len={ref_len}, source_id={source_id}, "
             f"phase_scale={phase_scale})")

    return pe


def _patched_process_input(self, x, keyframe_idxs, denoise_mask, **kwargs):
    """Patched LTXAVModel._process_input - injects reference tokens."""
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    call_id = _CALL_COUNTER

    transformer_options = kwargs.get("transformer_options", {}) or {}

    # Priority: kwargs (comfy unpacks transformer_options into kwargs),
    # then the nested dict, then the attribute side-channel.
    reference_latent = kwargs.get("reference_latent")
    if reference_latent is None:
        reference_latent = kwargs.get("memory_video")  # legacy key
    if reference_latent is None and isinstance(transformer_options, dict):
        reference_latent = transformer_options.get("reference_latent")
        if reference_latent is None:
            reference_latent = transformer_options.get("memory_video")
    if reference_latent is None:
        reference_latent = getattr(self, "_ltx_reference_latent", None)
        if reference_latent is None:
            reference_latent = getattr(self, "_echo_memory_video", None)

    position_mode = kwargs.get("reference_position_mode") \
        or kwargs.get("memory_position_mode") \
        or "reference"
    if isinstance(transformer_options, dict) and position_mode == "reference":
        position_mode = transformer_options.get("reference_position_mode") \
            or transformer_options.get("memory_position_mode") \
            or "reference"

    source_id = kwargs.get("reference_source_id")
    if source_id is None and isinstance(transformer_options, dict):
        source_id = transformer_options.get("reference_source_id")
    if source_id is None:
        source_id = 0.0
    source_id = float(source_id)

    phase_scale = kwargs.get("reference_phase_scale")
    if phase_scale is None and isinstance(transformer_options, dict):
        phase_scale = transformer_options.get("reference_phase_scale")
    if phase_scale is None:
        phase_scale = 1.0
    phase_scale = float(phase_scale)

    result = _ORIGINAL_PROCESS_INPUT(self, x, keyframe_idxs, denoise_mask, **kwargs)
    tokens_list, coords_list, additional_args = result

    self._pending_ref_seq_len = 0
    self._pending_source_id = 0.0
    self._pending_phase_scale = 0.0
    self._pending_ref_segments = None

    if reference_latent is None:
        return result

    if reference_latent.dim() != 5:
        _log(f"reference_latent has wrong dim: expected 5D [B,C,F,H,W], "
             f"got {reference_latent.dim()}D shape={tuple(reference_latent.shape)}")
        return result

    vx = tokens_list[0]
    reference_latent = reference_latent.to(device=vx.device, dtype=vx.dtype)

    # Spatial alignment fallback (tiled samplers / shape drift): latent-space
    # bilinear resize to the target's H/W.
    target_orig_shape = additional_args.get("orig_shape")
    if target_orig_shape is not None and len(target_orig_shape) >= 5:
        H_target = int(target_orig_shape[3])
        W_target = int(target_orig_shape[4])
        H_mem = int(reference_latent.shape[3])
        W_mem = int(reference_latent.shape[4])

        if (H_mem, W_mem) != (H_target, W_target):
            # Almost always a two-stage refine pass sampling an upscaled grid.
            # The resize keeps the geometry valid, but re-reading a full-frame
            # face overlay while RE-NOISING nearly-finished content bleeds its
            # colors into the output (verified live: red face-tone blotches on
            # the subject at the x2 refine; clean when the refine runs on the
            # pre-reference model). Identity belongs to the BASE pass - warn.
            _log(f"reference grid {H_mem}x{W_mem} != sampled grid "
                 f"{H_target}x{W_target} - resizing. If this is a refine/"
                 f"second-stage sampler, wire its MODEL from BEFORE the "
                 f"reference/reinforcer node instead: reference attention "
                 f"during a re-noise pass smears face colors onto the video.")
            B, C, F_mem_dim, _, _ = reference_latent.shape
            flat = reference_latent.permute(0, 2, 1, 3, 4).reshape(
                B * F_mem_dim, C, H_mem, W_mem)
            flat = F.interpolate(flat, size=(H_target, W_target),
                                 mode="bilinear", align_corners=False)
            reference_latent = flat.reshape(
                B, F_mem_dim, C, H_target, W_target
            ).permute(0, 2, 1, 3, 4).contiguous()

            if not hasattr(self, "_ltx_ref_seen_mismatches"):
                self._ltx_ref_seen_mismatches = set()
            key = (H_mem, W_mem, H_target, W_target)
            if key not in self._ltx_ref_seen_mismatches:
                self._ltx_ref_seen_mismatches.add(key)
                _log(f"  auto-resized reference latent {H_mem}x{W_mem} -> "
                     f"{H_target}x{W_target} (latent-space fallback; wire "
                     f"target_latent in the Conditioning node for the "
                     f"pixel-space path)")

    # Patchify the reference through the model's own patchifier.
    try:
        ref_tokens, ref_latent_coords = self.patchifier.patchify(reference_latent)
    except Exception as e:  # noqa: BLE001
        _log(f"patchify failed: {type(e).__name__}: {e}")
        return result

    _, _, latent_to_pixel_coords = _import_comfy()
    try:
        ref_pixel_coords = latent_to_pixel_coords(
            latent_coords=ref_latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )
    except Exception as e:  # noqa: BLE001
        _log(f"pixel coords failed: {type(e).__name__}: {e}")
        return result

    # Overlap layout by default (Best-Face-ID convention): reference tokens
    # keep the target's own coordinate grid. prefix modes shift them to
    # precede the target temporally.
    if position_mode == "prefix_continuous" or position_mode == "prefix":
        try:
            ref_temporal_end = float(ref_pixel_coords[:, 0, :, 1].max().item())
            ref_pixel_coords = ref_pixel_coords.clone()
            ref_pixel_coords[:, 0, :, :] -= ref_temporal_end
            _log("prefix mode: shifted reference to precede target")
        except Exception as e:  # noqa: BLE001
            _log(f"prefix offset failed: {type(e).__name__}: {e}")
    else:
        _log(f"overlap layout: reference at same coords as target "
             f"(source_id={source_id}, phase_scale={phase_scale})")

    try:
        ref_tokens = self.patchify_proj(ref_tokens)
    except Exception as e:  # noqa: BLE001
        _log(f"patchify_proj failed: {type(e).__name__}: {e}")
        return result

    # CFG/tiling batch broadcast.
    if ref_tokens.shape[0] != vx.shape[0]:
        if ref_tokens.shape[0] == 1:
            ref_tokens = ref_tokens.expand(vx.shape[0], -1, -1)
            ref_pixel_coords = ref_pixel_coords.expand(vx.shape[0], -1, -1, -1)
        else:
            _log(f"  batch mismatch: ref batch {ref_tokens.shape[0]}, vx batch "
                 f"{vx.shape[0]}, neither is 1 - skipping injection this call.")
            return result

    vx_combined = torch.cat([ref_tokens, vx], dim=1)
    tokens_list[0] = vx_combined

    v_pixel_coords = coords_list[0]
    coords_list[0] = torch.cat([ref_pixel_coords, v_pixel_coords], dim=2)

    ref_seq_len = ref_tokens.shape[1]
    ref_frames = int(reference_latent.shape[2])
    target_seq_len = int(vx.shape[1])
    spatial = max(1, ref_seq_len // max(1, ref_frames))
    target_frames = max(1, target_seq_len // spatial)

    additional_args["reference_seq_len"] = ref_seq_len
    additional_args["reference_frames"] = ref_frames
    additional_args["target_seq_len"] = target_seq_len
    additional_args["target_frames"] = target_frames
    self._pending_ref_seq_len = ref_seq_len
    self._pending_source_id = source_id
    self._pending_phase_scale = phase_scale
    self._pending_ref_frames = ref_frames
    # Multi-ref: UNIFORM source_id on every segment (Best-Face-ID was
    # trained single-ref; distinct per-ref ids land in untrained territory).
    if ref_frames > 1:
        self._pending_ref_segments = [(i * spatial, spatial, source_id, i)
                                      for i in range(ref_frames)]
        global _V2_STRATA_DONE
        if not _V2_STRATA_DONE:
            _V2_STRATA_DONE = True
            logger.info("[LTX Ref] multi-ref: %d refs x %d tokens each, all at "
                        "source_id=%s (uniform)", ref_frames, spatial, source_id)
    else:
        self._pending_ref_segments = None

    _log(f"Prepending {ref_seq_len} ref tokens (target was {target_seq_len}, "
         f"now {vx_combined.shape[1]}, F_ref={ref_frames}, "
         f"F_tgt~{target_frames}) [call #{call_id}]")

    return tokens_list, coords_list, additional_args


def _extend_prefix_in_tensor(t: torch.Tensor, target_size: int,
                             prefix_size: int) -> torch.Tensor:
    """Extend dim 1 by replicating row 0 prefix_size times at the front."""
    if not isinstance(t, torch.Tensor) or t.dim() < 2 or t.shape[1] != target_size:
        return t
    prefix = t[:, 0:1, ...].expand(-1, prefix_size,
                                   *([t.shape[i] for i in range(2, t.dim())]))
    return torch.cat([prefix, t], dim=1)


def _walk_and_extend_item(obj, target_seq_len, ref_seq_len,
                          target_frames, ref_frames,
                          zero_ref_timesteps, depth=0):
    """Walk a timestep object, extending tensors to include the reference
    prefix (CompressedTimestep-aware; see the source pack's docstring)."""
    if depth > 5 or obj is None:
        return obj, 0, 0

    if isinstance(obj, list):
        ext, zer = 0, 0
        for i, item in enumerate(obj):
            new_item, e, z = _walk_and_extend_item(
                item, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_ref_timesteps, depth + 1)
            obj[i] = new_item
            ext += e
            zer += z
        return obj, ext, zer

    if isinstance(obj, tuple):
        ext, zer = 0, 0
        for item in obj:
            _, e, z = _walk_and_extend_item(
                item, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_ref_timesteps, depth + 1)
            ext += e
            zer += z
        return obj, ext, zer

    if (hasattr(obj, "data") and hasattr(obj, "num_frames")
            and hasattr(obj, "patches_per_frame")):
        try:
            data = obj.data
            num_frames = obj.num_frames
            patches_per_frame = obj.patches_per_frame

            if not isinstance(data, torch.Tensor) or data.dim() < 2:
                return obj, 0, 0

            if patches_per_frame == 1 and num_frames == 1:
                return obj, 0, 0

            if patches_per_frame > 1 and num_frames * patches_per_frame == target_seq_len:
                prefix = data[:, 0:1, :].expand(-1, ref_frames, -1).contiguous()
                if zero_ref_timesteps:
                    prefix = torch.zeros_like(prefix)
                new_data = torch.cat([prefix, data], dim=1).contiguous()
                obj.data = new_data
                obj.num_frames = num_frames + ref_frames
                _log(f"      extended CompressedTimestep: num_frames "
                     f"{num_frames} -> {obj.num_frames}")
                return obj, 1, (1 if zero_ref_timesteps else 0)

            if patches_per_frame == 1 and num_frames == target_seq_len:
                prefix = data[:, 0:1, :].expand(-1, ref_seq_len, -1).contiguous()
                if zero_ref_timesteps:
                    prefix = torch.zeros_like(prefix)
                new_data = torch.cat([prefix, data], dim=1).contiguous()
                obj.data = new_data
                obj.num_frames = num_frames + ref_seq_len
                _log(f"      extended CompressedTimestep (uncompressed): "
                     f"num_frames {num_frames} -> {obj.num_frames}")
                return obj, 1, (1 if zero_ref_timesteps else 0)
        except Exception as e:  # noqa: BLE001
            _log(f"      couldn't extend CompressedTimestep: "
                 f"{type(e).__name__}: {e}")
        return obj, 0, 0

    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 2:
            size = obj.shape[1]
            if size == target_seq_len:
                new_obj = _extend_prefix_in_tensor(obj, target_seq_len, ref_seq_len)
                if zero_ref_timesteps:
                    new_obj = new_obj.clone()
                    new_obj[:, :ref_seq_len] = 0.0
                return new_obj, 1, (1 if zero_ref_timesteps else 0)
            elif size == target_frames:
                new_obj = _extend_prefix_in_tensor(obj, target_frames, ref_frames)
                if zero_ref_timesteps:
                    new_obj = new_obj.clone()
                    new_obj[:, :ref_frames] = 0.0
                return new_obj, 1, (1 if zero_ref_timesteps else 0)
        return obj, 0, 0

    return obj, 0, 0


def _patched_prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
    """Extend adaLN modulation tensors to match the prepended reference."""
    ref_seq_len = int(kwargs.get("reference_seq_len", 0) or 0)

    if ref_seq_len == 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size,
                                          hidden_dtype, **kwargs)

    ref_frames = int(kwargs.get("reference_frames", 0) or 0)
    if ref_frames == 0:
        ref_frames = 1

    target_seq_len = int(kwargs.get("target_seq_len", 0) or 0)
    target_frames = int(kwargs.get("target_frames", 0) or 0)

    if target_seq_len == 0:
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size,
                                          hidden_dtype, **kwargs)

    zero_enabled = bool(getattr(self, "_ltx_zero_ref_timesteps", False))

    _log(f"_prepare_timestep ref_seq={ref_seq_len} ref_f={ref_frames} "
         f"tgt_seq={target_seq_len} tgt_f={target_frames}")

    result = _ORIGINAL_PREPARE_TIMESTEP(self, timestep, batch_size,
                                        hidden_dtype, **kwargs)

    if not isinstance(result, (tuple, list)):
        _log(f"  result is {type(result).__name__}, not iterable - skipping")
        return result

    was_tuple = isinstance(result, tuple)
    result_list = list(result) if was_tuple else result

    ext_total, zer_total = 0, 0
    for slot_idx in range(len(result_list)):
        slot = result_list[slot_idx]
        if isinstance(slot, list):
            for i, item in enumerate(slot):
                new_item, e, z = _walk_and_extend_item(
                    item, target_seq_len, ref_seq_len,
                    target_frames, ref_frames, zero_enabled, 0)
                slot[i] = new_item
                ext_total += e
                zer_total += z
        elif slot is not None:
            new_slot, e, z = _walk_and_extend_item(
                slot, target_seq_len, ref_seq_len,
                target_frames, ref_frames, zero_enabled, 0)
            result_list[slot_idx] = new_slot
            ext_total += e
            zer_total += z

    if ext_total > 0:
        _log(f"  extended {ext_total} modulation tensor(s)"
             + (f", zeroed {zer_total}" if zer_total > 0 else ""))
    else:
        _log(f"  no modulation tensors matched target sizes ({target_seq_len} "
             f"per-token or {target_frames} per-frame); block forward will "
             f"likely fail at adaLN broadcast.")

    return tuple(result_list) if was_tuple else result_list


# ── output-side strip + per-instance installation ───────────────────────────

def _strip_ref_from_timestep(emb, ref_seq_len, ref_frames):
    """Undo _prepare_timestep's prefix extension on one embedded-timestep
    entry: CompressedTimestep-like objects are trimmed in compressed form
    (the exact mirror of the two extension branches), per-token tensors by
    row slice. Broadcast [B, 1, D] entries are untouched."""
    if (hasattr(emb, "data") and hasattr(emb, "num_frames")
            and hasattr(emb, "patches_per_frame")):
        trim = ref_frames if emb.patches_per_frame > 1 else ref_seq_len
        if trim > 0 and emb.num_frames > trim:
            emb.data = emb.data[:, trim:].contiguous()
            emb.num_frames = emb.num_frames - trim
        return emb
    if isinstance(emb, torch.Tensor) and emb.dim() >= 2 and emb.shape[1] > 1:
        return emb[:, ref_seq_len:]
    return emb


class _ProcessOutputWrapper:
    """Strip the reference prefix from the video tokens BEFORE the original
    _process_output runs.

    The strip used to live in a patchifier.unpatchify wrap - too late: on
    the keyframe/guide path (i2v inplace holds, v2v guide frames)
    _process_output scatters the tokens back through grid_mask
    (``full_x[:, grid_mask, :] = x``) BEFORE unpatchify, and the
    still-prefixed tokens crashed it (found live on the 2.5 i2v graph:
    "value tensor of shape [11160, 128] cannot be broadcast to indexing
    result of shape [1, 10980, 128]" - 180 = the reference prefix). This
    seam is the one comfy itself uses for its reference-AUDIO tokens
    (av_model._process_output trims ref_audio_seq_len at the top)."""

    def __init__(self, original_process_output, model_ref):
        self._original = original_process_output
        self._model_ref = model_ref

    def __call__(self, x, embedded_timestep, keyframe_idxs, **kwargs):
        m = self._model_ref
        ref_seq_len = int(getattr(m, "_pending_ref_seq_len", 0) or 0)
        ref_frames = int(getattr(m, "_pending_ref_frames", 0) or 0) or 1
        if ref_seq_len > 0:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                # AV model: [video_tokens, audio_tokens] - only video carries
                # the reference prefix (comfy trims its ref AUDIO itself).
                vx = x[0][:, ref_seq_len:]
                v_emb = _strip_ref_from_timestep(
                    embedded_timestep[0], ref_seq_len, ref_frames)
                x = [vx, x[1]]
                embedded_timestep = [v_emb, embedded_timestep[1]]
            elif isinstance(x, torch.Tensor):
                x = x[:, ref_seq_len:]
                embedded_timestep = _strip_ref_from_timestep(
                    embedded_timestep, ref_seq_len, ref_frames)
            m._pending_ref_seq_len = 0
        return self._original(x, embedded_timestep, keyframe_idxs, **kwargs)


def _apply_process_output_wrap(model_instance):
    if getattr(model_instance, "_ltx_ref_out_wrapped", False):
        return False
    original = model_instance._process_output
    model_instance._process_output = _ProcessOutputWrapper(original, model_instance)
    model_instance._ltx_ref_out_wrapped = True
    model_instance._ltx_ref_original_process_output = original
    return True


def _install_reference_patches(model):
    """Install the three method overrides on THIS diffusion_model instance.

    Idempotent. Deliberately per-instance (bound-method shadowing) rather
    than the source pack's class-level patch - same behavior for the model
    being wired, zero effect on any other loaded model. Every override is a
    passthrough when no reference latent is attached.
    """
    import types as _types

    if not _capture_originals():
        raise RuntimeError(
            f"[LTX Reference] couldn't resolve LTXAVModel originals: "
            f"{_ORIGINALS_ERROR}")

    try:
        dm = model.model.diffusion_model
    except AttributeError as e:
        raise RuntimeError(
            f"[LTX Reference] couldn't access diffusion_model: {e}")

    if not hasattr(dm, "patchifier") or not hasattr(dm, "patchify_proj"):
        raise ValueError(
            "[LTX Reference] this model is not an LTX-AV diffusion model "
            f"(got {type(dm).__name__}) - load an LTX-2.3/2.5 checkpoint "
            "via the LTX loaders.")

    if not getattr(dm, "_cctech_ltx_ref_installed", False):
        dm._process_input = _types.MethodType(_patched_process_input, dm)
        dm._prepare_timestep = _types.MethodType(_patched_prepare_timestep, dm)
        if _ORIGINAL_PREPARE_PE is not None:
            dm._prepare_positional_embeddings = _types.MethodType(
                _patched_prepare_positional_embeddings, dm)
        dm._cctech_ltx_ref_installed = True
        logger.info("[LTX Ref] instance patches installed on %s",
                    type(dm).__name__)

    _apply_process_output_wrap(dm)
    return dm


# ── face detection (ported from the pack's latent_likeness_guide.py) ────────

_YUNET_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                    "face_detection_yunet/face_detection_yunet_2023mar.onnx")
_YUNET_DETECTOR = None
_YUNET_LOAD_TRIED = False


def _download_yunet_model(target_path: str, debug: bool = False) -> bool:
    """Download the YuNet ONNX face detector (~350 KB)."""
    try:
        import urllib.request
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        if debug:
            logger.info("[face_detect] downloading YuNet model to %s", target_path)
        urllib.request.urlretrieve(_YUNET_MODEL_URL, target_path)
        return True
    except Exception as e:  # noqa: BLE001
        if debug:
            logger.info("[face_detect] YuNet download failed: %s: %s",
                        type(e).__name__, e)
        return False


def _get_yunet_detector(debug: bool = False):
    """Lazy-load OpenCV YuNet face detector, cached across calls."""
    global _YUNET_DETECTOR, _YUNET_LOAD_TRIED
    if _YUNET_DETECTOR is not None:
        return _YUNET_DETECTOR
    if _YUNET_LOAD_TRIED:
        return None
    _YUNET_LOAD_TRIED = True

    try:
        import cv2

        # Reuse the source pack's cached download when present, else our own.
        model_paths = [
            os.path.expanduser("~/.cache/10s_comfy/face_detection_yunet_2023mar.onnx"),
            os.path.expanduser("~/.cache/cctech_comfy/face_detection_yunet_2023mar.onnx"),
        ]
        model_path = None
        for p in model_paths:
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            target = model_paths[1]
            if _download_yunet_model(target, debug=debug):
                model_path = target
            else:
                return None

        _YUNET_DETECTOR = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), 0.5, 0.3, 5000)
        if debug:
            logger.info("[face_detect] YuNet loaded from %s", model_path)
        return _YUNET_DETECTOR
    except Exception as e:  # noqa: BLE001
        if debug:
            logger.info("[face_detect] YuNet load failed: %s: %s",
                        type(e).__name__, e)
        return None


def _detect_face_bbox(image_np, padding=0.15, debug=False):
    """Detect the largest face in an HxWx3 uint8 image; normalized
    (x1,y1,x2,y2) or None. Backend priority: YuNet DNN, MediaPipe,
    OpenCV Haar cascade - the source pack's exact chain."""
    H, W = image_np.shape[:2]

    def _pad_and_return(x1, y1, x2, y2):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half_w = (x2 - x1) / 2 * (1.0 + padding)
        half_h = (y2 - y1) / 2 * (1.0 + padding)
        return (max(0.0, cx - half_w), max(0.0, cy - half_h),
                min(1.0, cx + half_w), min(1.0, cy + half_h))

    try:
        import cv2
        detector = _get_yunet_detector(debug=debug)
        if detector is not None:
            detector.setInputSize((W, H))
            bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            _, faces = detector.detect(bgr)
            if faces is not None and len(faces) > 0:
                best = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = best[0], best[1], best[2], best[3]
                x1, y1 = max(0.0, x / W), max(0.0, y / H)
                x2, y2 = min(1.0, (x + w) / W), min(1.0, (y + h) / H)
                bbox = _pad_and_return(x1, y1, x2, y2)
                if debug:
                    logger.info("[face_detect] YuNet found face: %s", bbox)
                return bbox
            elif debug:
                logger.info("[face_detect] YuNet: no face; trying MediaPipe")
    except Exception as e:  # noqa: BLE001
        if debug:
            logger.info("[face_detect] YuNet error: %s: %s", type(e).__name__, e)

    try:
        import mediapipe as mp
        with mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as detector:
            results = detector.process(image_np)
            if results.detections:
                best = None
                best_area = 0
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    area = box.width * box.height
                    if area > best_area:
                        best_area = area
                        best = box
                if best is not None:
                    x1 = max(0.0, best.xmin)
                    y1 = max(0.0, best.ymin)
                    x2 = min(1.0, best.xmin + best.width)
                    y2 = min(1.0, best.ymin + best.height)
                    bbox = _pad_and_return(x1, y1, x2, y2)
                    if debug:
                        logger.info("[face_detect] MediaPipe found face: %s", bbox)
                    return bbox
    except ImportError:
        if debug:
            logger.info("[face_detect] mediapipe not installed; trying OpenCV")
    except Exception as e:  # noqa: BLE001
        if debug:
            logger.info("[face_detect] MediaPipe error: %s: %s",
                        type(e).__name__, e)

    try:
        import cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            if debug:
                logger.info("[face_detect] OpenCV cascade load failed")
            return None
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            if debug:
                logger.info("[face_detect] OpenCV: no face found")
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        bbox = _pad_and_return(x / W, y / H, (x + w) / W, (y + h) / H)
        if debug:
            logger.info("[face_detect] OpenCV Haar found face: %s", bbox)
        return bbox
    except ImportError:
        if debug:
            logger.info("[face_detect] opencv-python not installed")
        return None
    except Exception as e:  # noqa: BLE001
        if debug:
            logger.info("[face_detect] OpenCV error: %s: %s", type(e).__name__, e)
        return None


# ── image-prep helpers (ported verbatim from the reinforcer) ────────────────

def _pad_image_to_multiple(image_bhwc: torch.Tensor, divisor: int = 32) -> torch.Tensor:
    """Pad (B, H, W, C) so H and W are multiples of divisor."""
    if image_bhwc.dim() != 4:
        raise ValueError(
            f"Expected IMAGE tensor (B, H, W, C), got {tuple(image_bhwc.shape)}")
    B, H, W, C = image_bhwc.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return image_bhwc
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x.permute(0, 2, 3, 1).contiguous()


def _video_latent_samples(target_latent):
    """The VIDEO half of a target latent, whatever shape it arrives in.

    This pack's own prep nodes emit joint AV latents as NestedTensor
    (video, audio) - the source pack predates that and indexed .dim()
    directly, which NestedTensor doesn't implement (found live on GPU:
    'NestedTensor' object has no attribute 'dim'). The reference
    machinery only ever cares about the video stream's spatial grid.
    """
    if isinstance(target_latent, dict) and "samples" in target_latent:
        target_latent = target_latent["samples"]
    if target_latent is not None and getattr(target_latent, "is_nested", False):
        target_latent = target_latent.unbind()[0]
    return target_latent


def _resize_image_to_latent(image_bhwc: torch.Tensor, target_latent,
                            vae_scale: int = 32) -> torch.Tensor:
    """Resize (B,H,W,C) so its VAE latent matches the target's spatial dims."""
    target_latent = _video_latent_samples(target_latent)
    if target_latent.dim() == 5:
        Ht = int(target_latent.shape[3]) * vae_scale
        Wt = int(target_latent.shape[4]) * vae_scale
    elif target_latent.dim() == 4:
        Ht = int(target_latent.shape[2]) * vae_scale
        Wt = int(target_latent.shape[3]) * vae_scale
    else:
        return image_bhwc

    B, H, W, C = image_bhwc.shape
    if (H, W) == (Ht, Wt):
        return image_bhwc
    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(Ht, Wt), mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _align_to_reference_bbox(image_bhwc, face_bbox, target_face_bbox,
                             target_H, target_W, debug=False):
    """Scale/paste an image so its face bbox lands where the primary
    reference's face sits (multi-ref proportion sync). Ported verbatim."""
    B, sH, sW, C = image_bhwc.shape
    fx1, fy1, fx2, fy2 = face_bbox
    tx1, ty1, tx2, ty2 = target_face_bbox

    src_face_w = (fx2 - fx1) * sW
    src_face_h = (fy2 - fy1) * sH
    if src_face_w <= 0 or src_face_h <= 0:
        x = image_bhwc.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(target_H, target_W),
                          mode="bicubic", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), target_face_bbox

    tgt_face_w = (tx2 - tx1) * target_W
    tgt_face_h = (ty2 - ty1) * target_H

    scale_x = tgt_face_w / src_face_w
    scale_y = tgt_face_h / src_face_h
    scale = (scale_x + scale_y) * 0.5

    new_sW = int(round(sW * scale))
    new_sH = int(round(sH * scale))
    if new_sW <= 0 or new_sH <= 0:
        x = image_bhwc.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(target_H, target_W),
                          mode="bicubic", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0), target_face_bbox

    x = image_bhwc.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(new_sH, new_sW), mode="bicubic",
                      align_corners=False)
    x = x.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)

    face_cx_scaled = ((fx1 + fx2) * 0.5) * new_sW
    face_cy_scaled = ((fy1 + fy2) * 0.5) * new_sH
    tgt_cx = ((tx1 + tx2) * 0.5) * target_W
    tgt_cy = ((ty1 + ty2) * 0.5) * target_H

    origin_x = tgt_cx - face_cx_scaled
    origin_y = tgt_cy - face_cy_scaled

    canvas = torch.zeros(B, target_H, target_W, C,
                         dtype=image_bhwc.dtype, device=image_bhwc.device)

    src_x1 = max(0, int(round(-origin_x)))
    src_y1 = max(0, int(round(-origin_y)))
    src_x2 = min(new_sW, int(round(target_W - origin_x)))
    src_y2 = min(new_sH, int(round(target_H - origin_y)))

    dst_x1 = max(0, int(round(origin_x)))
    dst_y1 = max(0, int(round(origin_y)))
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[:, dst_y1:dst_y2, dst_x1:dst_x2, :] = x[:, src_y1:src_y2,
                                                       src_x1:src_x2, :]

    if dst_y1 > 0 and dst_y2 > dst_y1:
        canvas[:, :dst_y1, dst_x1:dst_x2, :] = canvas[:, dst_y1:dst_y1 + 1,
                                                      dst_x1:dst_x2, :]
    if dst_y2 < target_H and dst_y2 > dst_y1:
        canvas[:, dst_y2:, dst_x1:dst_x2, :] = canvas[:, dst_y2 - 1:dst_y2,
                                                      dst_x1:dst_x2, :]
    if dst_x1 > 0:
        canvas[:, :, :dst_x1, :] = canvas[:, :, dst_x1:dst_x1 + 1, :]
    if dst_x2 < target_W:
        canvas[:, :, dst_x2:, :] = canvas[:, :, dst_x2 - 1:dst_x2, :]

    if debug:
        logger.info("[Reinforcer] aligned ref2 to ref1: src %dx%d scaled "
                    "%.2fx -> %dx%d, canvas %dx%d",
                    sW, sH, scale, new_sW, new_sH, target_W, target_H)

    return canvas, target_face_bbox


def _auto_face_crop(image_bhwc, face_bbox, zoom_factor=2.0,
                    target_aspect=None, debug=False):
    """Crop around the detected face with context padding, matching the
    target aspect. Returns (cropped_image, new_face_bbox). Ported verbatim
    (including the invalid-bounds fallback returning the image alone)."""
    B, H, W, C = image_bhwc.shape
    x1, y1, x2, y2 = face_bbox
    fx1_px, fy1_px = x1 * W, y1 * H
    fx2_px, fy2_px = x2 * W, y2 * H
    face_w = fx2_px - fx1_px
    face_h = fy2_px - fy1_px
    fcx = (fx1_px + fx2_px) * 0.5
    fcy = (fy1_px + fy2_px) * 0.5

    if target_aspect is None:
        target_aspect = W / max(H, 1)

    base_h = face_h * zoom_factor
    base_w = face_w * zoom_factor

    if base_w / max(base_h, 1e-6) < target_aspect:
        base_w = base_h * target_aspect
    else:
        base_h = base_w / target_aspect

    cx1 = fcx - base_w * 0.5
    cy1 = fcy - base_h * 0.5
    cx2 = fcx + base_w * 0.5
    cy2 = fcy + base_h * 0.5

    pad_left = max(0.0, -cx1)
    pad_top = max(0.0, -cy1)
    pad_right = max(0.0, cx2 - W)
    pad_bottom = max(0.0, cy2 - H)

    cx1c = int(max(0, cx1))
    cy1c = int(max(0, cy1))
    cx2c = int(min(W, cx2))
    cy2c = int(min(H, cy2))

    if cx2c <= cx1c or cy2c <= cy1c:
        if debug:
            logger.info("[Reinforcer] auto-crop bounds invalid, using full image")
        return image_bhwc

    cropped = image_bhwc[:, cy1c:cy2c, cx1c:cx2c, :]

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        x = cropped.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, (int(pad_left), int(pad_right),
                      int(pad_top), int(pad_bottom)), mode="replicate")
        cropped = x.permute(0, 2, 3, 1).contiguous()

    new_H, new_W = cropped.shape[1], cropped.shape[2]
    new_fx1 = (fx1_px - cx1c) + pad_left
    new_fy1 = (fy1_px - cy1c) + pad_top
    new_fx2 = new_fx1 + face_w
    new_fy2 = new_fy1 + face_h
    new_bbox = (
        max(0.0, new_fx1 / max(new_W, 1)),
        max(0.0, new_fy1 / max(new_H, 1)),
        min(1.0, new_fx2 / max(new_W, 1)),
        min(1.0, new_fy2 / max(new_H, 1)),
    )

    if debug:
        logger.info("[Reinforcer] auto-crop: %dx%d -> %dx%d, new bbox %s",
                    W, H, new_W, new_H, new_bbox)

    return cropped, new_bbox


def _make_face_mask_latent(face_bbox, latent_shape, gating_mode="mask_soft",
                           dilation=0.10):
    """Latent-space face mask: 1 inside the (dilated) bbox, cosine falloff
    to 0 by 1.5x outside (soft) or binary (hard). None for 'off'/no bbox."""
    if gating_mode == "off" or face_bbox is None:
        return None

    B, C, F_lat, H, W = latent_shape
    x1, y1, x2, y2 = face_bbox

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    hw = (x2 - x1) * (0.5 + dilation)
    hh = (y2 - y1) * (0.5 + dilation)
    x1d = max(0.0, cx - hw)
    y1d = max(0.0, cy - hh)
    x2d = min(1.0, cx + hw)
    y2d = min(1.0, cy + hh)

    mask = torch.zeros(H, W, dtype=torch.float32)

    if gating_mode == "mask_hard":
        px1 = int(x1d * W)
        py1 = int(y1d * H)
        px2 = int(x2d * W)
        py2 = int(y2d * H)
        mask[py1:py2, px1:px2] = 1.0
    else:
        yy = torch.arange(H, dtype=torch.float32).view(-1, 1) / H
        xx = torch.arange(W, dtype=torch.float32).view(1, -1) / W
        dx = (xx - cx).abs() / max(hw, 1e-6)
        dy = (yy - cy).abs() / max(hh, 1e-6)
        dist = torch.maximum(dx, dy)
        soft = torch.where(
            dist <= 1.0,
            torch.ones_like(dist),
            torch.where(
                dist >= 1.5,
                torch.zeros_like(dist),
                0.5 * (1.0 + torch.cos((dist - 1.0) * math.pi / 0.5))),
        )
        mask = soft

    return mask.view(1, 1, 1, H, W)


# ── shared attach/clear plumbing for the conditioning nodes ─────────────────

def _clear_reference_state(model):
    """strength=0 / bypass path: clone and strip all reference keys."""
    model = model.clone()
    if hasattr(model, "model_options") and isinstance(model.model_options, dict):
        to = model.model_options.get("transformer_options")
        if isinstance(to, dict):
            for k in ("reference_latent", "reference_position_mode",
                      "reference_source_id", "reference_phase_scale",
                      "reference_spatial_mask", "reference_mask_gating",
                      "memory_video", "memory_position_mode"):
                to.pop(k, None)
    try:
        dm = model.model.diffusion_model
        for attr in ("_ltx_reference_latent", "_echo_memory_video",
                     "_pending_ref_seq_len", "_pending_ref_frames",
                     "_pending_memory_seq_len", "_pending_memory_frames"):
            if hasattr(dm, attr):
                try:
                    delattr(dm, attr)
                except Exception:  # noqa: BLE001
                    setattr(dm, attr, None)
    except Exception:  # noqa: BLE001
        pass
    return model


def _normalize_reference_latent(model, reference_latent, label, verbose):
    """process_latent_in normalization - without it the raw VAE latent has a
    different magnitude distribution than the target tokens (red tint)."""
    try:
        base_model = model.model
        if hasattr(base_model, "process_latent_in"):
            pre = (float(reference_latent.mean().item()),
                   float(reference_latent.std().item()))
            reference_latent = base_model.process_latent_in(reference_latent)
            post = (float(reference_latent.mean().item()),
                    float(reference_latent.std().item()))
            if verbose:
                logger.info("[%s] normalized latent via process_latent_in: "
                            "mean %.3f->%.3f, std %.3f->%.3f",
                            label, pre[0], post[0], pre[1], post[1])
        else:
            logger.warning("[%s] model has no process_latent_in - reference "
                           "latent stays raw (may cause color artifacts)", label)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] latent normalization failed: %s: %s - using raw "
                       "latent", label, type(e).__name__, e)
    return reference_latent


def _attach_reference(model, reference_latent, position_mode):
    """Clone + attach via both channels (transformer_options + attribute)."""
    model = model.clone()
    if not hasattr(model, "model_options") or model.model_options is None:
        model.model_options = {}
    if "transformer_options" not in model.model_options:
        model.model_options["transformer_options"] = {}
    to = model.model_options["transformer_options"]
    to["reference_latent"] = reference_latent
    to["reference_position_mode"] = position_mode
    # Legacy Echo keys kept for parity with the source pack.
    to["memory_video"] = reference_latent
    to["memory_position_mode"] = position_mode
    try:
        dm = model.model.diffusion_model
        dm._ltx_reference_latent = reference_latent
        dm._echo_memory_video = reference_latent
    except Exception as e:  # noqa: BLE001
        logger.warning("[LTX Reference] couldn't set attribute side-channel: %s", e)
    return model


def _resize_to_target_latent_px(image, target_latent, label, verbose):
    """Pixel-space resize of IMAGE frames to the target latent's H*32/W*32."""
    if target_latent is None:
        return image
    try:
        tl = _video_latent_samples(target_latent)
        if isinstance(tl, torch.Tensor) and tl.dim() == 5:
            H_lat = int(tl.shape[3])
            W_lat = int(tl.shape[4])
            target_H_px = H_lat * 32
            target_W_px = W_lat * 32
            img_H = int(image.shape[1])
            img_W = int(image.shape[2])
            if (img_H, img_W) != (target_H_px, target_W_px):
                img_chw = image.permute(0, 3, 1, 2).contiguous()
                img_chw = F.interpolate(
                    img_chw, size=(target_H_px, target_W_px),
                    mode="bilinear", align_corners=False, antialias=True)
                image = img_chw.permute(0, 2, 3, 1).contiguous()
                if verbose:
                    logger.info("[%s] resized image %dx%d -> %dx%d to match "
                                "target latent", label, img_H, img_W,
                                target_H_px, target_W_px)
        else:
            logger.warning("[%s] target_latent didn't have the expected "
                           "(B,C,F,H,W) shape (got %s) - using image as-is",
                           label, type(tl).__name__)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] couldn't use target_latent for resize: %s: %s - "
                       "using image at its native size",
                       label, type(e).__name__, e)
    return image


def _encode_reference_image(vae, image, label):
    """VAE-encode and normalize the result to 5D (B,C,F,H,W)."""
    try:
        encoded = vae.encode(image)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"[{label}] VAE encode failed: {type(e).__name__}: {e}. "
            f"Image shape (B,H,W,C) = {tuple(image.shape)}. "
            f"Ensure the VAE is the LTX video VAE.")

    if isinstance(encoded, dict):
        for k in ("samples", "latent", "x"):
            if k in encoded:
                encoded = encoded[k]
                break
    if not isinstance(encoded, torch.Tensor):
        raise RuntimeError(
            f"[{label}] VAE returned non-tensor: {type(encoded).__name__}")

    if encoded.dim() == 5:
        return encoded.contiguous()
    if encoded.dim() == 4:
        return encoded.unsqueeze(2).contiguous()
    raise RuntimeError(
        f"[{label}] unexpected VAE output dimensionality: "
        f"{encoded.dim()}D, shape {tuple(encoded.shape)}")


# ── nodes ───────────────────────────────────────────────────────────────────

class CCTechLTXReferenceConditioning:
    """Encode an image (or frame window) to a reference latent and attach it.

    Self-contained: installs the per-instance forward patch itself (the old
    separate Enable node was folded in here), so always take MODEL from this
    node's output.
    """

    CATEGORY = LTX23_REF_CATEGORY
    TITLE = "LTX-2.3 Reference Conditioning ⚡"
    SEARCH_ALIASES = ['reference', 'identity', 'reference image', 'memory',
                      'prefix injection', 'best face id']
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    DESCRIPTION = (
        "Encodes an image through the LTX video VAE and attaches the "
        "resulting latent to the MODEL as a reference for prefix injection "
        "during sampling. The forward patch installs itself on the model "
        "instance - always take MODEL from this node's output. A batched "
        "IMAGE input plus start_frame/num_frames selects a multi-frame "
        "reference window (motion/temporal style context). Complementary "
        "to standard i2v latent conditioning - different intervention "
        "points, can be combined. strength 0.0 is a clean bypass that "
        "also clears prior reference state.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE", {
                    "tooltip": "Reference image. A multi-frame batch (video "
                               "frames) is windowed via start_frame/"
                               "num_frames below.",
                }),
            },
            "optional": {
                "target_latent": ("LATENT", {
                    "tooltip": "Optional. Wire the same LATENT that goes "
                               "into your sampler. The image will be resized "
                               "in pixel space to match this latent's "
                               "spatial dimensions before VAE encoding, "
                               "guaranteeing memory and target have matching "
                               "patches-per-frame. Without this, you'll get "
                               "a tensor mismatch error if your image and "
                               "target latent don't match exactly.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales the reference latent magnitude. "
                               "1.0 = native VAE output. <1.0 reduces "
                               "influence; >1.0 boosts (may distort). Set to "
                               "0.0 to bypass completely and clear any prior "
                               "reference state.",
                }),
                "position_mode": (["reference", "prefix_continuous"], {
                    "default": "reference",
                    "tooltip": "How reference tokens are positioned in the "
                               "attention sequence. 'reference' (default): "
                               "tokens overlap target's first frame "
                               "temporally - uniform identity influence "
                               "across all generated frames. "
                               "'prefix_continuous': tokens placed before "
                               "target temporally - equivalent to standard "
                               "i2v prior-context conditioning.",
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0, "max": 240, "step": 1,
                    "tooltip": "For a batched IMAGE input: which frame to "
                               "start the reference window at. Frames before "
                               "this are discarded.",
                }),
                "num_frames": ("INT", {
                    "default": 1, "min": 1, "max": 25, "step": 1,
                    "tooltip": "How many frames from start_frame to use as "
                               "reference. 1 (default) = single-image "
                               "reference. 9 (1 + 8k matches LTX temporal "
                               "compression) gives ~2 latent frames; 17, 25 "
                               "add temporal context at higher cost.",
                }),
                "zero_ref_timesteps": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mark reference tokens as sigma=0 (clean "
                               "reference). Default OFF based on empirical "
                               "testing - most LTX2.3 checkpoints produce "
                               "better output when reference tokens share "
                               "target's noise sigma. Enable only if a "
                               "checkpoint was trained for clean-reference "
                               "memory.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print detailed per-call info to the console.",
                }),
            },
        }

    def attach(self, model, vae, image, target_latent=None,
               strength=1.0, position_mode="reference",
               start_frame=0, num_frames=1,
               zero_ref_timesteps=False, verbose=False):
        global _VERBOSE
        _VERBOSE = bool(verbose)

        if strength == 0.0:
            logger.info("[LTX Reference Conditioning] strength=0, bypassing "
                        "and clearing state.")
            return (_clear_reference_state(model),)

        dm = _install_reference_patches(model)
        dm._ltx_zero_ref_timesteps = bool(zero_ref_timesteps)

        if image.dim() != 4:
            raise ValueError(
                f"[LTX Reference Conditioning] Expected IMAGE shape "
                f"(B, H, W, C), got {tuple(image.shape)}")

        if image.shape[-1] > 3:
            image = image[..., :3]

        # Frame window (the old Sequence node's semantics; the defaults
        # 0/1 reproduce single-image behavior exactly).
        total_frames = int(image.shape[0])
        actual_start = max(0, min(start_frame, total_frames - 1))
        actual_end = min(actual_start + num_frames, total_frames)
        sequence = image[actual_start:actual_end]
        actual_count = int(sequence.shape[0])

        if actual_count < 1:
            raise ValueError(
                f"[LTX Reference Conditioning] No frames after slicing "
                f"(start={actual_start}, end={actual_end}, "
                f"total_frames={total_frames})")

        if verbose and total_frames > 1:
            logger.info("[LTX Reference Conditioning] using frames [%d:%d] "
                        "(%d frames) from input of %d total",
                        actual_start, actual_end, actual_count, total_frames)

        sequence = _resize_to_target_latent_px(
            sequence, target_latent, "LTX Reference Conditioning", verbose)
        sequence = _pad_image_to_multiple(sequence, divisor=32)

        try:
            encoded = vae.encode(sequence)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"[LTX Reference Conditioning] VAE encode failed: "
                f"{type(e).__name__}: {e}. Image shape (B,H,W,C) = "
                f"{tuple(sequence.shape)}. Ensure the VAE is the LTX video "
                f"VAE.")

        if isinstance(encoded, dict):
            for k in ("samples", "latent", "x"):
                if k in encoded:
                    encoded = encoded[k]
                    break
        if not isinstance(encoded, torch.Tensor):
            raise RuntimeError(
                f"[LTX Reference Conditioning] VAE returned non-tensor: "
                f"{type(encoded).__name__}")

        if encoded.dim() == 5:
            reference_latent = encoded.contiguous()
        elif encoded.dim() == 4:
            # N independent image latents -> stack along the F dim (for a
            # single image this is identical to unsqueeze(2)).
            reference_latent = encoded.unsqueeze(0).transpose(1, 2).contiguous()
            if verbose and encoded.shape[0] > 1:
                logger.info("[LTX Reference Conditioning] VAE returned 4D - "
                            "stacking N=%d as F dim", encoded.shape[0])
        else:
            raise RuntimeError(
                f"[LTX Reference Conditioning] unexpected VAE output "
                f"dimensionality: {encoded.dim()}D, shape "
                f"{tuple(encoded.shape)}")

        reference_latent = _normalize_reference_latent(
            model, reference_latent, "LTX Reference Conditioning", verbose)

        if strength != 1.0:
            reference_latent = reference_latent * strength

        if reference_latent.shape[1] != 128:
            logger.warning("[LTX Reference Conditioning] VAE produced %d "
                           "latent channels, expected 128 for LTX - "
                           "mismatched VAE?", reference_latent.shape[1])

        model = _attach_reference(model, reference_latent, position_mode)
        if verbose:
            logger.info("[LTX Reference Conditioning] attached: shape=%s "
                        "(B,C,F,H,W), mode=%s, strength=%s",
                        tuple(reference_latent.shape), position_mode, strength)
        return (model,)

class CCTechLTXFaceIdentityReinforcer:
    """Drop-in identity reinforcer for LTX-Best-Face-ID LoRA workflows."""

    CATEGORY = LTX23_REF_CATEGORY
    TITLE = "LTX-2.3 Face Identity Reinforcer \u26a1"
    SEARCH_ALIASES = ['face identity', 'best face id', 'identity reinforcer',
                      'face reference', 'source phase']
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "reinforce"
    DESCRIPTION = (
        "Unified LTX-Best-Face-ID identity reinforcer. Combines reference "
        "latent injection, RoPE source-phase tagging, face detection, and "
        "spatial mask gating in a single node. Makes Best-Face-ID work "
        "correctly alongside i2v frame 0 conditioning by placing reference "
        "tokens at a distinct RoPE position while preserving the "
        "source_id=2 tag the LoRA was trained against. Load the "
        "Best-Face-ID LoRA on the MODEL path before this node.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "target_latent": ("LATENT",),
            },
            "optional": {
                "identity_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scales reference latent magnitude. "
                               "1.0 = Best-Face-ID default.",
                }),
                "face_padding": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Face bbox expansion - captures hair/neck "
                               "context.",
                }),
                "auto_face_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When a face is detected, auto-crop the "
                               "reference image around the face at "
                               "zoom_factor extent and match target aspect "
                               "ratio. Dramatically improves identity "
                               "transfer for wide/full-body references by "
                               "giving the VAE much more face detail to "
                               "encode. Turn off if reference is already "
                               "tightly cropped.",
                }),
                "crop_zoom_factor": ("FLOAT", {
                    "default": 2.0, "min": 1.2, "max": 4.0, "step": 0.1,
                    "tooltip": "How much context around the face to include. "
                               "2.0 = crop is 2x the face bbox (shoulders + "
                               "hair). 1.5 = very tight (face + hair only). "
                               "3.0 = wide (upper body). Ignored if "
                               "auto_face_crop off.",
                }),
                "spatial_gating": (["mask_soft", "mask_hard", "off"], {
                    "default": "mask_soft",
                    "tooltip": "Constrain identity influence to face region. "
                               "mask_soft = cosine falloff (recommended). "
                               "mask_hard = binary. off = uniform (raw "
                               "Best-Face-ID).",
                }),
                "placement_mode": (["i2v_safe", "t2v_overlap", "prefix"], {
                    "default": "i2v_safe",
                    "tooltip": "i2v_safe / t2v_overlap = pure overlap layout "
                               "(Best-Face-ID's default). Reference reuses "
                               "target's coord grid, disambiguated by clean/"
                               "noisy state and sequence position. prefix = "
                               "additive offset (legacy).",
                }),
                "source_id": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 8.0, "step": 1.0,
                    "tooltip": "RoPE source tag applied via phase rotation. "
                               "Best-Face-ID LoRA expects 2.0. source_id=0 "
                               "disables rotation (overlap-only behavior).",
                }),
                "phase_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Phase rotation magnitude multiplier. "
                               "Best-Face-ID LoRA expects 1.0. Lower values "
                               "reduce reference/target separation strength.",
                }),
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Optional secondary reference (multi-subject).",
                }),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    def reinforce(self, model, vae, reference_image, target_latent,
                  identity_strength: float = 1.0,
                  face_padding: float = 0.15,
                  auto_face_crop: bool = True,
                  crop_zoom_factor: float = 2.0,
                  spatial_gating: str = "mask_soft",
                  placement_mode: str = "i2v_safe",
                  source_id: float = 2.0,
                  phase_scale: float = 1.0,
                  reference_image_2=None,
                  debug: bool = False):
        # 1. Patches (per-instance).
        _install_reference_patches(model)

        # 2. Face detection FIRST (best quality at native resolution).
        def _detect_on(img_tensor):
            try:
                img_np = (img_tensor[0].cpu().clamp(0, 1) * 255.0
                          ).to(torch.uint8).numpy()
                return _detect_face_bbox(img_np, padding=face_padding,
                                         debug=debug)
            except Exception as e:  # noqa: BLE001
                if debug:
                    logger.info("[Reinforcer] face detect failed: %s", e)
                return None

        face_bbox = _detect_on(reference_image)
        face_bbox_secondary = None
        if reference_image_2 is not None:
            face_bbox_secondary = _detect_on(reference_image_2)

        if debug:
            logger.info("[Reinforcer] primary face bbox: %s",
                        face_bbox if face_bbox is not None else "no face found")

        # 3. Auto-crop references around detected faces.
        _tlat = _video_latent_samples(target_latent)
        if _tlat.dim() == 5:
            _tH = int(_tlat.shape[3])
            _tW = int(_tlat.shape[4])
        elif _tlat.dim() == 4:
            _tH = int(_tlat.shape[2])
            _tW = int(_tlat.shape[3])
        else:
            _tH, _tW = 1, 1
        _target_aspect = _tW / max(_tH, 1)

        primary_post_crop_bbox = None

        def _prepare_ref(img_tensor, bbox):
            if img_tensor is None:
                return None, None
            processed = img_tensor
            post_crop_bbox = bbox
            if auto_face_crop and bbox is not None:
                cropped = _auto_face_crop(
                    processed, bbox, zoom_factor=crop_zoom_factor,
                    target_aspect=_target_aspect, debug=debug)
                # invalid-bounds fallback returns just the image
                if isinstance(cropped, tuple):
                    processed, post_crop_bbox = cropped
                else:
                    processed = cropped
            processed = _resize_image_to_latent(processed, target_latent,
                                                vae_scale=32)
            processed = _pad_image_to_multiple(processed, divisor=32)
            return processed, post_crop_bbox

        ref_prepared = []
        primary_prepared, primary_post_crop_bbox = _prepare_ref(
            reference_image, face_bbox)
        if primary_prepared is not None:
            ref_prepared.append(primary_prepared)
        if reference_image_2 is not None:
            if (primary_prepared is not None
                    and primary_post_crop_bbox is not None
                    and face_bbox_secondary is not None
                    and auto_face_crop):
                _, prim_H, prim_W, _ = primary_prepared.shape
                aligned_ref2, _ = _align_to_reference_bbox(
                    reference_image_2, face_bbox_secondary,
                    primary_post_crop_bbox, prim_H, prim_W, debug=debug)
                ref_prepared.append(
                    _pad_image_to_multiple(aligned_ref2, divisor=32))
                if debug:
                    logger.info("[Reinforcer] ref2 aligned to match ref1 "
                                "face position/scale")
            else:
                sec_prepared, _ = _prepare_ref(reference_image_2,
                                               face_bbox_secondary)
                if sec_prepared is not None:
                    ref_prepared.append(sec_prepared)

        # 4. VAE encode all prepared references.
        ref_latents = []
        for idx, img in enumerate(ref_prepared):
            try:
                lat = vae.encode(img)
            except Exception as e:  # noqa: BLE001
                if debug:
                    logger.info("[Reinforcer] VAE encode failed on ref %d: %s",
                                idx, e)
                continue
            if isinstance(lat, dict):
                lat = lat.get("samples", lat)
            if lat.dim() == 4:
                lat = lat.unsqueeze(2)
            ref_latents.append(lat * identity_strength)

        if not ref_latents:
            if debug:
                logger.info("[Reinforcer] no valid reference latents; "
                            "passing through")
            return (model,)

        if len(ref_latents) > 1:
            primary_ref = torch.cat(ref_latents, dim=2)
            if debug:
                logger.info("[Reinforcer] concatenated %d refs -> latent "
                            "shape %s", len(ref_latents),
                            tuple(primary_ref.shape))
        else:
            primary_ref = ref_latents[0]

        if auto_face_crop and primary_post_crop_bbox is not None:
            face_bbox_for_mask = primary_post_crop_bbox
        else:
            face_bbox_for_mask = face_bbox

        if debug:
            logger.info("[Reinforcer] mask bbox: %s",
                        face_bbox_for_mask
                        if face_bbox_for_mask is not None
                        else "gating disabled (no bbox)")

        # 5. Spatial gating mask.
        if isinstance(target_latent, dict):
            target_shape = tuple(target_latent["samples"].shape)
        else:
            target_shape = tuple(target_latent.shape)
        if len(target_shape) == 4:
            target_shape = (target_shape[0], target_shape[1], 1,
                            target_shape[2], target_shape[3])

        face_mask = _make_face_mask_latent(
            face_bbox=face_bbox_for_mask,
            latent_shape=target_shape,
            gating_mode=spatial_gating,
            dilation=face_padding)

        # 6. Attach via transformer_options.
        m = model.clone()
        model_options = m.model_options.setdefault("transformer_options", {})
        model_options["reference_latent"] = primary_ref
        model_options["reference_position_mode"] = placement_mode
        model_options["reference_source_id"] = source_id
        model_options["reference_phase_scale"] = phase_scale
        if face_mask is not None:
            model_options["reference_spatial_mask"] = face_mask
            model_options["reference_mask_gating"] = spatial_gating

        if debug:
            logger.info("[Reinforcer] attached: strength=%s, placement=%s, "
                        "source_id=%s, phase=%s, gating=%s",
                        identity_strength, placement_mode, source_id,
                        phase_scale, spatial_gating)

        return (m,)


class CCTechLTX25ReferenceConditioning(CCTechLTXReferenceConditioning):
    """The same node listed under the LTX-2.5 menu (identical behavior -
    the mechanism reads every shape off the model at runtime)."""

    CATEGORY = LTX25_REF_CATEGORY
    TITLE = "LTX-2.5 Reference Conditioning ⚡"


class CCTechLTX25FaceIdentityReinforcer(CCTechLTXFaceIdentityReinforcer):
    """The same node listed under the LTX-2.5 menu. NOTE: the Best-Face-ID
    LoRA the phase tag pairs with is 2.3-trained - loading it on a 2.5 model
    is cross-version territory (the ID-LoRAs load cleanly there, but this
    combination is unverified on 2.5)."""

    CATEGORY = LTX25_REF_CATEGORY
    TITLE = "LTX-2.5 Face Identity Reinforcer ⚡"


NODE_CLASS_MAPPINGS = {
    "CCTechLTXReferenceConditioning": CCTechLTXReferenceConditioning,
    "CCTechLTXFaceIdentityReinforcer": CCTechLTXFaceIdentityReinforcer,
    "CCTechLTX25ReferenceConditioning": CCTechLTX25ReferenceConditioning,
    "CCTechLTX25FaceIdentityReinforcer": CCTechLTX25FaceIdentityReinforcer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCTechLTXReferenceConditioning": CCTechLTXReferenceConditioning.TITLE,
    "CCTechLTXFaceIdentityReinforcer": CCTechLTXFaceIdentityReinforcer.TITLE,
    "CCTechLTX25ReferenceConditioning": CCTechLTX25ReferenceConditioning.TITLE,
    "CCTechLTX25FaceIdentityReinforcer": CCTechLTX25FaceIdentityReinforcer.TITLE,
}
