"""Small SeedVC integration helpers that do not require a ComfyUI import."""

import math


def select_seedvc_identity(skip_vc, chunk_count, explicit_reference, generated_reference):
    if skip_vc or (chunk_count <= 1 and explicit_reference is None):
        return None
    return explicit_reference if explicit_reference is not None else generated_reference


def seedvc_output_is_usable(converted, source, length_tolerance=0.05):
    """Reject a conversion that is finite but structurally wrong.

    Voice conversion may only re-time the source slightly and must stay audible.
    A result that is empty, non-finite, silent, or a markedly different shape or
    length is not a polished take of ``source`` — it is a failure that happened
    not to raise, and the caller must keep the unpolished audio.
    """
    if converted is None or source is None:
        return False
    if converted.ndim != source.ndim or converted.shape[:-1] != source.shape[:-1]:
        return False
    length, source_length = converted.shape[-1], source.shape[-1]
    if length < 1 or source_length < 1:
        return False
    if not bool(converted.isfinite().all()):
        return False
    if abs(length - source_length) > math.ceil(source_length * length_tolerance):
        return False
    return float(converted.abs().max()) > 1e-4
