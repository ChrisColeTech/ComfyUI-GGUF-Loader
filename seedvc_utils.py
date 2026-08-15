"""Small SeedVC integration helpers that do not require a ComfyUI import."""


def select_seedvc_identity(skip_vc, chunk_count, explicit_reference, generated_reference):
    if skip_vc or (chunk_count <= 1 and explicit_reference is None):
        return None
    return explicit_reference if explicit_reference is not None else generated_reference
