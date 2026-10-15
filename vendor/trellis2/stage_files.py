# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Per-stage TRELLIS.2 / Pixal3D files -> ComfyUI core's merged layouts.

The owner publishes the checkpoints split by stage, the way upstream ships
them (``ckpts/<stage>.safetensors`` or ``split/<group>/<stage>.gguf``). ComfyUI
core loads one merged diffusion model and one VAE per decoder group. The
tensors are the same; only the key prefix differs, so merging is a rename:

==========================================  ==========================  ===========
per-stage file                              core prefix                 core object
==========================================  ==========================  ===========
ss_flow_img_dit_1_3B_64                     ``structure_model.``        MODEL
slat_flow_img2shape_dit_1_3B_512            ``img2shape_512.``          MODEL
slat_flow_img2shape_dit_1_3B_1024           ``img2shape.``              MODEL
slat_flow_imgshape2tex_dit_1_3B_1024        ``shape2txt.``              MODEL
ss_dec_conv3d_16l8                          ``struct_dec.``             shape VAE
shape_dec_next_dc_f16c32                    ``shape_dec.``              shape VAE
tex_dec_next_dc_f16c32                      ``txt_dec.``                texture VAE
==========================================  ==========================  ===========

Every key and shape was checked against core's modules (TRELLIS.2 fp16
safetensors 640/640 per DiT, Pixal3D GGUF 700/700 incl. the projection heads,
decoders 74/292/284). Pixal3D's sparse-structure GGUF carries one extra
buffer, ``rope_phases``, which core recomputes; it is dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

# stage -> (core prefix, filename test)
DIT_STAGES = {
    "sparse_structure": ("structure_model.", lambda n: "ss_flow" in n),
    "shape_512": ("img2shape_512.", lambda n: "img2shape" in n and re.search(r"(?<!\d)512(?!\d)", n)),
    "shape_1024": ("img2shape.", lambda n: "img2shape" in n and re.search(r"(?<!\d)1024(?!\d)", n)),
    "texture": ("shape2txt.", lambda n: "imgshape2tex" in n and re.search(r"(?<!\d)1024(?!\d)", n)),
}
DECODER_STAGES = {
    "structure_decoder": ("struct_dec.", lambda n: "ss_dec" in n),
    "shape_decoder": ("shape_dec.", lambda n: "shape_dec" in n),
    "texture_decoder": ("txt_dec.", lambda n: "tex_dec" in n),
}
# Buffers some per-stage exports carry that core computes itself.
DROP_KEYS = ("rope_phases",)
WEIGHT_EXTENSIONS = (".safetensors", ".gguf")


def matches(stage_table: dict, stage: str, name: str) -> bool:
    return bool(stage_table[stage][1](Path(name).name.lower()))


def merge(parts: dict, stage_table: dict) -> dict:
    """``{stage: state_dict}`` -> one state dict with core's prefixes."""
    merged = {}
    for stage, sd in parts.items():
        prefix = stage_table[stage][0]
        for key, value in sd.items():
            if key in DROP_KEYS:
                continue
            merged[prefix + key] = value
    return merged


def is_pixal3d(state_dict) -> bool:
    """Pixal3D's flow models add projection heads (``cross_attn.proj_linear``)."""
    return any(".cross_attn.proj_linear." in k for k in state_dict)
