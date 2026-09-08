"""Fresh GGUF -> comfy conversion for Qwen3 / Qwen3-VL text encoders.

Purpose-built for the Krea2 pipeline (Qwen3-VL-4B, 12-layer conditioning tap),
used by Krea2ModelLoader's GGUF clip branch. Other node families keep going
through loader.py's legacy path - this module deliberately does not touch it.

Design (clean-room, anchored on gguf-py, not on the legacy string-replace map):

  1. The llama.cpp-native tensor name table is INVERTED from gguf-py's own
     constants (``gguf.constants.MODEL_TENSORS`` / ``TENSOR_NAMES``): every
     gguf name maps to an exact ``MODEL_TENSOR`` class + block id. No
     substring replacement - an unrecognized tensor is a hard error naming it,
     never a silently mangled key.
  2. Comfy keys are emitted from a per-tensor-class table that satisfies
     comfy sd.py's ``detect_te_model`` probes exactly:
     text under ``model.layers.N...`` (q_norm/k_norm included), vision under
     ``model.visual...`` (``deepstack_merger_list`` is the Qwen3-VL detection
     anchor). comfy's own loader then re-prefixes (``model.visual.`` ->
     ``visual.``) inside the qwen3vl branches.
  3. Per-tensor-class transforms only where llama.cpp's convert_hf_to_gguf.py
     ``modify_tensors`` demands an inverse:
       - Qwen3/Qwen3-VL text stack: NO q/k permutation (llama.cpp's
         Qwen3Model does not permute - it has per-head q/k RMSNorms instead;
         un-permuting here, the llama-family transform, would corrupt it).
       - tied lm_head is legal (``output.weight`` may be absent).
       - mmproj sibling merged in: fused attn qkv (or a split q/k/v variant
         re-fused), the temporal-patch pair ``v.patch_embd.weight`` +
         ``.weight.1`` stacked back to the 5D Conv3d kernel, DeepStack
         mergers mapped by ordinal of their vision-layer id.

Also handled here (identity guards - see the mislabeled-file trap below):

  * Every GGUF text encoder's ``general.name`` is read and LOGGED at load.
  * When the caller expects a VL/krea2 model but the metadata identity lacks
    VL lineage (e.g. ``general.name = "Qwen3-4B-text-encoder-diffusers"`` -
    that is the Z-Image text encoder, seen in the wild mislabeled as
    ``Qwen3-VL-4B-Q4_K_M.gguf``), a prominent warning names the file, the
    metadata identity, and the fix. This exact trap cost a full debugging
    session: same shapes, different trained weights, embeddings only ~0.83
    cosine to the real Qwen3-VL-4B.
  * A sibling mmproj whose ``general.name`` lineage does not match the text
    model's is warned about too - merging it would silently graft a VL vision
    tower onto a non-VL text stack.
"""
import logging
import os

import torch

import gguf
from gguf.constants import MODEL_ARCH, MODEL_TENSOR, MODEL_TENSORS, TENSOR_NAMES

from ..loader import (gguf_sd_loader, get_field, strip_quant_suffix,
                      squash_name)
from ..ops.dequant import dequantize_tensor, is_quantized

# Text-stack architectures this module accepts. Some quantizers tag a VL
# model's text-only conversion "qwen3" instead of "qwen3vl"; both use the
# identical tensor name table (verified against gguf-py's MODEL_TENSORS).
_TEXT_ARCHES = {"qwen3": MODEL_ARCH.QWEN3, "qwen3vl": MODEL_ARCH.QWEN3VL}

# MODEL_TENSOR class -> comfy key template for the text stack.
# {bid} is the block id. NO per-class numeric transform is needed for any of
# these (qwen3: no q/k permute, no norm offset - unlike gemma3's -1.0).
_TEXT_CLASS_TO_COMFY = {
    MODEL_TENSOR.TOKEN_EMBD: "model.embed_tokens",
    MODEL_TENSOR.OUTPUT_NORM: "model.norm",
    MODEL_TENSOR.OUTPUT: "model.lm_head",       # tied/absent is legal
    MODEL_TENSOR.ATTN_NORM: "model.layers.{bid}.input_layernorm",
    MODEL_TENSOR.ATTN_Q: "model.layers.{bid}.self_attn.q_proj",
    MODEL_TENSOR.ATTN_Q_NORM: "model.layers.{bid}.self_attn.q_norm",
    MODEL_TENSOR.ATTN_K: "model.layers.{bid}.self_attn.k_proj",
    MODEL_TENSOR.ATTN_K_NORM: "model.layers.{bid}.self_attn.k_norm",
    MODEL_TENSOR.ATTN_V: "model.layers.{bid}.self_attn.v_proj",
    MODEL_TENSOR.ATTN_OUT: "model.layers.{bid}.self_attn.o_proj",
    MODEL_TENSOR.FFN_NORM: "model.layers.{bid}.post_attention_layernorm",
    MODEL_TENSOR.FFN_GATE: "model.layers.{bid}.mlp.gate_proj",
    MODEL_TENSOR.FFN_UP: "model.layers.{bid}.mlp.up_proj",
    MODEL_TENSOR.FFN_DOWN: "model.layers.{bid}.mlp.down_proj",
}

# Classes that may appear in the file but have no comfy-side consumer.
_TEXT_CLASS_DROP = {MODEL_TENSOR.ROPE_FREQS}

# HF-style key stems (a diffusers/HF export written into GGUF without
# llama.cpp renaming - the mislabeled Z-Image TE is stored this way). These
# are already comfy names minus the "model." prefix; identity-map them.
_HF_TEXT_STEM_SUFFIXES = (
    "input_layernorm", "post_attention_layernorm",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
_HF_TEXT_TOP_LEVEL = {"embed_tokens", "norm", "lm_head"}


def _split_suffix(name):
    for suffix in (".weight", ".bias"):
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    return name, ""


def _invert_text_table(arch, n_blocks):
    """gguf tensor stem -> (MODEL_TENSOR class, bid) from gguf-py's tables."""
    table = {}
    for tclass in MODEL_TENSORS[arch]:
        template = TENSOR_NAMES[tclass]
        if "{bid}" in template:
            for bid in range(n_blocks):
                table[template.format(bid=bid)] = (tclass, bid)
        else:
            table[template] = (tclass, None)
    return table


def _read_general(path):
    reader = gguf.GGUFReader(path)
    arch = get_field(reader, "general.architecture", str)
    name = get_field(reader, "general.name", str)
    n_blocks = None
    if arch:
        f = reader.get_field(f"{arch}.block_count")
        if f is not None:
            n_blocks = int(f.parts[f.data[-1]].item())
    del reader
    return arch, name, n_blocks


def _looks_vl(*identity_strings):
    """Does any identity string (arch tag, general.name) claim VL lineage?"""
    for s in identity_strings:
        if s and "vl" in squash_name(s):
            return True
    return False


def _lineage_tokens(name):
    """Distinctive lowercase tokens of a model identity string."""
    import re
    generic = {"qwen", "qwen3", "qwen3vl", "vl", "instruct", "model", "text",
               "encoder", "diffusers", "gguf", "bf16", "f16", "fp8", ""}
    return {t for t in re.split(r"[^0-9a-zA-Z]+", (name or "").lower())
            if t not in generic and not t.isdigit()}


def _warn_banner(lines):
    bar = "!" * 78
    logging.warning("\n%s\n%s\n%s", bar, "\n".join(lines), bar)


def convert_text_sd(sd, arch, path, n_blocks=None):
    """Map a qwen3/qwen3vl text-stack GGUF state dict to comfy keys.

    Accepts both llama.cpp-native names (``blk.N.attn_q.weight``) and HF-style
    names (``layers.N.self_attn.q_proj.weight`` / ``model.layers...``).
    Unmapped tensors are a hard error naming every offender.
    """
    # Which naming convention is this file in?
    llama_named = any(k.startswith("blk.") or k in ("token_embd.weight",)
                      for k in sd)

    out = {}
    unmapped = []
    if llama_named:
        if n_blocks is None:
            bids = [int(k.split(".")[1]) for k in sd if k.startswith("blk.")]
            n_blocks = (max(bids) + 1) if bids else 0
        table = _invert_text_table(_TEXT_ARCHES[arch], n_blocks)
        for key, value in sd.items():
            stem, suffix = _split_suffix(key)
            hit = table.get(stem)
            if hit is None:
                unmapped.append(key)
                continue
            tclass, bid = hit
            if tclass in _TEXT_CLASS_DROP:
                continue
            template = _TEXT_CLASS_TO_COMFY.get(tclass)
            if template is None:
                unmapped.append(key)
                continue
            out[template.format(bid=bid) + suffix] = value
    else:
        # HF/diffusers naming: validate stems against the known HF vocabulary
        # instead of trusting arbitrary keys.
        for key, value in sd.items():
            stem, suffix = _split_suffix(key)
            bare = stem[len("model."):] if stem.startswith("model.") else stem
            ok = bare in _HF_TEXT_TOP_LEVEL
            if not ok and bare.startswith("layers."):
                parts = bare.split(".", 2)
                ok = (len(parts) == 3 and parts[1].isdigit()
                      and parts[2] in _HF_TEXT_STEM_SUFFIXES)
            if not ok:
                unmapped.append(key)
                continue
            out["model." + bare + suffix] = value

    if unmapped:
        raise ValueError(
            f"GGUF text-encoder conversion: {len(unmapped)} tensor(s) in "
            f"'{os.path.basename(path)}' did not map to any known "
            f"qwen3/qwen3vl tensor class: {sorted(unmapped)[:12]}"
            f"{' ...' if len(unmapped) > 12 else ''}. Refusing to load a "
            "partially-converted text encoder."
        )

    # Token embedding: dequantize to fp16 up front (same rationale as the
    # pack's other TE paths - the lazily-dequantized 150k x hidden embedding
    # otherwise OOMs at first use).
    temb_key = "model.embed_tokens.weight"
    if temb_key in out and out[temb_key].shape[0] >= (64 * 1024) and is_quantized(out[temb_key]):
        logging.warning("Dequantizing %s to prevent runtime OOM.", temb_key)
        out[temb_key] = dequantize_tensor(out[temb_key], dtype=torch.float16)
    return out


# ── mmproj (vision tower) ─────────────────────────────────────────────────

# gguf mmproj stem -> comfy stem under "model.visual." (fixed names)
_VISION_FIXED = {
    "v.patch_embd": "patch_embed.proj",
    "v.position_embd": "pos_embed",
    "v.post_ln": "merger.norm",
    "mm.0": "merger.linear_fc1",
    "mm.2": "merger.linear_fc2",
}

# per-block: gguf leaf -> comfy leaf under "model.visual.blocks.{bid}."
_VISION_BLOCK = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}


def convert_mmproj_sd(vsd, path):
    """Map a qwen3vl mmproj GGUF to comfy ``model.visual.*`` keys.

    Inverse of llama.cpp's mmproj conversion for Qwen3-VL:
      - the 5D Conv3d patch kernel is stored as two 4D halves
        (``v.patch_embd.weight`` + ``.weight.1``, split on the temporal
        patch axis) -> stacked back to (out, in, 2, ph, pw);
      - attention is fused qkv (split q/k/v variants are re-fused);
      - DeepStack mergers ``v.deepstack.{layer}.{norm,fc1,fc2}`` -> ordinal
        ``deepstack_merger_list.{i}`` (comfy indexes them by position in
        ``deepstack_visual_indexes``, not by vision-layer id).
    Unmapped tensors are a hard error naming every offender.
    """
    vsd = dict(vsd)

    # 5D patch kernel: stack the two temporal halves back together.
    if "v.patch_embd.weight.1" in vsd:
        w0 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w0, w1], dim=2)

    # Split q/k/v variant: re-fuse to the qkv comfy expects.
    split_stems = sorted({k.rsplit(".attn_", 1)[0] for k in vsd
                          if ".attn_q." in k or ".attn_k." in k or ".attn_v." in k})
    for stem in split_stems:
        for suffix in (".weight", ".bias"):
            parts = []
            for side in ("q", "k", "v"):
                key = f"{stem}.attn_{side}{suffix}"
                if key in vsd:
                    parts.append(dequantize_tensor(vsd.pop(key), dtype=torch.float32))
            if parts:
                vsd[f"{stem}.attn_qkv{suffix}"] = torch.cat(parts, dim=0)

    # DeepStack ordinal map: vision-layer id -> list index.
    ds_layers = sorted({int(k.split(".")[2]) for k in vsd
                        if k.startswith("v.deepstack.")})
    ds_index = {layer: i for i, layer in enumerate(ds_layers)}

    out = {}
    unmapped = []
    for key, value in vsd.items():
        stem, suffix = _split_suffix(key)
        if stem in _VISION_FIXED:
            out["model.visual." + _VISION_FIXED[stem] + suffix] = value
            continue
        parts = stem.split(".")
        if stem.startswith("v.blk.") and len(parts) == 4 and parts[2].isdigit():
            leaf = _VISION_BLOCK.get(parts[3])
            if leaf is not None:
                out[f"model.visual.blocks.{parts[2]}.{leaf}{suffix}"] = value
                continue
        if stem.startswith("v.deepstack.") and len(parts) == 4 and parts[2].isdigit():
            leaf = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}.get(parts[3])
            if leaf is not None:
                i = ds_index[int(parts[2])]
                out[f"model.visual.deepstack_merger_list.{i}.{leaf}{suffix}"] = value
                continue
        unmapped.append(key)

    if unmapped:
        raise ValueError(
            f"GGUF mmproj conversion: {len(unmapped)} tensor(s) in "
            f"'{os.path.basename(path)}' did not map to any known Qwen3-VL "
            f"vision tensor: {sorted(unmapped)[:12]}"
            f"{' ...' if len(unmapped) > 12 else ''}. Refusing to load a "
            "partially-converted vision tower."
        )
    return out


def find_sibling_mmproj(path):
    """Path of the mmproj GGUF belonging to this text encoder, or None.

    Same punctuation-insensitive filename matching the pack's legacy path
    uses (a TE and its mmproj are rarely punctuated identically).
    """
    stem = strip_quant_suffix(os.path.splitext(os.path.basename(path))[0].lower())
    root = os.path.dirname(path)
    hits = []
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf" or "mmproj" not in name.lower():
            continue
        if squash_name(stem) in squash_name(name):
            hits.append(os.path.join(root, fname))
    if not hits:
        return None
    if len(hits) > 1:
        logging.info("Ambiguous mmproj for '%s'; using first match.",
                     os.path.basename(path))
    return sorted(hits)[0]


def load_qwen3_te_sd(path, expect_vl=False, expect_label="this pipeline"):
    """Load a qwen3/qwen3vl text-encoder GGUF (+ sibling mmproj) as a comfy
    state dict, with identity logging and lineage guards.

    expect_vl: the caller needs a VL model (e.g. krea2's Qwen3-VL-4B tap) -
    warn prominently when the file's metadata identity says otherwise.
    """
    arch, name, n_blocks = _read_general(path)
    base = os.path.basename(path)
    logging.info("GGUF TE '%s': general.architecture=%r general.name=%r",
                 base, arch, name)

    if arch == "clip":
        raise ValueError(
            f"'{base}' is an mmproj vision tower, not a text encoder. Select "
            "the text-encoder GGUF instead - the mmproj is merged in "
            "automatically from the same folder."
        )
    if arch not in _TEXT_ARCHES:
        raise ValueError(
            f"'{base}' has architecture {arch!r} - this Krea2 loader only "
            "accepts qwen3/qwen3vl text encoders. Use the generic GGUF CLIP "
            "loader for other architectures."
        )

    if expect_vl and not _looks_vl(arch, name):
        _warn_banner([
            f"'{base}' does not look like a Qwen3-VL model:",
            f"  general.architecture = {arch!r}, general.name = {name!r}.",
            f"  {expect_label} needs a real Qwen3-VL text encoder; a plain",
            "  'Qwen3-4B...' identity here usually means the file is the",
            "  Z-Image text encoder mislabeled as Qwen3-VL-4B. Same shapes,",
            "  DIFFERENT trained weights - it will load and produce garbage",
            "  conditioning. Download the genuine Qwen3-VL-4B TE instead.",
        ])

    sd, extra = gguf_sd_loader(path, is_text_model=True)
    out = convert_text_sd(sd, arch, path, n_blocks=n_blocks)

    mmproj = find_sibling_mmproj(path)
    if mmproj is None:
        if expect_vl:
            logging.warning(
                "No mmproj vision tower found next to '%s'; the vision/"
                "grounding path (identity edit, Ostris edit encode) will not "
                "work, and comfy will detect this as a plain Qwen3 model.", base)
    else:
        m_arch, m_name, _ = _read_general(mmproj)
        logging.info("GGUF mmproj '%s': general.architecture=%r general.name=%r",
                     os.path.basename(mmproj), m_arch, m_name)
        graft = []
        if not _looks_vl(arch, name):
            graft = [
                f"Merging mmproj '{os.path.basename(mmproj)}' onto '{base}',",
                f"but the text model's own identity ({name!r}) has no VL",
                "lineage. This grafts a VL vision tower onto a non-VL text",
                "stack: it will pass comfy's Qwen3-VL detection while the",
                "text weights remain the wrong model. Check that the text",
                "GGUF really is the Qwen3-VL conversion.",
            ]
        else:
            t_tok, m_tok = _lineage_tokens(name), _lineage_tokens(m_name)
            if t_tok and m_tok and not (t_tok & m_tok):
                graft = [
                    f"mmproj '{os.path.basename(mmproj)}' (name={m_name!r})",
                    f"does not share lineage with text model '{base}'",
                    f"(name={name!r}). The merged vision tower may belong to",
                    "a different checkpoint.",
                ]
        if graft:
            _warn_banner(graft)
        vsd, _vextra = gguf_sd_loader(mmproj, is_text_model=True)
        out.update(convert_mmproj_sd(vsd, mmproj))
        logging.info("Merged mmproj '%s' into the text encoder.",
                     os.path.basename(mmproj))
    return out
