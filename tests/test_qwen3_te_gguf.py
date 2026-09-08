"""Exact-table GGUF -> comfy conversion for Qwen3 / Qwen3-VL text encoders.

nodes/gguf_qwen3_te.py replaces the pack's generic string-replace mapping for
the one path that needs to be exact: Krea2's Qwen3-VL-4B tap. The keys it
emits are only correct if they satisfy comfy sd.py's `detect_te_model` probes
verbatim, so those literal probe strings are pinned here - they are the whole
contract, and a silently mangled key means comfy detects a plain Qwen3 model
and never attaches the vision tower.

Nothing here needs a real checkpoint: the text-side fixtures are generated
from gguf-py's own MODEL_TENSORS/TENSOR_NAMES tables, so they track whatever
gguf version is installed rather than a copy of the names frozen in a test.
"""
import importlib.util
import sys
import types

import pytest
import torch
from gguf.constants import MODEL_ARCH, MODEL_TENSORS, TENSOR_NAMES

from conftest import ROOT, _PKG, load_pack_module

# comfy/sd.py detect_te_model, the Qwen3-VL-4B/8B branch. Both of these must
# be present for a GGUF Krea2 TE to be recognised as VL at all.
COMFY_VL_PROBE = "model.visual.deepstack_merger_list.0.norm.weight"
COMFY_VL_SHAPE_PROBE = "model.visual.merger.linear_fc2.weight"
# ... and the plain-Qwen3 branch it falls through to, which the text stack
# has to satisfy regardless of whether an mmproj is present.
COMFY_TEXT_PROBES = (
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_norm.weight",
)

N_BLOCKS = 2


def _import_te_module():
    """Import nodes/gguf_qwen3_te.py without running nodes/__init__.py."""
    load_pack_module("loader")  # installs the comfy stubs + _PKG package
    nodes_pkg = f"{_PKG}.nodes"
    if nodes_pkg not in sys.modules:
        package = types.ModuleType(nodes_pkg)
        package.__path__ = [str(ROOT / "nodes")]
        sys.modules[nodes_pkg] = package
    full = f"{nodes_pkg}.gguf_qwen3_te"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, ROOT / "nodes" / "gguf_qwen3_te.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


te = _import_te_module()
loader = load_pack_module("loader")


def _llama_named_text_sd(arch=MODEL_ARCH.QWEN3, n_blocks=N_BLOCKS):
    """A full llama.cpp-named qwen3 text stack, straight from gguf-py."""
    sd = {}
    for tclass in MODEL_TENSORS[arch]:
        template = TENSOR_NAMES[tclass]
        names = ([template.format(bid=b) for b in range(n_blocks)]
                 if "{bid}" in template else [template])
        for name in names:
            sd[f"{name}.weight"] = torch.zeros(4, 4)
    return sd


# -- text stack -----------------------------------------------------------

def test_text_conversion_satisfies_comfy_detection():
    out = te.convert_text_sd(_llama_named_text_sd(), "qwen3", "fake.gguf",
                             n_blocks=N_BLOCKS)
    for probe in COMFY_TEXT_PROBES:
        assert probe in out, "comfy's detect_te_model probes " + probe
    assert "model.embed_tokens.weight" in out
    assert "model.norm.weight" in out
    assert "model.lm_head.weight" in out


def test_every_gguf_tensor_is_placed_and_rope_freqs_dropped():
    sd = _llama_named_text_sd()
    out = te.convert_text_sd(sd, "qwen3", "fake.gguf", n_blocks=N_BLOCKS)
    # rope_freqs has no comfy consumer and is deliberately dropped; every
    # other tensor in the file must come out the far side exactly once.
    assert "rope_freqs.weight" in sd
    assert len(out) == len(sd) - 1
    assert not any("rope" in k for k in out)


def test_qwen3vl_arch_tag_uses_the_same_table():
    text = _llama_named_text_sd(MODEL_ARCH.QWEN3VL)
    a = te.convert_text_sd(text, "qwen3", "fake.gguf", n_blocks=N_BLOCKS)
    b = te.convert_text_sd(text, "qwen3vl", "fake.gguf", n_blocks=N_BLOCKS)
    assert set(a) == set(b)


def test_block_count_inferred_when_metadata_omits_it():
    out = te.convert_text_sd(_llama_named_text_sd(), "qwen3", "fake.gguf",
                             n_blocks=None)
    assert "model.layers.%d.self_attn.q_proj.weight" % (N_BLOCKS - 1) in out


def test_unknown_text_tensor_is_a_hard_error_naming_it():
    sd = _llama_named_text_sd()
    sd["blk.0.attn_nonsense.weight"] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="attn_nonsense"):
        te.convert_text_sd(sd, "qwen3", "fake.gguf", n_blocks=N_BLOCKS)


def test_out_of_range_block_is_not_silently_mangled():
    """A block id past block_count has no table entry - it must error, not
    fall through to a partially-converted state dict."""
    sd = _llama_named_text_sd()
    stray = "blk.%d.attn_q.weight" % (N_BLOCKS + 5)
    sd[stray] = torch.zeros(4, 4)
    with pytest.raises(ValueError, match=r"blk\.%d" % (N_BLOCKS + 5)):
        te.convert_text_sd(sd, "qwen3", "fake.gguf", n_blocks=N_BLOCKS)


def test_hf_named_file_is_identity_mapped_under_model_prefix():
    sd = {
        "layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "layers.0.self_attn.q_norm.weight": torch.zeros(4),
        "layers.0.post_attention_layernorm.weight": torch.zeros(4),
        "model.embed_tokens.weight": torch.zeros(4, 4),
    }
    out = te.convert_text_sd(sd, "qwen3", "fake.gguf")
    for probe in COMFY_TEXT_PROBES:
        assert probe in out
    assert "model.embed_tokens.weight" in out


def test_hf_named_junk_still_errors():
    with pytest.raises(ValueError, match="not_a_layer"):
        te.convert_text_sd({"layers.0.not_a_layer.weight": torch.zeros(4)},
                           "qwen3", "fake.gguf")


# -- mmproj / vision tower ------------------------------------------------

VISION_BLOCKS = 2


def _mmproj_sd(deepstack_layers, split_qkv=False, patch_halves=True):
    vsd = {
        "v.position_embd.weight": torch.zeros(9, 4),
        "v.post_ln.weight": torch.zeros(4),
        "mm.0.weight": torch.zeros(4, 4),
        "mm.2.weight": torch.zeros(6, 4),
    }
    if patch_halves:
        vsd["v.patch_embd.weight"] = torch.zeros(4, 3, 2, 2)
        vsd["v.patch_embd.weight.1"] = torch.ones(4, 3, 2, 2)
    else:
        vsd["v.patch_embd.weight"] = torch.zeros(4, 3, 2, 2, 2)
    for b in range(VISION_BLOCKS):
        vsd["v.blk.%d.ln1.weight" % b] = torch.zeros(4)
        vsd["v.blk.%d.ln2.weight" % b] = torch.zeros(4)
        vsd["v.blk.%d.attn_out.weight" % b] = torch.zeros(4, 4)
        vsd["v.blk.%d.ffn_up.weight" % b] = torch.zeros(8, 4)
        vsd["v.blk.%d.ffn_down.weight" % b] = torch.zeros(4, 8)
        if split_qkv:
            for i, side in enumerate(("q", "k", "v")):
                vsd["v.blk.%d.attn_%s.weight" % (b, side)] = torch.full((4, 4), float(i))
        else:
            vsd["v.blk.%d.attn_qkv.weight" % b] = torch.zeros(12, 4)
    for layer in deepstack_layers:
        vsd["v.deepstack.%d.norm.weight" % layer] = torch.zeros(4)
        vsd["v.deepstack.%d.fc1.weight" % layer] = torch.zeros(4, 4)
        vsd["v.deepstack.%d.fc2.weight" % layer] = torch.zeros(6, 4)
    return vsd


# The two DeepStack index sets comfy actually ships (text_encoders/qwen3vl.py
# QWEN3VL_VISION): 4B uses [5, 11, 17], 8B/32B use [8, 16, 24].
@pytest.mark.parametrize("layers", ([5, 11, 17], [8, 16, 24]))
def test_deepstack_mergers_are_indexed_by_ordinal_not_layer_id(layers):
    out = te.convert_mmproj_sd(_mmproj_sd(layers), "fake-mmproj.gguf")
    for i in range(3):
        assert "model.visual.deepstack_merger_list.%d.norm.weight" % i in out
        assert "model.visual.deepstack_merger_list.%d.linear_fc1.weight" % i in out
        assert "model.visual.deepstack_merger_list.%d.linear_fc2.weight" % i in out
    assert COMFY_VL_PROBE in out
    assert COMFY_VL_SHAPE_PROBE in out
    # No merger key may still carry a raw vision-layer id.
    merger_keys = [k for k in out if "deepstack_merger_list" in k]
    assert len(merger_keys) == 9
    for layer in layers:
        assert not any(".%d." % layer in k for k in merger_keys if layer > 2)


def test_deepstack_ordinals_survive_an_unseen_layer_id_set():
    """The legacy string-replace map hardcodes the two known id sets; the
    ordinal here is computed from the file, so a third set still maps."""
    out = te.convert_mmproj_sd(_mmproj_sd([3, 9, 21]), "fake-mmproj.gguf")
    assert COMFY_VL_PROBE in out
    assert "model.visual.deepstack_merger_list.2.linear_fc2.weight" in out


def test_pos_embed_carries_the_model_prefix():
    """comfy's Qwen35VisionModel owns a learned pos_embed table; a key
    spelled 'visual.pos_embed.weight' loads as an unexpected key."""
    out = te.convert_mmproj_sd(_mmproj_sd([5, 11, 17]), "fake-mmproj.gguf")
    assert "model.visual.pos_embed.weight" in out
    assert "visual.pos_embed.weight" not in out


def test_patch_kernel_halves_are_stacked_back_to_5d():
    out = te.convert_mmproj_sd(_mmproj_sd([5, 11, 17]), "fake-mmproj.gguf")
    w = out["model.visual.patch_embed.proj.weight"]
    assert tuple(w.shape) == (4, 3, 2, 2, 2), "Conv3d kernel is (out, in, t, ph, pw)"
    assert torch.equal(w[:, :, 0], torch.zeros(4, 3, 2, 2))
    assert torch.equal(w[:, :, 1], torch.ones(4, 3, 2, 2))


def test_already_5d_patch_kernel_is_left_alone():
    out = te.convert_mmproj_sd(_mmproj_sd([5, 11, 17], patch_halves=False),
                               "fake-mmproj.gguf")
    w = out["model.visual.patch_embed.proj.weight"]
    assert tuple(w.shape) == (4, 3, 2, 2, 2)


def test_split_qkv_is_refused_in_q_k_v_order():
    out = te.convert_mmproj_sd(_mmproj_sd([5, 11, 17], split_qkv=True),
                               "fake-mmproj.gguf")
    qkv = out["model.visual.blocks.0.attn.qkv.weight"]
    assert tuple(qkv.shape) == (12, 4)
    assert torch.equal(qkv[0:4], torch.zeros(4, 4))            # q
    assert torch.equal(qkv[4:8], torch.ones(4, 4))             # k
    assert torch.equal(qkv[8:12], torch.full((4, 4), 2.0))     # v
    assert not any("attn_q" in k for k in out)


def test_block_leaves_map_to_comfy_names():
    out = te.convert_mmproj_sd(_mmproj_sd([5, 11, 17]), "fake-mmproj.gguf")
    for b in range(VISION_BLOCKS):
        for leaf in ("norm1", "norm2", "attn.qkv", "attn.proj",
                     "mlp.linear_fc1", "mlp.linear_fc2"):
            assert "model.visual.blocks.%d.%s.weight" % (b, leaf) in out


def test_unknown_vision_tensor_is_a_hard_error_naming_it():
    vsd = _mmproj_sd([5, 11, 17])
    vsd["v.blk.0.mystery.weight"] = torch.zeros(4)
    with pytest.raises(ValueError, match="mystery"):
        te.convert_mmproj_sd(vsd, "fake-mmproj.gguf")


# -- identity guards ------------------------------------------------------

@pytest.mark.parametrize("identity, expected", [
    ("qwen3vl", True),
    ("Qwen3-VL-4B-Instruct", True),
    ("Qwen3VL_4B", True),
    ("qwen3", False),
    # The exact trap this module exists to catch: the Z-Image text encoder
    # shipped mislabelled as Qwen3-VL-4B. Same shapes, different weights.
    ("Qwen3-4B-text-encoder-diffusers", False),
])
def test_vl_lineage_detection(identity, expected):
    assert bool(te._looks_vl(identity)) is expected


def test_lineage_tokens_drop_boilerplate_but_keep_the_size():
    """Family/precision words carry no lineage, so they are stripped. The
    size does - it is what makes a 4B text encoder paired with an 8B mmproj
    an empty intersection, and therefore a warning."""
    tokens = te._lineage_tokens("Qwen3-VL-4B-Instruct-bf16")
    assert tokens == {"4b"}
    assert "zimage" in te._lineage_tokens("ZImage-Qwen3-4B-text-encoder")


def test_mismatched_sizes_share_no_lineage():
    te_tokens = te._lineage_tokens("Qwen3-VL-4B-Instruct")
    mmproj_tokens = te._lineage_tokens("mmproj-Qwen3-VL-8B-f16")
    assert not (te_tokens & mmproj_tokens)


def test_matching_pair_shares_lineage():
    te_tokens = te._lineage_tokens("Qwen3-VL-4B-Q4_K_M")
    mmproj_tokens = te._lineage_tokens("mmproj-Qwen3-VL-4B-f16")
    assert te_tokens & mmproj_tokens


def test_find_sibling_mmproj_is_punctuation_insensitive(tmp_path):
    text = tmp_path / "Qwen3-VL-4B-Q4_K_M.gguf"
    text.write_bytes(b"")
    mmproj = tmp_path / "mmproj_Qwen3VL4B_f16.gguf"
    mmproj.write_bytes(b"")
    assert te.find_sibling_mmproj(str(text)) == str(mmproj)


def test_find_sibling_mmproj_returns_none_when_absent(tmp_path):
    text = tmp_path / "Qwen3-VL-4B-Q4_K_M.gguf"
    text.write_bytes(b"")
    (tmp_path / "mmproj-gemma4-12b-f16.gguf").write_bytes(b"")
    assert te.find_sibling_mmproj(str(text)) is None


# -- legacy path regression -----------------------------------------------

def test_legacy_deepstack_map_also_prefixes_pos_embed():
    """loader.py's CLIP_VISION_QWEN3_MAP serves the MiniMax-H3 / generic
    CLIP-loader route. It had the same pos_embed prefix bug; pin the fix so
    the two paths cannot drift apart again."""
    out = loader.map_mmproj_sd(_mmproj_sd([8, 16, 24]))
    assert "model.visual.pos_embed.weight" in out
    assert COMFY_VL_PROBE in out


# -- end-to-end load from real GGUF files ---------------------------------

import gguf  # noqa: E402
import numpy as np  # noqa: E402


def _write_text_gguf(path, arch="qwen3vl", name="Qwen3-VL-4B-Instruct"):
    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_name(name)
    writer.add_block_count(N_BLOCKS)
    for key in _llama_named_text_sd():
        writer.add_tensor(key, np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_mmproj_gguf(path, name="Qwen3-VL-4B-Instruct-mmproj"):
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_name(name)
    for key, value in _mmproj_sd([5, 11, 17]).items():
        writer.add_tensor(key, value.numpy().astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_end_to_end_load_merges_the_sibling_mmproj(tmp_path):
    """The whole seam: GGUFReader metadata -> gguf_sd_loader -> both
    converters -> one state dict comfy can detect as Qwen3-VL."""
    text = tmp_path / "qwen3vl-4b-Q4_K_M.gguf"
    _write_text_gguf(text)
    _write_mmproj_gguf(tmp_path / "mmproj-qwen3vl-4b-f16.gguf")

    out = te.load_qwen3_te_sd(str(text), expect_vl=True, expect_label="Krea2")

    assert COMFY_VL_PROBE in out, "vision tower did not merge in"
    assert COMFY_VL_SHAPE_PROBE in out
    for probe in COMFY_TEXT_PROBES:
        assert probe in out
    assert "model.visual.pos_embed.weight" in out


def test_end_to_end_load_without_mmproj_is_text_only(tmp_path):
    text = tmp_path / "qwen3-4b-Q4_K_M.gguf"
    _write_text_gguf(text, arch="qwen3", name="Qwen3-4B")
    out = te.load_qwen3_te_sd(str(text))
    for probe in COMFY_TEXT_PROBES:
        assert probe in out
    assert not any(k.startswith("model.visual.") for k in out)


def test_mislabelled_non_vl_file_warns_but_still_loads(tmp_path, caplog):
    """The Z-Image TE shipped as Qwen3-VL-4B: same shapes, wrong weights.
    It must be loud, not fatal - the user may know better than the metadata."""
    text = tmp_path / "Qwen3-VL-4B-Q4_K_M.gguf"
    _write_text_gguf(text, arch="qwen3", name="Qwen3-4B-text-encoder-diffusers")
    with caplog.at_level("WARNING"):
        out = te.load_qwen3_te_sd(str(text), expect_vl=True, expect_label="Krea2")
    assert "does not look like a Qwen3-VL model" in caplog.text
    assert "model.embed_tokens.weight" in out


def test_handing_an_mmproj_to_the_text_loader_is_refused(tmp_path):
    mmproj = tmp_path / "mmproj-qwen3vl-4b-f16.gguf"
    _write_mmproj_gguf(mmproj)
    with pytest.raises(ValueError, match="vision tower, not a text encoder"):
        te.load_qwen3_te_sd(str(mmproj))


def test_wrong_architecture_is_refused_by_name(tmp_path):
    path = tmp_path / "gemma.gguf"
    writer = gguf.GGUFWriter(str(path), "gemma3")
    writer.add_name("Gemma-3")
    writer.add_tensor("token_embd.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="gemma3"):
        te.load_qwen3_te_sd(str(path))
