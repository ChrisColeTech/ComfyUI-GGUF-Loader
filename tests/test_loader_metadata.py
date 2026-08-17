import gguf
import numpy as np
import pytest
import torch

from conftest import load_pack_module

loader = load_pack_module("loader")


@pytest.fixture(scope="module")
def sample_gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("gguf") / "sample.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen2")
    # llama.cpp writes counts as UINT32, which the old reader did not even list.
    writer.add_uint32("qwen2.attention.head_count", 28)
    writer.add_uint32("qwen2.attention.head_count_kv", 4)
    writer.add_uint32("qwen2.block_count", 28)
    writer.add_float32("qwen2.rope.freq_base", 1000000.0)
    writer.add_bool("qwen2.attention.causal", True)
    writer.add_string("general.name", "sample")
    writer.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _write_image_gguf(path, architecture, tensor_names):
    writer = gguf.GGUFWriter(str(path), architecture)
    for name in tensor_names:
        writer.add_tensor(name, np.zeros((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_numeric_metadata_is_not_silently_dropped(sample_gguf):
    # The parts are 1-element 1-D arrays; numpy>=2 rejects int() on those, and a
    # bare except used to swallow it, leaving metadata string-only.
    md = loader.get_gguf_metadata(gguf.GGUFReader(str(sample_gguf)))
    assert md["qwen2.attention.head_count"] == 28
    assert md["qwen2.attention.head_count_kv"] == 4
    assert md["qwen2.block_count"] == 28
    assert md["qwen2.rope.freq_base"] == pytest.approx(1e6)
    assert md["qwen2.attention.causal"] is True
    assert md["general.name"] == "sample"


def test_head_counts_come_from_the_file_not_a_hardcoded_default(sample_gguf):
    md = loader.get_gguf_metadata(gguf.GGUFReader(str(sample_gguf)))
    assert loader.llama_head_counts(md, "qwen2") == (28, 4)
    # Missing metadata still has to produce something usable.
    assert loader.llama_head_counts({}, "qwen2") == (32, 8)


def test_permute_is_shape_preserving_and_head_count_sensitive():
    # 28 and 32 both divide 3584, so the wrong count reshapes without error and
    # only shows up as garbled output. Guard that they really do differ.
    weight = torch.arange(3584 * 4, dtype=torch.float32).reshape(3584, 4)
    a = loader.llama_permute({"q_proj.weight": weight.clone()}, 28, 4)["q_proj.weight"]
    b = loader.llama_permute({"q_proj.weight": weight.clone()}, 32, 8)["q_proj.weight"]
    assert a.shape == weight.shape == b.shape
    assert not torch.equal(a, b)


def test_ltx2_minimax_h3_signature_is_accepted_and_reported_truthfully(tmp_path):
    path = tmp_path / "minimax_h3.gguf"
    _write_image_gguf(path, "ltx2", loader.MINIMAX_H3_TENSOR_SIGNATURE)

    state_dict, extra = loader.gguf_sd_loader(str(path))

    assert set(state_dict) == loader.MINIMAX_H3_TENSOR_SIGNATURE
    assert extra["arch_str"] == "minimax_h3"
    assert extra["metadata"]["general.architecture"] == "ltx2"


@pytest.mark.parametrize("tensor_names", [
    {"patch_proj.weight"},
    {"video_patch_proj.weight"},
])
def test_unrelated_or_incomplete_ltx2_is_rejected(tmp_path, tensor_names):
    path = tmp_path / "unrelated_ltx2.gguf"
    _write_image_gguf(path, "ltx2", tensor_names)

    with pytest.raises(ValueError, match="Unexpected architecture type.*'ltx2'"):
        loader.gguf_sd_loader(str(path))


# --------------------------------------------------------------------------
# Qwen2.5-VL TE (Qwen-Image) loading — PR: multi-angle component support
# --------------------------------------------------------------------------

def _write_te_gguf(path, architecture):
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_qwen2vl_mmproj(path):
    """Synthetic qwen2vl mmproj: split Q/K/V, split patch embed, pos table."""
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("general.type", "mmproj")
    f32 = np.float32
    t = lambda *s: np.zeros(s, dtype=f32)
    writer.add_tensor("v.blk.0.attn_q.weight", t(4, 4))
    writer.add_tensor("v.blk.0.attn_q.bias", t(4))
    writer.add_tensor("v.blk.0.attn_k.weight", t(4, 4))
    writer.add_tensor("v.blk.0.attn_k.bias", t(4))
    writer.add_tensor("v.blk.0.attn_v.weight", t(4, 4))
    writer.add_tensor("v.blk.0.attn_v.bias", t(4))
    writer.add_tensor("v.blk.0.attn_out.weight", t(4, 4))
    writer.add_tensor("v.blk.0.ln1.weight", t(4))
    writer.add_tensor("v.blk.0.ln2.weight", t(4))
    writer.add_tensor("v.blk.0.ffn_gate.weight", t(8, 4))
    writer.add_tensor("v.blk.0.ffn_up.weight", t(8, 4))
    writer.add_tensor("v.blk.0.ffn_down.weight", t(4, 8))
    writer.add_tensor("mm.0.weight", t(16, 16))
    writer.add_tensor("mm.2.weight", t(4, 16))
    writer.add_tensor("v.post_ln.weight", t(4))
    writer.add_tensor("v.patch_embd.weight", t(4, 3, 2, 2))       # (out,c,h,w)
    writer.add_tensor("v.patch_embd.weight.1", t(4, 3, 2, 2))     # temporal half
    writer.add_tensor("v.position_embd.weight", t(4, 16))         # unused by comfy
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_qwen2vl_mmproj_qkv_recombine_and_cleanup(tmp_path):
    # Sibling naming mirrors the real pair the user loads:
    # "X-q4_0.gguf" + "X-mmproj-f16.gguf" in the same directory.
    _write_te_gguf(tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf", "qwen2vl")
    _write_qwen2vl_mmproj(tmp_path / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf")

    vsd = loader.gguf_mmproj_loader(str(tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf"))

    # fused qkv produced with q/k/v concatenated along dim 0
    assert vsd["visual.blocks.0.attn.qkv.weight"].shape == (12, 4)
    assert "visual.blocks.0.attn.qkv.bias" in vsd
    # consumed per-side tensors must NOT survive into the merged state dict
    for side in ("q", "k", "v"):
        assert f"visual.blocks.0.attn_{side}.weight" not in vsd
        assert f"visual.blocks.0.attn_{side}.bias" not in vsd
    # comfy's vision tower positions via rope; the pos table must be dropped
    assert "v.position_embd.weight" not in vsd
    # split temporal patch embeds concat to a single 5D conv weight
    assert vsd["visual.patch_embed.proj.weight"].shape == (4, 3, 2, 2, 2)
    assert "v.patch_embd.weight" not in vsd


def test_validate_te_type_requires_qwen_image_for_qwen2vl(tmp_path):
    te = tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf"
    _write_te_gguf(te, "qwen2vl")

    assert loader.read_gguf_arch(str(te)) == "qwen2vl"
    loader.validate_te_type(str(te), "qwen_image")  # correct type passes

    with pytest.raises(ValueError, match="type 'qwen_image'"):
        loader.validate_te_type(str(te), "sdxl")
    with pytest.raises(ValueError, match="type 'qwen_image'"):
        loader.validate_te_type(str(te), "stable_diffusion")


def test_mmproj_picked_by_hand_loads_as_a_vision_tower(tmp_path):
    # DualCLIPLoaderGGUF(clip_name1=<VL TE>, clip_name2=<mmproj>) used to hand
    # comfy two state dicts; TE detection then fell through to SDXLClipModel and
    # its clip_g never got loaded, dying at the first forward with
    # "'NoneType' object has no attribute 'device'". An explicitly picked mmproj
    # must map to the same vision keys the sibling auto-detection produces.
    mmproj = tmp_path / "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
    _write_qwen2vl_mmproj(mmproj)
    _write_te_gguf(tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf", "qwen2vl")

    assert loader.is_vision_projector(str(mmproj))
    assert not loader.is_vision_projector(str(tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf"))

    picked = loader.gguf_clip_loader(str(mmproj))
    auto = loader.gguf_mmproj_loader(str(tmp_path / "Qwen2.5-VL-7B-Instruct-q4_0.gguf"))
    assert set(picked) == set(auto)
    assert picked["visual.blocks.0.attn.qkv.weight"].shape == (12, 4)
    # no text-encoder keys: it must never look like a second TE to comfy
    assert not any(k.startswith(("model.", "text_model.")) for k in picked)

    # type validation is a no-op for it - the mmproj carries no type of its own
    for type_str in ("qwen_image", "sdxl", "stable_diffusion"):
        loader.validate_te_type(str(mmproj), type_str)


def test_validate_te_type_is_a_noop_without_a_requirement(tmp_path):
    te = tmp_path / "some-llama-te.gguf"
    _write_te_gguf(te, "llama")
    st = tmp_path / "not-a-gguf.safetensors"

    loader.validate_te_type(str(te), "anything")   # no known requirement
    loader.validate_te_type(str(st), "anything")   # not a gguf
    loader.validate_te_type(None, "anything")      # unresolved path from nodes
