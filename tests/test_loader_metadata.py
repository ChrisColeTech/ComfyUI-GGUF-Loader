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
