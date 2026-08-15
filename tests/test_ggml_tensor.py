import torch

import dequant
from conftest import load_pack_module

ops = load_pack_module("ops")


def _plain(values):
    """A GGMLTensor that carries no packed quantized payload."""
    return ops.GGMLTensor(values, tensor_type=None, tensor_shape=values.shape)


def test_clone_of_unquantized_tensor_is_a_real_copy():
    # comfy snapshots activations with .clone() and then writes the source
    # buffer in place (llama's per-layer `all_intermediate` capture). Handing
    # back an alias silently collapses every captured layer into the last one.
    tensor = _plain(torch.zeros(4))
    copy = tensor.clone()
    tensor.add_(1.0)
    assert not isinstance(copy, ops.GGMLTensor)
    assert copy.data_ptr() != tensor.data_ptr()
    assert float(copy.abs().max()) == 0.0


def test_clone_of_quantized_tensor_stays_shared():
    quantized = torch.zeros(64, dtype=torch.uint8)
    tensor = ops.GGMLTensor(
        quantized, tensor_type=dequant.gguf.GGMLQuantizationType.Q4_K,
        tensor_shape=torch.Size([2, 64]))
    assert dequant.is_quantized(tensor)
    assert tensor.clone() is tensor
