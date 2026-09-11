import torch

from conftest import load_pack_module

ops = load_pack_module("ops")
dequant = ops.dequant


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


def test_quantized_embedding_emits_compute_dtype_not_float32():
    """SenseNova prefix is embed_tokens(Q4_K). Reporting dtype=bf16 made the
    Embedding path wipe out_dtype and dequant to float32, which then mixed
    with the bf16 image stream in attention."""
    import numpy as np
    gguf = dequant.gguf
    table = np.random.randn(256, 256).astype(np.float32)
    packed = gguf.quants.quantize(table, gguf.GGMLQuantizationType.Q8_0)
    weight = ops.GGMLTensor(
        torch.from_numpy(np.array(packed)),
        tensor_type=gguf.GGMLQuantizationType.Q8_0,
        tensor_shape=torch.Size([256, 256]),
    )
    assert weight.dtype == torch.bfloat16
    emb = ops.GGMLOps.Embedding(256, 256)
    object.__setattr__(emb, "weight", weight)
    out = emb.forward_ggml_cast_weights(torch.tensor([[0, 1, 2]]))
    assert out.dtype == torch.bfloat16, out.dtype
    assert tuple(out.shape) == (1, 3, 256)
    assert torch.isfinite(out.float()).all()
