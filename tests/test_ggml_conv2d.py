"""SenseNova's fm_head / vision stem stay F16 in GGUF. Those must run as real convs."""
import gguf
import torch

from conftest import load_pack_module

ops = load_pack_module("ops")
loader = load_pack_module("loader")


def _f16_conv_weight(shape=(8, 4, 3, 3)):
    plain = torch.randn(*shape, dtype=torch.float16)
    wrapped = ops.GGMLTensor(
        plain.clone(),
        tensor_type=gguf.GGMLQuantizationType.F16,
        tensor_shape=torch.Size(shape),
    )
    return plain, wrapped


def test_conv2d_copy_from_ggml_f16_matches_plain():
    plain, wrapped = _f16_conv_weight()
    dest = torch.zeros_like(plain)
    dest.copy_(wrapped)
    assert torch.equal(dest, plain)


def test_conv2d_forward_with_ggml_f16_weight_matches_plain():
    plain, wrapped = _f16_conv_weight()
    x = torch.randn(1, 4, 16, 16, dtype=torch.float16)
    conv = torch.nn.Conv2d(4, 8, 3, padding=1, bias=False).to(torch.float16)
    conv.weight.data.copy_(plain)
    ref = conv(x)
    conv.weight = torch.nn.Parameter(wrapped, requires_grad=False)
    got = conv(x)
    if isinstance(got, ops.GGMLTensor):
        got = got.as_subclass(torch.Tensor)
    assert got.shape == ref.shape
    assert torch.allclose(got.float(), ref.float(), atol=1e-3, rtol=1e-3), (
        f"maxabs={(got.float() - ref.float()).abs().max().item():.4f}"
    )


def test_loader_keeps_f16_conv_as_plain_tensor(tmp_path):
    """F16 4-D weights must not stay GGMLTensor — Conv2d never dequants that subclass."""
    import numpy as np
    from pathlib import Path

    path = tmp_path / "conv.gguf"
    w = gguf.GGUFWriter(str(path), arch="sensenova_u15")
    w.add_name("probe")
    arr = np.random.randn(8, 4, 3, 3).astype(np.float16)
    w.add_tensor("fm_modules.fm_head.conv1.weight", arr, raw_dtype=gguf.GGMLQuantizationType.F16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Bypass arch allowlist by calling the inner loop via gguf_sd_loader;
    # sensenova_u15 is on IMG_ARCH_LIST.
    sd, _ = loader.gguf_sd_loader(str(path), handle_prefix=None)
    t = sd["fm_modules.fm_head.conv1.weight"]
    assert tuple(t.shape) == (8, 4, 3, 3)
    assert not isinstance(t, ops.GGMLTensor), type(t)
    assert t.dtype == torch.float16
