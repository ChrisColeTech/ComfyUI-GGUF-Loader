import importlib.util
import sys
import types
from pathlib import Path

import torch

import dequant

ROOT = Path(__file__).resolve().parents[1]


def _make_ops_module():
    """Import ops.py without pulling in ComfyUI, which tests do not install."""
    name = "comfy_gguf_test_pkg"
    if f"{name}.ops" in sys.modules:
        return sys.modules[f"{name}.ops"]
    for mod in ("comfy", "comfy.ops", "comfy.lora", "comfy.model_management"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    for attr in ("ops", "lora", "model_management"):
        setattr(sys.modules["comfy"], attr, sys.modules[f"comfy.{attr}"])
    sys.modules["comfy.ops"].manual_cast = type("manual_cast", (), {
        "Linear": torch.nn.Linear, "Conv2d": torch.nn.Conv2d,
        "Embedding": torch.nn.Embedding, "LayerNorm": torch.nn.LayerNorm,
        "GroupNorm": torch.nn.GroupNorm})

    package = types.ModuleType(name)
    package.__path__ = [str(ROOT)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(f"{name}.ops", ROOT / "ops.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ops = _make_ops_module()


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
