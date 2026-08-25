import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_PKG = "comfy_gguf_under_test"


def _stub_comfy():
    """Minimal ComfyUI stand-in so pack modules import without a Comfy install.

    Only the submodules the pack imports at module scope are registered. In
    particular `comfy.utils` is left absent: the pack's own __init__.py probes
    for it to decide whether it is running as a custom node, and stubbing it
    sends pytest's rootdir import of __init__ down the node-registration path.
    """
    for mod in ("comfy", "comfy.ops", "comfy.lora", "comfy.model_management"):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    for attr in ("ops", "lora", "model_management"):
        setattr(sys.modules["comfy"], attr, sys.modules[f"comfy.{attr}"])
    if not hasattr(sys.modules["comfy.ops"], "manual_cast"):
        sys.modules["comfy.ops"].manual_cast = type("manual_cast", (), {
            "Linear": torch.nn.Linear, "Conv2d": torch.nn.Conv2d,
            "Embedding": torch.nn.Embedding, "LayerNorm": torch.nn.LayerNorm,
            "GroupNorm": torch.nn.GroupNorm})


def load_pack_module(name):
    """Import a repo module that uses relative imports (ops.py, loader.py, ...)."""
    full = f"{_PKG}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    _stub_comfy()
    if _PKG not in sys.modules:
        package = types.ModuleType(_PKG)
        package.__path__ = [str(ROOT)]
        sys.modules[_PKG] = package
    pkg_init = ROOT / name / "__init__.py"
    if pkg_init.is_file():
        # name is now a package (e.g. ops/) rather than a flat module.
        spec = importlib.util.spec_from_file_location(
            full, pkg_init, submodule_search_locations=[str(ROOT / name)])
    else:
        spec = importlib.util.spec_from_file_location(full, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module
