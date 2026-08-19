"""Type-widget construction for the GGUF CLIP loader nodes.

All four GGUF CLIP loaders (single/dual/triple/quadruple) expose ONE
canonical type list: the union of every type list core's own loader classes
provide. Core deliberately splits its vocabulary (the single list carries
qwen_image/minimax, the dual list sdxl/flux, triple/quadruple may only exist
as comfy_extras nodes on newer cores), but the GGUF loaders take arbitrary
TE files regardless of slot count, so the owner-facing rule is: identical
options everywhere. These tests pin that under every core layout the pack
can meet in the wild — plus the repo rule that a type widget is never
silently dropped.

Import isolation note: nodes.py needs `comfy.utils` etc. to be importable
at module scope, but the pack's own __init__.py (which pytest imports as
a root-level Package) must NOT see comfy.utils or it walks the custom-node
registration path and dies on a relative import (see conftest). So every
stub registered here is removed again before the import returns.
"""
import importlib.util
import sys
import types

import pytest

from conftest import ROOT, _PKG

# Stands for whatever core's single CLIPLoader lists (has qwen_image on
# every core that supports Qwen-Image at all).
SINGLE_TYPES = ["stable_diffusion", "sdxl", "sd3", "flux", "qwen_image"]
DUAL_TYPES = ["sdxl", "sd3", "flux", "hunyuan_video"]  # no qwen_image by design
LEGACY_MULTI_TYPES = ["sd3", "flux"]
# The canonical union: single order first, dual-only values appended.
UNION_TYPES = SINGLE_TYPES + ["hunyuan_video"]

_COMFY_SUBMODULES = (
    "comfy.sd", "comfy.float", "comfy.utils", "comfy.sample",
    "comfy.nested_tensor", "comfy.model_patcher",
)


def _loader_class(type_values):
    class _Loader:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"clip_name": (["x.safetensors"],), "type": (type_values,)}}
    return _Loader


def _core_layout(with_single=True, with_dual=True, with_triple=False):
    mod = types.ModuleType("nodes")
    if with_single:
        mod.CLIPLoader = _loader_class(SINGLE_TYPES)
    if with_dual:
        mod.DualCLIPLoader = _loader_class(DUAL_TYPES)
    if with_triple:
        # Legacy layout: Triple/Quadruple still live in core's nodes.py.
        mod.TripleCLIPLoader = _loader_class(LEGACY_MULTI_TYPES)
        mod.QuadrupleCLIPLoader = _loader_class(LEGACY_MULTI_TYPES)
    return mod


def _import_nodes_module():
    """Import the pack's nodes.py against stub Comfy modules.

    The heavyweight sibling modules (scenema, stems, ...) are pre-stubbed
    in sys.modules so nodes.py's tail imports do not drag their deps in;
    only the type-widget plumbing itself is under test here. sys.modules
    is restored before returning so no other test can see the stubs.
    """
    import torch
    saved_modules = dict(sys.modules)
    comfy = sys.modules.setdefault("comfy", types.ModuleType("comfy"))
    saved_attrs = {n: hasattr(comfy, n) for n in _COMFY_SUBMODULES}
    try:
        for name in ("comfy", "comfy.ops", "comfy.lora", "comfy.model_management"):
            sys.modules.setdefault(name, types.ModuleType(name))
        for attr in ("ops", "lora", "model_management"):
            if not hasattr(comfy, attr):
                setattr(comfy, attr, sys.modules[f"comfy.{attr}"])
        if not hasattr(sys.modules["comfy.ops"], "manual_cast"):
            sys.modules["comfy.ops"].manual_cast = type("manual_cast", (), {
                "Linear": torch.nn.Linear, "Conv2d": torch.nn.Conv2d,
                "Embedding": torch.nn.Embedding, "LayerNorm": torch.nn.LayerNorm,
                "GroupNorm": torch.nn.GroupNorm})
        for name in _COMFY_SUBMODULES:
            full = sys.modules.setdefault(name, types.ModuleType(name))
            setattr(comfy, name.split(".", 1)[1], full)
        if not hasattr(sys.modules["comfy.model_patcher"], "ModelPatcher"):
            sys.modules["comfy.model_patcher"].ModelPatcher = type("ModelPatcher", (), {})

        fp = sys.modules.setdefault("folder_paths", types.ModuleType("folder_paths"))
        fp.folder_names_and_paths = {}
        fp.get_filename_list = lambda key: []
        fp.get_full_path = lambda key, name: None
        fp.get_folder_paths = lambda key: []

        sys.modules.setdefault("nodes", _core_layout())

        if _PKG not in sys.modules:
            package = types.ModuleType(_PKG)
            package.__path__ = [str(ROOT)]
            sys.modules[_PKG] = package
        for name in ("nodes_extra", "nodes_scenema", "nodes_minimax_music",
                     "nodes_stems", "nodes_minimax_h3", "nodes_ltx25"):
            full = f"{_PKG}.{name}"
            if full not in sys.modules:
                stub = types.ModuleType(full)
                stub.NODE_CLASS_MAPPINGS = {}
                stub.NODE_DISPLAY_NAME_MAPPINGS = {}
                sys.modules[full] = stub

        full = f"{_PKG}.nodes"
        spec = importlib.util.spec_from_file_location(full, ROOT / "nodes.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for key in list(sys.modules):
            if key not in saved_modules:
                del sys.modules[key]
        for name, had in saved_attrs.items():
            if not had and hasattr(comfy, name):
                delattr(comfy, name)


nodes_mod = _import_nodes_module()

ALL_LOADER_CLASSES = (
    "CLIPLoaderGGUF",
    "DualCLIPLoaderGGUF",
    "TripleCLIPLoaderGGUF",
    "QuadrupleCLIPLoaderGGUF",
)


def _type_of(cls):
    return cls.INPUT_TYPES()["required"]["type"]


def _required_of(cls):
    return cls.INPUT_TYPES()["required"]


@pytest.fixture
def modern_core(monkeypatch):
    """Triple/Quadruple moved to comfy_extras (ComfyUI 0.33-era layout)."""
    monkeypatch.setattr(nodes_mod, "nodes", _core_layout(with_triple=False))
    return nodes_mod


@pytest.fixture
def legacy_core(monkeypatch):
    monkeypatch.setattr(nodes_mod, "nodes", _core_layout(with_triple=True))
    return nodes_mod


def test_every_loader_shows_the_same_union_list(modern_core):
    for name in ALL_LOADER_CLASSES:
        widget = _type_of(getattr(modern_core, name))
        assert widget == (UNION_TYPES,), f"{name} list diverged: {widget}"
        assert "qwen_image" in widget[0]


def test_dual_loader_offers_qwen_image(modern_core):
    # The owner-facing bug: dual used to show core's dual-only list, so
    # qwen_image was unselectable on a dual-template workflow.
    assert "qwen_image" in _type_of(modern_core.DualCLIPLoaderGGUF)[0]
    # Dual-only values survive the merge too.
    assert "hunyuan_video" in _type_of(modern_core.CLIPLoaderGGUF)[0]


def test_slot_widgets_survive(modern_core):
    for cls, names in (
        (modern_core.CLIPLoaderGGUF, ("clip_name",)),
        (modern_core.DualCLIPLoaderGGUF, ("clip_name1", "clip_name2")),
        (modern_core.TripleCLIPLoaderGGUF, ("clip_name1", "clip_name2", "clip_name3")),
        (modern_core.QuadrupleCLIPLoaderGGUF, ("clip_name1", "clip_name2", "clip_name3", "clip_name4")),
    ):
        required = _required_of(cls)
        assert set(names) <= set(required)
        assert required["type"] == (UNION_TYPES,)


def test_legacy_core_layout_merges_the_same_union(legacy_core):
    # Legacy multi lists are a subset here; the union must be identical
    # across all four loaders and stable regardless of core layout.
    for name in ALL_LOADER_CLASSES:
        assert _type_of(getattr(legacy_core, name)) == (UNION_TYPES,), name


def test_no_core_lists_at_all_raises(monkeypatch):
    monkeypatch.setattr(nodes_mod, "nodes", _core_layout(with_single=False, with_dual=False))
    with pytest.raises(RuntimeError, match="type.*dropdown"):
        nodes_mod.CLIPLoaderGGUF.INPUT_TYPES()
    with pytest.raises(RuntimeError, match="type.*dropdown"):
        nodes_mod.DualCLIPLoaderGGUF.INPUT_TYPES()


def test_broken_core_input_types_falls_back_to_remaining_lists(monkeypatch):
    # One broken class must not take the dropdown down with it; only when
    # NOTHING yields a list may the loader raise.
    class _Broken:
        @classmethod
        def INPUT_TYPES(cls):
            raise KeyError("required")

    mod = _core_layout(with_single=False)
    mod.CLIPLoader = _Broken  # single broken, dual still fine
    monkeypatch.setattr(nodes_mod, "nodes", mod)
    widget = _type_of(nodes_mod.CLIPLoaderGGUF)
    assert widget == (DUAL_TYPES,)


def test_lists_are_not_shared_mutable_state(modern_core):
    # Mutating one loader's returned list must not leak into another's.
    a = _type_of(modern_core.CLIPLoaderGGUF)[0]
    a.append("garbage")
    b = _type_of(modern_core.DualCLIPLoaderGGUF)[0]
    assert "garbage" not in b
