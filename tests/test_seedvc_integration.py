import ast
from pathlib import Path

import torch

from seedvc_utils import select_seedvc_identity, seedvc_output_is_usable


ROOT = Path(__file__).resolve().parents[1]


def test_seedvc_trigger_and_identity_precedence():
    generated = object()
    explicit = object()
    assert select_seedvc_identity(False, 2, None, generated) is generated
    assert select_seedvc_identity(False, 1, explicit, generated) is explicit
    assert select_seedvc_identity(False, 2, explicit, generated) is explicit
    assert select_seedvc_identity(False, 1, None, generated) is None
    assert select_seedvc_identity(True, 2, explicit, generated) is None


def test_unusable_conversions_never_replace_generated_audio():
    source = torch.randn(1, 2, 48000) * 0.3
    assert seedvc_output_is_usable(source.clone(), source)
    # A conversion may re-time slightly, but not reshape or truncate the take.
    assert seedvc_output_is_usable(source[..., :47000], source)
    assert not seedvc_output_is_usable(source[..., :20000], source)
    assert not seedvc_output_is_usable(source[:, :1], source)
    assert not seedvc_output_is_usable(torch.zeros_like(source), source)
    assert not seedvc_output_is_usable(torch.full_like(source, float("nan")), source)
    assert not seedvc_output_is_usable(None, source)


def test_generate_validates_seedvc_output_before_replacing_audio():
    tree = ast.parse((ROOT / "nodes_scenema.py").read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "seedvc_output_is_usable" in called


def test_conversion_owns_or_borrows_its_bundle_but_never_leaks_one():
    tree = ast.parse((ROOT / "seedvc.py").read_text(encoding="utf-8"))
    convert = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "convert_voice")
    names = {arg.arg for arg in convert.args.kwonlyargs}
    assert {"bundle", "seed"} <= names
    assert "unload_after" not in names


def test_scenema_node_mappings_include_voice_clone():
    tree = ast.parse((ROOT / "nodes_scenema.py").read_text(encoding="utf-8"))
    mappings = [node for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                        for target in node.targets)]
    assert len(mappings) == 1
    keys = {key.value for key in mappings[0].value.keys}
    assert "ScenemaAudioGenerate" in keys
    assert "ScenemaAudioVoiceClone" in keys


def test_seedvc_modules_compile_and_use_relative_imports():
    for name in ("seedvc.py", "seedvc_arch.py", "seedvc_utils.py", "nodes_scenema.py"):
        compile((ROOT / name).read_bytes(), name, "exec")
    seedvc_tree = ast.parse((ROOT / "seedvc.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(seedvc_tree) if isinstance(node, ast.ImportFrom)]
    assert any(node.level == 1 and node.module == "seedvc_arch" for node in imports)
