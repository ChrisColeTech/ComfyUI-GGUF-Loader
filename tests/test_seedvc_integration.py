import ast
from pathlib import Path

from seedvc_utils import select_seedvc_identity


ROOT = Path(__file__).resolve().parents[1]


def test_seedvc_trigger_and_identity_precedence():
    generated = object()
    explicit = object()
    assert select_seedvc_identity(False, 2, None, generated) is generated
    assert select_seedvc_identity(False, 1, explicit, generated) is explicit
    assert select_seedvc_identity(False, 2, explicit, generated) is explicit
    assert select_seedvc_identity(False, 1, None, generated) is None
    assert select_seedvc_identity(True, 2, explicit, generated) is None


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
