"""CheckpointLoaderGGUF is the GGUF equivalent of stock Load Checkpoint."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _class_src(name: str) -> ast.ClassDef:
    tree = ast.parse((ROOT / "nodes" / "gguf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in nodes/gguf.py")


def test_checkpoint_loader_gguf_matches_stock_outputs():
    cls = _class_src("CheckpointLoaderGGUF")
    assigns = {n.targets[0].id: n.value for n in cls.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    ret = assigns["RETURN_TYPES"]
    assert [elt.value for elt in ret.elts] == ["MODEL", "CLIP", "VAE"]
    names = assigns["RETURN_NAMES"]
    assert [elt.value for elt in names.elts] == ["model", "clip", "vae"]
    assert assigns["FUNCTION"].value == "load_checkpoint"


def test_checkpoint_loader_gguf_is_registered():
    tree = ast.parse((ROOT / "nodes" / "gguf.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NODE_CLASS_MAPPINGS" for t in node.targets):
            keys = [k.value for k in node.value.keys]
            assert "CheckpointLoaderGGUF" in keys
            assert keys[0] == "CheckpointLoaderGGUF"
            return
    raise AssertionError("NODE_CLASS_MAPPINGS not found")
