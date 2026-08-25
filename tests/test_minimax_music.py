import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "nodes" / "minimax_music.py").read_text(encoding="utf-8")


def _generate_body():
    tree = ast.parse(SOURCE)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "generate")
    return node


def test_empty_lyrics_are_an_instrumental_not_an_error():
    # comfy's own MiniMaxMusic3TextEncode accepts empty lyrics, and
    # minimax_music.prompt.normalize_lyrics("") builds a valid "[start]"
    # section, so rejecting them only blocks instrumental tracks.
    raised = {ast.unparse(n.exc) for n in ast.walk(_generate_body())
              if isinstance(n, ast.Raise) and n.exc is not None}
    assert not any("lyrics" in text for text in raised), raised


def test_caption_is_still_required():
    raised = {ast.unparse(n.exc) for n in ast.walk(_generate_body())
              if isinstance(n, ast.Raise) and n.exc is not None}
    assert any("caption must not be empty" in text for text in raised), raised
