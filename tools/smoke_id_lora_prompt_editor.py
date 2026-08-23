"""CPU-only smoke test for LTXV23IDLoraAssembler in nodes_ltx23.py.

Stubs the comfy-internal modules nodes_ltx23.py imports at the top level
(none of their real behavior is needed - LTXV23IDLoraAssembler is a pure
string formatter with no comfy/torch dependency of its own) so it can be
verified without a running ComfyUI or GPU.

Usage:  python tools/smoke_id_lora_prompt_editor.py
"""
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for name in ("comfy", "comfy.model_management", "comfy.nested_tensor",
            "comfy.sample", "comfy.samplers", "comfy.sd", "comfy.utils",
            "folder_paths", "node_helpers", "nodes"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["comfy"].model_management = sys.modules["comfy.model_management"]
sys.modules["comfy"].nested_tensor = sys.modules["comfy.nested_tensor"]
sys.modules["comfy"].sample = sys.modules["comfy.sample"]
sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]
sys.modules["comfy"].sd = sys.modules["comfy.sd"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]

sys.path.insert(0, str(REPO_ROOT))
import nodes_ltx23 as ltx23  # noqa: E402


def test_assemble_combines_all_three_fields_in_order():
    node = ltx23.LTXV23IDLoraAssembler()
    (tagged,) = node.assemble(
        visual="a woman in a kitchen",
        speech="Hello there.",
        sounds="warm, close mic, soft room tone",
    )
    assert tagged == (
        "[VISUAL]: a woman in a kitchen\n"
        "[SPEECH]: Hello there.\n"
        "[SOUNDS]: warm, close mic, soft room tone"
    )
    print("[ok] assemble: combines visual/speech/sounds in the canonical order")


def test_assemble_handles_blank_fields():
    node = ltx23.LTXV23IDLoraAssembler()
    (tagged,) = node.assemble(visual="", speech="", sounds="")
    assert tagged == "[VISUAL]: \n[SPEECH]: \n[SOUNDS]: "
    print("[ok] assemble: blank fields degrade cleanly, not a crash")


if __name__ == "__main__":
    test_assemble_combines_all_three_fields_in_order()
    test_assemble_handles_blank_fields()
    print("[ok] all LTXV23IDLoraAssembler smoke tests passed")
