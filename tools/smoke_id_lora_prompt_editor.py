"""CPU-only smoke test for LTXV23IDLoraPromptEditor in nodes_ltx23.py.

Stubs the comfy-internal modules nodes_ltx23.py imports at the top level
(none of their real behavior is needed here) so the parsing and the
keep-the-edit-or-refresh state machine can be verified without a running
ComfyUI or GPU.

The JS write-back itself is NOT covered here - it needs a real browser.

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

SAMPLE = """It's funny, isn't it? How we build these walls around ourselves.

---

[VISUAL]: Medium close-up shot, handheld, a woman (30s) in a bright living room.

[SPEECH]: "It's funny, isn't it? How we build these walls around ourselves."

[SOUNDS]: Female, early 30s. American accent, warm alto with a slight breathiness.
"""

SAMPLE_2 = """A totally different take.

---

[VISUAL]: Wide shot of a man on a beach at sunset.

[SPEECH]: "The tide always comes back."

[SOUNDS]: Male, 40s, gravelly, waves in the background.
"""


def _fresh(node_id="n1"):
    """A node id nothing has been remembered for yet."""
    ltx23._LAST_SOURCE.pop(node_id, None)
    return node_id


# ── parsing ─────────────────────────────────────────────────────────────────

def test_parse_extracts_all_three_fields():
    visual, speech, sounds = ltx23._parse_id_lora_prompt(SAMPLE)
    assert visual.startswith("Medium close-up shot")
    assert speech.startswith('"It')
    assert sounds.startswith("Female, early 30s")
    print("[ok] parse: all three fields extracted")


def test_speech_does_not_swallow_sounds():
    _, speech, _ = ltx23._parse_id_lora_prompt(SAMPLE)
    assert "[SOUNDS]" not in speech and "Female, early 30s" not in speech
    print("[ok] parse: SPEECH stops before SOUNDS, not greedy to end-of-string")


def test_parse_ignores_part1_preamble():
    visual, _, _ = ltx23._parse_id_lora_prompt(SAMPLE)
    assert "how we build these walls" not in visual.lower()
    print("[ok] parse: Part-1 preamble before the tags is ignored")


# ── state machine ───────────────────────────────────────────────────────────

def test_first_run_with_empty_boxes_fills_them():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    out = node.assemble(SAMPLE, "", "", "", unique_id=nid)
    tagged, v, s, snd = out["result"]
    expected = ltx23._parse_id_lora_prompt(SAMPLE)
    assert (v, s, snd) == expected
    assert tagged == f"[VISUAL]: {v}\n[SPEECH]: {s}\n[SOUNDS]: {snd}"
    assert out["ui"] == {"visual": [v], "speech": [s], "sounds": [snd]}
    print("[ok] first run, empty boxes: parsed from source, ui matches result")


def test_first_run_with_filled_boxes_preserves_them():
    # The restart case: ComfyUI restarted (no memory) but the workflow still
    # carries the user's saved edits in the widgets - do not clobber them.
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    _, v, s, snd = node.assemble(
        SAMPLE, "my visual", "my speech", "my sounds", unique_id=nid)["result"]
    assert (v, s, snd) == ("my visual", "my speech", "my sounds")
    print("[ok] first run, filled boxes: saved edits survive a restart")


def test_same_source_keeps_user_edit():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    node.assemble(SAMPLE, "", "", "", unique_id=nid)          # run 1: fills
    _, v, s, snd = node.assemble(                              # run 2: edited
        SAMPLE, "EDITED visual", "EDITED speech", "EDITED sounds",
        unique_id=nid)["result"]
    assert (v, s, snd) == ("EDITED visual", "EDITED speech", "EDITED sounds")
    print("[ok] same source: a manual edit survives the next run")


def test_changed_source_refreshes_all_three():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    node.assemble(SAMPLE, "", "", "", unique_id=nid)
    node.assemble(SAMPLE, "EDITED", "EDITED", "EDITED", unique_id=nid)
    _, v, s, snd = node.assemble(                              # new caption
        SAMPLE_2, "EDITED", "EDITED", "EDITED", unique_id=nid)["result"]
    assert (v, s, snd) == ltx23._parse_id_lora_prompt(SAMPLE_2)
    assert v.startswith("Wide shot") and "EDITED" not in v
    print("[ok] changed source: all three boxes refresh, stale edits dropped")


def test_two_nodes_keep_independent_state():
    node = ltx23.LTXV23IDLoraPromptEditor()
    a, b = _fresh("a"), _fresh("b")
    node.assemble(SAMPLE, "", "", "", unique_id=a)
    # b has never run: its own empty boxes should still get filled.
    _, v, _, _ = node.assemble(SAMPLE_2, "", "", "", unique_id=b)["result"]
    assert v.startswith("Wide shot")
    # a's remembered source is untouched by b's run.
    assert ltx23._LAST_SOURCE[a] == SAMPLE
    print("[ok] state is per-node: two editors do not share remembered source")


def test_blank_source_degrades_to_empty_fields():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    tagged, v, s, snd = node.assemble("", "", "", "", unique_id=nid)["result"]
    assert (v, s, snd) == ("", "", "")
    assert tagged == "[VISUAL]: \n[SPEECH]: \n[SOUNDS]: "
    print("[ok] blank source: degrades to empty fields, not a crash")


if __name__ == "__main__":
    test_parse_extracts_all_three_fields()
    test_speech_does_not_swallow_sounds()
    test_parse_ignores_part1_preamble()
    test_first_run_with_empty_boxes_fills_them()
    test_first_run_with_filled_boxes_preserves_them()
    test_same_source_keeps_user_edit()
    test_changed_source_refreshes_all_three()
    test_two_nodes_keep_independent_state()
    test_blank_source_degrades_to_empty_fields()
    print("[ok] all LTXV23IDLoraPromptEditor smoke tests passed")
