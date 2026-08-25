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

sys.path.insert(0, str(REPO_ROOT.parent))
# Fake the `cctech_gguf_pkg` and `cctech_gguf_pkg.nodes` packages (pointed at
# the repo root / its nodes/ dir) so importing nodes.ltx23 doesn't run either
# the real root __init__.py (needs comfy.utils) or the real nodes/__init__.py
# (aggregates every other node module).
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
import importlib
ltx23 = importlib.import_module("cctech_gguf_pkg.nodes.ltx23")  # noqa: E402

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
    tagged, v, s, snd, batch = out["result"]
    expected = ltx23._parse_id_lora_prompt(SAMPLE)
    assert (v, s, snd) == expected
    assert tagged == f"[VISUAL]: {v}\n[SPEECH]: {s}\n[SOUNDS]: {snd}"
    assert out["ui"] == {"visual": [v], "speech": [s], "sounds": [snd]}
    assert batch == [s]  # SAMPLE's SPEECH is a single line -> one clip
    print("[ok] first run, empty boxes: parsed from source, ui matches result")


def test_first_run_with_filled_boxes_preserves_them():
    # The restart case: ComfyUI restarted (no memory) but the workflow still
    # carries the user's saved edits in the widgets - do not clobber them.
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    _, v, s, snd, _ = node.assemble(
        SAMPLE, "my visual", "my speech", "my sounds", unique_id=nid)["result"]
    assert (v, s, snd) == ("my visual", "my speech", "my sounds")
    print("[ok] first run, filled boxes: saved edits survive a restart")


def test_same_source_keeps_user_edit():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    node.assemble(SAMPLE, "", "", "", unique_id=nid)          # run 1: fills
    _, v, s, snd, _ = node.assemble(                            # run 2: edited
        SAMPLE, "EDITED visual", "EDITED speech", "EDITED sounds",
        unique_id=nid)["result"]
    assert (v, s, snd) == ("EDITED visual", "EDITED speech", "EDITED sounds")
    print("[ok] same source: a manual edit survives the next run")


def test_changed_source_refreshes_all_three():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    node.assemble(SAMPLE, "", "", "", unique_id=nid)
    node.assemble(SAMPLE, "EDITED", "EDITED", "EDITED", unique_id=nid)
    _, v, s, snd, _ = node.assemble(                            # new caption
        SAMPLE_2, "EDITED", "EDITED", "EDITED", unique_id=nid)["result"]
    assert (v, s, snd) == ltx23._parse_id_lora_prompt(SAMPLE_2)
    assert v.startswith("Wide shot") and "EDITED" not in v
    print("[ok] changed source: all three boxes refresh, stale edits dropped")


def test_two_nodes_keep_independent_state():
    node = ltx23.LTXV23IDLoraPromptEditor()
    a, b = _fresh("a"), _fresh("b")
    node.assemble(SAMPLE, "", "", "", unique_id=a)
    # b has never run: its own empty boxes should still get filled.
    _, v, _, _, _ = node.assemble(SAMPLE_2, "", "", "", unique_id=b)["result"]
    assert v.startswith("Wide shot")
    # a's remembered source is untouched by b's run.
    assert ltx23._LAST_SOURCE[a] == SAMPLE
    print("[ok] state is per-node: two editors do not share remembered source")


def test_blank_source_degrades_to_empty_fields():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    tagged, v, s, snd, batch = node.assemble("", "", "", "", unique_id=nid)["result"]
    assert (v, s, snd) == ("", "", "")
    assert tagged == "[VISUAL]: \n[SPEECH]: \n[SOUNDS]: "
    assert batch == [""]  # never an empty list - nothing to index
    print("[ok] blank source: degrades to empty fields, not a crash")


# ── speech_text_batch / selector ─────────────────────────────────────────────

SAMPLE_MULTILINE = """A three-line delivery.

---

[VISUAL]: a woman speaking to camera

[SPEECH]: First clip line.
Second clip line.
Third clip line.

[SOUNDS]: calm, close mic
"""


def test_batch_splits_on_non_blank_lines():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    _, _, _, _, batch = node.assemble(
        SAMPLE_MULTILINE, "", "", "", unique_id=nid)["result"]
    assert batch == ["First clip line.", "Second clip line.", "Third clip line."]
    print("[ok] speech_text_batch: one clip per non-blank SPEECH line")


def test_batch_ignores_blank_separator_lines():
    node = ltx23.LTXV23IDLoraPromptEditor()
    nid = _fresh()
    _, _, _, _, batch = node.assemble(
        "\n---\n[VISUAL]: x\n[SPEECH]: line a\n\nline b\n[SOUNDS]: y",
        "", "", "", unique_id=nid)["result"]
    assert batch == ["line a", "line b"]  # the blank line is a separator, not a clip
    print("[ok] speech_text_batch: a blank line between clips is not itself a clip")


def test_selector_picks_the_requested_index():
    node = ltx23.LTXV23SpeechBatchSelector()
    batch = ["a", "b", "c"]
    assert node.select(batch, [1]) == ("b", 3)
    print("[ok] selector: picks the requested index")


def test_selector_supports_negative_index():
    node = ltx23.LTXV23SpeechBatchSelector()
    batch = ["a", "b", "c"]
    assert node.select(batch, [-1]) == ("c", 3)
    print("[ok] selector: negative index counts from the end, like Python")


def test_selector_clamps_out_of_range_index():
    node = ltx23.LTXV23SpeechBatchSelector()
    batch = ["a", "b", "c"]
    assert node.select(batch, [99]) == ("c", 3)
    assert node.select(batch, [-99]) == ("a", 3)
    print("[ok] selector: out-of-range index clamps instead of crashing")


def test_selector_count_matches_batch_length():
    node = ltx23.LTXV23SpeechBatchSelector()
    _, count = node.select(["x", "y", "z", "w"], [0])
    assert count == 4
    print("[ok] selector: count output matches the batch length, for driving a for-each loop")


# ── LTXV23IDLoraAssembler ────────────────────────────────────────────────────

def test_assembler_combines_all_three_fields_in_order():
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
    print("[ok] assembler: combines visual/speech/sounds in the canonical order")


def test_assembler_matches_editor_output_for_the_same_fields():
    # Both nodes format identically - the assembler is the editor's
    # formatting step, exposed standalone with no source/state.
    visual, speech, sounds = ltx23._parse_id_lora_prompt(SAMPLE)
    editor_tagged = ltx23.LTXV23IDLoraPromptEditor().assemble(
        SAMPLE, visual, speech, sounds, unique_id="cmp")["result"][0]
    assembler_tagged = ltx23.LTXV23IDLoraAssembler().assemble(visual, speech, sounds)[0]
    assert editor_tagged == assembler_tagged
    print("[ok] assembler: output matches LTXV23IDLoraPromptEditor's own formatting")


def test_assembler_handles_blank_fields():
    node = ltx23.LTXV23IDLoraAssembler()
    (tagged,) = node.assemble(visual="", speech="", sounds="")
    assert tagged == "[VISUAL]: \n[SPEECH]: \n[SOUNDS]: "
    print("[ok] assembler: blank fields degrade cleanly, not a crash")


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
    test_batch_splits_on_non_blank_lines()
    test_batch_ignores_blank_separator_lines()
    test_selector_picks_the_requested_index()
    test_selector_supports_negative_index()
    test_selector_clamps_out_of_range_index()
    test_selector_count_matches_batch_length()
    test_assembler_combines_all_three_fields_in_order()
    test_assembler_matches_editor_output_for_the_same_fields()
    test_assembler_handles_blank_fields()
    print("[ok] all LTXV23IDLoraPromptEditor smoke tests passed")
