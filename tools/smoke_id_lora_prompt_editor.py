"""CPU-only smoke test for LTXV23IDLoraPromptEditor in nodes_ltx23.py.

Stubs the comfy-internal modules nodes_ltx23.py imports at the top level
(none of their real behavior is needed - only LTXV23IDLoraPromptEditor and
_parse_id_lora_prompt are exercised here) so the parsing/override/UI-payload
logic can be verified without a running ComfyUI or GPU.

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

[VISUAL]: Medium close-up shot, handheld camera angle, looking directly at a woman (30s) in a casual sweater and jeans standing in a bright living room.

[SPEECH]: "It's funny, isn't it? How we build these walls around ourselves."

[SOUNDS]: Female, early 30s. American accent, friendly and approachable. Warm alto voice with a slight breathiness.
"""


def test_parse_extracts_all_three_fields():
    visual, speech, sounds = ltx23._parse_id_lora_prompt(SAMPLE)
    assert visual.startswith("Medium close-up shot")
    assert speech.startswith('"It')
    assert sounds.startswith("Female, early 30s")
    print("[ok] _parse_id_lora_prompt: all three fields extracted")


def test_speech_does_not_swallow_sounds():
    # The exact bug a greedy `(.*)$` on a middle tag would reproduce.
    _, speech, _ = ltx23._parse_id_lora_prompt(SAMPLE)
    assert "[SOUNDS]" not in speech
    assert "Female, early 30s" not in speech
    print("[ok] _parse_id_lora_prompt: SPEECH stops before SOUNDS, not greedy to EOF")


def test_parse_ignores_part1_preamble():
    visual, _, _ = ltx23._parse_id_lora_prompt(SAMPLE)
    assert "how we build these walls" not in visual.lower()
    print("[ok] _parse_id_lora_prompt: Part-1 preamble before the tags is ignored")


def test_assemble_no_overrides_matches_parsed_fields():
    node = ltx23.LTXV23IDLoraPromptEditor()
    out = node.assemble(SAMPLE)
    visual, speech, sounds = ltx23._parse_id_lora_prompt(SAMPLE)
    tagged, v, s, snd = out["result"]
    assert (v, s, snd) == (visual, speech, sounds)
    assert tagged == f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}"
    assert out["ui"] == {"visual": [visual], "speech": [speech], "sounds": [sounds]}
    print("[ok] assemble: no overrides reproduces the parsed fields + ui payload")


def test_assemble_override_wins_and_others_untouched():
    node = ltx23.LTXV23IDLoraPromptEditor()
    visual, _, sounds = ltx23._parse_id_lora_prompt(SAMPLE)
    out = node.assemble(SAMPLE, speech_override="A brand new custom line.")
    tagged, v, s, snd = out["result"]
    assert s == "A brand new custom line."
    assert v == visual and snd == sounds
    assert "A brand new custom line." in tagged
    print("[ok] assemble: a single override wins for its field, others pass through")


def test_assemble_blank_source_produces_empty_fields():
    node = ltx23.LTXV23IDLoraPromptEditor()
    tagged, v, s, snd = node.assemble("")["result"]
    assert (v, s, snd) == ("", "", "")
    assert tagged == "[VISUAL]: \n[SPEECH]: \n[SOUNDS]: "
    print("[ok] assemble: blank source degrades to empty fields, not a crash")


if __name__ == "__main__":
    test_parse_extracts_all_three_fields()
    test_speech_does_not_swallow_sounds()
    test_parse_ignores_part1_preamble()
    test_assemble_no_overrides_matches_parsed_fields()
    test_assemble_override_wins_and_others_untouched()
    test_assemble_blank_source_produces_empty_fields()
    print("[ok] all LTXV23IDLoraPromptEditor smoke tests passed")
