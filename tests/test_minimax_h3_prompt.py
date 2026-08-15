import pytest

from minimax_h3_prompt import (build_prompt, duration_seconds, ensure_shot_marker,
                               looks_structured, on_frame_grid, parse_fields,
                               reference_header, schema_problems, snap_to_frame_grid,
                               strip_shot1_timestamp)


# The instruction lines are fixed by MiniMax and must be reproduced verbatim, so
# these are the exact strings from the guide's canonical examples.

def test_i2va_instruction_is_verbatim():
    assert reference_header("i2va", 10.13) == (
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced.")


def test_fl2va_instruction_is_verbatim():
    assert reference_header("fl2va", 8.0) == (
        "How the reference pictures align with the target video — "
        "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
        "Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.")


def test_l2va_instruction_is_verbatim():
    assert reference_header("l2va", 6.0) == (
        "How the reference pictures align with the target video — "
        "<Picture 1> (from [Shot 1]) aligns with the 6.00-second mark of the target video.")


def test_t2va_has_no_instruction_line():
    assert reference_header("t2va", 10.13) == ""
    prompt = build_prompt("A ridge at dawn.", "Wind.", "Strings.",
                          mode="t2va", duration_s=10.13)
    assert prompt.startswith("integrated_multimodal_description:")


def test_duration_is_two_decimals_even_when_it_rounds():
    # S.SS must be exactly two decimals; 481 frames is 20.0416... seconds.
    assert "20.04-second mark" in reference_header("fl2va", duration_seconds(481))


def test_envelope_layout_is_blank_line_separated():
    prompt = build_prompt("[Shot 1] A ridge.", "Wind moves through pines.",
                          "Sparse high strings.", mode="i2va", duration_s=5.17)
    header, body = prompt.split("\n\n", 1)
    assert header.startswith("For the target video")
    assert body == ("integrated_multimodal_description: [Shot 1] A ridge.\n\n"
                    "overall_soundscape: Wind moves through pines.\n\n"
                    "non_diegetic_music: Sparse high strings.")
    assert looks_structured(prompt)


def test_missing_audio_fields_are_omitted_not_left_as_bare_labels():
    prompt = build_prompt("[Shot 1] A ridge.", None, "", mode="t2va")
    assert prompt == "integrated_multimodal_description: [Shot 1] A ridge."
    # "N/A" is a deliberate empty slot and must survive.
    assert "non_diegetic_music: N/A" in build_prompt("x", "y", "N/A", mode="t2va")


def test_shot_marker_added_only_when_absent():
    assert ensure_shot_marker("A ridge.") == "[Shot 1] A ridge."
    assert ensure_shot_marker("[Shot 1] A ridge.") == "[Shot 1] A ridge."
    assert ensure_shot_marker("[Shot 2] Later.") == "[Shot 2] Later."


@pytest.mark.parametrize("frames,valid", [(124, True), (243, True), (362, True),
                                          (481, True), (240, False), (100, False)])
def test_frame_grid(frames, valid):
    assert on_frame_grid(frames) is valid
    assert on_frame_grid(snap_to_frame_grid(frames))


def test_parse_fields_extracts_the_three_bodies():
    visual, soundscape, music = parse_fields(
        "integrated_multimodal_description: [Shot 1] A ridge.\n\n"
        "overall_soundscape: Wind.\n\n"
        "non_diegetic_music: Strings.")
    assert visual == "[Shot 1] A ridge."
    assert soundscape == "Wind."
    assert music == "Strings."


def test_parse_fields_survives_preamble_fences_and_bold_labels():
    # Small instruct models wrap output like this even when told not to.
    visual, soundscape, music = parse_fields(
        "Sure! Here is your prompt:\n\n```\n"
        "**integrated_multimodal_description:** [Shot 1] A ridge.\n\n"
        "**overall_soundscape:** Wind.\n\n"
        "**non_diegetic_music:** N/A\n```")
    assert visual == "[Shot 1] A ridge."
    assert soundscape == "Wind."
    assert music == "N/A"


def test_parse_fields_drops_reasoning_blocks():
    visual, _, _ = parse_fields(
        "<think>The user wants a ridge.</think>\n"
        "integrated_multimodal_description: [Shot 1] A ridge.")
    assert visual == "[Shot 1] A ridge."


def test_parse_fields_reports_nothing_when_labels_are_absent():
    assert parse_fields("just some prose about a ridge") == (None, None, None)


def test_cut_times_must_increase_and_stay_inside_the_clip():
    # Both are anti-patterns in the guide and neither fails at generation time.
    assert schema_problems("[Shot 2] At 00:03.000, cut.", 10.13) == []
    assert schema_problems("[Shot 2] At 00:12.000, cut.", 10.13)
    assert schema_problems(
        "[Shot 2] At 00:05.000, cut. [Shot 3] At 00:02.000, cut.", 10.13)


def test_shot_one_must_not_carry_a_timestamp():
    assert schema_problems("[Shot 1] Live-action, a wide shot frames the tower.", 10.13) == []
    assert schema_problems(
        "[Shot 1] Live-action, a wide shot at 00:00.000 frames the tower.", 10.13)


def test_closed_lips_belong_only_to_voiceover():
    ok = ("The man (S1) says in an off-screen voiceover: <d>[English] I remember.</d> "
          "while his lips remain completely closed.")
    assert schema_problems(ok, 10.13) == []
    # On-screen dialogue plus the voiceover clause is the failure we saw in practice.
    assert schema_problems(
        "The man (S1) says: <d>[English] Hello.</d> His lips remain closed.", 10.13)


def test_shot_one_timestamp_is_repaired_not_just_reported():
    # Models add this persistently even when told not to, and the clause is
    # removable without disturbing the rest of the sentence.
    prompt = build_prompt(
        "[Shot 1] At 00:00.000, a static shot frames the tower.", mode="t2va")
    assert "integrated_multimodal_description: [Shot 1] a static shot frames the tower."\
        == prompt
    assert schema_problems(prompt, 10.13) == []
    # A later shot's cut time must survive.
    kept = build_prompt("[Shot 1] A tower. [Shot 2] At 00:03.000, the camera cuts to sea.",
                        mode="t2va")
    assert "At 00:03.000" in kept
