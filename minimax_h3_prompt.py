# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""MiniMax H3 base-mode prompt schema (T2VA / I2VA / FL2VA / L2VA).

Source of truth: MiniMax's *Video Prompt Writing Guide* (base). A final prompt is

    <Part One instruction>          (keyframe modes only)
    <blank line>
    integrated_multimodal_description: [Shot 1] ...
    <blank line>
    overall_soundscape: ...
    <blank line>
    non_diegetic_music: ...

The Part One instruction strings are fixed by MiniMax and must be reproduced
verbatim, so they are built here rather than written by a language model.
"""

import re

SECTIONS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")

MODES = ("t2va", "i2va", "fl2va", "l2va")

FPS = 24.0
# H3 accepts frame counts on a n % 17 == 5 grid.
FRAME_GRID_MODULUS = 17
FRAME_GRID_REMAINDER = 5


def duration_seconds(frames):
    return int(frames) / FPS


def on_frame_grid(frames):
    return int(frames) % FRAME_GRID_MODULUS == FRAME_GRID_REMAINDER


def snap_to_frame_grid(frames):
    """Nearest valid H3 frame count, never below the first grid point."""
    frames = max(FRAME_GRID_REMAINDER, int(frames))
    down = frames - ((frames - FRAME_GRID_REMAINDER) % FRAME_GRID_MODULUS)
    up = down + FRAME_GRID_MODULUS
    return down if frames - down <= up - frames else up


def reference_header_i2va():
    return ("For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.")


def reference_header_fl2va(duration_s, last_shot_index=1):
    return ("How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {int(last_shot_index)}) aligns with the "
            f"{float(duration_s):.2f}-second mark of the target video.")


def reference_header_l2va(duration_s, last_shot_index=1):
    return ("How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {int(last_shot_index)}]) aligns with the "
            f"{float(duration_s):.2f}-second mark of the target video.")


def reference_header(mode, duration_s, last_shot_index=1):
    mode = (mode or "t2va").lower()
    if mode == "i2va":
        return reference_header_i2va()
    if mode == "fl2va":
        return reference_header_fl2va(duration_s, last_shot_index)
    if mode == "l2va":
        return reference_header_l2va(duration_s, last_shot_index)
    return ""


def ensure_shot_marker(visual):
    visual = (visual or "").strip()
    if not visual or "[Shot 1]" in visual or "[Shot 2]" in visual:
        return visual
    return f"[Shot 1] {visual}"


_SHOT1_LEADING_TIME_RE = re.compile(
    r"(\[Shot 1\]\s*)(?:at\s+\d{1,2}:\d{2}(?:\.\d{1,3})?\s*,?\s*)", re.IGNORECASE)


def strip_shot1_timestamp(visual):
    """Drop a cut time from the first shot.

    Shot 1 opens the clip and must carry no timestamp, but language models add
    "At 00:00.000" to it persistently even when told not to. The clause is
    removable without touching anything else, so repair rather than warn.
    """
    return _SHOT1_LEADING_TIME_RE.sub(r"\1", visual or "", count=1)


def build_prompt(visual, soundscape=None, music=None, *, mode="t2va",
                 duration_s=None, last_shot_index=1):
    """Assemble the Part One instruction and the three core fields.

    Empty soundscape/music are omitted rather than emitted as bare labels; pass
    the string ``"N/A"`` for an explicit empty slot, which the guide allows for
    deliberate silence or a scoreless clip.
    """
    visual = strip_shot1_timestamp(ensure_shot_marker(visual))
    chunks = [f"integrated_multimodal_description: {visual}"]
    if soundscape and str(soundscape).strip():
        chunks.append(f"overall_soundscape: {str(soundscape).strip()}")
    if music and str(music).strip():
        chunks.append(f"non_diegetic_music: {str(music).strip()}")
    body = "\n\n".join(chunks)

    header = reference_header(mode, duration_s, last_shot_index) if duration_s else ""
    return f"{header}\n\n{body}" if header else body


def looks_structured(text):
    return bool(text) and "integrated_multimodal_description:" in text


# Fences can appear anywhere once a model wraps only part of its reply.
_FENCE_RE = re.compile(r"^[ \t]*```[a-zA-Z]*[ \t]*$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_CUT_RE = re.compile(r"\bAt\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?", re.IGNORECASE)


def _label_pattern(name):
    # Tolerate the model bulleting or bolding the label it was asked for, with
    # the emphasis closing on either side of the colon (`**label:**`, `**label**:`).
    return re.compile(rf"^[\s>*\-#]*{name}\s*\**\s*:\s*\**[ \t]*",
                      re.IGNORECASE | re.MULTILINE)


def parse_fields(text):
    """Pull the three core fields out of a model's reply.

    Returns ``(visual, soundscape, music)`` with missing fields as ``None``.
    Anything before the first recognised label is dropped, which discards
    preambles like "Here is your prompt:".
    """
    text = _THINK_RE.sub("", text or "")
    text = _FENCE_RE.sub("", text.strip())

    hits = []
    for name in SECTIONS:
        match = _label_pattern(name).search(text)
        if match:
            hits.append((match.start(), match.end(), name))
    if not hits:
        return None, None, None

    hits.sort()
    found = {}
    for index, (_, body_start, name) in enumerate(hits):
        end = hits[index + 1][0] if index + 1 < len(hits) else len(text)
        value = text[body_start:end].strip()
        # A model that repeats the label inside its own answer would otherwise
        # smuggle it into the value.
        value = _label_pattern(name).sub("", value).strip()
        found[name] = value or None
    return tuple(found.get(name) for name in SECTIONS)


_SHOT1_TIME_RE = re.compile(r"\[Shot 1\][^[]{0,120}?\b(?:at|At)\s+\d{1,2}:\d{2}")
_VOICEOVER_RE = re.compile(r"off-screen voiceover", re.IGNORECASE)
_LIPS_RE = re.compile(r"lips remain", re.IGNORECASE)


def schema_problems(visual, duration_s):
    """Report timeline text that breaks the H3 contract without failing at run time.

    These are the anti-patterns the guide calls out that no downstream component
    rejects: the model still generates, just worse.
    """
    problems = []
    visual = visual or ""

    previous = None
    for match in _CUT_RE.finditer(visual):
        minutes, seconds, millis = match.group(1), match.group(2), match.group(3) or "0"
        at = int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000.0
        stamp = match.group(0)
        if at >= duration_s:
            problems.append(f"{stamp} is at or past the {duration_s:.2f}s clip end")
        if previous is not None and at <= previous:
            problems.append(f"{stamp} does not increase past the previous cut")
        previous = at

    if _SHOT1_TIME_RE.search(visual):
        problems.append("[Shot 1] carries a timestamp; the first shot starts the clip")

    # "lips remain closed" is the voiceover rule. On ordinary dialogue it asks
    # for a closed mouth over spoken words.
    if _LIPS_RE.search(visual) and len(_LIPS_RE.findall(visual)) > len(
            _VOICEOVER_RE.findall(visual)):
        problems.append(
            "'lips remain closed' appears more often than 'off-screen voiceover'; "
            "it belongs only after a voiceover line")
    return problems


# Kept for callers that only care about the timeline times.
cut_time_problems = schema_problems
