# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Vocal / instrumental stem separation with MelBandRoformer.

Splits any Comfy AUDIO into an acapella and an instrumental. Useful on its own,
and as a front end for music captioning: an isolated stem gives an audio-language
model a much cleaner read on the arrangement or the vocal than the full mix.

The weights are the public ``MelBandRoformer_fp16`` checkpoint; only the
inference architecture is ported (``melband_arch.py``), and the model is staged
through Comfy model management like every other model in this pack.
"""

import logging
from pathlib import Path

import torch
import torchaudio

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

logger = logging.getLogger(__name__)

MELBAND_SR = 44100
MODEL_FOLDER = "scenema-audio"
SUBDIR = "mel-band-roformer"
CATEGORY = "🤖 CCTech/MiniMax Music"

# ~8 s at 44.1 kHz. The model is fully attentional over the chunk, so this is a
# memory/quality knob, not a correctness one.
CHUNK_SAMPLES = 352800
OVERLAP_DIVISOR = 2


def _extras_roots():
    try:
        roots = folder_paths.get_folder_paths(MODEL_FOLDER)
    except KeyError:
        roots = [str(Path(folder_paths.models_dir) / MODEL_FOLDER)]
    return [Path(root) / "extras" / SUBDIR for root in roots]


def available_models():
    names = []
    for root in _extras_roots():
        if root.is_dir():
            names += [p.name for p in sorted(root.glob("*.safetensors"))]
    return names or ["none"]


def resolve_model(name):
    for root in _extras_roots():
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{name} not found. Place the MelBandRoformer checkpoint under "
        f"{_extras_roots()[0]}")


def _load_patched(path, device):
    from ..vendor.melband_arch import MelBandRoformer

    model = MelBandRoformer().eval()
    state = comfy.utils.load_torch_file(str(path), safe_load=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    del state
    if missing or unexpected:
        raise RuntimeError(
            f"MelBandRoformer checkpoint mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}")
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, load_device=device,
        offload_device=comfy.model_management.unet_offload_device())
    return model, patcher


@torch.inference_mode()
def _separate(model, wave, device):
    """``wave`` [2, T] at 44.1 kHz -> the isolated stem, same shape."""
    total = wave.shape[-1]
    overlap = CHUNK_SAMPLES // OVERLAP_DIVISOR
    step = CHUNK_SAMPLES - overlap
    fade_in = torch.linspace(0, 1, overlap)
    fade_out = torch.linspace(1, 0, overlap)

    result = torch.zeros_like(wave)
    weight = torch.zeros(total)
    position = 0
    while position < total:
        end = min(position + CHUNK_SAMPLES, total)
        chunk = wave[:, position:end]
        if chunk.shape[-1] < CHUNK_SAMPLES:
            chunk = torch.nn.functional.pad(chunk, (0, CHUNK_SAMPLES - chunk.shape[-1]))
        stem = model(chunk.unsqueeze(0).to(device))[0].float().cpu()[:, :end - position]

        length = end - position
        window = torch.ones(length)
        if position > 0:
            window[:min(overlap, length)] *= fade_in[:min(overlap, length)]
        if end < total:
            window[-min(overlap, length):] *= fade_out[:min(overlap, length)]
        result[:, position:end] += stem * window
        weight[position:end] += window
        position += step
    return result / weight.clamp(min=1e-8)


def separate_stems(audio, model_name):
    """Return ``(vocals, instrumental)`` AUDIO dicts at the source rate/layout."""
    waveform = audio["waveform"]
    source_sr = int(audio["sample_rate"])
    if waveform.ndim != 3 or waveform.shape[0] < 1:
        raise ValueError("audio waveform must have shape [B, C, T]")
    wave = waveform[0].detach().float()
    channels = wave.shape[0]
    if not torch.isfinite(wave).all():
        raise ValueError("audio contains NaN or infinite samples")

    if source_sr != MELBAND_SR:
        wave = torchaudio.functional.resample(wave, source_sr, MELBAND_SR)
    if channels == 1:
        wave = wave.repeat(2, 1)
    elif channels > 2:
        wave = wave[:2]

    device = comfy.model_management.get_torch_device()
    model, patcher = _load_patched(resolve_model(model_name), device)
    try:
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        logger.info("MelBandRoformer: separating %.1fs of audio",
                    wave.shape[-1] / MELBAND_SR)
        vocals = _separate(model, wave, device)
    finally:
        try:
            comfy.model_management.unload_model_and_clones(patcher)
            patcher.cleanup()
        except Exception:
            logger.exception("Error unloading MelBandRoformer")
        comfy.model_management.soft_empty_cache()

    # The model estimates one stem; the other is exactly what it left behind.
    instrumental = wave - vocals

    out = []
    for stem in (vocals, instrumental):
        if source_sr != MELBAND_SR:
            stem = torchaudio.functional.resample(stem, MELBAND_SR, source_sr)
        if channels == 1:
            stem = stem.mean(0, keepdim=True)
        out.append({"waveform": stem.unsqueeze(0).contiguous(),
                    "sample_rate": source_sr})
    return out[0], out[1]


class AudioStemSplit:
    """Split a track into acapella and instrumental."""

    CATEGORY = CATEGORY
    TITLE = "Audio Stem Split ⚡"
    SEARCH_ALIASES = ['split audio', 'vocal removal', 'stem separation', 'isolate vocals', 'extract instrumental']

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
            "model_name": (available_models(), {
                "tooltip": "MelBandRoformer checkpoint from "
                           "models/scenema-audio/extras/mel-band-roformer."}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "instrumental")
    FUNCTION = "split"
    DESCRIPTION = ("Separate a song into an acapella and an instrumental with "
                   "MelBandRoformer. Sample rate and channel count are preserved.")

    def split(self, audio, model_name):
        return separate_stems(audio, model_name)


NODE_CLASS_MAPPINGS = {
    "AudioStemSplit": AudioStemSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioStemSplit": AudioStemSplit.TITLE,
}
