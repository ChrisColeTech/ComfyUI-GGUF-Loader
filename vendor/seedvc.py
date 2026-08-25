# Apache-2.0. Adapted from billwuhao/ComfyUI_Seed-VC and Plachtaa/seed-vc.
"""Comfy-native, offline SeedVC inference for Scenema Audio."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torchaudio
import yaml
from torch.nn import functional as F

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths
from comfy.audio_encoders.whisper import AudioEncoder, WhisperFeatureExtractor
from comfy.ldm.mmaudio.vae.bigvgan import BigVGANVocoder

from .seedvc_arch import CAMPPlus, SeedVCModel

logger = logging.getLogger(__name__)

SEEDVC_SR = 22050
FEATURE_SR = 16000
DEFAULT_STEPS = 25
DEFAULT_CFG_RATE = 0.5
DIT_FILENAME = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
CONFIG_FILENAME = "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
MODEL_FOLDER = "scenema-audio"

# Keep this independent of node registration so the runtime can also be used by
# other custom nodes while respecting Comfy's configured models root.
folder_paths.add_model_folder_path(
    MODEL_FOLDER, str(Path(folder_paths.models_dir) / MODEL_FOLDER), is_default=True)


def _extras_roots():
    try:
        roots = folder_paths.get_folder_paths(MODEL_FOLDER)
    except KeyError:
        roots = [str(Path(folder_paths.models_dir) / MODEL_FOLDER)]
    return [Path(root) / "extras" for root in roots]


def resolve_model_paths(root=None):
    """Resolve and validate the four offline component directories/files."""
    roots = [Path(root)] if root is not None else _extras_roots()
    required = {
        "dit": Path("seedvc") / DIT_FILENAME,
        "config": Path("seedvc") / CONFIG_FILENAME,
        "campplus": Path("campplus") / "campplus_cn_common.bin",
        "bigvgan_config": Path("bigvgan") / "config.json",
        "bigvgan": Path("bigvgan") / "bigvgan_generator.pt",
        "whisper_config": Path("whisper-small") / "config.json",
        "whisper_preprocessor": Path("whisper-small") / "preprocessor_config.json",
        "whisper": Path("whisper-small") / "model.safetensors",
    }
    for candidate in roots:
        paths = {name: candidate / rel for name, rel in required.items()}
        if all(path.is_file() for path in paths.values()):
            paths["root"] = candidate
            return paths
    expected = roots[0]
    missing = [str(rel) for rel in required.values() if not (expected / rel).is_file()]
    raise FileNotFoundError(f"SeedVC files missing under {expected}: {', '.join(missing)}")


def is_available(root=None):
    try:
        resolve_model_paths(root)
        return True
    except FileNotFoundError:
        return False


def _strip_module_in_place(sd):
    for key in list(sd):
        if key.startswith("module."):
            sd[key[7:]] = sd.pop(key)
    return sd


def _load_checkpoint(path):
    return torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)


def _load_whisper_encoder(path):
    from safetensors import safe_open

    prefix = "model.encoder."
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return {key[len(prefix):]: checkpoint.get_tensor(key)
                for key in checkpoint.keys() if key.startswith(prefix)}


def _fuse_weight_norm(sd):
    for key in list(sd):
        if not key.endswith(".weight_g"):
            continue
        stem = key[:-9]
        vkey = stem + ".weight_v"
        if vkey in sd:
            v = sd.pop(vkey)
            g = sd.pop(key)
            dims = tuple(range(1, v.ndim))
            norm = torch.linalg.vector_norm(v.float(), dim=dims, keepdim=True)
            sd[stem + ".weight"] = v * (g.float() / norm.clamp_min(1e-12)).to(v.dtype)
    return sd


class _SeedVCVocoder(BigVGANVocoder):
    def __init__(self, config):
        super().__init__(config)
        self.use_tanh_at_final = config.get("use_tanh_at_final", True)
        if not config.get("use_bias_at_final", True):
            channels = config["upsample_initial_channel"] // (2 ** len(config["upsample_rates"]))
            self.conv_post = comfy.ops.disable_weight_init.Conv1d(
                channels, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            x = sum(self.resblocks[i * self.num_kernels + j](x)
                    for j in range(self.num_kernels)) / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else x.clamp(-1, 1)


class SeedVCBundle:
    """Loaded component owner. All large models are Comfy CoreModelPatchers."""

    def __init__(self, paths, device=None):
        self.paths = paths
        self.device = device or model_management.get_torch_device()
        self.offload_device = model_management.unet_offload_device()
        self.patchers = []
        self.feature_extractor = None
        self.seed = self.camp = self.whisper = self.vocoder = None
        try:
            self._load()
        except Exception:
            self.unload()
            raise

    def _patch(self, model):
        model.eval()
        patcher = comfy.model_patcher.CoreModelPatcher(
            model, load_device=self.device, offload_device=self.offload_device)
        self.patchers.append(patcher)
        return model, patcher

    def _load(self):
        config = yaml.safe_load(self.paths["config"].read_text(encoding="utf-8"))
        spect = config["preprocess_params"]["spect_params"]
        if int(config["preprocess_params"]["sr"]) != SEEDVC_SR:
            raise ValueError(f"SeedVC config must use {SEEDVC_SR} Hz audio")
        self.hop_length = int(spect["hop_length"])
        self.max_context_window = SEEDVC_SR // self.hop_length * 30
        self.overlap_frames = 16
        self.overlap_samples = self.overlap_frames * self.hop_length

        checkpoint = _load_checkpoint(self.paths["dit"])
        nested = checkpoint.pop("net")
        seed, self.seed_patcher = self._patch(SeedVCModel())
        cfm_sd = _strip_module_in_place(nested.pop("cfm"))
        missing, unexpected = seed.cfm.load_state_dict(cfm_sd, strict=False)
        del cfm_sd
        if missing or unexpected:
            raise RuntimeError(f"SeedVC CFM checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        missing, unexpected = seed.length_regulator.load_state_dict(
            _strip_module_in_place(nested.pop("length_regulator")), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"SeedVC regulator checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.seed = seed
        del nested, checkpoint

        camp, self.camp_patcher = self._patch(CAMPPlus())
        camp_sd = comfy.utils.load_torch_file(str(self.paths["campplus"]), safe_load=True)
        camp_sd = {key.replace("xvector.stats", "stats").replace("xvector.dense", "dense"): value
                   for key, value in camp_sd.items()}
        missing, unexpected = camp.load_state_dict(camp_sd, strict=False)
        missing = [key for key in missing if not key.startswith("stats.")]
        if missing or unexpected:
            raise RuntimeError(f"CAMPPlus checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.camp = camp

        whisper_cfg = json.loads(self.paths["whisper_config"].read_text(encoding="utf-8"))
        preprocessor = json.loads(
            self.paths["whisper_preprocessor"].read_text(encoding="utf-8"))
        expected_preprocessor = {
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 80,
            "sampling_rate": FEATURE_SR,
            "hop_length": 160,
            "n_fft": 400,
            "chunk_length": 30,
        }
        invalid = {key: (preprocessor.get(key), value)
                   for key, value in expected_preprocessor.items()
                   if preprocessor.get(key) != value}
        mel_filters = preprocessor.get("mel_filters")
        if (not isinstance(mel_filters, list) or len(mel_filters) != 80
                or any(not isinstance(row, list) or len(row) != 201
                       for row in mel_filters)):
            invalid["mel_filters"] = ("invalid shape", "[80, 201]")
        if invalid:
            raise ValueError(f"Unsupported Whisper preprocessor_config values: {invalid}")
        whisper_dtype = (torch.float16 if self.device.type == "cuda"
                         and model_management.should_use_fp16(device=self.device)
                         else torch.float32)
        whisper = AudioEncoder(
            n_mels=int(whisper_cfg.get("num_mel_bins", 80)),
            n_ctx=int(whisper_cfg.get("max_source_positions", 1500)),
            n_state=int(whisper_cfg.get("d_model", 768)),
            n_head=int(whisper_cfg.get("encoder_attention_heads", 12)),
            n_layer=int(whisper_cfg.get("encoder_layers", 12)),
            dtype=whisper_dtype,
            device=self.offload_device,
            operations=comfy.ops.disable_weight_init,
        )
        whisper, self.whisper_patcher = self._patch(whisper)
        whisper_sd = _load_whisper_encoder(self.paths["whisper"])
        missing, unexpected = whisper.load_state_dict(whisper_sd, strict=False, assign=self.whisper_patcher.is_dynamic())
        del whisper_sd
        if missing or unexpected:
            raise RuntimeError(f"Whisper checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.whisper = whisper
        self.feature_extractor = WhisperFeatureExtractor(
            n_mels=expected_preprocessor["feature_size"], device=self.device)

        vocoder_config = json.loads(self.paths["bigvgan_config"].read_text(encoding="utf-8"))
        vocoder, self.vocoder_patcher = self._patch(_SeedVCVocoder(vocoder_config))
        vocoder_checkpoint = _load_checkpoint(self.paths["bigvgan"])
        vocoder_sd = vocoder_checkpoint.pop("generator")
        missing, unexpected = vocoder.load_state_dict(_fuse_weight_norm(vocoder_sd), strict=False)
        del vocoder_sd, vocoder_checkpoint
        if missing or unexpected:
            raise RuntimeError(f"BigVGAN checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.vocoder = vocoder

        model_management.load_models_gpu(self.patchers, force_full_load=True)

    def unload(self):
        for patcher in reversed(self.patchers):
            try:
                model_management.unload_model_and_clones(patcher)
                patcher.cleanup()
            except Exception:
                logger.exception("Error unloading a SeedVC component")
        self.patchers.clear()
        for name in ("seed", "camp", "whisper", "vocoder",
                     "seed_patcher", "camp_patcher", "whisper_patcher",
                     "vocoder_patcher"):
            setattr(self, name, None)
        self.feature_extractor = None
        model_management.soft_empty_cache()

    def semantics(self, audio_16k):
        features = self.feature_extractor(audio_16k.unsqueeze(1)).to(self.device)
        features = features.to(next(self.whisper.parameters()).dtype)
        hidden, _ = self.whisper(features)
        return hidden[:, :audio_16k.shape[-1] // 320 + 1].float()


def load_seed_vc(device=None, root=None):
    """Load an owned, non-singleton bundle from registered offline model paths."""
    return SeedVCBundle(resolve_model_paths(root), device=device)


def unload_seed_vc(bundle):
    if bundle is not None:
        bundle.unload()


def _mono_audio(audio, name):
    if isinstance(audio, dict):
        audio = audio.get("waveform")
    if not isinstance(audio, torch.Tensor):
        raise TypeError(f"{name} must be a tensor or Comfy AUDIO dictionary")
    if audio.ndim == 1:
        audio = audio[None, None]
    elif audio.ndim == 2:
        audio = audio[None]
    if audio.ndim != 3 or audio.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,C,T], [C,T], or [T]")
    if audio.shape[1] < 1 or audio.shape[2] < 1:
        raise ValueError(f"{name} must not be empty")
    if not torch.is_floating_point(audio):
        raise TypeError(f"{name} waveform must have a floating-point dtype")
    if not torch.isfinite(audio).all():
        raise ValueError(f"{name} contains NaN or infinite samples")
    return audio.detach().float().mean(1), audio.shape[1]


def _resample(audio, source_sr, target_sr):
    if not isinstance(source_sr, int) or isinstance(source_sr, bool) or source_sr <= 0:
        raise ValueError("sample rates must be positive integers")
    if not isinstance(target_sr, int) or isinstance(target_sr, bool) or target_sr <= 0:
        raise ValueError("sample rates must be positive integers")
    return audio if source_sr == target_sr else torchaudio.functional.resample(audio, source_sr, target_sr)


def _mel(audio):
    audio = F.pad(audio.unsqueeze(1), (384, 384), mode="reflect").squeeze(1)
    window = torch.hann_window(1024, device=audio.device)
    spec = torch.stft(audio, 1024, 256, 1024, window, center=False, return_complex=True)
    magnitude = torch.sqrt(spec.abs().square() + 1e-9)
    mel = torchaudio.functional.melscale_fbanks(513, 0, 11025, 80, 22050,
                                                norm="slaney", mel_scale="slaney").to(audio.device)
    return torch.log(torch.clamp(mel.transpose(0, 1) @ magnitude, min=1e-5))


def _long_semantics(bundle, audio):
    if audio.shape[-1] <= FEATURE_SR * 30:
        return bundle.semantics(audio)
    overlap = FEATURE_SR * 5
    parts, buffer, offset = [], None, 0
    while offset < audio.shape[-1]:
        fresh = FEATURE_SR * (30 if buffer is None else 25)
        chunk = audio[:, offset:offset + fresh]
        if buffer is not None:
            chunk = torch.cat([buffer, chunk], -1)
        encoded = bundle.semantics(chunk)
        parts.append(encoded if buffer is None else encoded[:, 250:])
        buffer = chunk[:, -overlap:]
        offset += fresh
    return torch.cat(parts, 1)


def _crossfade(previous, current, overlap):
    overlap = min(overlap, previous.shape[-1], current.shape[-1])
    phase = torch.linspace(0, torch.pi / 2, overlap, device=current.device)
    current = current.clone()
    current[..., :overlap] = current[..., :overlap] * phase.sin().square() + previous[..., -overlap:] * phase.cos().square()
    return current


@torch.inference_mode()
def _convert(bundle, source, reference, steps, cfg_rate, length_adjust, generator=None):
    source_22k = source.to(bundle.device)
    reference_22k = reference[..., :SEEDVC_SR * 25].to(bundle.device)
    if source_22k.shape[-1] < 1024:
        raise ValueError("source must contain at least 1024 samples at 22.05 kHz")
    if reference_22k.shape[-1] < 1024:
        raise ValueError("reference must contain at least 1024 samples at 22.05 kHz")
    source_16k = _resample(source_22k, SEEDVC_SR, FEATURE_SR)
    reference_16k = _resample(reference_22k, SEEDVC_SR, FEATURE_SR)
    source_semantics = _long_semantics(bundle, source_16k)
    reference_semantics = bundle.semantics(reference_16k)
    source_mel, prompt_mel = _mel(source_22k), _mel(reference_22k)

    fbank = torchaudio.compliance.kaldi.fbank(reference_16k, num_mel_bins=80, dither=0,
                                              sample_frequency=FEATURE_SR)
    fbank = fbank - fbank.mean(0, keepdim=True)
    camp_dtype = next(bundle.camp.parameters()).dtype
    style = bundle.camp(fbank.unsqueeze(0).to(camp_dtype))
    source_len = torch.tensor([int(source_mel.shape[-1] * length_adjust)], device=bundle.device)
    prompt_len = torch.tensor([prompt_mel.shape[-1]], device=bundle.device)
    cond = bundle.seed.length_regulator(source_semantics, source_len, 3)[0]
    prompt_cond = bundle.seed.length_regulator(reference_semantics, prompt_len, 3)[0]

    max_source = bundle.max_context_window - prompt_mel.shape[-1]
    if max_source <= bundle.overlap_frames:
        raise ValueError("SeedVC reference leaves no source context; use a shorter reference")
    chunks, previous, processed = [], None, 0
    while processed < cond.shape[1]:
        chunk_cond = cond[:, processed:processed + max_source]
        last = processed + max_source >= cond.shape[1]
        joined = torch.cat([prompt_cond, chunk_cond], 1)
        lengths = torch.tensor([joined.shape[1]], device=bundle.device)
        with torch.autocast(bundle.device.type, dtype=torch.float16, enabled=bundle.device.type == "cuda"):
            mel = bundle.seed.cfm.inference(joined, lengths, prompt_mel, style, steps,
                                            cfg_rate, generator=generator)
            mel = mel[..., prompt_mel.shape[-1]:]
        wave = bundle.vocoder(mel.float())[0, 0]
        if previous is not None:
            wave = _crossfade(previous, wave, bundle.overlap_samples)
        if last:
            chunks.append(wave)
            break
        chunks.append(wave[:-bundle.overlap_samples])
        previous = wave[-bundle.overlap_samples:]
        processed += mel.shape[-1] - bundle.overlap_frames
    return torch.cat(chunks).clamp(-1, 1).cpu()[None]


def convert_voice(source, source_sr, reference, reference_sr, *, steps=DEFAULT_STEPS,
                  cfg_rate=DEFAULT_CFG_RATE, length_adjust=1.0, output_sr=None,
                  seed=None, bundle=None, device=None, root=None):
    """Convert Comfy AUDIO-like source identity; return ``([1,C,T], sample_rate)``.

    Pass ``bundle`` to reuse a caller-owned bundle across several conversions;
    the caller then also owns unloading it. Without one, this call loads and
    unloads its own bundle, so no component is ever left resident by accident.
    ``seed`` ties the flow-matching noise to the workflow seed; ``None`` draws
    from the global RNG the way the reference implementation does.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 200:
        raise ValueError("steps must be an integer from 1 to 200")
    if not isinstance(cfg_rate, (int, float)) or isinstance(cfg_rate, bool) or not math.isfinite(cfg_rate) or cfg_rate < 0:
        raise ValueError("cfg_rate must be a finite non-negative number")
    if not isinstance(length_adjust, (int, float)) or isinstance(length_adjust, bool) or not math.isfinite(length_adjust) or length_adjust <= 0:
        raise ValueError("length_adjust must be a finite positive number")
    if not isinstance(source_sr, int) or isinstance(source_sr, bool) or source_sr <= 0:
        raise ValueError("source_sr must be a positive integer")
    if not isinstance(reference_sr, int) or isinstance(reference_sr, bool) or reference_sr <= 0:
        raise ValueError("reference_sr must be a positive integer")
    source, channels = _mono_audio(source, "source")
    reference, _ = _mono_audio(reference, "reference")
    source = _resample(source, int(source_sr), SEEDVC_SR)
    reference = _resample(reference, int(reference_sr), SEEDVC_SR)
    owned = bundle is None
    try:
        if owned:
            bundle = load_seed_vc(device=device, root=root)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=bundle.device)
            generator.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFF)
        result = _convert(bundle, source, reference, int(steps), float(cfg_rate),
                          float(length_adjust), generator=generator)
    finally:
        if owned:
            unload_seed_vc(bundle)
    output_sr = source_sr if output_sr is None else output_sr
    if not isinstance(output_sr, int) or isinstance(output_sr, bool) or output_sr <= 0:
        raise ValueError("output_sr must be a positive integer")
    result = _resample(result, SEEDVC_SR, output_sr).unsqueeze(0)
    if channels > 1:
        result = result.repeat(1, channels, 1)
    return result, output_sr


def apply_seed_vc_to_result(combined, sample_rate, identity_ref, identity_sr, **kwargs):
    return convert_voice(combined, sample_rate, identity_ref, identity_sr,
                         output_sr=sample_rate, **kwargs)[0]
