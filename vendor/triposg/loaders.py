# MIT (VAST-AI-Research/TripoSG). Component loading after the owner's POC
# ``pipelines/triposg_v1/{loaders,transformer_loader,vae_loader,image_encoder_loader}.py``.
"""Local-only loading of the TripoSG split checkpoint and RMBG-1.4.

Same files, same strict key checks and the same decoder-only VAE filter
(``encoder.*`` / ``quant.*`` dropped) as the POC. Two memory changes, neither
touching the math: the DiT and DINOv2 are built on the meta device and their
weights assigned straight from the file in the target dtype, so a 5.8 GB fp32
DiT never sits in RAM twice; the VAE is built normally because its Fourier
``frequencies`` buffer is non-persistent (computed in ``__init__``, absent from
the file). The values are what the POC's ``load fp32 -> .to(dtype)`` produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .models.conditioner import BitImageProcessor, DINOv2Config, DINOv2Model
from .models.transformer import TripoSGDiTConfig, TripoSGDiTModel
from .models.vae import TripoSGVAEConfig, TripoSGVAEModel
from .ops.scheduler import RectifiedFlowScheduler

WEIGHTS = "diffusion_pytorch_model.safetensors"

# Files the POC's TripoSGConfig.validate_local_files requires (minus the
# upstream source checkout, which this port no longer needs).
REQUIRED_FILES = (
    "model_index.json",
    "transformer/config.json",
    f"transformer/{WEIGHTS}",
    "vae/config.json",
    f"vae/{WEIGHTS}",
    "scheduler/scheduler_config.json",
    "image_encoder_dinov2/config.json",
    "image_encoder_dinov2/model.safetensors",
    "feature_extractor_dinov2/preprocessor_config.json",
)
RMBG_FILES = ("config.json", "model.safetensors")
_VAE_UNUSED_PREFIXES = ("encoder.", "quant.")


def missing_files(root, required=REQUIRED_FILES):
    root = Path(root)
    return [name for name in required if not (root / name).is_file()]


def is_triposg_dir(path) -> bool:
    """A TripoSG image-to-mesh root (the scribble variant has a text encoder)."""
    path = Path(path)
    return not missing_files(path) and not (path / "text_encoder").exists()


def is_rmbg_dir(path) -> bool:
    path = Path(path)
    if missing_files(path, RMBG_FILES):
        return False
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return "BriaRMBG" in (config.get("architectures") or [])


def _strict(model, state_dict, label):
    expected = set(model.state_dict().keys())
    actual = set(state_dict.keys())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"TripoSG {label}: strict load failed - {len(missing)} missing "
            f"{missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}")
    model.load_state_dict(state_dict, strict=True, assign=True)


def _assert_materialized(model, label):
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.device.type == "meta"]
    if left:
        raise RuntimeError(f"TripoSG {label}: tensors left on meta device: {left[:5]}")


def _cast(state_dict, dtype):
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state_dict.items()}


def load_transformer(root, dtype) -> TripoSGDiTModel:
    folder = Path(root) / "transformer"
    config = TripoSGDiTConfig.from_pretrained(folder, local_files_only=True)
    with torch.device("meta"):
        model = TripoSGDiTModel(config)
    _strict(model, _cast(load_file(str(folder / WEIGHTS)), dtype), "transformer")
    _assert_materialized(model, "transformer")
    return model.eval()


def load_vae(root, dtype) -> TripoSGVAEModel:
    folder = Path(root) / "vae"
    config = TripoSGVAEConfig.from_pretrained(folder, local_files_only=True)
    model = TripoSGVAEModel(config)
    state = {k: v for k, v in load_file(str(folder / WEIGHTS)).items()
             if not k.startswith(_VAE_UNUSED_PREFIXES)}
    _strict(model, state, "vae")
    return model.to(dtype=dtype).eval()


def load_image_encoder(root, dtype) -> DINOv2Model:
    folder = Path(root) / "image_encoder_dinov2"
    config = DINOv2Config.from_pretrained(folder, local_files_only=True)
    with torch.device("meta"):
        model = DINOv2Model(config)
    _strict(model, _cast(load_file(str(folder / "model.safetensors")), dtype), "image encoder")
    _assert_materialized(model, "image encoder")
    return model.eval()


def load_feature_extractor(root) -> BitImageProcessor:
    return BitImageProcessor.from_pretrained(
        str(root), subfolder="feature_extractor_dinov2", local_files_only=True)


def scheduler_config(root) -> dict:
    payload = json.loads(
        (Path(root) / "scheduler" / "scheduler_config.json").read_text(encoding="utf-8"))
    return {
        "num_train_timesteps": payload.get("num_train_timesteps", 1000),
        "shift": payload.get("shift", 1.0),
        "use_dynamic_shifting": payload.get("use_dynamic_shifting", False),
    }


def make_scheduler(config: dict) -> RectifiedFlowScheduler:
    """A fresh scheduler per run: its step index is per-instance state."""
    return RectifiedFlowScheduler(**config)


def load_rmbg(folder):
    """RMBG-1.4, float32 (the POC runs background removal in float32)."""
    from .briarmbg import BriaRMBG

    model = BriaRMBG.from_pretrained(str(folder), local_files_only=True)
    return model.float().eval()
