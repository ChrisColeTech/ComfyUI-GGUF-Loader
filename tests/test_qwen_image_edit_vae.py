"""qwen_image_edit_vae is a Wan VAE in Diffusers keys — remap before VAE()."""
import importlib.util
from pathlib import Path

import pytest
import torch

from conftest import ROOT

_spec = importlib.util.spec_from_file_location(
    "vae_load_under_test", ROOT / "nodes" / "vae_load.py")
vae_load = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vae_load)


def _fake_edit_sd():
    """Minimal tensors so is_qwen_image_edit_vae is true and remap copies names."""
    t5 = torch.zeros(96, 3, 3, 3, 3)
    t1 = torch.zeros(1)
    sd = {
        "encoder.conv_in.weight": t5,
        "encoder.conv_in.bias": t1,
        "encoder.conv_out.weight": t5,
        "encoder.conv_out.bias": t1,
        "encoder.norm_out.gamma": t1,
        "decoder.conv_in.weight": torch.zeros(384, 16, 3, 3, 3),
        "decoder.conv_in.bias": t1,
        "decoder.conv_out.weight": t5,
        "decoder.conv_out.bias": t1,
        "decoder.norm_out.gamma": t1,
        "quant_conv.weight": torch.zeros(32, 32, 1, 1, 1),
        "quant_conv.bias": t1,
        "post_quant_conv.weight": torch.zeros(16, 16, 1, 1, 1),
        "post_quant_conv.bias": t1,
    }

    def resnet(prefix, shortcut=False):
        sd[f"{prefix}.norm1.gamma"] = t1
        sd[f"{prefix}.conv1.weight"] = t5
        sd[f"{prefix}.conv1.bias"] = t1
        sd[f"{prefix}.norm2.gamma"] = t1
        sd[f"{prefix}.conv2.weight"] = t5
        sd[f"{prefix}.conv2.bias"] = t1
        if shortcut:
            sd[f"{prefix}.conv_shortcut.weight"] = t5
            sd[f"{prefix}.conv_shortcut.bias"] = t1

    def attn(prefix):
        sd[f"{prefix}.norm.gamma"] = t1
        sd[f"{prefix}.proj.weight"] = t5
        sd[f"{prefix}.proj.bias"] = t1
        sd[f"{prefix}.to_qkv.weight"] = t5
        sd[f"{prefix}.to_qkv.bias"] = t1

    def resample(prefix, time=False):
        sd[f"{prefix}.resample.1.weight"] = t5
        sd[f"{prefix}.resample.1.bias"] = t1
        if time:
            sd[f"{prefix}.time_conv.weight"] = t5
            sd[f"{prefix}.time_conv.bias"] = t1

    resnet("encoder.down_blocks.0")
    resnet("encoder.down_blocks.1")
    resample("encoder.down_blocks.2")
    resnet("encoder.down_blocks.3", shortcut=True)
    resnet("encoder.down_blocks.4")
    resample("encoder.down_blocks.5", time=True)
    resnet("encoder.down_blocks.6", shortcut=True)
    resnet("encoder.down_blocks.7")
    resample("encoder.down_blocks.8", time=True)
    resnet("encoder.down_blocks.9")
    resnet("encoder.down_blocks.10")
    resnet("encoder.mid_block.resnets.0")
    attn("encoder.mid_block.attentions.0")
    resnet("encoder.mid_block.resnets.1")
    resnet("decoder.mid_block.resnets.0")
    attn("decoder.mid_block.attentions.0")
    resnet("decoder.mid_block.resnets.1")
    for b, up in ((0, True), (1, True), (2, True), (3, False)):
        for j in range(3):
            resnet(f"decoder.up_blocks.{b}.resnets.{j}", shortcut=(b == 1 and j == 0))
        if up:
            resample(f"decoder.up_blocks.{b}.upsamplers.0", time=(b in (0, 1)))
    return sd


def test_wan_vae_is_left_alone():
    sd = {"decoder.middle.0.residual.0.gamma": torch.zeros(384, 1, 1, 1),
          "encoder.conv1.weight": torch.zeros(96, 3, 3, 3, 3)}
    assert vae_load.remap_qwen_image_edit_vae(sd) is sd


def test_edit_vae_remaps_to_wan_keys():
    sd = _fake_edit_sd()
    assert vae_load.is_qwen_image_edit_vae(sd)
    out = vae_load.remap_qwen_image_edit_vae(sd)
    assert "encoder.conv1.weight" in out
    assert "decoder.conv1.weight" in out
    assert "conv1.weight" in out and "conv2.weight" in out
    assert "encoder.downsamples.5.time_conv.weight" in out
    assert "decoder.upsamples.4.shortcut.weight" in out
    assert "encoder.conv_in.weight" not in out
    assert out["encoder.conv1.weight"].shape == (96, 3, 3, 3, 3)
    assert out["decoder.conv1.weight"].shape == (384, 16, 3, 3, 3)


def test_edit_vae_remap_covers_real_file():
    path = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\vae\qwen_image_edit_vae.safetensors")
    official = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI\models\vae\qwen_image_vae.safetensors")
    if not path.is_file() or not official.is_file():
        pytest.skip("qwen image VAE files not on this machine")
    from safetensors import safe_open

    def inv(p):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}

    edit = {k: torch.empty(shp) for k, shp in inv(path).items()}
    out = vae_load.remap_qwen_image_edit_vae(edit)
    want = inv(official)
    assert set(out) == set(want)
    for k in want:
        assert tuple(out[k].shape) == want[k], k
