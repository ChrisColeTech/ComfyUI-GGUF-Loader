# (c) CCTech || Apache-2.0
"""Load a VAE the way Krea2 / Qwen-Image expect.

`qwen_image_edit_vae.safetensors` is the same Wan 3-D VAE as
`qwen_image_vae.safetensors` (96-ch, 16 latent, 5-D convs) but stored in a
Diffusers-style layout (`encoder.conv_in`, `down_blocks`, `mid_block`). Comfy's
`VAE()` autodetection sees `decoder.conv_in.weight` with z=16 and builds a 2-D
AutoencoderKL (ch=128, 4-D convs) — then `load_state_dict` raises a size
mismatch. Remap to the native Wan key layout before constructing the VAE.
"""
from __future__ import annotations


def is_qwen_image_edit_vae(sd: dict) -> bool:
    w = sd.get("encoder.conv_in.weight")
    return (
        w is not None
        and getattr(w, "ndim", 0) == 5
        and "encoder.down_blocks.0.conv1.weight" in sd
        and "decoder.middle.0.residual.0.gamma" not in sd
    )


def remap_qwen_image_edit_vae(sd: dict) -> dict:
    """Diffusers-style Qwen-Image-Edit VAE keys → Comfy WanVAE keys.

    Leaves any other state dict untouched.
    """
    if not is_qwen_image_edit_vae(sd):
        return sd

    out = {}
    copied = set()

    def take(src, dst):
        if src in sd:
            out[dst] = sd[src]
            copied.add(src)

    def take_resnet(src, dst):
        take(f"{src}.norm1.gamma", f"{dst}.residual.0.gamma")
        take(f"{src}.conv1.weight", f"{dst}.residual.2.weight")
        take(f"{src}.conv1.bias", f"{dst}.residual.2.bias")
        take(f"{src}.norm2.gamma", f"{dst}.residual.3.gamma")
        take(f"{src}.conv2.weight", f"{dst}.residual.6.weight")
        take(f"{src}.conv2.bias", f"{dst}.residual.6.bias")
        take(f"{src}.conv_shortcut.weight", f"{dst}.shortcut.weight")
        take(f"{src}.conv_shortcut.bias", f"{dst}.shortcut.bias")

    def take_attn(src, dst):
        take(f"{src}.norm.gamma", f"{dst}.norm.gamma")
        take(f"{src}.proj.weight", f"{dst}.proj.weight")
        take(f"{src}.proj.bias", f"{dst}.proj.bias")
        take(f"{src}.to_qkv.weight", f"{dst}.to_qkv.weight")
        take(f"{src}.to_qkv.bias", f"{dst}.to_qkv.bias")

    def take_resample(src, dst):
        take(f"{src}.resample.1.weight", f"{dst}.resample.1.weight")
        take(f"{src}.resample.1.bias", f"{dst}.resample.1.bias")
        take(f"{src}.time_conv.weight", f"{dst}.time_conv.weight")
        take(f"{src}.time_conv.bias", f"{dst}.time_conv.bias")

    take("quant_conv.weight", "conv1.weight")
    take("quant_conv.bias", "conv1.bias")
    take("post_quant_conv.weight", "conv2.weight")
    take("post_quant_conv.bias", "conv2.bias")

    take("encoder.conv_in.weight", "encoder.conv1.weight")
    take("encoder.conv_in.bias", "encoder.conv1.bias")
    take("encoder.norm_out.gamma", "encoder.head.0.gamma")
    take("encoder.conv_out.weight", "encoder.head.2.weight")
    take("encoder.conv_out.bias", "encoder.head.2.bias")

    take("decoder.conv_in.weight", "decoder.conv1.weight")
    take("decoder.conv_in.bias", "decoder.conv1.bias")
    take("decoder.norm_out.gamma", "decoder.head.0.gamma")
    take("decoder.conv_out.weight", "decoder.head.2.weight")
    take("decoder.conv_out.bias", "decoder.head.2.bias")

    take_resnet("encoder.mid_block.resnets.0", "encoder.middle.0")
    take_attn("encoder.mid_block.attentions.0", "encoder.middle.1")
    take_resnet("encoder.mid_block.resnets.1", "encoder.middle.2")
    take_resnet("decoder.mid_block.resnets.0", "decoder.middle.0")
    take_attn("decoder.mid_block.attentions.0", "decoder.middle.1")
    take_resnet("decoder.mid_block.resnets.1", "decoder.middle.2")

    for i in range(11):
        src = f"encoder.down_blocks.{i}"
        dst = f"encoder.downsamples.{i}"
        if f"{src}.resample.1.weight" in sd:
            take_resample(src, dst)
        else:
            take_resnet(src, dst)

    # Wan decoder.upsamples is flat: 0-2 res, 3 up; 4-6 res, 7 up; 8-10 res, 11 up; 12-14 res.
    stages = ((0, 0, 3), (1, 4, 7), (2, 8, 11), (3, 12, None))
    for block, res_start, up_idx in stages:
        for j in range(3):
            take_resnet(f"decoder.up_blocks.{block}.resnets.{j}",
                        f"decoder.upsamples.{res_start + j}")
        if up_idx is not None:
            take_resample(f"decoder.up_blocks.{block}.upsamplers.0",
                          f"decoder.upsamples.{up_idx}")

    leftover = [k for k in sd if k not in copied]
    if leftover:
        raise ValueError(
            "qwen_image_edit VAE remap left unmapped keys: "
            + ", ".join(leftover[:8])
        )
    return out


def load_vae(vae_name: str):
    """folder_paths VAE file → comfy.sd.VAE, with Qwen-Image-Edit remap."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    path = folder_paths.get_full_path_or_raise("vae", vae_name)
    sd = comfy.utils.load_torch_file(path)
    sd = remap_qwen_image_edit_vae(sd)
    return comfy.sd.VAE(sd=sd)
