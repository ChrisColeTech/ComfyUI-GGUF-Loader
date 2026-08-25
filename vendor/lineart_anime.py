# MIT License, Copyright (c) 2022 Caroline Chan (pytorch-CycleGAN-and-
# pix2pix UnetGenerator architecture this checkpoint was trained with).
#
# Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
# src/custom_controlnet_aux/lineart_anime/, consolidated into this repo's
# flat-file vendor convention (see depth_anything_v2.py, hed.py).
"""Anime-style line-art extractor: a pix2pix-style U-Net generator
(8 downsampling levels, InstanceNorm, no dropout) that turns a photo/render
into a clean anime-style line drawing.
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

import folder_paths

logger = logging.getLogger(__name__)

LINEART_ANIME_MODELS_DIR = os.path.join(folder_paths.models_dir, "lineart_anime")

MODEL_REPO_IDS = {
    "netG.pth": "lllyasviel/Annotators",
}


# ── architecture (pix2pix-style UnetGenerator) ───────────────────────────

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)  # add skip connection


class UnetGenerator(nn.Module):
    """Create a Unet-based generator, built from the innermost layer outward."""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(LINEART_ANIME_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("Lineart Anime: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(LINEART_ANIME_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=LINEART_ANIME_MODELS_DIR)


class LineartAnimeDetector:
    """Loads a checkpoint (downloading if needed) and estimates an anime-style
    line drawing for one image."""

    def __init__(self, ckpt_name="netG.pth"):
        model_path = _download_checkpoint(ckpt_name)
        # InstanceNorm2d(affine=False, track_running_stats=False) has no
        # learnable params/buffers, so it contributes no state_dict keys.
        self.model = UnetGenerator(3, 1, 8, 64, norm_layer=nn.InstanceNorm2d, use_dropout=False)
        ckpt = torch.load(model_path, map_location="cpu")
        for key in list(ckpt.keys()):
            if "module." in key:
                ckpt[key.replace("module.", "")] = ckpt[key]
                del ckpt[key]
        self.model.load_state_dict(ckpt)
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def estimate(self, image_hwc_uint8, resolution=512):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3]."""
        h, w = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        # Pad up to a multiple of 64 (edge-replicate), matching the source's
        # resize_image_with_pad utility that this preprocessor is fed through.
        pad_h = int(np.ceil(target_h / 64.0) * 64) - target_h
        pad_w = int(np.ceil(target_w / 64.0) * 64) - target_w
        padded = np.pad(resized, [[0, pad_h], [0, pad_w], [0, 0]], mode="edge")
        Hp, Wp = padded.shape[:2]

        # The 8-downsampling U-Net additionally needs its input to be a
        # multiple of 256 (2**8); the source stretch-resizes to this size
        # rather than padding further, so replicate that exactly.
        Hn = 256 * int(np.ceil(Hp / 256.0))
        Wn = 256 * int(np.ceil(Wp / 256.0))
        net_input = cv2.resize(padded, (Wn, Hn), interpolation=cv2.INTER_CUBIC)

        # Device must be derived from the model's own parameters, never
        # re-detected independently - see depth_anything_v2.py's note on
        # this exact bug.
        device = next(self.model.parameters()).device

        with torch.no_grad():
            image_feed = torch.from_numpy(net_input).float().to(device)
            image_feed = image_feed / 127.5 - 1.0
            image_feed = image_feed.permute(2, 0, 1).unsqueeze(0)  # h w c -> 1 c h w
            line = self.model(image_feed)[0, 0] * 127.5 + 127.5
            line = line.cpu().numpy().clip(0, 255).astype(np.uint8)

        line_rgb = np.repeat(line[:, :, None], 3, axis=2)
        # A1111 uses INTER_AREA for downscaling, so this preprocessor does too.
        line_rgb = cv2.resize(line_rgb, (Wp, Hp), interpolation=cv2.INTER_AREA)
        line_rgb = line_rgb[:target_h, :target_w]  # remove the multiple-of-64 pad
        line_rgb = 255 - line_rgb

        if (target_h, target_w) != (h, w):
            line_rgb = cv2.resize(line_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return line_rgb
