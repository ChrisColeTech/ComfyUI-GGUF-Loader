# MIT License, Copyright (c) 2021 Miaomiao Li (MangaLineExtraction_PyTorch)
# https://github.com/ljsabc/MangaLineExtraction_PyTorch (mit-license.org)
"""Manga line-art extraction: a "res_skip" residual U-Net (grayscale in,
grayscale line map out) that pulls clean line art out of manga/anime-style
images. Intended as a preprocessor for the lineart_anime ControlNet.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/manga_line/ (model_torch.py + __init__.py),
consolidated into this repo's flat-file vendor convention (see
depth_anything_v2.py, hed.py, lineart.py).
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

import folder_paths

logger = logging.getLogger(__name__)

MANGA_LINE_MODELS_DIR = os.path.join(folder_paths.models_dir, "manga_line")

MODEL_REPO_IDS = {
    "erika.pth": "lllyasviel/Annotators",
}


# ── architecture (res_skip) ───────────────────────────────────────────────

class _bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample,
                     padding=(fw // 2, fh // 2), padding_mode="zeros"),
        )

    def forward(self, x):
        return self.model(x)


class _u_bn_relu_conv(nn.Module):
    def __init__(self, in_filters, nb_filters, fw, fh, subsample=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_filters, eps=1e-3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_filters, nb_filters, (fw, fh), stride=subsample, padding=(fw // 2, fh // 2)),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

    def forward(self, x):
        return self.model(x)


class _shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample=1):
        super().__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters or subsample != 1:
            self.process = True
            self.model = nn.Sequential(nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample))

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        return x + y


class _u_shortcut(nn.Module):
    def __init__(self, in_filters, nb_filters, subsample):
        super().__init__()
        self.process = False
        self.model = None
        if in_filters != nb_filters:
            self.process = True
            self.model = nn.Sequential(
                nn.Conv2d(in_filters, nb_filters, (1, 1), stride=subsample, padding_mode="zeros"),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )

    def forward(self, x, y):
        if self.process:
            return self.model(x) + y
        return x + y


class basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super().__init__()
        self.conv1 = _bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.residual(x1)
        return self.shortcut(x, x2)


class _u_basic_block(nn.Module):
    def __init__(self, in_filters, nb_filters, init_subsample=1):
        super().__init__()
        self.conv1 = _u_bn_relu_conv(in_filters, nb_filters, 3, 3, subsample=init_subsample)
        self.residual = _bn_relu_conv(nb_filters, nb_filters, 3, 3)
        self.shortcut = _u_shortcut(in_filters, nb_filters, subsample=init_subsample)

    def forward(self, x):
        y = self.residual(self.conv1(x))
        return self.shortcut(x, y)


class _residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions, is_first_layer=False):
        super().__init__()
        layers = []
        for i in range(repetitions):
            init_subsample = 1
            if i == repetitions - 1 and not is_first_layer:
                init_subsample = 2
            if i == 0:
                l = basic_block(in_filters=in_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters, init_subsample=init_subsample)
            layers.append(l)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _upsampling_residual_block(nn.Module):
    def __init__(self, in_filters, nb_filters, repetitions):
        super().__init__()
        layers = []
        for i in range(repetitions):
            if i == 0:
                l = _u_basic_block(in_filters=in_filters, nb_filters=nb_filters)
            else:
                l = basic_block(in_filters=nb_filters, nb_filters=nb_filters)
            layers.append(l)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class res_skip(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = _residual_block(in_filters=1, nb_filters=24, repetitions=2, is_first_layer=True)
        self.block1 = _residual_block(in_filters=24, nb_filters=48, repetitions=3)
        self.block2 = _residual_block(in_filters=48, nb_filters=96, repetitions=5)
        self.block3 = _residual_block(in_filters=96, nb_filters=192, repetitions=7)
        self.block4 = _residual_block(in_filters=192, nb_filters=384, repetitions=12)

        self.block5 = _upsampling_residual_block(in_filters=384, nb_filters=192, repetitions=7)
        self.res1 = _shortcut(in_filters=192, nb_filters=192)

        self.block6 = _upsampling_residual_block(in_filters=192, nb_filters=96, repetitions=5)
        self.res2 = _shortcut(in_filters=96, nb_filters=96)

        self.block7 = _upsampling_residual_block(in_filters=96, nb_filters=48, repetitions=3)
        self.res3 = _shortcut(in_filters=48, nb_filters=48)

        self.block8 = _upsampling_residual_block(in_filters=48, nb_filters=24, repetitions=2)
        self.res4 = _shortcut(in_filters=24, nb_filters=24)

        self.block9 = _residual_block(in_filters=24, nb_filters=16, repetitions=2, is_first_layer=True)
        self.conv15 = _bn_relu_conv(in_filters=16, nb_filters=1, fh=1, fw=1, subsample=1)

    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x5 = self.block5(x4)
        res1 = self.res1(x3, x5)

        x6 = self.block6(res1)
        res2 = self.res2(x2, x6)

        x7 = self.block7(res2)
        res3 = self.res3(x1, x7)

        x8 = self.block8(res3)
        res4 = self.res4(x0, x8)

        x9 = self.block9(res4)
        y = self.conv15(x9)
        return y


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(MANGA_LINE_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("Manga Line: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(MANGA_LINE_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=MANGA_LINE_MODELS_DIR)


def _pad_amount(x, multiple=64):
    """Amount of padding needed to round x up to a multiple of `multiple`
    (source's util.pad64)."""
    return int(np.ceil(float(x) / multiple) * multiple) - x


class MangaLineDetector:
    """Loads a checkpoint (downloading if needed) and estimates clean manga
    line art for one image."""

    def __init__(self, ckpt_name="erika.pth"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = res_skip()
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

        # Source rounds the requested detect_resolution up to a multiple of
        # 256 before resizing (LineartMangaDetector.__call__).
        detect_resolution = 256 * int(np.ceil(float(resolution) / 256.0))
        k = float(detect_resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        # This CNN downsamples 4x via strided convs then upsamples back
        # symmetrically, so the input must be a multiple of 16; the source
        # pads to a multiple of 64 with edge-replication (resize_image_with_pad,
        # mode='edge') and crops the padding back off afterward.
        pad_h, pad_w = _pad_amount(target_h), _pad_amount(target_w)
        padded = np.pad(resized, [[0, pad_h], [0, pad_w], [0, 0]], mode="edge")
        gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)

        # Device must be derived from the model's own parameters, never
        # re-detected independently - see depth_anything_v2.py's note on
        # this exact bug.
        device = next(self.model.parameters()).device

        with torch.no_grad():
            image_feed = torch.from_numpy(gray).float().to(device)
            image_feed = image_feed.unsqueeze(0).unsqueeze(0)  # h w -> 1 1 h w
            line = self.model(image_feed)
            line = line.cpu().numpy()[0, 0]
            line = line.clip(0, 255).astype(np.uint8)

        line = line[:target_h, :target_w]
        # Source inverts the raw output (255 - line) so lines come out
        # white-on-black, matching lineart_anime ControlNet's expected input.
        line = 255 - line
        line_rgb = np.repeat(line[:, :, None], 3, axis=2)

        if (target_h, target_w) != (h, w):
            line_rgb = cv2.resize(line_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return line_rgb
