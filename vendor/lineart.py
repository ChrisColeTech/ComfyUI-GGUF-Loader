# MIT License, Copyright (c) 2022 Caroline Chan (informative-drawings)
# (mit-license.org)
"""Realistic Lineart detector: a ResNet encoder-decoder GAN generator
(9 residual blocks) that turns a photo into a clean line drawing.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/lineart/__init__.py, consolidated into this
repo's flat-file vendor convention (see depth_anything_v2.py, hed.py).
Two checkpoints share this one architecture - "fine" (sk_model.pth) and
"coarse" (sk_model2.pth) - selected via LineartDetector(coarse=...).
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

import folder_paths

logger = logging.getLogger(__name__)

LINEART_MODELS_DIR = os.path.join(folder_paths.models_dir, "lineart")

MODEL_REPO_IDS = {
    "sk_model.pth": "lllyasviel/Annotators",
    "sk_model2.pth": "lllyasviel/Annotators",
}

norm_layer = nn.InstanceNorm2d


# ── architecture (informative-drawings Generator) ────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            norm_layer(in_features),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super().__init__()

        # Initial convolution block
        model0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            norm_layer(64),
            nn.ReLU(inplace=True),
        ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        # Residual blocks
        model2 = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                norm_layer(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]
        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)
        return out


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(LINEART_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("Lineart: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(LINEART_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=LINEART_MODELS_DIR)


class LineartDetector:
    """Loads the fine and/or coarse checkpoint (downloading if needed) and
    estimates a realistic line drawing for one image."""

    def __init__(self, coarse=False, fine_ckpt="sk_model.pth", coarse_ckpt="sk_model2.pth"):
        self.coarse = coarse
        ckpt_name = coarse_ckpt if coarse else fine_ckpt
        model_path = _download_checkpoint(ckpt_name)
        self.model = Generator(3, 1, 3)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
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

        # Device must be derived from the model's own parameters, never
        # re-detected independently - see depth_anything_v2.py's note on
        # this exact bug.
        device = next(self.model.parameters()).device

        with torch.no_grad():
            image = torch.from_numpy(resized).float().to(device)
            image = image / 255.0
            image = image.permute(2, 0, 1).unsqueeze(0)  # h w c -> 1 c h w
            line = self.model(image)[0][0]
            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        # Source inverts the raw sigmoid output (255 - line) so lines come
        # out dark-on-white rather than the network's native white-on-black.
        line = 255 - line
        line_rgb = np.repeat(line[:, :, None], 3, axis=2)

        if (target_h, target_w) != (h, w):
            line_rgb = cv2.resize(line_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return line_rgb
