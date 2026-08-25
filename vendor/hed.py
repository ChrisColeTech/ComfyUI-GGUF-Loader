# Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# "This is an improved version and model of HED edge detection with Apache
# License, Version 2.0. [...] Different from official models and other
# implementations, this is an RGB-input model (rather than BGR)."
# - from the original ControlNetHED_Apache2 source header.
"""HED soft-edge detector: a small VGG-like multi-scale CNN
(ControlNetHED_Apache2) that outputs a single fused soft-edge map.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/hed/__init__.py, consolidated into this repo's
flat-file vendor convention (see depth_anything_v2.py). Only the core
detector is ported - not the "Fake Scribble" postprocessing (NMS + Gaussian
blur + threshold) that the source pack's node_wrappers/hed.py builds on top
of the same detector via a separate node, and not the source's padding to a
multiple of 64 in resize_image_with_pad: this network resizes every
multi-scale projection back to the input's exact (H, W) with cv2.resize
before fusing them, so pre-padding to a stride-friendly size only mattered
for other preprocessors sharing that same utility, not for correctness here.
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

import folder_paths

logger = logging.getLogger(__name__)

HED_MODELS_DIR = os.path.join(folder_paths.models_dir, "hed")

MODEL_REPO_IDS = {
    "ControlNetHED.pth": "lllyasviel/Annotators",
}


# ── architecture (ControlNetHED_Apache2) ─────────────────────────────────

class DoubleConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                    kernel_size=(3, 3), stride=(1, 1), padding=1))
        for _ in range(1, layer_number):
            self.convs.append(nn.Conv2d(in_channels=output_channel, out_channels=output_channel,
                                        kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = nn.Conv2d(in_channels=output_channel, out_channels=1,
                                    kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def forward(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(HED_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("HED: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(HED_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=HED_MODELS_DIR)


def _safe_step(x, step=2):
    """Quantize into `step` discrete levels (source's util.safe_step)."""
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class HEDDetector:
    """Loads a checkpoint (downloading if needed) and estimates a soft-edge
    map for one image."""

    def __init__(self, ckpt_name="ControlNetHED.pth"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = ControlNetHED_Apache2()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.float().eval()

    def to(self, device):
        self.model.to(device)
        return self

    def estimate(self, image_hwc_uint8, resolution=512, safe=False):
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
            image_hed = torch.from_numpy(resized).float().to(device)
            image_hed = image_hed.permute(2, 0, 1).unsqueeze(0)  # h w c -> 1 c h w
            edges = self.model(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = _safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        edge_rgb = np.repeat(edge[:, :, None], 3, axis=2)
        if (target_h, target_w) != (h, w):
            edge_rgb = cv2.resize(edge_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return edge_rgb
