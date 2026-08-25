# Copyright 2021-present NAVER Corp.
# Apache License, Version 2.0 (apache.org/licenses/LICENSE-2.0)
"""M-LSD: a MobileNetV2-backbone line-segment detector that outputs a
line-map image (detected straight segments drawn on a black canvas).

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/mlsd/ (models/mbv2_mlsd_large.py + utils.py),
consolidated into this repo's flat-file vendor convention (see
depth_anything_v2.py, hed.py). Only MobileV2_MLSD_Large (the "large"
variant used by the source pack's node) and pred_lines are ported - not
mbv2_mlsd_tiny.py (an alternate, smaller backbone never wired up by the
source's node_wrappers/mlsd.py) and not pred_squares (the source's
polygon/rectangle-detection postprocessing built on the same raw model
output, which is a separate, unused ComfyUI node in the source pack).
MobileNetV2's `pretrained=True` path (_load_pretrained_model, downloading
torchvision's ImageNet MobileNetV2 weights to warm-start training) was
also dropped: MobileV2_MLSD_Large always constructs it with
pretrained=False, so that path is dead code for loading this checkpoint
and doesn't affect the module hierarchy or state_dict keys.

Also simplified: the source's resize_image_with_pad pads the resized
input up to a multiple of 64 before running pred_lines, then crops the
output back down. pred_lines always internally resizes whatever image it
is given down to a fixed 512x512 for the network regardless, and scales
the predicted line coordinates back up using the *passed-in* image's own
(padded) height/width - so the padding only nudges that scale factor by
at most 63px and otherwise adds a pad+crop round trip. This port instead
follows the plain resize-to-resolution / resize-back-to-original pattern
already used by depth_anything_v2.py and hed.py, which is behaviorally
equivalent short of that sub-pixel scale-factor difference.
"""
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger(__name__)

MLSD_MODELS_DIR = os.path.join(folder_paths.models_dir, "mlsd")

MODEL_REPO_IDS = {
    "mlsd_large_512_fp32.pth": "lllyasviel/Annotators",
}


# ── MobileNetV2 backbone ─────────────────────────────────────────────────

def _make_divisible(v, divisor, min_value=None):
    """Ensures all layers have a channel number divisible by 8 (from the
    original TF slim mobilenet repo)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        # TFLite uses slightly different padding than PyTorch.
        padding = 0 if stride == 2 else (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features = nn.Sequential(*features)
        self.fpn_selected = [1, 3, 6, 10, 13]

    def forward(self, x):
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)
        c1, c2, c3, c4, c5 = fpn_features
        return c1, c2, c3, c4, c5


# ── M-LSD head ────────────────────────────────────────────────────────────

class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True),
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x


class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MobileV2_MLSD_Large(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV2()

        self.block15 = BlockTypeA(in_c1=64, in_c2=96, out_c1=64, out_c2=64, upscale=False)
        self.block16 = BlockTypeB(128, 64)

        self.block17 = BlockTypeA(in_c1=32, in_c2=64, out_c1=64, out_c2=64)
        self.block18 = BlockTypeB(128, 64)

        self.block19 = BlockTypeA(in_c1=24, in_c2=64, out_c1=64, out_c2=64)
        self.block20 = BlockTypeB(128, 64)

        self.block21 = BlockTypeA(in_c1=16, in_c2=64, out_c1=64, out_c2=64)
        self.block22 = BlockTypeB(128, 64)

        self.block23 = BlockTypeC(64, 16)

    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)

        x = self.block15(c4, c5)
        x = self.block16(x)

        x = self.block17(c3, x)
        x = self.block18(x)

        x = self.block19(c2, x)
        x = self.block20(x)

        x = self.block21(c1, x)
        x = self.block22(x)
        x = self.block23(x)
        x = x[:, 7:, :, :]

        return x


# ── postprocessing (utils.py) ────────────────────────────────────────────

def _deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
    """tpMap: center = tpMap[1,0,:,:], displacement = tpMap[1,1:5,:,:]."""
    b, c, h, w = tpMap.shape
    assert b == 1, "only support bsize==1"
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(-1)

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def _pred_lines(image, model, input_shape=(512, 512), score_thr=0.10, dist_thr=20.0):
    h, w, _ = image.shape

    # Device must be derived from the model's own parameters, never
    # re-detected independently - see depth_anything_v2.py's note on
    # this exact bug.
    device = next(iter(model.parameters())).device
    h_ratio, w_ratio = h / input_shape[0], w / input_shape[1]

    resized_image = np.concatenate(
        [cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA),
         np.ones([input_shape[0], input_shape[1], 1])], axis=-1)

    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype("float32")
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).float().to(device)
    with torch.no_grad():
        outputs = model(batch_image)
    pts, pts_score, vmap = _deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 -> 512
    if lines.size:
        lines[:, 0] = lines[:, 0] * w_ratio
        lines[:, 1] = lines[:, 1] * h_ratio
        lines[:, 2] = lines[:, 2] * w_ratio
        lines[:, 3] = lines[:, 3] * h_ratio

    return lines


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(MLSD_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("MLSD: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(MLSD_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=MLSD_MODELS_DIR)


class MLSDDetector:
    """Loads a checkpoint (downloading if needed) and estimates a
    line-segment map for one image."""

    def __init__(self, ckpt_name="mlsd_large_512_fp32.pth"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = MobileV2_MLSD_Large()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def estimate(self, image_hwc_uint8, resolution=512, score_threshold=0.1, dist_threshold=0.1):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3]
        with detected line segments drawn white-on-black."""
        h, w = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        try:
            lines = _pred_lines(resized, self.model, (512, 512), score_threshold, dist_threshold)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(round(v)) for v in line]
                cv2.line(canvas, (x_start, y_start), (x_end, y_end), (255, 255, 255), 1)
        except Exception:
            logger.exception("MLSD: line prediction failed")

        if (target_h, target_w) != (h, w):
            canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        return canvas
