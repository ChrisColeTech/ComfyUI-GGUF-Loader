# Copyright (c) 2021 Zhuo Su, Wenzhe Liu
# MIT-style license (see src/custom_controlnet_aux/pidi/LICENSE in the
# ported-from repo) - the original authors' LICENSE text adds a research-use
# note: "It is just for research purpose, and commercial use should be
# contacted with authors first."
#
# Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
# src/custom_controlnet_aux/pidi/ (model.py + __init__.py), consolidated
# here into one flat file per this repo's own convention.
"""PiDiNet soft-edge/boundary detector, inference only.

Pixel Difference Convolution (PDC) CNN: a 4-stage fully-convolutional
backbone (init_block + 4x{3-4 PDCBlocks}) where several convs are replaced by
learned Pixel Difference Convolutions (central/angular/radial difference
variants) instead of vanilla convolution, followed by per-stage dilated
"CDCM" context modules, "CSAM" spatial-attention modules, 1x1 MapReduce
edge-map heads, and a final 1x1 fusion classifier - all four intermediate
edge maps and the fused output are bilinearly upsampled back to input
resolution and passed through sigmoid.

Simplified: the source's `nets` dict defines 14 PDC layer-assignment presets
(baseline, c-v15, a-v15, r-v15, cvvv4, avvv4, rvvv4, cccv4, aaav4, rrrv4,
c16, a16, r16, carv4) plus a `PDCBlock_converted` block (for reparameterizing
CPDC/APDC into vanilla 3x3 convs and RPDC into a vanilla 5x5 conv). The
shipped checkpoint (table5_pidinet.pth) is only ever built via the source's
`pidinet()` factory, which hardcodes `config_model('carv4')`, `convert=False`,
`dil=24`, `sa=True`. So only the "carv4" PDC layer assignment is wired in
here (as a plain list, replacing the `nets` dict + `config_model` lookup),
and only the non-converted `PDCBlock` path is kept - `PDCBlock_converted`
and the other 13 presets are dead code for this checkpoint and were dropped.
The `PiDiNet.__init__` branches for `sa=False`/`dil=None` were dropped too
for the same reason (this checkpoint always uses `sa=True, dil=24`, i.e. the
CDCM+CSAM branch). None of this changes the state_dict keys the checkpoint
loads into.

Weights auto-download from HuggingFace (lllyasviel/Annotators,
table5_pidinet.pth) on first use, same pattern as this repo's other vendor
ports: no config file, just a plain local folder under models/pidinet/.
"""
import logging
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger(__name__)

PIDINET_MODELS_DIR = os.path.join(folder_paths.models_dir, "pidinet")

MODEL_REPO_IDS = {
    "table5_pidinet.pth": "lllyasviel/Annotators",
}

# The "carv4" PDC layer-assignment preset (source: model.py's `nets['carv4']`),
# the only variant the shipped checkpoint's `pidinet()` factory ever builds.
_CARV4_OPS = ["cd", "ad", "rd", "cv"] * 4


# ── Pixel Difference Convolutions ────────────────────────────────────────

def _createConvFunc(op_type):
    assert op_type in ["cv", "cd", "ad", "rd"], "unknown op type: %s" % str(op_type)
    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for cd_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for cd_conv should be 3x3"
            assert padding == dilation, "padding for cd_conv set wrong"
            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == "ad":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for ad_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for ad_conv should be 3x3"
            assert padding == dilation, "padding for ad_conv set wrong"
            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == "rd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], "dilation for rd_conv should be in 1 or 2"
            assert weights.size(2) == 3 and weights.size(3) == 3, "kernel size for rd_conv should be 3x3"
            padding = 2 * dilation
            shape = weights.shape
            buffer = torch.zeros(shape[0], shape[1], 5 * 5, device=weights.device, dtype=weights.dtype)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func


class _PDCConv2d(nn.Module):
    """A Conv2d whose forward pass runs one of the PDC ops instead of a plain conv."""

    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ── attention / context modules ──────────────────────────────────────────

class CSAM(nn.Module):
    """Compact Spatial Attention Module."""

    def __init__(self, channels):
        super().__init__()
        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y


class CDCM(nn.Module):
    """Compact Dilation Convolution based Module."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class MapReduce(nn.Module):
    """Reduce feature maps into a single edge map."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
        self.conv1 = _PDCConv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            x = self.shortcut(x)
        return y + x


# ── PiDiNet backbone ──────────────────────────────────────────────────────

class PiDiNet(nn.Module):
    """Fixed to the checkpoint's actual config: inplane=60, carv4 PDCs,
    dil=24, sa=True, convert=False (i.e. always the CDCM+CSAM branch and
    always plain PDCBlock, never PDCBlock_converted)."""

    def __init__(self, inplane=60, pdcs=None, dil=24, sa=True):
        super().__init__()
        if pdcs is None:
            pdcs = [_createConvFunc(op) for op in _CARV4_OPS]
        self.sa = sa
        self.dil = dil
        self.fuseplanes = []

        self.inplane = inplane
        self.init_block = _PDCConv2d(pdcs[0], 3, self.inplane, kernel_size=3, padding=1)

        self.block1_1 = PDCBlock(pdcs[1], self.inplane, self.inplane)
        self.block1_2 = PDCBlock(pdcs[2], self.inplane, self.inplane)
        self.block1_3 = PDCBlock(pdcs[3], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block2_1 = PDCBlock(pdcs[4], inplane, self.inplane, stride=2)
        self.block2_2 = PDCBlock(pdcs[5], self.inplane, self.inplane)
        self.block2_3 = PDCBlock(pdcs[6], self.inplane, self.inplane)
        self.block2_4 = PDCBlock(pdcs[7], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 2C

        inplane = self.inplane
        self.inplane = self.inplane * 2
        self.block3_1 = PDCBlock(pdcs[8], inplane, self.inplane, stride=2)
        self.block3_2 = PDCBlock(pdcs[9], self.inplane, self.inplane)
        self.block3_3 = PDCBlock(pdcs[10], self.inplane, self.inplane)
        self.block3_4 = PDCBlock(pdcs[11], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.block4_1 = PDCBlock(pdcs[12], self.inplane, self.inplane, stride=2)
        self.block4_2 = PDCBlock(pdcs[13], self.inplane, self.inplane)
        self.block4_3 = PDCBlock(pdcs[14], self.inplane, self.inplane)
        self.block4_4 = PDCBlock(pdcs[15], self.inplane, self.inplane)
        self.fuseplanes.append(self.inplane)  # 4C

        self.conv_reduces = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.dilations = nn.ModuleList()
        for i in range(4):
            self.dilations.append(CDCM(self.fuseplanes[i], self.dil))
            self.attentions.append(CSAM(self.dil))
            self.conv_reduces.append(MapReduce(self.dil))

        self.classifier = nn.Conv2d(4, 1, kernel_size=1)  # has bias
        nn.init.constant_(self.classifier.weight, 0.25)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.init_block(x)

        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        x3 = self.block3_1(x2)
        x3 = self.block3_2(x3)
        x3 = self.block3_3(x3)
        x3 = self.block3_4(x3)

        x4 = self.block4_1(x3)
        x4 = self.block4_2(x4)
        x4 = self.block4_3(x4)
        x4 = self.block4_4(x4)

        x_fuses = []
        for i, xi in enumerate([x1, x2, x3, x4]):
            x_fuses.append(self.attentions[i](self.dilations[i](xi)))

        outputs = []
        for i in range(4):
            e = self.conv_reduces[i](x_fuses[i])
            e = F.interpolate(e, (H, W), mode="bilinear", align_corners=False)
            outputs.append(e)

        output = self.classifier(torch.cat(outputs, dim=1))
        outputs.append(output)
        outputs = [torch.sigmoid(r) for r in outputs]
        return outputs


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(PIDINET_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("PiDiNet: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(PIDINET_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=PIDINET_MODELS_DIR)


def _pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def _safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


class PiDiNetDetector:
    """Loads a checkpoint (downloading if needed) and estimates a soft-edge
    map for one IMAGE tensor."""

    def __init__(self, ckpt_name="table5_pidinet.pth"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = PiDiNet()
        state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def estimate(self, image_hwc_uint8, resolution=512, safe=False):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3]."""
        h_raw, w_raw = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h_raw, w_raw))
        target_h, target_w = int(round(h_raw * k)), int(round(w_raw * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        pad_h, pad_w = _pad64(target_h), _pad64(target_w)
        padded = np.pad(resized, [[0, pad_h], [0, pad_w], [0, 0]], mode="edge")

        # The source pack feeds BGR (it does `detected_map[:, :, ::-1]` on an
        # RGB input before this point) - match that.
        bgr = padded[:, :, ::-1].copy()

        device = next(self.model.parameters()).device
        with torch.no_grad():
            tensor = torch.from_numpy(bgr).float().to(device) / 255.0
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # h w c -> 1 c h w
            edge = self.model(tensor)[-1]
            edge = edge.cpu().numpy()
            if safe:
                edge = _safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]
        detected_map = detected_map[:target_h, :target_w]  # remove_pad
        detected_map = np.ascontiguousarray(detected_map.copy())
        detected_map = np.repeat(detected_map[:, :, None], 3, axis=2)

        if (target_h, target_w) != (h_raw, w_raw):
            detected_map = cv2.resize(detected_map, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR)
        return detected_map
