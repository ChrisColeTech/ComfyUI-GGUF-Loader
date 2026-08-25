# OpenPose: Multiperson Keypoint Detection - Carnegie Mellon University.
# "OPENPOSE: MULTIPERSON KEYPOINT DETECTION SOFTWARE LICENSE AGREEMENT -
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY [...]
# Licensor retains exclusive ownership of any copy of the Software [...]
# grants to Licensee a personal, non-exclusive, non-transferable license to
# use the Software for noncommercial research purposes [...] You may not
# sell, rent, lease, sublicense, lend, time-share or transfer, in whole or
# in part, or provide third parties access to prior or present versions (or
# any parts thereof) of the Software." - full text in the source pack's
# src/custom_controlnet_aux/open_pose/LICENSE. The body/hand/face weights
# used here (lllyasviel/Annotators re-uploads) trace back to this CMU
# research-only license; do not treat this file or its checkpoints as
# available for commercial use.
#
# Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
# src/custom_controlnet_aux/open_pose/ (body.py, hand.py, face.py, model.py,
# util.py, __init__.py), which is itself a chain of ports: CMU's original
# Caffe OpenPose -> Hzzone/pytorch-openpose -> lllyasviel/ControlNet -> this
# pack. Consolidated into this repo's flat-file vendor convention (see
# depth_anything_v2.py / hed.py).
"""OpenPose body/hand/face keypoint estimation, rendered as an OpenPose-style
stick-figure skeleton image.

Three classic (pre-DWPose) multi-stage Convolutional Pose Machine networks,
each with its own checkpoint:
  - body: VGG-style backbone + 6 CPM refinement stages producing 18-point
    body keypoint heatmaps and 19-pair part-affinity fields (PAFs), grouped
    into per-person skeletons via greedy PAF-guided bipartite matching.
  - hand: the same CPM stage structure specialised to 21 hand keypoints, run
    on crops proposed from each detected body's wrist/elbow/shoulder.
  - face: the same CPM stage structure specialised to 70 facial landmarks,
    run on a crop proposed from each detected body's eyes/ears/head.

Simplified/dropped relative to the source:
  - The source's `resize_image_with_pad` (pad to a multiple of 64, detect,
    then crop the padding back off) is replaced with this repo's own
    min-side `cv2.resize` convention (see hed.py/depth_anything_v2.py):
    the body/hand/face networks are already fully convolutional and pad
    themselves to their own stride internally (see `_pad_right_down_corner`
    below), so pre-padding to 64 only mattered for other preprocessors
    sharing that same shared utility, not for correctness here.
  - `scipy.ndimage.gaussian_filter` (body/hand heatmap smoothing) is
    replaced with `cv2.GaussianBlur(x, (0, 0), sigma)` - a behavior-equivalent
    Gaussian smoothing using a dependency this repo already has, avoiding a
    new scipy requirement.
  - `skimage.measure.label` (hand.py's connected-components step, used to
    keep only the heatmap blob with the largest score-sum before taking its
    peak) is replaced with `cv2.connectedComponents(..., connectivity=8)`,
    which is the same 8-connectivity the source requested via
    `connectivity=binary.ndim` on a 2D array - again avoiding a new
    dependency (scikit-image) this repo doesn't otherwise need.
  - `matplotlib.colors.hsv_to_rgb` (hand-skeleton edge rainbow coloring) is
    replaced with the stdlib `colorsys.hsv_to_rgb`, which computes the exact
    same conversion for a single color and needs no extra dependency.
  - The source's `__init__.py` `hand_and_face` deprecated kwarg and the
    `output_type="pil"` / `image_and_json` toggle machinery are dropped -
    this port always returns a numpy array plus the keypoint dict (see
    `OpenPoseDetector.estimate` below), never a PIL Image or an image-only
    mode with a hidden dict.
None of the above changes state_dict key structure or keypoint semantics -
only unreached control flow, padding strategy, and non-`torch` dependencies
were touched.

Weights auto-download from HuggingFace (lllyasviel/Annotators) on first use,
into models/openpose/, same pattern as this repo's other vendor detectors.
"""
import colorsys
import logging
import math
import os
from collections import OrderedDict
from typing import List, NamedTuple, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths

logger = logging.getLogger(__name__)

OPENPOSE_MODELS_DIR = os.path.join(folder_paths.models_dir, "openpose")

MODEL_REPO_IDS = {
    "body_pose_model.pth": "lllyasviel/Annotators",
    "hand_pose_model.pth": "lllyasviel/Annotators",
    "facenet.pth": "lllyasviel/Annotators",
}


# ── keypoint containers ──────────────────────────────────────────────────

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


class BodyResult(NamedTuple):
    keypoints: List[Union[Keypoint, None]]
    total_score: float
    total_parts: int


class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Union[List[Keypoint], None]
    right_hand: Union[List[Keypoint], None]
    face: Union[List[Keypoint], None]


# ── shared CPM-network building blocks (model.py) ────────────────────────

def _make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
    return nn.Sequential(OrderedDict(layers))


def _transfer_state_dict(model, raw_state_dict):
    """Re-key a flat Caffe-converted checkpoint (e.g. "conv1_1.weight") onto
    this module's own nested submodule names (e.g. "model0.conv1_1.weight").
    Matches util.py's `transfer` exactly - required for body/hand checkpoint
    compatibility."""
    out = {}
    for name in model.state_dict().keys():
        out[name] = raw_state_dict[".".join(name.split(".")[1:])]
    return out


# ── body sub-network (model.py: bodypose_model) ──────────────────────────

class BodyPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = ["conv5_5_CPM_L1", "conv5_5_CPM_L2", "Mconv7_stage2_L1",
                          "Mconv7_stage2_L2", "Mconv7_stage3_L1", "Mconv7_stage3_L2",
                          "Mconv7_stage4_L1", "Mconv7_stage4_L2", "Mconv7_stage5_L1",
                          "Mconv7_stage5_L2", "Mconv7_stage6_L1", "Mconv7_stage6_L1"]
        blocks = {}
        block0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]),
            ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]),
            ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]),
            ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3_CPM", [512, 256, 3, 1, 1]),
            ("conv4_4_CPM", [256, 128, 3, 1, 1]),
        ])

        block1_1 = OrderedDict([
            ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
            ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
        ])
        block1_2 = OrderedDict([
            ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
            ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
            ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
        ])
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2

        self.model0 = _make_layers(block0, no_relu_layers)

        for i in range(2, 7):
            blocks["block%d_1" % i] = OrderedDict([
                ("Mconv1_stage%d_L1" % i, [185, 128, 7, 1, 3]),
                ("Mconv2_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv4_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L1" % i, [128, 128, 7, 1, 3]),
                ("Mconv6_stage%d_L1" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L1" % i, [128, 38, 1, 1, 0]),
            ])
            blocks["block%d_2" % i] = OrderedDict([
                ("Mconv1_stage%d_L2" % i, [185, 128, 7, 1, 3]),
                ("Mconv2_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv4_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d_L2" % i, [128, 128, 7, 1, 3]),
                ("Mconv6_stage%d_L2" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d_L2" % i, [128, 19, 1, 1, 0]),
            ])

        for k in blocks.keys():
            blocks[k] = _make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]

        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1, out6_2


# ── hand sub-network (model.py: handpose_model) ──────────────────────────

class HandPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        no_relu_layers = ["conv6_2_CPM", "Mconv7_stage2", "Mconv7_stage3",
                          "Mconv7_stage4", "Mconv7_stage5", "Mconv7_stage6"]
        block1_0 = OrderedDict([
            ("conv1_1", [3, 64, 3, 1, 1]),
            ("conv1_2", [64, 64, 3, 1, 1]),
            ("pool1_stage1", [2, 2, 0]),
            ("conv2_1", [64, 128, 3, 1, 1]),
            ("conv2_2", [128, 128, 3, 1, 1]),
            ("pool2_stage1", [2, 2, 0]),
            ("conv3_1", [128, 256, 3, 1, 1]),
            ("conv3_2", [256, 256, 3, 1, 1]),
            ("conv3_3", [256, 256, 3, 1, 1]),
            ("conv3_4", [256, 256, 3, 1, 1]),
            ("pool3_stage1", [2, 2, 0]),
            ("conv4_1", [256, 512, 3, 1, 1]),
            ("conv4_2", [512, 512, 3, 1, 1]),
            ("conv4_3", [512, 512, 3, 1, 1]),
            ("conv4_4", [512, 512, 3, 1, 1]),
            ("conv5_1", [512, 512, 3, 1, 1]),
            ("conv5_2", [512, 512, 3, 1, 1]),
            ("conv5_3_CPM", [512, 128, 3, 1, 1]),
        ])
        block1_1 = OrderedDict([
            ("conv6_1_CPM", [128, 512, 1, 1, 0]),
            ("conv6_2_CPM", [512, 22, 1, 1, 0]),
        ])

        blocks = {"block1_0": block1_0, "block1_1": block1_1}

        for i in range(2, 7):
            blocks["block%d" % i] = OrderedDict([
                ("Mconv1_stage%d" % i, [150, 128, 7, 1, 3]),
                ("Mconv2_stage%d" % i, [128, 128, 7, 1, 3]),
                ("Mconv3_stage%d" % i, [128, 128, 7, 1, 3]),
                ("Mconv4_stage%d" % i, [128, 128, 7, 1, 3]),
                ("Mconv5_stage%d" % i, [128, 128, 7, 1, 3]),
                ("Mconv6_stage%d" % i, [128, 128, 1, 1, 0]),
                ("Mconv7_stage%d" % i, [128, 22, 1, 1, 0]),
            ])

        for k in blocks.keys():
            blocks[k] = _make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks["block1_0"]
        self.model1_1 = blocks["block1_1"]
        self.model2 = blocks["block2"]
        self.model3 = blocks["block3"]
        self.model4 = blocks["block4"]
        self.model5 = blocks["block5"]
        self.model6 = blocks["block6"]

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


# ── face sub-network (face.py: FaceNet) ───────────────────────────────────

class FaceNet(nn.Module):
    """Model the cascading heatmaps."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.max_pooling_2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3_CPM = nn.Conv2d(512, 128, 3, 1, 1)

        # stage1
        self.conv6_1_CPM = nn.Conv2d(128, 512, 1, 1, 0)
        self.conv6_2_CPM = nn.Conv2d(512, 71, 1, 1, 0)

        # stages 2-6
        for stage in range(2, 7):
            in_ch = 199
            setattr(self, f"Mconv1_stage{stage}", nn.Conv2d(in_ch, 128, 7, 1, 3))
            setattr(self, f"Mconv2_stage{stage}", nn.Conv2d(128, 128, 7, 1, 3))
            setattr(self, f"Mconv3_stage{stage}", nn.Conv2d(128, 128, 7, 1, 3))
            setattr(self, f"Mconv4_stage{stage}", nn.Conv2d(128, 128, 7, 1, 3))
            setattr(self, f"Mconv5_stage{stage}", nn.Conv2d(128, 128, 7, 1, 3))
            setattr(self, f"Mconv6_stage{stage}", nn.Conv2d(128, 128, 1, 1, 0))
            setattr(self, f"Mconv7_stage{stage}", nn.Conv2d(128, 71, 1, 1, 0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.bias, 0)

    def _stage(self, stage, h, feature_map):
        h = torch.cat([h, feature_map], dim=1)
        h = self.relu(getattr(self, f"Mconv1_stage{stage}")(h))
        h = self.relu(getattr(self, f"Mconv2_stage{stage}")(h))
        h = self.relu(getattr(self, f"Mconv3_stage{stage}")(h))
        h = self.relu(getattr(self, f"Mconv4_stage{stage}")(h))
        h = self.relu(getattr(self, f"Mconv5_stage{stage}")(h))
        h = self.relu(getattr(self, f"Mconv6_stage{stage}")(h))
        return getattr(self, f"Mconv7_stage{stage}")(h)

    def forward(self, x):
        """Return a list of heatmaps, one per CPM stage (stage6's is used)."""
        heatmaps = []

        h = self.relu(self.conv1_1(x))
        h = self.relu(self.conv1_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv2_1(h))
        h = self.relu(self.conv2_2(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv3_1(h))
        h = self.relu(self.conv3_2(h))
        h = self.relu(self.conv3_3(h))
        h = self.relu(self.conv3_4(h))
        h = self.max_pooling_2d(h)
        h = self.relu(self.conv4_1(h))
        h = self.relu(self.conv4_2(h))
        h = self.relu(self.conv4_3(h))
        h = self.relu(self.conv4_4(h))
        h = self.relu(self.conv5_1(h))
        h = self.relu(self.conv5_2(h))
        h = self.relu(self.conv5_3_CPM(h))
        feature_map = h

        h = self.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)

        for stage in range(2, 7):
            h = self._stage(stage, h, feature_map)
            heatmaps.append(h)

        return heatmaps


# ── postprocessing helpers (util.py, minus matplotlib/scipy/skimage) ─────

def _smart_resize(x, size):
    Ht, Wt = size
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co in (1, 3):
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)),
                          interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    return np.stack([_smart_resize(x[:, :, i], size) for i in range(Co)], axis=2)


def _smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co in (1, 3):
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)),
                          interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    return np.stack([_smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def _pad_right_down_corner(img, stride, pad_value):
    h, w = img.shape[0], img.shape[1]
    pad = [0, 0,
          0 if (h % stride == 0) else stride - (h % stride),
          0 if (w % stride == 0) else stride - (w % stride)]

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + pad_value, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + pad_value, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)
    return img_padded, pad


def _npmax(array):
    """Index of the global max of a 2D array, as (row, col)."""
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def _gaussian_filter(x, sigma):
    """Behavior-equivalent stand-in for scipy.ndimage.gaussian_filter using
    cv2 (already a dependency of this repo) instead of adding scipy."""
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigmaX=sigma,
                            borderType=cv2.BORDER_REFLECT101)


def _label_components(binary):
    """Behavior-equivalent stand-in for
    skimage.measure.label(binary, return_num=True, connectivity=2) (the
    source's `connectivity=binary.ndim` on a 2D array is 8-connectivity)
    using cv2.connectedComponents instead of adding scikit-image.
    Returns (label_image, num_labels_excluding_background)."""
    num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return labels, num_labels - 1


# ── body detector (body.py: Body) ─────────────────────────────────────────

class _Body:
    def __init__(self, model_path):
        self.model = BodyPoseModel()
        raw = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(_transfer_state_dict(self.model, raw))
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, oriImg):
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        pad_value = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

        # Device is always read from the model's own parameters, never
        # re-detected independently - see depth_anything_v2.py's note on
        # this exact bug.
        device = next(self.model.parameters()).device

        for m in range(len(multiplier)):
            scale = multiplier[m]
            image_to_test = _smart_resize_k(oriImg, fx=scale, fy=scale)
            image_to_test_padded, pad = _pad_right_down_corner(image_to_test, stride, pad_value)
            im = np.transpose(np.float32(image_to_test_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float().to(device)
            with torch.no_grad():
                mconv7_l1, mconv7_l2 = self.model(data)
            mconv7_l1 = mconv7_l1.cpu().numpy()
            mconv7_l2 = mconv7_l2.cpu().numpy()

            heatmap = np.transpose(np.squeeze(mconv7_l2), (1, 2, 0))
            heatmap = _smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
            heatmap = _smart_resize(heatmap, (oriImg.shape[0], oriImg.shape[1]))

            paf = np.transpose(np.squeeze(mconv7_l1), (1, 2, 0))
            paf = _smart_resize_k(paf, fx=stride, fy=stride)
            paf = paf[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
            paf = _smart_resize(paf, (oriImg.shape[0], oriImg.shape[1]))

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += paf / len(multiplier)

        all_peaks = []
        peak_counter = 0

        for part in range(18):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = _gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:, :] = one_heatmap[:-1, :]
            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1, :] = one_heatmap[1:, :]
            map_up = np.zeros(one_heatmap.shape)
            map_up[:, 1:] = one_heatmap[:, :-1]
            map_down = np.zeros(one_heatmap.shape)
            map_down[:, :-1] = one_heatmap[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (one_heatmap >= map_left, one_heatmap >= map_right,
                 one_heatmap >= map_up, one_heatmap >= map_down, one_heatmap > thre1))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                  [55, 56], [37, 38], [45, 46]]

        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(map_idx)):
            score_mid = paf_avg[:, :, [x - 19 for x in map_idx[k]]]
            cand_a = all_peaks[limb_seq[k][0] - 1]
            cand_b = all_peaks[limb_seq[k][1] - 1]
            n_a = len(cand_a)
            n_b = len(cand_b)
            if n_a != 0 and n_b != 0:
                connection_candidate = []
                for i in range(n_a):
                    for j in range(n_b):
                        vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        norm = max(0.001, norm)
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                            np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                                         for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                                         for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                        if len(connection) >= min(n_a, n_b):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(map_idx)):
            if k not in special_k:
                part_as = connection_all[k][:, 0]
                part_bs = connection_all[k][:, 1]
                index_a, index_b = np.array(limb_seq[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_b] != part_bs[i]:
                            subset[j][index_b] = part_bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][index_b] = part_bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[index_a] = part_as[i]
                        row[index_b] = part_bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        delete_idx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                delete_idx.append(i)
        subset = np.delete(subset, delete_idx, axis=0)

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total
        # score, 19 is the total parts. candidate: x, y, score, id.
        return candidate, subset

    @staticmethod
    def format_body_result(candidate: np.ndarray, subset: np.ndarray) -> List[BodyResult]:
        return [
            BodyResult(
                keypoints=[
                    Keypoint(
                        x=candidate[candidate_index][0],
                        y=candidate[candidate_index][1],
                        score=candidate[candidate_index][2],
                        id=candidate[candidate_index][3],
                    ) if candidate_index != -1 else None
                    for candidate_index in person[:18].astype(int)
                ],
                total_score=person[18],
                total_parts=person[19],
            )
            for person in subset
        ]


# ── hand detector (hand.py: Hand) ─────────────────────────────────────────

class _Hand:
    def __init__(self, model_path):
        self.model = HandPoseModel()
        raw = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(_transfer_state_dict(self.model, raw))
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, ori_img_raw):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        boxsize = 368
        stride = 8
        pad_value = 128
        thre = 0.05
        multiplier = [x * boxsize for x in scale_search]

        wsize = 128
        heatmap_avg = np.zeros((wsize, wsize, 22))

        Hr, Wr, _Cr = ori_img_raw.shape
        ori_img = cv2.GaussianBlur(ori_img_raw, (0, 0), 0.8)

        device = next(self.model.parameters()).device

        for m in range(len(multiplier)):
            scale = multiplier[m]
            image_to_test = _smart_resize(ori_img, (scale, scale))
            image_to_test_padded, pad = _pad_right_down_corner(image_to_test, stride, pad_value)
            im = np.transpose(np.float32(image_to_test_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float().to(device)
            with torch.no_grad():
                output = self.model(data).cpu().numpy()

            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
            heatmap = _smart_resize_k(heatmap, fx=stride, fy=stride)
            heatmap = heatmap[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
            heatmap = _smart_resize(heatmap, (wsize, wsize))

            heatmap_avg += heatmap / len(multiplier)

        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = _gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)

            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue

            label_img, label_numbers = _label_components(binary)
            max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = _npmax(map_ori)
            y = int(float(y) * float(Hr) / float(wsize))
            x = int(float(x) * float(Wr) / float(wsize))
            all_peaks.append([x, y])
        return np.array(all_peaks)


# ── face detector (face.py: Face) ─────────────────────────────────────────

class _Face:
    """
    Args:
        inference_size: inference image size, suggested 368/736/1312, default 736.
        gaussian_sigma: blur the heatmaps, default 2.5.
        heatmap_peak_thresh: return landmark if over threshold, default 0.1.
    """

    def __init__(self, model_path, inference_size=None, gaussian_sigma=None, heatmap_peak_thresh=None):
        self.inference_size = inference_size or 736
        self.sigma = gaussian_sigma or 2.5
        self.threshold = heatmap_peak_thresh or 0.1
        self.model = FaceNet()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, face_img):
        H, W, _C = face_img.shape
        w_size = 384
        device = next(self.model.parameters()).device

        x_data = torch.from_numpy(_smart_resize(face_img, (w_size, w_size))).permute([2, 0, 1]) / 256.0 - 0.5
        x_data = x_data.to(device)

        with torch.no_grad():
            hs = self.model(x_data[None, ...])
            heatmaps = F.interpolate(hs[-1], (H, W), mode="bilinear", align_corners=True).cpu().numpy()[0]
        return heatmaps

    def compute_peaks_from_heatmaps(self, heatmaps):
        all_peaks = []
        for part in range(heatmaps.shape[0]):
            map_ori = heatmaps[part].copy()
            binary = np.ascontiguousarray(map_ori > 0.05, dtype=np.uint8)

            if np.sum(binary) == 0:
                continue

            positions = np.where(binary > 0.5)
            intensities = map_ori[positions]
            mi = np.argmax(intensities)
            y, x = positions[0][mi], positions[1][mi]
            all_peaks.append([x, y])

        return np.array(all_peaks)


# ── hand/face region proposal from body keypoints (util.py) ──────────────

def _hand_detect(body: BodyResult, oriImg) -> List[Tuple[int, int, int, bool]]:
    """See CMU OpenPose's handDetector.cpp for the geometry this mirrors."""
    ratio_wrist_elbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]

    keypoints = body.keypoints
    left_shoulder = keypoints[5]
    left_elbow = keypoints[6]
    left_wrist = keypoints[7]
    right_shoulder = keypoints[2]
    right_elbow = keypoints[3]
    right_wrist = keypoints[4]

    has_left = all(k is not None for k in (left_shoulder, left_elbow, left_wrist))
    has_right = all(k is not None for k in (right_shoulder, right_elbow, right_wrist))
    if not (has_left or has_right):
        return []

    hands = []
    if has_left:
        hands.append([left_shoulder.x, left_shoulder.y, left_elbow.x, left_elbow.y,
                     left_wrist.x, left_wrist.y, True])
    if has_right:
        hands.append([right_shoulder.x, right_shoulder.y, right_elbow.x, right_elbow.y,
                     right_wrist.x, right_wrist.y, False])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        x = x3 + ratio_wrist_elbow * (x3 - x2)
        y = y3 + ratio_wrist_elbow * (y3 - y2)
        distance_wrist_elbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distance_elbow_shoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.5 * max(distance_wrist_elbow, 0.9 * distance_elbow_shoulder)
        x -= width / 2
        y -= width / 2
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        width1 = width
        width2 = width
        if x + width > image_width:
            width1 = image_width - x
        if y + width > image_height:
            width2 = image_height - y
        width = min(width1, width2)
        if width >= 20:
            detect_result.append((int(x), int(y), int(width), is_left))

    return detect_result


def _face_detect(body: BodyResult, oriImg) -> Union[Tuple[int, int, int], None]:
    image_height, image_width = oriImg.shape[0:2]
    keypoints = body.keypoints
    head = keypoints[0]
    left_eye = keypoints[14]
    right_eye = keypoints[15]
    left_ear = keypoints[16]
    right_ear = keypoints[17]

    if head is None or all(k is None for k in (left_eye, right_eye, left_ear, right_ear)):
        return None

    width = 0.0
    x0, y0 = head.x, head.y

    for kp, mult in ((left_eye, 3.0), (right_eye, 3.0), (left_ear, 1.5), (right_ear, 1.5)):
        if kp is not None:
            d = max(abs(x0 - kp.x), abs(y0 - kp.y))
            width = max(width, d * mult)

    x, y = x0, y0
    x -= width
    y -= width
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    width1 = width * 2
    width2 = width * 2
    if x + width > image_width:
        width1 = image_width - x
    if y + width > image_height:
        width2 = image_height - y
    width = min(width1, width2)

    if width >= 20:
        return int(x), int(y), int(width)
    return None


# ── rendering (util.py: draw_bodypose / draw_handpose / draw_facepose) ───

_EPS = 0.01


def draw_bodypose(canvas: np.ndarray, keypoints: List[Union[Keypoint, None]],
                  xinsr_stick_scaling: bool = False) -> np.ndarray:
    """Expects keypoint x/y normalized to [0, 1]."""
    H, W, _C = canvas.shape
    stickwidth = 4
    max_side = max(H, W)
    stick_scale = (1 if max_side < 500 else min(2 + (max_side // 1000), 7)) if xinsr_stick_scaling else 1

    limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
             [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
             [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limb_seq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]
        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth * stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        x = int(keypoint.x * W)
        y = int(keypoint.y * H)
        cv2.circle(canvas, (x, y), 4, color, thickness=-1)

    return canvas


def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas

    H, W, _C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
            [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        x1, y1 = int(k1.x * W), int(k1.y * H)
        x2, y2 = int(k2.x * W), int(k2.y * H)
        if x1 > _EPS and y1 > _EPS and x2 > _EPS and y2 > _EPS:
            r, g, b = colorsys.hsv_to_rgb(ie / float(len(edges)), 1.0, 1.0)
            cv2.line(canvas, (x1, y1), (x2, y2), (r * 255, g * 255, b * 255), thickness=2)

    for keypoint in keypoints:
        x, y = int(keypoint.x * W), int(keypoint.y * H)
        if x > _EPS and y > _EPS:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    if not keypoints:
        return canvas

    H, W, _C = canvas.shape
    for keypoint in keypoints:
        x, y = int(keypoint.x * W), int(keypoint.y * H)
        if x > _EPS and y > _EPS:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True,
               xinsr_stick_scaling=False) -> np.ndarray:
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)
        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand)
            canvas = draw_handpose(canvas, pose.right_hand)
        if draw_face:
            canvas = draw_facepose(canvas, pose.face)
    return canvas


# ── POSE_KEYPOINT JSON export (optional second estimate() return value) ──

def _encode_poses_as_dict(poses: List[PoseResult], canvas_height: int, canvas_width: int) -> dict:
    """OpenPose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md"""

    def compress(keypoints):
        if not keypoints:
            return None
        return [
            value
            for keypoint in keypoints
            for value in ([float(keypoint.x), float(keypoint.y), 1.0] if keypoint is not None else [0.0, 0.0, 0.0])
        ]

    return {
        "people": [
            {
                "pose_keypoints_2d": compress(pose.body.keypoints),
                "face_keypoints_2d": compress(pose.face),
                "hand_left_keypoints_2d": compress(pose.left_hand),
                "hand_right_keypoints_2d": compress(pose.right_hand),
            }
            for pose in poses
        ],
        "canvas_height": canvas_height,
        "canvas_width": canvas_width,
    }


# ── weight download + high-level detector ─────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(OPENPOSE_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("OpenPose: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(OPENPOSE_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=OPENPOSE_MODELS_DIR)


class OpenPoseDetector:
    """Loads all three checkpoints (downloading if needed) and estimates
    body/hand/face poses for one image, returning a rendered OpenPose-style
    skeleton IMAGE plus the POSE_KEYPOINT-style dict."""

    def __init__(self, body_filename="body_pose_model.pth", hand_filename="hand_pose_model.pth",
                face_filename="facenet.pth"):
        self.body_estimation = _Body(_download_checkpoint(body_filename))
        self.hand_estimation = _Hand(_download_checkpoint(hand_filename))
        self.face_estimation = _Face(_download_checkpoint(face_filename))

    def to(self, device):
        self.body_estimation.to(device)
        self.hand_estimation.to(device)
        self.face_estimation.to(device)
        return self

    def detect_hands(self, body: BodyResult, oriImg):
        left_hand = None
        right_hand = None
        H, W, _C = oriImg.shape
        for x, y, w, is_left in _hand_detect(body, oriImg):
            peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :]).astype(np.float32)
            if peaks.ndim == 2 and peaks.shape[1] == 2:
                peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
                peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
                hand_result = [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]
                if is_left:
                    left_hand = hand_result
                else:
                    right_hand = hand_result
        return left_hand, right_hand

    def detect_face(self, body: BodyResult, oriImg):
        face = _face_detect(body, oriImg)
        if face is None:
            return None
        x, y, w = face
        H, W, _C = oriImg.shape
        heatmaps = self.face_estimation(oriImg[y:y + w, x:x + w, :])
        peaks = self.face_estimation.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
            return [Keypoint(x=peak[0], y=peak[1]) for peak in peaks]
        return None

    def detect_poses(self, oriImg, include_hand=False, include_face=False) -> List[PoseResult]:
        """oriImg: RGB uint8 numpy [H,W,3] (converted to BGR internally to
        match the body/hand/face networks' training convention)."""
        oriImg = oriImg[:, :, ::-1].copy()
        H, W, _C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            bodies = self.body_estimation.format_body_result(candidate, subset)

            results = []
            for body in bodies:
                left_hand, right_hand, face = None, None, None
                if include_hand:
                    left_hand, right_hand = self.detect_hands(body, oriImg)
                if include_face:
                    face = self.detect_face(body, oriImg)

                results.append(PoseResult(BodyResult(
                    keypoints=[
                        Keypoint(x=keypoint.x / float(W), y=keypoint.y / float(H)) if keypoint is not None else None
                        for keypoint in body.keypoints
                    ],
                    total_score=body.total_score,
                    total_parts=body.total_parts,
                ), left_hand, right_hand, face))

            return results

    def estimate(self, image_hwc_uint8, resolution=512, include_body=True, include_hand=True,
                include_face=True, xinsr_stick_scaling=False):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3].
        Returns (rendered_skeleton_rgb_uint8 [H,W,3], pose_keypoint_dict)."""
        h, w = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        poses = self.detect_poses(resized, include_hand=include_hand, include_face=include_face)
        canvas = draw_poses(poses, target_h, target_w, draw_body=include_body,
                            draw_hand=include_hand, draw_face=include_face,
                            xinsr_stick_scaling=xinsr_stick_scaling)
        pose_dict = _encode_poses_as_dict(poses, target_h, target_w)

        if (target_h, target_w) != (h, w):
            canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        return canvas, pose_dict
