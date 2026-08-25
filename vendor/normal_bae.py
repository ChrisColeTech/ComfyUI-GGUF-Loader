# Copyright (c) 2022 Caroline Chan (NNET surface-normal-uncertainty architecture)
# MIT License (opensource.org/licenses/MIT)
"""Bae et al. surface normal estimator: EfficientNet-B5 encoder (via `timm`) +
BatchNorm upsampling decoder with an uncertainty-aware "kappa" output head
(NNET, "Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal
Estimation"), inference only.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/normalbae/ - consolidated from its nested
nets/{NNET.py, submodules/{encoder,decoder,submodules}.py} package into one
flat file per this repo's own convention.

Simplified for eval-only, single-image inference: the source's `from_pretrained`
always builds the decoder with `args.architecture = 'BN'`, so the parallel
`UpSampleGN` path (GroupNorm + weight-standardized `Conv2d`) is dead code for
every checkpoint this loads and was dropped, along with Decoder's `mode='train'`
branches and the uncertainty-guided `sample_points()` helper they alone call
(point sampling only matters when computing a training loss over a sparse set
of pixels - eval mode already evaluates every pixel densely). None of this
affects module hierarchy or state_dict keys.

This module has one dependency beyond this repo's existing ones: `timm`
(`pip install timm`), used to build the `tf_efficientnet_b5.ap_in1k` encoder
backbone exactly as the source does - it is not vendored here because
reproducing timm's EfficientNet module tree byte-for-byte (exact block/SE/
stem layout the "scannet.pt" checkpoint's state_dict expects) is infeasible
to hand-port safely.

Weights auto-download from HuggingFace on first use (lllyasviel/Annotators,
scannet.pt), same pattern as this repo's other vendor detectors: no config
file, no symlink cache tricks, just a plain local folder under
models/normal_bae/.
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

NORMAL_BAE_MODELS_DIR = os.path.join(folder_paths.models_dir, "normal_bae")

MODEL_REPO_IDS = {
    "scannet.pt": "lllyasviel/Annotators",
}


# ── EfficientNet-B5 encoder (nets/submodules/encoder.py) ────────────────

class Encoder(nn.Module):
    """Wraps timm's tf_efficientnet_b5.ap_in1k and collects every top-level
    submodule's output (unpacking `blocks` into its individual stages) into
    a flat list, exactly as the source does - the decoder indexes into this
    list positionally (features[3], features[4], features[5], features[7],
    features[10]), so the ordering must match timm's own module order."""

    def __init__(self):
        super().__init__()
        import timm
        self.original_model = timm.create_model("tf_efficientnet_b5.ap_in1k", pretrained=False, num_classes=0)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# ── decoder (nets/submodules/submodules.py + decoder.py) ────────────────

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features), nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features), nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode="bilinear", align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


def norm_normalize(norm_out):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    kappa = F.elu(kappa) + 1.0 + min_kappa
    return torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)


class Decoder(nn.Module):
    """BN-architecture decoder only (the source's from_pretrained() always
    passes architecture='BN'). Eval-mode forward only: every pixel is
    evaluated densely at each resolution (no uncertainty-guided sparse
    point sampling, which the source only uses to compute a training loss)."""

    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
        self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)

        # produces 1/8 res output
        self.out_conv_res8 = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(512 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1))

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1))

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(128 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1))

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[3], features[4], features[5], features[7], features[10])

        x_d0 = self.conv2(x_block4)          # 1/32 res
        x_d1 = self.up1(x_d0, x_block3)       # 1/16 res
        x_d2 = self.up2(x_d1, x_block2)       # 1/8 res
        x_d3 = self.up3(x_d2, x_block1)       # 1/4 res
        x_d4 = self.up4(x_d3, x_block0)       # 1/2 res

        out_res8 = self.out_conv_res8(x_d2)
        out_res8 = norm_normalize(out_res8)

        # 1/4 res
        feat_map = F.interpolate(x_d2, scale_factor=2, mode="bilinear", align_corners=True)
        init_pred = F.interpolate(out_res8, scale_factor=2, mode="bilinear", align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)
        B, _, H, W = feat_map.shape
        out_res4 = self.out_conv_res4(feat_map.view(B, 512 + 4, -1))
        out_res4 = norm_normalize(out_res4).view(B, 4, H, W)

        # 1/2 res
        feat_map = F.interpolate(x_d3, scale_factor=2, mode="bilinear", align_corners=True)
        init_pred = F.interpolate(out_res4, scale_factor=2, mode="bilinear", align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)
        B, _, H, W = feat_map.shape
        out_res2 = self.out_conv_res2(feat_map.view(B, 256 + 4, -1))
        out_res2 = norm_normalize(out_res2).view(B, 4, H, W)

        # 1/1 res
        feat_map = F.interpolate(x_d4, scale_factor=2, mode="bilinear", align_corners=True)
        init_pred = F.interpolate(out_res2, scale_factor=2, mode="bilinear", align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)
        B, _, H, W = feat_map.shape
        out_res1 = self.out_conv_res1(feat_map.view(B, 128 + 4, -1))
        out_res1 = norm_normalize(out_res1).view(B, 4, H, W)

        return [out_res8, out_res4, out_res2, out_res1]


class NNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img):
        return self.decoder(self.encoder(img))


# ── weight download + high-level detector ────────────────────────────────

def _load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location="cpu")["model"]
    load_dict = {}
    for k, v in ckpt.items():
        load_dict[k[len("module."):] if k.startswith("module.") else k] = v
    model.load_state_dict(load_dict)
    return model


def _download_checkpoint(filename):
    model_path = os.path.join(NORMAL_BAE_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("Normal BAE: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(NORMAL_BAE_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=NORMAL_BAE_MODELS_DIR)


class NormalBAEDetector:
    """Loads a checkpoint (downloading if needed) and estimates a surface
    normal map for one IMAGE tensor."""

    def __init__(self, ckpt_name="scannet.pt"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = NNET()
        self.model = _load_checkpoint(model_path, self.model)
        self.model = self.model.eval()
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to(self, device):
        self.model.to(device)
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        return self

    @torch.no_grad()
    def estimate(self, image_hwc_uint8, resolution=512):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3]."""
        h, w = image_hwc_uint8.shape[:2]
        k = float(resolution) / float(min(h, w))
        target_h, target_w = int(round(h * k)), int(round(w * k))
        resized = cv2.resize(image_hwc_uint8, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)

        device = next(self.model.parameters()).device
        image = torch.from_numpy(resized).float().to(device) / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        image = (image - self.norm_mean) / self.norm_std

        normal = self.model(image)
        normal = normal[-1][:, :3]
        normal = ((normal + 1) * 0.5).clip(0, 1)
        normal = normal[0].permute(1, 2, 0).cpu().numpy()
        normal_rgb = (normal * 255.0).clip(0, 255).astype(np.uint8)

        if (target_h, target_w) != (h, w):
            normal_rgb = cv2.resize(normal_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return normal_rgb
