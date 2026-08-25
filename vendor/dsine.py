# Copyright (c) Imperial College London (DSINE architecture)
# DSINE Software Licence Agreement (research/non-commercial use) - see
# https://github.com/baegwangbin/DSINE/blob/main/LICENSE for full terms.
# Rotation-matrix helpers (axis_angle_to_quaternion / quaternion_to_matrix)
# are NOTE'd by the source as "from PyTorch3D" (BSD-3-Clause).
"""DSINE surface normal estimator: EfficientNet-B5 encoder (via `timm`) +
a ray-direction-aware decoder with an iterative neighborhood-rotation
refinement (Bae & Davison, "Rethinking Inductive Biases for Surface Normal
Estimation", CVPR 2024), inference only. Camera-intrinsics-aware: takes a
per-pixel viewing-ray direction (derived from a field-of-view estimate)
into account when predicting normals.

Ported from Fannovel16/comfyui_controlnet_aux (Apache-2.0)'s
src/custom_controlnet_aux/dsine/ - consolidated from its nested
models/{dsine_arch.py, submodules/{standalone_encoder,submodules}.py} and
utils/{rotation.py, utils.py} package into one flat file per this repo's
own convention (depth_anything_v2.py, normal_bae.py).

This module has one dependency beyond this repo's existing ones: `timm`
(`pip install timm`), used to build the `tf_efficientnet_b5.ap_in1k` encoder
backbone exactly as the source does - same as this repo's normal_bae.py,
which uses the identical backbone; not vendored here because reproducing
timm's EfficientNet module tree byte-for-byte is infeasible to hand-port
safely.

Simplified for eval-only, single-image inference: DSINE()'s decoder is
always constructed with `BN=False`, so the parallel `UpSampleBN` path
(BatchNorm instead of GroupNorm + weight-standardized Conv2d) is dead code
for every checkpoint this loads and was dropped. None of this affects
module hierarchy or state_dict keys.

Weights auto-download from HuggingFace on first use (hr16/Diffusion-Edge,
dsine.pt - this is the exact, if oddly-named, repo the source's own
`DsineDetector.from_pretrained()` default downloads from), same pattern as
this repo's other vendor detectors: no config file, no symlink cache
tricks, just a plain local folder under models/dsine/.
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

DSINE_MODELS_DIR = os.path.join(folder_paths.models_dir, "dsine")

MODEL_REPO_IDS = {
    "dsine.pt": "hr16/Diffusion-Edge",
}


# ── EfficientNet-B5 encoder (models/submodules/standalone_encoder.py) ───

class Encoder(nn.Module):
    """Wraps timm's tf_efficientnet_b5.ap_in1k and collects every top-level
    submodule's output (unpacking `blocks` into its individual stages) into
    a flat list, exactly as the source does - the decoder indexes into this
    list positionally (features[5], features[7], features[10]), so the
    ordering must match timm's own module order."""

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


# ── decoder building blocks (models/submodules/submodules.py) ───────────

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, ks=3):
        super().__init__()
        p = (ks - 1) // 2
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, ks, padding=p)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, ks, padding=p)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, ks, padding=p)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class RayReLU(nn.Module):
    def __init__(self, eps=1e-2):
        super().__init__()
        self.eps = eps

    def forward(self, pred_norm, ray):
        cos = torch.cosine_similarity(pred_norm, ray, dim=1).unsqueeze(1)  # (B, 1, H, W)
        norm_along_view = ray * cos
        norm_along_view_relu = ray * (torch.relu(cos - self.eps) + self.eps)
        diff = norm_along_view_relu - norm_along_view
        new_pred_norm = pred_norm + diff
        return F.normalize(new_pred_norm, dim=1)


class Conv2d_WS(nn.Conv2d):
    """Weight-standardized Conv2d."""

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class UpSampleGN(nn.Module):
    """UpSample with GroupNorm (the only path DSINE's decoder actually uses)."""

    def __init__(self, skip_input, output_features, align_corners=True):
        super().__init__()
        self._net = nn.Sequential(
            Conv2d_WS(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, output_features), nn.LeakyReLU(),
            Conv2d_WS(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, output_features), nn.LeakyReLU())
        self.align_corners = align_corners

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)],
                             mode="bilinear", align_corners=self.align_corners)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


def convex_upsampling(out, up_mask, k):
    # out: low-resolution output (B, C, H, W); up_mask: (B, 9*k*k, H, W)
    B, C, H, W = out.shape
    up_mask = up_mask.view(B, 1, 9, k, k, H, W)
    up_mask = torch.softmax(up_mask, dim=2)

    out = F.pad(out, pad=(1, 1, 1, 1), mode="replicate")
    up_out = F.unfold(out, [3, 3], padding=0)          # (B, C X 3*3, H*W)
    up_out = up_out.view(B, C, 9, 1, 1, H, W)

    up_out = torch.sum(up_mask * up_out, dim=2)        # (B, C, k, k, H, W)
    up_out = up_out.permute(0, 1, 4, 2, 5, 3)           # (B, C, H, k, W, k)
    return up_out.reshape(B, C, k * H, k * W)


def get_unfold(pred_norm, ps, pad):
    B, C, H, W = pred_norm.shape
    pred_norm = F.pad(pred_norm, pad=(pad, pad, pad, pad), mode="replicate")
    pred_norm_unfold = F.unfold(pred_norm, [ps, ps], padding=0)  # (B, C X ps*ps, h*w)
    return pred_norm_unfold.view(B, C, ps * ps, H, W)


def get_prediction_head(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Conv2d(input_dim, hidden_dim, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim, 1), nn.ReLU(inplace=True),
        nn.Conv2d(hidden_dim, output_dim, 1))


INPUT_CHANNELS_DICT = {5: [2048, 176, 64, None, None]}  # EfficientNet-B5 only


# ── rotation helpers (utils/rotation.py, "from PyTorch3D") ──────────────

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


# ── decoder + DSINE model (models/dsine_arch.py) ─────────────────────────

class Decoder(nn.Module):
    def __init__(self, output_dims, B=5, NF=2048, downsample_ratio=8):
        super().__init__()
        input_channels = INPUT_CHANNELS_DICT[B]
        output_dim, feature_dim, hidden_dim = output_dims
        features = bottleneck_features = NF
        self.downsample_ratio = downsample_ratio

        self.conv2 = nn.Conv2d(bottleneck_features + 2, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleGN(skip_input=features // 1 + input_channels[1] + 2, output_features=features // 2, align_corners=False)
        self.up2 = UpSampleGN(skip_input=features // 2 + input_channels[2] + 2, output_features=features // 4, align_corners=False)

        i_dim = features // 4
        h_dim = 128
        self.normal_head = get_prediction_head(i_dim + 2, h_dim, output_dim)
        self.feature_head = get_prediction_head(i_dim + 2, h_dim, feature_dim)
        self.hidden_head = get_prediction_head(i_dim + 2, h_dim, hidden_dim)

    def forward(self, features, uvs):
        x_block2, x_block3, x_block4 = features[5], features[7], features[10]
        uv_32, uv_16, uv_8 = uvs

        x_d0 = self.conv2(torch.cat([x_block4, uv_32], dim=1))
        x_d1 = self.up1(x_d0, torch.cat([x_block3, uv_16], dim=1))
        x_feat = self.up2(x_d1, torch.cat([x_block2, uv_8], dim=1))
        x_feat = torch.cat([x_feat, uv_8], dim=1)

        normal = F.normalize(self.normal_head(x_feat), dim=1)
        f = self.feature_head(x_feat)
        h = self.hidden_head(x_feat)
        return normal, f, h


class DSINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample_ratio = 8
        self.ps = 5           # patch size
        self.num_iter = 5     # num iterations (overridable per-call)

        self.encoder = Encoder()

        self.output_dim = output_dim = 3
        self.feature_dim = feature_dim = 64
        self.hidden_dim = hidden_dim = 64
        self.decoder = Decoder([output_dim, feature_dim, hidden_dim], B=5, NF=2048)

        self.ray_relu = RayReLU(eps=1e-2)

        # pixel_coords (1, 3, H, W) - plain tensor attribute (not a buffer,
        # matching the source), moved manually in DSINEDetector.to().
        h = w = 2000
        pixel_coords = np.ones((3, h, w)).astype(np.float32)
        x_range = np.concatenate([np.arange(w).reshape(1, w)] * h, axis=0)
        y_range = np.concatenate([np.arange(h).reshape(h, 1)] * w, axis=1)
        pixel_coords[0, :, :] = x_range + 0.5
        pixel_coords[1, :, :] = y_range + 0.5
        self.pixel_coords = torch.from_numpy(pixel_coords).unsqueeze(0)

        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=feature_dim + 2, ks=self.ps)
        self.pad = (self.ps - 1) // 2

        self.prob_head = get_prediction_head(self.hidden_dim + 2, 64, self.ps * self.ps)
        self.xy_head = get_prediction_head(self.hidden_dim + 2, 64, self.ps * self.ps * 2)
        self.angle_head = get_prediction_head(self.hidden_dim + 2, 64, self.ps * self.ps)
        self.up_prob_head = get_prediction_head(self.hidden_dim + 2, 64, 9 * self.downsample_ratio * self.downsample_ratio)

    def get_ray(self, intrins, H, W, orig_H, orig_W, return_uv=False):
        B, _, _ = intrins.shape
        fu = intrins[:, 0, 0][:, None, None] * (W / orig_W)
        cu = intrins[:, 0, 2][:, None, None] * (W / orig_W)
        fv = intrins[:, 1, 1][:, None, None] * (H / orig_H)
        cv = intrins[:, 1, 2][:, None, None] * (H / orig_H)

        ray = self.pixel_coords[:, :, :H, :W].repeat(B, 1, 1, 1)
        ray[:, 0, :, :] = (ray[:, 0, :, :] - cu) / fu
        ray[:, 1, :, :] = (ray[:, 1, :, :] - cv) / fv

        if return_uv:
            return ray[:, :2, :, :]
        return F.normalize(ray, dim=1)

    def upsample(self, h, pred_norm, uv_8):
        up_mask = self.up_prob_head(torch.cat([h, uv_8], dim=1))
        up_pred_norm = convex_upsampling(pred_norm, up_mask, self.downsample_ratio)
        return F.normalize(up_pred_norm, dim=1)

    def refine(self, h, feat_map, pred_norm, intrins, orig_H, orig_W, uv_8, ray_8):
        B, C, H, W = pred_norm.shape
        fu = intrins[:, 0, 0][:, None, None, None] * (W / orig_W)
        cu = intrins[:, 0, 2][:, None, None, None] * (W / orig_W)
        fv = intrins[:, 1, 1][:, None, None, None] * (H / orig_H)
        cv = intrins[:, 1, 2][:, None, None, None] * (H / orig_H)

        h_new = self.gru(h, feat_map)

        nghbr_prob = self.prob_head(torch.cat([h_new, uv_8], dim=1)).unsqueeze(1)
        nghbr_prob = torch.sigmoid(nghbr_prob)

        nghbr_normals = get_unfold(pred_norm, ps=self.ps, pad=self.pad)

        nghbr_xys = self.xy_head(torch.cat([h_new, uv_8], dim=1))
        nghbr_xs, nghbr_ys = torch.split(nghbr_xys, [self.ps * self.ps, self.ps * self.ps], dim=1)
        nghbr_xys = torch.cat([nghbr_xs.unsqueeze(1), nghbr_ys.unsqueeze(1)], dim=1)
        nghbr_xys = F.normalize(nghbr_xys, dim=1)

        nghbr_angle = self.angle_head(torch.cat([h_new, uv_8], dim=1)).unsqueeze(1)
        nghbr_angle = torch.sigmoid(nghbr_angle) * np.pi

        nghbr_pixel_coord = get_unfold(self.pixel_coords[:, :, :H, :W], ps=self.ps, pad=self.pad)

        nghbr_axes = torch.zeros_like(nghbr_normals)

        du_over_fu = nghbr_xys[:, 0, ...] / fu
        dv_over_fv = nghbr_xys[:, 1, ...] / fv

        term_u = (nghbr_pixel_coord[:, 0, ...] + nghbr_xys[:, 0, ...] - cu) / fu
        term_v = (nghbr_pixel_coord[:, 1, ...] + nghbr_xys[:, 1, ...] - cv) / fv

        nx = nghbr_normals[:, 0, ...]
        ny = nghbr_normals[:, 1, ...]
        nz = nghbr_normals[:, 2, ...]

        nghbr_delta_z_num = -(du_over_fu * nx + dv_over_fv * ny)
        nghbr_delta_z_denom = (term_u * nx + term_v * ny + nz)
        nghbr_delta_z_denom[torch.abs(nghbr_delta_z_denom) < 1e-8] = \
            1e-8 * torch.sign(nghbr_delta_z_denom[torch.abs(nghbr_delta_z_denom) < 1e-8])
        nghbr_delta_z = nghbr_delta_z_num / nghbr_delta_z_denom

        nghbr_axes[:, 0, ...] = du_over_fu + nghbr_delta_z * term_u
        nghbr_axes[:, 1, ...] = dv_over_fv + nghbr_delta_z * term_v
        nghbr_axes[:, 2, ...] = nghbr_delta_z
        nghbr_axes = F.normalize(nghbr_axes, dim=1)

        invalid = torch.sum(torch.logical_or(torch.isnan(nghbr_axes), torch.isinf(nghbr_axes)).float(), dim=1) > 0.5
        nghbr_axes[:, 0, ...][invalid] = 0.0
        nghbr_axes[:, 1, ...][invalid] = 0.0
        nghbr_axes[:, 2, ...][invalid] = 0.0

        nghbr_axes_angle = nghbr_axes * nghbr_angle
        nghbr_axes_angle = nghbr_axes_angle.permute(0, 2, 3, 4, 1)  # (B, ps*ps, h, w, 3)
        nghbr_R = axis_angle_to_matrix(nghbr_axes_angle)            # (B, ps*ps, h, w, 3, 3)

        nghbr_normals_rot = torch.bmm(
            nghbr_R.reshape(B * self.ps * self.ps * H * W, 3, 3),
            nghbr_normals.permute(0, 2, 3, 4, 1).reshape(B * self.ps * self.ps * H * W, 3).unsqueeze(-1),
        ).reshape(B, self.ps * self.ps, H, W, 3, 1).squeeze(-1).permute(0, 4, 1, 2, 3)
        nghbr_normals_rot = F.normalize(nghbr_normals_rot, dim=1)

        nghbr_normals_rot = torch.cat([
            self.ray_relu(nghbr_normals_rot[:, :, i, :, :], ray_8).unsqueeze(2)
            for i in range(nghbr_normals_rot.size(2))
        ], dim=2)

        pred_norm = torch.sum(nghbr_prob * nghbr_normals_rot, dim=2)  # (B, C, H, W)
        pred_norm = F.normalize(pred_norm, dim=1)

        up_mask = self.up_prob_head(torch.cat([h_new, uv_8], dim=1))
        up_pred_norm = convex_upsampling(pred_norm, up_mask, self.downsample_ratio)
        up_pred_norm = F.normalize(up_pred_norm, dim=1)

        return h_new, pred_norm, up_pred_norm

    def forward(self, img, intrins):
        features = self.encoder(img)

        B, _, orig_H, orig_W = img.shape
        intrins = intrins.clone()
        intrins[:, 0, 2] += 0.5
        intrins[:, 1, 2] += 0.5
        uv_32 = self.get_ray(intrins, orig_H // 32, orig_W // 32, orig_H, orig_W, return_uv=True)
        uv_16 = self.get_ray(intrins, orig_H // 16, orig_W // 16, orig_H, orig_W, return_uv=True)
        uv_8 = self.get_ray(intrins, orig_H // 8, orig_W // 8, orig_H, orig_W, return_uv=True)
        ray_8 = self.get_ray(intrins, orig_H // 8, orig_W // 8, orig_H, orig_W)

        pred_norm, feat_map, h = self.decoder(features, uvs=(uv_32, uv_16, uv_8))
        pred_norm = self.ray_relu(pred_norm, ray_8)

        feat_map = torch.cat([feat_map, uv_8], dim=1)

        up_pred_norm = self.upsample(h, pred_norm, uv_8)
        for _ in range(self.num_iter):
            h, pred_norm, up_pred_norm = self.refine(
                h, feat_map, pred_norm.detach(), intrins, orig_H, orig_W, uv_8, ray_8)
        return up_pred_norm


# ── camera-intrinsics-from-FOV (utils/utils.py) ──────────────────────────

def get_intrins_from_fov(new_fov, H, W, device):
    """Builds an intrinsics matrix assuming a square-pixel camera whose
    field of view (along the longer image side) is `new_fov` degrees - the
    default (and only) way the source's node wrapper supplies intrinsics,
    since it never has real camera metadata for a plain input image."""
    if W >= H:
        new_fu = new_fv = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        new_fu = new_fv = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))

    new_cu = (W / 2.0) - 0.5
    new_cv = (H / 2.0) - 0.5

    return torch.tensor([
        [new_fu, 0, new_cu],
        [0, new_fv, new_cv],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)


# ── preprocessing (node_wrappers/dsine.py's common helpers) ─────────────

def _hwc3(x):
    if x.ndim == 2:
        x = x[:, :, None]
    H, W, C = x.shape
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    color = x[:, :, 0:3].astype(np.float32)
    alpha = x[:, :, 3:4].astype(np.float32) / 255.0
    y = color * alpha + 255.0 * (1.0 - alpha)
    return y.clip(0, 255).astype(np.uint8)


def _pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def _resize_image_with_pad(input_image, resolution):
    img = _hwc3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = _pad64(H_target), _pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode="constant")
    return np.ascontiguousarray(img_padded), (H_target, W_target)


def _get_pad(orig_H, orig_W):
    if orig_W % 64 == 0:
        l, r = 0, 0
    else:
        new_W = 64 * ((orig_W // 64) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l
    if orig_H % 64 == 0:
        t, b = 0, 0
    else:
        new_H = 64 * ((orig_H // 64) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b


# ── weight download + high-level detector ────────────────────────────────

def _download_checkpoint(filename):
    model_path = os.path.join(DSINE_MODELS_DIR, filename)
    if os.path.exists(model_path):
        return model_path
    from huggingface_hub import hf_hub_download
    logger.info("DSINE: downloading %s from %s ...", filename, MODEL_REPO_IDS[filename])
    os.makedirs(DSINE_MODELS_DIR, exist_ok=True)
    return hf_hub_download(repo_id=MODEL_REPO_IDS[filename], filename=filename, local_dir=DSINE_MODELS_DIR)


def _load_checkpoint(fpath, model):
    """Compatible-weights-only load (matches the source's own lenient
    load_checkpoint): a key is loaded only if it exists in the model and
    its checkpoint shape matches; mismatches are skipped with a warning
    instead of raising, exactly as the source pack does."""
    ckpt = torch.load(fpath, map_location="cpu")["model"]
    load_dict = {}
    for k, v in ckpt.items():
        load_dict[k[len("module."):] if k.startswith("module.") else k] = v

    model_state = model.state_dict()
    compatible_dict = {}
    skipped = []
    for k, v in load_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                skipped.append(k)
        else:
            skipped.append(k)
    if skipped:
        logger.warning("DSINE: skipped %d incompatible/unknown checkpoint keys (e.g. %s)",
                       len(skipped), skipped[:5])
    model.load_state_dict(compatible_dict, strict=False)
    return model


class DSINEDetector:
    """Loads a checkpoint (downloading if needed) and estimates a surface
    normal map for one IMAGE tensor."""

    def __init__(self, ckpt_name="dsine.pt"):
        model_path = _download_checkpoint(ckpt_name)
        self.model = DSINE()
        self.model = _load_checkpoint(model_path, self.model)
        self.model = self.model.eval()
        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to(self, device):
        self.model.to(device)
        self.model.pixel_coords = self.model.pixel_coords.to(device)
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        return self

    @torch.no_grad()
    def estimate(self, image_hwc_uint8, fov=60.0, iterations=5, resolution=512):
        """image_hwc_uint8: RGB uint8 numpy [H,W,3]. Returns RGB uint8 numpy [H,W,3].

        fov: assumed horizontal/vertical field of view in degrees, used to
        build a synthetic intrinsics matrix (no real camera metadata is
        available for a plain input image) - default 60.0, matching the
        source node wrapper's own default.
        iterations: number of iterative refinement steps (source default 5).
        """
        self.model.num_iter = iterations
        orig_H, orig_W = image_hwc_uint8.shape[:2]
        l, r, t, b = _get_pad(orig_H, orig_W)
        padded, (target_h, target_w) = _resize_image_with_pad(image_hwc_uint8, resolution)

        device = next(self.model.parameters()).device
        image = torch.from_numpy(padded).float().to(device) / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        image = (image - self.norm_mean) / self.norm_std

        intrins = get_intrins_from_fov(new_fov=fov, H=orig_H, W=orig_W, device=device).unsqueeze(0)
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        normal = self.model(image, intrins)[0]
        normal = ((normal + 1) * 0.5).clip(0, 1)
        normal = normal.permute(1, 2, 0).cpu().numpy()
        normal_rgb = (normal * 255.0).clip(0, 255).astype(np.uint8)

        normal_rgb = _hwc3(normal_rgb)
        # Drop the bottom/right multiple-of-64 padding before resizing back
        # to the caller's original resolution (the source pack itself never
        # removes this padding - it just returns the padded-resolution
        # image - but this repo's own convention, matching
        # DepthAnythingV2Detector/NormalBAEDetector, is to always hand back
        # an array shaped exactly like the input).
        normal_rgb = normal_rgb[:target_h, :target_w]
        if (target_h, target_w) != (orig_H, orig_W):
            normal_rgb = cv2.resize(normal_rgb, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
        return normal_rgb
