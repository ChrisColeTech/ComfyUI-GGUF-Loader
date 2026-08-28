"""LTX-2.3 A/V nodes: kit loader, T2V/I2V/A2V/IA2V prep, distilled sampler.

  LTXV23ModelsLoader   DiT + Gemma-3 TE (+ dual projection) + video VAE + audio VAE
  LTXV23ImgToVideo     prompts, init latent and noise masks for every mode
  LTXV23KSampler       euler on the DMD / distilled / refine sigma schedules
  LTXV23RefineSampler  base pass -> spatial x2 latent upscale -> refine pass
  LTXV23AVDecode       joint AV latent -> muxed VIDEO (video + audio VAE decode)

ID-LoRA talking-head pipelines (e.g. the ``ltxv23_talking_head`` gallery
workflow) layer a distilled LoRA + an ID-LoRA onto the model and a reference
audio clip onto the conditioning before sampling. Both of those are already
correctly served by stock core nodes -- ``LoraLoaderModelOnly`` (chain it
twice: distilled strength ~0.5, then the ID-LoRA at strength ~1.0) and
``LTXVReferenceAudio`` -- so this module does not wrap them; only the pieces
core leaves as loose multi-node wiring (the two-stage refine sampler, and the
final decode+mux) get a home here.

The conditioning path mirrors comfy_extras/nodes_lt.py rather than
reinterpreting it, because every deviation is a way to get a plausible-looking
video that is subtly wrong:

  * the init latent is [B, 128, (L-1)//8+1, H//32, W//32] and the encoded image
    is written into its FIRST latent frames (core LTXVImgToVideo);
  * the i2v hold is a PER-FRAME mask of shape [B, 1, T, 1, 1] carrying
    1 - strength on the held frames. Core's shape, not a full-size mask -
    sampling resizes masks, and a mask that already matches the latent skips
    that path entirely, so the two are not interchangeable by inspection;
  * the joint AV latent and its mask are NestedTensor((video, audio)), which is
    the only form core's samplers and LTXVSeparateAVLatent understand;
  * frame_rate rides on the conditioning (the DiT's RoPE reads it) via
    node_helpers.conditioning_set_values, the same call LTXVConditioning makes.

Audio-to-video follows the IA2V recipe: the encoded reference becomes the audio
stream with a ZERO mask, so it is held while the DiT generates matching video.
A clip shorter than the video is zero-padded with mask 1 on the tail, so the
model generates the remainder.
"""
import logging
import math
import re
import types
from fractions import Fraction

import comfy.ldm.common_dit
import comfy.ldm.lightricks.model
import comfy.model_management
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.quant_ops
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from comfy_api.latest import InputImpl as _VideoInputImpl
    from comfy_api.latest import Types as _VideoTypes
except ImportError:  # pragma: no cover - older comfy without the video API
    _VideoInputImpl = _VideoTypes = None

logger = logging.getLogger(__name__)

LTX23_CATEGORY = "\U0001F916 CCTech/LTX-2.3"

# LTX-2 distilled denoising schedule (ltx_pipelines.utils.constants
# .DISTILLED_SIGMAS). The kit has the distill LoRA pre-fused, so this schedule
# at cfg 1.0 is the intended way to run it.
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                    0.909375, 0.725, 0.421875, 0.0]
# Stage-2 refine schedule from the official workflows (after a latent x2
# spatial upscale): 3 steps, cfg 1.0, euler.
REFINE_SIGMAS = [0.85, 0.725, 0.4219, 0.0]

# Schedules for a DMD-distilled bake (TenStrip LTX2.3_DMD_reshaped_r256 fused
# at 1.0). DMD replaces the distill LoRA rather than stacking with it, so these
# are alternatives to DISTILLED_SIGMAS, not refinements of it.
#
# DMD_SIGMAS is what 10Eros_10SNodes_I2V_DMD_v1.json actually runs: the
# distilled curve with 0.98125 dropped and 0.78 inserted in the 0.909->0.725
# gap. That workflow feeds it through EchoDMDSigmaRemap(interpolate), which is
# an identity function -- interpolating at the input value returns the input --
# so the list below is the effective schedule, no remap needed.
DMD_SIGMAS = [1.0, 0.99375, 0.9875, 0.975,
              0.909375, 0.78, 0.725, 0.421875, 0.0]
# The model card's own recommendation, a smooth descent that shares no interior
# anchors with the distilled curve.
DMD_CARD_SIGMAS = [1.0, 0.955, 0.893, 0.812, 0.715,
                   0.603, 0.482, 0.241, 0.121, 0.0]
# Stage-2 upscale pass for DMD, from the same workflow.
DMD_UPSCALE_SIGMAS = [0.92, 0.909375, 0.725, 0.421875, 0.0]

# Order matters: this is the dropdown order, and the default sits first.
SIGMA_SETS = {
    "dmd (8 steps)": DMD_SIGMAS,
    "dmd card (9 steps)": DMD_CARD_SIGMAS,
    "dmd upscale (4 steps)": DMD_UPSCALE_SIGMAS,
    "distilled (8 steps)": DISTILLED_SIGMAS,
    "refine (3 steps)": REFINE_SIGMAS,
}

# LTXVImgToVideoInplace's value in the official I2V/IA2V first pass.
I2V_STRENGTH = 0.7
FPS = 24.0
_EXPECTED_LAYERS = 48

# Same geometry family as LTX-2.5; core builds exactly these numbers.
VIDEO_LATENT_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8


def _unet_filename_list():
    files = folder_paths.get_filename_list("unet")
    files += [f for f in folder_paths.get_filename_list("unet_gguf") if f not in files]
    return sorted(files)


def _clip_filename_list():
    files = folder_paths.get_filename_list("clip")
    files += [f for f in folder_paths.get_filename_list("clip_gguf") if f not in files]
    return sorted(files)


def _align_length(length):
    """Round a frame count up to the 8k+1 grid the video VAE tiles on."""
    length = max(9, int(length))
    while (length - 1) % VIDEO_TEMPORAL_RATIO != 0:
        length += 1
    return length


def distilled_sigma_schedule(steps, denoise=1.0, sigmas=None):
    """The LTX-2 distilled sigmas, resampled to ``steps``.

    At the trained 8 steps this is exactly the reference curve; other counts
    are a linear resample of it. ``denoise`` slices the head off, matching how
    KSampler treats denoise.
    """
    base = torch.tensor(sigmas if sigmas is not None else DISTILLED_SIGMAS,
                        dtype=torch.float32)
    steps = max(1, int(steps))
    if steps + 1 == len(base):
        out = base
    else:
        idx = torch.linspace(0.0, len(base) - 1.0, steps + 1)
        lo = idx.floor().long().clamp(0, len(base) - 2)
        frac = idx - lo
        out = base[lo] * (1.0 - frac) + base[lo + 1] * frac
    denoise = min(max(float(denoise), 0.0), 1.0)
    if denoise < 1.0:
        out = out[int(round(steps * (1.0 - denoise))):]
    return out


# ── loading ─────────────────────────────────────────────────────────────────

# Gemma-3 12B + projections costs ~24 GB and many seconds to build, and comfy
# re-runs loader nodes on every prompt edit. Cache the most recent pair; the
# CLIP owns its ModelPatcher so comfy still manages its VRAM.
_ENCODER_CACHE = {}


def _load_ltxv_clip(te_name, projections_name):
    key = (te_name, projections_name)
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]

    from ..loader import gguf_clip_loader
    from ..ops import GGMLOps

    te_path = folder_paths.get_full_path("clip", te_name) \
        or folder_paths.get_full_path_or_raise("text_encoders", te_name)
    proj_path = folder_paths.get_full_path("clip", projections_name) \
        or folder_paths.get_full_path_or_raise("text_encoders", projections_name)

    proj_sd, _ = comfy.utils.load_torch_file(proj_path, return_metadata=True)
    if "text_embedding_projection.audio_aggregate_embed.bias" not in proj_sd:
        raise ValueError(
            f"{projections_name} carries no text_embedding_projection tensors - "
            "it is not an LTX-2 projections file. Use the *_projections.safetensors "
            "that ships with the kit."
        )

    if te_path.lower().endswith(".gguf"):
        te_sd = gguf_clip_loader(te_path)
        te_options = {"custom_operations": GGMLOps(),
                      "initial_device": comfy.model_management.text_encoder_offload_device()}
    else:
        te_sd, _ = comfy.utils.load_torch_file(te_path, return_metadata=True)
        te_options = {"initial_device": comfy.model_management.text_encoder_offload_device()}

    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=comfy.sd.CLIPType.LTXV,
        state_dicts=[te_sd, proj_sd],
        model_options=te_options,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    del te_sd

    projection = type(getattr(clip.cond_stage_model,
                              "text_embedding_projection", None)).__name__
    if projection != "DualLinearProjection":
        raise ValueError(
            f"{te_name} + {projections_name} built a {projection}, not "
            "DualLinearProjection - the DiT's embeddings connectors cannot "
            "consume it."
        )

    _ENCODER_CACHE.clear()
    _ENCODER_CACHE[key] = clip
    return clip


def _load_vae(vae_name, want_audio):
    path = folder_paths.get_full_path("vae", vae_name) \
        or folder_paths.get_full_path_or_raise("vae", vae_name)
    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    is_audio = hasattr(vae.first_stage_model, "num_of_latents_from_frames")
    if want_audio and not is_audio:
        raise ValueError(f"{vae_name} is not an LTX audio VAE (looks like the "
                         f"video VAE); audio_vae wants *_audio_vae.safetensors.")
    if not want_audio and is_audio:
        raise ValueError(f"{vae_name} is the audio VAE; video_vae wants "
                         f"*_video_vae.safetensors.")
    return vae


class LTXV23ModelsLoader:
    """Load the whole LTX-2.3 A/V kit in one node.

    DiT GGUFs stay quantized (dequantized per layer at forward time). Outputs
    are plain comfy MODEL / CLIP / VAE objects, so they compose with comfy's
    own LTXV nodes as well as these.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Models Loader ⚡"
    SEARCH_ALIASES = ['load model', 'model loader', 'load vae', 'audio vae', 'video vae', 'load clip', 'text encoder']
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE")
    RETURN_NAMES = ("model", "clip", "vae", "audio_vae")
    FUNCTION = "load"
    DESCRIPTION = ("Load the LTX-2.3 A/V components (DiT, Gemma-3 text encoder "
                   "with dual projection, video VAE, audio VAE).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (_unet_filename_list(), {
                    "tooltip": "LTX-2.3 A/V DiT from models/diffusion_models. "
                               "GGUF stays quantized."}),
                "text_encoder_name": (_clip_filename_list(), {
                    "tooltip": "Gemma-3 12B text encoder from models/text_encoders."}),
                "projections_name": (_clip_filename_list(), {
                    "tooltip": "The kit's *_projections.safetensors — the dual "
                               "4096/2048 projection paired with the encoder."}),
                "video_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 video VAE."}),
                "audio_vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "LTX-2 audio VAE + vocoder."}),
            },
        }

    def load(self, unet_name, text_encoder_name, projections_name,
             video_vae_name, audio_vae_name):
        if unet_name.lower().endswith(".gguf"):
            from .gguf import UnetLoaderGGUF
            model, = UnetLoaderGGUF().load_unet(unet_name)
        else:
            model = comfy.sd.load_diffusion_model(
                folder_paths.get_full_path("unet", unet_name)
                or folder_paths.get_full_path_or_raise("unet", unet_name))

        cfg = model.model.model_config.unet_config
        if cfg.get("image_model") != "ltxav" or cfg.get("num_layers") != _EXPECTED_LAYERS:
            raise ValueError(
                f"{unet_name} built as image_model={cfg.get('image_model')} "
                f"num_layers={cfg.get('num_layers')}, expected "
                f"ltxav/{_EXPECTED_LAYERS}. Not an LTX-2.3 A/V checkpoint."
            )
        logger.info("LTX-2.3: %s -> ltxav/%d layers (%.2f GiB stored)",
                    unet_name, cfg.get("num_layers"), model.model_size() / 1024 ** 3)

        return (model,
                _load_ltxv_clip(text_encoder_name, projections_name),
                _load_vae(video_vae_name, want_audio=False),
                _load_vae(audio_vae_name, want_audio=True))


# ── EditAnything reference conditioning ─────────────────────────────────────
#
# Ported from D:\Projects\Wan2GP-main\models\ltx2\editanything.py (a separate
# multi-model video UI's from-scratch LTX-2.3 pipeline - read directly, not
# guessed) - the mechanism behind Lightricks/DeepBeepMeep's public
# "EditAnything" LoRA (huggingface.co/DeepBeepMeep/LTX-2,
# edit_anything_reference_v0.1_r128_*): injects one reference photo's
# identity into a generation via THREE simultaneous paths, not one:
#
#   1. the reference image gets VAE-encoded and appended as extra guide
#      tokens the model cross-attends to (comfy-core's own LTXVAddGuide -
#      the exact mechanism LTXV23VidToVideo's ic_lora selector already uses,
#      nothing new needed for this path);
#   2. a small extra cross-attention ("ref_attn") on transformer blocks 12-35
#      only, run against a 32-token pooled summary of the reference latent,
#      added as a tiny residual (scale 0.01) right after each block's own
#      attn2 - EDITANYTHING_REF_START/END_BLOCK below;
#   3. a global AdaLN modulation vector (scale 2.0), pooled from the same
#      reference latent, added directly into the model's embedded timestep
#      before any transformer block runs.
#
# Paths 2 and 3 are NOT LoRA weights - the .module.safetensors half of the
# EditAnything release is real extra architecture (new nn.Linear/LayerNorm
# layers with their own trained weights) that has to be loaded and wired
# into the model's forward pass directly. comfy has no built-in support for
# this, so it's grafted on here via the same clone-then-patch discipline
# this pack already uses (Krea2's _apply_identity_edit_patch,
# Krea2ControlLoRALoader's wrapper) - never touching comfy's own classes,
# only cloned model instances.
#
# comfy/ldm/lightricks/model.py's BasicTransformerBlock.forward() (verified
# by direct read this session) is an ordinary Python method on an ordinary
# nn.Module - patchable per-INSTANCE (never per-class, that would leak into
# every other loaded model) via a bound-method replacement. The patched
# version below is a faithful copy of that method's real body (not a
# re-derivation) with the ref_attn residual inserted at the exact point
# Wan2GP's own transformer.py:281-294 inserts it: after attn2, before the
# feed-forward. THIS WILL DRIFT if comfy's own BasicTransformerBlock.forward
# changes in a future ComfyUI update - the module docstring on
# _patched_block_forward below carries the same warning, check it first if
# EditAnything output ever looks wrong after a ComfyUI update.

EDITANYTHING_REF_START_BLOCK = 12
EDITANYTHING_REF_END_BLOCK = 35
EDITANYTHING_REF_CONTEXT_SCALE = 0.01
EDITANYTHING_REF_TOKEN_SCALE = 0.25
EDITANYTHING_ADALN_SCALE = 2.0

# The .module.safetensors file isn't a LoRA (comfy's own LoRA loader can't
# apply it - see _apply_editanything_patch's docstring), but it ships from
# the SAME HuggingFace release as its .standard.safetensors LoRA half and
# belongs in the same place a user would look for either: the real comfy
# `loras` folder_paths category (models/loras and whatever extra roots are
# registered there), not a bespoke category of its own - deploy both files
# side by side (e.g. models/loras/ltxv/) same as any other LTX-2.3 LoRA.
# Also register this pack's own D: archive location as an extra `loras`
# search root, so a fresh download there is discoverable without a manual
# copy into the portable install's models/loras.
import os as _os
_EDITANYTHING_EXTRA_DIR = r"D:\models\image-models-dev\ltxv23\edit_anything"
if _os.path.isdir(_EDITANYTHING_EXTRA_DIR):
    folder_paths.add_model_folder_path("loras", _EDITANYTHING_EXTRA_DIR)


class _EditAnythingLoRALinear(nn.Module):
    """A frozen base nn.Linear plus a LoRA delta, both real weights loaded
    from the .module.safetensors file - not comfy's own LoRA machinery
    (this isn't a patch onto an existing Linear's weight, it's a standalone
    extra attention module comfy never had in the first place)."""

    def __init__(self, base_linear, lora_a, lora_b):
        super().__init__()
        object.__setattr__(self, "base_linear", base_linear)
        self.lora_A = nn.Parameter(lora_a, requires_grad=False)
        self.lora_B = nn.Parameter(lora_b, requires_grad=False)

    def forward(self, x):
        out = self.base_linear(x)
        lora_dtype = self.lora_A.dtype
        lora_out = F.linear(F.linear(x.to(dtype=lora_dtype), self.lora_A), self.lora_B)
        return out.add(lora_out.to(device=out.device, dtype=out.dtype))


class _EditAnythingRefAttention(nn.Module):
    """The extra ref_attn cross-attention grafted onto one transformer
    block - reuses that block's own attn2 for q/k/v/out projection shapes
    and normalization (q_norm/k_norm), LoRA-augmented from the module file's
    per-block weights. Ported from editanything.py's EditAnythingRefAttention."""

    def __init__(self, base_attn, state, prefix):
        super().__init__()
        object.__setattr__(self, "base_attn", base_attn)
        self.heads = int(base_attn.heads)
        self.dim_head = int(base_attn.dim_head)
        self.to_q = _EditAnythingLoRALinear(base_attn.to_q, state[f"{prefix}to_q.lora_A.weight"], state[f"{prefix}to_q.lora_B.weight"])
        self.to_k = _EditAnythingLoRALinear(base_attn.to_k, state[f"{prefix}to_k.lora_A.weight"], state[f"{prefix}to_k.lora_B.weight"])
        self.to_v = _EditAnythingLoRALinear(base_attn.to_v, state[f"{prefix}to_v.lora_A.weight"], state[f"{prefix}to_v.lora_B.weight"])
        self.to_out = _EditAnythingLoRALinear(base_attn.to_out[0], state[f"{prefix}to_out.0.lora_A.weight"], state[f"{prefix}to_out.0.lora_B.weight"])

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        b = q.shape[0]
        q = q.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return self.to_out(out)


class _EditAnythingRefVisualProj(nn.Module):
    """Pools a VAE-encoded reference latent into 32 tokens (path 2 above).
    Ported from editanything.py's EditAnythingRefVisualProj - real weights
    (fc1/proj/norm/pos_embed), loaded strict from the module file."""

    def __init__(self, state):
        super().__init__()
        fc1_w, proj_w = state["fc1.weight"], state["proj.weight"]
        self.fc1 = nn.Linear(fc1_w.shape[1], fc1_w.shape[0], bias="fc1.bias" in state)
        self.proj = nn.Linear(proj_w.shape[1], proj_w.shape[0], bias="proj.bias" in state)
        self.norm = nn.LayerNorm(proj_w.shape[0])
        self.pos_embed = nn.Parameter(state["pos_embed"], requires_grad=False)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)

    def forward(self, ref_latent, token_scale=EDITANYTHING_REF_TOKEN_SCALE):
        ref_frame = ref_latent.mean(dim=2)  # (B,C,T,H,W) -> (B,C,H,W), collapse time
        local = F.adaptive_avg_pool2d(ref_frame, (4, 8)).permute(0, 2, 3, 1).reshape(ref_frame.shape[0], 32, -1)
        global_mean = ref_frame.mean(dim=(-2, -1))
        global_std = ref_frame.std(dim=(-2, -1), unbiased=False)
        stats = torch.cat([global_mean, global_std], dim=-1).unsqueeze(1).expand(-1, local.shape[1], -1)
        tokens = torch.cat([local, stats], dim=-1)
        tokens = self.proj(F.silu(self.fc1(tokens)))
        tokens = self.norm(tokens)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]].to(device=tokens.device, dtype=tokens.dtype)
        return tokens * float(token_scale)


class _EditAnythingRefAdaLNProj(nn.Module):
    """Pools the same reference latent into the global AdaLN modulation
    vector (path 3 above). Ported from editanything.py's
    EditAnythingRefAdaLNProj."""

    def __init__(self, state):
        super().__init__()
        fc1_w, proj_w = state["fc1.weight"], state["proj.weight"]
        self.fc1 = nn.Linear(fc1_w.shape[1], fc1_w.shape[0], bias="fc1.bias" in state)
        self.proj = nn.Linear(proj_w.shape[1], proj_w.shape[0], bias="proj.bias" in state)
        self.load_state_dict(state, strict=True)
        self.requires_grad_(False)

    def forward(self, ref_latent, adaln_scale=EDITANYTHING_ADALN_SCALE):
        ref_frame = ref_latent.mean(dim=2)
        avg_1x1 = F.adaptive_avg_pool2d(ref_frame, (1, 1)).flatten(1)
        avg_2x2 = F.adaptive_avg_pool2d(ref_frame, (2, 2)).flatten(1)
        max_1x1 = F.adaptive_max_pool2d(ref_frame, (1, 1)).flatten(1)
        pooled = torch.cat([avg_1x1, avg_2x2, max_1x1], dim=-1)
        return self.proj(F.silu(self.fc1(pooled))) * float(adaln_scale)


def _patched_block_forward(self, x, context=None, attention_mask=None, timestep=None, pe=None,
                           transformer_options={}, self_attention_mask=None, prompt_timestep=None):
    """Instance-level replacement for ONE patched BasicTransformerBlock's
    forward - a faithful copy of comfy/ldm/lightricks/model.py's real
    BasicTransformerBlock.forward (verified against that file directly this
    session; re-check it if EditAnything output ever looks wrong after a
    ComfyUI update, this is a real fork, not a wrapper) with exactly one
    addition: the ref_attn residual, inserted at the identical point
    Wan2GP's own transformer.py:281-294 inserts it - after attn2, before
    the feed-forward. Only ever bound onto blocks EDITANYTHING_REF_START_BLOCK..
    _END_BLOCK on a CLONED model (see _install_editanything_module) - never
    onto comfy's own class, which would leak into every other loaded model."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.scale_shift_table[None, None, :6].to(device=x.device, dtype=x.dtype)
        + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, :6, :]
    ).unbind(dim=2)

    if comfy.model_management.in_training:
        norm_x = comfy.ldm.common_dit.rms_norm(x) * (1 + scale_msa) + shift_msa
    else:
        norm_x = comfy.quant_ops.ck.rms_adaln(x, scale_msa, shift_msa)

    x += self.attn1(norm_x, pe=pe, mask=self_attention_mask, transformer_options=transformer_options) * gate_msa

    if self.cross_attention_adaln:
        shift_q_mca, scale_q_mca, gate_mca = (
            self.scale_shift_table[None, None, 6:9].to(device=x.device, dtype=x.dtype)
            + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)[:, :, 6:9, :]
        ).unbind(dim=2)
        x += comfy.ldm.lightricks.model.apply_cross_attention_adaln(
            x, context, self.attn2, shift_q_mca, scale_q_mca, gate_mca,
            self.prompt_scale_shift_table, prompt_timestep, attention_mask, transformer_options,
        )
    else:
        x += self.attn2(x, context=context, mask=attention_mask, transformer_options=transformer_options)

    # ── the one addition: EditAnything's ref_attn residual ──
    ref_context = transformer_options.get("editanything_ref_context")
    if ref_context is not None and EDITANYTHING_REF_START_BLOCK <= self.idx <= EDITANYTHING_REF_END_BLOCK:
        ref_out = self.ref_attn(comfy.ldm.common_dit.rms_norm(x), ref_context.to(device=x.device, dtype=x.dtype))
        x = x + ref_out * EDITANYTHING_REF_CONTEXT_SCALE

    y = comfy.ldm.common_dit.rms_norm(x)
    y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
    x.addcmul_(self.ff(y), gate_mlp)

    return x


def _install_editanything_module(model, module_path):
    """Load the EditAnything .module.safetensors file (NOT a LoRA) onto a
    CLONED model's diffusion_model: attach the two proj modules, attach and
    patch ref_attn on blocks EDITANYTHING_REF_START_BLOCK.._END_BLOCK only
    (the file carries ref_attn weights for all 48 blocks, but Wan2GP's own
    hardcoded start/end block range only ever fires 12-35 - the rest are
    dead weight in the file itself, skip loading them). Mutates `model` (a
    ModelPatcher, expected already cloned by the caller) in place; returns
    nothing. Also patches _prepare_timestep on the SAME clone's diffusion
    model to add ref_adaln - see LTXV23EditAnythingPatch.patch() for how
    ref_context/ref_adaln get computed and threaded through."""
    sd = comfy.utils.load_torch_file(module_path)
    dm = model.model.diffusion_model
    # comfy.utils.load_torch_file() loads onto CPU regardless of where the
    # rest of the model lives - the module's new submodules must be moved
    # onto the model's real compute device or their first real forward call
    # raises a CPU/CUDA mismatch (confirmed via a real traceback: "mat1 is
    # on cuda:0, different from other tensors on cpu").
    #
    # next(dm.parameters()).device is WRONG here and was tried first - it
    # reports wherever the model's weights CURRENTLY sit at this exact
    # moment (_install_editanything_module runs during node prep, before
    # comfy's own model management has necessarily streamed the model onto
    # GPU for the actual sampling step under offload/low-vram modes), not
    # where they'll actually run. comfy.model_management.get_torch_device()
    # is the real target compute device regardless of current placement -
    # the SAME call ref_latent below already (correctly) uses to land on
    # cuda:0, per the traceback itself ("mat1 is on cuda:0" - the input
    # activations were already using this convention; only the newly
    # attached weights weren't).
    target_device = comfy.model_management.get_torch_device()

    def _strip(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    visual_state = _strip("ref_visual_proj.")
    if not visual_state:
        raise ValueError(f"{module_path}: no ref_visual_proj.* tensors found - "
                         "not an EditAnything .module.safetensors file?")
    adaln_state = _strip("ref_adaln_proj.")
    dm.editanything_ref_visual_proj = _EditAnythingRefVisualProj(visual_state).to(device=target_device)
    dm.editanything_ref_adaln_proj = _EditAnythingRefAdaLNProj(adaln_state).to(device=target_device)

    patched = 0
    for i, block in enumerate(dm.transformer_blocks):
        if not (EDITANYTHING_REF_START_BLOCK <= i <= EDITANYTHING_REF_END_BLOCK):
            continue
        prefix = f"diffusion_model.transformer_blocks.{i}.ref_attn."
        if f"{prefix}to_q.lora_A.weight" not in sd:
            continue
        block.idx = i
        block.ref_attn = _EditAnythingRefAttention(block.attn2, sd, prefix).to(device=target_device)
        block.forward = types.MethodType(_patched_block_forward, block)
        patched += 1
    if patched == 0:
        raise ValueError(f"{module_path}: no ref_attn weights matched blocks "
                         f"{EDITANYTHING_REF_START_BLOCK}-{EDITANYTHING_REF_END_BLOCK}")
    logger.info("LTX-2.3 EditAnything: module installed, %d block(s) patched", patched)

    original_prepare_timestep = dm._prepare_timestep

    def _prepare_timestep_with_ref_adaln(self, *args, **kwargs):
        timestep, embedded_timestep, prompt_timestep = original_prepare_timestep(*args, **kwargs)
        ref_adaln = getattr(self, "_editanything_ref_adaln", None)
        if ref_adaln is not None:
            ref_adaln = ref_adaln.to(device=timestep.device, dtype=timestep.dtype)
            if ref_adaln.ndim == 2:
                ref_adaln = ref_adaln.unsqueeze(1)
            timestep = timestep + ref_adaln
        return timestep, embedded_timestep, prompt_timestep

    dm._prepare_timestep = types.MethodType(_prepare_timestep_with_ref_adaln, dm)


def _apply_editanything_patch(model, vae, reference_image, module_path, reference_mode="per_batch_item"):
    """Clone `model` and add LTX-2.3's EditAnything reference-conditioning
    path to the clone - a small extra cross-attention on transformer blocks
    12-35 plus a global timestep modulation, both driven by `reference_image`.
    See the "EditAnything reference conditioning" module docstring above for
    the full mechanism and where it was ported from. Shared by
    LTXV23VidToVideo's built-in `editanything_lora`/`editanything_module_path`
    selectors (the primary way to use this - see its docstring, mirrors
    Krea2Img2Img's `identity_edit` toggle) and the standalone
    LTXV23EditAnythingPatch node (for patching a model ahead of a different
    conditioning-prep node, e.g. LTXV23ImgToVideo).

    Needs BOTH EditAnything files, from the same HuggingFace release
    (DeepBeepMeep/LTX-2, edit_anything_reference_v0.1_r128_*):
      - the .standard.safetensors half is an ORDINARY LoRA - load it via
        comfy-core's real LoraLoaderModelOnly (this function never touches
        LoRA weights itself, only the extra .module architecture below).
      - the .module.safetensors half is what `module_path` points at here -
        real extra layers with their own trained weights, not something
        comfy's own LoRA loader can apply.
    Without the LoRA loaded, this patch alone is very unlikely to produce a
    recognizable result - the module's ref_attn/AdaLN paths were trained
    jointly with the LoRA's own attention deltas, not standalone.

    `reference_image` is VAE-encoded once here (not per sampling step).
    `reference_mode`: `per_batch_item` (default) encodes each image in the
    batch SEPARATELY and keeps them as distinct references, one per sample
    in the eventual sampling batch (tiled/truncated to fit - vae.encode()
    always collapses a single call's input batch into temporal frames of
    ONE video, comfy/sd.py VAE.encode, so a batch is encoded one image at a
    time and concatenated, never blended); `first_frame_only` uses only
    `reference_image[0]`, logging how many extra images were dropped - the
    right choice for vid2vid, where the reference batch isn't meant to line
    up with the sampling batch. `model` is cloned before any patching - the
    source model object passed in is never mutated.
    """
    m = model.clone()
    _install_editanything_module(m, folder_paths.get_full_path_or_raise(
        "loras", module_path))
    dm = m.model.diffusion_model

    pixels = reference_image[:, :, :, :3]
    if reference_mode == "first_frame_only":
        if pixels.shape[0] > 1:
            logger.info("LTX-2.3 EditAnything: reference_mode=first_frame_only, "
                        "using only the first of %d images", pixels.shape[0])
        pixels = pixels[:1]

    ref_latents = [vae.encode(pixels[i:i + 1]) for i in range(pixels.shape[0])]
    ref_latent = torch.cat(ref_latents, dim=0)
    ref_latent = m.model.process_latent_in(ref_latent).to(
        device=comfy.model_management.get_torch_device())

    visual_param = next(dm.editanything_ref_visual_proj.parameters())
    ref_context = dm.editanything_ref_visual_proj(
        ref_latent.to(dtype=visual_param.dtype)).detach()
    adaln_param = next(dm.editanything_ref_adaln_proj.parameters())
    ref_adaln = dm.editanything_ref_adaln_proj(
        ref_latent.to(dtype=adaln_param.dtype)).detach()

    def wrapper(executor, x, timesteps, context, attention_mask, frame_rate=25,
                transformer_options={}, keyframe_idxs=None, denoise_mask=None, **kwargs):
        transformer_options = dict(transformer_options)
        transformer_options["editanything_ref_context"] = _match_batch(ref_context, x.shape[0])
        dm._editanything_ref_adaln = _match_batch(ref_adaln, x.shape[0])
        return executor(x, timesteps, context, attention_mask, frame_rate,
                        transformer_options, keyframe_idxs, denoise_mask=denoise_mask, **kwargs)

    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "editanything_ref", wrapper)
    logger.info("LTX-2.3 EditAnything: patch applied, %d reference(s) (%s), shape %s",
                pixels.shape[0], reference_mode, tuple(pixels.shape))
    return m


class LTXV23EditAnythingPatch:
    """Standalone form of LTX-2.3's EditAnything reference-conditioning
    patch - see LTXV23VidToVideo's `editanything_lora`/`editanything_module_path`
    selectors instead for the primary, one-node way to use this (mirrors
    Krea2Img2Img's `identity_edit` toggle: the patch is built into the main
    conditioning node there too). Use THIS node only when you need the
    patched model ahead of a DIFFERENT conditioning-prep node - e.g.
    LTXV23ImgToVideo, or a plain txt2video graph with no video/vid2vid
    involved at all - since LTXV23VidToVideo's selectors only exist on that
    one node.

    Wire the SAME reference photo into this node's `reference_image` AND
    whatever conditioning node's `image` input for the intended full
    recipe (the guide-token append and the EditAnything path are
    complementary, not alternatives).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 EditAnything Reference Patch ⚡"
    SEARCH_ALIASES = ['edit anything', 'reference conditioning', 'identity insert',
                      'add person', 'reference image', 'ic-lora']
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    DESCRIPTION = ("Standalone EditAnything reference-conditioning patch (extra "
                   "cross-attention + timestep modulation from one or more "
                   "reference photos) for use ahead of a conditioning-prep node "
                   "other than LTXV23VidToVideo, which has this built in via the "
                   "editanything_lora/editanything_module_path selectors. Load "
                   "the EditAnything LoRA separately (stock LoraLoaderModelOnly) "
                   "before this.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE", {"tooltip": "Video VAE - encodes reference_image into "
                                "the same latent space the module's proj layers expect."}),
                "reference_image": ("IMAGE", {"tooltip": "The person/subject to inject. "
                                    "A batch of N images gives N distinct references, one "
                                    "per generation in the sampling batch (tiled/truncated "
                                    "to fit) - see reference_mode for the vid2vid case."}),
                "module_path": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "The EditAnything .module.safetensors file, from the "
                               "loras folder (it's not a LoRA itself, but ships from "
                               "the same release - keep it next to its .standard.safetensors "
                               "half, which loads separately via LoraLoaderModelOnly)."}),
                "reference_mode": (["per_batch_item", "first_frame_only"], {
                    "default": "per_batch_item",
                    "tooltip": "per_batch_item: each image in reference_image's batch is "
                               "encoded and used as its OWN distinct reference (not "
                               "blended) - image i drives sample i of the sampling batch, "
                               "tiled/truncated if the counts don't match. "
                               "first_frame_only: use only reference_image[0], ignore the "
                               "rest - the vid2vid recipe (one clean reference identity "
                               "against a single video), avoids a mismatched-count surprise "
                               "when reference_image's batch isn't meant to map 1:1 onto "
                               "the sampling batch."}),
            },
        }

    @torch.inference_mode()
    def patch(self, model, vae, reference_image, module_path, reference_mode="per_batch_item"):
        return (_apply_editanything_patch(model, vae, reference_image, module_path, reference_mode),)


# ── conditioning / latent prep ──────────────────────────────────────────────

def _encode_reference_audio(audio_vae, audio, duration_s):
    """Encode an AUDIO reference into a held audio latent [B, C, T, bins]."""
    fsm = getattr(audio_vae, "first_stage_model", None)
    if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
        raise ValueError("audio_vae is not an LTX audio VAE; reference_audio "
                         "needs the kit's *_audio_vae.safetensors.")

    waveform = audio["waveform"][0]          # [C, T] - one reference
    sr = audio["sample_rate"]
    max_samples = int(duration_s * sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    comfy.model_management.load_models_gpu(
        [audio_vae.patcher],
        force_full_load=getattr(audio_vae, "disable_offload", False))
    # The VAE is on the compute device now; the waveform must follow it or
    # torch raises "Input type (CPU) and weight type (CUDA) should be the same".
    waveform = waveform.to(device=audio_vae.device,
                           dtype=getattr(audio_vae, "vae_dtype", torch.float32))

    latent = fsm.encode(waveform.unsqueeze(0), sample_rate=sr)   # [1,C,T,bins]
    latent = latent.to(comfy.model_management.intermediate_device()).float()
    return latent, torch.zeros_like(latent)  # 0 = held: audio drives the video


def _match_batch(t, n):
    """Bring ``t`` to exactly ``n`` along dim 0 by tiling, then truncating.

    Deliberately idempotent: calling it twice is the same as calling it once.
    ``Tensor.repeat`` is not -- it multiplies -- so a value that has already
    been expanded once must never be repeated again.
    """
    have = t.shape[0]
    if have == n:
        return t
    if have > n:
        return t.narrow(0, 0, n)
    reps = -(-n // have)                      # ceil
    return t.repeat(reps, *([1] * (t.dim() - 1))).narrow(0, 0, n)


def _fit_audio_latent(audio, mask, target_shape):
    """Trim or zero-pad the audio latent along time to ``target_shape``.

    Port of core LTXVConcatAVLatent.fit_audio: a padded tail keeps mask 1 so
    the model generates it, which is what a clip shorter than the video means.

    Time only. The pad is allocated at the audio's own batch rather than
    ``target_shape[0]`` so this cannot silently expand the batch on the pad
    path while leaving it alone on the trim path -- that asymmetry is what
    made a following .repeat() produce batch_size**2 for short clips only.
    """
    dim, length = 2, target_shape[2]
    if audio.shape[dim] > length:
        audio = audio.narrow(dim, 0, length)
        if mask is not None:
            mask = mask.narrow(dim, 0, length)
    elif audio.shape[dim] < length:
        pad_shape = [audio.shape[0]] + list(target_shape[1:])
        pad = torch.zeros(pad_shape, device=audio.device, dtype=audio.dtype)
        pad[:, :, :audio.shape[dim]] = audio
        if mask is not None:
            pmask = torch.ones_like(pad)
            pmask[:, :, :mask.shape[dim]] = mask
            mask = pmask
        audio = pad
    return audio, mask


class LTXV23ImgToVideo:
    """Prompts + init latent for LTX-2.3: T2V, I2V, A2V and IA2V in one node.

      no image, no reference_audio -> text-to-video
      image                        -> image-to-video (first frames held at
                                      ``image_strength``)
      reference_audio              -> audio-to-video (audio stream held, video
                                      generated to match: lip sync, Foley)
      image + reference_audio      -> image-audio-to-video

    Feed the outputs into the LTX-2.3 KSampler; split its result with core
    LTXVSeparateAVLatent, decode video with VAE Decode and audio with
    LTXVAudioVAEDecode.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Img/Audio to Video ⚡"
    SEARCH_ALIASES = ['image to video', 'audio to video', 'img2vid', 'empty latent', 'video conditioning', 'text to video']
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts, init latent and noise masks for LTX-2.3 "
                   "T2V/I2V/A2V/IA2V. Feed into the LTX-2.3 KSampler.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "Describe the scene and its motion. "
                                                 "A caption, not an instruction."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 121, "min": 9,
                                   "max": nodes.MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Frames; 8k+1 tiles exactly (9, 97, 121...). "
                                              "Ignored when length_from_audio is on."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0,
                                         "step": 0.01,
                                         "tooltip": "24 is the LTX-2 convention. Match "
                                                    "this in CreateVideo or playback drifts."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "First frame. Resized and CENTER-CROPPED "
                                               "to width x height here - do not scale it "
                                               "upstream."}),
                "reference_audio": ("AUDIO",),
                "image_strength": ("FLOAT", {
                    "default": I2V_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "i2v only. How much of the init image to keep. 0.7 is "
                               "the official value; 1.0 locks the first frames hard."}),
                "length_from_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "With reference_audio: size the video to the clip."}),
            },
        }

    @torch.inference_mode()
    def prepare(self, clip, vae, audio_vae, prompt, negative_prompt, width,
                height, length, frame_rate, batch_size, image=None,
                reference_audio=None, image_strength=I2V_STRENGTH,
                length_from_audio=True):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use the kit's "
                             "*_audio_vae.safetensors in the audio_vae slot.")

        # ── audio stream ──
        audio = audio_mask = None
        if reference_audio is not None:
            seconds = reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
            if length_from_audio:
                length = _align_length(seconds * frame_rate + 1)
                logger.info("LTX-2.3 a2v: %.2fs of audio -> %d frames @ %.2f fps",
                            seconds, length, frame_rate)
            audio, audio_mask = _encode_reference_audio(
                audio_vae, reference_audio, length / frame_rate)

        # ── video stream, core LTXVImgToVideo's geometry ──
        length = _align_length(length)
        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        device = comfy.model_management.intermediate_device()
        video = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO],
            device=device)

        # Core shape: per-frame, broadcast over channels and space. Sampling
        # resizes masks, so this is NOT interchangeable with a full-size one.
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)
        if image is not None:
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            t = _match_batch(vae.encode(pixels[:, :, :, :3]), batch_size)
            video[:, :, :t.shape[2]] = t.to(video.device, video.dtype)
            video_mask[:, :, :t.shape[2]] = 1.0 - image_strength

        # ── audio geometry: empty, or the reference fitted to the video ──
        n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
        channels = int(getattr(audio_vae, "latent_channels", fsm.latent_channels))
        target = [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)]
        if audio is None:
            audio = torch.zeros(target, device=video.device)
            audio_mask = torch.ones_like(audio)
        else:
            audio, audio_mask = _fit_audio_latent(audio, audio_mask, target)
            audio = _match_batch(audio, batch_size).to(video.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video.device)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        # The DiT's RoPE reads frame_rate off the conditioning; same call
        # core's LTXVConditioning makes.
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        logger.info("LTX-2.3 prep: video %s mask %s, audio %s mask %s%s%s",
                    tuple(video.shape), tuple(video_mask.shape),
                    tuple(audio.shape), tuple(audio_mask.shape),
                    ", image held @ %.2f" % image_strength if image is not None else "",
                    ", audio locked" if reference_audio is not None else "")
        return (positive, negative, latent)


class LTXV23VidToVideo:
    """Prompts + init latent for LTX-2.3: T2V/I2V/V2V, with LoRA selection
    and reference-conditioning all built into this one node - no external
    LoraLoaderModelOnly wiring, no separate "is X attached" boolean toggles.
    Every mechanism here is driven purely by whether its selector is set
    (a real filename) or left at "none" - matching this node's `model`
    input/output design (patched model out, or the original unpatched
    model straight through, if nothing was selected).

    `mode` (required) declares which of the three base behaviors this call
    is: `t2v` (images/video must be disconnected - pure text-to-video),
    `i2v` (`images` required - ordinary first-frame hold, VAE-encoded and
    held at `image_strength` via the noise mask), or `v2v` (`video`
    required - IC-LoRA guide injection, see `ic_lora` below). `i2v`/`v2v`
    can still layer on top of each other exactly as before (image hold +
    video guide in the same call) - `mode` only says which one is the
    PRIMARY, required source; the other stays optional and combinable.

    `ic_lora` (optional selector, "none" = off) loads an IC-LoRA (in-context
    LoRA) task adapter directly onto `model` inside this node (comfy-core's
    real `LoraLoaderModelOnly.load_lora_model_only`, not reimplemented) at
    `ic_lora_strength`, then injects `video`'s frames as extra "guide"
    tokens the model attends to at the SAME timeline position as what it's
    generating (comfy-core's keyframe_idxs RoPE mechanism,
    comfy_extras/nodes_lt.py's LTXVAddGuide) - a genuinely different thing
    from `images`'s first-frame hold, and from Krea2Img2Img's partial-denoise
    img2img. `ic_lora="none"` (default): `video`, if connected, is only
    used for length/frame_rate/original audio, ignored for guidance. When
    `video` is connected, `length`/`frame_rate` are taken FROM it
    (overriding the widgets, logged); `width`/`height` always come from the
    widgets regardless of source, same as `images`. `keep_original_audio`
    (default on, only relevant with `video`) keeps the source clip's own
    audio unchanged in the output using this file's own already-proven
    `_encode_reference_audio`/`_fit_audio_latent` helpers - mechanically
    identical to `reference_audio` below, just fed the source video's own
    audio track instead of a separate clip. Off = the model generates new
    audio from scratch instead.

    `reference_audio` (optional, mutually exclusive in effect with
    `video`+`keep_original_audio` - the LAST one prepared wins if both are
    wired, though wiring both makes little sense) - LTXV23ImgToVideo's own
    A2V path, unchanged, for driving generation from a voice/sound clip
    with no source video at all.

    `editanything_lora` + `editanything_module_path` (both optional
    selectors, "none" = off - EITHER alone does nothing useful, they're
    trained jointly) add LTX-2.3's EditAnything reference-conditioning
    patch to `model` in this same node - mirrors Krea2Img2Img's own
    `identity_edit` toggle (patch built into the main conditioning node,
    reusing the SAME `images` input for both purposes rather than a
    second reference-photo slot - `identity_edit=True` switches what
    `images` means there instead of adding a new input; this node does
    the same), except the LoRA half is now ALSO loaded here (same
    `LoraLoaderModelOnly` delegation as `ic_lora` above, at
    `editanything_lora_strength`) instead of requiring external wiring.
    Needs `images` connected too (the person/subject to inject - one or
    more photos, see `reference_mode` below for what a batch means here;
    the SAME photo(s) drive both the ordinary i2v hold, if
    `image_strength` > 0, AND the EditAnything reference; set
    `image_strength=0` to use `images` for EditAnything only, with no
    first-frame hold effect) - see `_apply_editanything_patch`'s docstring
    for the full mechanism and what `reference_mode` controls. The
    returned `model` is the patched clone (or the original, unpatched, if
    neither selector is set); always take `model` from THIS node's output,
    not the original upstream model, whenever any selector here is used.

    All three LoRA-adjacent selectors (`ic_lora`, `editanything_lora`,
    `editanything_module_path`) list comfy's real `loras` folder_paths
    category - the same place a plain `LoraLoaderModelOnly` looks, so any
    file placed there for one is visible to all.

    This node calls comfy-core's real LTXVAddGuide.get_latent_index()/
    append_keyframe() directly (comfy_extras.nodes_lt, always present
    regardless of any custom_nodes pack's state) rather than re-deriving the
    RoPE coordinate math by hand - the same call the official Lightricks
    IC-LoRA nodes make internally (ComfyUI-LTXVideo/iclora.py), just wired
    into this pack's own joint-AV-latent convention
    (comfy.nested_tensor.NestedTensor((video, audio)), matching
    LTXV23ImgToVideo) instead of the ~10-node chain the official example
    workflows wire by hand. `frame_idx` is always 0 (guide spans the whole
    clip from the start) - the only sensible mode for a plain vid2vid node;
    an offset guide needs the raw core/Lightricks nodes directly.

    Feed the outputs into LTXV23KSampler UNCHANGED - it already samples any
    joint AV latent + noise mask generically, so no custom sampler is
    needed here. Then LTXV23CropVideoGuide before LTXV23AVDecode to strip
    the reference frames back out (a no-op if `video`/`ic_lora` were never
    used).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Video to Video (IC-LoRA) ⚡"
    SEARCH_ALIASES = ['video to video', 'vid2vid', 'v2v', 'ic-lora', 'ic lora',
                       'video guide', 'video conditioning', 'image to video', 'img2vid']
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "FLOAT")
    RETURN_NAMES = ("model", "positive", "negative", "latent", "frame_rate")
    FUNCTION = "prepare"
    DESCRIPTION = ("Prompts and init latent for LTX-2.3 T2V/I2V/V2V - LoRA "
                   "selection and EditAnything reference-conditioning both "
                   "built in, no external LoraLoaderModelOnly wiring "
                   "needed. Feed the outputs into LTXV23KSampler, then "
                   "LTXV23CropVideoGuide before LTXV23AVDecode.")

    @classmethod
    def INPUT_TYPES(s):
        lora_choices = ["none"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "mode": (["t2v", "i2v", "v2v"], {"default": "t2v",
                          "tooltip": "Which base behavior this call is. t2v: images/video "
                                     "must be disconnected. i2v: images required (video may "
                                     "still layer on top as an IC-LoRA guide). v2v: video "
                                     "required (images may still layer on top as a first-"
                                     "frame hold)."}),
                "vae": ("VAE", {"tooltip": "The loader's video_vae output."}),
                "audio_vae": ("VAE", {"tooltip": "The loader's audio_vae output."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "tooltip": "With ic_lora set: describe the OUTPUT "
                                                 "you want - most IC-LoRAs are trained on an "
                                                 "instruction-style caption describing the "
                                                 "transformed result. Otherwise: describe the "
                                                 "scene and its motion, a caption not an "
                                                 "instruction."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 121, "min": 9,
                                   "max": nodes.MAX_RESOLUTION, "step": 8,
                                   "tooltip": "Frames; 8k+1 tiles exactly. Ignored (taken "
                                              "from the clip instead) when video is "
                                              "connected."}),
                "frame_rate": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0, "step": 0.01,
                                         "tooltip": "Ignored (taken from the clip instead) "
                                                    "when video is connected."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "First frame(s) for image-to-video (ordinary "
                                    "i2v hold, independent of video/ic_lora below) - "
                                    "resized and CENTER-CROPPED to width x height. Also the "
                                    "EditAnything reference photo(s) when "
                                    "editanything_module_path is set (see reference_mode for "
                                    "how a batch is used there) - set image_strength=0 for a "
                                    "pure EditAnything reference with no i2v-hold effect."}),
                "image_strength": ("FLOAT", {
                    "default": I2V_STRENGTH, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "images only. How much of the init image(s) to keep. 0.7 is "
                               "the official value; 1.0 locks the first frames hard."}),
                "video": ("VIDEO", {"tooltip": "Source clip for IC-LoRA video-to-video "
                                    "(see ic_lora) and/or its original audio (see "
                                    "keep_original_audio). Sets length/frame_rate from itself."}),
                "ic_lora": (lora_choices, {"default": "none",
                            "tooltip": "video only. The IC-LoRA task adapter (beard removal, "
                                       "HDR grading, motion tracking, ...) - loaded onto model "
                                       "HERE (no external LoraLoaderModelOnly needed) at "
                                       "ic_lora_strength, then drives the actual vid2vid guide-"
                                       "injection mechanism. \"none\" = video is used only for "
                                       "length/frame_rate/original audio, ignored for guidance "
                                       "- useful for A/B-ing whether the IC-LoRA is doing "
                                       "anything."}),
                "ic_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0,
                                     "step": 0.01,
                                     "tooltip": "ic_lora only. Same as LoraLoaderModelOnly's "
                                                "strength_model."}),
                "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "ic_lora only. How strongly the reference is "
                                              "held. 1.0 = fully held (official default)."}),
                "keep_original_audio": ("BOOLEAN", {"default": True,
                               "tooltip": "video only. On = output keeps the source clip's "
                                          "own audio unchanged. Off = the model generates "
                                          "new audio from scratch instead."}),
                "latent_downscale_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0,
                                            "step": 1.0,
                                            "tooltip": "ic_lora only. Only for IC-LoRAs "
                                                       "trained on a downscaled reference "
                                                       "grid (rare - check the LoRA's model "
                                                       "card / reference_downscale_factor "
                                                       "metadata; most, including every "
                                                       "official example, use 1.0)."}),
                "reference_audio": ("AUDIO", {"tooltip": "Drive generation from a voice/sound "
                                              "clip with no source video (LTXV23ImgToVideo's "
                                              "A2V path). Not meant to be combined with "
                                              "video+keep_original_audio."}),
                "length_from_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "With reference_audio and no video: size the video to the clip."}),
                "editanything_lora": (lora_choices, {"default": "none",
                                      "tooltip": "The EditAnything .standard.safetensors LoRA "
                                                 "half - loaded onto model HERE at "
                                                 "editanything_lora_strength (no external "
                                                 "LoraLoaderModelOnly needed). \"none\" = off. "
                                                 "Needs editanything_module_path set too - "
                                                 "either alone does nothing useful, they're "
                                                 "trained jointly."}),
                "editanything_lora_strength": ("FLOAT", {"default": 1.0, "min": -100.0,
                                               "max": 100.0, "step": 0.01,
                                               "tooltip": "editanything_lora only. Same as "
                                                          "LoraLoaderModelOnly's strength_model."}),
                "editanything_module_path": (lora_choices, {"default": "none",
                    "tooltip": "The EditAnything .module.safetensors file (NOT a LoRA - real "
                               "extra layers, loaded by this pack's own patch mechanism), "
                               "from the loras folder. \"none\" = off. Needs "
                               "editanything_lora and `images` connected too."}),
                "reference_mode": (["per_batch_item", "first_frame_only"], {
                    "default": "first_frame_only",
                    "tooltip": "Needs editanything_module_path set. Controls how `images`'s "
                               "batch is used as the EditAnything reference (independent of "
                               "its ordinary i2v-hold use). per_batch_item: each image in "
                               "the batch is encoded and used as its OWN distinct reference "
                               "(not blended) - image i drives sample i of the sampling "
                               "batch, tiled/truncated if the counts don't match. "
                               "first_frame_only (default here): use only images[0], ignore "
                               "the rest - the vid2vid recipe (one clean reference identity "
                               "against a single video)."}),
            },
        }

    @torch.inference_mode()
    def prepare(self, model, clip, vae, audio_vae, mode, prompt, negative_prompt, width, height,
                length, frame_rate, batch_size, images=None, image_strength=I2V_STRENGTH,
                video=None, ic_lora="none", ic_lora_strength=1.0, guide_strength=1.0,
                keep_original_audio=True, latent_downscale_factor=1.0, reference_audio=None,
                length_from_audio=True, editanything_lora="none", editanything_lora_strength=1.0,
                editanything_module_path="none", reference_mode="first_frame_only"):
        fsm = getattr(audio_vae, "first_stage_model", None)
        if fsm is None or not hasattr(fsm, "num_of_latents_from_frames"):
            raise ValueError("audio_vae is not an LTX audio VAE; use the kit's "
                             "*_audio_vae.safetensors in the audio_vae slot.")

        ic_lora_attached = ic_lora not in (None, "none", "")
        editanything_lora_attached = editanything_lora not in (None, "none", "")
        edit_anything = editanything_module_path not in (None, "none", "")
        if editanything_lora_attached != edit_anything:
            raise ValueError("LTX-2.3 v2v: editanything_lora and editanything_module_path must "
                             "be set together (both \"none\" or both a real file) - either "
                             "alone does nothing, they're trained jointly.")

        # images is dual-purpose: ordinary i2v hold AND (when edit_anything
        # is on) the EditAnything reference photo(s) - mode=t2v only
        # forbids it when it would otherwise ONLY be doing the (unwanted,
        # in t2v) i2v hold job; with edit_anything on, images connected
        # under t2v is the "no first-frame hold, just inject this
        # identity" recipe (pair with image_strength=0 to suppress the
        # hold entirely).
        if mode == "t2v" and video is not None:
            raise ValueError("LTX-2.3 v2v: mode=t2v but video is connected - disconnect it "
                             "or pick v2v.")
        if mode == "t2v" and images is not None and not edit_anything:
            raise ValueError("LTX-2.3 v2v: mode=t2v but images is connected - disconnect it, "
                             "pick i2v, or set editanything_lora/editanything_module_path if "
                             "images is meant as an EditAnything reference, not an i2v hold.")
        if mode == "i2v" and images is None:
            raise ValueError("LTX-2.3 v2v: mode=i2v needs images connected.")
        if mode == "v2v" and video is None:
            raise ValueError("LTX-2.3 v2v: mode=v2v needs video connected.")

        if ic_lora_attached:
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, ic_lora, ic_lora_strength)[0]
        if editanything_lora_attached:
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(
                model, editanything_lora, editanything_lora_strength)[0]
        if edit_anything:
            if images is None:
                raise ValueError("LTX-2.3 v2v: editanything_module_path is set but "
                                 "images isn't connected (images doubles as the EditAnything "
                                 "reference photo(s) here - no separate reference_image slot).")
            model = _apply_editanything_patch(
                model, vae, images, editanything_module_path, reference_mode)

        video_frames = video_audio = None
        if video is not None:
            components = video.get_components()
            video_frames = components.images
            video_audio = components.audio
            src_length = _align_length(video_frames.shape[0])
            src_fps = float(components.frame_rate)
            if src_length != _align_length(length) or src_fps != frame_rate:
                logger.info("LTX-2.3 v2v: video connected - length %d -> %d, "
                           "frame_rate %.2f -> %.2f (taken from the clip)",
                           length, src_length, frame_rate, src_fps)
            length, frame_rate = src_length, src_fps
        elif reference_audio is not None and length_from_audio:
            seconds = reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
            length = _align_length(seconds * frame_rate + 1)
            logger.info("LTX-2.3 a2v: %.2fs of audio -> %d frames @ %.2f fps",
                        seconds, length, frame_rate)
        else:
            length = _align_length(length)

        t_latent = ((length - 1) // VIDEO_TEMPORAL_RATIO) + 1
        device = comfy.model_management.intermediate_device()
        video_samples = torch.zeros(
            [batch_size, VIDEO_LATENT_CHANNELS, t_latent,
             height // VIDEO_SPATIAL_RATIO, width // VIDEO_SPATIAL_RATIO], device=device)
        video_mask = torch.ones((batch_size, 1, t_latent, 1, 1),
                                dtype=torch.float32, device=device)
        if images is not None:
            pixels = comfy.utils.common_upscale(
                images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            t = _match_batch(vae.encode(pixels[:, :, :, :3]), batch_size)
            video_samples[:, :, :t.shape[2]] = t.to(video_samples.device, video_samples.dtype)
            video_mask[:, :, :t.shape[2]] = 1.0 - image_strength

        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})

        if video is not None and ic_lora_attached:
            import comfy_extras.nodes_lt as nodes_lt

            scale_factors = vae.downscale_index_formula
            time_scale = scale_factors[0]
            n = video_frames.shape[0]
            guide_frames = video_frames[:((n - 1) // time_scale) * time_scale + 1]
            target_w = int(width / latent_downscale_factor)
            target_h = int(height / latent_downscale_factor)
            guide_pixels = comfy.utils.common_upscale(
                guide_frames.movedim(-1, 1), target_w, target_h, "bilinear", "center"
            ).movedim(1, -1)[:, :, :, :3]
            guide_latent = _match_batch(vae.encode(guide_pixels), batch_size)

            guide_mask = None
            if latent_downscale_factor > 1:
                if width % latent_downscale_factor != 0 or height % latent_downscale_factor != 0:
                    raise ValueError(
                        f"LTX-2.3 v2v: target size {width}x{height} must be divisible by "
                        f"latent_downscale_factor {latent_downscale_factor}.")
                guide_latent, guide_mask = nodes_lt.LTXVAddGuide.dilate_latent(
                    guide_latent, latent_downscale_factor)

            frame_idx, _ = nodes_lt.LTXVAddGuide.get_latent_index(
                positive, t_latent, guide_latent.shape[2], 0, scale_factors,
                latent_shape=video_samples.shape)
            positive, negative, video_samples, video_mask = nodes_lt.LTXVAddGuide.append_keyframe(
                positive, negative, frame_idx, video_samples, video_mask, guide_latent,
                guide_strength, scale_factors, guide_mask=guide_mask,
                latent_downscale_factor=latent_downscale_factor, causal_fix=True)
            logger.info("LTX-2.3 v2v: guide %s appended @ strength %.2f (frame_idx=%d)",
                        tuple(guide_latent.shape), guide_strength, frame_idx)
        elif video is not None:
            logger.info("LTX-2.3 v2v: ic_lora=none - video used for length/frame_rate/"
                        "held audio only, ignored for guidance")

        n_latents = int(fsm.num_of_latents_from_frames(length, frame_rate))
        channels = int(getattr(audio_vae, "latent_channels", fsm.latent_channels))
        target = [batch_size, channels, n_latents, int(fsm.latent_frequency_bins)]
        if video is not None and keep_original_audio and video_audio is not None:
            audio_latent, audio_mask = _encode_reference_audio(
                audio_vae, video_audio, length / frame_rate)
            audio_latent, audio_mask = _fit_audio_latent(audio_latent, audio_mask, target)
            audio_latent = _match_batch(audio_latent, batch_size).to(video_samples.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video_samples.device)
        elif video is None and reference_audio is not None:
            audio_latent, audio_mask = _encode_reference_audio(
                audio_vae, reference_audio, length / frame_rate)
            audio_latent, audio_mask = _fit_audio_latent(audio_latent, audio_mask, target)
            audio_latent = _match_batch(audio_latent, batch_size).to(video_samples.device)
            audio_mask = _match_batch(audio_mask, batch_size).to(video_samples.device)
        else:
            audio_latent = torch.zeros(target, device=video_samples.device)
            audio_mask = torch.ones_like(audio_latent)

        latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_samples, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
            "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
        }
        logger.info("LTX-2.3 v2v prep: video %s mask %s, audio %s mask %s%s%s",
                    tuple(video_samples.shape), tuple(video_mask.shape),
                    tuple(audio_latent.shape), tuple(audio_mask.shape),
                    ", image held @ %.2f" % image_strength if images is not None else "",
                    ", audio held" if (video is not None and keep_original_audio and video_audio is not None)
                    or (video is None and reference_audio is not None) else "")
        return (model, positive, negative, latent, frame_rate)


class LTXV23KSampler:
    """Euler on the official LTX-2 distilled schedules.

    The stock schedulers (simple/karras/...) do not reproduce the trained
    distilled schedule, and a distilled model on the wrong schedule looks like
    a broken model. This passes the exact sigmas through; 8 steps is the
    trained configuration, other counts resample the same curve.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 KSampler (distilled) ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'txt2img', 'img2img']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Sample a joint AV latent on the official LTX-2 schedules "
                   "(distilled 8-step or refine 3-step, cfg 1.0, euler).")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000,
                                  "tooltip": "8 for dmd/distilled, 4 for dmd upscale, "
                                             "3 for refine."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                  "tooltip": "1.0 - both the DMD and distilled bakes "
                                             "are trained without CFG."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "schedule": (list(SIGMA_SETS), {"default": "dmd (8 steps)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, schedule, denoise):
        samples = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask", None)
        sigmas = distilled_sigma_schedule(
            steps, denoise, sigmas=SIGMA_SETS[schedule]).to(
            device=comfy.model_management.intermediate_device(), dtype=torch.float32)
        noise = comfy.sample.prepare_noise(samples, seed,
                                           latent_image.get("batch_index", None))

        logger.info("LTX-2.3 sample: %s, %d steps from sigma %.4f, mask %s",
                    schedule, len(sigmas) - 1, float(sigmas[0]),
                    "yes" if noise_mask is not None else "NONE")

        out = comfy.sample.sample(
            model, noise, steps=len(sigmas) - 1, cfg=cfg,
            sampler_name=sampler_name, scheduler="simple",
            positive=positive, negative=negative, latent_image=samples,
            sigmas=sigmas, seed=seed, noise_mask=noise_mask,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        latent = latent_image.copy()
        latent["samples"] = out
        latent.pop("noise_mask", None)
        return (latent,)


# ── two-stage refine + decode ───────────────────────────────────────────────

def _upsample_video_latent(latent, upscale_model, vae):
    """Port of core LTXVLatentUpsampler, restricted to the video branch.

    The upsample model only understands a single video-shaped tensor, not a
    joint AV one, so the latent must be split before calling it and rejoined
    after - calling it directly on the concatenated AV tensor does not error,
    it silently treats part of the audio latent as video channels.
    """
    samples = latent["samples"]
    video, audio = samples.unbind()

    device = upscale_model.load_device
    model = upscale_model.model
    model_dtype = upscale_model.model_dtype()
    input_dtype = video.dtype

    memory_required = math.prod(video.shape) * 3000.0  # matches core's estimate
    comfy.model_management.load_models_gpu([upscale_model], memory_required=memory_required)

    video = video.to(dtype=model_dtype, device=device)
    video = vae.first_stage_model.per_channel_statistics.un_normalize(video)
    video = model(video)
    video = vae.first_stage_model.per_channel_statistics.normalize(video)
    video = video.to(dtype=input_dtype, device=comfy.model_management.intermediate_device())
    audio = audio.to(video.device)

    out = latent.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out.pop("noise_mask", None)  # the upsampled latent has no held frames left
    return out


class LTXV23RefineSampler:
    """Base pass -> spatial x2 latent upscale -> refine pass, in one node.

    Reuses LTXV23KSampler for both passes (same schedule/noise semantics,
    verified separately) so there is exactly one place that owns the sigma
    math. The only genuinely new logic is the upscale hop between them: see
    ``_upsample_video_latent`` for why the joint latent has to be split and
    rejoined around the upscale model rather than fed to it directly.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Two-Stage Sampler (base + refine) ⚡"
    SEARCH_ALIASES = ['sampler', 'sample', 'generate', 'denoise', 'diffuse', 'refine latent', 'upscale latent', 'enlarge latent']
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    DESCRIPTION = ("Base sampling pass, spatial x2 latent upscale, then a "
                   "refine pass - the official LTX-2.3 two-stage recipe.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE", {"tooltip": "Video VAE - normalizes the latent "
                                          "around the upscale model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True}),
                "base_schedule": (list(SIGMA_SETS), {"default": "dmd (8 steps)"}),
                "base_steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "refine_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                        "control_after_generate": True}),
                "refine_schedule": (list(SIGMA_SETS), {"default": "refine (3 steps)"}),
                "refine_steps": ("INT", {"default": 3, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            },
        }

    def sample(self, model, positive, negative, latent_image, upscale_model, vae,
               seed, base_schedule, base_steps, refine_seed, refine_schedule,
               refine_steps, cfg, sampler_name):
        sampler = LTXV23KSampler()
        base_latent, = sampler.sample(
            model, positive, negative, latent_image, seed, base_steps, cfg,
            sampler_name, base_schedule, denoise=1.0)

        upscaled = _upsample_video_latent(base_latent, upscale_model, vae)
        logger.info("LTX-2.3 refine: base %s -> upscaled %s",
                    tuple(base_latent["samples"].unbind()[0].shape),
                    tuple(upscaled["samples"].unbind()[0].shape))

        refined, = sampler.sample(
            model, positive, negative, upscaled, refine_seed, refine_steps, cfg,
            sampler_name, refine_schedule, denoise=1.0)
        return (refined,)


class LTXV23CropVideoGuide:
    """Strip the video-guide reference frames LTXV23VidToVideo appended,
    after sampling and before LTXV23AVDecode.

    A thin wrapper around comfy-core's own LTXVCropGuides (comfy_extras/
    nodes_lt.py's get_keyframe_idxs) applied to just the video half of the
    joint AV latent - core's version only understands a plain video latent,
    not this pack's NestedTensor((video, audio)) convention, so the two
    streams get split before the crop and rejoined after (same reason
    _upsample_video_latent above has to do this for the refine sampler's
    upscale model). A no-op (returns the latent unchanged) when no guide was
    ever appended (ic_lora=none upstream, or nothing to crop) -
    matches core's own early-return behavior.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Crop Video Guide ⚡"
    SEARCH_ALIASES = ['crop guide', 'remove guide', 'strip reference', 'ic-lora',
                      'vid2vid', 'v2v']
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "crop"
    DESCRIPTION = ("Strip the video-guide reference frames LTXV23VidToVideo "
                   "appended, after sampling. No-op if none were appended.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
            },
        }

    def crop(self, positive, negative, latent):
        import comfy_extras.nodes_lt as nodes_lt

        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("LTX-2.3 Crop Video Guide: latent isn't a joint AV "
                             "latent - expected LTXV23KSampler's output.")
        video, audio = samples.unbind()

        video_mask = audio_mask = None
        noise_mask = latent.get("noise_mask")
        if noise_mask is not None:
            video_mask, audio_mask = noise_mask.unbind()

        _, num_keyframes = nodes_lt.get_keyframe_idxs(positive, video.shape)
        if num_keyframes == 0:
            return (positive, negative, latent)

        cropped_video = video[:, :, :-num_keyframes]
        cropped_mask = video_mask[:, :, :-num_keyframes] if video_mask is not None else None

        positive = node_helpers.conditioning_set_values(
            positive, {"keyframe_idxs": None, "guide_attention_entries": None})
        negative = node_helpers.conditioning_set_values(
            negative, {"keyframe_idxs": None, "guide_attention_entries": None})

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((cropped_video, audio))
        if cropped_mask is not None and audio_mask is not None:
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((cropped_mask, audio_mask))
        else:
            out.pop("noise_mask", None)

        logger.info("LTX-2.3 crop guide: removed %d guide frame(s), video %s -> %s",
                    num_keyframes, tuple(video.shape), tuple(cropped_video.shape))
        return (positive, negative, out)


class LTXV23AVDecode:
    """Joint AV latent -> muxed VIDEO, in one node.

    Wraps core VAEDecodeTiled (video) + LTXVAudioVAEDecode (audio) +
    CreateVideo (mux), threading one ``fps`` through all three - the
    plain-node version needs the same value typed into two different widgets
    that have no wire between them, and a mismatch there is a silent
    audio/video drift, not an error.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 AV Decode ⚡"
    SEARCH_ALIASES = ['decode', 'decode latent', 'latent to video', 'latent to audio', 'video decode', 'audio decode']
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "decode"
    DESCRIPTION = "Decode a joint AV latent to a muxed VIDEO output."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE", {"tooltip": "Video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE."}),
                "fps": ("FLOAT", {"default": FPS, "min": 1.0, "max": 120.0, "step": 0.01}),
            },
            "optional": {
                "tile_size": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 96, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 16, "min": 4, "max": 4096, "step": 4}),
            },
        }

    def decode(self, latent, vae, audio_vae, fps, tile_size=768, overlap=64,
               temporal_size=96, temporal_overlap=16):
        if _VideoInputImpl is None:
            raise RuntimeError(
                "This ComfyUI is too old to expose comfy_api.latest's VIDEO "
                "type; update ComfyUI or wire VAEDecodeTiled + "
                "LTXVAudioVAEDecode + CreateVideo by hand.")

        images, = nodes.VAEDecodeTiled().decode(
            vae, latent, tile_size, overlap, temporal_size, temporal_overlap)

        samples = latent["samples"]
        audio_latent = samples.unbind()[-1] if samples.is_nested else samples
        waveform = audio_vae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
        sample_rate = int(audio_vae.first_stage_model.output_sample_rate)
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        video = _VideoInputImpl.VideoFromComponents(
            _VideoTypes.VideoComponents(images=images, audio=audio,
                                        frame_rate=Fraction(fps)))
        logger.info("LTX-2.3 AV decode: %s frames @ %.2f fps, audio %s @ %d Hz",
                    tuple(images.shape), fps, tuple(waveform.shape), sample_rate)
        return (video,)


# ── ID-LoRA prompt editing ──────────────────────────────────────────────────

# Lazily matches each [TAG]: body up to the next [TAG]: or end of string, so
# it works whether SOUNDS (the last tag) or a middle tag like SPEECH is being
# pulled out - a greedy `(.*)$` makes SPEECH swallow SOUNDS too. Tolerant of
# a Part-1 spoken-script preamble before the tags (the captioner's raw
# two-part output) since it anchors on the tags themselves, not on position.
_ID_LORA_TAG_RE = re.compile(
    r"\[(VISUAL|SPEECH|SOUNDS)\]:\s*(.*?)\s*(?=\[(?:VISUAL|SPEECH|SOUNDS)\]:|$)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_id_lora_prompt(source):
    fields = {"VISUAL": "", "SPEECH": "", "SOUNDS": ""}
    for tag, body in _ID_LORA_TAG_RE.findall(source or ""):
        fields[tag.upper()] = body.strip()
    return fields["VISUAL"], fields["SPEECH"], fields["SOUNDS"]


# node unique_id -> the source text that was last parsed for it. Lets
# assemble() tell "the boxes hold the user's edit" from "the boxes hold
# stale auto-fill" without guessing from widget content (undecidable - see
# the class docstring) or making the user flip a switch.
_LAST_SOURCE = {}


class LTXV23IDLoraPromptEditor:
    """Show a captioner's [VISUAL]/[SPEECH]/[SOUNDS] in three editable
    boxes, and reassemble them into an ID-LoRA prompt string.

    ``source`` is whatever the upstream captioner (e.g. LMStudioVisionPrompt)
    produced - its full two-part output (spoken script, ``---``, tagged
    block) or just the tagged block; the parser only looks for the three
    ``[TAG]:`` markers, ignoring a Part-1 preamble. It is ``forceInput`` so
    it renders as a socket only: a widget-backed input that is wired greys
    out, blanks its own text and stops accepting clicks
    (comfy frontend ``LGraphNode.updateComputedDisabled``), so a visible box
    there would be dead weight.

    The three boxes auto-fill with the parsed values after a run and stay
    directly editable; an edit survives later runs, and a genuinely new
    ``source`` refreshes them. That combination does not exist as a widget
    flag in comfy-core - its only populate-from-own-execution widget,
    TEXT_PREVIEW (PreviewAny/SaveText), is hard-coded read-only and
    ``serialize: False`` - so the state lives here instead:

      * this run's ``source`` differs from the one remembered for this node
        -> the boxes are stale, re-parse and overwrite them;
      * same ``source`` as last run -> the boxes are the truth, keep them
        (this is where a manual edit survives);
      * nothing remembered yet (fresh ComfyUI start) -> fill only empty
        boxes, so edits saved into the workflow survive a restart.

    The paired ``web/`` JS then writes the resolved values back into the
    widgets unconditionally. That is safe precisely because the decision
    already happened here: it either echoes the user's own edit back or
    shows the fresh parse. Earlier versions put that decision in the JS
    ("fill only if the widget is empty"), which cannot work - once
    auto-filled a widget is non-empty forever, so it filled once and then
    appeared frozen.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 ID-LoRA Prompt Editor ⚡"
    SEARCH_ALIASES = ['edit prompt', 'prompt editor', 'visual speech sounds', 'id lora prompt', 'prompt writer']
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("tagged_prompt", "visual_text", "speech_text", "sounds_text",
                    "speech_text_batch")
    OUTPUT_IS_LIST = (False, False, False, False, True)
    FUNCTION = "assemble"
    OUTPUT_NODE = True          # required for the ui payload to reach the frontend
    DESCRIPTION = ("Edit a captioner's [VISUAL]/[SPEECH]/[SOUNDS] fields and "
                   "reassemble them into an ID-LoRA prompt.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("STRING", {"forceInput": True,
                    "tooltip": "The captioner's raw output - its full "
                               "two-part text, or just the tagged block."}),
                "visual": ("STRING", {"multiline": True, "default": ""}),
                "speech": ("STRING", {"multiline": True, "default": ""}),
                "sounds": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def assemble(self, source, visual, speech, sounds, unique_id=None):
        previous = _LAST_SOURCE.get(unique_id)
        if previous is None:
            if not (visual.strip() or speech.strip() or sounds.strip()):
                visual, speech, sounds = _parse_id_lora_prompt(source)
        elif previous != source:
            visual, speech, sounds = _parse_id_lora_prompt(source)
        _LAST_SOURCE[unique_id] = source

        tagged_prompt = f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}"
        # One clip per non-blank line. A blank line is treated as a separator,
        # not an empty clip - "line one\n\nline two" is 2 clips, not 3.
        speech_batch = [line.strip() for line in speech.splitlines() if line.strip()]
        if not speech_batch:
            speech_batch = [""]  # never return an empty list - nothing to index

        return {
            "ui": {"visual": [visual], "speech": [speech], "sounds": [sounds]},
            "result": (tagged_prompt, visual, speech, sounds, speech_batch),
        }


class LTXV23SpeechBatchSelector:
    """Pick one clip out of a speech_text_batch list by index.

    ``batch`` must arrive as the whole list in one call, not fanned out one
    call per item - INPUT_IS_LIST=True (comfy's execution.py:245,
    _async_map_node_over_list) does exactly that, at the cost of every input
    (including ``index``) arriving wrapped in a length-1 list, unwrapped
    below.

    ``count`` is the batch's total length - wire it into a loop node's
    iteration-count input to drive a for-each over the batch (e.g. bump
    ``index`` from 0 to ``count - 1`` across iterations).
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 Speech Batch Selector ⚡"
    SEARCH_ALIASES = ['select from batch', 'pick clip', 'batch subset', 'index selector', 'choose from list']
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_text", "count")
    FUNCTION = "select"
    INPUT_IS_LIST = True
    DESCRIPTION = "Pick one clip from a speech_text_batch list by index."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch": ("STRING", {"forceInput": True,
                    "tooltip": "A speech_text_batch list, e.g. from the "
                               "ID-LoRA Prompt Editor."}),
                "index": ("INT", {"default": 0, "min": -0xffffffff, "max": 0xffffffff,
                    "tooltip": "0-based; negative counts from the end like "
                               "Python (-1 = last clip). Out-of-range clamps "
                               "to the nearest valid index."}),
            },
        }

    def select(self, batch, index):
        index = int(index[0])
        n = len(batch)
        clamped = max(-n, min(index, n - 1))
        if clamped != index:
            logger.warning("LTX-2.3 Speech Batch Selector: index %d out of "
                           "range for %d clip(s), clamped to %d", index, n, clamped)
        return (batch[clamped], n)


class LTXV23IDLoraAssembler:
    """Merge three separate visual/speech/sounds strings into the canonical
    ID-LoRA prompt string.

    The counterpart to ``LTXV23IDLoraPromptEditor``: that node parses ONE
    combined source string apart into three editable fields; this node goes
    the other direction, combining three ALREADY-SEPARATE strings (e.g. one
    edited by hand, one picked from ``LTXV23SpeechBatchSelector``, one from
    somewhere else entirely) into one formatted string. No source input, no
    parsing, no state - a pure formatter, same job
    ``LTXV23IDLoraPromptEditor.assemble()`` does internally, exposed on its
    own for pipelines that already have the three pieces from elsewhere.
    """

    CATEGORY = LTX23_CATEGORY
    TITLE = "LTX-2.3 ID-LoRA Assembler ⚡"
    SEARCH_ALIASES = ["merge prompt", "combine prompt", "format prompt",
                      "assemble prompt", "id lora prompt", "visual speech sounds"]
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tagged_prompt",)
    FUNCTION = "assemble"
    DESCRIPTION = "Combine separate visual/speech/sounds strings into the ID-LoRA prompt format."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "visual": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [VISUAL] section text."}),
                "speech": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [SPEECH] section text."}),
                "sounds": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "The [SOUNDS] section text."}),
            },
        }

    def assemble(self, visual, speech, sounds):
        return (f"[VISUAL]: {visual}\n[SPEECH]: {speech}\n[SOUNDS]: {sounds}",)


NODE_CLASS_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader,
    "LTXV23EditAnythingPatch": LTXV23EditAnythingPatch,
    "LTXV23ImgToVideo": LTXV23ImgToVideo,
    "LTXV23VidToVideo": LTXV23VidToVideo,
    "LTXV23KSampler": LTXV23KSampler,
    "LTXV23RefineSampler": LTXV23RefineSampler,
    "LTXV23CropVideoGuide": LTXV23CropVideoGuide,
    "LTXV23AVDecode": LTXV23AVDecode,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor,
    "LTXV23SpeechBatchSelector": LTXV23SpeechBatchSelector,
    "LTXV23IDLoraAssembler": LTXV23IDLoraAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV23ModelsLoader": LTXV23ModelsLoader.TITLE,
    "LTXV23EditAnythingPatch": LTXV23EditAnythingPatch.TITLE,
    "LTXV23ImgToVideo": LTXV23ImgToVideo.TITLE,
    "LTXV23VidToVideo": LTXV23VidToVideo.TITLE,
    "LTXV23KSampler": LTXV23KSampler.TITLE,
    "LTXV23RefineSampler": LTXV23RefineSampler.TITLE,
    "LTXV23CropVideoGuide": LTXV23CropVideoGuide.TITLE,
    "LTXV23AVDecode": LTXV23AVDecode.TITLE,
    "LTXV23IDLoraPromptEditor": LTXV23IDLoraPromptEditor.TITLE,
    "LTXV23SpeechBatchSelector": LTXV23SpeechBatchSelector.TITLE,
    "LTXV23IDLoraAssembler": LTXV23IDLoraAssembler.TITLE,
}
