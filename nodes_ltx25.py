# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Empty joint AV latent for LTX-2.5.

Comfy core ships ``EmptyLTXVLatentVideo`` (video stream only) and the
``LTXVConcatAVLatent`` / ``LTXVSeparateAVLatent`` pair, but nothing that hands
the LTX-2.5 AV DiT a ready-made video+audio latent. The gap is easy to fill with
the wrong thing: the MiniMax H3 empty-AV node produces a nested latent of the
same *type*, so it wires up cleanly and then dies inside ``patchify_proj`` — H3
video is 24 channels on a /16 spatial grid, LTX-2.5 wants 128 on /32, and the
audio streams disagree on both rank and layout.

This node builds the LTX-2.5 geometry directly:

    video  [B, 128, (length - 1) // 8 + 1, height // 32, width // 32]
    audio  [B, z_channels, n_latents, frequency_bins]

The audio side is read off the audio VAE rather than hardcoded, because
``n_latents`` depends on the clip duration and on the VAE's mel hop
(``sample_rate / mel_hop_length / 8`` latents per second) — the same derivation
``LTXVReferenceAudio`` relies on when it encodes real audio.
"""

import logging

import nodes
import torch

import comfy.model_management
import comfy.nested_tensor

logger = logging.getLogger(__name__)

LTX25_CATEGORY = "🤖 CCTech/LTX-2.5"

# The video VAE's compression: /32 in space, /8 in time with a causal first frame.
VIDEO_LATENT_CHANNELS = 128
VIDEO_SPATIAL_RATIO = 32
VIDEO_TEMPORAL_RATIO = 8


def video_latent_t(length):
    """Latent frame count for ``length`` pixel frames (first frame is causal)."""
    return ((max(1, int(length)) - 1) // VIDEO_TEMPORAL_RATIO) + 1


def audio_latent_shape(audio_vae, length, frame_rate, batch_size):
    """Read the audio stream's geometry off the LTX-2.5 audio VAE.

    Returns ``(shape, latents_per_second)``. Raises if the VAE is not an LTX
    audio VAE, which is the common miswiring (the *video* VAE plugged into the
    audio slot).
    """
    model = getattr(audio_vae, "first_stage_model", None)
    required = ("num_of_latents_from_frames", "latent_frequency_bins")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "audio_vae is not an LTX-2.5 audio VAE (missing {}); load "
            "ltx-2.5-audio-vae-bf16.safetensors with the stock VAE Loader, not "
            "the video VAE.".format(", ".join(missing))
        )

    n_latents = int(model.num_of_latents_from_frames(int(length), float(frame_rate)))
    if n_latents < 1:
        raise ValueError(
            "length {} at {} fps is too short to produce an audio latent "
            "(need at least {:.3f}s).".format(
                length, frame_rate, 1.0 / float(model.latents_per_second))
        )

    channels = int(getattr(audio_vae, "latent_channels", model.latent_channels))
    bins = int(model.latent_frequency_bins)
    return ([int(batch_size), channels, n_latents, bins],
            float(model.latents_per_second))


def empty_av_latent(audio_vae, width, height, length, frame_rate, batch_size=1):
    """Build the nested (video, audio) latent the LTX-2.5 AV DiT samples."""
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [int(batch_size), VIDEO_LATENT_CHANNELS, video_latent_t(length),
         int(height) // VIDEO_SPATIAL_RATIO, int(width) // VIDEO_SPATIAL_RATIO],
        device=device)
    shape, _ = audio_latent_shape(audio_vae, length, frame_rate, batch_size)
    audio = torch.zeros(shape, device=device)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "downscale_ratio_spacial": VIDEO_SPATIAL_RATIO,
    }


class LTXV25EmptyLatentAVBatch:
    """Empty video+audio latent shaped for the LTX-2.5 AV transformer.

    Feeding this to the sampler instead of the MiniMax H3 empty-AV node is the
    difference between a clip and a channel-mismatch traceback: the two models
    share the nested-latent container but not a single dimension of it.
    """

    CATEGORY = LTX25_CATEGORY
    TITLE = "Empty LTX-2.5 AV Latent (Batch) ⚡"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_vae": ("VAE", {
                    "tooltip": "ltx-2.5-audio-vae-bf16.safetensors. Only its "
                               "geometry is read here (latent channels, "
                               "frequency bins, latents per second) — no "
                               "encoding happens, so it costs nothing."}),
                "width": ("INT", {"default": 768, "min": 64,
                                  "max": nodes.MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64,
                                   "max": nodes.MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {
                    "default": 97, "min": 1, "max": nodes.MAX_RESOLUTION,
                    "step": 8,
                    "tooltip": "Frame count. The video VAE compresses 8:1 in "
                               "time with a causal first frame, so 8k+1 values "
                               "(9, 97, 121...) tile exactly."}),
                "frame_rate": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Sets the clip duration, which sets the audio "
                               "latent length. Use the same value on LTXV "
                               "Conditioning or the two streams drift apart."}),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 64,
                    "tooltip": "Clips per job, each with its own noise from the "
                               "sampler seed. The LTX-2.5 DiT is batch-aware, "
                               "so unlike MiniMax H3 no patch node is needed — "
                               "but VRAM scales with the batch."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    DESCRIPTION = ("Joint video+audio latent in LTX-2.5 geometry: video "
                   "[B,128,(len-1)//8+1,H/32,W/32] and audio "
                   "[B,z,n_latents,bins] read from the audio VAE. Use this, not "
                   "the MiniMax H3 empty-AV node, to drive an LTX-2.5 sampler.")

    def generate(self, audio_vae, width, height, length, frame_rate, batch_size):
        latent = empty_av_latent(audio_vae, width, height, length, frame_rate,
                                 batch_size)
        video, audio = latent["samples"].unbind()
        logger.info("LTX-2.5 empty AV latent: video %s, audio %s (%d frames @ "
                    "%.2f fps = %.2fs)", tuple(video.shape), tuple(audio.shape),
                    length, frame_rate, length / float(frame_rate))
        return (latent,)


NODE_CLASS_MAPPINGS = {
    "LTXV25EmptyLatentAVBatch": LTXV25EmptyLatentAVBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXV25EmptyLatentAVBatch": LTXV25EmptyLatentAVBatch.TITLE,
}
