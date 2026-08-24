# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Convenience loaders that pair with the GGUF nodes.

Two nodes live here, both born out of the MiniMax-H3 pipeline:

DualVAELoader     H3 decodes video and audio through two separate VAEs, so every
                  workflow needs two VAELoader nodes side by side. This is that
                  pair in one node, with the outputs named for their role.

ClipProjLoader    load a text encoder and project it into a larger one's space,
                  in a single node, reading GGUF as well as safetensors. The
                  projection itself is in clipproj.py, ported from
                  ComfyUI-ClipProj (MIT, nicolab28).
"""

import logging

import nodes
import comfy.sd
import folder_paths

from .nodes import CLIPLoaderGGUF
from . import clipproj

# The projection folder has to exist before INPUT_TYPES first reads it, or the
# combo comes up with the controls alone on a fresh install.
clipproj.register_folder()

# The only families this projects from. ClipProj's tokenisation is MiniMax-H3's
# and its tap reads a Qwen3-VL hidden state, so offering the full CLIPType list
# would only invite picking one that cannot work.
QWEN3VL_TYPES = ("krea2", "boogu", "minimax")

# Size -> the matrix that goes with it, for error messages.
MATRIX_FOR = {"krea2": "mmh3-4b-*", "boogu": "mmh3-8b-*", "minimax": "mmh3-32b-*"}


class DualVAELoader:
    """Load two VAEs at once, named for the streams they decode."""

    @classmethod
    def INPUT_TYPES(s):
        # Reuse the stock loader's list so taesd and any other special entry it
        # synthesises stay available here too.
        vae_list = nodes.VAELoader.INPUT_TYPES()["required"]["vae_name"]
        return {
            "required": {
                "video_vae_name": vae_list,
                "audio_vae_name": vae_list,
            }
        }

    RETURN_TYPES = ("VAE", "VAE")
    RETURN_NAMES = ("video_vae", "audio_vae")
    FUNCTION = "load_vaes"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "Dual VAE Loader (video + audio) ⚡"
    SEARCH_ALIASES = ['load vae', 'dual vae', 'video vae', 'audio vae', 'vae loader']
    DESCRIPTION = ("Load a video VAE and an audio VAE in one node. For MiniMax "
                   "H3: minimax_h3_video_vae_* on the first, "
                   "minimax_h3_audio_vae_* on the second.")

    def load_vaes(self, video_vae_name, audio_vae_name):
        """Delegate to the stock loader so both VAEs load exactly as usual."""
        loader = nodes.VAELoader()
        video_vae = loader.load_vae(video_vae_name)[0]
        # Loaded separately even when both names match: a VAE carries per-node
        # state once patched, and handing the same object to two decoders has
        # bitten people before.
        audio_vae = loader.load_vae(audio_vae_name)[0]
        return (video_vae, audio_vae)


class ClipProjLoader(CLIPLoaderGGUF):
    """Load a text encoder (GGUF or safetensors) and project it, in one node."""

    @classmethod
    def INPUT_TYPES(s):
        stock = list(nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
        types = ["auto"] + [t for t in stock if t in QWEN3VL_TYPES]
        return {
            "required": {
                "clip_name": (s.get_filename_list(), {
                    "tooltip": "A full Qwen3-VL text encoder, .gguf or "
                               ".safetensors. Not an mmproj file, which is the "
                               "vision projector alone and carries no text "
                               "model. It must be a VL model: a text-only Qwen3 "
                               "of the same size loads without complaint and "
                               "produces conditioning that ignores your prompt."}),
                "type": (types, {
                    "default": "auto",
                    "tooltip": "auto reads the file header and picks the "
                               "architecture. Override only if that fails: "
                               "krea2 = 4B, boogu = 8B, minimax = 32B."}),
                "projection": (clipproj.list_projections(), {
                    "tooltip": "Learned matrix from models/clip_projections/, or "
                               "a <control:...> baseline. mmh3-4b-* goes with a "
                               "4B, mmh3-8b-* with an 8B; they are not "
                               "interchangeable. Run the controls first: they "
                               "show what the diffusion model does on its own."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_projected"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "Text Encoder + ClipProj Loader ⚡"
    SEARCH_ALIASES = ['load clip', 'text encoder', 'clip loader', 'clip projection', 'clipproj']
    DESCRIPTION = ("Load a small text encoder and project it into a large one's "
                   "space, so a 4B or 8B Qwen3-VL can stand in for the 32B "
                   "MiniMax H3 expects. Matrices go in models/clip_projections/.")

    def resolve_type(self, clip_path, clip_name, requested):
        """Settle the architecture, checking the file when it can be read.

        Detection runs even when the type is given by hand: it looks for the
        vision tower, so it tells a Qwen3-VL from an ordinary Qwen3 -- which the
        hidden width does not, both families share 2560 and 4096. It also
        catches a file that is not a text encoder at all, before a load that can
        be ten gigabytes.
        """
        found = clipproj.detect_arch(clip_path)
        if found is None:
            if requested == "auto":
                raise ValueError(
                    "Could not identify %s as a Qwen3-VL text encoder, so there "
                    "would be nothing for the projection to read.\n"
                    "  - an mmproj file is the vision projector on its own, a "
                    "few hundred MB against several GB, and carries no text "
                    "model at all. That is the usual mistake.\n"
                    "  - a text-only Qwen3 has no vision tower either; it would "
                    "load and then ignore your prompt.\n"
                    "Set 'type' by hand to load it anyway: krea2 = 4B, "
                    "boogu = 8B, minimax = 32B." % clip_name)
            logging.warning(
                "[ClipProj] no vision tower found in %s. If it is a text-only "
                "Qwen3 rather than a Qwen3-VL, the projection will load without "
                "complaint and produce conditioning that ignores your prompt.",
                clip_name)
            return requested
        detected, label = found
        if requested == "auto":
            logging.info("[ClipProj] %s detected as a %s", clip_name, label)
            return detected
        if requested != detected:
            raise ValueError(
                "%s is a %s, but 'type' is set to %s. Use auto, or %s. The "
                "matrix has to match as well: %s."
                % (clip_name, label, requested, detected,
                   MATRIX_FOR.get(detected, "the matching mmh3-* file")))
        return requested

    def load_clip_projected(self, clip_name, type, projection):
        """Load through the GGUF or the stock path, then wrap in the projection."""
        clip_path = folder_paths.get_full_path("clip", clip_name)
        if clip_path is None:
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)

        resolved = self.resolve_type(clip_path, clip_name, type)
        clip_type = getattr(comfy.sd.CLIPType, resolved.upper(),
                            comfy.sd.CLIPType.STABLE_DIFFUSION)

        # Drop the previous run's device-side projection before the new encoder
        # lands, not after: ComfyUI only replaces this node's output once the
        # load has finished, so otherwise both sets sit on the card at the
        # moment it is most loaded.
        clipproj.purge_projection_caches()

        if clip_path.lower().endswith(".gguf"):
            clip = self.load_patcher([clip_path], clip_type, self.load_data([clip_path]))
        else:
            # Stock path for safetensors. Deliberately NOT the GGUF one: it
            # refuses scaled-fp8 checkpoints, and qwen3vl_*_fp8_scaled is the
            # encoder most people running this actually have.
            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
            )

        # Last guard, on the built model rather than the file. ComfyUI falls back
        # to CLIP-L when it cannot identify a state dict: it logs every key as
        # missing, returns a randomly initialised model, and raises nothing.
        try:
            sub = clipproj.submodel(clip)
        except Exception:
            sub = None
        tr = getattr(sub, "transformer", None)
        if tr is None or not hasattr(tr, "preprocess_embed"):
            # tr.__class__ rather than type(tr): `type` is this function's
            # parameter name, matching the widget.
            built = tr.__class__.__name__ if tr is not None else "no text model"
            raise ValueError(
                "%s loaded as %s, not as a Qwen3-VL text encoder, so the "
                "projection has nothing to read. Check that the file is a full "
                "text encoder and that 'type' matches its size."
                % (clip_name, built))

        return (clipproj.wrap(clip, projection),)


NODE_CLASS_MAPPINGS = {
    "CCTechDualVAELoader": DualVAELoader,
    "CCTechClipProjLoader": ClipProjLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCTechDualVAELoader": DualVAELoader.TITLE,
    "CCTechClipProjLoader": ClipProjLoader.TITLE,
}
