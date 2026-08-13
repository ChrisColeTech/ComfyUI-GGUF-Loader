# (c) CCTech || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Convenience loaders that pair with the GGUF nodes.

Two nodes live here, both born out of the MiniMax-H3 pipeline:

DualVAELoader     H3 decodes video and audio through two separate VAEs, so every
                  workflow needs two VAELoader nodes side by side. This is that
                  pair in one node, with the outputs named for their role.

ClipProjLoader    "load a text encoder and project it" in a single node, with
                  the GGUF path wired in. ComfyUI-ClipProj's own all-in-one
                  loader goes through comfy.sd.load_clip, which cannot read a
                  .gguf; this one routes .gguf files through the GGUF loader and
                  everything else through the stock path, then applies the
                  projection to whichever came back.

The ClipProj node is optional: it needs the ComfyUI-ClipProj pack installed and
degrades to an explicit error, not an import failure, when it is absent.
"""

import sys
import logging

import nodes
import comfy.sd
import folder_paths

from .nodes import CLIPLoaderGGUF

# Shown in the projection combo when the ClipProj pack is not installed. Picking
# it raises with the URL rather than failing on a missing attribute.
CLIPPROJ_MISSING = "<ComfyUI-ClipProj not installed>"

CLIPPROJ_URL = "https://github.com/nicolab28/ComfyUI-ClipProj"

_CLIPPROJ = {}


def clipproj_module():
    """Return ComfyUI-ClipProj's node module, or None when it is not installed.

    Resolved through sys.modules rather than by importing a path, so the pack is
    found whatever folder name it was cloned under and, more importantly, so we
    share its module state: the projection cache and the pinning registry are
    module-level, and a second copy loaded from file would quietly hold its own.
    """
    mod = _CLIPPROJ.get("mod")
    if mod is not None:
        return mod
    for name, m in list(sys.modules.items()):
        if m is None or not name.endswith("clipproj_nodes"):
            continue
        if hasattr(m, "_wrap") and hasattr(m, "list_projections"):
            _CLIPPROJ["mod"] = m
            return m
    return None


def list_projections():
    """Projection matrices offered by the ClipProj pack, or the missing marker."""
    mod = clipproj_module()
    if mod is None:
        return [CLIPPROJ_MISSING]
    try:
        return list(mod.list_projections())
    except Exception as e:
        logging.warning("[CCTech] could not list ClipProj projections: %s", e)
        return [CLIPPROJ_MISSING]


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
        types = list(nodes.CLIPLoader.INPUT_TYPES()["required"]["type"][0])
        return {
            "required": {
                "clip_name": (s.get_filename_list(), {
                    "tooltip": "Small encoder: a Qwen3-VL-4B or -8B, .gguf or "
                               ".safetensors. It must be a VL model — a "
                               "text-only Qwen3 of the same size loads without "
                               "complaint and ignores your prompt."}),
                "type": (types, {
                    "default": "krea2" if "krea2" in types else types[0],
                    "tooltip": "krea2 = 4B, boogu = 8B, minimax = 32B."}),
                "projection": (list_projections(), {
                    "tooltip": "Learned matrix from models/clip_projections/, or "
                               "a <control:...> baseline. mmh3-4b-* goes with a "
                               "4B, mmh3-8b-* with an 8B; they are not "
                               "interchangeable."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip_projected"
    CATEGORY = "🤖 CCTech/GGUF"
    TITLE = "Text Encoder + ClipProj Loader ⚡"
    DESCRIPTION = ("Load a small text encoder and project it into the large "
                   "one's space. Reads GGUF encoders, which ComfyUI-ClipProj's "
                   "own loader cannot. Requires ComfyUI-ClipProj for the "
                   "projection itself.")

    def load_clip_projected(self, clip_name, type, projection):
        """Load through the GGUF or the stock path, then wrap in the projection."""
        mod = clipproj_module()
        if mod is None:
            raise RuntimeError(
                "This node needs the ComfyUI-ClipProj pack for the projection "
                "itself — it is not installed. Clone it into custom_nodes and "
                "restart:\n    git clone %s\nThe matrices go in "
                "models/clip_projections/." % CLIPPROJ_URL)
        if projection == CLIPPROJ_MISSING:
            raise ValueError(
                "No projection selected. Put a matrix in "
                "models/clip_projections/ (see %s) and pick it here." % CLIPPROJ_URL)

        clip_path = folder_paths.get_full_path("clip", clip_name)
        if clip_path is None:
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

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
        return (mod._wrap(clip, projection),)


NODE_CLASS_MAPPINGS = {
    "CCTechDualVAELoader": DualVAELoader,
    "CCTechClipProjLoader": ClipProjLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CCTechDualVAELoader": DualVAELoader.TITLE,
    "CCTechClipProjLoader": ClipProjLoader.TITLE,
}
