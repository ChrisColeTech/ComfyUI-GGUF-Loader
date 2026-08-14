"""Smoke test: Scenema node helpers under real comfy (CPU).

Run: python tools/smoke_scenema.py [--models-dir D:/models/image-models/scenema-audio]

Exercises the load paths ported in nodes_scenema.py against real checkpoints:
  1. transformer: int8 expansion + key normalization + connector merge +
     detection pad + comfy LTXAV detection
  2. VAE: AudioVAE construction from the pipeline checkpoint
  3. text encoder: gemma GGUF -> LTXAVTEModel with dual projection (optional;
     slow, needs sentencepiece+protobuf)
No GPU or sampling.
"""
import argparse
import logging
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir",
                        default=r"D:\models\image-models\scenema-audio")
    parser.add_argument("--skip-te", action="store_true",
                        help="skip the (slow) Gemma GGUF text encoder load")
    args = parser.parse_args()

    # CPU-only; paths must be set before comfy imports
    sys.argv = [sys.argv[0], "--cpu"]
    from pathlib import Path
    comfy_root = Path(__file__).resolve().parents[2] / "ComfyUI" / "upstream" / "ComfyUI"
    if not (comfy_root / "comfy").is_dir():
        comfy_root = Path(__file__).resolve().parents[2] / "ComfyUI" / "ComfyUI-master"
    sys.path.insert(0, str(comfy_root))
    import comfy.options
    comfy.options.args_parsing = True

    import torch  # noqa: F401
    logging.basicConfig(level=logging.INFO)

    import comfy.sd
    import comfy.utils

    import types
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("cctech_gguf_pkg")
    pkg.__path__ = [str(root)]
    sys.modules["cctech_gguf_pkg"] = pkg
    ns = __import__("importlib").import_module("cctech_gguf_pkg.nodes_scenema")

    base = args.models_dir
    t_path = base + r"\split\diffusion_models\scenema-audio-transformer-int8.safetensors"
    p_path = base + r"\split\vae\scenema-audio-pipeline.safetensors"

    # ── transformer ──
    t_sd, t_meta, _ = ns._load_file_sd(t_path)
    t_sd = ns._expand_int8(t_sd)
    t_sd = ns._normalize_transformer_keys(t_sd)
    assert "model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight" in t_sd

    p_sd, p_meta, _ = ns._load_file_sd(p_path)
    merged = 0
    for k, v in p_sd.items():
        if k.startswith("model.diffusion_model.") and k not in t_sd:
            t_sd[k] = v
            merged += 1
    metadata = t_meta if t_meta and "config" in t_meta else p_meta
    t_sd = ns._pad_detection_key(t_sd, metadata)

    model = comfy.sd.load_diffusion_model_state_dict(t_sd, metadata=metadata)
    assert model.model.model_config.unet_config.get("image_model") == "ltxav"
    assert model.model.model_config.unet_config.get("num_layers") == 48
    print("[ok] transformer loads as comfy LTXAV (48 layers)")
    del t_sd, model

    # ── VAE ──
    vae_sd = {k: v for k, v in p_sd.items() if k.startswith(("audio_vae.", "vocoder."))}
    vae = comfy.sd.VAE(sd=vae_sd, metadata=p_meta)
    fsm = vae.first_stage_model
    assert (fsm.latent_channels, fsm.latent_frequency_bins) == (8, 16)
    assert fsm.sample_rate == 16000 and fsm.output_sample_rate == 48000
    assert fsm.num_of_latents_from_frames(10 * 24 + 1, 24) == 251
    print("[ok] VAE is AudioVAE (16k in / 48k BWE out)")
    del vae_sd, vae

    # ── text encoder ──
    if not args.skip_te:
        from cctech_gguf_pkg.loader import gguf_clip_loader
        from cctech_gguf_pkg.ops import GGMLOps
        import comfy.model_management
        import glob
        ggufs = sorted(glob.glob(base + r"\split\text_encoders\gemma*.gguf"))
        assert ggufs, "no gemma gguf found"
        te_sd = gguf_clip_loader(ggufs[0])
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.LTXV,
            state_dicts=[te_sd, p_sd],
            model_options={
                "custom_operations": GGMLOps(),
                "initial_device": comfy.model_management.text_encoder_offload_device(),
            },
        )
        csm = clip.cond_stage_model
        assert type(getattr(csm, "text_embedding_projection", None)).__name__ == "DualLinearProjection"
        assert len(clip.tokenize("hello world")["gemma3_12b"][0]) == 1024
        print("[ok] Gemma GGUF -> LTXAV CLIP (dual projection, tokenizer rebuilt)")

    print("SCENEMA SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
