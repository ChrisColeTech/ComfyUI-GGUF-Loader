"""Smoke test: LTX-2.3 kit nodes under real comfy (CPU).

Run with the portable python:
  P:/Projects/ComfyUI_windows_portable_nvidia/ComfyUI_windows_portable/
    python_embeded/python.exe tools/smoke_ltx23.py

Exercises nodes_ltx23.py against the real 10Eros kit:
  1. LTXV23ModelsLoader: Q4 GGUF DiT + Gemma-3 GGUF TE + projections +
     both VAEs -> MODEL (ltxav/48), CLIP (DualLinearProjection), VAEs
  2. same loader with the fp8mixed safetensors DiT (TE cached)
  3. Q6_K / Q8_0 through the full UnetLoaderGGUF path
  4. LTXV23EmptyLatentAV geometry (video + audio read off the audio VAE)
  5. the distilled sigma schedule helper
No GPU or sampling (a 22B forward is not a CPU job).
"""
import sys


def main():
    # CPU-only; paths must be set before comfy imports
    sys.argv = [sys.argv[0], "--cpu"]
    from pathlib import Path
    comfy_root = Path(r"P:\Projects\ComfyUI_windows_portable_nvidia"
                      r"\ComfyUI_windows_portable\ComfyUI")
    sys.path.insert(0, str(comfy_root))
    import comfy.options
    comfy.options.args_parsing = True

    import logging
    logging.basicConfig(level=logging.WARNING)

    import torch  # noqa: F401
    import folder_paths

    # point comfy's model folders at the kit before importing the nodes
    kit = r"D:\models\image-models-dev\ltxv23"
    for key in ("unet", "diffusion_models"):
        folder_paths.add_model_folder_path(key, kit)
    folder_paths.add_model_folder_path("vae", kit)
    # TEs live in text_encoders/, the projections sidecar ships in the kit
    # root next to the DiT - register both dirs for the clip keys.
    for key in ("clip", "text_encoders"):
        folder_paths.add_model_folder_path(key, kit + r"\text_encoders")
        folder_paths.add_model_folder_path(key, kit)

    import types
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("cctech_gguf_pkg")
    pkg.__path__ = [str(root)]
    sys.modules["cctech_gguf_pkg"] = pkg
    ns = __import__("importlib").import_module("cctech_gguf_pkg.nodes_ltx23")

    loader = ns.LTXV23ModelsLoader()
    model, clip, vae, audio_vae = loader.load(
        "10Eros_v1.4_distilled-r72_Q4_K_M.gguf",
        "gemma-3-12b-it-ablit-norms-biproj-Q4_K_M.gguf",
        "10Eros_v1.4_projections.safetensors",
        "10Eros_v1.4_video_vae.safetensors",
        "10Eros_v1.4_audio_vae.safetensors",
    )
    cfg = model.model.model_config.unet_config
    assert cfg.get("image_model") == "ltxav" and cfg.get("num_layers") == 48
    w = model.model.diffusion_model.transformer_blocks[0].attn1.to_q.weight
    assert type(w).__name__ == "GGMLTensor", type(w).__name__
    print(f"[ok] Q4 GGUF -> ltxav/48, GGMLTensor weights, "
          f"{model.model_size() / 1024**3:.2f} GiB stored")

    projection = type(getattr(clip.cond_stage_model,
                              "text_embedding_projection", None)).__name__
    assert projection == "DualLinearProjection", projection
    toks = clip.tokenize("hello world")["gemma3_12b"]
    assert len(toks[0]) == 1024, len(toks[0])
    print("[ok] Gemma Q4 + projections -> DualLinearProjection, tokenizer ok")

    assert type(vae.first_stage_model).__name__ != "AudioVAE"
    fsm = audio_vae.first_stage_model
    assert (fsm.latent_channels, fsm.latent_frequency_bins) == (8, 16)
    print(f"[ok] VAEs: video {type(vae.first_stage_model).__name__}, "
          f"audio {type(fsm).__name__} ({fsm.sample_rate}->"
          f"{fsm.output_sample_rate} Hz)")
    del model

    # fp8mixed safetensors DiT through the same node (TE comes from cache)
    model, clip2, _, audio_vae2 = loader.load(
        "10Eros_v1.4_distilled-r72_fp8mixed.safetensors",
        "gemma-3-12b-it-ablit-norms-biproj-Q4_K_M.gguf",
        "10Eros_v1.4_projections.safetensors",
        "10Eros_v1.4_video_vae.safetensors",
        "10Eros_v1.4_audio_vae.safetensors",
    )
    cfg = model.model.model_config.unet_config
    assert cfg.get("image_model") == "ltxav" and cfg.get("num_layers") == 48
    assert clip2 is clip  # encoder cache hit
    print("[ok] fp8mixed safetensors -> ltxav/48 (config from file metadata)")
    del model

    # remaining quants through the full UnetLoaderGGUF path
    from cctech_gguf_pkg.nodes import UnetLoaderGGUF
    for name in ("10Eros_v1.4_distilled-r72_Q6_K.gguf",
                 "10Eros_v1.4_distilled-r72_Q8_0.gguf"):
        m, = UnetLoaderGGUF().load_unet(name)
        c = m.model.model_config.unet_config
        assert c.get("image_model") == "ltxav" and c.get("num_layers") == 48
        print(f"[ok] {name} -> ltxav/48")
        del m

    # empty AV latent geometry
    latent, = ns.LTXV23EmptyLatentAV().generate(
        audio_vae2, width=768, height=512, length=121, frame_rate=24.0,
        batch_size=1)
    video, audio = latent["samples"].unbind()
    assert tuple(video.shape) == (1, 128, 16, 16, 24), tuple(video.shape)
    assert tuple(audio.shape)[0:1] == (1,) and tuple(audio.shape)[3] == 16
    n_latents = audio.shape[2]
    assert n_latents > 0
    print(f"[ok] empty AV latent: video {tuple(video.shape)}, "
          f"audio {tuple(audio.shape)}")

    # distilled schedule
    sig8 = ns.distilled_sigma_schedule(8)
    assert torch.allclose(sig8, torch.tensor(ns.DISTILLED_SIGMAS))
    sig16 = ns.distilled_sigma_schedule(16)
    assert len(sig16) == 17 and bool((sig16[:-1] > sig16[1:]).all())
    sigd = ns.distilled_sigma_schedule(8, denoise=0.5)
    # comfy's convention: start_step = steps * (1 - denoise) -> sigma index 4
    assert abs(float(sigd[0]) - 0.975) < 1e-6, float(sigd[0])
    print("[ok] distilled sigma schedule (8 exact, 16 resampled, denoise slice)")

    print("LTX-2.3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
