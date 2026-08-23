"""Delta check for the 1.5.2 fixes - seconds, not minutes.

  1. nodes_ltx23 imports cleanly (guard revert + device fix edits)
  2. _load_vae builds BOTH repaired uncensored VAEs with the right geometry
  3. _encode_reference_audio runs end-to-end on CPU (fake 5s clip)

No DiT, no TE - the full smoke_ltx23.py already validated those and they are
unchanged since.
"""
import sys


def main():
    sys.argv = [sys.argv[0], "--cpu"]
    from pathlib import Path
    port = Path(r"P:\Projects\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable")
    sys.path.insert(0, str(port / "ComfyUI"))
    import comfy.options
    comfy.options.args_parsing = True

    import logging
    logging.basicConfig(level=logging.ERROR)

    import torch
    import folder_paths

    vae_dir = port / "ComfyUI" / "models" / "vae"
    folder_paths.add_model_folder_path("vae", str(vae_dir))

    import types
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("cctech_gguf_pkg")
    pkg.__path__ = [str(root)]
    sys.modules["cctech_gguf_pkg"] = pkg
    ns = __import__("importlib").import_module("cctech_gguf_pkg.nodes_ltx23")
    print("[ok] nodes_ltx23 imports")

    video = ns._load_vae("ltxv23_uncensored_video_vae.safetensors", want_audio=False)
    assert type(video.first_stage_model).__name__ == "VideoVAE"
    w = video.first_stage_model.encoder.down_blocks[7].conv.conv.weight
    assert tuple(w.shape) == (128, 1024, 3, 3, 3), tuple(w.shape)
    print(f"[ok] uncensored video VAE -> VideoVAE, down7 conv {tuple(w.shape)}")

    audio = ns._load_vae("ltxv23_uncensored_audio_vae.safetensors", want_audio=True)
    fsm = audio.first_stage_model
    assert type(fsm).__name__ == "AudioVAE"
    assert (fsm.latent_channels, fsm.latent_frequency_bins) == (8, 16)
    print(f"[ok] uncensored audio VAE -> AudioVAE {fsm.sample_rate}->{fsm.output_sample_rate} Hz")

    sr = 48000
    fake = {"waveform": torch.randn(1, 2, sr * 5), "sample_rate": sr}
    latent, mask = ns._encode_reference_audio(audio, fake, 6.0)
    assert latent.dim() == 4 and latent.shape[0] == 1
    assert mask.shape == latent.shape and bool((mask == 0).all())
    print(f"[ok] reference-audio encode: latent {tuple(latent.shape)}, mask all-zero (locked)")

    print("DELTA CHECK PASSED")


if __name__ == "__main__":
    main()
