"""CPU smoke test for the MiniMax Music 3 loader paths.

Run: python tools/smoke_minimax_music3.py [--models-dir PATH] [--skip-te]
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir", default=r"D:\models\image-models\minimax-music\split")
    parser.add_argument("--skip-te", action="store_true")
    args = parser.parse_args()

    sys.argv = [sys.argv[0], "--cpu"]
    from pathlib import Path

    comfy_root = Path(__file__).resolve().parents[2] / "ComfyUI" / "upstream" / "ComfyUI"
    sys.path.insert(0, str(comfy_root))
    import comfy.options
    comfy.options.args_parsing = True
    import torch

    import types
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("cctech_gguf_pkg")
    package.__path__ = [str(root)]
    sys.modules["cctech_gguf_pkg"] = package
    music = __import__("importlib").import_module("cctech_gguf_pkg.nodes_minimax_music")

    conditioning = [[
        torch.ones(1, 2, 3),
        {"conditioning_scale": torch.ones(1, 1, 1)},
    ]]
    negative = music._zero_conditioning(conditioning)
    assert torch.count_nonzero(negative[0][0]) == 0
    assert torch.count_nonzero(negative[0][1]["conditioning_scale"]) == 0
    assert set(music.NODE_CLASS_MAPPINGS) == {
        "MiniMaxMusic3ModelsLoader", "MiniMaxMusic3AudioGenerate"}

    base = Path(args.models_dir)
    dit_path = base / "diffusion_models" / "minimax_music3_dit_Q8_0.gguf"
    dav_path = base / "vae" / "minimax_music3_dav.safetensors"
    te_path = base / "text_encoders" / "minimax_music3_text_encoder_pruned_Q8_0.gguf"

    model = music._load_diffusion(str(dit_path))
    config = model.model.model_config.unet_config
    assert config.get("audio_model") == "minimax_music3"
    assert model.model.latent_format.latent_channels == 128
    print("[ok] MiniMax Music 3 DiT detected through native Comfy model")

    vae = music._load_dav(str(dav_path))
    assert vae.latent_channels == 128
    assert vae.audio_sample_rate == 44100
    assert vae.upscale_ratio == 512
    print("[ok] DAV detected (128-channel latent, 44.1 kHz stereo)")

    if not args.skip_te:
        clip = music._load_text_encoder(str(te_path))
        assert type(clip.cond_stage_model).__name__ == "MiniMaxMusic3TEModel"
        tokens = clip.tokenize(
            "Global Metadata: acoustic pop, 96 BPM.",
            lyrics="[verse]\nMorning light",
            seed=0,
            max_audio_frames=25,
            cfg_scale=1.5,
            top_k=50,
        )
        assert tokens["minimax_music3"]
        print("[ok] pruned AR GGUF and embedded tokenizer detected")

    print("MINIMAX MUSIC 3 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
