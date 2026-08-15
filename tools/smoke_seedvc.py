"""Opt-in real-checkpoint smoke test for the offline SeedVC bundle.

Loads every component from ``--extras`` and, unless ``--load-only`` is passed,
runs a full end-to-end conversion so the whole path — Whisper semantics,
CAMPPlus identity, the flow-matching DiT and BigVGAN — is exercised against the
real weights, not just checked for state-dict compatibility.
"""

import argparse
import importlib.util
import math
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-path", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--load-only", action="store_true",
                        help="Skip the conversion and only verify the bundle loads.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy_path))
    sys.argv = [sys.argv[0]] + (["--cpu"] if args.device == "cpu" else [])
    import comfy.options
    comfy.options.args_parsing = True

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("comfy_gguf_seedvc_smoke")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"{package.__name__}.{name}", root / f"{name}.py")
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    module_utils = load("seedvc_utils")
    module = load("seedvc")

    import torch

    device = torch.device(args.device)
    bundle = None
    try:
        bundle = module.load_seed_vc(device=device, root=args.extras)
        assert bundle.seed is not None
        assert bundle.camp is not None
        assert bundle.whisper is not None
        assert bundle.vocoder is not None
        print(f"SeedVC bundle loaded successfully on {device}")

        if not args.load_only:
            sr = module.SEEDVC_SR
            # A pitched, amplitude-shaped tone stands in for speech: it gives the
            # encoders real structure to follow without shipping an audio fixture.
            t = torch.arange(sr * 4, dtype=torch.float32) / sr
            wave = (0.4 * torch.sin(2 * math.pi * 140 * t)
                    * (0.6 + 0.4 * torch.sin(2 * math.pi * 3 * t)))
            source = wave.reshape(1, 1, -1)
            reference = (wave.flip(-1) * 0.8).reshape(1, 1, -1)

            out, out_sr = module.convert_voice(
                source, sr, reference, sr, steps=args.steps, seed=1234,
                bundle=bundle)
            again, _ = module.convert_voice(
                source, sr, reference, sr, steps=args.steps, seed=1234,
                bundle=bundle)

            assert out_sr == sr, out_sr
            assert out.ndim == 3 and out.shape[1] == 1, tuple(out.shape)
            assert torch.isfinite(out).all(), "conversion produced NaN/inf"
            assert module_utils.seedvc_output_is_usable(out, source), (
                f"unusable output: shape={tuple(out.shape)} peak={float(out.abs().max())}")
            assert torch.equal(out, again), "seeded conversion is not reproducible"
            print(f"SeedVC conversion OK: {out.shape[-1] / out_sr:.2f}s @ {out_sr} Hz, "
                  f"peak={float(out.abs().max()):.3f}, reproducible with seed=1234")
    finally:
        module.unload_seed_vc(bundle)
        if bundle is not None:
            assert all(getattr(bundle, name) is None
                       for name in ("seed", "camp", "whisper", "vocoder"))


if __name__ == "__main__":
    main()
