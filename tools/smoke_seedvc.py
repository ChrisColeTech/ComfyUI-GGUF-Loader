"""Opt-in real-checkpoint load smoke test for the offline SeedVC bundle."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-path", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.comfy_path))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.args_parsing = True

    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("comfy_gguf_seedvc_smoke")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.seedvc", root / "seedvc.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    bundle = None
    try:
        bundle = module.load_seed_vc(device=__import__("torch").device("cpu"), root=args.extras)
        assert bundle.seed is not None
        assert bundle.camp is not None
        assert bundle.whisper is not None
        assert bundle.vocoder is not None
        print("SeedVC bundle loaded successfully on CPU")
    finally:
        module.unload_seed_vc(bundle)
        if bundle is not None:
            assert all(getattr(bundle, name) is None
                       for name in ("seed", "camp", "whisper", "vocoder"))


if __name__ == "__main__":
    main()
