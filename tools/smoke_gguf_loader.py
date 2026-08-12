from __future__ import annotations
import ast, importlib.util, re, sys, time, types
from pathlib import Path
import gguf, torch

ROOT = Path(r"D:\Projects\Comfy-GGUF")

def load_arch_lists():
    src = (ROOT / "loader.py").read_text(encoding="utf-8")
    img_m = re.search(r"IMG_ARCH_LIST\s*=\s*(\{.*?\n\})", src, re.S)
    txt_m = re.search(r"TXT_ARCH_LIST\s*=\s*(\{.*?\n\})", src, re.S)
    return set(ast.literal_eval(img_m.group(1))), set(ast.literal_eval(txt_m.group(1)))

def load_dequant():
    pkg = types.ModuleType("comfyui_gguf")
    pkg.__path__ = [str(ROOT)]
    sys.modules["comfyui_gguf"] = pkg
    spec = importlib.util.spec_from_file_location("comfyui_gguf.dequant", ROOT / "dequant.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_gguf.dequant"] = mod
    spec.loader.exec_module(mod)
    return mod

def get_field(reader, name):
    field = reader.get_field(name)
    if field is None:
        return None
    if len(field.types) == 1 and field.types[0] == gguf.GGUFValueType.STRING:
        return str(field.parts[field.data[-1]], encoding="utf-8")
    return None

def dequant_one(dq, reader):
    skip = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16}
    for t in reader.tensors:
        q = t.tensor_type
        if q in skip or q not in dq.dequantize_functions:
            continue
        raw = torch.from_numpy(t.data.copy())
        shape = tuple(int(x) for x in reversed(t.shape))
        block_size, type_size = gguf.GGML_QUANT_SIZES[q]
        rows = raw.reshape((-1, raw.shape[-1])).view(torch.uint8)
        n_blocks = rows.numel() // type_size
        if n_blocks <= 0:
            continue
        blocks = rows.reshape((n_blocks, type_size))
        out = dq.dequantize_functions[q](blocks, block_size, type_size, torch.float16)
        try:
            out = out.reshape(shape)
        except Exception:
            out = out.reshape(-1)
        sample = out.reshape(-1)[:2048].float()
        if not torch.isfinite(sample).all():
            raise ValueError("non-finite dequant for %s" % t.name)
        return "%s shape=%s q=%s" % (t.name, tuple(out.shape), getattr(q, "name", q))
    return "n/a"

def main():
    print("1) allowlists")
    IMG, TXT = load_arch_lists()
    print("  IMG", sorted(IMG))
    print("  TXT", sorted(TXT))

    print("2) dequant")
    dq = load_dequant()
    print("  IQ", getattr(dq, "_HAS_EXTRA_IQ", None), "qtypes", len(dq.dequantize_functions))

    sys.path.insert(0, str(ROOT / "tools"))
    import convert

    detect_cases = [
        ("ltx2", {"adaln_t_table": 1, "blocks.0.adaln_proj.linear.weight": 1, "audio_patch_proj.weight": 1}, "ltx2"),
        ("zimage", {"all_x_embedder.2-1.weight": 1, "all_final_layer.2-1.adaLN_modulation.1.weight": 1, "cap_embedder.1.weight": 1}, "zimage"),
        ("ideogram4", {"embed_image_indicator.weight": 1, "adaln_proj.weight": 1, "final_layer.adaln_modulation.weight": 1}, "ideogram4"),
        ("qwen_image", {"img_in.weight": 1, "norm_out.linear.weight": 1, "proj_out.weight": 1, "transformer_blocks.0.attn.to_q.weight": 1}, "qwen_image"),
        ("krea2", {"blocks.0.attn.qknorm.qnorm.scale": 1, "blocks.0.mod.lin": 1, "txtfusion.projector.weight": 1}, "qwen_image"),
        ("ltxv", {"adaln_single.emb.timestep_embedder.linear_2.weight": 1, "transformer_blocks.27.scale_shift_table": 1, "caption_projection.linear_2.weight": 1}, "ltxv"),
    ]
    load_cases = [
        (r"D:\models\image-models\minimax-h3\split\diffusion_models\minimax_h3_fl2va_turbo_Q6_K.gguf", False, "ltx2"),
        (r"D:\models\image-models\ltxv25\split\diffusion_models\ltx-2.5-22b-distilled-transformer-Q6_K.gguf", False, "ltxv"),
        (r"D:\models\image-models\z-image\split\z-image-turbo-q6k-fresh.gguf", False, "zimage"),
        (r"D:\models\image-models\ideogram\split\ideogram4-turbo-Q6_K.gguf", False, "ideogram4"),
        (r"D:\models\image-models\ideogram\split\ideogram4_unconditional_Q6_K.gguf", False, None),
        (r"D:\models\image-models\krea2\split\diffusion_models\krea2_turbo_edit-Q4_K_M.gguf", False, "qwen_image"),
        (r"D:\models\image-models\qwen-image-edit\split\qwen-image-edit-2509-turbo-Q4_K_M.gguf", False, "qwen_image"),
        (r"D:\models\image-models\flux1-dev\split\flux1-turbo-Q4_K_M.gguf", False, "flux"),
        (r"D:\models\image-models\flux2-dev\split\flux2-dev-turbo-Q4_K_M-mixed.gguf", False, "flux"),
        (r"D:\models\image-models\pixal3d\split\shape\slat_flow_img2shape_dit_1_3B_512_bf16.gguf", False, "flux"),
        (r"D:\models\image-models\flux2-dev\split\text_encoders\Ministral-3-14B-Instruct-2512-Q3_K_S.gguf", True, "mistral3"),
        (r"D:\models\image-models\ltxv25\packaging\gemma4-12b-ltx-2.5-Q6_K.gguf", True, "gemma4"),
    ]

    failed = passed = skipped = 0
    print("3) convert detect_arch")
    for label, keys, want in detect_cases:
        got = convert.detect_arch(keys).arch
        ok = got == want
        print("  [%s] %s: %s" % ("PASS" if ok else "FAIL", label, got))
        passed += int(ok)
        failed += int(not ok)

    print("4) GGUF arch + dequant")
    for path, is_text, expect in load_cases:
        p = Path(path)
        if not p.exists():
            print("  [SKIP]", p.name)
            skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            reader = gguf.GGUFReader(str(p))
            arch = get_field(reader, "general.architecture")
            n = len(reader.tensors)
            if arch is None:
                names = {t.name for t in reader.tensors}
                arch = convert.detect_arch({k: 1 for k in names}).arch
            allow = TXT if is_text else IMG
            if arch not in allow:
                raise ValueError("arch %r not in allowlist" % arch)
            if expect is not None and arch != expect:
                raise ValueError("want %r got %r" % (expect, arch))
            deq = dequant_one(dq, reader)
            dt = time.perf_counter() - t0
            print("  [PASS] %s: arch=%r n=%d %.2fs deq=%s" % (p.name, arch, n, dt, deq))
            passed += 1
        except Exception as e:
            print("  [FAIL] %s: %s: %s" % (p.name, type(e).__name__, e))
            failed += 1

    print("RESULT: %d passed, %d failed, %d skipped" % (passed, failed, skipped))
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())