"""FLUX.1 / FLUX.2 text-encoder GGUF regressions found on ComfyUI 0.39.0 (2.19.2).

1. A full CLIP-L text encoder GGUF tagged `general.architecture = clip`
   (flux1-dev-turbo-GGUF clip-Q6_K.gguf) was taken for an llama.cpp mmproj and
   merged into the T5 TE -> KSampler "mat1 and mat2 shapes cannot be multiplied
   (77x768 and 4096x3072)".
2. The tekken blob rebuilt from a Ministral-3 GGUF lacked
   `config.default_num_special_tokens`, which ComfyUI 0.39's
   `bpe_tokenizer.from_tekken_json` reads -> KeyError at CLIPLoaderGGUF.
"""
import importlib.util
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from conftest import load_pack_module

loader = load_pack_module("loader")


def _write_gguf(path, arch, tensor_names, fields=()):
    writer = gguf.GGUFWriter(str(path), arch)
    for key, value in fields:
        writer.add_string(key, value)
    for name in tensor_names:
        writer.add_tensor(name, np.zeros((2, 2), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


# What the real flux1 clip-Q6_K.gguf holds: HF CLIPModel names, no metadata
# beyond general.* (header probed 2026-10-15).
CLIP_L_TENSORS = [
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_model.encoder.layers.1.mlp.fc1.weight",
    "text_model.final_layer_norm.weight",
    "text_projection.weight",
    "vision_model.embeddings.patch_embedding.weight",
    "vision_model.encoder.layers.0.mlp.fc1.weight",
    "vision_model.post_layernorm.weight",
    "visual_projection.weight",
]
# Ministral-3 pixtral mmproj (real header): v.* tower + mm.* projector.
PIXTRAL_MMPROJ_TENSORS = [
    "v.blk.0.attn_q.weight", "v.patch_embd.weight", "v.pre_ln.weight",
    "v.token_embd.img_break", "mm.1.weight", "mm.2.weight",
    "mm.input_norm.weight", "mm.patch_merger.weight",
]


def test_clip_l_text_encoder_is_not_a_vision_projector(tmp_path):
    p = _write_gguf(tmp_path / "clip-Q6_K.gguf", "clip", CLIP_L_TENSORS,
                    [("general.name", "flux1-clip-text-encoder")])
    assert loader.read_gguf_arch(str(p)) == "clip"
    assert not loader.is_vision_projector(str(p))
    # gguf_clip_loader hands back the HF keys comfy's FluxClipModel.load_sd
    # routes to clip_l, not an (empty) mapped vision tower.
    sd = loader.gguf_clip_loader(str(p))
    assert "text_model.encoder.layers.1.mlp.fc1.weight" in sd
    assert not any(k.startswith(("visual.", "model.visual.")) for k in sd)


def test_text_model_tensors_veto_mmproj_metadata(tmp_path):
    # Even with mmproj-looking metadata, text_model.* means a real text encoder.
    p = _write_gguf(tmp_path / "clip.gguf", "clip", CLIP_L_TENSORS,
                    [("general.type", "mmproj")])
    assert not loader.is_vision_projector(str(p))


@pytest.mark.parametrize("names,fields", [
    (PIXTRAL_MMPROJ_TENSORS,
     [("general.type", "mmproj"), ("clip.projector_type", "pixtral")]),
    # older converters: no general.type, tensor names alone decide
    (PIXTRAL_MMPROJ_TENSORS, []),
    # gemma3-style projector
    (["v.blk.0.attn_q.weight", "mm.input_projection.weight",
      "mm.soft_emb_norm.weight"], []),
    # metadata alone (no recognisable tensors)
    (["something.weight"], [("clip.projector_type", "qwen2vl_merger")]),
])
def test_mmproj_files_are_still_vision_projectors(tmp_path, names, fields):
    p = _write_gguf(tmp_path / "mmproj-F16.gguf", "clip", names, fields)
    assert loader.is_vision_projector(str(p))


def test_mmproj_rule_needs_the_clip_arch():
    assert not loader.is_mmproj_content("qwen2vl", PIXTRAL_MMPROJ_TENSORS)
    assert not loader.is_mmproj_content("clip", ["something.weight"])
    assert loader.is_mmproj_content("clip", PIXTRAL_MMPROJ_TENSORS)


# --- tekken -----------------------------------------------------------------

def _write_tekken_gguf(path, tokens, types):
    writer = gguf.GGUFWriter(str(path), "mistral3")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_tensor("token_embd.weight", np.zeros((2, 2), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


# 3 control tokens first (as every tekken GGUF stores them), then BPE pieces.
TEKKEN_TOKENS = ["<unk>", "<s>", "</s>", "a", "b", "ab", "Ġ", "Ġab"]
TEKKEN_TYPES = [3, 3, 3, 1, 1, 1, 1, 1]


def _tekken_blob(tmp_path):
    p = _write_tekken_gguf(tmp_path / "Ministral-3-14B.gguf", TEKKEN_TOKENS, TEKKEN_TYPES)
    blob = loader.gguf_tekken_tokenizer_loader(str(p), (131072, 5120))
    return blob.numpy().tobytes()


def test_tekken_config_has_every_field_comfy_reads(tmp_path):
    config = json.loads(_tekken_blob(tmp_path))["config"]
    # ComfyUI 0.39 from_tekken_json reads exactly these two.
    assert config["default_vocab_size"] == 131072
    assert config["default_num_special_tokens"] == 3


def test_tekken_num_special_tokens_rule():
    specials = [{"rank": i} for i in range(1000)]
    assert loader.tekken_num_special_tokens(specials) == 1000
    assert loader.tekken_num_special_tokens([]) == loader.TEKKEN_DEFAULT_NUM_SPECIAL_TOKENS == 1000


def _installed_bpe_tokenizer():
    for root in (r"N:\ComfyUI_windows_portable_nvidia\ComfyUI",
                 r"D:\Projects\ComfyUI\upstream\ComfyUI"):
        src = Path(root) / "comfy" / "text_encoders" / "bpe_tokenizer.py"
        if src.is_file():
            spec = importlib.util.spec_from_file_location("_comfy_bpe_under_test", src)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("no ComfyUI >= 0.39 checkout with bpe_tokenizer.py on this machine")


def test_tekken_blob_builds_with_installed_comfy_and_keeps_gguf_ids(tmp_path):
    bpe = _installed_bpe_tokenizer()
    tok = bpe.from_tekken_json(_tekken_blob(tmp_path))
    # ids must be the GGUF row indices, i.e. the embedding rows of the TE
    assert tok("ab")["input_ids"] == [TEKKEN_TOKENS.index("ab")]
    assert tok(" ab")["input_ids"] == [TEKKEN_TOKENS.index("Ġab")]
    assert tok("<s>")["input_ids"] == [1]
