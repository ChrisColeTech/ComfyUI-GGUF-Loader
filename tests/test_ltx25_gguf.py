"""LTX-2.5 GGUF loading: Gemma-4 sidecars and transformer-config translation.

The ltxv25 split ships a Gemma-4 TE GGUF (tokenizer.json + layer_scalar
sidecars), a distilled 22B DiT GGUF plus a `-metadata.json` config sidecar in
LTX's own key names, and ltx-v2-projections.safetensors. These tests pin the
loader bridges that make that set loadable.
"""
import json

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from conftest import load_pack_module
from test_clip_type_widgets import _import_nodes_module

loader = load_pack_module("loader")
nodes_mod = _import_nodes_module()


def _write_gemma4_gguf(path, tokens):
    writer = gguf.GGUFWriter(str(path), "gemma4")
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_array("tokenizer.ggml.scores", [0.0] * len(tokens))
    writer.add_array("tokenizer.ggml.token_type", [1] * len(tokens))
    writer.add_tensor("token_embd.weight", np.zeros((len(tokens), 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_tokenizer_sidecar(path, vocab):
    doc = {"model": {"type": "BPE", "vocab": vocab}, "added_tokens": []}
    path.write_text(json.dumps(doc), encoding="utf-8")


TOKENS = ["<pad>", "a", "b", "c"]


def test_gemma4_without_tokenizer_sidecar_raises(tmp_path):
    gguf_path = tmp_path / "gemma4.gguf"
    _write_gemma4_gguf(gguf_path, TOKENS)

    with pytest.raises(ValueError, match="tokenizer.json"):
        loader.gguf_clip_loader(str(gguf_path))


def test_gemma4_sidecar_vocab_mismatch_is_rejected(tmp_path):
    gguf_path = tmp_path / "gemma4.gguf"
    _write_gemma4_gguf(gguf_path, TOKENS)
    _write_tokenizer_sidecar(tmp_path / "gemma4.gguf.tokenizer.json",
                             {"x": 0, "y": 1, "z": 2})  # 3 != 4 tokens

    with pytest.raises(ValueError, match="different model"):
        loader.gguf_clip_loader(str(gguf_path))


def test_gemma4_without_layer_scalar_fixup_raises(tmp_path):
    # The tokenizer sidecar is fine but the GGUF carries no layer_scalar: the
    # model would run on uninitialized scalars, so the loader must refuse.
    gguf_path = tmp_path / "gemma4.gguf"
    _write_gemma4_gguf(gguf_path, TOKENS)
    _write_tokenizer_sidecar(tmp_path / "gemma4.gguf.tokenizer.json",
                             {t: i for i, t in enumerate(TOKENS)})

    with pytest.raises(ValueError, match="layer_scalar"):
        loader.gguf_clip_loader(str(gguf_path))


def test_gemma4_sidecars_load_into_the_state_dict(tmp_path):
    gguf_path = tmp_path / "gemma4.gguf"
    _write_gemma4_gguf(gguf_path, TOKENS)
    _write_tokenizer_sidecar(tmp_path / "gemma4.gguf.tokenizer.json",
                             {t: i for i, t in enumerate(TOKENS)})
    save_file({"model.layers.0.layer_scalar": torch.full([1], 0.5)},
              str(tmp_path / "gemma4.gguf.fixup.safetensors"))

    sd = loader.gguf_clip_loader(str(gguf_path))

    tok = sd["tokenizer_json"]
    assert isinstance(tok, torch.Tensor) and tok.dtype == torch.uint8
    assert json.loads(bytes(tok.tolist()).decode("utf-8"))["model"]["vocab"] == \
        {t: i for i, t in enumerate(TOKENS)}
    # fixup tensor rides through the GEMMA3 key map (comfy-layout keys only)
    assert sd["model.layers.0.layer_scalar"].shape == (1,)
    assert float(sd["model.layers.0.layer_scalar"]) == 0.5


def test_translate_ltx_transformer_cfg_maps_ltx_names():
    out = nodes_mod._translate_ltx_transformer_cfg({
        "cross_attn_mod": True, "gated_attn": True, "rope_theta": 10000.0,
        "cross_attn_timestep_scale_multiplier": 1000.0,
        "num_attention_heads": 32, "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "pos_embed_max_pos": 20, "base_height": 2048, "base_width": 2048,
        "audio_pos_embed_max_pos": 20,
    })
    assert out["cross_attention_adaln"] is True
    assert out["apply_gated_attention"] is True
    assert out["positional_embedding_theta"] == 10000.0
    assert out["av_ca_timestep_scale_multiplier"] == 1000.0
    assert out["positional_embedding_max_pos"] == [20, 2048, 2048]
    assert out["audio_positional_embedding_max_pos"] == [20]
    # connector width follows the main attention geometry (32*128=4096)
    assert out["connector_num_attention_heads"] == 32
    assert out["audio_connector_num_attention_heads"] == 32
    assert out["audio_connector_attention_head_dim"] == 64
    for dropped in ("cross_attn_mod", "gated_attn", "rope_theta",
                    "pos_embed_max_pos", "base_height", "base_width"):
        assert dropped not in out


LTX_SIDECAR_CONFIG = {
    "transformer": {
        "num_layers": 48, "num_attention_heads": 32, "attention_head_dim": 128,
        "cross_attn_mod": True, "gated_attn": True,
        "cross_attention_dim": 4096, "caption_channels": 3840,
        "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
        "audio_cross_attention_dim": 2048,
    },
}


def _write_metadata_sidecar(path):
    path.write_text(json.dumps({"config": json.dumps(LTX_SIDECAR_CONFIG)}),
                    encoding="utf-8")


def _fake_ltx_sd():
    return {
        "transformer_blocks.0.scale_shift_table": torch.zeros(9, 4096),
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight":
            torch.zeros(4096, 4096),
        "video_embeddings_connector.transformer_1d_blocks.0.attn1.to_gate_logits.weight":
            torch.zeros(4096, 4096),
    }


def test_unet_metadata_sidecar_translates_and_forces_from_weights(tmp_path):
    gguf_path = tmp_path / "ltx.gguf"
    gguf_path.write_bytes(b"")  # only the sidecar path matters here
    _write_metadata_sidecar(tmp_path / "ltx-transformer-metadata.json")

    metadata = nodes_mod._unet_metadata_sidecar(
        str(gguf_path), {}, _fake_ltx_sd())

    t = json.loads(metadata["config"])["transformer"]
    assert t["cross_attention_adaln"] is True
    assert t["apply_gated_attention"] is True
    # forced from the weights: 9-row adaln table, connector gate logits
    assert t["connector_apply_gated_attention"] is True
    # no caption_projection weights + connectors => pre-projected context
    assert t["caption_proj_before_connector"] is True
    assert t["caption_projection_first_linear"] is False


def test_unet_metadata_sidecar_keeps_checkpoint_caption_projection(tmp_path):
    # A checkpoint that HAS caption_projection weights must keep the
    # caption_channels split path (caption_proj_before_connector False).
    gguf_path = tmp_path / "ltx.gguf"
    gguf_path.write_bytes(b"")
    _write_metadata_sidecar(tmp_path / "ltx-transformer-metadata.json")
    sd = _fake_ltx_sd()
    sd["caption_projection.linear_1.weight"] = torch.zeros(4, 4)

    metadata = nodes_mod._unet_metadata_sidecar(str(gguf_path), {}, sd)

    t = json.loads(metadata["config"])["transformer"]
    assert not t.get("caption_proj_before_connector", False)


def test_unet_metadata_sidecar_string_config_is_parsed(tmp_path):
    # The real sidecar stores config as a JSON *string* (safetensors layout);
    # the translation has to see through that instead of passing it through.
    gguf_path = tmp_path / "ltx.gguf"
    gguf_path.write_bytes(b"")
    _write_metadata_sidecar(tmp_path / "ltx.gguf-metadata.json")

    metadata = nodes_mod._unet_metadata_sidecar(str(gguf_path), {}, None)

    assert "config" in metadata
    t = json.loads(metadata["config"])["transformer"]
    assert "cross_attention_adaln" in t  # translated, not raw LTX names


# --- sidecar ownership -------------------------------------------------------
# models/diffusion_models is a shared drawer. A folder-level sidecar merged onto
# the wrong GGUF builds a model with the wrong block count and head geometry,
# which survives loading and then dies inside attention.

def _sidecar_folder(tmp_path, sidecar_name, gguf_names, body=None):
    folder = tmp_path / "diffusion_models"
    folder.mkdir()
    (folder / sidecar_name).write_text(
        json.dumps(body if body is not None else {"config": {"transformer": {"gated_attn": True}}}),
        encoding="utf-8")
    for name in gguf_names:
        (folder / name).write_bytes(b"")
    return folder


def test_folder_sidecar_applies_to_its_own_family(tmp_path):
    folder = _sidecar_folder(tmp_path, "ltx-2.5-transformer-metadata.json",
                             ["ltx-2.5-22b-distilled-transformer-Q6_K.gguf"])
    meta = nodes_mod._unet_metadata_sidecar(
        str(folder / "ltx-2.5-22b-distilled-transformer-Q6_K.gguf"), {})
    assert "config" in meta


def test_folder_sidecar_does_not_leak_onto_an_unrelated_gguf(tmp_path):
    """The v1.4.4 regression: the LTX sidecar was merged onto a MiniMax H3 DiT."""
    folder = _sidecar_folder(tmp_path, "ltx-2.5-transformer-metadata.json",
                             ["minimax_h3_ref2va_turbo_Q6_K.gguf"])
    meta = nodes_mod._unet_metadata_sidecar(
        str(folder / "minimax_h3_ref2va_turbo_Q6_K.gguf"), {})
    assert "config" not in meta


def test_applies_to_globs_win_over_the_name_heuristic(tmp_path):
    folder = _sidecar_folder(
        tmp_path, "shared-metadata.json", ["someones_dit_Q6_K.gguf"],
        body={"applies_to": ["someones_*"], "config": {"transformer": {}}})
    assert "config" in nodes_mod._unet_metadata_sidecar(
        str(folder / "someones_dit_Q6_K.gguf"), {})


def test_applies_to_excludes_everything_it_does_not_list(tmp_path):
    folder = _sidecar_folder(
        tmp_path, "shared-metadata.json", ["other_dit_Q6_K.gguf"],
        body={"applies_to": ["someones_*"], "config": {"transformer": {}}})
    assert "config" not in nodes_mod._unet_metadata_sidecar(
        str(folder / "other_dit_Q6_K.gguf"), {})


def test_applies_to_is_not_merged_into_the_model_metadata(tmp_path):
    folder = _sidecar_folder(
        tmp_path, "ltx-2.5-transformer-metadata.json",
        ["ltx-2.5-22b-Q6_K.gguf"],
        body={"applies_to": ["ltx-2.5-*"], "config": {"transformer": {}}})
    meta = nodes_mod._unet_metadata_sidecar(str(folder / "ltx-2.5-22b-Q6_K.gguf"), {})
    assert "applies_to" not in meta


def test_a_sidecar_named_after_the_gguf_still_wins_outright(tmp_path):
    folder = _sidecar_folder(tmp_path, "ltx-2.5-transformer-metadata.json",
                             ["minimax_h3_ref2va_turbo_Q6_K.gguf"])
    unet = folder / "minimax_h3_ref2va_turbo_Q6_K.gguf"
    (folder / (unet.name + "-metadata.json")).write_text(
        json.dumps({"config": {"transformer": {}}}), encoding="utf-8")
    assert "config" in nodes_mod._unet_metadata_sidecar(str(unet), {})


def test_generic_metadata_json_is_still_taken_at_face_value(tmp_path):
    """Nothing in a bare `metadata.json` name to check against, so trust it."""
    folder = _sidecar_folder(tmp_path, "metadata.json", ["anything_Q6_K.gguf"])
    assert "config" in nodes_mod._unet_metadata_sidecar(
        str(folder / "anything_Q6_K.gguf"), {})


def test_quant_tags_alone_do_not_make_a_match(tmp_path):
    folder = _sidecar_folder(tmp_path, "krea2-Q6_K-metadata.json",
                             ["qwen-image-edit-Q6_K.gguf"])
    assert "config" not in nodes_mod._unet_metadata_sidecar(
        str(folder / "qwen-image-edit-Q6_K.gguf"), {})
