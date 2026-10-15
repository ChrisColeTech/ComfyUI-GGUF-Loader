"""Transformers-free UniRig skeleton loading (Merge-P0.1).

Builds the skeleton stack straight from ``skeleton.safetensors``:

- tokenizer from the vendored inlined config (pure python),
- Michelangelo mesh encoder via ``unirig.parse_encoder`` (pure torch),
- the OPT decoder as :class:`OPTDecoderLM` (pure torch, PhaseB.16),

wrapped in :class:`PureUniRigAR`, which is UniRigAR-shaped where it matters:
``encode_mesh_cond`` / ``generate`` / ``get_dtype`` / ``tokenizer`` — so the
wrapper's cache seeding, ``direct.predict_skeleton_from_mesh``, and the
skinning path work unchanged. The vendored ``unirig_ar`` module (and with it
the transformers import, real or stubbed) is never touched on this path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch
from torch import nn

from .opt_decoder import OPTDecoderLM
from .pure_skeleton import install_pure_decode

log = logging.getLogger("unirig")


class PureUniRigAR(nn.Module):
    """UniRigAR-shaped skeleton spec without the transformers scaffold."""

    def __init__(self, tokenizer, mesh_encoder, output_proj, decoder: OPTDecoderLM, dtype):
        super().__init__()
        self.tokenizer = tokenizer
        self.mesh_encoder = mesh_encoder
        self.output_proj = output_proj
        self.opt_decoder = decoder
        self.dtype = dtype
        # Alias (not a second registered submodule — that would double the
        # expected state-dict keys).
        self.__dict__["_pure_decoder"] = decoder
        # _pure_generate probes self.transformer to decide whether an HF copy
        # needs evicting; point it at a plain namespace exposing the decoder
        # (bypasses nn.Module attr registration on purpose).
        self.__dict__["transformer"] = SimpleNamespace(_pure_decoder=decoder)
        install_pure_decode(self)  # binds .generate

    def get_dtype(self):
        return self.dtype or torch.float32

    @torch.no_grad()
    def encode_mesh_cond(self, vertices, normals):
        """Mirror of vendored UniRigAR.encode_mesh_cond (michelangelo branch)."""
        if self.dtype is not None:
            vertices = vertices.to(dtype=self.dtype)
            normals = normals.to(dtype=self.dtype)
        assert not torch.isnan(vertices).any()
        assert not torch.isnan(normals).any()
        if len(vertices.shape) == 3:
            _, latents, _, _ = self.mesh_encoder.encode_latents(pc=vertices, feats=normals)
        else:
            _, latents, _, _ = self.mesh_encoder.encode_latents(
                pc=vertices.unsqueeze(0), feats=normals.unsqueeze(0)
            )
        return self.output_proj(latents)


def load_skeleton_pure(checkpoint_path: str, dtype=None, attn_backend: str = "sdpa"):
    """Pure replacement for ``direct._load_skeleton_model``.

    Returns (PureUniRigAR on CPU eval, tokenizer) — same contract.
    """
    # Vendored pure modules (no unirig_ar / model_parse / transformers).
    from .configs import AR_MODEL_CONFIG, TOKENIZER_CONFIG
    from .parse_encoder import get_mesh_encoder
    from .tokenizer_parse import get_tokenizer
    from .tokenizer_spec import TokenizerConfig

    import comfy.ops
    import comfy.utils

    model_dtype = dtype if not isinstance(dtype, str) else {
        "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32
    }.get(dtype, torch.float32)
    operations = (
        comfy.ops.pick_operations(model_dtype, model_dtype)
        if model_dtype is not None
        else comfy.ops.disable_weight_init
    )

    tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=TOKENIZER_CONFIG))

    llm = AR_MODEL_CONFIG["llm"]
    mesh_cfg = dict(AR_MODEL_CONFIG["mesh_encoder"])
    mesh_cfg.setdefault("dtype", model_dtype)
    mesh_cfg.setdefault("device", None)
    mesh_cfg.setdefault("operations", operations)
    if attn_backend == "flash_attn":
        raise RuntimeError("pure loader supports sdpa only (flash_attn path lives upstream)")

    log.info("pure loader: building mesh encoder + OPT decoder (no transformers)")
    mesh_encoder = get_mesh_encoder(**mesh_cfg)
    decoder = OPTDecoderLM(
        vocab_size=tokenizer.vocab_size,
        hidden=llm["hidden_size"],
        num_layers=llm["num_hidden_layers"],
        heads=llm["num_attention_heads"],
        ffn=llm["ffn_dim"],
        max_positions=llm["max_position_embeddings"],
    )
    output_proj = operations.Linear(mesh_encoder.width, llm["hidden_size"])

    model = PureUniRigAR(tokenizer, mesh_encoder, output_proj, decoder, model_dtype)

    state = comfy.utils.load_torch_file(str(checkpoint_path))
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}

    # transformer.* -> the pure decoder's own names; the rest load by name.
    own = {}
    for k, v in state.items():
        if k.startswith("transformer.model.decoder.embed_tokens"):
            own["opt_decoder.embed_tokens" + k[len("transformer.model.decoder.embed_tokens"):]] = v
        elif k.startswith("transformer.model.decoder.embed_positions"):
            own["opt_decoder.embed_positions" + k[len("transformer.model.decoder.embed_positions"):]] = v
        elif k.startswith("transformer.model.decoder.final_layer_norm"):
            own["opt_decoder.final_layer_norm" + k[len("transformer.model.decoder.final_layer_norm"):]] = v
        elif k.startswith("transformer.model.decoder.layers"):
            own["opt_decoder.layers" + k[len("transformer.model.decoder.layers"):]] = v
        elif k.startswith("transformer.lm_head"):
            own["opt_decoder.lm_head" + k[len("transformer.lm_head"):]] = v
        else:
            own[k] = v  # mesh_encoder.*, output_proj.*

    result = model.load_state_dict(own, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith("transformer.")]
    if unexpected:
        raise RuntimeError(f"pure loader: unexpected checkpoint keys: {unexpected[:8]}")
    missing = [k for k in result.missing_keys if not k.endswith("frequencies")]
    if missing:
        raise RuntimeError(f"pure loader: missing checkpoint keys: {missing[:8]}")

    if model_dtype is not None:
        model = model.to(dtype=model_dtype)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("pure skeleton spec ready: %.1fM params, dtype=%s", n_params / 1e6, model_dtype)
    return model, tokenizer
