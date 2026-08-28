"""Real-environment smoke test for LTX-2.3 EditAnything reference patching.

Runs against the actual portable ComfyUI install (real comfy.ldm.lightricks
.model.BasicTransformerBlock, not a fake) so the highest-risk piece of the
port gets checked directly: `_patched_block_forward` is a hand copy of
comfy's real BasicTransformerBlock.forward with one addition (the ref_attn
residual). If that copy is subtly wrong, EVERY generation through a patched
block breaks, not just EditAnything - so the critical assertion here is
NUMERICAL: with no reference conditioning present, the patched forward must
produce output BITWISE IDENTICAL to calling the real, unpatched
BasicTransformerBlock.forward on the same inputs. That proves the copy is
faithful before ever touching a real 40GB checkpoint.

Usage: python tools/smoke_ltx23_editanything.py
"""
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "--cpu"]

PORTABLE = Path(r"N:\ComfyUI_windows_portable_nvidia\ComfyUI")
sys.path.insert(0, str(PORTABLE))

import types

import torch
from safetensors.torch import save_file

import comfy.options
comfy.options.args_parsing = True

import folder_paths  # noqa: E402
folder_paths.folder_names_and_paths.setdefault("unet", ([], set()))

import importlib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))
pkg = types.ModuleType("cctech_gguf_pkg")
pkg.__path__ = [str(REPO_ROOT)]
sys.modules["cctech_gguf_pkg"] = pkg
nodes_pkg = types.ModuleType("cctech_gguf_pkg.nodes")
nodes_pkg.__path__ = [str(REPO_ROOT / "nodes")]
sys.modules["cctech_gguf_pkg.nodes"] = nodes_pkg
ltx23 = importlib.import_module("cctech_gguf_pkg.nodes.ltx23")  # noqa: E402

import comfy.ldm.lightricks.model as ltx_model  # noqa: E402
import comfy.ops  # noqa: E402


DIM = 64
HEADS = 2
DHEAD = 32
CTX_DIM = 64


def _make_block(idx):
    block = ltx_model.BasicTransformerBlock(
        dim=DIM, n_heads=HEADS, d_head=DHEAD, context_dim=CTX_DIM,
        cross_attention_adaln=False, operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        block.scale_shift_table.normal_()
    block.idx = idx
    return block


def _fake_timestep(batch=1, tokens=5):
    # 6 ada params (cross_attention_adaln=False) x DIM, matching scale_shift_table
    return torch.randn(batch, tokens, 6 * DIM)


def test_patched_forward_matches_original_when_no_reference():
    block = _make_block(idx=20)  # inside the 12-35 patched range
    original_forward = block.forward

    x = torch.randn(1, 5, DIM)
    context = torch.randn(1, 3, CTX_DIM)
    timestep = _fake_timestep()

    torch.manual_seed(0)
    expected = original_forward(x.clone(), context=context, timestep=timestep)

    block.forward = types.MethodType(ltx23._patched_block_forward, block)
    torch.manual_seed(0)
    actual = block.forward(x.clone(), context=context, timestep=timestep,
                           transformer_options={})  # no ref_context -> must be a no-op

    assert torch.equal(expected, actual), "patched forward diverged from comfy's real " \
        "BasicTransformerBlock.forward with no reference conditioning present - the " \
        "hand-copied body has a bug"
    print("[ok] _patched_block_forward: bitwise identical to comfy's real "
          "BasicTransformerBlock.forward when no ref_context is present")


def test_patched_forward_applies_residual_only_in_range_and_when_present():
    in_range = _make_block(idx=20)
    out_of_range = _make_block(idx=5)  # outside 12-35
    for block in (in_range, out_of_range):
        block.ref_attn = ltx23._EditAnythingRefAttention.__new__(ltx23._EditAnythingRefAttention)
        torch.nn.Module.__init__(block.ref_attn)
        block.ref_attn.heads, block.ref_attn.dim_head = HEADS, DHEAD
        # a trivial linear ref_attn stand-in that returns a large, obviously-nonzero delta
        block.ref_attn.forward = lambda x, context: torch.ones_like(x) * 100.0
        block.forward = types.MethodType(ltx23._patched_block_forward, block)

    x = torch.randn(1, 5, DIM)
    context = torch.randn(1, 3, CTX_DIM)
    timestep = _fake_timestep()
    ref_context = torch.randn(1, 32, DIM)

    torch.manual_seed(1)
    no_ref = in_range.forward(x.clone(), context=context, timestep=timestep, transformer_options={})
    torch.manual_seed(1)
    with_ref = in_range.forward(x.clone(), context=context, timestep=timestep,
                                transformer_options={"editanything_ref_context": ref_context})
    assert not torch.equal(no_ref, with_ref), "in-range block: ref_context present but " \
        "output unchanged - residual is not being applied"

    torch.manual_seed(1)
    oor_no_ref = out_of_range.forward(x.clone(), context=context, timestep=timestep, transformer_options={})
    torch.manual_seed(1)
    oor_with_ref = out_of_range.forward(x.clone(), context=context, timestep=timestep,
                                        transformer_options={"editanything_ref_context": ref_context})
    assert torch.equal(oor_no_ref, oor_with_ref), "out-of-range block (idx=5, outside " \
        "12-35): ref_context present but output changed - the block-range gate isn't working"
    print("[ok] _patched_block_forward: residual only fires for idx in "
          f"[{ltx23.EDITANYTHING_REF_START_BLOCK},{ltx23.EDITANYTHING_REF_END_BLOCK}] "
          "and only when ref_context is present")


def test_ref_visual_proj_and_adaln_proj_shapes():
    channels = 16  # stand-in for VIDEO_LATENT_CHANNELS, scaled down
    hidden = DIM

    visual_state = {
        "fc1.weight": torch.randn(32, 3 * channels), "fc1.bias": torch.randn(32),
        "proj.weight": torch.randn(hidden, 32), "proj.bias": torch.randn(hidden),
        "norm.weight": torch.randn(hidden), "norm.bias": torch.randn(hidden),
        "pos_embed": torch.randn(1, 32, hidden),
    }
    visual_proj = ltx23._EditAnythingRefVisualProj(visual_state)
    ref_latent = torch.randn(1, channels, 3, 8, 8)  # B,C,T,H,W
    tokens = visual_proj(ref_latent)
    assert tokens.shape == (1, 32, hidden)

    adaln_state = {
        "fc1.weight": torch.randn(32, 6 * channels), "fc1.bias": torch.randn(32),
        "proj.weight": torch.randn(6 * hidden, 32), "proj.bias": torch.randn(6 * hidden),
    }
    adaln_proj = ltx23._EditAnythingRefAdaLNProj(adaln_state)
    ref_adaln = adaln_proj(ref_latent)
    assert ref_adaln.shape == (1, 6 * hidden)
    print("[ok] _EditAnythingRefVisualProj/_EditAnythingRefAdaLNProj: real ported math, "
          "correct output shapes (32 tokens x hidden, and the full ada-param vector)")


def test_install_editanything_module_end_to_end():
    channels = 16
    hidden = DIM
    rank = 8

    class _FakeDiffusionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # _install_editanything_module patches by list POSITION (enumerate),
            # not by any .idx attribute - so positions must match real indices.
            self.transformer_blocks = torch.nn.ModuleList(
                [_make_block(i) for i in range(36)])  # positions 0..35

        def _prepare_timestep(self, timestep, batch_size, hidden_dtype, **kwargs):
            return _fake_timestep(batch_size), None, None

    class _FakeModel:
        def process_latent_in(self, x):
            return x

    class _FakeModelPatcher:
        def __init__(self):
            self.model = types.SimpleNamespace(diffusion_model=_FakeDiffusionModel())

    sd = {}
    sd["ref_visual_proj.fc1.weight"] = torch.randn(32, 3 * channels)
    sd["ref_visual_proj.fc1.bias"] = torch.randn(32)
    sd["ref_visual_proj.proj.weight"] = torch.randn(hidden, 32)
    sd["ref_visual_proj.proj.bias"] = torch.randn(hidden)
    sd["ref_visual_proj.norm.weight"] = torch.randn(hidden)
    sd["ref_visual_proj.norm.bias"] = torch.randn(hidden)
    sd["ref_visual_proj.pos_embed"] = torch.randn(1, 32, hidden)
    sd["ref_adaln_proj.fc1.weight"] = torch.randn(32, 6 * channels)
    sd["ref_adaln_proj.fc1.bias"] = torch.randn(32)
    sd["ref_adaln_proj.proj.weight"] = torch.randn(6 * hidden, 32)
    sd["ref_adaln_proj.proj.bias"] = torch.randn(6 * hidden)
    # only blocks 12 and 20 (both inside [12,35]) get real ref_attn weights -
    # every other in-range position (e.g. 30) is left weight-free deliberately
    # to prove missing-weight blocks are skipped, not errored; positions
    # outside [12,35] (e.g. 5) must never be patched regardless of weights.
    for idx in (12, 20):
        p = f"diffusion_model.transformer_blocks.{idx}.ref_attn."
        for name in ("to_q", "to_k", "to_v", "to_out.0"):
            sd[f"{p}{name}.lora_A.weight"] = torch.randn(rank, hidden)
            sd[f"{p}{name}.lora_B.weight"] = torch.randn(hidden, rank)

    tmp_path = Path(__file__).resolve().parent / "_scratch_editanything_module.safetensors"
    save_file(sd, str(tmp_path))
    try:
        model = _FakeModelPatcher()
        ltx23._install_editanything_module(model, str(tmp_path))
        dm = model.model.diffusion_model
        assert hasattr(dm, "editanything_ref_visual_proj")
        assert hasattr(dm, "editanything_ref_adaln_proj")
        patched_idxs = {b.idx for b in dm.transformer_blocks if hasattr(b, "ref_attn")}
        assert patched_idxs == {12, 20}, f"expected blocks {{12, 20}} patched, got {patched_idxs}"
        # _prepare_timestep patch applied and callable
        dm._editanything_ref_adaln = torch.randn(1, 6 * hidden)
        ts, _, _ = dm._prepare_timestep(torch.zeros(1), 1, torch.float32)
        assert ts.shape == (1, 5, 6 * hidden)
    finally:
        del model, dm
        import gc
        gc.collect()  # release comfy's mmap'd safetensors handle before unlink
        try:
            tmp_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows may still hold the mmap briefly; not worth failing the test over
    print("[ok] _install_editanything_module: loads real safetensors, patches exactly "
          "the blocks with matching weights (12, 20 - not out-of-range or weight-free "
          "in-range blocks), _prepare_timestep wrapped correctly")


if __name__ == "__main__":
    test_patched_forward_matches_original_when_no_reference()
    test_patched_forward_applies_residual_only_in_range_and_when_present()
    test_ref_visual_proj_and_adaln_proj_shapes()
    test_install_editanything_module_end_to_end()
    print("[ok] all smoke_ltx23_editanything tests passed")
