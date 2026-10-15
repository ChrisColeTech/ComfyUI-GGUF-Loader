"""CPU-light checks of the vendored TripoSG stages (no weights, no ComfyUI).

Needs Pillow (always present in ComfyUI); skipped in an environment without it.
tools/smoke_triposg.py runs the same stage checks against the real ComfyUI.
"""
import types

import numpy as np
import pytest
import torch

Image = pytest.importorskip("PIL.Image")

from vendor.triposg import loaders, pipeline  # noqa: E402
from vendor.triposg.models.transformer import TripoSGDiTConfig, TripoSGDiTModel  # noqa: E402
from vendor.triposg.preprocessing import prepare_image  # noqa: E402

SCHED = {"num_train_timesteps": 1000, "shift": 1.0, "use_dynamic_shifting": False}


def _tiny_dit():
    torch.manual_seed(0)
    return TripoSGDiTModel(TripoSGDiTConfig(num_attention_heads=2, width=32, in_channels=8,
                                            num_layers=2, cross_attention_dim=16)).eval()


def test_sampler_is_seeded_and_batches_unconditional_first():
    dit = _tiny_dit()
    seen = []
    forward = dit.forward

    def spy(x, t, encoder_hidden_states=None, **kw):
        seen.append(encoder_hidden_states.clone())
        return forward(x, t, encoder_hidden_states=encoder_hidden_states, **kw)

    dit.forward = spy
    cond = torch.randn(1, 5, 16)

    def run(seed, guidance=7.0):
        return pipeline.sample_latents(
            dit, loaders.make_scheduler(SCHED), cond, torch.zeros_like(cond),
            num_inference_steps=3, guidance_scale=guidance, num_tokens=12,
            generator=torch.Generator().manual_seed(seed), device=torch.device("cpu"))

    a, b, c = run(1), run(1), run(2)
    assert a.shape == (1, 12, 8)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert seen[0].shape[0] == 2
    assert not seen[0][0].any() and torch.equal(seen[0][1], cond[0])
    seen.clear()
    run(1, guidance=1.0)
    assert seen[0].shape[0] == 1


def test_sampler_reports_every_step_and_can_be_interrupted():
    dit = _tiny_dit()
    cond = torch.randn(1, 5, 16)
    steps = []
    pipeline.sample_latents(dit, loaders.make_scheduler(SCHED), cond, torch.zeros_like(cond),
                            num_inference_steps=4, num_tokens=8,
                            generator=torch.Generator().manual_seed(0),
                            device=torch.device("cpu"), step_callback=lambda d, t: steps.append(d))
    assert steps == [1, 2, 3, 4]

    def stop(done, total):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        pipeline.sample_latents(dit, loaders.make_scheduler(SCHED), cond, torch.zeros_like(cond),
                                num_inference_steps=4, num_tokens=8,
                                generator=torch.Generator().manual_seed(0),
                                device=torch.device("cpu"), step_callback=stop)


def test_validate_geometry_rejects_what_the_poc_rejects():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    out_v, out_f = pipeline.validate_geometry(v, f)
    assert out_v.dtype == np.float32 and out_f.shape == (2, 3)
    with pytest.raises(RuntimeError, match="extraction failed"):
        pipeline.validate_geometry(None, None)
    with pytest.raises(RuntimeError, match="out-of-range"):
        pipeline.validate_geometry(v, np.array([[0, 1, 9]]))
    with pytest.raises(RuntimeError, match="non-finite"):
        pipeline.validate_geometry(np.array([[np.nan, 0, 0]] * 3), np.array([[0, 1, 2]]))
    with pytest.raises(RuntimeError, match="non-positive extents"):
        pipeline.validate_geometry(np.zeros((3, 3)), np.array([[0, 1, 2]]))


def test_merge_vertices_matches_trimesh_process():
    rs = np.random.RandomState(3)
    v = rs.rand(40, 3).astype(np.float32)
    v[7] = v[3]                       # exact duplicate
    v[9] = v[3] + 1e-10               # equal at 8 decimals
    f = rs.randint(0, 39, (60, 3))     # vertex 39 unreferenced
    mv, mf = pipeline.merge_vertices(v, f)
    assert len(mv) < 40 and mf.max() < len(mv)
    assert np.allclose(mv[mf], v[f], atol=1e-7)  # every face keeps its corner positions
    trimesh = pytest.importorskip("trimesh")
    ref = trimesh.Trimesh(v, f)
    assert np.array_equal(np.asarray(ref.vertices, np.float32), mv)
    assert np.array_equal(np.asarray(ref.faces), mf)


def test_extract_mesh_on_an_analytic_sphere():
    pytest.importorskip("skimage")

    class Sphere:
        def decode(self, z, sampled_points):
            r = sampled_points.float().norm(dim=-1, keepdim=True)
            return types.SimpleNamespace(sample=(0.6 - r) * 20.0)

    v, f = pipeline.extract_mesh(Sphere(), torch.zeros(1, 4, 4), torch.device("cpu"),
                                 dense_octree_depth=5, hierarchical_octree_depth=6)
    assert abs(np.linalg.norm(v, axis=1).mean() - 0.6) < 0.02


def test_prepare_image_uses_meaningful_alpha_without_rmbg():
    rgba = np.zeros((120, 100, 4), np.uint8)
    rgba[30:90, 20:60] = (10, 200, 30, 255)  # opaque 40x60 box, rest transparent
    out = prepare_image(Image.fromarray(rgba, "RGBA"), rmbg=None, device="cpu")
    assert out.mode == "RGB" and out.size == (72, 72)  # 60 + 2 * int(6.0)
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((36, 36)) == (10, 200, 30)


def test_folder_detection(tmp_path):
    root = tmp_path / "TripoSG"
    for rel in loaders.REQUIRED_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}")
    assert loaders.is_triposg_dir(root)
    (root / "text_encoder").mkdir()
    assert not loaders.is_triposg_dir(root)  # the scribble variant is a different pipeline
    rmbg = tmp_path / "RMBG-1.4"
    rmbg.mkdir()
    (rmbg / "model.safetensors").write_bytes(b"")
    (rmbg / "config.json").write_text('{"architectures": ["BriaRMBG"]}')
    assert loaders.is_rmbg_dir(rmbg) and not loaders.is_rmbg_dir(root)
