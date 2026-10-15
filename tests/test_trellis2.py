"""CPU-light checks of the TRELLIS.2 / Pixal3D glue and vendored sidecar stages."""
import numpy as np
import pytest
import torch

from vendor.trellis2 import stage_files as sf


@pytest.mark.parametrize("name,stage", [
    ("trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", "sparse_structure"),
    ("pixal3d/split/Sparse/ss_flow_img_dit_1_3B_64_bf16.gguf", "sparse_structure"),
    ("trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", "shape_512"),
    ("pixal3d/split/shape/slat_flow_img2shape_dit_1_3B_1024_bf16.gguf", "shape_1024"),
    ("trellis2/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors", "texture"),
])
def test_stage_files_are_recognised_by_name(name, stage):
    hits = [s for s in sf.DIT_STAGES if sf.matches(sf.DIT_STAGES, s, name)]
    assert hits == [stage]


def test_texture_512_is_not_offered_for_the_1024_cascade():
    assert not sf.matches(sf.DIT_STAGES, "texture", "slat_flow_imgshape2tex_dit_1_3B_512_bf16.gguf")
    assert sf.matches(sf.DECODER_STAGES, "structure_decoder", "ss_dec_conv3d_16l8_fp16.safetensors")
    assert sf.matches(sf.DECODER_STAGES, "texture_decoder", "tex_dec_next_dc_f16c32_fp16.safetensors")


def test_merge_prefixes_and_drops_recomputed_buffers():
    t = torch.zeros(1)
    merged = sf.merge({"sparse_structure": {"blocks.0.x": t, "rope_phases": t},
                       "shape_512": {"blocks.0.x": t},
                       "shape_1024": {"blocks.0.x": t},
                       "texture": {"blocks.0.x": t}}, sf.DIT_STAGES)
    assert sorted(merged) == ["img2shape.blocks.0.x", "img2shape_512.blocks.0.x",
                              "shape2txt.blocks.0.x", "structure_model.blocks.0.x"]
    dec = sf.merge({"structure_decoder": {"a": t}, "shape_decoder": {"b": t}}, sf.DECODER_STAGES)
    assert sorted(dec) == ["shape_dec.b", "struct_dec.a"]


def test_pixal3d_is_detected_from_projection_heads():
    assert sf.is_pixal3d({"blocks.0.cross_attn.proj_linear.weight": 0})
    assert not sf.is_pixal3d({"blocks.0.cross_attn.to_q.weight": 0})


def _sphere(r=24, radius=9):
    zz, yy, xx = np.mgrid[:r, :r, :r]
    return ((zz - r // 2) ** 2 + (yy - r // 2) ** 2 + (xx - r // 2) ** 2) < radius ** 2


@pytest.mark.parametrize("algorithm", ["morphological_closing", "flood_fill"])
def test_structure_fill_and_shell(algorithm):
    pytest.importorskip("scipy")
    from vendor.trellis2.structure import fill_holes
    arr = _sphere()[None].copy()
    arr[0, 12, 12, 12] = False
    # a 1-voxel closing kernel is a no-op (sidecar default 1); 3 closes a 1-voxel hole
    out = fill_holes(arr, algorithm, hole_structure=3, keep_only_shell=False)
    assert out[0, 12, 12, 12]  # the enclosed hole is filled
    shell = fill_holes(arr, algorithm, hole_structure=3, keep_only_shell=True)
    assert not shell[0, 12, 12, 12] and shell[0].sum() < out[0].sum()
    assert shell[0, 12, 12, 4]  # surface voxel (radius 9) kept


def test_structure_remove_small_holes_matches_the_sidecar_default():
    pytest.importorskip("skimage")
    from vendor.trellis2.structure import fill_holes
    arr = _sphere()[None].copy()
    arr[0, 12, 12, 12] = False
    out = fill_holes(arr)  # remove_small_holes, structure 1, shell on
    assert out[0, 12, 12, 4] and not out[0, 12, 12, 12]
    with pytest.raises(RuntimeError, match="No voxels remain"):
        fill_holes(np.zeros((1, 8, 8, 8), bool))


def test_prepare_image_crop_matches_the_sidecar_arithmetic():
    Image = pytest.importorskip("PIL.Image")
    from vendor.trellis2.preprocess import preprocess_image, to_rgba
    rgb = np.zeros((300, 400, 3), np.float32)
    rgb[100:200, 150:230] = (0.2, 0.6, 1.0)
    alpha = np.zeros((300, 400), np.float32)
    alpha[100:200, 150:230] = 1.0
    rgba = to_rgba(rgb, alpha)
    a = preprocess_image(rgba, "trellis2")
    b = preprocess_image(rgba, "pixal3d")
    assert a.size == (98, 98) and b.size == (108, 108)
    assert a.getpixel((0, 0)) == (0, 0, 0) and a.getpixel((49, 49))[2] == 255
    big = to_rgba(np.ones((2048, 1024, 3), np.float32), np.ones((2048, 1024), np.float32))
    assert preprocess_image(big, "trellis2").size[1] <= 1024  # 1024 cap
    with pytest.raises(ValueError):
        preprocess_image(to_rgba(rgb, np.zeros_like(alpha)), "trellis2")
    with pytest.raises(ValueError):
        preprocess_image(rgba, "other")
    assert isinstance(a, Image.Image)
