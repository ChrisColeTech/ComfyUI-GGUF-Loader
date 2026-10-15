# MIT (microsoft/TRELLIS.2, TencentARC/Pixal3D). Vendored from the owner's giga-models
# sidecar: src/sidecar/vendor/pipelines/trellis2_v1 and pixal3d_v1.
"""The owner's TRELLIS.2 / Pixal3D pipeline stages that ComfyUI core does not have.

ComfyUI 0.39 ships a pure-PyTorch TRELLIS.2 + Pixal3D implementation
(``comfy/ldm/trellis2``: the flow DiTs, structure/shape/texture decoders,
DINOv3 + NAF conditioning, cascade upsampling, mesh decode, texture bake).
The owner's per-stage checkpoints load into it key-for-key (checked tensor
by tensor, see AGENTS.md), so the pack's nodes load those files into core's
models and core's stage nodes run the networks.

The sidecar's own model code (``trellis2_v1/trellis2``, ``pixal3d_v1/pixal3d``)
is NOT vendored: it needs CUDA-only extension wheels the ComfyUI portable does
not have (flex_gemm/triton sparse conv, o_voxel and cumesh mesh ops,
flash_attn/xformers sparse attention). What IS vendored is the behaviour of the
owner's validated runs that core's nodes do not reproduce:

* ``preprocess.py`` - ``Trellis2ImageTo3DPipeline.preprocess_image`` (TRELLIS.2,
  pad 1.0) and ``Pixal3DImageTo3DPipeline.preprocess_image`` (Pixal3D, pad 1.1):
  1024 cap, alpha > 0.8 bounding box, centred square crop, premultiplied
  composite on the background.
* ``structure.py`` - the hole-filling / shell options of
  ``Trellis2ImageTo3DPipeline.sample_sparse_structure``, which the sidecar's
  TRELLIS.2 engine runs with ``fill_holes=True, hole_iterations=1,
  hole_fill_algorithm="remove_small_holes", keep_only_shell=True``.
* ``stage_files.py`` - this pack's own glue: which per-stage file is which
  stage, and the key prefixes core's merged model expects.
"""
