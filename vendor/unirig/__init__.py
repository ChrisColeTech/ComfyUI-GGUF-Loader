# GPL-3.0 (see LICENSE here). From PozzettiAndrea/ComfyUI-UniRig's inference-only
# ``nodes/unirig`` package (itself derived from VAST-AI-Research/UniRig, MIT), as
# vendored and extended by the owner's unirig_v1 pipeline
# (D:\Projects\poc\hymotion-unirig-poc, giga-models src/sidecar/vendor/pipelines/unirig_v1).
"""UniRig auto-rigging: mesh -> skeleton (autoregressive) -> skin weights (PTv3).

Provenance, file by file:

* upstream ComfyUI-UniRig ``nodes/unirig`` (verbatim unless noted):
  ``attention, configs, data_order, data_spec, data_vertex_group,
  michelangelo_encoder, model_spec, parse_encoder, ptv3_encoder, serialization,
  tokenizer_parse, tokenizer_part, tokenizer_spec, unirig_skin``.
  This package already ran inside ComfyUI, so it imports the real ``comfy.ops``
  / ``comfy.utils`` / ``comfy.ldm.modules.attention`` (the POC needed shims).
* the owner's unirig_v1 ``ops/`` (verbatim, imports made relative):
  ``opt_decoder, beam_sample, pure_loader, pure_skeleton`` (transformers-free
  skeleton decode), ``bone_names`` (vroid -> ``mixamorig:*``), ``skin_post``
  (upstream reskin edge diffusion), ``native_fallbacks`` (pure-torch
  ``fps`` / ``segment_csr`` / varlen attention).
* this pack: ``sparse_conv`` (pure-torch ``SubMConv3d``), ``rig`` (the POC
  wrapper's ``rig()`` split into skeleton / skin stages) and ``glb`` (the POC's
  ``compose_v1.gltf_export.export_skinned_glb`` without pygltflib).

Edits to copied files, each marked ``CCTech`` in place: ``ptv3_encoder``,
``unirig_skin`` and ``michelangelo_encoder`` import the pure-torch fallbacks
instead of spconv / torch_scatter / torch_cluster / comfy_sparse_attn;
``data_order.Order.make_names`` carries the POC's tolerant-truncation patch.
Not vendored: ``unirig_ar`` / ``model_parse`` (the transformers OPT path the
pure decoder replaces), ``download`` (hub fetch) and the bpy/FBX modules.
"""
