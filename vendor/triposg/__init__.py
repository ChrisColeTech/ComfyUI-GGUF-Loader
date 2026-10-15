# MIT (VAST-AI-Research/TripoSG). Pure-PyTorch port from the owner's triposg-poc
# (D:\Projects\poc\triposg-poc, src/pipelines/triposg_v1), as vendored by
# giga-models (src/sidecar/vendor/pipelines/triposg_v1).
"""TripoSG image-to-mesh, pure PyTorch: no diffusers / transformers at runtime.

Copied verbatim from the POC (``models/``, ``ops/geometry.py``,
``ops/scheduler.py``, ``preprocessing.py``) and from giga-models
(``briarmbg.py``, the RMBG-1.4 architecture with a local safetensors loader).
The only edit to a copied file: ``models/transformer/attention.py`` drops the
giga-models sidecar's pluggable-attention import and calls
``F.scaled_dot_product_attention`` directly, which is byte-identical to the
POC's own file. ``pipeline.py`` and ``loaders.py`` are this pack's: the POC's
``TripoSGPipeline.__call__`` split into its encode / sample / extract stages so
ComfyUI can run them as separate nodes, and loaders that read ComfyUI model
folders through ``comfy.utils.load_torch_file``.
"""
