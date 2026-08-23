# FP8 safetensors → GGUF

Two steps: dequantize the FP8 checkpoint into an F16 GGUF tagged with an arch
this repo's `loader.py` accepts, then hand that to a patched `llama-quantize`.

```
python scripts/fp8_to_gguf/01_fp8_to_f16_gguf.py \
    --src D:/models/.../model_fp8.safetensors \
    --dst D:/models/.../_work/model-F16.gguf

python scripts/fp8_to_gguf/02_quantize.py \
    D:/models/.../_work/model-F16.gguf \
    D:/models/.../ \
    --types Q4_K_M Q6_K Q8_0
```

Step 1 reuses `tools/convert.py` for arch detection, `keys_hiprec`, the
1D→F32 rule and the name-length limit, so the output matches what the loader
in this repo expects. Everything it adds is the FP8 handling that
`convert.py` has no notion of.

## FP8 layouts handled

| layout | looks like | dequant |
| --- | --- | --- |
| unscaled | `<key>` is `float8_e4m3fn`, no companion tensor | cast to f16 |
| scaled (ComfyUI `comfy_quant`) | `<key>`, `<key>_scale` (f32 scalar), `<layer>.comfy_quant` (uint8 JSON) | `fp8.to(f16) * scale` |

`_scale` and `comfy_quant` tensors are dropped from the GGUF. A `_scale`
suffix only counts as a helper when the base key exists, so genuine weights
like `attn.qknorm.qnorm.scale` survive.

Dequantizing to f16 rather than bf16 is deliberate: the largest magnitude
either of these models reaches after scaling is ~6, far inside f16 range, and
f16 keeps `convert.py`'s "bf16/f32 1D → F32" rule from firing on weights that
should be quantized.

Mixed checkpoints (DiT + VAE in one file) are narrowed by `convert.py`'s
`strip_prefix()`: with a `model.diffusion_model.` prefix present, everything
outside it — including `vae.*` — is dropped.

## NaN sanitization

`llama-quantize` rejects an entire file with
`tensor '<name>' has invalid data` if one non-finite value survives. Step 1
zeroes non-finite values and logs each affected tensor loudly, because a NaN
in a checkpoint is merge damage rather than signal. Check that warning before
trusting the output — see the note on `norm_final.weight` below.

## Quantizer

`02_quantize.py` prefers `D:/Projects/llama.cpp/bin_cuda120`, which carries
City96's image-model patch (`tools/lcpp.patch`). The stock build at
`D:/Projects/llama.cpp/bin` fails immediately with
`unknown model architecture: 'zimage'`.

## Models converted with this

### `z_image_turbo_uncensored_fp8.safetensors` → arch `zimage`

698 tensors: 454 DiT (`model.diffusion_model.`) + 244 VAE (dropped).
210 unscaled FP8 weights.

Its key layout is the single-resolution repack (`x_embedder`, `final_layer`),
not the bucketed `all_x_embedder.2-1` / `all_final_layer.2-1` form the
existing `ModelZImage` template matched — it was falling through to
`lumina2`. `ModelZImage.keys_detect` gained a tuple keyed on
`cap_pad_token` + `x_pad_token`, which is what separates a Z-Image checkpoint
from a real Lumina-2 one on the shared code path (ComfyUI
`model_detection.py`, `dim == 3840` branch).

**`norm_final.weight` in this checkpoint holds 856 NaNs out of 3840.** The
tensor is inert: ComfyUI's Z-Image path has `self.norm_final` commented out
in `comfy/ldm/lumina/model.py`, and the upstream FP8 release does not carry
the key at all. It is merge residue. Step 1 zeroes it.

### `Krea2_turbo_uncensored_fp8.safetensors` → arch `qwen_image`

942 tensors: 256 scaled FP8 weights with their `_scale` and `comfy_quant`
companions (512 helpers dropped), 174 bf16. No non-finite values.
Matches the Krea-2 tuple already in `ModelQwenImage.keys_detect`.
