# Open PR backports (city96/ComfyUI-GGUF)

Upstream is slow to merge. This fork selectively ports **high-value open PRs**
from https://github.com/city96/ComfyUI-GGUF/pulls into `main`.

## Ported

| PR | Title | Files | Notes |
|----|-------|-------|-------|
| [#472](https://github.com/city96/ComfyUI-GGUF/pull/472) | Cache device-side constants in dequant kernels | `dequant.py` | ~1.4–3× dequant; LTX-class speedups |
| [#433](https://github.com/city96/ComfyUI-GGUF/pull/433) | Torch dequant for IQ1/IQ2/IQ3 | `dequant.py` | Unsloth UD / smaller TE quants |
| [#470](https://github.com/city96/ComfyUI-GGUF/pull/470) | QK-norm `.scale`→`.weight` | `loader.py` | Fixes silent NaN/black Flux-compat GGUFs |
| [#467](https://github.com/city96/ComfyUI-GGUF/pull/467) | Dequant bare nn.Parameters | `loader.py` | LTX-2 `learnable_registers` etc. |
| [#392](https://github.com/city96/ComfyUI-GGUF/pull/392) | Lumina2 / Z-Image pad token shape | `loader.py` | `x_pad_token` / `cap_pad_token` 1D→2D |
| [#456](https://github.com/city96/ComfyUI-GGUF/pull/456) | Float `dtype` for quantized GGMLTensor | `ops.py` | MPS / Ideogram-4 `x.to(weight.dtype)` |
| [#468](https://github.com/city96/ComfyUI-GGUF/pull/468) | `GGMLTensor.dequantize()` | `ops.py` | Core generic cast path |
| [#461](https://github.com/city96/ComfyUI-GGUF/pull/461) | WeightAdapters in `move_patch_to_device` | `ops.py` | Soft-deps `comfy.weight_adapter` |
| [#469](https://github.com/city96/ComfyUI-GGUF/pull/469) | Force patch on partial load/unload | `nodes.py` | LowVRAM correctness |
| [#440](https://github.com/city96/ComfyUI-GGUF/pull/440) / [#436](https://github.com/city96/ComfyUI-GGUF/pull/436) | `mistral3` TE | `loader.py` | Ministral-3 (Flux2 TE) |
| [#438](https://github.com/city96/ComfyUI-GGUF/pull/438) | `qwen35` TE | `loader.py` | Allowlist + clip path |
| [#455](https://github.com/city96/ComfyUI-GGUF/pull/455) / [#460](https://github.com/city96/ComfyUI-GGUF/pull/460) | Ideogram-4 arch | already in fork | Detectors merged |
| [#473](https://github.com/city96/ComfyUI-GGUF/pull/473) | MiniMax-H3 mmproj (partial) | `loader.py` | Qwen3-VL deepstack mmproj map only |

Also local (not from upstream PRs): DiT allowlist for all arch tags under
`D:\models\image-models` (`ltx2`, `zimage`, `ideogram4`, `flux2`, …) plus
`qwen2` TE (issue [#397](https://github.com/city96/ComfyUI-GGUF/issues/397)).

## Explicitly skipped

| PR | Why |
|----|-----|
| [#473](https://github.com/city96/ComfyUI-GGUF/pull/473) full | Large rewrite (`LazyGGUFReader`, `quant_ops.py`, dynamic path) — cherry-picked mmproj only |
| [#459](https://github.com/city96/ComfyUI-GGUF/pull/459) | Mega dump (IDE files, UI zip, unrelated docs) |
| [#445](https://github.com/city96/ComfyUI-GGUF/pull/445) | Depends on newer Comfy `eject_model` / hook APIs |
| [#336](https://github.com/city96/ComfyUI-GGUF/pull/336) draft | Triton dequant framework — large, draft |
| [#252](https://github.com/city96/ComfyUI-GGUF/pull/252) draft | Auto-convert / Gradio space WIP |
| Registry / Gradio / convert-GUI PRs | Out of scope for sidecar loader |
| [#412](https://github.com/city96/ComfyUI-GGUF/pull/412) | Mixed large loader rewrite — needs careful re-review |

## Re-check

When upstream merges any of the above, drop the corresponding notes and
rebase onto `upstream/main` if conflicts appear.
