"""Pure-torch skeleton generation: UniRigAR.generate without transformers.

Binds a replacement ``generate`` onto a loaded UniRigAR model spec that runs
the OPT decode through :mod:`opt_decoder` + :mod:`beam_sample` instead of
``transformers.generate``. The mesh encoder (Michelangelo) and tokenizer are
reused as-is; the HF transformer's weights are converted once and cached.

The replacement mirrors upstream ``UniRigAR.generate`` exactly: mesh cond ->
[bos, cls] start tokens -> constrained beam-sample decode -> prepend start
tokens -> tokenizer.detokenize.
"""

from __future__ import annotations

import logging
import types
from typing import Optional, Union

import torch

from .beam_sample import beam_sample
from .opt_decoder import OPTDecoderLM

log = logging.getLogger("unirig")


def _get_pure_decoder(model_spec) -> OPTDecoderLM:
    cached = getattr(model_spec, "_pure_decoder", None)
    if cached is None:
        carrier_decoder = getattr(model_spec.transformer, "_pure_decoder", None)
        if carrier_decoder is not None:
            # transformers-free path: the stub carrier already IS the pure
            # decoder (modules shared) — zero conversion, zero copy.
            cached = carrier_decoder
            log.info("pure-torch OPT decoder taken from stub carrier (shared weights)")
        else:
            state = {f"model.transformer.{k}": v for k, v in model_spec.transformer.state_dict().items()}
            cached = OPTDecoderLM.from_state_dict(state)
            log.info("pure-torch OPT decoder built from loaded transformer weights")
        model_spec._pure_decoder = cached
    return cached


@torch.no_grad()
def _pure_generate(self, vertices, normals, cls: Union[str, None] = None, **kwargs):
    """Replacement for UniRigAR.generate (single sample, pure-torch decode)."""
    cond = self.encode_mesh_cond(vertices=vertices, normals=normals)
    decoder = _get_pure_decoder(self).to(cond.device)
    if getattr(self.transformer, "_pure_decoder", None) is None:
        # Real HF transformer loaded: its GPU copy is dead weight during pure
        # decode — evict to CPU so the beam KV cache has headroom. (The stub
        # carrier SHARES the decoder's modules; evicting it would undo the
        # .to(device) above.)
        self.transformer.to("cpu")
    cond = cond.to(dtype=next(decoder.parameters()).dtype)

    start_tokens = [self.tokenizer.bos]
    if cls is not None:
        start_tokens.append(self.tokenizer.cls_name_to_token(cls=cls))
    start_ids = torch.tensor(start_tokens, device=cond.device)
    cond = torch.cat([cond, decoder.embed(start_ids.unsqueeze(0))], dim=1)

    vocab = decoder.lm_head.out_features
    start_list = list(start_tokens)

    def constrain(generated):
        import numpy as np

        ids = np.asarray(start_list + list(generated), dtype=np.int64)
        allowed = self.tokenizer.next_posible_token(ids=ids)
        mask = torch.zeros(vocab, dtype=torch.bool, device=cond.device)
        mask[torch.as_tensor(list(allowed), device=cond.device)] = True
        return mask

    seed = kwargs.pop("seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=cond.device)
        generator.manual_seed(int(seed))

    output_ids = beam_sample(
        decoder,
        cond,
        constrain=constrain,
        eos_token_id=self.tokenizer.eos,
        num_beams=int(kwargs.get("num_beams", 15)),
        max_new_tokens=int(kwargs.get("max_new_tokens", 2048)),
        do_sample=bool(kwargs.get("do_sample", True)),
        temperature=float(kwargs.get("temperature", 1.5)),
        top_k=int(kwargs.get("top_k", 5)),
        top_p=float(kwargs.get("top_p", 0.95)),
        repetition_penalty=float(kwargs.get("repetition_penalty", 3.0)),
        generator=generator,
    )

    import numpy as np

    full = np.asarray(start_list + list(output_ids), dtype=np.int64)
    return self.tokenizer.detokenize(ids=full)


def install_pure_decode(model_spec) -> None:
    """Rebind ``generate`` on a loaded UniRigAR instance to the pure decoder."""
    model_spec.generate = types.MethodType(_pure_generate, model_spec)
    log.info("UniRigAR.generate -> pure-torch beam-sample decode")
