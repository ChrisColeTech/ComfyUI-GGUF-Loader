"""Pure-torch beam-sample decoding (transformers.generate replacement).

Replicates the exact algorithm HF's generate() runs for UniRig's skeleton
generation settings — beam sampling (num_beams>1, do_sample=True) with
repetition penalty, a constraint mask (VocabSwitching), temperature, top-k,
top-p, and BeamSearchScorer bookkeeping (length_penalty=1.0,
early_stopping=False, batch of 1, one returned sequence).

Processor/warper order matches HF: log_softmax -> repetition penalty ->
constraint mask -> + beam scores -> temperature -> top-k -> top-p -> softmax
-> multinomial(2*num_beams) per step. Greedy (num_beams=1, do_sample=False)
is token-identical to transformers (see tests); the sampled path is
algorithmically identical but draws from a different RNG stream, so outputs
are equivalent-quality rather than bitwise-equal.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch.nn import functional as F


def _apply_repetition_penalty(scores: torch.Tensor, generated: List[List[int]], penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return scores
    for b, seq in enumerate(generated):
        if not seq:
            continue
        idx = torch.tensor(sorted(set(seq)), device=scores.device)
        vals = scores[b, idx]
        scores[b, idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return scores


def _warp(scores: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature != 1.0:
        scores = scores / temperature
    if top_k and top_k > 0:
        kth = torch.topk(scores, min(top_k, scores.shape[-1]))[0][..., -1, None]
        scores = scores.masked_fill(scores < kth, -float("inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=False)
        cum = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        # HF TopPLogitsWarper: remove tokens with cumulative prob <= 1 - top_p
        # (keeps at least 1 token; min_tokens_to_keep=1).
        remove = cum <= (1.0 - top_p)
        remove[..., -1] = False
        mask = remove.scatter(-1, sorted_idx, remove)
        scores = scores.masked_fill(mask, -float("inf"))
    return scores


class _Hypotheses:
    """BeamSearchScorer's finished-hypothesis pool (length_penalty=1.0)."""

    def __init__(self, num_beams: int, length_penalty: float = 1.0):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.pool: List[Tuple[float, List[int]]] = []

    def add(self, tokens: List[int], sum_logprob: float) -> None:
        score = sum_logprob / (max(len(tokens), 1) ** self.length_penalty)
        self.pool.append((score, tokens))
        self.pool.sort(key=lambda x: -x[0])
        del self.pool[self.num_beams:]

    def worst_score(self) -> float:
        return self.pool[-1][0] if len(self.pool) >= self.num_beams else -float("inf")

    def is_done(self, best_possible: float) -> bool:
        # early_stopping=False heuristic from HF: done when the best running
        # score cannot beat the worst finished hypothesis.
        return len(self.pool) >= self.num_beams and self.worst_score() >= best_possible


@torch.no_grad()
def beam_sample(
    model,
    cond_embeds: torch.Tensor,
    *,
    constrain: Callable[[List[int]], torch.Tensor],
    eos_token_id: int,
    num_beams: int = 15,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    temperature: float = 1.5,
    top_k: int = 5,
    top_p: float = 0.95,
    repetition_penalty: float = 3.0,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Decode one sequence from an ``OPTDecoderLM`` given condition embeds.

    Args:
        model: OPTDecoderLM.
        cond_embeds: (1, L, H) prompt embeddings (mesh cond + start tokens).
        constrain: maps the generated-token list to a boolean allow-mask (V,)
            over the vocabulary (the VocabSwitching grammar).
        generator: torch RNG for reproducible sampling.

    Returns:
        Generated token ids (without the prompt), ending with eos when reached.
    """
    device = cond_embeds.device
    k = num_beams if do_sample or num_beams > 1 else 1

    # Preallocated in-place KV cache (fp16 on CUDA: a 15-beam fp32 cache at
    # the full token budget would be ~45 GB and spill VRAM — the PhaseB.16
    # slowdown). Prefill once with batch 1, then broadcast to all beams.
    cache_dtype = torch.float16 if device.type == "cuda" else torch.float32
    prompt_len = cond_embeds.shape[1]
    cache = model.make_cache(k, prompt_len + max_new_tokens + 1, device, dtype=cache_dtype)
    logits = model(cond_embeds, cache)
    cache.expand_beams(k)
    logits = logits.expand(k, -1, -1)

    beam_tokens: List[List[int]] = [[] for _ in range(k)]
    beam_scores = torch.zeros(k, device=device)
    if k > 1:
        beam_scores[1:] = -1e9  # all beams start identical; keep only one live
    hyps = _Hypotheses(num_beams)
    vocab = logits.shape[-1]

    for step in range(max_new_tokens):
        next_scores = F.log_softmax(logits[:, -1].float(), dim=-1)          # (k, V)
        next_scores = _apply_repetition_penalty(next_scores, beam_tokens, repetition_penalty)
        for b in range(k):
            allow = constrain(beam_tokens[b])
            next_scores[b] = next_scores[b].masked_fill(~allow, -float("inf"))

        if num_beams == 1:
            if do_sample:
                warped = _warp(next_scores, temperature, top_k, top_p)
                probs = warped.softmax(dim=-1)
                nxt = int(torch.multinomial(probs[0], 1, generator=generator))
            else:
                nxt = int(next_scores[0].argmax())
            beam_tokens[0].append(nxt)
            if nxt == eos_token_id:
                return beam_tokens[0]
            tok = model.embed(torch.tensor([[nxt]], device=device))
            logits = model(tok, cache)
            continue

        scores_sum = next_scores + beam_scores[:, None]                     # (k, V)
        if do_sample:
            warped = _warp(scores_sum, temperature, top_k, top_p)
            flat = warped.reshape(-1)                                       # (k*V,)
            probs = flat.softmax(dim=-1)
            cand = torch.multinomial(probs, 2 * k, generator=generator)
            cand_scores = scores_sum.reshape(-1)[cand]
            order = torch.argsort(cand_scores, descending=True)
            cand, cand_scores = cand[order], cand_scores[order]
        else:
            cand_scores, cand = torch.topk(scores_sum.reshape(-1), 2 * k)

        next_beams: List[Tuple[int, int, float]] = []  # (src_beam, token, score)
        for rank in range(cand.shape[0]):
            src = int(cand[rank] // vocab)
            tok_id = int(cand[rank] % vocab)
            score = float(cand_scores[rank])
            if tok_id == eos_token_id:
                if rank < k:  # only top-k eos candidates finalize (HF rule)
                    hyps.add(beam_tokens[src] + [eos_token_id], score)
                continue
            next_beams.append((src, tok_id, score))
            if len(next_beams) == k:
                break

        if not next_beams:
            break
        srcs = torch.tensor([b[0] for b in next_beams], device=device)
        beam_tokens = [beam_tokens[int(s)] + [t] for s, t, _ in next_beams]
        beam_scores = torch.tensor([s for _, _, s in next_beams], device=device)

        if (step + 1) % 100 == 0:
            import logging

            logging.getLogger("unirig").info(
                "beam_sample: step %d, best=%.2f, finished=%d", step + 1,
                float(beam_scores.max()), len(hyps.pool),
            )
        if hyps.is_done(float(beam_scores.max()) / (step + 1)) or step == max_new_tokens - 1:
            break

        # Reorder the KV cache in place to the surviving beams, advance a step.
        if not torch.equal(srcs, torch.arange(k, device=device)):
            cache.reorder(srcs)
        toks = torch.tensor([[b[1]] for b in next_beams], device=device)
        logits = model(model.embed(toks), cache)

    # Finalize any running beams (HF adds them un-eos'ed at max length).
    for b in range(len(beam_tokens)):
        hyps.add(beam_tokens[b], float(beam_scores[b]))
    return max(hyps.pool, key=lambda x: x[0])[1]
