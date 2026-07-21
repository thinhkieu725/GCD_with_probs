"""Word-synchronized beam search fusing a word-level GF grammar with an LM.

Phase-2 counterpart of `GFProbConstraint`: instead of stepping one LM token at
a time over a token-ID grammar, the beam advances one *grammar word* at a time
over the natural word-level grammar (no tokenizer-specific rewriting, works
with any tokenizer):

- the grammar proposes candidate next words with probabilities
  (`GFProbRuntime.next_token_logprobs`),
- the LM scores each candidate word as the sum of the log-probabilities of
  its subword tokens (one batched forward pass per *hypothesis* per step,
  reusing that hypothesis's KV cache — see below),
- hypotheses grow by `lambda_gf * ln P_GF + (1 - lambda_gf) * ln P_LM` and the
  top `beam_size` survive.

Each surviving hypothesis carries its own KV cache (`past_key_values`) for
"prompt + hyp.words", plus the logits row predicting whatever comes right
after it (`next_start_logits`). At each step, per hypothesis:

1. The candidate words' own tokens are the *only* new input fed to the model;
   the shared prefix is already baked into the cache (`batch_repeat_interleave`
   fans the one cached prefix out across all of that hypothesis's candidates
   without recomputing it).
2. `ln P(candidate's first token | prefix)` comes from the *previous* step's
   `next_start_logits` (a plain lookup, no forward pass needed) — a fresh
   forward pass only ever scores tokens *after* the first one of a candidate.
3. Once global pruning picks the survivors for this step, each survivor's own
   row is sliced out of that batched result (cropped to its real token count,
   dropping any padding used to batch same-step candidates of differing
   subword length) and becomes its cache for the next step.

This avoids ever re-encoding the growing "prompt + words so far" prefix from
scratch — the previous design's `_score_expansions` did that once per
candidate, per step (see git history), which made the LM side of this search
scale roughly quadratically with `max_words`.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import DynamicCache

from gf_gcd.gf_probs import GF_EOS, GFProbRuntime


@dataclass
class WordHypothesis:
    words: List[str] = field(default_factory=list)
    score: float = 0.0  # cumulative fused score
    gf_logprob: float = 0.0  # cumulative ln P_GF
    llm_logprob: float = 0.0  # cumulative ln P_LM
    finished: bool = False
    # incremental-decoding state for "prompt + words"; None only for
    # `finished` hypotheses, which never generate further.
    past_key_values: Optional[DynamicCache] = None
    next_start_logits: Optional[torch.Tensor] = None  # logits predicting the token right after `words`

    @property
    def text(self) -> str:
        return " ".join(self.words)


@dataclass
class _Expansion:
    """One (hypothesis, candidate word) pair, already scored, carrying its
    own (small, pre-cropped) KV-cache slice in case it survives pruning."""

    hyp: WordHypothesis
    word: str
    gf_lp: float
    llm_lp: float
    cache: DynamicCache  # batch-1, cropped to exactly this candidate's real length
    next_start_logits: torch.Tensor  # (vocab,) — logits predicting what comes after `word`


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    legacy = cache.to_legacy_cache()
    return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in legacy))


def _extract_row_cache(cache: DynamicCache, row: int, length: int) -> DynamicCache:
    """Slice out one batch row, cropped to `length`, as a fresh batch-1 cache."""
    legacy = cache.to_legacy_cache()
    new_legacy = tuple(
        (k[row : row + 1, :, :length, :].clone(), v[row : row + 1, :, :length, :].clone())
        for k, v in legacy
    )
    return DynamicCache.from_legacy_cache(new_legacy)


class WordBeamSearcher:
    """Beam search over grammar words, scored jointly by grammar and LM."""

    def __init__(
        self,
        model,
        tokenizer,
        runtime: GFProbRuntime,
        lambda_gf: float = 0.5,
        beam_size: int = 4,
        gf_temperature: float = 1.0,
        prefix_space: bool = True,
        scoring_chunk_size: int = 64,
    ):
        """
        Args:
            model / tokenizer: a HF causal LM and its tokenizer (e.g. the
                `.model` / `.tokenizer` of a `ConstrainedGenerator`).
            runtime: probability runtime over the *word-level* .pgf.
            lambda_gf: interpolation weight, score = λ·lnP_GF + (1-λ)·lnP_LM.
            prefix_space: encode candidate words as " word" (mid-sentence
                position), matching how subword tokenizers see them.
            scoring_chunk_size: max candidates scored in one forward pass
                (see `_score_hyp_candidates`). Lower it if scoring still runs
                out of GPU memory on a step with many legal candidates;
                raise it for more throughput if memory allows.
        """
        if not 0.0 <= lambda_gf <= 1.0:
            raise ValueError(f"lambda_gf must be in [0, 1], got {lambda_gf}")
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.lambda_gf = lambda_gf
        self.beam_size = beam_size
        self.gf_temperature = gf_temperature
        self.prefix_space = prefix_space
        self.scoring_chunk_size = scoring_chunk_size

    @torch.no_grad()
    def search(
        self,
        prompt: str,
        max_words: int = 20,
        pgf_name: Optional[str] = None,
        num_return: Optional[int] = None,
    ) -> List[WordHypothesis]:
        """Return finished hypotheses, best fused score first.

        Hypotheses that hit `max_words` without the grammar allowing an end
        are dropped; if *nothing* finishes, the best partial hypotheses are
        returned with `finished=False` so callers can inspect what happened.
        """
        device = next(self.model.parameters()).device
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], device=device)
        out = self.model(input_ids=prompt_ids, use_cache=True)
        init_cache = out.past_key_values
        if not isinstance(init_cache, DynamicCache):
            init_cache = DynamicCache.from_legacy_cache(init_cache)
        init_logits = out.logits[0, -1, :].float()

        beams = [WordHypothesis(past_key_values=init_cache, next_start_logits=init_logits)]
        done: List[WordHypothesis] = []

        for _ in range(max_words):
            # ---- grammar proposals + LM scores, one batched forward per hypothesis
            expansions: List[_Expansion] = []
            for hyp in beams:
                dist = self.runtime.next_token_logprobs(
                    hyp.words, pgf_name=pgf_name, temperature=self.gf_temperature
                )
                if not dist:
                    continue
                words_i = list(dist.keys())
                gf_lps = list(dist.values())
                scored = self._score_hyp_candidates(hyp, words_i)
                for word, gf_lp, (llm_lp, cache, next_start_logits) in zip(words_i, gf_lps, scored):
                    expansions.append(_Expansion(hyp, word, gf_lp, llm_lp, cache, next_start_logits))
            if not expansions:
                break

            # ---- fuse; EOS candidates are always kept (like the grammar side,
            # they don't compete for the beam_size live slots), continuations
            # compete for the top beam_size — only *those* survivors pay for
            # extracting their own KV-cache row.
            continuations: List[Tuple[float, _Expansion]] = []
            for e in expansions:
                step_score = self.lambda_gf * e.gf_lp + (1.0 - self.lambda_gf) * e.llm_lp
                total_score = e.hyp.score + step_score
                if e.word == GF_EOS:
                    done.append(
                        replace(
                            e.hyp,
                            words=list(e.hyp.words),
                            score=total_score,
                            gf_logprob=e.hyp.gf_logprob + e.gf_lp,
                            llm_logprob=e.hyp.llm_logprob + e.llm_lp,
                            finished=True,
                            past_key_values=None,
                            next_start_logits=None,
                        )
                    )
                else:
                    continuations.append((total_score, e))

            continuations.sort(key=lambda t: t[0], reverse=True)
            new_beams: List[WordHypothesis] = []
            for total_score, e in continuations[: self.beam_size]:
                new_beams.append(
                    replace(
                        e.hyp,
                        words=list(e.hyp.words) + [e.word],
                        score=total_score,
                        gf_logprob=e.hyp.gf_logprob + e.gf_lp,
                        llm_logprob=e.hyp.llm_logprob + e.llm_lp,
                        past_key_values=e.cache,
                        next_start_logits=e.next_start_logits,
                    )
                )
            beams = new_beams  # already sorted best-first, at most beam_size long

            # nothing alive can beat what is already finished
            if done and beams:
                if max(h.score for h in beams) < min(h.score for h in done):
                    if len(done) >= (num_return or self.beam_size):
                        break
            if not beams:
                break

        done.sort(key=lambda h: h.score, reverse=True)
        results = done or sorted(beams, key=lambda h: h.score, reverse=True)
        return results[: num_return or self.beam_size]

    # ------------------------------------------------------------- scoring

    def _candidate_ids(self, word: str) -> List[int]:
        if word == GF_EOS:
            return [self.tokenizer.eos_token_id]
        text = (" " + word) if self.prefix_space else word
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _score_hyp_candidates(
        self, hyp: WordHypothesis, words: Sequence[str]
    ) -> List[Tuple[float, DynamicCache, torch.Tensor]]:
        """Score every candidate word for `hyp`, one (llm_lp, cache, next_start_logits)
        triple per candidate, in the same order as `words`.

        Feeds only each candidate's own tokens, using `hyp`'s cached
        "prompt + hyp.words" as `past_key_values` (expanded across a chunk of
        candidates, not recomputed per candidate). A candidate's first token
        is scored from `hyp.next_start_logits` — already known from the
        previous step, no forward pass needed for it.

        The candidate set can run into the thousands at a single step (a
        real grammar, unlike a toy one, can have hundreds-to-thousands of
        legal next words), so this chunks the forward pass — batching all of
        them into one call blows past GPU memory just for the logits tensor
        (vocab-sized, times candidate count, times sequence position). Each
        candidate's cache/logits are cropped down to its own real length
        immediately after its chunk computes, so what's kept around after
        this call is small (one candidate's worth) rather than a whole
        chunk's padded batch.
        """
        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        token_lists = [self._candidate_ids(w) for w in words]
        past_len = hyp.past_key_values.get_seq_length()
        first_lp = torch.log_softmax(hyp.next_start_logits.float(), dim=-1)

        results: List[Tuple[float, DynamicCache, torch.Tensor]] = []
        for start in range(0, len(token_lists), self.scoring_chunk_size):
            chunk = token_lists[start : start + self.scoring_chunk_size]
            m = len(chunk)
            max_k = max(len(ids) for ids in chunk)

            input_ids = torch.full((m, max_k), pad_id, dtype=torch.long, device=device)
            new_mask = torch.zeros((m, max_k), dtype=torch.long, device=device)
            for i, ids in enumerate(chunk):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                new_mask[i, : len(ids)] = 1

            cache = _clone_cache(hyp.past_key_values)
            cache.batch_repeat_interleave(m)
            attention_mask = torch.cat(
                [torch.ones((m, past_len), dtype=torch.long, device=device), new_mask], dim=1
            )
            position_ids = torch.arange(past_len, past_len + max_k, device=device).unsqueeze(0).expand(m, -1)
            cache_position = torch.arange(past_len, past_len + max_k, device=device)

            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
            logprobs = torch.log_softmax(out.logits.float(), dim=-1)  # (m, max_k, vocab)

            for i, ids in enumerate(chunk):
                k = len(ids)
                lp = first_lp[ids[0]].item()
                for j in range(1, k):
                    lp += logprobs[i, j - 1, ids[j]].item()
                row_cache = _extract_row_cache(out.past_key_values, i, past_len + k)
                row_logits = logprobs[i, k - 1, :].clone()
                results.append((lp, row_cache, row_logits))

            del out, logprobs, cache

        return results
