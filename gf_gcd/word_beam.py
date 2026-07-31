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
3. Once global pruning picks the survivors for this step (typically a
   handful out of many hundreds-to-thousands of scored candidates), each
   survivor gets one more small forward pass, over just its own tokens, to
   materialize the cache it carries forward -- see `_materialize`. This is
   deliberately *not* done during scoring for every candidate: with beam_size
   in the single digits, almost every scored candidate is pruned away, so
   slicing a KV-cache row per candidate up front would spend most of that
   work on hypotheses that are immediately discarded.

This avoids ever re-encoding the growing "prompt + words so far" prefix from
scratch — the previous design's `_score_expansions` did that once per
candidate, per step (see git history), which made the LM side of this search
scale roughly quadratically with `max_words`.

The grammar itself only ever accepts *one* sentence, though, so the words
proposed at each step are the continuations of the hypothesis's current
sentence (`WordHypothesis.sentence_words`), not its whole history
(`WordHypothesis.words`). Once a sentence has no legal continuation left,
`search` forks it into ending there or starting another sentence from
scratch, so one `search()` call can produce a `max_words`-bounded passage of
several grammar-conformant sentences. See `WordBeamSearcher.search`.

Three optional knobs make the search itself smarter, independent of the
grammar/LM tradeoff `lambda_gf` controls (see their docstrings in
`WordBeamSearcher.__init__` for details):

- `length_penalty` -- GNMT-style normalization so hypotheses that finish at
  different word counts (e.g. a 3-word vs. an 8-word sentence) are compared
  fairly instead of the raw cumulative log-prob favoring whichever is shorter.
- `diversity_penalty` -- discourages every surviving hypothesis from picking
  the same next word, which otherwise collapses a beam of size N into N
  near-identical continuations.
- `no_repeat_ngram_size` -- blocks candidates that would recreate an n-gram
  already generated earlier in the same hypothesis.

`do_sample=True` switches to a second, independent decoding mode: instead of
pooling every hypothesis's candidates into one list and keeping the globally
best `beam_size` (deterministic beam search, the default), each of the
`beam_size` hypotheses independently draws its own next word from its own
temperature/top-k/top-p/min-p/typical-p-filtered distribution -- the same
"chat model" style sampling `do_sample=True` triggers in
`transformers.generate()`, just hand-rolled over grammar words instead of LM
tokens. It does not reuse the pooled top-`beam_size` cut at all: hypotheses
never compete with each other, `diversity_penalty` (which only makes sense
for that pooled competition) is rejected alongside it, and `GF_EOS` is just
another candidate in the draw rather than an automatic, uncontested survivor.
See `WordBeamSearcher._sample_index` and the `do_sample` argument docstring.
"""

import time
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import DynamicCache

from gf_gcd.gf_probs import GF_EOS, GFProbRuntime


@dataclass
class WordHypothesis:
    words: List[str] = field(default_factory=list)  # every word so far, all sentences
    # words of the *current* grammar sentence only -- what gets queried
    # against the grammar; reset to [] whenever a sentence completes and
    # the hypothesis starts another one. See `WordBeamSearcher.search`.
    sentence_words: List[str] = field(default_factory=list)
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
    """One (hypothesis, candidate word) pair, already scored by the LM.

    Deliberately carries no KV-cache: at any step, thousands of candidates
    get scored but only `beam_size` survive pruning, so extracting a cache
    slice per candidate would throw away >99% of that work. Survivors get
    their cache materialized after pruning instead, see `_materialize`."""

    hyp: WordHypothesis
    word: str
    gf_lp: float
    llm_lp: float
    # True when `word` begins a *new* sentence (offered because the
    # hypothesis's current sentence had no legal continuation left), rather
    # than continuing `hyp.sentence_words`.
    new_sentence: bool = False


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


# ---------------------------------------------------------------------------
# Sampling filters -- 1-D ports of `transformers.generation.logits_process`'s
# TopKLogitsWarper / TopPLogitsWarper / MinPLogitsWarper / TypicalLogitsWarper,
# adapted from vocab-sized batched tensors to the (much smaller) per-step
# candidate set a grammar step offers. Each takes a 1-D tensor of fused
# `step_score`s and returns a copy with excluded entries set to -inf;
# `min_tokens_to_keep` guarantees at least one survivor so a step never ends
# up with nowhere to sample from.


def _filter_top_k(logits: torch.Tensor, top_k: int, min_tokens_to_keep: int = 1) -> torch.Tensor:
    top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])
    kth_value = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def _filter_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_remove = cumulative_probs <= (1 - top_p)
    sorted_remove[-min_tokens_to_keep:] = False
    remove = torch.zeros_like(sorted_remove).scatter(0, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def _filter_min_p(logits: torch.Tensor, min_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    threshold = min_p * probs.max()
    remove = probs < threshold
    sorted_indices = torch.argsort(logits, descending=True)
    sorted_remove = remove.gather(0, sorted_indices)
    sorted_remove[:min_tokens_to_keep] = False
    remove = torch.zeros_like(remove).scatter(0, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def _filter_typical_p(logits: torch.Tensor, typical_p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    normalized = torch.log_softmax(logits, dim=-1)
    p = torch.exp(normalized)
    entropy = -(normalized * p).nansum()
    shifted = torch.abs((-normalized) - entropy)
    sorted_scores, sorted_indices = torch.sort(shifted, descending=False)
    sorted_logits = logits.gather(0, sorted_indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    last_ind = min(int((cumulative_probs < typical_p).sum().item()), sorted_scores.shape[-1] - 1)
    sorted_remove = sorted_scores > sorted_scores[last_ind]
    sorted_remove[:min_tokens_to_keep] = False
    remove = torch.zeros_like(sorted_remove).scatter(0, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


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
        length_penalty: float = 0.0,
        diversity_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        seed: Optional[int] = None,
        verbose: bool = False,
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
            length_penalty: GNMT-style length-normalization exponent, applied
                only when *comparing* hypotheses that may have a different
                number of words (finished-vs-live early stopping, and the
                final ranking of `done`) — never when picking survivors
                within a single step, since those are always the same length.
                Raw cumulative log-probs get more negative every extra word,
                which biases those comparisons toward whichever hypothesis
                happened to finish first; 0.0 (default) reproduces that raw
                behavior, ~0.6-1.0 corrects for it.
            diversity_penalty: when selecting the `beam_size` survivors of a
                step, each candidate word already claimed by a higher-ranked
                survivor *this step* has this much subtracted from its score
                before the next pick (only for selection, not from the score
                that gets carried forward). Counters beam collapse, where
                every surviving hypothesis picks the same top word and the
                beam stops exploring alternatives. 0.0 (default) disables it
                and falls back to a plain top-`beam_size` sort.
            no_repeat_ngram_size: block a candidate word if appending it
                would recreate an n-gram (of this size) already present
                earlier in the hypothesis's full word history. 0 (default)
                disables it. Mirrors `transformers`' generation-time
                `no_repeat_ngram_size`, adapted to whole-word candidates.
            do_sample: use independent per-hypothesis sampling instead of
                deterministic beam search (see the module docstring). Under
                sampling, `beam_size` is repurposed as "number of independent
                parallel chains" (like HF's `num_return_sequences` with
                `do_sample=True, num_beams=1`) rather than a competing beam
                width. Mutually exclusive with `diversity_penalty > 0`
                (diversity penalty only makes sense for the pooled
                competition that sampling mode skips).
            temperature: scales the *fused* `step_score` before top-k/top-p/
                min-p/typical-p filtering, only when `do_sample=True`. This is
                distinct from `gf_temperature` above, which flattens the
                grammar's own distribution *before* fusion -- the ordering
                mirrors Phase 1 (`GFProbLogitsProcessor` fuses first, HF's
                sampling warpers run after). `temperature <= 1e-5` is treated
                as greedy (argmax) to avoid dividing by ~0.
            top_k: keep only the `top_k` highest-fused-score candidates before
                sampling. 0 (default) disables it. Only used when
                `do_sample=True`.
            top_p: nucleus sampling -- keep the smallest set of candidates
                whose cumulative probability reaches `top_p`. 1.0 (default)
                disables it. Only used when `do_sample=True`.
            min_p: keep candidates whose probability is at least
                `min_p * (probability of the top candidate)`. 0.0 (default)
                disables it. An alternative to `top_p`; only used when
                `do_sample=True`.
            typical_p: locally typical sampling -- keep the smallest set of
                candidates (by closeness of their surprisal to the
                distribution's entropy) whose cumulative probability reaches
                `typical_p`. 1.0 (default) disables it. Another alternative
                to `top_p`; only used when `do_sample=True`.
            seed: if given, used to seed the `torch.Generator` driving every
                sampling draw, for reproducible runs. Only used when
                `do_sample=True`.
            verbose: print per-word-step timing (grammar proposals via `pgf
                complete()`, LM forward passes scoring candidates, and
                fusing/pruning or sampling the beam), so processing time per
                stage can be eyeballed during a run.
        """
        if not 0.0 <= lambda_gf <= 1.0:
            raise ValueError(f"lambda_gf must be in [0, 1], got {lambda_gf}")
        if do_sample and diversity_penalty > 0.0:
            raise ValueError(
                "do_sample and diversity_penalty are mutually exclusive: diversity_penalty "
                "only corrects the pooled top-beam_size selection of deterministic beam "
                "search, which do_sample bypasses entirely (each hypothesis samples "
                "independently, so there is no shared competition to diversify)."
            )
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {min_p}")
        if not 0.0 < typical_p <= 1.0:
            raise ValueError(f"typical_p must be in (0, 1], got {typical_p}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.model = model
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.lambda_gf = lambda_gf
        self.beam_size = beam_size
        self.gf_temperature = gf_temperature
        self.prefix_space = prefix_space
        self.scoring_chunk_size = scoring_chunk_size
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.typical_p = typical_p
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self.verbose = verbose

    @torch.no_grad()
    def search(
        self,
        prompt: str,
        max_words: int = 20,
        pgf_name: Optional[str] = None,
        num_return: Optional[int] = None,
    ) -> List[WordHypothesis]:
        """Return finished hypotheses, best fused score first.

        The grammar accepts exactly one sentence, so each hypothesis is
        proposed continuations of its *current* sentence only
        (`hyp.sentence_words`), not its whole `words` history. Whenever that
        current sentence has no legal continuation left (`GF_EOS` is the
        *only* candidate the grammar offers), the hypothesis forks: one
        branch ends there (`GF_EOS` consumed, moved to `done`, scored via
        the real EOS token like today), and one branch per legal
        start-of-sentence word begins another sentence (`sentence_words`
        reset to `[word]`, `words` keeps growing) -- so a single `search()`
        call can produce a multi-sentence `max_words`-bounded passage of
        grammar-conformant sentences, not just one. An *optional* stop
        (`GF_EOS` alongside live continuations of the current sentence)
        only competes with those continuations for ending the whole
        hypothesis; it does not fork a new sentence, since a word that is
        both a legal continuation and a legal sentence start would make
        each fork's cumulative score ambiguous.

        Hypotheses that hit `max_words` without the grammar allowing an end
        are dropped; if *nothing* finishes, the best partial hypotheses are
        returned with `finished=False` so callers can inspect what happened.
        """
        device = next(self.model.parameters()).device
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], device=device)
        t0 = time.perf_counter()
        out = self.model(input_ids=prompt_ids, use_cache=True)
        if self.verbose:
            print(
                f"[word-step init] prompt tokens={prompt_ids.shape[1]}  "
                f"LLM forward (encode prompt)={(time.perf_counter() - t0) * 1000:7.1f} ms",
                flush=True,
            )
        init_cache = out.past_key_values
        if not isinstance(init_cache, DynamicCache):
            init_cache = DynamicCache.from_legacy_cache(init_cache)
        init_logits = out.logits[0, -1, :].float()

        if self.do_sample:
            # unlike pooled beam search -- which starts from one hypothesis and
            # *grows* to beam_size as the pooled top-k selects several distinct
            # continuations of it -- independent sampling advances each live
            # hypothesis by exactly one word per step, so it can never fan out
            # on its own. Seed `beam_size` independent chains up front instead
            # (sharing the same read-only initial cache/logits is safe: both
            # `_score_hyp_candidates` and `_materialize` clone a hypothesis's
            # cache before ever touching it).
            beams = [
                WordHypothesis(past_key_values=init_cache, next_start_logits=init_logits)
                for _ in range(self.beam_size)
            ]
        else:
            beams = [WordHypothesis(past_key_values=init_cache, next_start_logits=init_logits)]
        done: List[WordHypothesis] = []

        for step_idx in range(max_words):
            step_start = time.perf_counter()
            n_hyps = len(beams)
            grammar_s = 0.0  # time in `pgf complete()` (grammar proposals)
            llm_s = 0.0  # time in `_score_hyp_candidates` (LM forward passes)
            n_candidates = 0

            # ---- grammar proposals + LM scores, one batched forward per hypothesis
            expansions: List[_Expansion] = []  # pooled path (do_sample=False) only
            sampled_beams: List[WordHypothesis] = []  # sampling path (do_sample=True) only
            select_s = 0.0  # sampling path only: time spent drawing each hypothesis's next word
            for hyp in beams:
                t0 = time.perf_counter()
                dist = self.runtime.next_token_logprobs(
                    hyp.sentence_words, pgf_name=pgf_name, temperature=self.gf_temperature
                )
                grammar_s += time.perf_counter() - t0
                if not dist:
                    continue
                words_i = list(dist.keys())
                gf_lps = list(dist.values())
                new_sentence_flags = [False] * len(words_i)

                if GF_EOS in dist and set(dist) == {GF_EOS}:
                    # current sentence has no legal continuation left: also
                    # offer starting a fresh sentence, in addition to (and
                    # scored independently of) ending the hypothesis here
                    stop_lp = dist[GF_EOS]
                    t0 = time.perf_counter()
                    start_dist = self.runtime.next_token_logprobs(
                        [], pgf_name=pgf_name, temperature=self.gf_temperature
                    )
                    grammar_s += time.perf_counter() - t0
                    for word, start_lp in start_dist.items():
                        if word == GF_EOS:
                            continue  # an empty sentence isn't a real continuation
                        words_i.append(word)
                        gf_lps.append(stop_lp + start_lp)
                        new_sentence_flags.append(True)

                if self.no_repeat_ngram_size > 0:
                    words_i, gf_lps, new_sentence_flags = self._filter_repeats(
                        hyp.words, words_i, gf_lps, new_sentence_flags
                    )

                t0 = time.perf_counter()
                llm_lps = self._score_hyp_candidates(hyp, words_i)
                llm_s += time.perf_counter() - t0
                n_candidates += len(words_i)

                if self.do_sample:
                    # independent draw for *this* hypothesis alone: unlike the
                    # pooled path below, GF_EOS is just one more candidate in
                    # the distribution (it can be sampled, but doesn't
                    # automatically win), and this hypothesis never competes
                    # with any other hypothesis's candidates for a shared slot.
                    t0 = time.perf_counter()
                    step_scores = torch.tensor(
                        [
                            self.lambda_gf * gf_lp + (1.0 - self.lambda_gf) * llm_lp
                            for gf_lp, llm_lp in zip(gf_lps, llm_lps)
                        ],
                        dtype=torch.float32,
                    )
                    idx = self._sample_index(step_scores)
                    select_s += time.perf_counter() - t0
                    word, gf_lp, llm_lp, new_sentence = (
                        words_i[idx],
                        gf_lps[idx],
                        llm_lps[idx],
                        new_sentence_flags[idx],
                    )
                    total_score = hyp.score + step_scores[idx].item()
                    if word == GF_EOS:
                        done.append(
                            replace(
                                hyp,
                                words=list(hyp.words),
                                score=total_score,
                                gf_logprob=hyp.gf_logprob + gf_lp,
                                llm_logprob=hyp.llm_logprob + llm_lp,
                                finished=True,
                                past_key_values=None,
                                next_start_logits=None,
                            )
                        )
                    else:
                        new_sentence_words = [word] if new_sentence else list(hyp.sentence_words) + [word]
                        cache, next_start_logits = self._materialize(hyp, word)
                        sampled_beams.append(
                            replace(
                                hyp,
                                words=list(hyp.words) + [word],
                                sentence_words=new_sentence_words,
                                score=total_score,
                                gf_logprob=hyp.gf_logprob + gf_lp,
                                llm_logprob=hyp.llm_logprob + llm_lp,
                                past_key_values=cache,
                                next_start_logits=next_start_logits,
                            )
                        )
                else:
                    for word, gf_lp, new_sentence, llm_lp in zip(
                        words_i, gf_lps, new_sentence_flags, llm_lps
                    ):
                        expansions.append(_Expansion(hyp, word, gf_lp, llm_lp, new_sentence))

            if self.do_sample:
                if n_candidates == 0:
                    # every live hypothesis's dist was empty (a genuine grammar
                    # dead end, not "everyone sampled GF_EOS") -- leave `beams`
                    # untouched rather than overwriting it with the empty
                    # `sampled_beams`, so the last valid (unfinished) hypotheses
                    # are still returned instead of losing them to an empty
                    # `results` (mirrors the pooled path's `if not expansions`
                    # check below).
                    if self.verbose:
                        print(f"[word-step {step_idx:3d}] no live expansions, stopping", flush=True)
                    break
                beams = sampled_beams
                if self.verbose:
                    print(
                        f"[word-step {step_idx:3d}] hyps={n_hyps:2d} candidates={n_candidates:5d} "
                        f"live={len(beams)} done={len(done)}  |  "
                        f"pgf complete() [parse]: {grammar_s * 1000:7.1f} ms  |  "
                        f"LLM forward [_score_hyp_candidates]: {llm_s * 1000:8.1f} ms  |  "
                        f"sample: {select_s * 1000:6.2f} ms  |  "
                        f"step total: {(time.perf_counter() - step_start) * 1000:8.1f} ms",
                        flush=True,
                    )
                # No early-stop heuristic here: independent chains aren't
                # optimized against each other the way pooled beam search is
                # (a currently-low-scoring live chain can still wander into a
                # good finish), so just run until every chain finishes or
                # max_words is reached.
                if not beams:
                    break
                continue

            if not expansions:
                if self.verbose:
                    print(f"[word-step {step_idx:3d}] no live expansions, stopping", flush=True)
                break

            # ---- fuse; EOS candidates are always kept (like the grammar side,
            # they don't compete for the beam_size live slots), continuations
            # compete for the top beam_size — only *those* survivors pay for
            # extracting their own KV-cache row.
            t_combine_start = time.perf_counter()
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

            if self.diversity_penalty > 0.0 and len(continuations) > self.beam_size:
                survivors = self._select_diverse(continuations)
            else:
                continuations.sort(key=lambda t: t[0], reverse=True)
                survivors = continuations[: self.beam_size]

            combine_s = time.perf_counter() - t_combine_start

            # ---- only the (beam_size-ish) survivors ever need real KV-cache
            # state to carry forward, so materialize it here instead of for
            # every scored candidate above.
            t_materialize_start = time.perf_counter()
            new_beams: List[WordHypothesis] = []
            for total_score, e in survivors:
                new_sentence_words = [e.word] if e.new_sentence else list(e.hyp.sentence_words) + [e.word]
                cache, next_start_logits = self._materialize(e.hyp, e.word)
                new_beams.append(
                    replace(
                        e.hyp,
                        words=list(e.hyp.words) + [e.word],
                        sentence_words=new_sentence_words,
                        score=total_score,
                        gf_logprob=e.hyp.gf_logprob + e.gf_lp,
                        llm_logprob=e.hyp.llm_logprob + e.llm_lp,
                        past_key_values=cache,
                        next_start_logits=next_start_logits,
                    )
                )
            beams = new_beams  # already sorted best-first, at most beam_size long
            materialize_s = time.perf_counter() - t_materialize_start

            if self.verbose:
                print(
                    f"[word-step {step_idx:3d}] hyps={n_hyps:2d} candidates={n_candidates:5d} "
                    f"live={len(beams)} done={len(done)}  |  "
                    f"pgf complete() [parse]: {grammar_s * 1000:7.1f} ms  |  "
                    f"LLM forward [_score_hyp_candidates]: {llm_s * 1000:8.1f} ms  |  "
                    f"combine+prune: {combine_s * 1000:6.2f} ms  |  "
                    f"materialize survivors: {materialize_s * 1000:6.2f} ms  |  "
                    f"step total: {(time.perf_counter() - step_start) * 1000:8.1f} ms",
                    flush=True,
                )

            # nothing alive can beat what is already finished -- compare with
            # length normalization since live beams still have fewer words
            # than hypotheses that already finished (or vice versa)
            if done and beams:
                if max(self._normalized_score(h) for h in beams) < min(
                    self._normalized_score(h) for h in done
                ):
                    if len(done) >= (num_return or self.beam_size):
                        break
            if not beams:
                break

        # under do_sample, chains are independent (like num_return_sequences),
        # so a chain finishing before another one is not a reason to drop the
        # still-running one: report *both* finished and still-live chains,
        # not just whichever happened to reach GF_EOS first. The deterministic
        # path keeps its original behavior (only fall back to `beams` when
        # nothing at all finished), unaffected by this.
        results = done + beams if self.do_sample else (done or beams)
        results.sort(key=self._normalized_score, reverse=True)
        return results[: num_return or self.beam_size]

    # ------------------------------------------------------------- scoring

    def _length_norm(self, n_words: int) -> float:
        """GNMT-style length normalization factor; 1.0 (no-op) when disabled."""
        if self.length_penalty == 0.0:
            return 1.0
        return ((5.0 + max(n_words, 1)) / 6.0) ** self.length_penalty

    def _normalized_score(self, hyp: WordHypothesis) -> float:
        return hyp.score / self._length_norm(len(hyp.words))

    def _select_diverse(
        self, continuations: List[Tuple[float, "_Expansion"]]
    ) -> List[Tuple[float, "_Expansion"]]:
        """Greedily pick `beam_size` survivors, penalizing words already
        claimed by a higher-ranked pick this step (Hamming diversity) so the
        beam doesn't collapse onto near-duplicate continuations."""
        remaining = list(continuations)
        word_counts: dict = {}
        selected: List[Tuple[float, _Expansion]] = []
        while remaining and len(selected) < self.beam_size:
            remaining.sort(
                key=lambda t: t[0] - self.diversity_penalty * word_counts.get(t[1].word, 0),
                reverse=True,
            )
            picked = remaining.pop(0)
            selected.append(picked)
            word_counts[picked[1].word] = word_counts.get(picked[1].word, 0) + 1
        return selected

    def _sample_index(self, step_scores: torch.Tensor) -> int:
        """Sample one candidate index from `step_scores` (this hypothesis's
        fused per-word log-scores), applying temperature then top-k/top-p/
        min-p/typical-p filtering -- the same cascade order HF's sampling
        warpers use, just over this step's candidate set instead of the
        full vocabulary. Only called when `do_sample=True`."""
        n = step_scores.shape[0]
        if n <= 1:
            return 0
        if self.temperature <= 1e-5:
            return int(torch.argmax(step_scores).item())
        logits = step_scores / self.temperature
        if self.top_k > 0:
            logits = _filter_top_k(logits, self.top_k)
        if self.top_p < 1.0:
            logits = _filter_top_p(logits, self.top_p)
        if self.min_p > 0.0:
            logits = _filter_min_p(logits, self.min_p)
        if self.typical_p < 1.0:
            logits = _filter_typical_p(logits, self.typical_p)
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self._generator).item())

    def _filter_repeats(
        self,
        history: List[str],
        words: List[str],
        gf_lps: List[float],
        flags: List[bool],
    ) -> Tuple[List[str], List[float], List[bool]]:
        """Drop candidates that would recreate an n-gram already present in
        `history` (`no_repeat_ngram_size`). `GF_EOS` is never blocked, and if
        blocking would remove every candidate, the unfiltered lists are
        returned instead (never leaves a hypothesis with nowhere to go)."""
        ctx_len = self.no_repeat_ngram_size - 1
        if len(history) < ctx_len:
            return words, gf_lps, flags
        context = tuple(history[len(history) - ctx_len :]) if ctx_len > 0 else ()
        banned = {
            history[i + ctx_len]
            for i in range(len(history) - ctx_len)
            if tuple(history[i : i + ctx_len]) == context
        }
        if not banned:
            return words, gf_lps, flags
        kept = [
            (w, lp, f) for w, lp, f in zip(words, gf_lps, flags) if w == GF_EOS or w not in banned
        ]
        if not kept:
            return words, gf_lps, flags
        kept_words, kept_lps, kept_flags = zip(*kept)
        return list(kept_words), list(kept_lps), list(kept_flags)

    def _candidate_ids(self, word: str) -> List[int]:
        if word == GF_EOS:
            return [self.tokenizer.eos_token_id]
        text = (" " + word) if self.prefix_space else word
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _score_hyp_candidates(self, hyp: WordHypothesis, words: Sequence[str]) -> List[float]:
        """Score every candidate word for `hyp`, returning one `llm_lp` per
        candidate, in the same order as `words`.

        Feeds only each candidate's own tokens, using `hyp`'s cached
        "prompt + hyp.words" as `past_key_values` (expanded across a chunk of
        candidates, not recomputed per candidate). A candidate's first token
        is scored from `hyp.next_start_logits` — already known from the
        previous step, no forward pass needed for it.

        The candidate set can run into the thousands at a single step (a
        real grammar, unlike a toy one, can have hundreds-to-thousands of
        legal next words), so this chunks the forward pass — batching all of
        them into one call blows past GPU memory just for the logits tensor
        (vocab-sized, times candidate count, times sequence position).

        This only computes log-probabilities — no KV-cache is extracted here.
        Of the (up to) thousands of candidates scored per hypothesis, only
        `beam_size` survive pruning each step, so eagerly slicing a cache per
        candidate would waste the vast majority of that work; survivors get
        their cache materialized separately, see `_materialize`. Per-chunk
        results are pulled off the GPU with a single batched `.tolist()`
        rather than one `.item()` per token, which avoids a GPU/CPU sync per
        token — the dominant cost once candidate counts run into the
        thousands.
        """
        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        token_lists = [self._candidate_ids(w) for w in words]
        past_len = hyp.past_key_values.get_seq_length()
        first_lp = torch.log_softmax(hyp.next_start_logits.float(), dim=-1)

        results: List[float] = []
        n_chunks = 0
        forward_s = 0.0  # model forward + log_softmax + gather over its logits
        postprocess_s = 0.0  # pulling the batched result off the GPU
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

            t0 = time.perf_counter()
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
            logprobs = torch.log_softmax(out.logits.float(), dim=-1)  # (m, max_k, vocab)

            # lp(candidate) = first_lp[ids[0]] + sum_{j=1..k-1} logprobs[j-1, ids[j]],
            # vectorized over the whole chunk instead of a per-token Python loop.
            first_ids = input_ids[:, 0]  # (m,)
            step_lp = first_lp[first_ids]  # (m,)
            if max_k > 1:
                targets = input_ids[:, 1:]  # (m, max_k-1)
                gathered = logprobs[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (m, max_k-1)
                valid = new_mask[:, 1:].to(gathered.dtype)  # 0 past each candidate's real length
                step_lp = step_lp + (gathered * valid).sum(dim=1)
            t1 = time.perf_counter()
            forward_s += t1 - t0
            n_chunks += 1

            results.extend(step_lp.tolist())  # one GPU/CPU sync for the whole chunk
            postprocess_s += time.perf_counter() - t1
            del out, logprobs, cache

        if self.verbose:
            print(
                f"    [llm-score] candidates={len(token_lists):5d} chunks={n_chunks}  "
                f"forward+softmax={forward_s * 1000:8.1f} ms  "
                f"postprocess={postprocess_s * 1000:6.2f} ms",
                flush=True,
            )

        return results

    def _materialize(self, hyp: WordHypothesis, word: str) -> Tuple[DynamicCache, torch.Tensor]:
        """Run the forward pass for exactly one candidate word (a pruning
        survivor) and return the (cache, next_start_logits) it carries
        forward as the new hypothesis's incremental-decoding state.

        Only called for the `beam_size`-ish survivors of a step, never for
        every scored candidate — see `_score_hyp_candidates` and `_Expansion`.
        """
        device = next(self.model.parameters()).device
        ids = self._candidate_ids(word)
        k = len(ids)
        past_len = hyp.past_key_values.get_seq_length()

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones((1, past_len + k), dtype=torch.long, device=device)
        position_ids = torch.arange(past_len, past_len + k, device=device).unsqueeze(0)
        cache_position = torch.arange(past_len, past_len + k, device=device)
        cache = _clone_cache(hyp.past_key_values)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        next_start_logits = out.logits[0, -1, :].float().clone()
        new_cache = _extract_row_cache(out.past_key_values, 0, past_len + k)
        return new_cache, next_start_logits
