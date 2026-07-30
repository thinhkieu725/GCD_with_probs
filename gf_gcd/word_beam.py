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
    """One (hypothesis, candidate word) pair, already scored, carrying its
    own (small, pre-cropped) KV-cache slice in case it survives pruning."""

    hyp: WordHypothesis
    word: str
    gf_lp: float
    llm_lp: float
    cache: DynamicCache  # batch-1, cropped to exactly this candidate's real length
    next_start_logits: torch.Tensor  # (vocab,) — logits predicting what comes after `word`
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
            verbose: print per-word-step timing (grammar proposals via `pgf
                complete()`, LM forward passes scoring candidates, and
                fusing/pruning the beam), so processing time per stage can be
                eyeballed during a run.
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
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
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

        beams = [WordHypothesis(past_key_values=init_cache, next_start_logits=init_logits)]
        done: List[WordHypothesis] = []

        for step_idx in range(max_words):
            step_start = time.perf_counter()
            n_hyps = len(beams)
            grammar_s = 0.0  # time in `pgf complete()` (grammar proposals)
            llm_s = 0.0  # time in `_score_hyp_candidates` (LM forward passes)
            n_candidates = 0

            # ---- grammar proposals + LM scores, one batched forward per hypothesis
            expansions: List[_Expansion] = []
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
                scored = self._score_hyp_candidates(hyp, words_i)
                llm_s += time.perf_counter() - t0
                n_candidates += len(words_i)
                for word, gf_lp, new_sentence, (llm_lp, cache, next_start_logits) in zip(
                    words_i, gf_lps, new_sentence_flags, scored
                ):
                    expansions.append(
                        _Expansion(hyp, word, gf_lp, llm_lp, cache, next_start_logits, new_sentence)
                    )
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

            new_beams: List[WordHypothesis] = []
            for total_score, e in survivors:
                new_sentence_words = [e.word] if e.new_sentence else list(e.hyp.sentence_words) + [e.word]
                new_beams.append(
                    replace(
                        e.hyp,
                        words=list(e.hyp.words) + [e.word],
                        sentence_words=new_sentence_words,
                        score=total_score,
                        gf_logprob=e.hyp.gf_logprob + e.gf_lp,
                        llm_logprob=e.hyp.llm_logprob + e.llm_lp,
                        past_key_values=e.cache,
                        next_start_logits=e.next_start_logits,
                    )
                )
            beams = new_beams  # already sorted best-first, at most beam_size long
            combine_s = time.perf_counter() - t_combine_start

            if self.verbose:
                print(
                    f"[word-step {step_idx:3d}] hyps={n_hyps:2d} candidates={n_candidates:5d} "
                    f"live={len(beams)} done={len(done)}  |  "
                    f"pgf complete() [parse]: {grammar_s * 1000:7.1f} ms  |  "
                    f"LLM forward [_score_hyp_candidates]: {llm_s * 1000:8.1f} ms  |  "
                    f"combine+prune: {combine_s * 1000:6.2f} ms  |  "
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

        done.sort(key=self._normalized_score, reverse=True)
        results = done or sorted(beams, key=self._normalized_score, reverse=True)
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
        n_chunks = 0
        forward_s = 0.0  # model forward + log_softmax over its logits
        postprocess_s = 0.0  # per-candidate cache/logit extraction
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
            t1 = time.perf_counter()
            forward_s += t1 - t0
            n_chunks += 1

            for i, ids in enumerate(chunk):
                k = len(ids)
                lp = first_lp[ids[0]].item()
                for j in range(1, k):
                    lp += logprobs[i, j - 1, ids[j]].item()
                row_cache = _extract_row_cache(out.past_key_values, i, past_len + k)
                row_logits = logprobs[i, k - 1, :].clone()
                results.append((lp, row_cache, row_logits))

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
