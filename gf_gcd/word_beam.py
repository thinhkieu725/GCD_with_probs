"""Word-synchronized beam search fusing a word-level GF grammar with an LM.

Phase-2 counterpart of `GFProbConstraint`: instead of stepping one LM token at
a time over a token-ID grammar, the beam advances one *grammar word* at a time
over the natural word-level grammar (no tokenizer-specific rewriting, works
with any tokenizer):

- the grammar proposes candidate next words with probabilities
  (`GFProbRuntime.next_token_logprobs`),
- the LM scores each candidate word as the sum of the log-probabilities of
  its subword tokens (one batched forward pass per step),
- hypotheses grow by `lambda_gf * ln P_GF + (1 - lambda_gf) * ln P_LM` and the
  top `beam_size` survive.

Scoring recomputes the full sequence each step instead of carrying KV caches
across steps — simpler and version-proof; fine at the scale of grammar-
constrained decoding (tens of words, tens of candidates). Swap in cache reuse
if profiling ever says otherwise.
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple

import torch

from gf_gcd.gf_probs import GF_EOS, GFProbRuntime


@dataclass
class WordHypothesis:
    words: List[str] = field(default_factory=list)
    score: float = 0.0  # cumulative fused score
    gf_logprob: float = 0.0  # cumulative ln P_GF
    llm_logprob: float = 0.0  # cumulative ln P_LM
    finished: bool = False

    @property
    def text(self) -> str:
        return " ".join(self.words)


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
    ):
        """
        Args:
            model / tokenizer: a HF causal LM and its tokenizer (e.g. the
                `.model` / `.tokenizer` of a `ConstrainedGenerator`).
            runtime: probability runtime over the *word-level* .pgf.
            lambda_gf: interpolation weight, score = λ·lnP_GF + (1-λ)·lnP_LM.
            prefix_space: encode candidate words as " word" (mid-sentence
                position), matching how subword tokenizers see them.
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
        beams = [WordHypothesis()]
        done: List[WordHypothesis] = []

        for _ in range(max_words):
            # ---- grammar proposals for every live hypothesis
            expansions: List[Tuple[WordHypothesis, str, float]] = []
            for hyp in beams:
                dist = self.runtime.next_token_logprobs(
                    hyp.words, pgf_name=pgf_name, temperature=self.gf_temperature
                )
                for word, gf_lp in dist.items():
                    expansions.append((hyp, word, gf_lp))
            if not expansions:
                break

            # ---- LM scores for all (hypothesis, word) pairs in one forward
            llm_lps = self._score_expansions(prompt, expansions)

            # ---- fuse, extend, prune
            new_beams: List[WordHypothesis] = []
            for (hyp, word, gf_lp), llm_lp in zip(expansions, llm_lps):
                step_score = self.lambda_gf * gf_lp + (1.0 - self.lambda_gf) * llm_lp
                ext = replace(
                    hyp,
                    words=list(hyp.words),  # replace() copies the reference only
                    score=hyp.score + step_score,
                    gf_logprob=hyp.gf_logprob + gf_lp,
                    llm_logprob=hyp.llm_logprob + llm_lp,
                )
                if word == GF_EOS:
                    ext.finished = True
                    done.append(ext)
                else:
                    ext.words.append(word)
                    new_beams.append(ext)

            new_beams.sort(key=lambda h: h.score, reverse=True)
            beams = new_beams[: self.beam_size]

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

    def _encode_context(self, prompt: str, words: Sequence[str]) -> List[int]:
        text = prompt + "".join(" " + w for w in words)
        return self.tokenizer.encode(text)

    def _candidate_ids(self, word: str) -> List[int]:
        if word == GF_EOS:
            return [self.tokenizer.eos_token_id]
        text = (" " + word) if self.prefix_space else word
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _score_expansions(
        self, prompt: str, expansions: Sequence[Tuple[WordHypothesis, str, float]]
    ) -> List[float]:
        """ln P_LM(word | prompt, hyp.words) for each (hyp, word) pair."""
        device = next(self.model.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        ctx_cache = {}
        rows, ctx_lens, cand_lens = [], [], []
        for hyp, word, _ in expansions:
            key = tuple(hyp.words)
            if key not in ctx_cache:
                ctx_cache[key] = self._encode_context(prompt, hyp.words)
            ctx = ctx_cache[key]
            cand = self._candidate_ids(word)
            rows.append(ctx + cand)
            ctx_lens.append(len(ctx))
            cand_lens.append(len(cand))

        max_len = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1

        logits = self.model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        scores = []
        for i, row in enumerate(rows):
            # token at position p is predicted by logits at position p-1
            lp = 0.0
            for p in range(ctx_lens[i], ctx_lens[i] + cand_lens[i]):
                lp += logprobs[i, p - 1, row[p]].item()
            scores.append(lp)
        return scores
