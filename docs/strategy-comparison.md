# Token-level vs. word-level grammar–LM fusion

Two decoding strategies fuse a probability-annotated GF grammar
(`gf -make --probs=...`) with a causal LM. They are **alternative search
drivers**: each defines what "one step of beam search" means, so a single
generation run uses exactly one of them. They share everything below the
search loop — [`GFProbRuntime.next_token_logprobs`](../gf_gcd/gf_probs.py)
turns the GF C runtime's completions into per-step conditional
distributions for both.

|                        | Phase 1 — token-level                          | Phase 2 — word-level                       |
|------------------------|------------------------------------------------|--------------------------------------------|
| entry point            | `GFProbConstraint` → `generator.generate()`    | `WordBeamSearcher.search()`                |
| beam step advances by  | one **LM subword token**                       | one **grammar word**                       |
| search loop owned by   | HuggingFace `generate(num_beams=B)`            | hand-rolled loop in `word_beam.py`         |
| grammar format         | token-ID grammar (terminals `"t31" ++ "t287"`) | natural word-level grammar                 |
| tokenizer coupling     | grammar recompiled per tokenizer               | tokenizer-independent                      |
| fusion formula         | `ln P_LM + λ·ln P_GF`, λ ≥ 0                   | `λ·ln P_GF + (1−λ)·ln P_LM`, λ ∈ [0,1]     |
| λ = 0 means            | binary GCD (mask only)                         | LM chooses among legal words               |
| grammar cost lands on  | first subtoken of each word                    | the word, atomically                       |
| HF features (sampling, length penalty, batching, …) | free                    | only what `word_beam.py` implements        |

Both formulas are log-linear interpolations; they differ only in
normalization of λ.

## What problem each solves

**Phase 1** solves *"add grammar probabilities to the existing GCD pipeline
with minimal machinery."* The old pipeline already stepped HF generation
through a token-ID grammar, masking illegal tokens. Phase 1 keeps that
architecture and upgrades the mask to a soft reweighting: beam search then
optimizes `P_LM · P_GF^λ` with no custom search code.

**Phase 2** solves the two structural problems Phase 1 cannot fix from
inside HF's token loop:

1. the grammar must be rewritten into one tokenizer's IDs (recompile per
   model), and
2. the *unfair mid-word score* problem (below).

## Step-by-step workflow

### Phase 1 (token-level)

1. **Prepare** — rewrite terminals to letter-prefixed token IDs
   (`tokenize_concrete_grammar(..., id_prefix="t")` in
   [`grammar_tools/tokenize.py`](../gf_gcd/grammar_tools/tokenize.py));
   compile with `compile_grammar(..., probs_file=...)`. The letter prefix
   is required: digit-only terminals are lexed as `Int` literals by the C
   runtime, which breaks `complete()` (bug found in gf-core
   `runtime/c/pgf/parser.c`, `pgf_complete`).
2. **Wire** — `generator.generate(constraint=GFProbConstraint(...),
   num_beams=B)`. The generator prefers a constraint's `logits_processor()`
   over the binary `prefix_allowed_tokens_fn`
   ([`generator.py`](../gf_gcd/generator.py), constraint dispatch block).
3. **Every decoding step** — HF calls
   [`GFProbLogitsProcessor.__call__`](../gf_gcd/constraints.py) with one
   row per beam hypothesis:
   - strip the prompt, then locate the current *sentence* boundary within
     what's left (`_SentenceCursor` — the grammar accepts one sentence, so
     only the tail generated since the last boundary is rendered to
     grammar tokens, e.g. `["t100", "t300"]`);
   - query `next_token_logprobs` → `{token: ln P_GF}` incl. `GF_EOS`; at a
     forced boundary (no legal continuation left) also fuse in the
     grammar's own start-of-sentence tokens, so a beam can chain into
     another sentence instead of only being allowed to stop;
   - `fused[t] = scores[t] + λ·ln P_GF(t)` for allowed `t`, `−inf`
     otherwise; `GF_EOS` maps to the real `eos_token_id`;
   - finished rows (EOS already emitted) pass through EOS/pad only.
4. **Selection** — HF ranks all (beam, token) candidates by cumulative
   fused score and keeps the top B. Stopping, length penalties,
   `num_return_sequences` are HF's.

### Phase 2 (word-level)

Loop in [`WordBeamSearcher.search`](../gf_gcd/word_beam.py); each
iteration grows every hypothesis by one word:

1. **Propose** — per hypothesis, `next_token_logprobs(hyp.sentence_words)`
   (the current *sentence*'s words, not the hypothesis's whole history —
   the grammar accepts only one sentence) lists legal next words with
   `ln P_GF` (incl. `GF_EOS` when stopping is legal). At a forced boundary
   (`GF_EOS` the only option), the proposer also offers the grammar's
   start-of-sentence words, each forking the hypothesis into "begin
   another sentence" alongside "end here". The grammar is the proposer:
   nothing else can enter a beam.
2. **Score** — `_score_expansions` computes
   `ln P_LM(word | prompt + words)` as the **sum of the word's subtoken
   log-probs** — e.g.
   `P(" delicious") = P(" del") · P("icious" | " del")` — for all
   (hypothesis, word) pairs in one padded forward pass.
3. **Fuse & prune** — `step = λ·ln P_GF + (1−λ)·ln P_LM`; extensions
   sorted, top `beam_size` kept. `GF_EOS` moves a hypothesis to `done`
   (its LM term is the real EOS token's probability).
4. **Stop** — beam empty, or enough finished hypotheses that the best
   live score can no longer beat the worst finished one (scores only
   decrease). If `max_words` is exhausted, best partials return with
   `finished=False`.

`WordHypothesis` keeps `gf_logprob` and `llm_logprob` separately, so you
can always see which expert drove a result.

## The unfair mid-word score problem

The grammar assigns probability to **words** (via abstract functions); a
subword LM spends **several steps** per word. In Phase 1 the grammar's
entire opinion of a word lands on the step where it becomes committed —
the first subtoken — and the word's remaining subtokens get grammar cost
≈ 0 (once inside " Ital", the only legal continuation is "ian", and a
one-candidate distribution renormalizes to probability 1).

Example with GPT-2 tokenization of the Food grammar, probs
`Delicious 0.4`, `Italian 0.3`. `" delicious"` is 1 GPT-2 token;
`" Italian"` splits into `" Ital" + "ian"`. Two beams after
`"this wine is"`, λ = 1 for clarity:

```
                      step k                 step k+1
beam A:  + " delicious"  gf: ln 0.4     + <eos>   gf: ln 1
beam B:  + " Ital"       gf: ln 0.3     + "ian"   gf: ln 1  (forced)
```

At the **end of step k** the beams are compared for pruning:

- A carries the grammar cost of a **complete** word (`ln 0.4 ≈ −0.92`);
- B already carries the **full** grammar cost of "Italian"
  (`ln 0.3 ≈ −1.20`) but only the LM cost of *half* the word — the cheap
  forced `"ian"` LM factor hasn't been paid yet.

B is momentarily judged on full-grammar-cost/partial-LM-cost while A is
judged on full/full. Whenever the beam cut falls between those steps,
multi-token words are pruned slightly too eagerly (or too laxly, if their
first piece is LM-cheap). It is a *comparability* skew between hypotheses
at different intra-word positions — cumulative totals are correct again
at every word boundary; the bias only shows where the beam is cut, and it
never violates the grammar.

Phase 2 eliminates it by construction: hypotheses are only ever compared
at word boundaries, each candidate word entering with its complete
`λ·ln P_GF + (1−λ)·ln P_LM` in one atomic step (grammar and LM mass
aligned per word).

## Pros and cons

**Phase 1 — token-level**

- (+) ~100 lines on top of HF `generate`; sampling, batching, length
  penalties, per-sample grammars all inherited.
- (+) KV caching handled by HF → one forward pass per step regardless of
  candidate count.
- (+) Drop-in replacement for the binary `GFConstraint` (λ = 0 is exactly
  the old behavior — used as the regression test).
- (−) Grammar is tokenizer-specific; recompile to switch models.
- (−) Mid-word score skew (above).
- (−) Terminal encoding workaround (`id_prefix="t"`) required by the
  C-runtime digit-lexing bug.

**Phase 2 — word-level**

- (+) One word-level grammar for any model; no rewriting, no `id_prefix`.
- (+) Probabilistically clean: fusion aligned at word boundaries; EOS
  competes against continuations on both sides of the interpolation.
- (+) Transparent output (`gf_logprob` / `llm_logprob` per hypothesis).
- (−) Hand-rolled search: no sampling, no HF conveniences; more code to
  maintain.
- (−) Recomputes the full context every step (no KV-cache reuse across
  steps — deliberate, see docstring; negligible at tens of words but the
  first thing to optimize for long outputs).
- (−) One forward pass per step with batch = #candidates × #beams, vs.
  Phase 1's single-row-per-beam pass.

## Can they be combined?

Not within one run — each defines the beam's step unit, so one beam
cannot be advanced by both. Sequential composition works naturally:

- **Generate-then-rerank**: decode with Phase 1 (fast, HF-native,
  `num_return_sequences > 1`), then re-score the finished candidates with
  Phase 2's word-aligned objective (or simply with `parse` tree costs +
  LM word scores) and pick the best. This removes the mid-word skew's
  effect on the *final ranking*, though not on what was pruned mid-search.
- **Shared tuning**: λ and `gf_temperature` mean the same thing in both;
  sweep once, apply to either (`examples/quickstart_probs.py` runs the
  sweep side by side).

## When to use which

- Production-style decoding, many prompts, want sampling or HF options →
  **Phase 1**.
- Model-agnostic setups, research on the fusion itself, need faithful
  word-level probabilities or interpretable per-expert scores →
  **Phase 2**.
- Highest-quality final ranking on a budget → Phase 1 with
  `num_return_sequences`, reranked by Phase 2's objective.
