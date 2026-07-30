# gf-gcd

Grammar-constrained decoding (GCD) for HuggingFace causal LMs using
[Grammatical Framework](https://www.grammaticalframework.org/) (GF) grammars.

This is the task-agnostic core extracted from
[epfl-dlab/GCD](https://github.com/epfl-dlab/GCD) ("Grammar-Constrained Decoding
for Structured NLP Tasks without Finetuning", EMNLP 2023), with the experiment
scaffolding (PyTorch Lightning, Hydra, wandb, IE/CP/ED task code) removed.

## How it works

1. You write a grammar in GF describing the language the LM is allowed to generate.
2. The grammar's terminals are rewritten into the **token IDs of your tokenizer**
   (so the grammar and the LM share an alphabet) and compiled to a `.pgf`.
3. At every decoding step, a pool of `gf --server` processes is asked which
   token IDs the grammar allows next; the answer is fed to
   `model.generate(prefix_allowed_tokens_fn=...)`, masking all other tokens.

## Probability-fused decoding (grammar × LM beam search)

Beyond binary masking, a grammar compiled with `gf -make --probs=<file>`
(probabilities per abstract function, e.g. estimated from a treebank) can be
*fused* with the LM instead of merely filtering it. Two granularities:

- **Token-level** (`GFProbConstraint`): drop-in replacement for `GFConstraint`
  backed by the GF C runtime (`pip install pgf`) instead of `gf --server`.
  Each step adds `lambda_gf * ln P_GF(token | prefix)` to the LM's log-probs
  and masks disallowed tokens; HF's `generate(num_beams=B)` then performs beam
  search over `log P_LM + λ·log P_GF`. `lambda_gf=0` reproduces pure masking.
  **Terminal encoding**: the C runtime's `complete()` chokes on digit-only
  terminals (they lex as `Int` literals), so token-ID grammars for this path
  must use a letter prefix — pass `id_prefix="t"` to
  `tokenize_concrete_grammar` and to the constraint.
- **Word-level** (`WordBeamSearcher`): custom beam search that advances one
  *grammar word* at a time over the natural word-level grammar (tokenizer
  independent). The grammar proposes words with probabilities; the LM scores
  each candidate as the sum of its subword log-probs; hypotheses grow by
  `λ·ln P_GF + (1-λ)·ln P_LM`. End-of-sentence competes on the same scale:
  when the prefix parses, GF's full-tree probability is normalized against
  the continuations, and the LM contributes its EOS-token probability.

Both share `GFProbRuntime`, which turns the C runtime's cumulative
`(cost, token, cat, fun)` completions into per-step conditionals (merging
duplicate derivations by logsumexp, renormalizing per step; `gf_temperature`
flattens or sharpens the grammar's opinion). See
[examples/quickstart_probs.py](examples/quickstart_probs.py) for a λ sweep
over both modes, and `grammars/toy/*.probs` for the probs-file format
(`FunName<TAB>probability`, normalized per result category).

## Requirements

- Python ≥ 3.9 with `torch`, `transformers`, `requests` (`pip install -e .`)
- The GF binary on PATH ([install](https://www.grammaticalframework.org/download/))
- An LLM in HF format (model + tokenizer). Logits-level access is required, so
  API-only backends (e.g. Ollama) are not supported; GGUF weights can be loaded
  through `transformers`' `gguf_file` support.

## Quickstart

```python
from gf_gcd import ConstrainedGenerator, GFConstraint

generator = ConstrainedGenerator("gpt2")

with GFConstraint(
    grammar_dir="path/to/pgf_dir",
    default_grammar="MyGrammar.pgf",          # terminals = token IDs of gpt2's tokenizer
    eos_token_id=generator.tokenizer.eos_token_id,  # allow stopping once the derivation is complete
) as constraint:
    result = generator.generate(
        "My prompt:",
        constraint=constraint,
        max_new_tokens=32,
        num_beams=2,
        return_token_logprobs=True,
    )

print(result.first_texts[0])
print(result.token_logprobs[0])
```

See [examples/quickstart.py](examples/quickstart.py) for the full pipeline,
including turning a word-level grammar into a token-ID grammar.

## Preparing a grammar

Write your grammar at the word level (abstract + concrete `.gf`), then:

```python
from transformers import AutoTokenizer
from gf_gcd.grammar_tools import tokenize_concrete_grammar, compile_grammar

tokenizer = AutoTokenizer.from_pretrained("your-model")
token_level_src = tokenize_concrete_grammar(tokenizer, open("MyGramEng.gf").read())
# write token_level_src next to the abstract grammar, then:
compile_grammar("MyGramTok.gf", output_dir="grammars/pgf")
```

`tokenize_concrete_grammar` is a textual literal-rewriter that covers flat
lexical grammars; for grammars that compute strings, see the original authors'
[GF_helper](https://github.com/Saibo-creator/GF_helper). The grammar is
**tokenizer-specific**: switching models with a different tokenizer means
re-running this step.

Pitfalls: subword tokenizers encode a string differently at sequence start vs.
after a space (`prefix_space`), and the grammar must eventually allow EOS —
either encode the EOS token id in the grammar or set `eos_token_id` on
`GFConstraint` as a fallback once the derivation is complete.

**A compiled grammar accepts exactly one sentence.** All three decoding
paths (`GFConstraint`, `GFProbConstraint`, `WordBeamSearcher`) query the
grammar with only the tokens/words generated since the last sentence
boundary, not the whole output — so `max_new_tokens`/`max_words` can span
several grammar-conformant sentences rather than being wasted once the
first one completes, or breaking the grammar's parse once a second
sentence's tokens get appended to the first's. A new sentence can only
begin right where the grammar forces the previous one to end (no legal
continuation left); at that point generation may also just stop
(`eos_token_id`/`GF_EOS`). See `_SentenceCursor` in
[gf_gcd/constraints.py](gf_gcd/constraints.py) and `WordHypothesis.sentence_words`
in [gf_gcd/word_beam.py](gf_gcd/word_beam.py) for the mechanism.

## Layout

```
gf_gcd/
├── runtime.py        # GF HTTP server pool + token-ID completion queries
├── gf_probs.py       # GF C runtime wrapper: next-token *distributions* (needs pip install pgf)
├── constraints.py    # Constraint interface + GFConstraint (mask) + GFProbConstraint (fusion)
├── word_beam.py      # word-synchronized beam search fusing grammar and LM scores
├── generator.py      # HF model wrapper: generate() with constraint + token logprobs
└── grammar_tools/    # terminal tokenization + gf -make compilation helpers (--probs)
grammars/toy/         # Food (word-level) and Seq (token-ID-style) demo grammars + .probs
tests/                # run with: pytest  (needs `gf` on PATH; no GPU/model needed)
```

## Extending

- **New constraint sources** (trie, regex, JSON schema): implement
  `Constraint.prefix_allowed_tokens_fn`.
- **New backends** (vLLM, llama.cpp): both expose logits processors; write an
  adapter that calls the same `GFServerRuntimeForLM.get_prefix_allowed_tokens_for_LM`.
- **Per-sample grammars**: pass `grammars=[...]` to `generate` — one `.pgf`
  filename per prompt, for input-dependent constraints.
