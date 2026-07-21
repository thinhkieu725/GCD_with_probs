# `chainify_grammar`: scope and risky edge cases

> For usage instructions and the full API reference, see
> [`docs/chainify.md`](chainify.md). This document covers scope, guarantees,
> and the edge cases the transform deliberately does not handle.

[`gf_gcd/grammar_tools/chainify.py`](../gf_gcd/grammar_tools/chainify.py)
restructures a token-ID GF grammar (the output of `tokenize_concrete_grammar`,
or any hand-written grammar with the same shape) so that every function
contributes at most one literal token to its own linearization. That is the
fix for the GF C runtime's mid-literal `pgf_complete` bug documented in
`examples/repro_midliteral.py`: `pgf_complete` raises `ParseError` when the
parse position lands strictly inside a function's own run of consecutive
literal tokens (e.g. after `t1416` of `Delicious = "t1416" ++ "t563"`), but
handles positions on a function-application boundary correctly. Once no
function contributes more than one token, every completion position is a
function boundary, so the bug cannot fire.

This has been verified end to end, not just argued: `tests/test_chainify.py`
compiles a chainified grammar and shows `complete("t100 ")` — which raises
`pgf.ParseError` on the original `SeqTok.pgf` — returns the correct
`{t200, t300}` candidates on the rewritten one, and that
`GFProbRuntime.next_token_logprobs` goes from returning `{}` (the documented
dead-end fallback) to the correct probability-weighted distribution.

## Why source-to-source, not the compiled `.pgf`

The task allowed either starting point. The compiled `.pgf` was rejected:
the Python `pgf` bindings only expose a *reader* API (`pgf.readPGF` and
query methods) — there is no supported way to add categories/functions to a
loaded grammar and write a new `.pgf` back out. Reconstructing GF source and
recompiling with `gf -make` is the only route that produces a grammar the
runtime can actually load, so it is also the more "stable" one in the sense
the task asked about.

## What is guaranteed

- **Never produces an invalid grammar.** Every function is rewritten only
  when the tool is certain of its shape (see below); otherwise it is copied
  through unchanged. Leaving a function alone is always type-safe — the
  invariant the whole design leans on. A file that `chainify_grammar`
  can partially transform is not a partial success in the risky sense: the
  parts it didn't touch are provably byte-identical to the input, so the
  worst outcome is "this word still has the bug," never a broken grammar.
- **Never silently mistransforms.** If the tool cannot fully understand a
  module's structure (not just one function's RHS), it raises
  `GFSyntaxError` and produces no output at all, rather than guessing.
- **Idempotent.** Running it on its own output is a no-op (verified in
  `tests/test_chainify.py`).
- **Probabilities carry over unchanged.** A `--probs=` file that lists the
  *original* function names keeps working after chainification, because
  every synthetic tail category has exactly one producing function, which
  gets probability 1 after PGF's per-category normalization regardless of
  whether it's listed in the probs file (verified empirically, not just
  argued, in the same test file).

## What a function must look like to be rewritten

A function `F` is eligible only if **all** of the following hold:

1. Its abstract `fun F : C1 -> ... -> Cn -> C0` declaration parses as a
   plain arrow-chain of bare category names (no higher-order argument types).
2. `C0`'s concrete `lincat` is exactly `Str` (a flat string category — not a
   record, not a table type, not a discontinuous/tuple type).
3. Its `lin F x1 ... xn = <rhs>` rule has a plain parameter pattern (bare
   identifiers only, no record patterns) and `<rhs>` is *provably* a flat
   `++`-chain of string literals and references to `x1..xn` — nothing else.

If any of these fails, `F` is left exactly as written and recorded in
`ChainifyReport.skipped` with a reason. This is deliberately conservative:
the tool would rather under-fix a grammar than risk emitting something that
doesn't compile or changes meaning.

## Known limitations / risky edge cases

These are constructs the tool does not attempt to handle. None of them
crash the tool or corrupt the grammar — they just mean the affected
function(s) keep the mid-literal bug, which is reported via
`ChainifyReport.skipped` so the caller can see exactly what wasn't fixed.

- **Tables, `case`, `variants`, `pre {...}`.** Any parameter-dependent or
  conditional choice of string in a linearization is out of scope. This is
  the big one in practice: it means resource-grammar-style entries (agreement
  tables, inflection paradigms) are *not* rewritten. `examples/rgl_token_level.py`
  already works around the RGL's parameter-dependence a different way (via
  `-output-format=canonical_gf`, which evaluates tables to a fixed choice at
  compile time before this tool would even see it); grammars that still have
  live tables/cases in their concrete syntax will see most functions skipped.
  This tool's realistic target is flat, lexicalized grammars — exactly what
  `tokenize_concrete_grammar` produces from a word-level grammar — not
  general resource grammars.
- **Glue (`+`) vs concatenation (`++`).** Character-level string glue is
  refused (a function using it is skipped), since after tokenization the two
  sides of a `+` no longer correspond to independently meaningful token
  boundaries.
- **Calls to `oper`s, other functions, records, projections, lambdas, table
  application (`!`).** Any RHS token that isn't a string literal, a `++`, a
  parenthesis, or a reference to the function's own parameter aborts that
  function's rewrite.
- **Module extension (`abstract X = Y ** {...}`, `concrete X of Y = Z **
  {...}`).** Rejected outright with `GFSyntaxError` — resolving inherited
  declarations from the base module is out of scope.
- **One concrete syntax per invocation.** `chainify_grammar` takes one
  abstract/concrete pair and renames both (default suffix `Chain`) so the
  output never collides with the input. It does not attempt to keep one
  abstract module in sync with *multiple* concrete syntaxes that each need
  different rewrites — every abstract function gains an extra argument for
  the concrete it was rewritten against, and a second concrete for the same
  (unrenamed) abstract would then need a matching extra `lin` argument it
  doesn't have. If you have several concretes over one abstract, run the
  tool once per concrete and treat each result as its own standalone grammar
  pair, or restrict token-ID grammars to a single concrete syntax per model
  (the normal setup for this project, since the token-ID rewrite is already
  tokenizer-specific).
- **Multiple modules per file, or content outside the single module block.**
  The tool assumes one `abstract`/`concrete` module per source string
  (matching this project's one-module-per-file convention, and
  `tokenize_concrete_grammar`'s output). Anything after the module's closing
  `}` is silently ignored rather than validated.
- **Nested block comments** (`{- {- ... -} -}`). The tokenizer's block
  comment match is non-nested and closes at the first `-}`; a nested comment
  will be mis-tokenized (and most likely surface as a `GFSyntaxError` from
  the resulting unbalanced braces, rather than silent corruption, but this
  isn't guaranteed for all inputs).
- **Comments and formatting inside `fun`/`lin` sections are not preserved.**
  Those two sections are always fully reparsed and regenerated one
  declaration per line (grouped declarations like `Wine, Cheese, Fish : Kind
  ;` are split apart if *any* member needs rewriting, so that the group's
  other members can keep their original signature). `cat`, `lincat`,
  `flags`, and any section keyword the tool doesn't specifically handle
  (`oper`, `param`, `def`, `data`, `printname`, `instance`, ...) are always
  copied through byte-for-byte, comments included — those sections are never
  inspected for rewrite candidates in the first place.
- **Fresh-name collisions are checked only against identifiers appearing in
  the two provided files.** If the grammar is split across additional files
  reachable only via `gf`'s `--path` (not the case for
  `tokenize_concrete_grammar`'s single-file output, but possible for
  hand-written multi-file grammars), a synthetic `Tail_...` name could in
  principle collide with an identifier defined elsewhere. The `Tail_<token
  text>_<counter>` naming scheme makes this very unlikely in practice but it
  is not ruled out.
- **Grammar growth is not bounded.** Each literal run of length *k* adds up
  to *k − 1* categories and functions, minus whatever is deduplicated by
  exact-tail sharing (two functions whose token runs end in the identical
  literal sequence reuse the same chain from that point on). For a
  vocabulary where multi-token words rarely share exact BPE suffixes,
  expect roughly one new category/function pair per extra token per word —
  the same order of growth the "one function per subtoken" approach was
  already known to cost; this tool's contribution is doing that split
  mechanically and only where needed; it does not change that fixed cost.

## If a function you need isn't being rewritten

Check `ChainifyReport.skipped[name]` for the reason, then either:

- restructure that one lexical entry by hand into the same chained shape
  `chainify_grammar` would produce (see the module docstring for the
  target shape), or
- fall back to the word-synchronized search in
  [`word_beam.py`](../gf_gcd/word_beam.py), which never queries the grammar
  mid-word in the first place and so is unaffected by this bug regardless of
  grammar shape.
