# `chainify`: working around the GF mid-literal completion bug

`gf_gcd/grammar_tools/chainify.py` rewrites a token-ID GF grammar so that
every abstract-syntax function contributes at most one literal token to its
own linearization, then (optionally) compiles the result straight to a
`.pgf`. This is a fix for a real bug in the GF C runtime: `pgf_complete`
raises `ParseError` when the parse position lands strictly inside a
function's own run of consecutive literal tokens (e.g. between `t1416` and
`t563` of `Delicious = "t1416" ++ "t563"`), even though the same position
parses to a complete sentence just fine and completion works correctly at
every function-application boundary. See
[`docs/chainify_limitations.md`](chainify_limitations.md) for the mechanism,
the verified before/after behavior, and exactly which grammars this applies
to.

This document is the "how do I actually use it" reference.

## When you need this

You have a grammar whose terminals are LLM token IDs (typically produced by
[`tokenize_concrete_grammar`](../gf_gcd/grammar_tools/tokenize.py)) and at
least one word encodes to more than one token under your tokenizer. If every
word in your grammar is exactly one token under the target tokenizer (as in
the toy `Food`/GPT-2 demo), you don't need this — the bug never fires.

## Quick start

```python
from gf_gcd.grammar_tools import chainify_and_compile

pgf_path, report = chainify_and_compile(
    "grammars/toy/Seq.gf",       # abstract .gf -- read-only, never modified
    "grammars/toy/SeqTok.gf",    # concrete .gf (token-ID terminals) -- read-only
    output_dir="build",          # where the compiled .pgf lands
    probs_file="grammars/toy/Seq.probs",  # optional, same file as before chainify
)

print(report)          # what got rewritten, what got skipped and why
print(pgf_path)        # build/SeqChain.pgf -- use this with GFProbRuntime
```

```python
from gf_gcd.gf_probs import GFProbRuntime

runtime = GFProbRuntime(pgf_path.parent, default_pgf=pgf_path.name, concrete="SeqTokChain")
runtime.next_token_logprobs(["t100"])   # now returns real candidates instead of {}
```

That's the whole workflow for the common case: rewrite, compile, point
`GFProbRuntime` (or `GFProbConstraint`) at the new `.pgf`. The intermediate
rewritten `.gf` source is written to a private temporary directory and
deleted once the `.pgf` is built — it's an implementation detail, not
something you need to manage.

## Where this fits in the pipeline

```
word-level grammar (Food.gf / FoodEng.gf)
        |  tokenize_concrete_grammar(tokenizer, ...)
        v
token-ID grammar source (has multi-token words -- may hit the bug)
        |  chainify_and_compile(...)
        v
build/FoodChain.pgf   (every function <= 1 literal token -- bug-free)
        |
        v
GFProbRuntime / GFProbConstraint / WordBeamSearcher
```

`chainify_and_compile` is meant to sit right after the tokenizer rewrite
and right before compiling for real use — it's a drop-in replacement for a
plain `compile_grammar(...)` call once you know your vocabulary has
multi-token words.

## API reference

### `chainify_and_compile(...)` — recommended entry point

```python
chainify_and_compile(
    abstract_path, concrete_path, output_dir,
    output_name=None,
    new_abstract_name=None, new_concrete_name=None,
    gf_binary="gf", probs_file=None,
    keep_gf_files=False,
) -> (pgf_path: Path, report: ChainifyReport)
```

- `abstract_path` / `concrete_path`: the original `.gf` pair. **Read-only —
  never written to, regardless of any other argument.**
- `output_dir`: where the compiled `.pgf` is placed (created if missing).
  Safe to pass the same directory the originals live in; only the `.pgf`
  (and, with `keep_gf_files=True`, a renamed copy of the rewritten `.gf`
  pair) is written there, never a file that collides with the originals.
- `output_name`: base name for the `.pgf`, forwarded to `compile_grammar`
  (defaults to the rewritten abstract module's name, e.g. `SeqChain.pgf`).
- `new_abstract_name` / `new_concrete_name`: names for the rewritten
  modules (default: original name + `"Chain"`, e.g. `Seq` → `SeqChain`).
  Renaming is what lets the rewritten grammar coexist with the original
  without a module-name clash.
- `probs_file`: same `--probs=` file you'd already use — it lists the
  *original* function names, and keeps working unmodified after
  chainification (see "Probabilities" below).
- `keep_gf_files`: if `True`, also copies the rewritten `.gf` pair into
  `output_dir` (useful for debugging what changed). Default `False`: the
  rewritten source is deleted once compilation succeeds.
- Raises `GFSyntaxError` if the input can't be safely parsed (see
  `chainify_grammar` below), `ValueError` if `keep_gf_files=True` and the
  computed destination would collide with an original file (only possible
  if you explicitly reuse the original module name *and* the original
  directory), or `RuntimeError` if `gf -make` itself fails.

### `chainify_files(...)` — rewrite only, no compile

```python
chainify_files(
    abstract_path, concrete_path, output_dir,
    new_abstract_name=None, new_concrete_name=None,
) -> (new_abstract_path: Path, new_concrete_path: Path, report: ChainifyReport)
```

Use this when you want the rewritten `.gf` source itself — to inspect it,
hand-edit a skipped function, or compile it yourself with custom `gf -make`
flags. Also read-only on the inputs; writes the rewritten pair into
`output_dir`.

### `chainify_grammar(...)` — pure in-memory transform

```python
chainify_grammar(
    abstract_src: str, concrete_src: str,
    new_abstract_name=None, new_concrete_name=None, verify=True,
) -> (new_abstract_src: str, new_concrete_src: str, report: ChainifyReport)
```

The transform with no filesystem I/O at all — takes and returns source
text. `chainify_files` and `chainify_and_compile` are both thin wrappers
around this. `verify=True` (default) re-parses the tool's own output and
raises `AssertionError` if any function it claims to have fixed still has a
multi-literal run — a self-check against bugs in this module, not in your
input.

### `ChainifyReport`

```python
report.transformed   # List[str] -- function names that were rewritten
report.skipped        # Dict[str, str] -- function name -> why it was left alone
report.new_categories # int -- synthetic categories added
report.new_functions   # int -- synthetic functions added
str(report)            # human-readable summary of all of the above
```

Always inspect `report.skipped` after a run on a new grammar — every entry
is a function that still has the mid-literal bug, because the tool
couldn't prove it safe to rewrite (see
[`docs/chainify_limitations.md`](chainify_limitations.md) for exactly which
constructs cause a skip, and what to do about it).

### `GFSyntaxError`

Raised (not caught internally) when the module-level structure can't be
parsed at all — mismatched abstract/concrete pair, `**` module extension,
or a file that isn't a single `abstract`/`concrete` module. This is
different from a per-function skip: it means the tool refuses to guess
about the *whole file*, and produces no output.

## Probabilities

If you compile with `probs_file=...` (a `FunName <prob>` list), pass the
same file you were already using — it lists your *original* function
names (`Short`, `Delicious`, ...), and those names are unaffected by
chainification. The new synthetic tail functions never need entries: each
one is the only producer of its category, so PGF's per-category
normalization gives it probability 1 whether or not it's listed.

## Guarantees this module gives you

- **Originals are never modified.** Every entry point only reads
  `abstract_path`/`concrete_path`; the rewrite always targets a fresh
  location (a temp dir for `chainify_and_compile`, `output_dir` for
  `chainify_files`), never the source paths.
- **Never produces an invalid grammar.** A function is rewritten only when
  its shape is fully understood; anything else is left byte-for-byte
  unchanged and reported in `report.skipped`, which is always safe.
- **Idempotent.** Running it again on its own output is a no-op.
- **Self-verifying.** `chainify_grammar(..., verify=True)` (the default)
  checks its own output before returning.

For the full list of constructs that cause a skip (tables, `case`,
`variants`, `pre {...}`, glue `+`, non-`Str` lincats, ...) and why full
generality across all valid GF grammars isn't achievable, see
[`docs/chainify_limitations.md`](chainify_limitations.md).

## Tests / verification

`tests/test_chainify.py` covers the pure-Python transform (no `gf` binary
needed) plus, gated behind the project's `requires_gf` marker,
compile-level round-trips through the real `gf` compiler and the `pgf` C
runtime bindings — including a direct regression test showing
`complete("t100 ")` moving from `pgf.ParseError` to a correct candidate set
on the chainified grammar, and `GFProbRuntime.next_token_logprobs` moving
from `{}` (dead end) to a correct probability-weighted distribution.
