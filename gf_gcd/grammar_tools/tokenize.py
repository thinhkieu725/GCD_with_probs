"""Helpers to turn a word-level GF grammar into a token-ID-level grammar.

Grammar-constrained decoding requires the grammar and the LM to share an
alphabet: every terminal in the compiled .pgf must be a token ID of the target
tokenizer, written as a string (e.g. `"31" ++ "287"`). These helpers cover the
mechanical part of that transformation; for complex grammars see the original
authors' tooling at https://github.com/Saibo-creator/GF_helper.

Caveats to keep in mind when tokenizing terminals:
- Subword tokenizers encode the same string differently at the start of a
  sequence vs. after a space; choose `prefix_space` per terminal accordingly.
- The grammar must eventually allow the EOS token id, or generation cannot
  stop (alternatively set `eos_token_id` on GFConstraint as a fallback).
"""

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence, Union


def tokenize_terminal(tokenizer, text: str, prefix_space: bool = True) -> List[int]:
    """Encode one terminal string to token IDs, without special tokens.

    `prefix_space=True` encodes the terminal as it appears mid-sentence
    (preceded by a space), which is the common case for word-level terminals.
    """
    if prefix_space:
        text = " " + text
    return tokenizer.encode(text, add_special_tokens=False)


def ids_to_gf_lin(ids: Sequence[int], id_prefix: str = "") -> str:
    """Render token IDs as a GF linearization: [31, 287] -> '"31" ++ "287"'.

    `id_prefix` prepends a marker to every ID (`"t31" ++ "t287"`). Digit-only
    terminals are lexed as Int literals by the GF C runtime, which breaks its
    `complete` (next-token) queries — grammars meant for the probability-aware
    runtime (`GFProbConstraint`) must therefore use a non-numeric prefix.
    The `gf --server` path is indifferent to the prefix.
    """
    if len(ids) == 0:
        return '""'
    return " ++ ".join(f'"{id_prefix}{i}"' for i in ids)


_STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def tokenize_concrete_grammar(
    tokenizer,
    concrete_gf_source: str,
    prefix_space: bool = True,
    id_prefix: str = "",
) -> str:
    """Rewrite every string literal in a concrete .gf source into token IDs.

    This is a purely textual transformation: each double-quoted literal is
    replaced by the `++`-joined token IDs of its content. It handles flat
    lexical grammars well; grammars that compute strings (glue `+`, tables,
    regex patterns) need manual treatment.

    Pass a non-numeric `id_prefix` (e.g. "t") when the grammar is meant for
    `GFProbConstraint` — see `ids_to_gf_lin`.
    """

    def replace(match: re.Match) -> str:
        content = match.group(1)
        if content == "":
            return '""'
        ids = tokenize_terminal(tokenizer, content, prefix_space=prefix_space)
        return f"({ids_to_gf_lin(ids, id_prefix=id_prefix)})"

    return _STRING_LITERAL.sub(replace, concrete_gf_source)


def compile_grammar(
    gf_files: Union[str, Path, Sequence[Union[str, Path]]],
    output_dir: Union[str, Path],
    output_name: str = None,
    gf_binary: str = "gf",
    probs_file: Union[str, Path, None] = None,
) -> Path:
    """Compile concrete .gf file(s) into a .pgf with `gf -make`.

    `probs_file` is a GF probabilities file (lines of `FunName <prob>`,
    probabilities normalized per result category) passed as `--probs=`;
    the resulting .pgf ranks parses/completions by these probabilities.
    Probabilities attach to abstract functions, so the same probs file works
    for a word-level grammar and its token-ID rewriting.

    Returns the path of the produced .pgf inside `output_dir`.
    """
    if shutil.which(gf_binary) is None:
        raise RuntimeError(f"GF binary {gf_binary!r} not found on PATH")

    if isinstance(gf_files, (str, Path)):
        gf_files = [gf_files]
    gf_files = [Path(f).resolve() for f in gf_files]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [gf_binary, "-make"]
    if probs_file is not None:
        probs_file = Path(probs_file).resolve()
        if not probs_file.exists():
            raise FileNotFoundError(f"probs file does not exist: {probs_file}")
        cmd += [f"--probs={probs_file}"]
    if output_name:
        cmd += ["-name", output_name]
    cmd += [str(f) for f in gf_files]

    result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gf -make failed ({shlex.join(cmd)}):\n{result.stdout}\n{result.stderr}"
        )

    pgfs = sorted(output_dir.glob("*.pgf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pgfs:
        raise RuntimeError(f"gf -make succeeded but produced no .pgf in {output_dir}")
    return pgfs[0]
