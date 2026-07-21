"""Token-level grammar-LM fusion on the full RGL English grammar (Lang).

Tests the current token-level constraint (`GFProbConstraint` + HF beam search)
at real scale: the Resource Grammar Library's `Lang` abstract with its English
concretization `LangEng` and the RGL's treebank-estimated probability file
(gf-rgl/src/test.probs, pulled into grammars/rgl/).

The token-level constraint needs a token-ID grammar, but the RGL computes
strings through smart paradigms, so `tokenize_concrete_grammar` cannot be
applied to the source .gf files. Bridge (build step, cached in
grammars/rgl/build):

  1. compile LangEng.gf with `--probs=test.probs`, additionally dumping the
     compiled grammar back to *canonical* GF (`-output-format=canonical_gf`):
     all paradigms evaluated, every terminal a plain string literal;
  2. rewrite the canonical concrete's literals into token IDs — the same
     transformation as `tokenize_concrete_grammar`, with three RGL-specific
     accommodations (see `tokenize_canonical`):
       - `pre {...}` prefix alternations ("a"/"an") collapse to their default
         branch, since the conditions can never match a token-ID stream;
       - BIND-attached clitics ("'s", "'ll", ...) and punctuation-only
         terminals are tokenized without the leading space, as they appear
         in real text;
       - remaining BIND/SOFT_BIND (digits, markup, symb) degrade to the
         empty token — spacing inside those constructions is lost;
  3. compile the rewritten concrete as LangTok.pgf with the same probs file.

Requires the `gf` binary on PATH and `pip install pgf`; asks for the prompt
on stdin:

    python examples/rgl_token_level.py [model_name] [lambda ...]

The token grammar is tokenizer-specific: delete grammars/rgl/build when
switching to a model with a different tokenizer.
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from gf_gcd import ConstrainedGenerator, GFProbConstraint
from gf_gcd.grammar_tools import compile_grammar, ids_to_gf_lin, tokenize_terminal

PROJECT_ROOT = Path(__file__).parent.parent
RGL = PROJECT_ROOT / "grammars" / "rgl"
BUILD = RGL / "build"
SRC_DIRS = ["abstract", "common", "english", "prelude", "api"]

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
LAMBDAS = [float(x) for x in sys.argv[2:]] or [0.0, 0.3, 1.0]


def tokenize_canonical(tokenizer, src: str, id_prefix: str = "t") -> str:
    """Rewrite a canonical-GF concrete's terminals into token IDs.

    Same literal rewriting as `tokenize_concrete_grammar`, preceded by the
    canonical/RGL-specific normalizations documented in the module docstring.
    """
    # pre {...} -> its default branch (prefix conditions can't fire on IDs)
    default_branch = re.compile(r'_ =>\s*"((?:[^"\\]|\\.)*)"\s*\}$', re.DOTALL)

    def sub_pre(m: re.Match) -> str:
        d = default_branch.search(m.group(0))
        if d is None:
            raise ValueError(f"pre block without default branch: {m.group(0)}")
        return f'"{d.group(1)}"'

    src = re.sub(r"pre \{[^{}]*\}", sub_pre, src, flags=re.DOTALL)

    # BIND ++ "clitic" -> the clitic's IDs without a leading space
    def sub_clitic(m: re.Match) -> str:
        ids = tokenize_terminal(tokenizer, m.group(1), prefix_space=False)
        return f"({ids_to_gf_lin(ids, id_prefix=id_prefix)})"

    src = re.sub(r'Predef\.BIND \+\+\s*"((?:[^"\\]|\\.)*)"', sub_clitic, src)

    # leftover BIND/SOFT_BIND precede non-literals (digits, markup): drop them
    src = re.sub(r"Predef\.(SOFT_)?BIND", '""', src)

    # canonical printer emits Predef_nonExist; the compiler wants it qualified
    src = src.replace("Predef_nonExist", "Predef.nonExist")

    # literal rewriting as in tokenize_concrete_grammar, except that
    # punctuation-only terminals attach without a leading space
    def sub_literal(m: re.Match) -> str:
        content = m.group(1)
        if content == "":
            return '""'
        prefix_space = any(ch.isalnum() for ch in content)
        ids = tokenize_terminal(tokenizer, content, prefix_space=prefix_space)
        return f"({ids_to_gf_lin(ids, id_prefix=id_prefix)})"

    return re.sub(r'"((?:[^"\\]|\\.)*)"', sub_literal, src)


def build_grammars(tokenizer) -> None:
    """Produce build/Lang.pgf (word-level) and build/LangTok.pgf (token-ID)."""
    marker = BUILD / ".tokenizer"
    if (BUILD / "LangTok.pgf").exists() and marker.exists() and marker.read_text() == MODEL:
        return
    BUILD.mkdir(exist_ok=True)

    # 1. word-level compile with probs + canonical dump (writes Lang.pgf and
    #    canonical/{Lang,LangEng}.gf into BUILD)
    cmd = [
        "gf", "-make", "-output-format=canonical_gf",
        f"--probs={RGL / 'test.probs'}",
        f"--gfo-dir={BUILD}",
        "--path=" + ":".join(str(RGL / d) for d in SRC_DIRS),
        str(RGL / "english" / "LangEng.gf"),
    ]
    print("compiling RGL LangEng (one-time, takes a few minutes)...")
    result = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gf -make failed ({shlex.join(cmd)}):\n{result.stdout}\n{result.stderr}"
        )

    # 2. token-ID rewriting of the canonical concrete
    print(f"rewriting terminals to {MODEL} token IDs...")
    canonical = (BUILD / "canonical" / "LangEng.gf").read_text()
    token_level = tokenize_canonical(tokenizer, canonical).replace(
        "concrete LangEng of Lang =", "concrete LangTok of Lang =", 1
    )
    (BUILD / "LangTok.gf").write_text(token_level)
    (BUILD / "Lang.gf").write_text((BUILD / "canonical" / "Lang.gf").read_text())

    # 3. compile the token-ID grammar with the same probs
    compile_grammar(
        [BUILD / "Lang.gf", BUILD / "LangTok.gf"], BUILD,
        output_name="LangTok", probs_file=RGL / "test.probs",
    )
    marker.write_text(MODEL)
    print(f"built {BUILD / 'LangTok.pgf'}\n")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    build_grammars(tokenizer)

    # device_map=None: plain CPU/default placement, no accelerate dependency
    generator = ConstrainedGenerator(MODEL, device_map=None)
    prompt = input("Prompt: ").rstrip()
    if not prompt:
        sys.exit("empty prompt, nothing to do")

    print(f"\nmodel={MODEL}  prompt={prompt!r}")
    print("token-level fusion (GFProbConstraint + num_beams=4):")
    for lam in LAMBDAS:
        constraint = GFProbConstraint(
            grammar_dir=str(BUILD),
            default_grammar="LangTok.pgf",
            eos_token_id=generator.tokenizer.eos_token_id,
            pad_token_id=generator.tokenizer.pad_token_id,
            lambda_gf=lam,
            id_prefix="t",
        )
        result = generator.generate(
            prompt, constraint=constraint, max_new_tokens=24, num_beams=4
        )
        print(f"  lambda={lam:4.1f}  ->  {result.first_texts[0]!r}")


if __name__ == "__main__":
    main()
