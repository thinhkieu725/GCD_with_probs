"""Full pipeline: raw RGL grammar source -> token-ID grammar -> constrained LLM.

One straight-line script covering every stage between "grammar on disk" and
"generated text", for the RGL English grammar (`Lang`/`LangEng`) plus its
treebank-estimated probability file (`test.probs`, see
grammars/rgl/test.probs):

  Stage 0  locate sources        grammars/rgl/{abstract,common,english,prelude,api}
  Stage 1  compile + canonicalize  gf -make --probs=... -output-format=canonical_gf
  Stage 2  tokenize terminals      canonical literals -> "<id_prefix><token id>"
  Stage 3  compile token grammar   gf -make --probs=...  (same probs file)
  Stage 4  load LLM + constraint   GFProbConstraint + ConstrainedGenerator
  Stage 5  generate                prompt from stdin, one run per lambda

All paths, names, and generation knobs are constants in the CONFIG block
below — edit those to point at a different grammar/model instead of editing
the pipeline logic.

Requires the `gf` binary on PATH and `pip install pgf`.

    python examples/full_pipeline.py
"""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from gf_gcd import ConstrainedGenerator, GFProbConstraint, GFProbRuntime
from gf_gcd.grammar_tools import compile_grammar, ids_to_gf_lin, tokenize_terminal

# ============================================================== CONFIG =====

PROJECT_ROOT = Path(__file__).parent.parent

# Stage 0/1 — grammar source
GRAMMAR_SRC_DIR = PROJECT_ROOT / "grammars" / "rgl"          # where the .gf sources live
GRAMMAR_SRC_SUBDIRS = ["abstract", "common", "english", "prelude", "api"]  # -> gf --path
ABSTRACT_NAME = "Lang"                                        # abstract module name
CONCRETE_NAME = "LangEng"                                     # concrete module to compile
CONCRETE_FILE = GRAMMAR_SRC_DIR / "english" / f"{CONCRETE_NAME}.gf"
PROBS_FILE = GRAMMAR_SRC_DIR / "test.probs"                   # gf -make --probs=

# Stage 1-3 — compile output
BUILD_DIR = GRAMMAR_SRC_DIR / "build"                         # all generated artifacts land here
WORD_LEVEL_PGF = "Lang.pgf"                                   # gf -make default output name
TOKEN_CONCRETE_NAME = "LangTok"                               # renamed concrete for the token grammar
TOKEN_LEVEL_PGF = f"{TOKEN_CONCRETE_NAME}.pgf"
GF_BINARY = "gf"

# Stage 2 — tokenization
ID_PREFIX = "t"   # non-numeric marker required by the C runtime's complete() (see grammar_tools/tokenize.py)

# Stage 4/5 — model + decoding
# Device/quantization setup below is CUDA-specific — see docs/device_setup.md
# for what to change to run on CPU-only or Apple Silicon (mps) instead.
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # ungated on HF; alternative: "meta-llama/Llama-3.1-8B-Instruct" (gated, needs HF token + accepted license)
DEVICE_MAP = "cuda"       # single-GPU placement; requires `accelerate`
HALF_PRECISION = True     # float16 for the non-quantized layers (embeddings, norms)
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)  # requires `bitsandbytes`; keeps a 7-8B model comfortably under 12GB VRAM
LAMBDAS = [0.0, 0.3, 1.0]  # grammar weight sweep; 0.0 = pure masking (binary GCD)
GF_TEMPERATURE = 1.0       # >1 flattens the grammar's distribution towards uniform-over-allowed
NUM_BEAMS = 2
# The GF-side constraint (gf_gcd/gf_probs.py) reparses the whole prefix from
# scratch every step (the `pgf` C binding has no incremental/chart-reuse
# API), so its cost grows ~quadratically with MAX_NEW_TOKENS. Kept small
# here for fast iteration; raise it for an occasional slower "final" run.
MAX_NEW_TOKENS = 48
INCLUDE_BASELINE_WITHOUT_CONSTRAINT = True  # if True, also run the model without any GFProbConstraint (baseline)

# ============================================================= /CONFIG =====


def stage1_compile_and_canonicalize() -> Path:
    """gf -make the concrete with probabilities, dumping canonical GF alongside.

    Canonical GF has every paradigm evaluated down to plain string literals,
    which is what makes stage 2's textual tokenization possible on a grammar
    that (unlike the toy grammars) computes strings through smart paradigms.

    Returns the path to the canonicalized concrete syntax.
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    canonical_concrete = BUILD_DIR / "canonical" / f"{CONCRETE_NAME}.gf"
    if canonical_concrete.exists():
        print(f"[stage 1] reusing cached {canonical_concrete}")
        return canonical_concrete

    cmd = [
        GF_BINARY, "-make", "-output-format=canonical_gf",
        f"--probs={PROBS_FILE}",
        f"--gfo-dir={BUILD_DIR}",  # keep .gfo build litter out of the source tree
        "--path=" + ":".join(str(GRAMMAR_SRC_DIR / d) for d in GRAMMAR_SRC_SUBDIRS),
        str(CONCRETE_FILE),
    ]
    print(f"[stage 1] compiling {CONCRETE_FILE.name} (one-time, takes a few minutes)...")
    result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gf -make failed ({shlex.join(cmd)}):\n{result.stdout}\n{result.stderr}")
    print(f"[stage 1] wrote {BUILD_DIR / WORD_LEVEL_PGF} and {canonical_concrete}")
    return canonical_concrete


def stage2_tokenize_canonical(tokenizer, canonical_src: str) -> str:
    """Rewrite a canonical-GF concrete's terminals into token IDs.

    Same literal rewriting as `tokenize_concrete_grammar` (grammar_tools),
    preceded by three accommodations needed for canonical RGL output:

      - `pre {...}` prefix alternations (e.g. "a"/"an") collapse to their
        default branch, since the condition can never match a token-ID
        stream;
      - BIND-attached clitics ("'s", "'ll", ...) and punctuation-only
        terminals are tokenized *without* a leading space, matching how they
        actually attach in text;
      - remaining BIND/SOFT_BIND (in digits, markup, symbols) degrade to the
        empty token — spacing inside those constructions is lost.
    """
    default_branch = re.compile(r'_ =>\s*"((?:[^"\\]|\\.)*)"\s*\}$', re.DOTALL)

    def sub_pre(m: re.Match) -> str:
        d = default_branch.search(m.group(0))
        if d is None:
            raise ValueError(f"pre block without default branch: {m.group(0)}")
        return f'"{d.group(1)}"'

    src = re.sub(r"pre \{[^{}]*\}", sub_pre, canonical_src, flags=re.DOTALL)

    def sub_clitic(m: re.Match) -> str:
        ids = tokenize_terminal(tokenizer, m.group(1), prefix_space=False)
        return f"({ids_to_gf_lin(ids, id_prefix=ID_PREFIX)})"

    src = re.sub(r'Predef\.BIND \+\+\s*"((?:[^"\\]|\\.)*)"', sub_clitic, src)
    src = re.sub(r"Predef\.(SOFT_)?BIND", '""', src)
    src = src.replace("Predef_nonExist", "Predef.nonExist")  # canonical printer quirk

    def sub_literal(m: re.Match) -> str:
        content = m.group(1)
        if content == "":
            return '""'
        prefix_space = any(ch.isalnum() for ch in content)
        ids = tokenize_terminal(tokenizer, content, prefix_space=prefix_space)
        return f"({ids_to_gf_lin(ids, id_prefix=ID_PREFIX)})"

    return re.sub(r'"((?:[^"\\]|\\.)*)"', sub_literal, src)


def stage3_compile_token_grammar(tokenizer, canonical_concrete_path: Path) -> Path:
    """Tokenize the canonical concrete and compile it into a token-ID .pgf."""
    marker = BUILD_DIR / ".tokenizer"
    token_pgf = BUILD_DIR / TOKEN_LEVEL_PGF
    if token_pgf.exists() and marker.exists() and marker.read_text() == MODEL_NAME:
        print(f"[stage 2-3] reusing cached {token_pgf}")
        return token_pgf

    print(f"[stage 2] rewriting terminals to {MODEL_NAME} token IDs...")
    canonical_src = canonical_concrete_path.read_text()
    token_src = stage2_tokenize_canonical(tokenizer, canonical_src).replace(
        f"concrete {CONCRETE_NAME} of {ABSTRACT_NAME} =",
        f"concrete {TOKEN_CONCRETE_NAME} of {ABSTRACT_NAME} =",
        1,
    )
    token_concrete_file = BUILD_DIR / f"{TOKEN_CONCRETE_NAME}.gf"
    token_concrete_file.write_text(token_src)

    abstract_file = BUILD_DIR / f"{ABSTRACT_NAME}.gf"
    abstract_file.write_text((canonical_concrete_path.parent / f"{ABSTRACT_NAME}.gf").read_text())

    print(f"[stage 3] compiling {token_concrete_file.name}...")
    compile_grammar(
        [abstract_file, token_concrete_file],
        BUILD_DIR,
        output_name=TOKEN_CONCRETE_NAME,
        probs_file=PROBS_FILE,
        gf_binary=GF_BINARY,
    )
    marker.write_text(MODEL_NAME)
    print(f"[stage 3] wrote {token_pgf}")
    return token_pgf


def stage4_load_model_and_constraint(
    runtime: GFProbRuntime, eos_token_id, pad_token_id, lambda_gf: float
) -> GFProbConstraint:
    """Build the token-level, probability-weighted grammar constraint.

    Takes a shared `runtime` rather than a `grammar_dir`: the GF-side
    completion distribution doesn't depend on `lambda_gf` (it's only scaled
    afterward in the fused score), so every lambda in the sweep can reuse
    one loaded grammar and one completion cache instead of each rebuilding
    its own from scratch.
    """
    return GFProbConstraint(
        runtime=runtime,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        lambda_gf=lambda_gf,
        gf_temperature=GF_TEMPERATURE,
        id_prefix=ID_PREFIX,
    )


def stage5_generate(generator: ConstrainedGenerator, prompt: str) -> None:
    """Run one generation per configured lambda and print the results."""
    print(f"\nmodel={MODEL_NAME}  prompt={prompt!r}")

    if INCLUDE_BASELINE_WITHOUT_CONSTRAINT:
        print(f"baseline (no GFProbConstraint, num_beams={NUM_BEAMS}):")
        result = generator.generate(
            prompt, constraint=None, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
        )
        print(f"  unconstrained  ->  {result.first_texts[0]!r}")

    print(f"token-level fusion (GFProbConstraint + num_beams={NUM_BEAMS}):")
    runtime = GFProbRuntime(grammar_dir=str(BUILD_DIR), default_pgf=TOKEN_LEVEL_PGF)
    for lam in LAMBDAS:
        constraint = stage4_load_model_and_constraint(
            runtime=runtime,
            eos_token_id=generator.tokenizer.eos_token_id,
            pad_token_id=generator.tokenizer.pad_token_id,
            lambda_gf=lam,
        )
        result = generator.generate(
            prompt, constraint=constraint, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
        )
        print(f"  lambda={lam:4.1f}  ->  {result.first_texts[0]!r}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    canonical_concrete_path = stage1_compile_and_canonicalize()
    stage3_compile_token_grammar(tokenizer, canonical_concrete_path)

    generator = ConstrainedGenerator(
        MODEL_NAME,
        half_precision=HALF_PRECISION,
        device_map=DEVICE_MAP,
        hf_model_kwargs={"quantization_config": QUANTIZATION_CONFIG},
    )

    prompt = input("Prompt: ").rstrip()
    if not prompt:
        sys.exit("empty prompt, nothing to do")

    stage5_generate(generator, prompt)


if __name__ == "__main__":
    main()
