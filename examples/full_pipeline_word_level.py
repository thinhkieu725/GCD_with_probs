"""Full pipeline: raw RGL grammar source -> word-level constrained LLM.

Word-level counterpart of `full_pipeline.py`. Same grammar and probability
file, same CONFIG-block style, but two of the token-level pipeline's stages
disappear: word-level fusion (`WordBeamSearcher`) advances the beam one
*grammar word* at a time over the natural word-level grammar, so nothing
needs to be rewritten into a tokenizer's token IDs.

  Stage 0  locate sources        grammars/rgl/{abstract,common,english,prelude,api}
  Stage 1  compile with probs     gf -make --probs=...  (word-level Lang.pgf)
  Stage 2  tokenize terminals     SKIPPED — no token-ID grammar is needed;
                                    the word-level grammar is used as-is
  Stage 3  compile token grammar  SKIPPED — same reason as stage 2
  Stage 4  load LLM + runtime     GFProbRuntime + ConstrainedGenerator
  Stage 5  generate                prompt from stdin, WordBeamSearcher per lambda

One consequence worth knowing: this pipeline is tokenizer-independent (the
same compiled Lang.pgf works for any model), whereas `full_pipeline.py` must
recompile its token-ID grammar whenever MODEL_NAME's tokenizer changes.

Cost warning specific to a real (non-toy) grammar: `WordBeamSearcher` scores
every (beam, grammar-legal word) pair with one batched LM forward pass per
step, so the batch width is beam_size * |legal continuations|. The toy Food
grammar has a handful of legal words per step; Lang has hundreds to
thousands (1593 at the very first step). Measured on CPU/gpt2, step 0 alone
(1593 candidates, beam_size=1) took ~44s; deeper steps and beam_size=3 push
this to minutes per step. BEAM_SIZE/MAX_WORDS below are kept small
accordingly — raise them only if your hardware (typically a GPU, which
batches this forward pass far more efficiently than CPU) can absorb it.

All paths, names, and decoding knobs are constants in the CONFIG block below.

Requires the `gf` binary on PATH and `pip install pgf`.

    python examples/full_pipeline_word_level.py
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

from gf_gcd import ConstrainedGenerator, GFProbRuntime, WordBeamSearcher

# ============================================================== CONFIG =====

PROJECT_ROOT = Path(__file__).parent.parent

# Stage 0/1 — grammar source
GRAMMAR_SRC_DIR = PROJECT_ROOT / "grammars" / "rgl"          # where the .gf sources live
GRAMMAR_SRC_SUBDIRS = ["abstract", "common", "english", "prelude", "api"]  # -> gf --path
ABSTRACT_NAME = "Lang"                                        # abstract module name
CONCRETE_NAME = "LangEng"                                     # concrete module to compile
CONCRETE_FILE = GRAMMAR_SRC_DIR / "english" / f"{CONCRETE_NAME}.gf"
PROBS_FILE = GRAMMAR_SRC_DIR / "test.probs"                   # gf -make --probs=

# Stage 1 — compile output (no stage 2/3 output: no token-ID grammar exists)
BUILD_DIR = GRAMMAR_SRC_DIR / "build"                         # all generated artifacts land here
WORD_LEVEL_PGF = "Lang.pgf"                                   # gf -make default output name
GF_BINARY = "gf"

# Stage 4/5 — model + decoding
# Device setup below is Apple Silicon (mps) — see docs/device_setup.md for
# what to change to run on CUDA or CPU-only instead.
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # small enough to fit unquantized in unified memory
# MODEL_NAME = "gpt2"  # small enough to fit unquantized in unified memory
DEVICE_MAP = "mps"        # single-GPU placement; requires `accelerate`
HALF_PRECISION = True     # float16 works fine on mps and halves memory
QUANTIZATION_CONFIG = None  # bitsandbytes doesn't support mps; no 4-bit path here
if DEVICE_MAP == "mps":
    # transformers' parallel weight-loading thread pool races on the MPS backend
    # (multiple threads calling tensor.to(device="mps") concurrently) and segfaults
    # during from_pretrained — force sequential loading. See docs/device_setup.md.
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
LAMBDAS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.3]  # grammar weight sweep; lambda_gf must be in [0, 1] here
GF_TEMPERATURE = 1.0       # >1 flattens the grammar's distribution towards uniform-over-allowed
BEAM_SIZE = 2              # kept small — see the cost warning in the module docstring
MAX_WORDS = 40              # ditto
NUM_RETURN = 1
# Search-quality knobs (see WordBeamSearcher.__init__ for the full rationale
# behind each). None of these change lambda_gf's grammar/LM tradeoff -- they
# make the beam search around it smarter.
LENGTH_PENALTY = 0.8       # GNMT-style exponent; 0 = off (raw log-prob sum, biased toward
                           # whichever hypothesis finishes fewer words first); ~0.6-1.0 corrects it
DIVERSITY_PENALTY = 0.3    # per-step penalty on a word already claimed by a higher-ranked
                           # survivor this step; 0 = off, larger = more lexically diverse beam
NO_REPEAT_NGRAM_SIZE = 3   # block candidates that would recreate an already-seen n-gram; 0 = off

# Decoding mode: "beam" (deterministic, pooled word-synchronized beam search,
# the knobs above) or "sample" (independent per-chain temperature/top-p/top-k
# sampling -- see WordBeamSearcher.__init__'s `do_sample` docstring). Switch
# by editing this one constant; DIVERSITY_PENALTY above is ignored (must be 0)
# in "sample" mode since it only makes sense for the pooled beam competition.
CANDIDATE_TOP_K = 0             # keep only the top_k highest-gf-score candidates; 0 = off
CANDIDATE_TOP_P = 1.0         # nucleus sampling cumulative-probability cutoff; 1.0 = off

DECODING_MODE = "sample"     # "beam" | "sample"
SAMPLE_NUM_CHAINS = 4      # independent parallel chains in "sample" mode (replaces BEAM_SIZE's role)
SAMPLE_TEMPERATURE = 0.85   # scales the *fused* score before top-k/top-p/min-p/typical-p; ~0 = greedy
SAMPLE_TOP_K = 60           # keep only the top_k highest-fused-score candidates; 0 = off
SAMPLE_TOP_P = 0.9         # nucleus sampling cumulative-probability cutoff; 1.0 = off
SAMPLE_MIN_P = 0.0         # keep candidates >= min_p * (top candidate's probability); 0.0 = off
SAMPLE_TYPICAL_P = 1.0     # locally typical sampling cumulative-probability cutoff; 1.0 = off
SAMPLE_SEED = None         # set an int for reproducible sampling runs

INCLUDE_BASELINE_WITHOUT_CONSTRAINT = True  # if True, also run the model without any grammar fusion (baseline)
BASELINE_MAX_NEW_TOKENS = 48  # unconstrained generation has no word-level stopping rule, so it needs its own token budget
SHOW_TIMER = True  # if True, log the time spent in GFProbConstraint.complete() and .score() (useful for profiling)

# ============================================================= /CONFIG =====


def stage1_compile_word_level_grammar() -> Path:
    """gf -make the concrete with probabilities; no canonicalization needed.

    Unlike the token-level pipeline, this grammar is used exactly as
    `WordBeamSearcher` finds it: terminals stay natural-language words, so
    there is nothing downstream that needs the grammar's paradigms evaluated
    down to plain literals.

    Returns the path to the compiled word-level .pgf.
    """
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    word_pgf = BUILD_DIR / WORD_LEVEL_PGF
    if word_pgf.exists():
        print(f"[stage 1] reusing cached {word_pgf}")
        return word_pgf

    cmd = [
        GF_BINARY, "-make",
        f"--probs={PROBS_FILE}",
        f"--gfo-dir={BUILD_DIR}",  # keep .gfo build litter out of the source tree
        "--path=" + ":".join(str(GRAMMAR_SRC_DIR / d) for d in GRAMMAR_SRC_SUBDIRS),
        str(CONCRETE_FILE),
    ]
    print(f"[stage 1] compiling {CONCRETE_FILE.name} (one-time, takes a few minutes)...")
    result = subprocess.run(cmd, cwd=BUILD_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gf -make failed ({shlex.join(cmd)}):\n{result.stdout}\n{result.stderr}")
    print(f"[stage 1] wrote {word_pgf}")
    return word_pgf


def stage5_generate(generator: ConstrainedGenerator, runtime: GFProbRuntime, prompt: str) -> None:
    """Run one word-synchronized search per configured lambda and print the results."""
    if DECODING_MODE not in ("beam", "sample"):
        raise ValueError(f"DECODING_MODE must be 'beam' or 'sample', got {DECODING_MODE!r}")

    print(f"\nmodel={MODEL_NAME}  prompt={prompt!r}  decoding_mode={DECODING_MODE!r}")
    display_beam_size = BEAM_SIZE if DECODING_MODE == "beam" else SAMPLE_NUM_CHAINS

    if INCLUDE_BASELINE_WITHOUT_CONSTRAINT:
        if DECODING_MODE == "beam":
            print(f"baseline (no grammar fusion, num_beams={display_beam_size}):")
            baseline_kwargs = dict(num_beams=display_beam_size)
        else:
            # match WordBeamSearcher's sampling mode below, so the baseline
            # is a fair comparison instead of silently falling back to beam
            # search while the constrained side samples
            print("baseline (no grammar fusion, do_sample=True):")
            baseline_kwargs = dict(
                do_sample=True,
                temperature=SAMPLE_TEMPERATURE,
                top_k=SAMPLE_TOP_K,
                top_p=SAMPLE_TOP_P,
                min_p=SAMPLE_MIN_P,
                typical_p=SAMPLE_TYPICAL_P,
                seed=SAMPLE_SEED,
            )
        result = generator.generate(
            prompt, constraint=None, max_new_tokens=BASELINE_MAX_NEW_TOKENS, **baseline_kwargs
        )
        print(f"  unconstrained  ->  {result.first_texts[0]!r}")

    searcher_kwargs = dict(
        gf_temperature=GF_TEMPERATURE,
        length_penalty=LENGTH_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        verbose=SHOW_TIMER,
    )
    if DECODING_MODE == "beam":
        print(f"word-level fusion (WordBeamSearcher, beam={BEAM_SIZE}):")
        searcher_kwargs.update(beam_size=BEAM_SIZE, diversity_penalty=DIVERSITY_PENALTY)
    else:
        print(f"word-level fusion (WordBeamSearcher, do_sample=True, chains={SAMPLE_NUM_CHAINS}):")
        searcher_kwargs.update(
            candidate_top_k=CANDIDATE_TOP_K,
            candidate_top_p=CANDIDATE_TOP_P,
            beam_size=SAMPLE_NUM_CHAINS,
            do_sample=True,
            temperature=SAMPLE_TEMPERATURE,
            top_k=SAMPLE_TOP_K,
            top_p=SAMPLE_TOP_P,
            min_p=SAMPLE_MIN_P,
            typical_p=SAMPLE_TYPICAL_P,
            seed=SAMPLE_SEED,
        )

    for lam in LAMBDAS:
        searcher = WordBeamSearcher(generator.model, generator.tokenizer, runtime, lambda_gf=lam, **searcher_kwargs)
        hyps = searcher.search(prompt, max_words=MAX_WORDS, num_return=NUM_RETURN)
        best = hyps[0]
        print(
            f"  lambda={lam:4.3f}  ->  {best.text!r}   "
            f"(gf={best.gf_logprob:6.2f}, lm={best.llm_logprob:7.2f})"
        )


def main():
    stage1_compile_word_level_grammar()
    # stage 2 (tokenize terminals) and stage 3 (compile token grammar) are
    # skipped — word-level fusion needs no token-ID grammar, see module docstring

    generator = ConstrainedGenerator(
        MODEL_NAME,
        half_precision=HALF_PRECISION,
        device_map=DEVICE_MAP,
    )
    runtime = GFProbRuntime(BUILD_DIR, default_pgf=WORD_LEVEL_PGF)

    prompt = input("Prompt: ").rstrip()
    if not prompt:
        sys.exit("empty prompt, nothing to do")

    stage5_generate(generator, runtime, prompt)


if __name__ == "__main__":
    main()
