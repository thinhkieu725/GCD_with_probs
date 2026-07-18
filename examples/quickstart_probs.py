"""Probability-fused grammar-constrained decoding, both granularities.

Builds the toy Food grammar with a probability file (`--probs`), then decodes
with GPT-2 three ways and prints a lambda sweep:

  1. binary GCD baseline           (GFConstraint, gf --server, mask only)
  2. token-level fusion            (GFProbConstraint + HF beam search)
  3. word-level fusion             (WordBeamSearcher, custom beam)

Requires the `gf` binary on PATH; downloads GPT-2 on first run:

    python examples/quickstart_probs.py [model_name] [lambda ...]
"""

import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from gf_gcd import ConstrainedGenerator, GFProbConstraint, GFProbRuntime, WordBeamSearcher
from gf_gcd.grammar_tools import compile_grammar, tokenize_concrete_grammar

PROJECT_ROOT = Path(__file__).parent.parent
TOY = PROJECT_ROOT / "grammars" / "toy"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
LAMBDAS = [float(x) for x in sys.argv[2:]] or [0.0, 0.3, 1.0]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    generator = ConstrainedGenerator(MODEL)
    prompt = "Here is a comment about food:"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # word-level grammar with probabilities
        (tmp / "Food.gf").write_text((TOY / "Food.gf").read_text())
        (tmp / "FoodEng.gf").write_text((TOY / "FoodEng.gf").read_text())
        compile_grammar(tmp / "FoodEng.gf", tmp, probs_file=TOY / "Food.probs")

        # token-ID grammar: terminals rewritten to "t<id>" (letter prefix is
        # required — digit-only terminals break the C runtime's complete())
        token_level = tokenize_concrete_grammar(
            tokenizer, (TOY / "FoodEng.gf").read_text(), prefix_space=True, id_prefix="t"
        ).replace("concrete FoodEng of", "concrete FoodTok of")
        (tmp / "FoodTok.gf").write_text(token_level)
        compile_grammar(
            tmp / "FoodTok.gf", tmp, output_name="FoodTok", probs_file=TOY / "Food.probs"
        )

        print(f"model={MODEL}  prompt={prompt!r}\n")

        # ---- Phase 1: token-level fusion via HF beam search
        print("token-level fusion (GFProbConstraint + num_beams=4):")
        for lam in LAMBDAS:
            constraint = GFProbConstraint(
                grammar_dir=str(tmp),
                default_grammar="FoodTok.pgf",
                eos_token_id=generator.tokenizer.eos_token_id,
                pad_token_id=generator.tokenizer.pad_token_id,
                lambda_gf=lam,
                id_prefix="t",
            )
            result = generator.generate(
                prompt, constraint=constraint, max_new_tokens=16, num_beams=4
            )
            print(f"  lambda={lam:4.1f}  ->  {result.first_texts[0]!r}")

        # ---- Phase 2: word-level fusion via custom beam
        print("\nword-level fusion (WordBeamSearcher, beam=4):")
        runtime = GFProbRuntime(tmp, default_pgf="Food.pgf")
        for lam in LAMBDAS:
            searcher = WordBeamSearcher(
                generator.model, generator.tokenizer, runtime, lambda_gf=lam, beam_size=4
            )
            hyps = searcher.search(prompt, max_words=10, num_return=2)
            best = hyps[0]
            print(
                f"  lambda={lam:4.1f}  ->  {best.text!r}   "
                f"(gf={best.gf_logprob:6.2f}, lm={best.llm_logprob:7.2f})"
            )


if __name__ == "__main__":
    main()
