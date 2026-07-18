"""End-to-end example: build a token-ID grammar for a real tokenizer and decode with it.

Requires the `gf` binary on PATH and downloads a small HF model on first run:

    python examples/quickstart.py [model_name]
"""

import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from gf_gcd import ConstrainedGenerator, GFConstraint
from gf_gcd.grammar_tools import compile_grammar, tokenize_concrete_grammar

PROJECT_ROOT = Path(__file__).parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # 1. Take the word-level toy grammar and rewrite its terminals into the
    #    token IDs of the target tokenizer.
    word_level = (PROJECT_ROOT / "grammars" / "toy" / "FoodEng.gf").read_text()
    token_level = tokenize_concrete_grammar(tokenizer, word_level, prefix_space=True)
    token_level = token_level.replace("concrete FoodEng of", "concrete FoodTok of")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "Food.gf").write_text((PROJECT_ROOT / "grammars" / "toy" / "Food.gf").read_text())
        (tmp / "FoodTok.gf").write_text(token_level)

        # 2. Compile to .pgf.
        pgf_path = compile_grammar(tmp / "FoodTok.gf", tmp)
        print(f"Compiled grammar: {pgf_path.name}")

        # 3. Generate with and without the constraint.
        generator = ConstrainedGenerator(MODEL)
        prompt = "Here is a comment about food:"

        with GFConstraint(
            grammar_dir=str(tmp),
            default_grammar=pgf_path.name,
            eos_token_id=generator.tokenizer.eos_token_id,
        ) as constraint:
            constrained = generator.generate(
                prompt,
                constraint=constraint,
                max_new_tokens=12,
                num_beams=2,
                return_token_logprobs=True,
            )

        unconstrained = generator.generate(prompt, max_new_tokens=12)

        print(f"\nPrompt:        {prompt!r}")
        print(f"Unconstrained: {unconstrained.first_texts[0]!r}")
        print(f"Constrained:   {constrained.first_texts[0]!r}")
        print(f"Token logprobs: {constrained.token_logprobs[0].tolist()}")


if __name__ == "__main__":
    main()
