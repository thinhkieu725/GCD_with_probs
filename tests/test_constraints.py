import torch

from gf_gcd.constraints import GFConstraint

from .conftest import requires_gf


@requires_gf
def test_prefix_allowed_tokens_fn_strips_prompt(compiled_grammar_dir):
    with GFConstraint(grammar_dir=str(compiled_grammar_dir), default_grammar="Seq.pgf") as constraint:
        # prompt is 3 tokens; only tokens after it are sent to the grammar
        fn = constraint.prefix_allowed_tokens_fn(prompt_lengths=[3])

        prompt_only = torch.tensor([11, 22, 33])
        assert fn(0, prompt_only) == [100]

        mid_generation = torch.tensor([11, 22, 33, 100])
        assert sorted(fn(0, mid_generation)) == [200, 300]


@requires_gf
def test_eos_fallback_when_grammar_finished(compiled_grammar_dir):
    with GFConstraint(
        grammar_dir=str(compiled_grammar_dir), default_grammar="Seq.pgf", eos_token_id=2
    ) as constraint:
        fn = constraint.prefix_allowed_tokens_fn(prompt_lengths=[0])
        finished = torch.tensor([100, 200])
        # the sentence is complete with no legal continuation left: offer
        # both ending generation (eos_token_id) and starting a fresh
        # sentence (the grammar's own start-of-sentence tokens)
        assert sorted(fn(0, finished)) == [2, 100]


@requires_gf
def test_chains_a_second_sentence_after_the_first_completes(compiled_grammar_dir):
    with GFConstraint(
        grammar_dir=str(compiled_grammar_dir), default_grammar="Seq.pgf", eos_token_id=2
    ) as constraint:
        fn = constraint.prefix_allowed_tokens_fn(prompt_lengths=[0])
        # "100 200" (a complete first sentence) followed by "100": the mask
        # must be computed against the tail of the *second* sentence alone
        # (`[100]`), not the whole five-token history, which the grammar
        # cannot parse as one derivation
        second_sentence_started = torch.tensor([100, 200, 100])
        assert sorted(fn(0, second_sentence_started)) == [200, 300]


@requires_gf
def test_per_sample_grammar_selection(compiled_grammar_dir):
    with GFConstraint(grammar_dir=str(compiled_grammar_dir), default_grammar="Seq.pgf") as constraint:
        fn = constraint.prefix_allowed_tokens_fn(
            prompt_lengths=[0, 0], grammars=["Seq.pgf", "missing.pgf"]
        )
        # missing grammar falls back to the default
        assert fn(0, torch.tensor([100, 300])) == [400]
        assert fn(1, torch.tensor([100, 300])) == [400]
