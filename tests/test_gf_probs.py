import math

import pytest

from gf_gcd.gf_probs import GF_EOS, GFProbRuntime

from .conftest import requires_gf


def probs(dist):
    return {t: math.exp(lp) for t, lp in dist.items()}


@pytest.fixture(scope="module")
def seq_runtime(prob_grammar_dir):
    return GFProbRuntime(prob_grammar_dir, default_pgf="SeqTok.pgf")


@pytest.fixture(scope="module")
def food_runtime(prob_grammar_dir):
    return GFProbRuntime(prob_grammar_dir, default_pgf="Food.pgf")


@requires_gf
class TestSeqDistributions:
    def test_start_distribution(self, seq_runtime):
        dist = probs(seq_runtime.next_token_logprobs([]))
        assert dist == pytest.approx({"t100": 1.0})

    def test_branch_probabilities_follow_probs_file(self, seq_runtime):
        # Seq.probs: Short 0.9, Long 0.1
        dist = probs(seq_runtime.next_token_logprobs(["t100"]))
        assert dist == pytest.approx({"t200": 0.9, "t300": 0.1}, abs=1e-4)

    def test_eos_when_derivation_complete(self, seq_runtime):
        dist = probs(seq_runtime.next_token_logprobs(["t100", "t200"]))
        assert dist == pytest.approx({GF_EOS: 1.0})

    def test_distribution_sums_to_one(self, seq_runtime):
        dist = probs(seq_runtime.next_token_logprobs(["t100"]))
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_temperature_flattens(self, seq_runtime):
        hot = probs(seq_runtime.next_token_logprobs(["t100"], temperature=100.0))
        assert hot["t200"] == pytest.approx(hot["t300"], abs=0.02)

    def test_parses(self, seq_runtime):
        assert seq_runtime.parses(["t100", "t200"])
        assert seq_runtime.parses(["t100", "t300", "t400"])
        assert not seq_runtime.parses(["t100"])


@requires_gf
class TestFoodDistributions:
    def test_probability_ranking(self, food_runtime):
        # Food.probs: Wine 0.5 > Cheese 0.3 > Fish 0.1
        dist = probs(food_runtime.next_token_logprobs(["this"]))
        assert dist["wine"] > dist["cheese"] > dist["fish"]
        assert sum(dist.values()) == pytest.approx(1.0)

    def test_zero_prob_function_is_excluded(self, food_runtime):
        # Boring has probability 0.0 -> infinite cost -> never proposed
        dist = food_runtime.next_token_logprobs(["this", "wine", "is"])
        assert "boring" not in dist
        assert "delicious" in dist

    def test_eos_competes_with_continuations(self, food_runtime):
        # "this wine is delicious" is complete but could continue ("very ...")
        # -> no continuation exists in this grammar after a final Quality,
        # so eos should dominate; a mid-sentence prefix has no eos at all.
        assert GF_EOS in food_runtime.next_token_logprobs(["this", "wine", "is", "delicious"])
        assert GF_EOS not in food_runtime.next_token_logprobs(["this", "wine"])

    def test_unknown_prefix_is_dead_end(self, food_runtime):
        assert food_runtime.next_token_logprobs(["borogoves"]) == {}


@requires_gf
def test_missing_grammar_raises(prob_grammar_dir):
    rt = GFProbRuntime(prob_grammar_dir, default_pgf="nope.pgf")
    with pytest.raises(FileNotFoundError):
        rt.next_token_logprobs([])
