from gf_gcd.runtime import GFServerPoolRuntime, GFServerRuntimeForLM

from .conftest import requires_gf


@requires_gf
def test_complete_word_level_grammar(compiled_grammar_dir):
    with GFServerPoolRuntime(default_pgf="Food.pgf", grammar_dir=str(compiled_grammar_dir)) as rt:
        first = rt.complete([])
        assert set(first) == {"this", "that"}

        after_this = rt.complete(["this"])
        # a Kind or a Quality modifier can follow
        assert "wine" in after_this and "fresh" in after_this

        finished = rt.complete(["this", "wine", "is", "delicious"])
        assert finished == []


@requires_gf
def test_get_prefix_allowed_tokens_for_LM(compiled_grammar_dir):
    with GFServerRuntimeForLM(default_pgf="Seq.pgf", grammar_dir=str(compiled_grammar_dir)) as rt:
        assert rt.get_prefix_allowed_tokens_for_LM([]) == [100]
        assert sorted(rt.get_prefix_allowed_tokens_for_LM([100])) == [200, 300]
        assert rt.get_prefix_allowed_tokens_for_LM([100, 300]) == [400]
        assert rt.get_prefix_allowed_tokens_for_LM([100, 200]) == []


@requires_gf
def test_round_robin_pool(compiled_grammar_dir):
    with GFServerPoolRuntime(
        default_pgf="Seq.pgf", grammar_dir=str(compiled_grammar_dir), pool_size=2
    ) as rt:
        assert len(rt.urls) == 2
        # both servers answer consistently
        for _ in range(4):
            assert rt.complete([]) == ["100"]
