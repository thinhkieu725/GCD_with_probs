import math
import re

import pytest

from gf_gcd.grammar_tools import (
    GFSyntaxError,
    chainify_and_compile,
    chainify_files,
    chainify_grammar,
    compile_grammar,
)

from .conftest import TOY_GRAMMAR_SRC, requires_gf

# Two string literals joined directly by `++` with nothing (no argument)
# between them: the shape that trips the GF C runtime's mid-literal
# `pgf_complete` bug. A correctly chainified grammar must have none of these
# left in any function it claims to have transformed.
_ADJACENT_LITERALS = re.compile(r'"(?:[^"\\]|\\.)*"\s*\+\+\s*"(?:[^"\\]|\\.)*"')


# ------------------------------------------------------------------ unit tests
# Pure source-to-source transform; no `gf` binary required.


def test_already_single_token_grammar_is_left_untouched():
    abstract_src = (TOY_GRAMMAR_SRC / "Food.gf").read_text()
    concrete_src = (TOY_GRAMMAR_SRC / "FoodEng.gf").read_text()  # plain words, one token each

    abs_out, conc_out, report = chainify_grammar(abstract_src, concrete_src)

    assert report.transformed == []
    assert report.skipped == {}
    assert report.new_categories == 0
    assert report.new_functions == 0


def test_chains_a_zero_arg_multi_token_function():
    abstract_src = (TOY_GRAMMAR_SRC / "Seq.gf").read_text()
    concrete_src = (TOY_GRAMMAR_SRC / "SeqTok.gf").read_text()

    abs_out, conc_out, report = chainify_grammar(abstract_src, concrete_src)

    assert set(report.transformed) == {"Short", "Long"}
    assert _ADJACENT_LITERALS.search(conc_out) is None, (
        "chainified output still has two literals joined with no argument in "
        "between -- the mid-literal completion bug would still apply"
    )
    # every rewritten function keeps its original name and gains exactly the
    # extra argument(s) needed to carry the rest of its literal run
    assert "Short " in abs_out and "Short :" in abs_out
    assert "Long " in abs_out and "Long :" in abs_out


def test_chains_a_literal_run_between_two_arguments():
    # `Pred item quality = item ++ "is1" ++ "is2" ++ quality`: the run sits
    # strictly inside the RHS, sandwiched by two parameter references -- the
    # case a pure "split leading literal" approach would miss.
    abstract_src = """
    abstract A = {
      cat S ; N ;
      fun Pred : N -> N -> S ; Dog : N ; Cat : N ;
    }
    """
    concrete_src = """
    concrete AEng of A = {
      lincat S, N = Str ;
      lin
        Pred item quality = item ++ "is1" ++ "is2" ++ quality ;
        Dog = "dog" ;
        Cat = "cat" ;
    }
    """
    abs_out, conc_out, report = chainify_grammar(abstract_src, concrete_src)

    assert report.transformed == ["Pred"]
    assert _ADJACENT_LITERALS.search(conc_out) is None
    # Pred keeps taking `item` then `quality` before the new tail argument;
    # its own contribution shrinks to exactly one literal token
    pred_line = next(l for l in conc_out.splitlines() if l.strip().startswith("Pred "))
    assert pred_line.count('"is1"') == 1
    assert '"is2"' not in pred_line  # pushed into a chained function instead


def test_unsupported_construct_is_skipped_not_fatal():
    # `table` is out of scope (see docs/chainify_limitations.md); the
    # transform must still complete for everything else in the file.
    abstract_src = """
    abstract A = { cat S ; N ; fun Good : S ; Weird : N -> S ; Dog : N ; }
    """
    concrete_src = """
    concrete AEng of A = {
      lincat S, N = Str ;
      lin
        Good = "t1" ++ "t2" ++ "t3" ;
        Weird n = table { _ => "t9" } ! n ;
        Dog = "t7" ++ "t8" ;
    }
    """
    abs_out, conc_out, report = chainify_grammar(abstract_src, concrete_src)

    assert set(report.transformed) == {"Good", "Dog"}
    assert "Weird" in report.skipped
    # the untouched declaration is preserved byte-for-byte
    assert "table { _ => \"t9\" } ! n" in conc_out


def test_mismatched_abstract_concrete_raises():
    abstract_src = "abstract A = { cat S ; fun F : S ; }"
    concrete_src = 'concrete BEng of B = { lincat S = Str ; lin F = "x" ; }'
    with pytest.raises(GFSyntaxError):
        chainify_grammar(abstract_src, concrete_src)


def test_module_extension_is_rejected_not_silently_wrong():
    abstract_src = "abstract A = B ** { fun F : S ; }"
    concrete_src = 'concrete AEng of A = { lincat S = Str ; lin F = "x" ; }'
    with pytest.raises(GFSyntaxError):
        chainify_grammar(abstract_src, concrete_src)


def test_identical_tails_are_shared_across_functions():
    # Two words whose token runs share an exact trailing subsequence should
    # reuse the same synthetic category instead of duplicating it.
    abstract_src = "abstract A = { cat S ; fun One : S ; Two : S ; }"
    concrete_src = """
    concrete AEng of A = {
      lincat S = Str ;
      lin
        One = "a" ++ "shared" ++ "tail" ;
        Two = "b" ++ "shared" ++ "tail" ;
    }
    """
    _, _, report = chainify_grammar(abstract_src, concrete_src)
    # 2 tokens of tail per word, but the (shared, tail) suffix is reused ->
    # 2 categories total, not 4
    assert report.new_categories == 2
    assert report.new_functions == 2


def test_fresh_names_avoid_existing_identifiers():
    # "Tail_t2_1" is deliberately pre-declared: it is exactly the name
    # `_TailChains._fresh` would try first for F's run ("t1" ++ "t2"), since
    # its counter starts at 1 and its hint is derived from the second token.
    # If the generator picked it anyway, the output would redeclare the same
    # category name twice -- invalid GF (and it would silently alias F's
    # synthetic tail onto G's unrelated pre-existing category).
    abstract_src = "abstract A = { cat S ; Tail_t2_1 ; fun F : S ; G : Tail_t2_1 -> S ; }"
    concrete_src = """
    concrete AEng of A = {
      lincat S, Tail_t2_1 = Str ;
      lin
        F = "t1" ++ "t2" ;
        G x = x ;
    }
    """
    abs_out, _, report = chainify_grammar(abstract_src, concrete_src)

    assert report.transformed == ["F"]
    # "Tail_t2_1" must occur exactly where it originally did (the `cat`
    # declaration and G's signature) and nowhere new
    assert len(re.findall(r"\bTail_t2_1\b", abs_out)) == 2


def test_chainify_files_writes_renamed_pair(tmp_path):
    abs_path, conc_path, report = chainify_files(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path
    )
    assert abs_path.name == "SeqChain.gf"
    assert conc_path.name == "SeqTokChain.gf"
    assert abs_path.exists() and conc_path.exists()
    assert set(report.transformed) == {"Short", "Long"}


# ------------------------------------------------------------- compile-level
# Round-trips the rewritten grammar through the real `gf` compiler and (where
# available) the C-runtime `pgf` bindings, to confirm the rewrite is not just
# syntactically plausible but actually fixes the bug in `gf_probs.py`.


@requires_gf
def test_chainified_grammar_compiles(tmp_path):
    abs_path, conc_path, _ = chainify_files(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path
    )
    pgf_path = compile_grammar([abs_path, conc_path], tmp_path, output_name="SeqChain")
    assert pgf_path.exists()


@requires_gf
def test_chainified_grammar_fixes_mid_literal_completion(tmp_path):
    """Regression test for the bug `chainify_grammar` exists to work around.

    On the original SeqTok grammar, `next_token_logprobs(["t100"])` returns
    `{}` (a false dead end -- `complete("t100 ")` raises `pgf.ParseError`
    because the parse position is inside Short/Long's own literal run). The
    chainified grammar restructures both functions so that position falls on
    a function boundary, and the query must return real candidates.
    """
    pytest.importorskip("pgf")

    abs_path, conc_path, _ = chainify_files(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path
    )
    compile_grammar(
        [abs_path, conc_path], tmp_path, output_name="SeqChain",
        probs_file=TOY_GRAMMAR_SRC / "Seq.probs",
    )

    from gf_gcd.gf_probs import GFProbRuntime

    runtime = GFProbRuntime(tmp_path, default_pgf="SeqChain.pgf", concrete="SeqTokChain")
    dist = runtime.next_token_logprobs(["t100"])

    assert dist, "chainified grammar still hits the mid-literal dead end"
    probs = {tok: math.exp(lp) for tok, lp in dist.items()}
    # Seq.probs: Short 0.9, Long 0.1 -- unaffected by the rewrite, since the
    # new tail categories always have exactly one producer (probability 1)
    assert probs == pytest.approx({"t200": 0.9, "t300": 0.1}, abs=1e-4)


@requires_gf
def test_chainify_and_compile_produces_only_the_pgf(tmp_path):
    pgf_path, report = chainify_and_compile(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path
    )
    assert pgf_path.exists()
    assert pgf_path.parent == tmp_path
    assert set(report.transformed) == {"Short", "Long"}
    # no intermediate .gf/.gfo left behind -- just the compiled grammar
    assert [p.name for p in tmp_path.iterdir()] == [pgf_path.name]


@requires_gf
def test_chainify_and_compile_keep_gf_files(tmp_path):
    pgf_path, _ = chainify_and_compile(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path,
        keep_gf_files=True,
    )
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {pgf_path.name, "SeqChain.gf", "SeqTokChain.gf"}


@requires_gf
def test_chainify_and_compile_never_touches_the_originals(tmp_path):
    abstract_path = TOY_GRAMMAR_SRC / "Seq.gf"
    concrete_path = TOY_GRAMMAR_SRC / "SeqTok.gf"
    before = (abstract_path.read_bytes(), concrete_path.read_bytes())

    chainify_and_compile(abstract_path, concrete_path, tmp_path)

    after = (abstract_path.read_bytes(), concrete_path.read_bytes())
    assert before == after


@requires_gf
def test_chainify_and_compile_refuses_to_overwrite_originals(tmp_path):
    abstract_path = TOY_GRAMMAR_SRC / "Seq.gf"
    concrete_path = TOY_GRAMMAR_SRC / "SeqTok.gf"
    before = (abstract_path.read_bytes(), concrete_path.read_bytes())

    with pytest.raises(ValueError):
        chainify_and_compile(
            abstract_path,
            concrete_path,
            TOY_GRAMMAR_SRC,
            new_abstract_name="Seq",
            new_concrete_name="SeqTok",
            keep_gf_files=True,
        )

    # the guard must fire before anything is written back to the source dir
    after = (abstract_path.read_bytes(), concrete_path.read_bytes())
    assert before == after


@requires_gf
def test_chainify_and_compile_fixes_mid_literal_completion(tmp_path):
    """Same regression as the `chainify_files` version above, but through the
    one-call `chainify_and_compile` entry point most callers should use."""
    pytest.importorskip("pgf")

    pgf_path, _ = chainify_and_compile(
        TOY_GRAMMAR_SRC / "Seq.gf", TOY_GRAMMAR_SRC / "SeqTok.gf", tmp_path,
        probs_file=TOY_GRAMMAR_SRC / "Seq.probs",
    )

    from gf_gcd.gf_probs import GFProbRuntime

    runtime = GFProbRuntime(tmp_path, default_pgf=pgf_path.name, concrete="SeqTokChain")
    dist = runtime.next_token_logprobs(["t100"])

    assert dist, "chainified grammar still hits the mid-literal dead end"
    probs = {tok: math.exp(lp) for tok, lp in dist.items()}
    assert probs == pytest.approx({"t200": 0.9, "t300": 0.1}, abs=1e-4)
