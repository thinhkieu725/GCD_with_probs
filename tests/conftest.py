import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
TOY_GRAMMAR_SRC = PROJECT_ROOT / "grammars" / "toy"

requires_gf = pytest.mark.skipif(shutil.which("gf") is None, reason="gf binary not on PATH")


@pytest.fixture(scope="session")
def compiled_grammar_dir(tmp_path_factory) -> Path:
    """Compile the toy grammars into a session-scoped directory of .pgf files."""
    from gf_gcd.grammar_tools import compile_grammar

    out = tmp_path_factory.mktemp("pgf")
    compile_grammar(TOY_GRAMMAR_SRC / "FoodEng.gf", out)
    compile_grammar(TOY_GRAMMAR_SRC / "SeqIds.gf", out)
    assert (out / "Food.pgf").exists()
    assert (out / "Seq.pgf").exists()
    return out


@pytest.fixture(scope="session")
def prob_grammar_dir(tmp_path_factory) -> Path:
    """Toy grammars compiled with probability files (for GFProbRuntime & co).

    SeqTok uses letter-prefixed terminals ("t100"): digit-only terminals break
    the C runtime's complete().
    """
    from gf_gcd.grammar_tools import compile_grammar

    out = tmp_path_factory.mktemp("pgf_probs")
    compile_grammar(TOY_GRAMMAR_SRC / "FoodEng.gf", out, probs_file=TOY_GRAMMAR_SRC / "Food.probs")
    compile_grammar(
        TOY_GRAMMAR_SRC / "SeqTok.gf",
        out,
        output_name="SeqTok",
        probs_file=TOY_GRAMMAR_SRC / "Seq.probs",
    )
    assert (out / "Food.pgf").exists()
    assert (out / "SeqTok.pgf").exists()
    return out
