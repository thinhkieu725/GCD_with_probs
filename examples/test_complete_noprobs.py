"""Does the mid-literal complete() bug depend on probability annotations?

Runs the same three probes as repro_midliteral.py against two builds of the
identical SeqTok grammar (see compile_seq_noprobs.sh):

  - build_seq/noprobs/Seq.pgf    compiled with no --probs flag
  - build_seq/withprobs/Seq.pgf  compiled with --probs=Seq.probs (Short 0.9, Long 0.1)

If complete("t100 ") raises ParseError in both, the bug is in the completion
machinery itself and unrelated to probabilities.
"""

import pathlib
import pgf

BUILD = pathlib.Path(__file__).parent / "build_seq"


def probe(label: str, pgf_path: pathlib.Path) -> None:
    print(f"===== {label}: {pgf_path} =====")
    concr = pgf.readPGF(str(pgf_path)).languages["SeqTok"]

    print('complete("")')
    for cost, token, cat, fun in concr.complete(""):
        print(f"  ok: token={token!r} cost={cost:.4f} fun={fun}")

    print('complete("t100 ")')
    try:
        for cost, token, cat, fun in concr.complete("t100 "):
            print(f"  ok: token={token!r} cost={cost:.4f} fun={fun}")
    except pgf.ParseError as e:
        print(f"  pgf.ParseError: {e}")

    for sentence in ("t100 t200", "t100 t300 t400"):
        cost, tree = next(concr.parse(sentence))
        print(f'parse("{sentence}") -> {tree} (cost {cost:.4f})')
    print()


probe("NO PROBS", BUILD / "noprobs" / "Seq.pgf")
probe("WITH PROBS", BUILD / "withprobs" / "Seq.pgf")
