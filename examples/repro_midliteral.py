"""Reproduce the GF C-runtime mid-literal completion bug.

The grammar (SeqTok, compiled into Seq.pgf) linearizes:
    Short = "t100" ++ "t200"
    Long  = "t100" ++ "t300" ++ "t400"

Expected: complete("t100 ") offers {t200, t300}.
Actual:   pgf.ParseError, even though "t100 t200" parses fine.

Usage: python repro_midliteral.py [path/to/Seq.pgf]
"""

import sys
import pgf

PGF_PATH = sys.argv[1] if len(sys.argv) > 1 else "grammars/Seq.pgf"

grammar = pgf.readPGF(PGF_PATH)
concr = grammar.languages["SeqTok"]


def show_complete(prefix: str) -> None:
    print(f'complete({prefix!r})')
    try:
        for cost, token, cat, fun in concr.complete(prefix):
            print(f"  ok: token={token!r} cost={cost} cat={cat} fun={fun}")
    except pgf.ParseError as e:
        print(f"  pgf.ParseError: {e}")
    print()


# 1. Start of derivation: works, offers the first token of both chains.
show_complete("")

# 2. Mid-literal: the parse position is inside "t100" ++ ..., after t100.
#    Should offer t200 (Short) and t300 (Long); instead raises ParseError.
show_complete("t100 ")

# 3. Sanity check: full sentences parse fine, so the grammar itself is sound
#    and t100/t200 are real tokens -- only the completion path is broken.
for sentence in ("t100 t200", "t100 t300 t400"):
    prob, tree = next(concr.parse(sentence))
    print(f"parse({sentence!r}) -> {tree}")

# 4. What an unhandled mid-literal completion looks like in the terminal:
print("\n--- now unhandled, to show the raw traceback ---\n")
list(concr.complete("t100 "))
