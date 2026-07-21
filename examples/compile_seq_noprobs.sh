#!/usr/bin/env bash
# Compile SeqTok.gf twice: without any probability file, and with Seq.probs.
# Outputs land in examples/build_seq/{noprobs,withprobs}/Seq.pgf so the
# mid-literal complete() bug can be tested on both (test_complete_noprobs.py).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
toy="$root/grammars/toy"
build="$root/examples/build_seq"

rm -rf "$build"
mkdir -p "$build/noprobs" "$build/withprobs"

echo "== compiling WITHOUT probs =="
gf --make --output-dir="$build/noprobs" "$toy/SeqTok.gf"

echo "== compiling WITH Seq.probs =="
gf --make --output-dir="$build/withprobs" --probs="$toy/Seq.probs" "$toy/SeqTok.gf"

echo
echo "done:"
ls -l "$build/noprobs/Seq.pgf" "$build/withprobs/Seq.pgf"
