"""Probability-aware GF runtime built on the GF C runtime (the `pgf` module).

Where `GFServerPoolRuntime` answers "which tokens are allowed next" (a set),
this runtime answers "how likely is each allowed token" — the grammar must be
compiled with `gf -make --probs=...` for the probabilities to be informative
(without a probs file every function gets a uniform share, which degrades to
the plain allowed-set behaviour).

Semantics recovered from the C runtime:
- `Concr.complete(prefix)` yields one `(cost, token, cat, fun)` tuple per
  *derivation*, where cost is the cumulative -ln p of the best parse of
  `prefix + token` (Viterbi prefix cost, on the same scale across candidates
  of one step). Duplicate tokens (same surface, different derivations) must
  be merged by summing their probabilities (logsumexp on -cost).
- `Concr.parse(sentence)` iterates trees best-first, one cost per distinct
  derivation of the whole sentence -- the parse-tree analog of `complete`'s
  per-derivation costs. Just like `complete`, ambiguity is resolved by
  merging (logsumexp on -cost) rather than keeping only the best tree, so
  P(stop here) = Σ_trees P(tree), on the same cumulative scale as the
  continuation costs; normalizing over {stop} ∪ {continuations} yields a
  proper conditional including P(end of sentence | prefix). Only the
  `max_eos_trees` best trees are pulled (parse ambiguity can otherwise be
  combinatorial); costs decay quickly in a best-first iterator, so the
  truncated tail contributes negligible probability mass in practice.
"""

import math
import threading
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

try:
    import pgf
except ImportError:  # binary masking via gf --server works without it
    pgf = None

#: Dictionary key used for "the derivation can stop here" in the returned
#: next-token distributions. GF tokens are grammar terminals (words or token
#: IDs rendered as strings), so this sentinel cannot collide with them in
#: practice.
GF_EOS = "<eos>"


def _logsumexp(values) -> float:
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


class GFProbRuntime:
    """In-process runtime over compiled .pgf grammars with probability queries.

    Grammars are loaded lazily and cached per filename. Thread-safe for the
    loading step; the C runtime objects themselves are used from one thread
    at a time per call (HF's generate loop is single-threaded).
    """

    def __init__(
        self,
        grammar_dir: Union[str, Path],
        default_pgf: Optional[str] = None,
        concrete: Optional[str] = None,
        cache_size: int = 4096,
        max_eos_trees: int = 10,
    ):
        """
        Args:
            grammar_dir: directory containing compiled .pgf files.
            default_pgf: .pgf filename used when a query names no grammar.
            concrete: concrete syntax name to query; default: the (single)
                concrete in the grammar, error if it is ambiguous.
            cache_size: number of (grammar, prefix) -> distribution entries
                kept; beams revisit the same prefixes constantly.
            max_eos_trees: max number of best-first parse trees merged into
                the "stop here" probability (see class docstring); best-first
                order means the trees left out are the least likely ones.
        """
        if pgf is None:
            raise ImportError(
                "GFProbRuntime needs the GF C runtime binding: pip install pgf"
            )
        self.grammar_dir = Path(grammar_dir)
        if not self.grammar_dir.is_dir():
            raise FileNotFoundError(f"grammar_dir does not exist: {self.grammar_dir}")
        self.default_pgf = default_pgf
        self.concrete_name = concrete
        self._grammars: Dict[str, "pgf.Concr"] = {}
        self._lock = threading.Lock()
        self._cache: Dict[tuple, Dict[str, float]] = {}
        self._cache_size = cache_size
        self._max_eos_trees = max_eos_trees

    # ------------------------------------------------------------- grammars

    def _concr(self, pgf_name: Optional[str]) -> "pgf.Concr":
        name = pgf_name or self.default_pgf
        if name is None:
            raise RuntimeError("no grammar given and no default_pgf configured")
        with self._lock:
            if name not in self._grammars:
                path = self.grammar_dir / name
                if not path.exists():
                    raise FileNotFoundError(f"grammar not found: {path}")
                grammar = pgf.readPGF(str(path))
                if self.concrete_name is not None:
                    concr = grammar.languages[self.concrete_name]
                else:
                    if len(grammar.languages) != 1:
                        raise RuntimeError(
                            f"{name} has {len(grammar.languages)} concrete syntaxes "
                            f"({sorted(grammar.languages)}); pass `concrete=` to pick one"
                        )
                    concr = next(iter(grammar.languages.values()))
                self._grammars[name] = concr
            return self._grammars[name]

    # -------------------------------------------------------------- queries

    def next_token_logprobs(
        self,
        prefix_tokens: Sequence[str],
        pgf_name: Optional[str] = None,
        temperature: float = 1.0,
        include_eos: bool = True,
    ) -> Dict[str, float]:
        """Conditional distribution over the grammar's next tokens.

        Returns {token: ln P(token | prefix)}, normalized over the candidate
        set of this step (plus GF_EOS when the prefix already parses as a
        complete derivation and `include_eos` is set). Empty dict means the
        prefix is a dead end (cannot happen for prefixes generated under the
        same grammar).

        `temperature` rescales the grammar scores before normalization:
        T -> 0 approaches argmax, T -> inf approaches uniform-over-allowed
        (i.e. plain grammar masking).
        """
        key = (pgf_name or self.default_pgf, tuple(prefix_tokens), temperature, include_eos)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        concr = self._concr(pgf_name)
        prefix = "".join(t + " " for t in prefix_tokens)

        # -cost of each continuation token, merged over derivations
        scores: Dict[str, list] = {}
        try:
            completions = concr.complete(prefix)
        except pgf.ParseError:
            completions = []
        for cost, token, _cat, _fun in completions:
            if not isinstance(token, str):
                # pgf.BIND / pgf.SOFT_BIND: "glue the next literal to the
                # previous one without a space" (contractions, compounds).
                # Grammars whose terminals are all plain string literals
                # (the toy grammars, or a tokenize_concrete_grammar output)
                # never produce this; grammars compiled straight from
                # source (e.g. the RGL) can. Neither token-level nor
                # word-level candidates are meaningful sub-word glue
                # instructions, so drop the candidate rather than crash.
                continue
            scores.setdefault(token, []).append(-cost)

        merged = {token: _logsumexp(vals) for token, vals in scores.items()}

        if include_eos and prefix_tokens:
            try:
                tree_costs = [
                    -cost
                    for cost, _tree in concr.parse(
                        prefix.strip(), n=self._max_eos_trees
                    )
                ]
            except pgf.ParseError:
                tree_costs = []
            if tree_costs:
                merged[GF_EOS] = _logsumexp(tree_costs)

        if not merged:
            result: Dict[str, float] = {}
        else:
            scaled = {t: s / temperature for t, s in merged.items()}
            log_z = _logsumexp(list(scaled.values()))
            result = {t: s - log_z for t, s in scaled.items()}

        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = result
        return result

    def parses(self, tokens: Sequence[str], pgf_name: Optional[str] = None) -> bool:
        """Whether the token sequence is a complete derivation."""
        concr = self._concr(pgf_name)
        try:
            next(concr.parse(" ".join(tokens)))
            return True
        except (pgf.ParseError, StopIteration):
            return False
