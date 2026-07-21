"""Restructure a token-ID GF grammar so every function contributes one token.

Fix for the GF C runtime's mid-literal `pgf_complete` bug (see
`docs/chainify_limitations.md` and `examples/repro_midliteral.py`): the
runtime raises `ParseError` when the parse position lands strictly inside a
function's own literal run, e.g. after `"t1416"` of
`Delicious = "t1416" ++ "t563"`. Completion positions that fall on a
function-application boundary are unaffected, so a grammar where every
function contributes at most one literal token never hits the bug.

`chainify_grammar` performs that restructuring as a source-to-source
transform on an abstract/concrete .gf pair (typically the output of
`tokenize_concrete_grammar`): for a function whose linearization contains a
run of two or more consecutive literal tokens with no argument in between,
it moves tokens 2..k of that run into a chain of fresh single-token
categories/functions, e.g.

    fun  Delicious : Quality ;
    lin  Delicious = "t1416" ++ "t563" ;

becomes

    fun  Delicious : Tail_t563_1 -> Quality ;
    fun  Tail_t563_1 : Tail_t563_1 ;
    lin  Delicious xtail1 = "t1416" ++ xtail1 ;
    lin  Tail_t563_1 = "t563" ;

`Tail_t563_1`'s only producer is itself, so parsing/completing it is a
forced, probability-1 step -- every completion position in the rewritten
grammar sits on a function boundary. Identical tail token sequences are
shared across functions (e.g. two words ending in the same subword) to
limit how much the grammar grows.

This is a syntactic, best-effort transform: a function is rewritten only
when its linearization is provably a flat `++`-chain of string literals and
its own parameters (the shape `tokenize_concrete_grammar` produces).
Anything else -- tables, records, `case`, `variants`, glue (`+`), calls to
other opers, non-`Str` lincats, module extension (`**`) -- is left
untouched and reported as skipped; leaving a function alone is always
grammar-valid, it just means that function keeps the bug. See
`docs/chainify_limitations.md` for the full list of edge cases and why full
generality is not achievable.

For most callers, `chainify_and_compile` is the entry point to use: it
rewrites the given abstract/concrete pair, compiles it straight to a
`.pgf` with `gf -make`, and deletes the intermediate rewritten .gf source
once compilation succeeds (the rewritten source is rarely interesting on
its own; `keep_gf_files=True` keeps a copy if you want to inspect it). The
original grammar files passed in are only ever read, never modified.
`chainify_files` (rewrite only) and `chainify_grammar` (pure in-memory
transform, no filesystem I/O) are available for lower-level use. See
`docs/chainify.md` for a full usage guide.
"""

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from gf_gcd.grammar_tools.tokenize import compile_grammar

Atom = Tuple[str, str]  # ("lit", '"t100"') or ("var", "quality")


class GFSyntaxError(ValueError):
    """The source could not be parsed at the module/section level.

    Unlike a per-function skip (always safe -- the function is left as-is),
    this means `chainify_grammar` cannot make any safety guarantee about the
    file and refuses to proceed.
    """


class _Unsupported(Exception):
    """Internal control-flow signal: this one function can't be rewritten."""


# --------------------------------------------------------------- tokenizer

_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<linecomment>--[^\n]*)
    | (?P<blockcomment>\{-.*?-\})
    | (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<arrow>->)
    | (?P<fatarrow>=>)
    | (?P<concat>\+\+)
    | (?P<plus>\+)
    | (?P<punct>[{}\[\]();,:=|.*!$<>\\@?])
    | (?P<number>\d+(?:\.\d+)?)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_']*)
    """,
    re.VERBOSE | re.DOTALL,
)


@dataclass
class _Tok:
    kind: str
    text: str
    start: int
    end: int


def _tokenize(src: str) -> List[_Tok]:
    toks = []
    pos, n = 0, len(src)
    while pos < n:
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise GFSyntaxError(f"unrecognized character {src[pos]!r} at offset {pos}")
        kind = m.lastgroup
        if kind not in ("ws", "linecomment", "blockcomment"):
            toks.append(_Tok(kind, m.group(), m.start(), m.end()))
        pos = m.end()
    return toks


def _raw(src: str, toks: Sequence[_Tok]) -> str:
    if not toks:
        return ""
    return src[toks[0].start : toks[-1].end]


def _match_brace(toks: Sequence[_Tok], open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(toks)):
        if toks[i].text == "{":
            depth += 1
        elif toks[i].text == "}":
            depth -= 1
            if depth == 0:
                return i
    raise GFSyntaxError("unbalanced braces in module body")


# ------------------------------------------------------------- module head


@dataclass
class _Module:
    kind: str  # "abstract" | "concrete"
    name: str
    of_name: Optional[str]  # concrete only
    body: List[_Tok]


def _parse_module_header(src: str, toks: List[_Tok]) -> _Module:
    if not toks or toks[0].kind != "ident":
        raise GFSyntaxError("expected a module starting with 'abstract' or 'concrete'")
    head = toks[0].text
    if head == "abstract":
        if len(toks) < 4:
            raise GFSyntaxError("truncated 'abstract' header")
        if toks[2].text == "*":
            raise GFSyntaxError(
                "module extension ('Name ** {...}' / 'Name = Base ** {...}') is not supported"
            )
        if toks[1].kind != "ident" or toks[2].text != "=" or toks[3].text != "{":
            raise GFSyntaxError("expected 'abstract Name = { ... }'")
        close = _match_brace(toks, 3)
        return _Module("abstract", toks[1].text, None, toks[4:close])
    if head == "concrete":
        if len(toks) < 6:
            raise GFSyntaxError("truncated 'concrete' header")
        if toks[2].text != "of" or toks[1].kind != "ident" or toks[3].kind != "ident":
            raise GFSyntaxError("expected 'concrete Name of Abstract = { ... }'")
        if toks[4].text == "*":
            raise GFSyntaxError("module extension ('**') is not supported")
        if toks[4].text != "=" or toks[5].text != "{":
            raise GFSyntaxError("expected 'concrete Name of Abstract = { ... }'")
        close = _match_brace(toks, 5)
        return _Module("concrete", toks[1].text, toks[3].text, toks[6:close])
    raise GFSyntaxError("expected a module starting with 'abstract' or 'concrete'")


# GF top-level section keywords we recognize as *starting* a new section.
# Anything not in this set (comments aside) is assumed to live inside one of
# these sections; a body that starts with something else, or nests an
# unknown top-level construct, is out of scope (see docs/chainify_limitations.md).
_SECTION_KEYWORDS = {
    "flags",
    "cat",
    "fun",
    "def",
    "data",
    "lincat",
    "lindef",
    "lin",
    "oper",
    "param",
    "printname",
    "instance",
}


@dataclass
class _Section:
    keyword: str
    content: List[_Tok]  # tokens after the keyword, up to the next section


def _split_sections(body: List[_Tok]) -> List[_Section]:
    depth = 0
    starts = []
    for idx, t in enumerate(body):
        if t.text in ("{", "("):
            depth += 1
        elif t.text in ("}", ")"):
            depth -= 1
        elif depth == 0 and t.kind == "ident" and t.text in _SECTION_KEYWORDS:
            starts.append(idx)
    if not body:
        return []
    if not starts or starts[0] != 0:
        raise GFSyntaxError("module body does not start with a recognized section keyword")
    starts.append(len(body))
    return [
        _Section(body[starts[i]].text, body[starts[i] + 1 : starts[i + 1]])
        for i in range(len(starts) - 1)
    ]


def _split_chunks(content: List[_Tok]) -> List[List[_Tok]]:
    """Split section content into `;`-terminated top-level declarations."""
    depth = 0
    chunks = []
    start = 0
    for i, t in enumerate(content):
        if t.text in ("{", "("):
            depth += 1
        elif t.text in ("}", ")"):
            depth -= 1
        elif t.text == ";" and depth == 0:
            chunks.append(content[start:i])
            start = i + 1
    if content[start:]:
        raise GFSyntaxError("declaration not terminated with ';'")
    return chunks


def _split_top_level(toks: List[_Tok], sep_text: str) -> List[List[_Tok]]:
    depth = 0
    parts, start = [], 0
    for i, t in enumerate(toks):
        if t.text in ("{", "("):
            depth += 1
        elif t.text in ("}", ")"):
            depth -= 1
        elif depth == 0 and t.text == sep_text:
            parts.append(toks[start:i])
            start = i + 1
    parts.append(toks[start:])
    return parts


# -------------------------------------------------------------- fun / lin


@dataclass
class _FunChunk:
    names: List[str]
    arg_cats: Optional[List[str]]  # None if unparseable
    res_cat: Optional[str]
    raw: str


def _parse_fun_chunk(src: str, chunk: List[_Tok]) -> _FunChunk:
    raw = _raw(src, chunk)
    colon = next((i for i, t in enumerate(chunk) if t.text == ":"), None)
    if colon is None:
        return _FunChunk([], None, None, raw)
    name_parts = _split_top_level(chunk[:colon], ",")
    if any(len(p) != 1 or p[0].kind != "ident" for p in name_parts):
        return _FunChunk([], None, None, raw)
    names = [p[0].text for p in name_parts]
    type_parts = _split_top_level(chunk[colon + 1 :], "->")
    if any(len(p) != 1 or p[0].kind != "ident" for p in type_parts):
        return _FunChunk(names, None, None, raw)
    cats = [p[0].text for p in type_parts]
    return _FunChunk(names, cats[:-1], cats[-1], raw)


@dataclass
class _LinChunk:
    names: List[str]
    params: Optional[List[str]]
    rhs_tokens: Optional[List[_Tok]]
    rhs_raw: str
    raw: str


def _parse_lin_chunk(src: str, chunk: List[_Tok]) -> _LinChunk:
    raw = _raw(src, chunk)
    eq = next((i for i, t in enumerate(chunk) if t.text == "="), None)
    if eq is None:
        return _LinChunk([], None, None, "", raw)
    lhs, rhs = chunk[:eq], chunk[eq + 1 :]
    if not lhs or lhs[0].kind != "ident":
        return _LinChunk([], None, None, _raw(src, rhs), raw)
    names = [lhs[0].text]
    i = 1
    while i < len(lhs) and lhs[i].text == ",":
        if i + 1 >= len(lhs) or lhs[i + 1].kind != "ident":
            return _LinChunk(names, None, None, _raw(src, rhs), raw)
        names.append(lhs[i + 1].text)
        i += 2
    params = []
    while i < len(lhs):
        if lhs[i].kind != "ident":
            return _LinChunk(names, None, None, _raw(src, rhs), raw)
        params.append(lhs[i].text)
        i += 1
    return _LinChunk(names, params, rhs, _raw(src, rhs), raw)


def _flatten(rhs_tokens: List[_Tok], param_names: set) -> List[Atom]:
    """Parse a lin RHS as a flat `++`-chain of literals/params, or bail out."""
    pos = 0

    def parse_expr() -> List[Atom]:
        nonlocal pos
        atoms = parse_term()
        while pos < len(rhs_tokens) and rhs_tokens[pos].text in ("++", "+"):
            if rhs_tokens[pos].text == "+":
                raise _Unsupported("glue operator '+' (string-level paradigm) not supported")
            pos += 1
            atoms = atoms + parse_term()
        return atoms

    def parse_term() -> List[Atom]:
        nonlocal pos
        if pos >= len(rhs_tokens):
            raise _Unsupported("unexpected end of linearization expression")
        t = rhs_tokens[pos]
        if t.kind == "string":
            pos += 1
            return [("lit", t.text)]
        if t.kind == "ident":
            pos += 1
            if t.text in param_names:
                return [("var", t.text)]
            raise _Unsupported(f"reference to non-parameter identifier {t.text!r}")
        if t.text == "(":
            pos += 1
            inner = parse_expr()
            if pos >= len(rhs_tokens) or rhs_tokens[pos].text != ")":
                raise _Unsupported("unbalanced parentheses in linearization")
            pos += 1
            return inner
        raise _Unsupported(f"unsupported construct {t.text!r} in linearization")

    if not rhs_tokens:
        raise _Unsupported("empty linearization body")
    result = parse_expr()
    if pos != len(rhs_tokens):
        raise _Unsupported("trailing tokens after linearization expression")
    return result


# ------------------------------------------------------------ tail chains


class _TailChains:
    """Builds (and dedupes) the chain of single-token categories/functions
    that replace tokens 2..k of a literal run."""

    def __init__(self, used_idents: set):
        self.used = set(used_idents)
        self._cache: Dict[Tuple[str, ...], str] = {}
        self._counter = 0
        self.new_cats: List[str] = []
        self.new_funs: List[Tuple[str, List[str], str]] = []  # (name, arg_cats, res_cat)
        self.new_lins: List[Tuple[str, List[str], List[Atom]]] = []

    def _fresh(self, hint: str) -> str:
        base = re.sub(r"[^A-Za-z0-9]", "", hint) or "X"
        base = base[:24]
        while True:
            self._counter += 1
            cand = f"Tail_{base}_{self._counter}"
            if cand not in self.used:
                self.used.add(cand)
                return cand

    def get(self, tail: Tuple[str, ...]) -> str:
        """`tail` is a non-empty tuple of raw literal texts (e.g. '"t563"')."""
        if tail in self._cache:
            return self._cache[tail]
        first, rest = tail[0], tail[1:]
        cat_name = self._fresh(first.strip('"'))
        fun_name = self._fresh(first.strip('"') + "c")
        if rest:
            rest_cat = self.get(rest)
            self.new_cats.append(cat_name)
            self.new_funs.append((fun_name, [rest_cat], cat_name))
            self.new_lins.append((fun_name, ["xc"], [("lit", first), ("var", "xc")]))
        else:
            self.new_cats.append(cat_name)
            self.new_funs.append((fun_name, [], cat_name))
            self.new_lins.append((fun_name, [], [("lit", first)]))
        self._cache[tail] = cat_name
        return cat_name


def _split_runs(atoms: List[Atom], chains: _TailChains, existing_params: set):
    """Rewrite consecutive-literal runs (len >= 2) into literal + tail-var.

    Returns None if nothing needed to change.
    """
    new_atoms: List[Atom] = []
    extra_params: List[Tuple[str, str]] = []  # (param_name, tail_cat)
    used_names = set(existing_params)
    changed = False
    i, n, run_idx = 0, len(atoms), 0
    while i < n:
        if atoms[i][0] == "lit":
            j = i
            while j < n and atoms[j][0] == "lit":
                j += 1
            run = atoms[i:j]
            if len(run) >= 2:
                changed = True
                run_idx += 1
                tail_cat = chains.get(tuple(a[1] for a in run[1:]))
                pname = f"xtail{run_idx}"
                while pname in used_names:
                    run_idx += 1
                    pname = f"xtail{run_idx}"
                used_names.add(pname)
                extra_params.append((pname, tail_cat))
                new_atoms.append(("lit", run[0][1]))
                new_atoms.append(("var", pname))
            else:
                new_atoms.append(run[0])
            i = j
        else:
            new_atoms.append(atoms[i])
            i += 1
    if not changed:
        return None
    return extra_params, new_atoms


# --------------------------------------------------------------- render


def _render_fun_sig(name: str, arg_cats: Sequence[str], res_cat: str) -> str:
    """One `fun` declaration body, without the leading 'fun' keyword (the
    section header supplies it once)."""
    if arg_cats:
        return f"{name} : {' -> '.join(arg_cats)} -> {res_cat} ;"
    return f"{name} : {res_cat} ;"


def _render_atoms(atoms: Sequence[Atom]) -> str:
    if not atoms:
        return '""'
    return " ++ ".join(a[1] for a in atoms)


def _render_lin_decl(name: str, params: Sequence[str], atoms: Sequence[Atom]) -> str:
    """One `lin` declaration body, without the leading 'lin' keyword."""
    lhs = " ".join([name, *params])
    return f"{lhs} = {_render_atoms(atoms)} ;"


# ------------------------------------------------------------------ report


@dataclass
class ChainifyReport:
    """Summary of what `chainify_grammar` did to one abstract/concrete pair."""

    transformed: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)  # fun name -> reason
    new_categories: int = 0
    new_functions: int = 0

    def __str__(self) -> str:
        lines = [
            f"chainify: {len(self.transformed)} function(s) rewritten, "
            f"{len(self.skipped)} skipped, "
            f"{self.new_categories} new categories, {self.new_functions} new functions"
        ]
        for name in self.transformed:
            lines.append(f"  rewrote  {name}")
        for name, reason in self.skipped.items():
            lines.append(f"  skipped  {name}: {reason}")
        return "\n".join(lines)


# --------------------------------------------------------------- driver


def chainify_grammar(
    abstract_src: str,
    concrete_src: str,
    new_abstract_name: Optional[str] = None,
    new_concrete_name: Optional[str] = None,
    verify: bool = True,
) -> Tuple[str, str, ChainifyReport]:
    """Rewrite an abstract/concrete .gf pair so multi-token functions chain.

    Only functions whose linearization is a flat `++`-chain of string
    literals and the function's own parameters -- and whose result category
    has `lincat = Str` -- are eligible. Anything else (tables, records,
    `case`/`variants`, glue `+`, calls to other opers, non-`Str` lincats,
    `**` module extension) is left byte-for-byte unchanged and reported in
    `ChainifyReport.skipped`; that is always grammar-valid, it just means
    the mid-literal completion bug still applies to that one function.
    See docs/chainify_limitations.md.

    Returns `(new_abstract_src, new_concrete_src, report)`. Compile the
    result with `compile_grammar([new_abstract_path, new_concrete_path], ...)`.
    """
    abs_toks = _tokenize(abstract_src)
    conc_toks = _tokenize(concrete_src)
    abs_mod = _parse_module_header(abstract_src, abs_toks)
    conc_mod = _parse_module_header(concrete_src, conc_toks)
    if abs_mod.kind != "abstract":
        raise GFSyntaxError("abstract_src is not an 'abstract' module")
    if conc_mod.kind != "concrete":
        raise GFSyntaxError("concrete_src is not a 'concrete' module")
    if conc_mod.of_name != abs_mod.name:
        raise GFSyntaxError(
            f"concrete module is 'of {conc_mod.of_name}', but the abstract module is "
            f"{abs_mod.name!r} -- pass the matching pair"
        )

    used_idents = {t.text for t in abs_toks if t.kind == "ident"}
    used_idents |= {t.text for t in conc_toks if t.kind == "ident"}
    chains = _TailChains(used_idents)

    abs_sections = _split_sections(abs_mod.body)
    conc_sections = _split_sections(conc_mod.body)

    # ---- gather fun signatures (name -> chunk parse, section index, chunk index)
    fun_sigs: Dict[str, _FunChunk] = {}
    fun_locations: Dict[str, Tuple[int, int]] = {}
    abs_fun_sections: List[Tuple[int, List[List[_Tok]]]] = []
    for si, sec in enumerate(abs_sections):
        if sec.keyword != "fun":
            continue
        chunks = _split_chunks(sec.content)
        abs_fun_sections.append((si, chunks))
        for ci, chunk in enumerate(chunks):
            parsed = _parse_fun_chunk(abstract_src, chunk)
            for name in parsed.names:
                fun_sigs[name] = parsed
                fun_locations[name] = (si, ci)

    # ---- gather flat (lincat = Str) categories
    flat_cats: set = set()
    for sec in conc_sections:
        if sec.keyword != "lincat":
            continue
        for chunk in _split_chunks(sec.content):
            eq = next((i for i, t in enumerate(chunk) if t.text == "="), None)
            if eq is None:
                continue
            names = _split_top_level(chunk[:eq], ",")
            rhs = chunk[eq + 1 :]
            if len(rhs) == 1 and rhs[0].kind == "ident" and rhs[0].text == "Str":
                for p in names:
                    if len(p) == 1 and p[0].kind == "ident":
                        flat_cats.add(p[0].text)

    # ---- gather lin entries
    lin_chunks: Dict[str, _LinChunk] = {}
    conc_lin_sections: List[Tuple[int, List[List[_Tok]]]] = []
    for si, sec in enumerate(conc_sections):
        if sec.keyword != "lin":
            continue
        chunks = _split_chunks(sec.content)
        conc_lin_sections.append((si, chunks))
        for chunk in chunks:
            parsed = _parse_lin_chunk(concrete_src, chunk)
            for name in parsed.names:
                lin_chunks[name] = parsed

    # ---- decide, per function, whether to transform
    report = ChainifyReport()
    transformed: Dict[str, Tuple[List[str], List[str], List[Atom]]] = {}
    # name -> (new_arg_cats_full, new_params_full, new_atoms)
    all_names = set(fun_sigs) | set(lin_chunks)
    for name in sorted(all_names):
        fsig = fun_sigs.get(name)
        lin = lin_chunks.get(name)
        if fsig is None or fsig.arg_cats is None:
            report.skipped[name] = "abstract 'fun' signature not a plain type (higher-order arg, missing decl, ...)"
            continue
        if lin is None or lin.params is None or lin.rhs_tokens is None:
            report.skipped[name] = "no plain 'lin' rule (record pattern, missing rule, ...)"
            continue
        if fsig.res_cat not in flat_cats:
            report.skipped[name] = f"result category {fsig.res_cat!r} does not have lincat = Str"
            continue
        try:
            atoms = _flatten(lin.rhs_tokens, set(lin.params))
        except _Unsupported as e:
            report.skipped[name] = str(e)
            continue
        result = _split_runs(atoms, chains, set(lin.params))
        if result is None:
            continue  # already fine: no run of >=2 consecutive literals
        extra_params, new_atoms = result
        new_arg_cats = list(fsig.arg_cats) + [cat for _, cat in extra_params]
        new_params = list(lin.params) + [p for p, _ in extra_params]
        transformed[name] = (new_arg_cats, new_params, new_atoms)
        report.transformed.append(name)

    report.new_categories = len(chains.new_cats)
    report.new_functions = len(chains.new_funs)

    # ---- emit abstract
    new_abs_name = new_abstract_name or (abs_mod.name + "Chain")
    new_conc_name = new_concrete_name or (conc_mod.name + "Chain")

    abs_out = [f"abstract {new_abs_name} = {{"]
    for sec in abs_sections:
        if sec.keyword == "fun":
            chunks = _split_chunks(sec.content)
            lines = []
            for chunk in chunks:
                parsed = _parse_fun_chunk(abstract_src, chunk)
                if parsed.arg_cats is None or not any(n in transformed for n in parsed.names):
                    lines.append(parsed.raw + " ;")
                    continue
                for n in parsed.names:
                    if n in transformed:
                        new_args, _, _ = transformed[n]
                        lines.append(_render_fun_sig(n, new_args, parsed.res_cat))
                    else:
                        lines.append(_render_fun_sig(n, parsed.arg_cats, parsed.res_cat))
            abs_out.append("fun\n  " + "\n  ".join(lines))
        else:
            raw = _raw(abstract_src, sec.content)
            abs_out.append(f"{sec.keyword}\n  {raw}" if raw else sec.keyword)
    if chains.new_cats:
        abs_out.append("cat\n  " + "\n  ".join(f"{c} ;" for c in chains.new_cats))
    if chains.new_funs:
        abs_out.append(
            "fun\n  " + "\n  ".join(_render_fun_sig(n, a, r) for n, a, r in chains.new_funs)
        )
    abs_out.append("}")
    abstract_out = "\n".join(abs_out) + "\n"

    # ---- emit concrete
    conc_out = [f"concrete {new_conc_name} of {new_abs_name} = {{"]
    for sec in conc_sections:
        if sec.keyword == "lin":
            chunks = _split_chunks(sec.content)
            lines = []
            for chunk in chunks:
                parsed = _parse_lin_chunk(concrete_src, chunk)
                if parsed.params is None or not any(n in transformed for n in parsed.names):
                    lines.append(parsed.raw + " ;")
                    continue
                for n in parsed.names:
                    if n in transformed:
                        _, new_params, new_atoms = transformed[n]
                        lines.append(_render_lin_decl(n, new_params, new_atoms))
                    else:
                        lines.append(f"lin {' '.join([n, *parsed.params])} = {parsed.rhs_raw} ;")
            conc_out.append("lin\n  " + "\n  ".join(lines))
        else:
            raw = _raw(concrete_src, sec.content)
            conc_out.append(f"{sec.keyword}\n  {raw}" if raw else sec.keyword)
    if chains.new_cats:
        conc_out.append("lincat\n  " + "\n  ".join(f"{c} = Str ;" for c in chains.new_cats))
    if chains.new_lins:
        conc_out.append(
            "lin\n  "
            + "\n  ".join(_render_lin_decl(n, p, a) for n, p, a in chains.new_lins)
        )
    conc_out.append("}")
    concrete_out = "\n".join(conc_out) + "\n"

    if verify:
        _verify_no_multi_literal_runs(concrete_out, set(transformed) | {n for n, _, _ in chains.new_funs})

    return abstract_out, concrete_out, report


def _verify_no_multi_literal_runs(concrete_src: str, must_be_clean: set) -> None:
    """Self-check: re-parse our own output and confirm every function we
    claimed to fix has no remaining run of >=2 consecutive literal tokens.
    Raises AssertionError on failure -- a bug in this module, not the input.
    """
    toks = _tokenize(concrete_src)
    mod = _parse_module_header(concrete_src, toks)
    for sec in _split_sections(mod.body):
        if sec.keyword != "lin":
            continue
        for chunk in _split_chunks(sec.content):
            parsed = _parse_lin_chunk(concrete_src, chunk)
            if parsed.params is None or parsed.rhs_tokens is None:
                continue
            targets = [n for n in parsed.names if n in must_be_clean]
            if not targets:
                continue
            try:
                atoms = _flatten(parsed.rhs_tokens, set(parsed.params))
            except _Unsupported:
                continue
            run = 0
            for a in atoms:
                run = run + 1 if a[0] == "lit" else 0
                if run >= 2:
                    raise AssertionError(
                        f"chainify internal error: {targets} still has a multi-literal run "
                        f"after rewriting"
                    )


def chainify_files(
    abstract_path: Union[str, Path],
    concrete_path: Union[str, Path],
    output_dir: Union[str, Path],
    new_abstract_name: Optional[str] = None,
    new_concrete_name: Optional[str] = None,
) -> Tuple[Path, Path, ChainifyReport]:
    """`chainify_grammar` for files on disk; writes the rewritten pair.

    Only reads `abstract_path` / `concrete_path` -- they are never modified.
    Returns `(new_abstract_path, new_concrete_path, report)`. Compile with
    `compile_grammar([new_abstract_path, new_concrete_path], output_dir)`, or
    use `chainify_and_compile` to get a `.pgf` directly without keeping the
    intermediate .gf source around.
    """
    abstract_path = Path(abstract_path)
    concrete_path = Path(concrete_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    abstract_src = abstract_path.read_text()
    concrete_src = concrete_path.read_text()
    abs_out, conc_out, report = chainify_grammar(
        abstract_src, concrete_src, new_abstract_name, new_concrete_name
    )

    abs_name = new_abstract_name or (abstract_path.stem + "Chain")
    conc_name = new_concrete_name or (concrete_path.stem + "Chain")
    abs_path = output_dir / f"{abs_name}.gf"
    conc_path = output_dir / f"{conc_name}.gf"
    abs_path.write_text(abs_out)
    conc_path.write_text(conc_out)
    return abs_path, conc_path, report


def chainify_and_compile(
    abstract_path: Union[str, Path],
    concrete_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_name: Optional[str] = None,
    new_abstract_name: Optional[str] = None,
    new_concrete_name: Optional[str] = None,
    gf_binary: str = "gf",
    probs_file: Union[str, Path, None] = None,
    keep_gf_files: bool = False,
) -> Tuple[Path, ChainifyReport]:
    """Chainify an abstract/concrete .gf pair and compile it straight to a `.pgf`.

    This is the recommended entry point for most callers: the rewritten .gf
    source is an intermediate artifact you don't normally need to keep, so
    it is written to a private temporary directory, compiled with
    `gf -make`, and discarded -- only the resulting `.pgf` lands in
    `output_dir` (plus a copy of the rewritten .gf pair too, if
    `keep_gf_files=True`).

    `abstract_path` / `concrete_path` are only ever *read*. The rewrite
    always targets a fresh temporary directory, never `output_dir`, so the
    original grammar files are guaranteed unmodified no matter what
    `output_dir` is set to -- including the (unusual but not prohibited)
    case where it's the same directory the originals live in.

    Returns `(pgf_path, report)`. Raises `GFSyntaxError` if the source can't
    be safely rewritten (see `chainify_grammar`), or `RuntimeError` if
    `gf -make` fails on the rewritten grammar -- the latter would indicate a
    bug in this module, since `chainify_grammar` self-verifies its own
    output before returning.
    """
    abstract_path = Path(abstract_path)
    concrete_path = Path(concrete_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="chainify_") as tmp:
        tmp_dir = Path(tmp)
        abs_path, conc_path, report = chainify_files(
            abstract_path,
            concrete_path,
            tmp_dir,
            new_abstract_name=new_abstract_name,
            new_concrete_name=new_concrete_name,
        )

        dest_abs = output_dir / abs_path.name
        dest_conc = output_dir / conc_path.name
        if keep_gf_files:
            for dest, original in ((dest_abs, abstract_path), (dest_conc, concrete_path)):
                if dest.resolve() == original.resolve():
                    raise ValueError(
                        f"refusing to overwrite the original grammar file {original} -- "
                        f"pass a different output_dir, new_abstract_name, or "
                        f"new_concrete_name"
                    )

        pgf_path = compile_grammar(
            [abs_path, conc_path],
            tmp_dir,
            output_name=output_name,
            gf_binary=gf_binary,
            probs_file=probs_file,
        )
        dest_pgf = output_dir / pgf_path.name
        shutil.copy2(pgf_path, dest_pgf)

        if keep_gf_files:
            shutil.copy2(abs_path, dest_abs)
            shutil.copy2(conc_path, dest_conc)

    return dest_pgf, report
