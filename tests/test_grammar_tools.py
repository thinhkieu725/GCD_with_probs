from gf_gcd.grammar_tools import ids_to_gf_lin, tokenize_concrete_grammar


class FakeTokenizer:
    """Encodes each character to its codepoint; enough to test the rewriting."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_ids_to_gf_lin():
    assert ids_to_gf_lin([31, 287]) == '"31" ++ "287"'
    assert ids_to_gf_lin([]) == '""'


def test_tokenize_concrete_grammar_rewrites_literals():
    src = 'lin Wine = "ab" ;'
    out = tokenize_concrete_grammar(FakeTokenizer(), src, prefix_space=False)
    assert out == f'lin Wine = ("{ord("a")}" ++ "{ord("b")}") ;'


def test_tokenize_concrete_grammar_keeps_empty_strings():
    src = 'lin Empty = "" ;'
    assert tokenize_concrete_grammar(FakeTokenizer(), src) == src
