"""WordBeamSearcher tests with a controllable fake LM (hermetic, no downloads)."""

import math

import pytest
import torch
from transformers import DynamicCache

from gf_gcd.gf_probs import GFProbRuntime
from gf_gcd.word_beam import WordBeamSearcher

from .conftest import requires_gf

FOOD_WORDS = [
    "p", "this", "that", "wine", "cheese", "fish", "is", "very",
    "fresh", "warm", "Italian", "expensive", "delicious", "boring",
]


class FakeTokenizer:
    """Whitespace tokenizer over a closed vocabulary; id 0 is EOS."""

    def __init__(self, words):
        self.vocab = {w: i + 1 for i, w in enumerate(words)}
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.vocab_size = len(words) + 1

    def encode(self, text, add_special_tokens=True):
        return [self.vocab[w] for w in text.split()]


class FakeModel(torch.nn.Module):
    """Causal-LM stand-in: position-independent logits, optionally biased."""

    def __init__(self, vocab_size, bias=None):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        logits = torch.zeros(vocab_size)
        for idx, value in (bias or {}).items():
            logits[idx] = value
        self.register_buffer("base_logits", logits)

    def forward(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        batch, length = input_ids.shape
        logits = self.base_logits.expand(batch, length, -1)
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        # dummy 1-layer cache: content is never read (logits are
        # position-independent), only its sequence length matters
        dummy = torch.zeros(batch, 1, past_len + length, 1)
        cache = DynamicCache.from_legacy_cache(((dummy, dummy),))
        return type("Out", (), {"logits": logits, "past_key_values": cache})()


@pytest.fixture(scope="module")
def food_runtime(prob_grammar_dir):
    return GFProbRuntime(prob_grammar_dir, default_pgf="Food.pgf")


def make_searcher(food_runtime, lambda_gf, bias=None, beam_size=4, **kwargs):
    tok = FakeTokenizer(FOOD_WORDS)
    bias_ids = {tok.vocab[w]: v for w, v in (bias or {}).items()}
    model = FakeModel(tok.vocab_size, bias_ids)
    return WordBeamSearcher(
        model, tok, food_runtime, lambda_gf=lambda_gf, beam_size=beam_size, **kwargs
    )


@requires_gf
def test_pure_grammar_follows_probs_file(food_runtime):
    # lambda=1: LM is ignored; most probable derivation is
    # This(0.7) Wine(0.5) Pred Delicious(0.4) -> "this wine is delicious"
    searcher = make_searcher(food_runtime, lambda_gf=1.0)
    best = searcher.search("p", max_words=8)[0]
    assert best.finished
    assert best.text == "this wine is delicious"
    # P = This(0.7) * Wine(0.5) * Delicious(0.4|quality step) * eos(1)
    assert best.score == pytest.approx(best.gf_logprob)
    assert best.gf_logprob == pytest.approx(math.log(0.7) + math.log(0.5) + math.log(0.4), abs=0.15)


@requires_gf
def test_pure_lm_choice_within_grammar(food_runtime):
    # lambda=0: grammar only masks; the LM strongly prefers "fish" and "warm"
    searcher = make_searcher(food_runtime, lambda_gf=0.0, bias={"fish": 8.0, "warm": 8.0, "that": 4.0})
    best = searcher.search("p", max_words=8)[0]
    assert best.finished
    assert best.text == "that fish is warm"
    assert best.score == pytest.approx(best.llm_logprob)


@requires_gf
def test_fusion_trades_off_both(food_runtime):
    # LM pushes towards "fish", grammar towards "wine"; a mild LM bias should
    # win under lambda=0 but lose under lambda close to 1
    bias = {"fish": 2.0}
    lm_led = make_searcher(food_runtime, lambda_gf=0.0, bias=bias).search("p")[0]
    gf_led = make_searcher(food_runtime, lambda_gf=0.95, bias=bias).search("p")[0]
    assert "fish" in lm_led.words
    assert "wine" in gf_led.words


@requires_gf
def test_score_is_convex_combination(food_runtime):
    lam = 0.3
    best = make_searcher(food_runtime, lambda_gf=lam).search("p")[0]
    assert best.score == pytest.approx(lam * best.gf_logprob + (1 - lam) * best.llm_logprob)


@requires_gf
def test_unfinished_when_max_words_too_small(food_runtime):
    searcher = make_searcher(food_runtime, lambda_gf=1.0)
    results = searcher.search("p", max_words=2)
    assert results and not results[0].finished


@requires_gf
def test_num_return(food_runtime):
    results = make_searcher(food_runtime, lambda_gf=1.0).search("p", num_return=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


@requires_gf
def test_invalid_lambda_rejected(food_runtime):
    with pytest.raises(ValueError):
        make_searcher(food_runtime, lambda_gf=1.5)


@requires_gf
def test_search_can_chain_multiple_sentences(food_runtime):
    # the Food grammar accepts one Comment; with a budget wide enough for
    # two, and enough beam/return width to reach past the (always
    # higher-scoring, per the fused formula) single-Comment completions,
    # `search` should surface hypotheses that chain a second Comment after
    # the first one completes.
    searcher = make_searcher(food_runtime, lambda_gf=1.0, beam_size=64)
    results = searcher.search("p", max_words=10, num_return=64)
    chained = [h for h in results if h.finished and h.words.count("is") > 1]
    assert chained, "expected at least one chained (multi-sentence) result"

    best_chain = chained[0]
    # words spans both sentences, but sentence_words -- what the next
    # grammar query would see -- is only the *last* one
    assert best_chain.words.count("is") == 2
    assert best_chain.sentence_words.count("is") == 1
    assert best_chain.words[-len(best_chain.sentence_words) :] == best_chain.sentence_words
    assert best_chain.sentence_words != best_chain.words
