"""WordBeamSearcher tests with a controllable fake LM (hermetic, no downloads)."""

import math

import pytest
import torch
from transformers import DynamicCache

from gf_gcd.gf_probs import GF_EOS, GFProbRuntime
from gf_gcd.word_beam import (
    WordBeamSearcher,
    WordHypothesis,
    _Expansion,
    _filter_min_p,
    _filter_top_k,
    _filter_top_p,
    _filter_typical_p,
)

from .conftest import requires_gf

FOOD_WORDS = [
    "p", "this", "that", "wine", "cheese", "fish", "is", "very",
    "fresh", "warm", "Italian", "expensive", "delicious", "boring",
    ".", "?", "!",  # sentence-terminal marks: not grammar words, but the LM
    # must be able to score/encode them (see WordBeamSearcher._SENTENCE_END_MARKS)
    "This", "That",  # capitalized forms `_candidate_ids` scores/materializes
    # a sentence-initial "this"/"that" as -- see WordBeamSearcher._candidate_ids
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
    # lambda=1: LM is ignored *for word/mark choice*; most probable derivation
    # is This(0.7) Wine(0.5) Pred Delicious(0.4) -> "this wine is delicious",
    # closed by whichever mark ties first (all three cost the grammar the
    # same -- see test_sentence_end_mark_... below) and then stopped for real
    searcher = make_searcher(food_runtime, lambda_gf=1.0)
    best = searcher.search("p", max_words=8)[0]
    assert best.finished
    assert best.text == "This wine is delicious."
    # P = This(0.7) * Wine(0.5) * Delicious(0.4|quality step) * eos(1);
    # closing with "." also costs the grammar nothing (same eos(1))
    assert best.gf_logprob == pytest.approx(math.log(0.7) + math.log(0.5) + math.log(0.4), abs=0.15)
    # unlike the grammar-priced words/mark above, *stopping the whole
    # answer* right after "." is priced purely by the LM regardless of
    # lambda_gf (see module docstring) -- so, unlike before this hypothesis
    # could finish for free, `score` now always carries that real LM cost
    assert best.score < best.gf_logprob


@requires_gf
def test_pure_lm_choice_within_grammar(food_runtime):
    # lambda=0: grammar only masks; the LM strongly prefers "fish" and "warm".
    # "that" is only ever legal sentence-initially in this grammar, so it's
    # scored/materialized as "That" (see WordBeamSearcher._candidate_ids) --
    # bias that surface form, not the never-scored lowercase one.
    searcher = make_searcher(food_runtime, lambda_gf=0.0, bias={"fish": 8.0, "warm": 8.0, "That": 4.0})
    best = searcher.search("p", max_words=8)[0]
    assert best.finished
    assert best.text == "That fish is warm"
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
    # capped short of finishing: once a hypothesis reaches its free "stop the
    # whole answer" step (see module docstring), that step's LM cost isn't
    # scaled by lambda_gf, so `score` stops being a pure convex combination
    # of gf_logprob/llm_logprob -- restrict to words still mid-sentence,
    # where every step so far was an ordinary grammar-priced word
    best = make_searcher(food_runtime, lambda_gf=lam).search("p", max_words=2)[0]
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
    # two -- each now also spending one word-step on its closing mark, plus
    # one more step to decide against stopping the whole answer in between
    # -- and enough beam/return width to reach past the (always
    # higher-scoring, per the fused formula) single-Comment completions,
    # `search` should surface hypotheses that chain a second Comment after
    # the first one completes rather than being forced to stop there. A
    # mild bias on "This" (the sentence-initial surface form -- see
    # WordBeamSearcher._candidate_ids -- of the legal sentence-starting word
    # "this") is needed so the deterministic "stop the whole answer" gate
    # (see module docstring) has something to lose to: with a fully unbiased
    # LM the real eos token ties every continuation candidate exactly, and
    # ties favor stopping. This `FakeModel` is position-independent, so that
    # same bias makes the gate favor continuing identically every time it's
    # offered -- chained hypotheses in this test are therefore never
    # `finished` within `max_words`, only still-live and demonstrably past
    # one sentence.
    searcher = make_searcher(food_runtime, lambda_gf=1.0, beam_size=64, bias={"This": 3.0})
    results = searcher.search("p", max_words=12, num_return=64)
    chained = [h for h in results if h.words.count("is") > 1]
    assert chained, "expected at least one chained (multi-sentence) result"

    best_chain = chained[0]
    assert best_chain.words.count("is") >= 2
    # the "." between the two sentences closed the first one (reset
    # `sentence_words`) without ending the hypothesis outright
    dot_idx = best_chain.words.index(".")
    assert best_chain.words[dot_idx + 1] == "this"


# ---------------------------------------------------------------------------
# Sentence-terminal punctuation (`_SENTENCE_END_MARKS`): the grammar has no
# productions for "." "?" "!" at all (see module docstring for why), so these
# tests exercise the escape hatch `search` offers whenever a legal stop
# exists -- each mark priced with the grammar's own `GF_EOS` cost blended
# against the LM by `lambda_gf` -- and its interaction with `sentence_words`
# resetting so a hypothesis can still chain into another sentence afterward.


@requires_gf
def test_punctuation_offered_and_chosen_when_lm_prefers_it(food_runtime):
    # a huge bias on "." makes it the obvious choice the moment the grammar
    # says stopping is legal (here: a forced stop, "this wine is delicious");
    # wide beam/return so the (necessarily longer, hence lower-scoring under
    # length_penalty=0) punctuated completions aren't cut by num_return
    # before we get to inspect one
    searcher = make_searcher(food_runtime, lambda_gf=0.3, bias={".": 20.0}, beam_size=64)
    results = searcher.search("p", max_words=8, num_return=64)
    closed = [h for h in results if "." in h.words]
    assert closed, "expected at least one hypothesis to close with '.'"
    hyp = closed[0]
    # the mark closes the sentence: it's the last word of `words` but is not
    # part of `sentence_words` (which resets to [] right after it)
    assert hyp.words[-1] == "."
    assert hyp.sentence_words == []


class _OptionalStopRuntime:
    """Fake runtime whose sentence can *optionally* end at any length >= 1:
    every step offers both GF_EOS and a live continuation ("more"), a shape
    the toy Food grammar (forced stops only) cannot produce on its own."""

    def next_token_logprobs(self, words, pgf_name=None, temperature=1.0):
        if not words:
            return {"more": math.log(1.0)}  # must start with a real word
        return {GF_EOS: math.log(0.5), "more": math.log(0.5)}


def test_sentence_end_mark_prices_stop_lp_and_blends_with_lambda_gf():
    # a sentence-end mark's gf-side cost is now `GF_EOS`'s own cost -- the
    # same `stop_lp` that used to end the hypothesis outright -- not a
    # hardcoded 0.0, and it *does* blend with `lambda_gf` like an ordinary
    # word. At lambda_gf=0.0 the LM alone picks which mark wins ("." here,
    # via a huge bias), but `gf_logprob` still accumulates `stop_lp`
    # (log(0.5), from `_OptionalStopRuntime`) regardless of that weighting.
    tok = FakeTokenizer(["p", "more", "More", ".", "?", "!"])  # "more": _OptionalStopRuntime's
    # only word; "More" is its sentence-initial surface form (see
    # WordBeamSearcher._candidate_ids) -- must be encodable even unbiased
    bias = {".": 20.0}  # overwhelm "more" and the other marks
    model = FakeModel(tok.vocab_size, {tok.vocab[w]: v for w, v in bias.items()})
    searcher = WordBeamSearcher(model, tok, _OptionalStopRuntime(), lambda_gf=0.0, beam_size=8)

    results = searcher.search("p", max_words=3, num_return=8)
    hyp = next(h for h in results if h.words[:2] == ["more", "."])
    # "more" (step 0, forced, gf_lp=log(1.0)=0.0) + "." (step 1, gf_lp=stop_lp)
    assert hyp.gf_logprob == pytest.approx(math.log(0.5), abs=1e-6)


def test_stop_whole_answer_step_ignores_lambda_gf():
    # unlike the sentence-end mark above, *stopping the whole answer* right
    # after one is still priced purely by the LM (gf_lp=0.0, `gf_free=True`)
    # -- the grammar only ever describes a single sentence, so it has
    # nothing to say about when a multi-sentence answer as a whole is done.
    # At lambda_gf=1.0 the mark choice itself becomes LM-blind (its LM term
    # is zeroed out), but the free real-stop step's LM cost still isn't --
    # so `score` comes out strictly below the fully-grammar-priced
    # `gf_logprob`, instead of the degenerate `score == gf_logprob` a
    # fully grammar-priced hypothesis would give
    # (`test_pure_grammar_follows_probs_file`).
    tok = FakeTokenizer(["p", "more", "More", ".", "?", "!"])
    bias = {".": 6.0, "!": -3.0}  # distinct, non-degenerate logits per mark
    model = FakeModel(tok.vocab_size, {tok.vocab[w]: v for w, v in bias.items()})
    searcher = WordBeamSearcher(model, tok, _OptionalStopRuntime(), lambda_gf=1.0, beam_size=8)

    results = searcher.search("p", max_words=3, num_return=8)
    hyp = next(h for h in results if h.finished and h.words == ["more", "."])
    assert hyp.gf_logprob == pytest.approx(math.log(0.5), abs=1e-6)
    assert hyp.score < hyp.gf_logprob


class _TwoStartWordsRuntime:
    """Sentence always ends after exactly one word (forced); starting a new
    sentence offers two candidate opening words, so the stop-the-whole-
    answer gate (see module docstring) has more than one alternative to
    compare the real eos token's score against."""

    def next_token_logprobs(self, words, pgf_name=None, temperature=1.0):
        if not words:
            return {"this": math.log(0.5), "that": math.log(0.5)}
        return {GF_EOS: math.log(1.0)}


def test_stop_gate_compares_raw_llm_scores_ignoring_lambda_gf():
    # the deterministic "stop the whole answer" gate picks whichever of
    # {real eos, each legal fresh-sentence opening word} has the highest
    # *raw* LM log-probability -- lambda_gf plays no part on either side of
    # that comparison. lambda_gf=1.0 here isolates this: it zeroes out any
    # LM influence on ordinary word/mark choices, so if biasing "That" (the
    # sentence-initial surface form of "that" -- see
    # WordBeamSearcher._candidate_ids) still changes whether the hypothesis
    # stops (even though "that" itself never actually gets chosen afterward,
    # since lambda_gf=1.0 makes the subsequent word-choice grammar-only too),
    # that change can only be coming from the LM-only gate comparison, not
    # from any blended score.
    tok = FakeTokenizer(["p", "this", "that", "This", "That", ".", "?", "!"])

    eos_wins = WordBeamSearcher(
        FakeModel(tok.vocab_size, {tok.eos_token_id: 5.0}),
        tok,
        _TwoStartWordsRuntime(),
        lambda_gf=1.0,
        beam_size=8,
    ).search("p", max_words=6, num_return=8)[0]
    assert eos_wins.finished
    assert eos_wins.words == ["this", "."]

    continues = WordBeamSearcher(
        FakeModel(tok.vocab_size, {tok.vocab["That"]: 5.0}),
        tok,
        _TwoStartWordsRuntime(),
        lambda_gf=1.0,
        beam_size=8,
    ).search("p", max_words=6, num_return=8)[0]
    assert not continues.finished
    assert len(continues.words) > 2  # kept going past the first mark


def test_length_penalty_lets_the_stop_gate_win_after_running_on():
    # a mild, constant bias on "That" (the sentence-initial surface form of
    # "that" -- see WordBeamSearcher._candidate_ids) narrowly beats the real
    # eos token's raw (unbiased) score every single time the stop gate is
    # offered -- with length_penalty=0 that's the same fixed comparison
    # forever, so the hypothesis never stops within budget. A positive
    # length_penalty divides eos's score by a factor that grows with the
    # hypothesis's current length (`_length_norm`), softening its cost over
    # time, so the same fixed comparison eventually flips in eos's favor once
    # the hypothesis has run on long enough -- without changing anything
    # about the very first sentence, where that factor is close to a no-op.
    tok = FakeTokenizer(["p", "this", "that", "This", "That", ".", "?", "!"])
    model = FakeModel(tok.vocab_size, {tok.vocab["That"]: 0.5})

    unpenalized = WordBeamSearcher(
        model, tok, _TwoStartWordsRuntime(), lambda_gf=0.5, beam_size=8, length_penalty=0.0
    ).search("p", max_words=9, num_return=8)[0]
    assert not unpenalized.finished

    penalized = WordBeamSearcher(
        model, tok, _TwoStartWordsRuntime(), lambda_gf=0.5, beam_size=8, length_penalty=0.8
    ).search("p", max_words=9, num_return=8)[0]
    assert penalized.finished
    assert penalized.words == ["that", ".", "that", "."]


@requires_gf
def test_punctuation_can_chain_into_another_sentence(food_runtime):
    # after closing with ".", the hypothesis should still be able to start a
    # brand new Comment rather than being forced to stop right there. Besides
    # biasing "." for the mark choice, "This" (the sentence-initial surface
    # form of "this" -- see WordBeamSearcher._candidate_ids) also needs a
    # mild bias so the deterministic stop-the-whole-answer gate has something
    # to lose to (see test_search_can_chain_multiple_sentences).
    searcher = make_searcher(
        food_runtime, lambda_gf=1.0, beam_size=64, bias={".": 5.0, "This": 3.0}
    )
    results = searcher.search("p", max_words=12, num_return=64)
    chained = [h for h in results if h.words.count(".") >= 1 and h.words.count("is") > 1]
    assert chained, "expected a hypothesis that closed a sentence with '.' and then chained another"
    hyp = chained[0]
    # everything after the mark belongs to a fresh sentence: "is" appears
    # again after it
    dot_idx = hyp.words.index(".")
    assert "is" in hyp.words[dot_idx + 1 :]


class _SpaceSensitiveTokenizer:
    """Unlike `FakeTokenizer` (which `.split()`s and so can't distinguish a
    leading space), maps the exact encoded string to an id -- " wine" and
    "wine" are different tokens, like a real BPE tokenizer's are."""

    def __init__(self, words):
        surface = [w for word in words for w in (word, " " + word)]
        self.vocab = {s: i + 1 for i, s in enumerate(dict.fromkeys(surface))}
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        return [self.vocab[text]]


def test_candidate_ids_punctuation_has_no_leading_space():
    tok = _SpaceSensitiveTokenizer(["wine", ".", "?", "!"])
    model = FakeModel(len(tok.vocab) + 1)
    searcher = WordBeamSearcher(model, tok, runtime=None, prefix_space=True)
    # ordinary mid-sentence words get the leading-space form ("cat" -> " cat")
    assert searcher._candidate_ids("wine") == [tok.vocab[" wine"]]
    # sentence-terminal marks attach directly, with no leading space at all
    assert searcher._candidate_ids(".") == [tok.vocab["."]]
    assert searcher._candidate_ids("?") == [tok.vocab["?"]]
    assert searcher._candidate_ids("!") == [tok.vocab["!"]]


def test_candidate_ids_capitalize_only_uppercases_the_first_letter():
    tok = _SpaceSensitiveTokenizer(["wine", "Wine", "John", ".", "?", "!"])
    model = FakeModel(len(tok.vocab) + 1)
    searcher = WordBeamSearcher(model, tok, runtime=None, prefix_space=True)
    # sentence-initial candidates get their first letter uppercased for the
    # LM (real text essentially never starts a sentence lowercase, so an
    # uncapitalized candidate would be unfairly starved -- see module history)
    assert searcher._candidate_ids("wine", capitalize=True) == [tok.vocab[" Wine"]]
    # a terminal that's already capitalized (e.g. a proper noun) is
    # unaffected -- only the first character ever changes, not `.capitalize()`
    # (which would also lowercase the rest and mangle "John" into "John"... in
    # this case harmlessly, but not in general, e.g. an acronym)
    assert searcher._candidate_ids("John", capitalize=True) == [tok.vocab[" John"]]
    # capitalize is a no-op for GF_EOS and sentence-terminal marks
    assert searcher._candidate_ids(".", capitalize=True) == [tok.vocab["."]]


def test_hypothesis_text_has_no_space_before_punctuation():
    # `_candidate_ids` feeding the LM a space-less token for "." "?" "!" only
    # controls what the *model* sees; `WordHypothesis.text` reconstructs the
    # displayed string from `words` independently (`" ".join` would put a
    # space before every word, punctuation included) and must mark the same
    # marks as exceptions, or the rendered text still comes out "cat ." even
    # though the model was scored on "cat.".
    hyp = WordHypothesis(words=["this", "wine", "is", "delicious", ".", "is", "it", "fresh", "?"])
    assert hyp.text == "This wine is delicious. Is it fresh?"


def test_punctuation_offered_at_an_optional_stop_not_just_forced():
    # regression test: `_SENTENCE_END_MARKS` must be offered any time GF_EOS
    # is *one of* the legal continuations, not only when it's the *only* one
    tok = FakeTokenizer(["p", "more", "More", ".", "?", "!"])
    bias = {".": 20.0}  # overwhelm "more" so the searcher takes the mark
    model = FakeModel(tok.vocab_size, {tok.vocab[w]: v for w, v in bias.items()})
    searcher = WordBeamSearcher(
        model, tok, _OptionalStopRuntime(), lambda_gf=0.3, beam_size=8
    )
    results = searcher.search("p", max_words=4, num_return=8)
    assert any("." in h.words for h in results), (
        "expected '.' to be offered (and chosen) at an optional stop, "
        "where GF_EOS is legal but not the only continuation"
    )


# ---------------------------------------------------------------------------
# Pure-algorithm unit tests for the search-quality knobs (length_penalty,
# diversity_penalty, no_repeat_ngram_size) -- these helpers touch no
# model/tokenizer/runtime, so no GF grammar or fake LM is needed.


def _bare_searcher(**kwargs):
    return WordBeamSearcher(model=None, tokenizer=None, runtime=None, **kwargs)


def test_length_norm_is_noop_by_default():
    s = _bare_searcher()
    assert s._length_norm(1) == 1.0
    assert s._length_norm(10) == 1.0


def test_length_norm_narrows_the_gap_between_different_lengths():
    s = _bare_searcher(length_penalty=0.8)
    short = WordHypothesis(words=["a"], score=-1.0)
    long_ = WordHypothesis(words=["a"] * 8, score=-6.0)
    # raw score favors `short`; normalization should shrink that gap since
    # `long_` only "lost" by accumulating more (necessarily negative) words
    assert short.score > long_.score
    raw_gap = short.score - long_.score
    normalized_gap = s._normalized_score(short) - s._normalized_score(long_)
    assert 0 <= normalized_gap < raw_gap


def test_filter_repeats_blocks_recreated_ngram():
    s = _bare_searcher(no_repeat_ngram_size=2)
    words, gf_lps, flags = s._filter_repeats(
        history=["a", "b", "a"],
        words=["b", "c", GF_EOS],
        gf_lps=[0.1, 0.2, 0.3],
        flags=[False, False, False],
    )
    # "a" -> "b" already occurred once, so "b" is banned as the next word
    # after another "a"; "c" and GF_EOS are unaffected
    assert "b" not in words
    assert set(words) == {"c", GF_EOS}
    assert len(words) == len(gf_lps) == len(flags)


def test_filter_repeats_never_empties_the_candidate_list():
    s = _bare_searcher(no_repeat_ngram_size=2)
    words, gf_lps, flags = s._filter_repeats(
        history=["a", "b"], words=["b"], gf_lps=[0.1], flags=[False]
    )
    # banning the only candidate would leave nowhere to go -> fall back unfiltered
    assert words == ["b"]


def test_prune_candidates_noop_by_default():
    s = _bare_searcher()
    words, gf_lps, flags = s._prune_candidates(
        words=["a", "b", GF_EOS], gf_lps=[-0.1, -3.0, -5.0], flags=[False, False, False]
    )
    assert words == ["a", "b", GF_EOS]


def test_prune_candidates_top_k_keeps_highest_gf_lp():
    s = _bare_searcher(candidate_top_k=2)
    words, gf_lps, flags = s._prune_candidates(
        words=["a", "b", "c"], gf_lps=[-0.1, -5.0, -2.0], flags=[False, False, False]
    )
    assert set(words) == {"a", "c"}
    assert len(words) == len(gf_lps) == len(flags) == 2


def test_prune_candidates_always_keeps_gf_eos():
    s = _bare_searcher(candidate_top_k=1)
    # GF_EOS is by far the worst-scoring candidate here, but pruning it
    # would leave the hypothesis with no way to ever stop
    words, gf_lps, flags = s._prune_candidates(
        words=["a", "b", GF_EOS], gf_lps=[-0.1, -0.2, -9.0], flags=[False, False, False]
    )
    assert GF_EOS in words
    assert "a" in words  # the one real top-1 candidate
    assert len(words) == 2


def test_prune_candidates_top_p_drops_the_long_tail():
    s = _bare_searcher(candidate_top_p=0.9)
    # "a" alone already carries ~95% of the probability mass
    gf_lps = [math.log(0.95), math.log(0.03), math.log(0.02)]
    words, _, _ = s._prune_candidates(words=["a", "b", "c"], gf_lps=gf_lps, flags=[False] * 3)
    assert words == ["a"]


def test_select_diverse_avoids_two_survivors_with_the_same_word():
    s = _bare_searcher(beam_size=2, diversity_penalty=5.0)
    h1, h2 = WordHypothesis(), WordHypothesis()
    e_best_x = _Expansion(h1, "x", 0.0, 0.0)
    e_second_x = _Expansion(h2, "x", 0.0, -0.1)
    e_y = _Expansion(h1, "y", 0.0, -1.0)
    # without diversity, the top 2 by raw score would both be "x"
    continuations = [(0.0, e_best_x), (-0.1, e_second_x), (-1.0, e_y)]
    survivors = s._select_diverse(continuations)
    assert {e.word for _, e in survivors} == {"x", "y"}


# ---------------------------------------------------------------------------
# Sampling mode (do_sample=True): pure-algorithm tests for the filter
# cascade and `_sample_index` (bare, no grammar/model needed), plus a few
# GF-backed tests for end-to-end behavior with the toy Food grammar.


def test_filter_top_k_keeps_only_the_k_highest():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, -1.0])
    kept = torch.isfinite(_filter_top_k(logits, 2))
    assert kept.tolist() == [False, True, True, False, False]


def test_filter_top_p_always_keeps_the_top_candidate():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, -1.0])
    kept = torch.isfinite(_filter_top_p(logits, 1e-6))  # near-maximal cutoff
    assert kept[1].item() is True
    assert kept.sum().item() >= 1


def test_filter_min_p_always_keeps_the_top_candidate():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, -1.0])
    kept = torch.isfinite(_filter_min_p(logits, 0.9))  # aggressive relative cutoff
    assert kept[1].item() is True
    assert kept.sum().item() >= 1


def test_filter_typical_p_never_empties_the_candidate_set():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5, -1.0])
    assert torch.isfinite(_filter_typical_p(logits, 0.01)).sum().item() >= 1


def test_sample_index_near_zero_temperature_is_argmax():
    s = _bare_searcher(do_sample=True, temperature=1e-8)
    assert s._sample_index(torch.tensor([1.0, 3.0, 2.0])) == 1


def test_sample_index_top_k_1_is_deterministic():
    s = _bare_searcher(do_sample=True, top_k=1, seed=0)
    logits = torch.tensor([1.0, 3.0, 2.0])
    assert all(s._sample_index(logits) == 1 for _ in range(5))


def test_do_sample_and_diversity_penalty_are_mutually_exclusive():
    with pytest.raises(ValueError):
        _bare_searcher(do_sample=True, diversity_penalty=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_p": 1.5},
        {"top_p": 0.0},
        {"min_p": -0.1},
        {"min_p": 1.1},
        {"typical_p": 0.0},
        {"typical_p": 1.1},
        {"temperature": 0.0},
        {"temperature": -1.0},
    ],
)
def test_invalid_sampling_params_rejected(kwargs):
    with pytest.raises(ValueError):
        _bare_searcher(**kwargs)


@requires_gf
def test_do_sample_top_k_1_matches_deterministic_greedy(food_runtime):
    # top_k=1 collapses each hypothesis's sampling draw to the same per-step
    # argmax that deterministic beam_size=1 beam search already performs (a
    # single hypothesis's own top-1 candidate is the same whether it's found
    # by pooling-then-sorting or by a temperature/top-k-filtered multinomial
    # draw with k=1) -- this is a regression check that the new selection
    # path agrees with the existing one in this degenerate case.
    #
    # Unlike the pooled path -- which always drops "stop the whole answer"
    # into `done` the moment it's offered, regardless of its score relative
    # to continuing -- sampling only stops there if that candidate genuinely
    # wins the per-step draw. Since that step is now always LM-priced (see
    # module docstring) rather than free, greedy sampling isn't guaranteed to
    # prefer it on its own; a strong bias on the real eos token makes it the
    # top-1 pick once offered, so both paths agree on *when* to stop too.
    tok = FakeTokenizer(FOOD_WORDS)
    model = FakeModel(tok.vocab_size, {tok.eos_token_id: 10.0})
    greedy = WordBeamSearcher(model, tok, food_runtime, lambda_gf=1.0, beam_size=1).search(
        "p", max_words=8
    )[0]
    sampled = WordBeamSearcher(
        model, tok, food_runtime, lambda_gf=1.0, beam_size=1, do_sample=True, top_k=1, seed=0
    ).search("p", max_words=8)[0]
    assert sampled.finished
    assert sampled.text == greedy.text


@requires_gf
def test_do_sample_is_reproducible_with_a_seed(food_runtime):
    kwargs = dict(lambda_gf=0.3, beam_size=4, do_sample=True, temperature=1.5, seed=42)
    first = make_searcher(food_runtime, **kwargs).search("p", max_words=8, num_return=4)
    second = make_searcher(food_runtime, **kwargs).search("p", max_words=8, num_return=4)
    assert [h.text for h in first] == [h.text for h in second]


@requires_gf
def test_do_sample_beam_size_acts_as_independent_chain_count(food_runtime):
    n_chains = 3
    results = make_searcher(
        food_runtime, lambda_gf=0.3, beam_size=n_chains, do_sample=True, temperature=1.0, seed=1
    ).search("p", max_words=8, num_return=n_chains)
    assert 1 <= len(results) <= n_chains
    assert all(isinstance(h, WordHypothesis) for h in results)


@requires_gf
def test_do_sample_actually_explores_beam_size_independent_chains(food_runtime):
    # regression test: search() must seed `beam_size` independent starting
    # hypotheses under do_sample=True, not one hypothesis that (unlike pooled
    # beam search) can never fan out on its own since each hypothesis only
    # ever advances by exactly one word per step. Without that seeding, at
    # most 1 chain -- and therefore at most 1 distinct result -- could ever
    # exist, no matter how large beam_size is.
    results = make_searcher(
        food_runtime,
        lambda_gf=0.0,
        beam_size=8,
        # "this"/"that" are only ever legal sentence-initially in this
        # grammar, so they're scored/materialized as "This"/"That" (see
        # WordBeamSearcher._candidate_ids) -- bias those surface forms.
        bias={"This": 1.0, "That": 1.0, "wine": 1.0, "cheese": 1.0, "fish": 1.0},
        do_sample=True,
        temperature=2.0,
        seed=0,
    ).search("p", max_words=8, num_return=8)
    assert len({h.text for h in results}) > 1


class _DeadEndRuntime:
    """`next_token_logprobs` always returns no legal continuation -- a
    genuine grammar dead end, distinct from a forced stop (which returns
    `{GF_EOS: ...}`)."""

    def next_token_logprobs(self, words, pgf_name=None, temperature=1.0):
        return {}


def test_do_sample_all_chains_dead_end_keeps_last_valid_beams():
    # regression test: if every live chain's grammar continuation is empty in
    # the same step, `search` must fall back to the last valid hypotheses
    # instead of overwriting `beams` with the (empty) sampling result and
    # returning `[]` -- which crashed callers doing `results[0]` (do_sample's
    # per-step selection only reassigns `beams` unconditionally; the pooled
    # path avoids this because it never reassigns `beams` when `expansions`
    # comes back empty).
    tok = FakeTokenizer(FOOD_WORDS)
    model = FakeModel(tok.vocab_size)
    searcher = WordBeamSearcher(
        model, tok, _DeadEndRuntime(), lambda_gf=0.5, beam_size=3, do_sample=True, seed=0
    )
    results = searcher.search("p", max_words=5, num_return=3)
    assert results
    assert not results[0].finished
    assert results[0].words == []  # never advanced past the (dead-end) initial state
