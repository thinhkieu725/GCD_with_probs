import math

import pytest
import torch

from gf_gcd.constraints import GFProbConstraint

from .conftest import requires_gf

VOCAB = 500
EOS = 2
LAMBDA = 0.4


@pytest.fixture()
def constraint(prob_grammar_dir):
    return GFProbConstraint(
        grammar_dir=str(prob_grammar_dir),
        default_grammar="SeqTok.pgf",
        eos_token_id=EOS,
        lambda_gf=LAMBDA,
    )


def allowed_ids(fused_row):
    return sorted(torch.nonzero(fused_row > float("-inf")).flatten().tolist())


@requires_gf
class TestGFProbLogitsProcessor:
    def test_start_masks_everything_but_first_token(self, constraint):
        proc = constraint.logits_processor(prompt_lengths=[3])
        input_ids = torch.tensor([[7, 8, 9]])  # prompt only
        scores = torch.zeros((1, VOCAB))
        fused = proc(input_ids, scores)
        assert allowed_ids(fused[0]) == [100]
        # sole candidate: P_GF = 1, fused = 0 + lambda * ln 1 = 0
        assert fused[0, 100].item() == pytest.approx(0.0)

    def test_fused_scores_reflect_grammar_probs(self, constraint):
        proc = constraint.logits_processor(prompt_lengths=[0])
        input_ids = torch.tensor([[100]])
        scores = torch.zeros((1, VOCAB))
        fused = proc(input_ids, scores)
        assert allowed_ids(fused[0]) == [200, 300]
        assert fused[0, 200].item() == pytest.approx(LAMBDA * math.log(0.9), abs=1e-4)
        assert fused[0, 300].item() == pytest.approx(LAMBDA * math.log(0.1), abs=1e-4)

    def test_lambda_zero_is_pure_mask(self, prob_grammar_dir):
        constraint = GFProbConstraint(
            grammar_dir=str(prob_grammar_dir),
            default_grammar="SeqTok.pgf",
            eos_token_id=EOS,
            lambda_gf=0.0,
        )
        proc = constraint.logits_processor(prompt_lengths=[0])
        fused = proc(torch.tensor([[100]]), torch.zeros((1, VOCAB)))
        assert fused[0, 200].item() == pytest.approx(0.0)
        assert fused[0, 300].item() == pytest.approx(0.0)

    def test_complete_derivation_allows_ending_or_a_new_sentence(self, constraint):
        # "t100 t200" is complete with no legal continuation left: the
        # fusion also scores the grammar's own start-of-sentence token
        # (t100), so generation can chain into a second sentence instead of
        # only being allowed to stop
        proc = constraint.logits_processor(prompt_lengths=[0])
        fused = proc(torch.tensor([[100, 200]]), torch.zeros((1, VOCAB)))
        assert allowed_ids(fused[0]) == [EOS, 100]
        # both are certain (P=1) one-step decisions: end here, or start a
        # fresh sentence with the grammar's only possible first token
        assert fused[0, EOS].item() == pytest.approx(0.0)
        assert fused[0, 100].item() == pytest.approx(0.0)

    def test_chains_a_second_sentence_after_the_first_completes(self, constraint):
        proc = constraint.logits_processor(prompt_lengths=[0])
        # "t100 t200" (complete) followed by "t100": scored against the tail
        # of the second sentence alone (`[100]`), not the whole 3-token
        # history, which the grammar cannot parse as one derivation
        fused = proc(torch.tensor([[100, 200, 100]]), torch.zeros((1, VOCAB)))
        assert allowed_ids(fused[0]) == [200, 300]

    def test_finished_row_passes_eos_through(self, constraint):
        proc = constraint.logits_processor(prompt_lengths=[0])
        fused = proc(torch.tensor([[100, 200, EOS]]), torch.zeros((1, VOCAB)))
        assert allowed_ids(fused[0]) == [EOS]

    def test_beam_rows_map_to_batch_entries(self, constraint):
        # 1 prompt, 2 beams: both rows share prompt_lengths[0]
        proc = constraint.logits_processor(prompt_lengths=[1])
        input_ids = torch.tensor([[7, 100], [7, 100]])
        fused = proc(input_ids, torch.zeros((2, VOCAB)))
        assert allowed_ids(fused[0]) == allowed_ids(fused[1]) == [200, 300]

    def test_word_level_grammar_gives_clear_error(self, prob_grammar_dir):
        constraint = GFProbConstraint(
            grammar_dir=str(prob_grammar_dir),
            default_grammar="Food.pgf",
            eos_token_id=EOS,
        )
        proc = constraint.logits_processor(prompt_lengths=[0])
        with pytest.raises(ValueError, match="token-ID grammar"):
            proc(torch.tensor([[7]])[:, 1:], torch.zeros((1, VOCAB)))


@requires_gf
def test_prefix_allowed_tokens_fn_compat(constraint):
    fn = constraint.prefix_allowed_tokens_fn(prompt_lengths=[0])
    assert fn(0, torch.tensor([], dtype=torch.long)) == [100]
    assert sorted(fn(0, torch.tensor([100]))) == [200, 300]
    # complete, no legal continuation left: end here, or start a fresh sentence
    assert sorted(fn(0, torch.tensor([100, 200]))) == [EOS, 100]
    # a second sentence's tokens are masked against their own tail, not the
    # unparseable 3-token history
    assert sorted(fn(0, torch.tensor([100, 200, 100]))) == [200, 300]


@requires_gf
def test_generator_prefers_logits_processor(constraint):
    # the generator picks the reweighting path iff logits_processor() is non-None:
    # GFProbConstraint provides one, the mask-only GFConstraint inherits None
    from gf_gcd.constraints import GFConstraint

    assert constraint.logits_processor(prompt_lengths=[0]) is not None
    mask_only = GFConstraint.__new__(GFConstraint)
    assert mask_only.logits_processor(prompt_lengths=[0]) is None
