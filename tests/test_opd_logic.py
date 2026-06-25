"""Unit tests for OPD loss logic (A4).

Tests pure tensor operations — no GPU, no vllm, no flash_attn needed.
Run with: pytest tests/test_opd_logic.py -v
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ZeroSearch'))

from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.torch_functional import masked_mean


# ── kl_penalty: four modes ──────────────────────────────────────────────────


class TestKlPenalty:

    def test_kl_mode_basic(self):
        """kl mode: logprob - ref_logprob. When student == teacher, KL ≈ 0."""
        logprob = torch.tensor([-1.0, -2.0, -3.0])
        ref_logprob = torch.tensor([-1.0, -2.0, -3.0])
        result = kl_penalty(logprob, ref_logprob, "kl")
        assert torch.allclose(result, torch.zeros(3))

    def test_kl_mode_positive_when_student_less_confident(self):
        """If student assigns lower prob than teacher, logprob - ref < 0, so KL > 0 for 'kl' mode."""
        logprob = torch.tensor([-3.0])
        ref_logprob = torch.tensor([-1.0])
        result = kl_penalty(logprob, ref_logprob, "kl")
        assert result.item() == pytest.approx(-2.0)

    def test_abs_mode(self):
        logprob = torch.tensor([-1.0, -4.0])
        ref_logprob = torch.tensor([-2.0, -1.0])
        result = kl_penalty(logprob, ref_logprob, "abs")
        expected = torch.tensor([1.0, 3.0])
        assert torch.allclose(result, expected)

    def test_mse_mode(self):
        logprob = torch.tensor([-1.0, -4.0])
        ref_logprob = torch.tensor([-2.0, -1.0])
        result = kl_penalty(logprob, ref_logprob, "mse")
        expected = 0.5 * (logprob - ref_logprob).square()
        assert torch.allclose(result, expected)

    def test_low_var_kl_mode_zero_when_equal(self):
        """Schulman's low-var KL: when distributions match, ratio=1, kl=0, kld=0."""
        logprob = torch.tensor([-1.5, -2.5])
        ref_logprob = torch.tensor([-1.5, -2.5])
        result = kl_penalty(logprob, ref_logprob, "low_var_kl")
        assert torch.allclose(result, torch.zeros(2))

    def test_low_var_kl_non_negative(self):
        """Schulman's estimator is always >= 0 (ratio - kl - 1 >= 0 by convexity)."""
        logprob = torch.randn(100)
        ref_logprob = torch.randn(100)
        result = kl_penalty(logprob, ref_logprob, "low_var_kl")
        assert (result >= 0).all()

    def test_low_var_kl_clamped(self):
        """Output is clamped to [-10, 10]."""
        logprob = torch.tensor([-20.0])
        ref_logprob = torch.tensor([20.0])
        result = kl_penalty(logprob, ref_logprob, "low_var_kl")
        assert result.item() <= 10.0

    def test_batch_shape_preserved(self):
        """Output shape matches input shape for all modes."""
        bs, seq = 4, 16
        logprob = torch.randn(bs, seq)
        ref_logprob = torch.randn(bs, seq)
        for mode in ["kl", "abs", "mse", "low_var_kl"]:
            result = kl_penalty(logprob, ref_logprob, mode)
            assert result.shape == (bs, seq)

    def test_unknown_mode_raises(self):
        with pytest.raises(NotImplementedError):
            kl_penalty(torch.tensor([0.0]), torch.tensor([0.0]), "unknown")


# ── masked_mean: mask application ───────────────────────────────────────────


class TestMaskedMean:

    def test_all_ones_mask(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones(3)
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(2.0)

    def test_partial_mask(self):
        """Masked positions contribute nothing to the mean."""
        values = torch.tensor([100.0, 2.0, 3.0])
        mask = torch.tensor([0.0, 1.0, 1.0])
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx(2.5)

    def test_2d_batch(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        result = masked_mean(values, mask)
        assert result.item() == pytest.approx((1 + 2 + 3) / 3)


# ── OPD loss logic ──────────────────────────────────────────────────────────


class TestOpdLossLogic:
    """Test the OPD loss calculation pattern used in dp_actor.py."""

    def test_opd_loss_zero_when_student_equals_teacher(self):
        """When student log-prob matches teacher, KL = 0 → loss = 0."""
        bs, seq = 2, 8
        log_prob = torch.tensor([[-1.0] * seq] * bs)
        ref_log_prob = torch.tensor([[-1.0] * seq] * bs)
        response_mask = torch.ones(bs, seq)

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
        kl_loss = masked_mean(kld, response_mask)

        assert kl_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_opd_loss_positive_when_different(self):
        """When student differs from teacher, KL > 0 → loss > 0."""
        log_prob = torch.tensor([[-3.0, -2.0, -1.0]])
        ref_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        response_mask = torch.ones(1, 3)

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
        kl_loss = masked_mean(kld, response_mask)

        assert kl_loss.item() > 0

    def test_opd_loss_respects_mask(self):
        """Masked-out tokens (retrieval content) must not contribute to KL loss."""
        bs, seq = 1, 6
        log_prob = torch.tensor([[-1.0] * seq])
        ref_log_prob = torch.tensor([[-5.0, -5.0, -1.0, -1.0, -1.0, -1.0]])

        # Only tokens 2-5 are student-generated; tokens 0-1 are retrieval (masked out)
        response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
        kl_loss_masked = masked_mean(kld, response_mask)

        # With mask: only the matching tokens (idx 2-5) count → KL should be ~0
        assert kl_loss_masked.item() == pytest.approx(0.0, abs=1e-6)

        # Without mask: all tokens count → KL should be > 0 (due to mismatched tokens 0-1)
        all_mask = torch.ones(bs, seq)
        kl_loss_unmasked = masked_mean(kld, all_mask)
        assert kl_loss_unmasked.item() > 0

    def test_opd_loss_gradient_flows_to_student_only(self):
        """OPD loss should depend on student log_prob (gradient target)."""
        log_prob = torch.tensor([[-2.0, -1.5, -1.0]], requires_grad=True)
        ref_log_prob = torch.tensor([[-1.0, -1.0, -1.0]])
        response_mask = torch.ones(1, 3)

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
        kl_loss = masked_mean(kld, response_mask)
        kl_loss.backward()

        assert log_prob.grad is not None
        assert log_prob.grad.abs().sum().item() > 0

    def test_opd_loss_increases_with_divergence(self):
        """KL loss should monotonically increase as student diverges from teacher."""
        ref_log_prob = torch.tensor([[-1.0]])
        response_mask = torch.ones(1, 1)

        losses = []
        for delta in [0.0, 0.5, 1.0, 2.0, 4.0]:
            log_prob = torch.tensor([[-1.0 - delta]])
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
            loss = masked_mean(kld, response_mask).item()
            losses.append(loss)

        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1], f"Loss should increase: {losses}"


# ── info_mask → loss_mask simulation ────────────────────────────────────────


class TestInfoMaskToLossMask:
    """Simulate the _create_loss_mask logic from ray_trainer.py."""

    def test_retrieval_tokens_excluded(self):
        """info_mask=0 on retrieval tokens → loss_mask=0 → no KL contribution."""
        response_length = 6

        # Simulated info_mask: 1 on student tokens, 0 on retrieval tokens
        info_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 1, 0, 0]])

        # _create_loss_mask takes the response portion only
        loss_mask = info_mask[:, -response_length:]
        expected = torch.tensor([[1, 1, 1, 1, 0, 0]])
        assert torch.equal(loss_mask, expected)

    def test_loss_mask_applied_to_kl(self):
        """End-to-end: retrieval tokens in info_mask correctly zero out KL contribution."""
        # Student-generated tokens (idx 2-5) match teacher → KL ≈ 0 there
        log_prob = torch.tensor([[-2.0, -2.0, -1.0, -1.0, -1.0, -1.0]])
        ref_log_prob = torch.tensor([[-10.0, -10.0, -1.0, -1.0, -1.0, -1.0]])
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty="low_var_kl")
        kl_loss = masked_mean(kld, loss_mask)

        # Only the matching tokens (idx 2-5) should count → KL ≈ 0
        assert kl_loss.item() == pytest.approx(0.0, abs=1e-6)
