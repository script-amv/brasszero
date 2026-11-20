# SPDX-License-Identifier: Apache-2.0
"""Exact whole-list normalization, masking and singleton learning semantics."""
import pytest
import torch

from brass_rl.distributions import PolicyOutput, policy_probabilities, policy_stats


def test_policy_legal_normalized():
    logits = torch.tensor([1., 2., 999., -2., 4.], requires_grad=True)
    out = PolicyOutput(logits, torch.tensor([0, 3, 5]),
                       torch.tensor([True, True, False, True, True]), torch.full((2, 4), .25))
    probs = policy_probabilities(out)
    torch.testing.assert_close(probs[:2], torch.softmax(torch.tensor([1., 2.]), 0))
    assert probs[2].item() == 0
    assert probs[3:].sum().item() == pytest.approx(1.)
    stats = policy_stats(out, torch.tensor([1, 0]))
    assert stats.values.shape == (2, 4)
    assert torch.isfinite(stats.entropy).all()
    (-stats.log_prob.sum() - stats.entropy.sum()).backward()
    assert logits.grad[2].item() == 0


def test_singleton_has_zero_logp_entropy_and_policy_gradient():
    logits = torch.tensor([2., 9.], requires_grad=True)
    raw_value = torch.zeros((1, 4), requires_grad=True)
    out = PolicyOutput(logits, torch.tensor([0, 2]), torch.tensor([True, False]), raw_value.sigmoid())
    stats = policy_stats(out, torch.tensor([0]))
    assert stats.log_prob.item() == 0.
    assert stats.entropy.item() == 0.
    (stats.log_prob.sum() + stats.entropy.sum() + stats.values.sum()).backward()
    assert torch.equal(logits.grad, torch.zeros(2))
    assert (raw_value.grad != 0).all()


@pytest.mark.parametrize("offsets,valid,choice", [([0, 0], [], 0), ([0, 2], [False, False], 0),
                                                 ([0, 2], [True, False], 1),
                                                 ([0, 2], [True, True], -1)])
def test_invalid_distribution_rejected(offsets, valid, choice):
    out = PolicyOutput(torch.zeros(len(valid)), torch.tensor(offsets),
                       torch.tensor(valid, dtype=torch.bool), torch.zeros((1, 4)))
    with pytest.raises(ValueError):
        policy_stats(out, torch.tensor([choice]))


def test_nonfinite_real_logits_rejected_but_padding_ignored():
    out = PolicyOutput(torch.tensor([0., float("nan")]), torch.tensor([0, 2]),
                       torch.tensor([True, False]), torch.zeros((1, 4)))
    assert policy_probabilities(out).tolist() == [1., 0.]
    with pytest.raises(ValueError, match="finite"):
        policy_probabilities(PolicyOutput(out.logits, out.offsets, torch.ones(2, dtype=torch.bool), out.values))
