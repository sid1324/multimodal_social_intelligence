import pytest
import torch

from social_rl.grpo import clipped_token_objective, group_advantages


def test_binary_group_advantages_are_centered_and_standardized():
    a = group_advantages([1, 0, 0, 0])
    assert a.mean().item() == pytest.approx(0, abs=1e-6)
    assert a.std(unbiased=False).item() == pytest.approx(1, abs=1e-6)
    assert a[0] > 0 and (a[1:] < 0).all()
    assert torch.equal(group_advantages([1, 1, 1, 1]), torch.zeros(4))


def test_clipping_stops_favorable_extra_policy_gradient():
    old = torch.tensor([0.0])
    ref = torch.tensor([0.0])
    # At beta=0, a positive advantage should stop growing once the ratio
    # exceeds 1+epsilon, while an unclipped ratio still has gradient.
    logps = torch.tensor([0.5], requires_grad=True)
    loss, _, fraction = clipped_token_objective(logps, old, ref, 1.0, 0.2, 0.0)
    loss.backward()
    assert fraction.item() == 1.0
    assert logps.grad.item() == pytest.approx(0.0, abs=1e-7)
    logps2 = torch.tensor([0.0], requires_grad=True)
    loss2, _, fraction2 = clipped_token_objective(logps2, old, ref, 1.0, 0.2, 0.0)
    loss2.backward()
    assert fraction2.item() == 0.0
    assert logps2.grad.item() < 0
