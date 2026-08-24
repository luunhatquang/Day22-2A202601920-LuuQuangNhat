import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_computation() -> None:
    # When policy and ref have identical log-ratios, logit is 0 -> loss = ln(2) ~ 0.693147
    pi_c = np.array([-1.0])
    pi_r = np.array([-2.0])
    ref_c = np.array([-1.0])
    ref_r = np.array([-2.0])
    loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert np.isclose(loss, np.log(2.0), atol=1e-4)

    # When policy aligns better with chosen than ref, loss decreases below ln(2)
    pi_c_better = np.array([-0.2])
    loss_better = dpo_loss(pi_c_better, pi_r, ref_c, ref_r, beta=1.0)
    assert loss_better < np.log(2.0)


def test_dpo_loss_numerical_stability() -> None:
    # Extreme positive and negative values should not produce NaN or Inf
    extreme_large = np.array([1000.0])
    extreme_small = np.array([-1000.0])
    zeros = np.array([0.0])

    loss_large = dpo_loss(extreme_large, zeros, zeros, zeros, beta=1.0)
    assert not np.isnan(loss_large) and not np.isinf(loss_large)
    assert np.isclose(loss_large, 0.0, atol=1e-4)

    loss_small = dpo_loss(extreme_small, zeros, zeros, zeros, beta=1.0)
    assert not np.isnan(loss_small) and not np.isinf(loss_small)
    assert loss_small > 0.0


def test_orpo_loss_computation() -> None:
    sft_nll = np.array([1.2])
    chosen_logp = np.array([-0.5])
    rejected_logp = np.array([-1.5])
    loss = orpo_loss(sft_nll, chosen_logp, rejected_logp, lambda_orpo=0.1)
    assert isinstance(loss, float)
    assert loss > 0.0
    # Loss should be greater than SFT alone due to non-zero OR penalty
    assert loss > 1.2


def test_orpo_loss_numerical_stability() -> None:
    sft_nll = np.array([0.5, 1.0])
    chosen_logp = np.array([-0.00001, -50.0])
    rejected_logp = np.array([-50.0, -0.00001])
    loss = orpo_loss(sft_nll, chosen_logp, rejected_logp, lambda_orpo=0.1)
    assert not np.isnan(loss) and not np.isinf(loss)
    assert loss > 0.0
