from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def dpo_loss(
    policy_chosen_logps: FloatArray,
    policy_rejected_logps: FloatArray,
    ref_chosen_logps: FloatArray,
    ref_rejected_logps: FloatArray,
    beta: float = 0.1,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    DPO Loss = -E [ log sigma(beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))) ]
    Using np.logaddexp(0, -logits) for numerical stability (since -log(sigma(z)) = log(1 + exp(-z))).
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)

    # -log(sigmoid(logits)) == log(1 + exp(-logits)) == logaddexp(0, -logits)
    losses = np.logaddexp(0, -logits)
    return float(np.mean(losses))


def orpo_loss(
    sft_nll: FloatArray,
    chosen_logps: FloatArray,
    rejected_logps: FloatArray,
    lambda_orpo: float = 0.1,
    eps: float = 1e-7,
) -> float:
    """Compute ORPO objective (SFT loss + odds-ratio preference penalty).

    odds(y|x) = P(y|x) / (1 - P(y|x))
    log_odds(y|x) = logp - log(1 - exp(logp))
    L_OR = -log sigma(log_odds(chosen) - log_odds(rejected))
    L_ORPO = E[SFT_NLL] + lambda_orpo * E[L_OR]
    """
    # Ensure logps do not cause log(0) in log(1 - exp(logp))
    safe_chosen_logps = np.clip(chosen_logps, -100.0, -eps)
    safe_rejected_logps = np.clip(rejected_logps, -100.0, -eps)

    log_odds_chosen = safe_chosen_logps - np.log1p(-np.exp(safe_chosen_logps))
    log_odds_rejected = safe_rejected_logps - np.log1p(-np.exp(safe_rejected_logps))

    log_odds_ratio = log_odds_chosen - log_odds_rejected
    or_penalty = np.logaddexp(0, -log_odds_ratio)

    total_loss = np.mean(sft_nll) + lambda_orpo * np.mean(or_penalty)
    return float(total_loss)
