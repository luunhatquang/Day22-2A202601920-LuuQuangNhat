from __future__ import annotations

import json
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
    tie_weight: float = 0.5,
) -> float:
    """Return fraction where chosen score is greater than rejected score.

    Validates that example and score lengths match, and awards `tie_weight` (default 0.5)
    for exact ties between chosen and rejected scores.
    """
    if not examples:
        return 0.0

    if len(examples) != len(chosen_scores) or len(examples) != len(rejected_scores):
        raise ValueError(
            f"Length mismatch: examples={len(examples)}, "
            f"chosen_scores={len(chosen_scores)}, rejected_scores={len(rejected_scores)}"
        )

    score_sum = 0.0
    for c, r in zip(chosen_scores, rejected_scores, strict=True):
        if c > r:
            score_sum += 1.0
        elif c == r:
            score_sum += tie_weight

    return score_sum / len(examples)


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
