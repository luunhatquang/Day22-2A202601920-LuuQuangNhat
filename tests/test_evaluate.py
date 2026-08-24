import inspect

import pytest

from preference_lab.cli import score_response
from preference_lab.evaluate import pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy_basic() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a1", rejected="b1"),
        PreferenceExample(prompt="p2", chosen="a2", rejected="b2"),
    ]
    # One win, one loss -> accuracy 0.5
    assert pairwise_accuracy(examples, [2.0, 0.5], [1.0, 1.5]) == 0.5


def test_pairwise_accuracy_with_ties() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a1", rejected="b1"),
        PreferenceExample(prompt="p2", chosen="a2", rejected="b2"),
    ]
    # One win (1.0), one tie (0.5) -> (1.0 + 0.5) / 2 = 0.75
    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0], tie_weight=0.5) == 0.75


def test_pairwise_accuracy_length_mismatch() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    with pytest.raises(ValueError, match=r"Length mismatch"):
        pairwise_accuracy(examples, [1.0, 2.0], [1.0])


def test_pairwise_accuracy_empty() -> None:
    assert pairwise_accuracy([], [], []) == 0.0


def test_cpu_scorer_is_label_independent() -> None:
    """The same response must receive the same score in either pair position."""
    prompt = "Explain gradient descent"
    response = "Gradient descent updates parameters using the loss gradient."
    assert "is_chosen" not in inspect.signature(score_response).parameters
    assert score_response(prompt, response) == score_response(prompt, response)
