import pytest
from pydantic import ValidationError

from preference_lab.schemas import PreferenceExample


def test_valid_preference_example() -> None:
    ex = PreferenceExample(
        prompt="Explain backprop",
        chosen="It computes gradients.",
        rejected="It is an activation function.",
    )
    assert ex.prompt == "Explain backprop"
    assert ex.chosen == "It computes gradients."
    assert ex.rejected == "It is an activation function."


def test_strip_whitespace() -> None:
    ex = PreferenceExample(
        prompt="  Prompt text   ",
        chosen="   Chosen text  ",
        rejected="   Rejected text  ",
    )
    assert ex.prompt == "Prompt text"
    assert ex.chosen == "Chosen text"
    assert ex.rejected == "Rejected text"


def test_reject_identical_chosen_and_rejected() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(
            prompt="Prompt",
            chosen="Identical answer",
            rejected="Identical answer",
        )


def test_reject_near_duplicate_differing_only_by_case_or_spaces() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(
            prompt="Prompt",
            chosen="Identical Answer",
            rejected="   identical   answer  ",
        )


def test_reject_near_duplicate_differing_only_by_punctuation() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(
            prompt="Prompt",
            chosen="Identical answer.",
            rejected="Identical answer!",
        )
