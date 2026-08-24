from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_split_returns_all_examples_without_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5, seed=42)
    assert len(train) + len(val) == len(examples)

    # Check zero leakage between train and validation prompts
    train_prompts = {ex.prompt.strip().lower() for ex in train}
    val_prompts = {ex.prompt.strip().lower() for ex in val}
    assert len(train_prompts & val_prompts) == 0


def test_load_jsonl_line_error_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"prompt": "valid", "chosen": "a", "rejected": "b"}\n{malformed json}\n')
    with pytest.raises(ValueError, match=r"Line 2: Invalid JSON"):
        load_jsonl(bad_file)


def test_load_jsonl_schema_validation_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "identical.jsonl"
    bad_file.write_text('{"prompt": "test", "chosen": "same answer", "rejected": "same answer"}\n')
    with pytest.raises(ValueError, match=r"Line 1: Schema validation error"):
        load_jsonl(bad_file)


def test_load_jsonl_duplicate_prompt_error(tmp_path: Path) -> None:
    dup_file = tmp_path / "dup.jsonl"
    dup_file.write_text(
        '{"prompt": "What is AI?", "chosen": "Artificial Intelligence", "rejected": "A game"}\n'
        '{"prompt": "What is AI?", "chosen": "Machine learning system", "rejected": "Wrong answer"}\n'
    )
    with pytest.raises(ValueError, match=r"Line 2: Duplicate prompt detected"):
        load_jsonl(dup_file)


def test_load_jsonl_pii_check(tmp_path: Path) -> None:
    pii_file = tmp_path / "pii.jsonl"
    pii_file.write_text(
        '{"prompt": "Contact me at alice@example.com", "chosen": "Valid answer", "rejected": "Bad answer"}\n'
    )
    # Should succeed without PII check
    assert len(load_jsonl(pii_file, check_pii=False)) == 1
    # Should fail with PII check
    with pytest.raises(ValueError, match=r"Line 1: PII detected"):
        load_jsonl(pii_file, check_pii=True)
