from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample

PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"(?:\+?1[-. ]?)?\(?[0-9]{3}\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}\b"),  # Phone
    re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,})\b"),  # API keys
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),  # Credit card
]


def contains_pii(text: str) -> bool:
    """Check if text contains potential PII patterns."""
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def load_jsonl(
    path: str | Path,
    check_pii: bool = False,
    allow_duplicate_prompts: bool = False,
) -> list[PreferenceExample]:
    """Load preference examples from JSONL.

    Includes line-numbered error messages, duplicate prompt detection,
    and optional PII guardrails.
    """
    examples: list[PreferenceExample] = []
    seen_prompts: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            trimmed = line.strip()
            if not trimmed:
                continue

            try:
                data = json.loads(trimmed)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num}: Invalid JSON - {exc.msg}") from exc

            try:
                example = PreferenceExample.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"Line {line_num}: Schema validation error - {exc}") from exc

            norm_prompt = " ".join(example.prompt.strip().lower().split())
            if not allow_duplicate_prompts and norm_prompt in seen_prompts:
                raise ValueError(f"Line {line_num}: Duplicate prompt detected: '{example.prompt}'")
            seen_prompts.add(norm_prompt)

            if check_pii:
                combined_text = f"{example.prompt} {example.chosen} {example.rejected}"
                if contains_pii(combined_text):
                    raise ValueError(f"Line {line_num}: PII detected in preference example")

            examples.append(example)

    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid data leakage between splits.

    Groups examples by prompt, deterministically shuffles prompt groups using seed,
    and splits them according to validation_ratio.
    """
    if not examples:
        return [], []

    prompt_groups: dict[str, list[PreferenceExample]] = defaultdict(list)
    for ex in examples:
        prompt_key = " ".join(ex.prompt.strip().lower().split())
        prompt_groups[prompt_key].append(ex)

    unique_prompts = list(prompt_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_prompts)

    if validation_ratio <= 0.0:
        val_prompts = set()
    elif validation_ratio >= 1.0:
        val_prompts = set(unique_prompts)
    else:
        num_val_prompts = max(1, round(len(unique_prompts) * validation_ratio))
        if num_val_prompts >= len(unique_prompts) and len(unique_prompts) > 1:
            num_val_prompts = len(unique_prompts) - 1
        val_prompts = set(unique_prompts[:num_val_prompts])

    train_examples: list[PreferenceExample] = []
    val_examples: list[PreferenceExample] = []

    for prompt_key, group in prompt_groups.items():
        if prompt_key in val_prompts:
            val_examples.extend(group)
        else:
            train_examples.extend(group)

    return train_examples, val_examples
