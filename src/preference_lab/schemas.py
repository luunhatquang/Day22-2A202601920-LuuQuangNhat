from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PreferenceExample(BaseModel):
    """One preference pair for DPO/ORPO-style alignment."""

    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, rejected: str, info: Any) -> str:
        chosen = info.data.get("chosen")
        if chosen is None:
            return rejected

        # Normalize whitespace and case
        norm_chosen = " ".join(chosen.strip().lower().split())
        norm_rejected = " ".join(rejected.strip().lower().split())

        if norm_chosen == norm_rejected:
            raise ValueError("chosen and rejected must differ")

        # Check for near duplicates differing only by punctuation
        clean_chosen = re.sub(r"[^\w\s]", "", norm_chosen)
        clean_rejected = re.sub(r"[^\w\s]", "", norm_rejected)
        if clean_chosen == clean_rejected:
            raise ValueError(
                "chosen and rejected must differ (near duplicate differing only by punctuation/case/whitespace)"
            )

        return rejected
