"""Decision logic."""

from __future__ import annotations

from typing import Any

from .constants import (
    DECISION_ACCEPT,
    DECISION_CORRECT,
    DECISION_MISMATCH,
    DECISION_REVIEW,
    FIELD_ABSENT,
    NO_CORRECTION,
    SOURCE_EXTRACTION,
    SOURCE_JUDGE_CORRECTION,
    VERDICT_PASS,
    VERDICT_UNCERTAIN,
)
from .models import JudgeFieldResult


def decide(
    extracted: Any,
    judge: JudgeFieldResult,
    threshold: float,
) -> tuple[str, Any, str]:
    """Return (decision, final_value, final_value_source)."""
    if judge.verdict is None:
        return DECISION_REVIEW, extracted, SOURCE_EXTRACTION

    confidence = judge.confidence if judge.confidence is not None else 0.0

    if judge.verdict == VERDICT_UNCERTAIN or confidence < threshold:
        return DECISION_REVIEW, extracted, SOURCE_EXTRACTION

    if judge.verdict == VERDICT_PASS:
        return DECISION_ACCEPT, extracted, SOURCE_EXTRACTION

    # FAIL at or above threshold
    correction = judge.correction_value
    if correction == FIELD_ABSENT:
        return DECISION_CORRECT, None, SOURCE_JUDGE_CORRECTION
    if correction and correction != NO_CORRECTION:
        return DECISION_CORRECT, correction, SOURCE_JUDGE_CORRECTION
    return DECISION_MISMATCH, extracted, SOURCE_EXTRACTION
