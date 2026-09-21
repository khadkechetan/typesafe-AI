"""
Core data model.

Ground-truth leakage prevention is STRUCTURAL: prompt builders accept a
DocumentView, which physically does not carry ground truth. GroundTruth is a
separate object that never reaches extraction or judging.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field as dc_field
from typing import Any

from .constants import REASON_SOURCE_UNAVAILABLE, SOURCE_NONE


@dataclass(frozen=True)
class DocumentView:
    """The ONLY document object that may reach a prompt. Carries no labels."""

    doc_id: str
    doc_index: int
    ocr_text: str
    ocr_lines: tuple[str, ...]


@dataclass(frozen=True)
class GroundTruth:
    """Labels. Never passed to extraction or judging; used only for scoring."""

    doc_id: str
    fields: dict[str, Any]  # value may be str | list[str] | None


@dataclass
class ExtractionResult:
    model: str
    values: dict[str, Any]
    latency_ms: float | None = None
    error: str | None = None
    raw_response: str | None = None
    usage: dict[str, int] | None = None
    json_mode_used: bool = True


@dataclass
class JudgeFieldResult:
    verdict: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    grounded_noul: float | None = None
    correction_value: str | None = None
    correction_confidence: float | None = None
    correction_probabilities: dict[str, float] | None = None
    reason: str | None = None
    reason_source: str = REASON_SOURCE_UNAVAILABLE
    error: str | None = None


@dataclass
class JudgeDocResult:
    model: str
    served_model: str | None = None
    fields: dict[str, JudgeFieldResult] = dc_field(default_factory=dict)
    latency_ms: float | None = None
    usage: dict[str, int] | None = None
    error: str | None = None


@dataclass
class FieldRecord:
    """One row per (document, model, field). Four disjoint value groups."""

    # identity
    run_id: str
    doc_id: str
    doc_index: int
    model: str
    field: str
    field_type: str

    # --- group 1: raw extraction (never overwritten) ---
    extracted_value: Any = None
    extraction_latency_ms: float | None = None
    extraction_error: str | None = None

    # --- group 2: judge output (never overwritten) ---
    judge_verdict: str | None = None
    judge_confidence: float | None = None
    judge_probabilities: str | None = None
    judge_grounded_noul: float | None = None
    judge_correction_value: str | None = None
    judge_correction_confidence: float | None = None
    judge_reason: str | None = None
    reason_source: str = REASON_SOURCE_UNAVAILABLE
    judge_latency_ms: float | None = None
    judge_error: str | None = None

    # --- group 3: final decision ---
    decision: str | None = None
    final_value: Any = None
    final_value_source: str = SOURCE_NONE

    # --- group 4: ground truth (filled only after all API calls) ---
    gt_value: Any = None
    gt_present: bool = False
    gt_multivalued: bool = False

    # scoring
    exact_match_before: bool | None = None
    normalized_match_before: bool | None = None
    exact_match_after: bool | None = None
    normalized_match_after: bool | None = None

    def to_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self)
        for key in ("extracted_value", "final_value", "gt_value"):
            value = row[key]
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, ensure_ascii=False)
        return row
