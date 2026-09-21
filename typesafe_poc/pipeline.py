"""Per-document pipeline: extraction -> judging -> record building, plus
run-to-run stability measurement."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Sequence

from .constants import FIELD_SCHEMA
from .extraction import extract_fields
from .judging import judge_document
from .models import DocumentView, ExtractionResult, FieldRecord, GroundTruth, JudgeDocResult, JudgeFieldResult
from .normalize import exact_match, normalized_match


def build_records(
    run_id: str,
    view: DocumentView,
    truth: GroundTruth,
    extraction: ExtractionResult,
    judge: JudgeDocResult | None,
) -> list[FieldRecord]:
    records: list[FieldRecord] = []
    for spec in FIELD_SCHEMA:
        extracted = extraction.values.get(spec.name)
        record = FieldRecord(
            run_id=run_id,
            doc_id=view.doc_id,
            doc_index=view.doc_index,
            model=extraction.model,
            field=spec.name,
            field_type=spec.kind,
            extracted_value=extracted,
            extraction_latency_ms=extraction.latency_ms,
            extraction_error=extraction.error,
        )

        if judge is not None:
            entry = judge.fields.get(spec.name, JudgeFieldResult())
            record.judge_verdict = entry.verdict
            record.judge_confidence = entry.confidence
            record.judge_probabilities = (
                json.dumps(entry.probabilities, ensure_ascii=False) if entry.probabilities else None
            )
            record.judge_grounded_noul = entry.grounded_noul
            record.judge_correction_value = entry.correction_value
            record.judge_correction_confidence = entry.correction_confidence
            record.judge_reason = entry.reason
            record.reason_source = entry.reason_source
            record.judge_latency_ms = judge.latency_ms
            record.judge_error = entry.error or judge.error

        # --- ground truth attached ONLY here, after all API calls ---
        gt_value = truth.fields.get(spec.name)
        record.gt_value = gt_value
        record.gt_present = gt_value is not None
        record.gt_multivalued = isinstance(gt_value, list)
        record.exact_match_before = exact_match(extracted, gt_value)
        record.normalized_match_before = normalized_match(extracted, gt_value, spec.kind)

        records.append(record)
    return records


def compute_stability(runs: Sequence[Sequence[FieldRecord]]) -> dict[str, Any] | None:
    if len(runs) < 2:
        return None
    baseline = runs[0]
    stable_extractions = 0
    stable_verdicts = 0
    n_judged = 0
    stdevs: list[float] = []

    for index in range(len(baseline)):
        values = [str(run[index].extracted_value) for run in runs]
        if len(set(values)) == 1:
            stable_extractions += 1

        verdicts = [run[index].judge_verdict for run in runs]
        if all(v is not None for v in verdicts):
            n_judged += 1
            if len(set(verdicts)) == 1:
                stable_verdicts += 1

        confidences = [run[index].judge_confidence for run in runs]
        if all(c is not None for c in confidences) and len(confidences) > 1:
            stdevs.append(statistics.pstdev(confidences))

    return {
        "repeats": len(runs),
        "n_slots": len(baseline),
        "stable_extractions": stable_extractions,
        "n_judged": n_judged,
        "stable_verdicts": stable_verdicts,
        "mean_confidence_stdev": statistics.fmean(stdevs) if stdevs else None,
    }


def process_document(
    args: argparse.Namespace,
    run_id: str,
    nvidia_client: Any,
    judge_client: Any,
    model: str,
    view: DocumentView,
    truth: GroundTruth,
) -> tuple[list[FieldRecord], JudgeDocResult | None]:
    extraction = extract_fields(nvidia_client, model, view, args.max_tokens, args.seed)

    judge: JudgeDocResult | None = None
    if judge_client is not None:
        if extraction.error:
            judge = JudgeDocResult(
                model=args.judge_model,
                fields={
                    spec.name: JudgeFieldResult(error="skipped: extraction failed")
                    for spec in FIELD_SCHEMA
                },
                error="skipped: extraction failed",
            )
        else:
            judge = judge_document(judge_client, args.judge_model, view, extraction, args.correct_on)

    return build_records(run_id, view, truth, extraction, judge), judge
