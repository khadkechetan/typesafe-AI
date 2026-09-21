"""Metrics."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from .constants import (
    CONFIDENCE_BINS,
    DECISION_REVIEW,
    FIELD_BY_NAME,
    LOW_CONFIDENCE_N,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN,
)
from .decision import decide
from .models import FieldRecord, JudgeFieldResult
from .normalize import exact_match, normalized_match


def safe_div(numerator: float, denominator: float) -> float | None:
    """Zero denominator yields None, never 0.0."""
    if not denominator:
        return None
    return numerator / denominator


def fmt_metric(value: float | None, denominator: int | None = None, pct: bool = True) -> str:
    if value is None:
        return "n/a"
    text = f"{value * 100:.1f}%" if pct else f"{value:.3f}"
    if denominator is not None and denominator < LOW_CONFIDENCE_N:
        text += " ⚠"
    return text


def accuracy_block(records: Sequence[FieldRecord], after: bool = False) -> dict[str, Any]:
    exact_key = "exact_match_after" if after else "exact_match_before"
    norm_key = "normalized_match_after" if after else "normalized_match_before"

    rows = [r for r in records if getattr(r, norm_key) is not None]
    present = [r for r in rows if r.gt_present]

    return {
        "n_slots": len(rows),
        "n_present": len(present),
        "exact": safe_div(sum(bool(getattr(r, exact_key)) for r in rows), len(rows)),
        "normalized": safe_div(sum(bool(getattr(r, norm_key)) for r in rows), len(rows)),
        "exact_present_only": safe_div(
            sum(bool(getattr(r, exact_key)) for r in present), len(present)
        ),
        "normalized_present_only": safe_div(
            sum(bool(getattr(r, norm_key)) for r in present), len(present)
        ),
    }


def judge_metrics(records: Sequence[FieldRecord], threshold: float) -> dict[str, Any]:
    """
    Score the judge as an error detector.

    Positive class = "the extraction is WRONG" (by normalized match).
    Review rows are NOT a PASS/FAIL prediction and are excluded from the
    matrix; they are reported separately and also folded into an all-rows view.
    """
    judged = [r for r in records if r.judge_verdict is not None and r.judge_error is None]

    tp = fp = fn = tn = 0
    review = 0
    for record in judged:
        wrong = not bool(record.normalized_match_before)
        confidence = record.judge_confidence if record.judge_confidence is not None else 0.0

        if record.judge_verdict == VERDICT_UNCERTAIN or confidence < threshold:
            review += 1
            continue
        if record.judge_verdict == VERDICT_FAIL:
            if wrong:
                tp += 1
            else:
                fp += 1
        elif record.judge_verdict == VERDICT_PASS:
            if wrong:
                fn += 1
            else:
                tn += 1

    decided = tp + fp + fn + tn
    total_wrong = sum(1 for r in judged if not bool(r.normalized_match_before))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if precision and recall else None

    # All-rows view: review counts as a missed detection.
    review_wrong = sum(
        1
        for r in judged
        if not bool(r.normalized_match_before)
        and (
            r.judge_verdict == VERDICT_UNCERTAIN
            or (r.judge_confidence if r.judge_confidence is not None else 0.0) < threshold
        )
    )
    recall_all = safe_div(tp, tp + fn + review_wrong)
    f1_all = (
        safe_div(2 * precision * recall_all, precision + recall_all)
        if precision and recall_all
        else None
    )

    return {
        "n_judged": len(judged),
        "n_decided": decided,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "correct_errors_detected": tp,
        "correct_values_incorrectly_rejected": fp,
        "incorrect_values_incorrectly_accepted": fn,
        "fields_routed_to_manual_review": review,
        "review_rate": safe_div(review, len(judged)),
        "total_actual_errors": total_wrong,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_all_rows": recall_all,
        "f1_all_rows": f1_all,
        "false_automation_rate": safe_div(fn, decided) if decided else None,
    }


def apply_decisions(records: Sequence[FieldRecord], threshold: float) -> None:
    """Recompute decision + after-scores for a given threshold, in place."""
    for record in records:
        judge = JudgeFieldResult(
            verdict=record.judge_verdict,
            confidence=record.judge_confidence,
            correction_value=record.judge_correction_value,
        )
        decision, final, source = decide(record.extracted_value, judge, threshold)
        record.decision = decision
        record.final_value = final
        record.final_value_source = source
        kind = FIELD_BY_NAME[record.field].kind
        record.exact_match_after = exact_match(final, record.gt_value)
        record.normalized_match_after = normalized_match(final, record.gt_value, kind)


def accuracy_bounds(records: Sequence[FieldRecord]) -> dict[str, Any]:
    """Automation-only lower bound and oracle-review upper bound."""
    scored = [r for r in records if r.normalized_match_after is not None]
    if not scored:
        return {"automation_only": None, "oracle_review": None, "n": 0}

    automation = sum(bool(r.normalized_match_after) for r in scored)
    oracle = sum(
        1 if r.decision == DECISION_REVIEW else bool(r.normalized_match_after) for r in scored
    )
    return {
        "automation_only": safe_div(automation, len(scored)),
        "oracle_review": safe_div(oracle, len(scored)),
        "n": len(scored),
    }


def threshold_sweep(records: Sequence[FieldRecord], step: float = 0.05) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    value = 0.0
    while value <= 1.0001:
        threshold = round(value, 2)
        apply_decisions(records, threshold)
        metrics = judge_metrics(records, threshold)
        bounds = accuracy_bounds(records)
        rows.append(
            {
                "threshold": threshold,
                "accuracy_automation_only": bounds["automation_only"],
                "accuracy_oracle_review": bounds["oracle_review"],
                "review_rate": metrics["review_rate"],
                "false_automation_rate": metrics["false_automation_rate"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_decided": metrics["n_decided"],
            }
        )
        value += step
    return rows


def pick_best_threshold(sweep: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Maximise automation-only accuracy, then minimise false-automation rate,
    then review rate, then maximise F1.

    False-automation (a wrong value silently accepted) ranks above review rate
    because it is the failure that reaches downstream systems unnoticed, while
    a routed field only costs reviewer time. This ordering matters in practice:
    on the 5-document run, accuracy ties at 95.0% for every threshold from 0.00
    to 0.60, but false-automation varies from 5.0% down to 1.8% across that
    same range. Ranking review rate first would select the riskiest of the tied
    options.

    Only thresholds that actually decided something are eligible.
    """
    eligible = [
        row
        for row in sweep
        if row["n_decided"] > 0 and row["accuracy_automation_only"] is not None
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda r: (
            -r["accuracy_automation_only"],
            r["false_automation_rate"] if r["false_automation_rate"] is not None else 1.0,
            r["review_rate"] if r["review_rate"] is not None else 1.0,
            -(r["f1"] if r["f1"] is not None else 0.0),
        ),
    )[0]


def calibration_bins(records: Sequence[FieldRecord]) -> list[dict[str, Any]]:
    """Observed judge correctness per claimed-confidence bin."""
    judged = [
        r for r in records
        if r.judge_verdict is not None and r.judge_confidence is not None and r.judge_error is None
    ]
    out: list[dict[str, Any]] = []
    for low, high in zip(CONFIDENCE_BINS, CONFIDENCE_BINS[1:]):
        bucket = [r for r in judged if low <= r.judge_confidence < high]
        correct = 0
        for record in bucket:
            wrong = not bool(record.normalized_match_before)
            said_wrong = record.judge_verdict == VERDICT_FAIL
            said_right = record.judge_verdict == VERDICT_PASS
            if (said_wrong and wrong) or (said_right and not wrong):
                correct += 1
        out.append(
            {
                "bin": f"[{low:.2f}, {min(high, 1.0):.2f})",
                "n": len(bucket),
                "mean_confidence": statistics.fmean([r.judge_confidence for r in bucket])
                if bucket
                else None,
                "observed_correct": safe_div(correct, len(bucket)),
            }
        )
    return out


def group_metrics(records: Sequence[FieldRecord], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[FieldRecord]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(record)

    rows: list[dict[str, Any]] = []
    for name, bucket in sorted(groups.items(), key=lambda kv: str(kv[0])):
        before = accuracy_block(bucket)
        after = accuracy_block(bucket, after=True)
        judged = [r for r in bucket if r.judge_verdict is not None]
        rows.append(
            {
                key: name,
                "n": len(bucket),
                "acc_before_norm": before["normalized"],
                "acc_after_norm": after["normalized"],
                "n_errors": sum(1 for r in bucket if r.normalized_match_before is False),
                "n_review": sum(1 for r in bucket if r.decision == DECISION_REVIEW),
                "mean_judge_confidence": statistics.fmean(
                    [r.judge_confidence for r in judged if r.judge_confidence is not None]
                )
                if any(r.judge_confidence is not None for r in judged)
                else None,
            }
        )
    return rows
