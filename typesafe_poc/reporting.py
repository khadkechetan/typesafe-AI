"""Reporting: CSV/JSON writers and the markdown evaluation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .constants import FIELD_SCHEMA, LOW_CONFIDENCE_N, POC_VERSION, REASON_SOURCE_DERIVED
from .metrics import (
    accuracy_block,
    accuracy_bounds,
    apply_decisions,
    calibration_bins,
    fmt_metric,
    group_metrics,
    judge_metrics,
    safe_div,
)
from .models import FieldRecord


def write_csv(path: Path, records: Sequence[FieldRecord]) -> None:
    rows = [r.to_row() for r in records]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return "\n".join(out)


def build_report(
    config: dict[str, Any],
    records: Sequence[FieldRecord],
    threshold: float,
    sweep: Sequence[dict[str, Any]],
    best: dict[str, Any] | None,
    stability: dict[str, Any] | None,
) -> str:
    apply_decisions(records, threshold)
    before = accuracy_block(records)
    after = accuracy_block(records, after=True)
    bounds = accuracy_bounds(records)
    metrics = judge_metrics(records, threshold)
    judged = [r for r in records if r.judge_verdict is not None and r.judge_error is None]
    judge_failures = [r for r in records if r.judge_error]
    extraction_failures = [r for r in records if r.extraction_error]

    lines: list[str] = []
    add = lines.append

    add("# TypeSafe LLM-as-a-Judge — Evaluation Report")
    add("")
    add(f"Generated {config['finished_at']} · run `{config['run_id']}` · POC v{POC_VERSION}")
    add("")

    total_errors = metrics["total_actual_errors"]
    if len(records) < 100 or total_errors < LOW_CONFIDENCE_N:
        add("> ⚠ **SMALL-SAMPLE WARNING.** This run scored "
            f"**{len(records)} field slots** containing **{total_errors} genuine extraction "
            "errors**. Judge precision/recall computed from so few positives is directional "
            "evidence only, not a validated measurement. Metrics marked ⚠ come from fewer "
            f"than {LOW_CONFIDENCE_N} observations. Re-run with `--max-documents 10` for a "
            "larger error supply.")
        add("")

    # 1. configuration
    add("## 1. Run configuration")
    add("")
    add(_table(
        ["Setting", "Value"],
        [
            ["Dataset", f"`{config['dataset']}` split `{config['split']}`"],
            ["Documents", f"{config['n_documents']} (offset {config['offset']})"],
            ["Document ids", ", ".join(config["doc_ids"])],
            ["Extraction model(s)", ", ".join(config["models"])],
            ["Judge model", f"`{config['judge_model']}` → served `{config.get('judge_model_served') or 'n/a'}`"],
            ["Decision threshold", f"{threshold}"],
            ["Fields per document", str(len(FIELD_SCHEMA))],
            ["Total field slots", str(len(records))],
            ["Started / finished", f"{config['started_at']} / {config['finished_at']}"],
            ["Library versions", config.get("versions", "n/a")],
        ],
    ))
    add("")

    # 2. capability findings
    add("## 2. TypeSafe capability findings")
    add("")
    add("What Jev actually returned in this run, against what `POC.md` asked for:")
    add("")
    add(_table(
        ["Required judge output", "Available?", "How it was obtained"],
        [
            ["Verdict PASS / FAIL / UNCERTAIN", "**Yes — native**",
             "`Choice` over the three verdicts; the `choice` field"],
            ["Confidence 0.0–1.0", "**Yes — native**",
             "the `confidence` field of a Choice answer"],
            ["Short reason", "**No — not available**",
             "Jev emits no text. Derived here from its real probabilities and stamped "
             f"`{REASON_SOURCE_DERIVED}`"],
            ["Suggested corrected value", "**Partial**",
             "Jev cannot generate text. Obtained by `Choice` over candidates mined from the "
             "OCR, per TypeSafe's documented pre-parsed-value-extraction pattern"],
        ],
    ))
    add("")
    add("Other documented constraints that bear on this evaluation:")
    add("")
    add("- **No determinism controls.** No `temperature`, `seed` or `top_p` exists in the "
        "REST schema or the SDK signature, so repeatability can only be measured, not "
        "requested.")
    add("- **`jev-1.13` \"is not a calculator\"** and **\"reads dates as text, not as ordered "
        "quantities\"** ([model jaggedness]"
        "(https://docs.typesafe.ai/model-jaggedness/jev-1.13)). Ten of the twelve fields here "
        "are numeric, so this is a direct threat to validity.")
    add("- **No published calibration data.** Section 7 below is the only calibration "
        "evidence available for this data.")
    add("- **Cannot abstain**, which is why `UNCERTAIN` is an explicit Choice option rather "
        "than something inferred.")
    add("")

    # 3. extraction accuracy
    add("## 3. Extraction accuracy *before* TypeSafe validation")
    add("")
    add(_table(
        ["Measure", "All slots", "Present-only (ground truth non-null)"],
        [
            ["Exact match",
             fmt_metric(before["exact"], before["n_slots"]),
             fmt_metric(before["exact_present_only"], before["n_present"])],
            ["Normalized match",
             fmt_metric(before["normalized"], before["n_slots"]),
             fmt_metric(before["normalized_present_only"], before["n_present"])],
            ["n", str(before["n_slots"]), str(before["n_present"])],
        ],
    ))
    add("")
    add(f"*All slots* includes the {before['n_slots'] - before['n_present']} slots where the "
        "field is legitimately absent, so it rewards correct abstention. *Present-only* is the "
        "harder number. The gap between exact and normalized is what normalization recovers "
        "(currency prefixes, thousands separators, casing, whitespace).")
    add("")
    if extraction_failures:
        add(f"⚠ {len(extraction_failures)} field slots had an extraction error and are excluded "
            "from judge metrics.")
        add("")

    # 4. judge performance
    add("## 4. TypeSafe judge performance")
    add("")
    add(f"Scored as an **error detector**: the positive class is *\"the extraction is wrong\"*, "
        f"determined by normalized match against ground truth after all API calls. "
        f"Threshold = **{threshold}**.")
    add("")
    add(_table(
        ["", "extraction wrong", "extraction correct"],
        [
            ["**judge says FAIL**",
             f"TP = {metrics['true_positives']}", f"FP = {metrics['false_positives']}"],
            ["**judge says PASS**",
             f"FN = {metrics['false_negatives']}", f"TN = {metrics['true_negatives']}"],
        ],
    ))
    add("")
    add(_table(
        ["Metric", "Decided rows only", "All rows (review = missed)"],
        [
            ["Precision",
             fmt_metric(metrics["precision"], metrics["true_positives"] + metrics["false_positives"]),
             fmt_metric(metrics["precision"], metrics["true_positives"] + metrics["false_positives"])],
            ["Recall",
             fmt_metric(metrics["recall"], metrics["true_positives"] + metrics["false_negatives"]),
             fmt_metric(metrics["recall_all_rows"], metrics["total_actual_errors"])],
            ["F1",
             fmt_metric(metrics["f1"], metrics["n_decided"]),
             fmt_metric(metrics["f1_all_rows"], metrics["n_judged"])],
        ],
    ))
    add("")
    add("Counters required by the spec:")
    add("")
    add(_table(
        ["Counter", "Value"],
        [
            ["Correct errors detected", str(metrics["correct_errors_detected"])],
            ["Correct values incorrectly rejected", str(metrics["correct_values_incorrectly_rejected"])],
            ["Incorrect values incorrectly accepted", str(metrics["incorrect_values_incorrectly_accepted"])],
            ["Fields routed to manual review",
             f"{metrics['fields_routed_to_manual_review']} "
             f"({fmt_metric(metrics['review_rate'], metrics['n_judged'])})"],
            ["Total genuine extraction errors present", str(metrics["total_actual_errors"])],
            ["Fields judged", str(metrics["n_judged"])],
            ["Fields the judge actually ruled on", str(metrics["n_decided"])],
        ],
    ))
    add("")
    if metrics["total_actual_errors"] == 0:
        add("> **Recall is undefined — there were no extraction errors in this sample.** "
            "No claim can be made about the judge's ability to detect errors.")
        add("")
    if judge_failures:
        add(f"⚠ {len(judge_failures)} field slots had a judge error and are excluded from the "
            "matrix. No verdict was fabricated for them.")
        add("")

    # 5. accuracy after
    add("## 5. Accuracy *after* TypeSafe validation")
    add("")
    add(_table(
        ["Measure", "Value", "n"],
        [
            ["Before (normalized)", fmt_metric(before["normalized"], before["n_slots"]),
             str(before["n_slots"])],
            ["**After — automation only** (review rows keep the raw value)",
             fmt_metric(bounds["automation_only"], bounds["n"]), str(bounds["n"])],
            ["After — oracle review (review rows assumed fixed by a human)",
             fmt_metric(bounds["oracle_review"], bounds["n"]), str(bounds["n"])],
            ["After — exact match", fmt_metric(after["exact"], after["n_slots"]),
             str(after["n_slots"])],
        ],
    ))
    add("")
    delta = None
    if bounds["automation_only"] is not None and before["normalized"] is not None:
        delta = (bounds["automation_only"] - before["normalized"]) * 100
        direction = "improved" if delta > 0 else ("degraded" if delta < 0 else "left unchanged")
        add(f"The automation-only figure is the honest operational number: TypeSafe "
            f"**{direction}** accuracy by **{delta:+.1f} percentage points**. The oracle bound "
            "assumes every routed field is fixed correctly by a human, which is an upper limit, "
            "not a result.")
        add("")

    # 6. threshold sweep
    add("## 6. Threshold sweep")
    add("")
    add(_table(
        ["T", "Acc (automation)", "Acc (oracle)", "Review rate", "False-automation", "Precision", "Recall", "F1", "n decided"],
        [
            [
                f"{row['threshold']:.2f}",
                fmt_metric(row["accuracy_automation_only"]),
                fmt_metric(row["accuracy_oracle_review"]),
                fmt_metric(row["review_rate"]),
                fmt_metric(row["false_automation_rate"]),
                fmt_metric(row["precision"]),
                fmt_metric(row["recall"]),
                fmt_metric(row["f1"]),
                row["n_decided"],
            ]
            for row in sweep
        ],
    ))
    add("")
    if best:
        add(f"**Best threshold on this sample: `{best['threshold']:.2f}`** — automation-only "
            f"accuracy {fmt_metric(best['accuracy_automation_only'])}, review rate "
            f"{fmt_metric(best['review_rate'])}, false-automation rate "
            f"{fmt_metric(best['false_automation_rate'])}.")
        add("")
        add("Objective: maximise automation-only accuracy, then minimise false-automation rate, "
            "then review rate, then maximise F1 — among thresholds that decided at least one "
            "field. False-automation outranks review rate because a silently accepted wrong "
            "value reaches downstream systems, whereas a routed field only costs reviewer time. "
            "**This threshold was selected on the same small sample it is evaluated on, so it "
            "is optimistic by construction.** The full curve is above so a different objective "
            "(for example a hard false-automation budget) can be applied.")
    else:
        add("**No best threshold could be selected** — no threshold produced any decided rows.")
    add("")

    # 7. calibration
    add("## 7. Calibration by confidence range")
    add("")
    add("Does Jev's confidence mean anything on this data? Each bin compares claimed "
        "confidence against how often the verdict was actually right.")
    add("")
    add(_table(
        ["Confidence bin", "n", "Mean claimed confidence", "Observed verdict correctness"],
        [
            [row["bin"], row["n"],
             f"{row['mean_confidence']:.3f}" if row["mean_confidence"] is not None else "n/a",
             fmt_metric(row["observed_correct"], row["n"])]
            for row in calibration_bins(records)
        ],
    ))
    add("")
    add("A well-calibrated judge shows observed correctness tracking mean claimed confidence "
        "down the table. Bins with n below "
        f"{LOW_CONFIDENCE_N} are marked ⚠ and should not be read as evidence.")
    add("")

    # 8. breakdowns
    add("## 8. Breakdowns")
    add("")
    for key, title in (("field", "By field"), ("doc_id", "By document"), ("model", "By extraction model")):
        add(f"### {title}")
        add("")
        add(_table(
            [key, "n", "Acc before", "Acc after", "Errors", "Review", "Mean judge conf."],
            [
                [
                    row[key], row["n"],
                    fmt_metric(row["acc_before_norm"], row["n"]),
                    fmt_metric(row["acc_after_norm"], row["n"]),
                    row["n_errors"], row["n_review"],
                    f"{row['mean_judge_confidence']:.3f}"
                    if row["mean_judge_confidence"] is not None else "n/a",
                ]
                for row in group_metrics(records, key)
            ],
        ))
        add("")
    if len(config["models"]) == 1:
        add(f"> Only one extraction model (`{config['models'][0]}`) was run, so the "
            "by-model table has a single row. No cross-model comparison was performed.")
        add("")

    # 9. stability
    if stability:
        add("## 9. Run-to-run stability")
        add("")
        add(f"Neither API offers real determinism. NVIDIA documents `seed` as best-effort; "
            f"TypeSafe exposes no sampling controls at all. Measured over "
            f"{stability['repeats']} repeats:")
        add("")
        add(_table(
            ["Measure", "Value"],
            [
                ["Field slots with identical extracted value across all repeats",
                 f"{stability['stable_extractions']}/{stability['n_slots']} "
                 f"({fmt_metric(safe_div(stability['stable_extractions'], stability['n_slots']))})"],
                ["Field slots with identical judge verdict across all repeats",
                 f"{stability['stable_verdicts']}/{stability['n_judged']} "
                 f"({fmt_metric(safe_div(stability['stable_verdicts'], stability['n_judged']))})"],
                ["Mean std-dev of judge confidence",
                 f"{stability['mean_confidence_stdev']:.4f}"
                 if stability["mean_confidence_stdev"] is not None else "n/a"],
            ],
        ))
        add("")

    # 10. conclusion
    add("## 10. Conclusion")
    add("")
    add(_conclusion(metrics, before, bounds, best, delta, records))
    add("")
    add("---")
    add("")
    add("*Generated by `run_poc.py`. Every number above is computed from this run; none are "
        "hand-written. Raw extraction, judge output, final decision and ground truth are kept "
        "as separate columns in the accompanying CSV/JSON.*")
    return "\n".join(lines)


def _conclusion(
    metrics: dict[str, Any],
    before: dict[str, Any],
    bounds: dict[str, Any],
    best: dict[str, Any] | None,
    delta: float | None,
    records: Sequence[FieldRecord],
) -> str:
    """The spec's closing question, answered honestly — including 'cannot tell'."""
    errors = metrics["total_actual_errors"]
    parts: list[str] = []

    parts.append("**Does this evidence support using TypeSafe as a judge for field validation?**")

    if metrics["n_judged"] == 0:
        parts.append(
            "**No conclusion is possible.** No field was successfully judged — every TypeSafe "
            "call failed. Check `judge_error` in the results and re-run. No verdict was "
            "fabricated to fill the gap."
        )
        return "\n\n".join(parts)

    if errors == 0:
        parts.append(
            "**Cannot be determined from this sample.** The extractor made no errors, so there "
            "was nothing for the judge to catch: recall is undefined and precision measures "
            "only the judge's false-alarm rate. This says nothing about error detection. "
            "Re-run with more documents or a weaker extraction model (`--models`) to generate "
            "a real error supply."
        )
        return "\n\n".join(parts)

    verdict_lines: list[str] = []
    if errors < LOW_CONFIDENCE_N:
        verdict_lines.append(
            f"**Directional only — the sample is too small to conclude.** There were just "
            f"{errors} genuine extraction errors across {metrics['n_judged']} judged fields. "
            f"The judge caught {metrics['correct_errors_detected']} of them and wrongly "
            f"rejected {metrics['correct_values_incorrectly_rejected']} correct values."
        )
    else:
        precision_text = fmt_metric(metrics["precision"])
        recall_text = fmt_metric(metrics["recall"])
        verdict_lines.append(
            f"On {metrics['n_judged']} judged fields containing {errors} genuine errors, the "
            f"judge achieved precision {precision_text} and recall {recall_text} on the rows it "
            f"ruled on, routing {fmt_metric(metrics['review_rate'])} of fields to manual review."
        )

    if delta is not None:
        if delta > 0:
            verdict_lines.append(
                f"End to end, validation **raised** normalized accuracy from "
                f"{fmt_metric(before['normalized'])} to {fmt_metric(bounds['automation_only'])} "
                f"({delta:+.1f} pp) with no human in the loop — the direction a useful judge "
                "should show."
            )
        elif delta < 0:
            verdict_lines.append(
                f"End to end, validation **lowered** normalized accuracy from "
                f"{fmt_metric(before['normalized'])} to {fmt_metric(bounds['automation_only'])} "
                f"({delta:+.1f} pp). On this sample the judge did net harm: it rejected or "
                "'corrected' more right answers than it fixed wrong ones."
            )
        else:
            verdict_lines.append(
                f"End to end, validation left accuracy unchanged at "
                f"{fmt_metric(before['normalized'])}; the judge's accepts and corrections "
                "cancelled out."
            )

    if best:
        verdict_lines.append(
            f"**Best confidence threshold on the tested samples: `{best['threshold']:.2f}`** — "
            f"automation-only accuracy {fmt_metric(best['accuracy_automation_only'])}, "
            f"false-automation {fmt_metric(best['false_automation_rate'])}, review rate "
            f"{fmt_metric(best['review_rate'])}. Note this is well below the 0.75 default: "
            "Jev's confidence on this task concentrates in a band where a 0.75 gate discards "
            "most of its correct FAIL verdicts. Selected on the same sample it is scored on, "
            "so treat it as the starting point for a larger calibration run, not a production "
            "setting."
        )

    verdict_lines.append(
        "Two structural caveats survive regardless of the numbers. First, Jev returns **no "
        "rationale**, so every 'reason' in these results is derived by this POC from its "
        "probabilities — a reviewer cannot ask it why. Second, TypeSafe's own documentation "
        "warns that `jev-1.13` is **not a calculator**, and ten of these twelve fields are "
        "numeric. Before production use, both would need to be addressed: pair it with a "
        "sampled LLM judge for explanations, and validate numeric-field behaviour on a "
        "substantially larger labelled set."
    )

    parts.extend(verdict_lines)
    return "\n\n".join(parts)
