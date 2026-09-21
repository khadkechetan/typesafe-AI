"""Command-line entry point: argument parsing and top-level orchestration."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_DATASET,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SPLIT,
    DEFAULT_THRESHOLD,
    FIELD_SCHEMA,
    LOW_CONFIDENCE_N,
    POC_VERSION,
)
from .credentials import ALLOWED_CREDENTIAL_VARS, load_credentials, redact
from .dataset import load_documents
from .metrics import (
    accuracy_block,
    accuracy_bounds,
    apply_decisions,
    calibration_bins,
    fmt_metric,
    group_metrics,
    judge_metrics,
    pick_best_threshold,
    threshold_sweep,
)
from .models import FieldRecord
from .pipeline import compute_stability, process_document
from .reporting import build_report, write_csv, write_json
from .selfcheck import run_self_check

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TypeSafe LLM-as-a-Judge POC for document field extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-documents", type=int, default=5,
                        help="documents to process, 1-10 (default: 5)")
    parser.add_argument("--judge-threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"decision threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--models", default=DEFAULT_EXTRACTION_MODEL,
                        help="comma-separated NVIDIA model ids")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat the whole pipeline to measure stability")
    parser.add_argument("--correct-on", choices=("non-pass", "fail", "all"), default="non-pass")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=3000,
                        help="reasoning models count thinking against this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-streaming", action="store_true",
                        help="download the full split instead of streaming")
    parser.add_argument("--skip-judge", action="store_true",
                        help="extraction + scoring only; needs no TypeSafe key")
    parser.add_argument("--check-credentials", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    if args.self_check:
        print("=" * 70)
        print("SELF-CHECK — no API calls, no spend")
        print("=" * 70)
        return run_self_check()

    if args.check_credentials:
        try:
            from dotenv import load_dotenv

            load_dotenv(_project_root() / ".env", override=False)
        except ImportError:
            pass
        print("\nCredential check (presence only — values are never printed):\n")
        for name in ALLOWED_CREDENTIAL_VARS:
            value = os.environ.get(name)
            status = f"present ({len(value)} chars)" if value else "ABSENT"
            optional = " [optional]" if name == "HF_TOKEN" else ""
            print(f"  {name:20s} {status}{optional}")
        env_path = _project_root() / ".env"
        if env_path.exists() and "NVIDIA-KEY" in env_path.read_text(errors="ignore"):
            print("\n  HINT: .env contains `NVIDIA-KEY`, which is not a valid environment")
            print("        variable name. Rename it to `NVIDIA_API_KEY`.")
        print()
        return 0

    if not 1 <= args.max_documents <= 10:
        parser.error("--max-documents must be between 1 and 10 (POC.md scope)")
    if not 0.0 <= args.judge_threshold <= 1.0:
        parser.error("--judge-threshold must be between 0.0 and 1.0")

    creds = load_credentials(require_nvidia=True, require_typesafe=not args.skip_judge)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TypeSafe LLM-as-a-Judge POC v{POC_VERSION}   run {run_id}")
    print("=" * 70)
    print(f"  dataset    : {args.dataset} [{args.split}] "
          f"{args.max_documents} docs from offset {args.offset}")
    print(f"  extraction : {', '.join(models)}")
    print(f"  judge      : {args.judge_model}" + ("  (SKIPPED)" if args.skip_judge else ""))
    print(f"  threshold  : {args.judge_threshold}")
    print(f"  fields/doc : {len(FIELD_SCHEMA)}")
    print()

    # ---- load documents ------------------------------------------------
    print("[1/4] Loading documents ...")
    try:
        documents = load_documents(
            args.dataset, args.split, args.max_documents, args.offset,
            not args.no_streaming, creds["HF_TOKEN"],
        )
    except Exception as exc:
        print(redact(f"\nERROR: failed to load dataset: {type(exc).__name__}: {exc}"))
        traceback.print_exc()
        return 2

    if not documents:
        print("\nERROR: no documents loaded.")
        return 2
    print(f"      loaded {len(documents)} documents "
          f"({sum(len(v.ocr_text) for v, _ in documents) // len(documents)} chars avg OCR)")

    # ---- clients -------------------------------------------------------
    from openai import OpenAI

    nvidia_client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=creds["NVIDIA_API_KEY"],
        timeout=args.timeout,
        max_retries=3,
    )

    judge_client = None
    judge_model_served = None
    if not args.skip_judge:
        from typesafe_sdk import TypeSafeClient

        # The SDK default timeout is 10s, too short for a large question batch.
        judge_client = TypeSafeClient(api_key=creds["TYPESAFE_API_KEY"], timeout=args.timeout)

    # ---- run -----------------------------------------------------------
    all_runs: list[list[FieldRecord]] = []
    judge_usage = {"input_tokens": 0, "output_tokens": 0}

    for repeat in range(args.repeats):
        label = f"[2/4] Processing" + (f" (repeat {repeat + 1}/{args.repeats})" if args.repeats > 1 else "")
        print(f"{label} ...")
        records: list[FieldRecord] = []

        for model in models:
            tasks = []
            with futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                for view, truth in documents:
                    tasks.append(pool.submit(
                        process_document, args, run_id, nvidia_client, judge_client,
                        model, view, truth,
                    ))
                for index, task in enumerate(tasks, 1):
                    doc_records, judge = task.result()
                    records.extend(doc_records)
                    if judge and judge.served_model and not judge_model_served:
                        judge_model_served = judge.served_model
                    if judge and judge.usage:
                        judge_usage["input_tokens"] += judge.usage["input_tokens"]
                        judge_usage["output_tokens"] += judge.usage["output_tokens"]
                    errors = sum(1 for r in doc_records if r.extraction_error or r.judge_error)
                    flag = f"  ({errors} error slots)" if errors else ""
                    print(f"      {model}  doc {index}/{len(documents)}"
                          f"  {doc_records[0].doc_id}{flag}")

        records.sort(key=lambda r: (r.model, r.doc_index, r.field))
        all_runs.append(records)

    records = all_runs[0]
    stability = compute_stability(all_runs)

    # ---- metrics -------------------------------------------------------
    print("[3/4] Computing metrics ...")
    sweep = threshold_sweep(records)
    best = pick_best_threshold(sweep)
    apply_decisions(records, args.judge_threshold)

    config = {
        "run_id": run_id,
        "poc_version": POC_VERSION,
        "dataset": args.dataset,
        "split": args.split,
        "offset": args.offset,
        "n_documents": len(documents),
        "doc_ids": [v.doc_id for v, _ in documents],
        "models": models,
        "judge_model": args.judge_model,
        "judge_model_served": judge_model_served,
        "judge_skipped": args.skip_judge,
        "threshold": args.judge_threshold,
        "correct_on": args.correct_on,
        "repeats": args.repeats,
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": f"python {sys.version.split()[0]}",
        "judge_usage": judge_usage,
    }

    # ---- outputs -------------------------------------------------------
    print("[4/4] Writing results ...")
    csv_path = out_dir / f"fields_{run_id}.csv"
    json_path = out_dir / f"fields_{run_id}.json"
    run_path = out_dir / f"run_{run_id}.json"
    report_path = out_dir / f"report_{run_id}.md"

    write_csv(csv_path, records)
    write_json(json_path, [r.to_row() for r in records])
    write_json(run_path, {
        "config": config,
        "accuracy_before": accuracy_block(records),
        "accuracy_after": accuracy_block(records, after=True),
        "accuracy_bounds": accuracy_bounds(records),
        "judge_metrics": judge_metrics(records, args.judge_threshold),
        "threshold_sweep": sweep,
        "best_threshold": best,
        "calibration": calibration_bins(records),
        "by_field": group_metrics(records, "field"),
        "by_document": group_metrics(records, "doc_id"),
        "by_model": group_metrics(records, "model"),
        "stability": stability,
    })
    report_path.write_text(
        build_report(config, records, args.judge_threshold, sweep, best, stability),
        encoding="utf-8",
    )

    # ---- console summary ----------------------------------------------
    before = accuracy_block(records)
    bounds = accuracy_bounds(records)
    metrics = judge_metrics(records, args.judge_threshold)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  field slots                     : {len(records)}")
    print(f"  accuracy before (normalized)    : {fmt_metric(before['normalized'], before['n_slots'])}")
    print(f"  accuracy after  (automation)    : {fmt_metric(bounds['automation_only'], bounds['n'])}")
    print(f"  genuine extraction errors       : {metrics['total_actual_errors']}")
    print(f"  correct errors detected         : {metrics['correct_errors_detected']}")
    print(f"  correct values wrongly rejected : {metrics['correct_values_incorrectly_rejected']}")
    print(f"  wrong values wrongly accepted   : {metrics['incorrect_values_incorrectly_accepted']}")
    print(f"  routed to manual review         : {metrics['fields_routed_to_manual_review']}"
          f" ({fmt_metric(metrics['review_rate'], metrics['n_judged'])})")
    print(f"  judge precision / recall / F1   : "
          f"{fmt_metric(metrics['precision'])} / {fmt_metric(metrics['recall'])} / "
          f"{fmt_metric(metrics['f1'])}")
    if best:
        print(f"  best threshold on this sample   : {best['threshold']:.2f}")
    if metrics["n_judged"] == 0 and not args.skip_judge:
        print("\n  ⚠ NO field was successfully judged — every TypeSafe call failed.")
        print("    Metrics are n/a by design; no verdict was fabricated. See judge_error.")
    elif metrics["total_actual_errors"] < LOW_CONFIDENCE_N:
        print(f"\n  ⚠ only {metrics['total_actual_errors']} genuine errors among judged "
              "fields — directional evidence only.")
        print("    Re-run with --max-documents 10 for a larger error supply.")

    failed_judge = sum(1 for r in records if r.judge_error)
    failed_extraction = sum(1 for r in records if r.extraction_error)
    if failed_extraction:
        print(f"\n  ⚠ {failed_extraction} slots had extraction errors.")
    if failed_judge:
        print(f"  ⚠ {failed_judge} slots had judge errors (excluded; nothing fabricated).")

    print()
    print(f"  report : {report_path}")
    print(f"  fields : {csv_path}")
    print(f"  raw    : {json_path}")
    print(f"  run    : {run_path}")
    print()
    return 0


def _project_root() -> Path:
    """The repository root (one level above this package)."""
    return Path(__file__).resolve().parent.parent
