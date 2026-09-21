"""Self-check: assertions that need no API calls and cost nothing."""

from __future__ import annotations

from .constants import (
    DECISION_ACCEPT,
    DECISION_MISMATCH,
    DECISION_REVIEW,
    FIELD_ABSENT,
    FIELD_BY_NAME,
    MAX_CHOICE_OPTIONS,
    NO_CORRECTION,
    TYPE_CURRENCY,
    TYPE_TEXT,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN,
)
from .credentials import _register_secret, redact
from .dataset import build_ocr_text
from .decision import decide
from .extraction import extract_fields
from .judging import build_verdict_questions, mine_candidates
from .metrics import fmt_metric, judge_metrics, safe_div
from .models import DocumentView, GroundTruth, JudgeFieldResult
from .normalize import normalize_date, normalize_value, normalized_match


def run_self_check() -> int:
    """Assertions that need no API calls and cost nothing."""
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {label}")
        else:
            failures.append(f"{label} {detail}".strip())
            print(f"  FAIL  {label} {detail}")

    print("\n[1] Number normalization (Indonesian thousands separators)")
    cases = [
        ("60.000", "60000"), ("1.00", "1"), ("16,500", "16500"),
        ("Rp. 111,000", "111000"), ("28.182", "28182"), ("2.818", "2818"),
        ("31.000", "31000"), ("9.00", "9"), ("1X", "1"), ("10%", "10"),
    ]
    for raw, want in cases:
        got = normalize_value(raw, TYPE_CURRENCY)
        check(f"normalize({raw!r}) == {want!r}", got == want, f"got {got!r}")

    print("\n[2] Arithmetic consistency proves the separator rule")
    subtotal = float(normalize_value("28.182", TYPE_CURRENCY))
    tax = float(normalize_value("2.818", TYPE_CURRENCY))
    total = float(normalize_value("31.000", TYPE_CURRENCY))
    check("28.182 + 2.818 == 31.000", abs(subtotal + tax - total) < 1e-6,
          f"{subtotal} + {tax} != {total}")

    print("\n[3] Date normalization (implemented; not exercised by the CORD schema)")
    check("25/12/2018 -> 2018-12-25", normalize_date("25/12/2018") == "2018-12-25")
    check("2018-12-25 -> 2018-12-25", normalize_date("2018-12-25") == "2018-12-25")

    print("\n[4] Null / empty handling")
    check("None -> None", normalize_value(None) is None)
    check("'' -> None", normalize_value("") is None)
    check("'-' -> None", normalize_value("-") is None)
    check("null == null matches", normalized_match(None, None, TYPE_TEXT))
    check("null != value", not normalized_match(None, "9.00", TYPE_CURRENCY))

    print("\n[5] Multi-valued ground truth (4 of 1200 CORD slots)")
    check("matches any element",
          normalized_match("60,000", ["1,213,000", "60,000"], TYPE_CURRENCY))
    check("rejects a non-element",
          not normalized_match("7", ["1,213,000", "60,000"], TYPE_CURRENCY))

    print("\n[6] Ground-truth leakage guard")
    sample = {
        "valid_line": [
            {"words": [{"text": "TOTAL", "quad": {"y1": 10, "x1": 5}},
                       {"text": "31.000", "quad": {"y1": 10, "x1": 60}}],
             "category": "total.total_price"},
            {"words": [{"text": "SUBTTL", "quad": {"y1": 5, "x1": 5}}],
             "category": "sub_total.subtotal_price"},
        ],
        "gt_parse": {"total": {"total_price": "31.000"}},
    }
    text, lines = build_ocr_text(sample)
    categories = [line["category"] for line in sample["valid_line"]]
    check("OCR text contains no CORD category label",
          not any(c in text for c in categories), f"text={text!r}")
    check("OCR text is geometrically ordered", lines[0] == "SUBTTL", f"lines={lines}")
    check("OCR text carries the words", "31.000" in text)

    try:
        extract_fields(None, "m", GroundTruth("d", {}), 10, 1)  # type: ignore[arg-type]
        check("extract_fields rejects a GroundTruth object", False, "no TypeError raised")
    except TypeError:
        check("extract_fields rejects a GroundTruth object", True)

    try:
        build_verdict_questions(GroundTruth("d", {}), {})  # type: ignore[arg-type]
        check("judge prompt builder rejects a GroundTruth object", False, "no TypeError raised")
    except TypeError:
        check("judge prompt builder rejects a GroundTruth object", True)
    except ImportError:
        print("  SKIP  judge prompt builder guard (typesafe_sdk not installed)")

    print("\n[7] Degenerate metrics return null, never 0.0")
    check("safe_div(0, 0) is None", safe_div(0, 0) is None)
    check("safe_div(5, 0) is None", safe_div(5, 0) is None)
    check("fmt_metric(None) == 'n/a'", fmt_metric(None) == "n/a")
    empty = judge_metrics([], 0.75)
    check("empty precision is None", empty["precision"] is None)
    check("empty recall is None", empty["recall"] is None)
    check("empty F1 is None", empty["f1"] is None)

    print("\n[8] Decision logic")
    passing = JudgeFieldResult(verdict=VERDICT_PASS, confidence=0.9)
    failing = JudgeFieldResult(verdict=VERDICT_FAIL, confidence=0.9, correction_value="31.000")
    failing_nofix = JudgeFieldResult(verdict=VERDICT_FAIL, confidence=0.9, correction_value=NO_CORRECTION)
    absent = JudgeFieldResult(verdict=VERDICT_FAIL, confidence=0.9, correction_value=FIELD_ABSENT)
    unsure = JudgeFieldResult(verdict=VERDICT_UNCERTAIN, confidence=0.99)
    lowconf = JudgeFieldResult(verdict=VERDICT_FAIL, confidence=0.4)
    check("PASS >= T -> accept", decide("9", passing, 0.75)[0] == DECISION_ACCEPT)
    check("FAIL >= T with correction -> correct", decide("9", failing, 0.75)[1] == "31.000")
    check("FAIL >= T without correction -> mismatch",
          decide("9", failing_nofix, 0.75)[0] == DECISION_MISMATCH)
    check("FAIL -> field absent -> final value None", decide("9", absent, 0.75)[1] is None)
    check("UNCERTAIN at high confidence -> review", decide("9", unsure, 0.75)[0] == DECISION_REVIEW)
    check("below threshold -> review", decide("9", lowconf, 0.75)[0] == DECISION_REVIEW)

    print("\n[9] Candidate mining respects the 255-option Choice limit")
    view = DocumentView("d", 0, "\n".join(f"ITEM {i} {i}.000" for i in range(400)),
                        tuple(f"ITEM {i} {i}.000" for i in range(400)))
    mined = mine_candidates(view, FIELD_BY_NAME["total.total_price"], "1.000")
    check(f"candidates capped at {MAX_CHOICE_OPTIONS}", len(mined) <= MAX_CHOICE_OPTIONS,
          f"got {len(mined)}")
    check("sentinels always present", NO_CORRECTION in mined and FIELD_ABSENT in mined)
    check("candidates are unique", len(mined) == len(set(mined)))

    print("\n[10] Credential redaction")
    _register_secret("nvapi-abcdefghijklmnopqrstuvwxyz1234567890")
    check("key is scrubbed from text",
          "nvapi-abcdefghijklmnopqrstuvwxyz1234567890" not in
          redact("boom: nvapi-abcdefghijklmnopqrstuvwxyz1234567890 failed"))
    check("unregistered key shapes are scrubbed too",
          "REDACTED" in redact("Bearer hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))

    print()
    if failures:
        print(f"SELF-CHECK FAILED — {len(failures)} assertion(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("SELF-CHECK PASSED — all assertions hold.")
    return 0
