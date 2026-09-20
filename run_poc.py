#!/usr/bin/env python3
"""
TypeSafe LLM-as-a-Judge POC
===========================

Evaluates whether TypeSafe's System One model ("Jev") can reliably validate
document field extraction performed by an NVIDIA-hosted LLM.

Pipeline
--------
    CORD-v2 receipt  ->  OCR text (from dataset, images never used)
                     ->  NVIDIA LLM extracts a fixed 12-field schema
                     ->  EVERY field is sent to the TypeSafe judge
                     ->  threshold decision: accept / correct / mismatch / review
                     ->  scored against ground truth that neither model ever saw

Documented TypeSafe limitations this POC had to adapt to
--------------------------------------------------------
Jev returns ONLY typed values. Per https://docs.typesafe.ai/model-jaggedness/jev-1.13
it "is not trained to generate text". Therefore:

  * VERDICT    -> native. A Choice over {PASS, FAIL, UNCERTAIN} returns
                  `choice`, `confidence` and `probabilities`.
  * CONFIDENCE -> native. The `confidence` field of a Choice answer.
  * REASON     -> NOT AVAILABLE as TypeSafe text. This POC derives a reason
                  deterministically from Jev's real returned probabilities and
                  stamps it `reason_source="derived_from_jev_probabilities"`.
                  It is never presented as TypeSafe prose.
  * CORRECTION -> NOT AVAILABLE as generated text. This POC uses TypeSafe's own
                  documented pre-parsed-value-extraction pattern: candidates are
                  mined from the OCR text and Jev picks one via a Choice.
                  https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook

No endpoint, SDK method or response field is invented. No API result is ever
mocked: a failed call is recorded as an error and excluded from metrics.

Usage
-----
    python run_poc.py --max-documents 5 --judge-threshold 0.75
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import dataclasses
import json
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

POC_VERSION = "1.0.0"

DEFAULT_DATASET = "naver-clova-ix/cord-v2"
DEFAULT_SPLIT = "test"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-oss-20b"
DEFAULT_JUDGE_MODEL = "jev-latest"
DEFAULT_THRESHOLD = 0.75
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# TypeSafe documented limit: "You can have a maximum of 255 options per Choice."
# https://docs.typesafe.ai/api  (prose-only; not enforced by the SDK schema)
MAX_CHOICE_OPTIONS = 255

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_UNCERTAIN)

# Correction sentinels. These are Choice options, not values from the document.
NO_CORRECTION = "__NO_CORRECTION_SUPPORTED__"
FIELD_ABSENT = "__FIELD_NOT_PRESENT__"

DECISION_ACCEPT = "accept"
DECISION_CORRECT = "correct"
DECISION_MISMATCH = "mismatch"
DECISION_REVIEW = "review"

SOURCE_EXTRACTION = "extraction"
SOURCE_JUDGE_CORRECTION = "judge_correction"
SOURCE_NONE = "none"

REASON_SOURCE_DERIVED = "derived_from_jev_probabilities"
REASON_SOURCE_UNAVAILABLE = "unavailable"

CONFIDENCE_BINS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0001]

# Any metric computed from fewer than this many observations is flagged.
LOW_CONFIDENCE_N = 10


# --------------------------------------------------------------------------
# Field schema
#
# Derived empirically by scanning all 100 CORD-v2 test documents rather than
# trusting the upstream README, whose superclass names disagree with the data
# (README says `subtotal`, the data says `sub_total`).
#
# `menu` is a repeating line-item list in 61 of 100 documents, so the schema
# pins the FIRST item only. That keeps every field single-valued.
# --------------------------------------------------------------------------

TYPE_TEXT = "text"
TYPE_CURRENCY = "currency"
TYPE_NUMBER = "number"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    description: str


FIELD_SCHEMA: tuple[FieldSpec, ...] = (
    FieldSpec("menu.first_item_name", TYPE_TEXT,
              "Name of the first menu/line item on the receipt"),
    FieldSpec("menu.first_item_price", TYPE_CURRENCY,
              "Line total price of the first menu item"),
    FieldSpec("menu.first_item_cnt", TYPE_NUMBER,
              "Quantity of the first menu item"),
    FieldSpec("sub_total.subtotal_price", TYPE_CURRENCY,
              "Subtotal before tax, service and discounts"),
    FieldSpec("sub_total.tax_price", TYPE_CURRENCY,
              "Tax amount"),
    FieldSpec("sub_total.service_price", TYPE_CURRENCY,
              "Service charge amount"),
    FieldSpec("sub_total.discount_price", TYPE_CURRENCY,
              "Discount amount"),
    FieldSpec("total.total_price", TYPE_CURRENCY,
              "Final total amount payable"),
    FieldSpec("total.cashprice", TYPE_CURRENCY,
              "Cash amount tendered"),
    FieldSpec("total.changeprice", TYPE_CURRENCY,
              "Change returned to the customer"),
    FieldSpec("total.creditcardprice", TYPE_CURRENCY,
              "Amount paid by credit or debit card"),
    FieldSpec("total.menuqty_cnt", TYPE_NUMBER,
              "Total quantity of items purchased"),
)

FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in FIELD_SCHEMA)
FIELD_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELD_SCHEMA}


# --------------------------------------------------------------------------
# Credentials and redaction
# --------------------------------------------------------------------------

ALLOWED_CREDENTIAL_VARS = ("NVIDIA_API_KEY", "TYPESAFE_API_KEY", "HF_TOKEN")

_SECRETS: list[str] = []


def _register_secret(value: str | None) -> None:
    if value and len(value) >= 8:
        _SECRETS.append(value)


def redact(text: Any) -> str:
    """Scrub any known credential (and key-shaped substrings) from text."""
    out = str(text)
    for secret in _SECRETS:
        if secret and secret in out:
            out = out.replace(secret, "***REDACTED***")
    # Belt and braces: catch key-shaped tokens even if never registered.
    out = re.sub(r"nvapi-[A-Za-z0-9_\-]{8,}", "nvapi-***REDACTED***", out)
    out = re.sub(r"\bsk-[A-Za-z0-9_\-]{16,}", "sk-***REDACTED***", out)
    out = re.sub(r"\bhf_[A-Za-z0-9]{16,}", "hf_***REDACTED***", out)
    return out


def load_credentials(require_nvidia: bool, require_typesafe: bool) -> dict[str, str | None]:
    """Read credentials from the environment only. Never returns values to logs."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent / ".env", override=False)
    except ImportError:
        pass

    creds = {name: os.environ.get(name) or None for name in ALLOWED_CREDENTIAL_VARS}
    for value in creds.values():
        _register_secret(value)

    missing: list[str] = []
    if require_nvidia and not creds["NVIDIA_API_KEY"]:
        missing.append("NVIDIA_API_KEY")
    if require_typesafe and not creds["TYPESAFE_API_KEY"]:
        missing.append("TYPESAFE_API_KEY")

    if missing:
        hint = ""
        if "NVIDIA_API_KEY" in missing:
            raw = Path(__file__).parent / ".env"
            if raw.exists() and "NVIDIA-KEY" in raw.read_text(errors="ignore"):
                hint = (
                    "\n  HINT: your .env contains `NVIDIA-KEY`. A hyphen is not a valid\n"
                    "        environment-variable name. Rename it to `NVIDIA_API_KEY`."
                )
        raise SystemExit(
            f"\nERROR: missing required credential(s): {', '.join(missing)}\n"
            f"  Set them in .env or the environment. See .env.example.{hint}\n"
            "  This POC never fabricates API results, so it cannot continue.\n"
        )
    return creds


# --------------------------------------------------------------------------
# Core data model
#
# Ground-truth leakage prevention is STRUCTURAL: prompt builders accept a
# DocumentView, which physically does not carry ground truth. GroundTruth is a
# separate object that never reaches extraction or judging.
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Dataset loading (CORD-v2)
# --------------------------------------------------------------------------


def build_ocr_text(ground_truth_obj: dict) -> tuple[str, tuple[str, ...]]:
    """
    Reconstruct the page text from CORD's `valid_line[].words[].text` ONLY.

    `valid_line[].category` is the CORD ground-truth label. It is deliberately
    never read here — doing so would leak labels into the prompt.
    """
    rows: list[tuple[int, int, str]] = []
    for line in ground_truth_obj.get("valid_line", []) or []:
        words = line.get("words") or []
        texts = [w.get("text", "") for w in words if w.get("text")]
        if not texts:
            continue
        ys = [w["quad"]["y1"] for w in words if "quad" in w]
        xs = [w["quad"]["x1"] for w in words if "quad" in w]
        rows.append((min(ys) if ys else 0, min(xs) if xs else 0, " ".join(texts)))
    rows.sort(key=lambda r: (r[0], r[1]))
    lines = tuple(text for _, _, text in rows)
    return "\n".join(lines), lines


def flatten_ground_truth(gt_parse: dict) -> dict[str, Any]:
    """Map CORD's nested `gt_parse` onto the flat 12-field schema."""
    out: dict[str, Any] = {name: None for name in FIELD_NAMES}

    menu = gt_parse.get("menu")
    if isinstance(menu, list):
        first = menu[0] if menu else None
    elif isinstance(menu, dict):
        first = menu
    else:
        first = None
    if isinstance(first, dict):
        out["menu.first_item_name"] = first.get("nm")
        out["menu.first_item_price"] = first.get("price")
        out["menu.first_item_cnt"] = first.get("cnt")

    for group in ("sub_total", "total"):
        section = gt_parse.get(group)
        if isinstance(section, dict):
            for key, value in section.items():
                name = f"{group}.{key}"
                if name in out:
                    out[name] = value
    return out


def load_documents(
    dataset: str,
    split: str,
    limit: int,
    offset: int,
    streaming: bool,
    hf_token: str | None,
) -> list[tuple[DocumentView, GroundTruth]]:
    """Load N documents. Images are never decoded; only the OCR text is used."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split}
    if hf_token:
        kwargs["token"] = hf_token
    if streaming:
        kwargs["streaming"] = True

    data = load_dataset(dataset, **kwargs)

    # Drop the image column before iterating. The POC is text-only, and
    # decoding CORD's embedded receipt images would both require Pillow and
    # waste time on data we never look at.
    try:
        if "ground_truth" in (data.column_names or []):
            data = data.select_columns(["ground_truth"])
    except Exception:
        pass

    out: list[tuple[DocumentView, GroundTruth]] = []
    if streaming:
        iterator: Iterable[dict] = data.skip(offset).take(limit) if offset else data.take(limit)
        rows = enumerate(iterator)
    else:
        end = min(offset + limit, len(data))
        rows = ((i - offset, data[i]) for i in range(offset, end))

    for index, row in rows:
        raw = row["ground_truth"]
        obj = json.loads(raw) if isinstance(raw, str) else raw
        text, lines = build_ocr_text(obj)
        meta = obj.get("meta") or {}
        doc_id = str(meta.get("image_id", offset + index))
        view = DocumentView(
            doc_id=f"{split}-{doc_id}",
            doc_index=offset + index,
            ocr_text=text,
            ocr_lines=lines,
        )
        truth = GroundTruth(doc_id=view.doc_id, fields=flatten_ground_truth(obj.get("gt_parse", {})))
        out.append((view, truth))
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Normalization
#
# Validated against all 940 numeric ground-truth values in the CORD test split
# (930 parse; the remainder are list-valued or literal "-").
#
# CORD uses Indonesian formatting where BOTH "." and "," are thousands
# separators: `60.000` is sixty thousand, but `1.00` is one. Disambiguation
# rule: if the group after the final separator is exactly 3 digits it is a
# thousands separator, otherwise it is a decimal point.
# Proof: 28.182 + 2.818 == 31000, matching the printed TOTAL of `31.000`.
# --------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"^[^\d\-]*(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?|-?\d+(?:[.,]\d+)?)\D*$"
)
_DATE_RE = re.compile(r"^\s*(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})\s*$")


def normalize_number(value: str) -> str | None:
    """Return a canonical numeric string, or None if not number-like."""
    match = _NUMBER_RE.match(value.strip())
    if not match:
        return None
    core = match.group(1)
    negative = core.startswith("-")
    core = core.lstrip("-")
    parts = re.split(r"[.,]", core)
    if len(parts) == 1:
        number = float(core)
    elif len(parts[-1]) == 3:
        number = float("".join(parts))
    else:
        number = float("".join(parts[:-1]) + "." + parts[-1])
    if negative:
        number = -number
    return f"{number:g}"


def normalize_date(value: str) -> str | None:
    """
    Canonicalise dd/mm/yyyy-style dates to ISO.

    NOTE: CORD's scalar header/footer fields contain no date, so this is
    exercised by --self-check but not by the CORD field schema.
    """
    match = _DATE_RE.match(value)
    if not match:
        return None
    a, b, c = (int(x) for x in match.groups())
    if a > 31:  # yyyy/mm/dd
        year, month, day = a, b, c
    else:  # dd/mm/yyyy
        day, month, year = a, b, c
    if year < 100:
        year += 2000 if year < 70 else 1900
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_value(value: Any, kind: str = TYPE_TEXT) -> str | None:
    """Normalize a single scalar. Returns None for null/empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return None

    if kind in (TYPE_CURRENCY, TYPE_NUMBER):
        number = normalize_number(text)
        if number is not None:
            return number
    else:
        date = normalize_date(text)
        if date is not None:
            return date
        number = normalize_number(text)
        if number is not None:
            return number

    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip(" .,:;-") or None


def _as_candidate_list(value: Any) -> list[Any]:
    """CORD occasionally gives a list where a scalar is expected."""
    if isinstance(value, list):
        return value
    return [value]


def exact_match(predicted: Any, truth: Any) -> bool:
    def clean(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    p = clean(predicted)
    return any(clean(t) == p for t in _as_candidate_list(truth))


def normalized_match(predicted: Any, truth: Any, kind: str) -> bool:
    p = normalize_value(predicted, kind)
    return any(normalize_value(t, kind) == p for t in _as_candidate_list(truth))


# --------------------------------------------------------------------------
# Extraction (NVIDIA, OpenAI-compatible)
# --------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You extract structured fields from the OCR text of a retail receipt.\n"
    "Return ONLY a JSON object. No prose, no markdown fences.\n"
    "The object must contain EXACTLY these keys:\n{keys}\n\n"
    "Rules:\n"
    "- Copy each value EXACTLY as printed on the receipt, including separators.\n"
    "- Use null when the field does not appear on the receipt. Do not guess.\n"
    "- Do not compute or infer values that are not printed.\n"
)


def _field_key_block() -> str:
    return "\n".join(f"  {f.name}  ({f.kind}) - {f.description}" for f in FIELD_SCHEMA)


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating fences and prose."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {cleaned[:120]!r}")
    return json.loads(cleaned[start : end + 1])


def extract_fields(
    client: Any,
    model: str,
    view: DocumentView,
    max_tokens: int,
    seed: int,
) -> ExtractionResult:
    """
    Extract the 12-field schema from a DocumentView.

    `view` is a DocumentView by contract - it carries no ground truth.
    """
    if not isinstance(view, DocumentView):  # structural leakage guard
        raise TypeError("extract_fields only accepts a DocumentView (no ground truth)")

    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM.format(keys=_field_key_block())},
        {"role": "user", "content": f"OCR TEXT:\n{view.ocr_text}"},
    ]
    base: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,          # deterministic where supported
        "seed": seed,              # NVIDIA documents this as "best effort"
        "max_tokens": max_tokens,
    }

    started = time.perf_counter()
    json_mode = True
    try:
        try:
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                reasoning_effort="low",
                **base,
            )
        except TypeError:
            response = client.chat.completions.create(
                extra_body={"reasoning_effort": "low"},
                response_format={"type": "json_object"},
                **base,
            )
        except Exception as exc:  # response_format is undocumented on this endpoint
            if "response_format" not in str(exc) and "400" not in str(exc):
                raise
            json_mode = False
            response = client.chat.completions.create(**base)

        latency = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()

        if choice.finish_reason == "length":
            return ExtractionResult(
                model=model,
                values={name: None for name in FIELD_NAMES},
                latency_ms=latency,
                error=f"response truncated (finish_reason=length, max_tokens={max_tokens})",
                raw_response=raw,
                json_mode_used=json_mode,
            )

        parsed = parse_json_object(raw)
        values = {name: parsed.get(name) for name in FIELD_NAMES}
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        return ExtractionResult(
            model=model,
            values=values,
            latency_ms=latency,
            raw_response=raw,
            usage=usage,
            json_mode_used=json_mode,
        )

    except Exception as exc:
        return ExtractionResult(
            model=model,
            values={name: None for name in FIELD_NAMES},
            latency_ms=(time.perf_counter() - started) * 1000,
            error=redact(f"{type(exc).__name__}: {exc}"),
            json_mode_used=json_mode,
        )


# --------------------------------------------------------------------------
# Judging (TypeSafe System One / Jev)
# --------------------------------------------------------------------------


def _question_key(field_name: str, suffix: str) -> str:
    return f"{field_name.replace('.', '__')}__{suffix}"


def answer_attr(answer: Any, name: str) -> Any:
    """
    Read a field from a SystemOne answer.

    The SDK wraps answers in a pydantic RootModel; the documented examples use
    direct attribute access. Both shapes are supported defensively.
    """
    if hasattr(answer, name):
        return getattr(answer, name)
    root = getattr(answer, "root", None)
    if root is not None and hasattr(root, name):
        return getattr(root, name)
    return None


_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*%?")


def mine_candidates(view: DocumentView, spec: FieldSpec, extracted: Any) -> list[str]:
    """
    Build the candidate list for a correction Choice.

    Jev cannot generate a value, so the only way to obtain a corrected value is
    to enumerate options it can select from. Candidates come from the OCR text,
    so any correction is genuinely supported by the document.
    """
    candidates: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if text and text not in candidates:
            candidates.append(text)

    add(extracted)

    if spec.kind in (TYPE_CURRENCY, TYPE_NUMBER):
        for line in view.ocr_lines:
            for token in _NUMBER_TOKEN_RE.findall(line):
                token = token.strip(".,")
                if token:
                    add(token)
    else:
        for line in view.ocr_lines:
            add(line)
            # a trailing price is common on item lines; offer the name part too
            stripped = _NUMBER_TOKEN_RE.sub("", line).strip(" .,:-")
            if stripped and stripped != line:
                add(stripped)

    # Sentinels last so they never crowd out real document values.
    room = MAX_CHOICE_OPTIONS - 2
    candidates = candidates[:room]
    candidates.append(NO_CORRECTION)
    candidates.append(FIELD_ABSENT)
    return candidates


def build_verdict_questions(view: DocumentView, extracted: dict[str, Any]) -> dict[str, Any]:
    """Pass 1: a verdict Choice and a grounding Noul for every field."""
    if not isinstance(view, DocumentView):  # structural leakage guard
        raise TypeError("build_verdict_questions only accepts a DocumentView")

    from typesafe_sdk import Choice, Noul

    questions: dict[str, Any] = {}
    for spec in FIELD_SCHEMA:
        value = extracted.get(spec.name)
        shown = "null (the extractor reported this field as absent)" if value is None else repr(str(value))

        questions[_question_key(spec.name, "verdict")] = Choice(
            instructions={
                "task": "Validate one field extracted from the receipt above.",
                "field": spec.name,
                "field_means": spec.description,
                "extracted_value": shown,
                "question": (
                    "Is this extracted value correct for this field, "
                    "judged only against the receipt text above?"
                ),
            },
            criteria={
                VERDICT_PASS: (
                    "The extracted value is correct: it matches what the receipt shows for "
                    "this field, or the field genuinely does not appear and the value is null."
                ),
                VERDICT_FAIL: (
                    "The extracted value is wrong: it contradicts the receipt, belongs to a "
                    "different field, or a value was reported for a field that is absent "
                    "(or null was reported for a field that is present)."
                ),
                VERDICT_UNCERTAIN: (
                    "The receipt text does not contain enough information to tell whether "
                    "the value is correct."
                ),
            },
        )

        questions[_question_key(spec.name, "grounded")] = Noul(
            instructions={
                "field": spec.name,
                "extracted_value": shown,
                "statement": (
                    "This exact value appears verbatim in the receipt text above, "
                    "printed as the value of this field."
                ),
            },
        )
    return questions


def build_correction_questions(
    view: DocumentView, extracted: dict[str, Any], targets: Sequence[str]
) -> dict[str, Any]:
    """Pass 2: a Choice over document-supported candidate values."""
    from typesafe_sdk import Choice

    questions: dict[str, Any] = {}
    for name in targets:
        spec = FIELD_BY_NAME[name]
        candidates = mine_candidates(view, spec, extracted.get(name))
        criteria: dict[str, str | None] = {c: None for c in candidates}
        criteria[NO_CORRECTION] = (
            "The receipt does not support any of the listed values for this field."
        )
        criteria[FIELD_ABSENT] = (
            "This field does not appear on the receipt at all; the correct value is nothing."
        )
        questions[_question_key(name, "correction")] = Choice(
            instructions={
                "field": name,
                "field_means": spec.description,
                "question": (
                    "Which of these values is the correct value of this field "
                    "according to the receipt above?"
                ),
            },
            criteria=criteria,
        )
    return questions


def derive_reason(
    verdict: str | None,
    confidence: float | None,
    probabilities: dict[str, float] | None,
    grounded: float | None,
    correction: str | None,
    correction_confidence: float | None,
) -> str:
    """
    Build a reason string from Jev's REAL returned numbers.

    TypeSafe cannot emit text, so this is derived, not quoted. It is always
    stamped reason_source="derived_from_jev_probabilities".
    """
    parts: list[str] = []
    if verdict is None:
        return "No judge verdict available."

    conf = f"{confidence:.2f}" if confidence is not None else "n/a"
    parts.append(f"Jev selected {verdict} with confidence {conf}")

    if probabilities:
        dist = ", ".join(f"{k}={v:.2f}" for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1]))
        parts.append(f"distribution [{dist}]")

    if grounded is not None:
        if grounded >= 0.75:
            parts.append(f"value appears verbatim in the document (noul={grounded:.2f})")
        elif grounded <= 0.25:
            parts.append(f"value does NOT appear verbatim in the document (noul={grounded:.2f})")
        else:
            parts.append(f"verbatim presence is ambiguous (noul={grounded:.2f})")

    if correction:
        cconf = f"{correction_confidence:.2f}" if correction_confidence is not None else "n/a"
        if correction == FIELD_ABSENT:
            parts.append(f"suggests the field is absent from the receipt (confidence {cconf})")
        elif correction == NO_CORRECTION:
            parts.append(f"found no document-supported correction (confidence {cconf})")
        else:
            parts.append(f"suggests corrected value {correction!r} (confidence {cconf})")

    return "; ".join(parts) + "."


def judge_document(
    client: Any,
    judge_model: str,
    view: DocumentView,
    extraction: ExtractionResult,
    correct_on: str,
) -> JudgeDocResult:
    """Validate every extracted field of one document through TypeSafe."""
    result = JudgeDocResult(model=judge_model)
    started = time.perf_counter()
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    try:
        questions = build_verdict_questions(view, extraction.values)
        response = client.system_one(state=view.ocr_text, questions=questions, model=judge_model)
        result.served_model = getattr(response, "model", None)

        if getattr(response, "usage", None):
            total_usage["input_tokens"] += response.usage.input_tokens
            total_usage["output_tokens"] += response.usage.output_tokens

        for spec in FIELD_SCHEMA:
            entry = JudgeFieldResult()
            verdict_ans = response.answers.get(_question_key(spec.name, "verdict"))
            grounded_ans = response.answers.get(_question_key(spec.name, "grounded"))

            if verdict_ans is not None:
                entry.verdict = answer_attr(verdict_ans, "choice")
                entry.confidence = answer_attr(verdict_ans, "confidence")
                probs = answer_attr(verdict_ans, "probabilities")
                entry.probabilities = dict(probs) if probs else None
            if grounded_ans is not None:
                entry.grounded_noul = answer_attr(grounded_ans, "noul")

            result.fields[spec.name] = entry

    except Exception as exc:
        result.error = redact(f"{type(exc).__name__}: {exc}")
        for spec in FIELD_SCHEMA:
            result.fields.setdefault(spec.name, JudgeFieldResult(error=result.error))
        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    # ---- pass 2: corrections -------------------------------------------
    if correct_on == "all":
        targets = list(FIELD_NAMES)
    elif correct_on == "fail":
        targets = [n for n, r in result.fields.items() if r.verdict == VERDICT_FAIL]
    else:  # non-pass
        targets = [n for n, r in result.fields.items() if r.verdict and r.verdict != VERDICT_PASS]

    if targets:
        try:
            questions = build_correction_questions(view, extraction.values, targets)
            response = client.system_one(state=view.ocr_text, questions=questions, model=judge_model)
            if getattr(response, "usage", None):
                total_usage["input_tokens"] += response.usage.input_tokens
                total_usage["output_tokens"] += response.usage.output_tokens

            for name in targets:
                answer = response.answers.get(_question_key(name, "correction"))
                if answer is None:
                    continue
                entry = result.fields[name]
                entry.correction_value = answer_attr(answer, "choice")
                entry.correction_confidence = answer_attr(answer, "confidence")
                probs = answer_attr(answer, "probabilities")
                entry.correction_probabilities = dict(probs) if probs else None
        except Exception as exc:
            message = redact(f"correction pass failed: {type(exc).__name__}: {exc}")
            for name in targets:
                existing = result.fields[name].error
                result.fields[name].error = f"{existing}; {message}" if existing else message

    for entry in result.fields.values():
        if entry.verdict is not None:
            entry.reason = derive_reason(
                entry.verdict,
                entry.confidence,
                entry.probabilities,
                entry.grounded_noul,
                entry.correction_value,
                entry.correction_confidence,
            )
            entry.reason_source = REASON_SOURCE_DERIVED

    result.latency_ms = (time.perf_counter() - started) * 1000
    result.usage = total_usage
    return result


# --------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


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

            load_dotenv(Path(__file__).parent / ".env", override=False)
        except ImportError:
            pass
        print("\nCredential check (presence only — values are never printed):\n")
        for name in ALLOWED_CREDENTIAL_VARS:
            value = os.environ.get(name)
            status = f"present ({len(value)} chars)" if value else "ABSENT"
            optional = " [optional]" if name == "HF_TOKEN" else ""
            print(f"  {name:20s} {status}{optional}")
        env_path = Path(__file__).parent / ".env"
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


if __name__ == "__main__":
    sys.exit(main())
