"""Judging (TypeSafe System One / Jev)."""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

from .constants import (
    FIELD_ABSENT,
    FIELD_BY_NAME,
    FIELD_NAMES,
    FIELD_SCHEMA,
    MAX_CHOICE_OPTIONS,
    NO_CORRECTION,
    REASON_SOURCE_DERIVED,
    TYPE_CURRENCY,
    TYPE_NUMBER,
    VERDICT_FAIL,
    VERDICT_PASS,
    VERDICT_UNCERTAIN,
    FieldSpec,
)
from .credentials import redact
from .models import DocumentView, ExtractionResult, JudgeDocResult, JudgeFieldResult


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
