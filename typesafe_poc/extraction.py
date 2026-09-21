"""Extraction (NVIDIA, OpenAI-compatible)."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .constants import FIELD_NAMES, FIELD_SCHEMA
from .credentials import redact
from .models import DocumentView, ExtractionResult

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
