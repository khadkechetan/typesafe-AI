"""
Normalization.

Validated against all 940 numeric ground-truth values in the CORD test split
(930 parse; the remainder are list-valued or literal "-").

CORD uses Indonesian formatting where BOTH "." and "," are thousands
separators: `60.000` is sixty thousand, but `1.00` is one. Disambiguation
rule: if the group after the final separator is exactly 3 digits it is a
thousands separator, otherwise it is a decimal point.
Proof: 28.182 + 2.818 == 31000, matching the printed TOTAL of `31.000`.
"""

from __future__ import annotations

import re
from typing import Any

from .constants import TYPE_CURRENCY, TYPE_NUMBER, TYPE_TEXT

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
