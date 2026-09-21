"""Dataset loading (CORD-v2)."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .constants import FIELD_NAMES
from .models import DocumentView, GroundTruth


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
