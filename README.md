# TypeSafe LLM-as-a-Judge POC

A minimal, runnable evaluation of whether **TypeSafe's System One model ("Jev") can reliably
validate LLM document-field extraction**.

```
CORD-v2 receipt  →  OCR text (from the dataset; images are never used)
                 →  NVIDIA-hosted LLM extracts a fixed 12-field schema
                 →  EVERY extracted field is validated by the TypeSafe judge
                 →  threshold decision: accept / correct / mismatch / review
                 →  scored against ground truth that neither model ever saw
```

---

## Quick start

Requires **Python 3.11+**.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then fill in your keys

python run_poc.py --self-check          # 42 assertions, no API calls, no spend
python run_poc.py --check-credentials   # reports presence only, never values

python run_poc.py --max-documents 5 --judge-threshold 0.75
```

Outputs land in `results/`:

| File | Contents |
|---|---|
| `report_<run_id>.md` | the evaluation report — every number computed from the run |
| `fields_<run_id>.csv` | one row per (document, model, field) |
| `fields_<run_id>.json` | the same rows as JSON |
| `run_<run_id>.json` | config, all metrics, threshold sweep, calibration |

> **Use `--max-documents 10`.** At 5 documents the extractor makes only ~4 mistakes, and
> recall computed from 4 positives is noise. 10 documents yields ~18 errors, which is thin but
> actually measurable. The report flags this itself.

---

## Credentials

Read **only** from the environment, never logged, never written to results, and scrubbed from
any error text:

| Variable | Required | Notes |
|---|---|---|
| `NVIDIA_API_KEY` | yes | https://build.nvidia.com |
| `TYPESAFE_API_KEY` | yes (unless `--skip-judge`) | https://console.typesafe.ai/keys |
| `HF_TOKEN` | no | CORD-v2 is public and ungated |

Environment variable names cannot contain hyphens. If your `.env` uses `NVIDIA-KEY` or
similar, rename it — `--check-credentials` will tell you.

---

## CLI

| Flag | Default | Purpose |
|---|---|---|
| `--max-documents` | `5` | documents to process, 1–10 |
| `--judge-threshold` | `0.75` | decision threshold `T` |
| `--dataset` / `--split` / `--offset` | `naver-clova-ix/cord-v2` / `test` / `0` | deterministic document window |
| `--models` | `openai/gpt-oss-20b` | comma-separated; enables per-model metrics |
| `--judge-model` | `jev-latest` | TypeSafe model |
| `--repeats` | `1` | >1 measures run-to-run stability |
| `--correct-on` | `non-pass` | when to ask for a correction: `non-pass`, `fail`, `all` |
| `--concurrency` | `4` | thread pool across documents |
| `--skip-judge` | off | extraction + scoring only; needs no TypeSafe key |
| `--no-streaming` | off | download the full 234 MB split instead of streaming |
| `--self-check` / `--check-credentials` | off | validate and exit |
| `--out-dir` | `results` | output location |

---

## TypeSafe capability limitations

**This is the most important section.** `POC.md` asked the judge for four things per field.
Jev can natively provide two of them. The other two required documented adaptations, because
Jev **emits no text at all** — its own docs state that `jev-1.13`
["is not trained to generate text"](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

| Required output | Status | How this POC obtains it |
|---|---|---|
| Verdict `PASS`/`FAIL`/`UNCERTAIN` | ✅ **native** | a `Choice` over the three verdicts → the `choice` field |
| Confidence `0.0`–`1.0` | ✅ **native** | the `confidence` field of a `Choice` answer |
| Short reason | ❌ **not available** | **derived** from Jev's real probability distribution and stamped `reason_source="derived_from_jev_probabilities"`. It is never presented as TypeSafe prose. |
| Suggested corrected value | ⚠️ **partial** | Jev cannot generate a value, so candidates are mined from the OCR text and Jev picks one via a `Choice` — TypeSafe's own [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook) pattern. Any correction is therefore genuinely supported by the document. |

Other documented constraints that shape the evaluation:

- **No determinism controls.** No `temperature`, `seed` or `top_p` exists in the REST schema or
  the SDK signature. Repeatability can only be *measured* (`--repeats N`), not requested.
  NVIDIA's `seed` is documented as best-effort, and in testing two identical 5-document runs
  produced 3 then 4 errors — so neither side is bit-deterministic.
- **`jev-1.13` "is not a calculator"** and **"reads dates as text, not as ordered quantities"**.
  Ten of the twelve fields here are numeric — a direct threat to validity, flagged in the report.
- **Cannot abstain.** `UNCERTAIN` is an explicit `Choice` option rather than something inferred.
- **`Choice` is capped at 255 options** (documented in prose but *not* enforced by the SDK's
  pydantic schema, so `run_poc.py` enforces it when mining candidates).
- **Text input only**, 32k state limit. Not a constraint here — CORD receipts average ~110
  characters of OCR text.

### Verified API surface

Confirmed against the live API and the actual `typesafe-sdk` 0.7.0 wheel, not from memory:

```
POST https://api.typesafe.ai/v1/systemone     Authorization: Bearer <key>

NoulAnswer    -> noul                                        (NO confidence field)
ChoiceAnswer  -> choice, confidence, probabilities
ScoreAnswer   -> score, confidence, legend, probabilities
```

No endpoint, SDK method or response field is invented anywhere in this POC.

---

## How it works

### Ground-truth leakage prevention is structural

Two dataclasses, and only one can reach a prompt:

```python
DocumentView   # doc_id, ocr_text, ocr_lines   <- the ONLY thing prompt builders accept
GroundTruth    # labels                        <- never passed to extraction or judging
```

`build_ocr_text()` joins **only** `valid_line[].words[].text`, sorted geometrically. It never
reads `valid_line[].category` — that key *is* the CORD label. Both prompt builders raise
`TypeError` on anything that isn't a `DocumentView`, and `--self-check` asserts it.

Ground truth is attached to a record only *after* every API call has returned.

### The 12-field schema

Derived by scanning all 100 CORD test documents rather than trusting the upstream README
(whose class names disagree with the data — it says `subtotal`, the data says `sub_total`).
Every document is asked for all 12 fields regardless of what it contains, so the POC measures
both extraction **and** correct abstention: **56% of slots are non-null, 44% are legitimately
empty**.

`menu.*` is a repeating line-item list in 61 of 100 documents, so the schema pins the **first**
item only, keeping every field single-valued.

### Normalization

CORD uses Indonesian formatting where **both `.` and `,` are thousands separators**: `60.000`
is sixty thousand, but `1.00` is one. The disambiguating rule — *if the group after the final
separator is exactly 3 digits it is a thousands separator, otherwise a decimal point* — parses
930 of the 940 numeric ground-truth values, and the arithmetic proves it:
`28.182 + 2.818 = 31000`, matching the printed `TOTAL 31.000`.

Both exact and normalized accuracy are always reported. A date normalizer is implemented and
unit-tested, but CORD's scalar fields contain no date, so it is not exercised by this schema —
the report says so rather than implying otherwise.

4 of 1200 slots have a *list* ground-truth value; these are treated as a set of acceptable
values and flagged `gt_multivalued`.

### Decision logic

```
verdict == PASS      and confidence >= T  ->  accept the extracted value
verdict == FAIL      and confidence >= T  ->  apply the correction, else mark mismatch
verdict == UNCERTAIN or confidence <  T   ->  route to manual review
```

### Metrics

The judge is scored as an **error detector** (positive class = "the extraction is wrong"):

| | extraction wrong | extraction correct |
|---|---|---|
| judge says `FAIL` | TP — *correct errors detected* | FP — *correct values incorrectly rejected* |
| judge says `PASS` | FN — *incorrect values incorrectly accepted* | TN |

Rows routed to review are not a PASS/FAIL prediction, so they are excluded from the matrix and
reported separately; both a *decided-only* and an *all-rows* view are given. Accuracy after
validation is reported as a **band**: an automation-only lower bound (review rows keep their
raw value) and an oracle-review upper bound.

**Honesty rules, enforced in code:** any metric with a zero denominator is emitted as `null`,
never `0.0`; every rate carries its raw numerator/denominator; metrics from fewer than 10
observations are marked ⚠; and if the extractor makes no errors the report states that recall
is undefined rather than claiming the judge works.

---

## Failure handling

No API result is ever mocked. A failed call is recorded as an error, that row is excluded from
judge metrics, and the run summary reports how many failed. A missing `TYPESAFE_API_KEY`
aborts before any spend.

---

## Files

```
run_poc.py         the entire POC
requirements.txt
.env.example
README.md
results/           generated output
```

`POC.md` (the brief) and `TypeSafe-vs-LLM-Report.md` (prior background research) are inputs,
not part of the deliverable.
