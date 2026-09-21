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

Package layout
--------------
    constants.py    schema, enums and tunables shared by every module
    credentials.py  environment credential loading + secret redaction
    models.py       the dataclasses that flow through the pipeline
    dataset.py      CORD-v2 loading and OCR reconstruction
    normalize.py    value normalization and match comparisons
    extraction.py   NVIDIA (OpenAI-compatible) field extraction
    judging.py      TypeSafe (Jev) field validation
    decision.py     accept / correct / mismatch / review logic
    metrics.py      accuracy, judge-performance and calibration metrics
    reporting.py    CSV/JSON writers and the markdown report
    selfcheck.py    offline assertions that need no API calls
    pipeline.py     per-document orchestration and stability measurement
    cli.py          argument parsing and the `main()` entry point
"""

from .constants import POC_VERSION

__version__ = POC_VERSION
