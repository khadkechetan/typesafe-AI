"""Constants and the fixed 12-field extraction schema."""

from __future__ import annotations

from dataclasses import dataclass

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
