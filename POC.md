Build a minimal end-to-end Python POC to evaluate whether **TypeSafe’s LLM-as-a-Judge capability can reliably validate document field extraction**.

### Target workflow

1. Load 1–10 labeled documents from a public Hugging Face dataset such as FUNSD, CORD, or SROIE.
2. Use NVIDIA-hosted LLMs to extract all predefined fields into structured JSON.
3. Pass **every extracted field**—not only low-confidence fields—to the TypeSafe judge, together with the relevant OCR/document context.
4. For each field, TypeSafe must return:

   * Validation verdict: `PASS`, `FAIL`, or `UNCERTAIN`
   * Judge confidence score from `0.0` to `1.0`
   * Short reason
   * Suggested corrected value, if supported by the document
5. Apply this configurable decision logic:

   * `PASS` with confidence ≥ threshold → accept extracted value
   * `FAIL` with confidence ≥ threshold → use the supported correction or mark as mismatch
   * Confidence below threshold or `UNCERTAIN` → send to manual review
6. Use `0.75` as the default threshold.

### Evaluation

Do not provide ground truth to the extraction model or TypeSafe judge. Use ground truth only after processing to calculate:

* Extraction accuracy before TypeSafe validation
* Final accuracy after TypeSafe validation
* Exact-match and normalized-match accuracy
* TypeSafe judge precision, recall, and F1
* Correct errors detected
* Correct values incorrectly rejected
* Incorrect values incorrectly accepted
* Fields routed to manual review
* Field-level mismatches and corrections
* Metrics by model, field, document, and confidence range

### Technical requirements

* Python 3.11+
* NVIDIA API using its official or OpenAI-compatible client
* TypeSafe API using only documented SDKs or REST endpoints
* Hugging Face `datasets`
* Use dataset OCR when available; otherwise use open-source OCR
* Read credentials only from:

  * `NVIDIA_API_KEY`
  * `TYPESAFE_API_KEY`
  * `HF_TOKEN`
* Provide a CLI such as:

```bash
python run_poc.py --max-documents 5 --judge-threshold 0.75
```

### Deliverables

Create only the minimum required files:

* `run_poc.py`
* `requirements.txt`
* `.env.example`
* `README.md`
* JSON/CSV field-level results
* Markdown evaluation report

### Guardrails

* First verify TypeSafe’s current documentation and actual LLM-as-a-Judge capabilities.
* Never invent TypeSafe endpoints, SDK methods, or response fields.
* If TypeSafe cannot return the required verdict, confidence, reason, or correction, clearly document the limitation and adapt the evaluation to its actual response.
* Validate every extracted field through TypeSafe.
* Keep raw extraction, judge output, final decision, and ground truth as separate values.
* Prevent ground-truth leakage into extraction and judging prompts.
* Do not treat self-reported extraction confidence as calibrated confidence.
* Use deterministic model settings where supported.
* Normalize dates, currency, whitespace, and casing, but report both raw and normalized accuracy.
* Do not mock successful API results when an API call fails.
* Never expose credentials in logs or reports.
* Keep the POC small, reproducible, and runnable locally.
* After running it, conclude whether the results support using TypeSafe as a judge and identify the best confidence threshold from the tested samples.
