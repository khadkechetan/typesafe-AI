# TypeSafe AI (Jev / System One) vs. General-Purpose LLMs
### An evidence-graded architecture assessment for a high-volume Document AI pipeline
*Prepared 20 September 2026 · All sources dated in §11 · Evidence levels: **High** = multiple independent sources · **Medium** = vendor docs or limited evidence · **Low** = architectural inference*

---

## 1. Executive summary

- **Jev is a decision model, not a small LLM.** It returns typed, calibrated choices/scores/booleans and **emits no text at all**. This is a category difference, not a quality difference. *(Medium — vendor docs)*
- **A hybrid is structurally required for this pipeline, not merely preferable.** Of the ten stated requirements, Jev can serve 1–6 and 9; requirements **7 (open-ended extraction)** and **8 (reviewer explanation)** are generative by definition and Jev cannot perform them at any quality level. *(High — follows directly from documented capability)*
- **The usage pattern determines whether Jev wins or loses.** Independently tested on 2,000 phishing emails: as *one* question Jev scored **62.6%** vs. Claude Haiku 4.5's **81.3%**; decomposed into *five* atomic questions with learned weighting it reached **95.0%**, while Haiku's five-question composite **declined** to 93.2%. Decomposition helped only Jev. *(High — independent, single task)*
- **Cost advantage is real and large; the headline multipliers are not independently confirmed.** At $0.042/MTok input and free output, Jev is ~24× cheaper than Haiku 4.5 and ~120× cheaper than Opus 5 on input. Independent testing measured ~12–27× cheaper and ~3–5× faster — directionally confirming, but well below the vendor's "193× / 444×". *(High for direction, Low for the specific multipliers)*
- **Treat vendor accuracy and confidence numbers as unvalidated.** TypeSafe's benchmarks score **agreement with GPT-6 Astra and Claude Fable 5.1**, not ground truth, with vendor-disclosed authoring bias. No calibration reporting (ECE/Brier) has been published. *(High — acknowledged by vendor and noted by independent reviewers)*
- **Deterministic code remains competitive and must be measured.** In the same independent test, a **two-line regex scored 91.8%**, beating Jev's single-signal 89.4%. *(High)*
- **LLM-as-a-judge is Jev's best-evidenced use case** (§11) — 91.5% verdict agreement with a frontier judge at ~206× lower cost across 6,003 rubric checks. It cannot explain its verdicts, so pair it with a sampled LLM judge for diagnosis. *(Medium)*
- **Recommendation: staged hybrid**, with Jev adopted only after local calibration validation. Expected outcome is a large cost reduction on the classification/validation majority of the pipeline, not an accuracy improvement.

---

## 2. What TypeSafe AI is, and how it differs from a generative LLM

TypeSafe AI (founded by Diogo Almeida, ex-OpenAI; ~$40M funding; Jev early access opened **15 Sep 2026**) markets Jev as the first **"System One" model** — fast, intuitive judgment, as opposed to the deliberative "System Two" of a reasoning LLM. It is trained via **Reinforcement Learning for Calibrated Decisions (RLCD)** rather than RLHF/RLVR, optimising probabilities against outcomes rather than human preference. *(Medium — vendor; company facts corroborated by The Register)*

You send a **state** (the context) plus one or more **typed questions**, evaluated in parallel in a single pass:

| Primitive | Returns | Limit |
|---|---|---|
| **Choice** | one option + probability per option + confidence | **≤ 255 options** |
| **Score** | a level + probability per level + confidence | **2–10 levels** |
| **Noul** | probability a yes/no statement is true | no separate confidence — the probability *is* the uncertainty |

**The critical distinction — "cannot hallucinate" is true only in a narrow sense.** The output is *guaranteed schema-valid* because the answer space is fixed before inference. That is a **type-safety guarantee, not a correctness guarantee**. Jev can be confidently wrong; it simply cannot be malformed or invent a category. Schema-valid ≠ semantically correct, and this is the claim most likely to be misread by non-technical stakeholders. *(High — logical necessity, and consistent with the 62.6% single-question result)*

**The fit test:** *can you enumerate every valid answer before you ask?*

| Fits Jev | Does not fit Jev |
|---|---|
| Classification, triage, routing, tagging | Open-ended extraction (names, addresses, clause text) |
| Policy / compliance checks | Summaries, explanations, drafting |
| Quality and completeness scoring | Multi-step reasoning needing a chain of thought |
| Risk gating (auto-approve vs. escalate) | Images, audio, video — **text input only** |
| Replacing an LLM that returns one enum and discards the prose | Evidence spans or citations |

---

## 3. Side-by-side comparison

| Dimension | TypeSafe Jev | General-purpose LLM | Ev. |
|---|---|---|---|
| Primary purpose | Typed decisions for software | Text generation for humans/software | High |
| Input modalities | **Text only** (string/JSON/text arrays) | Text, image, PDF, audio (varies) | Med |
| Classification | Native; strong **when decomposed** | Strong zero-shot, single-call | High |
| Scoring / ranking | Native (2–10 ordered levels) | Via prompt; scale adherence unreliable | Med |
| Binary decisions | Native (Noul, true probability) | Via prompt or schema | Med |
| Open-ended extraction | **Not possible** | Native strength | High |
| Generation / summarisation | **Not possible** | Native strength | High |
| Multi-step reasoning | **Not supported** (no chain of thought) | Native (extended thinking) | Med |
| Typed / structured output | Guaranteed by construction | Guaranteed via constrained decoding (`output_config.format`, `strict:true`) | High |
| Probability distributions | **Full distribution over all options** | Not exposed by major APIs | Med |
| Confidence & calibration | Explicitly optimised (RLCD); **no public ECE/Brier** | Verbalised confidence systematically overconfident | High |
| Hallucination risk | Cannot produce invalid types; **can still be wrong** | Can fabricate content and categories | High |
| Output consistency | High (constrained space) | Varies with sampling; improves at temp 0 | Med |
| Explainability | **None** — decision only, no rationale | Native rationale (quality varies) | High |
| Auditability | Distributions are logged and comparable | Log prompt/response; rationale may not reflect true cause | Med |
| Prompt-injection exposure | **Materially lower** — no instruction-following surface, fixed answer space | Significant; active research area | Low |
| Context window | **64K total; 32K state + longest question** | 200K–1M | Med |
| Latency | 70–500 ms claimed; ~100–500 ms measured | ~0.7 s (Haiku) to minutes (reasoning) | High |
| Batch / parallel questions | **Core design** — state ingested once, N questions in one pass | Batch API (50% off), but per-call re-read | Med |
| Scalability | 250K tok/s, 1,200 rpm (vendor says dynamic) | Mature, high, multi-provider | Med |
| Price (input / output per MTok) | **$0.042 / $0** | Haiku 4.5 $1/$5 · Sonnet 5 $2/$10 · Opus 5 $5/$25 · GPT-5.6-luna $0.20/$1.20 · Gemini 3.1 Pro $2/$12 | High |
| SDK maturity | Python, JS + community SDKs; **~6 months old** | Years of production hardening, 7+ languages | High |
| Deployment / privacy | US-hosted API only; no training on customer data; ZDR for enterprise; **no on-prem or open weights** | Multi-cloud (Bedrock/Vertex/Foundry), ZDR, regional pinning | Med |
| Vendor lock-in | **High** — single vendor, single region, no local fallback | Lower — multiple interchangeable providers | High |
| Human-review integration | Confidence is a first-class field, purpose-built for gating | Requires separate confidence estimation | Med |
| Monitoring / eval | Vendor eval dashboard; ecosystem immature | Mature tooling ecosystem | Med |
| Known limitations | Text-only; no generation; 255-option cap; English-optimised (CJK degraded); vendor-noted "1.13 jaggedness" | Cost, latency, calibration, injection | Med |

---

## 4. Document AI task-to-technology mapping

| # | Task | Recommended | Rationale |
|---|---|---|---|
| 1 | Classify into 500+ types | **Jev, 2-stage** | 255-option cap → family (≈30) then type (≈20). Hierarchy also lifts accuracy. |
| 2 | Required fields present? | **Rules → Jev** | Regex/layout rules first (cheap, auditable); Noul per field for the rest. |
| 3 | Validate against business rules | **Rules → Jev** | Deterministic for format/arithmetic/date logic; Noul for judgment calls. |
| 4 | Policy / compliance violations | **Jev (fan-out)** | One Noul per policy, evaluated in parallel against one state. |
| 5 | Quality & completeness score | **Jev (Score)** | Ordered rubric is exactly the Score primitive. |
| 6 | Route low-confidence to review | **Jev confidence + own thresholds** | Only after local calibration (§6). |
| 7 | Extract names, addresses, clauses | **LLM required** | Free text — structurally impossible for Jev. |
| 8 | Reviewer explanation / summary | **LLM required** | Generative by definition. |
| 9 | Schema-compliant results | **Both** | Jev by construction; LLM via constrained decoding. |
| 10 | Millions of docs/week | **Jev-first** | Cost and latency advantage concentrates here. |

**Upstream OCR / multimodal consequence.** Because Jev accepts text only, **OCR quality becomes the hard accuracy ceiling** for tasks 1–6. There is no pixel fallback, so anything carried by layout or image — signatures, stamps, checkbox marks, table geometry, handwriting, redactions — is invisible unless OCR serialises it into text. *Mitigation:* have OCR emit a structured text state (text + coordinates + page + confidence as JSON) rather than a flat string, and route image-dependent questions to a vision-capable LLM. *(High — direct consequence of the documented text-only constraint)*

**Long documents.** The 32K state limit means many contracts and filings will not fit. Retrieve-then-decide (select relevant passages, then ask Jev) is required — which adds a retrieval component that is itself a source of error.

---

## 5. Strengths and limitations

**Jev** — *Strengths:* very low cost; sub-second latency; parallel multi-question evaluation over one state; full probability distributions; guaranteed schema validity; low prompt-injection surface. *Limitations:* no text output, no reasoning, no evidence spans; 255 options; 32K state; text-only; English-optimised; single US vendor with no on-prem option; immature ecosystem; unproven at the stated scale.

**Small LLM (Haiku 4.5 / GPT-5.6-luna)** — *Strengths:* strong single-call classification, generation, extraction, multimodal, mature tooling; Haiku 4.5 shows the best measured verbalised calibration among tested models (ECE 0.122). *Limitations:* ~24× Jev's input cost; slower; injection-exposed.

**Frontier LLM (Opus 5 / Fable 5.1 / GPT-5.6-sol / Gemini 3.1 Pro)** — *Strengths:* best reasoning, long context, citations with verbatim source spans. *Limitations:* 100×+ Jev's cost; seconds-to-minutes latency; overkill for enum answers.

**Deterministic code** — *Strengths:* free, instant, perfectly auditable and reproducible; a regex beat Jev's single signal at 91.8% vs. 89.4%. *Limitations:* brittle to format drift; cannot handle semantic judgment; high maintenance across 500 doc types.

---

## 6. Proposed benchmark methodology

**Dataset.** 5,000–10,000 human-labelled documents, stratified across document types (including rare ones), with a deliberate **ambiguous/none-of-the-above stratum** and an **unseen-category stratum**. Split train / validation / test; the test set is touched once. Labels must be human ground truth — *never* LLM-generated, which is precisely the flaw in the vendor's own evaluation.

**Five arms, identical inputs and decision criteria:** (1) Deterministic rules; (2) Jev; (3) Small LLM; (4) Frontier reasoning LLM; (5) Hybrid. Run Jev in **both** single-question and decomposed-fan-out configurations — the independent evidence says this is the largest single variable.

**Metrics.** Accuracy, precision, recall, macro-F1 (macro matters — 500 classes are heavily imbalanced); **ECE and Brier score** with reliability diagrams; coverage at confidence thresholds; human-review rate; **false-automation rate** (auto-approved and wrong — the metric that carries real business risk); schema-valid rate; semantic correctness (human-audited sample); consistency across 5 repeated runs; P50/P95 latency; cost per 1,000 documents; sustained throughput; failure and retry rates.

**Indicative cost per 1,000 documents** at ~4K tokens each (input only): Jev **$0.17** · GPT-5.6-luna $0.80 · Haiku 4.5 $4.00 · Sonnet 5 $8.00 · Opus 5 $20.00. Jev's free output and single-pass multi-question design widen this further when many questions are asked per document. *(Medium — arithmetic on published rates; excludes LLM output tokens)*

**Calibration validation — the non-negotiable gate.** Vendor confidence must be treated as an **uncalibrated score** until proven otherwise on your own data:
1. Score the validation set; bin predictions by reported confidence (10 bins).
2. Plot observed accuracy per bin against reported confidence — the reliability diagram. Compute ECE and decompose Brier into reliability / resolution / uncertainty.
3. If miscalibrated, **fit your own mapping** (Platt scaling or isotonic regression) on validation and apply it in production. Never ship the raw vendor number as a probability.
4. Set the auto-approve threshold from the *calibrated* curve against a stated false-automation budget (e.g. ≤0.5%), not from a round number like 0.9.
5. Re-run this monthly and on every model-version change — `jev-1.13` will not be the last version, and calibration drifts silently.

**Governance rule:** no auto-approval path goes live until steps 1–4 pass on that document family. Until then Jev runs in shadow mode, scoring traffic without acting on it.

---

## 7. Recommended hybrid architecture

```
OCR  →  Structured text state (text + coords + page + OCR confidence, JSON)
          │
   ┌──────▼──────────────────────────────────────────────┐
   │ STAGE 0 — Deterministic rules                        │  free, instant
   │ format/checksum/date/arithmetic validation,          │  resolves the
   │ field presence, obvious routing                      │  easy majority
   └──────┬───────────────────────────────────────────────┘
          │ unresolved
   ┌──────▼──────────────────────────────────────────────┐
   │ STAGE 1 — Jev fan-out (ONE state, N questions)       │  ~$0.17/1k docs
   │  Choice  : doc family (≈30)  →  doc type (≈20)       │  70–500 ms
   │  Noul ×N : each policy / compliance rule             │
   │  Score   : quality, completeness                     │
   │  → calibrated confidence per answer                  │
   └──────┬───────────────────────────────────────────────┘
          │
   ┌──────▼───────────┬──────────────────┬────────────────┐
   │ HIGH confidence  │ Needs free text  │ LOW confidence │
   │ auto-process     │ → LLM extraction │ → frontier LLM │
   │ (calibrated      │   + reviewer      │   adjudication │
   │  threshold)      │   summary         │   → human      │
   └──────────────────┴──────────────────┴────────────────┘
                              │
                     Schema validation → output
```

**Design rules that follow from the evidence:**
- **Decompose everything.** Ask many narrow questions, not one broad one, and combine them in your code (learned weights beat hand-tuned). This is where Jev's measured advantage lives. *(High)*
- **But keep related facts in one state.** Decomposition loses cross-field relationships — *"is the signature date after the effective date?"* must be asked as one question, not two. Decompose by *independent signal*, never by *related fact*. *(Low — inference)*
- **Jev decides; the LLM explains.** Send the LLM only the documents that need prose, with Jev's decisions as input. This keeps the expensive model off the majority of traffic.
- **Evidence spans come from the LLM, not Jev.** Jev returns a decision with no rationale or source span. If auditors need "show me where in the document," that requires a separate retrieval or LLM component — Claude's Citations API returns verbatim `cited_text` with char/page locations natively. *(High — answers analytical question 8)*

---

## 8. Decision matrix

| Situation | Use |
|---|---|
| Answer is one of a known, enumerable set; high volume; no explanation needed | **Jev only** |
| Output is free text, a summary, or open-ended extraction | **LLM only** |
| Needs images, layout, handwriting, or signatures | **LLM only (multimodal)** |
| Needs a citation or evidence span | **LLM only** (or Jev + retrieval) |
| Multi-step reasoning, novel/unseen categories, genuine ambiguity | **Frontier LLM** |
| Rule is exact, stable, and expressible in code | **Deterministic code** |
| High-volume typed decisions **plus** some free text or explanation | **Hybrid** ← *this pipeline* |
| Regulated workload requiring on-prem or open weights | **Not Jev** (no such option publicly available) |
| Volume low enough that LLM cost is immaterial | **LLM only** — Jev's advantage is cost; don't add a vendor for nothing |

---

## 9. Risks, unanswered questions, and evidence gaps

| Risk | Severity | Note |
|---|---|---|
| **Circular vendor benchmarks** | High | Accuracy is measured as agreement with two frontier LLMs, not ground truth; vendor discloses possible authoring bias. Read as "matches an LLM," never "is correct." |
| **No published calibration data** | High | Despite calibration being the core claim, no ECE/Brier/reliability diagrams have been published. Must be established locally. |
| **Vendor lock-in** | High | Single US-hosted API; no on-prem, no open weights, no second source. Every decision leaves your network. |
| **Compliance posture** | High | **SOC 2 / ISO 27001 status: not publicly verified.** ZDR is referenced for enterprise but terms are not public. Must be confirmed contractually before any regulated data flows. |
| **Immaturity** | Med | ~6 months old; vendor acknowledges "1.13 jaggedness"; rate limits described as "adjusting dynamically." |
| **Version drift** | Med | No public guarantee of version pinning or deprecation notice beyond `jev-1.13`. Recalibration must be triggered by version change. |
| **Prompt-sensitivity** | Med | Independent testing showed accuracy swinging 62.6% → 95.0% on question framing alone. Question design is a first-class engineering artefact needing version control and regression tests. |
| **Unproven at scale** | Med | No public case study at millions of documents/week. **Not publicly verified.** |
| **OCR ceiling** | Med | Text-only input makes OCR the accuracy bound for all Jev-served tasks. |
| **Non-English** | Med | Vendor states reduced accuracy for CJK and other languages. |

**Open questions to put to the vendor:** SOC 2/ISO status and audit report; contractual ZDR terms and data residency; version pinning and deprecation policy; per-class calibration data; sustained throughput SLA and support for burst volume; roadmap for image input and for >255 options.

---

## 10. Final recommendation

**Adopt a staged hybrid — Jev for typed decisions, an LLM for text, rules first — but gate adoption on local validation.** *(Confidence: Medium-High; the architecture follows from documented capability, the accuracy gain does not)*

The case for Jev here is **cost and latency at volume**, not accuracy. On a pipeline of millions of documents per week, moving classification, validation, policy checks and scoring off a general-purpose LLM is a 1–2 order-of-magnitude reduction in inference spend on the majority of traffic, with sub-second latency. That is a genuine and material benefit. But every independent signal says Jev matches rather than exceeds a small LLM on accuracy, and only when questions are decomposed properly.

**Phased plan:**
1. **Weeks 1–4 — Baseline.** Build the labelled set. Implement deterministic rules first and measure them honestly; they may resolve more than expected. Run the five-arm benchmark (§6) on 2–3 document families, testing Jev both single-question and decomposed.
2. **Weeks 5–8 — Calibrate and shadow.** Run the calibration protocol. Fit your own confidence mapping. Run Jev in shadow mode against production traffic with no authority to act.
3. **Weeks 9–12 — Narrow production.** Enable auto-approval for the 2–3 families that passed, at a threshold set by the calibrated curve against a stated false-automation budget. Keep the LLM path for extraction and summaries throughout.
4. **Ongoing — Expand and monitor.** Add families only after each passes the same gate. Monthly recalibration; treat any version change as a trigger for full revalidation.

**Go/no-go gates:** (a) calibrated ECE below your threshold on your data; (b) false-automation rate within budget at a coverage level that makes the economics work; (c) contractual confirmation of compliance and data-handling terms; (d) a documented fallback to the LLM path if the vendor has an outage. **Do not proceed past shadow mode on any of these failing.**

The realistic outcome is a pipeline that costs far less to run at similar accuracy — with an added vendor dependency that must be managed deliberately.

---

## 11. Jev as an LLM-as-a-Judge for classification tasks

A judge scores against a rubric — a closed answer space — so this is the use case Jev fits best, and the one with the most third-party evidence. The rubric maps straight onto the primitives: **binary criterion → Noul, categorical → Choice, numeric scale → Score.** Eval platforms (Langfuse, Braintrust, Arize, Pydantic AI) shipped Jev judge integrations within days of launch.

| Your question | Answer | Evidence | Ev. |
|---|---|---|---|
| **1. How effective is it?** | Effective for rubric-shaped judging — reaches an LLM judge's verdict at a fraction of the cost. Not usable for open-ended critique. | 6,003 rubric checks: **91.5% agreement** with Claude Fable 5.1 at **$160 vs $33,000** per million graded (~206×). Moderation: 96% vs 86% for Gemini Flash-Lite, 58× cheaper. Spam: 98.3%. | Med |
| **2. No knowledge source like an LLM?** | Correct — and it's the sharpest limit. Jev's weights are not a knowledge store: no browsing, no retrieval, no world-knowledge prior. It is a **reference-based judge** — *"does this match the reference?"* works; *"is this claim true?"* does not unless you put the truth in the state. | 32K state cap bounds how much reference fits; accuracy degrades when the state is padded with irrelevant material. | Med |
| **3. How can we rely on its output?** | Through **calibrated bands validated on your own labels** — not through the raw score. Two hard gaps: it **cannot abstain** and gives **no rationale**, so it can't tell you *why* something failed. | Spam judge: 0.9+ band was **99.9%** correct, 0.5–0.6 band only **38%** — bands mean what they say. LLM judges by contrast cluster at 90–100% confidence with accuracy well below. | Med |
| **4. Can we fine-tune it?** | **No.** No fine-tuning, no LoRA, no per-account weights — the same weights serve every customer. You adapt at request time only. | Vendor docs: adaptation via `state` (your reference material), `instructions` (domain rules), `criteria` (option/level definitions). | Med |

**Three practical consequences**

- **What you *can* train is the layer around it.** Two things recover much of the value of fine-tuning: your own **calibration mapping** (Platt/isotonic on your labels) and a **learned combination over decomposed questions** — the same move that took the phishing task from 62.6% to 95.0%. For a narrow, stable, high-volume judging task, a fine-tuned encoder classifier remains the real competitor, because it *can* be trained on your data.
- **Pair it for the "why."** Run Jev at 100% coverage for the verdict, and sample the failures through an LLM judge to get explanations. Cheap coverage plus expensive diagnosis on a slice.
- **Add an explicit "unclear" option**, since it cannot abstain — and treat a flat probability distribution as an abstention signal in your own code.

⚠️ **Read 91.5% as *agreement with an LLM*, not accuracy.** Like TypeSafe's own benchmarks, that figure measures how often Jev matched another model's verdict, not how often it was right. Before trusting any judge — Jev or LLM — score it against human labels on your data.

---

## 12. Sources

**Vendor (TypeSafe AI)** — *treat as vendor-documented facts and vendor claims, not independent evidence*
- [Introduction](https://docs.typesafe.ai/introduction) · [System One concepts](https://docs.typesafe.ai/concepts/system-one) · [Primitives](https://docs.typesafe.ai/primitives) · [Models & limits](https://docs.typesafe.ai/models) — accessed 20 Sep 2026
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — 15 Sep 2026
- [Privacy policy](https://typesafe.ai/legal/privacy-policy) — accessed 20 Sep 2026

**Independent reporting and evaluation**
- [The Register — TypeSafe AI debuts model for machines that plays Doom](https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711) — 16 Sep 2026 (funding, leadership, launch)
- [XenoSpectrum — Jev, the AI That Never Writes a Sentence](https://xenospectrum.com/en/jev-typesafe-bert-classifier-decomposition/) — ~17 Sep 2026 (2,000-email phishing test; decomposition; regex baseline)
- [The Cherry Creek News — Its Own Eval Scores Against Two Other Models' Answers](https://thecherrycreeknews.com/typesafe-jev-system-one-model-claims-evals-independent-tests-cherry_creek/) — Sep 2026 (benchmark circularity)
- [explainX — Jev's Speed/Cost Claims: Fact-Checked](https://www.explainx.ai/blog/jev-speed-cost-claims-fact-check-2026) — Sep 2026
- [MarkTechPost — TypeSafe AI Releases Jev](https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/) — 19 Sep 2026
- [Kingy AI — TypeSafe Jev Review](https://kingy.ai/blog/typesafe-jev-review-the-ai-model-that-doesnt-generate-text/) — Sep 2026

**LLM-as-a-judge (§11)**
- [Arize — TypeSafe Jev: Can Decision Models Replace LLM Judges?](https://arize.com/blog/typesafe-jev-llm-judge/) — Sep 2026 (calibration bands; no-explanation limitation; moderation and spam results)
- [Langfuse — Using TypeSafe's Jev for evals](https://langfuse.com/blog/2026-09-18-using-typesafes-jev-for-evals) — 18 Sep 2026 (rubric→primitive mapping; Good Start Labs 6,003-check figures; cannot abstain; context rot)
- [Braintrust — Eval agent responses with Jev](https://www.braintrust.dev/blog/evaluate-agent-responses-with-jev) — Sep 2026
- [Pydantic AI — TypeSafe (Jev) model docs](https://pydantic.dev/docs/ai/models/typesafe/) — Sep 2026 (no fine-tuning; adaptation via state/instructions/criteria)
- [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/pdf/2410.21819) — 2024 · [LLMs-as-Judges: A Comprehensive Survey](https://arxiv.org/pdf/2412.05579) — Dec 2024 *(position, verbosity and self-preference bias — the biases a non-generative judge structurally avoids)*

**Comparator model documentation**
- [Anthropic pricing & models](https://docs.claude.com/en/docs/about-claude/pricing) — Haiku 4.5 $1/$5 (200K); Sonnet 5 $2/$10; Opus 5 $5/$25; Fable 5.1 $10/$50 (1M) — cached 24 Jun 2026
- [Anthropic structured outputs & Citations API](https://docs.claude.com/en/docs/build-with-claude/tool-use) — `output_config.format`, `strict:true`, verbatim `cited_text` spans
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing) — GPT-5.6-luna $0.20/$1.20; terra $2/$12; sol $5/$30 — Aug–Sep 2026
- [Google Gemini pricing](https://ai.google.dev/pricing) — Gemini 3.1 Pro $2/$12 (≤200K) — 2026

**Peer-reviewed / academic**
- [Let Me Speak Freely? Impact of Format Restrictions on LLM Performance](https://arxiv.org/pdf/2408.02442) — EMNLP 2024. *Constrained decoding hinders reasoning but **improves classification accuracy** — the direct answer to "does typing improve judgment?"*
- [JSONSchemaBench: A Rigorous Benchmark of Structured Outputs](https://arxiv.org/pdf/2501.10868) — 2025
- [On Verbalized Confidence Scores for LLMs](https://arxiv.org/pdf/2412.14737) — Dec 2024
- [Overconfidence is Key: Verbalized Uncertainty Evaluation](https://arxiv.org/pdf/2405.02917) — Groot & Valdenegro-Toro, 2024
- [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/pdf/2410.09724) — 2024
- [The Dunning-Kruger Effect in LLMs: Confidence Calibration](https://arxiv.org/html/2603.09985) — 2026 (Haiku 4.5 ECE 0.122; Kimi K2 ECE 0.726)

---

### Appendix — the ten analytical distinctions, answered

1. **Does typing improve judgment, or only consumability?** Mostly consumability — but not only. Constrained decoding *improves* classification accuracy while *degrading* reasoning (EMNLP 2024). For this pipeline's enum-shaped tasks, typing plausibly helps a little; it is not a general accuracy gain. *(High)*
2. **How does Jev's confidence differ?** It is a trained output optimised against outcomes (RLCD) — closer to a true classification probability than to LLM token probability (which measures token likelihood, not correctness) or self-reported confidence (systematically overconfident). But *optimised for calibration ≠ calibrated on your data*, and no independent ECE has been published. *(Medium)*
3. **Does schema-valid guarantee semantically correct?** **No.** Type safety and correctness are orthogonal — Jev's 62.6% single-question score was 100% schema-valid. *(High)*
4. **Can LLM structured outputs reproduce Jev?** Largely yes for *shape* — constrained decoding and `strict:true` give guaranteed-valid enums. What they don't reproduce: the price, the sub-second latency, the full probability distribution over options, and single-pass multi-question evaluation over one state. *(High)*
5. **What remains materially different?** Cost/latency at volume; exposed probability distributions; parallel multi-question evaluation; a much smaller prompt-injection surface. *(Medium)*
6. **Handling of edge cases.** *Missing/ambiguous evidence* → surfaces as low confidence or a flat distribution (the distribution, not the argmax, is the signal). *None-of-the-above* → you must add that option explicitly; it will not be invented. *Large option sets* → 255 cap forces hierarchy. *Unseen categories* → cannot be returned at all; needs an "other" option plus an LLM fallback. *Long documents* → 32K state forces retrieval. *(Medium)*
7. **When does decomposition help vs. hurt?** Helps when signals are genuinely independent — measured 62.6% → 95.0%. Hurts when it severs a relationship between facts; any question requiring two fields to be compared must remain a single question. *(High for the gain, Low for the boundary)*
8. **Can Jev provide evidence spans?** **No** — it returns a decision with no rationale. Spans require a separate retrieval or LLM component. *(High)*
9. **How to validate vendor confidence?** Reliability diagrams + ECE + Brier decomposition on your own labelled data; refit with Platt/isotonic scaling; set thresholds from the calibrated curve; recalibrate monthly and on every version change. Full protocol in §6. *(High)*
10. **Public evidence on security, compliance, throughput, maturity?** Thin. Privacy policy and ZDR references exist; **SOC 2/ISO certification, production case studies, and sustained throughput at scale are not publicly verified.** *(High — confirmed absence)*
