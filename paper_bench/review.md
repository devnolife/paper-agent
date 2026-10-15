# Review — main.tex

Critics (model that actually answered): copilot:gpt-6.1-sol, copilot:claude-sonnet-5.5

Recommendations: {'copilot:gpt-6.1-sol': 'major revision', 'copilot:claude-sonnet-5.5': 'major revision'}
Mean scores: {'clarity': 4.0, 'novelty': 3.0, 'soundness': 3.0, 'evidence': 3.0, 'reproducibility': 3.0}
Issues: 16 (0 raised by both critics)

## Issues

- **[major · single]** The benchmark's keys, prose judgments, and extraction labels lack independent human validation despite their central role in establishing quality. — _Tasks and items; Judges and human raters; Limitations: 'Neither the LLM judges nor the item keys have yet been checked by people.'_
  - fix: Complete expert review of all item keys and a stratified sample of prose and extraction outputs, reporting disagreements, corrections, and human–judge agreement.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** Extraction scores compare vendors against different reference labels and potentially different scored fields, compromising direct comparability. — _Extraction: 'a candidate is scored against the majority of the panel members not of its own vendor'._
  - fix: Create a single expert-adjudicated extraction gold set with a common field mask, then rerun the extraction and composite comparisons.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** Programme-specific outline quality is equated with prevalence in one university's theses without validating whether these patterns represent appropriate or required structures. — _Structure: families occurring 'in at least 25% of that programme's theses' and regex-based mapping._
  - fix: Validate reference structures with programme guidelines and faculty experts, audit regex precision and recall, and test transfer to another institution.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** The deterministic grounded-writing checks measure citation syntax and number overlap rather than claim-level evidential support. — _Grounded writing; Limitations: checks do not test 'whether each cited claim is supported'._
  - fix: Add expert-annotated citation entailment and unsupported-claim evaluations, including adversarial outputs that satisfy surface checks while misrepresenting sources.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** Single generations and unadjusted comparisons against the empirically selected leader leave uncertainty about ranking stability and significance. — _Statistics: 'intervals are not adjusted for multiplicity'; Models: 'one run per model and item'._
  - fix: Repeat runs for representative models, incorporate generation variability, and use simultaneous paired comparisons or explicitly exploratory inference.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** Local cost and energy comparisons are confounded by shared hardware, unequal concurrency, and a context-window change during one model's run. — _Experimental Setup, Models and Cost._
  - fix: Repeat representative local runs on an isolated server under fixed contexts and matched concurrency, documenting GPU allocation, throughput, idle subtraction, and rental-time accounting.
  - raised by: copilot:gpt-6.1-sol
- **[major · single]** Items and keys were drafted by LLMs and audited only by another LLM; no human validation of keys, judges or silver labels. — _Sec. 3.2; Limitations_
  - fix: Complete the human review of the methodology keys and the blind-sheet judge comparison before publication.
  - raised by: copilot:claude-sonnet-5.5
- **[major · single]** The panel of silver-label and judge models (gpt-6.1-sol, claude-sonnet-5, gemini-4-argon) is also among the evaluated candidates. Cross-vendor exclusion only partly removes circularity and gives different reference labels to each vendor. — _Extraction paragraph; Judges_
  - fix: Add human-labelled gold for a subset and report extraction scores against it, with sensitivity to excluding panel models.
  - raised by: copilot:claude-sonnet-5.5
- **[major · single]** Single run per model and item at low effort, with ceiling effects on two tasks. The top-3 ranking and the 'not significant' claim rest on a small per-task item count, and run-to-run variance is unmeasured. — _Sec. 4 Statistics; Limitations_
  - fix: Repeat runs (at least 3) for the top models, and report per-task CIs and sensitivity to task weighting.
  - raised by: copilot:claude-sonnet-5.5
- **[major · single]** The local-vs-API cost comparison is confounded: rental price for a shared server at 3x concurrency against vendor AI units, with 4-bit quantisation and context changes mid-run for one model. — _Cost paragraph; Sec. 5.4_
  - fix: Measure cost at fixed concurrency on a dedicated server, and add a sensitivity analysis over concurrency and utilisation.
  - raised by: copilot:claude-sonnet-5.5
- **[minor · single]** The paper does not fully specify several score transformations needed to independently reconstruct the leaderboard. — _Judges and human raters; Extraction; Coding; Statistics._
  - fix: State the 1–5 to 0–1 judge transformation, calibration clipping policy, extraction field weights, evidence contribution, and exact coding-score aggregation, including repair treatment.
  - raised by: copilot:gpt-6.1-sol
- **[minor · single]** Reproduction of the complete benchmark is constrained by private extraction inputs and insufficiently pinned artifact information in the text. — _Introduction, 'An open kit'; Experimental Setup; extraction items 'published as hashes only'._
  - fix: Provide an immutable release identifier, model artifact hashes and full inference settings, frozen lookup responses, and a public extraction substitute or controlled-access replication procedure.
  - raised by: copilot:gpt-6.1-sol
- **[minor · single]** Comparisons are against the leader only, with no multiplicity adjustment. The claim of 'no significant difference' mixes absence of evidence with equivalence. — _Sec. 5.1_
  - fix: Run pairwise tests or equivalence margins with correction.
  - raised by: copilot:claude-sonnet-5.5
- **[minor · single]** The SahabatAI vs Gemma 4 comparison confounds model generation, size and quantisation (Q8 vs Q4), and the Indonesian-tuning conclusion is drawn from a language gap of 0.891 vs 0.925 with no CI. — _Sec. 5.4; Discussion_
  - fix: Soften the claim or compare with the base Gemma 2 9B at matched quantisation.
  - raised by: copilot:claude-sonnet-5.5
- **[minor · single]** The reference score gives abstention the same credit as an unverifiable reference, which penalises cautious models. The structure reference comes from a single university's theses. The premium tier is absent, so the title question 'which model' is only partially answered. — _References, Structure, Models_
  - fix: Report a precision-based metric and scope the claims; add the premium models or narrow the title.
  - raised by: copilot:claude-sonnet-5.5
- **[minor · single]** Fictional model names and dates, plus API account-specific availability, limit replication. Provenance is checked only through the SDK's own report. — _Setup; Provenance_
  - fix: Release raw logs and model identifiers, and add an independent provenance probe.
  - raised by: copilot:claude-sonnet-5.5

## copilot:gpt-6.1-sol (requested copilot:gpt-6.1-sol, 36.1 s)

**Summary.** SkripsiBench introduces 255 Indonesian thesis-workflow items across eight tasks, combining deterministic checks, cross-vendor LLM judgments, bibliographic verification, and execution-based evaluation. It compares 15 API models and 10 local models while recording serving-stack provenance, billing, and local GPU resource use. The reported results show closely ranked leading API models, substantial price differences, weak reference generation by local models, and ceiling effects on methodology and formal-language tasks.

**Strengths**
- The eight tasks address a coherent, practically relevant Indonesian academic workflow rather than generic language understanding.
- Execution-based code evaluation, compiled LaTeX checks, and multi-database reference auditing provide useful objective evidence.
- The paper explicitly distinguishes database absence from proven fabrication and quantifies the effect of references recovered by secondary databases.
- Cost accounting includes judging costs, and the provenance claims are appropriately qualified as relying on SDK reports.
- The limitations candidly acknowledge single runs, private items, silver labels, missing human validation, and shared-server energy measurements.

**Overclaims**
- 'API models keep to the given sources.' — Low unsupported-number counts and citation-format compliance do not establish that substantive claims are supported; human citation-entailment validation is absent.
- 'Within the precision of this benchmark the leading API models perform alike, so price can decide.' — Failure to detect a difference is not an equivalence test, and the composite can hide task-specific tradeoffs; the results section correctly acknowledges non-equivalence.
- 'gpt-oss 20B is the most efficient.' — Efficiency is not formally defined, and shared-server energy attribution and unmatched operating conditions do not establish a general efficiency ranking.
- 'SkripsiBench measures what Indonesian thesis work needs from an LLM.' — The benchmark covers selected short-form tasks, with institutional structure references from one university and no human validation of whether scores reflect thesis usefulness.

**Missing**
- Completed human validation of item keys, extraction records, and prose judgments.
- Independent citation-entailment and reference-audit error-rate assessments.
- Multiple generation runs and multiplicity-aware uncertainty estimates.
- Expert, template-based, and retrieval-assisted baselines for relevant tasks.
- Cross-university evaluation of programme-specific structures.
- Controlled local throughput, cost, and energy measurements.
- Confidence intervals and explicit sampling units for judge reliability and correlation statistics.
- Sensitivity analysis for composite task weights, extraction field masks, and scoring thresholds.

## copilot:claude-sonnet-5.5 (requested copilot:claude-sonnet-5.5, 20.3 s)

**Summary.** The paper introduces SkripsiBench, a 255-item benchmark of eight Indonesian thesis-writing tasks (outlines, methodology MCQ, extraction, grounded review, references, formal Indonesian, coding, LaTeX). It evaluates 15 Copilot-SDK API models and 10 local models on quality, cost, energy and served-model provenance. Main findings: the top three API models are statistically indistinguishable while cost differs about 100x, references separate models most, and local models trail API models.

**Strengths**
- Tasks reflect a real workflow and are mostly scored deterministically (code execution, Crossref lookup, LaTeX compile).
- Reference audit across Crossref, doi.org, Scopus and OpenAlex qualifies the Crossref-only misses, and the paper shows the audit leaves the ranking unchanged.
- Cost, energy and served-model provenance are reported alongside quality, with paired bootstrap CIs and a cost-quality front.
- Limitations are stated candidly, including the unrun human validation, single runs and LLM-drafted items.
- Judges come from other vendors than the candidates, with leniency calibration and reliability statistics.

**Overclaims**
- The SDK reported the requested model as the serving model for every call (provenance verified). — The check relies on the SDK's own usage event, not independent verification, so it cannot detect silent substitution.
- Local models are not cheaper per item than API models. — This depends on rental price, shared-server concurrency and measurement artefacts, so it is not a general result.
- On average local models trail API models on every task, least on formal Indonesian. — Local models were run with quantisation and a different context, and the Indonesian ceiling at near 0.96 makes the task weakly discriminative.
- Judging a stratified sample may suffice (rho=0.994 deterministic-only). — Not tested, and the paper's own judge-vs-deterministic correlations are only moderate across answers (0.53 and 0.69).

**Missing**
- Human gold labels and a human-judge agreement study
- Repeated runs and variance estimates
- Higher reasoning effort and premium-tier models
- Per-task item-level CIs, and a sensitivity analysis of the equal task weighting
- Comparison with existing Indonesian or code benchmarks, or an ablation of the LLM-drafted items
- Contamination analysis of public items
