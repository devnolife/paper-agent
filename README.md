<div align="center">

# AcademicBench

**Which LLM should Indonesian students, supervisors and thesis tools use — and at what cost? 255 items in eight
thesis tasks, 25 local and API models, the cost and the served model of every call.**

[![CI](https://github.com/devnolife/paper-agent/actions/workflows/ci.yml/badge.svg?branch=bench)](https://github.com/devnolife/paper-agent/actions/workflows/ci.yml?query=branch%3Abench)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper EN](https://img.shields.io/badge/paper-English%20PDF-8a2be2.svg)](paper_bench/build/main.pdf)
[![Paper ID](https://img.shields.io/badge/paper-Bahasa%20Indonesia%20PDF-8a2be2.svg)](paper_bench/build/main_id.pdf)

[At a glance](#at-a-glance) · [For reviewers](#for-reviewers) · [Tasks](#the-benchmark) · [Key findings](#key-findings) ·
[Full results](#full-results) · [Run it](#run-it) · [Paper](#paper) · [Citation](#citation)

</div>

> [!NOTE]
> This is the **`bench`** branch: the AcademicBench benchmark and its paper. The paper-agent tool it runs on and the
> paper on silent model substitution are on the [`translate`](https://github.com/devnolife/paper-agent/tree/translate) branch; [`main`](https://github.com/devnolife/paper-agent/tree/main) holds both.

## At a glance

| | |
|---|---|
| **Question** | Vendor leaderboards rank models on English knowledge and reasoning; none says which model to use for a *skripsi*, the Indonesian undergraduate thesis, or for the analysis code behind it |
| **Benchmark** | 255 items in 8 tasks from the thesis workflow — programme-specific chapter outlines, methodology questions, methods extraction from real theses, grounded literature reviews, reference lists, formal Indonesian, data-analysis code, LaTeX |
| **Models** | 15 API models of the GitHub Copilot SDK and 10 local open-weight models (Ollama, two L40S) — 7,356 API calls, US$46.65, 6.1 GPU-hours |
| **Findings** | the three leaders are not significantly different, yet cost differs 104× between models 0.04 apart; references separate the models (0–31 % of an API model's references are found in no database, 61–85 % for local models); the best local model stays below the weakest API model |
| **Paper** | [English](paper_bench/build/main.pdf) (8 pages) · [Indonesian](paper_bench/build/main_id.pdf) |
| **Check it** | `python experiments/bench/verify_release.py` — 9 checks, about 2 seconds, no API key, GPU or private data |

<p align="center">
  <img src="docs/img/bench_pareto.png" alt="Overall score against cost per 100 items" width="62%">
</p>

---

## For reviewers

> [!TIP]
> The paper's headline numbers can be checked from this branch alone — no API key, no GPU, no private data. CI runs
> step 1 and the test suite on every push (Python 3.10 and 3.12).

### 1 · Verify the released results (about 2 seconds)

```bash
git clone -b bench https://github.com/devnolife/paper-agent.git && cd paper-agent
pip install -e .
python experiments/bench/verify_release.py
```

It recomputes, from the 6,375 released per-item scores, the task and overall scores, the ranking, the
task-stratified rank bootstrap with the paired difference to the leader, the three sensitivity analyses, the served
models of the main run, the SHA-256 of the public items, and the 24 key numbers printed in the paper:

```text
ok    task and overall scores (25 models x 8 tasks, 6375 item scores)                 max |diff| 0.000067 vs summary.json
ok    overall ranking                                                                 same order as summary.json
ok    rank bootstrap: P(rank 1), rank CIs, paired difference to the leader [95 % CI]  max |diff| P(top1) 0.0000, difference/CI 0.00000; rank CIs identical
ok    ranking with prose scored by automatic checks only / judges only                Spearman rho 0.994 / 0.996
ok    ranking without the private extraction task                                     Spearman rho 0.999, largest rank shift 1
ok    ranking with the works found by the reference audit credited                    40 references, Spearman rho 0.999, largest rank shift 1
ok    served model = requested model (main run, API models)                           3825 calls of 15 models
ok    public items match bench/MANIFEST.json (SHA-256)                                225 public items identical, 30 private items listed as hashes only
ok    numbers printed in the paper (generated/numbers.tex)                            24 macros recomputed

9/9 checks agree
```

### 2 · Rebuild the paper from the data (about a minute; needs TeX Live)

```bash
pip install matplotlib
python experiments/bench/make_figures.py     # figures from summary.json, analysis.json, refs_audit.json
python paper_bench/make_tables.py            # tables and every number in the text -> paper_bench/generated/
python experiments/bench/make_results_md.py --check   # the result tables of this README match the data
paper-agent compile paper_bench/main.tex     # 8 pages, 0 undefined references or citations
paper-agent check-citations --tex paper_bench/main.tex --bib paper_bench/references.bib \
                            --report paper_bench/references_report.md
```

No number in the text is typed by hand: each is a macro in
[`generated/numbers.tex`](paper_bench/generated/numbers.tex), written by
[`make_tables.py`](paper_bench/make_tables.py).

### 3 · Where each claim comes from

| Claim (AcademicBench) | Value | Evidence |
|---|---|---|
| The three leaders are not significantly different | paired difference to the leader 0.006 [−0.014, 0.027] and 0.010 [−0.010, 0.028] | [`analysis.json`](experiments/bench/analysis.json) → `rank_bootstrap.*.diff_ci` |
| Price differs about 100× among comparable models | US$3.12 vs US$0.030 per 100 items for 0.941 vs 0.901 | [`summary.json`](experiments/bench/summary.json) → `models.*.cost.usd_per_100` |
| References separate the models | 0–31 % of an API model's references are found in no database (local models: 61–85 %) | [`refs_audit.json`](experiments/bench/refs_audit.json) → `models.*.no_match_rate` |
| The ranking does not hinge on the judges, the private task or Crossref coverage | Spearman ρ 0.994 (no judges), 0.999 (no private task), 0.999 (audit credited) | `analysis.json` → `judge_sensitivity`, `public_only`, `verification_sensitivity` |
| The judges agree | Krippendorff's α = 0.86 (interval), on 480 answers rated by all three | `analysis.json` → `judges` |
| No silent substitution in this study | served = requested for every call of the main run (3,825) | `summary.json` → `models.*.cost.served` |
| Every citation in the paper is verified | 23 in ≥ 2 independent sources, 5 manual (software, web, manuscript) | [`references_report.md`](paper_bench/references_report.md) |

### 4 · What is released, and what is not

| Released | Not released, and why |
|---|---|
| 225 public items, word lists, item generators, model roster ([`bench/`](bench/README.md)) | the 30 extraction items: masked excerpts of real theses (privacy); their SHA-256 are in [`MANIFEST.json`](bench/MANIFEST.json) |
| runner, scorers, sandbox, judges, statistics ([`paper_agent/bench/`](paper_agent/bench/)) | raw answers and judge ratings: they include the answers on the private excerpts |
| per-item scores of all 25 models, summary, analyses, reference audit (aggregates only) | API credentials: bring your own Copilot token; local models run on any Ollama server |

### 5 · Re-running the benchmark

`paper-agent bench plan` counts the calls and estimates their cost without calling any model; `run` is resumable and
refuses the premium tier and high reasoning effort without `--approve-high-cost`, and `--budget-usd` stops new calls
when a spending cap is reached. This study cost US$46.65 in API calls (answers 21.52, judges 24.50, cost probes 0.63)
and 6.1 GPU-hours. Environment: GitHub Copilot SDK 1.0.9 (Copilot CLI 1.0.73), Ollama 0.22 on two NVIDIA L40S; API
models as enabled for the author's account on 14 October 2026; fixed bootstrap seeds. Hosted models change without
notice, so a re-run measures the models of its own date. Commands: [Run it](#run-it).

### 6 · Threats to validity

One run per model and item; no human validation yet of item keys, judges and extraction silver labels (blind sheets
for two raters are generated by `paper-agent bench human-sheet`); structure references from the theses of one
university; local energy measured on a shared server; the premium tier not included (see [Limitations](#limitations)).

---

## The benchmark

| Task | What the model does | Items | Scored by |
|---|---|---:|---|
| `structure` | outline chapter III (methods) and IV (results) for a thesis title in a study programme | 35 | prevalence-weighted F1 of sub-section families vs. the programme's typical structure (aggregated from a local thesis dataset) |
| `methodology` | multiple-choice questions on research methods and statistics | 60 | accuracy |
| `extraction` | fill a closed-vocabulary methods record from a masked thesis excerpt (**private items**) | 30 | field agreement with cross-vendor silver labels (no model graded by its own vendor) + verbatim-evidence check |
| `grounded` | literature review from six given abstracts, citing `[P1]…[P6]` | 30 | citation coverage/validity, sentence citation rate, unsupported numbers, length; two LLM judges |
| `references` | list five real references with DOI for a topic | 20 | Crossref lookup by DOI, then title + first author (not-found rate, DOI correctness); misses audited in doi.org, Scopus and OpenAlex |
| `language` | abstract from a research summary; formal rewrite of an informal draft | 30 | length, English leakage, non-standard words, kept/fabricated numbers, key terms, sentence length; two LLM judges |
| `coding` | Python data analysis on a synthetic CSV (regression, t/Mann–Whitney, ANOVA+Tukey, Cronbach, N-gain, …) | 30 | the program runs offline in a sandbox; JSON outputs compared within tolerance; pass@1 and pass after one repair |
| `latex` | fix a broken document, typeset a booktabs table, convert a reference to BibTeX | 20 | pdflatex in the sandbox, PDF text, cell-by-cell comparison, BibTeX fields |

Prompts are in Indonesian. Items were written for the benchmark (14 October 2026) except the extraction excerpts,
which come from a local thesis dataset of one Indonesian university and stay private. Full scoring rules:
[`bench/README.md`](bench/README.md).

```mermaid
flowchart LR
    I["255 items, 8 tasks<br/>225 public · 30 private (hashed)"] --> R["runner<br/>cost gate · budget cap<br/>served model · tokens · AI units · GPU energy"]
    R --> A1["15 API models<br/>(GitHub Copilot SDK)"]
    R --> A2["10 local models<br/>(Ollama, 2× L40S)"]
    A1 & A2 --> D["deterministic scorers<br/>sandboxed code and LaTeX · Crossref · word lists"]
    A1 & A2 --> J["3 judges of other vendors<br/>leniency-calibrated"]
    D --> AU["reference audit<br/>doi.org · Scopus · OpenAlex"]
    D & J --> S["6,375 per-item scores"]
    S --> T["bootstrap CIs · rank stability<br/>sensitivity analyses"]
    T --> P["paper: every number<br/>a generated macro"]
```

**Why the scores can be trusted.** Six of the eight tasks are scored without an LLM, and the two prose tasks are
half deterministic. Model-written programs and LaTeX run in a sandbox (Linux user, mount, network and PID namespaces,
no network, capabilities dropped, cgroup limits). References are looked up in Crossref with paper-agent's matching
rule, and every miss is audited again in doi.org, Scopus and OpenAlex. A judge never rates models of its own vendor,
and each judge's leniency is removed before averaging. Failed calls score 0 instead of disappearing.

---

## Key findings

<p align="center">
  <img src="docs/img/bench_pareto.png" alt="Overall score against cost per 100 items" width="48%">
  &nbsp;
  <img src="docs/img/bench_references.png" alt="References not found in Crossref and in no database" width="48%">
</p>
<p align="center">
  <img src="docs/img/bench_heatmap.png" alt="Per-task scores of the 25 models" width="48%">
  &nbsp;
  <img src="docs/img/bench_energy.png" alt="GPU energy per 100 items of the local models" width="48%">
</p>

* **The three leaders are not significantly different.** The paired difference to kimi-k3 is 0.006 [−0.014, 0.027]
  for gpt-5.4 and 0.010 [−0.010, 0.028] for gpt-6.1-sol (task-stratified bootstrap). That is not equivalence, and two
  tasks (methodology questions, formal Indonesian) are near ceiling for every API model.
* **Price is what differs.** Per 100 items kimi-k3 costs US$3.12 (it reasons at length: median 34.5 s per call),
  gpt-6.1-sol US$0.46, and gpt-6-luna reaches 0.901 for US$0.030 — 104 times less for 0.04 less. Tier names are a
  poor guide: the most expensive model per item was grouped with the economy models.
* **References separate the models.** 0–40 % of the references an API model listed are not found in Crossref. The
  second-source audit ([`audit_refs.py`](experiments/bench/audit_refs.py)) finds 17 % of those misses with matching
  metadata elsewhere; 0–31 % of an API model's references are found in no database, against 61–85 % for local
  models. Verify every reference a model produces.
* **Local models trail on every task, on average.** The best, Qwen3.6 36B (0.745), stays below the weakest API model
  (claude-haiku-4.5, 0.777), and at rental prices local models were not cheaper per item; they come closest on formal
  Indonesian. The Indonesian-tuned SahabatAI 9B did not write formal Indonesian better than the general Gemma 4 8B.
* **Provenance and judges.** The SDK reported the requested model as the serving model for all 7,356 calls (no
  substitution; the incident that motivated this check is the subject of the [paper-agent paper](https://github.com/devnolife/paper-agent/tree/translate)). The three judges agree (Krippendorff's α = 0.86), and
  scoring the prose by the deterministic checks alone barely changes the ranking (Spearman ρ = 0.994).

> [!NOTE]
> Not in this version: the nine premium-tier models (estimated at about US$50 for all items; the runner gates them
> behind an explicit approval and a budget cap), a high-effort check of the leaders, repeated runs, and the
> comparison of the judges with two human raters.

---

## Full results

25 models ran on 14–15 October 2026: 15 API models of the GitHub Copilot SDK (grouped by name into an economy and a
standard tier, low reasoning effort where a model accepts one) and 10 local open-weight models on a shared Ollama
server with two NVIDIA L40S GPUs, each on all 255 items, one run per model and item. Every table below is generated
from the released data ([`summary.json`](experiments/bench/summary.json),
[`analysis.json`](experiments/bench/analysis.json), [`refs_audit.json`](experiments/bench/refs_audit.json)) by
[`make_results_md.py`](experiments/bench/make_results_md.py); CI fails when they are out of date. Item-level scores:
[`per_item.csv`](experiments/bench/per_item.csv).

<!-- results:begin -->
### Leaderboard

Overall score = unweighted mean of the eight task scores, with its 95 % task-stratified bootstrap interval. Academic: the six writing and research tasks; Code: data-analysis code and LaTeX. US$ per 100 items: API models in the vendor's AI units at US$0.01, local models as GPU time at US$1.09/h (on-demand L40S). kWh: GPU board energy of the local models (shared server, upper bound).

| # | Model | Tier | Overall [95 % CI] | Academic | Code | US$ / 100 | kWh / 100 | Median s / call |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **kimi-k3** | economy | **0.941 [0.92, 0.96]** | 0.930 | 0.975 | 3.119 | – | 34.5 |
| 2 | **gpt-5.4** | standard | **0.935 [0.92, 0.95]** | 0.913 | 1.000 | 0.864 | – | 7.9 |
| 3 | **gpt-6.1-sol** | standard | **0.931 [0.92, 0.95]** | 0.908 | 1.000 | 0.461 | – | 8.3 |
| 4 | claude-sonnet-5.5 | standard | 0.913 [0.89, 0.93] | 0.890 | 0.983 | 0.783 | – | 4.8 |
| 5 | gpt-5.6-terra | standard | 0.913 [0.89, 0.94] | 0.898 | 0.958 | 0.585 | – | 5.9 |
| 6 | gemini-3.7-flash | economy | 0.911 [0.90, 0.92] | 0.882 | 1.000 | 0.141 | – | 15.4 |
| 7 | gemini-4-argon | standard | 0.901 [0.88, 0.92] | 0.868 | 1.000 | 0.396 | – | 5.8 |
| 8 | gpt-6-luna | standard | 0.901 [0.87, 0.93] | 0.901 | 0.900 | 0.030 | – | 7.0 |
| 9 | claude-haiku-5.5 | economy | 0.895 [0.87, 0.92] | 0.868 | 0.975 | 0.056 | – | 6.8 |
| 10 | gemini-3.8-flash | economy | 0.893 [0.87, 0.91] | 0.857 | 1.000 | 0.145 | – | 18.6 |
| 11 | gpt-5.4-mini | economy | 0.883 [0.86, 0.91] | 0.868 | 0.931 | 0.252 | – | 4.8 |
| 12 | claude-sonnet-5 | standard | 0.876 [0.85, 0.90] | 0.834 | 1.000 | 0.711 | – | 5.8 |
| 13 | mai-code-1.1-flash | economy | 0.837 [0.81, 0.86] | 0.808 | 0.922 | 0.157 | – | 7.9 |
| 14 | gpt-5-mini | economy | 0.816 [0.78, 0.85] | 0.790 | 0.892 | 0.136 | – | 7.7 |
| 15 | claude-haiku-4.5 | economy | 0.777 [0.74, 0.81] | 0.762 | 0.823 | 0.598 | – | 10.6 |
| 16 | Qwen3.6 36B | local | 0.745 [0.71, 0.77] | 0.719 | 0.825 | 0.655 | 0.1106 | 57.7 |
| 17 | gpt-oss 20B | local | 0.723 [0.70, 0.74] | 0.661 | 0.908 | 0.087 | 0.0189 | 4.1 |
| 18 | Gemma 3 27B | local | 0.691 [0.66, 0.72] | 0.680 | 0.727 | 0.170 | 0.0286 | 9.6 |
| 19 | Gemma 4 8B | local | 0.668 [0.64, 0.70] | 0.671 | 0.660 | 0.183 | 0.0357 | 13.6 |
| 20 | Phi-4 reasoning plus 14.7B | local | 0.639 [0.60, 0.67] | 0.586 | 0.801 | 0.693 | 0.0713 | 8.7 |
| 21 | Qwen2.5-Coder 32B | local | 0.621 [0.59, 0.65] | 0.604 | 0.674 | 0.259 | 0.0460 | 20.2 |
| 22 | Qwen3-Coder 30B-A3B | local | 0.618 [0.58, 0.65] | 0.597 | 0.681 | 0.236 | 0.0259 | 3.2 |
| 23 | Qwen2.5 7B Instruct | local | 0.518 [0.48, 0.55] | 0.523 | 0.503 | 0.083 | – | 4.7 |
| 24 | Gemma2 9B CPT SahabatAI | local | 0.517 [0.48, 0.55] | 0.571 | 0.357 | 0.299 | 0.0520 | 5.9 |
| 25 | Llama 3.2 3B | local | 0.368 [0.34, 0.40] | 0.408 | 0.248 | 0.067 | 0.0124 | 3.4 |

### Scores per task

Mean item score in [0, 1]; best per column in bold. Items per task: outline 35, method. 60, extract. 30, grounded 30, refs 20, formal id 30, code 30, latex 20.

| Model | Outline | Method. | Extract. | Grounded | Refs | Formal ID | Code | LaTeX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi-k3 | **0.79** | **1.00** | 0.93 | 0.95 | **0.94** | 0.98 | **1.00** | 0.95 |
| gpt-5.4 | 0.76 | **1.00** | 0.95 | **0.96** | 0.84 | 0.97 | **1.00** | **1.00** |
| gpt-6.1-sol | 0.74 | **1.00** | 0.97 | 0.96 | 0.81 | 0.97 | **1.00** | **1.00** |
| claude-sonnet-5.5 | 0.75 | **1.00** | 0.93 | 0.95 | 0.73 | 0.99 | 0.97 | **1.00** |
| gpt-5.6-terra | 0.76 | 0.98 | 0.95 | 0.94 | 0.79 | 0.97 | 0.97 | 0.95 |
| gemini-3.7-flash | 0.73 | **1.00** | 0.94 | 0.95 | 0.71 | 0.97 | **1.00** | **1.00** |
| gemini-4-argon | 0.73 | **1.00** | **0.97** | 0.89 | 0.65 | 0.96 | **1.00** | **1.00** |
| gpt-6-luna | 0.77 | **1.00** | 0.95 | 0.93 | 0.79 | 0.97 | **1.00** | 0.80 |
| claude-haiku-5.5 | 0.72 | **1.00** | 0.93 | 0.88 | 0.69 | **0.99** | **1.00** | 0.95 |
| gemini-3.8-flash | 0.71 | **1.00** | 0.92 | 0.95 | 0.61 | 0.97 | **1.00** | **1.00** |
| gpt-5.4-mini | 0.74 | **1.00** | 0.92 | 0.92 | 0.66 | 0.96 | 0.97 | 0.90 |
| claude-sonnet-5 | 0.74 | **1.00** | 0.90 | 0.93 | 0.45 | 0.98 | **1.00** | **1.00** |
| mai-code-1.1-flash | 0.57 | **1.00** | 0.87 | 0.89 | 0.56 | 0.96 | 0.89 | 0.95 |
| gpt-5-mini | 0.65 | **1.00** | 0.85 | 0.92 | 0.40 | 0.92 | 0.93 | 0.85 |
| claude-haiku-4.5 | 0.68 | 0.98 | 0.88 | 0.84 | 0.23 | 0.96 | 0.86 | 0.79 |
| Qwen3.6 36B | 0.66 | **1.00** | 0.81 | 0.82 | 0.06 | 0.96 | 0.80 | 0.85 |
| gpt-oss 20B | 0.50 | 0.93 | 0.80 | 0.75 | 0.06 | 0.93 | 0.87 | 0.95 |
| Gemma 3 27B | 0.69 | 0.80 | 0.76 | 0.81 | 0.09 | 0.93 | 0.58 | 0.88 |
| Gemma 4 8B | 0.65 | 0.82 | 0.72 | 0.89 | 0.03 | 0.93 | 0.49 | 0.83 |
| Phi-4 reasoning plus 14.7B | 0.54 | 0.75 | 0.62 | 0.59 | 0.13 | 0.88 | 0.81 | 0.79 |
| Qwen2.5-Coder 32B | 0.53 | 0.77 | 0.76 | 0.65 | 0.04 | 0.87 | 0.42 | 0.93 |
| Qwen3-Coder 30B-A3B | 0.53 | 0.77 | 0.62 | 0.72 | 0.07 | 0.89 | 0.53 | 0.83 |
| Qwen2.5 7B Instruct | 0.41 | 0.68 | 0.52 | 0.68 | 0.04 | 0.80 | 0.31 | 0.69 |
| Gemma2 9B CPT SahabatAI | 0.49 | 0.70 | 0.61 | 0.66 | 0.07 | 0.89 | 0.11 | 0.60 |
| Llama 3.2 3B | 0.38 | 0.40 | 0.41 | 0.41 | 0.03 | 0.82 | 0.09 | 0.40 |

### How stable is the ranking?

1,000 task-stratified bootstrap samples of the items (seed 3): how often each model ranks first, its median rank and 95 % rank interval, and its paired difference to the leader (kimi-k3). *Significant*: the difference interval excludes zero (no multiplicity adjustment).

| Model | P(rank 1) % | Median rank | Rank 95 % interval | Δ to leader [95 % CI] | Significant |
|---|---:|---:|---:|---:|:---:|
| kimi-k3 | 69.7 | 1.0 | 1–3 | – | – |
| gpt-5.4 | 24.9 | 2.0 | 1–4 | 0.006 [−0.014, 0.027] | no |
| gpt-6.1-sol | 5.3 | 3.0 | 1–4 | 0.010 [−0.010, 0.028] | no |
| claude-sonnet-5.5 | 0.1 | 5.0 | 3–9 | 0.028 [0.006, 0.050] | yes |
| gpt-5.6-terra | 0.0 | 5.0 | 2–10 | 0.028 [0.008, 0.052] | yes |
| gemini-3.7-flash | 0.0 | 6.0 | 4–9 | 0.030 [0.009, 0.048] | yes |
| gemini-4-argon | 0.0 | 8.0 | 5–11 | 0.040 [0.019, 0.060] | yes |
| gpt-6-luna | 0.0 | 8.0 | 4–11 | 0.040 [0.017, 0.067] | yes |
| claude-haiku-5.5 | 0.0 | 9.0 | 6–11 | 0.046 [0.028, 0.064] | yes |
| gemini-3.8-flash | 0.0 | 9.0 | 6–12 | 0.048 [0.024, 0.073] | yes |
| gpt-5.4-mini | 0.0 | 11.0 | 7–12 | 0.058 [0.033, 0.084] | yes |
| claude-sonnet-5 | 0.0 | 12.0 | 10–12 | 0.066 [0.043, 0.088] | yes |
| mai-code-1.1-flash | 0.0 | 13.0 | 13–14 | 0.105 [0.074, 0.135] | yes |
| gpt-5-mini | 0.0 | 14.0 | 13–14 | 0.126 [0.092, 0.161] | yes |
| claude-haiku-4.5 | 0.0 | 15.0 | 15–16 | 0.164 [0.136, 0.195] | yes |
| Qwen3.6 36B | 0.0 | 16.0 | 15–17 | 0.196 [0.167, 0.226] | yes |
| gpt-oss 20B | 0.0 | 17.0 | 16–18 | 0.218 [0.190, 0.246] | yes |
| Gemma 3 27B | 0.0 | 18.0 | 17–19 | 0.250 [0.215, 0.283] | yes |
| Gemma 4 8B | 0.0 | 19.0 | 18–20 | 0.273 [0.236, 0.309] | yes |
| Phi-4 reasoning plus 14.7B | 0.0 | 20.0 | 19–22 | 0.302 [0.269, 0.334] | yes |
| Qwen2.5-Coder 32B | 0.0 | 21.0 | 20–22 | 0.320 [0.286, 0.353] | yes |
| Qwen3-Coder 30B-A3B | 0.0 | 22.0 | 20–22 | 0.323 [0.292, 0.357] | yes |
| Qwen2.5 7B Instruct | 0.0 | 23.0 | 23–24 | 0.423 [0.388, 0.459] | yes |
| Gemma2 9B CPT SahabatAI | 0.0 | 24.0 | 23–24 | 0.424 [0.391, 0.459] | yes |
| Llama 3.2 3B | 0.0 | 25.0 | 25–25 | 0.573 [0.539, 0.605] | yes |

### Sensitivity of the ranking

The overall ranking recomputed under alternative scoring choices, against the published ranking.

| Alternative | Spearman ρ | Largest rank shift | Same leader | Same top four |
|---|---:|---:|:---:|:---:|
| prose scored by the automatic checks only (no LLM judges) | 0.994 | 3 | yes | no |
| prose scored by the LLM judges only | 0.996 | 2 | yes | yes |
| without the private extraction task (public items only) | 0.999 | 1 | yes | yes |
| references found by the second-source audit credited (40 references) | 0.999 | 1 | yes | yes |

### What separates the models

References: share of listed references not found in Crossref, found in none of Crossref, doi.org, Scopus and OpenAlex, and listed with the correct DOI; references listed per topic (5 requested). Code: share of tasks solved by the first program and after one repair turn. LaTeX: broken documents that compile after the fix. Grounded: numbers per review found in no source; reviews with author–year citations (forbidden). Extraction: quoted evidence found verbatim. Outline: chapter IV built around statistical tests where the programme's theses have none.

| Model | Refs not in Crossref % | Refs found nowhere % | Correct DOI % | Refs / topic | Code pass@1 | + repair | LaTeX fix | Unsupported numbers | Author–year % | Verbatim evidence | Generic ch. IV % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi-k3 | 2 | 1 | 89 | 5.0 | 1.00 | 1.00 | 0.88 | 0.03 | 3 | 0.82 | 6 |
| gpt-5.4 | 0 | 0 | 91 | 4.4 | 1.00 | 1.00 | 1.00 | 0.00 | 0 | 0.96 | 23 |
| gpt-6.1-sol | 0 | 0 | 98 | 4.1 | 1.00 | 1.00 | 1.00 | 0.10 | 0 | 1.00 | 26 |
| claude-sonnet-5.5 | 7 | 2 | 84 | 4.1 | 0.97 | 1.00 | 1.00 | 0.10 | 0 | 0.90 | 34 |
| gpt-5.6-terra | 2 | 2 | 89 | 4.2 | 0.97 | 1.00 | 0.88 | 0.10 | 0 | 0.96 | 17 |
| gemini-3.7-flash | 21 | 19 | 63 | 5.0 | 1.00 | 1.00 | 1.00 | 0.00 | 0 | 0.93 | 23 |
| gemini-4-argon | 3 | 1 | 90 | 3.5 | 1.00 | 1.00 | 1.00 | 0.07 | 0 | 0.92 | 26 |
| gpt-6-luna | 8 | 6 | 82 | 4.5 | 1.00 | 1.00 | 0.50 | 0.07 | 0 | 0.88 | 14 |
| claude-haiku-5.5 | 3 | 1 | 83 | 3.8 | 1.00 | 1.00 | 0.88 | 0.07 | 10 | 0.99 | 29 |
| gemini-3.8-flash | 27 | 26 | 62 | 4.5 | 1.00 | 1.00 | 1.00 | 0.10 | 0 | 0.95 | 26 |
| gpt-5.4-mini | 8 | 7 | 57 | 4.4 | 0.97 | 1.00 | 0.75 | 0.00 | 0 | 0.89 | 17 |
| claude-sonnet-5 | 14 | 8 | 66 | 3.0 | 1.00 | 1.00 | 1.00 | 0.00 | 7 | 0.89 | 31 |
| mai-code-1.1-flash | 27 | 22 | 45 | 4.8 | 0.87 | 0.97 | 0.88 | 0.17 | 3 | 0.88 | 14 |
| gpt-5-mini | 2 | 0 | 69 | 2.4 | 0.93 | 0.97 | 0.62 | 0.00 | 3 | 0.94 | 34 |
| claude-haiku-4.5 | 40 | 31 | 25 | 2.8 | 0.83 | 0.97 | 0.50 | 0.07 | 0 | 0.47 | 23 |
| Qwen3.6 36B | 74 | 63 | 9 | 1.8 | 0.80 | 0.90 | 0.62 | 0.00 | 0 | 0.92 | 20 |
| gpt-oss 20B | 86 | 77 | 1 | 3.5 | 0.87 | 0.90 | 0.88 | 0.13 | 43 | 0.68 | 26 |
| Gemma 3 27B | 83 | 70 | 1 | 5.0 | 0.53 | 0.60 | 0.75 | 0.07 | 40 | 0.90 | 20 |
| Gemma 4 8B | 91 | 85 | 0 | 3.4 | 0.43 | 0.73 | 0.62 | 0.00 | 3 | 0.97 | 20 |
| Phi-4 reasoning plus 14.7B | 77 | 61 | 2 | 4.7 | 0.80 | 0.93 | 0.62 | 0.17 | 77 | 0.49 | 14 |
| Qwen2.5-Coder 32B | 90 | 74 | 0 | 4.7 | 0.40 | 0.80 | 0.88 | 0.17 | 87 | 0.82 | 20 |
| Qwen3-Coder 30B-A3B | 90 | 78 | 3 | 5.0 | 0.50 | 0.77 | 0.62 | 0.00 | 67 | 0.67 | 6 |
| Qwen2.5 7B Instruct | 86 | 75 | 0 | 3.2 | 0.13 | 0.30 | 0.50 | 0.10 | 43 | 0.66 | 26 |
| Gemma2 9B CPT SahabatAI | 84 | 73 | 1 | 3.9 | 0.10 | 0.13 | 0.25 | 0.00 | 77 | 0.87 | 26 |
| Llama 3.2 3B | 95 | 85 | 0 | 4.5 | 0.03 | 0.03 | 0.25 | 0.80 | 93 | 0.57 | 6 |

### Reference audit

Every reference that Crossref does not know, looked up again with the same matching rule (title similarity ≥ 0.90, year ± 1, first author): a given DOI at doi.org (all registration agencies), the title in Scopus and OpenAlex. Shares are of the references not found in Crossref.

| Group | Listed | Not in Crossref | Found elsewhere, matching | Title only (wrong year / author) | Found nowhere | Found via |
|---|---:|---:|---:|---:|---:|---|
| API models | 1,206 | 130 (11 %) | 22 (17 %) | 5 (4 %) | 103 (79 %) | openalex 10, scopus 12 |
| local models | 790 | 681 (86 %) | 18 (3 %) | 76 (11 %) | 587 (86 %) | openalex 15, scopus 3 |

### Which tasks discriminate?

Spread of the task scores among API models (ceiling: score ≥ 0.95) and the gap between the API and the local group means.

| Task | API range | API SD | API at ceiling | Local mean | Local range | API − local gap |
|---|---:|---:|---:|---:|---:|---:|
| Outline | 0.57–0.79 | 0.053 | 0 / 15 | 0.54 | 0.38–0.69 | 0.18 |
| Method. | 0.98–1.00 | 0.006 | 15 / 15 | 0.76 | 0.40–1.00 | 0.24 |
| Extract. | 0.85–0.97 | 0.035 | 5 / 15 | 0.66 | 0.41–0.81 | 0.26 |
| Grounded | 0.84–0.96 | 0.034 | 3 / 15 | 0.70 | 0.41–0.89 | 0.23 |
| Refs | 0.23–0.94 | 0.178 | 0 / 15 | 0.06 | 0.03–0.13 | 0.59 |
| Formal ID | 0.92–0.99 | 0.017 | 14 / 15 | 0.89 | 0.80–0.96 | 0.08 |
| Code | 0.86–1.00 | 0.043 | 12 / 15 | 0.50 | 0.09–0.87 | 0.47 |
| LaTeX | 0.79–1.00 | 0.073 | 11 / 15 | 0.78 | 0.40–0.95 | 0.17 |

### LLM judges

Three judges of three vendors rated 1,500 prose answers (3,480 ratings); a judge never rates models of its own vendor. Agreement on the 480 answers that all three rated: Krippendorff's α = 0.86 (interval, mean rating). Judges against the deterministic checks (Spearman ρ): grounded 0.69 across 750 answers, 0.82 across 25 model means; language 0.53 across 750 answers, 0.82 across 25 model means.

| Judge | Mean rating (0–1) | Leniency offset removed |
|---|---:|---:|
| claude-sonnet-5 | 0.845 | 0.046 |
| gemini-4-argon | 0.814 | −0.028 |
| gpt-6.1-sol | 0.784 | −0.018 |

| Task | Criterion | Krippendorff's α (ordinal) |
|---|---|---:|
| grounded | formal Indonesian | 0.74 |
| grounded | coverage | 0.80 |
| grounded | instruction compliance | 0.69 |
| grounded | coherence | 0.74 |
| grounded | faithfulness to the sources | 0.57 |
| language | formal Indonesian | 0.57 |
| language | completeness | 0.76 |
| language | instruction compliance | 0.46 |
| language | coherence | 0.61 |
| language | faithfulness to the summary | 0.54 |

### Extraction silver labels

Agreement of the three-model panel per field over the 30 private excerpts; fields without a majority are not scored.

| Field | Unanimous | Majority | No majority (not scored) |
|---|---:|---:|---:|
| approach | 28 | 2 | 0 |
| data type | 26 | 3 | 1 |
| collection technique | 10 | 18 | 2 |
| analysis technique | 12 | 12 | 6 |
| software | 30 | 0 | 0 |
| sample size | 28 | 1 | 1 |
| hypothesis | 28 | 2 | 0 |

### Cost of the study

API tokens: 3.4 M in, 2.3 M out. Local models: 2,580 calls, 6.1 GPU-hours, 0.98 kWh gross GPU board energy.

| Stage | API calls | AI units | US$ |
|---|---:|---:|---:|
| LLM judges | 3,480 | 2,450 | 24.50 |
| answers (main run) | 3,476 | 1,912 | 19.12 |
| answers (pilot) | 376 | 240 | 2.40 |
| premium cost probes | 24 | 63 | 0.63 |
| **total** | **7,356** | **4,665** | **46.65** |
<!-- results:end -->

---

## Run it

```bash
pip install -e ".[copilot,dev]"                 # Python >= 3.11 for the vendor SDK >= 1.0
export BENCH_PRIVATE_DIR=~/bench-private          # private items, silver/gold labels, raw runs
export BENCH_PYTHON=/path/to/python-with-numpy-pandas-scipy-statsmodels-sklearn   # runs model-written code
# API models: an already running Copilot runtime, or a token for a spawned CLI
export COPILOT_RUNTIME_URL=127.0.0.1:4100 COPILOT_CONNECTION_TOKEN=…   # or COPILOT_GITHUB_TOKEN=…

paper-agent bench check                                    # items load, prompts build, sandbox available
paper-agent bench plan --tiers economy standard            # calls and estimated cost, no model calls
paper-agent bench run  --tiers economy standard local --limit-per-task 3 --stage pilot
paper-agent bench run  --tiers economy standard local      # resumable; re-running only does what is missing
paper-agent bench plan --tiers premium                     # read the estimate first …
paper-agent bench run  --tiers premium --approve-high-cost # … premium tier and high effort are gated
paper-agent bench silver --panel copilot:gpt-6.1-sol@low copilot:claude-sonnet-5@low copilot:gemini-4-argon
paper-agent bench score
paper-agent bench judge                                    # judges of other vendors than the candidate's
paper-agent bench human-sheet && paper-agent bench human-ingest   # two human raters, blind
paper-agent bench report --out experiments/bench
python experiments/bench/audit_refs.py                     # second-source audit of the references not in Crossref
python experiments/bench/analyze.py                        # cost, diagnostics, judges, robustness -> analysis.json
python experiments/bench/make_figures.py && python paper_bench/make_tables.py
```

[`bench/models.yaml`](bench/models.yaml) is the study roster (tier and vendor family per model; local models with their
`think` setting and context cap); any model can be added with `--models engine:model[@effort]`. The background
orchestration used for this study (resumable rounds, cost gate, human-rating import) is described in
[`bench/README.md`](bench/README.md).

### Built on paper-agent

AcademicBench runs on [paper-agent](https://github.com/devnolife/paper-agent/tree/translate)'s engines, which record the served model of every call (the SDK's
usage event for API models, Ollama's response for local ones). Its reference scorer uses paper-agent's verification
rule, and the Indonesian version of the paper was translated by paper-agent's structure-safe `translate-tex`. The tool
is included here because the benchmark needs it; its documentation and its own paper are on the
[`translate`](https://github.com/devnolife/paper-agent/tree/translate) branch.

---

## Paper

**AcademicBench: Which Language Model for Indonesian Academic Writing and Research Code? Quality, Cost and Provenance
of Local and API Models.** [`paper_bench/`](paper_bench/) — IEEE conference format, 8 pages, 28 references (23 verified in ≥ 2 bibliographic
sources, 5 software/web/manuscript references marked `MANUAL`).

<p align="center">
  <a href="paper_bench/build/main.pdf"><img src="docs/img/paper_bench_en_p1-1.png" alt="AcademicBench, English version, page 1" width="44%"></a>
  &nbsp;&nbsp;
  <a href="paper_bench/build/main_id.pdf"><img src="docs/img/paper_bench_id_p1-1.png" alt="AcademicBench, Indonesian version, page 1" width="44%"></a>
</p>

Every number in the text is a macro generated from the run data, so a new run updates the paper by regenerating and
recompiling. The Indonesian version was translated by `paper-agent translate-tex` from the same sections (111 units,
none left untranslated, glossary compliance 94 %, cosine 0.94 to the back-translation; served models GPT-5 mini,
Claude Haiku 4.5, Gemma 3 27B) and then post-edited in 89 places that [`postedit_id.py`](paper_bench/postedit_id.py)
lists and re-applies. Translating it exposed two masking bugs, now fixed and tested: a `\caption{…}` spanning lines
lost its last character and gained a stray `}`, and a `\paragraph{Title}` followed by text on the same line kept
its title untranslated. [`review.md`](paper_bench/review.md) is the output of `paper-agent review` on the English
paper (critics: gpt-6.1-sol and claude-sonnet-5.5 at low effort): four overclaims it found were softened before
release, and its main points — no human validation yet, single runs, the premium tier missing, confounded local cost
— are the limitations the paper states.

---

## Citation

```bibtex
@unpublished{arya2026academicbench,
  author = {Andi Agung Dwi Arya B},
  title  = {AcademicBench: Which Language Model for Indonesian Academic Writing and Research Code?
            Quality, Cost and Provenance of Local and {API} Models},
  year   = {2026},
  note   = {Universitas Hasanuddin. Code and data: \url{https://github.com/devnolife/paper-agent}}
}
```

The provenance check builds on the paper-agent paper:

```bibtex
@unpublished{arya2026paperagent,
  author = {Andi Agung Dwi Arya B},
  title  = {Which Model Actually Answered? Silent Model Substitution, Deterministic Validators,
            and Multi-Model Routing in an {LLM} Paper Agent},
  year   = {2026},
  note   = {Universitas Hasanuddin. Code and data: \url{https://github.com/devnolife/paper-agent}}
}
```

---

## Repository layout

```
bench/                  items (public), model roster, word lists, item generators, MANIFEST.json (hashes of all items)
paper_agent/bench/      runner (cost gate, budget cap, provenance), scorers per task, sandbox, judges,
                        silver labels, human sheets, statistics, report
paper_agent/            the paper-agent package the benchmark runs on (engines, reference verification, translation)
experiments/bench/      results: leaderboard, per-item scores, summary, analyses, reference audit,
                        verify_release.py, make_results_md.py, analyze.py, audit_refs.py, make_figures.py
paper_bench/            the paper: EN and ID sections, generated tables and number macros, figures,
                        verified bibliography, post-edits of the ID version, both PDFs, review
tests/                  deterministic tests (no network, no models)
```

---

## Limitations

One run per model and item at low reasoning effort, so run-to-run variation is not measured; the structure
references and the extraction excerpts come from the theses of one university; extraction is scored against silver
labels that differ by vendor; the LLM judges and the item keys have not yet been checked by people; reference
verification depends on the coverage of Crossref, doi.org, Scopus and OpenAlex; the served-model check relies on the
SDK's own report; local energy was measured on a shared server and local cost depends on the concurrency used; the
nine premium-tier models are not included.

## License

MIT — see [LICENSE](LICENSE).
