# AcademicBench

**Which model should an Indonesian student or lecturer use for thesis work — and which one for the research code?**
Vendor leaderboards do not measure this. AcademicBench does, on eight tasks taken from the *skripsi* (Indonesian
undergraduate thesis) workflow, for local models (Ollama) and API models (vendor SDK) under the same prompts, with
scores that are deterministic wherever possible, and with the cost and the model that actually answered recorded
for every call.

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

Prompts are in Indonesian. Items are new (written for this benchmark, 2026-10-14) except the `extraction` excerpts,
which come from a local undergraduate thesis dataset of one Indonesian university and cannot be redistributed: the
repository holds their SHA-256 in [`MANIFEST.json`](MANIFEST.json), not the text. `grounded` items store OpenAlex
work ids; titles and abstracts are fetched once and frozen in a private snapshot (`sources_sha256` lets you check a
reconstruction).

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

[`models.yaml`](models.yaml) is the study roster (tier and vendor family per model; local models with their
`think` setting and context cap). Any model can be added with `--models engine:model[@effort]`.

**What is recorded per call** (`<run>/raw/<model>.jsonl`): requested and *served* model, reasoning effort, input,
output and reasoning tokens, Copilot AI units (from the SDK usage event), latency, the answer, follow-up turns and
sandbox results, and for local models the GPU, its idle baseline and the measured board energy of the call
(``nvidia-smi`` sampled every 250 ms; gross and minus idle).

**Cost** is reported in US$: API calls as AI units × US$0.01 (`BENCH_USD_PER_AIU`); local runs as GPU time × an
on-demand L40S price (`BENCH_GPU_USD_PER_HOUR`, default source in the run script) and as measured kWh, to be
multiplied by your electricity tariff (`BENCH_TARIFF_PER_KWH`, `BENCH_TARIFF_CURRENCY`).

**Sandbox.** Model-written programs and LaTeX run under `unshare` (user, mount, network and PID namespaces): no
network, fresh tmpfs over `/home`, `/root`, `/tmp`, `/var/tmp`, `/dev/shm`, `/run`, `/srv`, `/mnt`, `/media`, the
interpreter bound read-only, a minimal environment without tokens, a wall-clock timeout, and — with
`systemd-run --user --scope` — memory, task and CPU limits.

## Scoring rules in one place

* An item score is in [0, 1]. A failed model call counts as 0; an item the scorer cannot grade (e.g. no silver label)
  is left out and reported.
* Judged tasks (`grounded`, `language`): 0.5 × deterministic score + 0.5 × mean judge rating (five criteria, 1–5,
  rescaled). Three judges from three vendors rate every answer except those of their own vendor (two or three
  ratings per answer); each judge's leniency, estimated on the answers all three rated, is subtracted so vendor
  families rated by different judge subsets stay comparable. Agreement with two human raters is reported
  (Krippendorff's α between the raters, Spearman's ρ between judges and humans).
* Composite scores are unweighted means of task scores: *academic* (six tasks), *coding* (`coding`, `latex`),
  *overall* (eight). CIs: percentile bootstrap over items (stratified by task for composites). `analyze.py` gives the
  paired difference of every model to the leader and the rank stability from task-stratified resamples (the
  weighting of the overall score); `summary.json` also lists item-level Wilcoxon/sign tests among the top models.
* `references`: credit 1 for a work found in Crossref with the correct DOI, 0.75 without a DOI, 0.5 with a wrong DOI,
  0 if Crossref has no matching record (title similarity ≥ 0.90, year ± 1, first-author surname); Crossref errors
  leave the item unscored. Not found in Crossref is not the same as fabricated, so `experiments/bench/audit_refs.py`
  looks every miss up again — a given DOI at doi.org (other registration agencies), the title in Scopus and OpenAlex
  through the studio-revisi browser-agent (`BROWSER_AGENT_URL`, `BROWSER_AGENT_API_KEY`) — and classifies it as
  *exact* (a real work the score missed), *title* (a real title with wrong year or author), *none* or *unchecked*.
  OpenAlex bills each search against a daily budget (US$1 with a free key, shared by everything that uses the key),
  so the audit makes at most `AUDIT_OPENALEX_DAILY_CAP` searches (default 600) per UTC day and checks the rest on the
  next run. The audit does not change scores; `analyze.py` reports the ranking with the works it finds credited.

## Limitations

One run per model at low reasoning effort (a high-effort check on the top models is gated with the premium tier);
one university's theses behind the `structure` reference and the `extraction` items; silver labels for `extraction`
(a candidate is scored against the panel members of other vendors, so labels differ by vendor; a human-verified
subset is prepared but not yet rated); LLM judges for prose, not yet compared with the two human raters; reference
verification depends on the coverage of Crossref, doi.org, Scopus and OpenAlex (Semantic Scholar refused our
requests during the audit); the served-model check rests on the SDK's own usage event; vendor model versions
change, so results are dated (see `summary.json`).
