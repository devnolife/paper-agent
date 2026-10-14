# SkripsiBench

**Which model should an Indonesian student or lecturer use for thesis work — and which one for the research code?**
Vendor leaderboards do not measure this. SkripsiBench does, on eight tasks taken from the *skripsi* (Indonesian
undergraduate thesis) workflow, for local models (Ollama) and API models (vendor SDK) under the same prompts, with
scores that are deterministic wherever possible, and with the cost and the model that actually answered recorded
for every call.

| Task | What the model does | Items | Scored by |
|---|---|---:|---|
| `structure` | outline chapter III (methods) and IV (results) for a thesis title in a study programme | 35 | prevalence-weighted F1 of sub-section families vs. the programme's typical structure (aggregated from a local thesis dataset) |
| `methodology` | multiple-choice questions on research methods and statistics | 60 | accuracy |
| `extraction` | fill a closed-vocabulary methods record from a masked thesis excerpt (**private items**) | 30 | field agreement with cross-vendor silver labels (no model graded by its own vendor) + verbatim-evidence check |
| `grounded` | literature review from six given abstracts, citing `[P1]…[P6]` | 30 | citation coverage/validity, sentence citation rate, unsupported numbers, length; two LLM judges |
| `references` | list five real references with DOI for a topic | 20 | Crossref/OpenAlex/Semantic Scholar verification (hallucination rate, DOI correctness) |
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
  *overall* (eight). CIs: percentile bootstrap over items (stratified by task for composites); top models are
  compared item by item (Wilcoxon signed-rank with Holm correction, sign test).

## Limitations

One run per model at low reasoning effort (a small high-effort check on the top models); one university's theses
behind the `structure` reference and the `extraction` items; silver labels for `extraction` (a human-verified
subset estimates their accuracy); LLM judges for prose, calibrated against only two raters; vendor model versions
change, so results are dated (see `summary.json`).
