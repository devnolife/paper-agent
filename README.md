# paper-agent

An agent for scientific papers whose **every LLM call records the model that actually answered**.

It grew out of a finding while building a research agent on top of the GitHub Copilot SDK: the Copilot
CLI accepts *any* model name and, when the account is not entitled to it, silently serves another
model (`claude-opus-4.8-fast` requested → `claude-haiku-4.5` served). A pipeline that labels its output
with the *requested* model therefore misreports its own provenance. `paper-agent` reads the
`assistant.usage` event of every call, routes work to models by role, and wraps every model output in
deterministic validators. The accompanying paper (`paper/`) reports the experiments.

## What it does

| Command | Purpose |
|---|---|
| `paper-agent translate-tex` | EN → target-language translation of LaTeX sections. Commands, math, `\cite`/`\ref`, identifiers become placeholders `⟦n⟧`, emphasis becomes `<k>…</k>` tags; the multiset and number-gluing of placeholders is verified per unit. Pipeline `multi` = draft model → post-editor model → back-translation QA (local model) + cross-lingual cosine → per-unit selection. |
| `paper-agent translate-docx` | Translate a Word manuscript in place; run formatting (bold/italic/superscript) preserved via tags. |
| `paper-agent verify-refs` | Verify a reference seed (YAML) against Crossref, OpenAlex, Semantic Scholar, Scopus, arXiv (≥ 2 sources = VERIFIED); emit `references.bib` + report. Citations are never written from memory. |
| `paper-agent check-citations` | Gate: every `\cite` key exists in the `.bib` and is VERIFIED/MANUAL. |
| `paper-agent compile` | `latexmk` build + QA (pages, undefined refs/cites, TODO markers). |
| `paper-agent review` | Two *independent* critics (different models) review the paper against a rubric; findings merged, each tagged with the model that produced it. |
| `paper-agent docx2tex` | Convert a DOCX manuscript into a compilable IEEEtran project (sections, tables, figures, numeric citations → `\cite{refN}`, reference list → `.bib` + verification seed). |
| `paper-agent run "<request>"` | Natural-language task → plan of whitelisted tool calls (planner model, keyword fallback) → execution → report with served models per step. |
| `paper-agent serve` | The same as an HTTP API (FastAPI). |

## Install

```bash
pip install -e ".[copilot,qa,api,dev]"      # from a clone
# system tools: TeX Live (texlive-latex-extra texlive-publishers latexmk; texlive-lang-other for babel bahasa), poppler-utils
```

Configuration is by environment (`.env` in the working directory is loaded):

```ini
COPILOT_GITHUB_TOKEN=github_pat_…     # fine-grained PAT, permission "Copilot Requests" (or `copilot login`)
OLLAMA_URL=http://127.0.0.1:11434
CROSSREF_EMAIL=you@example.org
BROWSER_AGENT_URL=http://127.0.0.1:8780   # optional: Scopus / Semantic Scholar through studio-revisi's browser-agent
BROWSER_AGENT_API_KEY=…
PAPER_AGENT_ROLES={"draft":["copilot","gpt-5-mini"],"review":["copilot","claude-haiku-4.5"],"qa":["ollama","gemma3:27b"]}
```

## Quick start

```bash
paper-agent status
paper-agent translate-tex --src paper/sections --out paper/sections_id --glossary paper/glossary_id.yaml
paper-agent compile paper/main_id.tex
paper-agent verify-refs --seed paper/references_seed.yaml --bib paper/references.bib --report paper/references_report.md
paper-agent check-citations --tex paper/main.tex --bib paper/references.bib --report paper/references_report.md
paper-agent review paper/main.tex
paper-agent docx2tex manuscript.docx -o converted
paper-agent run "terjemahkan paper ke bahasa Indonesia lalu kompilasi" --workdir .
```

Every report (`paper/translation/<pipeline>_report.md`, `review.md`, `run` output) lists
`served_models` — the models that answered, not the ones requested.

## Using it inside another project (e.g. a thesis repo)

```make
paper-translate: ; paper-agent translate-tex --src paper/sections --out paper/sections_id --glossary paper/glossary_id.yaml
paper-id:        ; paper-agent compile paper/main_id.tex
paper-check:     ; paper-agent check-citations --tex paper/main.tex --bib paper/references.bib --report paper/references_report.md
```

A VS Code custom agent that drives these commands from Copilot Chat is in
[`integrations/vscode/paper-agent.agent.md`](integrations/vscode/paper-agent.agent.md) — copy it to
`.github/agents/` of your project.

## Paper

`paper/` contains *"Which Model Actually Answered? Silent Model Substitution, Deterministic Validators and
Multi-Model Routing in an LLM Paper Agent"* (IEEE conference format, English and Indonesian — the Indonesian
version was produced by `paper-agent translate-tex` itself). `experiments/` holds the per-unit metrics of the
three translation configurations compared in the paper.

## Design principles

1. **Provenance over labels** — the served model is read from the provider's usage event; substitution is logged.
2. **Deterministic validators first** — placeholder multisets, number-glued placeholders, glossary hits,
   back-translation-echo detection, citation gates. They caught the errors no model (judge included) noticed.
3. **No single judge** — the post-editor is a different model from the translator; two critics review independently.
4. **Local models for bulk, API models for the final pass** — roles, not hard-coded models (`PAPER_AGENT_ROLES`).

## License

MIT
