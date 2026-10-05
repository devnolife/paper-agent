"""Task agent: a natural-language request -> a plan of whitelisted tool calls -> execution -> report.

The planner is an LLM (role ``planner``) that must answer with JSON naming only tools from
``TOOLS``; arguments that are paths are resolved inside the working directory and rejected
otherwise. When no planner model is reachable (``--no-llm``), a keyword router builds the plan.
Each step's result records the models that actually served it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from . import engines
from .config import roles as default_roles

TOOLS: Dict[str, dict] = {
    "status": {"doc": "engine/model availability", "args": {}},
    "translate_tex": {"doc": "translate LaTeX sections EN->target with the multi-model pipeline",
                      "args": {"src_dir": "path", "out_dir": "path", "pipeline": "multi|single:<engine>:<model>",
                               "glossary": "path?", "target": "language", "qa": "bool"}},
    "translate_docx": {"doc": "translate a DOCX in place (format preserved)",
                       "args": {"src": "path", "out": "path?", "engine": "ollama|copilot", "model": "str",
                                "target": "language", "glossary": "path?"}},
    "verify_refs": {"doc": "verify references of a seed YAML against Crossref/OpenAlex/...; writes .bib + report",
                    "args": {"seed": "path", "bib": "path", "report": "path", "reuse_verified": "bool"}},
    "check_citations": {"doc": "gate: every \\cite key is in the .bib and VERIFIED/MANUAL",
                        "args": {"tex": "path", "bib": "path", "report": "path?"}},
    "compile": {"doc": "latexmk build + QA (pages, undefined refs/cites, TODO markers)", "args": {"main_tex": "path"}},
    "review": {"doc": "two independent LLM critics review the paper; findings merged with provenance",
               "args": {"main_tex": "path", "out": "path?"}},
    "docx2tex": {"doc": "convert a DOCX manuscript into an IEEEtran LaTeX project", "args": {"src": "path", "out_dir": "path"}},
}

PLANNER_SYSTEM = """You plan work for a paper-handling agent. Available tools (name: purpose; arguments):
{tools}
Rules: answer ONLY JSON {{"steps": [{{"tool": "<name>", "args": {{...}}}}, ...], "note": "<one sentence>"}}.
Use only listed tools and argument names. Paths are relative to the working directory whose listing is given.
Prefer the obvious minimal plan (e.g. translate then compile the translated main file if one exists).
If the request cannot be done with these tools, return {{"steps": [], "note": "<why>"}}."""


def _ls(workdir: Path, max_entries: int = 120) -> str:
    out = []
    for p in sorted(workdir.rglob("*")):
        if any(part.startswith(".") or part in ("build", "node_modules", "__pycache__") for part in p.relative_to(workdir).parts):
            continue
        if p.suffix in (".tex", ".bib", ".yaml", ".yml", ".docx", ".md", ".pdf") or p.is_dir():
            out.append(str(p.relative_to(workdir)) + ("/" if p.is_dir() else ""))
        if len(out) >= max_entries:
            break
    return "\n".join(out)


def plan_with_llm(request: str, workdir: Path, roles=None) -> dict:
    roles = roles or default_roles()
    engine, model = roles["planner"]
    tools = "\n".join(f"- {k}: {v['doc']}; args {json.dumps(v['args'])}" for k, v in TOOLS.items())
    user = f"Working directory listing:\n{_ls(workdir)}\n\nRequest: {request}"
    data = engines.chat_json(engine, model, PLANNER_SYSTEM.format(tools=tools), user, retries=1)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return {"steps": [], "note": "planner returned no valid plan", "planner": engines.last_served()}
    data["planner"] = engines.last_served() or f"{engine}:{model}"
    return data


def plan_by_keywords(request: str, workdir: Path) -> dict:
    """Fallback router without an LLM."""
    r = request.lower()
    steps: List[dict] = []
    main = next((str(p.relative_to(workdir)) for p in workdir.rglob("main.tex")), "paper/main.tex")
    paper_dir = str(Path(main).parent)
    docx = next((str(p.relative_to(workdir)) for p in workdir.rglob("*.docx") if not p.name.startswith("~$")), None)
    if any(k in r for k in ("translate", "terjemah", "indonesia", "bahasa")):
        if docx and ("docx" in r or "word" in r):
            steps.append({"tool": "translate_docx", "args": {"src": docx, "engine": "ollama", "model": "gemma3:27b"}})
        else:
            steps.append({"tool": "translate_tex", "args": {"src_dir": f"{paper_dir}/sections", "out_dir": f"{paper_dir}/sections_id",
                                                            "pipeline": "multi", "glossary": f"{paper_dir}/glossary_id.yaml"}})
            if (workdir / paper_dir / "main_id.tex").exists():
                steps.append({"tool": "compile", "args": {"main_tex": f"{paper_dir}/main_id.tex"}})
    if any(k in r for k in ("verify", "verifikasi", "referen", "bibtex", "daftar pustaka")):
        steps.append({"tool": "verify_refs", "args": {"seed": f"{paper_dir}/references_seed.yaml", "bib": f"{paper_dir}/references.bib",
                                                      "report": f"{paper_dir}/references_report.md", "reuse_verified": True}})
    if any(k in r for k in ("cite", "sitasi", "citation")):
        steps.append({"tool": "check_citations", "args": {"tex": main, "bib": f"{paper_dir}/references.bib",
                                                          "report": f"{paper_dir}/references_report.md"}})
    if any(k in r for k in ("compile", "build", "kompil", "pdf")) and not any(s["tool"] == "compile" for s in steps):
        steps.append({"tool": "compile", "args": {"main_tex": main}})
    if any(k in r for k in ("review", "kritik", "tinjau", "reviewer")):
        steps.append({"tool": "review", "args": {"main_tex": main, "out": f"{paper_dir}/review.md"}})
    if any(k in r for k in ("docx2tex", "convert", "konversi", "ke latex", "to latex")) and docx:
        steps.append({"tool": "docx2tex", "args": {"src": docx, "out_dir": "converted"}})
    if any(k in r for k in ("status", "model", "engine")) and not steps:
        steps.append({"tool": "status", "args": {}})
    return {"steps": steps, "note": "keyword router (no LLM)", "planner": "keywords"}


def _path(workdir: Path, value: Any) -> Path:
    p = (workdir / str(value)).resolve()
    if workdir.resolve() not in p.parents and p != workdir.resolve():
        raise ValueError(f"path escapes the working directory: {value}")
    return p


def execute(plan: dict, workdir: Path, log: Callable[[str], None] = print) -> dict:
    from . import build, citations, docx2tex, references, review, translate_docx, translate_tex
    results = []
    for i, step in enumerate(plan.get("steps", []), 1):
        tool, args = step.get("tool"), dict(step.get("args") or {})
        t0 = time.time()
        before = dict(engines.served_summary())
        log(f"[{i}/{len(plan['steps'])}] {tool} {json.dumps(args, ensure_ascii=False)}")
        try:
            if tool not in TOOLS:
                raise ValueError(f"unknown tool {tool!r}")
            if tool == "status":
                out = engines.engine_status()
            elif tool == "translate_tex":
                out = translate_tex.translate_tex(_path(workdir, args["src_dir"]), _path(workdir, args["out_dir"]),
                                                  pipeline=args.get("pipeline", "multi"),
                                                  glossary=_path(workdir, args["glossary"]) if args.get("glossary") else None,
                                                  target=args.get("target", "Bahasa Indonesia"),
                                                  qa=bool(args.get("qa", True)), log=log)
            elif tool == "translate_docx":
                out = translate_docx.translate_docx(_path(workdir, args["src"]),
                                                    _path(workdir, args["out"]) if args.get("out") else None,
                                                    engine=args.get("engine", "ollama"), model=args.get("model", "gemma3:27b"),
                                                    target=args.get("target", "Indonesian"),
                                                    glossary=_path(workdir, args["glossary"]) if args.get("glossary") else None, log=log)
            elif tool == "verify_refs":
                out = references.verify_references(_path(workdir, args["seed"]), _path(workdir, args["bib"]),
                                                   _path(workdir, args["report"]), reuse_verified=bool(args.get("reuse_verified", True)), log=log)
            elif tool == "check_citations":
                out = citations.check_citations(_path(workdir, args["tex"]), _path(workdir, args["bib"]),
                                                _path(workdir, args["report"]) if args.get("report") else None)
                log(citations.format_result(out))
            elif tool == "compile":
                out = build.compile_tex(_path(workdir, args["main_tex"]))
                log(build.format_result(out))
            elif tool == "review":
                out = review.review_paper(_path(workdir, args["main_tex"]), out=_path(workdir, args["out"]) if args.get("out") else None, log=log)
                out = {"merged": out["merged"], "critics": [r["_model_served"] for r in out["reviews"]], "out": args.get("out")}
            elif tool == "docx2tex":
                out = docx2tex.docx2tex(_path(workdir, args["src"]), _path(workdir, args["out_dir"]))
            else:  # pragma: no cover
                raise ValueError(tool)
            ok = bool(out.get("ok", True)) if isinstance(out, dict) else True
        except Exception as exc:  # noqa: BLE001 - report, continue with the next step
            out, ok = {"error": f"{type(exc).__name__}: {exc}"}, False
            log(f"    error: {out['error']}")
        after = engines.served_summary()
        results.append({"tool": tool, "args": args, "ok": ok, "seconds": round(time.time() - t0, 1),
                        "served_models": {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)},
                        "result": _compact(out)})
    return {"plan": plan, "results": results, "ok": all(r["ok"] for r in results) and bool(results)}


def _compact(out: Any, limit: int = 4000) -> Any:
    s = json.dumps(out, ensure_ascii=False, default=str)
    return out if len(s) <= limit else {"truncated": s[:limit]}


def run(request: str, workdir: Path, use_llm: bool = True, dry_run: bool = False, log=print) -> dict:
    workdir = Path(workdir).resolve()
    plan = plan_with_llm(request, workdir) if use_llm else plan_by_keywords(request, workdir)
    if not plan.get("steps") and use_llm:
        kw = plan_by_keywords(request, workdir)
        if kw["steps"]:
            kw["note"] = f"planner: {plan.get('note')}; fell back to keyword router"
            plan = kw
    log(f"plan ({plan.get('planner')}): {plan.get('note', '')}")
    for s in plan.get("steps", []):
        log(f"  - {s.get('tool')} {json.dumps(s.get('args', {}), ensure_ascii=False)}")
    if dry_run:
        return {"plan": plan, "results": [], "ok": True, "dry_run": True}
    return execute(plan, workdir, log)


def tool_catalogue_markdown() -> str:
    return "\n".join(f"- `{k}` — {v['doc']}; args: {', '.join(f'{a} ({t})' for a, t in v['args'].items()) or '—'}"
                     for k, v in TOOLS.items())


__all__ = ["TOOLS", "run", "execute", "plan_with_llm", "plan_by_keywords", "tool_catalogue_markdown"]
