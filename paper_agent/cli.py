"""Command-line interface: ``paper-agent <command> [options]``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import load_env


def _p(s: Optional[str]) -> Optional[Path]:
    return Path(s) if s else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="paper-agent",
                                 description="Agent for scientific papers: structure-safe translation, reference "
                                             "verification, citation gate, LaTeX build/QA, multi-model review, "
                                             "DOCX->IEEE LaTeX. Every LLM call records the model that actually answered.")
    ap.add_argument("--version", action="version", version=f"paper-agent {__version__}")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the result as JSON")
    sub = ap.add_subparsers(dest="cmd", required=True, parser_class=lambda **kw: argparse.ArgumentParser(parents=[common], **kw))

    s = sub.add_parser("status", help="engines and models available (Ollama tags, Copilot auth)")

    s = sub.add_parser("translate-tex", help="translate paper/sections/*.tex -> paper/sections_id/ (multi-model)")
    s.add_argument("--src", default="paper/sections")
    s.add_argument("--out", default="paper/sections_id")
    s.add_argument("--files", nargs="*", help="section names without .tex (default: all)")
    s.add_argument("--pipeline", default="multi", help="multi | single:<engine>:<model>")
    s.add_argument("--glossary", default=None, help="YAML EN->target glossary")
    s.add_argument("--target", default="Bahasa Indonesia")
    s.add_argument("--batch-chars", type=int, default=2500)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--no-qa", action="store_true", help="skip back-translation + cosine QA")
    s.add_argument("--limit", type=int, default=0, help="first N units only (smoke test)")
    s.add_argument("--dry-run", action="store_true", help="masking statistics + round-trip check only")

    s = sub.add_parser("translate-docx", help="translate a DOCX in place, formatting preserved")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    s.add_argument("--engine", choices=["ollama", "copilot"], default="ollama")
    s.add_argument("--model", default="gemma3:27b")
    s.add_argument("--target", default="Indonesian")
    s.add_argument("--glossary")
    s.add_argument("--batch-chars", type=int, default=3000)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--keep-ids", default="", help="comma-separated unit ids to leave untranslated")
    s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("verify-refs", help="verify a reference seed YAML; write .bib + report")
    s.add_argument("--seed", default="paper/references_seed.yaml")
    s.add_argument("--bib", default="paper/references.bib")
    s.add_argument("--report", default="paper/references_report.md")
    s.add_argument("--only")
    s.add_argument("--no-agent", action="store_true", help="skip the browser-agent, direct APIs only")
    s.add_argument("--reuse-verified", action="store_true")

    s = sub.add_parser("check-citations", help="gate: every \\cite is in the .bib and VERIFIED/MANUAL")
    s.add_argument("--tex", default="paper/main.tex")
    s.add_argument("--bib", default="paper/references.bib")
    s.add_argument("--report", default="paper/references_report.md")
    s.add_argument("--allow-single", action="store_true")

    s = sub.add_parser("compile", help="latexmk build + QA")
    s.add_argument("main_tex", nargs="?", default="paper/main.tex")
    s.add_argument("--clean", action="store_true")
    s.add_argument("--outdir", default="build")

    s = sub.add_parser("review", help="two independent LLM critics review the paper (provenance recorded)")
    s.add_argument("main_tex", nargs="?", default="paper/main.tex")
    s.add_argument("-o", "--out", default=None, help="markdown report path (default: <paper dir>/review.md)")

    s = sub.add_parser("docx2tex", help="convert a DOCX manuscript to an IEEEtran LaTeX project")
    s.add_argument("input")
    s.add_argument("-o", "--out-dir", default="converted")
    s.add_argument("--title")

    s = sub.add_parser("run", help="natural-language task -> plan of tool calls -> execution")
    s.add_argument("request")
    s.add_argument("--workdir", default=".")
    s.add_argument("--no-llm", action="store_true", help="keyword router instead of the planner model")
    s.add_argument("--dry-run", action="store_true", help="show the plan only")

    s = sub.add_parser("serve", help="run the HTTP API (FastAPI)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8790)

    args = ap.parse_args(argv)
    load_env()
    log = (lambda *_a, **_k: None) if args.json else (lambda m="": print(m, file=sys.stderr, flush=True))

    if args.cmd == "status":
        from .engines import engine_status
        res = engine_status()
    elif args.cmd == "translate-tex":
        from . import translate_tex as tt
        if args.dry_run:
            res = tt.dry_run(Path(args.src), args.files)
        else:
            res = tt.translate_tex(Path(args.src), Path(args.out), args.files, args.pipeline, _p(args.glossary),
                                   args.target, args.batch_chars, args.workers, not args.no_qa, args.limit, log=log)
    elif args.cmd == "translate-docx":
        from .translate_docx import translate_docx
        keep = {int(x) for x in args.keep_ids.split(",") if x.strip()}
        res = translate_docx(Path(args.input), _p(args.output), args.engine, args.model, args.target, _p(args.glossary),
                             args.batch_chars, args.workers, args.limit, keep, args.dry_run, log=log)
    elif args.cmd == "verify-refs":
        from .references import verify_references
        res = verify_references(Path(args.seed), Path(args.bib), Path(args.report), args.only, not args.no_agent,
                                args.reuse_verified, log=log)
    elif args.cmd == "check-citations":
        from .citations import check_citations, format_result
        res = check_citations(Path(args.tex), Path(args.bib), _p(args.report), args.allow_single)
        log(format_result(res))
        if not res["ok"] and not args.json:
            return 1
    elif args.cmd == "compile":
        from .build import compile_tex, format_result
        res = compile_tex(Path(args.main_tex), args.outdir, args.clean)
        log(format_result(res))
        if not res.get("ok") and not args.json:
            return 1
    elif args.cmd == "review":
        from .review import review_paper
        out = _p(args.out) or Path(args.main_tex).parent / "review.md"
        res = review_paper(Path(args.main_tex), out=out, log=log)
        log(f"wrote {out}")
        res = {"out": str(out), "merged": res["merged"], "critics": [r["_model_served"] for r in res["reviews"]]}
    elif args.cmd == "docx2tex":
        from .docx2tex import docx2tex
        res = docx2tex(Path(args.input), Path(args.out_dir), args.title)
        log(f"wrote {res['out_dir']}: {res['headings']} headings, {res['paragraphs']} paragraphs, {res['tables']} tables, "
            f"{res['figures']} figures, {res['references']} references")
    elif args.cmd == "run":
        from .agent import run
        res = run(args.request, Path(args.workdir), use_llm=not args.no_llm, dry_run=args.dry_run, log=log)
        if not res.get("ok") and not args.json:
            print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
            return 1
    elif args.cmd == "serve":
        from .api import serve
        serve(args.host, args.port)
        return 0
    else:  # pragma: no cover
        ap.error("unknown command")
        return 2
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
