"""HTTP API (FastAPI) mirroring the CLI. Paths are resolved inside ``PAPER_AGENT_WORKDIR`` (default: cwd).

    paper-agent serve --port 8790
    curl -X POST localhost:8790/v1/run -H 'content-type: application/json' -d '{"request": "translate the paper to Indonesian"}'
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install 'paper-agent[api]'") from exc

from . import __version__, agent, build, citations, engines
from .config import load_env

load_env()
WORKDIR = Path(os.getenv("PAPER_AGENT_WORKDIR") or Path.cwd()).resolve()
app = FastAPI(title="paper-agent", version=__version__)


def _inside(p: str) -> Path:
    full = (WORKDIR / p).resolve()
    if WORKDIR not in full.parents and full != WORKDIR:
        raise HTTPException(400, f"path escapes workdir: {p}")
    return full


class RunRequest(BaseModel):
    request: str
    use_llm: bool = True
    dry_run: bool = False


class TranslateTexRequest(BaseModel):
    src_dir: str = "paper/sections"
    out_dir: str = "paper/sections_id"
    pipeline: str = "multi"
    glossary: Optional[str] = None
    target: str = "Bahasa Indonesia"
    qa: bool = True


class PathRequest(BaseModel):
    main_tex: str = "paper/main.tex"


class CitationsRequest(BaseModel):
    tex: str = "paper/main.tex"
    bib: str = "paper/references.bib"
    report: Optional[str] = "paper/references_report.md"


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "workdir": str(WORKDIR), "served_models": engines.served_summary()}


@app.get("/v1/status")
def status() -> dict:
    return engines.engine_status()


@app.get("/v1/tools")
def tools() -> dict:
    return agent.TOOLS


@app.post("/v1/run")
def run(req: RunRequest) -> dict:
    return agent.run(req.request, WORKDIR, use_llm=req.use_llm, dry_run=req.dry_run, log=lambda *_: None)


@app.post("/v1/translate/tex")
def translate_tex(req: TranslateTexRequest) -> dict:
    from .translate_tex import translate_tex as tt
    return tt(_inside(req.src_dir), _inside(req.out_dir), pipeline=req.pipeline,
              glossary=_inside(req.glossary) if req.glossary else None, target=req.target, qa=req.qa, log=lambda *_: None)


@app.post("/v1/compile")
def compile_paper(req: PathRequest) -> dict:
    return build.compile_tex(_inside(req.main_tex))


@app.post("/v1/citations/check")
def check(req: CitationsRequest) -> dict:
    return citations.check_citations(_inside(req.tex), _inside(req.bib), _inside(req.report) if req.report else None)


@app.post("/v1/review")
def review(req: PathRequest) -> dict:
    from .review import review_paper
    res = review_paper(_inside(req.main_tex), out=_inside(req.main_tex).parent / "review.md", log=lambda *_: None)
    return {"merged": res["merged"], "critics": [r["_model_served"] for r in res["reviews"]]}


def serve(host: str = "127.0.0.1", port: int = 8790) -> None:  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host=host, port=port)
