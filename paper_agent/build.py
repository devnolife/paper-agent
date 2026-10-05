"""Compile a LaTeX paper with latexmk and run deterministic QA on the result."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def compile_tex(main_tex: Path, outdir: str = "build", clean: bool = False, timeout: int = 600) -> dict:
    main_tex = Path(main_tex).resolve()
    cwd = main_tex.parent
    out = cwd / outdir
    if clean and out.exists():
        shutil.rmtree(out)
    if not shutil.which("latexmk"):
        return {"ok": False, "error": "latexmk not found (install TeX Live: texlive-latex-extra texlive-publishers latexmk)"}
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={outdir}", main_tex.name]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    pdf = out / main_tex.with_suffix(".pdf").name
    log = out / main_tex.with_suffix(".log").name
    res = {"ok": proc.returncode == 0 and pdf.exists(), "pdf": str(pdf) if pdf.exists() else None,
           "log": str(log) if log.exists() else None, "returncode": proc.returncode}
    if not res["ok"]:
        err = re.findall(r"^!.*$", proc.stdout + proc.stderr, re.MULTILINE)
        res["error"] = "\n".join(err[:10]) or (proc.stdout + proc.stderr)[-2000:]
    if log.exists():
        res.update(qa_log(log))
    if pdf.exists():
        res.update(qa_pdf(pdf))
    return res


def qa_log(log_path: Path) -> dict:
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    return {"undefined_refs": len(re.findall(r"Reference `[^']+' on page \d+ undefined", text)),
            "undefined_cites": len(re.findall(r"Citation `[^']+' on page \d+ undefined", text)),
            "overfull_hboxes": len(re.findall(r"^Overfull \\hbox", text, re.MULTILINE)),
            "multiply_defined": len(re.findall(r"multiply[- ]defined", text, re.IGNORECASE)),
            "todos_in_log": len(re.findall(r"\[TODO:", text))}


def qa_pdf(pdf: Path, todo_marker: str = "[TODO:") -> dict:
    res: dict = {}
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        res["pages"] = int(m.group(1)) if m else None
    if shutil.which("pdftotext"):
        text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        res["todo_markers"] = text.count(todo_marker)
        res["question_marks_refs"] = len(re.findall(r"\[\?\]|\?\?", text))
        res["chars"] = len(text)
    return res


def render_pages(pdf: Path, pages: Optional[List[int]] = None, dpi: int = 70, out_prefix: Optional[Path] = None) -> List[str]:
    """PNG previews via pdftoppm (for visual QA by a multimodal reviewer)."""
    if not shutil.which("pdftoppm"):
        return []
    out_prefix = Path(out_prefix or Path(pdf).with_suffix(""))
    files: List[str] = []
    for p in pages or [1]:
        subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p), "-r", str(dpi), "-png", str(pdf), str(out_prefix)],
                       capture_output=True)
        cands = sorted(out_prefix.parent.glob(f"{out_prefix.name}-*{p}.png"))
        files += [str(c) for c in cands if c.stem.endswith(f"-{p}") or c.stem.endswith(f"-{p:02d}") or c.stem.endswith(f"-{p:03d}")]
    return sorted(set(files))


def format_result(res: dict) -> str:
    if not res.get("ok"):
        return f"BUILD FAILED\n{res.get('error', '')}"
    parts = [f"OK {res['pdf']}"]
    for k in ("pages", "undefined_refs", "undefined_cites", "overfull_hboxes", "todo_markers"):
        if k in res:
            parts.append(f"{k}={res[k]}")
    return "  ".join(parts)
