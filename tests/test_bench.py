"""AcademicBench: deterministic tests (no network, no models; engines are monkeypatched)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_agent import engines
from paper_agent.bench import power, runner, sandbox, schema, stats
from paper_agent.bench.models import ModelSpec, family_of, load_roster, select
from paper_agent.bench.tasks import Context, code_block, number_forms, parse_json_loose
from paper_agent.bench.tasks import coding, extraction, grounded, language, latex, methodology, references, structure

REPO = Path(__file__).resolve().parents[1]


def mcq(i: int, answer: str = "B") -> dict:
    return {"id": f"methodology-{i:03d}", "task": "methodology", "privacy": "public",
            "input": {"question": f"Soal {i}?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}},
            "reference": {"answer": answer}, "meta": {}}


# ----------------------------------------------------------------------------- schema & models
def test_items_manifest_roundtrip(tmp_path: Path):
    schema.write_jsonl(tmp_path / "methodology.jsonl", [mcq(1), mcq(2)])
    items = schema.load_items([tmp_path], ["methodology"])
    man = schema.build_manifest(items)
    assert man["tasks"]["methodology"]["public"] == 2 and schema.check_manifest(man, items) == []
    items[0].input["question"] = "changed"
    assert schema.check_manifest(man, items) == ["item changed methodology-001"]
    schema.write_jsonl(tmp_path / "methodology.jsonl", [mcq(1), mcq(1)])
    with pytest.raises(ValueError):
        schema.load_items([tmp_path], ["methodology"])


def test_model_specs_and_roster():
    s = ModelSpec.parse("ollama:qwen2.5:7b-instruct")
    assert (s.engine, s.model, s.effort) == ("ollama", "qwen2.5:7b-instruct", None)
    s = ModelSpec.parse("copilot:gpt-5-mini@high")
    assert (s.model, s.effort, s.label) == ("gpt-5-mini", "high", "copilot:gpt-5-mini@high")
    assert "/" not in ModelSpec.parse("ollama:hf.co/x/y-GGUF:Q8_0").slug
    assert [family_of(m) for m in ("claude-opus-5", "gpt-oss", "gemma3:27b", "kimi-k3", "phi4-reasoning:plus")] == \
        ["anthropic", "openai", "google", "moonshot", "microsoft"]
    roster = load_roster(REPO / "bench" / "models.yaml")
    tiers = {t: sum(1 for m in roster if m.tier == t) for t in ("economy", "standard", "premium", "local")}
    assert tiers == {"economy": 8, "standard": 7, "premium": 9, "local": 10}
    assert all(m.high_cost for m in roster if m.tier == "premium") and not any(m.high_cost for m in roster if m.tier != "premium")
    assert next(m for m in roster if m.spec.model == "gemini-4-argon").spec.effort is None
    hi = select(roster, tiers=["standard"], effort="high")
    assert all(m.high_cost for m in hi if m.spec.effort) and any(m.spec.effort is None for m in hi)
    local = next(m for m in roster if m.spec.model == "qwen3.6")
    opts = local.engine_options(512)
    assert opts["think"] is True and opts["num_ctx"] == 16384 and opts["num_predict"] == 512 + 6144


# ----------------------------------------------------------------------------- engines
def test_usage_events_and_thinking_split():
    ev = [SimpleNamespace(model="m1", input_tokens=10, output_tokens=5, reasoning_tokens=2, cache_read_tokens=0,
                          copilot_usage=SimpleNamespace(total_nano_aiu=2.5e9), cost=1.0, time_to_first_token=None,
                          reasoning_effort="low", finish_reason="stop"),
          SimpleNamespace(model="m2", input_tokens=7, output_tokens=3, copilot_usage=None, cost=None)]
    u, served = engines.usage_from_events(ev)
    assert served == "m2" and u.input_tokens == 17 and u.output_tokens == 8 and u.aiu == 2.5 and u.api_calls == 2
    assert engines.split_thinking("<think>hmm</think>\n{\"a\": 1}") == ('{"a": 1}', "hmm")
    assert engines.split_thinking("plain") == ("plain", None)


# ----------------------------------------------------------------------------- stats
def test_krippendorff_reference_values():
    n = None
    units = list(zip([1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n], [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
                     [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n], [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n]))
    got = [round(stats.krippendorff_alpha(units, lv), 3) for lv in ("nominal", "ordinal", "interval")]
    assert got == [0.743, 0.815, 0.849]


def test_stats_helpers():
    assert stats.holm([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert stats.binom_two_sided(3, 15) == pytest.approx(0.03515625)
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40])["rho"] == pytest.approx(1.0)
    assert stats.t_two_sided_p(2.1, 17) == pytest.approx(0.05096, abs=1e-4)
    lo, hi = stats.bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0
    assert stats.pareto_front([(1.0, 0.5, "a"), (2.0, 0.4, "b"), (3.0, 0.9, "c")]) == ["a", "c"]


def test_power_integration():
    pts = [(0.0, 100.0), (1.0, 100.0), (2.0, 200.0), (3.0, 200.0)]
    assert power.integrate(pts, 0.0, 3.0) == pytest.approx(100 + 150 + 200)
    assert power.integrate(pts, 0.5, 1.5) == pytest.approx(50 + 0.5 * 125)
    assert power.integrate(pts, 10.0, 12.0) is None
    assert power.model_gpus({0: (80, 1000, 0), 1: (90, 5000, 0)}, {0: (85, 20000, 0), 1: (90, 5100, 0)}) == [0]


# ----------------------------------------------------------------------------- parsing helpers
def test_parsing_helpers():
    assert parse_json_loose('Berikut:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert parse_json_loose('teks {"b": "x}"} lagi') == {"b": "x}"}
    assert parse_json_loose('{"bab3": ["a"], "bab4": ["b"]]') == {"bab3": ["a"], "bab4": ["b"]}   # wrong closer
    assert parse_json_loose('jawab: {"a": [1, 2') == {"a": [1, 2]}                                  # cut off
    assert parse_json_loose('{"x": "] di string", "y": [1]]') == {"x": "] di string", "y": [1]}
    assert parse_json_loose("tidak ada json") is None
    assert code_block("a\n```python\nprint(1)\n```\nb\n```python\nprint(22222)\n```", ("python",)) == "print(22222)\n"
    assert code_block("```python\nimport x\n", ("python",)) == "import x\n"
    assert number_forms("1.250") == {"1250", "1.25"} and number_forms("0,85") == {"0.85"}
    assert number_forms("85%") == {"85"}


# ----------------------------------------------------------------------------- task scorers
def test_methodology_choice():
    assert methodology.chosen('{"jawaban": "C"}') == "C"
    assert methodology.chosen("Jawaban: (B) karena ...") == "B"
    assert methodology.chosen("A atau D") is None
    it = schema.item_from_dict(mcq(1, "C"))
    assert methodology.MethodologyTask().score(it, {"text": '{"jawaban":"C"}'}, None)["score"] == 1.0


def test_structure_score():
    it = schema.item_from_dict({"id": "structure-001", "task": "structure", "privacy": "public",
                                "input": {"program": "P", "faculty": "F", "title": "T"},
                                "reference": {"bab3": [["Lokasi dan waktu penelitian", 80], ["Populasi dan sampel", 60],
                                                       ["Teknik analisis data", 60]],
                                              "bab4": [["Hasil penelitian", 70], ["Pembahasan", 70]]}})
    ans = {"bab3": ["3.1 Jenis Penelitian", "3.2 Lokasi dan Waktu", "3.3 Populasi dan Sampel", "3.4 Teknik Analisis Data"],
           "bab4": ["4.1 Hasil Penelitian", "4.2 Uji Asumsi Klasik", "4.3 Pembahasan"]}
    res = structure.StructureTask().score(it, {"text": json.dumps(ans)}, None)
    assert res["ch3"]["recall"] == 1.0 and res["ch3"]["precision"] == 0.75
    assert res["generic_ch4"] is True and 0 < res["score"] < 1
    double = {"bab3": json.dumps({"bab3": ans["bab3"]}), "bab4": "4.1 Hasil Penelitian\n4.2 Pembahasan"}
    assert structure.headings(double, "III") == ans["bab3"] and structure.headings(double, "IV")[1] == "4.2 Pembahasan"


def test_extraction_scoring():
    obj = {"pendekatan": "Kuantitatif", "data_jenis": "primer", "teknik_pengumpulan": ["kuesioner", "Dokumentasi"],
           "analisis": ["regresi_linier_berganda", "bukan_kosakata"], "alat": "SPSS", "n_sampel": "96 responden",
           "hipotesis": True, "bukti": {"pendekatan": "penelitian ini kuantitatif", "data": "tidak ada di teks"}}
    got = extraction.normalize(obj)
    assert got["pendekatan"] == "kuantitatif" and got["analisis"] == ["regresi_linier_berganda"]
    assert got["alat"] == ["spss"] and got["n_sampel"] == 96
    ref = {"pendekatan": "kuantitatif", "teknik_pengumpulan": ["kuesioner"], "teknik_pengumpulan__undecided": ["dokumentasi"],
           "n_sampel": 100, "hipotesis": True}
    fs = extraction.field_scores(got, ref)
    assert fs == {"pendekatan": 1.0, "teknik_pengumpulan": 1.0, "n_sampel": 0.0, "hipotesis": 1.0}
    assert extraction.evidence_verbatim(obj, "Jenis: Penelitian ini kuantitatif. Data ...") == 0.5
    assert extraction.reference_for({"*": {"a": 1}, "openai": {"a": 2}}, "openai") == {"a": 2}


def test_grounded_checks():
    text = ("Media sosial berkaitan dengan kecemasan remaja [P1]. Studi lain menemukan efek 0,35 [P2, P3]. "
            "Smith (2020) menyebut 42 persen [P9].")
    assert grounded.cited_numbers(text) == [1, 2, 3, 9]
    n, bad, which = grounded.number_support(text, [{"title": "x", "abstract": "effect of 0.35 was found", "year": 2020}])
    assert (n, bad, which) == (2, 1, ["42"])
    assert sum(len(rx.findall(text)) for rx in grounded._FOREIGN) == 1
    assert grounded.abstract_text({"world": [1], "hello": [0]}) == "hello world"
    assert grounded.words_to_digits("Fifty-two percent; one hundred and twenty-five pupils") == "52 percent; 125 pupils"
    assert grounded.number_support("52% rumah tangga [P1].", [{"title": "", "abstract": "Fifty-two percent of"}])[1] == 0
    assert grounded.clip("Kalimat satu. " * 200, 100).endswith(".")


def test_language_scoring(tmp_path: Path):
    wl = tmp_path / "bench" / "wordlists"
    wl.mkdir(parents=True)
    (wl / "id_nonstandard.tsv").write_text("analisa\tanalisis\ngak\ttidak\n", encoding="utf-8")
    (wl / "id_stopwords.txt").write_text("dan\nyang\ndi\nini\n", encoding="utf-8")
    (wl / "en_stopwords.txt").write_text("the\nand\nof\n", encoding="utf-8")
    it = schema.item_from_dict({"id": "language-001", "task": "language", "privacy": "public",
                                "input": {"kind": "paraphrase", "instruction": "x", "text": "ada 85 orang naik 12,4%",
                                          "summary": None, "min_words": 5, "max_words": 40},
                                "reference": {"numbers": ["85", "12,4%"], "key_terms": ["responden"]}})
    ctx = Context(repo=tmp_path)
    good = language.LanguageTask().score(it, {"text": "Jumlah responden ini 85 orang dan proporsinya naik 12,4% di desa."}, ctx)
    bad = language.LanguageTask().score(it, {"text": "Analisa ini gak pakai the data and of 99 orang."}, ctx)
    assert good["parts"]["numbers_kept"] == 1.0 and good["nonstandard_count"] == 0 and good["score"] > bad["score"]
    assert bad["nonstandard"] == ["analisa", "gak"] and bad["numbers_fabricated"] == ["99"]


def test_coding_compare():
    exp = {"r2": 0.51234, "n": 30, "test": "Mann-Whitney", "coefs": [1.0, 2.0]}
    tol = {"default": {"abs": 1e-4, "rel": 1e-3}}
    out = {"r2": "0.5124", "n": 30, "test": "mann-whitney ", "coefs": [1.0, 2.0005]}
    assert coding.compare(out, exp, tol) == {"r2": True, "n": True, "test": True, "coefs": True}
    assert coding.compare({"r2": 0.6}, exp, tol)["r2"] is False and coding.compare(None, exp, tol)["n"] is False
    assert coding.last_json('log\n{"a": 1}\n') == {"a": 1}


def test_latex_parsers():
    tab = r"""\begin{table}[h]\centering\caption{Hasil}\label{tab:a}
\begin{tabular}{@{}lp{3cm}r@{}}\toprule
Item & Ket & Nilai \\ \midrule
X1 & Valid \& reliabel & 85\% \\ \bottomrule
\end{tabular}\end{table}"""
    cells = latex.tabular_cells(tab)
    assert cells == [["item", "ket", "nilai"], ["x1", "valid & reliabel", "85%"]]
    assert latex.cell_accuracy(cells, [["Item", "Ket", "Nilai"], ["X1", "Valid & reliabel", "85%"]]) == 1.0
    e = latex.parse_bibtex("@article{k1, author = {Hake, R. R. and Doe, J.}, pages = {64--74}, year = 1998}")
    assert e["entry_type"] == "article" and latex.family_names(e["fields"]["author"]) == ["hake", "doe"]
    assert latex.bib_field_ok("pages", "64-74", "64--74") and latex.bib_field_ok("doi", "https://doi.org/10.1/X", "10.1/x")


def test_reference_seeds_and_credit():
    ans = {"rujukan": [{"penulis": ["Hake, Richard"], "tahun": "1998", "judul": "Interactive engagement", "doi": "10.1/x"},
                       {"penulis": "A; B", "tahun": 2020, "judul": "Interactive engagement"},
                       {"judul": "Second", "tahun": None}]}
    seeds = references.seeds_from_answer(ans, 5)
    assert [s["title"] for s in seeds] == ["Interactive engagement", "Second"]
    assert seeds[0]["authors"] == ["Richard Hake"] and seeds[0]["year"] == 1998 and seeds[1]["year"] is None
    assert references.credit(True, True, True) == 1.0 and references.credit(True, False, False) == 0.75
    assert references.credit(True, True, False) == 0.5 and references.credit(False, True, True) == 0.0


def test_crossref_client_separates_missing_from_unavailable(tmp_path: Path, monkeypatch):
    import requests
    monkeypatch.setenv("PAPER_AGENT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(references.time, "sleep", lambda *_: None)
    replies = {"a": [404], "b": [429, 200], "c": [500] * 10}
    calls = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        k = url.rsplit("/", 1)[-1]
        calls[k] = calls.get(k, 0) + 1
        code = replies[k][min(calls[k], len(replies[k])) - 1]
        return SimpleNamespace(status_code=code, ok=code == 200, headers={},
                               json=lambda: {"message": {"title": ["T"], "DOI": "10.1/b"}})
    monkeypatch.setattr(requests, "get", fake_get)
    cr = references.Crossref()
    assert cr.get("https://x/a") is None and cr.get("https://x/a") is None and calls["a"] == 1   # cached 404
    assert cr.get("https://x/b")["message"]["DOI"] == "10.1/b" and calls["b"] == 2              # retried 429
    with pytest.raises(references.Unavailable):
        cr.get("https://x/c")


# ----------------------------------------------------------------------------- runner (fake engine)
def _fake_reply(engine, model, system, user, timeout=600, json_mode=True, effort=None, options=None):
    return engines.Reply('{"jawaban": "B"}', engine, model, model, effort, 0.01,
                         engines.Usage(input_tokens=10, output_tokens=2, nano_aiu=1e8, api_calls=1))


def test_runner_resume_gate_score_report(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(engines, "chat_reply", _fake_reply)
    items_dir = tmp_path / "items"
    schema.write_jsonl(items_dir / "methodology.jsonl", [mcq(1, "B"), mcq(2, "A"), mcq(3, "B")])
    items = schema.load_items([items_dir])
    roster = load_roster(REPO / "bench" / "models.yaml")
    models = select(roster, ["copilot:gpt-5-mini@low"])
    ctx = Context(repo=REPO)
    cfg = runner.RunConfig(run_dir=tmp_path / "run", workers=2)
    out = runner.run(items, models, ctx, cfg, log=lambda *_: None)
    assert out["api"]["calls"] == 3 and out["api"]["ok"] == 3 and out["api"]["aiu"] == pytest.approx(0.3)
    again = runner.run(items, models, ctx, cfg, log=lambda *_: None)
    assert again["api"]["calls"] == 0                                  # resumed: nothing left
    with pytest.raises(runner.GateError):
        runner.run(items, select(roster, ["copilot:claude-opus-5@low"]), ctx, cfg, log=lambda *_: None)
    from paper_agent.bench import report, score
    tot = score.score_runs(items, ctx, cfg.run_dir, log=lambda *_: None)
    assert tot["scored"] == 3
    summ = report.build_summary(items, cfg.run_dir)
    t = summ["models"]["copilot:gpt-5-mini@low"]["tasks"]["methodology"]
    assert t["n"] == 3 and t["mean"] == pytest.approx(2 / 3, abs=1e-3)
    assert summ["models"]["copilot:gpt-5-mini@low"]["cost"]["aiu"] == pytest.approx(0.3)


@pytest.mark.skipif(not sandbox.available(), reason="needs Linux user namespaces")
def test_sandbox_blocks_network_and_keeps_env_clean(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "x")
    code = ("import os, socket, json\nok = True\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
            "except OSError:\n    ok = False\nprint(json.dumps({'net': ok, 'secret': 'SECRET_TOKEN' in os.environ, "
            "'data': open('d.txt').read()}))")
    r = sandbox.run_python(code, {"d.txt": b"hi"}, sandbox.default_python(), timeout=60)
    assert r.ok and json.loads(r.stdout) == {"net": False, "secret": False, "data": "hi"}


@pytest.mark.skipif(not sandbox.available(), reason="needs Linux user namespaces")
def test_sandbox_program_cannot_unmount_the_hiding_layers():
    code = ("import ctypes, json, os\nlibc = ctypes.CDLL(None, use_errno=True)\n"
            "rcs = [libc.umount2(d.encode(), 2) for d in ('/home', '/run', '/tmp', '/proc')]\n"
            "cap = [l for l in open('/proc/self/status') if l.startswith('CapEff')][0].split()[1]\n"
            "print(json.dumps({'rcs': rcs, 'cap': cap, 'run': os.listdir('/run'), 'home': os.listdir('/home')}))")
    r = sandbox.run_python(code, {}, sandbox.default_python(), timeout=60)
    out = json.loads(r.stdout)
    assert out["rcs"] == [-1, -1, -1, -1] and int(out["cap"], 16) == 0
    assert "docker.sock" not in out["run"] and "sbx" in out["home"]


def test_local_costs_mix_sequential_calls_and_parallel_segments(tmp_path: Path):
    from paper_agent.bench import report
    from paper_agent.bench.cost import Prices
    raw = tmp_path / "raw"
    raw.mkdir()
    base = {"engine": "ollama", "model": "ollama:m", "tier": "local", "family": "meta", "requested": "m", "served": "m",
            "task": "methodology", "rep": 0, "text": "x", "seconds": 1.0, "usage": {}}
    rows = [dict(base, item="a", wall_s=10.0, energy={"gross_j": 3.6e5, "marginal_j": 1.8e5}),   # sequential call
            dict(base, item="b", wall_s=30.0),                                                  # parallel call
            dict(base, item="c", wall_s=30.0)]
    (raw / "ollama__m__default.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    seg = tmp_path / "energy"
    seg.mkdir()
    segs = [{"model": "ollama:m", "t0": 0, "t1": 10, "calls": 1, "workers": 1, "gross_j": 3.6e5},   # duplicate: ignored
            {"model": "ollama:m", "t0": 100, "t1": 120, "calls": 2, "workers": 4, "gross_j": 7.2e5, "marginal_j": 3.6e5}]
    (seg / "ollama__m__default.jsonl").write_text("".join(json.dumps(s) + "\n" for s in segs), encoding="utf-8")
    c = report.costs(tmp_path, Prices(gpu_usd_per_hour=3.6))["ollama:m"]
    assert c["gpu_hours"] == pytest.approx(30 / 3600, abs=1e-4)    # 10 s sequential + 20 s parallel segment
    assert c["kwh_gross"] == pytest.approx(0.3) and c["kwh_per_100"] == pytest.approx(10.0)
    assert c["rental_usd"] == pytest.approx(0.03)
    assert c["timed_calls"] == 3 and c["usd_per_100"] == pytest.approx(1.0)       # US$0.03 over 3 timed calls


def _costly_reply(engine, model, system, user, timeout=600, json_mode=True, effort=None, options=None):
    return engines.Reply('{"jawaban": "B"}', engine, model, model, effort, 0.01,
                         engines.Usage(input_tokens=10, output_tokens=2, nano_aiu=1e9, api_calls=1))   # 1 AIU per call


def test_budget_cap_stops_new_calls(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(engines, "chat_reply", _costly_reply)
    items_dir = tmp_path / "items"
    schema.write_jsonl(items_dir / "methodology.jsonl", [mcq(i) for i in range(1, 11)])
    items = schema.load_items([items_dir])
    models = select(load_roster(REPO / "bench" / "models.yaml"), ["copilot:gpt-5-mini@low"])
    cfg = runner.RunConfig(run_dir=tmp_path / "run", workers=1, budget_usd=0.025)       # 2.5 AIU at US$0.01/AIU
    out = runner.run(items, models, Context(repo=REPO), cfg, log=lambda *_: None)["api"]
    assert out["stopped_by_budget"] and out["ok"] == 3 and out["skipped"] == 7      # one worker: no overshoot
    cfg.budget_usd = None                                                              # resume without a cap
    again = runner.run(items, models, Context(repo=REPO), cfg, log=lambda *_: None)["api"]
    assert again["calls"] == 10 - out["ok"] and not again["stopped_by_budget"]


def test_changed_answer_key_is_rescored(tmp_path: Path, monkeypatch):
    from paper_agent.bench import score
    monkeypatch.setattr(engines, "chat_reply", _fake_reply)                            # always answers B
    items_dir = tmp_path / "items"
    schema.write_jsonl(items_dir / "methodology.jsonl", [mcq(1, "B")])
    cfg = runner.RunConfig(run_dir=tmp_path / "run", workers=1)
    runner.run(schema.load_items([items_dir]), select(load_roster(REPO / "bench" / "models.yaml"),
                                                       ["copilot:gpt-5-mini@low"]), Context(repo=REPO), cfg,
               log=lambda *_: None)
    score.score_runs(schema.load_items([items_dir]), Context(repo=REPO), cfg.run_dir, log=lambda *_: None)
    schema.write_jsonl(items_dir / "methodology.jsonl", [mcq(1, "A")])                # corrected key, same prompt
    tot = score.score_runs(schema.load_items([items_dir]), Context(repo=REPO), cfg.run_dir, log=lambda *_: None)
    rows = score.load_scores(cfg.run_dir)["copilot:gpt-5-mini@low"]
    assert tot["scored"] == 1 and tot["reused"] == 0 and rows[("methodology-001", 0)]["score"] == 0.0


def test_mount_script_mounts_targets_under_the_temp_root_after_its_tmpfs():
    script = sandbox._mount_script(["/tmp/ci/venv", "/home/u/runtime"], "/tmp/pa-sbx-1/hold", "/tmp/pa-sbx-1/work",
                                   {"A": "b"}).splitlines()
    tmp_tmpfs = script.index('mount -t tmpfs -o size=256m,mode=1777 tmpfs "/tmp"')
    venv = next(i for i, line in enumerate(script) if line.startswith("mount --bind") and line.endswith('"/tmp/ci/venv"'))
    runtime = next(i for i, line in enumerate(script) if line.endswith('"/home/u/runtime"') and "--bind" in line)
    work = next(i for i, line in enumerate(script) if line.endswith(f'"{sandbox.WORK}"'))
    assert venv > tmp_tmpfs > max(runtime, work)
    assert any(line.startswith("umount ") for line in script[venv:])
