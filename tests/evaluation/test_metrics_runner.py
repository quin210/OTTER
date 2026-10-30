import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from otter.evaluation import stats
from otter.evaluation.aggregate import build_tables
from otter.evaluation.dummy_systems import EvalDirect, OracleSystem, get_dummy
from otter.evaluation.mcq import load_mcq, load_mcq_index, validate_mcq
from otter.evaluation.metrics import citation_check, evaluate, extract_choice, load_answers
from otter.evaluation.runner import import_external, run_system
from otter.llm.interface import MockLLM
from otter.schemas.reasoning import AnswerClaim, Citation, FinalAnswer

ROOT = Path(__file__).resolve().parents[2]


def test_chi2_and_binom():
    # chi2 sf(7.815, 3) ~ 0.05 ; sf(0, k) = 1
    assert abs(stats.chi2_sf(7.8147, 3) - 0.05) < 1e-3
    assert abs(stats.chi2_sf(3.8415, 1) - 0.05) < 1e-3
    assert stats.chi2_sf(0, 3) == 1.0
    assert abs(stats.binom_test_two_sided(0, 10) - 2 * 0.5 ** 10) < 1e-12
    m = stats.mcnemar_exact([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
    assert m["a_only"] == 10 and m["b_only"] == 0 and m["p_value"] < 0.01


def test_bootstrap_ci_contains_mean():
    v = [1] * 70 + [0] * 30
    m, lo, hi = stats.bootstrap_ci(v, n_boot=2000)
    assert m == 0.7 and lo < 0.7 < hi and hi - lo < 0.25
    pb = stats.paired_bootstrap(v, [0] * 100, n_boot=2000)
    assert pb["diff"] == 0.7 and pb["p_value"] < 0.01
    pg = stats.paired_bootstrap(v, [0] * 100, n_boot=2000, groups=[str(i % 7) for i in range(100)])
    assert pg["lo"] > 0




def test_validator_flags_errors(tmp_path):
    rows = [{"id": "X", "domain": "physics", "question": "q?", "options": {"A": "a", "B": "a", "C": "c", "D": "d"}, "answer": "E"},
            {"id": "X", "domain": "physics", "question": "q?", "options": {"A": "a", "B": "b", "C": "c"}, "answer": "A"}]
    p = tmp_path / "b.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    rep = validate_mcq(load_mcq(p))
    checks = {e["check"] for e in rep["errors"]}
    assert {"duplicate_id", "duplicate_option_text", "answer_not_single_letter", "options_keys"} <= checks
    assert rep["exact_duplicate_questions"] == [["X", "X"]]


def test_extract_choice():
    assert extract_choice("Đáp án đúng là C vì ...") == "C"
    assert extract_choice("B. Niutơn") == "B"
    assert extract_choice("không biết") is None


def test_citation_check(corpus):
    ok = Citation(book_id="bk_phys", span_id="bk_phys:p0001:u001", page_number="11", subject="physics", grade=10)
    assert citation_check(ok, corpus)["valid"]
    invented = Citation(book_id="bk_phys", span_id="bk_phys:p0001:u002", page_number="11")  # page not verified
    r = citation_check(invented, corpus)
    assert not r["valid"] and r["invented_page"]
    unknown = Citation(book_id="bk_phys", span_id="bk_phys:p9999:u001")
    assert not citation_check(unknown, corpus)["span_exists"]
    wrong_grade = Citation(book_id="bk_bio", span_id="bk_bio:p0001:u001", grade=10)
    assert not citation_check(wrong_grade, corpus)["valid"]
    pdfp = Citation(book_id="bk_bio", span_id="bk_bio:p0001:u001", pdf_page_number=2)  # index 1 -> 1-based 2
    assert citation_check(pdfp, corpus)["valid"]


def test_evaluate_oracle_and_first(bench_path, corpus):
    items = load_mcq(bench_path)
    idx = {it.id: it for it in items}
    oracle = OracleSystem(idx, corpus)
    ans = {it.id: oracle.answer(it.as_system_input()) for it in items}
    res = evaluate(items, ans, corpus=corpus, n_boot=500)
    o = res["overall"]
    assert o["accuracy"] == 1.0 and o["citation_validity"] == 1.0 and o["ev_recall"] == 1.0 and o["ev_precision"] == 1.0
    first = get_dummy("first")
    res2 = evaluate(items, {it.id: first.answer(it.as_system_input()) for it in items}, corpus=corpus, n_boot=500)
    assert res2["overall"]["accuracy"] == 0.25 and res2["by_domain"]["biology"]["accuracy"] == 0.5
    # missing answers count as wrong
    res3 = evaluate(items, {}, corpus=corpus, n_boot=200)
    assert res3["overall"]["accuracy"] == 0 and res3["overall"]["missing"] == 4


class Boom:
    name = "boom"

    def answer(self, item):
        if item["id"] == "Q2":
            raise RuntimeError("kaput")
        return FinalAnswer(question_id=item["id"], system=self.name, answer="A", answer_choice="A")


def test_runner_resume_and_errors(bench_path, tmp_path):
    items = load_mcq(bench_path)
    out = tmp_path / "run"
    meta = run_system(Boom(), items, out, workers=3)
    assert meta["n_answers"] == 4 and meta["n_failed"] == 1
    a = load_answers(out / "answers.jsonl")
    assert "kaput" in a["Q2"].error and a["Q1"].tool_usage["wall_latency_ms"] >= 0
    meta2 = run_system(Boom(), items, out, workers=2)          # resume: nothing to do
    assert meta2["n_todo"] == 0
    meta3 = run_system(get_dummy("first"), items, out, retry_failed=True)
    assert meta3["n_todo"] == 1 and meta3["n_failed"] == 0


def test_eval_direct_with_mock_llm(bench_path):
    llm = MockLLM(handler=lambda m: '{"answer": "B", "confidence": 90}')
    s = EvalDirect(llm=llm)
    fa = s.answer(load_mcq(bench_path)[0].as_system_input())
    assert fa.answer_choice == "B" and fa.tool_usage["llm_calls"] == 1
    bad = EvalDirect(llm=MockLLM(handler=lambda m: "garbage"))
    assert bad.answer(load_mcq(bench_path)[0].as_system_input()).status == "failed"


def test_import_external(bench_path, tmp_path):
    items = load_mcq(bench_path)
    src = tmp_path / "ext.jsonl"
    src.write_text(FinalAnswer(question_id="Q1", system="ext", answer="B", answer_choice="B").model_dump_json() + "\nnot json\n")
    meta = import_external(src, items, tmp_path / "out")
    assert meta["n_answers"] == 1 and meta["invalid_lines"] == 1 and meta["n_missing"] == 3


def test_cli_run_and_aggregate(bench_path, tmp_path):
    env = {"OTTER_LLM_MOCK": "1", "PATH": "/usr/bin:/bin", "OTTER_EVAL_LOG_DIR": str(tmp_path / "logs")}
    exp = tmp_path / "runs" / "exp1"
    for spec, name in (("dummy:oracle", None), ("dummy:first", None), ("dummy:random", None)):
        out = exp / spec.replace("dummy:", "dummy_") / "main"
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval/run.py"), "--system", spec, "--bench", str(bench_path),
                            "--out", str(out), "--workers", "2"], env=env, capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert (out / "scores.json").exists()
    # an ablation variant of the oracle
    out = exp / "dummy_oracle+no_ws=True" / "main"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/eval/run.py"), "--system", "dummy:first", "--bench", str(bench_path),
                        "--out", str(out)], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    w = build_tables(exp, tmp_path / "tables", ref_system="dummy_oracle", order=["dummy_first", "dummy_oracle"])
    assert {"main", "subject", "cost", "ablation", "significance"} <= set(w)
    tex = (tmp_path / "tables" / "exp1_main.tex").read_text()
    assert r"\toprule" in tex and r"\textbf{100.0" in tex
    main_csv = (tmp_path / "tables" / "exp1_main.csv").read_text().splitlines()
    assert main_csv[1].startswith("dummy_first")


def test_combined_table(bench_path, tmp_path):
    from otter.evaluation.aggregate import build_combined
    env = {"OTTER_LLM_MOCK": "1", "PATH": "/usr/bin:/bin", "OTTER_EVAL_LOG_DIR": str(tmp_path / "logs")}
    for exp in ("b1", "b2"):
        for spec in ("dummy:oracle", "dummy:first"):
            out = tmp_path / "runs" / exp / spec.replace("dummy:", "dummy_") / "small"
            r = subprocess.run([sys.executable, str(ROOT / "scripts/eval/run.py"), "--system", spec, "--bench",
                                str(bench_path), "--out", str(out)], env=env, capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, r.stderr
    p = build_combined([("B1", tmp_path / "runs/b1"), ("B2", tmp_path / "runs/b2")], tmp_path / "tables",
                       order=["dummy_first", "dummy_oracle"])
    lines = p.read_text().splitlines()
    assert lines[0].startswith("System,Backbone,B1 Acc.,B2 Acc.") and lines[1].startswith("dummy_first")
    assert "booktabs" not in p.with_suffix(".tex").read_text() and r"\toprule" in p.with_suffix(".tex").read_text()


def test_gold_context_system(bench_path, corpus):
    items = load_mcq(bench_path)
    idx = {it.id: it for it in items}
    seen = []
    def h(m):
        seen.append(m[-1]["content"])
        return '{"answer": "B", "confidence": 80}'
    s = get_dummy("gold_context", bench=idx, corpus=corpus, llm=MockLLM(handler=h))
    fa = s.answer(items[0].as_system_input())
    assert fa.answer_choice == "B" and fa.claims[0].citations[0].span_id == "bk_phys:p0001:u001"
    assert "niutơn" in seen[0]


