import json

from otter.evaluation import answer_judge as aj
from otter.evaluation.aggregate import build_tables
from otter.llm.interface import MockLLM
from otter.schemas.reasoning import AnswerClaim, Citation, FinalAnswer
from tests.evaluation.conftest import mcq_rows


def _fa(qid, choice, *, cited=("bk_phys:p0001:u001",), status="supported", text=None):
    cl = [AnswerClaim(claim_id="c1", text=f"Đáp án {choice}", citations=[Citation(book_id="bk_phys", span_id=s) for s in cited])] if cited else []
    return FinalAnswer(question_id=qid, system="SECRET_SYSTEM_NAME", answer=text or f"Đáp án {choice}. Lực đo bằng niutơn",
                       answer_choice=choice, status=status, claims=cl)


def _reply(c=2, cm=2, g=2):
    return json.dumps({"reason": "ok", "correctness": c, "completeness": cm, "grounding": g})


def test_prompt_hides_system_and_uses_span_text(corpus):
    msgs, info = aj.build_prompt(mcq_rows()[0], _fa("Q1", "B"), corpus)
    p = msgs[0]["content"]
    assert "SECRET_SYSTEM_NAME" not in p
    assert "Lực là đại lượng vectơ" in p and info["n_cited_resolved"] == 1       # text comes from the corpus
    assert "Đáp án đúng: B. Niutơn" in p and "[S1]" in p
    msgs2, info2 = aj.build_prompt(mcq_rows()[0], _fa("Q1", "B", cited=("nope:p9:u9",)), corpus)
    assert info2["n_cited_resolved"] == 0 and "không tìm thấy" in msgs2[0]["content"]


def test_judge_item_scores_and_rules(corpus):
    llm = MockLLM(handler=lambda m: _reply(2, 2, 2))
    r = aj.judge_item(llm, mcq_rows()[0], _fa("Q1", "B"), corpus, judge_id="qwen3-32b")
    assert r["judge_score"] == 1.0 and r["strict"] == 1 and r["rules"] == []
    # wrong letter -> correctness forced to 0 even if the judge says 2; raw value kept
    r = aj.judge_item(llm, mcq_rows()[0], _fa("Q1", "A"), corpus, judge_id="qwen3-32b")
    assert r["correctness"] == 0 and r["correctness_raw"] == 2 and r["strict"] == 0 and r["rules"]
    # no resolvable citation -> grounding 0
    r = aj.judge_item(llm, mcq_rows()[0], _fa("Q1", "B", cited=()), corpus, judge_id="qwen3-32b")
    assert r["grounding"] == 0 and abs(r["judge_score"] - 4 / 6) < 1e-9 and abs(r["ans_score"] - 1.0) < 1e-9
    # failed answer -> zeros, no model call
    boom = MockLLM(handler=lambda m: (_ for _ in ()).throw(AssertionError("must not be called")))
    f = FinalAnswer(question_id="Q1", system="x", answer="", status="failed", error="boom")
    r = aj.judge_item(boom, mcq_rows()[0], f, corpus, judge_id="qwen3-32b")
    assert r["judge_score"] == 0 and r["called"] is False


def test_unanswerable_rules(corpus):
    item = {**mcq_rows()[0], "id": "U1", "answerable": False, "answer": None, "abstain_type": "out_of_corpus"}
    llm = MockLLM(handler=lambda m: _reply(0, 1, 0))
    ab = _fa("U1", "E", cited=(), status="abstain", text="Đáp án E. Không đủ căn cứ")
    r = aj.judge_item(llm, item, ab, corpus, judge_id="qwen3-8b")
    assert r["correctness"] == 2 and r["grounding"] == 0 and r["abstained"]          # abstention keeps the judge's grounding (no no-cite rule)
    r = aj.judge_item(MockLLM(handler=lambda m: _reply(2, 2, 2)), item, _fa("U1", "B"), corpus, judge_id="qwen3-8b")
    assert r["correctness"] == 0


def test_parse_and_retry(corpus):
    seq = ["not json at all", "```json\n" + _reply(1, 2, 0) + "\n```"]
    llm = MockLLM(replies=list(seq))
    r = aj.judge_item(llm, mcq_rows()[0], _fa("Q1", "B"), corpus, judge_id="qwen3-32b")
    assert r["parse_ok"] and r["attempts"] == 2 and (r["correctness"], r["completeness"], r["grounding"]) == (1, 2, 0)
    bad = MockLLM(handler=lambda m: '{"correctness": 5, "completeness": 1, "grounding": 1, "reason": ""}')
    r = aj.judge_item(bad, mcq_rows()[0], _fa("Q1", "B"), corpus, judge_id="qwen3-32b")
    assert not r["parse_ok"] and "out of range" in r["error"]


def test_judge_run_resumable_and_cache(corpus, tmp_path):
    calls = []
    llm = MockLLM(handler=lambda m: calls.append(1) or _reply())
    items = mcq_rows()[:2]
    answers = {"Q1": _fa("Q1", "B"), "Q2": _fa("Q2", "B", cited=("bk_phys:p0001:u002",))}
    out = tmp_path / "answer_judgments.qwen3-32b.jsonl"
    st = aj.judge_run(items, answers, corpus, llm, out, judge_id="qwen3-32b", workers=2)
    assert st["judged"] == 2 and len(calls) == 2
    st = aj.judge_run(items, answers, corpus, llm, out, judge_id="qwen3-32b", workers=2)
    assert st["todo"] == 0 and st["cached"] == 2 and len(calls) == 2                  # nothing re-judged
    answers["Q2"] = _fa("Q2", "B", cited=("bk_phys:p0001:u002",), text="Đáp án B. Khác hẳn")
    st = aj.judge_run(items, answers, corpus, llm, out, judge_id="qwen3-32b", workers=2)
    assert st["judged"] == 1 and len(calls) == 3                                      # changed answer -> new key
    rows = aj.load_judgments(out)
    assert set(rows) == {"Q1", "Q2"}
    s = aj.summarize_judgments(rows, ["Q1", "Q2"], n_boot=200)
    assert s["judge"] == 1.0 and s["strict"] == 1.0 and s["n_judged"] == 2


def test_agreement_stats():
    a = [{"correctness": c, "completeness": c, "grounding": c} for c in (0, 1, 2, 2, 1, 0)]
    assert aj.quadratic_weighted_kappa([0, 1, 2], [0, 1, 2]) == 1.0
    st = aj.agreement_stats(a, a)
    assert st["correctness"]["exact"] == 1.0 and st["correctness"]["mean_diff"] == 0
    b = [{**r, "grounding": min(2, r["grounding"] + 1)} for r in a]
    st = aj.agreement_stats(a, b)
    assert st["grounding"]["mean_diff"] > 0 and st["grounding"]["judge_higher"] > 0 and st["correctness"]["qwk"] == 1.0


def test_aggregate_shows_llm_judge_column(corpus, tmp_path):
    d = tmp_path / "exp" / "sysA" / "main"
    d.mkdir(parents=True)
    recs = [{"id": r["id"], "correct": True, "domain": "physics"} for r in mcq_rows()]
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (d / "scores.json").write_text(json.dumps({"overall": {"n": 4, "accuracy": 1.0, "acc_ci_lo": 1.0, "acc_ci_hi": 1.0}}))
    (d / "run_meta.json").write_text(json.dumps({"system": "sysA", "backbone": "main", "backbone_model": "Qwen/Qwen3-32B"}))
    rows = [{"question_id": r["id"], "judge_version": aj.JUDGE_VERSION, "parse_ok": True, "judge_score": 5 / 6, "ans_score": 1.0,
             "strict": 0, "correctness": 2, "completeness": 2, "grounding": 1} for r in mcq_rows()]
    (d / "answer_judgments.qwen3-32b.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    w = build_tables(tmp_path / "exp", tmp_path / "out")
    txt = w["main"].read_text()
    assert "LLM-J 32B" in txt and "0.833" in txt and "*" in txt           # same model as the backbone is starred
