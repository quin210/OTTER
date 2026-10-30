import json
from pathlib import Path

import pytest

from otter.evaluation.abstain_build import absent_topics, toc_entries
from otter.evaluation.abstain_metrics import (ABSTAIN_KEY, ABSTAIN_OPTION_TEXT, evaluate_abstain, response_kind,
                                              with_abstain_option)
from otter.evaluation.corpus import Span
from otter.evaluation.exclusion import ExcludingStore, current_exclusion, excluded_spans, install_exclusion
from otter.evaluation.mcq import MCQItem, load_mcq, validate_mcq
from otter.evaluation.runner import run_system
from otter.schemas.reasoning import AnswerClaim, Citation, FinalAnswer

ROOT = Path(__file__).resolve().parents[2]
OPTS = {"A": "a1", "B": "b1", "C": "c1", "D": "d1"}


def item(i, answerable, answer=None, excl=(), typ=None, ref=None):
    return MCQItem(id=i, question="q?", options=OPTS, answer=answer, domain="physics", subject="Vật lí", grade=10,
                   answerable=answerable, abstain_type=typ, excluded_span_ids=list(excl), provenance={"reference_key": ref})


def fa(i, choice=None, status="supported", cite=()):
    claims = [AnswerClaim(claim_id="c1", text="x", citations=[Citation(book_id="b", span_id=s) for s in cite])] if cite else []
    return FinalAnswer(question_id=i, system="s", answer=choice or "", answer_choice=choice, status=status, claims=claims)


def test_response_kind():
    assert response_kind(fa("x", "B")) == "answer"
    assert response_kind(fa("x", "E")) == "abstain"
    assert response_kind(fa("x", None, "abstain")) == "abstain"
    assert response_kind(fa("x", "B", "qualified")) == "answer"
    assert response_kind(fa("x", None, "failed")) == "failed"
    assert response_kind(fa("x", None, "supported")) == "failed"       # unparsable
    assert response_kind(None) == "failed"


def test_abstain_metrics_counts():
    items = [item("u1", False, typ="out_of_corpus", ref="A"), item("u2", False, typ="out_of_corpus", ref="A"),
             item("u3", False, typ="missing_premise", excl=["s9"], ref="B"), item("u4", False, typ="missing_premise", ref="B"),
             item("c1", True, "A"), item("c2", True, "B"), item("c3", True, "C"), item("c4", True, "D")]
    ans = {a.question_id: a for a in [
        fa("u1", "E"), fa("u2", "A", cite=["s1"]), fa("u3", "C", cite=["s9"]), fa("u4", None, "abstain"),   # TP FN FN TP
        fa("c1", "A"), fa("c2", "E"), fa("c3", "A"), fa("c4", None, "failed")]}                          # ok, FP, wrong, failed
    m = evaluate_abstain(items, ans)["overall"]
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 2, 1, 2)
    assert m["abstain_precision"] == pytest.approx(2 / 3) and m["abstain_recall"] == 0.5
    assert m["abstain_f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert m["unsupported_inference_rate"] == 0.5 and m["uir_reference_key"] == 0.25   # u2 chose reference key A; u3 chose C != B
    assert m["uir_with_citation"] == 0.5 and m["cited_excluded_span"] == 1
    assert m["over_abstention_rate"] == pytest.approx(1 / 3)
    assert m["control_accuracy"] == 0.25 and m["control_selective_accuracy"] == 0.5     # abstain & failure count wrong
    assert m["n_failed_answerable"] == 1
    r = evaluate_abstain(items, ans)
    assert r["by_abstain_type"]["missing_premise"]["abstain_recall"] == 0.5


def test_abstain_metrics_degenerate():
    items = [item("u1", False, typ="x"), item("c1", True, "A")]
    m = evaluate_abstain(items, {"u1": fa("u1", "A"), "c1": fa("c1", "A")})["overall"]
    assert m["abstain_precision"] is None and m["abstain_recall"] == 0.0 and m["abstain_f1"] is None
    assert m["unsupported_inference_rate"] == 1.0 and m["control_accuracy"] == 1.0


def test_with_abstain_option_and_input():
    it = item("u1", False, excl=["s1"])
    plain = it.as_system_input()
    assert plain["excluded_span_ids"] == ["s1"] and "E" not in plain["options"]
    d = it.as_system_input(abstain_option=True)
    assert d["options"][ABSTAIN_KEY] == ABSTAIN_OPTION_TEXT and ABSTAIN_KEY not in OPTS and "E" in d["question"]
    assert with_abstain_option(plain)["options"]["E"] == ABSTAIN_OPTION_TEXT
    # ordinary benchmark items are unaffected
    old = MCQItem(id="o", question="q", options=OPTS, answer="A", domain="physics")
    assert set(old.as_system_input()) == {"id", "question", "options", "subject", "domain", "grade"}


class FakeStore:
    def __init__(self):
        self.calls = []
        self.spans = [Span(f"s{i}", "b", f"t{i}") for i in range(10)]

    def search_spans(self, query, *, subject=None, grade=None, top_k=10):
        self.calls.append(top_k)
        return [(s, 1.0) for s in self.spans[:top_k]]

    def get_spans(self, ids):
        return [s for s in self.spans if s.span_id in ids]

    def search_knowledge(self, q, **kw):
        return [{"id": 1, "source_span_ids": ["s1"]}, {"id": 2, "source_span_ids": ["s1", "s2"]}, {"id": 3, "source_span_ids": []}]

    def other(self):
        return "passthrough"


def test_excluding_store():
    st = ExcludingStore(FakeStore())
    assert [s.span_id for s, _ in st.search_spans("q", top_k=3)] == ["s0", "s1", "s2"]      # transparent without exclusion
    with excluded_spans(["s1", "s2"]):
        assert current_exclusion() == {"s1", "s2"}
        got = [s.span_id for s, _ in st.search_spans("q", top_k=3)]
        assert got == ["s0", "s3", "s4"] and st._inner.calls[-1] == 5                       # over-fetch keeps top-k full
        assert [s.span_id for s in st.get_spans(["s1", "s3"])] == ["s3"]
        assert [h["id"] for h in st.search_knowledge("q")] == [3]                            # all-excluded evidence dropped
        assert st.other() == "passthrough"
    assert current_exclusion() == frozenset()


def test_runner_hook(tmp_path):
    class Sys:
        name = "probe"

        def __init__(self):
            self.store = FakeStore()
            self.seen = {}

        def answer(self, inp):
            hits = self.store.search_spans("q", top_k=3)
            self.seen[inp["id"]] = ([s.span_id for s, _ in hits], sorted(inp["options"]), inp.get("excluded_span_ids"))
            return fa(inp["id"], "A")
    s = Sys()
    assert install_exclusion(s) and isinstance(s.store, ExcludingStore)
    items = [item("u1", False, excl=["s0"]), item("c1", True, "A")]
    run_system(s, items, tmp_path / "o1", workers=2)
    assert s.seen["u1"] == (["s1", "s2", "s3"], ["A", "B", "C", "D"], ["s0"])
    assert s.seen["c1"] == (["s0", "s1", "s2"], ["A", "B", "C", "D"], None)
    run_system(s, [item("u2", False)], tmp_path / "o2", workers=1, abstain_option=True)
    assert s.seen["u2"][1] == ["A", "B", "C", "D", "E"]


def test_toc_parse_and_absent_topics():
    toc = Span("b:p0004:u001", "b", "| | Trang |\n|---|---|\n| Bài 1. Lực và tác dụng của lực | 7 |\n| Bài 2. Khối lượng riêng của vật | 12 |\n"
                                    "| CHƯƠNG II. ĐỘNG LỰC HỌC | 20 |\n| Bài 3. Định luật Newton thứ nhất | 21 |\n| Bài 4. Tốc độ và vận tốc | 26 |",
               pdf_page_index=4, printed_page_label="3")
    es = toc_entries([toc])
    assert [(e["num"], e["page"]) for e in es] == [(1, 7), (2, 12), (3, 21), (4, 26)]
    assert es[2]["chapter"].startswith("CHƯƠNG II")
    from otter.evaluation.corpus import SpanCorpus
    body = Span("b:p0020:u000", "b", "Nội dung bài 3 " * 5, pdf_page_index=20, printed_page_label="22", domain="physics", grade=10,
                book_title="Vật lí 10")
    toc.domain, toc.grade, toc.book_title = "physics", 10, "Vật lí 10"
    tp = absent_topics(SpanCorpus([toc, body]))
    assert [t["num"] for t in tp] == [4]                                                     # only lessons beyond the last present page


def test_released_benchmark_if_built():
    p = ROOT / "benchmarks/abstain/all.jsonl"
    if not p.exists():
        pytest.skip("OTTER-Abstain not built")
    items = load_mcq(p)
    rep = validate_mcq(items)
    assert all(e["check"] not in ("options_keys", "empty_option", "duplicate_id", "answer_not_single_letter") for e in rep["errors"])
    for it in items:
        assert it.answerable == (it.answer is not None)
        assert (it.abstain_type is None) == it.answerable
        assert not (set(it.gold_span_ids) & set(it.excluded_span_ids))
        if it.abstain_type == "missing_premise":
            assert it.excluded_span_ids
    dev = {it.id for it in load_mcq(ROOT / "benchmarks/abstain/dev.jsonl")}
    test = {it.id for it in load_mcq(ROOT / "benchmarks/abstain/test.jsonl")}
    assert not dev & test and len(dev) + len(test) == len(items)


def test_install_exclusion_wraps_v2_delegated_calc_system():
    """otter_v2 delegates calculation items to an otter_full instance: its store must be wrapped too."""
    class Master:
        store = FakeStore()

    class Calc:
        master = Master()

    class V2:
        store = FakeStore()
        calc_system = Calc()
    v = V2()
    assert install_exclusion(v)
    assert isinstance(v.store, ExcludingStore) and isinstance(v.calc_system.master.store, ExcludingStore)
