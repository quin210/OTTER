"""Offline tests for ViEduHop-Deep: leave-one-out necessity check and fold (synthetic spans, MockLLM)."""
import json

from otter.evaluation import hard_deep as hd
from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import assemble_options
from otter.llm.interface import MockLLM

TXT = {f"b:p000{i}:u000": f"Đoạn thử nghiệm số {i}: thông tin mắt xích {i} của chuỗi suy luận, nói về khái niệm k{i} và tính chất t{i}." for i in range(5)}


def _corpus():
    spans = [Span(span_id=k, book_id="b", text=v, domain="biology", grade=10, heading_path=["Bài 1"], section_id="b:ch00:l00:s00",
                  pdf_page_index=i) for i, (k, v) in enumerate(TXT.items())]
    return SpanCorpus(spans)


def _cand():
    gold = list(TXT)[:5]
    opts, ans = assemble_options("cand_x", "Đáp án đúng", ["Nhiễu một", "Nhiễu hai", "Nhiễu ba"])
    return {"cand_id": "cand_x", "gold_span_ids": gold, "options": opts, "answer": ans, "question": "Câu hỏi?"}


def test_loo_fold_all_necessary():
    gold = ["a", "b", "c", "d"]
    ok = {s: [{"sufficient": False, "answer": "none"}] * 2 for s in gold}
    r = hd.loo_fold(gold, "B", ok)
    assert r["loo_all_necessary"] and not r["loo_derivable_without"] and not r["loo_missing"]


def test_loo_fold_rejects_derivable_without_one_span():
    gold = ["a", "b", "c", "d"]
    recs = {s: [{"sufficient": False, "answer": "none"}] * 2 for s in gold}
    recs["c"] = [{"sufficient": False, "answer": "none"}, {"sufficient": True, "answer": "B"}]     # one permutation derives the key
    r = hd.loo_fold(gold, "B", recs)
    assert not r["loo_all_necessary"] and r["loo_derivable_without"] == ["c"]


def test_loo_fold_wrong_answer_does_not_count_as_derivable_but_missing_blocks():
    gold = ["a", "b", "c", "d"]
    recs = {s: [{"sufficient": True, "answer": "A"}] * 2 for s in gold}                          # sufficient for a different option
    assert hd.loo_fold(gold, "B", recs)["loo_all_necessary"]
    recs["d"] = [recs["d"][0], None]                                                              # a check is absent -> cannot pass
    r = hd.loo_fold(gold, "B", recs)
    assert r["loo_missing"] == ["d"] and not r["loo_all_necessary"]


def test_check_loo_removes_exactly_the_held_out_span():
    corpus, c = _corpus(), _cand()
    seen = []

    def handler(messages):
        seen.append(messages[-1]["content"])
        return json.dumps({"sufficient": False, "answer": "none", "missing": "thiếu"})
    llm = MockLLM(handler=handler)
    held = c["gold_span_ids"][2]
    r = hd.check_loo(c, held, corpus, llm, perm=1)
    prompt = seen[0]
    assert TXT[held] not in prompt and all(TXT[s] in prompt for s in c["gold_span_ids"] if s != held)
    assert r["held_out"] == held and r["perm"] == 1 and r["sufficient"] is False and r["answer"] == "none"


def test_loo_verdict_end_to_end_with_mock():
    corpus, c = _corpus(), _cand()
    key_text = c["options"][c["answer"]]

    def handler(messages):                         # judge says "sufficient with the key" only when span 3 is present AND span 0 is held out
        p = messages[-1]["content"]
        if TXT[list(TXT)[0]] not in p and TXT[list(TXT)[3]] in p:
            letter = next(l for l in "ABCD" if f"{l}. {key_text}" in p)
            return json.dumps({"sufficient": True, "answer": letter, "missing": ""})
        return json.dumps({"sufficient": False, "answer": "none", "missing": "x"})
    llm = MockLLM(handler=handler)
    recs = {}
    for s in c["gold_span_ids"]:
        for p in range(2):
            recs[f"loo:{s}:{p}"] = {"result": hd.check_loo(c, s, corpus, llm, p)}
    v = hd.loo_verdict(c, lambda name: (recs.get(name) or {}).get("result"), 2)
    assert v["loo_derivable_without"] == [list(TXT)[0]] and not v["loo_all_necessary"]


def test_deep_structural_requires_four_spans_and_grounded_chained_steps():
    corpus, c = _corpus(), _cand()
    ids = c["gold_span_ids"]
    c.update(feasible=True, correct_answer=c["options"][c["answer"]], distractors=[v for k, v in c["options"].items() if k != c["answer"]],
             seed_span_ids=ids, question="Một tình huống thực tế hỏi về kết quả cuối cùng của chuỗi suy luận nhiều bước?",
             reasoning_steps=[{"span_id": s, "step": "x"} for s in ids],
             chain=[{"span_id": s, "input": f"khái niệm k{i - 1} tính chất t{i - 1}", "fact": TXT[s].split(": ")[1], "output": f"khái niệm k{i} tính chất t{i}"}
                    for i, s in enumerate(ids)])
    f = hd.deep_structural(c, corpus)
    assert f["deep_min_spans"] and f["deep_facts_grounded"] and f["deep_chain_linked"] and f["deep_chain_covers_gold"], f
    c3 = dict(c, gold_span_ids=ids[:3])
    assert not hd.deep_structural(c3, corpus)["deep_min_spans"]
    c_bad = dict(c, chain=[dict(st, fact="điều hoàn toàn bịa đặt không có trong đoạn nào cả") for st in c["chain"]])
    assert not hd.deep_structural(c_bad, corpus)["deep_facts_grounded"]
