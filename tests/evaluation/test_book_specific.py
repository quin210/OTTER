"""Offline tests of the book_specific pipeline pieces (synthetic self-written spans, MockLLM)."""
import json

from otter.evaluation import book_specific as bs
from otter.evaluation.corpus import Span, SpanCorpus
from otter.evaluation.hard_gen import structural_checks
from otter.evaluation.hard_seeds import Seed
from otter.llm.interface import MockLLM

T1 = ("Trong thí nghiệm của sách, người ta đun 200 g nước từ 20 °C đến 80 °C và đo được nhiệt lượng thu vào là 50,4 kJ. "
      "Kết quả cho thấy nhiệt dung riêng của nước vào khoảng 4,2 kJ/kg.K, lớn hơn nhiều so với kim loại.")
T2 = ("Định luật chỉ áp dụng khi hệ kín và nhiệt độ không đổi; trường hợp có trao đổi chất với môi trường thì không áp dụng. "
      "Ngoài ra, điều kiện cần là áp suất phải ổn định trong suốt quá trình, nếu không kết quả sẽ sai lệch đáng kể.")


def corpus():
    sp = [Span("bkS:p0001:u001", "bkS", T1, 1, "bkS:ch01:l01:s01", ["Bài 1", "Nhiệt"], pedagogical_type="concept_explanation", grade=10,
               domain="physics", book_title="Vật lí 10 (synthetic)"),
          Span("bkS:p0002:u001", "bkS", T2, 2, "bkS:ch01:l02:s01", ["Bài 2", "Định luật"], pedagogical_type="law", grade=10,
               domain="physics", book_title="Vật lí 10 (synthetic)"),
          Span("bkS:p0003:u001", "bkS", "Em hãy tính nhiệt lượng cần thiết để đun nóng 2 kg nước?", 3, "bkS:ch01:l02:s02", ["Bài 2", "Luyện tập"],
               pedagogical_type="exercise", grade=10, domain="physics")]
    return SpanCorpus(sp, {"bkS": {"book_id": "bkS", "grade": 10, "subject_normalized": "physics", "title": "Vật lí 10 (synthetic)"}})


def seed(c, ids, tts):
    return bs._mk(ids, c, tts, [], "t")


def test_quote_and_numbers():
    assert bs.quote_in_span("nhiệt dung riêng của nước vào khoảng 4,2 kJ/kg.K", T1)
    assert not bs.quote_in_span("nước sôi ở một trăm độ khi áp suất bằng một atm", T1)
    assert bs.nontrivial_numbers("khoảng 4,2 kJ và 3 lần, 50") == ["4.2", "50"]


def test_trick_scores_prefers_condition_and_numeric():
    c = corpus()
    s1, s2 = c.get("bkS:p0001:u001"), c.get("bkS:p0002:u001")
    assert "condition_exception" in bs.trick_scores(s2)
    assert "example_case" in bs.trick_scores(s1) and "table_calc" in bs.trick_scores(s1)


def test_statement_assembly_is_deterministic_and_unique():
    sts = [{"text": f"phát biểu {i}", "true": t} for i, t in enumerate([True, False, True, False])]
    a = bs.assemble_statement_item("Cho các phát biểu về nhiệt", sts, "k1")
    assert a == bs.assemble_statement_item("Cho các phát biểu về nhiệt", sts, "k1")
    assert a["n_true"] == 2 and len({a["correct_answer"], *a["distractors"]}) == 4
    assert bs.assemble_statement_item("x", sts[:3], "k") is None
    assert bs.assemble_statement_item("x", [dict(s, true=False) for s in sts], "k") is None
    # all-true -> count form only; key is "4" and not the unique longest
    a4 = bs.assemble_statement_item("x", [dict(s, true=True) for s in sts], "k")
    assert a4["correct_answer"] == "4" and a4["statement_form"] == "count"


def _item(tt="table_lookup", **kw):
    it = {"trick_type": tt, "hop_type": "bridge", "question": "Theo sách giáo khoa Vật lí lớp 10, nhiệt dung riêng của nước trong thí nghiệm xấp xỉ bao nhiêu?",
          "correct_answer": "4,2 kJ/kg.K", "distractors": [{"text": "0,9 kJ/kg.K", "why_wrong": "nhôm"}, {"text": "2,1 kJ/kg.K", "why_wrong": "nước đá"},
                                                            {"text": "8,4 kJ/kg.K", "why_wrong": "gấp đôi"}],
          "gold_spans": ["S1"], "evidence": {"S1": "nhiệt dung riêng của nước vào khoảng 4,2 kJ/kg.K"}, "bloom": "nhận biết"}
    it.update(kw)
    return it


def test_generate_and_structural_checks():
    c = corpus()
    sd = seed(c, ["bkS:p0001:u001"], ["table_lookup", "example_case"])
    llm = MockLLM(handler=lambda m: json.dumps({"items": [_item(), _item(correct_answer="9,9 kJ/kg.K")]}, ensure_ascii=False))
    cands = bs.generate_bs(sd, c, llm)
    assert len(cands) == 2 and all(x["feasible"] and x["book_specific"] for x in cands)
    f0 = bs.bs_structural(cands[0], c)
    assert f0["bs_structural_ok"], f0
    f1 = bs.bs_structural(cands[1], c)             # invented number in the key -> not grounded in the span
    assert not f1["key_numbers_grounded"] and not f1["bs_structural_ok"]
    assert structural_checks(cands[0], c)["gold_spans_not_exercise"]
    assert bs.needs_numeric_check(cands[0], c) is True


def test_bad_evidence_and_foreign_gold_rejected():
    c = corpus()
    sd = seed(c, ["bkS:p0001:u001"], ["table_lookup"])
    llm = MockLLM(handler=lambda m: json.dumps({"items": [_item(evidence={"S1": "một câu không có trong sách giáo khoa này"})]}, ensure_ascii=False))
    x = bs.generate_bs(sd, c, llm)[0]
    assert "evidence_verbatim" in bs.bs_structural(x, c)["bs_structural_fail"]
    x["gold_span_ids"] = ["bkS:p0003:u001"]                  # exercise span, not in seed
    f = bs.bs_structural(x, c)
    assert not f["gold_in_seed"]
    assert not structural_checks(x, c)["gold_spans_not_exercise"]


def test_statement_combo_generation_and_check():
    c = corpus()
    sd = seed(c, ["bkS:p0002:u001"], ["statement_combo"])
    sts = [{"text": "Định luật chỉ áp dụng cho hệ kín", "true": True, "span": "S1", "quote": "Định luật chỉ áp dụng khi hệ kín và nhiệt độ không đổi"},
           {"text": "Định luật áp dụng cả khi có trao đổi chất", "true": False, "span": "S1", "quote": "trường hợp có trao đổi chất với môi trường thì không áp dụng"},
           {"text": "Áp suất phải ổn định", "true": True, "span": "S1", "quote": "áp suất phải ổn định trong suốt quá trình"},
           {"text": "Nhiệt độ thay đổi tùy ý", "true": False, "span": "S1", "quote": "nhiệt độ không đổi"}]
    llm = MockLLM(handler=lambda m: json.dumps({"items": [{"trick_type": "statement_combo", "stem": "Cho các phát biểu về định luật", "statements": sts,
                                                           "gold_spans": ["S1"], "evidence": {"S1": "điều kiện cần là áp suất phải ổn định trong suốt quá trình"}}]}, ensure_ascii=False))
    x = bs.generate_bs(sd, c, llm)[0]
    assert x["feasible"] and len(x["options"]) == 4 and x["options"][x["answer"]] == x["correct_answer"]
    assert bs.bs_structural(x, c)["bs_structural_ok"]
    chk = MockLLM(handler=lambda m: json.dumps({"I": "đúng", "II": "sai", "III": "đúng", "IV": "sai"}))
    assert bs.check_statements(x, c, chk)["match"] is True
    chk2 = MockLLM(handler=lambda m: json.dumps({"I": "đúng", "II": "sai", "III": "đúng", "IV": "không xác định"}))
    assert bs.check_statements(x, c, chk2)["match"] is False


def test_independent_check_maps_letters_back():
    c = corpus()
    sd = seed(c, ["bkS:p0001:u001"], ["table_lookup"])
    x = bs.generate_bs(sd, c, MockLLM(handler=lambda m: json.dumps({"items": [_item()]}, ensure_ascii=False)))[0]

    def handler(m):
        p = m[-1]["content"]
        import re
        opts = dict(re.findall(r"^([ABCD])\. (.*)$", p, re.M))
        return json.dumps({"answer": next(k for k, v in opts.items() if v == "4,2 kJ/kg.K")})
    assert bs.check_independent(x, c, MockLLM(handler=handler), 0)["correct"] is True
    wrong = bs.check_independent(x, c, MockLLM(handler=lambda m: '{"answer": "A"}'), 1)
    assert wrong["correct"] == (wrong["answer"] == x["answer"])


def test_seed_builder_runs_and_uses_expository_only():
    c = corpus()
    seeds = bs.build_seeds(c, {}, {}, round=0, max_per_domain=10)
    ids = {i for s in seeds for i in s.span_ids}
    assert "bkS:p0003:u001" not in ids                         # exercise span never a seed
    assert all(s.strategy == "book_specific" and s.link["trick_types"] for s in seeds)


def test_pick_distractors_prefers_length_balance_and_not_shortest_key():
    key = "cầu nối disulfide giữa các mạch"
    ds = ["x", "cầu nối peroxit giữa các mạch", "cầu nối carbon giữa hai mạch", "cầu nối nitro giữa các mạch polymer", "ngắn", "cầu nối ion giữa các mạch cao su"]
    sel = bs.pick_distractors(key, ds, "salt")
    assert len(sel) == 3 and "x" not in sel and "ngắn" not in sel
    assert max(len(d) for d in sel) >= len(key)


def test_option_truth_and_audit_checks():
    c = corpus()
    sd = seed(c, ["bkS:p0001:u001"], ["negation"])
    x = bs.generate_bs(sd, c, MockLLM(handler=lambda m: json.dumps({"items": [_item("negation")]}, ensure_ascii=False)))[0]
    import re

    def handler(m):
        p = m[-1]["content"]
        opts = dict(re.findall(r"^([ABCD])\. (.*)$", p, re.M))
        return json.dumps({k: ("sai" if v == "4,2 kJ/kg.K" else "đúng") for k, v in opts.items()}, ensure_ascii=False)
    r = bs.check_option_truth(x, c, MockLLM(handler=handler), 0)
    assert r["ok"] is True and r["truth"][x["answer"]] is False
    bad = bs.check_option_truth(x, c, MockLLM(handler=lambda m: json.dumps({k: "đúng" for k in "ABCD"})), 1)
    assert bad["ok"] is False
    good = bs.check_audit(x, MockLLM(handler=lambda m: json.dumps({"self_contained": 5, "also_correct": [], "subjective": False})))
    assert good["ok"] is True
    # 'also_correct' letters are mapped back to the original option letters and never include the key
    alt = bs.check_audit(x, MockLLM(handler=lambda m: json.dumps({"self_contained": 5, "also_correct": ["A", "B", "C", "D"], "subjective": False})))
    assert alt["ok"] is False and x["answer"] not in alt["also_correct"] and len(alt["also_correct"]) == 3
    assert not bs.deictic_free("Trong văn bản này, nhân vật nào xuất hiện?") and bs.deictic_free("Theo sách giáo khoa Vật lí 10, nhiệt dung riêng của nước là bao nhiêu?")


def test_names_grounded_blocks_hallucinated_titles():
    spans = ["Tuy nhiên, ông thường xuyên bị thanh tra Gia-ve rình mò.", "Bài 7 QUYỀN NĂNG CỦA NGƯỜI KỂ CHUYỆN"]
    assert not bs.names_grounded("Trong truyện ngắn 'Chuyện người con gái Nam Xương', ai bị thanh tra Gia-ve rình mò?", spans)
    assert bs.names_grounded("Theo sách giáo khoa Ngữ văn lớp 12, ai bị thanh tra Gia-ve rình mò?", spans)
    assert not bs.names_grounded("Theo sách giáo khoa Lịch sử lớp 12, ai chỉ huy chiến dịch Điện Biên Phủ?", ["Chiến dịch lịch sử"])


def test_strict_review_all_criteria_required():
    c = corpus()
    sd = seed(c, ["bkS:p0001:u001"], ["table_lookup"])
    x = bs.generate_bs(sd, c, MockLLM(handler=lambda m: json.dumps({"items": [_item()]}, ensure_ascii=False)))[0]
    ok = {"key_supported": True, "unique_key": True, "question_clear": True, "facts_accurate": True, "options_distinct": True, "issues": ""}
    assert bs.check_strict_review(x, c, MockLLM(handler=lambda m: json.dumps(ok)))["ok"] is True
    bad = bs.check_strict_review(x, c, MockLLM(handler=lambda m: json.dumps(dict(ok, unique_key=False, issues="UAC cũng đúng"))))
    assert bad["ok"] is False and "UAC" in bad["issues"]
