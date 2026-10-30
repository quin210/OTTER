"""Offline tests of the OTTER-Hard pipeline (synthetic self-written spans, MockLLM)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from otter.evaluation import hard_build as hb
from otter.evaluation.corpus import SpanCorpus
from otter.evaluation.embeddings import MockEmbedder
from otter.evaluation.hard_gen import (JsonlStore, LeakageIndex, assemble_options, check_key, permute, run_checks,
                                       run_generation, structural_checks, summarize_checks)
from otter.evaluation.hard_seeds import SeedSampler, eligible, hop_cues
from otter.evaluation.mcq import LETTERS, load_mcq
from otter.llm.interface import MockLLM

ROOT = Path(__file__).resolve().parents[2]

# Synthetic, self-written biology-like paragraphs (not textbook content).
TEXTS = {
    ("bkA", 10, "l01"): [
        "Enzim xúc tác là chất có bản chất protein giúp tăng tốc độ phản ứng trong tế bào. Mỗi loại enzim xúc tác chỉ "
        "liên kết với một cơ chất đặc hiệu tại vùng trung tâm hoạt động, nhờ vậy phản ứng diễn ra nhanh hơn nhiều lần.",
        "Nhiệt độ ảnh hưởng rõ rệt tới hoạt tính enzim. Khi nhiệt độ vượt quá mức tối ưu thì cấu trúc trung tâm hoạt động "
        "bị biến tính, do đó enzim xúc tác mất khả năng gắn cơ chất và tốc độ phản ứng giảm mạnh.",
        "Một số đoạn mô tả lịch sử phát hiện kính hiển vi quang học, giúp các nhà khoa học quan sát được vi khuẩn, nấm men "
        "và nhiều dạng sống nhỏ bé khác mà mắt thường không nhìn thấy trong môi trường nước ao hồ tự nhiên quanh ta.",
        "Ở độ pH thích hợp, hoạt tính enzim đạt cực đại; nếu môi trường quá axit hoặc quá kiềm thì liên kết trong trung "
        "tâm hoạt động thay đổi, làm cho enzim không còn nhận ra cơ chất đặc hiệu của mình nữa trong tế bào sống.",
    ],
    ("bkA", 10, "l02"): [
        "Ti thể là bào quan tạo năng lượng ATP thông qua hô hấp tế bào. Quá trình hô hấp tế bào cần nhiều enzim xúc tác "
        "khác nhau nằm trên màng trong của ti thể, vì vậy tế bào cơ hoạt động mạnh thường chứa rất nhiều ti thể.",
        "Lục lạp có ở tế bào thực vật, là nơi diễn ra quang hợp. Năng lượng ánh sáng được hấp thụ bởi diệp lục rồi chuyển "
        "thành năng lượng hóa học tích lũy trong phân tử đường, cung cấp nguyên liệu cho hô hấp tế bào về sau.",
        "Màng sinh chất có cấu trúc khảm động gồm lớp kép phospholipid xen kẽ các phân tử protein. Nhờ tính bán thấm, màng "
        "kiểm soát sự ra vào của các chất, giữ cho môi trường bên trong tế bào luôn ổn định trước biến động bên ngoài.",
    ],
    ("bkB", 11, "l05"): [
        "Ở thực vật, quang hợp phụ thuộc cường độ ánh sáng, nồng độ khí cacbonic và nhiệt độ. Khi nhiệt độ quá cao, các "
        "enzim xúc tác của pha tối bị biến tính nên cường độ quang hợp giảm, cây sinh trưởng chậm lại rõ rệt.",
        "Hô hấp tế bào ở thực vật diễn ra liên tục cả ngày lẫn đêm, phân giải đường tạo năng lượng ATP. Nếu rễ cây bị ngập "
        "úng lâu ngày thì thiếu ôxi, hô hấp hiếu khí bị ức chế và rễ chuyển sang lên men, sinh ra chất độc cho cây.",
        "Thoát hơi nước qua khí khổng tạo lực kéo giúp vận chuyển nước và ion khoáng từ rễ lên lá. Ngoài ra, thoát hơi nước "
        "còn làm giảm nhiệt độ bề mặt lá vào những ngày nắng nóng, bảo vệ bộ máy quang hợp khỏi bị tổn thương.",
    ],
}
BOOKS = [{"book_id": "bkA", "grade": 10, "subject_normalized": "biology", "title": "Sinh học 10 (synthetic)"},
         {"book_id": "bkB", "grade": 11, "subject_normalized": "biology", "title": "Sinh học 11 (synthetic)"}]
EXERCISE = "Em hãy giải thích vì sao khi nhiệt độ quá cao enzim xúc tác lại mất hoạt tính?"


@pytest.fixture
def hard_corpus(tmp_path):
    for b in BOOKS:
        d = tmp_path / "parsed" / b["book_id"]
        d.mkdir(parents=True)
        rows, page = [], 0
        for (bid, g, les), texts in TEXTS.items():
            if bid != b["book_id"]:
                continue
            for k, t in enumerate(texts):
                page += 1
                rows.append({"span_id": f"{bid}:p{page:04d}:u001", "book_id": bid, "section_id": f"{bid}:ch01:{les}:s{k:02d}",
                             "heading_path": [f"Bài {les}", f"Mục {k}"], "pdf_page_index": page, "start_offset": 0,
                             "end_offset": len(t), "text": t, "source_file_sha256": "0" * 64})
        if b["book_id"] == "bkA":
            rows.append({"span_id": "bkA:p0099:u001", "book_id": "bkA", "section_id": "bkA:ch01:l01:s99",
                         "heading_path": ["Bài l01", "Luyện tập"], "pdf_page_index": 99, "start_offset": 0,
                         "end_offset": len(EXERCISE), "text": EXERCISE, "source_file_sha256": "0" * 64})
        (d / "spans.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "books.jsonl").write_text("\n".join(json.dumps(b) for b in BOOKS), encoding="utf-8")
    c = SpanCorpus.load(parsed_dir=tmp_path / "parsed", db=None, registry=reg)
    return c, tmp_path


MARK = "đúng chuẩn"


def mock_handler(messages):
    p = messages[-1]["content"]
    if "NHIỆM VỤ: Viết MỘT câu hỏi" in p:
        hop = re.search(r'"hop_type"', p)
        return json.dumps({"feasible": True, "hop_type": "causal_chain", "link": "enzim",
                           "reasoning_steps": [{"span": "S1", "step": "bước một"}, {"span": "S2", "step": "bước hai"}],
                           "question": "Vì sao khi trời quá nóng thì cường độ quang hợp của cây giảm theo cơ chế nào?",
                           "correct_answer": f"Enzim bị biến tính {MARK}",
                           "distractors": [{"text": "Khí khổng mở rộng hơn", "why_wrong": "x"},
                                           {"text": "Diệp lục tăng gấp đôi", "why_wrong": "y"},
                                           {"text": "Rễ hút nhiều ôxi hơn", "why_wrong": "z"}],
                           "gold_spans": ["S1", "S2"]}, ensure_ascii=False)
    lines = {m.group(1): m.group(2) for m in re.finditer(r"^([ABCD])\. (.*)$", p, re.M)}
    key = next((k for k, v in lines.items() if MARK in v), "A")
    if "đánh giá từng phương án" in p:
        return json.dumps({"option_status": {k: "correct" if k == key else "incorrect" for k in LETTERS},
                           "answer": key, "used_spans": ["E1", "E2"], "confidence": 90})
    if "MỘT đoạn trích" in p:
        return json.dumps({"sufficient": False, "answer": "none", "missing": "thiếu"})
    return json.dumps({"answer": "A", "confidence": 40})


def test_eligibility_and_cues(hard_corpus):
    c, _ = hard_corpus
    ex = c.get("bkA:p0099:u001")
    assert ex.is_exercise() and not eligible(ex)
    assert sum(eligible(s) for s in c.spans) == 10
    assert "conditional_rule" in hop_cues("Nếu trời mưa thì đất ướt") and "temporal" in hop_cues("Năm 1945 ...")


def test_seed_sampler_constraints(hard_corpus):
    c, _ = hard_corpus
    c.attach_embedder(MockEmbedder())
    sm = SeedSampler(c, seed=1, min_link=1.0, dense_lo=-1.0)
    seeds = sm.sample(per_cell=6)
    assert seeds, "expected at least one linked seed"
    used = {}
    for s in seeds:
        assert 2 <= len(s.span_ids) <= 3 and len(set(s.span_ids)) == len(s.span_ids)
        assert all(not c.get(i).is_exercise() for i in s.span_ids)
        assert s.target_hop_type in s.compatible_hop_types
        for i in s.span_ids:
            used[i] = used.get(i, 0) + 1
    assert max(used.values()) <= 2
    fs = [frozenset(s.span_ids) for s in seeds]
    assert len(fs) == len(set(fs))
    assert {s.strategy for s in seeds} & {"same_lesson", "cross_lesson", "cross_book", "chain"}


def test_options_balanced_and_permute():
    pos = [assemble_options(f"c{i}", "đúng", ["x", "y", "z"])[1] for i in range(400)]
    assert all(pos.count(k) > 60 for k in LETTERS)
    o, a = assemble_options("c1", "đúng", ["x", "y", "z"])
    assert o[a] == "đúng"
    no, na, back = permute(o, a, "salt")
    assert no[na] == "đúng" and back[na] == a and sorted(no.values()) == sorted(o.values())


def test_structural_and_leakage(hard_corpus):
    c, _ = hard_corpus
    ids = ["bkA:p0002:u001", "bkB:p0001:u001"]
    good = {"cand_id": "x", "domain": "biology", "feasible": True, "question": "Vì sao cường độ quang hợp giảm khi trời quá nóng?",
            "correct_answer": "Enzim bị biến tính", "distractors": ["Khí khổng mở", "Diệp lục tăng", "Rễ hút ôxi"],
            "gold_span_ids": ids, "reasoning_steps": [{"span_id": ids[0], "step": "a"}, {"span_id": ids[1], "step": "b"}]}
    good["options"], good["answer"] = assemble_options("x", good["correct_answer"], good["distractors"])
    f = structural_checks(good, c)
    assert f["structural_ok"], f["structural_fail"]
    bad = dict(good, question="Theo đoạn trích S1, vì sao enzim bị biến tính?")
    assert "no_meta_reference" in structural_checks(bad, c)["structural_fail"]
    one = dict(good, gold_span_ids=ids[:1])
    assert not structural_checks(one, c)["structural_ok"]
    longans = dict(good, correct_answer="Enzim bị biến tính " * 6)
    longans["options"], longans["answer"] = assemble_options("x", longans["correct_answer"], good["distractors"])
    assert "no_length_cue" in structural_checks(longans, c)["structural_fail"]
    leak = LeakageIndex(c, [("M1", "Đơn vị của lực là gì?")])
    assert leak.check(good)["leak_ok"]
    assert not leak.check(dict(good, question=EXERCISE))["leak_ok"]


def test_generation_checks_assembly(hard_corpus, tmp_path):
    c, _ = hard_corpus
    c.attach_embedder(MockEmbedder())
    seeds = SeedSampler(c, seed=1, min_link=1.0, dense_lo=-1.0).sample(per_cell=6)
    llm = MockLLM(handler=mock_handler, model="mock-gen")
    store = JsonlStore(tmp_path / "w" / "cands.jsonl", key_fn=lambda r: r["cand_id"])
    run_generation(seeds, c, llm, store, workers=2)
    assert len(store.rows) == len(seeds)
    cands = list(store.rows.values())
    assert all(cd["options"][cd["answer"]].endswith(MARK) for cd in cands)
    # resumable: a second run adds nothing
    n_lines = len((tmp_path / "w" / "cands.jsonl").read_text().splitlines())
    run_generation(seeds, c, llm, JsonlStore(tmp_path / "w" / "cands.jsonl", key_fn=lambda r: r["cand_id"]), workers=2)
    assert len((tmp_path / "w" / "cands.jsonl").read_text().splitlines()) == n_lines
    checks = JsonlStore(tmp_path / "w" / "checks.jsonl", key_fn=check_key)
    run_checks(cands, c, checks, main=llm, small=MockLLM(handler=mock_handler, model="mock-small"), workers=2)
    s = summarize_checks(cands[0], checks.rows)
    assert s["verify_ok"] and s["multi_hop_ok"] and s["difficulty"] in ("hard", "medium")
    leak = LeakageIndex(c, [])
    ann = hb.annotate_candidates(cands, checks.rows, c, leak)
    kept, counts = hb.funnel(ann)
    assert counts["generated"] == len(cands) and counts["not_single_span_answerable"] == len(kept)
    kept, dups = hb.dedup(kept)
    assert len(kept) == 1 and len(dups) == len(cands) - 1      # mock always writes the same question
    rows = hb.to_rows(kept, c)
    r = rows[0]
    assert r["id"].startswith("OTTER-HARD-BIOLOGY-") and r["hop_count"] == 2 and r["reasoning_type"] == r["hop_type"]
    split = hb.lesson_disjoint_split(rows, dev_size=1)
    assert set(split.values()) <= {"dev", "test"}
    st = hb.bench_stats([dict(r, split="test")], c)
    assert st["n"] == 1 and st["by_hop_count"] == {2: 1}
    csv_p, html_p = hb.write_review_sheet(rows, c, tmp_path / "sheet", n=5)
    assert csv_p.exists() and "OTTER-Hard review" in html_p.read_text()
    out, summ = hb.apply_review(rows, [{"id": r["id"], "decision": "drop"}])
    assert out == [] and summ["n_reviewed"] == 1 and summ["drop"] == 1
    out, summ = hb.apply_review(rows, [{"id": r["id"], "decision": "fix", "corrected_answer": "c"}])
    assert out[0]["answer"] == "C" and out[0]["review_status"] == "human_fixed"


def test_split_is_lesson_disjoint():
    rows = [{"id": f"I{i}", "domain": "physics" if i % 2 else "biology", "lesson_keys": [f"L{i // 3}", f"L{i // 3 + 100}"]}
            for i in range(60)]
    sp = hb.lesson_disjoint_split(rows, dev_size=12, seed=3)
    dev_l = {k for r in rows if sp[r["id"]] == "dev" for k in r["lesson_keys"]}
    assert 6 <= sum(v == "dev" for v in sp.values()) <= 16
    assert not any(set(r["lesson_keys"]) & dev_l for r in rows if sp[r["id"]] == "test")


def test_stratified_cap_and_easy_cap():
    cs = [{"cand_id": f"c{i}", "domain": "physics" if i < 30 else "history", "hop_type": "bridge" if i % 3 else "temporal",
           "filters": {"difficulty": "easy" if i % 2 else "hard"}} for i in range(60)]
    capped, dropped = hb.cap_easy(cs, frac=0.25)
    for d in ("physics", "history"):
        sub = [c for c in capped if c["domain"] == d]
        assert sum(c["filters"]["difficulty"] == "easy" for c in sub) <= 0.25 * len(sub) + 1e-9
    assert len(hb.stratified_cap(cs, 20)) == 20




@pytest.mark.parametrize("text,heading,expected", [
    ("? Đọc thông tin trong mục 1, hãy trình bày tính chất của nước biển.", ["Bài 9", "1. Nước biển"], True),
    ("- Bạn hãy kể tên một số tác giả mà bạn yêu thích và chia sẻ cảm nhận.", ["VĂN BẢN 1"], True),
    ("- Nêu được đặc điểm của thể thơ.\n- Phân tích được tác dụng của biện pháp tu từ.", ["Bài 2"], True),
    ("Enzim là chất xúc tác sinh học có bản chất protein, làm tăng tốc độ phản ứng trong tế bào.", ["BÀI 1 PHẦN MỞ ĐẦU", "I. Enzim"], False),
    ("Nội dung bất kì của phần luyện tập dành cho học sinh tự làm ở nhà.", ["Bài 3", "LUYỆN TẬP"], True),
])
def test_exercise_detection(text, heading, expected):
    from otter.evaluation.corpus import Span
    assert Span(span_id="x", book_id="b", text=text, heading_path=heading).is_exercise() is expected
