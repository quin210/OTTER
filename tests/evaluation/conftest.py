import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OTTER_LLM_MOCK", "1")

SPANS = [
    # synthetic, self-written texts (no textbook content)
    ("bk_phys:p0001:u001", "bk_phys", 1, "bk_phys:ch01:l01:s01", ["Bài 1", "Lực"],
     "Lực là đại lượng vectơ đặc trưng cho tác dụng của vật này lên vật khác. Đơn vị của lực là niutơn (N)."),
    ("bk_phys:p0001:u002", "bk_phys", 1, "bk_phys:ch01:l01:s02", ["Bài 1", "Định luật II"],
     "Gia tốc của một vật cùng hướng với hợp lực và có độ lớn bằng hợp lực chia cho khối lượng: a = F/m."),
    ("bk_phys:p0002:u001", "bk_phys", 2, "bk_phys:ch01:l02:s01", ["Bài 2", "Khối lượng"],
     "Khối lượng là đại lượng đặc trưng cho mức quán tính của vật; vật có khối lượng càng lớn thì càng khó thay đổi vận tốc."),
    ("bk_phys:p0002:u002", "bk_phys", 2, "bk_phys:ch01:l02:s02", ["Bài 2", "Luyện tập"],
     "Em hãy tính gia tốc của vật khối lượng 2 kg chịu lực 4 N?"),
    ("bk_bio:p0001:u001", "bk_bio", 1, "bk_bio:ch01:l01:s01", ["Bài 1", "Tế bào"],
     "Tế bào là đơn vị cấu trúc và chức năng cơ bản của mọi cơ thể sống. Màng sinh chất bao bọc tế bào."),
    ("bk_bio:p0001:u002", "bk_bio", 1, "bk_bio:ch01:l01:s02", ["Bài 1", "Ti thể"],
     "Ti thể là bào quan thực hiện hô hấp tế bào, tạo ra năng lượng ATP cho tế bào; nếu ti thể bị tổn thương thì tế bào thiếu năng lượng."),
    ("bk_bio:p0003:u001", "bk_bio", 3, "bk_bio:ch02:l03:s01", ["Bài 3", "Lục lạp"],
     "Lục lạp có ở tế bào thực vật, là nơi diễn ra quang hợp, chuyển hóa quang năng thành hóa năng."),
]
BOOKS = [
    {"book_id": "bk_phys", "grade": 10, "subject_normalized": "physics", "title": "Vật lí 10 (synthetic)"},
    {"book_id": "bk_bio", "grade": 11, "subject_normalized": "biology", "title": "Sinh học 11 (synthetic)"},
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    for b in BOOKS:
        d = tmp_path / "parsed" / b["book_id"]
        d.mkdir(parents=True)
        rows = [s for s in SPANS if s[1] == b["book_id"]]
        with open(d / "spans.jsonl", "w", encoding="utf-8") as f:
            for sid, bid, page, sec, hp, text in rows:
                f.write(json.dumps({"span_id": sid, "book_id": bid, "section_id": sec, "heading_path": hp,
                                    "pdf_page_index": page, "printed_page_label": str(page + 10),
                                    "printed_page_verified": sid.endswith("u001"), "start_offset": 0,
                                    "end_offset": len(text), "text": text, "pedagogical_type": "unknown",
                                    "source_file_sha256": "0" * 64}, ensure_ascii=False) + "\n")
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "books.jsonl").write_text("\n".join(json.dumps(b, ensure_ascii=False) for b in BOOKS), encoding="utf-8")
    return tmp_path


@pytest.fixture
def corpus(corpus_dir):
    from otter.evaluation.corpus import SpanCorpus
    return SpanCorpus.load(parsed_dir=corpus_dir / "parsed", db=None, registry=corpus_dir / "registry")


def mcq_rows():
    return [
        {"id": "Q1", "domain": "physics", "subject": "Vật lí", "grade": 10, "question": "Đơn vị của lực là gì?",
         "options": {"A": "Jun", "B": "Niutơn", "C": "Oát", "D": "Pascal"}, "answer": "B",
         "reasoning_type": "multi_hop", "gold_span_ids": ["bk_phys:p0001:u001"]},
        {"id": "Q2", "domain": "physics", "subject": "Vật lí", "grade": 10, "question": "Vật 2 kg chịu lực 4 N có gia tốc bao nhiêu?",
         "options": {"A": "1 m/s²", "B": "2 m/s²", "C": "4 m/s²", "D": "8 m/s²"}, "answer": "B",
         "reasoning_type": "calculation", "gold_span_ids": ["bk_phys:p0001:u002"]},
        {"id": "Q3", "domain": "biology", "subject": "Sinh học", "grade": 11, "question": "Bào quan nào tạo ATP?",
         "options": {"A": "Lục lạp", "B": "Ribôxôm", "C": "Ti thể", "D": "Nhân"}, "answer": "C",
         "reasoning_type": "conditional_rule", "gold_span_ids": ["bk_bio:p0001:u002", "bk_bio:p0001:u001"]},
        {"id": "Q4", "domain": "biology", "subject": "Sinh học", "grade": 11, "question": "Quang hợp diễn ra ở đâu?",
         "options": {"A": "Lục lạp", "B": "Ti thể", "C": "Không bào", "D": "Màng"}, "answer": "A",
         "reasoning_type": "multi_hop", "gold_span_ids": ["bk_bio:p0003:u001"]},
    ]


@pytest.fixture
def bench_path(tmp_path: Path) -> Path:
    p = tmp_path / "bench.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mcq_rows()), encoding="utf-8")
    return p
