"""Structure recovery + span offsets on self-written synthetic Markdown (no textbook text)."""
import json
import tempfile
import unittest
from pathlib import Path

from otter.schemas.textbook import SourceSpan, StructureNode
from otter.textbooks.ocr import quality_metrics, quality_score, repetition_score, split_page_tag, trim_degenerate_tail
from otter.textbooks.pages import resolve_page_labels
from otter.textbooks.structure import build_structure, process_book, split_blocks

PAGE0 = """# CHƯƠNG I
# THẾ GIỚI GIẢ ĐỊNH

## Bài 1. Hạt mầm xanh

### I. Khái niệm hạt mầm

Hạt mầm xanh là đơn vị giả định dùng để minh hoạ cho bài kiểm thử.

Câu hỏi
Vì sao hạt mầm xanh cần ánh sáng?

#### 1. Công thức tăng trưởng

$$
g = \\frac{m}{t}
$$

| Đại lượng | Kí hiệu |
|---|---|
| Khối lượng | m |

[Hình 1.1. Sơ đồ hạt mầm giả định]
"""

PAGE1 = """#### 2. Điều kiện nảy mầm
Hạt mầm chỉ nảy mầm khi đủ độ ẩm trong mô hình giả định này.

## Em có biết
Trong mô hình này, mỗi hạt mầm có một mã số riêng.

## BÀI 2
## CÂY NON GIẢ ĐỊNH

### I. Cấu tạo
- Rễ giả định
- Thân giả định
"""


def _pages():
    sha = "0" * 64
    return [
        {"book_id": "bk", "span_id": "bk:p0000", "pdf_page_index": 0, "text": PAGE0, "printed_page_label": "6",
         "printed_page_verified": True, "source_file_sha256": sha, "extraction_method": "vlm_ocr", "extractor_version": "t"},
        {"book_id": "bk", "span_id": "bk:p0001", "pdf_page_index": 1, "text": PAGE1, "printed_page_label": "7",
         "printed_page_verified": False, "source_file_sha256": sha, "extraction_method": "vlm_ocr", "extractor_version": "t"},
    ]


class StructureSpanTests(unittest.TestCase):
    def test_block_offsets_exact(self):
        for text in (PAGE0, PAGE1, "  lead space para\n\n\nsecond para  \n", "$$\nx=1\n"):
            for b in split_blocks(text):
                self.assertEqual(text[b.start:b.end], b.text)
                self.assertEqual(b.text, b.text.strip())
        kinds = [b.kind for b in split_blocks(PAGE0)]
        self.assertIn("table", kinds); self.assertIn("figure", kinds); self.assertIn("formula", kinds)

    def test_hierarchy_and_spans(self):
        pages = _pages()
        nodes, spans, page_secs = build_structure("bk", pages, "Sách giả định")
        by_level = {}
        for n in nodes:
            StructureNode.model_validate(n.model_dump())
            by_level.setdefault(n.level, []).append(n)
        self.assertEqual(len(by_level["chapter"]), 1)
        self.assertEqual(by_level["chapter"][0].title, "THẾ GIỚI GIẢ ĐỊNH")
        self.assertEqual([n.ordinal for n in by_level["lesson"]], ["Bài 1", "BÀI 2"])
        self.assertEqual(by_level["lesson"][1].title, "CÂY NON GIẢ ĐỊNH")          # title on the next heading line
        self.assertEqual(by_level["lesson"][0].parent_id, by_level["chapter"][0].section_id)
        subs = by_level["subsection"]
        self.assertEqual([s.ordinal for s in subs], ["1.", "2."])
        self.assertEqual(subs[1].pdf_page_start, 1)                                  # continues across the page break
        self.assertEqual(by_level["lesson"][0].printed_page_start, "6")
        self.assertIsNone(by_level["lesson"][1].printed_page_start)                  # unverified label not used
        self.assertTrue(by_level["lesson"][0].heading_path[-1].startswith("Bài 1"))
        texts = {p["pdf_page_index"]: p["text"] for p in pages}
        for s in spans:
            SourceSpan.model_validate(s.model_dump())
            self.assertEqual(texts[s.pdf_page_index][s.start_offset:s.end_offset], s.text)
        def ptype(prefix):
            return next(s.pedagogical_type for s in spans if s.text.startswith(prefix))
        self.assertEqual(ptype("Hạt mầm xanh là"), "definition")
        self.assertEqual(ptype("Câu hỏi"), "question")
        self.assertEqual(ptype("Trong mô hình này"), "reading_text")
        self.assertEqual(ptype("| Đại lượng"), "table")
        self.assertEqual(ptype("[Hình 1.1"), "figure_caption")
        self.assertEqual(ptype("$$"), "formula")
        ids = [s.span_id for s in spans]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(i.startswith("bk:p000") and ":u" in i for i in ids))
        self.assertIsNotNone(page_secs[1])

    def test_process_book_writes_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "bk"; d.mkdir()
            (d / "pages.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in _pages()), encoding="utf-8")
            n_nodes, n_spans = process_book(d, "Sách giả định")
            self.assertGreater(n_nodes, 5); self.assertGreater(n_spans, 10)
            spans = [json.loads(l) for l in (d / "spans.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("span_kind" in s and "chapter_id" in s for s in spans))
            pages = [json.loads(l) for l in (d / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(p["section_id"] for p in pages))

    def test_toc_page_does_not_create_lessons(self):
        toc = {"book_id": "bk", "span_id": "bk:p0000", "pdf_page_index": 0, "text": "# MỤC LỤC\n\nBài 1. Hạt mầm 5\n\nBài 2. Cây non 9\n",
               "source_file_sha256": "0" * 64}
        nodes, spans, _ = build_structure("bk", [toc])
        self.assertEqual([n.level for n in nodes], ["book"])

    def test_ocr_helpers(self):
        body, label, present = split_page_tag("# Tiêu đề\n\nNội dung.\n<<SỐ_TRANG: 12>>")
        self.assertEqual((body, label, present), ("# Tiêu đề\n\nNội dung.", "12", True))
        self.assertIsNone(split_page_tag("x\n<<SỐ_TRANG: không>>")[1])
        loop = "Dòng mở đầu.\n" + "lặp lại mãi\n" * 30
        self.assertGreater(repetition_score(loop), 0.3)
        trimmed, changed = trim_degenerate_tail(loop)
        self.assertTrue(changed); self.assertLess(trimmed.count("lặp lại mãi"), 5)
        good = "Đây là một đoạn văn tiếng Việt có dấu đầy đủ để kiểm tra chất lượng nhận dạng. " * 3
        m = quality_metrics(good)
        self.assertGreater(m["vi_diacritic_ratio"], 0.4)
        score, review, _ = quality_score(m)
        self.assertGreaterEqual(score, 0.9); self.assertFalse(review)
        bad = quality_metrics("D?y l? m?t ?o?n v?n kh?ng c? d?u " * 10)
        self.assertLess(quality_score(bad)[0], 0.9)

    def test_page_label_resolution(self):
        raw = {0: {"printed_page_label_tag": "10", "printed_page_label_crop": "10"},
               1: {"printed_page_label_tag": "11", "printed_page_label_crop": "11"},
               2: {"printed_page_label_tag": "12", "printed_page_label_crop": "17"},
               3: {"printed_page_label_tag": None, "printed_page_label_crop": None},
               9: {"printed_page_label_tag": "40", "printed_page_label_crop": "40"}}
        r = resolve_page_labels(raw)
        self.assertEqual(r[0][:2], ("10", True)); self.assertEqual(r[1][:2], ("11", True))
        self.assertEqual(r[2][:2], ("12", False))       # reads disagree -> neighbour-consistent read, unverified
        self.assertEqual(r[3][:2], (None, False))
        self.assertEqual(r[9][:2], ("40", False))       # isolated: no neighbour support


if __name__ == "__main__":
    unittest.main()
