"""Recover Book -> Chapter -> Lesson -> Section -> Subsection structure and paragraph SourceSpans
from OCR Markdown page text.

* Structure comes only from printed headings (CHƯƠNG / CHỦ ĐỀ / CHUYÊN ĐỀ / PHẦN, Bài, I., 1.)
  found in the page text; nothing is inferred from semantic similarity. A node
  whose expected parent level is missing is attached to the nearest existing
  ancestor and marked ``link_status="unresolved"``.
* Spans are paragraph-level blocks (separated by blank lines; headings, tables,
  figure markers and display formulas are separate blocks) with exact character
  offsets into the page text: ``page_text[start_offset:end_offset] == span.text``.
* ``pedagogical_type`` is a transparent heuristic from Vietnamese textbook markers.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from otter.schemas.textbook import SourceSpan, StructureNode

STRUCTURE_VERSION = "otter-structure-0.3"

_ROMAN = r"[IVXLC]{1,6}"
CHAPTER_RE = re.compile(rf"^(?P<ord>(?:CHƯƠNG|Chương|CHỦ ĐỀ|Chủ đề|CHUYÊN ĐỀ|Chuyên đề|PHẦN|Phần)\s+(?:{_ROMAN}|\d{{1,2}}|MỘT|HAI|BA|BỐN|NĂM|SÁU|BẢY|TÁM|CHÍN|MƯỜI)\b)[\s.:–—-]*(?P<title>.*)$")
LESSON_RE = re.compile(r"^(?P<ord>(?:BÀI|Bài)\s+\d{1,3}[a-z]?)\b[\s.:–—-]*(?P<title>.*)$")
SECTION_RE = re.compile(rf"^(?P<ord>{_ROMAN})\s*[.)]\s*(?P<title>\S.*)$")
SUBSECTION_RE = re.compile(r"^(?P<ord>\d{1,2})\s*[.)]\s*(?P<title>\S.*)$")
TEXTSECTION_RE = re.compile(r"^(?P<ord>(?:VĂN BẢN|Văn bản)\s+\d{1,2})\b[\s.:–—-]*(?P<title>.*)$")
TOC_RE = re.compile(r"^\s*#*\s*(MỤC LỤC|Mục lục)\s*$", re.M)
# front matter ("how to use this book" pages show sample lesson headings): no structure until the next
# page that carries a chapter/lesson heading
FRONT_RE = re.compile(r"^\s*#*\s*\**\s*(HƯỚNG DẪN SỬ DỤNG SÁCH|Hướng dẫn sử dụng sách|MỤC LỤC|Mục lục)\b", re.M)
IMPERATIVE_RE = re.compile(r"^(Em hãy|Hãy|Nêu|Trình bày|Giải thích|Phân tích|Viết|Tính|Quan sát|Cho biết|So sánh|Vẽ|Lập|Tìm|"
                           r"Xác định|Kể|Mô tả|Chứng minh|Đọc|Thảo luận|Lấy ví dụ|Liệt kê|Dựa vào|Vận dụng|Chọn|"
                           r"Điền|Hoàn thành|Nhận xét|Đánh giá|Rút ra|Làm|Thực hiện|Theo em|Vì sao|Tại sao|Có|Khi nào)\b")

# Vietnamese pedagogical markers (box titles / paragraph starts) -> PedagogicalType
BOX_MARKERS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(em có biết|bạn có biết|có thể em chưa biết|thông tin thêm|tư liệu)", re.I), "reading_text", "Em có biết"),
    (re.compile(r"^(em đã học|tổng kết|ghi nhớ|kiến thức cốt lõi|kết luận|tóm tắt)", re.I), "summary", "Em đã học"),
    (re.compile(r"^(mục tiêu|yêu cầu cần đạt|học xong bài (học )?này|sau bài học này|sau khi học)", re.I), "objective", "Mục tiêu"),
    (re.compile(r"^(ví dụ|bài tập ví dụ|bài giải mẫu)", re.I), "example", "Ví dụ"),
    (re.compile(r"^(câu hỏi|trả lời câu hỏi|suy nghĩ và trả lời|câu hỏi và bài tập|dừng lại và suy ngẫm|\?\s)", re.I), "question", "Câu hỏi"),
    (re.compile(r"^(mở rộng|đọc thêm|tìm hiểu thêm)", re.I), "reading_text", "Mở rộng"),
    (re.compile(r"^(bài tập|luyện tập|vận dụng|em có thể|thực hành viết|viết|nói và nghe)\b", re.I), "exercise", "Bài tập"),
    (re.compile(r"^(hoạt động|thực hành|thí nghiệm|dự án|thảo luận|khám phá|tìm hiểu|trải nghiệm|mở đầu|khởi động|hình thành kiến thức)", re.I), "activity", "Hoạt động"),
    (re.compile(r"^(định nghĩa)", re.I), "definition", "Định nghĩa"),
    (re.compile(r"^(khái niệm)", re.I), "concept_explanation", "Khái niệm"),
    (re.compile(r"^(định luật)", re.I), "law", "Định luật"),
    (re.compile(r"^(nguyên lí|nguyên lý|quy tắc|quy luật)", re.I), "principle", "Nguyên lí"),
    (re.compile(r"^(công thức)", re.I), "formula", "Công thức"),
    (re.compile(r"^(điều kiện)", re.I), "condition", "Điều kiện"),
    (re.compile(r"^(quá trình|các bước|quy trình)", re.I), "process", "Quá trình"),
]
DEF_RE = re.compile(r"^[^.?!:]{2,120}?\s(?:là|được gọi là|gọi là|được định nghĩa là)\s", re.I)
CAPTION_RE = re.compile(r"^\**\s*(Hình|Sơ đồ|Bản đồ|Biểu đồ|Ảnh|Bảng)\s+\d+(?:[.,]\d+)*\s*[.:]", re.I)
COLOPHON_RE = re.compile(r"(nxbgd\.vn|ISBN|Chịu trách nhiệm xuất bản|Giá:\s*\d|In \d+ bản|Mã số:|Số đăng kí xuất bản)", re.I)
WORKED_RE = re.compile(r"\b(lời giải|hướng dẫn giải|giải\s*:)", re.I)


@dataclass
class Block:
    start: int
    end: int
    text: str
    kind: str                      # heading | paragraph | table | figure | formula | list


def split_blocks(text: str) -> list[Block]:
    """Split Markdown page text into blocks with exact [start, end) offsets."""
    blocks: list[Block] = []
    lines = text.splitlines(keepends=True)
    pos = 0
    cur: list[tuple[int, str]] = []   # (offset, line) of the current paragraph

    def kind_of(line: str) -> str:
        s = line.strip()
        if s.startswith("#"):
            return "heading"
        if s.startswith("|"):
            return "table"
        if re.match(r"^!?\[(Hình|Hinh|Ảnh|Sơ đồ|Biểu đồ|Bản đồ|Bảng)\b", s):
            return "figure"
        if s.startswith("$$"):
            return "formula"
        return "text"

    def flush():
        nonlocal cur
        if not cur:
            return
        start = cur[0][0]; last_off, last_line = cur[-1]
        end = last_off + len(last_line.rstrip())
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        start += lead
        if end > start:
            body = text[start:end]
            k = "list" if all(re.match(r"^\s*([-*•+]|\d+[.)]|[a-zđ][.)])\s", l) for _, l in cur if l.strip()) and len(cur) > 1 else "paragraph"
            blocks.append(Block(start, end, body, k))
        cur = []

    table: list[tuple[int, str]] = []
    in_display_formula = False
    for line in lines:
        off = pos; pos += len(line)
        s = line.strip()
        k = kind_of(line)
        if in_display_formula:
            cur.append((off, line))
            if s.endswith("$$") and len(s) >= 2:
                in_display_formula = False
                fs, fe = cur[0][0], off + len(line.rstrip())
                lead = len(text[fs:fe]) - len(text[fs:fe].lstrip())
                blocks.append(Block(fs + lead, fe, text[fs + lead:fe], "formula")); cur = []
            continue
        if k != "table" and table:
            ts, te = table[0][0] + (len(table[0][1]) - len(table[0][1].lstrip())), table[-1][0] + len(table[-1][1].rstrip())
            blocks.append(Block(ts, te, text[ts:te], "table")); table = []
        if not s:
            flush(); continue
        if k == "table":
            flush(); table.append((off, line)); continue
        if k in ("heading", "figure"):
            flush()
            st = off + (len(line) - len(line.lstrip())); en = off + len(line.rstrip())
            blocks.append(Block(st, en, text[st:en], k)); continue
        if k == "formula":
            flush()
            if s == "$$" or not (s.endswith("$$") and len(s) > 2):
                in_display_formula = True; cur.append((off, line)); continue
            st = off + (len(line) - len(line.lstrip())); en = off + len(line.rstrip())
            blocks.append(Block(st, en, text[st:en], "formula")); continue
        cur.append((off, line))
    if table:
        ts, te = table[0][0] + (len(table[0][1]) - len(table[0][1].lstrip())), table[-1][0] + len(table[-1][1].rstrip())
        blocks.append(Block(ts, te, text[ts:te], "table"))
    if in_display_formula:
        fs, fe = cur[0][0], cur[-1][0] + len(cur[-1][1].rstrip()); cur = []
        lead = len(text[fs:fe]) - len(text[fs:fe].lstrip())
        blocks.append(Block(fs + lead, fe, text[fs + lead:fe], "formula"))
    flush()
    blocks.sort(key=lambda b: b.start)
    return blocks


def _clean_heading(s: str) -> str:
    s = s.strip().lstrip("#").strip()
    s = re.sub(r"^\*\*(.*)\*\*$", r"\1", s).strip()
    return " ".join(s.split())


def classify_heading(line: str, is_md_heading: bool) -> Optional[tuple[str, str, str]]:
    """Return (level, ordinal, title) for a structural heading line, else None."""
    s = _clean_heading(line)
    if not s or len(s) > 200:
        return None
    for level, rx in (("chapter", CHAPTER_RE), ("lesson", LESSON_RE)):
        m = rx.match(s)
        if m:
            # plain (non-#) lines only count when short, to avoid sentences like "Bài 3 cho biết ..."
            if not is_md_heading and len(s) > 90:
                return None
            return level, " ".join(m.group("ord").split()), m.group("title").strip(" .:–—-|")
    if not is_md_heading:
        # I./1. numbering on a plain line counts only for a short standalone title-cased line without
        # final punctuation (the OCR does not always mark headings with #)
        if len(s) > 90 or s.endswith((".", ":", "?", ";", ",", "!")):
            return None
        for level, rx, suffix in (("section", SECTION_RE, "."), ("subsection", SUBSECTION_RE, ".")):
            m = rx.match(s)
            if m and _title_like(m.group("title")) and m.group("title")[:1].isupper():
                return level, m.group("ord") + suffix, m.group("title").strip()
        return None
    m = TEXTSECTION_RE.match(s)
    if m:
        return "section", " ".join(m.group("ord").split()), m.group("title").strip(" .:–—-|")
    m = SECTION_RE.match(s)
    if m and _title_like(m.group("title")):
        return "section", m.group("ord") + ".", m.group("title").strip()
    m = SUBSECTION_RE.match(s)
    if m and _title_like(m.group("title")):
        return "subsection", m.group("ord") + ".", m.group("title").strip()
    return None


def _title_like(t: str) -> bool:
    """Numbered headings that are really questions/exercises ("1. Viết công thức ... ?") are not structure."""
    t = t.strip().strip("*").strip()
    if len(t) > 110 or t.endswith(("?", ".", ":")) or IMPERATIVE_RE.match(t):
        return False
    if re.search(r"[.!?;]\s+\S", t):          # more than one sentence
        return False
    return True


_BOX_CONTAINER_TYPES = {"objective", "question", "exercise", "activity", "summary", "reading_text", "example"}


def _is_box_heading(text: str) -> bool:
    """Box titles (MỤC TIÊU, Câu hỏi, Em có biết, ...) - not content words such as "Định luật ..." / "Quy tắc ..."."""
    bm = box_marker(text)
    return bool(bm) and bm[0] in _BOX_CONTAINER_TYPES


def box_marker(text: str) -> Optional[tuple[str, str]]:
    s = _clean_heading(text).strip("*_ :")
    for rx, ptype, label in BOX_MARKERS:
        if rx.match(s):
            return ptype, label
    return None


def classify_paragraph(text: str, kind: str, context: Optional[tuple[str, str]], in_lesson: bool) -> tuple[str, Optional[str]]:
    """Heuristic PedagogicalType (+ Vietnamese marker label) for one block."""
    if kind == "table":
        return "table", None
    if kind == "figure" or CAPTION_RE.match(text):
        return "figure_caption", None
    if kind == "formula":
        return "formula", "Công thức"
    if kind == "heading":
        bm = box_marker(text)
        if bm:
            return bm[0], bm[1]
        if len(_clean_heading(text)) <= 80:
            return "other", None
        text = _clean_heading(text)            # long "heading" = OCR-marked sentence (question/exercise)
        if re.match(r"^\d{1,2}\s*[.)]\s", text) and not text.endswith("?"):
            return ("exercise", None) if context is None else (context[0], context[1])
    if COLOPHON_RE.search(text):
        return "other", "colophon"
    own = box_marker(text)
    if own:
        ptype, label = own
        if ptype == "example" and WORKED_RE.search(text):
            return "worked_example", label
        return ptype, label
    if context:
        ptype, label = context
        if ptype == "example" and WORKED_RE.search(text):
            return "worked_example", label
        return ptype, label
    s = text.strip()
    if s.endswith("?") and len(s) < 400:
        return "question", None
    if re.search(r"\bđịnh luật\b", s, re.I):
        return "law", "Định luật"
    if DEF_RE.match(s.lstrip("-*• ")):
        return "definition", None
    if s.count("$") >= 2 and len(re.sub(r"\$[^$]*\$", "", s).strip()) < 40:
        return "formula", "Công thức"
    return ("concept_explanation" if in_lesson else "other"), None


# extra (non-contract) per-span fields, keyed by span_id; written next to the SourceSpan fields
_EXTRAS: dict[str, dict] = {}


LEVELS = ["book", "chapter", "lesson", "section", "subsection"]
_ID_PREFIX = {"chapter": "ch", "lesson": "l", "section": "s", "subsection": "ss"}


def build_structure(book_id: str, pages: list[dict], book_title: Optional[str] = None,
                    source_sha256: str = "", extractor_version: str = "") -> tuple[list[StructureNode], list[SourceSpan], dict[int, Optional[str]]]:
    """Return (nodes, spans, page_section_ids)."""
    root = StructureNode(section_id=book_id, book_id=book_id, level="book", title=book_title or book_id, heading_path=[])
    nodes: list[StructureNode] = [root]
    stack: list[StructureNode] = [root]               # current path, root first
    counters = {"chapter": 0, "lesson": 0}
    child_counters: dict[str, int] = {}
    spans: list[SourceSpan] = []
    page_sections: dict[int, Optional[str]] = {}
    pending_title: Optional[StructureNode] = None     # chapter/lesson heading without title on its line
    max_seen: dict[str, int] = {}
    lessons_with_I: set[str] = set()
    title_cont: Optional[tuple] = None                # (node, heading level, n) for multi-line titles
    front_matter = False

    def labels_of(p):
        return (p.get("printed_page_label") if p.get("printed_page_verified") else None)

    def open_node(level: str, ordinal: str, title: str, page: dict, md: bool) -> StructureNode:
        if level == "subsection":
            cur = next((x for x in reversed(stack) if x.level in ("lesson", "section")), None)
            if cur is not None and (cur.level == "lesson" or re.match(r"^\d", cur.ordinal or "")):
                level = "section"          # "1." numbering directly under a lesson without Roman sections
        target = LEVELS.index(level)
        while len(stack) > 1 and LEVELS.index(stack[-1].level) >= target:
            stack.pop()
        parent = stack[-1]
        expected_parent = LEVELS[target - 1]
        unresolved = parent.level != expected_parent and not (level == "lesson" and parent.level == "book") \
            and not (level == "chapter" and parent.level == "book")
        if level in counters:
            counters[level] += 1; n = counters[level]
            if level == "chapter":
                sid = f"{book_id}:ch{n:02d}"
            else:
                chap = next((s for s in reversed(stack) if s.level == "chapter"), None)
                sid = f"{chap.section_id if chap else book_id + ':ch00'}:l{n:02d}"
        else:
            child_counters[parent.section_id] = child_counters.get(parent.section_id, 0) + 1
            sid = f"{parent.section_id}:{_ID_PREFIX[level]}{child_counters[parent.section_id]:02d}"
        full_title = f"{ordinal} {title}".strip() if level in ("chapter", "lesson") else f"{ordinal} {title}".strip()
        node = StructureNode(section_id=sid, book_id=book_id, level=level, title=title or ordinal, parent_id=parent.section_id,
                             ordinal=ordinal, pdf_page_start=page["pdf_page_index"], pdf_page_end=page["pdf_page_index"],
                             printed_page_start=labels_of(page), printed_page_end=labels_of(page),
                             heading_path=[*(s.title if s.level == "book" else _path_label(s) for s in stack[1:]), full_title],
                             link_status="unresolved" if unresolved else "resolved")
        nodes.append(node); stack.append(node)
        return node

    def adopt_lead(node: StructureNode, title: str, lead_heads: list, idx: int) -> None:
        node.title = title
        if node.ordinal:
            node.heading_path[-1] = f"{node.ordinal} {title}"
        page_sections[idx] = node.section_id
        for _, k in lead_heads:          # re-attach the title spans to the new node
            sp = spans[k]
            sp.section_id = node.section_id
            sp.heading_path = [_path_label(x) for x in stack[1:]]
            _EXTRAS[sp.span_id].update({
                "chapter_id": next((x.section_id for x in stack if x.level == "chapter"), None),
                "lesson_id": next((x.section_id for x in stack if x.level == "lesson"), None)})

    for page in sorted(pages, key=lambda p: p["pdf_page_index"]):
        text = page.get("text") or ""
        idx = page["pdf_page_index"]
        is_toc = bool(TOC_RE.search(text))
        if FRONT_RE.search(text):
            front_matter = True
        elif front_matter and any(
                (h := classify_heading(b.text, b.kind == "heading")) and h[0] in ("chapter", "lesson")
                for b in split_blocks(text) if b.kind in ("heading", "paragraph") and "\n" not in b.text):
            front_matter = False
        is_toc = is_toc or front_matter
        blocks = split_blocks(text)
        page_sections[idx] = stack[-1].section_id if len(stack) > 1 else None
        first_content = True
        context: Optional[tuple[str, str]] = None
        unit = 0
        lead_heads: list[tuple[Block, int]] = []   # leading non-structural headings of this page (block, span index)
        lead_open = True
        lesson_on_page = False
        page_top_heads: list[tuple[Block, int]] = []
        for b in blocks:
            md = b.kind == "heading"
            if title_cont is not None:
                node_c, hashes, n_cont = title_cont
                cand = _clean_heading(b.text)
                if (md and _hashes(b.text) == hashes and cand and cand.upper() == cand and not cand[:1].isdigit()
                        and classify_heading(b.text, md) is None and not box_marker(b.text) and n_cont < 2
                        and len(node_c.title) + len(cand) <= 200):
                    # multi-line printed title ("MỘT SỐ NỀN VĂN MINH THẾ GIỚI" / "THỜI CỔ – TRUNG ĐẠI")
                    node_c.title = f"{node_c.title} {cand}"
                    node_c.heading_path[-1] = f"{node_c.ordinal} {node_c.title}".strip()
                    title_cont = (node_c, hashes, n_cont + 1)
                    _emit(spans, book_id, page, b, unit, stack, "heading", "other", None, source_sha256, extractor_version); unit += 1
                    continue
                title_cont = None
            if pending_title is not None:
                cand = _clean_heading(b.text)
                if b.kind in ("heading", "paragraph") and cand and len(cand) <= 160 and classify_heading(b.text, md) is None \
                        and not ((box_marker(b.text) or ("",))[0] == "objective"
                                 or (box_marker(b.text) and len(cand) < 25 and pending_title.level != "chapter")) \
                        and (md or cand.upper() == cand):
                    pending_title.title = cand
                    pending_title.heading_path[-1] = f"{pending_title.ordinal} {cand}"
                    if md:
                        title_cont = (pending_title, _hashes(b.text), 0)
                    pending_title = None
                    _emit(spans, book_id, page, b, unit, stack, "heading", "other", None, source_sha256, extractor_version); unit += 1
                    continue
                pending_title = None
            h = None if is_toc else (classify_heading(b.text, md) if b.kind in ("heading", "paragraph") and "\n" not in b.text else None)
            if h is None and not is_toc and md and first_content and _hashes(b.text) <= 2 and "lesson" in max_seen:
                # "## 5 THỰC HÀNH: ..." at the top of a page: the graphic "BÀI" label was not transcribed but the
                # number was; accept only the next few lesson numbers after the last one seen
                m = re.match(r"^(\d{1,3})\s+(\S.*)$", _clean_heading(b.text))
                if m and m.group(2).upper() == m.group(2) and max_seen["lesson"] < int(m.group(1)) <= max_seen["lesson"] + 3:
                    h = ("lesson", m.group(1), m.group(2).strip(" .:–—-|"))
            if h and h[0] in ("chapter", "lesson"):
                num = _ordinal_value(h[1])
                key = "lesson" if h[0] == "lesson" else ("chapter" if re.match(r"(?i)^ch", h[1]) else None)
                if key and num is not None:
                    if num < max_seen.get(key, 0) or (num == max_seen.get(key, 0)
                                                      and any(x.level == h[0] and _ordinal_value(x.ordinal or "") == num
                                                              for x in stack)):
                        # running header / repeated title of the current or an earlier unit -> plain text,
                        # and it neither closes nor joins the page's leading title headings
                        _emit(spans, book_id, page, b, unit, stack, b.kind, "other", "running_header",
                              source_sha256, extractor_version); unit += 1
                        continue
                    max_seen[key] = num
            if h and h[0] == "section" and h[1] == "I." and page_top_heads and not lesson_on_page:
                cur_lesson = next((x for x in reversed(stack) if x.level == "lesson"), None)
                cand = " ".join(_clean_heading(x.text) for x, _ in page_top_heads)
                if cur_lesson is not None and cur_lesson.section_id in lessons_with_I and cand.upper() == cand \
                        and len(cand) <= 200 and not cand[:1].isdigit():
                    # a second "I." inside one lesson after an all-caps page-top title: new lesson whose
                    # "Bài n" label is a graphic the OCR did not transcribe
                    node = open_node("lesson", "", cand, page, True)
                    node.ordinal = None; node.heading_path[-1] = cand; node.link_status = "unresolved"
                    adopt_lead(node, cand, page_top_heads, idx)
                    lesson_on_page = True
            if h:
                level, ordinal, title = h
                node = open_node(level, ordinal, title, page, md)
                if level == "section" and ordinal == "I." and node.parent_id:
                    lessons_with_I.add(node.parent_id)
                if not title and level in ("chapter", "lesson") and lead_heads and lead_open:
                    # e.g. "# XU HƯỚNG BIẾN ĐỔI ..." / "## BÀI 7": the title is printed before the ordinal
                    cand = " ".join(_clean_heading(x.text) for x, _ in lead_heads)
                    if len(cand) <= 200:
                        adopt_lead(node, cand, lead_heads, idx)
                        title = cand
                lead_heads, lead_open = [], level == "chapter"
                if level == "lesson":
                    lesson_on_page = True
                if not title and (level in ("chapter", "lesson") or ordinal.upper().startswith("VĂN BẢN")):
                    pending_title = node
                elif title and md and level in ("chapter", "lesson"):
                    title_cont = (node, _hashes(b.text), 0)
                if first_content:
                    page_sections[idx] = node.section_id
                context = None
                _emit(spans, book_id, page, b, unit, stack, "heading", "other", None, source_sha256, extractor_version); unit += 1
                first_content = False
                continue
            if (not is_toc and lead_open and lead_heads and not lesson_on_page and b.kind in ("heading", "paragraph")
                    and (box_marker(b.text) or ("", ""))[0] == "objective"):
                # lesson whose "BÀI n" label is a graphic the OCR did not transcribe: top-of-page title heading(s)
                # immediately followed by the objectives box ("YÊU CẦU CẦN ĐẠT", "MỤC TIÊU", "Học xong bài này").
                heads = [_clean_heading(x.text) for x, _ in lead_heads]
                cand = " ".join(heads)
                if cand.upper() == cand and len(cand) <= 200:
                    m = re.match(r"^(\d{1,3})\s+(\S.*)$", cand)
                    ordinal, ltitle = (m.group(1), m.group(2)) if m else (None, cand)
                    node = open_node("lesson", ordinal or "", ltitle, page, True)
                    node.ordinal = ordinal
                    node.heading_path[-1] = ltitle if ordinal is None else f"{ordinal} {ltitle}"
                    node.link_status = "unresolved"    # inferred lesson boundary (ordinal not transcribed)
                    adopt_lead(node, ltitle, lead_heads, idx)
                    lesson_on_page = True; lead_heads, lead_open = [], False
            for s in stack[1:]:
                s.pdf_page_end = idx
                if labels_of(page):
                    s.printed_page_end = labels_of(page)
            in_lesson = any(s.level == "lesson" for s in stack)
            if b.kind == "heading":
                bm = box_marker(b.text)
                context = bm  # a non-structural heading starts (or ends) a box context
                ptype, label = classify_paragraph(b.text, b.kind, None, in_lesson)
            elif b.kind == "paragraph" and b.text.strip() in ("?", "❓", "**?**"):
                context = ("question", "Câu hỏi"); ptype, label = context
            elif b.kind == "paragraph" and len(b.text) < 60 and box_marker(b.text) and not b.text.rstrip().endswith("."):
                context = box_marker(b.text); ptype, label = context
            else:
                ptype, label = classify_paragraph(b.text, b.kind, context, in_lesson)
            if is_toc:
                ptype, label = "other", "Mục lục"
            if lead_open and b.kind == "heading" and not _is_box_heading(b.text) and len(lead_heads) < 4:
                lead_heads.append((b, len(spans)))
                if _hashes(b.text) <= 2:
                    page_top_heads.append((b, len(spans)))
            else:
                lead_open = False
            _emit(spans, book_id, page, b, unit, stack, b.kind, ptype, label, source_sha256, extractor_version); unit += 1
            first_content = False
    return nodes, spans, page_sections


_ROMAN_VAL = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_WORD_NUM = {"MỘT": 1, "HAI": 2, "BA": 3, "BỐN": 4, "NĂM": 5, "SÁU": 6, "BẢY": 7, "TÁM": 8, "CHÍN": 9, "MƯỜI": 10}


def _ordinal_value(ordinal: str) -> Optional[int]:
    if not ordinal or not ordinal.split():
        return None
    tok = ordinal.split()[-1].upper().rstrip(".")
    if tok.isdigit():
        return int(tok)
    m = re.match(r"^(\d+)[A-Z]$", tok)
    if m:
        return int(m.group(1))
    if tok in _WORD_NUM:
        return _WORD_NUM[tok]
    if re.fullmatch(r"[IVXLC]+", tok):
        total, prev = 0, 0
        for c in reversed(tok):
            v = _ROMAN_VAL[c]; total += -v if v < prev else v; prev = max(prev, v)
        return total
    return None


def _hashes(line: str) -> int:
    s = line.lstrip()
    return len(s) - len(s.lstrip("#"))


def _path_label(n: StructureNode) -> str:
    return f"{n.ordinal} {n.title}".strip() if n.ordinal and n.title and n.title != n.ordinal else (n.title or n.ordinal or "")


def _emit(spans: list, book_id: str, page: dict, b: Block, unit: int, stack: list, kind: str, ptype: str,
          label: Optional[str], sha: str, version: str) -> None:
    sec = stack[-1].section_id if len(stack) > 1 else None
    span = SourceSpan(
        span_id=f"{book_id}:p{page['pdf_page_index']:04d}:u{unit:03d}", book_id=book_id, section_id=sec,
        heading_path=[_path_label(s) for s in stack[1:]], pdf_page_index=page["pdf_page_index"],
        printed_page_label=page.get("printed_page_label"), printed_page_verified=bool(page.get("printed_page_verified")),
        start_offset=b.start, end_offset=b.end, text=b.text, pedagogical_type=ptype,
        source_file_sha256=page.get("source_file_sha256") or sha, extractor_version=page.get("extractor_version") or version,
        evidence_status="ocr_unverified" if page.get("extraction_method") in ("vlm_ocr", "tesseract_ocr", "paddle_ocr") else "extracted")
    _EXTRAS[span.span_id] = {"span_kind": kind, "text_sha1": hashlib.sha1(b.text.encode("utf-8")).hexdigest(), "pedagogical_label_vi": label, "page_span_id": page.get("span_id"),
                   "chapter_id": next((s.section_id for s in stack if s.level == "chapter"), None),
                   "lesson_id": next((s.section_id for s in stack if s.level == "lesson"), None),
                   "page_quality_score": page.get("quality_score"),
                   "requires_manual_review": bool(page.get("requires_manual_review"))}
    spans.append(span)


def span_row(span: SourceSpan) -> dict:
    """SourceSpan fields + WS1 extras (span_kind, pedagogical_label_vi, chapter_id, lesson_id, ...)."""
    row = span.model_dump()
    row.update(_EXTRAS.get(span.span_id, {}))
    return row


def process_book(parsed_dir: Path, book_title: Optional[str] = None) -> tuple[int, int]:
    """Read pages.jsonl, write sections.jsonl + spans.jsonl and update pages.jsonl section_ids."""
    pages_path = parsed_dir / "pages.jsonl"
    pages = [json.loads(l) for l in pages_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not pages:
        return 0, 0
    book_id = pages[0]["book_id"]
    nodes, spans, page_secs = build_structure(book_id, pages, book_title, pages[0].get("source_file_sha256", ""))
    for p in pages:
        p["section_id"] = page_secs.get(p["pdf_page_index"])
    _atomic_write(pages_path, (json.dumps(p, ensure_ascii=False, sort_keys=True) for p in pages))
    _atomic_write(parsed_dir / "sections.jsonl", (json.dumps(n.model_dump() | {"structure_version": STRUCTURE_VERSION}, ensure_ascii=False, sort_keys=True) for n in nodes))
    _atomic_write(parsed_dir / "spans.jsonl", (json.dumps(span_row(s), ensure_ascii=False, sort_keys=True) for s in spans))
    return len(nodes), len(spans)


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    tmp.replace(path)
