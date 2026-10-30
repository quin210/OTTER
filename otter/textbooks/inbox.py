"""Book records for unmatched local PDFs (``local_<sha256[:10]>``).

Metadata comes ONLY from the OCR'd cover / title pages of the file itself (never from the
file name, which is kept as a locator, and never invented). Every field derived here is a
candidate: the record is flagged ``metadata_review_needed`` and carries the evidence
(page index + the exact heading lines it was read from).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

SUBJECT_KEYWORDS = [  # (normalized subject, pattern on the OCR'd cover heading text)
    ("natural_science", r"KHOA HỌC\s+TỰ NHIÊN"),
    ("history_geography", r"LỊCH SỬ\s+(VÀ|&)\s+ĐỊA LÍ"),
    ("literature", r"NGỮ VĂN"),
    ("chemistry", r"HO[ÁÀ]\s*H[OỌ]C|HOÁ HỌC|HÓA HỌC"),
    ("physics", r"VẬT L[ÍÝ]"),
    ("biology", r"SINH HỌC"),
    ("history", r"LỊCH SỬ"),
    ("geography", r"ĐỊA L[ÍÝ]"),
]
SKIP_LINE = re.compile(r"(NHÀ XUẤT BẢN|NXB|Chủ biên|CHỦ BIÊN|BỘ GIÁO DỤC|Tái bản|TÁI BẢN|ISBN|Giá|GIÁ|bản quyền)", re.I)
SERIES_RE = re.compile(r"KẾT NỐI TRI THỨC\s*(VỚI CUỘC SỐNG)?|CHÂN TRỜI SÁNG TẠO|CÁNH DIỀU", re.I)
VOLUME_RE = re.compile(r"\b(TẬP\s+(MỘT|HAI|BA|1|2|3))\b", re.I)


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _pages(parsed_dir: Path, n: int = 5) -> list[dict]:
    f = parsed_dir / "pages.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["pdf_page_index"])
    return rows[:n]


def cover_metadata(pages: list[dict]) -> dict:
    """Candidate title/grade/subject/series/volume from the first OCR'd pages with title-like headings."""
    best = None
    for p in pages:
        text = _norm(p.get("text") or "")
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        heads = [l.lstrip("#").strip().strip("*") for l in lines if l.startswith("#")]
        heads = [h for h in heads if h and not SKIP_LINE.search(h)]
        if not heads:
            continue
        subj = next((s for s, rx in SUBJECT_KEYWORDS if re.search(rx, " ".join(heads).upper())), None)
        score = len(heads) + (5 if subj else 0)
        if best is None or score > best[0]:
            best = (score, p, heads, lines, subj)
        if subj:
            break
    if best is None:
        return {"title": None, "metadata_evidence": {"reason": "no title-like heading found on the first OCR'd pages"}}
    _, p, heads, lines, subj = best
    title_heads = heads[:4]
    title = " ".join(title_heads)
    title = re.sub(r"\s+", " ", title).strip()
    grade = None
    # grade = a standalone number 1..12 printed right after the title headings, or at the end of the title
    m = re.search(r"\b(1[0-2]|[1-9])\s*$", title)
    if m:
        grade = int(m.group(1))
    else:
        try:
            k = lines.index(next(l for l in lines if l.lstrip("#").strip().strip("*") == title_heads[-1]))
            for l in lines[k + 1:k + 3]:
                if re.fullmatch(r"#*\s*\**(1[0-2]|[1-9])\**", l):
                    grade = int(re.sub(r"\D", "", l)); title = f"{title} {grade}"; break
        except StopIteration:
            pass
    full = " ".join(lines)
    series = SERIES_RE.search(full)
    vol = VOLUME_RE.search(full)
    return {
        "title": title or None,
        "grade": grade,
        "subject_normalized": subj,
        "series": series.group(0).strip() if series else None,
        "volume": vol.group(1).lower() if vol else None,
        "metadata_evidence": {"pdf_page_index": p["pdf_page_index"], "page_span_id": p.get("span_id"),
                              "heading_lines": heads[:8],
                              "note": "read from the OCR'd cover/title page; candidates only, not verified against a catalog"},
    }


def local_book_record(book_id: str, parsed_dir: Path, job: Optional[dict] = None) -> dict:
    job = job or {}
    meta = cover_metadata(_pages(parsed_dir))
    pages = _pages(parsed_dir, 1)
    lf = job.get("local_file")
    if lf:
        try:
            lf = str(Path(lf).resolve().relative_to(parsed_dir.resolve().parents[3]))
        except (ValueError, IndexError):
            pass
    sha = job.get("sha256") or (pages[0].get("source_file_sha256") if pages else None)
    return {
        "book_id": book_id, "sha256": sha, "local_file": lf,
        "filename_locator": job.get("filename"),
        "catalog_match_status": "metadata_review_needed", "metadata_review_needed": True,
        "metadata_source": "ocr_cover_page", "access_status": "user_supplied",
        **meta,
    }
