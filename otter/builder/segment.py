"""Builder step 2: group spans by lesson/section into extraction chunks (~1.5-3k tokens).

Hierarchy detection itself is WS1's job (sections.jsonl); here we only pick the lesson-level
ancestor of each span's section and pack consecutive spans of the same lesson into chunks.
Small consecutive lessons are packed together; large lessons are split at span boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from otter.schemas.textbook import SourceSpan
from otter.storage.textutil import short_hash

SKIP_TYPES = frozenset({"exercise", "question"})      # avoid builder/benchmark leakage; see spec 7.4
CHARS_PER_TOKEN = 3.0                                  # Vietnamese on Qwen tokenizer (calibrated from usage)


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class Chunk:
    chunk_id: str
    book_id: str
    subject: str
    grade: Optional[int]
    book_title: str
    section_ids: list[str]
    heading_paths: list[list[str]]
    spans: list[SourceSpan] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(est_tokens(s.text) for s in self.spans)

    @property
    def lesson_section_id(self) -> Optional[str]:
        return self.section_ids[0] if self.section_ids else None

    def local_ids(self) -> dict[str, str]:
        """S1..Sn -> real span ids (short ids keep the LLM from mangling long ids)."""
        return {f"S{i + 1}": s.span_id for i, s in enumerate(self.spans)}


def lesson_of(section_id: Optional[str], sections: dict[str, dict]) -> Optional[str]:
    seen = set()
    cur = section_id
    best = section_id
    while cur and cur in sections and cur not in seen:
        seen.add(cur)
        lvl = sections[cur].get("level")
        if lvl == "lesson":
            return cur
        if lvl in ("book", "chapter"):
            return best
        best = cur
        cur = sections[cur].get("parent_id")
    return best


def segment_book(book_id: str, spans: list[SourceSpan], sections: dict[str, dict], *, subject: str,
                 grade: Optional[int], book_title: str, min_tokens: int = 1500, max_tokens: int = 3000,
                 include_exercises: bool = False) -> list[Chunk]:
    groups: list[tuple[Optional[str], list[SourceSpan]]] = []
    # front matter (cover, credits, table of contents) = unsectioned spans before the first sectioned page
    sectioned = [s.pdf_page_index for s in spans if s.section_id and s.section_id != book_id]
    first_page = min(sectioned) if sectioned else -1
    for s in spans:
        if not include_exercises and s.pedagogical_type in SKIP_TYPES:
            continue
        if (not s.section_id or s.section_id == book_id) and s.pdf_page_index < first_page:
            continue
        key = lesson_of(s.section_id, sections) if s.section_id else f"page:{s.pdf_page_index // 4}"
        if groups and groups[-1][0] == key:
            groups[-1][1].append(s)
        else:
            groups.append((key, [s]))

    chunks: list[Chunk] = []
    cur: Optional[Chunk] = None

    def new(key):
        return Chunk("", book_id, subject, grade, book_title, [key] if key and not key.startswith("page:") else [],
                     [])

    def close():
        nonlocal cur
        if cur and cur.spans:
            ids = [s.span_id for s in cur.spans]
            cur.chunk_id = f"{book_id}:ch:{short_hash(*ids, n=10)}"
            hp = []
            for s in cur.spans:
                if s.heading_path and s.heading_path not in hp:
                    hp.append(s.heading_path)
            cur.heading_paths = hp[:6]
            chunks.append(cur)
        cur = None

    for key, gspans in groups:
        gtok = sum(est_tokens(s.text) for s in gspans)
        if cur is not None and (cur.tokens >= min_tokens or cur.tokens + gtok > max_tokens):
            close()
        if cur is None:
            cur = new(key)
        elif key and not key.startswith("page:") and key not in cur.section_ids:
            cur.section_ids.append(key)
        for s in gspans:
            t = est_tokens(s.text)
            if cur.spans and cur.tokens + t > max_tokens:
                close()
                cur = new(key)
            cur.spans.append(s)
    close()
    # fold tiny chunks (< min_tokens / 5) into their predecessor when the result stays <= 1.2 * max_tokens
    merged: list[Chunk] = []
    for c in chunks:
        if merged and c.tokens < min_tokens / 5 and merged[-1].tokens + c.tokens <= 1.2 * max_tokens:
            p = merged[-1]
            p.spans += c.spans
            p.section_ids += [x for x in c.section_ids if x not in p.section_ids]
            p.heading_paths = (p.heading_paths + [h for h in c.heading_paths if h not in p.heading_paths])[:6]
            p.chunk_id = f"{book_id}:ch:{short_hash(*[s.span_id for s in p.spans], n=10)}"
        else:
            merged.append(c)
    return merged
