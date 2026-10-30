"""Builder step 1: load immutable book / section / span provenance produced by WS1.

Reads the provenance DB (`books`, `sections`, `spans`) read-only; falls back to
`data/textbooks/parsed/<book_id>/spans.jsonl` for books whose spans are not (yet) in the DB.
Never modifies provenance.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from otter.schemas.textbook import SourceSpan
from otter.storage.sqlite_store import SqliteOntologyStore

_UNIT_RE = re.compile(r":u\d+$")


@dataclass
class BookInfo:
    book_id: str
    subject: str
    grade: Optional[int]
    title: str
    meta: dict = field(default_factory=dict)


def _connect_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master")}


def load_books(db_path: Path, book_ids: Optional[list[str]] = None, registry: Optional[Path] = None
               ) -> dict[str, BookInfo]:
    metas: dict[str, dict] = {}
    if registry and (registry / "books.jsonl").exists():
        for line in (registry / "books.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                metas[d["book_id"]] = d
    if db_path.exists():
        con = _connect_ro(db_path)
        if "books" in _tables(con):
            for r in con.execute("SELECT * FROM books"):
                d = dict(r)
                try:
                    m = json.loads(d.pop("metadata_json", None) or "{}")
                except json.JSONDecodeError:
                    m = {}
                m.update({k: v for k, v in d.items() if v is not None})
                metas[d["book_id"]] = {**metas.get(d["book_id"], {}), **m}
        con.close()
    out = {}
    for bid, m in metas.items():
        if book_ids and bid not in book_ids:
            continue
        subj = m.get("subject") or m.get("subject_normalized") or "other"
        grade = m.get("grade")
        out[bid] = BookInfo(bid, subj, int(grade) if grade is not None else None, m.get("title") or bid, m)
    return out


def load_sections(db_path: Path, book_ids: list[str], parsed_dir: Optional[Path] = None) -> dict[str, dict]:
    secs: dict[str, dict] = {}
    if parsed_dir:
        for bid in book_ids:
            p = parsed_dir / bid / "sections.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        secs[d["section_id"]] = d
    if db_path.exists():
        con = _connect_ro(db_path)
        if "sections" in _tables(con) and book_ids:
            q = f"SELECT * FROM sections WHERE book_id IN ({','.join('?' * len(book_ids))})"
            for r in con.execute(q, book_ids):
                d = dict(r)
                try:
                    m = json.loads(d.pop("metadata_json", None) or "{}")
                except json.JSONDecodeError:
                    m = {}
                m.update({k: v for k, v in d.items() if v is not None})
                secs[d["section_id"]] = {**secs.get(d["section_id"], {}), **m}
        con.close()
    return secs


def load_spans(db_path: Path, book_ids: list[str], parsed_dir: Optional[Path] = None) -> dict[str, list[SourceSpan]]:
    """{book_id: [SourceSpan]} in reading order. Paragraph spans (`...:uNNN`) are preferred over
    page-level records when both exist for a book. Empty-text spans are skipped."""
    out: dict[str, list[SourceSpan]] = {b: [] for b in book_ids}
    if db_path.exists():
        con = _connect_ro(db_path)
        if "spans" in _tables(con):
            for bid in book_ids:
                for r in con.execute("SELECT * FROM spans WHERE book_id=?", (bid,)):
                    s = SqliteOntologyStore._span_from_row(r)
                    if s.text.strip():
                        out[bid].append(s)
        con.close()
    if parsed_dir:
        for bid in book_ids:
            if out[bid]:
                continue
            p = parsed_dir / bid / "spans.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        s = SourceSpan.model_validate(json.loads(line))
                        if s.text.strip():
                            out[bid].append(s)
    for bid, spans in out.items():
        if any(_UNIT_RE.search(s.span_id) for s in spans):
            spans[:] = [s for s in spans if _UNIT_RE.search(s.span_id)]
        spans.sort(key=lambda s: (s.pdf_page_index, s.start_offset, s.span_id))
    return out


def existing_span_ids(db_path: Path, span_ids: list[str]) -> set[str]:
    """Hard provenance check against the DB (not against an in-memory copy)."""
    found: set[str] = set()
    if not span_ids or not db_path.exists():
        return found
    con = _connect_ro(db_path)
    if "spans" in _tables(con):
        ids = list(dict.fromkeys(span_ids))
        for i in range(0, len(ids), 500):
            ch = ids[i:i + 500]
            found.update(r[0] for r in con.execute(
                f"SELECT span_id FROM spans WHERE span_id IN ({','.join('?' * len(ch))})", ch))
    con.close()
    return found
