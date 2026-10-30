"""CitationResolver: joins span ids against real builder/provenance metadata.

Never produces a citation from memory: unknown span ids resolve to ``None`` and are
reported as invalid. ``page_number`` is the printed page label only when the span's
``printed_page_verified`` flag is true; otherwise it is ``None`` (no confirmed page).
"""
from __future__ import annotations

import threading
from typing import Any, Iterable, Optional

from otter.schemas.reasoning import Citation


def _get(d: Any, key: str, default: Any = None) -> Any:
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None and str(x).strip() != "" else None
    except (TypeError, ValueError):
        return None


class CitationResolver:
    def __init__(self, store: Any) -> None:
        self.store = store
        self._book_cache: dict[str, Optional[dict]] = {}
        self._section_cache: dict[str, Optional[dict]] = {}
        self._lock = threading.Lock()

    def _book(self, book_id: str) -> Optional[dict]:
        with self._lock:
            if book_id in self._book_cache:
                return self._book_cache[book_id]
        try:
            meta = self.store.book_metadata(book_id)
        except Exception:
            meta = None
        with self._lock:
            self._book_cache[book_id] = meta
        return meta

    def _section(self, section_id: Optional[str]) -> Optional[dict]:
        if not section_id:
            return None
        with self._lock:
            if section_id in self._section_cache:
                return self._section_cache[section_id]
        try:
            sec = self.store.section(section_id)
        except Exception:
            sec = None
        with self._lock:
            self._section_cache[section_id] = sec
        return sec

    def _ancestors(self, section_id: Optional[str]) -> list[dict]:
        out, seen = [], set()
        cur = self._section(section_id)
        while cur is not None and len(out) < 8:
            sid = _get(cur, "section_id")
            if sid in seen:
                break
            seen.add(sid)
            out.append(cur)
            parent = _get(cur, "parent_id")
            if not parent and isinstance(cur, dict) and isinstance(cur.get("metadata_json"), dict):
                parent = cur["metadata_json"].get("parent_id")
            cur = self._section(parent) if parent else None
        return out

    def resolve_many(self, span_ids: Iterable[str], *, with_quote: bool = True, quote_chars: int = 300
                     ) -> tuple[list[Citation], list[str]]:
        ids = [s for s in dict.fromkeys(span_ids) if s]
        if not ids:
            return [], []
        try:
            spans = {s.span_id: s for s in self.store.get_spans(ids)}
        except Exception:
            spans = {}
        cites, invalid = [], []
        for sid in ids:
            sp = spans.get(sid)
            if sp is None:
                invalid.append(sid)
                continue
            book = self._book(sp.book_id) or {}
            chain = self._ancestors(sp.section_id)
            lesson = next((c for c in chain if _get(c, "level") == "lesson"), None)
            chapter = next((c for c in chain if _get(c, "level") == "chapter"), None)
            cites.append(Citation(
                book_id=sp.book_id,
                book_title=_get(book, "title"),
                subject=_get(book, "subject_normalized") or _get(book, "subject"),
                grade=_int_or_none(_get(book, "grade")),
                chapter_id=_get(chapter, "section_id"),
                lesson_id=_get(lesson, "section_id") or (sp.section_id if not chain else None),
                lesson_title=_get(lesson, "title"),
                page_number=sp.printed_page_label if (sp.printed_page_verified and sp.printed_page_label) else None,
                pdf_page_number=(sp.pdf_page_index + 1) if sp.pdf_page_index is not None else None,
                span_id=sid,
                quote=(sp.text[:quote_chars] if with_quote else None),
            ))
        return cites, invalid

    def resolve(self, span_id: str) -> Optional[Citation]:
        c, _ = self.resolve_many([span_id])
        return c[0] if c else None
