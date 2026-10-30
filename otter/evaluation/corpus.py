"""Read-only view over WS1's source spans for evaluation (gold evidence, OTTER-Hard, citation checks).

Sources, in order of preference:
1. `data/textbooks/parsed/<book_id>/spans.jsonl` (SourceSpan rows, WS1 contract);
2. DB table `spans` in `data/textbooks/ontology.sqlite3` (column set tolerated).
Book metadata (grade, subject, title) comes from the DB `books` table or `registry/books.jsonl`.
Retrieval: in-memory BM25 over syllable uni+bigrams (FTS5's default tokenizer strips Vietnamese
diacritics, so we do not rely on it) and bge-m3 dense vectors, fused with reciprocal-rank fusion.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from otter.evaluation.text import bigram_tokens

ROOT = Path(__file__).resolve().parents[2]

EXERCISE_TYPES = {"exercise", "question", "activity"}
_EXERCISE_HEAD = re.compile(
    r"(luyện tập|vận dụng|câu hỏi|bài tập|hoạt động|thảo luận|khởi động|mở đầu|trả lời câu hỏi|thực hành|"
    r"em hãy|hãy cho biết|câu \d+)", re.I)
_IMPERATIVE = re.compile(r"\b(em|bạn|các em|các bạn) (hãy|có thể|cần)\b|\bhãy (trình bày|cho biết|nêu|kể|giải thích|"
                         r"phân tích|so sánh|xác định|chia sẻ|tìm|liệt kê|vẽ|tính|nhận xét|đánh giá|viết|lập|sưu tầm|"
                         r"thảo luận|mô tả|chứng minh|dự đoán|đề xuất)\b|\bdựa vào (hình|bảng|sơ đồ|bản đồ|thông tin|"
                         r"tư liệu|đoạn)|^\s*câu hỏi\b", re.I | re.M)
_OBJECTIVE_LINE = re.compile(r"(^|\n)\s*[-–•]?\s*(nêu|trình bày|phân tích|giải thích|nhận biết|vận dụng|so sánh|"
                             r"giới thiệu|hiểu|biết|xác định|mô tả|liệt kê|viết|đọc)[^\n.]{0,80}\bđược\b", re.I)
_TITLE_HEAD = re.compile(r"^\s*(#+\s*)?(BÀI|CHƯƠNG|PHẦN|CHỦ ĐỀ|CHUYÊN ĐỀ)\b", re.I)
_EXERCISE_START = re.compile(r"^\s*(\d+[.)]\s*)?(em hãy|hãy|tại sao|vì sao|cho biết|trình bày|nêu|giải thích|"
                             r"so sánh|tính|xác định|quan sát|đọc|thảo luận|câu hỏi|bài tập|luyện tập|vận dụng)\b", re.I)


@dataclass
class Span:
    span_id: str
    book_id: str
    text: str
    pdf_page_index: Optional[int] = None
    section_id: Optional[str] = None
    heading_path: list[str] = field(default_factory=list)
    printed_page_label: Optional[str] = None
    printed_page_verified: bool = False
    pedagogical_type: str = "unknown"
    grade: Optional[int] = None
    domain: Optional[str] = None
    book_title: Optional[str] = None

    @property
    def lesson_key(self) -> str:
        """Lesson-level grouping key used for seed sampling and lesson-disjoint splits."""
        if self.section_id:
            parts = self.section_id.split(":")
            for i, p in enumerate(parts):
                if re.fullmatch(r"l\d+", p):
                    return ":".join(parts[: i + 1])
            return self.section_id
        if self.heading_path:
            return f"{self.book_id}:" + " / ".join(self.heading_path[:2])
        return f"{self.book_id}:pg{(self.pdf_page_index or 0) // 4}"

    @property
    def section_key(self) -> str:
        return self.section_id or self.lesson_key

    def is_exercise(self) -> bool:
        if self.pedagogical_type in EXERCISE_TYPES:
            return True
        for h in self.heading_path[-2:]:          # lesson/chapter titles ("BÀI 1 PHẦN MỞ ĐẦU") are not exercise headings
            if not _TITLE_HEAD.match(h) and _EXERCISE_HEAD.search(h):
                return True
        t = self.text.strip()
        if _EXERCISE_START.match(t) and (t.endswith("?") or len(t) < 300):
            return True
        if _IMPERATIVE.search(t) or len(_OBJECTIVE_LINE.findall(t)) >= 2:
            return True
        return t.count("?") >= 2 and len(t) < 600

    def meta(self) -> dict[str, Any]:
        return {"span_id": self.span_id, "book_id": self.book_id, "book_title": self.book_title, "grade": self.grade,
                "subject": self.domain, "section_id": self.section_id, "heading_path": self.heading_path,
                "pdf_page_index": self.pdf_page_index, "printed_page_label": self.printed_page_label,
                "printed_page_verified": self.printed_page_verified, "pedagogical_type": self.pedagogical_type}


def _load_books(db: Optional[Path], registry: Path) -> dict[str, dict]:
    books: dict[str, dict] = {}
    if db and db.exists():
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                for bid, mj in c.execute("SELECT book_id, metadata_json FROM books"):
                    books[bid] = json.loads(mj) if mj else {}
        except sqlite3.Error:
            pass
    reg = registry / "books.jsonl"
    if reg.exists():
        for line in reg.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                books.setdefault(r["book_id"], r)
    return books


class SpanCorpus:
    def __init__(self, spans: Iterable[Span], books: Optional[dict[str, dict]] = None):
        self.books = books or {}
        self.spans: list[Span] = []
        for s in spans:
            b = self.books.get(s.book_id, {})
            if s.grade is None:
                s.grade = b.get("grade")
            if s.domain is None:
                s.domain = b.get("subject_normalized") or b.get("subject")
            if s.book_title is None:
                s.book_title = b.get("title")
            self.spans.append(s)
        self.by_id = {s.span_id: s for s in self.spans}
        self._bm25 = None
        self._dense: Optional[np.ndarray] = None
        self._embedder = None

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, *, parsed_dir: Path = ROOT / "data/textbooks/parsed", db: Optional[Path] = ROOT / "data/textbooks/ontology.sqlite3",
             registry: Path = ROOT / "data/textbooks/registry", min_chars: int = 1,
             book_filter: Optional[callable] = None) -> "SpanCorpus":
        books = _load_books(db, registry)
        spans: list[Span] = []
        files = sorted(Path(parsed_dir).glob("*/spans.jsonl")) if parsed_dir and Path(parsed_dir).exists() else []
        files = [f for f in files if f.stat().st_size > 0]
        if files:
            for f in files:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    spans.append(Span(span_id=r["span_id"], book_id=r["book_id"], text=r.get("text", ""),
                                      pdf_page_index=r.get("pdf_page_index"), section_id=r.get("section_id"),
                                      heading_path=r.get("heading_path") or [],
                                      printed_page_label=r.get("printed_page_label"),
                                      printed_page_verified=bool(r.get("printed_page_verified")),
                                      pedagogical_type=r.get("pedagogical_type") or "unknown"))
        elif db and Path(db).exists():
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                cols = [r[1] for r in c.execute("PRAGMA table_info(spans)")]
                rows = c.execute(f"SELECT {','.join(cols)} FROM spans").fetchall()
            for row in rows:
                r = dict(zip(cols, row))
                hp = r.get("heading_path") or r.get("heading_path_json")
                if isinstance(hp, str):
                    try:
                        hp = json.loads(hp)
                    except json.JSONDecodeError:
                        hp = [hp]
                spans.append(Span(span_id=r["span_id"], book_id=r["book_id"], text=r.get("text") or "",
                                  pdf_page_index=r.get("pdf_page_index"), section_id=r.get("section_id"),
                                  heading_path=hp or [], printed_page_label=r.get("printed_page_label"),
                                  printed_page_verified=bool(r.get("printed_page_verified")),
                                  pedagogical_type=r.get("pedagogical_type") or "unknown"))
        spans = [s for s in spans if len(s.text.strip()) >= min_chars]
        if book_filter:
            spans = [s for s in spans if book_filter(books.get(s.book_id, {"book_id": s.book_id}))]
        return cls(spans, books)

    def __len__(self) -> int:
        return len(self.spans)

    def get(self, span_id: str) -> Optional[Span]:
        return self.by_id.get(span_id)

    # ------------------------------------------------------------ retrieval
    def _mask(self, domain: Optional[str], grades: Optional[Iterable[int]], book_ids: Optional[Iterable[str]],
              exclude_exercises: bool) -> np.ndarray:
        key = (domain, tuple(sorted(grades)) if grades else None, tuple(sorted(book_ids)) if book_ids else None,
               exclude_exercises)
        cache = self.__dict__.setdefault("_mask_cache", {})
        if key in cache:
            return cache[key]
        if exclude_exercises and "_is_ex" not in self.__dict__:
            self._is_ex = np.array([s.is_exercise() for s in self.spans], dtype=bool)
        g = set(grades) if grades else None
        b = set(book_ids) if book_ids else None
        m = np.array([(domain is None or s.domain == domain) and (g is None or s.grade in g)
                      and (b is None or s.book_id in b) for s in self.spans], dtype=bool)
        if exclude_exercises:
            m &= ~self._is_ex
        cache[key] = m
        return m

    def bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([bigram_tokens(s.text) or ["∅"] for s in self.spans])
        return np.asarray(self._bm25.get_scores(bigram_tokens(query)), dtype=float)

    def attach_embedder(self, embedder) -> None:
        self._embedder = embedder
        self._dense = None

    def dense_matrix(self) -> Optional[np.ndarray]:
        if self._embedder is None:
            return None
        if self._dense is None:
            self._dense = self._embedder.embed([s.text for s in self.spans])
        return self._dense

    def dense_scores(self, query: str) -> Optional[np.ndarray]:
        m = self.dense_matrix()
        if m is None:
            return None
        q = self._embedder.embed([query])[0]
        return m @ q

    def search(self, query: str, *, top_k: int = 10, domain: Optional[str] = None,
               grades: Optional[Iterable[int]] = None, book_ids: Optional[Iterable[str]] = None,
               exclude_exercises: bool = False, mode: str = "hybrid", rrf_k: int = 60) -> list[tuple[Span, float, dict]]:
        """Returns [(span, fused_score, {"bm25_rank","dense_rank"})]."""
        if not self.spans:
            return []
        mask = self._mask(domain, grades, book_ids, exclude_exercises)
        if not mask.any():
            return []
        ranks: dict[str, np.ndarray] = {}
        if mode in ("hybrid", "bm25"):
            s = np.where(mask, self.bm25_scores(query), -np.inf)
            ranks["bm25_rank"] = _ranks(s)
        if mode in ("hybrid", "dense"):
            d = self.dense_scores(query)
            if d is not None:
                ranks["dense_rank"] = _ranks(np.where(mask, d, -np.inf))
        fused = np.zeros(len(self.spans))
        for r in ranks.values():
            fused += 1.0 / (rrf_k + r)
        fused[~mask] = -np.inf
        order = np.argsort(-fused)[:top_k]
        return [(self.spans[i], float(fused[i]), {k: int(v[i]) for k, v in ranks.items()})
                for i in order if np.isfinite(fused[i])]


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each element (ties broken by index); -inf scores get a huge rank."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(scores) + 1)
    r = r.astype(float)
    r[~np.isfinite(scores)] = 1e9
    return r


CORE_PREFIX = "nxbgd_"


def core_book_filter(b: dict) -> bool:
    """Evaluation corpus "v1-core": the 18 core grade 10-12 textbooks (book_id prefix nxbgd_)."""
    return str(b.get("book_id", "")).startswith(CORE_PREFIX)


def corpus_manifest(corpus: Optional["SpanCorpus"] = None, manifest_dir: Path = ROOT / "runs/corpus") -> dict[str, Any]:
    """Frozen-corpus metadata (runs/corpus/CORPUS_V*.json, written by WS1/coordinator) + a check that the loaded
    spans match the manifest's per-book counts (for the books present in `corpus`)."""
    files = sorted(Path(manifest_dir).glob("CORPUS_V*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    if not files:
        return {"corpus_version": None, "spans_sha256": None, "manifest": None}
    m = json.loads(files[-1].read_text(encoding="utf-8"))
    out = {"corpus_version": m.get("corpus_version"), "spans_sha256": m.get("spans_sha256"),
           "manifest": str(files[-1].relative_to(ROOT)) if files[-1].is_relative_to(ROOT) else str(files[-1]),
           "frozen_at": m.get("frozen_at"), "ocr": m.get("ocr")}
    if corpus is not None and m.get("per_book"):
        from collections import Counter
        got = Counter(s.book_id for s in corpus.spans)
        # corpus may be filtered (min_chars, core books): only books whose counts can be compared exactly
        mism = {b: (n, m["per_book"].get(b)) for b, n in got.items() if m["per_book"].get(b) is None or n > m["per_book"][b]}
        out["books_checked"] = len(got)
        out["matches_manifest"] = not mism
        if got and all(b.startswith(CORE_PREFIX) for b in got):
            out["corpus_version"] = f"{m.get('corpus_version')}-core"
            out["base_corpus_version"] = m.get("corpus_version")
            out["core_spans_in_manifest"] = sum(v for b, v in m["per_book"].items() if b.startswith(CORE_PREFIX))
            out["note"] = "core textbooks only (book_id prefix nxbgd_); spans_sha256 refers to the full base corpus"
        if mism:
            out["mismatch"] = mism
    return out
