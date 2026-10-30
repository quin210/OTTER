"""Knowledge maintenance, part 2: ingest NEW books (other publishers / editions / grades) into an existing ontology.

ADD-ON module (like `revalidate`): it never changes answering code or configs and never writes the old ontology file.
Output is a NEW immutable version file (+ changelog, merge log, review list, summary).

Pipeline
  1. load the new books (parsed dir | spans.jsonl | sqlite), refuse books that are already in the ontology
  2. segment -> extract (existing Extractor + chunk cache) -> normalize (existing; within the new books only)
     -> structural checks -> SupportChecker (existing verify-v2, role `small`)           [only the NEW chunks/items]
  3. concept matching against the old ontology (same subject only): exact key, alias, bge-m3 label similarity
     (sim_threshold 0.93, `_compatible` guard).  Every decision is recorded: merged | kept_separate | ambiguous.
  4. agreement / conflict detection for facts, rules and concept definitions (otter/builder/claims.py):
     agrees | new | conflicts | ambiguous, lexical/value check first, small LLM only for uncertain pairs.
  5. write the new version:  copy of the old file + new items (own span links) + provenance ledger.
        agrees     -> the OLD item stays; the new book's spans are ADDED to its evidence (logged, `item_sources`)
        new        -> added as a new item
        ambiguous  -> added, visible, never strict-executable, listed for review
        conflicts  -> added but HIDDEN (item_status.valid = 0, listed for review);  `include_conflicts=True` makes them visible
                      as `alternative_source` (never strict-executable)
  Nothing in the old file is overwritten: old item rows change only by additive evidence/alias lists, each logged.

Not done (docs/KNOWLEDGE_MAINTENANCE.md): relations are only agreed/new (no contradiction check); conflicts are found
between a new item and *candidate* old items (shared concept or embedding neighbour), not by exhaustive comparison; a rephrased
claim with the same value is judged by the 8B model; `revalidate` run on an ingested file resets the hidden flags.
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.builder import claims as C
from otter.builder import extract as X
from otter.builder import validate as V
from otter.builder.ingest import BookInfo
from otter.builder.normalize import GENERIC_TYPES, Ontology, _compatible, normalize
from otter.builder.persist import EmbeddingCache, relation_text
from otter.builder.revalidate import KINDS, SPAN_COLS, SPAN_DDL, TABLES, load_items
from otter.builder.segment import segment_book
from otter.llm.interface import LLM, LLMResponse
from otter.schemas.ontology import (Atom, Concept, ConditionalRule, Fact, LearningObjective, PedagogicalLink,
                                    SemanticRelation, Validation)
from otter.schemas.textbook import SourceSpan
from otter.storage.embeddings import from_blob, to_blob
from otter.storage.schema import INGEST_DDL, MAINTENANCE_DDL
from otter.storage.textutil import norm_key, short_hash, slugify
from otter.storage.writer import OntologyWriter, concept_text, rule_text

log = logging.getLogger("otter.builder.ingest_books")
ROOT = Path(__file__).resolve().parents[2]
PROTECTED = (ROOT / "data/ontology/ontology.sqlite3", ROOT / "data/textbooks/ontology.sqlite3")
SUBJECTS = {"physics", "chemistry", "biology", "history", "geography", "literature", "other"}


# ------------------------------------------------------------------------------------------ config
@dataclass
class IngestConfig:
    old_db: Path
    new_spans: Path
    out: Path
    old_prov: Optional[Path] = None              # provenance DB of the OLD books (default: paths.db)
    books: Optional[list[str]] = None            # new book ids / fnmatch patterns (default: every book in new_spans)
    book_meta: dict = field(default_factory=dict)  # {book_id | "*": {publisher, series, edition, edition_year, subject, grade, title}}
    extractor_role: str = "main"
    verifier_role: str = "small"
    judge_role: str = "small"
    workers: int = 4                             # extraction workers (<= 6)
    verify_workers: int = 3
    judge_workers: int = 3
    min_tokens: int = 1500
    max_tokens: int = 3000
    max_output_tokens: int = 6144
    limit_chunks: Optional[int] = None           # smoke test: only the first N chunks of the new books
    sim_threshold: float = 0.93
    ambiguous_low: float = 0.88                  # label similarity in [low, threshold) -> `ambiguous`
    cand_cos: float = 0.78                       # dense neighbours (same subject) considered for claim comparison
    max_candidates: int = 6
    max_judge_pairs: int = 800                   # cap of LLM pairs per run (beyond: ambiguous, flagged)
    max_pairs_per_item: int = 3
    claim_rules: str = "v2"                      # "v1" = first claim rules (kept to reproduce the first synthetic run)
    cross_kind: bool = True                      # compare a new fact with old rules and vice versa (same subject)
    include_conflicts: bool = False              # conflicts visible as `alternative_source` (default: hidden, review list)
    confirm_conflicts: bool = True               # LLM re-check of lexical (same wording, other number) conflicts
    lexical_only: bool = False                   # no claim LLM: uncertain pairs -> ambiguous
    no_verify: bool = False                      # lexical-only support check (flagged)
    accumulate_provenance: bool = True           # `agrees`: add the new book's spans to the OLD item (logged)
    merge_aliases: bool = True                   # merged concepts: add the new label/aliases to the OLD concept (logged)
    embed: bool = True
    embed_old_provenance: bool = True            # copy provenance of the old books so the file is self-contained
    allow_reingest: bool = False
    force: bool = False                          # overwrite an existing --out (never old_db)
    reason: str = "new books ingested"
    build_id: Optional[str] = None
    report_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None             # extract/ verify/ judge/ emb_cache.sqlite3 (default <out>.cache)
    dry_run: bool = False


class CountingLLM(LLM):
    """Exact per-role call/token meter around any LLM (live calls only)."""

    def __init__(self, inner: LLM, name: str):
        self.inner, self.name = inner, name
        self.model = getattr(inner, "model", "")
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def chat(self, messages, **kw) -> LLMResponse:
        r = self.inner.chat(messages, **kw)
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        return r

    def usage(self) -> dict:
        return {"role": self.name, "model": self.model, "llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# ------------------------------------------------------------------------------------------ loading new books
@dataclass
class NewBooks:
    infos: dict[str, BookInfo] = field(default_factory=dict)
    spans: dict[str, list[SourceSpan]] = field(default_factory=dict)
    sections: dict[str, dict] = field(default_factory=dict)
    raw_spans: dict[str, dict] = field(default_factory=dict)
    raw_books: dict[str, dict] = field(default_factory=dict)
    origin: str = ""


_UNIT_RE = re.compile(r":u\d+$")


def _order(spans: list[SourceSpan]) -> list[SourceSpan]:
    spans = [s for s in spans if s.text.strip()]
    if any(_UNIT_RE.search(s.span_id) for s in spans):
        spans = [s for s in spans if _UNIT_RE.search(s.span_id)]
    spans.sort(key=lambda s: (s.pdf_page_index, s.start_offset, s.span_id))
    return spans


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def provenance_book_meta(prov: Optional[Path]) -> dict[str, dict]:
    """Metadata of the books known to a provenance DB / registry (used when the new-spans source carries none, e.g. the
    parsed dir of a core book: the extractor prompt includes title/subject/grade, so they must be the same to hit the chunk cache)."""
    out: dict[str, dict] = {}
    reg = ROOT / "data/textbooks/registry/books.jsonl"
    if reg.exists():
        for l in reg.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                out[d["book_id"]] = d
    if prov and Path(prov).exists():
        con = _connect_ro(Path(prov))
        if con.execute("SELECT 1 FROM sqlite_master WHERE name='books'").fetchone():
            for bid, mj in con.execute("SELECT book_id, metadata_json FROM books"):
                try:
                    out[bid] = {**out.get(bid, {}), **json.loads(mj or "{}")}
                except json.JSONDecodeError:
                    pass
        con.close()
    return out


def load_new_books(src: Path, patterns: Optional[list[str]] = None, meta: Optional[dict] = None,
                   fallback: Optional[dict] = None) -> NewBooks:
    """src: dir of <book>/spans.jsonl (parsed layout) | dir holding spans.jsonl | spans.jsonl file | sqlite with spans table."""
    src = Path(src)
    nb = NewBooks(origin=str(src))
    rows_by_book: dict[str, list[dict]] = defaultdict(list)
    if src.is_dir():
        files = [src / "spans.jsonl"] if (src / "spans.jsonl").exists() else sorted(src.glob("*/spans.jsonl"))
        for f in files:
            rows = _jsonl(f)
            for r in rows:
                rows_by_book[r["book_id"]].append(r)
            bid = rows[0]["book_id"] if rows else None
            if bid:
                if (f.parent / "book.json").exists():
                    nb.raw_books[bid] = json.loads((f.parent / "book.json").read_text(encoding="utf-8"))
                if (f.parent / "sections.jsonl").exists():
                    for d in _jsonl(f.parent / "sections.jsonl"):
                        nb.sections[d["section_id"]] = d
    elif src.suffix in (".jsonl", ".json"):
        for r in _jsonl(src):
            rows_by_book[r["book_id"]].append(r)
    else:
        con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
        if "spans" not in names:
            raise ValueError(f"{src} has no spans table")
        from otter.storage.sqlite_store import SqliteOntologyStore
        for r in con.execute("SELECT * FROM spans"):
            d = dict(r)
            sp = SqliteOntologyStore._span_from_row(r)
            d["heading_path"] = sp.heading_path
            rows_by_book[d["book_id"]].append(d)
        if "books" in names:
            for r in con.execute("SELECT * FROM books"):
                d = dict(r)
                m = json.loads(d.pop("metadata_json", None) or "{}") if d.get("metadata_json") else {}
                m.update({k: v for k, v in d.items() if v is not None and k != "metadata_json"})
                nb.raw_books[d["book_id"]] = m
        if "sections" in names:
            for r in con.execute("SELECT * FROM sections"):
                d = dict(r)
                try:
                    m = json.loads(d.pop("metadata_json", None) or "{}")
                except json.JSONDecodeError:
                    m = {}
                m.update({k: v for k, v in d.items() if v is not None})
                nb.sections[d["section_id"]] = m
        con.close()
    ids = sorted(rows_by_book)
    if patterns:
        ids = [b for b in ids if any(fnmatch.fnmatch(b, p) for p in patterns)]
    for bid in ids:
        spans = []
        for r in rows_by_book[bid]:
            rr = dict(r)
            if "heading_path" not in rr and rr.get("heading_path_json"):
                try:
                    rr["heading_path"] = json.loads(rr["heading_path_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            try:
                s = SourceSpan.model_validate({k: v for k, v in rr.items() if k in SourceSpan.model_fields})
            except Exception:
                continue
            spans.append(s)
            nb.raw_spans[s.span_id] = rr
        nb.spans[bid] = _order(spans)
        m = {**(fallback or {}).get(bid, {}), **nb.raw_books.get(bid, {})}
        nb.raw_books[bid] = m
        m = dict(m)
        m.update((meta or {}).get("*", {}))
        m.update((meta or {}).get(bid, {}))
        subj = m.get("subject") or m.get("subject_normalized") or "other"
        grade = m.get("grade")
        nb.infos[bid] = BookInfo(bid, subj if subj in SUBJECTS else "other", int(grade) if grade is not None else None,
                                 m.get("title") or bid, m)
    nb.spans = {b: s for b, s in nb.spans.items() if s}
    nb.infos = {b: i for b, i in nb.infos.items() if b in nb.spans}
    return nb


def book_meta_row(info: BookInfo) -> dict:
    m = info.meta
    return {"book_id": info.book_id, "title": info.title, "subject": info.subject, "grade": info.grade,
            "publisher": m.get("publisher"), "series": m.get("series"),
            "edition": m.get("edition") or m.get("edition_label"),
            "edition_year": m.get("edition_year") or m.get("year")}


# ------------------------------------------------------------------------------------------ helpers
def _connect_ro(p: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _book_of(span_id: str) -> str:
    return span_id.split(":")[0]


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class _OldIndex:
    """Read-only view of the old ontology used for matching (concepts, facts, rules + their vectors)."""

    def __init__(self, old_db: Path, items: dict, subjects: set[str], model: Optional[str]):
        con = _connect_ro(old_db)
        self.concepts: dict[str, Concept] = {i: o for (k, i), o in items.items() if k == "concept" and o.subject in subjects}
        self.facts: dict[str, Fact] = {i: o for (k, i), o in items.items() if k == "fact" and o.subject in subjects}
        self.rules: dict[str, ConditionalRule] = {i: o for (k, i), o in items.items() if k == "rule" and o.subject in subjects}
        self.relation_ids = {i for (k, i) in items if k == "relation"}
        self.all_ids = {k: {i for (kk, i) in items if kk == k} for k in KINDS}
        self.label_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        self.alias_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for cid, c in self.concepts.items():
            self.label_key[(c.subject, norm_key(c.label))].add(cid)
            for a in c.aliases:
                if norm_key(a):
                    self.alias_key[(c.subject, norm_key(a))].add(cid)
        self.fact_by_concept: dict[str, set[str]] = defaultdict(set)
        for aid, cid in con.execute("SELECT assertion_id, concept_id FROM fact_concepts"):
            if aid in self.facts:
                self.fact_by_concept[cid].add(aid)
        self.rule_by_concept: dict[str, set[str]] = defaultdict(set)
        for rid, cid in con.execute("SELECT rule_id, concept_id FROM rule_concepts"):
            if rid in self.rules:
                self.rule_by_concept[cid].add(rid)
        self.vec: dict[str, tuple[list[str], np.ndarray]] = {}
        if model:
            for kind in ("fact", "rule"):
                rows = con.execute("SELECT item_id, subject, vec FROM embeddings WHERE kind=? AND model=?", (kind, model)).fetchall()
                keep = self.facts if kind == "fact" else self.rules
                ids = [r[0] for r in rows if r[0] in keep]
                if ids:
                    byid = {r[0]: r[2] for r in rows}
                    self.vec[kind] = (ids, _unit(np.stack([from_blob(byid[i]) for i in ids])))
        con.close()
        self._pos = {k: {i: n for n, i in enumerate(v[0])} for k, v in self.vec.items()}
        self.concept_vec: dict[str, tuple[list[str], np.ndarray]] = {}

    def subject_of(self, kind: str, iid: str) -> str:
        return (self.facts if kind == "fact" else self.rules)[iid].subject


def _span_snips(con: Optional[sqlite3.Connection], ids: list[str], n: int = 2, chars: int = 420) -> list[dict]:
    out = []
    if con is None:
        return out
    for sid in ids[:n]:
        r = con.execute("SELECT text FROM spans WHERE span_id=?", (sid,)).fetchone()
        if r:
            out.append({"span_id": sid, "text": r[0][:chars]})
    return out


# ------------------------------------------------------------------------------------------ main
def ingest_books(cfg: IngestConfig, *, extractor_llm: Optional[LLM] = None, verifier_llm: Optional[LLM] = None,
                 judge_llm: Optional[LLM] = None, embedder=None, config: Optional[dict] = None) -> dict[str, Any]:
    from otter.llm.interface import get_llm, load_config
    t_all = time.time()
    timings: dict[str, float] = {}
    old_db, out = Path(cfg.old_db), Path(cfg.out)
    if out.resolve() == old_db.resolve():
        raise ValueError("refusing to overwrite the old ontology file: --out must be a NEW file")
    if any(out.resolve() == p.resolve() for p in PROTECTED if p.exists()):
        raise ValueError(f"refusing to write a protected file: {out}")
    if cfg.old_prov and out.resolve() == Path(cfg.old_prov).resolve():
        raise ValueError("--out must differ from the provenance DB")
    if out.exists() and not cfg.force and not cfg.dry_run:
        raise ValueError(f"{out} exists (use --force to replace a previous ingest output)")
    if cfg.workers > 6 or cfg.verify_workers > 6 or cfg.judge_workers > 6:
        raise ValueError("at most 6 concurrent LLM requests are allowed (shared GPUs)")
    config = config or load_config()
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else out.with_name(out.stem + ".cache")
    old_sha_before = sha256_file(old_db)

    # ---- 1. old ontology + new books
    t = time.time()
    items, concepts_all, support_old, old_build = load_items(old_db)
    old_build_id = old_build.get("build_id", "unknown")
    nb = load_new_books(cfg.new_spans, cfg.books, cfg.book_meta, provenance_book_meta(cfg.old_prov))
    if not nb.spans:
        raise ValueError(f"no books with spans found in {cfg.new_spans} matching {cfg.books!r}")
    old_books = sorted({_book_of(s) for o in items.values() for s in o.source_span_ids})
    clash = sorted(set(nb.spans) & set(old_books))
    if clash and not cfg.allow_reingest:
        raise ValueError(f"books already in the ontology: {clash} (use --allow-reingest to force)")
    old_span_ids = {s for o in items.values() for s in o.source_span_ids}
    new_span_ids = {s.span_id for ss in nb.spans.values() for s in ss}
    if old_span_ids & new_span_ids and not cfg.allow_reingest:
        raise ValueError("span ids of the new books collide with spans cited by the old ontology")
    spans: dict[str, SourceSpan] = {s.span_id: s for ss in nb.spans.values() for s in ss}
    subjects = {i.subject for i in nb.infos.values()}
    timings["load"] = round(time.time() - t, 2)

    # ---- 2. segment
    chunks, chunk_stats = [], {}
    for b, info in nb.infos.items():
        cs = segment_book(b, nb.spans[b], nb.sections, subject=info.subject, grade=info.grade, book_title=info.title,
                          min_tokens=cfg.min_tokens, max_tokens=cfg.max_tokens)
        chunk_stats[b] = {"spans": len(nb.spans[b]), "chunks": len(cs), "chunk_tokens_est": sum(c.tokens for c in cs)}
        chunks += cs
    if cfg.limit_chunks:
        chunks = chunks[: cfg.limit_chunks]
    log.info("%d new books, %d spans, %d chunks", len(nb.infos), len(spans), len(chunks))

    if cfg.dry_run:
        mdl = config["llm"][cfg.extractor_role]["model"]
        cached = sum(1 for c in chunks if (cache_dir / "extract" / X.chunk_hash(c, mdl)[:2] / f"{X.chunk_hash(c, mdl)}.json").exists())
        return {"dry_run": True, "books": list(nb.infos), "spans": len(spans), "chunks": len(chunks), "chunks_cached": cached,
                "chunks_to_extract": len(chunks) - cached, "chunk_tokens_est": sum(c.tokens for c in chunks),
                "chunk_stats": chunk_stats}

    # ---- 3. extract (existing extractor + cache)
    t = time.time()
    ext_llm = CountingLLM(extractor_llm or get_llm(cfg.extractor_role, config), cfg.extractor_role)
    extractor = X.Extractor(ext_llm, cache_dir / "extract", max_tokens=cfg.max_output_tokens)
    results = extractor.run(chunks, workers=cfg.workers)
    timings["extract"] = round(time.time() - t, 2)
    live = [r for r in results if not r.cached]
    ex_stats = {"model": ext_llm.model, "prompt_version": X.PROMPT_VERSION, "chunks": len(results),
                "llm_chunks": len(live), "cached_chunks": len(results) - len(live),
                "failed_chunks": sum(1 for r in results if r.status in ("parse_failed", "llm_error")),
                "live_prompt_tokens": ext_llm.prompt_tokens, "live_completion_tokens": ext_llm.completion_tokens,
                "live_llm_calls": ext_llm.calls,
                "cached_recorded_prompt_tokens": sum(r.prompt_tokens for r in results if r.cached),
                "cached_recorded_completion_tokens": sum(r.completion_tokens for r in results if r.cached),
                "item_schema_errors": sum(len(r.item_errors) for r in results)}
    if results and ex_stats["failed_chunks"] / len(results) > 0.2:
        raise SystemExit(f"{ex_stats['failed_chunks']}/{len(results)} chunks failed; not writing (extraction cache keeps the rest)")

    # ---- 4. normalize inside the new books (existing code), then validate (existing code)
    t = time.time()
    emb = None
    if cfg.embed:
        if embedder is None:
            from otter.storage.embeddings import Bgem3Embedder
            embedder = Bgem3Embedder.from_config(config)
        if embedder.available():
            emb = EmbeddingCache(embedder, cache_dir / "emb_cache.sqlite3")
        else:
            log.warning("embedding server unavailable: string-only concept matching, no dense candidates")
    new_onto = normalize(results, spans, embedder=emb, sim_threshold=cfg.sim_threshold)
    timings["normalize"] = round(time.time() - t, 2)

    t = time.time()
    tmp_prov = _tmp_provenance(cache_dir / "new_books_prov.sqlite3", nb)
    V.structural_checks(new_onto, tmp_prov)
    V.missing_guard_check(new_onto, spans)
    ver_llm = None
    if not cfg.no_verify:
        ver_llm = CountingLLM(verifier_llm or get_llm(cfg.verifier_role, config), cfg.verifier_role)
        if ver_llm.model == ext_llm.model and ver_llm.model not in ("", "mock"):
            raise ValueError("support verifier must be a different model than the extractor")
    checker = V.SupportChecker(ver_llm, cache_dir / "verify")
    checker.run(new_onto, spans, workers=cfg.verify_workers)
    timings["validate"] = round(time.time() - t, 2)
    ver_stats = {"model": getattr(ver_llm, "model", None), "verify_version": V.VERIFY_VERSION,
                 "mode": "lexical+llm" if ver_llm is not None else "lexical_only",
                 "live_llm_calls": ver_llm.calls if ver_llm else 0, "live_prompt_tokens": ver_llm.prompt_tokens if ver_llm else 0,
                 "live_completion_tokens": ver_llm.completion_tokens if ver_llm else 0,
                 "cached_batches": checker.stats["cached_batches"], "failed_batches": checker.stats["failed_batches"]}

    # ---- 5. match against the old ontology
    t = time.time()
    model = getattr(emb, "model", None) or old_build.get("embedding_model") or None
    if emb is not None and old_build.get("embedding_model") and emb.model != old_build["embedding_model"]:
        raise ValueError(f"embedder {emb.model!r} differs from the old ontology's {old_build['embedding_model']!r}")
    idx = _OldIndex(old_db, items, subjects, model if emb is not None else None)
    ingest_id = cfg.build_id or f"ingest-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    merge_log, cmap = match_concepts(new_onto, idx, emb, cfg, nb, ingest_id)
    timings["match_concepts"] = round(time.time() - t, 2)

    # ---- 6. agreement / conflict detection
    t = time.time()
    judge = None
    if not cfg.lexical_only:
        jl = judge_llm or get_llm(cfg.judge_role, config)
        judge = C.PairJudge(CountingLLM(jl, cfg.judge_role), cache_dir / f"judge_{cfg.claim_rules}", workers=cfg.judge_workers, rules=cfg.claim_rules)
    prov_old = _connect_ro(Path(cfg.old_prov)) if cfg.old_prov and Path(cfg.old_prov).exists() else None
    decisions = classify_items(new_onto, idx, cmap, merge_log, emb, cfg, judge, nb)
    timings["classify"] = round(time.time() - t, 2)
    judge_stats = ({**judge.stats, **judge.llm.usage()} if judge else {"llm_calls": 0})

    # ---- 7. write
    build_id = cfg.build_id or f"{old_build_id}+ing-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    summary: dict[str, Any] = {
        "build_id": build_id, "ingest_id": ingest_id, "parent_build_id": old_build_id, "reason": cfg.reason,
        "books": {b: {**book_meta_row(i), **chunk_stats[b]} for b, i in nb.infos.items()},
        "extraction": ex_stats, "verification": ver_stats, "claim_judge": judge_stats,
        "timings_s": timings, "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()
                                          if k not in ("book_meta",)}}
    t = time.time()
    info = _persist(cfg, build_id, ingest_id, old_build_id, new_onto, decisions, merge_log, cmap, nb, spans, idx, items,
                    emb, results, prov_old, summary, support_old)
    summary.update(info)
    timings["persist"] = round(time.time() - t, 2)
    timings["total"] = round(time.time() - t_all, 2)
    # totals
    cnt: dict[str, Counter] = {k: Counter() for k in KINDS}
    for d in decisions:
        cnt[d.kind][d.verdict] += 1
    summary["verdicts"] = {k: dict(v) for k, v in cnt.items()}
    summary["handling"] = dict(Counter(d.handling for d in decisions))
    summary["concept_merge"] = dict(Counter(f"{m['decision']}/{m['method']}" for m in merge_log.values()))
    summary["concept_merge_decisions"] = dict(Counter(m["decision"] for m in merge_log.values()))
    sup: dict[str, Counter] = {k: Counter() for k in ("concept", "relation", "fact", "rule")}
    for kind, iid, obj in new_onto.items():
        if kind in sup:
            sup[kind][obj.validation.source_support] += 1
    summary["support_new_items"] = {k: dict(v) for k, v in sup.items()}
    summary["rejected_new"] = dict(Counter(f"{r.get('kind')}:{r.get('reason')}" for r in new_onto.rejected))
    live_calls = ex_stats["live_llm_calls"] + ver_stats["live_llm_calls"] + judge_stats.get("llm_calls", 0)
    live_pt = ex_stats["live_prompt_tokens"] + ver_stats["live_prompt_tokens"] + judge_stats.get("prompt_tokens", 0)
    live_ct = ex_stats["live_completion_tokens"] + ver_stats["live_completion_tokens"] + judge_stats.get("completion_tokens", 0)
    summary["cost"] = {"live_llm_calls": live_calls, "live_prompt_tokens": live_pt, "live_completion_tokens": live_ct,
                       "live_total_tokens": live_pt + live_ct}
    summary["old_db_sha256"] = old_sha_before
    summary["old_db_unchanged"] = sha256_file(old_db) == old_sha_before
    summary["out_sha256"] = sha256_file(out)
    rdir = Path(cfg.report_dir) if cfg.report_dir else out.parent
    (rdir / f"{out.stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if prov_old:
        prov_old.close()
    return summary


def _tmp_provenance(path: Path, nb: NewBooks) -> Path:
    """A throw-away sqlite file with the new books' spans so the existing structural check (spans must exist) works."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE spans(span_id TEXT PRIMARY KEY, book_id TEXT)")
    con.executemany("INSERT OR IGNORE INTO spans VALUES(?,?)", [(s.span_id, s.book_id) for ss in nb.spans.values() for s in ss])
    con.commit()
    con.close()
    return path


# ------------------------------------------------------------------------------------------ concept matching
def match_concepts(new_onto: Ontology, idx: _OldIndex, emb, cfg: IngestConfig, nb: NewBooks, ingest_id: str
                   ) -> tuple[dict[str, dict], dict[str, str]]:
    """Returns (merge_log {new_cid: record}, cmap {new_cid: final concept id}). Same subject only; every decision is kept."""
    log_: dict[str, dict] = {}
    cmap: dict[str, str] = {}
    by_subj: dict[str, list[str]] = defaultdict(list)
    for cid, c in new_onto.concepts.items():
        by_subj[c.subject].append(cid)
    for subj, cids in by_subj.items():
        old_ids = [i for i, c in idx.concepts.items() if c.subject == subj]
        old_vecs = None
        new_vecs = None
        if emb is not None and old_ids:
            try:
                old_vecs = _unit(emb.embed([idx.concepts[i].label for i in old_ids]))
                new_vecs = _unit(emb.embed([new_onto.concepts[c].label for c in cids]))
            except Exception as e:                                   # embedding server down -> string-only
                log.warning("concept embedding skipped for %s: %s", subj, e)
                old_vecs = new_vecs = None
        for n, cid in enumerate(cids):
            c = new_onto.concepts[cid]
            k = norm_key(c.label)
            rec = {"ingest_id": ingest_id, "new_concept_id": cid, "new_label": c.label, "subject": subj,
                   "decision": "kept_separate", "method": "none", "old_concept_id": None, "sim": None, "evidence": {}}
            exact = idx.label_key.get((subj, k), set())
            via_alias = set()
            for a in c.aliases:
                ak = norm_key(a)
                if len(ak) >= 2:
                    via_alias |= idx.label_key.get((subj, ak), set())
            via_alias |= idx.alias_key.get((subj, k), set()) if len(k) >= 2 else set()
            if len(exact) == 1:
                rec.update(decision="merged", method="exact", old_concept_id=next(iter(exact)), sim=1.0)
            elif len(exact) > 1:
                rec.update(decision="ambiguous", method="exact_multiple", evidence={"old_candidates": sorted(exact)})
            elif len(via_alias) == 1:
                rec.update(decision="merged", method="alias", old_concept_id=next(iter(via_alias)))
            elif len(via_alias) > 1:
                rec.update(decision="ambiguous", method="alias_multiple", evidence={"old_candidates": sorted(via_alias)})
            if rec["decision"] == "merged" and rec["method"] in ("exact", "alias"):
                oc = idx.concepts[rec["old_concept_id"]]
                short = len(k) <= 3 or len(norm_key(oc.label)) <= 3          # symbols ("N", "Cl", "P") are homonyms across topics
                if short and not (c.concept_type == oc.concept_type or c.concept_type in GENERIC_TYPES or oc.concept_type in GENERIC_TYPES):
                    rec.update(decision="ambiguous", method=rec["method"] + "_short_label_type_mismatch",
                               evidence={"new_type": c.concept_type, "old_type": oc.concept_type})
            if rec["decision"] == "kept_separate" and old_vecs is not None:
                sims = old_vecs @ new_vecs[n]
                order = np.argsort(-sims)[:3]
                top = [(old_ids[int(j)], float(sims[int(j)])) for j in order]
                rec["evidence"]["nearest"] = [{"old_concept_id": i, "label": idx.concepts[i].label, "sim": round(s, 4),
                                               "compatible": _compatible(c.label, idx.concepts[i].label, c.concept_type,
                                                                         idx.concepts[i].concept_type)}
                                              for i, s in top]
                # units differ only by one letter ("km/s" vs "km/h"): the digit guard of `_compatible` cannot see that
                unitish = lambda lab, typ: "/" in lab or typ == "Unit"
                hi = [(i, s) for i, s in top if s >= cfg.sim_threshold and rec["evidence"]["nearest"][[x[0] for x in top].index(i)]["compatible"]
                      and not (unitish(c.label, c.concept_type) and unitish(idx.concepts[i].label, idx.concepts[i].concept_type))]
                if len(hi) >= 2 and hi[0][1] - hi[1][1] < 0.02:
                    rec.update(decision="ambiguous", method="embedding_tie", old_concept_id=hi[0][0], sim=round(hi[0][1], 4))
                elif hi:
                    rec.update(decision="merged", method="embedding", old_concept_id=hi[0][0], sim=round(hi[0][1], 4))
                else:
                    near = [(i, s, x["compatible"]) for (i, s), x in zip(top, rec["evidence"]["nearest"])]
                    if near and near[0][1] >= cfg.sim_threshold:                      # blocked by the compatibility guard
                        rec.update(decision="kept_separate", method="blocked_incompatible", old_concept_id=near[0][0],
                                   sim=round(near[0][1], 4))
                    elif near and near[0][1] >= cfg.ambiguous_low and near[0][2]:
                        rec.update(decision="ambiguous", method="embedding_below_threshold", old_concept_id=near[0][0],
                                   sim=round(near[0][1], 4))
                    elif near:
                        rec.update(sim=round(near[0][1], 4), old_concept_id=None)
            elif rec["decision"] == "kept_separate" and old_vecs is None:
                rec["method"] = "no_embedding_no_match"
            if rec["decision"] == "merged":
                cmap[cid] = rec["old_concept_id"]
                rec["final_concept_id"] = rec["old_concept_id"]
            else:
                final = cid
                if final in idx.all_ids["concept"]:                                  # same slug, different concept
                    final = f"{cid}_{short_hash(norm_key(c.label), *sorted(c.source_span_ids)[:1], n=6)}"
                cmap[cid] = final
                rec["final_concept_id"] = final
            log_[cid] = rec
    return log_, cmap


# ------------------------------------------------------------------------------------------ classification
@dataclass
class Decision:
    kind: str
    new_id: str
    final_id: str
    verdict: str                      # new | agrees | conflicts | ambiguous
    handling: str                     # add | accumulate | hide | alternative | review_only
    matched_old_id: Optional[str] = None
    obj: Any = None                   # remapped pydantic object (None for pure accumulations)
    book_ids: list[str] = field(default_factory=list)
    span_ids: list[str] = field(default_factory=list)
    pair: Optional[C.PairVerdict] = None
    pairs: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    new_text: str = ""
    old_text: str = ""
    extra: dict = field(default_factory=dict)


def _remap_atoms(atoms: list[Atom], cmap: dict[str, str]) -> list[Atom]:
    return [a.model_copy(update={"args": [cmap.get(x, x) for x in a.args]}) for a in atoms]


def _fixed_rule(r: ConditionalRule, cmap: dict[str, str]) -> ConditionalRule:
    vars_ = {}
    for k, v in r.variables.items():
        for o, n in cmap.items():
            if o != n and v.startswith(o):
                v = n + v[len(o):]
                break
        vars_[k] = v
    return r.model_copy(update={"premises": _remap_atoms(r.premises, cmap), "guards": _remap_atoms(r.guards, cmap),
                                "conclusions": _remap_atoms(r.conclusions, cmap),
                                "exceptions": _remap_atoms(r.exceptions, cmap), "variables": vars_}, deep=True)


def old_kind(oid: str, default: str = "fact") -> str:
    """Item ids carry their kind: <subject>:a:<h> (fact), :rule: (rule), :c: (concept), :rel:, :obj:."""
    return "rule" if ":rule:" in oid else "fact" if ":a:" in oid else "concept" if ":c:" in oid else default


def _candidates(kinds: tuple, concept_ids: list[str], text_vec: Optional[np.ndarray], idx: _OldIndex, cfg: IngestConfig,
                subject: str) -> list[tuple[str, Optional[float]]]:
    """Old facts/rules (same subject) that share a merged concept with the new item or are dense neighbours of its text."""
    cand: dict[str, Optional[float]] = {}
    for kind in kinds:
        by_concept = idx.fact_by_concept if kind == "fact" else idx.rule_by_concept
        pool = idx.facts if kind == "fact" else idx.rules
        ids, mat = idx.vec.get(kind, ([], None))
        mine: dict[str, Optional[float]] = {}
        for c in concept_ids:
            for oid in by_concept.get(c, ()):
                if pool[oid].subject == subject:
                    mine[oid] = None
        if text_vec is not None and mat is not None:
            sims = mat @ text_vec
            for j in np.argsort(-sims)[: cfg.max_candidates * 2]:
                if sims[j] >= cfg.cand_cos and pool[ids[int(j)]].subject == subject:
                    mine[ids[int(j)]] = float(sims[j])
            for oid in list(mine):
                if mine[oid] is None and oid in idx._pos.get(kind, {}):
                    mine[oid] = float(sims[idx._pos[kind][oid]])
        cand.update(mine)
    ranked = sorted(cand.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1))
    return ranked[: cfg.max_candidates]


def classify_items(new_onto: Ontology, idx: _OldIndex, cmap: dict[str, str], merge_log: dict[str, dict], emb,
                   cfg: IngestConfig, judge: Optional[C.PairJudge], nb: NewBooks) -> list[Decision]:
    decisions: list[Decision] = []
    all_pairs: list[tuple[Decision, str, C.PairVerdict]] = []

    def books_of(iid: str, obj) -> list[str]:
        b = sorted(new_onto.item_books.get(iid, set()))
        return b or sorted({_book_of(s) for s in obj.source_span_ids})

    # ---------------- concepts
    for cid, c in new_onto.concepts.items():
        rec = merge_log[cid]
        d = Decision("concept", cid, cmap[cid], "new", "add", obj=c, book_ids=books_of(cid, c), span_ids=list(c.source_span_ids),
                     new_text=concept_text(c))
        if rec["decision"] == "merged":
            old = idx.concepts[rec["old_concept_id"]]
            d.matched_old_id, d.verdict, d.handling, d.obj = old.concept_id, "agrees", "accumulate", None
            d.old_text = concept_text(old)
            if c.definition and old.definition and c.validation.source_support != "unsupported":
                pv = C.compare_lexical(old.definition, c.definition, rules=cfg.claim_rules)
                if pv.label != "unrelated":
                    all_pairs.append((d, old.concept_id, pv))
                    d.extra["compare"] = "definition"
                    d.extra["old_def"], d.extra["new_def"] = old.definition, c.definition
                else:
                    pv2 = C.PairVerdict("ambiguous", "lexical_pending", pv.jaccard, note="definitions_share_few_words")
                    all_pairs.append((d, old.concept_id, pv2))
                    d.extra["compare"] = "definition"
                    d.extra["old_def"], d.extra["new_def"] = old.definition, c.definition
        elif rec["decision"] == "ambiguous":
            d.verdict, d.handling = "ambiguous", "add"
            d.notes.append(f"possible_same_as:{rec.get('old_concept_id')}")
            d.matched_old_id = rec.get("old_concept_id")
            if d.matched_old_id in idx.concepts:
                d.old_text = concept_text(idx.concepts[d.matched_old_id])
            d.pair = C.PairVerdict("ambiguous", "embedding" if "embedding" in rec["method"] else "label", cosine=rec.get("sim"),
                                   note=f"concept_merge_{rec['method']}")
        decisions.append(d)

    # ---------------- relations (agree / new only)
    rels: dict[str, SemanticRelation] = {}
    rel_books: dict[str, set] = defaultdict(set)
    for rid, r in new_onto.relations.items():
        s, o = cmap.get(r.subject_id, r.subject_id), cmap.get(r.object_id, r.object_id)
        subj = (new_onto.concepts.get(r.subject_id) or Concept(concept_id="", label="", subject="other")).subject
        nid = f"{subj}:rel:{short_hash(s, r.relation_type, o)}"
        if nid in rels:
            rels[nid].source_span_ids = list(dict.fromkeys(rels[nid].source_span_ids + r.source_span_ids))
        else:
            rels[nid] = r.model_copy(update={"relation_id": nid, "subject_id": s, "object_id": o}, deep=True)
        rel_books[nid] |= set(books_of(rid, r))
    for nid, r in rels.items():
        d = Decision("relation", nid, nid, "new", "add", obj=r, book_ids=sorted(rel_books[nid]), span_ids=list(r.source_span_ids))
        if nid in idx.all_ids["relation"]:
            d.verdict, d.handling, d.matched_old_id, d.obj = "agrees", "accumulate", nid, None
        decisions.append(d)

    # ---------------- facts and rules
    fvec = rvec = None
    if emb is not None:
        facts_list = list(new_onto.facts.items())
        rules_list = list(new_onto.rules.items())
        if facts_list:
            fvec = dict(zip([i for i, _ in facts_list], _unit(emb.embed([f.text for _, f in facts_list]))))
        if rules_list:
            rvec = dict(zip([i for i, _ in rules_list], _unit(emb.embed([rule_text(r) for _, r in rules_list]))))
    for fid, f in new_onto.facts.items():
        f2 = f.model_copy(update={"concept_ids": list(dict.fromkeys(cmap.get(c, c) for c in f.concept_ids))}, deep=True)
        d = Decision("fact", fid, fid, "new", "add", obj=f2, book_ids=books_of(fid, f), span_ids=list(f.source_span_ids),
                     new_text=f.text)
        if fid in idx.all_ids["fact"]:
            d.verdict, d.handling, d.matched_old_id, d.obj = "agrees", "accumulate", fid, None
            d.pair = C.PairVerdict("agrees", "exact", 1.0, note="identical_normalised_text")
            d.old_text = idx.facts[fid].text if fid in idx.facts else ""
        elif f.validation.source_support == "unsupported":
            d.notes.append("unsupported_by_own_source:conflict_check_skipped")
        else:
            kinds = ("fact", "rule") if cfg.cross_kind else ("fact",)
            for oid, cos in _candidates(kinds, f2.concept_ids, fvec.get(fid) if fvec else None, idx, cfg, f.subject):
                pv = C.compare_lexical(_old_text(idx, old_kind(oid), oid), f.text, cosine=cos, rules=cfg.claim_rules,
                                       old_formula=idx.rules[oid].formula if old_kind(oid) == "rule" else None)
                all_pairs.append((d, oid, pv))
        decisions.append(d)
    for rid, r in new_onto.rules.items():
        r2 = _fixed_rule(r, cmap)
        txt = rule_text(r)
        d = Decision("rule", rid, rid, "new", "add", obj=r2, book_ids=books_of(rid, r), span_ids=list(r.source_span_ids),
                     new_text=txt)
        if rid in idx.all_ids["rule"]:
            d.verdict, d.handling, d.matched_old_id, d.obj = "agrees", "accumulate", rid, None
            d.pair = C.PairVerdict("agrees", "exact", 1.0, note="identical_rule_signature")
            d.old_text = rule_text(idx.rules[rid]) if rid in idx.rules else ""
        elif r.validation.source_support == "unsupported":
            d.notes.append("unsupported_by_own_source:conflict_check_skipped")
        else:
            cids = [x for a in [*r2.premises, *r2.guards, *r2.conclusions] for x in a.args if ":c:" in x]
            kinds = ("rule", "fact") if cfg.cross_kind else ("rule",)
            for oid, cos in _candidates(kinds, cids, rvec.get(rid) if rvec else None, idx, cfg, r.subject):
                pv = C.compare_lexical(_old_text(idx, old_kind(oid), oid), txt, cosine=cos, rules=cfg.claim_rules,
                                       old_formula=idx.rules[oid].formula if old_kind(oid) == "rule" else None, new_formula=r.formula)
                all_pairs.append((d, oid, pv))
        decisions.append(d)

    # ---------------- objectives: always new (ids carry the book)
    for oid, o in new_onto.objectives.items():
        decisions.append(Decision("objective", oid, oid, "new", "add", obj=o, book_ids=books_of(oid, o),
                                  span_ids=list(o.source_span_ids), new_text=o.text))

    # ---------------- adjudicate pending pairs (LLM only for the uncertain ones), cap per item and per run
    per_item: dict[int, list] = defaultdict(list)
    for d, oid, pv in all_pairs:
        per_item[id(d)].append((d, oid, pv))
    pending: list[tuple[Decision, str, C.PairVerdict]] = []
    final: dict[int, list[tuple[str, C.PairVerdict]]] = defaultdict(list)
    for key, lst in per_item.items():
        lst.sort(key=lambda x: -(x[2].jaccard + (x[2].cosine or 0) * 0.01))
        npend = 0
        for d, oid, pv in lst:
            if pv.tier != "lexical_pending":
                final[key].append((oid, pv))
                continue
            needs_llm = (pv.label != "conflicts") or cfg.confirm_conflicts
            if cfg.lexical_only or not needs_llm or npend >= cfg.max_pairs_per_item:
                note = "" if (cfg.lexical_only or not needs_llm) else "pair_cap_per_item"
                fp = C.finalize_pending(pv, lexical_only=cfg.lexical_only)
                if note:
                    fp.note += f"; {note}"
                final[key].append((oid, fp))
            else:
                npend += 1
                pending.append((d, oid, pv))
    if judge is not None and pending:
        capped = pending[: cfg.max_judge_pairs]
        over = pending[cfg.max_judge_pairs:]
        texts = [((d.extra.get("old_def") if d.extra.get("compare") == "definition" else _old_text(idx, old_kind(oid, d.kind), oid)),
                  (d.extra.get("new_def") if d.extra.get("compare") == "definition" else d.new_text)) for d, oid, pv in capped]
        js = judge.judge(texts)
        for (d, oid, pv), j in zip(capped, js):
            v = C.apply_judgment(pv, j)
            if d.extra.get("compare") == "definition" and v.label == "conflicts" and cfg.claim_rules == "v2":
                # definitions are paraphrased across publishers: a definition conflict needs differing VALUES on both sides
                vo, vn = C.number_values(d.extra["old_def"]), C.number_values(d.extra["new_def"])
                if not (vo and vn and not C._eq_values(vo, vn)):
                    v = C.PairVerdict("ambiguous", "llm", v.jaccard, v.cosine, v.slot, v.old_value, v.new_value,
                                      note=v.note + "; definition_conflict_without_value_difference", llm_label=v.llm_label)
            final[id(d)].append((oid, v))
        for d, oid, pv in over:
            fp = C.finalize_pending(pv, lexical_only=True)
            fp.note += "; judge_budget_exhausted"
            final[id(d)].append((oid, fp))
    for d in decisions:
        pairs = final.get(id(d), [])
        if not pairs:
            continue
        d.pairs = pairs
        verdict, oid, pv = C.aggregate(pairs)
        if d.kind == "concept":                                   # merged concept: the merge stays; the verdict reports the definition
            if verdict == "conflicts":
                d.verdict = "conflicts"
            elif verdict == "ambiguous":
                d.verdict = "ambiguous"
            d.pair = pv if verdict in ("conflicts", "ambiguous") else d.pair
            d.handling = "accumulate" if verdict not in ("conflicts",) else ("alternative" if cfg.include_conflicts else "review_only")
            if verdict in ("conflicts", "ambiguous"):
                d.extra["definition_conflict"] = True
            continue
        d.pair = pv
        if verdict == "agrees":
            d.verdict, d.matched_old_id = "agrees", oid
            d.old_text = _old_text(idx, old_kind(oid, d.kind), oid)
            if pv.tier == "llm":
                # the model judged the claims equivalent but the wording differs (maybe the new one is more specific):
                # keep the new item as a visible variant AND record the agreement; nothing is dropped
                d.handling = "add_variant"
            else:
                d.handling, d.obj = "accumulate", None
        elif verdict == "conflicts":
            d.verdict, d.matched_old_id, d.old_text = "conflicts", oid, _old_text(idx, old_kind(oid, d.kind), oid)
            d.handling = "alternative" if cfg.include_conflicts else "hide"
        elif verdict == "ambiguous":
            d.verdict, d.matched_old_id, d.old_text = "ambiguous", oid, _old_text(idx, old_kind(oid, d.kind), oid)
            d.handling = "add"
    return decisions


def _old_text(idx: _OldIndex, kind: str, oid: str) -> str:
    if kind == "fact":
        return idx.facts[oid].text
    if kind == "rule":
        return rule_text(idx.rules[oid])
    if kind == "concept":
        return concept_text(idx.concepts[oid])
    return oid


# ------------------------------------------------------------------------------------------ persistence
def _ensure_prov_tables(dst: sqlite3.Connection) -> None:
    dst.executescript(SPAN_DDL)
    have = {r[1] for r in dst.execute("PRAGMA table_info(spans)")}
    for col in SPAN_COLS:
        if col not in have:
            dst.execute(f"ALTER TABLE spans ADD COLUMN {col} {'REAL' if col == 'page_quality_score' else 'TEXT'}")
    hs = {r[1] for r in dst.execute("PRAGMA table_info(sections)")}
    for col in ("level", "parent_id", "ordinal", "heading_path_json", "link_status"):
        if col not in hs:
            dst.execute(f"ALTER TABLE sections ADD COLUMN {col} TEXT")


def _persist(cfg: IngestConfig, build_id: str, ingest_id: str, parent_id: str, new_onto: Ontology, decisions: list[Decision],
             merge_log: dict[str, dict], cmap: dict[str, str], nb: NewBooks, spans: dict[str, SourceSpan], idx: _OldIndex,
             items: dict, emb, results, prov_old: Optional[sqlite3.Connection], summary: dict, support_old: dict) -> dict:
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".building")
    for suf in ("", "-wal", "-shm", "-journal"):
        Path(str(tmp) + suf).unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{cfg.old_db}?mode=ro", uri=True, timeout=60)
    dst0 = sqlite3.connect(str(tmp))
    src.backup(dst0)
    src.close()
    dst0.execute("PRAGMA journal_mode=DELETE")
    dst0.executescript(MAINTENANCE_DDL)
    dst0.executescript(INGEST_DDL)
    have = {r[1] for r in dst0.execute("PRAGMA table_info(build_versions)")}
    for col in ("parent_build_id", "reason"):
        if col not in have:
            dst0.execute(f"ALTER TABLE build_versions ADD COLUMN {col} TEXT")
    dst0.commit()
    dst0.close()

    w = OntologyWriter(tmp)
    dst = w.db
    now = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    sup = {}
    book_pub = {b: book_meta_row(i) for b, i in nb.infos.items()}

    def note(obj, tags):
        v = getattr(obj, "validation", None)
        if v is not None:
            for t in tags:
                if t not in v.notes:
                    v.notes.append(t)

    # label / subject lookup by FINAL concept id (relation text, embeddings)
    labels = {c.concept_id: c.label for c in idx.concepts.values()}
    csubj = {c.concept_id: c.subject for c in idx.concepts.values()}
    for dd in decisions:
        if dd.kind == "concept" and dd.obj is not None:
            labels[dd.final_id], csubj[dd.final_id] = dd.obj.label, dd.obj.subject
    label_of = lambda cid: labels.get(cid, cid)

    vec_rows: dict[str, list[tuple[str, Optional[str], Optional[int], str]]] = defaultdict(list)
    changelog: list[dict] = []
    conflicts: list[dict] = []
    status_rows = []
    source_rows = []
    touched_old: dict[tuple[str, str], dict] = {}
    new_concept_label_alias: dict[str, list[str]] = defaultdict(list)

    for d in decisions:
        info0 = (book_pub.get(d.book_ids[0]) if d.book_ids else {}) or {}
        ev = {"ingest_id": ingest_id, "build_id": build_id, "item_kind": d.kind, "item_id": d.final_id, "new_item_id": d.new_id,
              "verdict": d.verdict, "handling": d.handling, "matched_old_id": d.matched_old_id, "books": d.book_ids,
              "span_ids": d.span_ids[:12], "new_text": d.new_text[:600], "old_text": d.old_text[:600], "notes": d.notes,
              "pair": d.pair.as_dict() if d.pair else None}
        grade = new_onto.grade.get(d.new_id)
        # ---- source ledger rows
        role = {"new": "origin", "agrees": "agrees", "ambiguous": "ambiguous", "conflicts": "alternative"}[d.verdict]
        if d.kind == "concept" and d.obj is None:                  # merged into an old concept: a second source
            role = "agrees"
        for b in (d.book_ids or [None]):
            if b is None:
                continue
            bspans = [s for s in d.span_ids if _book_of(s) == b] or d.span_ids
            source_rows.append((d.kind if d.handling in ("add", "alternative", "hide") else old_kind(d.matched_old_id or "", d.kind), d.final_id if d.handling in ("add", "alternative", "hide") else (d.matched_old_id or d.final_id),
                                b, role, json.dumps(bspans), d.matched_old_id, ingest_id, ""))
        # ---- accumulate: additive evidence on the OLD item
        if d.handling in ("accumulate", "add_variant") or (d.handling in ("review_only", "alternative") and d.kind == "concept"):
            kind, oid = old_kind(d.matched_old_id, d.kind), d.matched_old_id
            table, pk, _m = TABLES[kind]
            old = items[(kind, oid)]
            added_spans = [s for s in d.span_ids if s not in old.source_span_ids]
            added_alias: list[str] = []
            if d.kind == "concept" and cfg.merge_aliases and d.obj is None:
                rec = merge_log[d.new_id]
                newc = new_onto.concepts[d.new_id]
                have_keys = {norm_key(a) for a in [old.label, *old.aliases]}
                added_alias = [a for a in [newc.label, *newc.aliases] if norm_key(a) and norm_key(a) not in have_keys][:5]
            if d.kind == "concept" and d.handling == "review_only":
                added_spans = []                              # conflicting definition: the source spans stay with the review row
            if cfg.accumulate_provenance and (added_spans or added_alias):
                key = (kind, oid)
                cur = touched_old.get(key) or {"obj": old.model_copy(deep=True), "added_spans": [], "added_alias": [], "books": set()}
                cur["added_spans"] += [s for s in added_spans if s not in cur["added_spans"]]
                cur["added_alias"] += [a for a in added_alias if a not in cur["added_alias"]]
                cur["books"] |= set(d.book_ids)
                touched_old[key] = cur
            ev["op"] = "provenance_added" if (added_spans or added_alias) and cfg.accumulate_provenance else "agrees_recorded"
            ev["added_spans"], ev["added_aliases"] = len(added_spans), added_alias
            if d.handling != "add_variant":
                changelog.append(ev)
            if d.verdict in ("conflicts", "ambiguous") and d.kind == "concept":
                _add_conflict_row(conflicts, d, ingest_id, nb, book_pub, idx, prov_old, items, support_old)
            if d.handling != "add_variant":
                continue
            ev = {**ev, "op": "added_variant"}
            source_rows.append((d.kind, d.final_id, d.book_ids[0] if d.book_ids else "", "origin", json.dumps(d.span_ids), d.matched_old_id,
                                ingest_id, "variant wording of an agreeing item"))
        # ---- new item rows
        obj = d.obj
        tags = [f"ingest:{ingest_id}"] + ([f"ingest_book:{b}" for b in d.book_ids[:3]])
        status, valid = "ingested_new", 1
        if d.verdict == "ambiguous":
            tags += ["ingest:ambiguous", "excluded_from_strict_proof"]
            status = "ambiguous"
        elif d.verdict == "conflicts":
            tags += ["ingest:conflict", "excluded_from_strict_proof"]
            if d.handling == "alternative":
                status, valid = "alternative_source", 1
                tags.append("ingest:alternative_source")
            else:
                status, valid = "conflict_review", 0
        if d.kind != "objective":
            note(obj, tags)
        if d.kind == "concept":
            c = obj.model_copy(update={"concept_id": d.final_id})
            w.add_concept(c)
            vec_rows["concept"].append((c.concept_id, c.subject, c.grade, concept_text(c)))
        elif d.kind == "relation":
            subject = csubj.get(obj.subject_id) or "other"
            txt = f"{label_of(obj.subject_id)} {obj.relation_type} {label_of(obj.object_id)}" + (f" ({obj.qualifier})" if obj.qualifier else "")
            w.add_relation(obj, subject=subject, grade=grade, text=txt)
            vec_rows["relation"].append((obj.relation_id, subject, grade, txt))
        elif d.kind == "fact":
            w.add_fact(obj, grade=grade)
            vec_rows["fact"].append((obj.assertion_id, obj.subject, grade, obj.text))
        elif d.kind == "rule":
            w.add_rule(obj, grade=grade)
            vec_rows["rule"].append((obj.rule_id, obj.subject, grade, rule_text(obj)))
        elif d.kind == "objective":
            w.add_objective(obj)
            vec_rows["objective"].append((obj.objective_id, obj.subject, obj.grade, obj.text))
        support = getattr(getattr(obj, "validation", None), "source_support", "n/a")
        status_rows.append((d.kind, d.final_id, status, valid, None, support, f"{d.verdict}:{(d.pair.note if d.pair else '') or ','.join(d.notes)}",
                            len(d.span_ids), 0, 0, 0, 0, 0))
        ev["op"] = {"new": "added", "agrees": "added_variant", "ambiguous": "added_ambiguous", "conflicts": "hidden_conflict" if valid == 0 else "added_alternative"}[d.verdict]
        ev["support"] = support
        changelog.append(ev)
        if d.verdict in ("conflicts", "ambiguous") and d.matched_old_id:
            _add_conflict_row(conflicts, d, ingest_id, nb, book_pub, idx, prov_old, items, support_old)
    # ---- links
    n_links = 0
    for lid, l in new_onto.links.items():
        s, t_ = cmap.get(l.source_id, l.source_id), cmap.get(l.target_id, l.target_id)
        nl = l.model_copy(update={"source_id": s, "target_id": t_, "link_id": f"link:{short_hash(s, l.link_type, t_)}"})
        w.add_link(nl)
        n_links += 1
    for rj in new_onto.rejected:
        item = rj.get("item") or {}
        key = str(item.get("label") or item.get("text") or item.get("subject") or "")[:200] if isinstance(item, dict) else ""
        w.add_rejected(rj.get("kind", "?"), key, rj.get("reason", "?"), {**rj, "ingest_id": ingest_id}, rj.get("chunk_id"))
    w.commit()

    # ---- additive updates of OLD items (logged above)
    n_old_touched = 0
    for (kind, oid), cur in touched_old.items():
        table, pk, _m = TABLES[kind]
        o = cur["obj"]
        o.source_span_ids = list(dict.fromkeys(list(o.source_span_ids) + cur["added_spans"]))[: 40 if kind == "concept" else 10_000]
        if kind == "concept" and cur["added_alias"]:
            o.aliases = list(dict.fromkeys(list(o.aliases) + cur["added_alias"]))[:20]
            dst.executemany("INSERT OR IGNORE INTO concept_aliases VALUES(?,?,?,?)",
                            [(oid, a, norm_key(a), o.subject) for a in cur["added_alias"]])
            dst.execute("UPDATE concepts SET json=? WHERE concept_id=?", (o.model_dump_json(), oid))
        else:
            dst.execute(f"UPDATE {table} SET json=? WHERE {pk}=?", (o.model_dump_json(), oid))
        dst.executemany("INSERT OR IGNORE INTO evidence_links VALUES(?,?,?)", [(kind, oid, s) for s in cur["added_spans"]])
        n_old_touched += 1
    # ---- ledgers
    dst.executemany("INSERT OR REPLACE INTO item_status VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", status_rows)
    dst.executemany("INSERT OR REPLACE INTO item_sources VALUES(?,?,?,?,?,?,?,?)", source_rows)
    # old items get an `origin` row per book of their spans (once), so the ledger covers both sides
    old_cited = {}
    for (kind, iid), o in items.items():
        for b in {_book_of(s) for s in o.source_span_ids}:
            old_cited.setdefault((kind, iid, b), [s for s in o.source_span_ids if _book_of(s) == b])
    dst.executemany("INSERT OR IGNORE INTO item_sources VALUES(?,?,?,?,?,?,?,?)",
                    [(k, i, b, "origin", json.dumps(sp), None, "base", "") for (k, i, b), sp in old_cited.items()])
    dst.executemany("INSERT OR REPLACE INTO concept_merge_log VALUES(?,?,?,?,?,?,?,?,?,?)",
                    [(m["ingest_id"], m["new_concept_id"], m["new_label"], m["subject"], m["decision"], m["method"],
                      m.get("old_concept_id"), m.get("sim"), m.get("final_concept_id"), json.dumps(m.get("evidence") or {}, ensure_ascii=False))
                     for m in merge_log.values()])
    dst.executemany("INSERT OR REPLACE INTO item_conflicts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(c["conflict_id"], ingest_id, c["item_kind"], c["new_item_id"], c["old_item_id"], c["verdict"], c["tier"], c["slot"],
                      c["old_value"], c["new_value"], c["new_book_id"], json.dumps(c["old_book_ids"]), c["handling"], "open",
                      json.dumps(c["evidence"], ensure_ascii=False)) for c in conflicts])
    # book ledger: old books (role base) + new
    old_meta = _old_book_meta(cfg, sorted({b for (_, _, b) in old_cited}))
    for b, m in old_meta.items():
        dst.execute("INSERT OR IGNORE INTO book_sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (b, "base", m.get("title"), m.get("subject"), m.get("grade"), m.get("publisher"), m.get("series"),
                     m.get("edition"), m.get("edition_year"), m.get("n_spans"), None, None, json.dumps(m.get("meta") or {}, ensure_ascii=False, default=str)))
    for b, m in book_pub.items():
        dst.execute("INSERT OR REPLACE INTO book_sources VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (b, "ingested", m["title"], m["subject"], m["grade"], m["publisher"], m["series"], m["edition"], m["edition_year"],
                     len(nb.spans[b]), ingest_id, nb.origin, json.dumps(nb.infos[b].meta, ensure_ascii=False, default=str)))
    dst.commit()

    # ---- vectors (items + spans of the new books)
    emb_info: dict[str, Any] = {"model": getattr(emb, "model", None), "vectors": {}}
    if emb is not None:
        for kind, rows in vec_rows.items():
            if rows:
                w.add_embeddings(kind, emb.model, [(i, s, g) for i, s, g, _ in rows], emb.embed([t for *_, t in rows]))
                emb_info["vectors"][kind] = len(rows)
        ids = sorted(spans)
        for i in range(0, len(ids), 256):
            ch = ids[i:i + 256]
            vecs = emb.embed([spans[s].text[:6000] if spans[s].text.strip() else " " for s in ch])
            w.add_embeddings("span", emb.model, [(s, spans[s].book_id, None) for s in ch], vecs)
        emb_info["vectors"]["span"] = len(ids)
        w.commit()
    # ---- provenance of old books (self-contained file) + new books
    _write_provenance(dst, cfg, nb, prov_old, sorted({b for (_, _, b) in old_cited}))
    # ---- version row
    stats = {"verdicts": summary.get("verdicts"), "n_new_books": len(nb.infos)}
    dst.execute("INSERT OR REPLACE INTO build_versions(build_id, created_at, extractor, verifier, embedding_model, config_json,"
                " stats_json, notes, parent_build_id, reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (build_id, now, f"ingest-books of {parent_id}: {summary['extraction']['model']}|{X.PROMPT_VERSION}",
                 f"{summary['verification']['model']}|{V.VERIFY_VERSION}|{summary['verification']['mode']}",
                 emb_info["model"] or "", json.dumps({"ingest_id": ingest_id, "books": list(nb.infos), "new_spans": str(cfg.new_spans),
                                                      "old_db": str(cfg.old_db), "include_conflicts": cfg.include_conflicts,
                                                      "sim_threshold": cfg.sim_threshold, "lexical_only": cfg.lexical_only}),
                 json.dumps(stats, ensure_ascii=False, default=str), "knowledge-ingest version; see item_status, item_sources, item_conflicts",
                 parent_id, cfg.reason))
    dst.execute("INSERT OR REPLACE INTO ingest_runs VALUES(?,?,?,?,?,?,?)",
                (ingest_id, build_id, parent_id, now, json.dumps(list(nb.infos)), json.dumps(summary.get("config"), default=str), None))
    w.commit()
    dst.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('optimize')")
    w.commit()
    dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    dst.execute("PRAGMA journal_mode=DELETE")
    w.close()
    for suf in ("-wal", "-shm"):
        Path(str(tmp) + suf).unlink(missing_ok=True)
    os.replace(tmp, out)

    rdir = Path(cfg.report_dir) if cfg.report_dir else out.parent
    rdir.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    _write_jsonl(rdir / f"{stem}.changelog.jsonl", changelog)
    _write_jsonl(rdir / f"{stem}.merges.jsonl", merge_log.values())
    _write_jsonl(rdir / f"{stem}.conflicts.jsonl", conflicts)
    _write_review_csv(rdir / f"{stem}.conflicts.csv", conflicts)
    return {"db": str(out), "embeddings": emb_info, "changelog": str(rdir / f"{stem}.changelog.jsonl"),
            "conflicts_file": str(rdir / f"{stem}.conflicts.jsonl"), "merges_file": str(rdir / f"{stem}.merges.jsonl"),
            "old_items_touched": n_old_touched, "links_added": n_links, "conflict_rows": len(conflicts)}


def _old_book_meta(cfg: IngestConfig, book_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {b: {"title": b} for b in book_ids}
    reg = ROOT / "data/textbooks/registry/books.jsonl"
    meta: dict[str, dict] = {}
    if reg.exists():
        for l in reg.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                meta[d["book_id"]] = d
    prov = Path(cfg.old_prov) if cfg.old_prov else None
    if prov and prov.exists():
        con = _connect_ro(prov)
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
        if "books" in names:
            for r in con.execute("SELECT book_id, metadata_json FROM books"):
                try:
                    meta[r[0]] = {**meta.get(r[0], {}), **json.loads(r[1] or "{}")}
                except json.JSONDecodeError:
                    pass
        n = {}
        if "spans" in names:
            for b, c in con.execute("SELECT book_id, COUNT(*) FROM spans GROUP BY book_id"):
                n[b] = c
        con.close()
    else:
        n = {}
    for b in book_ids:
        m = meta.get(b, {})
        out[b] = {"title": m.get("title") or b, "subject": m.get("subject_normalized") or m.get("subject"), "grade": m.get("grade"),
                  "publisher": m.get("publisher"), "series": m.get("series"), "edition": m.get("edition"),
                  "edition_year": m.get("edition_year"), "n_spans": n.get(b), "meta": {k: m.get(k) for k in ("curriculum", "series", "publisher") if k in m}}
    return out


def _write_provenance(dst: sqlite3.Connection, cfg: IngestConfig, nb: NewBooks, prov_old: Optional[sqlite3.Connection],
                      old_books: list[str]) -> None:
    has_spans = bool(dst.execute("SELECT 1 FROM sqlite_master WHERE name='spans'").fetchone())
    _ensure_prov_tables(dst)
    if not has_spans and cfg.embed_old_provenance and cfg.old_prov and Path(cfg.old_prov).exists() and old_books:
        src = _connect_ro(Path(cfg.old_prov))
        ph = ",".join("?" * len(old_books))
        for table in ("books", "sections", "spans"):
            mine = [r[1] for r in dst.execute(f"PRAGMA table_info({table})")]
            theirs = {r[1] for r in src.execute(f"PRAGMA table_info({table})")}
            use = [c for c in mine if c in theirs]
            if not use:
                continue
            cur = src.execute(f"SELECT {','.join(use)} FROM {table} WHERE book_id IN ({ph})", old_books)
            while True:
                batch = cur.fetchmany(2000)
                if not batch:
                    break
                dst.executemany(f"INSERT OR IGNORE INTO {table}({','.join(use)}) VALUES({','.join('?' * len(use))})",
                                [tuple(r) for r in batch])
        dst.execute("INSERT INTO spans_fts(span_id, book_id, text) SELECT span_id, book_id, text FROM spans "
                    f"WHERE book_id IN ({ph})", old_books)
        dst.commit()
        src.close()
    # new books
    for b, info in nb.infos.items():
        meta = {**nb.raw_books.get(b, {}), **{k: v for k, v in book_meta_row(info).items() if v is not None}}
        meta.setdefault("subject_normalized", info.subject)
        dst.execute("INSERT OR REPLACE INTO books VALUES(?,?,?)", (b, json.dumps(meta, ensure_ascii=False, default=str), meta.get("sha256")))
    for sid, sec in nb.sections.items():
        if sec.get("book_id") in nb.infos:
            cols = [r[1] for r in dst.execute("PRAGMA table_info(sections)")]
            row = {c: sec.get(c) for c in cols}
            row["section_id"], row["book_id"] = sid, sec["book_id"]
            row["metadata_json"] = json.dumps(sec, ensure_ascii=False, default=str)
            for c in ("heading_path_json",):
                if row.get(c) is None and sec.get("heading_path") is not None:
                    row[c] = json.dumps(sec["heading_path"], ensure_ascii=False)
            dst.execute(f"INSERT OR REPLACE INTO sections({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                        [row[c] if not isinstance(row[c], (list, dict)) else json.dumps(row[c]) for c in cols])
    rows = []
    for b, ss in nb.spans.items():
        for s in ss:
            r = nb.raw_spans.get(s.span_id, {})
            hp = r.get("heading_path_json")
            if hp is None:
                hp = json.dumps(r.get("heading_path") or list(s.heading_path), ensure_ascii=False)
            page = r.get("pdf_page_index", s.pdf_page_index)
            rows.append((s.span_id, s.book_id, page, r.get("source_page_index") if r.get("source_page_index") is not None else (page or 0),
                         r.get("printed_page_label"), int(bool(r.get("printed_page_verified"))), r.get("section_id"), s.text,
                         r.get("source_sha256") or r.get("source_file_sha256") or "", r.get("start_offset"), r.get("end_offset"), hp,
                         r.get("pedagogical_type"), r.get("extractor_version"), r.get("evidence_status"), r.get("span_kind"),
                         r.get("pedagogical_label_vi"), r.get("chapter_id"), r.get("lesson_id"), r.get("page_span_id"),
                         r.get("page_quality_score"), int(bool(r.get("requires_manual_review"))),
                         r.get("text_sha1") or hashlib.sha1(s.text.encode("utf-8")).hexdigest()))
    dst.executemany(f"INSERT OR REPLACE INTO spans({','.join(SPAN_COLS)}) VALUES({','.join('?' * len(SPAN_COLS))})", rows)
    dst.executemany("INSERT INTO spans_fts VALUES(?,?,?)", [(s.span_id, s.book_id, s.text) for ss in nb.spans.values() for s in ss])
    dst.commit()


def _add_conflict_row(conflicts: list, d: Decision, ingest_id: str, nb: NewBooks, book_pub: dict, idx: _OldIndex,
                      prov_old: Optional[sqlite3.Connection], items: dict, support_old: dict) -> None:
    pv = d.pair
    old_item = items.get((old_kind(d.matched_old_id or "", d.kind), d.matched_old_id))
    old_spans = list(old_item.source_span_ids) if old_item is not None else []
    old_books = sorted({_book_of(s) for s in old_spans})
    new_b = d.book_ids[0] if d.book_ids else ""
    conflicts.append({
        "conflict_id": f"cf:{short_hash(ingest_id, d.kind, d.new_id, d.matched_old_id or '', n=14)}",
        "ingest_id": ingest_id, "item_kind": d.kind, "new_item_id": d.new_id, "old_item_id": d.matched_old_id or "",
        "verdict": d.verdict, "tier": pv.tier if pv else "", "slot": pv.slot if pv else "", "old_value": pv.old_value if pv else "",
        "new_value": pv.new_value if pv else "", "new_book_id": new_b, "old_book_ids": old_books, "handling": d.handling,
        "evidence": {
            "new_text": d.extra.get("new_def") or d.new_text, "old_text": d.extra.get("old_def") or d.old_text,
            "new_spans": d.span_ids[:6], "old_spans": old_spans[:6],
            "new_source": book_pub.get(new_b, {}), "old_support": support_old.get((old_kind(d.matched_old_id or "", d.kind), d.matched_old_id)),
            "new_snippets": [{"span_id": s, "text": nb.raw_spans[s]["text"][:420]} for s in d.span_ids[:2] if s in nb.raw_spans],
            "old_snippets": _span_snips(prov_old, old_spans, 2),
            "jaccard": pv.jaccard if pv else None, "cosine": pv.cosine if pv else None, "note": pv.note if pv else "",
            "llm_label": pv.llm_label if pv else None, "other_pairs": [
                {"old_id": o, **p.as_dict()} for o, p in d.pairs[:5] if o != d.matched_old_id]}})


def _write_review_csv(path: Path, conflicts: list[dict]) -> None:
    import csv
    cols = ["conflict_id", "item_kind", "verdict", "tier", "slot", "old_value", "new_value", "old_text", "new_text", "old_books",
            "new_book", "new_publisher", "new_edition", "old_item_id", "new_item_id", "human_label (conflict|not_conflict|same|unclear)", "human_note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for c in conflicts:
            ev = c["evidence"]
            ns = ev.get("new_source") or {}
            wr.writerow([c["conflict_id"], c["item_kind"], c["verdict"], c["tier"], c["slot"], c["old_value"], c["new_value"],
                         ev.get("old_text"), ev.get("new_text"), ";".join(c["old_book_ids"]), c["new_book_id"],
                         ns.get("publisher"), ns.get("edition"), c["old_item_id"], c["new_item_id"], "", ""])
