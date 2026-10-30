"""Read-only SQLite implementation of `otter.schemas.store_api.OntologyReader`.

Layout
------
* Ontology tables (WS2, see `otter/storage/schema.py`) live in `ontology_path`.
* Provenance tables (WS1: books, sections, spans, spans_fts) live either in the same file or in a
  separate `provenance_path`, which is ATTACHed read-only as schema `prov`. Unqualified table names
  resolve to whichever schema has them, so both layouts work transparently.

Retrieval
---------
* Lexical: SQLite FTS5 BM25 (`knowledge_fts` for ontology items, WS1's `spans_fts` for spans);
  if `spans_fts` is missing, an in-memory rank_bm25 index over spans is used.
* Dense: bge-m3 vectors stored in the `embeddings` table, pure-numpy cosine. Used only when the
  embedding server (:8002) answers; otherwise search silently degrades to lexical-only.
* Fusion: reciprocal-rank fusion (k=60). `score` is the RRF score; `score_components` keeps raw scores.

Versions / knowledge maintenance
--------------------------------
A version written by `python -m otter.builder revalidate` carries an `item_status` table. With
`valid_only=True` (the default whenever that table exists; `False` shows everything, `None` = auto) items
whose `valid` flag is 0 (orphaned or no longer supported by the new edition) are invisible to retrieval
(`search_knowledge`), typed access (`get_*`, `find_concepts`, `rules_*`, `facts_about`), the concept graph
and `neighbors`. Databases without `item_status` (the original build) are unaffected. Old versions are
immutable files; `SqliteOntologyStore.open_version(build_id, search_dir)` opens a given one.

Thread-safe: one SQLite connection per thread; in-memory caches are guarded by a lock.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional, get_args

import networkx as nx
import numpy as np

from otter.schemas.ontology import (Concept, ConditionalRule, Fact, LearningObjective, RelationType,
                                    SemanticRelation)
from otter.schemas.textbook import PedagogicalType, SourceSpan
from otter.storage.embeddings import Embedder, cosine_topk, from_blob
from otter.storage.textutil import fts_query, norm_key, syllables
from otter.storage.writer import is_strict_executable

RRF_K = 60
_PED_TYPES = set(get_args(PedagogicalType))
_EV_STATUS = {"extracted", "ocr_unverified", "human_verified"}
VALID_RELATION_TYPES = set(get_args(RelationType))
KIND_TABLE = {"concept": ("concepts", "concept_id"), "fact": ("facts", "assertion_id"),
              "rule": ("rules", "rule_id"), "relation": ("semantic_relations", "relation_id"),
              "objective": ("objectives", "objective_id")}


def _rrf(rank_lists: list[list[str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for lst in rank_lists:
        for r, key in enumerate(lst):
            out[key] = out.get(key, 0.0) + 1.0 / (RRF_K + r + 1)
    return out


class SqliteOntologyStore:
    def __init__(self, ontology_path: str | Path, *, provenance_path: str | Path | None = None,
                 embedder: Optional[Embedder] = None, use_dense: bool = True, valid_only: Optional[bool] = None):
        self.ontology_path = Path(ontology_path)
        if not self.ontology_path.exists():
            raise FileNotFoundError(self.ontology_path)
        self.provenance_path = Path(provenance_path) if provenance_path else None
        if self.provenance_path and self.provenance_path.resolve() == self.ontology_path.resolve():
            self.provenance_path = None
        self._local = threading.local()
        self._lock = threading.RLock()
        self._use_dense = use_dense
        self._embedder = embedder
        self._embedder_init = embedder is not None
        self._graph: Optional[nx.MultiDiGraph] = None
        self._dense: dict[str, tuple[list[str], np.ndarray, list, list]] = {}
        self._books: Optional[dict[str, dict]] = None
        self._books_at = 0.0
        self._span_bm25 = None
        self._valid_only_arg = valid_only
        self._invalid: Optional[frozenset] = None

    # ------------------------------------------------------------------ versions / validity
    @classmethod
    def open_version(cls, build_id: str, search_dir: str | Path, **kw) -> "SqliteOntologyStore":
        """Open the ontology DB file under `search_dir` (non-recursive `*.sqlite3`) that holds version `build_id`.
        A file whose NEWEST build_versions row is `build_id` wins (a maintained file also carries its parent's
        row, so asking for the parent id returns the parent's own immutable file when it is in the directory)."""
        for f in sorted(Path(search_dir).glob("*.sqlite3")):
            try:
                con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                ids = [r[0] for r in con.execute("SELECT build_id FROM build_versions ORDER BY created_at")]
                con.close()
            except sqlite3.Error:
                continue
            if ids and ids[-1] == build_id:
                return cls(f, **kw)
        for f in sorted(Path(search_dir).glob("*.sqlite3")):
            try:
                con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                has = con.execute("SELECT 1 FROM build_versions WHERE build_id=?", (build_id,)).fetchone()
                con.close()
            except sqlite3.Error:
                continue
            if has:
                return cls(f, **kw)
        raise FileNotFoundError(f"no ontology DB with build_id {build_id!r} under {search_dir}")

    def build_versions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM build_versions ORDER BY created_at, rowid")]

    @property
    def valid_only(self) -> bool:
        if self._valid_only_arg is not None:
            return bool(self._valid_only_arg)
        return self._has_table("item_status")

    def _invalid_keys(self) -> frozenset:
        """{'kind|id'} of items hidden in this version (empty when valid_only is off / no item_status)."""
        if self._invalid is None:
            with self._lock:
                if self._invalid is None:
                    keys: frozenset = frozenset()
                    if self.valid_only and self.db.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='item_status'").fetchone():
                        keys = frozenset(f"{r[0]}|{r[1]}" for r in self.db.execute(
                            "SELECT item_kind, item_id FROM item_status WHERE valid=0"))
                    self._invalid = keys
        return self._invalid

    def is_valid(self, kind: str, item_id: str) -> bool:
        return f"{kind}|{item_id}" not in self._invalid_keys()

    def item_status(self, kind: str, item_id: str) -> Optional[dict[str, Any]]:
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='item_status'").fetchone():
            return None
        r = self.db.execute("SELECT * FROM item_status WHERE item_kind=? AND item_id=?", (kind, item_id)).fetchone()
        return dict(r) if r else None

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_config(cls, config: Optional[dict] = None, **kw) -> "SqliteOntologyStore":
        """Default experiment store: ontology DB `paths.ontology_db` (falls back to `paths.db`) +
        provenance DB `paths.db`. Paths are relative to the repo root."""
        from otter.llm.interface import CONFIG_PATH, load_config
        cfg = config or load_config()
        root = CONFIG_PATH.parents[1]
        prov = root / cfg["paths"]["db"]
        onto = root / cfg["paths"].get("ontology_db", cfg["paths"]["db"])
        if not onto.exists():
            onto = prov
        return cls(onto, provenance_path=prov, **kw)

    @property
    def db(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.ontology_path}?mode=ro", uri=True, timeout=30,
                                  check_same_thread=False)
            con.row_factory = sqlite3.Row
            if self.provenance_path is not None:
                con.execute("ATTACH DATABASE ? AS prov", (f"file:{self.provenance_path}?mode=ro",))
            self._local.con = con
        return con

    def _has_table(self, name: str) -> bool:
        q = "SELECT 1 FROM sqlite_master WHERE name=?"
        if self.db.execute(q, (name,)).fetchone():
            return True
        if self.provenance_path is not None:
            return bool(self.db.execute("SELECT 1 FROM prov.sqlite_master WHERE name=?", (name,)).fetchone())
        return False

    def _embed_query(self, text: str) -> Optional[np.ndarray]:
        if not self._use_dense:
            return None
        if not self._embedder_init:
            with self._lock:
                if not self._embedder_init:
                    try:
                        from otter.storage.embeddings import Bgem3Embedder
                        self._embedder = Bgem3Embedder.from_config()
                    except Exception:
                        self._embedder = None
                    self._embedder_init = True
        if self._embedder is None or not self._embedder.available():
            return None
        try:
            return self._embedder.embed([text])[0]
        except Exception:
            return None

    def _dense_index(self, kind: str):
        with self._lock:
            if kind not in self._dense:
                model = getattr(self._embedder, "model", None)
                rows = self.db.execute("SELECT item_id, subject, grade, vec FROM embeddings WHERE kind=? AND model=?",
                                       (kind, model)).fetchall() if self._has_table("embeddings") else []
                ids = [r["item_id"] for r in rows]
                mat = np.stack([from_blob(r["vec"]) for r in rows]) if rows else np.zeros((0, 1), np.float32)
                self._dense[kind] = (ids, mat, [r["subject"] for r in rows], [r["grade"] for r in rows])
            return self._dense[kind]

    # --------------------------------------------------------------- metadata
    def _book_map(self) -> dict[str, dict]:
        with self._lock:
            if self._books is None or time.time() - self._books_at > 60:
                books = {}
                if self._has_table("books"):
                    for r in self.db.execute("SELECT * FROM books"):
                        d = dict(r)
                        meta = {}
                        try:
                            meta = json.loads(d.pop("metadata_json", None) or "{}")
                        except json.JSONDecodeError:
                            pass
                        meta.update({k: v for k, v in d.items() if v is not None})
                        meta.setdefault("subject", meta.get("subject_normalized"))
                        books[d["book_id"]] = meta
                self._books, self._books_at = books, time.time()
            return self._books

    def _books_for(self, subject: Optional[str], grade: Optional[int], book_ids: Optional[list[str]] = None
                   ) -> Optional[list[str]]:
        if subject is None and grade is None and book_ids is None:
            return None
        out = []
        for bid, m in self._book_map().items():
            if subject is not None and m.get("subject") != subject:
                continue
            if grade is not None and m.get("grade") is not None and int(m["grade"]) != int(grade):
                continue
            if book_ids is not None and bid not in book_ids:
                continue
            out.append(bid)
        return out

    def book_metadata(self, book_id: str) -> Optional[dict[str, Any]]:
        m = self._book_map().get(book_id)
        if m is None:
            self._books = None
            m = self._book_map().get(book_id)
        return dict(m) if m else None

    def section(self, section_id: str) -> Optional[dict[str, Any]]:
        if not section_id or not self._has_table("sections"):
            return None
        r = self.db.execute("SELECT * FROM sections WHERE section_id=?", (section_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        try:
            meta = json.loads(d.pop("metadata_json", None) or "{}")
        except json.JSONDecodeError:
            meta = {}
        meta.update({k: v for k, v in d.items() if v is not None})
        return meta

    # ------------------------------------------------------------- provenance
    @staticmethod
    def _span_from_row(row: sqlite3.Row) -> SourceSpan:
        d = dict(row)
        base: dict[str, Any] = {}
        for jcol in ("span_json", "json", "metadata_json"):
            if d.get(jcol):
                try:
                    base.update(json.loads(d.pop(jcol)))
                except (json.JSONDecodeError, TypeError):
                    pass
        base.update({k: v for k, v in d.items() if v is not None})
        hp = base.get("heading_path", base.get("heading_path_json"))
        if isinstance(hp, str):
            try:
                hp = json.loads(hp)
            except json.JSONDecodeError:
                hp = [hp]
        text = base.get("text") or ""
        ped = base.get("pedagogical_type") or "unknown"
        ev = base.get("evidence_status") or "ocr_unverified"
        return SourceSpan(
            span_id=base["span_id"], book_id=base["book_id"], section_id=base.get("section_id"),
            heading_path=list(hp or []),
            pdf_page_index=int(base.get("pdf_page_index") if base.get("pdf_page_index") is not None
                               else base.get("source_page_index") or 0),
            printed_page_label=base.get("printed_page_label"),
            printed_page_verified=bool(base.get("printed_page_verified") or False),
            start_offset=int(base.get("start_offset") or 0),
            end_offset=int(base["end_offset"]) if base.get("end_offset") is not None else len(text),
            text=text, pedagogical_type=ped if ped in _PED_TYPES else "unknown",
            source_file_sha256=base.get("source_file_sha256") or base.get("source_sha256") or "",
            extractor_version=base.get("extractor_version") or "",
            evidence_status=ev if ev in _EV_STATUS else "ocr_unverified")

    def get_spans(self, span_ids: list[str]) -> list[SourceSpan]:
        if not span_ids or not self._has_table("spans"):
            return []
        out: dict[str, SourceSpan] = {}
        ids = list(dict.fromkeys(span_ids))
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = f"SELECT * FROM spans WHERE span_id IN ({','.join('?' * len(chunk))})"
            for r in self.db.execute(q, chunk):
                out[r["span_id"]] = self._span_from_row(r)
        return [out[s] for s in ids if s in out]

    def span_exists(self, span_id: str) -> bool:
        if not span_id or not self._has_table("spans"):
            return False
        return self.db.execute("SELECT 1 FROM spans WHERE span_id=?", (span_id,)).fetchone() is not None

    def search_spans(self, query: str, *, subject: Optional[str] = None, grade: Optional[int] = None,
                     book_ids: Optional[list[str]] = None, top_k: int = 10) -> list[tuple[SourceSpan, float]]:
        allowed = self._books_for(subject, grade, book_ids)
        if allowed is not None and not allowed:
            return []
        n_cand = max(top_k * 4, 40)
        lex: list[str] = []
        comp: dict[str, dict[str, float]] = {}
        if self._has_table("spans_fts"):
            mq = fts_query(query)
            if mq:
                sql = "SELECT span_id, bm25(spans_fts) AS s FROM spans_fts WHERE spans_fts MATCH ?"
                args: list[Any] = [mq]
                if allowed is not None:
                    sql += f" AND book_id IN ({','.join('?' * len(allowed))})"
                    args += allowed
                sql += " ORDER BY s LIMIT ?"
                args.append(n_cand)
                try:
                    for r in self.db.execute(sql, args):
                        lex.append(r["span_id"])
                        comp.setdefault(r["span_id"], {})["bm25"] = -float(r["s"])
                except sqlite3.OperationalError:
                    lex = []
        else:
            lex = self._bm25_fallback(query, allowed, n_cand, comp)
        dense = self._dense_rank("span", query, subject=None, grade=None, n=n_cand, comp=comp,
                                 allowed_books=allowed)
        fused = _rrf([lex, dense]) if dense else {k: 1.0 / (RRF_K + i + 1) for i, k in enumerate(lex)}
        best = sorted(fused, key=lambda k: -fused[k])[: top_k]
        spans = {s.span_id: s for s in self.get_spans(best)}
        return [(spans[k], fused[k]) for k in best if k in spans]

    def _bm25_fallback(self, query: str, allowed, n: int, comp) -> list[str]:
        if not self._has_table("spans"):
            return []
        with self._lock:
            if self._span_bm25 is None:
                from rank_bm25 import BM25Okapi
                rows = self.db.execute("SELECT span_id, book_id, text FROM spans").fetchall()
                docs = [syllables(r["text"]) or ["_"] for r in rows]
                self._span_bm25 = (BM25Okapi(docs), [r["span_id"] for r in rows], [r["book_id"] for r in rows])
        bm, ids, books = self._span_bm25
        scores = bm.get_scores(syllables(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0 or len(out) >= n:
                break
            if allowed is not None and books[i] not in allowed:
                continue
            out.append(ids[i])
            comp.setdefault(ids[i], {})["bm25"] = float(scores[i])
        return out

    def _dense_rank(self, kind: str, query: str, *, subject, grade, n: int, comp, allowed_books=None) -> list[str]:
        if not self._has_table("embeddings"):
            return []
        qv = self._embed_query(query)
        if qv is None:
            return []
        ids, mat, subs, grades = self._dense_index(kind)
        if not ids or mat.shape[1] != qv.shape[0]:
            return []
        mask = np.ones(len(ids), dtype=bool)
        if kind == "span" and allowed_books is not None:
            allowed = set(allowed_books)
            mask &= np.array([s in allowed for s in subs])   # for kind=span the `subject` column holds book_id
        if subject is not None:
            mask &= np.array([s == subject for s in subs])
        if grade is not None:
            mask &= np.array([g is None or int(g) == int(grade) for g in grades])
        if not mask.any():
            return []
        sel = np.nonzero(mask)[0]
        hits = cosine_topk(qv, mat[sel], n)
        out = []
        for j, sim in hits:
            iid = ids[sel[j]]
            out.append(iid)
            comp.setdefault(iid, {})["dense"] = sim
        return out

    # ---------------------------------------------------------------- search
    def search_knowledge(self, query: str, *, subject: Optional[str] = None, grade: Optional[int] = None,
                         kinds: tuple[str, ...] = ("concept", "fact", "rule"), top_k: int = 10) -> list[dict[str, Any]]:
        kinds = tuple(k for k in kinds if k in KIND_TABLE)
        if not kinds or not self._has_table("knowledge_fts"):
            return []
        n_cand = max(top_k * 4, 40)
        comp: dict[str, dict[str, float]] = {}
        lex: list[str] = []
        bad = self._invalid_keys()
        n_cand_q = n_cand + (min(len(bad), 400) if bad else 0)
        mq = fts_query(query)
        if mq:
            sql = (f"SELECT kind, item_id, bm25(knowledge_fts) AS s FROM knowledge_fts WHERE knowledge_fts MATCH ? "
                   f"AND kind IN ({','.join('?' * len(kinds))})")
            args: list[Any] = [mq, *kinds]
            if subject is not None:
                sql += " AND subject = ?"
                args.append(subject)
            if grade is not None:
                sql += " AND (grade IS NULL OR grade = ?)"
                args.append(int(grade))
            sql += " ORDER BY s LIMIT ?"
            args.append(n_cand_q)
            try:
                for r in self.db.execute(sql, args):
                    key = f"{r['kind']}|{r['item_id']}"
                    if key in bad:
                        continue
                    lex.append(key)
                    comp.setdefault(key, {})["bm25"] = -float(r["s"])
            except sqlite3.OperationalError:
                lex = []
        dense_lists = []
        for kind in kinds:
            c2: dict[str, dict[str, float]] = {}
            d = self._dense_rank(kind, query, subject=subject, grade=grade, n=n_cand_q, comp=c2)
            if bad:
                d = [i for i in d if f"{kind}|{i}" not in bad]
            if d:
                dense_lists.append(([f"{kind}|{i}" for i in d], c2, kind))
        if dense_lists:
            merged = []
            for keys, c2, kind in dense_lists:
                for k in keys:
                    comp.setdefault(k, {}).update(c2[k.split("|", 1)[1]])
                merged.extend(keys)
            merged.sort(key=lambda k: -comp[k]["dense"])
            fused = _rrf([lex, merged])
        else:
            fused = {k: 1.0 / (RRF_K + i + 1) for i, k in enumerate(lex)}
        best = sorted(fused, key=lambda k: -fused[k])[:top_k]
        hits = []
        for key in best:
            kind, iid = key.split("|", 1)
            item = self._hit(kind, iid)
            if item is None:
                continue
            item["score"] = round(fused[key], 6)
            item["score_components"] = comp.get(key, {})
            hits.append(item)
        return hits

    def _hit(self, kind: str, iid: str) -> Optional[dict[str, Any]]:
        table, pk = KIND_TABLE[kind]
        r = self.db.execute(f"SELECT * FROM {table} WHERE {pk}=?", (iid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        if kind == "concept":
            txt = d["label"] + (f": {d['definition']}" if d.get("definition") else "")
        elif kind == "relation":
            obj = json.loads(d["json"])
            txt = f"{self._label(obj['subject_id'])} --{obj['relation_type']}--> {self._label(obj['object_id'])}"
        else:
            txt = d["text"]
        hit = {"kind": kind, "id": iid, "label_or_text": txt, "subject": d.get("subject"), "grade": d.get("grade"),
               "source_span_ids": self._evidence(kind, iid), "source_support": d.get("source_support"),
               "status": d.get("status")}
        if kind == "rule":
            hit["strict_executable"] = bool(d.get("strict_executable"))
        return hit

    def _label(self, concept_id: str) -> str:
        r = self.db.execute("SELECT label FROM concepts WHERE concept_id=?", (concept_id,)).fetchone()
        return r["label"] if r else concept_id

    def _evidence(self, kind: str, iid: str) -> list[str]:
        return [r[0] for r in self.db.execute(
            "SELECT span_id FROM evidence_links WHERE item_kind=? AND item_id=? ORDER BY span_id", (kind, iid))]

    # ----------------------------------------------------------- typed access
    def _get_json(self, table: str, pk: str, iid: str):
        if not iid:
            return None
        r = self.db.execute(f"SELECT json FROM {table} WHERE {pk}=?", (iid,)).fetchone()
        return json.loads(r[0]) if r else None

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        if not self.is_valid("concept", concept_id):
            return None
        d = self._get_json("concepts", "concept_id", concept_id)
        return Concept.model_validate(d) if d else None

    def get_fact(self, assertion_id: str) -> Optional[Fact]:
        if not self.is_valid("fact", assertion_id):
            return None
        d = self._get_json("facts", "assertion_id", assertion_id)
        return Fact.model_validate(d) if d else None

    def get_rule(self, rule_id: str) -> Optional[ConditionalRule]:
        if not self.is_valid("rule", rule_id):
            return None
        d = self._get_json("rules", "rule_id", rule_id)
        return ConditionalRule.model_validate(d) if d else None

    def get_relation(self, relation_id: str) -> Optional[SemanticRelation]:
        if not self.is_valid("relation", relation_id):
            return None
        d = self._get_json("semantic_relations", "relation_id", relation_id)
        return SemanticRelation.model_validate(d) if d else None

    def get_objective(self, objective_id: str) -> Optional[LearningObjective]:
        if not self.is_valid("objective", objective_id):
            return None
        d = self._get_json("objectives", "objective_id", objective_id)
        return LearningObjective.model_validate(d) if d else None

    def find_concepts(self, label: str, *, subject: Optional[str] = None) -> list[Concept]:
        """Exact (NFC/casefold, diacritic-preserving) label/alias lookup."""
        sql = "SELECT DISTINCT concept_id FROM concept_aliases WHERE alias_norm=?"
        args: list[Any] = [norm_key(label)]
        if subject is not None:
            sql += " AND subject=?"
            args.append(subject)
        return [c for c in (self.get_concept(r[0]) for r in self.db.execute(sql, args)) if c]

    def rules_concluding(self, predicate: str, *, subject: Optional[str] = None) -> list[ConditionalRule]:
        sql = ("SELECT DISTINCT c.rule_id FROM rule_conclusions c JOIN rules r ON r.rule_id=c.rule_id "
               "WHERE c.predicate=?")
        args: list[Any] = [predicate]
        if subject is not None:
            sql += " AND r.subject=?"
            args.append(subject)
        return [x for x in (self.get_rule(r[0]) for r in self.db.execute(sql + " ORDER BY c.rule_id", args)) if x]

    def rules_with_premise(self, predicate: str, *, subject: Optional[str] = None) -> list[ConditionalRule]:
        sql = ("SELECT DISTINCT p.rule_id FROM (SELECT rule_id, predicate FROM rule_premises UNION ALL "
               "SELECT rule_id, predicate FROM rule_guards) p JOIN rules r ON r.rule_id=p.rule_id WHERE p.predicate=?")
        args: list[Any] = [predicate]
        if subject is not None:
            sql += " AND r.subject=?"
            args.append(subject)
        return [x for x in (self.get_rule(r[0]) for r in self.db.execute(sql + " ORDER BY p.rule_id", args)) if x]

    def facts_about(self, concept_id: str) -> list[Fact]:
        return [f for f in (self.get_fact(r[0]) for r in self.db.execute(
            "SELECT assertion_id FROM fact_concepts WHERE concept_id=? ORDER BY assertion_id", (concept_id,))) if f]

    @staticmethod
    def is_strict_executable(rule: ConditionalRule) -> bool:
        return is_strict_executable(rule)

    # ---------------------------------------------------------------- graph
    def graph(self) -> nx.MultiDiGraph:
        with self._lock:
            if self._graph is None:
                g = nx.MultiDiGraph()
                bad = self._invalid_keys()
                for r in self.db.execute("SELECT concept_id, subject FROM concepts"):
                    if f"concept|{r[0]}" not in bad:
                        g.add_node(r[0], subject=r[1])
                for r in self.db.execute(
                        "SELECT relation_id, subject_id, relation_type, object_id, subject FROM semantic_relations"):
                    if f"relation|{r[0]}" not in bad and r[1] in g and r[3] in g:
                        g.add_edge(r[1], r[3], key=r[0], relation_type=r[2], subject=r[4])
                self._graph = g
            return self._graph

    def neighbors(self, node_ids: list[str], *, relation_types: Optional[list[str]] = None, max_hops: int = 1,
                  subject: Optional[str] = None, max_relations: int = 300) -> dict[str, Any]:
        g = self.graph()
        invalid = sorted(set(relation_types or []) - VALID_RELATION_TYPES)
        rtypes = set(relation_types) if relation_types else None
        seeds: set[str] = set()
        seed_rules: list[str] = []
        unknown: list[str] = []
        for nid in node_ids:
            if nid in g:
                seeds.add(nid)
            elif self.db.execute("SELECT 1 FROM rules WHERE rule_id=?", (nid,)).fetchone():
                seed_rules.append(nid)
                seeds.update(r[0] for r in self.db.execute("SELECT concept_id FROM rule_concepts WHERE rule_id=?", (nid,)))
            elif self.db.execute("SELECT 1 FROM facts WHERE assertion_id=?", (nid,)).fetchone():
                seeds.update(r[0] for r in self.db.execute("SELECT concept_id FROM fact_concepts WHERE assertion_id=?", (nid,)))
            else:
                unknown.append(nid)
        seeds = {s for s in seeds if s in g}
        if subject is not None:
            seeds = {s for s in seeds if g.nodes[s].get("subject") == subject}
        dist = {s: 0 for s in seeds}
        rel_ids: list[str] = []
        frontier = set(seeds)
        for hop in range(1, max(0, max_hops) + 1):
            nxt = set()
            for u in frontier:
                edges = [(u, v, k, d) for _, v, k, d in g.out_edges(u, keys=True, data=True)] + \
                        [(w, u, k, d) for w, _, k, d in g.in_edges(u, keys=True, data=True)]
                for a, b, k, d in edges:
                    if rtypes is not None and d["relation_type"] not in rtypes:
                        continue
                    if subject is not None and d.get("subject") != subject:
                        continue
                    if k not in rel_ids:
                        rel_ids.append(k)
                    other = b if a == u else a
                    if other not in dist:
                        dist[other] = hop
                        nxt.add(other)
                if len(rel_ids) >= max_relations:
                    break
            frontier = nxt
            if not frontier or len(rel_ids) >= max_relations:
                break
        rel_ids = rel_ids[:max_relations]
        concept_ids = list(dist)
        rule_ids = list(dict.fromkeys(seed_rules))
        fact_ids: list[str] = []
        for i in range(0, len(concept_ids), 500):
            ch = concept_ids[i:i + 500]
            ph = ",".join("?" * len(ch))
            rule_ids += [r[0] for r in self.db.execute(
                f"SELECT DISTINCT rule_id FROM rule_concepts WHERE concept_id IN ({ph}) ORDER BY rule_id", ch)]
            fact_ids += [r[0] for r in self.db.execute(
                f"SELECT DISTINCT assertion_id FROM fact_concepts WHERE concept_id IN ({ph}) ORDER BY assertion_id", ch)]
        rule_ids = list(dict.fromkeys(rule_ids))
        rules = [r for r in (self.get_rule(x) for x in rule_ids) if r and (subject is None or r.subject == subject)]
        return {
            "nodes": [c for c in (self.get_concept(x) for x in concept_ids) if c],
            "relations": [r for r in (self.get_relation(x) for x in rel_ids) if r],
            "rules": rules,
            "facts": [f for f in (self.get_fact(x) for x in dict.fromkeys(fact_ids)) if f],
            "hops": dist,
            "unknown_ids": unknown,
            "invalid_relation_types": invalid,
        }

    # ---------------------------------------------------------------- misc
    def stats(self) -> dict[str, int]:
        out = {}
        for t in ("concepts", "semantic_relations", "facts", "rules", "objectives", "pedagogical_links",
                  "evidence_links", "embeddings"):
            try:
                out[t] = self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                out[t] = 0
        out["strict_executable_rules"] = self.db.execute(
            "SELECT COUNT(*) FROM rules WHERE strict_executable=1").fetchone()[0]
        out["spans"] = self.db.execute("SELECT COUNT(*) FROM spans").fetchone()[0] if self._has_table("spans") else 0
        out["hidden_invalid_items"] = len(self._invalid_keys())
        return out

    def build_info(self) -> Optional[dict[str, Any]]:
        r = self.db.execute("SELECT * FROM build_versions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def close(self) -> None:
        con = getattr(self._local, "con", None)
        if con is not None:
            con.close()
            self._local.con = None
