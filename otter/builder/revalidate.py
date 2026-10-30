"""Knowledge maintenance: re-validate an existing ontology build against a NEW edition of the corpus.

ADD-ON module: it never changes the answering code or configs, and it never writes the old ontology file.
Output is a NEW immutable version file (+ changelog) whose `build_versions` row points at its parent.

Pipeline
  1. align old spans -> new spans (otter/builder/align.py; ids are not trusted, nothing is forced)
  2. classify every concept / relation / fact / rule / objective from the status of the spans it cites:
        unchanged     all cited spans identical (exact or whitespace-normalised, possibly moved)
        modified      at least one cited span changed (or part of the evidence is gone): the existing
                      source-support check (validate.SupportChecker, verifier model, different from the
                      extractor) is re-run against the NEW text -> supported / partially_supported / unsupported
        orphaned      every cited span is gone
        needs_review  alignment ambiguous, or the verifier could not decide
  3. write the new version: item JSON/columns updated only for items that changed, span ids remapped to the new
     edition, `item_status` (valid flag) + `span_alignment` tables, span vectors (reused when the text is identical),
     optional self-contained provenance (books/sections/spans/spans_fts of the NEW edition).
  `valid = 0` (orphaned, or modified and now unsupported, or a relation whose endpoint concept is invalid) items are hidden
  by `SqliteOntologyStore(valid_only=True)` (default for a maintained file) and rules lose `strict_executable`.
  Unchanged items keep their old label (including old 'unsupported' ones): maintenance only acts on what the edition changed.

Not done (see docs/KNOWLEDGE_MAINTENANCE.md): no re-extraction of NEW content, no rewriting of item texts, no concept
split/merge repair, no cross-item reasoning about rules whose premises sit in different spans beyond the re-check.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.builder.align import (AlignConfig, AlignRow, SpanRec, align_spans, load_span_records, span_category,
                                 summarize)
from otter.builder.validate import SupportChecker, claim_text, VERIFY_SYSTEM
from otter.llm.interface import LLM
from otter.schemas.ontology import (Concept, ConditionalRule, Fact, LearningObjective, PedagogicalLink,
                                    SemanticRelation, Validation)
from otter.storage.embeddings import from_blob, to_blob
from otter.storage.schema import MAINTENANCE_DDL
from otter.storage.textutil import lexical_overlap, norm_key
from otter.storage.writer import is_strict_executable

log = logging.getLogger("otter.builder.revalidate")

KINDS = ("concept", "relation", "fact", "rule", "objective")
TABLES = {"concept": ("concepts", "concept_id", Concept), "relation": ("semantic_relations", "relation_id", SemanticRelation),
          "fact": ("facts", "assertion_id", Fact), "rule": ("rules", "rule_id", ConditionalRule),
          "objective": ("objectives", "objective_id", LearningObjective)}
SPAN_COLS = ["span_id", "book_id", "pdf_page_index", "source_page_index", "printed_page_label", "printed_page_verified",
             "section_id", "text", "source_sha256", "start_offset", "end_offset", "heading_path_json", "pedagogical_type",
             "extractor_version", "evidence_status", "span_kind", "pedagogical_label_vi", "chapter_id", "lesson_id",
             "page_span_id", "page_quality_score", "requires_manual_review", "text_sha1"]
SPAN_DDL = """
CREATE TABLE IF NOT EXISTS books(book_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, source_sha256 TEXT);
CREATE TABLE IF NOT EXISTS sections(section_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT, pdf_page_start INTEGER,
  pdf_page_end INTEGER, printed_page_start TEXT, printed_page_end TEXT, metadata_json TEXT NOT NULL, level TEXT,
  parent_id TEXT, ordinal TEXT, heading_path_json TEXT, link_status TEXT);
CREATE TABLE IF NOT EXISTS spans(span_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, pdf_page_index INTEGER, source_page_index INTEGER NOT NULL,
  printed_page_label TEXT, printed_page_verified INTEGER NOT NULL, section_id TEXT, text TEXT NOT NULL, source_sha256 TEXT NOT NULL,
  start_offset INTEGER, end_offset INTEGER, heading_path_json TEXT, pedagogical_type TEXT, extractor_version TEXT, evidence_status TEXT,
  span_kind TEXT, pedagogical_label_vi TEXT, chapter_id TEXT, lesson_id TEXT, page_span_id TEXT, page_quality_score REAL,
  requires_manual_review INTEGER, text_sha1 TEXT);
CREATE INDEX IF NOT EXISTS ix_spans_book_page ON spans(book_id, pdf_page_index);
CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(span_id UNINDEXED, book_id UNINDEXED, text);
"""


@dataclass
class RevalConfig:
    old_db: Path
    new_spans: Path
    out: Path
    old_prov: Optional[Path] = None             # provenance DB of the OLD edition (default: paths.db of configs/default.yaml)
    old_spans: Optional[Path] = None            # alternative: spans.jsonl / dir / db for the old edition
    book_prefix: Optional[str] = None
    verifier_role: str = "small"
    workers: int = 8
    batch_size: int = 12
    lost_span_policy: str = "recheck"           # recheck | strict (any cited span gone => orphaned)
    lexical_only: bool = False                  # no LLM: lexical support label is final (flagged)
    lexical_shortcut: bool = False              # skip the LLM when claim/evidence overlap did not drop (cost option)
    embed: bool = True                          # span vectors for the new edition (bge-m3 :8002)
    embed_provenance: bool = True               # write books/sections/spans of the NEW edition into the new file
    reason: str = "new edition of the corpus"
    build_id: Optional[str] = None
    report_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    dry_run: bool = False                       # classify + count, write nothing
    align: AlignConfig = field(default_factory=AlignConfig)


# ------------------------------------------------------------------------------------------ item loading
def _connect_ro(p: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def load_items(old_db: Path) -> tuple[dict[tuple[str, str], Any], dict[str, Concept], dict[tuple[str, str], str], dict]:
    """({(kind,id): pydantic obj}, {concept_id: Concept}, {(kind,id): old support label}, build_row)."""
    con = _connect_ro(old_db)
    items: dict[tuple[str, str], Any] = {}
    for kind, (table, pk, model) in TABLES.items():
        for r in con.execute(f"SELECT {pk}, json FROM {table}"):
            items[(kind, r[0])] = model.model_validate_json(r[1])
    concepts = {i: o for (k, i), o in items.items() if k == "concept"}
    support = {}
    for key, o in items.items():
        v = getattr(o, "validation", None)
        support[key] = v.source_support if v is not None else "n/a"
    row = con.execute("SELECT * FROM build_versions ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    build = dict(row) if row else {}
    con.close()
    return items, concepts, support, build


# ------------------------------------------------------------------------------------------ shims for SupportChecker
class _Txt:
    __slots__ = ("text", "span_id")

    def __init__(self, span_id: str, text: str):
        self.span_id, self.text = span_id, text


class _Shim:
    """Duck-typed stand-in for `normalize.Ontology` (only what SupportChecker.run / claim_text read)."""

    def __init__(self, concepts: dict[str, Concept], work: list[tuple[str, str, Any]], groups: dict[str, str]):
        self.concepts = concepts
        self._work = work
        self.item_chunks = {iid: {g} for iid, g in groups.items()}

    def items(self):
        yield from self._work


def _objective_as_fact(o: LearningObjective) -> Fact:
    return Fact(assertion_id=o.objective_id, text=o.text, subject=o.subject, source_span_ids=list(o.source_span_ids))


# ------------------------------------------------------------------------------------------ classification
@dataclass
class ItemDecision:
    kind: str
    item_id: str
    status: str                      # unchanged | modified | orphaned | needs_review
    valid: bool
    support_old: str
    support_new: str
    reason: str = ""
    old_span_ids: list[str] = field(default_factory=list)
    new_span_ids: list[str] = field(default_factory=list)
    lost_span_ids: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=lambda: {"same": 0, "changed": 0, "gone": 0, "unresolved": 0})
    rechecked: bool = False
    score: Optional[float] = None
    notes: list[str] = field(default_factory=list)


def classify_item(kind: str, iid: str, obj: Any, rows: dict[str, AlignRow], policy: str) -> ItemDecision:
    sids = list(obj.source_span_ids)
    cnt = {"same": 0, "changed": 0, "gone": 0, "unresolved": 0}
    new_ids: list[str] = []
    lost: list[str] = []
    for s in sids:
        r = rows.get(s)
        cat = span_category(r.status) if r is not None else "unresolved"
        cnt[cat] += 1
        if r is not None and cat in ("same", "changed"):
            for n in r.new_ids:
                if n not in new_ids:
                    new_ids.append(n)
        elif cat == "gone":
            lost.append(s)
    sup_old = getattr(getattr(obj, "validation", None), "source_support", "n/a")
    d = ItemDecision(kind, iid, "unchanged", True, sup_old, sup_old, old_span_ids=sids, new_span_ids=new_ids,
                     lost_span_ids=lost, counts=cnt)
    if not sids:
        d.status, d.reason = "needs_review", "no_cited_spans"
    elif cnt["unresolved"]:
        d.status, d.reason = "needs_review", "ambiguous_alignment"
    elif cnt["gone"] == len(sids):
        d.status, d.valid, d.support_new, d.reason = "orphaned", False, "unsupported", "all_cited_spans_gone"
    elif cnt["changed"] == 0 and cnt["gone"] == 0:
        d.status = "unchanged"
    elif policy == "strict" and cnt["gone"]:
        d.status, d.valid, d.support_new, d.reason = "orphaned", False, "unsupported", "cited_span_gone(strict)"
    else:
        d.status = "modified"
        d.reason = ("cited_span_changed" if cnt["changed"] else "") + \
                   ("+" if cnt["changed"] and cnt["gone"] else "") + ("evidence_partially_gone" if cnt["gone"] else "")
    return d


# ------------------------------------------------------------------------------------------ cost helpers
def plan_batches(work: list[tuple[str, str, Any]], groups: dict[str, str], batch_size: int) -> list[list]:
    """Replicates SupportChecker.run's batching (concepts without definition are decided without the LLM)."""
    todo: dict[str, list] = defaultdict(list)
    for kind, iid, obj in work:
        if kind == "concept" and not obj.definition:
            continue
        todo[groups.get(iid, "_")].append((kind, iid, obj))
    batches = []
    for g, its in todo.items():
        for i in range(0, len(its), batch_size):
            batches.append(its[i:i + batch_size])
    return batches


def estimate_prompt_chars(batches: list[list], texts: dict[str, str], onto_concepts: dict[str, Concept],
                          max_span_chars: int = 1500) -> int:
    shim = _Shim(onto_concepts, [], {})
    total = 0
    for b in batches:
        ev = list(dict.fromkeys(s for _, _, o in b for s in o.source_span_ids[:6]))
        total += len(VERIFY_SYSTEM) + sum(min(len(texts.get(s, "")), max_span_chars) + 8 for s in ev)
        total += sum(len(claim_text(k, o, shim)) + 24 for k, _, o in b)
    return total


def _group_of(obj, span_lesson: dict[str, str]) -> str:
    for s in obj.source_span_ids:
        if s in span_lesson:
            return span_lesson[s]
    return "_"


# ------------------------------------------------------------------------------------------ main
def revalidate(cfg: RevalConfig, *, verifier: Optional[LLM] = None, embedder=None) -> dict[str, Any]:
    t0 = time.time()
    timings: dict[str, float] = {}
    old_db = Path(cfg.old_db)
    out = Path(cfg.out)
    if out.resolve() == old_db.resolve():
        raise ValueError("refusing to overwrite the old ontology file: --out must be a NEW file")
    items, concepts, support_old, old_build = load_items(old_db)
    old_build_id = old_build.get("build_id", "unknown")
    book_ids = sorted({s.split(":")[0] for o in items.values() for s in o.source_span_ids})
    if cfg.book_prefix:
        book_ids = [b for b in book_ids if b.startswith(cfg.book_prefix)]

    # ---- 1. spans of both editions + alignment
    t = time.time()
    old_src = cfg.old_spans or cfg.old_prov
    if old_src is None:
        from otter.llm.interface import CONFIG_PATH, load_config
        old_src = CONFIG_PATH.parents[1] / load_config()["paths"]["db"]
    old_recs = load_span_records(old_src, book_ids=book_ids)
    new_recs = load_span_records(cfg.new_spans, book_ids=book_ids)
    old_texts = {s.span_id: s.text for s in old_recs}
    new_texts = {s.span_id: s.text for s in new_recs}
    old_vecs: dict[str, np.ndarray] = {}
    con = _connect_ro(old_db)
    model_name = getattr(embedder, "model", None)
    if model_name is None:
        r0 = con.execute("SELECT model FROM embeddings WHERE kind='span' LIMIT 1").fetchone()
        model_name = r0[0] if r0 else None
    if model_name:
        for r in con.execute("SELECT item_id, vec FROM embeddings WHERE kind='span' AND model=?", (model_name,)):
            old_vecs[r[0]] = from_blob(r[1])
    con.close()
    rows, added = align_spans(old_recs, new_recs, embedder=embedder, old_vecs=old_vecs, cfg=cfg.align)
    align_summary = summarize(rows)
    timings["align"] = round(time.time() - t, 2)
    log.info("alignment: %s; %d new spans without an old counterpart", align_summary, len(added))

    # ---- 2. classify
    decisions: dict[tuple[str, str], ItemDecision] = {}
    for key, obj in items.items():
        decisions[key] = classify_item(key[0], key[1], obj, rows, cfg.lost_span_policy)
    by_status = Counter(d.status for d in decisions.values())
    log.info("classification: %s", dict(by_status))

    # ---- 3. re-check modified items against the NEW text
    span_lesson = {s.span_id: (s.raw.get("lesson_id") or s.book_id) for s in old_recs}
    new_lesson = {s.span_id: (s.raw.get("lesson_id") or s.book_id) for s in new_recs}
    work: list[tuple[str, str, Any]] = []
    groups: dict[str, str] = {}
    shortcut: dict[str, str] = {}
    copies: dict[tuple[str, str], Any] = {}
    for key, d in decisions.items():
        if d.status != "modified":
            continue
        obj = items[key]
        kind, iid = key
        if kind == "objective":
            c = _objective_as_fact(obj)
        else:
            c = obj.model_copy(deep=True)
            c.validation = Validation(schema_valid=True, source_ids_exist=True)
        c.source_span_ids = list(d.new_span_ids)
        claim = claim_text(kind if kind != "objective" else "fact", c, _Shim(concepts, [], {}))
        if cfg.lexical_shortcut and d.support_old == "supported" and kind != "objective":
            old_ev = " ".join(old_texts.get(s, "") for s in d.old_span_ids)
            new_ev = " ".join(new_texts.get(s, "") for s in d.new_span_ids)
            if lexical_overlap(claim, new_ev) >= lexical_overlap(claim, old_ev) - 0.02 and d.new_span_ids:
                d.rechecked, d.support_new, d.reason = False, "supported", d.reason + "+lexical_stable"
                shortcut[iid] = "supported"
                continue
        wkind = "fact" if kind == "objective" else kind
        wid = iid
        copies[key] = c
        work.append((wkind, wid, c))
        groups[wid] = _group_of(c, new_lesson)
    sc = SupportChecker(None if (cfg.lexical_only or verifier is None) else verifier,
                        cache_dir=Path(cfg.cache_dir) if cfg.cache_dir else None, batch_size=cfg.batch_size)
    batches = plan_batches(work, groups, cfg.batch_size)
    full_work = [(k[0] if k[0] != "objective" else "fact", k[1], (_objective_as_fact(o) if k[0] == "objective" else o))
                 for k, o in items.items()]
    full_groups = {k[1]: _group_of(o, span_lesson) for k, o in items.items()}
    full_batches = plan_batches(full_work, full_groups, cfg.batch_size)
    cost = {"items_rechecked": len(work), "items_total": len(items), "batches_planned": len(batches),
            "batches_full_revalidation": len(full_batches),
            "prompt_chars_planned": estimate_prompt_chars(batches, new_texts, concepts),
            "prompt_chars_full_revalidation": estimate_prompt_chars(full_batches, old_texts, concepts),
            "lexical_shortcut_skipped": len(shortcut)}
    t = time.time()
    if work and not cfg.dry_run:
        sc.run(_Shim(concepts, work, groups), {sid: _Txt(sid, tx) for sid, tx in new_texts.items()}, workers=cfg.workers)
    timings["recheck"] = round(time.time() - t, 2)
    cost.update({"llm_calls": sc.stats["llm_calls"], "cached_batches": sc.stats["cached_batches"],
                 "failed_batches": sc.stats["failed_batches"], "prompt_tokens": sc.stats["prompt_tokens"],
                 "completion_tokens": sc.stats["completion_tokens"]})

    for key, c in copies.items():
        d = decisions[key]
        d.rechecked = True
        if cfg.dry_run:
            d.support_new, d.valid = "needs_review", True
            continue
        v = c.validation
        lex_only = "support_lexical_only" in v.notes
        d.score = v.support_score
        d.notes = [n for n in v.notes if n not in ("support_lexical_only",)]
        if lex_only and not cfg.lexical_only:
            d.status, d.valid, d.support_new, d.reason = "needs_review", True, "needs_review", d.reason + "+verifier_failed"
        else:
            d.support_new = v.source_support
            d.valid = v.source_support != "unsupported"
            if not d.valid:
                d.reason += "+unsupported_after_change"
    # relation dependency on an invalid concept: an edge to a hidden node must not be traversable
    bad_c = {k[1] for k, d in decisions.items() if k[0] == "concept" and not d.valid}
    for key, d in decisions.items():
        if key[0] == "relation" and d.valid:
            r = items[key]
            if r.subject_id in bad_c or r.object_id in bad_c:
                d.valid, d.reason = False, (d.reason + "+" if d.reason else "") + "endpoint_concept_invalid"
    counts: dict[str, dict[str, int]] = {k: Counter() for k in KINDS}
    for (kind, _), d in decisions.items():
        counts[kind][d.status] += 1
        counts[kind]["invalid" if not d.valid else "valid"] += 1
    counts = {k: dict(v) for k, v in counts.items()}

    summary: dict[str, Any] = {
        "parent_build_id": old_build_id, "reason": cfg.reason, "alignment": align_summary,
        "added_spans_without_old_counterpart": len(added), "counts": counts,
        "totals": dict(Counter(d.status for d in decisions.values())),
        "invalid_items": sum(1 for d in decisions.values() if not d.valid),
        "cost": cost, "timings_s": timings, "lost_span_policy": cfg.lost_span_policy,
        "lexical_only": cfg.lexical_only, "verifier": getattr(verifier, "model", None),
        "n_old_spans": len(old_recs), "n_new_spans": len(new_recs), "books": book_ids}
    if cfg.dry_run:
        return summary

    # ---- 4. write the new immutable version
    build_id = cfg.build_id or f"{old_build_id}+rev-{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    summary["build_id"] = build_id
    t = time.time()
    info = _persist(cfg, build_id, old_build_id, items, copies, decisions, rows, old_recs, new_recs, old_vecs, embedder,
                    summary)
    summary.update(info)
    timings["persist"] = round(time.time() - t, 2)
    timings["total"] = round(time.time() - t0, 2)
    # ---- 5. changelog + alignment export
    rdir = Path(cfg.report_dir) if cfg.report_dir else out.parent
    rdir.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    summary["changelog"] = str(rdir / f"{stem}.changelog.jsonl")
    _write_jsonl(rdir / f"{stem}.changelog.jsonl", _changelog(decisions, rows, build_id, old_build_id, cfg.reason))
    _write_jsonl(rdir / f"{stem}.alignment.jsonl", (r.as_dict() for r in rows.values() if r.status not in ("identical",)))
    (rdir / f"{stem}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
    return summary


def _changelog(decisions: dict, rows: dict[str, AlignRow], build_id: str, parent: str, reason: str):
    for (kind, iid), d in sorted(decisions.items()):
        if d.status == "unchanged" and d.valid:
            continue
        yield {"build_id": build_id, "parent_build_id": parent, "reason": reason, "item_kind": kind, "item_id": iid,
               "status": d.status, "valid": d.valid, "support_old": d.support_old, "support_new": d.support_new,
               "why": d.reason, "rechecked": d.rechecked, "support_score": d.score, "span_counts": d.counts,
               "spans": [{"old": s, "new": rows[s].new_ids if s in rows else [],
                          "align": rows[s].status if s in rows else "missing", "jaccard": round(rows[s].score, 3) if s in rows else None}
                         for s in d.old_span_ids],
               "lost_span_ids": d.lost_span_ids, "notes": d.notes}


def _write_jsonl(path: Path, rows) -> int:
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    tmp.replace(path)
    return n


# ------------------------------------------------------------------------------------------ persistence
def _persist(cfg: RevalConfig, build_id: str, parent_id: str, items, copies, decisions: dict, rows: dict[str, AlignRow],
             old_recs: list[SpanRec], new_recs: list[SpanRec], old_vecs, embedder, summary) -> dict[str, Any]:
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".building")
    for suf in ("", "-wal", "-shm", "-journal"):
        Path(str(tmp) + suf).unlink(missing_ok=True)
    src = sqlite3.connect(f"file:{cfg.old_db}?mode=ro", uri=True, timeout=60)
    dst = sqlite3.connect(str(tmp))
    src.backup(dst)                                           # the old file is only read
    src.close()
    dst.execute("PRAGMA journal_mode=DELETE")
    dst.executescript(MAINTENANCE_DDL)
    have = {r[1] for r in dst.execute("PRAGMA table_info(build_versions)")}
    for col in ("parent_build_id", "reason"):
        if col not in have:
            dst.execute(f"ALTER TABLE build_versions ADD COLUMN {col} TEXT")
    dst.commit()

    # --- items
    n_updated = 0
    for key, d in decisions.items():
        kind, iid = key
        table, pk, _ = TABLES[kind]
        obj = items[key].model_copy(deep=True)
        changed_ids = d.new_span_ids != d.old_span_ids
        tag = f"maintenance:{d.status}" + ("" if d.valid else ":invalid")
        if d.status == "unchanged" and d.valid and not changed_ids:
            continue
        v = getattr(obj, "validation", None)
        if d.status in ("unchanged", "modified") and d.new_span_ids:
            obj.source_span_ids = list(d.new_span_ids)             # needs_review / orphaned keep the old ids
        if v is not None:
            if d.rechecked or d.status == "orphaned" or not d.valid:
                v.source_support = d.support_new if d.support_new in ("supported", "partially_supported", "unsupported", "needs_review") else v.source_support
            if d.score is not None:
                v.support_score = d.score
            if tag not in v.notes and d.status != "unchanged":
                v.notes.append(tag)
            if kind == "rule" and (not d.valid or d.support_new != "supported" or d.status == "needs_review"):
                if "excluded_from_strict_proof" not in v.notes and d.status != "unchanged":
                    v.notes.append("excluded_from_strict_proof")
        sets, args = ["json=?"], [obj.model_dump_json()]
        if kind in ("concept", "relation", "fact", "rule") and v is not None:
            sets.append("source_support=?")
            args.append(v.source_support)
        if kind == "rule":
            sets.append("strict_executable=?")
            args.append(int(bool(d.valid and d.status != "needs_review" and is_strict_executable(obj))))
        dst.execute(f"UPDATE {table} SET {', '.join(sets)} WHERE {pk}=?", [*args, iid])
        if d.status in ("unchanged", "modified") and changed_ids and d.new_span_ids:
            dst.execute("DELETE FROM evidence_links WHERE item_kind=? AND item_id=?", (kind, iid))
            dst.executemany("INSERT OR IGNORE INTO evidence_links VALUES(?,?,?)", [(kind, iid, s) for s in d.new_span_ids])
        n_updated += 1
    # pedagogical links: remap cited spans only
    for lid, js in list(dst.execute("SELECT link_id, json FROM pedagogical_links")):
        link = PedagogicalLink.model_validate_json(js)
        new_ids: list[str] = []
        for s in link.source_span_ids:
            r = rows.get(s)
            if r is not None and span_category(r.status) in ("same", "changed"):
                new_ids += [n for n in r.new_ids if n not in new_ids]
        if new_ids and new_ids != link.source_span_ids:
            link.source_span_ids = new_ids
            dst.execute("UPDATE pedagogical_links SET json=? WHERE link_id=?", (link.model_dump_json(), lid))
            dst.execute("DELETE FROM evidence_links WHERE item_kind='link' AND item_id=?", (lid,))
            dst.executemany("INSERT OR IGNORE INTO evidence_links VALUES('link',?,?)", [(lid, s) for s in new_ids])
    dst.executemany("INSERT OR REPLACE INTO item_status VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", [
        (k, i, d.status, int(d.valid), d.support_old, d.support_new, d.reason, len(d.old_span_ids), d.counts["same"],
         d.counts["changed"], d.counts["gone"], d.counts["unresolved"], int(d.rechecked))
        for (k, i), d in decisions.items()])
    dst.executemany("INSERT OR REPLACE INTO span_alignment VALUES(?,?,?,?,?,?,?)", [
        (r.old_id, json.dumps(r.new_ids), r.status, r.method, r.score, r.cosine, r.note) for r in rows.values()])
    dst.commit()

    # --- span vectors of the NEW edition
    emb_info: dict[str, Any] = {"span_vectors": 0, "span_vectors_reused": 0, "span_vectors_missing": 0}
    model = getattr(embedder, "model", None) or next(iter({r[0] for r in dst.execute(
        "SELECT DISTINCT model FROM embeddings WHERE kind='span'")}), None)
    if cfg.embed and model:
        reuse: dict[str, np.ndarray] = {}
        for r in rows.values():
            if r.status in ("identical", "identical_norm", "moved") and r.new_ids and r.old_id in old_vecs:
                reuse.setdefault(r.new_ids[0], old_vecs[r.old_id])
        todo = [s for s in new_recs if s.span_id not in reuse]
        vecs: dict[str, np.ndarray] = dict(reuse)
        if todo and embedder is not None and embedder.available():
            for i in range(0, len(todo), 256):
                ch = todo[i:i + 256]
                m = embedder.embed([s.text[:6000] if s.text.strip() else " " for s in ch])
                for s, v in zip(ch, m):
                    vecs[s.span_id] = np.asarray(v, dtype=np.float32)
        dst.execute("DELETE FROM embeddings WHERE kind='span'")
        dst.executemany("INSERT OR REPLACE INTO embeddings VALUES('span',?,?,?,NULL,?,?)",
                        [(s.span_id, model, s.book_id, len(vecs[s.span_id]), to_blob(vecs[s.span_id]))
                         for s in new_recs if s.span_id in vecs])
        emb_info.update(span_vectors=len(vecs), span_vectors_reused=len(reuse),
                        span_vectors_missing=len(new_recs) - len(vecs))
    elif cfg.embed:
        emb_info["note"] = "no embedding model known; span vectors not written"
    dst.commit()

    # --- self-contained provenance of the NEW edition
    if cfg.embed_provenance:
        prov_src = cfg.old_prov
        if prov_src is None:
            from otter.llm.interface import CONFIG_PATH, load_config
            prov_src = CONFIG_PATH.parents[1] / load_config()["paths"]["db"]
        ps = Path(prov_src)
        _write_provenance(dst, new_recs, ps if ps.is_file() and ps.suffix not in (".jsonl", ".json") else None)

    # --- version row
    stats = {k: summary[k] for k in ("alignment", "counts", "totals", "invalid_items", "cost", "added_spans_without_old_counterpart")}
    dst.execute("INSERT OR REPLACE INTO build_versions(build_id, created_at, extractor, verifier, embedding_model, config_json,"
                " stats_json, notes, parent_build_id, reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (build_id, _dt.datetime.now().astimezone().isoformat(timespec="microseconds"),
                 f"maintenance(revalidate) of {parent_id}", summary.get("verifier") or ("lexical" if cfg.lexical_only else ""),
                 model or "", json.dumps({"old_db": str(cfg.old_db), "new_spans": str(cfg.new_spans),
                                          "lost_span_policy": cfg.lost_span_policy, "lexical_only": cfg.lexical_only,
                                          "lexical_shortcut": cfg.lexical_shortcut, "align": cfg.align.__dict__}),
                 json.dumps(stats, ensure_ascii=False, default=str), "knowledge-maintenance version; see item_status",
                 parent_id, cfg.reason))
    dst.commit()
    dst.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    dst.close()
    for suf in ("-wal", "-shm"):
        Path(str(out) + suf).unlink(missing_ok=True)
    os.replace(tmp, out)
    emb_info["items_rewritten"] = n_updated
    emb_info["db"] = str(out)
    return emb_info


def _write_provenance(dst: sqlite3.Connection, new_recs: list[SpanRec], books_src: Optional[Path]) -> None:
    dst.executescript(SPAN_DDL)
    have = {r[1] for r in dst.execute("PRAGMA table_info(spans)")}
    for col in SPAN_COLS:
        if col not in have:
            dst.execute(f"ALTER TABLE spans ADD COLUMN {col} {'REAL' if col == 'page_quality_score' else 'TEXT'}")
    dst.execute("DELETE FROM spans")                         # the new file describes the NEW edition only
    dst.execute("DELETE FROM spans_fts")
    book_ids = sorted({s.book_id for s in new_recs})
    if books_src is not None and Path(books_src).exists():
        sc = _connect_ro(Path(books_src))
        names = {r[0] for r in sc.execute("SELECT name FROM sqlite_master")}
        ph = ",".join("?" * len(book_ids))
        if "books" in names and book_ids:
            dst.executemany("INSERT OR REPLACE INTO books VALUES(?,?,?)",
                            [tuple(r) for r in sc.execute(f"SELECT book_id, metadata_json, source_sha256 FROM books "
                                                          f"WHERE book_id IN ({ph})", book_ids)])
        if "sections" in names and book_ids:
            cols = [r[1] for r in dst.execute("PRAGMA table_info(sections)")]
            have = {r[1] for r in sc.execute("PRAGMA table_info(sections)")}
            use = [c for c in cols if c in have]
            q = f"SELECT {','.join(use)} FROM sections WHERE book_id IN ({ph})"
            dst.executemany(f"INSERT OR REPLACE INTO sections({','.join(use)}) VALUES({','.join('?' * len(use))})",
                            [tuple(r) for r in sc.execute(q, book_ids)])
        sc.close()
    else:
        for b in book_ids:
            dst.execute("INSERT OR IGNORE INTO books VALUES(?,?,NULL)", (b, json.dumps({"book_id": b})))
    rows = []
    for s in new_recs:
        r = s.raw
        hp = r.get("heading_path_json")
        if hp is None:
            hp = json.dumps(r.get("heading_path") or list(s.heading_path), ensure_ascii=False)
        page = r.get("pdf_page_index")
        rows.append((s.span_id, s.book_id, page, r.get("source_page_index") if r.get("source_page_index") is not None else (page or 0),
                     r.get("printed_page_label"), int(bool(r.get("printed_page_verified"))), r.get("section_id"), s.text,
                     r.get("source_sha256") or r.get("source_file_sha256") or "", r.get("start_offset"), r.get("end_offset"),
                     hp, r.get("pedagogical_type"), r.get("extractor_version"), r.get("evidence_status"), r.get("span_kind"),
                     r.get("pedagogical_label_vi"), r.get("chapter_id"), r.get("lesson_id"), r.get("page_span_id"),
                     r.get("page_quality_score"), int(bool(r.get("requires_manual_review"))), s.sha1))
    dst.executemany(f"INSERT OR REPLACE INTO spans({','.join(SPAN_COLS)}) VALUES({','.join('?' * len(SPAN_COLS))})", rows)
    dst.executemany("INSERT INTO spans_fts VALUES(?,?,?)", [(s.span_id, s.book_id, s.text) for s in new_recs])
    dst.commit()
