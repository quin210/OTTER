"""Builder step 6: materialise the validated ontology.

* SQLite ontology DB (WS2 tables, FTS5 `knowledge_fts`, bge-m3 vectors) written to a temp file, switched
  to rollback-journal mode and atomically renamed over the target (readers keep the old inode). The
  provenance DB (WS1) is never written.
* JSONL exports (`concepts/relations/facts/rules/objectives/links/rejected/merges.jsonl`) + networkx
  node-link graph (`graph.json`) in the export dir.
* Embedding cache (sqlite, keyed by model + sha256(text)) so incremental rebuilds only embed new texts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np

from otter.builder.normalize import Ontology
from otter.schemas.textbook import SourceSpan
from otter.storage.embeddings import from_blob, to_blob
from otter.storage.writer import OntologyWriter, concept_text, rule_text

log = logging.getLogger("otter.builder.persist")


class EmbeddingCache:
    """Wraps an embedder with a persistent text->vector cache."""

    def __init__(self, embedder, path: Optional[Path]):
        self.embedder, self.model = embedder, getattr(embedder, "model", "unknown")
        self.path = path
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(path)) as db:
                db.execute("CREATE TABLE IF NOT EXISTS emb(model TEXT, h TEXT, vec BLOB, PRIMARY KEY(model, h))")

    def available(self) -> bool:
        return self.embedder is not None and self.embedder.available()

    def embed(self, texts: list[str]) -> np.ndarray:
        hs = [hashlib.sha256((t or " ").encode("utf-8")).hexdigest() for t in texts]
        got: dict[str, np.ndarray] = {}
        if self.path:
            with self._lock, sqlite3.connect(str(self.path)) as db:
                uniq = list(dict.fromkeys(hs))
                for i in range(0, len(uniq), 500):
                    ch = uniq[i:i + 500]
                    for h, b in db.execute(f"SELECT h, vec FROM emb WHERE model=? AND h IN ({','.join('?' * len(ch))})",
                                           [self.model, *ch]):
                        got[h] = from_blob(b)
        todo = list(dict.fromkeys(h for h in hs if h not in got))
        self.hits += len(got)
        if todo:
            first = dict(zip(hs, texts))
            vecs = self.embedder.embed([first[h] for h in todo])
            self.misses += len(todo)
            for h, v in zip(todo, vecs):
                got[h] = np.asarray(v, dtype=np.float32)
            if self.path:
                with self._lock, sqlite3.connect(str(self.path)) as db:
                    db.executemany("INSERT OR REPLACE INTO emb VALUES(?,?,?)",
                                   [(self.model, h, to_blob(got[h])) for h in todo])
        if not hs:
            return np.zeros((0, 1024), dtype=np.float32)
        return np.stack([got[h] for h in hs]).astype(np.float32)


def relation_text(onto: Ontology, r) -> str:
    s, o = onto.concepts.get(r.subject_id), onto.concepts.get(r.object_id)
    t = f"{s.label if s else r.subject_id} {r.relation_type} {o.label if o else r.object_id}"
    return t + (f" ({r.qualifier})" if r.qualifier else "")


def item_texts(onto: Ontology) -> dict[str, list[tuple[str, str, Optional[int], str]]]:
    """kind -> [(item_id, subject, grade, text)] used for FTS and dense vectors."""
    out: dict[str, list] = {"concept": [], "fact": [], "rule": [], "relation": [], "objective": []}
    for cid, c in onto.concepts.items():
        out["concept"].append((cid, c.subject, c.grade, concept_text(c)))
    for fid, f in onto.facts.items():
        out["fact"].append((fid, f.subject, onto.grade.get(fid), f.text))
    for rid, r in onto.rules.items():
        out["rule"].append((rid, r.subject, onto.grade.get(rid), rule_text(r)))
    for rid, r in onto.relations.items():
        s = onto.concepts.get(r.subject_id)
        out["relation"].append((rid, s.subject if s else "other", onto.grade.get(rid), relation_text(onto, r)))
    for oid, o in onto.objectives.items():
        out["objective"].append((oid, o.subject, o.grade, o.text))
    return out


def _dump(path: Path, rows) -> int:
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    tmp.replace(path)
    return n


def to_networkx(onto: Ontology) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for cid, c in onto.concepts.items():
        g.add_node(cid, kind="concept", label=c.label, subject=c.subject, concept_type=c.concept_type, grade=c.grade)
    for rid, r in onto.relations.items():
        g.add_edge(r.subject_id, r.object_id, key=rid, kind="relation", relation_type=r.relation_type,
                   status=r.status, support=r.validation.source_support)
    for rid, r in onto.rules.items():
        g.add_node(rid, kind="rule", subject=r.subject, rule_type=r.rule_type, support=r.validation.source_support)
        for role, atoms in (("premise", r.premises), ("guard", r.guards), ("conclusion", r.conclusions),
                            ("exception", r.exceptions)):
            for a in atoms:
                pid = f"pred:{r.subject}:{a.predicate}"
                if pid not in g:
                    g.add_node(pid, kind="predicate", subject=r.subject, label=a.predicate)
                if role == "conclusion":
                    g.add_edge(rid, pid, key=f"{rid}:{role}:{a.predicate}", kind=role, negated=a.negated)
                else:
                    g.add_edge(pid, rid, key=f"{rid}:{role}:{a.predicate}", kind=role, negated=a.negated)
                for x in a.args:
                    if x in onto.concepts:
                        g.add_edge(rid, x, key=f"{rid}:mentions:{x}", kind="mentions")
    for fid, f in onto.facts.items():
        g.add_node(fid, kind="fact", subject=f.subject, support=f.validation.source_support)
        for c in f.concept_ids:
            if c in onto.concepts:
                g.add_edge(fid, c, key=f"{fid}:about:{c}", kind="about")
    for oid, o in onto.objectives.items():
        g.add_node(oid, kind="objective", subject=o.subject)
    for lid, l in onto.links.items():
        if l.source_id in g and l.target_id in g:
            g.add_edge(l.source_id, l.target_id, key=lid, kind=l.link_type)
    return g


def persist(onto: Ontology, out_path: Path, *, build_id: str, extractor: str, verifier: str,
            embedder=None, embed_spans: bool = True, spans: Optional[dict[str, SourceSpan]] = None,
            span_books: Optional[dict[str, str]] = None, config: Optional[dict] = None,
            stats: Optional[dict] = None, export_dir: Optional[Path] = None, notes: str = "") -> dict[str, Any]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".building")
    for suf in ("", "-wal", "-shm", "-journal"):
        Path(str(tmp) + suf).unlink(missing_ok=True)
    w = OntologyWriter(tmp)
    for c in onto.concepts.values():
        w.add_concept(c)
    texts = item_texts(onto)
    rel_text = {iid: t for iid, _, _, t in texts["relation"]}
    for rid, r in onto.relations.items():
        s = onto.concepts.get(r.subject_id)
        w.add_relation(r, subject=s.subject if s else "other", grade=onto.grade.get(rid), text=rel_text[rid])
    for fid, f in onto.facts.items():
        w.add_fact(f, grade=onto.grade.get(fid))
    for rid, r in onto.rules.items():
        w.add_rule(r, grade=onto.grade.get(rid))
    for o in onto.objectives.values():
        w.add_objective(o)
    for l in onto.links.values():
        w.add_link(l)
    for rj in onto.rejected:
        item = rj.get("item") or {}
        key = str(item.get("label") or item.get("text") or item.get("subject") or "")[:200] if isinstance(item, dict) else ""
        w.add_rejected(rj.get("kind", "?"), key, rj.get("reason", "?"), rj, rj.get("chunk_id"))
    w.commit()
    emb_info: dict[str, Any] = {"model": None, "vectors": {}}
    if embedder is not None and embedder.available():
        model = embedder.model
        emb_info["model"] = model
        for kind, rows in texts.items():
            if not rows:
                continue
            vecs = embedder.embed([t for *_, t in rows])
            w.add_embeddings(kind, model, [(iid, subj, grade) for iid, subj, grade, _ in rows], vecs)
            emb_info["vectors"][kind] = len(rows)
            w.commit()
        if embed_spans and spans:
            ids = sorted(spans)
            for i in range(0, len(ids), 4096):
                ch = ids[i:i + 4096]
                vecs = embedder.embed([spans[s].text for s in ch])
                # for kind=span the `subject` column holds the book_id (see SqliteOntologyStore._dense_rank)
                w.add_embeddings("span", model, [(s, spans[s].book_id, None) for s in ch], vecs)
                w.commit()
            emb_info["vectors"]["span"] = len(ids)
    w.add_build_version(build_id, extractor=extractor, verifier=verifier, embedding_model=emb_info["model"] or "",
                        config=config, stats=stats, notes=notes)
    w.commit()
    w.db.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('optimize')")
    w.commit()
    w.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    w.db.execute("PRAGMA journal_mode=DELETE")
    w.close()
    for suf in ("-wal", "-shm"):
        Path(str(out_path) + suf).unlink(missing_ok=True)
    os.replace(tmp, out_path)
    log.info("ontology written to %s", out_path)

    exports: dict[str, int] = {}
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        dump = lambda d: (json.loads(o.model_dump_json()) | {"_grade": onto.grade.get(k),
                                                            "_books": sorted(onto.item_books.get(k, []))}
                          for k, o in sorted(d.items()))
        exports["concepts"] = _dump(export_dir / "concepts.jsonl", dump(onto.concepts))
        exports["relations"] = _dump(export_dir / "relations.jsonl", dump(onto.relations))
        exports["facts"] = _dump(export_dir / "facts.jsonl", dump(onto.facts))
        exports["rules"] = _dump(export_dir / "rules.jsonl", dump(onto.rules))
        exports["objectives"] = _dump(export_dir / "objectives.jsonl", dump(onto.objectives))
        exports["links"] = _dump(export_dir / "links.jsonl", (json.loads(l.model_dump_json()) for l in onto.links.values()))
        exports["rejected"] = _dump(export_dir / "rejected.jsonl", onto.rejected)
        exports["merges"] = _dump(export_dir / "merges.jsonl", onto.merges)
        g = to_networkx(onto)
        try:
            data = nx.node_link_data(g, edges="links")
        except TypeError:
            data = nx.node_link_data(g)
        tmpg = export_dir / "graph.json.tmp"
        tmpg.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        tmpg.replace(export_dir / "graph.json")
        exports["graph_nodes"], exports["graph_edges"] = g.number_of_nodes(), g.number_of_edges()
    return {"db": str(out_path), "embeddings": emb_info, "exports": exports}
