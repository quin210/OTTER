"""Write side of the ontology store (used by the offline builder and the fixture builder only).

Runtime code must use the read-only `SqliteOntologyStore`.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from otter.schemas.ontology import (Atom, Concept, ConditionalRule, Fact, LearningObjective, PedagogicalLink,
                                    SemanticRelation)
from otter.storage.embeddings import to_blob
from otter.storage.schema import ONTOLOGY_DDL
from otter.storage.textutil import norm_key


def is_strict_executable(rule: ConditionalRule) -> bool:
    """Policy for strict (formal) proof execution.

    A rule is executable in strict mode only if its cited spans exist, the separate support check
    judged it `supported`, it is an explicit textbook rule, it has premises and conclusions, and the
    validator did not flag a suspected missing guard. Everything else stays in the store (searchable,
    usable for source-supported evidence chains) but must not license formal inferences.
    """
    v = rule.validation
    return bool(v.schema_valid and v.source_ids_exist and v.source_support == "supported"
                and rule.status == "explicit_rule" and rule.premises and rule.conclusions
                and "possible_missing_guard" not in v.notes and "excluded_from_strict_proof" not in v.notes)


def rule_text(rule: ConditionalRule) -> str:
    def j(atoms: list[Atom]) -> str:
        return "; ".join((("không " if a.negated else "") + (a.text or a.predicate)) for a in atoms)
    s = f"Nếu {j(rule.premises)}"
    if rule.guards:
        s += f" (điều kiện: {j(rule.guards)})"
    s += f" thì {j(rule.conclusions)}"
    if rule.exceptions:
        s += f" (ngoại lệ: {j(rule.exceptions)})"
    if rule.formula:
        s += f". Công thức: {rule.formula}"
    if rule.variables:
        s += " (" + ", ".join(f"{k}: {v}" for k, v in rule.variables.items()) + ")"
    return s


def concept_text(c: Concept) -> str:
    s = c.label
    if c.aliases:
        s += " (" + ", ".join(c.aliases) + ")"
    if c.definition:
        s += ": " + c.definition
    return s


class OntologyWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(ONTOLOGY_DDL)

    # ------------------------------------------------------------------ items
    def _evidence(self, kind: str, item_id: str, span_ids: Iterable[str]) -> None:
        self.db.executemany("INSERT OR IGNORE INTO evidence_links VALUES(?,?,?)",
                            [(kind, item_id, s) for s in span_ids])

    def _fts(self, kind: str, item_id: str, subject: str, grade: Optional[int], text: str) -> None:
        self.db.execute("INSERT INTO knowledge_fts(kind,item_id,subject,grade,text) VALUES(?,?,?,?,?)",
                        (kind, item_id, subject, grade, text))

    def add_concept(self, c: Concept) -> None:
        self.db.execute("INSERT OR REPLACE INTO concepts VALUES(?,?,?,?,?,?,?,?,?)",
                        (c.concept_id, c.label, norm_key(c.label), c.concept_type, c.subject, c.grade, c.definition,
                         c.validation.source_support, c.model_dump_json()))
        rows = {(c.concept_id, a, norm_key(a), c.subject) for a in [c.label, *c.aliases] if norm_key(a)}
        self.db.executemany("INSERT OR IGNORE INTO concept_aliases VALUES(?,?,?,?)", list(rows))
        self._evidence("concept", c.concept_id, c.source_span_ids)
        self._fts("concept", c.concept_id, c.subject, c.grade, concept_text(c))

    def add_relation(self, r: SemanticRelation, *, subject: str, grade: Optional[int] = None,
                     text: Optional[str] = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO semantic_relations VALUES(?,?,?,?,?,?,?,?,?)",
                        (r.relation_id, r.subject_id, r.relation_type, r.object_id, r.status, subject, grade,
                         r.validation.source_support, r.model_dump_json()))
        self._evidence("relation", r.relation_id, r.source_span_ids)
        if text:
            self._fts("relation", r.relation_id, subject, grade, text)

    def add_fact(self, f: Fact, *, grade: Optional[int] = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO facts VALUES(?,?,?,?,?,?,?)",
                        (f.assertion_id, f.text, f.subject, grade, f.status, f.validation.source_support,
                         f.model_dump_json()))
        self.db.executemany("INSERT OR IGNORE INTO fact_concepts VALUES(?,?)",
                            [(f.assertion_id, c) for c in f.concept_ids])
        self._evidence("fact", f.assertion_id, f.source_span_ids)
        self._fts("fact", f.assertion_id, f.subject, grade, f.text)

    def add_rule(self, r: ConditionalRule, *, grade: Optional[int] = None, concept_ids: Iterable[str] = ()) -> None:
        txt = rule_text(r)
        self.db.execute("INSERT OR REPLACE INTO rules VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (r.rule_id, r.subject, grade, r.rule_type, r.status, r.formula, txt,
                         r.validation.source_support, int(is_strict_executable(r)), r.model_dump_json()))
        for table, atoms in (("rule_premises", r.premises), ("rule_guards", r.guards),
                             ("rule_conclusions", r.conclusions), ("rule_exceptions", r.exceptions)):
            self.db.executemany(f"INSERT OR REPLACE INTO {table} VALUES(?,?,?,?,?,?)",
                                [(r.rule_id, i, a.predicate, json.dumps(a.args, ensure_ascii=False), a.text,
                                  int(a.negated)) for i, a in enumerate(atoms)])
        cids = set(concept_ids)
        for a in [*r.premises, *r.guards, *r.conclusions, *r.exceptions]:
            cids.update(x for x in a.args if ":c:" in x)
        self.db.executemany("INSERT OR IGNORE INTO rule_concepts VALUES(?,?)", [(r.rule_id, c) for c in cids])
        self._evidence("rule", r.rule_id, r.source_span_ids)
        self._fts("rule", r.rule_id, r.subject, grade, txt)

    def add_objective(self, o: LearningObjective) -> None:
        self.db.execute("INSERT OR REPLACE INTO objectives VALUES(?,?,?,?,?,?)",
                        (o.objective_id, o.text, o.lesson_section_id, o.subject, o.grade, o.model_dump_json()))
        self._evidence("objective", o.objective_id, o.source_span_ids)
        self._fts("objective", o.objective_id, o.subject, o.grade, o.text)

    def add_link(self, link: PedagogicalLink) -> None:
        self.db.execute("INSERT OR REPLACE INTO pedagogical_links VALUES(?,?,?,?,?)",
                        (link.link_id, link.source_id, link.link_type, link.target_id, link.model_dump_json()))
        self._evidence("link", link.link_id, link.source_span_ids)

    def add_rejected(self, kind: str, key: str, reason: str, payload: dict, chunk_id: Optional[str] = None) -> None:
        self.db.execute("INSERT INTO rejected_items VALUES(?,?,?,?,?)",
                        (kind, key, reason, chunk_id, json.dumps(payload, ensure_ascii=False, default=str)))

    def add_embeddings(self, kind: str, model: str, rows: list[tuple[str, Optional[str], Optional[int]]],
                       vecs: np.ndarray) -> None:
        assert len(rows) == len(vecs)
        dim = int(vecs.shape[1]) if len(vecs) else 0
        self.db.executemany("INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?,?,?,?)",
                            [(kind, iid, model, subj, grade, dim, to_blob(v)) for (iid, subj, grade), v in zip(rows, vecs)])

    def add_build_version(self, build_id: str, *, extractor: str = "", verifier: str = "", embedding_model: str = "",
                          config: Optional[dict] = None, stats: Optional[dict] = None, notes: str = "") -> None:
        self.db.execute("INSERT OR REPLACE INTO build_versions VALUES(?,?,?,?,?,?,?,?)",
                        (build_id, _dt.datetime.now().astimezone().isoformat(timespec="seconds"), extractor, verifier,
                         embedding_model, json.dumps(config or {}, ensure_ascii=False, default=str),
                         json.dumps(stats or {}, ensure_ascii=False, default=str), notes))

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()
