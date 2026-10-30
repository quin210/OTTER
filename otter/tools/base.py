"""Shared tool context and serialization helpers."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from otter.reasoning.text import truncate


@dataclass
class ToolContext:
    """Everything a tool may read. One instance per question (cheap to build).

    `store` is an `OntologyReader` (read-only). `judge_llm` is an optional (metered) LLM used
    only by `verify_evidence` for the disclosed, heuristic semantic-support check.
    """
    store: Any
    subject: Optional[str] = None
    grade: Optional[int] = None
    judge_llm: Any = None
    workspace: Any = None
    disabled: frozenset = frozenset()
    kinds: tuple = ("concept", "fact", "rule")
    support_threshold: float = 0.5
    executable_support: frozenset = frozenset({"supported"})
    max_hops: int = 3
    top_k_cap: int = 20
    extra: dict = field(default_factory=dict)


class LockedStore:
    """Serializes calls to a store whose connection may not be thread-safe."""

    def __init__(self, store: Any):
        self._store = store
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **kw):
            with self._lock:
                return attr(*a, **kw)
        return wrapped


def dump(obj: Any) -> Any:
    """pydantic model / list / dict -> JSON-able structure."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", by_alias=True)
    if isinstance(obj, (list, tuple)):
        return [dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    return obj


def atom_view(a: Any) -> dict[str, Any]:
    return {"predicate": a.predicate, "args": list(a.args), "negated": a.negated, "text": a.text}


def rule_view(r: Any, *, full: bool = True) -> dict[str, Any]:
    v = {"kind": "rule", "id": r.rule_id, "subject": r.subject, "rule_type": r.rule_type,
         "premises": [atom_view(a) for a in r.premises], "guards": [atom_view(a) for a in r.guards],
         "conclusions": [atom_view(a) for a in r.conclusions],
         "source_span_ids": list(r.source_span_ids), "source_support": r.validation.source_support}
    if r.exceptions:
        v["exceptions"] = [atom_view(a) for a in r.exceptions]
    if r.formula:
        v["formula"] = r.formula
        v["variables"] = dict(r.variables)
    return v


def concept_view(c: Any) -> dict[str, Any]:
    return {"kind": "concept", "id": c.concept_id, "label": c.label, "aliases": list(c.aliases),
            "type": c.concept_type, "subject": c.subject, "definition": truncate(c.definition or "", 300) or None,
            "source_span_ids": list(c.source_span_ids)}


def fact_view(f: Any) -> dict[str, Any]:
    return {"kind": "fact", "id": f.assertion_id, "text": truncate(f.text, 400), "subject": f.subject,
            "concept_ids": list(f.concept_ids), "source_span_ids": list(f.source_span_ids),
            "source_support": f.validation.source_support}


def relation_view(r: Any) -> dict[str, Any]:
    return {"kind": "relation", "id": r.relation_id, "subject_id": r.subject_id, "relation_type": r.relation_type,
            "object_id": r.object_id, "qualifier": r.qualifier, "source_span_ids": list(r.source_span_ids)}


def span_view(s: Any, n: int = 600) -> dict[str, Any]:
    return {"span_id": s.span_id, "book_id": s.book_id, "section_id": s.section_id,
            "heading_path": list(s.heading_path)[-3:], "text": truncate(s.text, n),
            "pdf_page_index": s.pdf_page_index,
            "printed_page_label": s.printed_page_label if s.printed_page_verified else None,
            "evidence_status": s.evidence_status}
