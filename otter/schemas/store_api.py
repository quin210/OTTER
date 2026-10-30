"""Read-only store API used by the online runtime (WS3 tools) and baselines.

WS2 implements it in `otter/storage/sqlite_store.py`; WS1 owns the provenance
tables (books, sections, spans, spans_fts) it reads from. Runtime code must not
mutate the canonical ontology.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from otter.schemas.ontology import Concept, ConditionalRule, Fact, SemanticRelation
from otter.schemas.reasoning import FinalAnswer
from otter.schemas.textbook import SourceSpan


class OntologyReader(Protocol):
    # --- retrieval -------------------------------------------------------
    def search_knowledge(self, query: str, *, subject: Optional[str] = None, grade: Optional[int] = None,
                         kinds: tuple[str, ...] = ("concept", "fact", "rule"), top_k: int = 10) -> list[dict[str, Any]]:
        """Hybrid (FTS/BM25 + bge-m3 dense) search; each hit = {"kind","id","label_or_text","score","subject","source_span_ids"}."""

    def search_spans(self, query: str, *, subject: Optional[str] = None, grade: Optional[int] = None,
                     book_ids: Optional[list[str]] = None, top_k: int = 10) -> list[tuple[SourceSpan, float]]: ...

    # --- typed access ----------------------------------------------------
    def get_concept(self, concept_id: str) -> Optional[Concept]: ...
    def get_fact(self, assertion_id: str) -> Optional[Fact]: ...
    def get_rule(self, rule_id: str) -> Optional[ConditionalRule]: ...
    def rules_concluding(self, predicate: str, *, subject: Optional[str] = None) -> list[ConditionalRule]: ...
    def neighbors(self, node_ids: list[str], *, relation_types: Optional[list[str]] = None, max_hops: int = 1,
                  subject: Optional[str] = None) -> dict[str, Any]:
        """{"nodes": [Concept...], "relations": [SemanticRelation...], "rules": [ConditionalRule...]}"""

    # --- provenance ------------------------------------------------------
    def get_spans(self, span_ids: list[str]) -> list[SourceSpan]:
        """Only existing spans are returned; callers treat missing ids as invalid citations."""
    def span_exists(self, span_id: str) -> bool: ...
    def book_metadata(self, book_id: str) -> Optional[dict[str, Any]]: ...
    def section(self, section_id: str) -> Optional[dict[str, Any]]: ...


class QASystem(Protocol):
    """Every system (OTTER, internal baselines, external graph-RAG adapters) implements this.

    `item` is one benchmark row: {"id", "question", "options": {"A":..,"B":..,"C":..,"D":..}, "subject",
    "domain", "grade", ...}. Must never raise: on error return FinalAnswer(status="failed", error=...).
    """
    name: str

    def answer(self, item: dict[str, Any]) -> FinalAnswer: ...


__all__ = ["OntologyReader", "QASystem", "Concept", "ConditionalRule", "Fact", "SemanticRelation", "SourceSpan"]
