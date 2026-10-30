"""`retrieve_source`: verbatim spans + true metadata (text channel / provenance)."""
from __future__ import annotations

from typing import Any, Optional

from otter.reasoning.citation_resolver import CitationResolver
from otter.tools.base import span_view


def retrieve_source(ctx: Any, span_ids: Optional[list[str]] = None, query: Optional[str] = None,
                    book_ids: Optional[list[str]] = None, top_k: int = 5) -> dict[str, Any]:
    if isinstance(span_ids, str):
        span_ids = [span_ids]
    span_ids = [s for s in (span_ids or []) if s]
    if not span_ids and not (query and str(query).strip()):
        return {"status": "error", "reason": "need_span_ids_or_query", "spans": []}
    resolver = ctx.extra.get("citation_resolver") or CitationResolver(ctx.store)
    out_spans, missing = [], []
    if span_ids:
        found = {s.span_id: s for s in ctx.store.get_spans(span_ids)}
        missing = [s for s in span_ids if s not in found]
        spans = [(found[s], None) for s in span_ids if s in found]
    else:
        top_k = max(1, min(int(top_k or 5), ctx.top_k_cap))
        spans = ctx.store.search_spans(str(query), subject=ctx.subject, grade=ctx.grade, book_ids=book_ids, top_k=top_k)
    cites, _ = resolver.resolve_many([s.span_id for s, _ in spans], with_quote=False)
    cmap = {c.span_id: c for c in cites}
    for s, score in spans:
        v = span_view(s)
        c = cmap.get(s.span_id)
        if c is not None:
            v["citation"] = {"book_title": c.book_title, "grade": c.grade, "lesson_title": c.lesson_title,
                             "page_number": c.page_number, "pdf_page_number": c.pdf_page_number}
        if score is not None:
            v["score"] = round(float(score), 4)
        out_spans.append(v)
    status = "ok" if out_spans and not missing else ("partial" if out_spans else ("invalid_span_ids" if missing else "no_results"))
    return {"status": status, "spans": out_spans, "missing_span_ids": missing, "mode": "ids" if span_ids else "query"}
