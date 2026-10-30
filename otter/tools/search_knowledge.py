"""`search_knowledge`: typed concept / fact / rule candidates from the ontology (graph channel)."""
from __future__ import annotations

from typing import Any, Optional

from otter.tools.base import concept_view, fact_view, rule_view

VALID_KINDS = ("concept", "fact", "rule")


def search_knowledge(ctx: Any, query: str, subject: Optional[str] = None, top_k: int = 8,
                     kinds: Optional[list[str]] = None, lesson_scope: Optional[str] = None) -> dict[str, Any]:
    if not query or not str(query).strip():
        return {"status": "error", "reason": "empty_query", "hits": []}
    notes = []
    if ctx.subject and subject and subject != ctx.subject:
        notes.append(f"subject_scope_enforced({subject}->{ctx.subject})")
    subject = ctx.subject or subject
    ks = tuple(k for k in (kinds or ctx.kinds) if k in VALID_KINDS and k in ctx.kinds) or tuple(ctx.kinds)
    top_k = max(1, min(int(top_k or 8), ctx.top_k_cap))
    raw = ctx.store.search_knowledge(str(query), subject=subject, grade=ctx.grade, kinds=ks, top_k=top_k * 2)
    hits, invalid = [], []
    for h in raw:
        kind, hid = h.get("kind"), h.get("id")
        obj = None
        if kind == "concept":
            obj = ctx.store.get_concept(hid)
            view = concept_view(obj) if obj else None
        elif kind == "fact":
            obj = ctx.store.get_fact(hid)
            view = fact_view(obj) if obj else None
        elif kind == "rule":
            obj = ctx.store.get_rule(hid)
            view = rule_view(obj) if obj else None
        else:
            view = None
        if view is None:
            invalid.append(hid)
            continue
        if subject and view.get("subject") and view["subject"] != subject:
            invalid.append(hid)          # out of the authorized scope
            continue
        if lesson_scope and view.get("source_span_ids"):
            spans = ctx.store.get_spans(view["source_span_ids"])
            if not any((s.section_id or "").startswith(lesson_scope) for s in spans):
                continue
        view["score"] = round(float(h.get("score") or 0.0), 4)
        hits.append(view)
        if len(hits) >= top_k:
            break
    return {"status": "ok" if hits else "no_results", "query": query, "subject": subject, "hits": hits,
            "invalid_ids": invalid, "notes": notes}
