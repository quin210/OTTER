"""Source-location ("Truy xuất nguồn" / where is the theory in the textbook) metrics.

Predicted location of a system = `FinalAnswer.where_to_read[0]` if present, otherwise the top citation (first citation of the first
claim); all citations / where_to_read entries are the "any" pool. Citations carry span ids; book / lesson / page are resolved from the
span corpus (DB), never from the system's own text. Gold = the item's `gold_span_ids`.

Per-item record keys (None when the item has no gold or the system gave no location counts as 0, not None):
  loc_pred            1 if the system produced any location
  loc_span_hit1       top location span in gold
  loc_span_hit_any    any cited / where_to_read span in gold
  loc_lesson1         top location lesson (book, lesson_key) among gold lessons
  loc_book1           top location book among gold books
  loc_page0           top location (book, PDF page) equals a gold (book, PDF page)
  loc_page1           same with +-1 page tolerance
  loc_printed_exact   top location printed page label equals gold printed label (only items whose gold pages are verified; else absent)
  loc_theory          lesson correct AND PDF page exact: "names the correct lesson + page for the theory behind the answer"
"""
from __future__ import annotations

from typing import Any, Optional


def predicted_locations(fa) -> list:
    wr = list(getattr(fa, "where_to_read", None) or [])
    cites = [c for cl in fa.claims for c in cl.citations]
    return wr + [c for c in cites if c not in wr]


def _info(corpus, span_id):
    s = corpus.get(span_id)
    if s is None:
        return None
    return {"book": s.book_id, "lesson": (s.book_id, s.lesson_key), "page": (s.book_id, s.pdf_page_index),
            "printed": (s.book_id, str(s.printed_page_label).strip()) if s.printed_page_verified and s.printed_page_label else None}


def score(gold_span_ids: list[str], fa, corpus) -> dict[str, Any]:
    gold = [g for g in (_info(corpus, s) for s in gold_span_ids) if g]
    if not gold:
        return {}
    preds = predicted_locations(fa)
    out: dict[str, Any] = {"loc_pred": int(bool(preds))}
    gs = set(gold_span_ids)
    out["loc_span_hit_any"] = int(any(c.span_id in gs for c in preds))
    top = _info(corpus, preds[0].span_id) if preds else None
    out["loc_span_hit1"] = int(bool(preds) and preds[0].span_id in gs)
    gl, gb, gp = {g["lesson"] for g in gold}, {g["book"] for g in gold}, {g["page"] for g in gold}
    out["loc_lesson1"] = int(bool(top) and top["lesson"] in gl)
    out["loc_book1"] = int(bool(top) and top["book"] in gb)
    out["loc_page0"] = int(bool(top) and top["page"] in gp)
    out["loc_page1"] = int(bool(top) and any(top["page"][0] == b and p is not None and top["page"][1] is not None and abs(top["page"][1] - p) <= 1 for b, p in gp))
    gpr = {g["printed"] for g in gold if g["printed"]}
    if gpr and len(gpr) == len({g["page"] for g in gold}):                 # all gold pages have verified printed labels
        out["loc_printed_exact"] = int(bool(top) and top["printed"] in gpr)
    out["loc_theory"] = int(out["loc_lesson1"] and out["loc_page0"])
    return out
