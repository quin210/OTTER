"""Evaluation metrics for MCQ QA with provenance (spec §7.2).

Conventions (stated in the paper):
* Accuracy counts failed / abstained / unparsable answers as incorrect (denominator = all bench items).
  Selective accuracy (over items with a choice) and coverage are reported separately.
* Citation validity is deterministic: the cited span exists in the corpus, its book id matches, the
  stated title/subject/grade agree with builder metadata, and no page number is asserted unless it equals
  a *verified* printed page label of that span (pdf page index must match too when given).
* Evidence precision/recall compare cited span ids with gold span ids (strict) and (book, pdf page)
  pairs (page-level, lenient: tolerant to span segmentation differences).
* Claim support comes from an LLM judge with a different model than the system backbone
  (see otter/evaluation/judge.py) plus a human-audited subset; it is heuristic and disclosed as such.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from otter.evaluation.mcq import LETTERS, MCQItem
from otter.evaluation.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap
from otter.schemas.reasoning import FinalAnswer

_CHOICE_PATTERNS = [
    re.compile(r"(?:đáp án|lựa chọn|chọn|answer)\s*(?:đúng)?\s*(?:là|:)?\s*\(?([ABCD])\b\)?", re.I),
    re.compile(r"^\s*\(?([ABCD])[).:\s]", re.M),
]


def extract_choice(text: str) -> Optional[str]:
    for p in _CHOICE_PATTERNS:
        m = p.search(text or "")
        if m:
            return m.group(1).upper()
    return None


def load_answers(path: str | Path) -> dict[str, FinalAnswer]:
    out: dict[str, FinalAnswer] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            fa = FinalAnswer.model_validate_json(line)
            out[fa.question_id] = fa          # later lines (retries) win
    return out


# ------------------------------------------------------------------ per item
def item_record(item: MCQItem, fa: Optional[FinalAnswer], *, corpus=None, parse_text_fallback: bool = True,
                pdf_page_base: int = 1) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": item.id, "domain": item.domain, "grade": item.grade,
        "reasoning_type": getattr(item, "reasoning_type", None) or "unknown",
        "hop_count": getattr(item, "hop_count", None), "lesson_key": getattr(item, "lesson_key", None) or (list(getattr(item, "lesson_keys", None) or []) or [None])[0],
        "difficulty": getattr(item, "difficulty", None), "split": getattr(item, "split", None),
        "gold": item.answer, "missing": fa is None,
    }
    if fa is None:
        rec.update(choice=None, correct=False, answered=False, status="missing", failed=True, n_claims=0,
                   n_citations=0, n_valid_citations=0)
        return rec
    choice = fa.answer_choice.strip().upper()[:1] if fa.answer_choice else None
    parsed = False
    if choice not in LETTERS and parse_text_fallback and fa.status != "abstain":
        choice = extract_choice(fa.answer)
        parsed = choice is not None
    if choice not in LETTERS:
        choice = None
    rec.update(choice=choice, choice_parsed_from_text=parsed, correct=choice == item.answer,
               answered=choice is not None, status=fa.status, failed=fa.status == "failed" or fa.error is not None,
               abstain=fa.status == "abstain", qualified=fa.status == "qualified")
    # cost
    tu = fa.tool_usage or {}
    for k in ("llm_calls", "prompt_tokens", "completion_tokens", "calls", "subagents"):
        v = tu.get(k)
        rec[f"cost_{k}"] = float(v) if isinstance(v, (int, float)) else None
    lat = tu.get("latency_ms") if isinstance(tu.get("latency_ms"), (int, float)) else tu.get("wall_latency_ms")
    rec["cost_latency_ms"] = float(lat) if isinstance(lat, (int, float)) else None
    # citations
    cites = [c for cl in fa.claims for c in cl.citations]
    rec["n_claims"] = len(fa.claims)
    rec["n_citations"] = len(cites)
    if corpus is not None:
        checks = [citation_check(c, corpus, pdf_page_base=pdf_page_base) for c in cites]
        rec["n_valid_citations"] = sum(1 for c in checks if c["valid"])
        rec["n_invented_page"] = sum(1 for c in checks if c["invented_page"])
        rec["n_unknown_span"] = sum(1 for c in checks if not c["span_exists"])
        valid_claims = 0
        for cl in fa.claims:
            if cl.citations and all(citation_check(c, corpus, pdf_page_base=pdf_page_base)["valid"] for c in cl.citations):
                valid_claims += 1
        rec["n_claims_all_valid_citations"] = valid_claims
    else:
        rec["n_valid_citations"] = None
    # evidence vs gold spans
    gold = list(getattr(item, "gold_span_ids", []) or [])
    cited_ids = list(dict.fromkeys(c.span_id for c in cites))
    rec["cited_span_ids"] = cited_ids
    if gold:
        g = set(gold)
        hit = len(g & set(cited_ids))
        rec["ev_recall"] = hit / len(g)
        rec["ev_precision"] = hit / len(cited_ids) if cited_ids else None
        rec["ev_all_gold_cited"] = hit == len(g)
        if corpus is not None:
            gp = {_page_key(corpus, s) for s in g} - {None}
            cp = {_page_key(corpus, s) for s in cited_ids} - {None}
            if gp:
                rec["ev_page_recall"] = len(gp & cp) / len(gp)
                rec["ev_page_precision"] = len(gp & cp) / len(cp) if cp else None
    if gold and corpus is not None:
        from otter.evaluation.source_location import score as _loc_score
        rec.update(_loc_score(gold, fa, corpus))
    return rec


def _page_key(corpus, span_id: str):
    s = corpus.get(span_id)
    return (s.book_id, s.pdf_page_index) if s is not None else None


def citation_check(c, corpus, *, pdf_page_base: int = 1) -> dict[str, Any]:
    """Deterministic validity of one Citation against the span corpus / builder metadata."""
    s = corpus.get(c.span_id)
    out = {"span_id": c.span_id, "span_exists": s is not None, "book_match": False, "metadata_ok": False,
           "page_ok": False, "invented_page": False}
    if s is None:
        out["invented_page"] = c.page_number is not None
        out["valid"] = False
        return out
    out["book_match"] = c.book_id == s.book_id
    meta_ok = True
    if c.book_title is not None and s.book_title is not None and c.book_title.strip() != s.book_title.strip():
        meta_ok = False
    if c.subject is not None and s.domain is not None and c.subject != s.domain:
        meta_ok = False
    if c.grade is not None and s.grade is not None and int(c.grade) != int(s.grade):
        meta_ok = False
    out["metadata_ok"] = meta_ok
    page_ok = True
    if c.page_number is not None:
        if not (s.printed_page_verified and s.printed_page_label is not None
                and str(c.page_number).strip() == str(s.printed_page_label).strip()):
            page_ok = False
            out["invented_page"] = True
    if c.pdf_page_number is not None and s.pdf_page_index is not None:
        if int(c.pdf_page_number) != int(s.pdf_page_index) + pdf_page_base:
            page_ok = False
    out["page_ok"] = page_ok
    out["valid"] = out["book_match"] and meta_ok and page_ok
    return out


# ------------------------------------------------------------- aggregation
def _mean(xs: Iterable) -> Optional[float]:
    v = [float(x) for x in xs if x is not None]
    return float(np.mean(v)) if v else None


def summarize(records: list[dict], *, judgments: Optional[dict] = None, n_boot: int = 10_000,
              seed: int = 0) -> dict[str, Any]:
    """Overall metrics over per-item records (see item_record)."""
    n = len(records)
    acc, lo, hi = bootstrap_ci([r["correct"] for r in records], n_boot=n_boot, seed=seed)
    answered = [r for r in records if r.get("answered")]
    out: dict[str, Any] = {
        "n": n, "accuracy": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "coverage": len(answered) / n if n else None,
        "selective_accuracy": _mean(r["correct"] for r in answered),
        "missing": sum(1 for r in records if r.get("missing")),
        "failed": sum(1 for r in records if r.get("failed")),
        "abstain_rate": _mean(r.get("abstain", False) for r in records),
        "qualified_rate": _mean(r.get("qualified", False) for r in records),
        "choice_parsed_from_text": sum(1 for r in records if r.get("choice_parsed_from_text")),
    }
    # citations
    tot_c = sum(r.get("n_citations") or 0 for r in records)
    out["citations_per_item"] = tot_c / n if n else None
    out["items_with_citation"] = _mean((r.get("n_citations") or 0) > 0 for r in records)
    if any(r.get("n_valid_citations") is not None for r in records):
        val = sum(r.get("n_valid_citations") or 0 for r in records)
        out["citation_validity"] = val / tot_c if tot_c else None
        out["invented_page_rate"] = (sum(r.get("n_invented_page") or 0 for r in records) / tot_c) if tot_c else None
        out["unknown_span_rate"] = (sum(r.get("n_unknown_span") or 0 for r in records) / tot_c) if tot_c else None
    # evidence vs gold
    for k in ("ev_recall", "ev_precision", "ev_page_recall", "ev_page_precision", "ev_all_gold_cited", "loc_pred", "loc_span_hit1",
              "loc_span_hit_any", "loc_lesson1", "loc_book1", "loc_page0", "loc_page1", "loc_printed_exact", "loc_theory"):
        vals = [r.get(k) for r in records if k in r]
        if vals:
            # items with gold but no citations: precision undefined (None) -> excluded; recall counts as 0
            out[k] = _mean(vals)
    # claim support from judge
    if judgments is not None:
        out.update(claim_support_summary(records, judgments))
    # cost
    for k in ("llm_calls", "prompt_tokens", "completion_tokens", "calls", "subagents", "latency_ms"):
        out[f"cost_{k}"] = _mean(r.get(f"cost_{k}") for r in records)
    return out


def claim_support_summary(records: list[dict], judgments: dict) -> dict[str, Any]:
    """judgments: {(question_id, claim_id): {"verdict": supported|partial|unsupported|no_citation}}"""
    ids = {r["id"] for r in records}
    v = [j["verdict"] for (qid, _), j in judgments.items() if qid in ids]
    if not v:
        return {"n_judged_claims": 0}
    n = len(v)
    return {"n_judged_claims": n,
            "claim_support_rate": sum(x == "supported" for x in v) / n,
            "claim_partial_rate": sum(x == "partial" for x in v) / n,
            "unsupported_inference_rate": sum(x in ("unsupported", "no_citation") for x in v) / n}


def by_group(records: list[dict], key: str, *, n_boot: int = 10_000, seed: int = 0) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[str(r.get(key))].append(r)
    res = {}
    for g, rs in sorted(groups.items()):
        a, lo, hi = bootstrap_ci([r["correct"] for r in rs], n_boot=n_boot, seed=seed)
        res[g] = {"n": len(rs), "accuracy": a, "acc_ci_lo": lo, "acc_ci_hi": hi}
    return res


def compare(rec_a: list[dict], rec_b: list[dict], *, n_boot: int = 10_000, seed: int = 0,
            cluster_key: Optional[str] = None) -> dict[str, Any]:
    """Paired significance of accuracy A vs B on the items both cover (aligned by id)."""
    b_by = {r["id"]: r for r in rec_b}
    pairs = [(r, b_by[r["id"]]) for r in rec_a if r["id"] in b_by]
    a = [p[0]["correct"] for p in pairs]
    b = [p[1]["correct"] for p in pairs]
    groups = [str(p[0].get(cluster_key)) for p in pairs] if cluster_key else None
    out = {"n": len(pairs), "acc_a": _mean(a), "acc_b": _mean(b), "mcnemar": mcnemar_exact(a, b),
           "paired_bootstrap": paired_bootstrap(a, b, n_boot=n_boot, seed=seed, groups=groups)}
    return out


def evaluate(items: list[MCQItem], answers: dict[str, FinalAnswer], *, corpus=None, judgments=None,
             n_boot: int = 10_000, group_keys: tuple[str, ...] = ("domain", "grade", "reasoning_type", "hop_count", "difficulty")) -> dict:
    records = [item_record(it, answers.get(it.id), corpus=corpus) for it in items]
    res = {"overall": summarize(records, judgments=judgments, n_boot=n_boot), "records": records}
    for k in group_keys:
        res[f"by_{k}"] = by_group(records, k, n_boot=n_boot)
    return res


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
