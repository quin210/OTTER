"""Abstention metrics for OTTER-Abstain (MCQ A-D plus an explicit fifth action "abstain").

A system *abstains* on an item when its FinalAnswer has `status == "abstain"` or `answer_choice == "E"` (the explicit option
"Không đủ căn cứ trong sách" added by `with_abstain_option`). It *answers* when `answer_choice` is one of A-D (any status
other than abstain/failed, i.e. "supported" and "qualified" both count as an answer). Failed / missing responses are
excluded from the rates (and counted separately).

Positive class = "should abstain" (unanswerable items). With TP = abstain on unanswerable, FP = abstain on answerable
control, FN = answer on unanswerable:
  abstain_precision = TP / (TP + FP)          abstain_recall = TP / (TP + FN)         abstain_f1
  unsupported_inference_rate (UIR) = FN / (TP + FN)  = 1 - abstain_recall  (committing to an option that no available
                                     evidence supports); `uir_reference_key` = share of FN that chose the would-be key
  over_abstention_rate = share of answerable controls on which the system abstained
  control_accuracy = correct / n_controls (abstentions count as wrong); control_selective_accuracy = correct / answered
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional

from otter.evaluation.mcq import LETTERS, MCQItem
from otter.evaluation.stats import bootstrap_ci
from otter.schemas.reasoning import FinalAnswer

ABSTAIN_KEY = "E"
ABSTAIN_OPTION_TEXT = "Không đủ căn cứ trong sách"
ABSTAIN_INSTRUCTION = (f" (Nếu kiến thức trong sách giáo khoa không đủ căn cứ để trả lời chắc chắn, hãy chọn "
                       f"{ABSTAIN_KEY}: “{ABSTAIN_OPTION_TEXT}”; không chọn A–D khi chỉ đoán.)")


def with_abstain_option(item: dict[str, Any]) -> dict[str, Any]:
    """System input with the explicit fifth option E; the original item dict is not modified."""
    d = dict(item)
    d["options"] = {**item["options"], ABSTAIN_KEY: ABSTAIN_OPTION_TEXT}
    d["question"] = item["question"].rstrip() + ABSTAIN_INSTRUCTION
    return d


def response_kind(fa: Optional[FinalAnswer]) -> str:
    """'abstain' | 'answer' | 'failed' (failed also covers missing and unparsable answers)."""
    if fa is None or fa.status == "failed" or fa.error is not None:
        return "failed"
    ch = (fa.answer_choice or "").strip().upper()[:1]
    if fa.status == "abstain" or ch == ABSTAIN_KEY:
        return "abstain"
    if ch in LETTERS:
        return "answer"
    return "failed"


def abstain_record(item: MCQItem, fa: Optional[FinalAnswer], excluded_cited: bool = False) -> dict[str, Any]:
    kind = response_kind(fa)
    ch = (fa.answer_choice or "").strip().upper()[:1] if fa is not None and kind == "answer" else None
    prov = getattr(item, "provenance", None) or {}
    cited = [c.span_id for cl in fa.claims for c in cl.citations] if fa is not None else []
    return {"id": item.id, "domain": item.domain, "grade": item.grade, "answerable": bool(item.answerable),
            "abstain_type": item.abstain_type or "control", "kind": kind, "choice": ch,
            "correct": bool(item.answerable and ch == item.answer),
            "chose_reference_key": (ch is not None and not item.answerable and ch == prov.get("reference_key")),
            "n_citations": len(cited),
            "cited_excluded": bool(set(cited) & set(item.excluded_span_ids))}


def _rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None or p + r == 0:
        return 0.0 if (p is not None and r is not None) else None
    return 2 * p * r / (p + r)


def summarize_abstain(recs: list[dict], *, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    U = [r for r in recs if not r["answerable"]]
    A = [r for r in recs if r["answerable"]]
    Uv = [r for r in U if r["kind"] != "failed"]
    Av = [r for r in A if r["kind"] != "failed"]
    tp = sum(r["kind"] == "abstain" for r in Uv)
    fn = sum(r["kind"] == "answer" for r in Uv)
    fp = sum(r["kind"] == "abstain" for r in Av)
    tn = sum(r["kind"] == "answer" for r in Av)
    prec, rec = _rate(tp, tp + fp), _rate(tp, tp + fn)
    out: dict[str, Any] = {
        "n": len(recs), "n_unanswerable": len(U), "n_answerable": len(A),
        "n_failed_unanswerable": len(U) - len(Uv), "n_failed_answerable": len(A) - len(Av),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "abstain_precision": prec, "abstain_recall": rec, "abstain_f1": _f1(prec, rec),
        "unsupported_inference_rate": _rate(fn, tp + fn),
        "uir_reference_key": _rate(sum(r["chose_reference_key"] for r in Uv), tp + fn),
        "uir_with_citation": _rate(sum(r["kind"] == "answer" and r["n_citations"] > 0 for r in Uv), tp + fn),
        "over_abstention_rate": _rate(fp, fp + tn),
        "abstain_rate_overall": _rate(tp + fp, tp + fp + fn + tn),
        "control_accuracy": _rate(sum(r["correct"] for r in A), len(A)),
        "control_selective_accuracy": _rate(sum(r["correct"] for r in Av if r["kind"] == "answer"), tn),
        "abstention_accuracy": _rate(tp + tn, tp + fn + fp + tn),     # right action (abstain vs answer), ignoring the letter
        "cited_excluded_span": sum(r["cited_excluded"] for r in recs),
    }
    if A and n_boot > 0:
        a, lo, hi = bootstrap_ci([r["correct"] for r in A], n_boot=n_boot, seed=seed)
        out.update(control_accuracy_ci=[lo, hi])
    if Uv and n_boot > 0:
        a, lo, hi = bootstrap_ci([r["kind"] == "answer" for r in Uv], n_boot=n_boot, seed=seed)
        out.update(uir_ci=[lo, hi])
    return out


def by_group_abstain(recs: list[dict], key: str) -> dict[str, dict]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        g[str(r.get(key))].append(r)
    res = {}
    for k, rs in sorted(g.items()):
        s = summarize_abstain(rs, n_boot=0)
        res[k] = {x: s[x] for x in ("n_unanswerable", "n_answerable", "abstain_precision", "abstain_recall", "abstain_f1",
                                    "unsupported_inference_rate", "over_abstention_rate", "control_accuracy")}
    return res


def evaluate_abstain(items: Iterable[MCQItem], answers: dict[str, FinalAnswer], *, n_boot: int = 2000) -> dict[str, Any]:
    items = list(items)
    recs = [abstain_record(it, answers.get(it.id)) for it in items]
    res = {"overall": summarize_abstain(recs, n_boot=n_boot), "records": recs}
    res["by_abstain_type"] = by_group_abstain([r for r in recs if not r["answerable"] or True], "abstain_type")
    res["by_domain"] = by_group_abstain(recs, "domain")
    return res
