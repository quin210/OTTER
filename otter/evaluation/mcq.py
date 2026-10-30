"""MCQ benchmark loader + validator (OTTER_MCQS_300 and OTTER-Hard share this format).

Row format (JSONL): {"id","question","options":{"A".."D"},"answer","subject","domain","grade", ...}.
OTTER-Hard rows additionally carry `gold_span_ids`, `reasoning_type`, `hop_count`, `split`, ...
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from otter.evaluation.stats import chi2_uniform_test
from otter.evaluation.text import _QUOTED, char_ngrams, jaccard, nfc, norm, template_of

LETTERS = ("A", "B", "C", "D")

DOMAIN_TO_SUBJECT_VI = {
    "physics": "Vật lí", "chemistry": "Hóa học", "biology": "Sinh học",
    "history": "Lịch sử", "geography": "Địa lí", "literature": "Ngữ văn",
}
SUBJECT_VI_TO_DOMAIN = {v: k for k, v in DOMAIN_TO_SUBJECT_VI.items()}


class MCQItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    question: str
    options: dict[str, str]
    answer: Optional[str] = None            # None only for unanswerable OTTER-Abstain items (answerable=False)
    domain: Optional[str] = None            # english normalized subject (physics, ...)
    subject: Optional[str] = None           # Vietnamese label as in the source file
    grade: Optional[int] = None
    topic: Optional[str] = None
    reasoning_type: Optional[str] = None
    difficulty: Optional[str] = None
    explanation: Optional[str] = None
    gold_span_ids: list[str] = Field(default_factory=list)
    # OTTER-Abstain only (absent in every other benchmark, so existing inputs/outputs are unchanged)
    answerable: bool = True
    abstain_type: Optional[str] = None
    excluded_span_ids: list[str] = Field(default_factory=list)

    def as_system_input(self, abstain_option: bool = False) -> dict[str, Any]:
        """What a QASystem receives: no answer, explanation or gold evidence.

        OTTER-Abstain: `excluded_span_ids` (spans to hide from retrieval, see otter/evaluation/exclusion.py) is passed only
        when non-empty; `abstain_option=True` appends the explicit fifth option E (otter/evaluation/abstain_metrics.py)."""
        d = {"id": self.id, "question": self.question, "options": dict(self.options),
             "subject": self.subject, "domain": self.domain, "grade": self.grade}
        if self.excluded_span_ids:
            d["excluded_span_ids"] = list(self.excluded_span_ids)
        if abstain_option:
            from otter.evaluation.abstain_metrics import with_abstain_option
            d = with_abstain_option(d)
        return d

    @property
    def answer_text(self) -> str:
        return self.options.get(self.answer, "")


def load_mcq(path: str | Path) -> list[MCQItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("domain") is None and row.get("subject") in SUBJECT_VI_TO_DOMAIN:
                row["domain"] = SUBJECT_VI_TO_DOMAIN[row["subject"]]
            if row.get("subject") is None and row.get("domain") in DOMAIN_TO_SUBJECT_VI:
                row["subject"] = DOMAIN_TO_SUBJECT_VI[row["domain"]]
            if row.get("gold_span_ids") is None:
                row["gold_span_ids"] = [row["source_span_id"]] if row.get("source_span_id") else []
            try:
                items.append(MCQItem(**row))
            except Exception as e:  # keep line number for the validator report
                raise ValueError(f"{path}:{ln}: invalid MCQ row: {e}") from e
    return items


def load_mcq_index(path: str | Path) -> dict[str, MCQItem]:
    return {it.id: it for it in load_mcq(path)}


# ---------------------------------------------------------------- validation
def validate_mcq(items: list[MCQItem], *, near_dup_threshold: float = 0.85) -> dict[str, Any]:
    """Structural + statistical checks; returns a JSON-serialisable report."""
    errors: list[dict] = []
    warnings: list[dict] = []

    ids = Counter(it.id for it in items)
    for i, c in ids.items():
        if c > 1:
            errors.append({"id": i, "check": "duplicate_id", "detail": f"{c} rows"})

    subj_pairs = Counter((it.domain, it.subject) for it in items)
    for it in items:
        keys = tuple(sorted(it.options))
        if keys != LETTERS:
            errors.append({"id": it.id, "check": "options_keys", "detail": str(keys)})
        texts = [" ".join(nfc(it.options.get(k, "")).lower().split()) for k in LETTERS]
        if any(not t for t in texts):
            errors.append({"id": it.id, "check": "empty_option", "detail": ""})
        if len(set(texts)) < len(texts):
            errors.append({"id": it.id, "check": "duplicate_option_text", "detail": str(it.options)})
        if it.answer is None and not it.answerable:
            pass                                                   # OTTER-Abstain unanswerable item
        elif it.answer not in LETTERS:
            errors.append({"id": it.id, "check": "answer_not_single_letter", "detail": it.answer})
        if not it.question.strip():
            errors.append({"id": it.id, "check": "empty_question", "detail": ""})
        if it.domain not in DOMAIN_TO_SUBJECT_VI:
            errors.append({"id": it.id, "check": "unknown_domain", "detail": str(it.domain)})
        elif it.subject and DOMAIN_TO_SUBJECT_VI[it.domain] != it.subject:
            errors.append({"id": it.id, "check": "subject_domain_mismatch", "detail": f"{it.domain}/{it.subject}"})
        if it.grade is None:
            warnings.append({"id": it.id, "check": "missing_grade", "detail": ""})
        # answer text literally contained in the question: possible giveaway
        at = norm(it.answer_text)
        if at and len(at) >= 6 and at in norm(it.question):
            warnings.append({"id": it.id, "check": "answer_text_in_question", "detail": it.answer_text[:80]})
        # options also appearing in the question (any distractor) are fine; only correct one is a giveaway
        if it.explanation and at and at not in norm(it.explanation):
            warnings.append({"id": it.id, "check": "explanation_does_not_mention_answer", "detail": ""})

    # duplicates / near duplicates
    exact = defaultdict(list)
    for it in items:
        exact[norm(it.question)].append(it.id)
    exact_dups = [v for v in exact.values() if len(v) > 1]
    grams = [char_ngrams(it.question) for it in items]
    near: list[dict] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = jaccard(grams[i], grams[j])
            if s >= near_dup_threshold and norm(items[i].question) != norm(items[j].question):
                same_ans = norm(items[i].answer_text) == norm(items[j].answer_text)
                near.append({"a": items[i].id, "b": items[j].id, "jaccard": round(s, 3), "same_answer_text": same_ans})
    # "same fact" duplicates: different wording, identical correct-answer text and same option set
    fact_dups = defaultdict(list)
    for it in items:
        fact_dups[(it.domain, norm(it.answer_text), frozenset(norm(v) for v in it.options.values()))].append(it.id)
    same_fact = [v for v in fact_dups.values() if len(v) > 1]

    # answer position bias
    pos = Counter(it.answer for it in items)
    counts = [pos.get(k, 0) for k in LETTERS]
    chi2, p = chi2_uniform_test(counts)
    by_subject_pos = {}
    for d in sorted({it.domain for it in items if it.domain}):
        c = Counter(it.answer for it in items if it.domain == d)
        cc = [c.get(k, 0) for k in LETTERS]
        by_subject_pos[d] = {"counts": dict(zip(LETTERS, cc)), "chi2": chi2_uniform_test(cc)[0],
                             "p": chi2_uniform_test(cc)[1]}

    # surface artifacts: correct answer is the (strictly) longest option
    longest = 0
    for it in items:
        lens = {k: len(it.options.get(k, "")) for k in LETTERS}
        mx = max(lens.values())
        if lens.get(it.answer) == mx and list(lens.values()).count(mx) == 1:
            longest += 1

    # mirror pairs: "which concept matches description X?" <-> "which description fits concept Y?"
    quoted = {it.id: {norm(m) for m in _QUOTED.findall(it.question)} for it in items}
    by_answer = defaultdict(list)
    for it in items:
        by_answer[(it.domain, norm(it.answer_text))].append(it)
    mirror = set()
    for it in items:
        for q in quoted[it.id]:
            for other in by_answer.get((it.domain, q), []):
                if other.id != it.id and norm(it.answer_text) in quoted[other.id]:
                    mirror.add(tuple(sorted((it.id, other.id))))
    templates = Counter(template_of(it.question) for it in items)
    report = {
        "n_items": len(items),
        "n_errors": len(errors),
        "errors": errors,
        "warnings": warnings,
        "by_domain": dict(Counter(it.domain for it in items)),
        "by_grade": {str(k): v for k, v in Counter(it.grade for it in items).items()},
        "by_reasoning_type": dict(Counter(it.reasoning_type for it in items)),
        "by_difficulty": dict(Counter(it.difficulty for it in items)),
        "by_topic": dict(Counter(it.topic for it in items)),
        "subject_label_pairs": {f"{d}|{s}": c for (d, s), c in subj_pairs.items()},
        "with_gold_spans": sum(1 for it in items if it.gold_span_ids),
        "exact_duplicate_questions": exact_dups,
        "near_duplicates": near,
        "near_dup_threshold": near_dup_threshold,
        "same_fact_duplicates": same_fact,
        "answer_position": {"counts": dict(zip(LETTERS, counts)), "chi2": chi2, "p": p, "by_domain": by_subject_pos},
        "correct_is_unique_longest": longest,
        "correct_is_unique_longest_rate": longest / max(1, len(items)),
        "mirror_pairs": sorted(mirror),
        "n_templates": len(templates),
        "top_templates": templates.most_common(15),
    }
    return report


def render_validation_md(report: dict[str, Any], *, title: str, source: str) -> str:
    L: list[str] = [f"# {title}", "", f"Source: `{source}`  ", f"Items: **{report['n_items']}**  ",
                    f"Structural errors: **{report['n_errors']}**  ",
                    f"Items with gold span ids: **{report['with_gold_spans']}**", ""]
    L += ["## Composition", "", "| field | distribution |", "|---|---|"]
    for k in ("by_domain", "by_grade", "by_reasoning_type", "by_difficulty", "by_topic"):
        dist = ", ".join(f"{a}: {b}" for a, b in sorted(report[k].items(), key=lambda x: str(x[0])))
        L.append(f"| {k[3:]} | {dist} |")
    L.append(f"| subject labels (domain\\|subject) | " +
             ", ".join(f"{a}: {b}" for a, b in sorted(report["subject_label_pairs"].items())) + " |")
    L += ["", "## Structural checks (unique ids, 4 options A–D, single valid answer, subject labels)", ""]
    if report["errors"]:
        L += ["| id | check | detail |", "|---|---|---|"]
        L += [f"| {e['id']} | {e['check']} | {str(e['detail'])[:120]} |" for e in report["errors"]]
    else:
        L.append("All structural checks passed.")
    wc = Counter(w["check"] for w in report["warnings"])
    L += ["", "### Warnings", ""]
    if wc:
        L += [f"- `{k}`: {v} item(s): " + ", ".join(w["id"] for w in report["warnings"] if w["check"] == k)[:600]
              for k, v in wc.items()]
    else:
        L.append("None.")
    ap = report["answer_position"]
    L += ["", "## Answer-position bias", "",
          "| | " + " | ".join(ap["counts"]) + " | χ² | p (uniform) |", "|---|" + "---|" * (len(ap["counts"]) + 2),
          "| all | " + " | ".join(str(v) for v in ap["counts"].values()) + f" | {ap['chi2']:.2f} | {ap['p']:.3f} |"]
    for d, v in ap["by_domain"].items():
        L.append(f"| {d} | " + " | ".join(str(x) for x in v["counts"].values()) + f" | {v['chi2']:.2f} | {v['p']:.3f} |")
    L += ["", f"Correct option is the unique longest option in {report['correct_is_unique_longest']} items "
          f"({100 * report['correct_is_unique_longest_rate']:.1f}%; chance ≈ 25% if lengths were uninformative).", ""]
    L += ["## Duplicates", "",
          f"- Exact duplicate questions (normalized text): {len(report['exact_duplicate_questions'])} group(s)"
          + (": " + "; ".join(", ".join(g) for g in report["exact_duplicate_questions"]) if report["exact_duplicate_questions"] else ""),
          f"- Near-duplicate question pairs (char-3gram Jaccard ≥ {report['near_dup_threshold']}): "
          f"{len(report['near_duplicates'])} "
          f"({sum(1 for x in report['near_duplicates'] if x['same_answer_text'])} with identical correct-answer text)",
          f"- Same-fact groups (identical option set and correct answer, different wording): "
          f"{len(report['same_fact_duplicates'])}"
          + (": " + "; ".join(", ".join(g) for g in report["same_fact_duplicates"][:20]) if report["same_fact_duplicates"] else ""),
          f"- Mirror pairs (item X asks concept→description, item Y asks description→concept for the same fact): "
          f"{len(report['mirror_pairs'])} pair(s) covering {len({i for p in report['mirror_pairs'] for i in p})} items",
          ""]
    if report["near_duplicates"]:
        L += ["| a | b | jaccard | same answer |", "|---|---|---|---|"]
        L += [f"| {x['a']} | {x['b']} | {x['jaccard']} | {x['same_answer_text']} |" for x in report["near_duplicates"][:60]]
        if len(report["near_duplicates"]) > 60:
            L.append(f"| … | {len(report['near_duplicates']) - 60} more | | |")
        L.append("")
    L += ["## Question templates", "",
          f"{report['n_templates']} distinct templates after masking quoted spans and numbers. Most frequent:", "",
          "| count | template |", "|---|---|"]
    L += [f"| {c} | {t[:140].replace('|', '/')} |" for t, c in report["top_templates"]]
    L.append("")
    return "\n".join(L)
