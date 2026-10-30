"""Unified ViEduHop release index: one JSONL with every tier, built from the frozen source files (read-only).

  .venv/bin/python scripts/build_vieduhop_all.py

Writes benchmarks/vieduhop_v2/{vieduhop_all.jsonl, STATS.json}. The source files are never modified. Deliberately NOT copied:
textbook passages and gold quotes (`quotes`, `reference_answer`), the explanations (they may quote the book), generator/verifier
internals and the closed-book probe details. Source pointers (book, lesson headings, printed page) and span ids are kept.
`headline` = Core/Cross items of the test split that are not `seen_in_development` (the 1,208 items of the pre-registered headline).
"""
import hashlib, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/vieduhop_v2"
SRC = {
    "core_cross": "benchmarks/otter_hard/otter_hard_all.jsonl",
    "abstain": "benchmarks/abstain/all.jsonl",
    "twist": "benchmarks/otter_hard/twist.jsonl",
    "twist_abstain": "benchmarks/abstain/twist_abstain.jsonl",
    "para": "benchmarks/otter_hard/para.jsonl",
    "typed": "benchmarks/typed/typed600.jsonl",
    "multihop": "benchmarks/otter_hard/multihop/multihop_all_v2.jsonl",
    "extra": "benchmarks/otter_hard/extra/extra_all.jsonl",
}
HARD = set(json.load(open(ROOT / "benchmarks/otter_hard/hard_tier_ids.json")))


def rows(key):
    return [json.loads(l) for l in open(ROOT / SRC[key], encoding="utf-8") if l.strip()]


def pointers(r):
    return [{k: g.get(k) for k in ("span_id", "book_id", "book_title", "grade", "lesson_key", "lesson", "printed_page", "pdf_page")}
            for g in (r.get("where_gold") or [])]


def mcq(r, tier, fmt, headline=False):
    subj = r.get("subject") or (r.get("subjects") or [None])[0]
    return {
        "id": r["id"], "tier": tier, "format": fmt, "split": r.get("split"), "headline": headline,
        "hard_tier": r["id"] in HARD, "seen_in_development": bool(r.get("seen_in_development")),
        "subject": subj, "subjects": r.get("subjects") or [subj], "grade": r.get("grade"), "grades": r.get("grades"),
        "question": r["question"], "options": r.get("options"), "answer": r.get("answer"),
        "hop_count": r.get("hop_count"), "hop_type": r.get("hop_type"), "difficulty": r.get("difficulty"),
        "book_specific": r.get("book_specific"), "bloom": r.get("bloom"),
        "gold_span_ids": r.get("gold_span_ids"), "book_ids": r.get("book_ids"), "lesson_keys": r.get("lesson_keys"),
        "source_pointers": pointers(r), "review_status": r.get("review_status"), "source_file": SRC[
            {"core": "core_cross", "cross": "core_cross"}.get(tier, tier)],
    }


def main():
    out = []
    for r in rows("core_cross"):
        tier = "cross" if r.get("subset") == "cross_subject" else "core"
        out.append(mcq(r, tier, "mcq4", headline=(r.get("split") == "test" and not r.get("seen_in_development"))))
    for key, tier in (("abstain", "abstain"), ("twist_abstain", "twist_abstain")):
        for r in rows(key):
            out.append({"id": r["id"], "tier": tier, "format": "mcq4_abstain", "split": r.get("split"), "headline": False,
                        "hard_tier": False, "seen_in_development": False, "subject": r.get("subject"), "grade": r.get("grade"),
                        "question": r["question"], "options": r["options"], "answerable": r.get("answerable"), "answer": r.get("answer"),
                        "abstain_type": r.get("abstain_type"), "control_type": r.get("control_type"), "hop_count": r.get("hop_count"),
                        "gold_span_ids": r.get("gold_span_ids"), "excluded_span_ids": r.get("excluded_span_ids"),
                        "lesson_keys": r.get("lesson_keys"), "twin_id": (r.get("provenance") or {}).get("twin_id"),
                        "review_status": r.get("review_status"), "source_file": SRC[key]})
    for r in rows("twist"):
        d = mcq(r, "twist", "mcq4"); d.update(twist_type=r.get("twist_type"), twist_variant=r.get("twist_variant")); out.append(d)
    for r in rows("para"):
        d = mcq(r, "para", "mcq4"); d.update(orig_id=r["orig_id"], orig_subset=r.get("orig_subset"), para_round=r.get("para_round"),
                                            overlap_orig=r.get("overlap_orig"), overlap_para=r.get("overlap_para")); out.append(d)
    for r in rows("multihop"):
        d = mcq(r, "multihop", "mcq4"); d.update(authoring=r.get("authoring"), closed_book_hard=r.get("closed_book_hard")); out.append(d)
    for r in rows("extra"):
        d = mcq(r, "extra", "mcq4"); d.update(authoring=r.get("authoring"), closed_book_hard=r.get("closed_book_hard"), extra_kind=r.get("subset"), gold_span_in_dev2=bool(r.get("gold_span_in_dev2"))); out.append(d)
    for r in rows("typed"):
        out.append({"id": r["id"], "tier": "typed", "format": {"multiple_choice": "mcq4"}.get(r["format"], r["format"]), "split": "eval_only", "headline": False, "hard_tier": False,
                    "seen_in_development": False, "subject": r["subject"], "grade": int(r["grade"]) if str(r["grade"]).isdigit() else r["grade"],
                    "topic": r.get("topic"), "question": r["question"], "options": r.get("choices"), "answer": r["answer"],
                    "group_id": r.get("group_id"), "negated": r.get("negated"), "alignment_status": r.get("alignment_status"),
                    "gold_span_ids": r.get("gold_span_ids") if r.get("alignment_status") == "auto_aligned" else [],
                    "review_status": r.get("review_status_user_statement"), "authorship": r.get("authorship_user_statement"),
                    "auto_verified": False, "source_file": SRC["typed"]})
    ids = [d["id"] for d in out]
    assert len(ids) == len(set(ids)), "duplicate ids"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "vieduhop_all.jsonl"
    p.chmod(0o644) if p.exists() else None
    with open(p, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    st = {"n_items": len(out), "by_tier": dict(collections.Counter(d["tier"] for d in out)),
          "by_format": dict(collections.Counter(d["format"] for d in out)),
          "headline_items": sum(d["headline"] for d in out), "hard_tier_items": sum(d["hard_tier"] for d in out),
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
          "sources": {k: hashlib.sha256((ROOT / v).read_bytes()).hexdigest() for k, v in SRC.items()}}
    (OUT / "STATS.json").write_text(json.dumps(st, indent=1, ensure_ascii=False))
    print(json.dumps(st, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
