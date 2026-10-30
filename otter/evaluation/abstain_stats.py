"""STATS.md for OTTER-Abstain, computed from the released jsonl rows."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from otter.evaluation.abstain_build import SUBJECTS, VERSION
from otter.evaluation.mcq import LETTERS
from otter.evaluation.text import char_ngrams, jaccard, norm

RESULTS_MARK = "<!-- RESULTS -->"


def _tab(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def stats(rows: list[dict], hard) -> dict:
    hdev, htest, hgold, hqs = hard[:4]
    hgold_test = hard[4] if len(hard) > 4 else hgold
    U = [r for r in rows if not r["answerable"]]
    C = [r for r in rows if r["answerable"]]
    out = {"n": len(rows), "n_unanswerable": len(U), "n_controls": len(C)}
    out["by_split"] = {s: {"unanswerable": sum(1 for r in U if r["split"] == s), "controls": sum(1 for r in C if r["split"] == s)}
                       for s in ("dev", "test")}
    out["subject_type"] = {d: dict(Counter(r["abstain_type"] for r in U if r["domain"] == d)) for d in SUBJECTS}
    out["subject_controls"] = {d: sum(1 for r in C if r["domain"] == d) for d in SUBJECTS}
    out["grade"] = {f"{'control' if a else 'unanswerable'}|{g}": n for (a, g), n in sorted(Counter((r["answerable"], r["grade"]) for r in rows).items())}
    # leakage vs OTTER-Hard
    lk = {k for r in rows for k in r["lesson_keys"]}
    out["leak"] = {
        "items_sharing_a_source_span_with_a_hard_final_item (allowed; see rule)": sum(1 for r in rows if (set(r["gold_span_ids"]) | set(r["excluded_span_ids"])) & hgold),
        "question_identical_to_hard_final": sum(1 for r in rows if r["question"] in hqs),
        "dev_items_sharing_a_span_with_a_hard_TEST_item": sum(1 for r in rows if r["split"] == "dev" and (set(r["gold_span_ids"]) | set(r["excluded_span_ids"])) & hgold_test),
        "dev_items_on_hard_test_lessons": sum(1 for r in rows if r["split"] == "dev" and set(r["lesson_keys"]) & htest),
        "test_items_on_hard_dev_only_lessons": sum(1 for r in rows if r["split"] == "test" and set(r["lesson_keys"]) & hdev and not set(r["lesson_keys"]) & htest),
        "abstain_dev_test_shared_lessons": len({k for r in rows if r["split"] == "dev" for k in r["lesson_keys"]} &
                                                {k for r in rows if r["split"] == "test" for k in r["lesson_keys"]}),
        "abstain_items_touching_hard_dev_lessons": sum(1 for r in rows if set(r["lesson_keys"]) & hdev),
        "abstain_items_touching_hard_test_lessons": sum(1 for r in rows if set(r["lesson_keys"]) & htest),
    }
    # structure + bias
    bad = 0
    for r in rows:
        t = [norm(r["options"].get(k, "")) for k in LETTERS]
        if tuple(sorted(r["options"])) != LETTERS or len(set(t)) < 4 or "" in t or r["answerable"] != (r["answer"] is not None):
            bad += 1
        if r["answerable"] and r["answer"] not in LETTERS:
            bad += 1
    out["structural_errors"] = bad
    out["control_answer_position"] = dict(Counter(r["answer"] for r in C))
    lng = 0
    for r in C:
        ls = {k: len(v) for k, v in r["options"].items()}
        mx = max(ls.values())
        lng += ls[r["answer"]] == mx and list(ls.values()).count(mx) == 1
    out["control_key_unique_longest"] = lng / max(1, len(C))
    ref = [r["provenance"].get("reference_key") for r in U]
    out["unanswerable_reference_key_position"] = dict(Counter(ref))
    qs = [(r["id"], char_ngrams(r["question"])) for r in rows]
    twins = {r["id"]: r["provenance"].get("twin_id") for r in rows}
    near = [(a, b) for i, (a, ga) in enumerate(qs) for b, gb in qs[i + 1:] if jaccard(ga, gb) >= 0.85 and twins.get(a) != b]
    out["near_duplicate_pairs_excl_twins"] = len(near)
    out["mp_pairs_same_question"] = sum(1 for r in U if r["abstain_type"] == "missing_premise")
    out["distinct_gold_spans"] = len({s for r in rows for s in r["gold_span_ids"]})
    out["excluded_spans"] = len({s for r in rows for s in r["excluded_span_ids"]})
    return out


def write_stats(rows: list[dict], funnel: dict, out: Path, hard) -> None:
    s = stats(rows, hard)
    L = [f"# OTTER-Abstain statistics ({VERSION})", "",
         "Auto-generated from `benchmarks/abstain/{dev,test}.jsonl` by `otter/evaluation/abstain_stats.py`. All items are "
         "LLM-generated and automatically filtered; `review_status` is `unreviewed` until the teacher pack "
         "(`annotation/teacher_review_abstain/`) is back. Corpus v1-core (18 `nxbgd_` books, 16,716 spans).", "",
         f"Items: **{s['n']}** = {s['n_unanswerable']} unanswerable + {s['n_controls']} answerable controls.", ""]
    L += ["## Split", ""] + _tab(["split", "unanswerable", "answerable controls"],
                                 [[k, v["unanswerable"], v["controls"]] for k, v in s["by_split"].items()]) + [""]
    types = ["out_of_corpus", "missing_premise", "counterfactual_guard"]
    L += ["## Subject x abstain type (unanswerable) and controls", ""]
    L += _tab(["subject", *types, "total", "controls"],
              [[d, *[s["subject_type"][d].get(t, 0) for t in types], sum(s["subject_type"][d].values()), s["subject_controls"][d]]
               for d in SUBJECTS] + [["all", *[sum(s["subject_type"][d].get(t, 0) for d in SUBJECTS) for t in types],
                                       s["n_unanswerable"], s["n_controls"]]]) + [""]
    L += ["## Grade (unanswerable|control)", "", ", ".join(f"{k}: {v}" for k, v in s["grade"].items()), ""]
    L += ["## Build funnel", "", "```json", json.dumps(funnel, ensure_ascii=False, indent=1), "```", ""]
    L += ["## Leakage / disjointness checks against OTTER-Hard (final set; no Hard labels are read, only span ids, questions and lesson keys)", ""]
    L += _tab(["check", "value"], [[k, v] for k, v in s["leak"].items()]) + [""]
    L += ["Rule: an abstain item on a lesson used by an OTTER-Hard test item can only be in abstain-test; dev items come from "
          "lessons not used by Hard-test. Abstain dev/test are lesson-disjoint among themselves (twin controls share the split). `test_items_on_hard_dev_only_lessons` counts abstain-test items on a lesson used only by Hard-dev; they are in test because their lesson group (transitively, via multi-lesson items) also touches a Hard-test lesson, so they never enter abstain-dev.", ""]
    L += ["## Bias and quality", "",
          f"- structural errors (4 distinct options, answer iff answerable): **{s['structural_errors']}**",
          f"- control answer-key positions: {s['control_answer_position']}",
          f"- control key is the unique longest option: {100 * s['control_key_unique_longest']:.1f}% (chance about 25%)",
          f"- would-be key position of unanswerable items (provenance.reference_key): {s['unanswerable_reference_key_position']}",
          f"- near-duplicate question pairs (char-3gram Jaccard >= 0.85, excluding twins): {s['near_duplicate_pairs_excl_twins']}",
          f"- missing-premise items (each has an identical-question full-evidence twin): {s['mp_pairs_same_question']}",
          f"- distinct gold spans: {s['distinct_gold_spans']}; distinct excluded spans: {s['excluded_spans']}", ""]
    L += [RESULTS_MARK, ""]
    p = out / "STATS.md"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    if RESULTS_MARK in old:                                   # keep the sanity-run section across rebuilds
        L = L[:L.index(RESULTS_MARK) + 1] + [old.split(RESULTS_MARK, 1)[1]]
    p.write_text("\n".join(L), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
