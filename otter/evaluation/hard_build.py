"""OTTER-Hard assembly: filtering funnel, dedup, stratified selection, lesson-disjoint dev/test split,
benchmark rows, statistics report and the human review sheet (+ merge of the filled sheet)."""
from __future__ import annotations

import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from otter.evaluation.corpus import SpanCorpus
from otter.evaluation.hard_gen import (CLOSED_PROMPT_VERSION, GEN_PROMPT_VERSION, SINGLE_PROMPT_VERSION,
                                       VERIFY_PROMPT_VERSION, LeakageIndex, structural_checks, summarize_checks)
from otter.evaluation.hard_seeds import HOP_TYPES, SEEDS_VERSION
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS
from otter.evaluation.sheets import span_block, write_sheet
from otter.evaluation.stats import chi2_uniform_test
from otter.evaluation.text import char_ngrams, jaccard, tokens

DIFF_RANK = {"hard": 0, "medium": 1, "easy": 2, None: 3}


def annotate_candidates(cands: list[dict], checks: dict[str, dict], corpus: SpanCorpus, leak: LeakageIndex,
                        **summ_kw) -> list[dict]:
    out = []
    for c in cands:
        c = dict(c)
        f = structural_checks(c, corpus)
        if f["structural_ok"]:
            f.update(leak.check(c))
            f.update(summarize_checks(c, checks, **summ_kw))
        c["filters"] = f
        out.append(c)
    return out


def funnel(cands: list[dict], *, require_multi_hop: bool = True) -> tuple[list[dict], dict[str, int]]:
    steps = [("generated", lambda c: True),
             ("feasible", lambda c: c["filters"]["feasible"]),
             ("structural", lambda c: c["filters"]["structural_ok"]),
             ("no_leakage", lambda c: c["filters"].get("leak_ok", False)),
             ("answerable_with_gold_unique_key", lambda c: c["filters"].get("verify_ok") is True)]
    steps.append(("coherent_question", lambda c: c["filters"].get("quality_ok", True) is not False))
    if require_multi_hop:
        steps.append(("not_single_span_answerable", lambda c: c["filters"].get("multi_hop_ok") is True))
    counts, cur = {}, list(cands)
    for name, fn in steps:
        cur = [c for c in cur if fn(c)]
        counts[name] = len(cur)
    return cur, counts


def dedup(cands: list[dict], *, thr: float = 0.6) -> tuple[list[dict], list[tuple[str, str, float]]]:
    """Greedy: prefer harder items, then lower answer-length ratio; drop near-duplicate questions
    (char-3gram Jaccard over question+answer) and identical (gold span set, answer) pairs."""
    order = sorted(cands, key=lambda c: (DIFF_RANK.get(c["filters"].get("difficulty")),
                                         c["filters"].get("answer_len_ratio", 1), c["cand_id"]))
    kept, grams, keys, dropped = [], [], set(), []
    for c in order:
        g = char_ngrams(c["question"] + " " + c["correct_answer"])
        key = (frozenset(c["gold_span_ids"]), c["correct_answer"].lower())
        hit = next(((k["cand_id"], s) for k, kg in zip(kept, grams) if (s := jaccard(g, kg)) >= thr), None)
        if key in keys or hit:
            dropped.append((c["cand_id"], hit[0] if hit else "same_gold_and_answer", round(hit[1], 3) if hit else 1.0))
            continue
        kept.append(c)
        grams.append(g)
        keys.add(key)
    return kept, dropped


def cap_easy(cands: list[dict], *, frac: float = 0.25, seed: int = 0) -> tuple[list[dict], int]:
    """Within each subject keep at most `frac` items labelled easy by the closed-book small-model probe."""
    rng = random.Random(seed)
    out, dropped = [], 0
    for dom in sorted({c["domain"] for c in cands}):
        cs = [c for c in cands if c["domain"] == dom]
        easy = [c for c in cs if c["filters"].get("difficulty") == "easy"]
        other = [c for c in cs if c["filters"].get("difficulty") != "easy"]
        k = int(frac / (1 - frac) * len(other)) if frac < 1 else len(easy)
        rng.shuffle(easy)
        out += other + easy[:k]
        dropped += max(0, len(easy) - k)
    return out, dropped


def stratified_cap(cands: list[dict], max_items: Optional[int], *, seed: int = 0) -> list[dict]:
    """Proportional allocation over (domain, hop_type) cells with a floor, preferring harder items."""
    if not max_items or len(cands) <= max_items:
        return cands
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        cells[(c["domain"], c["hop_type"])].append(c)
    rng = random.Random(seed)
    for v in cells.values():
        rng.shuffle(v)
        v.sort(key=lambda c: DIFF_RANK.get(c["filters"].get("difficulty")))
    alloc = {k: min(len(v), max(2, int(max_items * len(v) / len(cands)))) for k, v in cells.items()}
    while sum(alloc.values()) > max_items:
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    while sum(alloc.values()) < max_items:
        room = [k for k in alloc if alloc[k] < len(cells[k])]
        if not room:
            break
        k = max(room, key=lambda k: len(cells[k]) - alloc[k])
        alloc[k] += 1
    return [c for k, v in cells.items() for c in v[:alloc[k]]]


def lesson_disjoint_split(items: list[dict], *, dev_size: int = 50, seed: int = 0) -> dict[str, str]:
    """Union items sharing any lesson key into groups; fill dev per subject with whole groups."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for it in items:
        ks = it["lesson_keys"]
        for k in ks[1:]:
            parent[find(k)] = find(ks[0])
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[find(it["lesson_keys"][0])].append(it)
    rng = random.Random(seed)
    doms = sorted({it["domain"] for it in items})
    quota = {d: round(dev_size * sum(1 for it in items if it["domain"] == d) / max(1, len(items))) for d in doms}
    split = {it["id"]: "test" for it in items}
    got: Counter = Counter()
    gl = list(groups.values())
    rng.shuffle(gl)
    gl.sort(key=len)                                 # small groups first -> fine-grained quota filling
    for g in gl:
        d = Counter(it["domain"] for it in g).most_common(1)[0][0]
        if got[d] + len(g) <= quota[d] + 1 and len(g) <= max(3, quota[d] // 2):
            for it in g:
                split[it["id"]] = "dev"
            got[d] += len(g)
    return split


def to_rows(cands: list[dict], corpus: SpanCorpus, corpus_version: Optional[str] = None) -> list[dict]:
    rows = []
    per_dom: Counter = Counter()
    for c in sorted(cands, key=lambda c: (c["domain"], c.get("grade") or 0, c["cand_id"])):
        per_dom[c["domain"]] += 1
        spans = [corpus.get(s) for s in c["gold_span_ids"]]
        f = c["filters"]
        rows.append({
            "id": f"OTTER-HARD-{c['domain'].upper()}-{per_dom[c['domain']]:03d}",
            "question": c["question"], "options": c["options"], "answer": c["answer"],
            "domain": c["domain"], "subject": DOMAIN_TO_SUBJECT_VI.get(c["domain"]),
            "grade": max((s.grade for s in spans if s.grade is not None), default=c.get("grade")),
            "grades": sorted({s.grade for s in spans if s.grade is not None}),
            "hop_type": c["hop_type"], "reasoning_type": c["hop_type"], "hop_count": len(c["gold_span_ids"]),
            "gold_span_ids": c["gold_span_ids"], "reasoning_steps": c.get("reasoning_steps", []),
            "link": c.get("link", ""), "explanation": " ".join(st_["step"] for st_ in c.get("reasoning_steps", [])),
            "book_ids": sorted({s.book_id for s in spans}), "lesson_keys": sorted({s.lesson_key for s in spans}),
            "seed_strategy": c.get("strategy"), "seed_id": c.get("seed_id"), "cand_id": c["cand_id"],
            "difficulty": f.get("difficulty"),
            "closed_book": {r: {"correct": f.get(f"closed_{r}_correct"), "n": f.get(f"closed_{r}_n"),
                                "mean_conf": f.get(f"closed_{r}_conf")} for r in ("small", "main")},
            "verification": {"answer": f.get("verify_answer"), "confidence": f.get("verify_conf"),
                             "n_undetermined_distractors": f.get("verify_n_undetermined")},
            "generator": c.get("generator", {}), "review_status": "unreviewed", "corpus_version": corpus_version,
        })
    return rows


# ------------------------------------------------------------------ statistics
def _ls(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": round(st.mean(xs), 1), "median": st.median(xs), "max": max(xs)} if xs else None


def bench_stats(rows: list[dict], corpus: SpanCorpus) -> dict[str, Any]:
    pos = Counter(r["answer"] for r in rows)
    longest = sum(1 for r in rows if (lambda L: L[r["answer"]] == max(L.values()) and list(L.values()).count(max(L.values())) == 1)
                  ({k: len(r["options"][k]) for k in LETTERS}))
    by_hops = {}
    for h in sorted({r["hop_count"] for r in rows}):
        rs = [r for r in rows if r["hop_count"] == h]
        by_hops[h] = {"n": len(rs), "vocab": len({t for r in rs for t in tokens(r["question"] + " " + " ".join(r["options"].values()))}),
                      "question_syllables": _ls([len(tokens(r["question"])) for r in rs]),
                      "answer_syllables": _ls([len(tokens(r["options"][r["answer"]])) for r in rs]),
                      "evidence_syllables": _ls([sum(len(tokens(corpus.get(s).text)) for s in r["gold_span_ids"] if corpus.get(s)) for r in rs])}
    cb = {}
    for role in ("small", "main"):
        vals = [r["closed_book"][role] for r in rows if r["closed_book"][role]["n"]]
        if vals:
            cb[role] = {"n_items": len(vals), "acc_over_perms": round(sum(v["correct"] for v in vals) / sum(v["n"] for v in vals), 4)}
    return {
        "n": len(rows), "by_split": dict(Counter(r.get("split") for r in rows)),
        "by_domain": dict(Counter(r["domain"] for r in rows)),
        "by_grade": {str(k): v for k, v in Counter(r["grade"] for r in rows).items()},
        "by_domain_grade": {f"{d}|{g}": v for (d, g), v in Counter((r["domain"], r["grade"]) for r in rows).items()},
        "by_hop_type": dict(Counter(r["hop_type"] for r in rows)),
        "by_domain_hop": {f"{d}|{h}": v for (d, h), v in Counter((r["domain"], r["hop_type"]) for r in rows).items()},
        "by_hop_count": dict(Counter(r["hop_count"] for r in rows)),
        "by_strategy": dict(Counter(r["seed_strategy"] for r in rows)),
        "by_difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "cross_book_items": sum(1 for r in rows if len(r["book_ids"]) > 1),
        "cross_lesson_items": sum(1 for r in rows if len(r["lesson_keys"]) > 1),
        "answer_position": {"counts": {k: pos.get(k, 0) for k in LETTERS},
                            "p_uniform": chi2_uniform_test([pos.get(k, 0) for k in LETTERS])[1]},
        "correct_is_unique_longest_rate": round(longest / max(1, len(rows)), 4),
        "closed_book_probe": cb, "by_hop_count_lengths": by_hops,
        "n_gold_spans": len({s for r in rows for s in r["gold_span_ids"]}),
        "n_books": len({b for r in rows for b in r["book_ids"]}),
    }


def render_report(manifest: dict, stats: dict, funnel_counts: dict) -> str:
    L = ["# OTTER-Hard build report", "",
         f"Built {manifest['built_at']} · generator `{manifest['generator_model']}` (`{GEN_PROMPT_VERSION}`) · "
         f"verifier `{manifest.get('verifier_model')}` · difficulty probe `{manifest.get('probe_model')}`  ",
         f"Corpus `{(manifest.get('corpus') or {}).get('corpus_version')}` · spans_sha256 "
         f"`{(manifest.get('corpus') or {}).get('spans_sha256')}` · matches manifest: "
         f"{(manifest.get('corpus') or {}).get('matches_manifest')}", "",
         "All items are LLM-generated from textbook spans and automatically filtered; human review status is recorded "
         "per item (`review_status`). Numbers below are computed from the files in this directory.", "",
         "## Filtering funnel", "", "| stage | items |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in funnel_counts.items()]
    L += ["", "## Composition", "", "| field | distribution |", "|---|---|"]
    for k in ("by_split", "by_domain", "by_grade", "by_hop_type", "by_hop_count", "by_strategy", "by_difficulty"):
        L.append(f"| {k[3:]} | " + ", ".join(f"{a}: {b}" for a, b in sorted(stats[k].items(), key=lambda x: str(x[0]))) + " |")
    L += [f"| cross-book items | {stats['cross_book_items']} |", f"| cross-lesson items | {stats['cross_lesson_items']} |",
          f"| distinct gold spans / books | {stats['n_gold_spans']} / {stats['n_books']} |", ""]
    doms = sorted(stats["by_domain"])
    hops = [h for h in HOP_TYPES if h in stats["by_hop_type"]]
    L += ["### Subject × hop type", "", "| subject | " + " | ".join(hops) + " |", "|---|" + "---|" * len(hops)]
    for d in doms:
        L.append(f"| {d} | " + " | ".join(str(stats["by_domain_hop"].get(f"{d}|{h}", 0)) for h in hops) + " |")
    ap = stats["answer_position"]
    L += ["", "## Bias checks", "", f"- Answer position: {ap['counts']} (χ² uniform p = {ap['p_uniform']:.3f})",
          f"- Correct option is the unique longest: {100 * stats['correct_is_unique_longest_rate']:.1f}% (chance 25%)"]
    for role, v in stats["closed_book_probe"].items():
        L.append(f"- Closed-book {role} model accuracy over option permutations (filter-time probe): "
                 f"{100 * v['acc_over_perms']:.1f}% on {v['n_items']} items")
    L += ["", "## Length statistics by number of gold spans (syllables)", "",
          "| hops | n | vocab | question mean/med/max | answer mean/med/max | evidence mean/med/max |", "|---|---|---|---|---|---|"]
    f = lambda x: f"{x['mean']}/{x['median']}/{x['max']}" if x else "--"
    for h, v in stats["by_hop_count_lengths"].items():
        L.append(f"| {h} | {v['n']} | {v['vocab']} | {f(v['question_syllables'])} | {f(v['answer_syllables'])} | {f(v['evidence_syllables'])} |")
    L += ["", "## Versions", "", "```json", json.dumps(manifest, ensure_ascii=False, indent=1), "```", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ review sheet
REVIEW_FIELDS = [
    {"name": "answer_correct", "label": "Keyed answer correct (per the spans)?", "type": "choice", "options": ["yes", "no", "unsure"]},
    {"name": "unique_answer", "label": "Exactly one option is correct?", "type": "choice", "options": ["yes", "no", "unsure"]},
    {"name": "needs_all_spans", "label": "Needs ≥2 spans (true multi-hop)?", "type": "choice", "options": ["yes", "no", "unsure"]},
    {"name": "hop_type_ok", "label": "Hop type label fits?", "type": "choice", "options": ["yes", "no"]},
    {"name": "question_quality", "label": "Clear, natural, curriculum-appropriate?", "type": "choice", "options": ["good", "minor_issues", "bad"]},
    {"name": "outside_knowledge", "label": "Requires knowledge not in the spans?", "type": "choice", "options": ["no", "yes"]},
    {"name": "decision", "label": "Decision", "type": "choice", "options": ["keep", "fix", "drop"]},
    {"name": "corrected_answer", "label": "Corrected key letter (if wrong)", "type": "text", "required": False},
    {"name": "notes", "label": "Notes / fix", "type": "text", "required": False},
]


def review_sample(rows: list[dict], n: int = 150, seed: int = 0) -> list[dict]:
    """Stratified by (domain, hop_type), round-robin so every cell is covered before any cell gets a second item."""
    rng = random.Random(seed)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["domain"], r["hop_type"])].append(r)
    for v in cells.values():
        rng.shuffle(v)
    keys = sorted(cells)
    out = []
    while len(out) < min(n, len(rows)):
        for k in keys:
            if cells[k] and len(out) < n:
                out.append(cells[k].pop())
    return out


def write_review_sheet(rows: list[dict], corpus: SpanCorpus, out_dir: Path, *, n: int = 150, seed: int = 0,
                       name: str = "otter_hard_review") -> tuple[Path, Path]:
    sample = review_sample(rows, n, seed)
    srows = []
    for r in sample:
        steps = "\n".join(f"{i + 1}. [{s.get('span_id')}] {s.get('step')}" for i, s in enumerate(r.get("reasoning_steps", [])))
        srows.append({"id": r["id"], "split": r.get("split"), "domain": r["domain"], "grade": r["grade"],
                      "hop_type": r["hop_type"], "question": r["question"],
                      **{f"option_{k}": r["options"][k] for k in LETTERS}, "answer": r["answer"],
                      "gold_span_ids": r["gold_span_ids"], "difficulty": r.get("difficulty"),
                      "__blocks": [
                          {"kind": "text", "label": f"{r['subject']} · lớp {r['grade']} · {r['hop_type']} · {r['hop_count']} spans · "
                                                     f"{r.get('seed_strategy')} · {r.get('split')}", "value": r["question"]},
                          {"kind": "options", "label": "Options (keyed answer highlighted)", "value": r["options"], "answer": r["answer"]},
                          {"kind": "text", "label": "Generator's reasoning steps", "value": steps or "—"},
                          span_block("Gold evidence spans", [corpus.get(s) for s in r["gold_span_ids"]], max_chars=2200),
                      ]})
    instr = ("OTTER-Hard human review. For each item read the gold spans (textbook OCR text) and the question. "
             "Judge ONLY against the span texts (plus arithmetic). 'Needs ≥2 spans' = a student could NOT answer from any "
             "single span alone. Decision: keep (no change), fix (small fix described in Notes / corrected key), drop "
             "(wrong, ambiguous, not multi-hop, needs outside knowledge, or poor quality). Export CSV when done and save it "
             f"as annotation/otter_hard_review/{name}_filled.csv. Generator prompt `{GEN_PROMPT_VERSION}`.")
    ctx = ["id", "split", "domain", "grade", "hop_type", "question"] + [f"option_{k}" for k in LETTERS] + \
          ["answer", "gold_span_ids", "difficulty"]
    return write_sheet(srows, context_cols=ctx, fields=REVIEW_FIELDS, out_dir=out_dir, name=name,
                       title="OTTER-Hard review", instructions=instr)


def apply_review(rows: list[dict], filled: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    """Merge a filled review CSV: drop -> removed; fix with corrected key -> key updated; keep -> human_verified."""
    by = {r["id"]: r for r in filled if r.get("decision")}
    out, stats = [], Counter()
    for r in rows:
        h = by.get(r["id"])
        if h is None:
            out.append(r)
            continue
        r = dict(r)
        stats[h["decision"]] += 1
        for k in ("answer_correct", "unique_answer", "needs_all_spans", "hop_type_ok", "question_quality", "outside_knowledge"):
            stats[f"{k}={h.get(k)}"] += 1
        if h["decision"] == "drop":
            continue
        ck = (h.get("corrected_answer") or "").strip().upper()[:1]
        if h["decision"] == "fix" and ck in LETTERS:
            r["answer"] = ck
        r["review_status"] = "human_kept" if h["decision"] == "keep" else "human_fixed"
        r["review"] = {k: h.get(k) for k in [f["name"] for f in REVIEW_FIELDS]}
        out.append(r)
    n = sum(stats[k] for k in ("keep", "fix", "drop"))
    summary = {"n_reviewed": n, **dict(stats)}
    if n:
        summary["keep_rate"] = stats["keep"] / n
        summary["answer_correct_rate"] = stats["answer_correct=yes"] / n
        summary["multi_hop_rate"] = stats["needs_all_spans=yes"] / n
    return out, summary


def manifest(**kw) -> dict:
    return {"seeds_version": SEEDS_VERSION, "gen_prompt": GEN_PROMPT_VERSION, "verify_prompt": VERIFY_PROMPT_VERSION,
            "single_prompt": SINGLE_PROMPT_VERSION, "closed_prompt": CLOSED_PROMPT_VERSION, **kw}
