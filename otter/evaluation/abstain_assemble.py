"""Assemble OTTER-Abstain from the accepted work-file candidates: quotas, lesson-disjoint dev/test, ids, STATS.md."""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from otter.evaluation.abstain_build import ABBR, SUBJECTS, VERSION
from otter.evaluation.corpus import SpanCorpus
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI
from otter.evaluation.text import char_ngrams, jaccard

TYPE_CODE = {"out_of_corpus": "OOC", "missing_premise": "MPR", "counterfactual_guard": "GRD"}
QUOTA = {"missing_premise": 10, "counterfactual_guard": 7, "out_of_corpus": 8}      # per subject -> 25 -> 150 total
DEV_FRAC = 0.2
ROOT = Path(__file__).resolve().parents[2]


_DEICTIC = re.compile(r"(\bkia\b|nói trên|ở trên|như trên|đã nêu|vừa nêu|đã cho|sự kiện (thứ )?(nhất|hai)|hai sự kiện|hai (quá trình|đối tượng|khái niệm) (này|trên))", re.I)


def _rows(p: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if p.exists():
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["cand_id"] + "|" + r.get("check", "cand")] = r
    return out


def _res(rows: dict, cid: str, name: str) -> dict:
    return (rows.get(f"{cid}|{name}") or {}).get("result") or {}


def hard_lessons() -> tuple:
    rows = [json.loads(l) for l in open(ROOT / "benchmarks/otter_hard/otter_hard_all.jsonl", encoding="utf-8")]
    dev = {k for r in rows if r.get("split") == "dev" for k in r["lesson_keys"]}
    test = {k for r in rows if r.get("split") == "test" for k in r["lesson_keys"]}
    gold = {s for r in rows for s in r["gold_span_ids"]}
    gold_test = {s for r in rows if r.get("split") == "test" for s in r["gold_span_ids"]}
    qs = {r["question"] for r in rows}
    return dev, test, gold, qs, gold_test


def hard_cand_ids() -> set[str]:
    return {json.loads(l)["cand_id"] for l in open(ROOT / "benchmarks/otter_hard/otter_hard_all.jsonl", encoding="utf-8")}


def collect(corpus: SpanCorpus, work: Path) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Returns (unanswerable items with twin controls, ooc control pool, funnel)."""
    funnel: dict[str, Any] = {}
    U: list[dict] = []
    # ---- out of corpus
    cands, checks = _rows(work / "ooc_cands.jsonl"), _rows(work / "ooc_checks.jsonl")
    acc = []
    for k, c in cands.items():
        if not c.get("question"):
            continue
        r = _res(checks, c["cand_id"], "absent")
        if r.get("accepted") and r.get("leak_ok", True):
            acc.append((c, r))
    funnel["ooc"] = {"generated": sum(1 for c in cands.values() if c.get("question")), "accepted_absent": len(acc)}
    seen_topic, seen_q = Counter(), []
    for c, r in sorted(acc, key=lambda x: x[0]["cand_id"]):
        g = char_ngrams(c["question"])
        if seen_topic[c["topic"]["topic_id"]] >= 2 or any(jaccard(g, q) > 0.6 for q in seen_q):
            continue
        seen_topic[c["topic"]["topic_id"]] += 1
        seen_q.append(g)
        t = c["topic"]
        U.append({"abstain_type": "out_of_corpus", "domain": c["domain"], "grade": c["grade"], "question": c["question"],
                  "options": c["options"], "gold_span_ids": [], "excluded_span_ids": [],
                  "lesson_keys": [f"{t['book_id']}:toc:{t.get('num') or t['topic_id']}"], "control": None,
                  "reasoning_type": "out_of_corpus", "hop_count": 0,
                  "provenance": {"source": "toc_absent_lesson", "book_id": t["book_id"], "toc_title": t["title"], "toc_page": t["page"],
                                 "max_present_printed_page": t["max_present_page"], "reference_key": c["reference_key"],
                                 "generator": c["generator"], "absence_check": {"retrieved_top10": r["top_span_ids"],
                                                                                 "judge_status": r["status"], "judge_model": r["model"],
                                                                                 "exercise_sim": r.get("exercise_sim")}}})
    # ---- missing premise
    mcands, mchecks = _rows(work / "mp_cands.jsonl"), _rows(work / "mp_checks.jsonl")
    n_mp = 0
    for cid, c in mcands.items():
        rs = [r for k, x in mchecks.items() if k.startswith(c["cand_id"] + "|missing") and (r := x.get("result")) and r.get("accepted")]
        if not rs:
            continue
        r = sorted(rs, key=lambda r: r["excluded"])[0]
        n_mp += 1
        drop = r["excluded"]
        lk = sorted({corpus.get(s).lesson_key for s in c["gold_span_ids"] if corpus.get(s)})
        common = {"domain": c["domain"], "grade": c["grade"], "question": c["question"], "options": c["options"],
                  "lesson_keys": lk, "reasoning_type": c.get("hop_type"), "hop_count": len(c["gold_span_ids"])}
        U.append({**common, "abstain_type": "missing_premise", "gold_span_ids": r["remaining_gold"], "excluded_span_ids": [drop],
                  "reasoning_type": c.get("hop_type"),
                  "provenance": {"source": "otter_hard_candidate_not_in_final", "hard_cand_id": c["cand_id"], "original_gold_span_ids": c["gold_span_ids"],
                                 "reference_key": c["answer"], "generator": c["generator"],
                                 "absence_check": {"retrieved_top10": r["top_span_ids"], "judge_status": r["status"], "judge_model": r["model"],
                                                   "duplicate_neighbours_of_excluded": r["dup_neighbours"]}},
                  "control": {**common, "answer": c["answer"], "gold_span_ids": c["gold_span_ids"], "excluded_span_ids": [],
                              "provenance": {"source": "twin_full_evidence", "hard_cand_id": c["cand_id"]}}})
    funnel["missing_premise"] = {"checked": len(mcands), "accepted": n_mp}
    # ---- guard
    gc, gchk = _rows(work / "guard_cands.jsonl"), _rows(work / "guard_checks.jsonl")
    n_g = 0
    for cid, c in gc.items():
        if not c.get("question_guard"):
            continue
        r = _res(gchk, c["cand_id"], "guard")
        if not r.get("accepted"):
            continue
        n_g += 1
        common = {"domain": c["domain"], "grade": c["grade"], "options": c["options"], "lesson_keys": c["lesson_keys"],
                  "reasoning_type": "conditional_rule", "hop_count": 1}
        U.append({**common, "question": c["question_guard"], "abstain_type": "counterfactual_guard", "gold_span_ids": c["gold_span_ids"],
                  "excluded_span_ids": [],
                  "provenance": {"source": "rule_span", "rule_span_id": c["gold_span_ids"][0], "condition": c.get("condition"),
                                 "conclusion": c.get("conclusion"), "guard_kind": c.get("guard_kind"), "reference_key": c["answer"],
                                 "generator": c["generator"],
                                 "absence_check": {"judge_status": r["guard"]["status"], "judge_model": r["guard"]["model"]}},
                  "control": {**common, "question": c["question_satisfied"], "answer": c["answer"], "gold_span_ids": c["gold_span_ids"],
                              "excluded_span_ids": [], "provenance": {"source": "twin_condition_satisfied", "rule_span_id": c["gold_span_ids"][0]}}})
    funnel["counterfactual_guard"] = {"generated": sum(1 for c in gc.values() if c.get("question_guard")), "accepted": n_g}
    # ---- ooc control pool
    cc, cchk = _rows(work / "ctrl_cands.jsonl"), _rows(work / "ctrl_checks.jsonl")
    pool: dict[str, list[dict]] = defaultdict(list)
    for c in cc.values():
        r = _res(cchk, c["cand_id"], "control")
        if c.get("question") and r.get("accepted") and r.get("leak_ok", True):
            pool[c["domain"]].append({"domain": c["domain"], "grade": c["grade"], "question": c["question"], "options": c["options"],
                                      "answer": c["answer"], "gold_span_ids": c["gold_span_ids"], "excluded_span_ids": [],
                                      "lesson_keys": c["lesson_keys"], "reasoning_type": "single_span", "hop_count": 1,
                                      "provenance": {"source": "single_span_generated", "generator": c["generator"],
                                                     "verify": {"model": r["model"], "answer": r["answer"]}}})
    funnel["ooc_controls"] = {"generated": len(cc), "accepted": sum(len(v) for v in pool.values())}
    n0 = len(U)
    # literary prose has no rule-like conditionals ("nếu P thì Q" in a poem is not a curriculum rule): no guard items for Ngữ văn
    U = [u for u in U if not (u["abstain_type"] == "counterfactual_guard" and u["domain"] == "literature")]
    U = [u for u in U if not _DEICTIC.search(u["question"]) and not (u["control"] and _DEICTIC.search(u["control"]["question"]))]
    pool = {d: [c for c in v if not _DEICTIC.search(c["question"])] for d, v in pool.items()}
    funnel["dropped_context_dependent_wording"] = n0 - len(U)
    return U, pool, funnel


def select(U: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(list))
    for u in U:
        by[u["domain"]][u["abstain_type"]].append(u)
    for d in by:
        for t in by[d]:
            rng.shuffle(by[d][t])
    chosen, left = [], []
    for d in SUBJECTS:
        want = sum(QUOTA.values())
        got = []
        for t, q in QUOTA.items():
            got += by[d][t][:q]
        spare = [u for t in QUOTA for u in by[d][t][QUOTA[t]:]]
        short = want - len(got)
        # a subject short of one type is topped up from its own surplus of other types
        take = spare[:max(0, short)]
        chosen += got + take
        left += spare[max(0, short):]
    target = sum(QUOTA.values()) * len(SUBJECTS)
    if len(chosen) < target:                       # global top-up (subjects with no surplus stay below 25)
        cnt = Counter(u["domain"] for u in chosen)
        rng.shuffle(left)
        while len(chosen) < target and left:
            u = min(left, key=lambda u: cnt[u["domain"]])         # always feed the currently smallest subject
            left.remove(u)
            chosen.append(u)
            cnt[u["domain"]] += 1
    return chosen


def assign_splits(items: list[dict], seed: int, hdev: set, htest: set) -> None:
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for it in items:
        for k in it["lesson_keys"][1:]:
            parent[find(k)] = find(it["lesson_keys"][0])
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[find(it["lesson_keys"][0])].append(it)
    rng = random.Random(seed)
    utype = lambda it: it.get("control_type") if it.get("is_control") else it["abstain_type"]
    nU = Counter(it["domain"] for it in items if not it.get("is_control"))
    nT = Counter(it["abstain_type"] for it in items if not it.get("is_control"))
    target = {d: round(DEV_FRAC * n) for d, n in nU.items()}
    ttarget = {t: round(DEV_FRAC * n) for t, n in nT.items()}
    got: Counter = Counter()
    tgot: Counter = Counter()
    gl = list(groups.values())
    rng.shuffle(gl)
    prio = {"missing_premise": 0, "counterfactual_guard": 1, "out_of_corpus": 2}       # rarest/largest groups first
    gl.sort(key=lambda g: min(prio.get(it.get("abstain_type"), 3) for it in g))
    for g in gl:
        keys = {k for it in g for k in it["lesson_keys"]}
        us = [it for it in g if not it.get("is_control")]
        d = Counter(it["domain"] for it in g).most_common(1)[0][0]
        tc = Counter(it["abstain_type"] for it in us)
        fits = (got[d] + len(us) <= target.get(d, 0) + 2 and got[d] < target.get(d, 0)
                and all(tgot[t] + n <= ttarget.get(t, 0) + 1 and tgot[t] < ttarget.get(t, 0) for t, n in tc.items()))
        if keys & htest:
            sp = "test"
        elif keys & hdev:
            sp = "dev"
        else:
            sp = "dev" if fits else "test"
        if sp == "dev":
            got[d] += len(us)
            tgot.update(tc)
        for it in g:
            it["split"] = sp


def assemble(corpus: SpanCorpus, work: Path, out: Path, seed: int = 13) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    U, pool, funnel = collect(corpus, work)
    hdev, htest, hgold, hqs, hgold_test = hard_lessons()
    # no OTTER-Hard final *item* is reused (same candidate or same question); sharing a source span with a Hard item is allowed
    # but then the lesson-based rule below keeps the item out of abstain-dev whenever the lesson belongs to a Hard-test item.
    n0 = len(U)
    mp_hard = hard_cand_ids()
    U = [u for u in U if u["provenance"].get("hard_cand_id") not in mp_hard and u["question"] not in hqs]
    funnel["dropped_same_item_as_hard_final"] = n0 - len(U)
    U = select(U, seed)
    items = []
    for u in U:
        u["pair"] = u["control"] is not None
        items.append(u)
        if u["control"]:
            ctl = dict(u["control"], abstain_type=None, is_control=True, control_type=u["abstain_type"], pair_of=u)
            u["twin"] = ctl
            items.append(ctl)
    assign_splits(items, seed, hdev, htest)
    # OOC controls: matched on (split, subject), lessons disjoint across splits
    lesson_split = {k: it["split"] for it in items for k in it["lesson_keys"]}
    rng = random.Random(seed + 7)
    need = Counter((u["split"], u["domain"]) for u in U if u["abstain_type"] == "out_of_corpus")
    ou = {(u["split"], u["domain"]): [] for u in U}
    for u in U:
        if u["abstain_type"] == "out_of_corpus":
            ou[(u["split"], u["domain"])].append(u)
    for d in SUBJECTS:
        cands = pool.get(d, [])
        rng.shuffle(cands)
        for sp in ("dev", "test"):
            us = ou.get((sp, d), [])
            def allowed(c):
                ks = set(c["lesson_keys"])
                if ks & htest:
                    return sp == "test"
                if ks & hdev:
                    return sp == "dev"
                return True
            ok = [c for c in cands if all(lesson_split.get(k, sp) == sp for k in c["lesson_keys"]) and allowed(c) and not c.get("_used")]
            for u in us:
                if not ok:
                    break
                same = [c for c in ok if c["grade"] == u["grade"]] or ok       # prefer same grade
                c = same[0]
                ok.remove(c)
                c["_used"] = True
                c.update(abstain_type=None, is_control=True, control_type="out_of_corpus", pair_of=u, split=sp)
                for k in c["lesson_keys"]:
                    lesson_split[k] = sp
                u["twin"] = c
                items.append(c)
    # ids and rows
    counters: Counter = Counter()
    rows = []
    for it in sorted(items, key=lambda x: (x["domain"], x.get("control_type") or x["abstain_type"], x["question"])):
        ctl = it.get("is_control", False)
        typ = it.get("control_type") if ctl else it["abstain_type"]
        counters[(it["domain"], typ, ctl)] += 1
        it["id"] = f"OTTER-ABS-{ABBR[it['domain']]}-{TYPE_CODE[typ]}{'-C' if ctl else ''}-{counters[(it['domain'], typ, ctl)]:03d}"
    for it in items:
        twin = it.get("twin") if not it.get("is_control") else it["pair_of"]
        prov = dict(it["provenance"], corpus_version="v1-core", benchmark_version=VERSION)
        if twin is not None:
            prov["twin_id"] = twin["id"]
        rows.append({
            "id": it["id"], "subject": DOMAIN_TO_SUBJECT_VI[it["domain"]], "domain": it["domain"], "grade": it["grade"],
            "question": it["question"], "options": it["options"], "answerable": bool(it.get("is_control")),
            "answer": it.get("answer") if it.get("is_control") else None, "gold_span_ids": it["gold_span_ids"],
            "excluded_span_ids": it["excluded_span_ids"], "abstain_type": it["abstain_type"],
            "control_type": it.get("control_type"), "reasoning_type": it["reasoning_type"], "hop_count": it["hop_count"],
            "lesson_keys": it["lesson_keys"], "provenance": prov, "review_status": "unreviewed", "split": it["split"],
        })
    rows.sort(key=lambda r: r["id"])
    for name, rs in (("all", rows), ("dev", [r for r in rows if r["split"] == "dev"]), ("test", [r for r in rows if r["split"] == "test"])):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    from otter.evaluation.abstain_stats import write_stats
    write_stats(rows, funnel, out, (hdev, htest, hgold, hqs, hgold_test))
    return funnel
