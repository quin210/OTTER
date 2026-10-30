"""Build OTTER-Abstain (resumable stages; LLM calls go to the main model, <= --workers concurrent).

  .venv/bin/python scripts/eval/build_otter_abstain.py --stage ooc|ctrl|mp|guard|assemble|all

Work files: benchmarks/abstain/work/*.jsonl.  Outputs: benchmarks/abstain/{dev,test,all}.jsonl, STATS.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation import abstain_build as ab  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.embeddings import get_embedder  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, LeakageIndex  # noqa: E402
from otter.evaluation.hard_seeds import eligible  # noqa: E402
from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.llm.interface import get_llm  # noqa: E402

log = logging.getLogger("otter.abstain")
WORK = ROOT / "benchmarks/abstain/work"
HARD = ROOT / "benchmarks/otter_hard"


def store(name: str) -> JsonlStore:
    return JsonlStore(WORK / name, key_fn=lambda r: r["cand_id"] + "|" + r.get("check", "cand"))


def pmap(fn, items, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def hard_final_info():
    rows = [json.loads(l) for l in open(HARD / "otter_hard_all.jsonl", encoding="utf-8")]
    return ({r["cand_id"] for r in rows}, {s for r in rows for s in r["gold_span_ids"]})


# ------------------------------------------------------------------------------------------- stages
def stage_ooc(corpus, llm, a):
    topics = ab.absent_topics(corpus)
    (WORK / "topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
    cands, checks = store("ooc_cands.jsonl"), store("ooc_checks.jsonl")
    jobs = [(t, v) for t in topics for v in range(3 if t["domain"] == "literature" else 2)]
    todo = [(t, v) for t, v in jobs if not cands.done(f"ooc_{t['topic_id'][4:]}_{v}|cand")]
    log.info("ooc: %d topics, %d generation jobs to run", len(topics), len(todo))
    pmap(lambda tv: cands.add(ab.gen_ooc(tv[0], llm, tv[1])), todo, a.workers)
    good = [c for c in cands.rows.values() if c.get("question")]
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / "tests/textbooks/OTTER_MCQS_300.jsonl")])

    def chk(c):
        if checks.done(c["cand_id"] + "|absent"):
            return
        try:
            r = ab.check_absent(c, corpus, llm)
            r.update(leak.check(c))
            checks.add({"cand_id": c["cand_id"], "check": "absent", "result": r})
        except Exception as e:
            checks.add({"cand_id": c["cand_id"], "check": "absent", "error": repr(e)})
    pmap(chk, good, a.workers)
    ok = [c for c in good if (checks.rows.get(c["cand_id"] + "|absent") or {}).get("result", {}).get("accepted")]
    log.info("ooc: %d generated, %d accepted as absent", len(good), len(ok))


def span_pool(corpus, domain=None):
    return [s for s in corpus.spans if s.domain and (domain is None or s.domain == domain) and eligible(s)]


def stage_ctrl(corpus, llm, a):
    cands, checks = store("ctrl_cands.jsonl"), store("ctrl_checks.jsonl")
    rng = random.Random(a.seed)
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / "tests/textbooks/OTTER_MCQS_300.jsonl")])
    _, hard_gold = hard_final_info()
    picked = []
    for d in ab.SUBJECTS:
        pool = [s for s in span_pool(corpus, d) if s.span_id not in hard_gold]
        by_lesson = defaultdict(list)
        for s in pool:
            by_lesson[s.lesson_key].append(s)
        keys = sorted(by_lesson)
        rng.shuffle(keys)
        picked += [rng.choice(by_lesson[k]) for k in keys[: a.ctrl_per_subject]]
    todo = [s for s in picked if not cands.done("ctl_" + ab.sha1(s.span_id)[:12] + "|cand")]
    log.info("ctrl: %d spans, %d to generate", len(picked), len(todo))
    pmap(lambda s: cands.add(ab.gen_from_span(s, llm)), todo, a.workers)
    good = [c for c in cands.rows.values() if c.get("question")]

    def chk(c):
        if checks.done(c["cand_id"] + "|control"):
            return
        try:
            r = ab.check_control(c, corpus, llm)
            r.update(leak.check(c))
            checks.add({"cand_id": c["cand_id"], "check": "control", "result": r})
        except Exception as e:
            checks.add({"cand_id": c["cand_id"], "check": "control", "error": repr(e)})
    pmap(chk, good, a.workers)
    ok = [c for c in good if (checks.rows.get(c["cand_id"] + "|control") or {}).get("result", {}).get("accepted")]
    log.info("ctrl: %d generated, %d accepted", len(good), len(ok))


def stage_mp(corpus, llm, a):
    hard_ids, hard_gold = hard_final_info()
    ann = [json.loads(l) for l in open(HARD / "candidates_annotated.jsonl", encoding="utf-8")]
    pool = []
    for c in ann:
        f = c["filters"]
        if not (f.get("structural_ok") and f.get("leak_ok") and f.get("verify_ok") is True and f.get("multi_hop_ok") is True):
            continue
        if c["cand_id"] in hard_ids:        # never reuse an OTTER-Hard final item; shared source spans are handled by the lesson rule
            continue
        if f.get("q_natural", 0) < 3 or f.get("q_needs_both", 0) < 3:
            continue
        pool.append(c)
    # candidates of the OTTER-Hard v2 build (not final items, not rewrites of final items)
    from otter.evaluation import hard_build as hb
    from otter.evaluation.hard_gen import check_key
    W2 = HARD / "work_v2"
    if (W2 / "candidates.jsonl").exists() and (W2 / "checks.jsonl").exists():
        fin = [json.loads(l) for l in open(HARD / "otter_hard_all.jsonl", encoding="utf-8")]
        fid = {r["cand_id"] for r in fin} | {r.get("parent_cand_id") for r in fin}
        fq = {r["question"] for r in fin}
        cands2 = [c for c in (json.loads(l) for l in open(W2 / "candidates.jsonl", encoding="utf-8")) if not c.get("error")]
        leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / "tests/textbooks/OTTER_MCQS_300.jsonl")])
        ann2 = hb.annotate_candidates(cands2, JsonlStore(W2 / "checks.jsonl", key_fn=check_key).rows, corpus, leak)
        have = {c["cand_id"] for c in pool}
        for c in ann2:
            f = c["filters"]
            if (f.get("structural_ok") and f.get("leak_ok") and f.get("verify_ok") is True and f.get("multi_hop_ok") is True
                    and f.get("q_natural", 0) >= 3 and f.get("q_needs_both", 0) >= 3 and c["cand_id"] not in fid
                    and c["question"] not in fq and c["cand_id"] not in have):
                pool.append(c)
                have.add(c["cand_id"])
    log.info("mp: pool %d (OTTER-Hard candidates that are not Hard final items)", len(pool))
    rng = random.Random(a.seed)
    by_dom = defaultdict(list)
    for c in pool:
        by_dom[c["domain"]].append(c)
    sel = [c for d in ab.SUBJECTS for c in sorted(by_dom[d], key=lambda c: c["cand_id"])]
    checks = store("mp_checks.jsonl")

    def chk(job):
        c, drop, name = job
        if checks.done(c["cand_id"] + "|" + name):
            return
        gold = sorted(c["gold_span_ids"])
        rest = [corpus.get(g) for g in gold if g != drop]
        ds = corpus.get(drop)
        try:
            # content duplicated elsewhere? nearest neighbours of the dropped span (other than itself)
            dup = ab.retrieve(corpus, ds.text[:400], exclude={drop}, top_k=5)
            r = ab.check_absent(c, corpus, llm, exclude=frozenset({drop}), extra_spans=rest,
                                query_extra=ds.text[:200])
            r.update(excluded=drop, remaining_gold=[s.span_id for s in rest], dup_neighbours=[s.span_id for s in dup])
            checks.add({"cand_id": c["cand_id"], "check": name, "result": r})
        except Exception as e:
            checks.add({"cand_id": c["cand_id"], "check": name, "error": repr(e)})
    jobs = []
    for c in sel:        # every gold span is tried as the removed hop ("missing" = the first, hash-chosen one; legacy name)
        gold = sorted(c["gold_span_ids"])
        first = gold[int(ab.sha1(c["cand_id"] + "drop")[:6], 16) % len(gold)]
        order = [first] + [g for g in gold if g != first]
        for k, g in enumerate(order):
            jobs.append((c, g, "missing" if k == 0 else f"missing:{g}"))
    log.info("mp: %d candidates, %d (candidate, removed-span) checks", len(sel), len(jobs))
    pmap(chk, jobs, a.workers)
    cs = {c["cand_id"]: c for c in sel}
    ok = [k for k in cs if any((checks.rows.get(f"{k}|{n}") or {}).get("result", {}).get("accepted") for _, _, n in
                               [j for j in jobs if j[0]["cand_id"] == k])]
    (WORK / "mp_cands.jsonl").write_text("\n".join(json.dumps(cs[k], ensure_ascii=False) for k in cs) + "\n", encoding="utf-8")
    log.info("mp: %d/%d candidates have at least one accepted removed span", len(ok), len(sel))


_COND = re.compile(r"\bnếu\b", re.I)
_THI = re.compile(r"\bthì\b", re.I)
_CHIKHI = re.compile(r"\b(chỉ khi|với điều kiện|trong điều kiện|trong trường hợp)\b", re.I)
_KHI = re.compile(r"\bkhi\b.{3,120}\b(thì|sẽ)\b", re.I)


def stage_guard(corpus, llm, a):
    cands, checks = store("guard_cands.jsonl"), store("guard_checks.jsonl")
    rng = random.Random(a.seed + 1)
    picked = []
    for d in ab.SUBJECTS:
        pool = span_pool(corpus, d)
        def score(s):
            t = s.text
            return 3 * bool(_COND.search(t) and _THI.search(t)) + 2 * bool(_COND.search(t)) + bool(_CHIKHI.search(t)) + bool(_KHI.search(t))
        pool = [s for s in pool if score(s) > 0]
        rng.shuffle(pool)
        pool.sort(key=lambda s: -score(s))
        picked += pool[: a.guard_per_subject]
    todo = [s for s in picked if not cands.done("grd_" + ab.sha1(s.span_id)[:12] + "|cand")]
    log.info("guard: %d spans picked, %d to generate", len(picked), len(todo))
    pmap(lambda s: cands.add(ab.gen_guard(s, llm)), todo, a.workers)
    good = [c for c in cands.rows.values() if c.get("question_guard")]

    def chk(c):
        if checks.done(c["cand_id"] + "|guard"):
            return
        try:
            span = corpus.get(c["gold_span_ids"][0])
            ctrl = ab.check_control({**c, "question": c["question_satisfied"]}, corpus, llm)
            gq = {**c, "question": c["question_guard"], "reference_key": c["answer"]}
            g = ab.check_absent(gq, corpus, llm, extra_spans=[span], key_must_be_undetermined=True)
            checks.add({"cand_id": c["cand_id"], "check": "guard", "result": {"control": ctrl, "guard": g,
                                                                               "accepted": bool(ctrl["accepted"] and g["accepted"])}})
        except Exception as e:
            checks.add({"cand_id": c["cand_id"], "check": "guard", "error": repr(e)})
    pmap(chk, good, a.workers)
    ok = [c for c in good if (checks.rows.get(c["cand_id"] + "|guard") or {}).get("result", {}).get("accepted")]
    log.info("guard: %d generated, %d accepted", len(good), len(ok))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["ooc", "ctrl", "mp", "guard", "assemble", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--ctrl-per-subject", type=int, default=22)
    ap.add_argument("--mp-per-subject", type=int, default=20)
    ap.add_argument("--guard-per-subject", type=int, default=26)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    WORK.mkdir(parents=True, exist_ok=True)
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    corpus.attach_embedder(get_embedder())
    llm = get_llm("main")
    log.info("corpus %d spans; llm %s", len(corpus), llm.model)
    stages = ["ooc", "ctrl", "mp", "guard"] if a.stage == "all" else [a.stage]
    for st in stages:
        {"ooc": stage_ooc, "ctrl": stage_ctrl, "mp": stage_mp, "guard": stage_guard}.get(st, lambda *x: None)(corpus, llm, a)
    if a.stage in ("assemble", "all"):
        from otter.evaluation.abstain_assemble import assemble
        assemble(corpus, WORK, ROOT / "benchmarks/abstain", seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
