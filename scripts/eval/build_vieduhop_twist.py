"""ViEduHop-Twist builder (27 Oct 2026). Resumable stages; all work files in benchmarks/otter_hard/twist/ (raw LLM calls in twist/llm_calls/).

  thresholds  (done once, BEFORE generation, see twist/thresholds.json)  content-word stem/gold-span overlap on the existing test split
  seeds       per twist type, seed spans from the corpus (cue statistics, bge-m3 neighbours, existing cross-subject anchors)  -> seeds.jsonl, ae_seeds.jsonl
  generate    one type-specific generation per seed (prompt otter_hard_twist_gen_v1, main model)                              -> candidates.jsonl
  screen      deterministic checks + closed-book (8B,32B x 4 perms) + gold-only x3 + near-miss-only + leave-one-out + strict review -> checks.jsonl
  ae          almost_enough (abstain format): generate + absence check + twin-control check                                  -> ae_candidates.jsonl, ae_checks.jsonl
  assemble    dedup, selection (subject round-robin, hash order; never uses any system's answers), enrichment, write
              benchmarks/otter_hard/twist.jsonl and benchmarks/abstain/twist_abstain.jsonl, funnel.json

  OTTER_LLM_URL_MAIN=http://127.0.0.1:8005/v1 .venv/bin/python scripts/eval/build_vieduhop_twist.py --stage all --workers 4
Max concurrent LLM requests = --workers (default 4). Never writes outside benchmarks/otter_hard/twist/, twist.jsonl, benchmarks/abstain/twist_abstain.jsonl.
"""
from __future__ import annotations

import argparse, datetime as dt, json, logging, os, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OTTER_LLM_LOG_DIR", str(ROOT / "benchmarks/otter_hard/twist/llm_calls"))
from otter.evaluation import book_specific as bs, hard_deep as hd, hard_twist as tw, hard_v2 as v2  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter, corpus_manifest  # noqa: E402
from otter.evaluation.embeddings import get_embedder  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, LeakageIndex, check_closed, check_key, check_single  # noqa: E402
from otter.evaluation.hard_seeds import load_seeds, write_seeds  # noqa: E402
from otter.evaluation.mcq import DOMAIN_TO_SUBJECT_VI, LETTERS, load_mcq  # noqa: E402
from otter.evaluation.text import char_ngrams, jaccard, sha1  # noqa: E402
from otter.llm.interface import OpenAICompatLLM, get_llm, load_config  # noqa: E402

log = logging.getLogger("twist")
B = ROOT / "benchmarks/otter_hard"
T = B / "twist"
HARD_MAX = 0.5          # closed_book_hard: accuracy over 4 option permutations <= 0.5 for BOTH models (same as the book-specific pool)
PERMS_LOO = 2
MAX_PER_TYPE = 8


class BalancedLLM:
    """Dispatches each call to the replica with the smallest expected wait ((my in-flight + 1) x EWMA latency per output token); at most
    --workers calls are in flight in total, so the global cap on concurrent requests is unchanged. Same model on every URL."""

    def __init__(self, llms):
        import threading
        self.llms, self.model, self.lock = llms, llms[0].model, threading.Lock()
        self.inflight, self.ewma = [0] * len(llms), [1.0] * len(llms)

    def __getattr__(self, k):
        return getattr(self.llms[0], k)

    def chat(self, messages, **kw):
        import time
        with self.lock:
            i = min(range(len(self.llms)), key=lambda j: (self.inflight[j] + 1) * self.ewma[j])
            self.inflight[i] += 1
        t = time.perf_counter()
        try:
            r = self.llms[i].chat(messages, **kw)
            dt_ = (time.perf_counter() - t) / max(16, r.completion_tokens)
            with self.lock:
                self.ewma[i] = 0.7 * self.ewma[i] + 0.3 * dt_ if self.ewma[i] != 1.0 else dt_
            return r
        finally:
            with self.lock:
                self.inflight[i] -= 1


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pmap(fn, items, workers):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(fn, items))


def run_jobs(jobs, store, workers):
    jobs = [j for j in jobs if not store.done(f"{j[0]}|{j[1]}")]

    def one(j):
        cid, name, fn = j
        try:
            store.add({"cand_id": cid, "check": name, "result": fn()})
        except Exception as e:  # noqa: BLE001
            store.add({"cand_id": cid, "check": name, "error": f"{type(e).__name__}: {e}"[:300]})
    pmap(one, jobs, workers)
    return len(jobs)


def acc_of(c, checks, role, n=4):
    rs = [(checks.get(f"{c['cand_id']}|closed:{role}:{p}") or {}).get("result") for p in range(n)]
    if any(r is None for r in rs):
        return None
    return sum(1 for r in rs if r["answer"] == c["answer"]) / n


def leak_worst(leak, c, corpus):
    """LeakageIndex looks at exercise spans of ONE subject; multi-subject items take the worst case over their gold subjects."""
    doms = sorted({corpus.get(s).domain for s in c["gold_span_ids"] if corpus.get(s)}) or [c["domain"]]
    rs = [leak.check({**c, "domain": d}) for d in doms]
    best = max(rs, key=lambda r: r["exercise_sim"])
    return {**best, "other_bench_sim": max(r["other_bench_sim"] for r in rs), "leak_ok": all(r["leak_ok"] for r in rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seeds", "generate", "screen", "ae", "assemble", "all"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", default="lexical_gap:110,negation_exception:100,hidden_condition:110,numeric:120,misconception:110,unnamed_bridge:140,near_miss_lesson:140")
    ap.add_argument("--ae-n", type=int, default=70)
    ap.add_argument("--round", default="r0", help="seed round tag; a new round adds seeds on spans not used before")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N seeds per type")
    ap.add_argument("--emb-cache", default=str(ROOT / "runs/cache/embeddings.sqlite3"))
    ap.add_argument("--max-per-type", type=int, default=MAX_PER_TYPE)
    ap.add_argument("--types", default="")
    ap.add_argument("--gen-model", default="main", choices=["main", "small"], help="generator: main = main model (two replicas), small = small model (:8001); all checks use the main model")
    ap.add_argument("--seed-ids", default="", help="generate only for the seed ids listed in this file (one per line)")
    ap.add_argument("--gen-samples", type=int, default=1)
    ap.add_argument("--onto-bridge", type=int, default=0, help="seeds stage: add N ontology-chain unnamed_bridge seeds (round tag --round)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    T.mkdir(parents=True, exist_ok=True)
    thr = json.loads((T / "thresholds.json").read_text())
    cfg = load_config()
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    mc = cfg["llm"]["main"]
    urls = [u for u in os.environ.get("TWIST_MAIN_URLS", "http://127.0.0.1:8005/v1,http://127.0.0.1:8004/v1").split(",") if u]
    main_llm = BalancedLLM([OpenAICompatLLM(base_url=u, model=mc["model"]) for u in urls])
    think_llm = BalancedLLM([OpenAICompatLLM(base_url=u, model=mc["model"], enable_thinking=True) for u in urls])
    small_llm = get_llm("small")
    log.info("main replicas=%s small=%s", urls, small_llm.client.base_url)
    only = set(a.types.split(",")) if a.types else None

    # ------------------------------------------------------------------ seeds
    seeds_p, ae_seeds_p = T / "seeds.jsonl", T / "ae_seeds.jsonl"
    if a.stage in ("seeds", "all"):
        import numpy as np
        old = load_seeds(seeds_p) if seeds_p.exists() else []
        done_tags = {(s.strategy, s.link.get("round")) for s in old}
        used_by_type = defaultdict(set)
        for s_ in old:
            if s_.strategy != "misconception":          # the misconception pool is small: spans may be re-sampled in a later round
                used_by_type[s_.strategy] |= set(s_.span_ids)
        used_spans = {x for s_ in old for x in s_.span_ids}
        if not any(t[1] == a.round for t in done_tags):
            corpus.attach_embedder(get_embedder(cache_path=a.emb_cache))
            pageq = tw.load_page_quality(ROOT)
            pool = tw.twist_pool(corpus, pageq)
            pos = {s.span_id: i for i, s in enumerate(corpus.spans)}
            dense = corpus.dense_matrix()
            used_pairs = set()
            for r in read_jsonl(B / "otter_hard_all.jsonl"):
                g = r["gold_span_ids"]
                used_pairs |= {frozenset((x, y)) for i, x in enumerate(g) for y in g[i + 1:]}
            W = B / "work_cross"
            judged = {r["pair_id"]: r for r in read_jsonl(W / "cross2_judge.jsonl")}
            anchors = []
            for fn in ("cross2_pairs.jsonl", "cross2_pairs_b.jsonl"):
                for r in read_jsonl(W / fn):
                    if ({"concept", "bridge"} & set(r["sources"])) or judged.get(r["pair_id"], {}).get("score", 0) >= 3:
                        if corpus.get(r["a"]) and corpus.get(r["b"]) and corpus.get(r["a"]).domain != corpus.get(r["b"]).domain:
                            anchors.append(r)
            n = {k: int(v) for k, v in (x.split(":") for x in a.n.split(","))}
            new = tw.build_seeds(corpus, pool, dense, pos, used_pairs, anchors, n=n, seed=a.seed + len(done_tags), exclude_by_type=dict(used_by_type), round_tag=a.round)
            for s in new:
                s.link = dict(s.link or {}, round=a.round)
            aes = tw.almost_enough_seeds(pool, a.ae_n, seed=a.seed + 1 + len(done_tags), round_tag=a.round,
                                         exclude_spans=used_spans if old else None)
            for s in aes:
                s.link = dict(s.link or {}, round=a.round)
            write_seeds(old + new, seeds_p)
            oldae = load_seeds(ae_seeds_p) if ae_seeds_p.exists() else []
            write_seeds(oldae + aes, ae_seeds_p)
            log.info("seeds: +%d (%s), almost_enough +%d", len(new), dict(Counter(s.strategy for s in new)), len(aes))
            (T / f"seeds_stats_{a.round}.json").write_text(json.dumps({
                "by_type": Counter(s.strategy for s in new), "by_type_subject": {f"{t}|{d}": c for (t, d), c in Counter((s.strategy, "+".join(sorted({corpus.get(x).domain for x in s.span_ids}))) for s in new).items()},
                "almost_enough": len(aes), "pool_spans": len(pool), "anchor_pairs": len(anchors)}, ensure_ascii=False, indent=1, default=dict))
    if a.onto_bridge and a.stage in ("seeds", "all"):
        from otter.evaluation import hard_v2 as v2
        old = load_seeds(seeds_p)
        if not any(s_.link.get("round") == a.round and s_.link.get("kind") == "onto_chain" for s_ in old):
            used_pairs = set()
            for r in read_jsonl(B / "otter_hard_all.jsonl"):
                g = r["gold_span_ids"]
                used_pairs |= {frozenset((x, y)) for i, x in enumerate(g) for y in g[i + 1:]}
            pq = tw.load_page_quality(ROOT)
            pool_ids = {s_.span_id for s_ in corpus.spans if (pq.get((s_.book_id, s_.pdf_page_index)) or 1) >= 0.9 and not tw._META.search(s_.text)}
            new = tw.onto_bridge_seeds(corpus, v2.Onto(ROOT / "data/ontology/ontology.sqlite3"), used_pairs, pool_ids, n=a.onto_bridge, round_tag=a.round)
            write_seeds(old + new, seeds_p)
            log.info("onto-chain bridge seeds: +%d %s", len(new), dict(Counter(s_.domain for s_ in new)))
    seeds = load_seeds(seeds_p) if seeds_p.exists() else []
    if only:
        seeds = [s for s in seeds if s.strategy in only]
    if a.limit:
        by = defaultdict(list)
        for s in seeds:
            by[s.strategy].append(s)
        seeds = [x for v in by.values() for x in v[: a.limit]]
    if a.stage == "seeds":
        return 0

    # ------------------------------------------------------------------ generate
    cands = JsonlStore(T / "candidates.jsonl", key_fn=lambda r: r["cand_id"])
    if a.stage in ("generate", "all"):
        gen_llm = main_llm if a.gen_model == "main" else small_llm
        only_ids = set(open(a.seed_ids).read().split()) if a.seed_ids else None
        todo = [(s, k) for s in seeds for k in range(a.gen_samples) if (only_ids is None or s.seed_id in only_ids) and not cands.done(tw.cand_id_of(s, gen_llm.model, k))]
        log.info("generating %d candidates with %s", len(todo), gen_llm.model)
        pmap(lambda t: cands.add(tw.generate_twist(t[0], corpus, gen_llm, sample=t[1], temperature=0.7 if t[1] == 0 else 0.9)), todo, a.workers)
    seed_ids = {s.seed_id for s in seeds}
    allc = [c for c in cands.rows.values() if c.get("seed_id") in seed_ids]
    log.info("candidates: %d (errors %d, feasible %d)", len(allc), sum(1 for c in allc if c.get("error")), sum(1 for c in allc if c.get("feasible")))

    leak_q = [(it.id, it.question) for it in load_mcq(ROOT / cfg["paths"]["benchmark"])] + [(r["id"], r["question"]) for r in read_jsonl(B / "otter_hard_all.jsonl")]
    leak = LeakageIndex(corpus, leak_q)
    bench = read_jsonl(B / "otter_hard_all.jsonl")
    bgrams = [char_ngrams(r["question"] + " " + r["options"][r["answer"]]) for r in bench]
    ck = JsonlStore(T / "checks.jsonl", key_fn=check_key)

    def det(c):
        c = dict(c)
        c.pop("filters", None)
        f = {"feasible": bool(c.get("feasible"))}
        if f["feasible"]:
            f.update(tw.twist_structural(c, corpus, thr))
            if f["twist_structural_ok"]:
                f.update(leak_worst(leak, c, corpus))
                g = char_ngrams(c["question"] + " " + c["correct_answer"])
                f["bench_dup_sim"] = round(max((jaccard(g, x) for x in bgrams), default=0), 3)
                f["leak_ok"] = bool(f["leak_ok"]) and f["bench_dup_sim"] < 0.6
        c["filters"] = f
        return c

    def get_of(c):
        return lambda name, cid=c["cand_id"]: (ck.rows.get(f"{cid}|{name}") or {}).get("result")

    def annotate(c):
        c = det(c)
        f = c["filters"]
        if f["feasible"] and f.get("twist_structural_ok") and f.get("leak_ok"):
            g = get_of(c)
            f["closed_small_acc"], f["closed_main_acc"] = acc_of(c, ck.rows, "small"), acc_of(c, ck.rows, "main")
            f["closed_book_hard"] = bool(f["closed_small_acc"] is not None and f["closed_main_acc"] is not None and f["closed_small_acc"] <= HARD_MAX
                                         and f["closed_main_acc"] <= HARD_MAX)
            f.update(tw.fold_gold(c, g))
            if c["twist_type"] == "near_miss_lesson":
                f.update(tw.fold_nm(c, g))
            if len(c["gold_span_ids"]) >= 2:
                f.update(hd.loo_verdict(c, g, PERMS_LOO))
            ind = [g("indep:0"), g("indep:1")]
            f["indep_ok"] = None if any(x is None for x in ind) else all(x["correct"] for x in ind)
            au = g("audit")
            f["audit_ok"] = None if au is None else bool(au["ok"])
            ss = [g("strict:0"), g("strict:1")]
            f["strict_ok"] = None if any(x is None for x in ss) else all(x["ok"] for x in ss)
        return c

    # ------------------------------------------------------------------ screen
    def pass_det(c):
        f = c["filters"]
        return f["feasible"] and f.get("twist_structural_ok") and f.get("leak_ok") and f.get("overlap_ok")

    def need_cb(c):
        return c["twist_type"] in tw.CLOSED_BOOK_REQUIRED

    if a.stage in ("screen", "all"):
        ann = [det(c) for c in allc]
        stage1 = [c for c in ann if pass_det(c)]
        log.info("deterministic survivors: %d / %d", len(stage1), len(ann))
        n = run_jobs([(c["cand_id"], f"closed:small:{p}", lambda c=c, p=p: check_closed(c, small_llm, p)) for c in stage1 for p in range(4)], ck, a.workers)
        log.info("closed small: %d jobs", n)
        s2 = [c for c in stage1 if not need_cb(c) or (acc_of(c, ck.rows, "small") is not None and acc_of(c, ck.rows, "small") <= HARD_MAX)]
        n = run_jobs([(c["cand_id"], f"closed:main:{p}", lambda c=c, p=p: check_closed(c, main_llm, p)) for c in s2 for p in range(4)], ck, a.workers)
        log.info("closed main: %d jobs", n)
        s3 = [c for c in s2 if not need_cb(c) or (acc_of(c, ck.rows, "main") is not None and acc_of(c, ck.rows, "main") <= HARD_MAX)]
        jobs = []
        for c in s3:
            for i in range(3):
                jobs.append((c["cand_id"], f"gold:{i}", lambda c=c, i=i: tw.check_gold_sample(c, corpus, main_llm, i)))
            if c["twist_type"] == "negation_exception":
                for i in range(2):
                    jobs.append((c["cand_id"], f"optruth:{i}", lambda c=c, i=i: bs.check_option_truth(c, corpus, main_llm, i)))
        log.info("gold-only: %d jobs", run_jobs(jobs, ck, a.workers))
        s4 = [c for c in map(annotate, s3) if c["filters"].get("gold_ok") and c["filters"].get("gold_unique_ok")]
        jobs = []
        for c in s4:
            if c["twist_type"] == "near_miss_lesson":
                for i in range(3):
                    jobs.append((c["cand_id"], f"nm:{i}", lambda c=c, i=i: tw.check_nm_only(c, corpus, main_llm, i)))
            if len(c["gold_span_ids"]) >= 2:
                for s in c["gold_span_ids"]:
                    for p in range(PERMS_LOO):
                        jobs.append((c["cand_id"], f"loo:{s}:{p}", lambda c=c, s=s, p=p: hd.check_loo(c, s, corpus, main_llm, p)))
        log.info("necessity (near-miss / leave-one-out): %d jobs", run_jobs(jobs, ck, a.workers))
        s4b = [c for c in map(annotate, s4) if (c["twist_type"] != "near_miss_lesson" or c["filters"].get("nm_ok")) and
               (len(c["gold_span_ids"]) < 2 or c["filters"].get("loo_all_necessary"))]
        log.info("independent 8B gold-only: %d jobs", run_jobs([(c["cand_id"], f"indep:{i}", lambda c=c, i=i: bs.check_independent(c, corpus, small_llm, i))
                                                               for c in s4b if c["twist_type"] != "negation_exception" for i in range(2)], ck, a.workers))
        s5 = [c for c in map(annotate, s4b) if c["twist_type"] == "negation_exception" or c["filters"].get("indep_ok")]
        log.info("audit: %d jobs", run_jobs([(c["cand_id"], "audit", lambda c=c: bs.check_audit(c, main_llm)) for c in s5], ck, a.workers))
        s6 = [c for c in map(annotate, s5) if c["filters"].get("audit_ok")]
        log.info("strict review (2 permutations): %d jobs", run_jobs([(c["cand_id"], f"strict:{i}", lambda c=c, i=i: tw.check_strict_twist(c, corpus, think_llm, i)) for c in s6 for i in range(2)], ck, a.workers))

    # ------------------------------------------------------------------ almost_enough
    aes = load_seeds(ae_seeds_p) if ae_seeds_p.exists() else []
    aec = JsonlStore(T / "ae_candidates.jsonl", key_fn=lambda r: r["cand_id"])
    aek = JsonlStore(T / "ae_checks.jsonl", key_fn=check_key)
    if a.stage in ("ae", "all") and (not only or "almost_enough" in only):
        if a.limit:
            aes = aes[: a.limit]
        todo = [s for s in aes if not aec.done(tw.ae_cand_id(s, main_llm.model, 0))]
        log.info("almost_enough: generating %d", len(todo))
        pmap(lambda s: aec.add(tw.generate_ae(s, corpus, main_llm)), todo, a.workers)
        from otter.evaluation import abstain_build as ab
        from otter.evaluation.hard_gen import check_verify
        okc = [c for c in aec.rows.values() if c.get("feasible") and c.get("options") and c["seed_id"] in {s.seed_id for s in aes}]
        jobs = []
        for c in okc:
            for i in range(2):
                jobs.append((c["cand_id"], f"absent:{i}", lambda c=c, i=i: ab.check_absent(dict(c, cand_id=c["cand_id"] + f"#a{i}", reference_key=c["answer"]), corpus, main_llm,
                                                                                         extra_spans=[corpus.get(c["gold_span_ids"][0])])))
            for i in range(3):
                jobs.append((c["cand_id"], f"twin:{i}", lambda c=c, i=i: check_verify(dict(c, cand_id=c["cand_id"] + f"#t{i}", question=c["question_with_datum"]), corpus, main_llm)))
        log.info("almost_enough checks: %d jobs", run_jobs(jobs, aek, a.workers))
    if a.stage in ("generate", "screen", "ae", "seeds"):
        return 0

    # ------------------------------------------------------------------ assemble
    return assemble(a, corpus, thr, allc, annotate, ck, aec, aek, aes, seeds, main_llm, cfg)


def funnel_for(rows_ann, types):
    gates = [("generated", lambda c: True), ("feasible (model + >=3 distractors + key)", lambda c: c["filters"]["feasible"]),
             ("structural (+ type-specific deterministic checks)", lambda c: c["filters"].get("twist_structural_ok")),
             ("no exercise/benchmark leakage, not a near-duplicate of a ViEduHop question", lambda c: c["filters"].get("leak_ok")),
             ("stem overlap below threshold (rule d)", lambda c: c["filters"].get("overlap_ok")),
             ("closed-book hard, both models <= 0.5 of 4 perms (rule c; not required for negation_exception)",
              lambda c: c["twist_type"] not in tw.CLOSED_BOOK_REQUIRED or c["filters"].get("closed_book_hard")),
             ("gold-only main model picks key in >=2/3 samples (rule a)", lambda c: c["filters"].get("gold_ok")),
             ("unique key: only key 'correct' in >=2/3 samples (negation: per-option truth in 2/2)", lambda c: c["filters"].get("gold_unique_ok")),
             ("near-miss span alone does not give the key (near_miss_lesson only)", lambda c: c["twist_type"] != "near_miss_lesson" or c["filters"].get("nm_ok")),
             ("every gold span necessary, leave-one-out (rule b; multi-span only)", lambda c: len(c["gold_span_ids"]) < 2 or c["filters"].get("loo_all_necessary")),
             ("independent small model with gold spans only picks the key in 2/2 permutations (negation_exception exempt: per-option truth instead)",
              lambda c: c["twist_type"] == "negation_exception" or c["filters"].get("indep_ok")),
             ("audit (main model, no book): self-contained, no other defensible option, not subjective", lambda c: c["filters"].get("audit_ok") is True),
             ("strict review, main model thinking mode, both of 2 option permutations", lambda c: c["filters"].get("strict_ok") is True)]
    out = {}
    for t in types:
        cur = [c for c in rows_ann if c["twist_type"] == t]
        out[t] = {}
        for name, fn in gates:
            cur = [c for c in cur if fn(c)]
            out[t][name] = len(cur)
    return out, [g[0] for g in gates]


def assemble(a, corpus, thr, allc, annotate, ck, aec, aek, aes, seeds, main_llm, cfg):
    types = [t for t in tw.TWIST_TYPES]
    ann = [annotate(c) for c in allc]
    fun, gate_names = funnel_for(ann, types)
    final = [c for c in ann if all(g(c) for g in [lambda c: c["filters"]["feasible"], lambda c: c["filters"].get("twist_structural_ok"), lambda c: c["filters"].get("leak_ok"),
                                                   lambda c: c["filters"].get("overlap_ok"),
                                                   lambda c: c["twist_type"] not in tw.CLOSED_BOOK_REQUIRED or c["filters"].get("closed_book_hard"),
                                                   lambda c: c["filters"].get("gold_ok"), lambda c: c["filters"].get("gold_unique_ok"),
                                                   lambda c: c["twist_type"] != "near_miss_lesson" or c["filters"].get("nm_ok"),
                                                   lambda c: len(c["gold_span_ids"]) < 2 or c["filters"].get("loo_all_necessary"),
                                                   lambda c: c["twist_type"] == "negation_exception" or c["filters"].get("indep_ok"),
                                                   lambda c: c["filters"].get("audit_ok") is True, lambda c: c["filters"].get("strict_ok") is True])]
    # dedup within the pool (question+answer char-3gram Jaccard >= 0.5, or same gold set + answer) -- order by hash only (system-blind)
    final.sort(key=lambda c: sha1(c["cand_id"]))
    kept, grams = [], []
    dropped_dup = 0
    for c in final:
        g = char_ngrams(c["question"] + " " + c["correct_answer"])
        if any(jaccard(g, x) >= 0.5 for x in grams) or any(set(c["gold_span_ids"]) == set(k["gold_span_ids"]) and c["twist_type"] == k["twist_type"] and
                                                           k["correct_answer"].lower() == c["correct_answer"].lower() for k in kept):
            dropped_dup += 1
            continue
        kept.append(c)
        grams.append(g)
    fun_after_dedup = Counter(c["twist_type"] for c in kept)
    # selection: per type up to max_per_type, round-robin over subjects, hash order inside a subject
    sel = []
    for t in types:
        pool = [c for c in kept if c["twist_type"] == t]
        by = defaultdict(list)
        for c in pool:
            by[c["domain"]].append(c)
        order = sorted(by, key=lambda d: (len(by[d]), d))
        chosen = []
        while len(chosen) < a.max_per_type and any(by.values()):
            for d in order:
                if by[d] and len(chosen) < a.max_per_type:
                    chosen.append(by[d].pop(0))
        sel += chosen
    log.info("selected %d: %s", len(sel), dict(Counter(c["twist_type"] for c in sel)))
    es = JsonlStore(T / "enrich.jsonl", key_fn=lambda r: r["cand_id"])
    rows = [to_row(c, corpus) for c in sel]
    pmap(lambda r: es.add({"cand_id": r["cand_id"], **v2.enrich_item(r, corpus, main_llm)}) if not es.done(r["cand_id"]) else None, rows, a.workers)
    cnt = Counter()
    byc = {c["cand_id"]: c for c in sel}
    for r in sorted(rows, key=lambda r: (r["twist_type"], r["domain"], r["cand_id"])):
        c, e = byc[r["cand_id"]], es.rows.get(r["cand_id"]) or {}
        k = (r["twist_type"], r["domain"])
        cnt[k] += 1
        r["id"] = f"VIEDUHOP-TW-{tw.TYPE_CODE[r['twist_type']]}-{r['domain'][:3].upper()}-{cnt[k]:03d}"
        r.update(bloom=e.get("bloom"), ped_types=e.get("ped_types"), ped_label_ws1={s: (corpus.get(s).pedagogical_type if corpus.get(s) else None) for s in r["gold_span_ids"]},
                 reference_answer=e.get("reference_answer"), explanation=e.get("explanation") or r["explanation"], quotes=e.get("quotes"),
                 quotes_verbatim=e.get("quotes_verbatim"), enrich_ok=bool(e.get("reference_answer")))
    rows.sort(key=lambda r: r["id"])
    rebalance_positions(rows)
    (B / "twist.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    # ---- almost_enough
    ab_rows, ae_fun = assemble_ae(corpus, aec, aek, aes, main_llm)
    (ROOT / "benchmarks/abstain/twist_abstain.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ab_rows), encoding="utf-8")
    # per-check independent stats among those that reached it
    indep = {}
    for t in types:
        cs = [c for c in ann if c["twist_type"] == t and c["filters"].get("overlap_ok") and c["filters"].get("leak_ok")]
        indep[t] = {"reached_closed_book_probe_small": sum(1 for c in cs if c["filters"].get("closed_small_acc") is not None),
                    "small_pass": sum(1 for c in cs if c["filters"].get("closed_small_acc") is not None and c["filters"]["closed_small_acc"] <= HARD_MAX),
                    "reached_main_probe": sum(1 for c in cs if c["filters"].get("closed_main_acc") is not None),
                    "main_pass": sum(1 for c in cs if c["filters"].get("closed_main_acc") is not None and c["filters"]["closed_main_acc"] <= HARD_MAX)}
    structural_fail = defaultdict(Counter)
    for c in ann:
        f = c["filters"]
        if f["feasible"] and not f.get("twist_structural_ok"):
            for k in f.get("twist_structural_fail", []):
                structural_fail[c["twist_type"]][k] += 1
    infeasible = defaultdict(Counter)
    for c in ann:
        if not c["filters"]["feasible"]:
            infeasible[c["twist_type"]]["error" if c.get("error") else c.get("infeasible_reason", "other")] += 1
    rep = {"built_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "benchmark_version": tw.BENCHMARK_VERSION,
           "funnel_by_type": fun, "gate_names": gate_names, "after_pool_dedup": fun_after_dedup, "dedup_dropped": dropped_dup,
           "selected": Counter(c["twist_type"] for c in sel), "selected_by_subject": Counter(c["domain"] for c in sel),
           "selected_by_type_subject": {f"{t}|{d}": n for (t, d), n in Counter((c["twist_type"], c["domain"]) for c in sel).items()},
           "selected_by_variant": {f"{t}|{v}": n for (t, v), n in Counter((c["twist_type"], variant_of(c)) for c in sel).items() if v not in (None, "none")},
           "closed_book_probe_stats": indep, "structural_failure_reasons": structural_fail, "infeasible_reasons": infeasible,
           "almost_enough_funnel": ae_fun, "n_seeds": Counter(s.strategy for s in seeds), "thresholds": thr,
           "models": {"generator": main_llm.model + " (FP8; :8004/:8005 replicas, latency-balanced)", "verifier": main_llm.model, "strict_review": main_llm.model + " (thinking mode)",
                      "closed_book_probe": ["small model", main_llm.model], "enrichment": main_llm.model},
           "prompt_versions": {"generation": tw.TWIST_GEN_VERSION, "almost_enough_generation": tw.TWIST_AE_GEN_VERSION, "almost_enough_generation_r1plus": tw.TWIST_AE_GEN_VERSION_R1, "strict_review": tw.TWIST_STRICT_VERSION, "independent_8b": bs.BS_INDEP_VERSION, "generation_r1plus": tw.TWIST_GEN_VERSION_R1, "audit": bs.BS_AUDIT_VERSION,
                               "seeds": tw.TWIST_SEEDS_VERSION, "overlap": tw.OVERLAP_VERSION},
           "corpus": corpus_manifest(corpus)}
    (T / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=dict))
    (T / "prompts.json").write_text(json.dumps({"versions": rep["prompt_versions"], **tw.PROMPT_TEXTS}, ensure_ascii=False, indent=1))
    # closed-book probe sidecar for headline_analysis.py --probe
    with open(T / "closed_book_probe.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            c = byc[r["cand_id"]]["filters"]
            f.write(json.dumps({"id": r["id"], "split": "twist", "main_acc": c["closed_main_acc"], "small_acc": c["closed_small_acc"],
                                "closed_book_hard": bool(c["closed_book_hard"]), "note": "filter-time probe (4 permutations), same rule as closed_book_probe.py"},
                               ensure_ascii=False) + "\n")
    print(json.dumps({"funnel": fun, "selected": rep["selected"]}, ensure_ascii=False, indent=1, default=dict))
    return 0


def rebalance_positions(rows, key_fn=lambda r: r["cand_id"]):
    """After selection (no check depends on it: every LLM check already used re-permuted options) the key position is cycled A,B,C,D over the
    items in hash order, so the answer-position distribution is balanced to within one item. Distractor order is kept."""
    order = sorted(rows, key=lambda r: sha1(key_fn(r)))
    for i, r in enumerate(order):
        pos = i % 4
        key = r["options"][r["answer"]]
        ds = [r["options"][k] for k in LETTERS if k != r["answer"]]
        new = ds[:pos] + [key] + ds[pos:]
        r["options"], r["answer"] = dict(zip(LETTERS, new)), LETTERS[pos]


def variant_of(c):
    """Generator-declared variant (validated in hard_twist.generate_twist) or, for misconception / near_miss_lesson, the deterministic regex tag
    'superlative_stem' when the stem asks for a single greatest/most/only element ('... nhất', 'duy nhất')."""
    v = c.get("twist_variant", "none")
    if v == "none" and c["twist_type"] in ("misconception", "near_miss_lesson") and tw.SUPERLATIVE.search(c["question"]):
        return "superlative_stem"
    return v


def to_row(c, corpus):
    spans = [corpus.get(s) for s in c["gold_span_ids"]]
    f = c["filters"]
    sa, ma = f.get("closed_small_acc"), f.get("closed_main_acc")
    g = lambda acc: {"correct": None if acc is None else int(round(acc * 4)), "n": None if acc is None else 4, "mean_conf": None}
    doms = sorted({s.domain for s in spans})
    diff = None if sa is None else ("hard" if sa == 0 else ("easy" if sa == 1 else "medium"))
    return {
        "id": "", "question": c["question"], "options": c["options"], "answer": c["answer"], "domain": c["domain"], "subject": DOMAIN_TO_SUBJECT_VI.get(c["domain"]),
        "grade": max((s.grade for s in spans if s.grade is not None), default=c.get("grade")), "grades": sorted({s.grade for s in spans if s.grade is not None}),
        "hop_type": c["hop_type"], "reasoning_type": c["hop_type"], "hop_count": len(c["gold_span_ids"]), "gold_span_ids": c["gold_span_ids"],
        "reasoning_steps": c.get("reasoning_steps", []), "link": c.get("twist_note", ""), "explanation": c.get("twist_note", ""),
        "book_ids": sorted({s.book_id for s in spans}), "lesson_keys": sorted({s.lesson_key for s in spans}), "seed_strategy": c["twist_type"],
        "seed_id": c["seed_id"], "cand_id": c["cand_id"], "difficulty": diff, "closed_book": {"small": g(sa), "main": g(ma)},
        "verification": {"answer": c["answer"], "confidence": None, "n_undetermined_distractors": None, "gold_only_samples_key": f.get("gold_n_key")},
        "generator": c.get("generator", {}), "review_status": "auto_verified", "corpus_version": corpus_manifest(corpus).get("corpus_version"),
        "benchmark_name": "ViEduHop", "benchmark_version": tw.BENCHMARK_VERSION, "subset": "twist", "subjects": doms, "origin": "generated",
        "hop_levels": len(c["gold_span_ids"]), "bloom": None, "ped_types": None, "ped_label_ws1": None, "reference_answer": None, "quotes": None,
        "quotes_verbatim": None, "enrich_ok": False, "calc": c.get("calc"), "distractor_rewritten": False, "book_specific": False,
        "where_gold": [{"span_id": s.span_id, "book_id": s.book_id, "book_title": s.book_title, "grade": s.grade, "lesson_key": s.lesson_key,
                        "lesson": " › ".join(s.heading_path[-3:]), "pdf_page": (s.pdf_page_index + 1) if s.pdf_page_index is not None else None,
                        "printed_page": s.printed_page_label if s.printed_page_verified else None} for s in spans],
        "split": "twist", "seen_in_development": False,
        # twist-specific
        "twist_type": c["twist_type"], "twist_variant": variant_of(c), "twist_note": c.get("twist_note", ""),
        "near_miss_span_ids": [c["near_miss_span_id"]] if c.get("near_miss_span_id") else [], "bridge_concept": c.get("bridge_concept"),
        "avoided_terms": c.get("avoided_terms"), "stem_overlap": f.get("stem_overlap"), "stem_overlap_threshold": f.get("stem_overlap_threshold"),
        "twist_checks": {"gold_only_key_votes": f.get("gold_n_key"), "gold_only_unique_votes": f.get("gold_n_unique"),
                         "closed_book_small_acc": sa, "closed_book_main_acc": ma, "closed_book_hard": bool(f.get("closed_book_hard")),
                         "closed_book_required": c["twist_type"] in tw.CLOSED_BOOK_REQUIRED, "loo_all_necessary": f.get("loo_all_necessary") if c["hop_type"] else None,
                         "near_miss_only_key_votes": f.get("nm_n_key"), "strict_review": True},
        "twist_generator": {"model": c["generator"]["model"], "prompt_version": c["generator"]["prompt_version"]},
    }


def assemble_ae(corpus, aec, aek, aes, main_llm):
    sids = {s.seed_id for s in aes}
    cs = [c for c in aec.rows.values() if c.get("seed_id") in sids]
    fun = {"seeds": len(aes), "generated": len([c for c in cs if not c.get("error")]), "feasible_wellformed": len([c for c in cs if c.get("feasible") and c.get("options")]),
           "failed_reasons": dict(Counter(c.get("failed") or ("error" if c.get("error") else None) for c in cs if not c.get("feasible")))}
    g = lambda c, n: (aek.rows.get(f"{c['cand_id']}|{n}") or {}).get("result")
    good = []
    n_late = 0
    for c in cs:
        if not (c.get("feasible") and c.get("options")):
            continue
        if tw.AE_GIVEAWAY.search(c["question"]) or any(tw.OPT_PREFIX.match(x) for x in c["options"].values()):
            n_late += 1                        # round-r0 candidates generated before these two filters existed: dropped here, counted below
            continue
        ab = [g(c, f"absent:{i}") for i in range(2)]
        tw_ = [g(c, f"twin:{i}") for i in range(3)]
        if any(x is None for x in ab + tw_):
            continue
        c["_absent_ok"] = all(x["accepted"] for x in ab)
        k = c["answer"]
        c["_twin_n"] = sum(1 for x in tw_ if x["answer"] == k)
        c["_twin_unique"] = sum(1 for x in tw_ if x["option_status"].get(k) == "correct" and all(x["option_status"].get(o) != "correct" for o in LETTERS if o != k))
        c["_twin_ok"] = c["_twin_n"] >= 2 and c["_twin_unique"] >= 2
        good.append(c)
    fun["dropped_giveaway_or_option_prefix_late_filter"] = n_late
    fun["checks_complete"] = len(good)
    a1 = [c for c in good if c["_absent_ok"]]
    fun["no_option_determined_by_gold+top10_in_both_perms"] = len(a1)
    a2 = [c for c in a1 if c["_twin_ok"]]
    fun["twin_control_answerable_from_gold_with_datum(>=2/3 key, unique)"] = len(a2)
    a2.sort(key=lambda c: sha1(c["cand_id"]))
    kept, grams = [], []
    for c in a2:
        gm = char_ngrams(c["question"])
        if any(jaccard(gm, x) >= 0.5 for x in grams):
            continue
        kept.append(c)
        grams.append(gm)
    fun["after_dedup"] = len(kept)
    by = defaultdict(list)
    for c in kept:
        by[c["domain"]].append(c)
    sel = []
    while len(sel) < 15 and any(by.values()):
        for d in sorted(by, key=lambda d: (len(by[d]), d)):
            if by[d] and len(sel) < 15:
                sel.append(by[d].pop(0))
    fun["selected_unanswerable"] = len(sel)
    fun["selected_by_subject"] = dict(Counter(c["domain"] for c in sel))
    from otter.evaluation.abstain_build import ABBR
    cnt = Counter()
    rows = []
    for c in sorted(sel, key=lambda c: (c["domain"], c["cand_id"])):
        cnt[c["domain"]] += 1
        uid = f"OTTER-ABS-{ABBR[c['domain']]}-AEN-{cnt[c['domain']]:03d}"
        cid_ = f"OTTER-ABS-{ABBR[c['domain']]}-AEN-C-{cnt[c['domain']]:03d}"
        sp = corpus.get(c["gold_span_ids"][0])
        a1_ = g(c, "absent:0")
        prov = {"source": "almost_enough", "missing_datum": c.get("missing_datum"), "reference_key": c["answer"],
                "generator": {"model": c["generator"]["model"], "version": c["generator"]["version"]},
                "absence_check": {"judge_status": a1_["status"], "judge_model": a1_["model"], "evidence_span_ids": a1_["evidence_ids"],
                                  "accepted_in_both_permutations": True},
                "twin_check": {"key_votes_of_3": c["_twin_n"], "unique_votes_of_3": c["_twin_unique"]},
                "corpus_version": "v1-core", "benchmark_version": tw.BENCHMARK_VERSION, "twist_type": "almost_enough",
                "verification": "auto_verified (no teacher review)"}
        base = {"subject": DOMAIN_TO_SUBJECT_VI[c["domain"]], "domain": c["domain"], "grade": c["grade"], "options": c["options"],
                "gold_span_ids": c["gold_span_ids"], "excluded_span_ids": [], "reasoning_type": "almost_enough", "hop_count": 1,
                "lesson_keys": c["lesson_keys"], "review_status": "unreviewed", "split": "twist"}
        rows.append({"id": uid, **base, "question": c["question"], "answerable": False, "answer": None, "abstain_type": "almost_enough", "control_type": None,
                     "provenance": dict(prov, twin_id=cid_)})
        rows.append({"id": cid_, **base, "question": c["question_with_datum"], "answerable": True, "answer": c["answer"], "abstain_type": None,
                     "control_type": "almost_enough", "provenance": dict(prov, twin_id=uid)})
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["id"].replace("-C-", "-")].append(r)
    for i, (k, rs) in enumerate(sorted(pairs.items(), key=lambda kv: sha1(kv[0]))):
        ctl = next(r for r in rs if r["answerable"])
        pos = i % 4
        key = ctl["options"][ctl["answer"]]
        ds = [ctl["options"][o] for o in LETTERS if o != ctl["answer"]]
        new = dict(zip(LETTERS, ds[:pos] + [key] + ds[pos:]))
        for r in rs:
            r["options"] = dict(new)
            if r["answerable"]:
                r["answer"] = LETTERS[pos]
            else:
                r["provenance"]["reference_key"] = LETTERS[pos]
    rows.sort(key=lambda r: r["id"])
    return rows, fun


if __name__ == "__main__":
    raise SystemExit(main())
