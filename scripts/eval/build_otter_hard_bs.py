"""Build the ViEduHop `book_specific` pool (hard, tricky, textbook-specific items). Resumable; every stage appends to JSONL stores.

  .venv/bin/python scripts/eval/build_otter_hard_bs.py --stage all [--limit-seeds N]
Stages: seeds, generate, filter (deterministic + length rewrite), probe (closed-book 8B then 32B), verify, final (annotate, dedup, enrich, pool).
Output: benchmarks/otter_hard/work_bs/annotated_pool.jsonl (+ funnel.json, stats via scripts/eval/otter_hard_bs_stats.py).
Concurrency policy: main (:8004) is shared -> --main-workers <= 10; small (:8001) --small-workers.
"""
import argparse, glob, json, logging, random, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation import book_specific as bs, hard_build as hb, hard_v2 as v2  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.hard_gen import (JsonlStore, LeakageIndex, check_closed, check_key, check_quality, check_single,  # noqa: E402
                                       check_verify, structural_checks, summarize_checks)
from otter.evaluation.hard_seeds import load_seeds, write_seeds  # noqa: E402
from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.evaluation.text import sha1  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402

log = logging.getLogger("bs")
HARD_MAX = 0.5      # closed_book_hard: accuracy over the 4 option permutations must be <= 0.5 for BOTH models


def run_named(jobs, store, workers):
    """jobs: list of (cand_id, check_name, fn). Skips those already in the store; errors are stored and retried on the next run."""
    todo = [j for j in jobs if not store.done(f"{j[0]}|{j[1]}")]

    def one(j):
        cid, name, fn = j
        try:
            row = {"cand_id": cid, "check": name, "result": fn()}
        except Exception as e:  # noqa: BLE001
            row = {"cand_id": cid, "check": name, "error": f"{type(e).__name__}: {e}"}
        store.add(row)
    t0 = time.time()
    with ThreadPoolExecutor(max(1, workers)) as ex:
        list(ex.map(one, todo))
    log.info("ran %d check jobs in %.0fs", len(todo), time.time() - t0)


def acc_of(c, checks, cid, role, n=4):
    rs = [(checks.get(f"{cid}|closed:{role}:{p}") or {}).get("result") for p in range(n)]
    if any(r is None for r in rs):
        return None
    return sum(1 for r in rs if r["answer"] == c["answer"]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seeds", "generate", "filter", "probe", "verify", "final", "all"])
    ap.add_argument("--work-dir", default=str(ROOT / "benchmarks/otter_hard/work_bs"))
    ap.add_argument("--per-domain", type=int, default=0, help="max seeds per subject for this round (0 = all eligible spans)")
    ap.add_argument("--round", default="0", help="comma-separated seed rounds (each round = a different archetype per span)")
    ap.add_argument("--domains", default="")
    ap.add_argument("--hop-mix", default="0.58,0.30,0.12", help="share of 1/2/3-span seeds when a seed round is created")
    ap.add_argument("--limit-seeds", type=int, default=0, help="use only the first N (shuffled) seeds; 0 = all")
    ap.add_argument("--gen-model", default="main", choices=["main", "small"])
    ap.add_argument("--tier2-domains", default="physics,chemistry", help="subjects that also get the 'evidence_only' tier (not closed_book_hard)")
    ap.add_argument("--tier2-cap", type=int, default=220, help="max tier-2 candidates per subject sent to verification (lowest closed-book accuracy first)")
    ap.add_argument("--strict-role", default="index", help="LLM role for the thinking-mode final review (index = 2nd main model replica :8005)")
    ap.add_argument("--strict-workers", type=int, default=12)
    ap.add_argument("--rw-attempts", type=int, default=4)
    ap.add_argument("--reverse", action="store_true", help="process the rewrite todo list from the end (second worker process)")
    ap.add_argument("--rewrite-model", default="small", choices=["main", "small"])
    ap.add_argument("--main-workers", type=int, default=10)
    ap.add_argument("--small-workers", type=int, default=32)
    ap.add_argument("--min-indep", type=int, default=2, help="independent 8B gold-span check: need >= this many correct of 2")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    assert a.main_workers <= 24, "main :8004 is shared: coordinator allows <= ~24 concurrent requests in total for this job"
    W = Path(a.work_dir)
    W.mkdir(parents=True, exist_ok=True)
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    main_llm, small_llm = get_llm("main"), get_llm("small")
    gen_llm = main_llm if a.gen_model == "main" else small_llm
    gen_workers = a.main_workers if a.gen_model == "main" else a.small_workers
    rw_llm = main_llm if a.rewrite_model == "main" else small_llm      # length-matching rewrite (re-verified later by the 32B)

    # ---------------------------------------------------------------- seeds
    seeds = []
    for rnd in [int(x) for x in str(a.round).split(",")]:
        seeds_p = W / f"seeds_bs_r{rnd}.jsonl"
        if not seeds_p.exists():
            pageq = {}
            for p in glob.glob(str(ROOT / "data/textbooks/parsed/nxbgd_*/pages.jsonl")):
                for l in open(p, encoding="utf-8"):
                    r = json.loads(l)
                    pageq[(r["book_id"], r["pdf_page_index"])] = r.get("quality_score")
            from otter.evaluation.hard_seeds import load_span_concepts
            cfg = load_config()
            sc = load_span_concepts(ROOT / cfg["paths"]["ontology_db"])
            rs = bs.build_seeds(corpus, pageq, sc, round=rnd, domains=a.domains.split(",") if a.domains else None, max_per_domain=a.per_domain or None,
                               hop_mix=tuple(float(x) for x in a.hop_mix.split(",")))
            random.Random(5 + rnd).shuffle(rs)
            write_seeds(rs, seeds_p)
            log.info("round %d: wrote %d seeds", rnd, len(rs))
        seeds += load_seeds(seeds_p)
    random.Random(11).shuffle(seeds)
    if a.limit_seeds:
        seeds = seeds[:a.limit_seeds]
    funnel = {"seeds": len(seeds)}
    if a.stage == "seeds":
        return

    # ---------------------------------------------------------------- generate
    cs = JsonlStore(W / "cands_bs.jsonl", key_fn=lambda r: r["cand_id"])
    if a.stage in ("generate", "all"):
        todo = [s for s in seeds if not cs.done("cand_" + sha1(s.seed_id + bs.BS_GEN_VERSION + gen_llm.model + "0")[:12])]
        log.info("generating for %d seeds (%s, %d workers)", len(todo), gen_llm.model, gen_workers)

        def gen(sd):
            for c in bs.generate_bs(sd, corpus, gen_llm):
                cs.add(c)
        with ThreadPoolExecutor(gen_workers) as ex:
            list(ex.map(gen, todo))
    seed_ids = {s.seed_id for s in seeds}
    cands = [c for c in cs.rows.values() if c.get("seed_id") in seed_ids]
    funnel["candidates_generated"] = len([c for c in cands if not c.get("error")])
    funnel["feasible"] = len([c for c in cands if c.get("feasible") and c.get("options")])

    # ---------------------------------------------------------------- filter: deterministic + length-matched copy
    cfg = load_config()
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / cfg["paths"]["benchmark"])])
    ck = JsonlStore(W / "checks_bs.jsonl", key_fn=check_key)
    fs = JsonlStore(W / "finals_bs.jsonl", key_fn=lambda r: r["cand_id"])

    def det(c):
        c = dict(c)
        for k in ("filters",):
            c.pop(k, None)
        f = structural_checks(c, corpus)
        f.update(bs.bs_structural(c, corpus))
        if f["structural_ok"]:
            f.update(leak.check(c))
        cc = v2.check_calc(c)
        f["calc_ok"] = cc["calc_ok"] if cc["calc_present"] else None
        f["calc_gate_ok"] = (cc["calc_ok"] is True) if cc["calc_present"] else True
        c["filters"] = f
        return c
    feas = [det(c) for c in cands if c.get("feasible") and c.get("options")]
    # hard_gen.structural_checks length rule is a hard fail at ratio > 1.8 only; our stricter length_ok is handled by rewrite
    FIXABLE = {"no_length_cue"}          # length problems are repaired by the length-matching rewrite and re-checked on the final copy
    ok = [c for c in feas if set(c["filters"].get("structural_fail", [])) <= FIXABLE and c["filters"]["bs_structural_ok"]
          and (c["filters"].get("leak_ok") is not False) and c["filters"]["calc_gate_ok"]]
    funnel["structural_ok"] = len(ok)
    if a.stage in ("filter", "all"):
        need_rw = {c["cand_id"] for c in ok if c["trick_type"] != "statement_combo" and
                   (v2.needs_rewrite(c) or "no_length_cue" in c["filters"].get("structural_fail", []))}
        funnel["needs_rewrite"] = len(need_rw)

        def fin(c):
            base = {k: v for k, v in c.items() if k != "filters"}
            n = v2.rewrite_distractors(base, corpus, rw_llm, attempts=a.rw_attempts) if c["cand_id"] in need_rw else v2.finalize_copy(base)
            fs.add(n or {"cand_id": c["cand_id"] + "_F_failed", "failed": "rewrite"})
        todo = [c for c in ok if c["cand_id"] + "_F" not in fs.rows and c["cand_id"] + "_F_failed" not in fs.rows]
        # rewrite queue: probe the RAW options with the 8B (cheap, 4 permutations) and rewrite only items the small model does not always get
        # right, hardest first (the closed-book rule is re-applied on the rewritten copy, so this only saves rewrite calls)
        rw_todo = [c for c in todo if c["cand_id"] in need_rw]
        run_named([(c["cand_id"], f"closed:small:{p}", (lambda c=c, p=p: check_closed(c, small_llm, p))) for c in rw_todo for p in range(4)],
                  ck, a.small_workers)
        raw = {c["cand_id"]: acc_of(c, ck.rows, c["cand_id"], "small") for c in rw_todo}
        funnel["rewrite_skipped_raw_8B_all_correct"] = sum(1 for v in raw.values() if v is not None and v > 0.75)
        todo = [c for c in todo if c["cand_id"] not in need_rw or (raw.get(c["cand_id"]) is not None and raw[c["cand_id"]] <= 0.75)]
        todo.sort(key=lambda c: raw.get(c["cand_id"], 0.0) if c["cand_id"] in need_rw else -1.0)
        if a.reverse:
            todo = todo[::-1]
        with ThreadPoolExecutor(a.small_workers if rw_llm is small_llm else a.main_workers) as ex:
            list(ex.map(fin, todo))
    parent_ids = {c["cand_id"] for c in cands}
    finals = [det(c) for c in fs.rows.values() if not c.get("failed") and c.get("parent_cand_id") in parent_ids]
    finals = [c for c in finals if c["filters"]["structural_ok"] and c["filters"]["bs_structural_ok"] and c["filters"].get("leak_ok") and c["filters"]["calc_gate_ok"]
              and (c["trick_type"] == "statement_combo" or not bs.unique_longest(c["options"], c["answer"]))]
    funnel["finals_length_balanced"] = len(finals)

    # ---------------------------------------------------------------- probe: closed-book 8B then 32B (no retrieval)
    if a.stage in ("probe", "all"):
        run_named([(c["cand_id"], f"closed:small:{p}", (lambda c=c, p=p: check_closed(c, small_llm, p))) for c in finals for p in range(4)],
                  ck, a.small_workers)
        surv = [c for c in finals if (acc_of(c, ck.rows, c["cand_id"], "small") is not None and acc_of(c, ck.rows, c["cand_id"], "small") <= HARD_MAX)]
        log.info("8B closed-book <= %.2f: %d / %d", HARD_MAX, len(surv), len(finals))
        run_named([(c["cand_id"], f"closed:main:{p}", (lambda c=c, p=p: check_closed(c, main_llm, p))) for c in surv for p in range(4)],
                  ck, a.main_workers)
    def tier2_list(hard_ids):
        out_ = []
        for dom in [d for d in a.tier2_domains.split(",") if d and a.tier2_cap > 0]:
            cs = [c for c in finals if c["domain"] == dom and c["cand_id"] not in hard_ids and acc_of(c, ck.rows, c["cand_id"], "small") is not None
                  and not (acc_of(c, ck.rows, c["cand_id"], "main") is not None and acc_of(c, ck.rows, c["cand_id"], "main") <= HARD_MAX
                           and acc_of(c, ck.rows, c["cand_id"], "small") <= HARD_MAX)]
            cs.sort(key=lambda c: (acc_of(c, ck.rows, c["cand_id"], "small"), sha1(c["cand_id"])))
            out_ += cs[:a.tier2_cap]
        return out_
    if a.stage in ("probe", "all"):
        _h = {c["cand_id"] for c in finals if (acc_of(c, ck.rows, c["cand_id"], "small") or 9) <= HARD_MAX and (acc_of(c, ck.rows, c["cand_id"], "main") if acc_of(c, ck.rows, c["cand_id"], "main") is not None else 9) <= HARD_MAX}
        run_named([(c["cand_id"], f"closed:main:{p}", (lambda c=c, p=p: check_closed(c, main_llm, p))) for c in tier2_list(_h) for p in range(4)],
                  ck, a.main_workers)
    small_ok = [c for c in finals if (acc_of(c, ck.rows, c["cand_id"], "small") is not None and acc_of(c, ck.rows, c["cand_id"], "small") <= HARD_MAX)]
    hard = [c for c in small_ok if (acc_of(c, ck.rows, c["cand_id"], "main") is not None and acc_of(c, ck.rows, c["cand_id"], "main") <= HARD_MAX)]
    funnel["closed_small_le_0.5"] = len(small_ok)
    funnel["closed_book_hard_both"] = len(hard)
    hard_ids = {c["cand_id"] for c in hard}
    tier2 = tier2_list(hard_ids)
    tier2_ids = {c["cand_id"] for c in tier2}
    funnel["tier2_candidates"] = len(tier2)
    verify_set = hard + tier2

    # ---------------------------------------------------------------- verify (survivors only)
    if a.stage in ("verify", "all"):
        jobs = []
        for c in verify_set:
            cid = c["cand_id"]
            if c["trick_type"] == "negation":      # 'KHÔNG đúng' stems: per-option truth labels, not "which option is the answer"
                jobs.append((cid, "optruth:0", lambda c=c: bs.check_option_truth(c, corpus, main_llm, 0)))
                jobs.append((cid, "optruth:1", lambda c=c: bs.check_option_truth(c, corpus, main_llm, 1)))
            else:
                jobs.append((cid, "verify", lambda c=c: check_verify(c, corpus, main_llm)))
                jobs.append((cid, "verify2", lambda c=c: check_verify(dict(c, cand_id=c["cand_id"] + "#2"), corpus, main_llm)))
            jobs.append((cid, "audit", lambda c=c: bs.check_audit(c, main_llm)))
            if len(c["gold_span_ids"]) > 1:
                jobs.append((cid, "quality", lambda c=c: check_quality(c, corpus, main_llm)))
                for s in c["gold_span_ids"]:
                    jobs.append((cid, f"single:{s}", lambda c=c, s=s: check_single(c, s, corpus, main_llm)))
            if c["trick_type"] == "statement_combo":
                jobs.append((cid, "statements", lambda c=c: bs.check_statements(c, corpus, main_llm)))
        run_named(jobs, ck, a.main_workers)
        run_named([(c["cand_id"], f"indep:{p}", (lambda c=c, p=p: bs.check_independent(c, corpus, small_llm, p))) for c in verify_set for p in range(2)
                   if c["trick_type"] != "negation"],
                  ck, a.small_workers)

    # ---------------------------------------------------------------- annotate + gates
    ann = hb.annotate_candidates(finals, ck.rows, corpus, leak, max_undetermined=3)
    out = []
    for c in ann:
        f, cid = c["filters"], c["cand_id"]
        r = lambda n: (ck.rows.get(f"{cid}|{n}") or {}).get("result")
        f["closed_small_acc"] = acc_of(c, ck.rows, cid, "small")
        f["closed_main_acc"] = acc_of(c, ck.rows, cid, "main")
        f["closed_book_hard"] = bool(f["closed_small_acc"] is not None and f["closed_main_acc"] is not None
                                     and f["closed_small_acc"] <= HARD_MAX and f["closed_main_acc"] <= HARD_MAX)
        v2_ = r("verify2")
        if c["trick_type"] == "negation":
            ot = [r("optruth:0"), r("optruth:1")]
            f["verify_ok_generic"] = f.get("verify_ok")
            f["verify_ok"] = None if any(x is None for x in ot) else all(x["ok"] for x in ot)     # assemble.good() reads verify_ok
            f["verify2_ok"] = f["verify_ok"]
        elif v2_ is not None:
            st = v2_.get("option_status") or {}
            f["verify2_ok"] = v2_.get("answer") == c["answer"] and st.get(c["answer"]) == "correct" and \
                not any(st.get(k) == "correct" for k in "ABCD" if k != c["answer"])
        au = r("audit")
        f["audit_ok"] = None if au is None else bool(au["ok"])
        f["audit_self_contained"] = None if au is None else au["self_contained"]
        ind = [r(f"indep:{p}") for p in range(2)]
        if all(x is not None for x in ind):
            f["indep_correct"] = sum(1 for x in ind if x["correct"])
            f["indep_ok"] = f["indep_correct"] >= a.min_indep
        if c["trick_type"] == "statement_combo":
            sr = r("statements")
            f["statements_ok"] = None if sr is None else bool(sr["match"])
        f["bs_gates_ok"] = bool(f.get("verify_ok") is True and f.get("verify2_ok") is True and (f.get("indep_ok") is True or c["trick_type"] == "negation")
                              and f.get("audit_ok") is True and f.get("multi_hop_ok") is True and (f.get("quality_ok", True) is not False)
                              and (c["trick_type"] != "statement_combo" or f.get("statements_ok") is True))
        c["book_specific_tier"] = "closed_book_hard" if f["closed_book_hard"] else ("evidence_only" if cid in tier2_ids and f["closed_small_acc"] is not None
                                                                                  and f["closed_main_acc"] is not None else None)
        f["bs_all_ok"] = bool(f["bs_gates_ok"] and c["book_specific_tier"] is not None)
        c["needs_teacher_numeric_check"] = bs.needs_numeric_check(c, corpus)
        c["book_specific"] = True
        out.append(c)
    final_ok = [c for c in out if c["filters"]["bs_all_ok"]]
    funnel["verified_32B_two_perms"] = len([c for c in out if c["filters"]["closed_book_hard"] and c["filters"].get("verify_ok") is True and c["filters"].get("verify2_ok") is True])
    funnel["independent_8B_ok"] = len([c for c in out if c["filters"]["closed_book_hard"] and c["filters"].get("verify_ok") is True
                                       and c["filters"].get("verify2_ok") is True and c["filters"].get("indep_ok") is True])
    funnel["all_gates"] = len(final_ok)
    if a.stage in ("final", "all"):
        kept, dups = hb.dedup(final_ok, thr=0.5)
        funnel["after_lexical_dedup"] = len(kept)
        kept, edups = embed_dedup(kept, corpus)
        funnel["after_embedding_dedup"] = len(kept)
        # strict final review: main model in THINKING mode, gold spans + question: key supported, unique (also against general knowledge),
        # question clear, facts accurate, options distinct. Only on items that passed every other gate.
        from otter.llm.interface import OpenAICompatLLM
        mc = load_config()["llm"][a.strict_role]
        think_llm = OpenAICompatLLM(base_url=mc["base_url"], model=mc["model"], enable_thinking=True)
        run_named([(c["cand_id"], "strict", (lambda c=c: bs.check_strict_review(c, corpus, think_llm))) for c in kept], ck, a.strict_workers)
        for c in kept:
            c["filters"]["strict_ok"] = ((ck.rows.get(c["cand_id"] + "|strict") or {}).get("result") or {}).get("ok")
            c["strict_review"] = (ck.rows.get(c["cand_id"] + "|strict") or {}).get("result")
        kept = [c for c in kept if c["filters"]["strict_ok"] is True]
        funnel["after_strict_review_32B_thinking"] = len(kept)
        # enrichment: Bloom label, reference answer + explanation (Vietnamese), per-span verbatim quotes
        rows = hb.to_rows(kept, corpus, None)
        es = JsonlStore(W / "enrich_bs.jsonl", key_fn=lambda r: r["cand_id"])
        todo = [r for r in rows if not es.done(r["cand_id"])]
        log.info("enriching %d items", len(todo))
        with ThreadPoolExecutor(a.main_workers) as ex:
            list(ex.map(lambda r: es.add({"cand_id": r["cand_id"], **v2.enrich_item(r, corpus, main_llm)}), todo))
        for c in kept:
            c["enrich"] = es.rows.get(c["cand_id"])
            c["book_specific_meta"] = {
                "trick_type": c["trick_type"], "closed_book": {"small_acc": c["filters"]["closed_small_acc"], "main_acc": c["filters"]["closed_main_acc"],
                                                               "rule": "closed_book_hard: both <= 0.5 over 4 option permutations, no retrieval", "tier": c["book_specific_tier"]},
                "models": {"generator": c["generator"]["model"], "generator_prompt": c["generator"]["prompt_version"],
                           "verifier_gold_spans": main_llm.model, "strict_review": think_llm.model + " (thinking, FP8)", "independent_check": small_llm.model, "closed_book_probe": [small_llm.model, main_llm.model]}}
        kept = [c for c in kept if (c.get("enrich") or {}).get("reference_answer")]
        funnel["after_enrichment"] = len(kept)
        funnel["pool_by_tier"] = dict(Counter(c["book_specific_tier"] for c in kept))
        funnel["pool_by_subject_and_tier"] = {d: dict(Counter(c["book_specific_tier"] for c in kept if c["domain"] == d)) for d in sorted({c["domain"] for c in kept})}
        with open(W / "annotated_pool.jsonl", "w", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        (W / "funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1))
    print(json.dumps(funnel, ensure_ascii=False, indent=1))


def embed_dedup(kept, corpus, thr=0.88):
    """Drop near-duplicate questions (bge-m3 cosine >= thr, same domain), keeping the earlier (harder) one."""
    import numpy as np
    try:
        from otter.evaluation.embeddings import get_embedder
        emb = get_embedder()
        v = emb.embed([c["question"] + " " + c["correct_answer"] for c in kept])
    except Exception as e:  # noqa: BLE001
        log.warning("embedding dedup skipped: %s", e)
        return kept, []
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    keep_idx, dropped = [], []
    for i, c in enumerate(kept):
        hit = next((j for j in keep_idx if kept[j]["domain"] == c["domain"] and float(v[i] @ v[j]) >= thr), None)
        if hit is None:
            keep_idx.append(i)
        else:
            dropped.append((c["cand_id"], kept[hit]["cand_id"]))
    return [kept[i] for i in keep_idx], dropped


if __name__ == "__main__":
    main()
