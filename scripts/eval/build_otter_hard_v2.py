"""OTTER-Hard v2 build (resumable stages). v1 files are snapshotted in benchmarks/work/otter_hard_v1_snapshot.

  seeds    ontology rule seeds + relation-chain seeds + subject-quota concept-linked seeds  -> work_v2/seeds.jsonl
  generate new seeds (gen prompt v3) with the main model; v1 candidates are reused as the legacy pool
  round1   verify / quality / single-span checks for the new candidates (no closed-book yet)
  fixlen   rewrite distractors to match the key's length (+-20%) for survivors -> work_v2/final_cands.jsonl
  final    full re-verification of the final candidates (verify, quality, single, closed-book x4 small/main)
  assemble filters, dedup, balanced selection, dev/test, STATS.md, review sheet
  bias     answer-only baselines on the assembled benchmark

  .venv/bin/python scripts/eval/build_otter_hard_v2.py --stage all --workers 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation import hard_build as hb  # noqa: E402
from otter.evaluation import hard_v2 as v2  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter, corpus_manifest  # noqa: E402
from otter.evaluation.embeddings import get_embedder  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, LeakageIndex, check_key, run_checks, run_generation  # noqa: E402
from otter.evaluation.hard_seeds import SeedSampler, load_seeds, load_span_concepts, seed_stats, write_seeds  # noqa: E402
from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.evaluation.text import sha1  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402

log = logging.getLogger("otter.hard2")
V1 = ROOT / "benchmarks/work/otter_hard_v1_snapshot/work"
PER_DOMAIN = {"physics": 60, "chemistry": 50, "literature": 30, "geography": 30, "biology": 12, "history": 5}


def calc_gate(c: dict) -> bool:
    cc = v2.check_calc(c)
    c["calc_check"] = cc
    if c.get("hop_type") == "calculation" or cc["calc_present"]:
        return cc["calc_ok"] is True
    return True


def survivors(ann: list[dict], qmin: tuple[int, int, int]) -> list[dict]:
    out = []
    for c in ann:
        f = c["filters"]
        if not (f.get("structural_ok") and f.get("leak_ok") and f.get("verify_ok") is True and f.get("multi_hop_ok") is True):
            continue
        if "q_natural" in f and not (f["q_natural"] >= qmin[0] and f["q_needs_both"] >= qmin[1] and f["q_distractors"] >= qmin[2]):
            continue
        if "q_natural" not in f:
            continue
        if not c.get("calc_gate_ok", True):
            continue
        out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seeds", "generate", "round1", "fixlen", "final", "assemble", "bias", "all"])
    ap.add_argument("--work-dir", default=str(ROOT / "benchmarks/otter_hard/work_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks/otter_hard"))
    ap.add_argument("--sheet-dir", default=str(ROOT / "annotation/otter_hard_review"))
    ap.add_argument("--ontology-db", default=str(ROOT / "data/ontology/ontology.sqlite3"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--q-round1", default="3,3,3")
    ap.add_argument("--q-final", default="3,4,3", help="natural,needs_both,distractors minimum scores (quality, not correctness)")
    ap.add_argument("--max-undetermined", type=int, default=3)
    ap.add_argument("--dev-size", type=int, default=40)
    ap.add_argument("--total-target", type=int, default=340)
    ap.add_argument("--domain-floor", type=int, default=45)
    ap.add_argument("--domain-cap", type=int, default=75)
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--review-n", type=int, default=150)
    ap.add_argument("--closed-perms", type=int, default=4)
    ap.add_argument("--seeds-name", default="seeds.jsonl")
    ap.add_argument("--no-legacy", action="store_true", help="do not copy the v1 candidates/checks into the work dir")
    ap.add_argument("--pool-only", action="store_true", help="stop after the final checks and write annotated_pool.jsonl (no benchmark files)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    work, out = Path(a.work_dir), Path(a.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    q1 = tuple(int(x) for x in a.q_round1.split(","))
    qf = tuple(int(x) for x in a.q_final.split(","))
    corpus = SpanCorpus.load(parsed_dir=ROOT / "data/textbooks/parsed", db=ROOT / "data/textbooks/ontology.sqlite3",
                             registry=ROOT / "data/textbooks/registry", min_chars=1, book_filter=core_book_filter)
    log.info("corpus %d spans", len(corpus))
    seeds_p, cand_p, chk_p = work / a.seeds_name, work / "candidates.jsonl", work / "checks.jsonl"
    final_p, fchk_p = work / "final_cands.jsonl", work / "checks_final.jsonl"
    for src, dst in ((V1 / "candidates.jsonl", cand_p), (V1 / "checks.jsonl", chk_p)):
        if not dst.exists() and not a.no_legacy:
            shutil.copy(src, dst)          # legacy pool: v1 candidates and their (v1) check records

    # ---------------------------------------------------------------- seeds
    if a.stage in ("seeds", "all") and not seeds_p.exists():
        onto = v2.Onto(Path(a.ontology_db))
        old = load_seeds(V1 / "seeds.jsonl")
        new = v2.rule_seeds(corpus, onto) + v2.relation_chain_seeds(corpus, onto)
        corpus.attach_embedder(get_embedder())
        sampler = SeedSampler(corpus, span_concepts=load_span_concepts(Path(a.ontology_db)), seed=a.seed + 100)
        for sd in old + new:
            sampler._commit(sd.span_ids)
        gen = sampler.sample(per_cell=10, per_domain=PER_DOMAIN)
        seen = {s.seed_id for s in old}
        allnew = [s for s in new + gen if not (s.seed_id in seen or seen.add(s.seed_id))]
        write_seeds(allnew, seeds_p)
        st = seed_stats(allnew)
        st["by_domain_hop"] = {f"{d}|{h}": n for (d, h), n in Counter((s.domain, s.target_hop_type) for s in allnew).items()}
        (work / "seeds_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
        log.info("new seeds: %s", st)
    seeds = load_seeds(seeds_p) if seeds_p.exists() else []
    cands_store = JsonlStore(cand_p, key_fn=lambda r: r["cand_id"])
    main_llm = get_llm("main")
    small_llm = get_llm("small")

    # ---------------------------------------------------------------- generate
    if a.stage in ("generate", "all"):
        log.info("generating %d seeds with %s", len(seeds), main_llm.model)
        run_generation(seeds, corpus, main_llm, cands_store, workers=a.workers, extra_fn=v2.extra_prompt,
                       prompt_version=v2.GEN_V3_VERSION)
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / cfg["paths"]["benchmark"])])
    checks = JsonlStore(chk_p, key_fn=check_key)
    all_cands = [c for c in cands_store.rows.values() if not c.get("error")]

    def prep(cs):
        for c in cs:
            c["calc_gate_ok"] = calc_gate(c)
        return cs

    # ---------------------------------------------------------------- round 1
    if a.stage in ("round1", "all"):
        pre = hb.annotate_candidates(prep(list(all_cands)), {}, corpus, leak)
        todo = [c for c in pre if c["filters"]["structural_ok"] and c["filters"].get("leak_ok") and c.get("calc_gate_ok", True)]
        log.info("round1 checks for %d/%d candidates", len(todo), len(all_cands))
        run_checks(todo, corpus, checks, main=main_llm, small=None, workers=a.workers, closed_main=False, closed_perms=0)
    ann1 = hb.annotate_candidates(prep(list(all_cands)), checks.rows, corpus, leak, max_undetermined=a.max_undetermined)
    surv = survivors(ann1, q1)
    log.info("round1 survivors: %d (legacy %d, new %d)", len(surv), sum(1 for c in surv if c["generator"]["prompt_version"] != v2.GEN_V3_VERSION),
             sum(1 for c in surv if c["generator"]["prompt_version"] == v2.GEN_V3_VERSION))

    # ---------------------------------------------------------------- fixlen
    final_store = JsonlStore(final_p, key_fn=lambda r: r["cand_id"])
    if a.stage in ("fixlen", "all"):
        todo = [c for c in surv if c["cand_id"] + "_F" not in final_store.rows and c["cand_id"] + "_F_failed" not in final_store.rows]
        log.info("fixlen for %d survivors (%d need a rewrite)", len(todo), sum(1 for c in todo if v2.needs_rewrite(c)))

        def one(c):
            base = {k: v for k, v in c.items() if k not in ("filters",)}
            if v2.needs_rewrite(c):
                n = v2.rewrite_distractors(base, corpus, main_llm)
                if n is None:
                    final_store.add({"cand_id": c["cand_id"] + "_F_failed", "parent_cand_id": c["cand_id"], "failed": "rewrite"})
                    return
            else:
                n = v2.finalize_copy(base)
            final_store.add(n)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(one, todo))
    finals = [c for c in final_store.rows.values() if not c.get("failed")]
    log.info("final candidates: %d (rewritten %d, rewrite failures %d)", len(finals),
             sum(1 for c in finals if c.get("distractor_rewrite")), sum(1 for c in final_store.rows.values() if c.get("failed")))

    # ---------------------------------------------------------------- final checks
    fchecks = JsonlStore(fchk_p, key_fn=check_key)
    if a.stage in ("final", "all"):
        prep(finals)
        pre = hb.annotate_candidates(finals, {}, corpus, leak)
        todo = [c for c in pre if c["filters"]["structural_ok"] and c["filters"].get("leak_ok") and c.get("calc_gate_ok", True)]
        log.info("final checks for %d/%d candidates", len(todo), len(finals))
        run_checks(todo, corpus, fchecks, main=main_llm, small=small_llm, workers=a.workers, closed_perms=a.closed_perms)
    if a.pool_only or a.stage not in ("assemble", "all", "bias"):
        if a.pool_only:
            annp = hb.annotate_candidates(prep(finals), fchecks.rows, corpus, leak, max_undetermined=a.max_undetermined)
            with open(work / "annotated_pool.jsonl", "w", encoding="utf-8") as f:
                for c in annp:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            log.info("pool written: %d final candidates, %d verified", len(annp), sum(1 for c in annp if c["filters"].get("verify_ok") is True))
        return 0

    # ---------------------------------------------------------------- assemble
    annf = hb.annotate_candidates(prep(finals), fchecks.rows, corpus, leak, max_undetermined=a.max_undetermined)
    with open(out / "candidates_annotated_v2.jsonl", "w", encoding="utf-8") as f:
        for c in annf:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    funnel = {"seeds_new": len(seeds), "seeds_v1": len(load_seeds(V1 / "seeds.jsonl")), "candidates_total": len(all_cands),
              "round1_survivors": len(surv), "final_candidates": len(finals), "final_structural": sum(1 for c in annf if c["filters"]["structural_ok"]),
              "final_verified": sum(1 for c in annf if c["filters"].get("verify_ok") is True)}
    pool = [c for c in survivors(annf, qf) if "closed_small_n" in c["filters"]]
    funnel["final_after_quality_multihop"] = len(pool)
    pool, dups = hb.dedup(pool)
    funnel["after_dedup"] = len(pool)
    sel = v2.balanced_select(pool, total_target=a.total_target, max_total=a.max_items, domain_floor=a.domain_floor,
                             domain_cap=a.domain_cap, seed=a.seed)
    funnel["selected"] = len(sel)
    rows = hb.to_rows(sel, corpus, corpus_manifest(corpus).get("corpus_version"))
    by_cid = {c["cand_id"]: c for c in sel}
    for r in rows:
        c = by_cid[r["cand_id"]]
        r["benchmark_version"] = "otter_hard_v2"
        r["seed_kind"] = ((c.get("seed_kind")) or "")
        r["calc"] = c.get("calc_check") if c.get("calc") else None
        r["distractor_rewritten"] = bool(c.get("distractor_rewrite"))
        r["parent_cand_id"] = c.get("parent_cand_id")
    split = hb.lesson_disjoint_split(rows, dev_size=a.dev_size, seed=a.seed)
    for r in rows:
        r["split"] = split[r["id"]]
    for name, rs in (("otter_hard_all", rows), ("dev", [r for r in rows if r["split"] == "dev"]),
                     ("test", [r for r in rows if r["split"] == "test"])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = hb.bench_stats(rows, corpus)
    lessons_dev = {k for r in rows if r["split"] == "dev" for k in r["lesson_keys"]}
    stats["test_items_sharing_lesson_with_dev"] = sum(1 for r in rows if r["split"] == "test" and set(r["lesson_keys"]) & lessons_dev)
    man = hb.manifest(built_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"), benchmark_version="otter_hard_v2",
                      generator_model="main model (FP8)", verifier_model="main model (FP8)", probe_model="small model",
                      corpus=corpus_manifest(corpus), q_round1=q1, q_final=qf, max_undetermined=a.max_undetermined,
                      gen_prompt_v3=v2.GEN_V3_VERSION, rewrite_prompt=v2.REWRITE_VERSION, seeds_v2=v2.SEEDS_V2_VERSION,
                      closed_perms=a.closed_perms, dedup_dropped=len(dups))
    (out / "build_report_v2.json").write_text(json.dumps({"manifest": man, "funnel": funnel, "stats": stats}, ensure_ascii=False,
                                                         indent=1, default=str))
    (out / "build_report_v2.md").write_text(hb.render_report(man, stats, funnel), encoding="utf-8")
    hb.write_review_sheet(rows, corpus, Path(a.sheet_dir), n=a.review_n, seed=a.seed)
    print(json.dumps({"funnel": funnel, "by_split": stats["by_split"], "by_domain": stats["by_domain"],
                      "by_hop_type": stats["by_hop_type"], "by_difficulty": stats["by_difficulty"],
                      "unique_longest": stats["correct_is_unique_longest_rate"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
