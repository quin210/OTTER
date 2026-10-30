"""OTTER-Hard 1-hop (single-span, 'source location') items. Cheap funnel: generate -> structural -> length-matching rewrite ->
ONE verification pass (gold span only; unique key) + closed-book probes. Output: work_v2/annotated_1hop.jsonl (pool for the merge).
  .venv/bin/python scripts/eval/build_otter_hard_1hop.py --stage all
"""
import argparse, json, logging, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation import hard_build as hb, hard_v2 as v2  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, LeakageIndex, check_key, run_checks  # noqa: E402
from otter.evaluation.hard_seeds import load_seeds, write_seeds  # noqa: E402
from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.evaluation.text import sha1  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402

log = logging.getLogger("hard1")
PER_DOMAIN = {"physics": 170, "chemistry": 190, "biology": 150, "geography": 150, "history": 150, "literature": 150}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seeds", "generate", "rewrite", "check", "annotate", "all"])
    ap.add_argument("--work-dir", default=str(ROOT / "benchmarks/otter_hard/work_v2"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    W = Path(a.work_dir)
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    seeds_p = W / "seeds_1hop.jsonl"
    if not seeds_p.exists() and a.stage in ("seeds", "all"):
        write_seeds(v2.one_hop_seeds(corpus, per_domain={k: int(v * a.scale) for k, v in PER_DOMAIN.items()}), seeds_p)
    seeds = load_seeds(seeds_p)
    main_llm, small_llm = get_llm("main"), get_llm("small")
    cs = JsonlStore(W / "cands_1hop.jsonl", key_fn=lambda r: r["cand_id"])
    if a.stage in ("generate", "all"):
        todo = [s for s in seeds if not cs.done("cand_" + sha1(s.seed_id + v2.ONE_HOP_VERSION + main_llm.model)[:12])]
        log.info("generating %d 1-hop candidates", len(todo))
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda s: cs.add(v2.generate_one_hop(s, corpus, main_llm)), todo))
    cfg = load_config()
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / cfg["paths"]["benchmark"])])
    cands = [c for c in cs.rows.values() if not c.get("error")]
    for c in cands:
        cc = v2.check_calc(c)
        c["calc_gate_ok"] = cc["calc_ok"] is True if (cc["calc_present"]) else True
    pre = hb.annotate_candidates(cands, {}, corpus, leak)
    ok = [c for c in pre if c["filters"]["structural_ok"] and c["filters"].get("leak_ok") and c["calc_gate_ok"]]
    log.info("1-hop: %d generated, %d structurally ok", len(cands), len(ok))
    fs = JsonlStore(W / "final_1hop.jsonl", key_fn=lambda r: r["cand_id"])
    if a.stage in ("rewrite", "all"):
        todo = [c for c in ok if c["cand_id"] + "_F" not in fs.rows and c["cand_id"] + "_F_failed" not in fs.rows]
        def one(c):
            base = {k: v for k, v in c.items() if k != "filters"}
            n = v2.rewrite_distractors(base, corpus, main_llm) if v2.needs_rewrite(c) else v2.finalize_copy(base)
            fs.add(n or {"cand_id": c["cand_id"] + "_F_failed", "failed": "rewrite"})
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
    finals = [c for c in fs.rows.values() if not c.get("failed")]
    ck = JsonlStore(W / "checks_1hop.jsonl", key_fn=check_key)
    if a.stage in ("check", "all"):
        pre = hb.annotate_candidates(finals, {}, corpus, leak)
        todo = [c for c in pre if c["filters"]["structural_ok"] and c["filters"].get("leak_ok")]
        log.info("checks for %d finals", len(todo))
        run_checks(todo, corpus, ck, main=main_llm, small=small_llm, workers=a.workers, closed_perms=4, light=True)
    ann = hb.annotate_candidates(finals, ck.rows, corpus, leak, max_undetermined=3)
    with open(W / "annotated_1hop.jsonl", "w", encoding="utf-8") as f:
        for c in ann:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    good = [c for c in ann if c["filters"].get("verify_ok") is True and c["filters"].get("structural_ok") and c["filters"].get("leak_ok")]
    log.info("1-hop funnel: seeds %d, generated %d, structural %d, final %d, verified %d", len(seeds), len(cands), len(ok), len(finals), len(good))


if __name__ == "__main__":
    main()
