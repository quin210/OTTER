"""Build the OTTER-Hard span-grounded multi-hop MCQ benchmark (resumable, stage by stage).

  # full pipeline (main model generates + verifies, small model is the closed-book difficulty probe)
  .venv/bin/python scripts/eval/build_otter_hard.py --stage all --gen-role main --verify-role main --probe-role small
  # prompt development on the small model, few seeds, separate work dir
  .venv/bin/python scripts/eval/build_otter_hard.py --stage all --gen-role small --verify-role small \
      --probe-role small --limit-seeds 30 --work-dir runs/otter_hard_dev --out-dir runs/otter_hard_dev/out
  # merge the filled human review sheet
  .venv/bin/python scripts/eval/build_otter_hard.py --stage assemble --apply-review annotation/otter_hard_review/otter_hard_review_filled.csv

Work files (work dir): seeds.jsonl, candidates.jsonl, checks.jsonl.  Outputs (out dir): otter_hard_all.jsonl,
dev.jsonl, test.jsonl, candidates_annotated.jsonl, build_report.{md,json}; review sheet in annotation/otter_hard_review/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation import hard_build as hb  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter, corpus_manifest  # noqa: E402
from otter.evaluation.embeddings import get_embedder  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, LeakageIndex, check_key, run_checks, run_generation  # noqa: E402
from otter.evaluation.hard_seeds import SeedSampler, load_seeds, load_span_concepts, seed_stats, write_seeds  # noqa: E402
from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.evaluation.sheets import read_filled_csv  # noqa: E402
from otter.evaluation.text import sha1  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402

log = logging.getLogger("otter.hard")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["seeds", "generate", "check", "assemble", "all"])
    ap.add_argument("--work-dir", default=str(ROOT / "benchmarks/otter_hard/work"))
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks/otter_hard"))
    ap.add_argument("--sheet-dir", default=str(ROOT / "annotation/otter_hard_review"))
    ap.add_argument("--parsed-dir", default=str(ROOT / "data/textbooks/parsed"))
    ap.add_argument("--db", default=str(ROOT / "data/textbooks/ontology.sqlite3"))
    ap.add_argument("--registry", default=str(ROOT / "data/textbooks/registry"))
    ap.add_argument("--ontology-db", default=None, help="WS2 ontology DB for concept links (default: config paths.ontology_db)")
    ap.add_argument("--core-only", action="store_true", default=True, help="only the grade 10-12 core textbooks")
    ap.add_argument("--all-books", dest="core_only", action="store_false")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--per-cell", type=int, default=36, help="seeds per (subject, grade)")
    ap.add_argument("--limit-seeds", type=int, default=None)
    ap.add_argument("--extra-seeds", type=int, default=0,
                    help="append a further round of N seeds per cell (span-use caps / overlap bound include existing seeds)")
    ap.add_argument("--gen-role", default="main")
    ap.add_argument("--verify-role", default="main")
    ap.add_argument("--probe-role", default="small")
    ap.add_argument("--no-closed-main", action="store_true", help="skip closed-book runs of the verifier model")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--easy-cap", type=float, default=0.25)
    ap.add_argument("--max-undetermined", type=int, default=1)
    ap.add_argument("--dev-size", type=int, default=50)
    ap.add_argument("--review-n", type=int, default=150)
    ap.add_argument("--apply-review", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model-note", default="main = main-role model; "
                                         "small = small-role model")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    work, out = Path(a.work_dir), Path(a.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    def core(b: dict) -> bool:
        return (not a.core_only) or core_book_filter(b)
    corpus = SpanCorpus.load(parsed_dir=Path(a.parsed_dir), db=Path(a.db), registry=Path(a.registry), min_chars=1,
                             book_filter=core)
    log.info("corpus: %d spans with text in %d books", len(corpus), len({s.book_id for s in corpus.spans}))
    if not len(corpus):
        raise SystemExit("no spans with text (WS1 OCR not finished?)")
    if not a.no_dense:
        corpus.attach_embedder(get_embedder())
    seeds_p, cand_p, chk_p = work / "seeds.jsonl", work / "candidates.jsonl", work / "checks.jsonl"

    if a.stage in ("seeds", "all") and not seeds_p.exists():
        onto = Path(a.ontology_db) if a.ontology_db else ROOT / cfg["paths"].get("ontology_db", "")
        sc = load_span_concepts(onto)
        log.info("ontology concept links for %d spans (%s)", len(sc), onto)
        sampler = SeedSampler(corpus, span_concepts=sc, seed=a.seed)
        log.info("eligible spans: %d", len(sampler.spans))
        seeds = sampler.sample(per_cell=a.per_cell)
        write_seeds(seeds, seeds_p)
        (work / "seeds_stats.json").write_text(json.dumps({**seed_stats(seeds), "eligible_spans": len(sampler.spans),
                                                           "ontology_spans": len(sc), "dense": not a.no_dense},
                                                          ensure_ascii=False, indent=1))
        log.info("seeds: %s", seed_stats(seeds))
    if a.extra_seeds and seeds_p.exists():
        old_seeds = load_seeds(seeds_p)
        onto = Path(a.ontology_db) if a.ontology_db else ROOT / cfg["paths"].get("ontology_db", "")
        sampler = SeedSampler(corpus, span_concepts=load_span_concepts(onto), seed=a.seed + len(old_seeds))
        for sd in old_seeds:
            sampler._commit(sd.span_ids)
        new = [sd for sd in sampler.sample(per_cell=a.extra_seeds) if sd.seed_id not in {o.seed_id for o in old_seeds}]
        write_seeds(old_seeds + new, seeds_p)
        log.info("extra seeds: %d (total %d) %s", len(new), len(old_seeds) + len(new), seed_stats(new))
    seeds = load_seeds(seeds_p) if seeds_p.exists() else []
    if a.limit_seeds:
        seeds = seeds[:: max(1, len(seeds) // a.limit_seeds)][: a.limit_seeds]

    cands_store = JsonlStore(cand_p, key_fn=lambda r: r["cand_id"])
    if a.stage in ("generate", "all"):
        gen = get_llm(a.gen_role)
        log.info("generating with %s for %d seeds", gen.model, len(seeds))
        run_generation(seeds, corpus, gen, cands_store, workers=a.workers, temperature=a.temperature)
    seed_ids = {s.seed_id for s in seeds}
    cands = [c for c in cands_store.rows.values() if c["seed_id"] in seed_ids and not c.get("error")]
    leak = LeakageIndex(corpus, [(it.id, it.question) for it in load_mcq(ROOT / cfg["paths"]["benchmark"])])
    checks = JsonlStore(chk_p, key_fn=check_key)
    if a.stage in ("check", "all"):
        pre = hb.annotate_candidates(cands, {}, corpus, leak)
        todo = [c for c in pre if c["filters"]["structural_ok"] and c["filters"].get("leak_ok")]
        main_llm, small_llm = get_llm(a.verify_role), get_llm(a.probe_role)
        log.info("checking %d/%d structurally valid candidates (verifier %s, probe %s)", len(todo), len(cands),
                 main_llm.model, small_llm.model)
        run_checks(todo, corpus, checks, main=main_llm, small=small_llm, workers=a.workers,
                   closed_main=not a.no_closed_main)
    if a.stage not in ("assemble", "all"):
        return 0

    ann = hb.annotate_candidates(cands, checks.rows, corpus, leak, max_undetermined=a.max_undetermined)
    with open(out / "candidates_annotated.jsonl", "w", encoding="utf-8") as f:
        for c in ann:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    kept, counts = hb.funnel(ann)
    kept, dups = hb.dedup(kept)
    counts["after_dedup"] = len(kept)
    kept, n_easy = hb.cap_easy(kept, frac=a.easy_cap, seed=a.seed)
    counts[f"after_easy_cap_{a.easy_cap}"] = len(kept)
    kept = hb.stratified_cap(kept, a.max_items, seed=a.seed)
    counts["final"] = len(kept)
    counts = {"seeds": len(seeds), **counts}
    rows = hb.to_rows(kept, corpus, corpus_manifest(corpus).get("corpus_version"))
    split = hb.lesson_disjoint_split(rows, dev_size=a.dev_size, seed=a.seed)
    for r in rows:
        r["split"] = split[r["id"]]
    review = None
    if a.apply_review:
        rows, review = hb.apply_review(rows, read_filled_csv(Path(a.apply_review)))
    for name, rs in (("otter_hard_all", rows), ("dev", [r for r in rows if r["split"] == "dev"]),
                     ("test", [r for r in rows if r["split"] == "test"])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    gen_models = sorted({c["generator"]["model"] for c in cands})
    vm = sorted({r["result"].get("model") for r in checks.rows.values() if r["check"] == "verify" and r.get("result")})
    pm = sorted({r["result"].get("model") for r in checks.rows.values() if r["check"].startswith("closed:small") and r.get("result")})
    man = hb.manifest(built_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                      generator_model=",".join(gen_models), verifier_model=",".join(vm), probe_model=",".join(pm),
                      temperature=a.temperature, corpus_spans=len(corpus),
                      corpus_sha1=sha1("".join(s.span_id + s.text for s in corpus.spans)),
                      corpus=corpus_manifest(corpus),
                      books=sorted({s.book_id for s in corpus.spans}), dense=not a.no_dense, seed=a.seed,
                      easy_cap=a.easy_cap, max_undetermined=a.max_undetermined, max_items=a.max_items,
                      model_note=a.model_note, dedup_dropped=len(dups), easy_dropped=n_easy, review=review)
    stats = hb.bench_stats(rows, corpus)
    lessons_dev = {k for r in rows if r["split"] == "dev" for k in r["lesson_keys"]}
    stats["test_items_sharing_lesson_with_dev"] = sum(1 for r in rows if r["split"] == "test" and set(r["lesson_keys"]) & lessons_dev)
    (out / "build_report.json").write_text(json.dumps({"manifest": man, "funnel": counts, "stats": stats},
                                                      ensure_ascii=False, indent=1, default=str))
    (out / "build_report.md").write_text(hb.render_report(man, stats, counts), encoding="utf-8")
    if rows and not a.apply_review:
        hb.write_review_sheet(rows, corpus, Path(a.sheet_dir), n=a.review_n, seed=a.seed)
    print(json.dumps({"funnel": counts, "by_split": stats["by_split"], "by_domain": stats["by_domain"],
                      "by_hop_type": stats["by_hop_type"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
