"""Assemble ViEduHop (files keep the otter_hard names) from candidate pools, enrich, split, write benchmark files.

  .venv/bin/python scripts/eval/assemble_vieduhop.py --pools v2=benchmarks/otter_hard/work_v2/candidates_annotated.jsonl ... 
Defaults read the standard pools (multi-hop v2, 1-hop, cross-subject, optional extra multi-hop scale-up pools).
Selection: per hop level (1/2/3 spans) up to --per-hop items, per-subject balanced, non-easy first; all verified cross-subject items
(<= --cross-cap) form ViEduHop-Cross. Nothing here relaxes the correctness filters (verified unique key entailed by the gold spans,
not single-span answerable for >=2 hops, no exercise leakage, arithmetic verified for calculation items).
"""
import argparse, datetime as dt, json, logging, random, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation import hard_build as hb, hard_v2 as v2  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter, corpus_manifest  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore  # noqa: E402
from otter.llm.interface import get_llm  # noqa: E402

log = logging.getLogger("vieduhop")
W = ROOT / "benchmarks/otter_hard"
DEFAULT_POOLS = [W / "work_v2/annotated_pool.jsonl", W / "work_v2/annotated_1hop.jsonl", W / "work_cross/annotated_pool.jsonl"]
BS_POOL = W / "work_bs/annotated_pool.jsonl"     # book-specific (closed-book-hard, tricky) items; see STATS.md section 'book-specific'

SUBJ_VI = {"physics": "Vật lí", "chemistry": "Hóa học", "biology": "Sinh học", "geography": "Địa lí", "history": "Lịch sử", "literature": "Ngữ văn"}


def good(c, qmin):
    f = c["filters"]
    if not (f.get("structural_ok") and f.get("leak_ok") and f.get("verify_ok") is True and f.get("multi_hop_ok") is True):
        return False
    if "closed_small_n" not in f:
        return False
    if c.get("min_spans") != 1 and not ("q_natural" in f and f["q_natural"] >= qmin[0] and f["q_needs_both"] >= qmin[1] and f["q_distractors"] >= qmin[2]):
        return False
    if c.get("book_specific") and not f.get("bs_all_ok"):
        return False                       # every book-specific item passed all verification gates; tier = closed_book_hard or (physics/chemistry only) evidence_only
    cc = v2.check_calc(c)
    if (c.get("hop_type") == "calculation" or cc["calc_present"]) and cc["calc_ok"] is not True:
        return False
    return True


def pick(pool, target, per_dom_target, rng, hop_cap_frac=0.35):
    """Per-subject balanced, non-easy first, hop-type interleaved; returns <= target items."""
    rank = {"hard": 0, "medium": 1, "easy": 2, None: 3}
    by = defaultdict(list)
    for c in pool:
        by[c["domain"]].append(c)
    for v in by.values():
        rng.shuffle(v)
        v.sort(key=lambda c: rank[c["filters"].get("difficulty")])
    chosen, left = [], {}
    quota = {d: min(len(v), per_dom_target) for d, v in by.items()}
    for d, v in by.items():
        # hop-type cap inside a subject (reasoning types), preferring hard items
        keep, cnt = [], Counter()
        lim = max(6, int(hop_cap_frac * quota[d]))
        rest = []
        for c in v:
            if len(keep) < quota[d] and cnt[c["hop_type"]] < lim:
                keep.append(c); cnt[c["hop_type"]] += 1
            else:
                rest.append(c)
        chosen += keep
        left[d] = rest
    while len(chosen) < target:
        ds = [d for d in left if left[d]]
        if not ds:
            break
        d = min(ds, key=lambda d: sum(1 for c in chosen if c["domain"] == d))
        chosen.append(left[d].pop(0))
    return chosen[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="*", default=[str(p) for p in DEFAULT_POOLS])
    ap.add_argument("--per-hop", type=int, default=400)
    ap.add_argument("--hop3-cap", type=int, default=400)
    ap.add_argument("--cross-cap", type=int, default=250)
    ap.add_argument("--q-final", default="3,4,3")
    ap.add_argument("--dev-size", type=int, default=150)
    ap.add_argument("--out-dir", default=str(W))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--bs-pool", default=str(BS_POOL), help="book_specific pool (merged with subset=core, origin=generated, review_status=auto_verified)")
    ap.add_argument("--no-bs", action="store_true", help="do not merge the book_specific pool")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    qmin = tuple(int(x) for x in a.q_final.split(","))
    rng = random.Random(a.seed)
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    pool = []
    pools = list(a.pools) + ([] if a.no_bs or str(a.bs_pool) in a.pools else [a.bs_pool])
    for p in pools:
        p = Path(p)
        if p.exists():
            rows = [json.loads(l) for l in open(p) if l.strip()]
            g = [c for c in rows if good(c, qmin)]
            log.info("%s: %d candidates, %d pass filters", p.name, len(rows), len(g))
            pool += g
    # dedup across pools (prefer harder, then lower length ratio)
    bs_items = [c for c in pool if c.get("book_specific")]
    pool, dups = hb.dedup([c for c in pool if not c.get("book_specific")])
    if bs_items:                           # book-specific items are protected: near-duplicates among them were removed at build time,
        from otter.evaluation.text import char_ngrams, jaccard      # here only the other pools yield to them
        bg = [char_ngrams(c["question"] + " " + c["correct_answer"]) for c in bs_items]
        keep = []
        for c in pool:
            g = char_ngrams(c["question"] + " " + c["correct_answer"])
            if any(jaccard(g, x) >= 0.6 for x in bg):
                dups.append((c["cand_id"], "book_specific", 1.0))
            else:
                keep.append(c)
        pool = keep + bs_items
    log.info("after dedup: %d (dropped %d)", len(pool), len(dups))
    for c in pool:
        doms = sorted({corpus.get(s).domain for s in c["gold_span_ids"] if corpus.get(s)})
        c["subjects"] = doms
        c["subset"] = "cross_subject" if len(doms) >= 2 else "core"
    cross = [c for c in pool if c["subset"] == "cross_subject"]
    core = [c for c in pool if c["subset"] == "core"]
    sel = []
    sel += [c for c in core if c.get("book_specific")]          # all book-specific items are always kept
    log.info("book_specific items kept: %d", len(sel))
    core = [c for c in core if not c.get("book_specific")]
    for hop, tgt in ((1, a.per_hop), (2, a.per_hop), (3, a.hop3_cap)):
        cs = [c for c in core if len(c["gold_span_ids"]) == hop]
        s = pick(cs, tgt, max(1, tgt // 6), rng)
        log.info("core %d-hop: pool %d -> selected %d", hop, len(cs), len(s))
        sel += s
    # cross: non-easy first, balanced over subject pairs
    cross.sort(key=lambda c: ({"hard": 0, "medium": 1, "easy": 2}.get(c["filters"].get("difficulty"), 3), rng.random()))
    pairs = defaultdict(list)
    for c in cross:
        pairs["-".join(c["subjects"])].append(c)
    xs = []
    while len(xs) < a.cross_cap and any(pairs.values()):
        for k in sorted(pairs):
            if pairs[k] and len(xs) < a.cross_cap:
                xs.append(pairs[k].pop(0))
    log.info("cross: pool %d -> selected %d (%s)", len(cross), len(xs), dict(Counter("-".join(c["subjects"]) for c in xs)))
    sel += xs
    # enrichment
    llm = get_llm("main")
    rows = hb.to_rows(sel, corpus, corpus_manifest(corpus).get("corpus_version"))
    byc = {c["cand_id"]: c for c in sel}
    es = JsonlStore(Path(a.out_dir) / "work_v2/enrich.jsonl", key_fn=lambda r: r["cand_id"])
    for c in sel:                          # book-specific items arrive already enriched (work_bs/enrich_bs.jsonl)
        if c.get("enrich"):
            es.rows[c["cand_id"]] = {"cand_id": c["cand_id"], **c["enrich"]}
    if not a.no_enrich:
        todo = [r for r in rows if not es.done(r["cand_id"])]
        log.info("enriching %d items", len(todo))
        def one(r):
            es.add({"cand_id": r["cand_id"], **v2.enrich_item(r, corpus, llm)})
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(one, todo))
    for r in rows:
        c = byc[r["cand_id"]]
        e = es.rows.get(r["cand_id"]) or {}
        r.update(benchmark_name="ViEduHop", subset=c["subset"], subjects=c["subjects"], origin="generated",
                 review_status="auto_verified", hop_levels=len(r["gold_span_ids"]),
                 bloom=e.get("bloom") or c.get("bloom_target"), ped_types=e.get("ped_types"),
                 ped_label_ws1={s: (corpus.get(s).pedagogical_type if corpus.get(s) else None) for s in r["gold_span_ids"]},
                 reference_answer=e.get("reference_answer"), explanation=e.get("explanation"), quotes=e.get("quotes"),
                 quotes_verbatim=e.get("quotes_verbatim"), enrich_ok=bool(e.get("reference_answer")),
                 calc=v2.check_calc(c) if c.get("calc") else None, distractor_rewritten=bool(c.get("distractor_rewrite")),
                 where_gold=[{"span_id": s, "book_id": corpus.get(s).book_id, "book_title": corpus.get(s).book_title,
                              "grade": corpus.get(s).grade, "lesson_key": corpus.get(s).lesson_key,
                              "lesson": " › ".join(corpus.get(s).heading_path[-3:]),
                              "pdf_page": (corpus.get(s).pdf_page_index + 1) if corpus.get(s).pdf_page_index is not None else None,
                              "printed_page": corpus.get(s).printed_page_label if corpus.get(s).printed_page_verified else None}
                             for s in r["gold_span_ids"]])
        r["book_specific"] = bool(c.get("book_specific"))
        if r["book_specific"]:
            r.update(trick_type=c.get("trick_type"), needs_teacher_numeric_check=bool(c.get("needs_teacher_numeric_check")),
                     closed_book_hard=bool(c["filters"].get("closed_book_hard")), book_specific_tier=c.get("book_specific_tier"),
                     closed_book_acc={"qwen3_8b": c["filters"].get("closed_small_acc"), "qwen3_32b_fp8": c["filters"].get("closed_main_acc")},
                     book_specific_meta=c.get("book_specific_meta"), evidence_quotes=c.get("evidence"))
        if r["hop_levels"] == 1:
            r["hop_type"] = "single"
        r["id"] = r["id"].replace("OTTER-HARD", "VIEDUHOP")
    # stable ids per subset/subject
    cnt = Counter()
    for r in sorted(rows, key=lambda r: (r["subset"], r["domain"], r["hop_levels"], r["cand_id"])):
        k = (r["subset"], r["domain"], r["hop_levels"])
        cnt[k] += 1
        r["id"] = f"VIEDUHOP-{'X' if r['subset'] == 'cross_subject' else 'C'}{r['hop_levels']}-{r['domain'][:3].upper()}-{cnt[k]:03d}"
    split = hb.lesson_disjoint_split(rows, dev_size=a.dev_size, seed=a.seed)
    for r in rows:
        r["split"] = split[r["id"]]
    out = Path(a.out_dir)
    for name, rs in (("otter_hard_all", rows), ("dev", [r for r in rows if r["split"] == "dev"]), ("test", [r for r in rows if r["split"] == "test"]),
                     ("cross_all", [r for r in rows if r["subset"] == "cross_subject"])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = hb.bench_stats(rows, corpus)
    (out / "build_report_v2.json").write_text(json.dumps({"built_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "pools": a.pools, "dedup_dropped": len(dups), "stats": stats}, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"n": len(rows), "by_hop": Counter(r["hop_levels"] for r in rows), "by_subset": Counter(r["subset"] for r in rows),
                      "by_domain": Counter(r["domain"] for r in rows), "by_split": Counter(r["split"] for r in rows),
                      "difficulty": Counter(r["difficulty"] for r in rows), "unique_longest": stats["correct_is_unique_longest_rate"]},
                     ensure_ascii=False, default=dict, indent=1))


if __name__ == "__main__":
    main()
