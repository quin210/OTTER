"""Experiment A: leave-one-book-out for `ingest-books`.

  .venv/bin/python scripts/maintenance/ingest_loo.py base    --name physics11 --book nxbgd_g11_physics_ce4cc7d0
  .venv/bin/python scripts/maintenance/ingest_loo.py ingest  --name physics11 --book nxbgd_g11_physics_ce4cc7d0
  .venv/bin/python scripts/maintenance/ingest_loo.py compare --name physics11 --book nxbgd_g11_physics_ce4cc7d0

base    : a reduced build on the other 17 core books. Extraction is REPLAYED from the cache of the full build
          (runs/knowledge_ingest/cache_main = copy of data/ontology/cache), so it makes no extraction calls; support-check
          batches that changed because of the missing book are re-run on the small model.
ingest  : `ingest-books` of the left-out book into that base (extraction replayed from the same cache: the recorded tokens are reported
          as "replayed", not as live cost).
compare : overlap with the full build (data/ontology, read-only), cost, support rates, conflicts.
Everything is written to runs/knowledge_ingest/loo_<name>/.  The full build and the corpus are only read.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OTTER_RUN_TAG", "knowledge_ingest__loo")

from otter.storage.embeddings import from_blob            # noqa: E402
from otter.storage.textutil import norm_key               # noqa: E402

FULL_DB = ROOT / "data/ontology/ontology.sqlite3"
PROV = ROOT / "data/textbooks/ontology.sqlite3"
EXPORT = ROOT / "data/ontology/export"
CACHE = ROOT / "runs/knowledge_ingest/cache_main"
PARSED = ROOT / "data/textbooks/parsed"
KI = ROOT / "runs/knowledge_ingest"


def core_books() -> list[str]:
    return sorted(p.name for p in PARSED.glob("nxbgd_*") if (p / "spans.jsonl").exists())


def stage_base(a):
    from otter.builder.pipeline import BuildConfig, build
    d = KI / f"loo_{a.name}"
    d.mkdir(parents=True, exist_ok=True)
    books = [b for b in core_books() if b != a.book]
    cfg = BuildConfig(db=PROV, out=d / "base.sqlite3", export_dir=d / "base_export", cache_dir=CACHE, books=books, backbone="main",
                      verifier="small", workers=2, verify_workers=3, report_dir=d / "base_report", require_verifier=True,
                      corpus_manifest=ROOT / "runs/corpus/CORPUS_V1.json")
    rep = build(cfg)
    (d / "base_build_summary.json").write_text(json.dumps({k: rep[k] for k in ("build_id", "totals", "support", "extraction", "verification", "timings_s")},
                                                           ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"totals": rep["totals"], "extraction": {k: rep["extraction"][k] for k in ("chunks", "llm_chunks", "cached_chunks")},
                      "verification": rep["verification"]}, indent=1))


def ingest_cache() -> Path:
    """Extraction replayed from the full-build cache (symlink), but EMPTY verify/judge caches so those LLM calls are live and counted."""
    import shutil
    c = KI / "cache_ingest_only"
    for sub in ("verify", "judge_v2"):
        shutil.rmtree(c / sub, ignore_errors=True)                       # fresh: every verify/judge call is live and counted
    (c / "verify").mkdir(parents=True, exist_ok=True)
    (c / "judge_v2").mkdir(parents=True, exist_ok=True)
    for name in ("extract", "emb_cache.sqlite3"):
        if not (c / name).exists():
            (c / name).symlink_to(CACHE / name)
    return c


def stage_ingest(a):
    from otter.builder.ingest_books import IngestConfig, ingest_books
    d = KI / f"loo_{a.name}"
    cfg = IngestConfig(old_db=d / "base.sqlite3", new_spans=PARSED, out=d / "ingested.sqlite3", old_prov=PROV, books=[a.book],
                       cache_dir=ingest_cache() if not a.replay_verify else CACHE, report_dir=d, workers=2, verify_workers=3, judge_workers=3, force=True,
                       reason=f"leave-one-book-out: ingest {a.book}", book_meta={"*": {"publisher": "Nhà xuất bản Giáo dục Việt Nam",
                                                                                         "series": "Kết nối tri thức với cuộc sống"}})
    s = ingest_books(cfg)
    print(json.dumps({k: s[k] for k in ("verdicts", "handling", "concept_merge_decisions", "cost", "extraction", "verification", "claim_judge")},
                     ensure_ascii=False, indent=1, default=str))


# ------------------------------------------------------------------------------------------ compare
REF = KI / "reference_full"


def _items_target(book: str, target: str) -> tuple[dict[str, dict[str, dict]], Path]:
    exp, db = (REF / "export" / "export", REF / "reference.sqlite3") if target == "reference" else (EXPORT, FULL_DB)
    out = {}
    for kind, fn in (("concept", "concepts"), ("fact", "facts"), ("rule", "rules")):
        rows = [json.loads(l) for l in (exp / f"{fn}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        out[kind] = {r[{"concept": "concept_id", "fact": "assertion_id", "rule": "rule_id"}[kind]]: r for r in rows if book in r.get("_books", [])}
    return out, db


def _vec(db: Path, kind: str, ids: list[str]) -> dict[str, np.ndarray]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    res = {}
    for i in range(0, len(ids), 500):
        ch = ids[i:i + 500]
        for iid, b in con.execute(f"SELECT item_id, vec FROM embeddings WHERE kind=? AND item_id IN ({','.join('?' * len(ch))})", [kind, *ch]):
            v = from_blob(b)
            res[iid] = v / (np.linalg.norm(v) or 1)
    con.close()
    return res


def stage_compare(a):
    d = KI / f"loo_{a.name}"
    ing = d / "ingested.sqlite3"
    summ = json.loads((d / "ingested.summary.json").read_text(encoding="utf-8"))
    full, full_db = _items_target(a.book, a.target)
    log = [json.loads(l) for l in (d / "ingested.changelog.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    merges = [json.loads(l) for l in (d / "ingested.merges.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    con = sqlite3.connect(f"file:{ing}?mode=ro", uri=True)
    res: dict = {"book": a.book, "name": a.name, "target": a.target}
    for kind in ("concept", "fact", "rule"):
        ev = [e for e in log if e["item_kind"] == kind]
        if kind == "concept":
            fkeys = {norm_key(r["label"]) for r in full["concept"].values()}
            ikeys = {norm_key(m["new_label"]) for m in merges}
            final_ids = {m["final_concept_id"] for m in merges}
        else:
            fkeys = set(full[kind])
            ikeys = {e["new_item_id"] for e in ev}
            final_ids = {(e["matched_old_id"] if e["op"] in ("provenance_added", "agrees_recorded") and e["matched_old_id"] else e["item_id"]) for e in ev}
        inter = fkeys & ikeys
        res[kind] = {"target_items_of_book": len(fkeys), "ingested_items_of_book": len(ikeys), "intersection": len(inter),
                     "recall_vs_target": round(len(inter) / max(1, len(fkeys)), 4), "precision_vs_target": round(len(inter) / max(1, len(ikeys)), 4),
                     "only_in_target": len(fkeys - ikeys), "only_in_ingest": len(ikeys - fkeys)}
        fv = _vec(full_db, kind, list(full[kind]))
        iv = _vec(ing, kind, list(final_ids))
        if fv and iv:
            M = np.stack(list(iv.values()))
            best = np.array([float((M @ v).max()) for v in fv.values()])
            res[kind]["embedding_cov"] = {"n": len(best), "ge_0.95": round(float((best >= 0.95).mean()), 4),
                                          "ge_0.90": round(float((best >= 0.90).mean()), 4), "median": round(float(np.median(best)), 4)}
    # support acceptance of the new book's items: ingest vs the same book in the target build
    sup_i, sup_f = Counter(), Counter()
    for kind, table, pk in (("concept", "concepts", "concept_id"), ("fact", "facts", "assertion_id"), ("rule", "rules", "rule_id")):
        for e in log:
            if e["item_kind"] == kind and e["op"] in ("added", "added_ambiguous", "hidden_conflict", "added_alternative"):
                r = con.execute(f"SELECT json FROM {table} WHERE {pk}=?", (e["item_id"],)).fetchone()
                if r:
                    sup_i[json.loads(r[0])["validation"]["source_support"]] += 1
        for r in full[kind].values():
            sup_f[r["validation"]["source_support"]] += 1
    res["support"] = {"ingest_new_items": dict(sup_i), "target_items_of_book": dict(sup_f),
                      "accept_rate_ingest": round(sup_i["supported"] / max(1, sum(sup_i.values())), 4),
                      "accept_rate_target": round(sup_f["supported"] / max(1, sum(sup_f.values())), 4)}
    res["verdicts"] = summ["verdicts"]
    res["handling"] = summ["handling"]
    res["conflict_rows"] = [dict(zip(("verdict", "n"), r)) for r in con.execute("SELECT verdict, COUNT(*) FROM item_conflicts GROUP BY verdict")]
    full_rep = json.loads((ROOT / "data/ontology/build_report.json").read_text(encoding="utf-8"))
    ex, ve = full_rep["extraction"], full_rep["verification"]
    full_tokens = ex["prompt_tokens"] + ex["completion_tokens"] + ve["prompt_tokens"] + ve["completion_tokens"]
    e, v, j = summ["extraction"], summ["verification"], summ["claim_judge"]
    rep_cost = {}
    rp = REF / "summary.json"
    if rp.exists():
        rs = json.loads(rp.read_text(encoding="utf-8"))
        rep_cost = {"reference_build_verify_live_calls": rs["verification"]["llm_calls"], "reference_build_verify_cached_batches": rs["verification"]["cached_batches"]}
    res["cost"] = {
        "ingest_live_llm_calls": summ["cost"]["live_llm_calls"], "ingest_live_tokens": summ["cost"]["live_total_tokens"],
        "ingest_extraction_replayed_chunks": e["cached_chunks"], "ingest_extraction_replayed_tokens": e["cached_recorded_prompt_tokens"] + e["cached_recorded_completion_tokens"],
        "ingest_verify_batches_total": v["live_llm_calls"] + v["cached_batches"], "ingest_verify_live_calls": v["live_llm_calls"], "ingest_verify_live_tokens": v["live_prompt_tokens"] + v["live_completion_tokens"],
        "ingest_judge_live_calls": j.get("llm_calls", 0), "ingest_judge_live_tokens": j.get("prompt_tokens", 0) + j.get("completion_tokens", 0),
        "equivalent_tokens_if_extraction_were_live": summ["cost"]["live_total_tokens"] + e["cached_recorded_prompt_tokens"] + e["cached_recorded_completion_tokens"],
        "equivalent_calls_if_extraction_were_live": summ["cost"]["live_llm_calls"] + e["cached_chunks"],
        "official_full_build_extraction_calls": ex["llm_chunks"], "official_full_build_verify_calls": ve["llm_calls"],
        "official_full_build_tokens": full_tokens, **rep_cost}
    (d / f"compare_{a.target}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("httpx", "openai", "httpcore"):
        logging.getLogger(n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["base", "ingest", "compare"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--replay-verify", action="store_true", help="also reuse the full-build verification cache (cheap, not a cost measurement)")
    ap.add_argument("--target", default="reference", choices=["reference", "official"])
    a = ap.parse_args()
    {"base": stage_base, "ingest": stage_ingest, "compare": stage_compare}[a.stage](a)


if __name__ == "__main__":
    main()
