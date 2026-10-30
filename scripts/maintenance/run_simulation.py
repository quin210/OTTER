"""Knowledge-maintenance simulation: build a synthetic new edition, re-validate the ontology, score against ground truth.

  .venv/bin/python scripts/maintenance/run_simulation.py edition    --seed 0
  .venv/bin/python scripts/maintenance/run_simulation.py revalidate --seed 0        # verifier = small (:8001), bge-m3 (:8002)
  .venv/bin/python scripts/maintenance/run_simulation.py evaluate   --seed 0
  .venv/bin/python scripts/maintenance/run_simulation.py all --seeds 0 1 2

Resumable: every stage is skipped when its output exists (the verifier batch cache makes a re-run of `revalidate` cheap).
Outputs: runs/knowledge_maintenance/{editions,versions,reports}/seed<k>...  The old ontology file is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.builder.align import AlignConfig, align_spans, load_span_records, span_category      # noqa: E402
from otter.builder.revalidate import RevalConfig, load_items, revalidate                        # noqa: E402
from otter.builder.simulate_edition import find_entities, find_numbers, simulate_edition        # noqa: E402
from otter.builder.validate import claim_text                                                   # noqa: E402
from otter.storage.textutil import norm_key                                                     # noqa: E402

KM = ROOT / "runs/knowledge_maintenance"
OLD_ONTO = ROOT / "data/ontology/ontology.sqlite3"
OLD_PROV = ROOT / "data/textbooks/ontology.sqlite3"
BENCH = {"dev": ROOT / "benchmarks/otter_hard/dev.jsonl", "test": ROOT / "benchmarks/otter_hard/test.jsonl"}
log = logging.getLogger("km")


# ------------------------------------------------------------------------------------------ data
def core_rows() -> list[dict]:
    con = sqlite3.connect(f"file:{OLD_PROV}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM spans WHERE book_id LIKE 'nxbgd_%' ORDER BY book_id, pdf_page_index, start_offset, span_id")]
    con.close()
    for r in rows:
        r["heading_path"] = json.loads(r.get("heading_path_json") or "[]")
    return rows


def load_bench(split: str) -> list[dict]:
    return [json.loads(l) for l in BENCH[split].read_text(encoding="utf-8").splitlines() if l.strip()]


def _has(tok: str, text: str) -> bool:
    return f" {norm_key(tok)} " in f" {norm_key(text)} "


def select_targets(rows: list[dict], seed: int = 17, per_split: int = 20) -> dict:
    """Targeted changes for the QA part: for each split, `per_split` items whose gold span(s) are deleted and
    `per_split` items whose gold span is edited at an ANSWER-RELEVANT token (the old token occurs in the correct
    option or in the item's rationale). Items are disjoint and share no gold span. Seeded."""
    by_id = {r["span_id"]: r for r in rows}
    out = {"delete": {}, "edit": {}, "items": {}}
    used_spans: set[str] = set()
    for split in ("dev", "test"):
        items = [it for it in load_bench(split) if all(g in by_id for g in it["gold_span_ids"])]
        rng = random.Random(f"{seed}-{split}")
        rng.shuffle(items)
        n_edit = n_del = 0
        for it in items:
            gold = it["gold_span_ids"]
            if set(gold) & used_spans:
                continue
            correct = it["options"][it["answer"]]
            ctx = correct + " " + " ".join(s.get("step", "") for s in it.get("reasoning_steps", []))
            tok = None
            for g in gold:
                t = by_id[g]["text"]
                cands = [(x[2]) for x in find_numbers(t)] + [x[2] for x in find_entities(t)]
                cands = [c for c in cands if _has(c, ctx)]
                if cands:
                    tok = (g, rng.choice(cands))
                    break
            if tok and n_edit < per_split:
                out["edit"][tok[0]] = tok[1]
                out["items"][it["id"]] = {"split": split, "kind": "edit", "span": tok[0], "token": tok[1]}
                used_spans |= set(gold)
                n_edit += 1
            elif n_del < per_split and not (tok and n_edit < per_split):
                for g in gold:
                    out["delete"][g] = "targeted"
                out["items"][it["id"]] = {"split": split, "kind": "delete", "spans": gold}
                used_spans |= set(gold)
                n_del += 1
            if n_edit >= per_split and n_del >= per_split:
                break
    return out


# ------------------------------------------------------------------------------------------ stages
def stage_edition(seed: int, force: bool = False) -> Path:
    d = KM / "editions" / f"seed{seed}"
    spans_p = d / "spans.jsonl"
    if spans_p.exists() and not force:
        return d
    d.mkdir(parents=True, exist_ok=True)
    rows = core_rows()
    targets = select_targets(rows) if seed == 0 else {"delete": {}, "edit": {}, "items": {}}
    new, lg = simulate_edition(rows, seed=seed, forced_delete=targets["delete"], forced_edit=targets["edit"])
    tmp = spans_p.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new), encoding="utf-8")
    tmp.replace(spans_p)
    (d / "changes.json").write_text(json.dumps(
        {"seed": seed, "stats": lg.stats, "deleted": lg.deleted, "edited": lg.edited, "targets": targets["items"]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("edition seed %d: %s", seed, lg.stats)
    return d


def _embedder():
    from otter.storage.embeddings import Bgem3Embedder
    e = Bgem3Embedder.from_config()
    return e if e.available() else None


def stage_revalidate(seed: int, *, tag: str = "", **kw) -> dict:
    d = KM / "editions" / f"seed{seed}"
    out = KM / "versions" / f"seed{seed}{tag}.sqlite3"
    summ_p = KM / "versions" / f"seed{seed}{tag}.summary.json"
    if summ_p.exists() and out.exists():
        return json.loads(summ_p.read_text())
    from otter.llm.interface import get_llm
    cfg = RevalConfig(old_db=OLD_ONTO, old_prov=OLD_PROV, new_spans=d / "spans.jsonl", out=out, book_prefix="nxbgd_",
                      cache_dir=KM / "work" / f"seed{seed}{tag}.cache", workers=8,
                      reason=f"simulated new edition seed {seed} (5% spans deleted, 5% edited)", **kw)
    t = time.time()
    summ = revalidate(cfg, verifier=get_llm("small"), embedder=_embedder())
    summ["wall_s"] = round(time.time() - t, 1)
    summ_p.write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str))
    return summ


# ------------------------------------------------------------------------------------------ evaluation
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def _Shim_(concepts):
    from otter.builder.revalidate import _Shim
    return _Shim(concepts, [], {})


def ground_truth(items, concepts, changes, old_text, new_text_of) -> dict:
    """item -> class in {unaffected, edit_irrelevant, stale, orphaned, partial_loss}"""
    deleted, edited = set(changes["deleted"]), changes["edited"]
    from otter.builder.revalidate import _Shim
    shim = _Shim(concepts, [], {})
    gt = {}
    for (kind, iid), obj in items.items():
        spans = list(obj.source_span_ids)
        dl = [s for s in spans if s in deleted]
        ed = [s for s in spans if s in edited]
        if not dl and not ed:
            gt[(kind, iid)] = "unaffected"
            continue
        if spans and len(dl) == len(spans):
            gt[(kind, iid)] = "orphaned"
            continue
        claim = claim_text(kind if kind != "objective" else "fact", obj, shim)
        surviving = " ".join(new_text_of.get(s, "") for s in spans if s not in deleted)
        stale = any(_has(edited[s]["old"], claim) and not _has(edited[s]["old"], surviving) for s in ed)
        gt[(kind, iid)] = "stale" if stale else ("partial_loss" if dl else "edit_irrelevant")
    return gt


def stage_evaluate(seed: int) -> dict:
    d = KM / "editions" / f"seed{seed}"
    ver = KM / "versions" / f"seed{seed}.sqlite3"
    changes = json.loads((d / "changes.json").read_text())
    summ = json.loads((KM / "versions" / f"seed{seed}.summary.json").read_text())
    items, concepts, _, _ = load_items(OLD_ONTO)
    old_text = {r["span_id"]: r["text"] for r in core_rows()}
    new_rows = [json.loads(l) for l in (d / "spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    new_text = {r["span_id"]: r["text"] for r in new_rows}
    gt = ground_truth(items, concepts, changes, old_text, new_text)
    con = sqlite3.connect(ver)
    st = {(k, i): (s, v, r, sn, so) for k, i, s, v, r, sn, so in con.execute(
        "SELECT item_kind,item_id,status,valid,reason,support_new,support_old FROM item_status")}
    con.close()
    keys = sorted(gt)
    touched = {k for k in keys if gt[k] != "unaffected"}
    gt_bad = {k for k in keys if gt[k] in ("orphaned", "stale")}
    neutral = {k for k in keys if gt[k] in ("unaffected", "edit_irrelevant")}
    flagged = {k for k in keys if st[k][0] != "unchanged"}
    direct_bad = {k for k in keys if not st[k][1] and "endpoint_concept_invalid" not in st[k][2]}
    cascade_bad = {k for k in keys if not st[k][1] and "endpoint_concept_invalid" in st[k][2]}
    id_exists_bad = {k for k in keys if gt[k] == "orphaned"}               # ids are kept in the simulation: = all cited ids gone
    res = {"seed": seed, "n_items": len(keys), "gt_counts": dict(Counter(gt.values())),
           "status_counts": dict(Counter(v[0] for v in st.values())),
           "detection_touched": prf(len(flagged & touched), len(flagged - touched), len(touched - flagged))}
    methods = {"no_maintenance": set(), "id_exists_check": id_exists_bad, "hash_diff_exclude_touched": touched,
               "ours_direct": direct_bad, "ours_incl_cascade": direct_bad | cascade_bad}
    res["invalidation"] = {}
    for m, bad in methods.items():
        o = prf(len(bad & gt_bad), len(bad & neutral), len(gt_bad - bad))
        o["false_exclusion_rate"] = len(bad & neutral) / len(neutral) if neutral else float("nan")
        o["excluded"] = len(bad)
        res["invalidation"][m] = o
    # recall by class / edit type (ours_direct)
    rc = {}
    rc["orphaned"] = {"n": sum(1 for k in keys if gt[k] == "orphaned"),
                      "excluded": sum(1 for k in keys if gt[k] == "orphaned" and k in direct_bad)}
    for typ in ("number", "year", "entity"):
        ks = [k for k in keys if gt[k] == "stale" and any(
            s in changes["edited"] and changes["edited"][s]["type"] == typ and _has(changes["edited"][s]["old"],
            claim_text(k[0] if k[0] != "objective" else "fact", items[k], _Shim_(concepts))) for s in items[k].source_span_ids)]
        rc["stale_" + typ] = {"n": len(ks), "excluded": sum(1 for k in ks if k in direct_bad),
                              "flagged_needs_review": sum(1 for k in ks if st[k][0] == "needs_review")}
    res["recall_by_class"] = rc
    # per gt class x predicted status
    tab = defaultdict(Counter)
    for k in keys:
        tab[gt[k]][st[k][0] + ("" if st[k][1] else "|invalid")] += 1
    res["gt_vs_status"] = {c: dict(v) for c, v in tab.items()}
    by_kind = {}
    for kind in ("concept", "relation", "fact", "rule", "objective"):
        ks = {k for k in keys if k[0] == kind}
        gb = gt_bad & ks
        db = direct_bad & ks
        by_kind[kind] = {"n": len(ks), "touched": len(touched & ks), "gt_invalid": len(gb), "pred_invalid": len(db),
                         **{k2: v for k2, v in prf(len(db & gb), len(db & neutral & ks), len(gb - db)).items()
                            if k2 in ("precision", "recall")}}
    res["by_kind"] = by_kind
    # partial loss: how ours treats them
    pl = [k for k in keys if gt[k] == "partial_loss"]
    res["partial_loss"] = {"n": len(pl), "kept_valid": sum(1 for k in pl if st[k][1]), "excluded": sum(1 for k in pl if not st[k][1])}
    # rules: strict executable before/after
    con = sqlite3.connect(ver)
    res["strict_rules_new"] = con.execute("SELECT COUNT(*) FROM rules WHERE strict_executable=1").fetchone()[0]
    con.close()
    con = sqlite3.connect(f"file:{OLD_ONTO}?mode=ro", uri=True)
    res["strict_rules_old"] = con.execute("SELECT COUNT(*) FROM rules WHERE strict_executable=1").fetchone()[0]
    con.close()
    res["alignment"] = stage_alignment_eval(seed, d, changes, old_text, new_rows)
    (KM / "reports").mkdir(parents=True, exist_ok=True)
    (KM / "reports" / f"detection_seed{seed}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


def stage_alignment_eval(seed, d, changes, old_text, new_rows, scramble: bool = True) -> dict:
    """Alignment accuracy with span ids hidden (new ids scrambled) vs the simulation's ground truth."""
    old_recs = [r for r in load_span_records(OLD_PROV, book_prefix="nxbgd_")]
    new_recs = load_span_records(d / "spans.jsonl")
    remap = {}
    for r in new_recs:
        nid = "N" + hashlib.md5(r.span_id.encode()).hexdigest()[:12] if scramble else r.span_id
        remap[r.span_id] = nid
        r.raw = dict(r.raw)
        r.span_id = nid
    inv = {v: k for k, v in remap.items()}
    emb = _embedder()
    rows, added = align_spans(old_recs, new_recs, embedder=emb, old_vecs=None)
    deleted, edited = set(changes["deleted"]), changes["edited"]
    new_sha = {r.span_id: r.sha1 for r in new_recs}
    old_sha = {r.span_id: r.sha1 for r in old_recs}
    out = {"unchanged": Counter(), "edited": Counter(), "deleted": Counter()}
    for r in old_recs:
        row = rows[r.span_id]
        cat = span_category(row.status)
        sid = r.span_id
        if sid in deleted:
            key = "gone" if row.status == "gone" else ("ambiguous" if cat == "unresolved" else "FALSE_MATCH")
            out["deleted"][key] += 1
        elif sid in edited:
            exp = remap.get(sid)
            if row.status == "modified" and row.new_ids == [exp]:
                key = "modified_correct"
            elif cat == "unresolved":
                key = "ambiguous"
            elif row.status == "gone":
                key = "gone(MISSED)"
            else:
                key = "WRONG_MATCH"
            out["edited"][key] += 1
        else:
            if cat == "same" and row.new_ids and new_sha[row.new_ids[0]] == old_sha[sid]:
                key = "identical_correct"
            elif cat == "unresolved":
                key = "ambiguous"
            elif row.status == "gone":
                key = "gone(MISSED)"
            else:
                key = "WRONG_MATCH"
            out["unchanged"][key] += 1
    n_added_true = 0                                             # simulation adds no content
    return {"ids_scrambled": scramble, **{k: dict(v) for k, v in out.items()}, "new_spans_unaligned": len(added),
            "added_true": n_added_true, "n_old": len(old_recs), "n_new": len(new_recs)}


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["edition", "revalidate", "evaluate", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for s in (a.seeds if a.seeds is not None and a.stage == "all" else [a.seed]):
        if a.stage in ("edition", "all"):
            stage_edition(s)
        if a.stage in ("revalidate", "all"):
            stage_revalidate(s)
        if a.stage in ("evaluate", "all"):
            r = stage_evaluate(s)
            print(json.dumps({k: r[k] for k in ("seed", "gt_counts", "detection_touched", "invalidation")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
