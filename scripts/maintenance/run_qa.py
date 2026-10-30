"""Knowledge-maintenance QA effect (seed-0 simulated edition): do answers follow the edition?

  prepare   choose affected + control dev items, build the 'stale KB' variant (OLD ontology items + NEW spans)
  run       --arm otter_new | otter_stale_kb | rag_new   (answers on the dev subset; Qwen3-32B :8004, <= 8 workers)
  exposure  stale-hint exposure of search_knowledge (no LLM)
  score     metrics table -> runs/knowledge_maintenance/qa/{qa_table.csv,qa_table.tex,qa_scores.json}

OLD-edition systems are NOT re-run: their answers already exist (dev: runs/core_dev/{otter_v2,rag_hybrid}; test split graph-RAG:
runs/archive/ws4_baselines_prev_snapshot_*/runs/otter_hard_v2/{hipporag,lightrag}). They represent an index/graph built on the OLD
corpus that is not rebuilt (stale); the edition's changed-span set plays the role of `excluded_span_ids`.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/maintenance"))
import run_simulation as RS                                            # noqa: E402
from otter.storage.textutil import norm_key                           # noqa: E402

KM = RS.KM
QA = KM / "qa"
SEED = 0
OLD_RUNS = {"rag_hybrid (old index)": ROOT / "runs/core_dev/rag_hybrid/main/answers.jsonl",
            "otter_v2 (old index + old KB)": ROOT / "runs/core_dev/otter_v2/main/answers.jsonl"}
ARCH = ROOT / "runs/archive/ws4_baselines_prev_snapshot_20261026_122308/runs/otter_hard_v2"
GRAPH_RUNS = {"hipporag (old graph)": ARCH / "hipporag/main_prev_20261026_1219/answers.jsonl",
              "lightrag (old graph)": ARCH / "lightrag/main_prev_20261026_1219/answers.jsonl"}
NEW_ARMS = {"rag_new": "rag_hybrid (new spans)", "otter_stale_kb": "otter_v2 (new spans + old KB)",
            "otter_new": "otter_v2 (new spans + maintained KB)"}


def changes() -> dict:
    return json.loads((KM / "editions" / f"seed{SEED}" / "changes.json").read_text())


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def item_class(it: dict, ch: dict, by_text: dict) -> tuple[str, dict]:
    """class of a benchmark item under the edition: gold_deleted | gold_deleted_partial | edit_relevant | edit_other | control"""
    dl = [g for g in it["gold_span_ids"] if g in ch["deleted"]]
    ed = [g for g in it["gold_span_ids"] if g in ch["edited"]]
    if dl:
        return ("gold_deleted" if len(dl) == len(it["gold_span_ids"]) else "gold_deleted_partial"), {}
    if ed:
        e = ch["edited"][ed[0]]
        ctx = it["options"][it["answer"]] + " " + " ".join(s.get("step", "") for s in it.get("reasoning_steps", []))
        return ("edit_relevant" if RS._has(e["old"], ctx) else "edit_other"), e
    return "control", {}


# ------------------------------------------------------------------------------------------ prepare
def cmd_prepare(a):
    QA.mkdir(parents=True, exist_ok=True)
    ch = changes()
    rows = {r["span_id"]: r for r in RS.core_rows()}
    dev = RS.load_bench("dev")
    cls = {it["id"]: item_class(it, ch, rows) for it in dev}
    aff = [i for i, (c, _) in cls.items() if c != "control"]
    ctrl = sorted(i for i, (c, _) in cls.items() if c == "control")
    random.Random(7).shuffle(ctrl)
    ctrl = ctrl[:30]
    keep = set(aff) | set(ctrl)
    sub = [it for it in dev if it["id"] in keep]
    (QA / "dev_subset.jsonl").write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in sub), encoding="utf-8")
    (QA / "dev_subset_classes.json").write_text(json.dumps(
        {i: {"class": c, **({"old": e["old"], "new": e["new"], "type": e["type"]} if e else {})}
         for i, (c, e) in cls.items() if i in keep}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("dev subset:", Counter(cls[i][0] for i in keep), "of", len(dev), "dev items")
    # stale-KB variant: OLD ontology items, NEW edition spans + span vectors
    out = KM / "versions" / "seed0_staleKB.sqlite3"
    if not out.exists():
        new_v = KM / "versions" / f"seed{SEED}.sqlite3"
        src = sqlite3.connect(f"file:{RS.OLD_ONTO}?mode=ro", uri=True)
        tmp = out.with_name(out.name + ".building")
        tmp.unlink(missing_ok=True)
        dst = sqlite3.connect(tmp)
        src.backup(dst)
        src.close()
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.execute("ATTACH DATABASE ? AS nv", (str(new_v),))
        dst.execute("DELETE FROM embeddings WHERE kind='span'")
        dst.execute("INSERT INTO embeddings SELECT * FROM nv.embeddings WHERE kind='span'")
        for t in ("books", "sections", "spans"):
            dst.execute(f"CREATE TABLE {t} AS SELECT * FROM nv.{t}")
        dst.execute("CREATE VIRTUAL TABLE spans_fts USING fts5(span_id UNINDEXED, book_id UNINDEXED, text)")
        dst.execute("INSERT INTO spans_fts SELECT span_id, book_id, text FROM nv.spans")
        dst.commit()
        dst.execute("DETACH nv")
        dst.close()
        os.replace(tmp, out)
        print("wrote", out)


# ------------------------------------------------------------------------------------------ run
def cmd_run(a):
    from otter.agents.runtime import LRU
    from otter.evaluation.mcq import load_mcq
    from otter.evaluation.runner import run_system
    items = load_mcq(str(QA / "dev_subset.jsonl"))
    out_dir = QA / "answers" / a.arm
    os.environ["OTTER_RUN_TAG"] = f"knowledge_maintenance/{a.arm}"
    if a.arm == "rag_new":
        from otter.evaluation import rag_variants as RV
        from otter.evaluation.corpus import SpanCorpus, core_book_filter
        from otter.storage import SqliteOntologyStore
        import numpy as np
        from otter.storage.embeddings import from_blob
        from otter.llm.interface import get_llm
        ver = KM / "versions" / f"seed{SEED}.sqlite3"
        corpus = SpanCorpus.load(parsed_dir=KM / "none", db=ver, book_filter=core_book_filter)
        con = sqlite3.connect(f"file:{ver}?mode=ro", uri=True)
        vec = {i: np.frombuffer(b, dtype=np.float32) for i, b in
               con.execute("SELECT item_id, vec FROM embeddings WHERE kind='span' AND model='BAAI/bge-m3'")}
        con.close()
        mat = np.zeros((len(corpus.spans), len(next(iter(vec.values())))), dtype=np.float32)
        miss = 0
        for k, s in enumerate(corpus.spans):
            v = vec.get(s.span_id)
            if v is None:
                miss += 1
            else:
                mat[k] = v / (np.linalg.norm(v) or 1.0)
        from otter.evaluation.embeddings import get_embedder
        corpus.attach_embedder(get_embedder())
        corpus._dense = mat
        corpus._missing_dense = miss
        RV._INDEX["corpus"] = corpus
        print("rag_new corpus:", len(corpus.spans), "spans; missing vectors:", miss)
        store = SqliteOntologyStore(ver)
        system = RV.RagVariantSystem(get_llm("main"), mode="hybrid", top_k=5, name="rag_hybrid", llm_cache=LRU(50000), store=store)
    else:
        from otter.agents.systems import get_system
        db = KM / "versions" / ("seed0_staleKB.sqlite3" if a.arm == "otter_stale_kb" else f"seed{SEED}.sqlite3")
        system = get_system("otter_v2", "main", db_path=str(db), provenance_path=None, book_prefix="nxbgd_")
    meta = {"arm": a.arm, "seed": SEED, "bench": "dev_subset.jsonl"}
    run_system(system, items, out_dir, workers=min(8, a.workers), retry_failed=True, meta=meta)


# ------------------------------------------------------------------------------------------ exposure
def cmd_exposure(a):
    """Stale-hint exposure: for every dev+test question, run the SAME search_knowledge call otter_v2 uses for its hints
    on the OLD KB and on the maintained KB; count returned facts/rules whose evidence is GT-bad (orphaned/stale)."""
    from otter.builder.revalidate import load_items
    from otter.storage import SqliteOntologyStore
    ch = changes()
    items, concepts, _, _ = load_items(RS.OLD_ONTO)
    rows = RS.core_rows()
    old_text = {r["span_id"]: r["text"] for r in rows}
    new_rows = jl(KM / "editions" / f"seed{SEED}" / "spans.jsonl")
    new_text = {r["span_id"]: r["text"] for r in new_rows}
    gt = RS.ground_truth(items, concepts, ch, old_text, new_text)
    bad = {f"{k}|{i}" for (k, i), c in gt.items() if c in ("orphaned", "stale")}
    touched = {f"{k}|{i}" for (k, i), c in gt.items() if c != "unaffected"}
    stores = {"old KB (no maintenance)": SqliteOntologyStore(RS.OLD_ONTO, provenance_path=RS.OLD_PROV),
              "maintained KB": SqliteOntologyStore(KM / "versions" / f"seed{SEED}.sqlite3")}
    res = {}
    for split in ("dev", "test"):
        its = RS.load_bench(split)
        cls = {it["id"]: item_class(it, ch, None)[0] for it in its}
        for name, st in stores.items():
            agg = defaultdict(lambda: Counter())
            for it in its:
                q = it["question"] + " " + " ".join(it["options"].values())
                hits = st.search_knowledge(q, subject=it.get("domain"), grade=None, kinds=("fact", "rule"), top_k=8)
                hits = [h for h in hits if h.get("source_support") == "supported"][:4]           # as otter_v2._hints
                for grp in ("all", cls[it["id"]]):
                    agg[grp]["q"] += 1
                    agg[grp]["hints"] += len(hits)
                    agg[grp]["q_with_bad_hint"] += any(f"{h['kind']}|{h['id']}" in bad for h in hits)
                    agg[grp]["bad_hints"] += sum(f"{h['kind']}|{h['id']}" in bad for h in hits)
                    agg[grp]["q_with_touched_hint"] += any(f"{h['kind']}|{h['id']}" in touched for h in hits)
            res[f"{split}|{name}"] = {g: dict(c) for g, c in agg.items()}
    (QA / "exposure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res.items():
        print(k, v["all"])


# ------------------------------------------------------------------------------------------ score
def _cites(ans: dict) -> list[tuple[str, str]]:
    out = []
    for cl in ans.get("claims") or []:
        for c in cl.get("citations") or []:
            out.append((c.get("span_id"), c.get("quote") or ""))
    return out


def judge_answer(ans: dict, it: dict, ch: dict) -> dict:
    ch_del, ch_ed = ch["deleted"], ch["edited"]
    cites = _cites(ans)
    stale = False
    fresh_edit = False
    for sid, quote in cites:
        if sid in ch_del:
            stale = True
        elif sid in ch_ed and RS._has(ch_ed[sid]["old"], quote):
            stale = True
        elif sid in ch_ed and RS._has(ch_ed[sid]["new"], quote):
            fresh_edit = True
    choice = ans.get("answer_choice")
    follows = False
    for sid in it["gold_span_ids"]:
        if sid in ch_ed and choice and choice in it["options"] and RS._has(ch_ed[sid]["new"], it["options"][choice]):
            follows = True
    return {"choice": choice, "old_key": choice == it["answer"], "status": ans.get("status"),
            "not_supported": ans.get("status") in ("qualified", "abstain", "failed"), "stale_citation": stale,
            "cites_edited_fresh": fresh_edit, "follows_edit": follows,
            "old_key_and_stale": (choice == it["answer"]) and stale}


def cmd_score(a):
    ch = changes()
    classes = json.loads((QA / "dev_subset_classes.json").read_text())
    sub = {it["id"]: it for it in jl(QA / "dev_subset.jsonl")}
    systems: dict[str, dict[str, dict]] = {}
    for name, p in OLD_RUNS.items():
        systems[name] = {d["question_id"]: d for d in jl(p) if d["question_id"] in sub}
    for arm, name in NEW_ARMS.items():
        p = QA / "answers" / arm / "answers.jsonl"
        if p.exists():
            systems[name] = {d["question_id"]: d for d in jl(p) if d["question_id"] in sub}
    groups = ["gold_deleted", "gold_deleted_partial", "edit_relevant", "edit_other", "control"]
    out: dict = {"dev": {}}
    old_name = "otter_v2 (old index + old KB)"
    for name, ans in systems.items():
        res = {}
        for g in groups:
            ids = [i for i, c in classes.items() if c["class"] == g and i in ans]
            if not ids:
                continue
            js = [judge_answer(ans[i], sub[i], ch) for i in ids]
            n = len(js)
            res[g] = {"n": n, "old_key": sum(j["old_key"] for j in js) / n, "stale_citation": sum(j["stale_citation"] for j in js) / n,
                      "not_supported_status": sum(j["not_supported"] for j in js) / n,
                      "follows_edit": sum(j["follows_edit"] for j in js) / n if g.startswith("edit") else None,
                      "old_key_and_stale_citation": sum(j["old_key_and_stale"] for j in js) / n}
            if old_name in systems and name != old_name and g == "control":
                base = systems[old_name]
                res[g]["flip_vs_old_otter_v2"] = sum(1 for i in ids if ans[i].get("answer_choice") != base[i].get("answer_choice")) / n
        out["dev"][name] = res
    # graph-RAG proxy on the TEST split (existing answers; nothing re-run)
    test = {it["id"]: it for it in RS.load_bench("test")}
    tcls = {i: item_class(it, ch, None)[0] for i, it in test.items()}
    out["test_proxy"] = {}
    for name, p in GRAPH_RUNS.items():
        ans = {d["question_id"]: d for d in jl(p)}
        res = {}
        for g in ("gold_deleted", "gold_deleted_partial", "edit_relevant", "edit_other"):
            ids = [i for i, c in tcls.items() if c == g and i in ans]
            if not ids:
                continue
            js = [judge_answer(ans[i], test[i], ch) for i in ids]
            n = len(js)
            res[g] = {"n": n, "old_key": sum(j["old_key"] for j in js) / n, "stale_citation": sum(j["stale_citation"] for j in js) / n,
                      "old_key_and_stale_citation": sum(j["old_key_and_stale"] for j in js) / n}
        out["test_proxy"][name] = res
    (QA / "qa_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    r = sp.add_parser("run")
    r.add_argument("--arm", required=True, choices=list(NEW_ARMS))
    r.add_argument("--workers", type=int, default=8)
    sp.add_parser("exposure")
    sp.add_parser("score")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    {"prepare": cmd_prepare, "run": cmd_run, "exposure": cmd_exposure, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
