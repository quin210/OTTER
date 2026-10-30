"""Tables for `ingest-books` (CSV + LaTeX) from the saved results under runs/knowledge_ingest/.  Nothing is recomputed with an LLM.

  .venv/bin/python scripts/maintenance/ingest_tables.py
-> runs/knowledge_ingest/table_cost.{csv,tex}, table_synthetic.{csv,tex}, table_loo.csv, table_paper.tex
Live cost = calls/tokens actually sent in the run that created the output (extraction replayed from a cache is reported separately).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KI = ROOT / "runs/knowledge_ingest"
REPORT = json.loads((ROOT / "data/ontology/build_report.json").read_text(encoding="utf-8"))
EX, VE = REPORT["extraction"], REPORT["verification"]
FULL_CALLS = EX["llm_chunks"] + VE["llm_calls"]
FULL_TOK = EX["prompt_tokens"] + EX["completion_tokens"] + VE["prompt_tokens"] + VE["completion_tokens"]


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verdict_totals(v: dict) -> dict:
    out = {"new": 0, "agrees": 0, "conflicts": 0, "ambiguous": 0}
    for kind, d in v.items():
        for k, n in d.items():
            out[k] = out.get(k, 0) + n
    return out


def cost_of(s: dict) -> dict:
    e, v, jd = s["extraction"], s["verification"], s["claim_judge"]
    return {"extract_chunks": e["chunks"], "extract_live": e["live_llm_calls"], "verify_live": v["live_llm_calls"],
            "judge_live": jd.get("llm_calls", 0),
            "calls": e["live_llm_calls"] + v["live_llm_calls"] + jd.get("llm_calls", 0),
            "tokens": s["cost"]["live_total_tokens"],
            "replayed_tokens": e["cached_recorded_prompt_tokens"] + e["cached_recorded_completion_tokens"]}


def main():
    rows = []
    # natural new books: live extraction on `main` (summaries of the first, live run)
    for nm, label in (("phys11", "Chuyên đề Vật lí 11 (natural)"), ("chem11", "Chuyên đề Hóa học 11 (natural)")):
        live = j(KI / f"natural/{nm}.live_run1.summary.json")
        fin = j(KI / f"natural/{nm}.summary.json")
        c = cost_of(live)
        b = next(iter(fin["books"].values()))
        rows.append({"setting": label, "spans": b["spans"], **c, "mode": "live extraction (main)", **verdict_totals(fin["verdicts"])})
    # synthetic second edition (live)
    import re
    for nm, f in (("s1", "ingest_synth_s1_llm.log"), ("s2", "ingest_synth_s2_llm_v2.log")):
        t = (ROOT / "runs/logs" / f).read_text(encoding="utf-8")
        i = t.index('\n{\n "verdicts"')
        d = json.loads(t[i:t.index("\n}\n", i) + 2])
        ed = j(KI / f"synthetic/{nm}/edits.json")
        spans = sum(p["n_spans"] for p in ed["picked_chunks"])
        c = cost_of(d)
        rows.append({"setting": f"synthetic 2nd edition {nm} ({len(ed['edits'])} value edits)", "spans": spans, **c, "mode": "live extraction (main)",
                     **verdict_totals(d["verdicts"])})
    # leave-one-book-out: extraction replayed, verify/judge live
    for nm, label in (("physics11", "LOBO Vật lí 11"), ("history11", "LOBO Lịch sử 11")):
        d = j(KI / f"loo_{nm}/compare_reference.json")
        s = j(KI / f"loo_{nm}/ingested.summary.json")
        b = next(iter(s["books"].values()))
        e = s["extraction"]
        c = {"extract_chunks": e["chunks"], "extract_live": 0, "verify_live": d["cost"]["ingest_verify_live_calls"],
             "judge_live": d["cost"]["ingest_judge_live_calls"], "calls": d["cost"]["ingest_live_llm_calls"],
             "tokens": d["cost"]["ingest_live_tokens"], "replayed_tokens": d["cost"]["ingest_extraction_replayed_tokens"]}
        rows.append({"setting": label, "spans": b["spans"], **c, "mode": "extraction replayed from cache", **verdict_totals(s["verdicts"])})
    for r in rows:
        r["calls_equiv"] = r["calls"] + (r["extract_chunks"] if r["extract_live"] == 0 else 0)
        r["tokens_equiv"] = r["tokens"] + r["replayed_tokens"]
        r["pct_of_full_tokens"] = round(100 * r["tokens_equiv"] / FULL_TOK, 2)
    cols = ["setting", "mode", "spans", "extract_chunks", "extract_live", "verify_live", "judge_live", "calls", "tokens", "replayed_tokens",
            "calls_equiv", "tokens_equiv", "pct_of_full_tokens", "new", "agrees", "conflicts", "ambiguous"]
    with (KI / "table_cost.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({"setting": "FULL REBUILD (official build report, 18 core books)", "mode": "live", "spans": 16716, "extract_chunks": 312,
                    "extract_live": EX["llm_chunks"], "verify_live": VE["llm_calls"], "calls": FULL_CALLS, "tokens": FULL_TOK,
                    "calls_equiv": FULL_CALLS, "tokens_equiv": FULL_TOK, "pct_of_full_tokens": 100})
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    # synthetic detection
    det = []
    for nm, role in (("s1", "development (rules v2 were revised after seeing it)"), ("s2", "held-out")):
        for m in ("llm_v1", "llm_v2", "nonconfirm_v2", "lexical_v2"):
            e = j(KI / f"synthetic/{nm}/eval_{m}.json")["summary"]
            det.append({"set": nm, "role": role, "config": m, "true_conflict_items": e["gt_conflict_items"], "detected": e["detected_conflicts"],
                        "tp": e["tp_lenient"], "fp": e["fp"], "precision": e["precision_lenient"], "recall": e["recall_lenient"],
                        "unedited_items": e["unedited_items"], "false_conflicts_unedited": e["false_conflicts_unedited"]})
    with (KI / "table_synthetic.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(det[0]))
        w.writeheader()
        w.writerows(det)
    # LOBO overlap
    lo = []
    for nm in ("physics11", "history11"):
        d = j(KI / f"loo_{nm}/compare_reference.json")
        for k in ("concept", "fact", "rule"):
            if d[k]["target_items_of_book"]:
                lo.append({"book": nm, "kind": k, "reference_items": d[k]["target_items_of_book"], "ingested_items": d[k]["ingested_items_of_book"],
                           "recall_by_key": d[k]["recall_vs_target"], "precision_by_key": d[k]["precision_vs_target"],
                           "emb_ge_0.95": d[k]["embedding_cov"]["ge_0.95"], "support_accept_ingest": d["support"]["accept_rate_ingest"],
                           "support_accept_reference": d["support"]["accept_rate_target"]})
    with (KI / "table_loo.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lo[0]))
        w.writeheader()
        w.writerows(lo)

    # LaTeX (no textbook text)
    def tex_cost() -> str:
        L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Setting & spans & LLM calls & tokens & \% of rebuild \\", r"\midrule",
             f"Full rebuild (18 books) & 16,716 & {FULL_CALLS:,} & {FULL_TOK/1e6:.2f}M & 100 \\\\"]
        for r in rows:
            L.append(f"{r['setting'].replace('&', 'and')} & {r['spans']:,} & {r['calls_equiv']} & {r['tokens_equiv']/1e3:.0f}k & {r['pct_of_full_tokens']:.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(L)
    (KI / "table_cost.tex").write_text(tex_cost() + "\n", encoding="utf-8")

    def tex_syn() -> str:
        L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Set & configuration & true & detected & precision & recall & false conflicts (unedited) \\", r"\midrule"]
        for d in det:
            L.append(f"{d['set']} & {d['config'].replace('_', ' ')} & {d['true_conflict_items']} & {d['detected']} & {d['precision']:.2f} & {d['recall']:.2f} & "
                     f"{d['false_conflicts_unedited']}/{d['unedited_items']} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(L)
    (KI / "table_synthetic.tex").write_text(tex_syn() + "\n", encoding="utf-8")
    # compact paper table: cost only (the synthetic table is optional)
    paper = ["% Generated by scripts/maintenance/ingest_tables.py. Synthetic rows are a controlled value-swap edition, not real editions.",
             tex_cost()]
    (KI / "table_paper.tex").write_text("\n".join(paper) + "\n", encoding="utf-8")
    print((KI / "table_cost.csv").read_text(encoding="utf-8"))
    print((KI / "table_synthetic.csv").read_text(encoding="utf-8"))
    print((KI / "table_loo.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
