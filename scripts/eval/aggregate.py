"""Aggregate scored runs into runs/tables/<exp>_*.{csv,tex}.

  .venv/bin/python scripts/eval/aggregate.py --exp hard --ref otter_full \
      --order direct,text_rag,single_agent_tools,multi_agent_no_workspace,otter_full
  (add --rescore to recompute scores.json from answers.jsonl first, e.g. after new claim judgments)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation.aggregate import build_combined, build_tables  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="experiment dir name under runs/ (or a path)")
    ap.add_argument("--combined", default=None,
                    help="LABEL=exp,LABEL=exp,... -> runs/tables/combined_main.{csv,tex} (provenance columns from the last)")
    ap.add_argument("--out", default=str(ROOT / "runs/tables"))
    ap.add_argument("--ref", default=None, help="reference system for significance/ablation (e.g. otter_full)")
    ap.add_argument("--order", default=None, help="comma-separated system order")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    if a.combined:
        benches = []
        for kv in a.combined.split(","):
            lab, _, e = kv.partition("=")
            benches.append((lab, Path(e) if Path(e).exists() else ROOT / "runs" / e))
        p = build_combined(benches, Path(a.out), order=a.order.split(",") if a.order else None)
        print(f"combined: {p} (+ .tex)")
        if not a.exp:
            return 0
    exp = Path(a.exp) if Path(a.exp).exists() else ROOT / "runs" / a.exp
    if a.rescore:
        sys.path.insert(0, str(ROOT / "scripts/eval"))
        from run import load_corpus_if_any, score_dir  # noqa: E402
        from otter.evaluation.mcq import load_mcq  # noqa: E402
        corpus = load_corpus_if_any()
        for meta_p in sorted(exp.glob("*/*/run_meta.json")):
            meta = json.loads(meta_p.read_text())
            items = load_mcq(meta["bench"])
            if meta.get("item_ids"):
                keep = set(meta["item_ids"])
                items = [it for it in items if it.id in keep]
            score_dir(meta_p.parent, items, corpus)
    w = build_tables(exp, Path(a.out), ref_system=a.ref, order=a.order.split(",") if a.order else None)
    for k, p in w.items():
        print(f"{k}: {p} (+ .tex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
