"""Compact dev report for v2 runs: acc (bootstrap CI), coverage, gold-span recall, cost, grounding stats.
usage: v2_report.py <bench.jsonl> <run_dir> [<run_dir> ...]"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from otter.evaluation.mcq import load_mcq
from otter.evaluation.metrics import load_answers, item_record, bootstrap_ci

def main():
    items = load_mcq(sys.argv[1])
    for d in sys.argv[2:]:
        ans = load_answers(Path(d) / "answers.jsonl")
        recs = [item_record(it, ans.get(it.id)) for it in items]
        c = [r["correct"] for r in recs]
        a, lo, hi = bootstrap_ci(c, n_boot=5000, seed=0)
        g = lambda k: np.mean([r[k] for r in recs if r.get(k) is not None]) if any(r.get(k) is not None for r in recs) else float("nan")
        tok = [(ans[i.id].tool_usage.get("tokens") or (ans[i.id].tool_usage.get("prompt_tokens", 0) + ans[i.id].tool_usage.get("completion_tokens", 0))) for i in items if i.id in ans]
        loc = [ (len(set(i.gold_span_ids) & {c.span_id for c in ans[i.id].where_to_read}) > 0,
                 len(set(i.gold_span_ids) & {c.span_id for cl in ans[i.id].claims for c in cl.citations}) > 0)
                for i in items if i.id in ans and i.gold_span_ids]
        wl = f" where_to_read_hit@gold={np.mean([x[0] for x in loc]):.3f} cited_any_gold={np.mean([x[1] for x in loc]):.3f}" if loc else ""
        print(f"{d}: n={len(recs)} acc={a:.3f} [{lo:.3f},{hi:.3f}] cov={np.mean([r['answered'] for r in recs]):.2f} "
              f"gold_recall={g('ev_recall'):.3f} calls={g('cost_llm_calls'):.2f} tokens={np.mean(tok):.0f} cites/q={g('n_citations'):.2f}"+wl)
main()
