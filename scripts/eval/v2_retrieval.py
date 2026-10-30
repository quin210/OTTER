"""Retrieval analysis: gold-span recall of the retrieved set (not only the cited set) and retrieval calls / question.
usage: v2_retrieval.py <bench.jsonl> <run_dir> ...   (uses tool_usage.retrieved_span_ids / retrieved_by_round)"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from otter.evaluation.mcq import load_mcq
from otter.evaluation.metrics import load_answers

items = [i for i in load_mcq(sys.argv[1]) if i.answerable and i.gold_span_ids]
for d in sys.argv[2:]:
    ans = load_answers(Path(d) / "answers.jsonl")
    rec, rec1, any1, n_ret, calls = [], [], [], [], []
    for it in items:
        fa = ans.get(it.id)
        if fa is None:
            continue
        tu = fa.tool_usage; g = set(it.gold_span_ids)
        ids = list(tu.get("retrieved_span_ids") or [])
        r1 = (tu.get("retrieved_by_round") or [ids])[0]
        rec.append(len(g & set(ids)) / len(g)); rec1.append(len(g & set(r1)) / len(g))
        any1.append(float(bool(g & set(ids)))); n_ret.append(len(ids))
        calls.append(tu.get("retrieval_calls") or tu.get("calls") or 1)
    print(f"{d}: n={len(rec)} recall@retrieved={np.mean(rec):.3f} (round1 {np.mean(rec1):.3f}) any-gold={np.mean(any1):.3f} "
          f"|retrieved|={np.mean(n_ret):.1f} retrieval_calls={np.mean(calls):.1f}")
