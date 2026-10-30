"""Gold-span retrieval metrics (recall@5, recall@10, all-gold@10, MRR of the first gold span) from answers.jsonl files that
store tool_usage.retrieved_top10 (rag_* systems) -> runs/tables/<exp>_retrieval.{csv,tex}.
  .venv/bin/python scripts/eval/retrieval_metrics.py --exp hard_v2_reviewed
"""
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation.mcq import load_mcq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
a = ap.parse_args()
exp = ROOT / "runs" / a.exp
rows = []
for ans in sorted(exp.glob("*/*/answers.jsonl")):
    meta_p = ans.parent / "run_meta.json"
    if not meta_p.exists():
        continue
    meta = json.loads(meta_p.read_text())
    gold = {it.id: list(it.gold_span_ids or []) for it in load_mcq(meta["bench"])}
    R5, R10, ALL, MRR, n = [], [], [], [], 0
    for l in open(ans):
        fa = json.loads(l)
        top = (fa.get("tool_usage") or {}).get("retrieved_top10")
        g = set(gold.get(fa["question_id"], []))
        if top is None or not g:
            continue
        n += 1
        R5.append(len(g & set(top[:5])) / len(g)); R10.append(len(g & set(top[:10])) / len(g)); ALL.append(g <= set(top[:10]))
        MRR.append(next((1 / (i + 1) for i, s in enumerate(top[:10]) if s in g), 0.0))
    if n:
        m = lambda x: round(sum(x) / len(x), 4)
        rows.append({"system": ans.parent.parent.name, "backbone": ans.parent.name, "n": n, "recall@5": m(R5), "recall@10": m(R10),
                     "all_gold@10": m(ALL), "MRR@10": m(MRR)})
out = ROOT / "runs/tables"
out.mkdir(exist_ok=True, parents=True)
cols = ["system", "backbone", "n", "recall@5", "recall@10", "all_gold@10", "MRR@10"]
with open(out / f"{a.exp}_retrieval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
tex = ["\\begin{tabular}{llrrrrr}", "\\toprule", " & ".join(c.replace("_", "\\_").replace("@", "@") for c in cols) + " \\\\", "\\midrule"]
tex += [" & ".join(str(r[c]).replace("_", "\\_").replace("+", "{+}") for c in cols) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
(out / f"{a.exp}_retrieval.tex").write_text("\n".join(tex))
print(open(out / f"{a.exp}_retrieval.csv").read())
