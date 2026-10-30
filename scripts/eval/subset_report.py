"""Per-subset results (hop level, review_status, origin, subject, ...) + source-location metrics from records.jsonl.
  .venv/bin/python scripts/eval/subset_report.py --exp hard_v2_reviewed --bench benchmarks/otter_hard/test_reviewed.jsonl \
      --by hop_count review_status origin domain
Writes runs/tables/<exp>_by_<field>.csv and <exp>_location.csv/.tex (overall source-location table)."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--bench", required=True)
ap.add_argument("--by", nargs="*", default=["hop_count", "review_status", "origin", "domain", "bloom"])
a = ap.parse_args()
bench = {}
for l in open(a.bench):
    if l.strip():
        r = json.loads(l); bench[r["id"]] = r
LOC = ["loc_span_hit1", "loc_span_hit_any", "loc_lesson1", "loc_book1", "loc_page0", "loc_page1", "loc_printed_exact", "loc_theory"]
runs = []
for rp in sorted((ROOT / "runs" / a.exp).glob("*/*/records.jsonl")):
    recs = [json.loads(l) for l in open(rp) if l.strip()]
    runs.append((rp.parent.parent.name, rp.parent.name, recs))
out = ROOT / "runs/tables"; out.mkdir(exist_ok=True, parents=True)
m = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else ""
rows = []
for s, b, recs in runs:
    r = {"system": s, "backbone": b, "n": len(recs), "acc": m([x["correct"] for x in recs])}
    for k in LOC:
        r[k] = m([x[k] for x in recs if k in x])
    rows.append(r)
cols = ["system", "backbone", "n", "acc"] + LOC
with open(out / f"{a.exp}_location.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
esc = lambda x: str(x).replace("_", "\\_").replace("+", "{+}")
(out / f"{a.exp}_location.tex").write_text("\n".join(["\\begin{tabular}{ll" + "r" * (len(cols) - 2) + "}", "\\toprule", " & ".join(map(esc, cols)) + " \\\\", "\\midrule"]
    + [" & ".join(esc(r[c]) for c in cols) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]))
for fld in a.by:
    rows = []
    for s, b, recs in runs:
        g = defaultdict(list)
        for x in recs:
            g[str(bench.get(x["id"], {}).get(fld))].append(x)
        for v, rs in sorted(g.items()):
            rows.append({"system": s, "backbone": b, fld: v, "n": len(rs), "acc": m([x["correct"] for x in rs]),
                         "loc_theory": m([x["loc_theory"] for x in rs if "loc_theory" in x]), "loc_span_hit_any": m([x["loc_span_hit_any"] for x in rs if "loc_span_hit_any" in x])})
    if rows:
        with open(out / f"{a.exp}_by_{fld}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(open(out / f"{a.exp}_location.csv").read())
