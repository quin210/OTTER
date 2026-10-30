"""ViEduHop-Multi (60 hand-authored 3/4/5-hop items): accuracy, located, verified-correct and all-gold-cited by hop count for the finished runs."""
import json, collections, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
B = {r["id"]: r for r in map(json.loads, open(ROOT / "benchmarks/otter_hard/multihop/multihop_all.jsonl"))}
systems = ["direct", "rag_hybrid", "ircot", "otter_v2"]
R = {s: {r["id"]: r for r in map(json.loads, open(ROOT / f"runs/vieduhop_multihop/{s}/main/records.jsonl"))} for s in systems}
ids = [i for i in B if all(i in R[s] for s in systems)]
random.seed(5)
def boot(v):
    n = len(v); xs = sorted(sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(2000)); return xs[50], xs[1950]
def m(r, k):
    ok = bool(r.get("correct")) and not r.get("abstain")
    return {"acc": float(ok), "loc": float(bool(r.get("loc_span_hit1"))), "vc": float(ok and bool(r.get("loc_span_hit1"))),
            "allgold": float(bool(r.get("ev_all_gold_cited"))), "recall": float(r.get("ev_recall") or 0)}[k]
out = ["# ViEduHop-Multi (post hoc; 60 hand-authored items by an LLM, auto-checked, not teacher-reviewed)", "", f"n = {len(ids)} common items", ""]
for grp, pred in (("all", lambda b: True), ("3-hop", lambda b: int(b["hop_count"]) == 3), ("4-hop", lambda b: int(b["hop_count"]) == 4), ("5-hop", lambda b: int(b["hop_count"]) == 5)):
    g = [i for i in ids if pred(B[i])]
    out += [f"## {grp} (n={len(g)})", "", "| system | accuracy [95% CI] | located (top pointer in a gold span) | verified-correct | all gold spans cited | mean gold recall in citations | calls/q |", "|---|---|---|---|---|---|---|"]
    for s in systems:
        a = [m(R[s][i], "acc") for i in g]; lo, hi = boot(a)
        calls = sum((R[s][i].get("cost_llm_calls") or 0) for i in g) / len(g)
        out.append(f"| {s} | {sum(a)/len(a):.3f} [{lo:.3f},{hi:.3f}] | {sum(m(R[s][i],'loc') for i in g)/len(g):.3f} | {sum(m(R[s][i],'vc') for i in g)/len(g):.3f} | {sum(m(R[s][i],'allgold') for i in g)/len(g):.3f} | {sum(m(R[s][i],'recall') for i in g)/len(g):.3f} | {calls:.2f} |")
    out.append("")
(ROOT / "runs/vieduhop_multihop/multihop_report.md").write_text("\n".join(out))
print("\n".join(out))
