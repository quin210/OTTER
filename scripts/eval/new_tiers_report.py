"""Results of direct, rag_hybrid, ircot and otter_v2 on the two new tiers (ViEduHop-Multi 175 items, ViEduHop-Extra 291 items); post hoc.
Accuracy [95% bootstrap CI], located (top pointer in a gold span), verified-correct, all gold spans cited, calls/q; split by hop count, by
closed_book_hard and (Extra) by subject and kind."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SYS = ["direct", "rag_hybrid", "ircot", "otter_v2"]
random.seed(5)
def boot(v):
    n = len(v); xs = sorted(sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(2000)); return xs[50], xs[1950]
def m(r, k):
    ok = bool(r.get("correct")) and not r.get("abstain")
    return {"acc": float(ok), "loc": float(bool(r.get("loc_span_hit1"))), "vc": float(ok and bool(r.get("loc_span_hit1"))), "allgold": float(bool(r.get("ev_all_gold_cited")))}[k]
out = ["# New tiers: ViEduHop-Multi and ViEduHop-Extra (post hoc; LLM-written items, auto-checked, not teacher-reviewed)", ""]
for name, bench, exp in (("ViEduHop-Multi", "benchmarks/otter_hard/multihop/multihop_all_v2.jsonl", "vieduhop_multihop2"), ("ViEduHop-Extra", "benchmarks/otter_hard/extra/extra_all.jsonl", "vieduhop_extra")):
    B = {r["id"]: r for r in map(json.loads, open(ROOT / bench))}
    R = {s: {r["id"]: r for r in map(json.loads, open(ROOT / f"runs/{exp}/{s}/main/records.jsonl"))} for s in SYS}
    ids = [i for i in B if all(i in R[s] for s in SYS)]
    out += [f"## {name}: {len(ids)} of {len(B)} items answered by all four systems", ""]
    groups = [("all", lambda b: True)]
    if "Multi" in name:
        groups += [(f"{k}-hop", (lambda k: lambda b: int(b["hop_count"]) == k)(k)) for k in (3, 4, 5)]
    else:
        groups += [("1-hop", lambda b: int(b["hop_count"]) == 1), ("2-hop in subject", lambda b: int(b["hop_count"]) == 2 and b.get("subset") == "core"), ("cross-subject", lambda b: b.get("subset") == "cross_subject")]
        groups += [(f"subject {s}", (lambda s: lambda b: (b.get("domain") or b.get("subject")) == s)(s)) for s in ("physics", "chemistry", "biology", "history", "geography", "literature")]
    groups += [("closed_book_hard", lambda b: bool(b.get("closed_book_hard"))), ("not closed_book_hard", lambda b: not b.get("closed_book_hard"))]
    for g, pred in groups:
        ix = [i for i in ids if pred(B[i])]
        if len(ix) < 5: continue
        out += [f"### {g} (n={len(ix)})", "", "| system | accuracy [95% CI] | located | verified-correct | all gold cited | calls/q |", "|---|---|---|---|---|---|"]
        for s in SYS:
            a = [m(R[s][i], "acc") for i in ix]; lo, hi = boot(a)
            calls = sum((R[s][i].get("cost_llm_calls") or 0) for i in ix) / len(ix)
            out.append(f"| {s} | {sum(a)/len(a):.3f} [{lo:.3f},{hi:.3f}] | {sum(m(R[s][i],'loc') for i in ix)/len(ix):.3f} | {sum(m(R[s][i],'vc') for i in ix)/len(ix):.3f} | {sum(m(R[s][i],'allgold') for i in ix)/len(ix):.3f} | {calls:.2f} |")
        out.append("")
(ROOT / "runs/vieduhop_new_tiers_report.md").write_text("\n".join(out)); print("\n".join(out[:60]))
