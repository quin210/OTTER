"""Merge the multihop authors' finals into one tier with a deterministic span-conflict rule (dry run by default).

  .venv/bin/python scripts/merge_multihop.py [--write]

Rules: items of the first release (m3/m4/m5 *_final.jsonl, ids VIEDUHOP-M*-00xx) are kept; the x*_final.jsonl items are added greedily
in priority order (closed_book_hard first, then the stricter hop count, then batch order and temporary id) and an item is dropped if it shares
ANY gold span with an item already kept, or if its question has a char-3gram Jaccard >= 0.6 with a kept question. Kept x-items are renumbered
VIEDUHOP-M{k}-0021... in a fixed order. Writes benchmarks/otter_hard/multihop/multihop_all_v2.jsonl and merge_report.json with --write.
"""
import json, glob, re, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "benchmarks/otter_hard/multihop"
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def grams(s):
    s = re.sub(r"\s+", " ", s.lower()); return {s[i:i + 3] for i in range(len(s) - 2)}
def jac(a, b): return len(a & b) / max(1, len(a | b))
base = []
for k in (3, 4, 5):
    for r in load(D / f"m{k}_final.jsonl"):
        r["closed_book_hard"] = r.get("closed_book_hard", True); r["batch"] = f"m{k}"; base.append(r)
order = ["x3a", "x3b", "x4a", "x4b", "x5a", "x5b", "x3c", "x3d", "x4c", "x4d", "x5c", "x5d"]
# items removed after the independent oracle/key review (28 Oct 2026): ambiguous key year (X3C-001, Phuoc Long 1974 or 1975) and a contrived item using the
# lower bound "duoi 10 nghin" as the number 10 whose key had to be corrected once (X4D-008)
EXCLUDE = {"X3C-001", "X4D-008"}
cand = []
for b in order:
    f = D / (("x5a_final_alt_all.jsonl" if (D / "x5a_final_alt_all.jsonl").exists() else "x5a_final.jsonl") if b == "x5a" else f"{b}_final.jsonl")
    if f.exists():
        for r in load(f):
            if r.get("id") in EXCLUDE: continue
            r["batch"] = b; r.setdefault("closed_book_hard", False); cand.append(r)
def key(r): return (0 if r.get("closed_book_hard") else 1, order.index(r["batch"]), r.get("id", ""))
cand.sort(key=key)
used = {}; kept = []; grams_kept = [grams(r["question"]) for r in base]
for r in base:
    for s in r["gold_span_ids"]: used[s] = r["id"]
dropped = collections.Counter(); drops = []
for r in cand:
    k = int(r["hop_count"]); gs = set(r["gold_span_ids"])
    if len(gs) != k: dropped["wrong_hop_count"] += 1; continue
    if any(s in used for s in gs): dropped["shared_span"] += 1; drops.append((r["id"], "shared_span")); continue
    g = grams(r["question"])
    if any(jac(g, h) >= 0.6 for h in grams_kept): dropped["near_duplicate"] += 1; drops.append((r["id"], "near_dup")); continue
    kept.append(r); grams_kept.append(g)
    for s in gs: used[s] = r["id"]
cnt = collections.Counter(int(r["hop_count"]) for r in base + kept)
print("first-release kept:", len(base), "| x-items available:", len(cand), "| kept:", len(kept), "| dropped:", dict(dropped))
print("final tier by hop:", dict(sorted(cnt.items())), "| total", len(base) + len(kept), "| closed_book_hard:", sum(1 for r in base + kept if r.get("closed_book_hard")))
print("kept by batch:", dict(collections.Counter(r["batch"] for r in kept)))
if "--write" in sys.argv:
    nxt = {3: 21, 4: 21, 5: 21}; out = list(base)
    for r in sorted(kept, key=lambda r: (int(r["hop_count"]), key(r))):
        k = int(r["hop_count"]); r = dict(r); r["temp_id"] = r.get("id"); r["id"] = f"VIEDUHOP-M{k}-{nxt[k]:04d}"; nxt[k] += 1; out.append(r)
    with open(D / "multihop_all_v2.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            r["hop_count"] = int(r["hop_count"]); r["split"] = "multihop"; r["subset"] = "multihop"; r.setdefault("seen_in_development", False); r.setdefault("quotes", {})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (D / "merge_report.json").write_text(json.dumps({"kept": len(out), "by_hop": dict(cnt), "dropped": dict(dropped), "drops": drops}, indent=1, ensure_ascii=False))
    print("written", len(out))
