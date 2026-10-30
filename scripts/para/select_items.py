"""Step 1: fixed seeded stratified sample of 400 items (no system output used). Writes benchmarks/otter_hard/para/selection.json."""
import json, random, hashlib
from collections import Counter
from para_lib import ROOT, jl
import sys
ROUND = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SEED, N = (20261027, 400) if ROUND == 1 else (20261028, 400)
items = [r for r in jl(ROOT / "benchmarks/otter_hard/test.jsonl") if not r.get("seen_in_development")]
if ROUND == 2:
    r1 = set(json.load(open(ROOT / "benchmarks/otter_hard/para/selection.json"))["ids"])
    items = [r for r in items if r["id"] not in r1]
def key(r):
    return (r["subject"], min(r["hop_count"], 2), bool(r.get("book_specific")), r["subset"], r["difficulty"])
rng = random.Random(SEED)
strata = {}
for r in sorted(items, key=lambda r: (key(r), r["id"])):
    strata.setdefault(key(r), []).append(r)
order = []
for k in sorted(strata):
    g = strata[k]; rng.shuffle(g); order += g
step = len(order) / N
off = rng.random() * step
idx = [int(off + i * step) for i in range(N)]
sel = [order[i] for i in idx]
assert len({r["id"] for r in sel}) == N
out = {"seed": SEED, "n_population": len(items), "n_selected": N, "method": "strata sorted, shuffled within stratum (random.Random(seed)), systematic sample with random start",
       "strata_vars": ["subject", "hop_count(3->2)", "book_specific", "subset", "difficulty"],
       "population_sha256": hashlib.sha256("\n".join(sorted(r["id"] for r in items)).encode()).hexdigest(),
       "ids": [r["id"] for r in sel]}
(ROOT / ("benchmarks/otter_hard/para/selection.json" if ROUND == 1 else "benchmarks/otter_hard/para/selection_round2.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
for v in ["subject", "difficulty", "hop_count", "book_specific", "subset"]:
    print(v, dict(Counter(str(r[v]) for r in sel)), "| pop", dict(Counter(str(r[v]) for r in items)))
