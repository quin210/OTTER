"""Markdown tables for benchmarks/otter_hard/twist/README.md from twist/build_report.json and twist.jsonl (+ sub-variant funnel).
  .venv/bin/python scripts/eval/twist_report.py > benchmarks/otter_hard/twist/STATS.md"""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation import hard_twist as tw  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.hard_gen import JsonlStore, check_key  # noqa: E402

T = ROOT / "benchmarks/otter_hard/twist"
d = json.loads((T / "build_report.json").read_text())
rows = [json.loads(l) for l in open(ROOT / "benchmarks/otter_hard/twist.jsonl")]
types = tw.TWIST_TYPES
short = ["generated", "feasible", "structural+type checks", "leakage/dup", "(d) stem overlap", "(c) closed-book hard", "(a) gold-only 2/3", "unique key", "near-miss alone", "(b) leave-one-out", "indep. 8B 2/2", "audit", "strict x2"]
print("## Funnel (candidates remaining after each gate; gates applied in this order)\n")
print("| gate | " + " | ".join(t for t in types) + " |\n|---|" + "---|" * len(types))
for i, g in enumerate(d["gate_names"]):
    print(f"| {i + 1}. {short[i]} | " + " | ".join(str(list(d["funnel_by_type"][t].values())[i]) for t in types) + " |")
print("| after dedup within pool | " + " | ".join(str(d["after_pool_dedup"].get(t, 0)) for t in types) + " |")
print("| **selected** | " + " | ".join(str(d["selected"].get(t, 0)) for t in types) + " |")
print("\nNote: gate 6 is not applied to negation_exception (recorded only); gates 9/10 apply to near_miss_lesson / multi-span items only; gate 11 is not applied to negation_exception.\n")
print("## Selected items per type x subject\n")
subs = ["physics", "chemistry", "biology", "geography", "history", "literature"]
print("| type | " + " | ".join(subs) + " | total |\n|---|" + "---|" * (len(subs) + 1))
for t in types:
    print(f"| {t} | " + " | ".join(str(sum(1 for r in rows if r["twist_type"] == t and r["domain"] == s)) for s in subs) + f" | {sum(1 for r in rows if r['twist_type'] == t)} |")
print("| all | " + " | ".join(str(sum(1 for r in rows if r["domain"] == s)) for s in subs) + f" | {len(rows)} |")
print("\n## Selected items: generator model, grade, hop count, position\n")
print("generator:", dict(Counter(r["generator"]["model"] for r in rows)), "| grade:", dict(sorted(Counter(r["grade"] for r in rows).items())),
      "| hop_count:", dict(Counter(r["hop_count"] for r in rows)), "| answer position:", dict(sorted(Counter(r["answer"] for r in rows).items())))
ln = sum(1 for r in rows if len(r["options"][r["answer"]]) == max(len(v) for v in r["options"].values()) and list(len(v) for v in r["options"].values()).count(len(r["options"][r["answer"]])) == 1)
print(f"key is unique longest option: {ln}/{len(rows)}")
print("\n## Stem overlap (content-word recall) and closed-book accuracy of selected items\n")
print("| type | n | mean stem overlap | threshold | closed-book Qwen3-8B acc | closed-book Qwen3-32B acc | closed_book_hard |\n|---|---|---|---|---|---|---|")
for t in types:
    rs = [r for r in rows if r["twist_type"] == t]
    if not rs:
        print(f"| {t} | 0 | | | | | |")
        continue
    m = lambda k: sum(x for x in (r["twist_checks"][k] for r in rs)) / len(rs)
    print(f"| {t} | {len(rs)} | {sum(r['stem_overlap'] for r in rs) / len(rs):.3f} | {rs[0]['stem_overlap_threshold']:.3f} | {m('closed_book_small_acc'):.2f} | {m('closed_book_main_acc'):.2f} | {sum(r['twist_checks']['closed_book_hard'] for r in rs)}/{len(rs)} |")
# sub-variants
cands = {}
for l in open(T / "candidates.jsonl"):
    r = json.loads(l)
    cands[r["cand_id"]] = r
print("\n## Sub-variants (twist_variant) among generated candidates\n")
print("| type | variant | candidates tagged | selected |\n|---|---|---|---|")
vc = Counter((r["twist_type"], r.get("twist_variant")) for r in cands.values() if r.get("twist_variant") not in (None, "none"))
for (t, v), n in sorted(vc.items()):
    print(f"| {t} | {v} | {n} | {sum(1 for r in rows if r['twist_type'] == t and r['twist_variant'] == v)} |")
print(f"\nCandidates with a hint in the prompt: {dict(Counter((r['twist_type'], r.get('variant_hint')) for r in cands.values() if r.get('variant_hint')))}")
print("\n## Almost_enough (abstain) funnel\n")
for k, v in d["almost_enough_funnel"].items():
    print(f"- {k}: {v}")
print("\n## Deterministic-check failure reasons (feasible candidates failing structural+type checks; a candidate can fail several)\n")
for t in types:
    print(f"- {t}: {dict(sorted(d['structural_failure_reasons'].get(t, {}).items(), key=lambda kv: -kv[1]))}")
print("\n## Generation failures\n")
for t in types:
    print(f"- {t}: {d['infeasible_reasons'].get(t, {})}")
