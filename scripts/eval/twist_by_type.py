"""Per-twist-type (and per-subject) breakdown with the pre-registered metrics of scripts/eval/headline_analysis.py (same definitions, same
common-set rule). Writes runs/<exp>/by_type.md.
  .venv/bin/python scripts/eval/twist_by_type.py --exp vieduhop_twist --bench benchmarks/otter_hard/twist.jsonl"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/eval")); sys.path.insert(0, str(ROOT))
import headline_analysis as ha  # noqa: E402
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.source_location import score as loc_score  # noqa: E402
from otter.schemas.reasoning import FinalAnswer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="vieduhop_twist"); ap.add_argument("--bench", default=str(ROOT / "benchmarks/otter_hard/twist.jsonl"))
ap.add_argument("--systems", default="direct,rag_hybrid,ircot,otter_v2")
a = ap.parse_args()
bench = {r["id"]: r for r in (json.loads(l) for l in open(a.bench) if l.strip())}
runs = ha.load(a.exp)
names = [s for s in a.systems.split(",") if s in runs]
common = set.intersection(*[set(runs[s]) for s in names]) & set(bench)
corpus = SpanCorpus.load(book_filter=core_book_filter)
rec = {s: {} for s in names}
for s in names:
    for i in common:
        r, b = runs[s][i], bench[i]
        ok = bool(r.get("answer_choice") == b["answer"] and r.get("status") != "abstain")
        loc = {}
        if r.get("status") != "failed":
            try:
                loc = loc_score(b["gold_span_ids"], FinalAnswer.model_validate(r), corpus) or {}
            except Exception:
                loc = {}
        rec[s][i] = dict(acc=float(ok), loc=float(bool(loc.get("loc_span_hit1"))), les=float(bool(loc.get("loc_lesson1"))), vc=float(ok and bool(loc.get("loc_span_hit1"))))
groups = {}
for key in ("twist_type", "domain"):
    for v in sorted({b[key] for b in bench.values()}):
        groups[f"{key}={v}"] = sorted(i for i in common if bench[i][key] == v)
L = [f"# Twist per-group results `{a.exp}`, common set n={len(common)}", "", "cells: accuracy / located(span@1) / lesson_located / verified_correct (n items). Small n: read as descriptive counts, CIs are wide.", "",
     "| group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
for g, ids in groups.items():
    if not ids:
        continue
    cells = []
    for s in names:
        m = lambda k: sum(rec[s][i][k] for i in ids) / len(ids)
        cells.append(f"{m('acc'):.2f} / {m('loc'):.2f} / {m('les'):.2f} / {m('vc'):.2f}")
    L.append(f"| {g} | {len(ids)} | " + " | ".join(cells) + " |")
L += ["", "Per-item correctness (1 = correct): id | " + " | ".join(names)]
for i in sorted(common):
    L.append(f"- {i} ({bench[i]['twist_type']}): " + " ".join(f"{s}={int(rec[s][i]['acc'])}" for s in names))
out = ROOT / "runs" / a.exp / "by_type.md"
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
