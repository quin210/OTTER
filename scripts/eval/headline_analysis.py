"""Headline analysis for ViEduHop: accuracy, source location and "verified-correct" per system, overall and by subset.

  .venv/bin/python scripts/eval/headline_analysis.py --exp vieduhop_unrev [--bench benchmarks/otter_hard/test.jsonl] [--systems a,b,c]

Pre-registered definitions (fixed before the final numbers exist, see docs/HEADLINE_METRICS.md):
  accuracy            chosen option == key (abstention / failure counts as wrong)
  located             the top source pointer (where_to_read[0], else first citation) lies in a gold span  (loc_span_hit1)
  lesson_located      the top pointer lies in a gold lesson (loc_lesson1)
  verified_correct    accuracy AND located
  cost                LLM calls and tokens per question (tool_usage)
Only items answered by ALL listed systems are used (the common set), so rows are directly comparable. Items flagged
`seen_in_development` are excluded from the headline rows. Subsets: all, book_specific, closed_book_hard (probe sidecar), cross_subject,
non-book-specific core. 95% CIs: bootstrap over items; the last column is the paired bootstrap difference of verified_correct against the
best non-OTTER system. Nothing is tuned here; rows with a missing metric (e.g. `direct` has no pointer) are scored 0 on it, as defined.
"""
import argparse, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.source_location import score as loc_score  # noqa: E402
from otter.schemas.reasoning import FinalAnswer  # noqa: E402

ORDER = ["direct", "rag_bm25", "rag_dense", "rag_hybrid", "rag_hybrid+top_k=10", "ircot", "single_agent_tools",
         "multi_agent_no_workspace", "otter_full", "otter_v2", "hipporag", "lightrag", "minirag", "graphrag"]


def boot(v, n=2000, seed=7):
    rng = random.Random(seed)
    m = len(v)
    xs = sorted(sum(v[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return xs[int(.025 * n)], xs[int(.975 * n)]


def load(exp, bb="main"):
    out = {}
    for d in sorted((ROOT / "runs" / exp).glob(f"*/{bb}")):
        f = d / "answers.jsonl"
        if f.exists():
            out[d.parent.name] = {r["question_id"]: r for r in (json.loads(l) for l in open(f) if l.strip())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="vieduhop_unrev")
    ap.add_argument("--bench", default=str(ROOT / "benchmarks/otter_hard/test.jsonl"))
    ap.add_argument("--probe", default=str(ROOT / "benchmarks/otter_hard/closed_book_probe.jsonl"))
    ap.add_argument("--systems", default="")
    ap.add_argument("--bb", default="main")
    ap.add_argument("--include-seen", action="store_true")
    ap.add_argument("--ids", default=None, help="JSON list of question ids; restricts the common set (post hoc subset, label it as such)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    bench = {r["id"]: r for r in (json.loads(l) for l in open(a.bench) if l.strip())}
    cbh = set()
    if Path(a.probe).exists():
        for l in open(a.probe):
            r = json.loads(l)
            if r.get("closed_book_hard"):
                cbh.add(r.get("id") or r.get("question_id"))
    runs = load(a.exp, a.bb)
    names = [s for s in ORDER if s in runs] + [s for s in runs if s not in ORDER and "+" not in s]
    if a.systems:
        names = [s for s in a.systems.split(",") if s in runs]
    common = set.intersection(*[set(runs[s]) for s in names]) & set(bench)
    if not a.include_seen:
        common = {i for i in common if not bench[i].get("seen_in_development")}
    if a.ids:
        common &= set(json.load(open(a.ids)))
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    rec = {s: {} for s in names}
    for s in names:
        for i in common:
            r = runs[s][i]
            b = bench[i]
            ok = bool(r.get("answer_choice") == b["answer"] and r.get("status") != "abstain")
            loc = {}
            if r.get("status") != "failed":
                try:
                    loc = loc_score(b["gold_span_ids"], FinalAnswer.model_validate(r), corpus) or {}
                except Exception:
                    loc = {}
            tu = r.get("tool_usage") or {}
            rec[s][i] = dict(acc=float(ok), loc=float(bool(loc.get("loc_span_hit1"))), les=float(bool(loc.get("loc_lesson1"))),
                             vc=float(ok and bool(loc.get("loc_span_hit1"))), calls=tu.get("llm_calls") or 0,
                             tok=(tu.get("prompt_tokens") or 0) + (tu.get("completion_tokens") or 0))
    subsets = {"all": lambda b: True, "book_specific": lambda b: b.get("book_specific"),
               "closed_book_hard": lambda b: b["id"] in cbh, "cross_subject": lambda b: b.get("subset") == "cross_subject",
               "core_not_book_specific": lambda b: b.get("subset") == "core" and not b.get("book_specific")}
    lines = [f"# Headline analysis `{a.exp}` ({a.bb}), common set n={len(common)} (seen_in_development "
             f"{'included' if a.include_seen else 'excluded'})", "", f"systems: {', '.join(names)}", ""]
    for sub, pred in subsets.items():
        ids = sorted(i for i in common if pred(bench[i]))
        if not ids:
            continue
        lines += [f"## {sub} (n={len(ids)})", "", "| system | accuracy [95% CI] | located (span@1) | lesson_located | verified_correct [95% CI] | calls/q | tok/q |",
                  "|---|---|---|---|---|---|---|"]
        best = None
        base = [s for s in names if not s.startswith("otter")]
        if base:
            best = max(base, key=lambda s: sum(rec[s][i]["vc"] for i in ids))
        for s in names:
            g = lambda k: [rec[s][i][k] for i in ids]
            n = len(ids)
            acc, vc = g("acc"), g("vc")
            la, ha = boot(acc); lv, hv = boot(vc)
            d = ""
            if s.startswith("otter_v2") and best:
                dv = [rec[s][i]["vc"] - rec[best][i]["vc"] for i in ids]
                lo, hi = boot(dv)
                d = f" (vs {best}: {sum(dv)/n:+.3f} [{lo:+.3f},{hi:+.3f}])"
            lines.append(f"| {s} | {sum(acc)/n:.3f} [{la:.3f},{ha:.3f}] | {sum(g('loc'))/n:.3f} | {sum(g('les'))/n:.3f} | "
                         f"{sum(vc)/n:.3f} [{lv:.3f},{hv:.3f}]{d} | {sum(g('calls'))/n:.2f} | {sum(g('tok'))/n:.0f} |")
        lines.append("")
    text = "\n".join(lines)
    out = Path(a.out or ROOT / "runs" / a.exp / "headline.md")
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
