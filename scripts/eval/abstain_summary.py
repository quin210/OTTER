"""Markdown table of OTTER-Abstain runs: runs/<exp>/<system>/<backbone>/abstain_scores.json  (+ optional --write-stats).
  .venv/bin/python scripts/eval/abstain_summary.py abstain_dev abstain_dev_opt [--write-stats benchmarks/abstain/STATS.md]"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation.abstain_stats import RESULTS_MARK  # noqa: E402
from otter.evaluation.answer_judge import JUDGES, fmt_judge, run_judge_summaries  # noqa: E402

COLS = [("abstain_precision", "P_abs"), ("abstain_recall", "R_abs"), ("abstain_f1", "F1_abs"), ("unsupported_inference_rate", "UIR"),
        ("uir_reference_key", "UIR(key)"), ("over_abstention_rate", "over-abst."), ("control_accuracy", "ctrl acc"), ("control_selective_accuracy", "ctrl sel.acc")]


def f(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--write-stats", default=None)
    a = ap.parse_args()
    L = []
    for exp in a.exps:
        L += [f"### `runs/{exp}`", "", "| system | backbone | n (U/A) | " + " | ".join(c[1] for c in COLS) + " | recall by type (ooc / mp / guard) | behaviour acc (rule) | LLM-J 32B | LLM-J 8B | Full 32B | Full 8B |",
              "|" + "---|" * (len(COLS) + 9)]
        for p in sorted((ROOT / "runs" / exp).glob("*/*/abstain_scores.json")):
            r = json.loads(p.read_text())
            o, bt = r["overall"], r["by_abstain_type"]
            rec = " / ".join(f(bt.get(t, {}).get("abstain_recall")) for t in ("out_of_corpus", "missing_premise", "counterfactual_guard"))
            recs = [json.loads(l) for l in (p.parent / "abstain_records.jsonl").read_text().splitlines() if l.strip()] \
                if (p.parent / "abstain_records.jsonl").exists() else []
            # rule: abstain on unanswerable, correct choice on an answerable control (failures count as wrong)
            beh = sum((not r["answerable"] and r["kind"] == "abstain") or (r["answerable"] and r["correct"]) for r in recs) / len(recs) if recs else None
            js = run_judge_summaries(p.parent, [r["id"] for r in recs])
            L.append(f"| {p.parent.parent.name} | {p.parent.name} | {o['n_unanswerable']}/{o['n_answerable']} | " +
                     " | ".join(f(o[c[0]]) for c in COLS) + f" | {rec} | {f(beh)} | {fmt_judge(js.get('qwen3-32b'))} | {fmt_judge(js.get('qwen3-8b'))} | "
                     f"{fmt_judge(js.get('qwen3-32b'), 'strict', ci=False)} | {fmt_judge(js.get('qwen3-8b'), 'strict', ci=False)} |")
        L.append("")
    txt = "\n".join(L)
    print(txt)
    if a.write_stats:
        p = Path(a.write_stats)
        head = p.read_text(encoding="utf-8").split(RESULTS_MARK)[0]
        p.write_text(head + RESULTS_MARK + "\n\n## Sanity runs on dev\n\n" + txt, encoding="utf-8")


if __name__ == "__main__":
    main()
