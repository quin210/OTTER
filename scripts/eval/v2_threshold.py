"""Post-hoc abstention curve for an otter_v2 run on an abstain bench: abstain_below = tau in 0..3 (abstain when grounding < tau),
combined with the explicit option E when the run used --abstain-option (E chosen => abstain). tau=0 is "E only".
usage: v2_threshold.py <bench.jsonl> <run_dir> [--select]   (--select prints the tau with best F1 on THIS split; use DEV only)
Output columns: abstain precision / recall / F1, UIR = 1 - recall, over-abstention on answerable controls, accuracy on answerable
controls (abstain = wrong), selective accuracy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from otter.evaluation.mcq import load_mcq
from otter.evaluation.metrics import load_answers
items = load_mcq(sys.argv[1]); ans = load_answers(Path(sys.argv[2]) / "answers.jsonl")
f = lambda a, b: a / b if b else float("nan")
print("tau  abst_prec abst_rec  F1     UIR    over_abst  ctrl_acc  sel_acc")
best = None
for tau in (0, 1, 2, 3):
    tp = fp = fn = tn = corr = 0
    for it in items:
        fa = ans.get(it.id)
        if fa is None or fa.status == "failed":
            continue
        ch = (fa.answer_choice or "").strip().upper()[:1]
        ab = ch == "E" or ch == "" or fa.tool_usage.get("v2_grounding", 0) < tau
        if it.answerable:
            if ab: fp += 1
            else: tn += 1; corr += ch == it.answer
        else:
            if ab: tp += 1
            else: fn += 1
    p, r = f(tp, tp + fp), f(tp, tp + fn)
    F = f(2 * p * r, p + r) if p == p and r == r else float("nan")
    na = fp + tn
    print(f"{tau}    {p:.3f}     {r:.3f}    {F:.3f}  {1 - r:.3f}  {f(fp, na):.3f}      {f(corr, sum(1 for i in items if i.answerable)):.3f}     {f(corr, tn):.3f}")
    if F == F and (best is None or F > best[1] + 1e-9):
        best = (tau, F)
if "--select" in sys.argv:
    print(f"best tau by F1 (tie -> lower tau = fewer abstentions): {best[0]} (F1={best[1]:.3f})")
