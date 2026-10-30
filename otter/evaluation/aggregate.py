"""Collect runs/<exp>/<system>/<backbone>/ results into CSV + LaTeX (booktabs) tables.

Only numbers present in scores.json / records.jsonl are used. Missing cells are '--'.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from otter.evaluation.answer_judge import JUDGES, fmt_judge, run_judge_summaries
from otter.evaluation.metrics import compare

DOMAIN_LABEL = {"physics": "Phys.", "chemistry": "Chem.", "biology": "Bio.", "history": "Hist.",
                "geography": "Geo.", "literature": "Lit."}

# (key in overall, header, scale, decimals, higher_is_better)
MAIN_COLS = [
    ("accuracy", "Acc.", 100, 1, True),
    ("selective_accuracy", "Sel.Acc.", 100, 1, True),
    ("abstain_rate", "Abst.", 100, 1, None),
    ("citation_validity", "Cit.Val.", 100, 1, True),
    ("ev_recall", "Ev.R", 100, 1, True),
    ("ev_precision", "Ev.P", 100, 1, True),
    ("claim_support_rate", "Supp.", 100, 1, True),
    ("unsupported_inference_rate", "Unsupp.", 100, 1, False),
]
COST_COLS = [
    ("cost_llm_calls", "LLM calls", 1, 1, False),
    ("cost_prompt_tokens", "Prompt tok.", 1, 0, False),
    ("cost_completion_tokens", "Compl. tok.", 1, 0, False),
    ("cost_calls", "Tool calls", 1, 1, False),
    ("cost_subagents", "Subagents", 1, 1, False),
    ("cost_latency_ms", "Latency (s)", 0.001, 1, False),
]


def collect(exp_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for sp in sorted(exp_dir.glob("*/*/scores.json")):
        d = sp.parent
        meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
        recs = [json.loads(l) for l in (d / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] \
            if (d / "records.jsonl").exists() else []
        runs.append({"system": d.parent.name, "backbone": d.name, "dir": str(d), "meta": meta,
                     "scores": json.loads(sp.read_text()), "records": recs,
                     "judge": run_judge_summaries(d, [r["id"] for r in recs]) if recs else {}})
    return runs


def _fmt(v: Optional[float], scale: float, dec: int) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "--"
    return f"{v * scale:.{dec}f}"


def _order(runs, order: Optional[list[str]]):
    if not order:
        return runs
    pos = {s: i for i, s in enumerate(order)}
    return sorted(runs, key=lambda r: (pos.get(r["system"].split("+")[0], len(pos)), r["system"], r["backbone"]))


def _write(rows: list[list[str]], header: list[str], csv_path: Path, tex_path: Path, *, caption: str, label: str,
           bold_cols: dict[int, Optional[bool]] | None = None, raw: Optional[list[list[Optional[float]]]] = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    # bold the best value per column (separately per backbone group would be nicer; keep global per table)
    best: dict[int, int] = {}
    if bold_cols and raw:
        for j, hib in bold_cols.items():
            if hib is None:
                continue
            vals = [(i, r[j]) for i, r in enumerate(raw) if r[j] is not None]
            if len(vals) > 1:
                best[j] = (max if hib else min)(vals, key=lambda x: x[1])[0]
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}",
             r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    prev_bb = None
    for i, r in enumerate(rows):
        cells = [esc(c) for c in r]
        for j, bi in best.items():
            if bi == i:
                cells[j] = rf"\textbf{{{cells[j]}}}"
        if prev_bb is not None and r[1] != prev_bb:
            lines.append(r"\midrule")
        prev_bb = r[1]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sig_marker(run, ref_run, alpha=0.05) -> str:
    if ref_run is None or run is ref_run or not run["records"] or not ref_run["records"]:
        return ""
    c = compare(run["records"], ref_run["records"], n_boot=2000)
    return "$^\\dagger$" if c["mcnemar"]["p_value"] < alpha else ""


def build_tables(exp_dir: Path, out_dir: Path, *, ref_system: Optional[str] = None, order: Optional[list[str]] = None,
                 tag: Optional[str] = None) -> dict[str, Path]:
    exp_dir, out_dir = Path(exp_dir), Path(out_dir)
    tag = tag or exp_dir.name
    runs = _order(collect(exp_dir), order)
    written: dict[str, Path] = {}
    if not runs:
        return written
    by_key = {(r["system"], r["backbone"]): r for r in runs}

    # main
    has_j = any(r.get("judge") for r in runs)
    jcols = [(j, k) for j in JUDGES for k in ("judge", "strict")] if has_j else []
    jhead = [("LLM-J " if k == "judge" else "Full ") + JUDGES[j]["short"] for j, k in jcols]
    header = ["System", "Backbone", MAIN_COLS[0][1]] + jhead + [c[1] for c in MAIN_COLS[1:]] + ["N"]
    rows, raw = [], []
    for r in runs:
        o = r["scores"]["overall"]
        ref = by_key.get((ref_system, r["backbone"])) if ref_system else None
        acc = _fmt(o.get("accuracy"), 100, 1)
        if acc != "--" and o.get("acc_ci_lo") is not None:
            acc = f"{acc} ±{(o['acc_ci_hi'] - o['acc_ci_lo']) * 50:.1f}" + _sig_marker(r, ref)
        cells = [acc] + [fmt_judge(r["judge"].get(j), k, ci=(k == "judge")) for j, k in jcols] + \
            [_fmt(o.get(k), s, d) for k, _, s, d, _ in MAIN_COLS[1:]]
        rows.append([r["system"], r["backbone"]] + cells + [str(o.get("n"))])
        raw.append([None, None, o.get("accuracy")] + [(r["judge"].get(j) or {}).get(k) for j, k in jcols] +
                   [o.get(k) for k, *_ in MAIN_COLS[1:]] + [None])
    bold = {j + 2: c for j, c in enumerate([MAIN_COLS[0][4]] + [True] * len(jcols) + [c[4] for c in MAIN_COLS[1:]])}
    _write(rows, header, out_dir / f"{tag}_main.csv", out_dir / f"{tag}_main.tex",
           caption=f"Main results on {esc_caption(tag)} (accuracy with 95\\% bootstrap CI half-width; "
                   f"$\\dagger$: McNemar $p<0.05$ vs. {esc_caption(ref_system or '--')}). Accuracy in \\%; LLM-J = LLM-as-a-Judge score "
                   "(correctness+completeness+grounding)/6 in [0,1] with 95\\% CI half-width, Full = share fully correct and fully grounded; "
                   "* judge model = system backbone, v = judge is the otter\\_v2 verifier.",
           label=f"tab:{tag}_main", bold_cols=bold, raw=raw)
    written["main"] = out_dir / f"{tag}_main.csv"

    # per subject + per reasoning type
    for key, name, lab in (("by_domain", "subject", DOMAIN_LABEL), ("by_reasoning_type", "reasoning", {}),
                           ("by_grade", "grade", {}), ("by_hop_count", "hops", {}), ("by_difficulty", "difficulty", {})):
        groups = sorted({g for r in runs for g in r["scores"].get(key, {})})
        if not groups or groups == ["None"]:
            continue
        header = ["System", "Backbone"] + [lab.get(g, g) for g in groups]
        rows, raw = [], []
        for r in runs:
            sc = r["scores"].get(key, {})
            vals = [sc.get(g, {}).get("accuracy") for g in groups]
            rows.append([r["system"], r["backbone"]] + [_fmt(v, 100, 1) for v in vals])
            raw.append([None, None] + vals)
        _write(rows, header, out_dir / f"{tag}_{name}.csv", out_dir / f"{tag}_{name}.tex",
               caption=f"Accuracy (\\%) by {name} on {esc_caption(tag)}.", label=f"tab:{tag}_{name}",
               bold_cols={j + 2: True for j in range(len(groups))}, raw=raw)
        written[name] = out_dir / f"{tag}_{name}.csv"

    # cost
    header = ["System", "Backbone"] + [c[1] for c in COST_COLS]
    rows, raw = [], []
    for r in runs:
        o = r["scores"]["overall"]
        rows.append([r["system"], r["backbone"]] + [_fmt(o.get(k), s, d) for k, _, s, d, _ in COST_COLS])
        raw.append([None, None] + [o.get(k) for k, *_ in COST_COLS])
    _write(rows, header, out_dir / f"{tag}_cost.csv", out_dir / f"{tag}_cost.tex",
           caption=f"Mean cost per question on {esc_caption(tag)}.", label=f"tab:{tag}_cost")
    written["cost"] = out_dir / f"{tag}_cost.csv"

    # ablations: runs whose system name is "<ref>+flags"
    if ref_system:
        abl = [r for r in runs if r["system"].startswith(ref_system + "+")]
        if abl:
            header = ["Variant", "Backbone", "Acc.", "ΔAcc", "95% CI(Δ)", "McNemar p", "Ev.R", "Supp.", "LLM-J 32B", "LLM-J 8B"]
            rows = []
            for r in [by_key.get((ref_system, b)) for b in sorted({x["backbone"] for x in abl})] + abl:
                if r is None:
                    continue
                o = r["scores"]["overall"]
                ref = by_key.get((ref_system, r["backbone"]))
                if r is ref or ref is None:
                    d = ["--", "--", "--"]
                else:
                    c = compare(r["records"], ref["records"])
                    pb = c["paired_bootstrap"]
                    d = [f"{pb['diff'] * 100:+.1f}", f"[{pb['lo'] * 100:+.1f}, {pb['hi'] * 100:+.1f}]",
                         f"{c['mcnemar']['p_value']:.3f}"]
                name = "full" if r is ref else "− " + r["system"].split("+", 1)[1]
                rows.append([name, r["backbone"], _fmt(o.get("accuracy"), 100, 1)] + d +
                            [_fmt(o.get("ev_recall"), 100, 1), _fmt(o.get("claim_support_rate"), 100, 1),
                             fmt_judge(r["judge"].get("qwen3-32b"), ci=False), fmt_judge(r["judge"].get("qwen3-8b"), ci=False)])
            _write(rows, header, out_dir / f"{tag}_ablation.csv", out_dir / f"{tag}_ablation.tex",
                   caption=f"Ablations of {esc_caption(ref_system)} on {esc_caption(tag)} (paired bootstrap CI and exact McNemar test vs. full).",
                   label=f"tab:{tag}_ablation")
            written["ablation"] = out_dir / f"{tag}_ablation.csv"

        # pairwise significance vs ref
        header = ["System", "Backbone", "N paired", "ΔAcc vs ref", "95% CI", "McNemar p", "b01", "b10"]
        rows = []
        for r in runs:
            ref = by_key.get((ref_system, r["backbone"]))
            if ref is None or r is ref:
                continue
            c = compare(ref["records"], r["records"])
            pb, mc = c["paired_bootstrap"], c["mcnemar"]
            rows.append([r["system"], r["backbone"], str(c["n"]), f"{pb['diff'] * 100:+.1f}",
                         f"[{pb['lo'] * 100:+.1f}, {pb['hi'] * 100:+.1f}]", f"{mc['p_value']:.4f}",
                         str(mc["a_only"]), str(mc["b_only"])])
        if rows:
            _write(rows, header, out_dir / f"{tag}_significance.csv", out_dir / f"{tag}_significance.tex",
                   caption=f"Paired comparison of {esc_caption(ref_system)} (ref) against each system on {esc_caption(tag)}: "
                           "Δ = ref − system; b01/b10 = items only ref / only system answers correctly.",
                   label=f"tab:{tag}_significance")
            written["significance"] = out_dir / f"{tag}_significance.csv"
    return written


def esc_caption(s: Optional[str]) -> str:
    return str(s).replace("_", r"\_")


def build_combined(benches: list[tuple[str, Path]], out_dir: Path, *, order: Optional[list[str]] = None,
                   extra_cols: tuple[tuple[str, str, float, int], ...] = (
                       ("citation_validity", "Cit.Val.", 100, 1), ("ev_recall", "Ev.R", 100, 1),
                       ("claim_support_rate", "Supp.", 100, 1), ("unsupported_inference_rate", "Unsupp.", 100, 1),
                       ("abstain_rate", "Abst.", 100, 1)),
                   extra_on: Optional[str] = None, name: str = "combined_main") -> Optional[Path]:
    """One row per (system, backbone): accuracy ± CI on every bench + provenance columns on `extra_on`
    (default: the last bench). benches = [(column label, exp_dir), ...]."""
    per = [(lab, {(r["system"], r["backbone"]): r for r in collect(Path(d))}) for lab, d in benches]
    keys = sorted({k for _, m in per for k in m})
    if not keys:
        return None
    if order:
        pos = {s: i for i, s in enumerate(order)}
        keys.sort(key=lambda k: (k[1], pos.get(k[0].split("+")[0], len(pos)), k[0]))
    extra_on = extra_on or per[-1][0]
    header = ["System", "Backbone"] + [f"{lab} Acc." for lab, _ in per] + [f"{lab} LLM-J" for lab, _ in per] + \
        [f"{extra_on} {h}" for _, h, _, _ in extra_cols]
    rows, raw = [], []
    for k in keys:
        cells, rv = [], []
        for lab, m in per:
            o = m[k]["scores"]["overall"] if k in m else {}
            a = _fmt(o.get("accuracy"), 100, 1)
            if a != "--" and o.get("acc_ci_lo") is not None:
                a = f"{a} ±{(o['acc_ci_hi'] - o['acc_ci_lo']) * 50:.1f}"
            cells.append(a)
            rv.append(o.get("accuracy"))
        for lab, m in per:                 # LLM-as-a-Judge: the stronger judge (Qwen3-32B); * = same model as the backbone
            js = (m[k]["judge"] if k in m else {}).get("qwen3-32b") or next(iter((m[k]["judge"] if k in m else {}).values()), None)
            cells.append(fmt_judge(js, ci=False))
            rv.append(js.get("judge") if js else None)
        m = dict(per)[extra_on]
        o = m[k]["scores"]["overall"] if k in m else {}
        cells += [_fmt(o.get(c), s, d) for c, _, s, d in extra_cols]
        rv += [o.get(c) for c, *_ in extra_cols]
        rows.append([k[0], k[1]] + cells)
        raw.append([None, None] + rv)
    hib = [True] * (2 * len(per)) + [True, True, True, False, None]
    _write(rows, header, Path(out_dir) / f"{name}.csv", Path(out_dir) / f"{name}.tex",
           caption="Main results: accuracy (\\%, 95\\% bootstrap CI half-width) on each benchmark and provenance metrics on "
                   f"{esc_caption(extra_on)}.", label=f"tab:{name}",
           bold_cols={j + 2: hib[j] if j < len(hib) else None for j in range(len(header) - 2)}, raw=raw)
    return Path(out_dir) / f"{name}.csv"
