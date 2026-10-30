"""Unified post hoc report over the 3,015-row ViEduHop release, grouped into four subcategories.
  S1 Textbook QA   = Core + Cross (non-seen test items = the headline set) + Extra
  S2 Multi-hop     = Multi (3/4/5 hops)
  S3 Abstain       = Abstain (test split) + Twist-Abstain
  S4 Robustness    = Twist + Para + Typed (true/false and multiple-choice rows)
Rows not scored for baselines: Core/Cross dev and seen-in-development (147), Abstain dev (38), Typed fill-in (150, run for four systems only).
Pre-registered headline numbers are unchanged (runs/vieduhop_unrev/headline_final_candidate.md); this file is additional.
Output: runs/vieduhop_unified_report.md (Table A: all systems x subcategories; Table B: per tier within each subcategory)."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
random.seed(11)
ALL = {r["id"]: r for r in map(json.loads, open(ROOT / "benchmarks/vieduhop_v2/vieduhop_all.jsonl"))}
SYS = ["direct", "rag_bm25", "rag_dense", "rag_hybrid", "ircot", "single_agent_tools", "multi_agent_no_workspace", "hipporag", "lightrag", "minirag", "graphrag", "otter_v2"]
GRAPH = {"hipporag", "lightrag", "minirag", "graphrag"}
# tier -> (subcategory, label, experiment dir per system kind)
def core_dir(s): return "vieduhop_unrev_graph_scored" if s in GRAPH else "vieduhop_unrev"
TIERS = [  # (S, label, selector on row, exp function)
    ("S1", "Core", lambda r: r["tier"] == "core" and r["split"] == "test" and not r["seen_in_development"], core_dir),
    ("S1", "Cross", lambda r: r["tier"] == "cross" and r["split"] == "test", core_dir),
    ("S1", "Extra", lambda r: r["tier"] == "extra", lambda s: "vieduhop_extra"),
    ("S2", "Multi", lambda r: r["tier"] == "multihop", lambda s: "vieduhop_multihop2"),
    ("S3", "Abstain", lambda r: r["tier"] == "abstain" and r["split"] == "test", lambda s: "vieduhop_abstain_unrev"),
    ("S3", "Twist-Abstain", lambda r: r["tier"] == "twist_abstain", lambda s: "vieduhop_twist_abstain"),
    ("S4", "Twist", lambda r: r["tier"] == "twist", lambda s: "vieduhop_twist"),
    ("S4", "Para", lambda r: r["tier"] == "para", lambda s: "vieduhop_para"),
    ("S4", "Typed (T/F+MCQ)", lambda r: r["tier"] == "typed" and r["format"] != "fill_blank", lambda s: "vieduhop_typed"),
]
SNAME = {"S1": "S1 Textbook QA", "S2": "S2 Multi-hop", "S3": "S3 Abstain", "S4": "S4 Robustness and formats"}
def load(exp, s):
    p = ROOT / f"runs/{exp}/{s}/main/records.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else None
def ok(r): return float(bool(r.get("correct")) and not r.get("abstain"))
def ciw(v):
    n = len(v); xs = sorted(sum(v[random.randrange(n)] for _ in range(n)) / n for _ in range(1000)); return xs[25], xs[975]
cells = {}   # (S, tier, sys) -> list of (acc, vc, loc, calls); ids per tier
ids = {t[1]: [i for i, r in ALL.items() if t[2](r)] for t in TIERS}
for S, lab, sel, ef in TIERS:
    for s in SYS:
        R = load(ef(s), s)
        if R is None: continue
        got = [i for i in ids[lab] if i in R]
        cells[(lab, s)] = [(i, ok(R[i]), ok(R[i]) * float(bool(R[i].get("loc_span_hit1"))), float(bool(R[i].get("loc_span_hit1"))), R[i].get("cost_llm_calls") or 0) for i in got]
out = ["# Unified report: ViEduHop 3,015 rows in four subcategories (post hoc, additional to the pre-registered headline)", "",
       "Subcategories: " + "; ".join(f"{SNAME[k]} = " + " + ".join(f"{t[1]} ({len(ids[t[1]])})" for t in TIERS if t[0] == k) for k in SNAME) + ".", "",
       f"Scored rows per system at most {sum(len(v) for v in ids.values()):,} of 3,015; not scored: Core/Cross dev and seen-in-development (147), Abstain dev (38), Typed fill-in (150; four systems only). `n` = rows scored for that system; a cell with n below the full count means the run is incomplete.", ""]
def agg(S, s, tiers=None):
    rows = [x for t in TIERS if t[0] == S and (tiers is None or t[1] in tiers) for x in cells.get((t[1], s), [])]
    return rows
def abs_f1(s):
    d = "otter_v2+abstain_below=3,abstention=True" if s == "otter_v2" else s
    p = ROOT / f"runs/vieduhop_abstain_unrev/{d}/main/abstain_scores.json"
    return json.load(open(p))["overall"] if p.exists() else None
out += ["## Table A: all systems x four subcategories", "",
        "S1, S2 and S4: accuracy [95% CI] (n). S3: abstention F1 on the 262 Abstain test rows (official scorer; OTTER at its operating point tau=3 with the explicit option; other systems with the explicit option E). Metrics differ by column, so no macro average is shown.", "",
        "| system | " + " | ".join(SNAME[k] for k in SNAME) + " |", "|---|---|---|---|---|"]
for s in SYS:
    parts = []
    for k in SNAME:
        if k == "S3":
            f = abs_f1(s)
            if f and f["abstain_f1"] is None and f["abstain_recall"] == 0: parts.append(f"F1 0.000 (never abstains; n={f['n']})")   # precision undefined: no abstention at all
            else: parts.append(f"F1 {f['abstain_f1']:.3f} (n={f['n']})" if f else "n/a")
            continue
        rows = agg(k, s)
        if not rows: parts.append("n/a"); continue
        a = [x[1] for x in rows]; lo, hi = ciw(a); parts.append(f"{sum(a)/len(a):.3f} [{lo:.3f},{hi:.3f}] ({len(a)})")
    out.append(f"| {s} | " + " | ".join(parts) + " |")
out.append("")
out += ["## Table B: per subcategory, per tier (accuracy [95% CI]; verified-correct = accuracy and top source pointer inside a gold span, where the tier has gold spans; calls/q)", ""]
for k in SNAME:
    out += [f"### {SNAME[k]}", ""]
    tl = [t[1] for t in TIERS if t[0] == k]
    out += ["| system | " + " | ".join(f"{t} acc (n)" for t in tl) + " | " + " | ".join(f"{t} verified-corr." for t in tl if t not in ("Abstain", "Twist-Abstain", "Typed (T/F+MCQ)")) + " | calls/q |", "|---|" + "---|" * (len(tl) + len([t for t in tl if t not in ("Abstain", "Twist-Abstain", "Typed (T/F+MCQ)")]) + 1)]
    for s in SYS:
        pa, pv = [], []
        for t in tl:
            rows = cells.get((t, s), [])
            if not rows: pa.append("n/a"); pv.append("n/a"); continue
            a = [x[1] for x in rows]; lo, hi = ciw(a)
            pa.append(f"{sum(a)/len(a):.3f} [{lo:.3f},{hi:.3f}] ({len(a)})"); pv.append(f"{sum(x[2] for x in rows)/len(rows):.3f}")
        allr = agg(k, s)
        calls = f"{sum(x[4] for x in allr)/len(allr):.2f}" if allr else "n/a"
        pvv = [v for t, v in zip(tl, pv) if t not in ("Abstain", "Twist-Abstain", "Typed (T/F+MCQ)")]
        out.append(f"| {s} | " + " | ".join(pa) + " | " + " | ".join(pvv) + f" | {calls} |")
    out.append("")
(ROOT / "runs/vieduhop_unified_report.md").write_text("\n".join(out)); print("\n".join(out[:30]))
