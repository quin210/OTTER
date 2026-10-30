"""Step 5 analysis for ViEduHop-Para (post hoc: designed after the main results; criteria fixed before generation; see docs/VIEDUHOP_PARA.md).
Paired comparison of runs/vieduhop_unrev (ORIGINAL answers, not re-run) with runs/vieduhop_para on the common set. Imports (never edits)
scripts/eval/metrics_r5_em_f1.py (Boot/mean_ci/offline_retrieval) and the loc_score used by scripts/eval/headline_analysis.py."""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/eval")); sys.path.insert(0, str(ROOT / "scripts/para"))
import metrics_r5_em_f1 as M
from para_lib import jl
from otter.evaluation.corpus import SpanCorpus, core_book_filter
from otter.evaluation.source_location import score as loc_score
from otter.schemas.reasoning import FinalAnswer

SYS = ["direct", "rag_hybrid", "ircot", "otter_v2"]
LAB = "post hoc: designed after the main results; criteria fixed before generation"
ROUNDS = {int(x) for x in os.environ.get("PARA_ROUNDS", "1,2").split(",")}
SUF = "" if ROUNDS == {1, 2} else "_round" + "".join(map(str, sorted(ROUNDS)))
para = {r["orig_id"]: r for r in jl(ROOT / "benchmarks/otter_hard/para.jsonl") if r["para_round"] in ROUNDS}
test = {r["id"]: r for r in jl(ROOT / "benchmarks/otter_hard/test.jsonl")}
orig_runs = M.load_runs("vieduhop_unrev"); para_runs = M.load_runs("vieduhop_para")
for s in SYS:
    assert s in para_runs, f"missing para run {s}"
common = [i for i in para if all(i in orig_runs[s] and f"VIEDUHOP-PARA-{i}" in para_runs[s] for s in SYS)]
ids = sorted(common); n = len(ids)
corpus = SpanCorpus.load(book_filter=core_book_filter)


def score_row(r, gold, ans):
    ok = bool(r and r.get("answer_choice") == ans and r.get("status") != "abstain")
    loc = {}
    if r and r.get("status") != "failed":
        try:
            loc = loc_score(gold, FinalAnswer.model_validate(r), corpus) or {}
        except Exception:
            loc = {}
    tu = (r or {}).get("tool_usage") or {}
    h = bool(loc.get("loc_span_hit1"))
    return dict(acc=float(ok), loc=float(h), vc=float(ok and h), calls=float(tu.get("llm_calls") or 0),
                tok=float((tu.get("prompt_tokens") or 0) + (tu.get("completion_tokens") or 0)))


rec = {"orig": {s: {} for s in SYS}, "para": {s: {} for s in SYS}}
for s in SYS:
    for i in ids:
        g = test[i]["gold_span_ids"]; a = test[i]["answer"]
        rec["orig"][s][i] = score_row(orig_runs[s][i], g, a)
        rec["para"][s][i] = score_row(para_runs[s][f"VIEDUHOP-PARA-{i}"], g, a)

# ---- offline retrieval (cache for originals copied from the existing cache, never written back to it)
cache_p = ROOT / "runs/vieduhop_para/retrieval_cache.json"
base_cache = json.loads((ROOT / "runs/vieduhop_unrev/metrics_r5_em_f1_retrieval.json").read_text())
cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
for i in ids:
    if i in base_cache and i not in cache:
        cache[i] = base_cache[i]
cache_p.write_text(json.dumps(cache))
p_items = [dict(para[i], id=f"VIEDUHOP-PARA-{i}") for i in ids]
need_o = [test[i] for i in ids if i not in cache]
cache = M.offline_retrieval(need_o + p_items, cache_p, workers=3)
RET = {}
for s, key in [("rag_hybrid", "hybrid"), ("otter_v2", "otter_v2")]:
    for side in ("orig", "para"):
        k = (lambda i: i) if side == "orig" else (lambda i: f"VIEDUHOP-PARA-{i}")
        ranked = [cache[k(i)][s if s == "otter_v2" else "rag_hybrid"] for i in ids]
        gold = [test[i]["gold_span_ids"] for i in ids]
        h5, _ = M.hit_arrays(ranked, gold, 5); h10, _ = M.hit_arrays(ranked, gold, 10)
        RET[(s, side)] = (h5, h10)
for s in ("ircot",):   # stored first-N collected set (not a ranking), same unit for both sides
    for side, runs_, k in (("orig", orig_runs, lambda i: i), ("para", para_runs, lambda i: f"VIEDUHOP-PARA-{i}")):
        coll = [(runs_[s][k(i)].get("tool_usage") or {}).get("retrieved_all") or [] for i in ids]
        gold = [test[i]["gold_span_ids"] for i in ids]
        RET[(s, side)] = (M.hit_arrays(coll, gold, 5)[0], M.hit_arrays(coll, gold, 10)[0])

# ---- splits
q0 = np.array([para[i]["overlap_q_orig"] for i in ids])
def terc(x): return "T1 (<=0.45)" if x <= 0.45 else ("T2 (0.45-0.62]" if x <= 0.62 else "T3 (>0.62)")
splits = {"all": np.ones(n, bool)}
def addsplit(name, f):
    vals = [f(test[i]) if name != "tercile" else None for i in ids]
for v, f in [("subject", lambda b: b["subject"]), ("hop", lambda b: "1-hop" if b["hop_count"] == 1 else "2+-hop"),
             ("book_specific", lambda b: "book_specific" if b.get("book_specific") else "not book_specific"),
             ("cross_subject", lambda b: "cross_subject" if b["subset"] == "cross_subject" else "core")]:
    vals = np.array([f(test[i]) for i in ids])
    for u in sorted(set(vals)): splits[f"{v}={u}"] = vals == u
tv = np.array([terc(x) for x in q0])
for u in sorted(set(tv)): splits[f"orig_overlap_tercile={u}"] = tv == u

def vec(side, s, k): return np.array([rec[side][s][i][k] for i in ids])
def drops(s, k):
    if k in ("R@5", "R@10"):
        j = 0 if k == "R@5" else 1
        return RET[(s, "orig")][j] - RET[(s, "para")][j]
    return vec("orig", s, k) - vec("para", s, k)
def has_metric(s, k):
    if k == "loc" or k == "vc": return s != "direct" or True
    return True

METR = [("acc", "accuracy"), ("loc", "located"), ("vc", "verified_correct")]
RMETR = [("R@5", "R@5"), ("R@10", "R@10")]
L = [f"# ViEduHop-Para: paired paraphrase evaluation ({LAB})", "",
     f"Rounds {sorted(ROUNDS)} (round 1 = the pre-specified 400 selected items; round 2 = extension, see docs/VIEDUHOP_PARA.md Deviation 1). Common set n={n} items (answered by all four systems on both the original and the paraphrase). Systems {', '.join(SYS)}, backbone main (Qwen3-32B-FP8), exp `vieduhop_para`; ORIGINAL answers = `runs/vieduhop_unrev` (not re-run). "
     "Drop = original - paraphrase in percentage points (positive = worse on paraphrase); 95% paired percentile bootstrap over items (2,000 resamples, seed 7). `direct` has no retrieval/pointer (located = verified_correct = 0 by definition). "
     "ircot R@k = first-k of its collected set (not a ranking).", ""]
def pct(t): return f"{100*t[0]:+.1f} [{100*t[1]:+.1f}, {100*t[2]:+.1f}]"
def mean_pct(x, m=None):
    t = M.mean_ci(x, m); return None if t is None else t
J = {"n": n, "label": LAB, "overall": {}, "splits": {}}
L += ["## Overall (original -> paraphrase, drop)", "", "| system | metric | original | paraphrase | drop [95% CI] |", "|---|---|---|---|---|"]
for s in SYS:
    for k, kn in METR + RMETR:
        if s == "direct" and k in ("R@5", "R@10"): continue
        d = drops(s, k)
        if k in ("R@5", "R@10"):
            j = 0 if k == "R@5" else 1; o, p = RET[(s, "orig")][j].mean(), RET[(s, "para")][j].mean()
        else:
            o, p = vec("orig", s, k).mean(), vec("para", s, k).mean()
        t = M.mean_ci(d)
        L.append(f"| {s} | {kn} | {o:.3f} | {p:.3f} | {pct(t)} |")
        J["overall"][f"{s}|{kn}"] = dict(orig=float(o), para=float(p), drop=t[0], lo=t[1], hi=t[2])
L += ["", "## Cost per question (original -> paraphrase)", "", "| system | calls/q orig | calls/q para | tokens/q orig | tokens/q para |", "|---|---|---|---|---|"]
for s in SYS:
    L.append(f"| {s} | {vec('orig', s, 'calls').mean():.2f} | {vec('para', s, 'calls').mean():.2f} | {vec('orig', s, 'tok').mean():.0f} | {vec('para', s, 'tok').mean():.0f} |")
for k, kn in METR + RMETR:
    L += ["", f"## Drop in {kn} by split (pp, 95% CI); n per split in the header column", ""]
    L += M.md_table(["split", "n"] + [s for s in SYS if not (s == "direct" and k in ("R@5", "R@10"))],
                    [[nm, int(m.sum())] + [pct(M.mean_ci(drops(s, k), m)) if m.sum() else "-" for s in SYS if not (s == "direct" and k in ("R@5", "R@10"))] for nm, m in splits.items()])
    for nm, m in splits.items():
        for s in SYS:
            if s == "direct" and k in ("R@5", "R@10"): continue
            t = M.mean_ci(drops(s, k), m)
            J["splits"][f"{nm}|{s}|{kn}"] = dict(n=int(m.sum()), drop=t[0], lo=t[1], hi=t[2])
# ---- ranking vs hybrid RAG
L += ["", "## Does OTTER's rank relative to hybrid RAG change?", "",
      "Paired difference otter_v2 - rag_hybrid (pp) on originals and on paraphrases, and the difference of differences (para minus orig); same items.", "",
      "| metric | otter_v2 - rag_hybrid (orig) | otter_v2 - rag_hybrid (para) | change [95% CI] |", "|---|---|---|---|"]
def mval(side, s, k):
    if k in ("R@5", "R@10"): return RET[(s, side)][0 if k == "R@5" else 1]
    return vec(side, s, k)
J["rank"] = {}
for k, kn in METR + RMETR:
    do = mval("orig", "otter_v2", k) - mval("orig", "rag_hybrid", k); dp = mval("para", "otter_v2", k) - mval("para", "rag_hybrid", k)
    to, tp, tc = M.mean_ci(do), M.mean_ci(dp), M.mean_ci(dp - do)
    L.append(f"| {kn} | {pct(to)} | {pct(tp)} | {pct(tc)} |")
    J["rank"][kn] = dict(orig=to, para=tp, change=tc)
L += ["", "Ordering of the four systems (point estimates):", ""]
for k, kn in METR:
    for side in ("orig", "para"):
        order = sorted(SYS, key=lambda s: -vec(side, s, k).mean())
        L.append(f"* {kn}, {side}: " + " > ".join(f"{s} {vec(side, s, k).mean():.3f}" for s in order))
for k, kn in RMETR:
    for side in ("orig", "para"):
        ss = [s for s in SYS if s != "direct"]; order = sorted(ss, key=lambda s: -RET[(s, side)][0 if k == "R@5" else 1].mean())
        L.append(f"* {kn}, {side}: " + " > ".join(f"{s} {RET[(s, side)][0 if k == 'R@5' else 1].mean():.3f}" for s in order))
# ---- equal-drop test: is the drop of each system different from the drop of rag_hybrid?
L += ["", "## Drop of each system minus drop of rag_hybrid (paired; pp)", "", "| system | metric | extra drop vs rag_hybrid [95% CI] |", "|---|---|---|"]
for s in SYS:
    if s == "rag_hybrid": continue
    for k, kn in METR + RMETR:
        if s == "direct" and k != "acc": continue
        t = M.mean_ci(drops(s, k) - drops("rag_hybrid", k)); L.append(f"| {s} | {kn} | {pct(t)} |")
L += ["", "## Accuracy drop minus the drop of `direct` (closed-book control; paired; pp)", "",
      "`direct` uses no retrieval, so its accuracy drop measures what the paraphrase costs a model that does not depend on source wording (paraphrase noise, meaning drift, lost cues). A system's drop beyond it is the part not explained by that.", "",
      "| system | accuracy drop minus direct's drop [95% CI] |", "|---|---|"]
for s in SYS:
    if s != "direct":
        L.append(f"| {s} | {pct(M.mean_ci(drops(s, 'acc') - drops('direct', 'acc')))} |")
Path(ROOT / f"runs/vieduhop_para/para_summary{SUF}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
Path(ROOT / f"runs/vieduhop_para/para_summary{SUF}.json").write_text(json.dumps(J, ensure_ascii=False, indent=1))
# ---- LaTeX fragment
rows = []
for s in SYS:
    c = []
    for k, kn in METR + RMETR:
        if s == "direct" and k in ("R@5", "R@10"): c.append("--"); continue
        o = J["overall"][f"{s}|{kn}"]; c.append(f"{100*o['orig']:.1f}$\\to${100*o['para']:.1f} ({100*o['drop']:+.1f} [{100*o['lo']:+.1f},{100*o['hi']:+.1f}])")
    rows.append([M.tex_esc(s)] + c)
tex = M.tex_table(f"ViEduHop-Para ($n={n}$, post hoc: designed after the main results; criteria fixed before generation): original $\\to$ paraphrase, with the drop in points and its paired 95\\% bootstrap CI.",
                  f"tab:para{SUF.replace(chr(95), chr(45))}", ["system", "accuracy", "located", "verified-correct", "R@5", "R@10"], rows, "lccccc")
Path(ROOT / f"runs/tables/vieduhop_para{SUF}.tex").write_text("\n".join(tex) + "\n")
print("\n".join(L))
