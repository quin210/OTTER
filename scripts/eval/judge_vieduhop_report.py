"""Report for the final ViEduHop LLM-judge pass (judge_v2). Reads <run>/answer_judgments.<judge>.jsonl, never calls a model.

  .venv/bin/python scripts/eval/judge_vieduhop_report.py            # writes runs/vieduhop_unrev/llm_judge.md and runs/tables/vieduhop_llm_judge.{csv,tex}

Everything is computed on non-seen items (seen_in_development excluded). Per judge the table uses the COMMON set of items judged for every
listed system (systems with fewer than --min-n judged items are listed separately, not mixed in). Bootstrap: 5000 resamples over items, seed 0.
Nothing here changes the rubric or thresholds; the same numbers are printed for every system whichever way they come out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from otter.evaluation.answer_judge import DIMS, JUDGE_VERSION, load_judgments, quadratic_weighted_kappa  # noqa: E402

ORDER = ["direct", "rag_bm25", "rag_bm25+top_k=10", "rag_dense", "rag_dense+top_k=10", "rag_hybrid", "rag_hybrid+top_k=10", "ircot",
         "single_agent_tools", "multi_agent_no_workspace", "otter_full", "otter_v2", "hipporag", "lightrag", "minirag", "graphrag"]
JUDGES = ["qwen3-32b", "qwen3-8b"]
SHORT = {"qwen3-32b": "32B", "qwen3-8b": "8B"}
N_BOOT = 5000


def boot_ci(v, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    return boot_ci(d, seed)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    from collections import defaultdict
    def avg(v):
        r = np.zeros(len(v)); g = defaultdict(list)
        for i, a in enumerate(v):
            g[a].append(i)
        order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
        for a, ix in g.items():
            r[ix] = np.mean([order[i] for i in ix])
        return r
    rx, ry = avg(x), avg(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_bench(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip())}


def load_run(exp, jid):
    """{system: {"rows": {qid: judgment}, "answers": {qid: answer record}, "meta": judge meta or {}}}"""
    out = {}
    for d in sorted((ROOT / "runs" / exp).glob("*/main")):
        f = d / f"answer_judgments.{jid}.jsonl"
        if not f.exists():
            continue
        rows = load_judgments(f)
        if not rows:
            continue
        ans = {}
        for l in open(d / "answers.jsonl", encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                ans[r["question_id"]] = r
        meta = json.loads((d / f"answer_judgments.{jid}.meta.json").read_text()) if (d / f"answer_judgments.{jid}.meta.json").exists() else {}
        out[d.parent.name] = {"rows": rows, "answers": ans, "meta": meta}
    return out


def acc_of(row, bench_item):
    return float(bool(row.get("choice") == bench_item.get("answer") and not row.get("abstained")))


def mark(sysname, jid):
    # all main-backbone runs share the Qwen3-32B backbone -> 32B judge is a self-judge; the 8B judge is the otter_v2 verifier
    return ("*" if jid == "qwen3-32b" else "") + ("v" if jid == "qwen3-8b" and sysname.startswith("otter_v2") else "")


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} ±{(hi - lo) / 2:.{d}f}"


def system_stats(sysname, jid, ids, run, bench):
    rows = [run[sysname]["rows"][i] for i in ids]
    js = [r["judge_score"] for r in rows]
    acc = [acc_of(r, bench[i]) for r, i in zip(rows, ids)]
    out = {"n": len(ids), "acc": boot_ci(acc), "judge": boot_ci(js), "ans": boot_ci([r["ans_score"] for r in rows]),
           "strict": boot_ci([r["strict"] for r in rows])}
    for d in DIMS:
        out[d] = float(np.mean([r[d] for r in rows])) / 2
    A = np.array(acc)
    J = np.array(js)
    out["corr_ge1_given_acc"] = float(np.mean([r["correctness"] >= 1 for r, a in zip(rows, acc) if a])) if A.sum() else None
    out["corr_eq2_given_acc"] = float(np.mean([r["correctness"] == 2 for r, a in zip(rows, acc) if a])) if A.sum() else None
    out["corr0_given_wrong"] = float(np.mean([r["correctness"] == 0 for r, a in zip(rows, acc) if not a])) if (1 - A).sum() else None
    out["judge_given_acc"] = float(J[A == 1].mean()) if A.sum() else None
    out["judge_given_wrong"] = float(J[A == 0].mean()) if (1 - A).sum() else None
    out["pearson_acc_judge"] = float(np.corrcoef(A, J)[0, 1]) if A.std() > 0 and J.std() > 0 else None
    an = run[sysname]["answers"]
    L = [len((an.get(i) or {}).get("answer") or "") for i in ids]
    out["len_chars"] = float(np.mean(L))
    out["n_claims"] = float(np.mean([len((an.get(i) or {}).get("claims") or []) for i in ids]))
    comp = [r["completeness"] for r, a in zip(rows, acc) if a]
    Lc = [l for l, a in zip(L, acc) if a]
    out["spearman_len_compl_acc"] = spearman(Lc, comp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--out-md", default=str(ROOT / "runs/vieduhop_unrev/llm_judge.md"))
    ap.add_argument("--out-csv", default=str(ROOT / "runs/tables/vieduhop_llm_judge.csv"))
    a = ap.parse_args()
    bench = load_bench("benchmarks/otter_hard/test.jsonl")
    ab_bench = load_bench("benchmarks/abstain/test.jsonl")
    unseen = {i for i, r in bench.items() if not r.get("seen_in_development")}
    L = []
    P = L.append
    runs = {j: load_run("vieduhop_unrev", j) for j in JUDGES}
    table = {}          # (judge) -> {sys: stats}
    commons = {}
    notes = {}
    for j in JUDGES:
        r = runs[j]
        full = {s: v for s, v in r.items() if len(set(v["rows"]) & unseen) >= a.min_n}
        small = sorted(set(r) - set(full))
        names = [s for s in ORDER if s in full] + [s for s in full if s not in ORDER]
        common = set(unseen)
        for s in names:
            common &= set(full[s]["rows"])
        ids = sorted(common) if names else []
        commons[j] = ids
        table[j] = {s: system_stats(s, j, ids, r, bench) for s in names} if ids else {}
        notes[j] = (names, small)

    P("# LLM-as-a-Judge, ViEduHop final runs (`vieduhop_unrev`, backbone Qwen3-32B-FP8)\n")
    P(f"Rubric `{JUDGE_VERSION}` (docs/LLM_JUDGE.md, unchanged; prompt text, thresholds and deterministic rules exactly as frozen before these runs). "
      "Generated by `scripts/eval/judge_vieduhop_report.py` from the per-item verdicts `runs/vieduhop_unrev/<system>/main/answer_judgments.<judge>.jsonl` "
      "(fields: scores, raw scores, rules, reason, `judge_version`, `judge_model`, `prompt_sha1`; settings in `answer_judgments.<judge>.meta.json`: "
      "prompt-template SHA-1, served model, endpoint, temperature 0, thinking off, no guided decoding). `answers.jsonl` was not modified. "
      "Driver: `scripts/eval/run_vieduhop_judge.sh` -> `scripts/eval/judge_vieduhop_final.py` (resumable: cache key = judge_version + item + prompt SHA-1).\n")
    P("**The judge is calibrated only against an AI grader (150 pairs, Claude; `annotation/judge_calibration/`), not against humans** (docs/LLM_JUDGE.md section 6). "
      "On that sample both local judges were stricter than the AI grader (completeness and grounding), correctness agreed (0.98), total-score Pearson r 0.90-0.92. "
      "Use the numbers for ranking systems within one judge, not as absolute quality levels.\n")
    P("## 0. Reading (written after the final pass of 27 Oct 2026, 13:16; the numbers below are copied from the tables of this file at that time)\n")
    P("1. **Same verbosity pattern as on dev, not resolved.** On the 749 common items (32B judge) rag_hybrid 0.870 +-0.017, rag_hybrid+top_k=10 0.867, ircot 0.857, rag_bm25 0.852, rag_dense 0.844 score above otter_v2 0.771 +-0.020; "
      "the paired difference otter_v2 minus ircot is -0.085 [-0.105, -0.065] (8B judge, 1,208 items: -0.052 [-0.066, -0.039]). The ranking is therefore the same as on the dev runs; it is NOT evidence that OTTER is better on explanation quality.")
    P("2. **The gap is not a correctness gap.** On items where both systems chose the key (n=673 vs ircot) the gap is still -0.080 [-0.095, -0.066]; it is carried by completeness (mean component difference -0.166 vs ircot on the 0-1 scale) and, less, by grounding (-0.036). "
      "otter_v2 answers are ~71 characters (option text + 1.6 short claims) against ~260 for RAG/IRCoT. The judge_v2 rubric tells the judge not to penalise short answers, and within a system answer length does not predict completeness (Spearman -0.1 to -0.2 for the RAG variants), "
      "but across systems the compact answers still lose on completeness; I cannot separate 'compact answer states fewer reference points' from 'judge prefers longer text' without human grading. The rubric and thresholds were not touched.")
    P("3. **OTTER versus the agentic/graph baselines.** otter_v2 is above single_agent_tools (+0.089 [+0.067, +0.109]), multi_agent_no_workspace (+0.086 [+0.067, +0.106]) and minirag (+0.064 [+0.037, +0.091]) with the 32B judge, "
      "and indistinguishable from lightrag (-0.016 [-0.039, +0.008]; 8B: +0.010 [-0.006, +0.026]). The tool agents are penalised on completeness and grounding even where accuracy is similar (single_agent_tools accuracy 0.874, multi_agent_no_workspace 0.913 vs otter_v2 0.925 on these items).")
    P("4. **Agreement with accuracy.** LLM-J tracks accuracy across systems (Spearman 0.94 with the 32B judge over 11 systems, 0.97 with the 8B judge over 13), the judge's correctness is >= 1 on 99.5-100% of accurate answers and 0 on wrong ones by rule. "
      "Where it adds information is the within-accuracy spread: among accurate answers LLM-J is 0.91 for rag_hybrid but 0.83 for otter_v2, 0.78 for single_agent_tools, 0.75 for multi_agent_no_workspace (32B).")
    P("5. **Independence of the claim-support (grounding) score for otter_v2.** The 32B judge is not the otter_v2 verifier (that is Qwen3-8B), but it is the backbone that wrote the claims (self-judge, same for every system). "
      "The 8B judge is independent of the backbone but IS the verifier, so its otter_v2 grounding is not independent. Observed: otter_v2 grounding 0.806 (32B) vs 0.796 (8B); the 8B judge does not rate otter_v2 higher than the 32B judge does (in the paired grounding differences against the RAG variants the 8B judge gives the same gap as the 32B judge, and a slightly larger one against ircot), "
      "so there is no visible sign of verifier favouritism, which does not prove independence. Quote the 32B column for claims about otter_v2 and the 8B column as a check.")
    P("6. Both judges agree on the ordering of the main groups; pooled over 8,253 pairs the two judges agree exactly on completeness in 60.5% and on grounding in 79.2% (Pearson r of LLM-J 0.85). "
      "4 minirag items could not be parsed by the 32B judge after the retry (`parse_ok=false`, excluded), hence 749 instead of 750 common items.\n")
    P("## 1. What was and was not judged\n")
    P("* Benchmark: `benchmarks/otter_hard/test.jsonl`, 1,283 items; 75 items with `seen_in_development=true` are excluded and were never sent to a judge -> 1,208 eligible items. "
      "Items are processed in a fixed order (SHA-1 of `judge_final|<id>`, identical for all systems and judges), in prefixes (100, 200, 300 ... items), so every judged subset is a seeded random subset of the 1,208 and the same for every system.")
    for j in JUDGES:
        names, small = notes[j]
        P(f"* **{SHORT[j]} judge** ({'Qwen3-32B-FP8 on :8005 (replica of the backbone model)' if j == 'qwen3-32b' else 'Qwen3-8B on :8001'}): common set of "
          f"{len(commons[j])} non-seen items judged for every system in the table; systems: {', '.join(names) if names else 'none yet'}.")
        if small:
            P(f"  * partially judged, NOT in the table (fewer than {a.min_n} items): {', '.join(small)}.")
        allsys = sorted(d.parent.name for d in (ROOT / 'runs/vieduhop_unrev').glob('*/main') if (d / 'answers.jsonl').exists())
        miss = [s for s in allsys if s not in runs[j]]
        if miss:
            P(f"  * no verdicts for this judge: {', '.join(miss)} (run incomplete when the passes started, or deliberately not in the plan for this judge: the 32B replica was slow, so the top_k=10 variants of rag_bm25 / rag_dense were left to the 8B judge only).")
    P("* Judged only runs whose `answers.jsonl` had an answer row for all 1,283 items (incomplete runs are skipped by the driver). Failed answers count 0 (rule 1), as in accuracy.")
    present = sorted(d.parent.name for d in (ROOT / "runs/vieduhop_unrev").glob("*/main") if (d / "answers.jsonl").exists())
    gr = [g for g in ("hipporag", "lightrag", "minirag", "graphrag") if g in present]
    P(f"* Graph-RAG baselines with an `answers.jsonl` at report time: {', '.join(gr) if gr else 'none'}; the others of hipporag / lightrag / minirag / graphrag were not available and are not judged.")
    P("* `otter_full` was not judged: its run is incomplete (it was moved to `runs/vieduhop_unrev_partial/otter_full`, fewer than 1,283 answers). "
      "A system whose run is incomplete is never judged, and the judge never saw any of its partial answers in the final pass.\n")

    for j in JUDGES:
        names, small = notes[j]
        ids = commons[j]
        P(f"## 2{'a' if j == 'qwen3-32b' else 'b'}. Scores, {SHORT[j]} judge (n = {len(ids)} common non-seen items)\n")
        if not ids:
            P("No common judged items yet.\n")
            continue
        if j == "qwen3-32b":
            P("`*` = the judge is the backbone model of every main-backbone run (self-judge: possible self-preference, equal for all systems here). "
              "The 32B judge is NOT the otter_v2 verifier (that is Qwen3-8B), so for otter_v2 this is the judge to read.\n")
        else:
            P("`v` = the 8B judge is also the otter_v2 verifier (claim support is checked by the same model that then grades it): NOT independent for otter_v2; "
              "for the other systems the 8B judge is independent of the 32B backbone.\n")
        P("LLM-J = (correctness + completeness + grounding)/6, Ans = (correctness + completeness)/4 (answer-only; fair to `direct`, which cites nothing), Full = all three scores 2; "
          "mean ±95% bootstrap CI half-width over items. Acc = chosen option == key (abstention/failure wrong), on the same items.\n")
        P("| system | Acc | LLM-J | Ans | Full | correctness | completeness | grounding |")
        P("|---|---|---|---|---|---|---|---|")
        for s in names:
            t = table[j][s]
            P(f"| {s} | {t['acc'][0]:.3f} | {fmt(*t['judge'])}{mark(s, j)} | {fmt(*t['ans'])} | {t['strict'][0]:.3f} | {t['correctness']:.3f} | {t['completeness']:.3f} | {t['grounding']:.3f} |")
        P("")

    # --- agreement with accuracy / verbosity
    for j in JUDGES:
        names, _ = notes[j]
        if not table[j]:
            continue
        P(f"## 3{'a' if j == 'qwen3-32b' else 'b'}. Agreement with accuracy and verbosity, {SHORT[j]} judge\n")
        P("Accuracy is a rule (chosen == key); the judge's correctness is forced to 0 on wrong choices (rule 2), so the interesting directions are: how often an accurate answer "
          "is NOT scored correctness 2, and how far the LLM-J score is from accuracy. `len` = mean characters of the answer text, `claims` = mean number of claims; "
          "`rho(len,compl)` = Spearman correlation between answer length and the completeness score within the accurate answers of that system.\n")
        P("| system | Acc | LLM-J | corr>=1 given acc | corr=2 given acc | LLM-J given acc | LLM-J given wrong | r(acc, LLM-J) | len | claims | rho(len,compl) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        f3 = lambda x: "--" if x is None else f"{x:.3f}"
        for s in names:
            t = table[j][s]
            P(f"| {s} | {t['acc'][0]:.3f} | {t['judge'][0]:.3f} | {f3(t['corr_ge1_given_acc'])} | {f3(t['corr_eq2_given_acc'])} | {f3(t['judge_given_acc'])} | "
              f"{f3(t['judge_given_wrong'])} | {f3(t['pearson_acc_judge'])} | {t['len_chars']:.0f} | {t['n_claims']:.2f} | {f3(t['spearman_len_compl_acc'])} |")
        accs = [table[j][s]["acc"][0] for s in names]
        jus = [table[j][s]["judge"][0] for s in names]
        ans = [table[j][s]["ans"][0] for s in names]
        P(f"\nRank agreement across the {len(names)} systems: Spearman(Acc, LLM-J) = {f3(spearman(accs, jus))}, Spearman(Acc, Ans) = {f3(spearman(accs, ans))}, "
          f"Spearman(mean answer length, LLM-J) = {f3(spearman([table[j][s]['len_chars'] for s in names], jus))}, "
          f"Spearman(mean answer length, completeness) = {f3(spearman([table[j][s]['len_chars'] for s in names], [table[j][s]['completeness'] for s in names]))}.\n")

    # --- paired differences
    P("## 4. Paired differences in LLM-J (otter_v2 minus other system, same items, 95% bootstrap CI)\n")
    P("| judge | n | vs system | diff LLM-J | diff Ans | diff grounding(/2) |")
    P("|---|---|---|---|---|---|")
    for j in JUDGES:
        names, _ = notes[j]
        if "otter_v2" not in names:
            continue
        ids = commons[j]
        rv = runs[j]["otter_v2"]["rows"]
        for s in names:
            if s == "otter_v2":
                continue
            ro = runs[j][s]["rows"]
            d1 = boot_diff([rv[i]["judge_score"] for i in ids], [ro[i]["judge_score"] for i in ids])
            d2 = boot_diff([rv[i]["ans_score"] for i in ids], [ro[i]["ans_score"] for i in ids])
            d3 = boot_diff([rv[i]["grounding"] / 2 for i in ids], [ro[i]["grounding"] / 2 for i in ids])
            P(f"| {SHORT[j]} | {len(ids)} | {s} | {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}] | {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}] | {d3[0]:+.3f} [{d3[1]:+.3f}, {d3[2]:+.3f}] |")
    P("")

    # --- gap decomposition: where does the otter_v2 gap come from (correctness vs explanation style)?
    P("## 4b. Where the otter_v2 gap comes from (otter_v2 minus comparator, same items)\n")
    P("`both acc` = items on which both systems chose the key (the gap there is pure explanation/citation style, not correctness); `differ` = exactly one of the two is accurate; "
      "`book_specific` / `other` split the common items by the benchmark flag. Entries: mean difference in LLM-J, [95% CI]; n.\n")
    P("| judge | comparator | all | both acc | differ | book_specific | other | completeness diff (both acc, /2) | grounding diff (both acc, /2) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for j in JUDGES:
        names, _ = notes[j]
        if "otter_v2" not in names:
            continue
        ids = commons[j]
        rv = runs[j]["otter_v2"]["rows"]
        for c in ("ircot", "rag_hybrid", "single_agent_tools"):
            if c not in names:
                continue
            ro = runs[j][c]["rows"]
            def cell(sel):
                sel = [i for i in sel]
                if len(sel) < 5:
                    return f"n={len(sel)}"
                m, lo, hi = boot_diff([rv[i]["judge_score"] for i in sel], [ro[i]["judge_score"] for i in sel])
                return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(sel)}"
            both = [i for i in ids if acc_of(rv[i], bench[i]) and acc_of(ro[i], bench[i])]
            diff = [i for i in ids if acc_of(rv[i], bench[i]) != acc_of(ro[i], bench[i])]
            bs = [i for i in ids if bench[i].get("book_specific")]
            ot = [i for i in ids if not bench[i].get("book_specific")]
            cd = grd = float("nan")
            if both:
                cd = np.mean([rv[i]["completeness"] - ro[i]["completeness"] for i in both]) / 2
                grd = np.mean([rv[i]["grounding"] - ro[i]["grounding"] for i in both]) / 2
            P(f"| {SHORT[j]} | {c} | {cell(ids)} | {cell(both)} | {cell(diff)} | {cell(bs)} | {cell(ot)} | {cd:+.3f} | {grd:+.3f} |")
    P("")

    # --- 32B vs 8B judge consistency
    P("## 5. Consistency between the two judges (items judged by both, all systems pooled)\n")
    pairs = []
    for s in set(runs["qwen3-32b"]) & set(runs["qwen3-8b"]):
        for i in set(runs["qwen3-32b"][s]["rows"]) & set(runs["qwen3-8b"][s]["rows"]):
            if i in unseen:
                pairs.append((runs["qwen3-32b"][s]["rows"][i], runs["qwen3-8b"][s]["rows"][i]))
    if pairs:
        P(f"n = {len(pairs)} (system, item) pairs. diff = mean(8B - 32B).\n")
        P("| dimension | exact | QWK | diff |")
        P("|---|---|---|---|")
        for d in DIMS:
            x = [p[0][d] for p in pairs]; y = [p[1][d] for p in pairs]
            P(f"| {d} | {np.mean(np.array(x) == np.array(y)):.3f} | {quadratic_weighted_kappa(x, y):.3f} | {np.mean(np.array(y) - np.array(x)):+.3f} |")
        x = [p[0]["judge_score"] for p in pairs]; y = [p[1]["judge_score"] for p in pairs]
        P(f"| LLM-J | Pearson r = {np.corrcoef(x, y)[0, 1]:.3f} | | {np.mean(np.array(y) - np.array(x)):+.3f} |\n")
    else:
        P("No overlap yet.\n")

    # --- abstain
    P("## 6. OTTER-Abstain (`runs/vieduhop_abstain_unrev`, 262 items = 131 unanswerable + 131 controls)\n")
    any_ab = False
    for j in JUDGES:
        ra = load_run("vieduhop_abstain_unrev", j)
        if not ra:
            continue
        any_ab = True
        names = [s for s in ORDER if s in ra] + sorted(s for s in ra if s not in ORDER)
        common = set(ab_bench)
        for s in names:
            common &= set(ra[s]["rows"])
        ids = sorted(common)
        P(f"**{SHORT[j]} judge**, common judged items n = {len(ids)} ({sum(1 for i in ids if ab_bench[i].get('answerable') is False)} unanswerable). "
          "Behaviour = abstain on unanswerable, correct choice on controls (rule, `abstain_metrics.py` convention); LLM-J by item type.\n")
        P("| system | behaviour acc | LLM-J all | LLM-J unanswerable | LLM-J controls | Full |")
        P("|---|---|---|---|---|---|")
        for s in names:
            rows = {i: ra[s]["rows"][i] for i in ids}
            beh = [float(r["abstained"] if ab_bench[i].get("answerable") is False else acc_of(r, ab_bench[i])) for i, r in rows.items()]
            un = [r["judge_score"] for i, r in rows.items() if ab_bench[i].get("answerable") is False]
            co = [r["judge_score"] for i, r in rows.items() if ab_bench[i].get("answerable") is not False]
            allj = boot_ci([r["judge_score"] for r in rows.values()])
            P(f"| {s} | {np.mean(beh):.3f} | {fmt(*allj)}{mark(s, j)} | {np.mean(un) if un else float('nan'):.3f} | {np.mean(co) if co else float('nan'):.3f} | {np.mean([r['strict'] for r in rows.values()]):.3f} |")
        P("")
    if not any_ab:
        P("No Abstain verdicts yet.\n")

    P("## 7. Caveats (read before quoting any number)\n")
    P("1. Judge validity: calibrated on 150 pairs against an AI grader (Claude), not humans; both local judges are stricter than that grader (mainly completeness and grounding), so absolute levels are deflated by roughly 0.1.")
    P("2. Self-judge: every system in `vieduhop_unrev` runs on Qwen3-32B; the 32B judge is the same model (`*`). Any self-preference is common to all systems, but it can interact with answer style. The 8B judge is independent of the backbone, except that it is the otter_v2 verifier (`v`), i.e. not independent for exactly the system whose claim support it grades.")
    P("3. Verbosity / completeness: the reference answers of OTTER-Hard mostly restate the correct option. judge_v2 explicitly does not penalise short answers that state the key content, yet the dev findings (docs/LLM_JUDGE.md section 7) showed RAG/IRCoT outscoring compact OTTER answers; sections 3a/3b show answer length and the length-completeness correlation so the reader can see how much of any gap is style rather than correctness. The rubric was not changed after seeing these results.")
    P("4. `direct` cites nothing, hence grounding = 0 by rule and Full = 0; compare it only on Ans.")
    P("5. Subsets: the judged subset is a seeded random prefix of the 1,208 non-seen items, not the whole set (see section 1 for the size per judge); CIs reflect that. Benchmark label noise reaches the judge.")
    P("6. Accuracy and the headline metrics (`docs/HEADLINE_METRICS.md`) are unaffected: the LLM-judge is a secondary, explanation-quality metric.")
    Path(a.out_md).write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- csv / tex fragment
    hdr = ["System", "Backbone", "Acc.", "LLM-J 32B", "Ans 32B", "Full 32B", "N32", "LLM-J 8B", "Ans 8B", "Full 8B", "N8"]
    allsys = [s for s in ORDER if s in table["qwen3-32b"] or s in table["qwen3-8b"]]
    rowsout = []
    for s in allsys:
        r = [s, "main"]
        t32, t8 = table["qwen3-32b"].get(s), table["qwen3-8b"].get(s)
        t = t32 or t8
        r.append(f"{100 * t['acc'][0]:.1f} ±{100 * (t['acc'][2] - t['acc'][1]) / 2:.1f}")
        for tt, jid in ((t32, "qwen3-32b"), (t8, "qwen3-8b")):
            if tt:
                r += [fmt(*tt["judge"]) + mark(s, jid), fmt(*tt["ans"]), f"{tt['strict'][0]:.3f}" + mark(s, jid), str(tt["n"])]
            else:
                r += ["--", "--", "--", "0"]
        rowsout.append(r)
    import csv
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rowsout)
    esc = lambda x: x.replace("_", "\\_")
    tex = ["\\begin{table}[t]", "\\centering",
           "\\caption{LLM-as-a-Judge (judge\\_v2) on ViEduHop final runs, non-seen items only. Acc.\\ in \\% (judged subset of the 32B column), LLM-J = (correctness+completeness+grounding)/6 with 95\\% CI half-width, "
           "Ans = (correctness+completeness)/4, Full = share fully correct and fully grounded; N = common judged items per judge; * judge model = system backbone (self-judge), "
           "v = judge is the otter\\_v2 verifier. Judge calibrated against an AI grader, not humans.}",
           "\\label{tab:vieduhop_llm_judge}", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llrrrrrrrrr}", "\\toprule",
           " & ".join(esc(h) for h in hdr[:2] + hdr[2:]) + " \\\\", "\\midrule"]
    for r in rowsout:
        tex.append(" & ".join(esc(c) for c in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(a.out_csv).with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
