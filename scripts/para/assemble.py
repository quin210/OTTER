"""Step 3: assemble benchmarks/otter_hard/para.jsonl, funnel, stats, human-check CSV from raw/gen.jsonl + raw/checks.jsonl. No system output used."""
import csv, json, random
from collections import Counter, defaultdict
import numpy as np
from para_lib import *

OUT = ROOT / "benchmarks/otter_hard/para"
SEL = {1: json.load(open(OUT / "selection.json"))["ids"], 2: json.load(open(OUT / "selection_round2.json"))["ids"]}
RAW = {1: OUT / "raw", 2: OUT / "raw_r2"}
rnd_of = {i: r for r, ids in SEL.items() for i in ids}
sel = SEL[1] + SEL[2]
test = {r["id"]: r for r in jl(ROOT / "benchmarks/otter_hard/test.jsonl")}
gens = defaultdict(dict); checks = {}
for r_, d_ in RAW.items():
    for g in jl(d_ / "gen.jsonl"):
        gens[g["id"]][g["cand"]] = g
    checks.update({r["id"]: r for r in jl(d_ / "checks.jsonl")})
stages = ["gen", "a", "b", "c", "d"]
fround = defaultdict(Counter); funnel = Counter(); fsub = defaultdict(Counter); cand_funnel = Counter()
rows = []
for i in sel:
    it = test[i]; ck = checks.get(i)
    if ck is None:
        continue
    best = 0
    for c in ck["cands"]:
        k = stages.index(c["stage_reached"]); best = max(best, k)
        for s in range(k + 1):
            cand_funnel[stages[s]] += 1
    # item-level: items with >=1 candidate reaching stage s
    for s in range(best + 1):
        funnel[stages[s]] += 1; fsub[it["subject"]][stages[s]] += 1; fround[rnd_of[i]][stages[s]] += 1
    fsub[it["subject"]]["n"] += 1
    if ck["chosen"] is None:
        continue
    p = gens[i][ck["chosen"]]["parsed"]
    cc = next(c for c in ck["cands"] if c["cand"] == ck["chosen"])
    gt = gold_text(it["gold_span_ids"])
    r = dict(it)
    r.update(id="VIEDUHOP-PARA-" + i, orig_id=i, orig_question=it["question"], orig_options=it["options"], orig_subset=it["subset"],
             question=p["question"].strip(), options={k: p["options"][k].strip() for k in LETTERS},
             overlap_orig=cc["b"]["info"]["overlap_orig"], overlap_para=cc["b"]["info"]["overlap_para"],
             overlap_q_orig=overlap(it["question"], gt), overlap_q_para=overlap(p["question"], gt),
             para_candidate=ck["chosen"], para_round=rnd_of[i], para_checks={"a": cc["a"]["pass"], "b": cc["b"]["pass"], "c": cc["c"]["pass"], "d": cc["d"]["pass"],
                                                      "answerability_correct_of_3": cc["d"]["info"]["n_correct"], "judge": cc["c"]["info"].get("judge"),
                                                      "closed_book_correct_of_4": ck["closed_book_probe"]["correct"]},
             subset="para", review_status="auto_verified", seen_in_development=False, benchmark_name="ViEduHop-Para",
             para_generator={"model": "main model", "temperature": 0.7, "prompt_sha256": json.load(open(OUT / "prompts/HASHES.json"))["gen_sha256"]})
    for k in ("closed_book", "verification", "explanation", "reference_answer", "quotes", "quotes_verbatim", "reasoning_steps", "link"):
        pass  # kept as in the original (answer/gold unchanged); explanation/quotes describe the source, not the wording
    rows.append(r)
with open(ROOT / "benchmarks/otter_hard/para.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n_sel = len(sel)
stats = {"n_selected": n_sel, "n_checked": len(checks), "n_para": len(rows),
         "item_funnel": {s: funnel[s] for s in stages}, "candidate_funnel_evaluated": {s: cand_funnel[s] for s in stages},
         "per_subject": {s: dict(v) for s, v in fsub.items()}, "share_lost": 1 - len(rows) / max(1, len(checks)), "n_selected_round": {str(k): len(v) for k, v in SEL.items()}}
ov0 = np.array([r["overlap_orig"] for r in rows]); ov1 = np.array([r["overlap_para"] for r in rows])
q0 = np.array([r["overlap_q_orig"] for r in rows]); q1 = np.array([r["overlap_q_para"] for r in rows])
q = lambda a: [round(float(x), 3) for x in np.percentile(a, [5, 25, 50, 75, 95])] + [round(float(a.mean()), 3)]
stats["per_round_item_funnel"] = {str(k): dict(v) for k, v in fround.items()}
stats["overlap_stem_options"] = {"orig_p5_25_50_75_95_mean": q(ov0), "para_p5_25_50_75_95_mean": q(ov1)}
stats["overlap_question_only"] = {"orig_p5_25_50_75_95_mean": q(q0), "para_p5_25_50_75_95_mean": q(q1)}
# selected vs accepted composition
sel_ov = np.array([overlap(mcq_text(test[i]["question"], test[i]["options"]), gold_text(test[i]["gold_span_ids"])) for i in sel])
stats["selected_800_orig_overlap_p5_25_50_75_95_mean"] = q(sel_ov)
comp = {}
for v in ["subject", "difficulty", "hop_count", "book_specific", "subset"]:
    comp[v] = {"selected": dict(Counter(str(test[i][v]) for i in sel)), "para": dict(Counter(str(r[v] if v != "subset" else r["orig_subset"]) for r in rows))}
stats["composition"] = comp
stats["closed_book_correct_of_4_para"] = dict(Counter(r["para_checks"]["closed_book_correct_of_4"] for r in rows))
stats["answerability_correct_of_3"] = dict(Counter(r["para_checks"]["answerability_correct_of_3"] for r in rows))
stats["chosen_candidate"] = dict(Counter(r["para_candidate"] for r in rows))
# first-failure reasons (candidate level)
why = Counter()
for ck in checks.values():
    for c in ck["cands"]:
        if c["passed_all"]:
            continue
        st = c["stage_reached"]; nxt = stages[stages.index(st) + 1]
        if nxt == "a":
            why["a:" + str(c["a"]["info"]).split(":")[0]] += 1
        elif nxt == "b":
            why["b:overlap_abs" if c["b"]["info"]["overlap_para"] > 0.35 else "b:overlap_rel"] += 1
        elif nxt == "c":
            why["c:judge"] += 1
        else:
            why["d:answerability"] += 1
stats["first_failure_reasons_per_candidate"] = dict(why.most_common())
(OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
# human check sample
rng = random.Random(20261028)
samp = rng.sample(rows, min(40, len(rows)))
with open(OUT / "human_check_40.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["para_id", "orig_id", "subject", "key", "orig_question", "orig_A", "orig_B", "orig_C", "orig_D", "para_question", "para_A", "para_B", "para_C", "para_D",
                "overlap_orig", "overlap_para", "same_meaning (1/0)", "key_still_correct (1/0)", "distractors_still_wrong (1/0)", "fluent (1/0)", "note"])
    for r in samp:
        w.writerow([r["id"], r["orig_id"], r["subject"], r["answer"], r["orig_question"], *[r["orig_options"][k] for k in LETTERS], r["question"], *[r["options"][k] for k in LETTERS],
                    round(r["overlap_orig"], 3), round(r["overlap_para"], 3), "", "", "", "", ""])
print(json.dumps(stats, ensure_ascii=False, indent=1))
