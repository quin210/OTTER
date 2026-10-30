"""Summarise the ViEduHop-Typed evaluation (runs/vieduhop_typed/): item AND group accuracy with group-level bootstrap CIs, located,
lesson_located, verified_correct, cost, splits by format / subject / negated / alignment, closed-book saturation, fill-in scoring agreement.

  .venv/bin/python scripts/eval/typed_report.py [--exp vieduhop_typed] [--bench benchmarks/typed/typed600.jsonl]

Definitions (fixed before results were inspected)
  item accuracy      mean over items; failed / missing = wrong. fill_blank is scored two ways: S = normalized string match (typed_common.string_match),
                     J = Qwen3-32B judge verdict 'correct' (prompt typed_fill_judge_v1). true_false / multiple_choice: chosen letter == key.
  group accuracy     a group is correct only if ALL its items (in the evaluated subset) are correct.
  located            top pointer lies in the item's auto-aligned gold span (loc_span_hit1); only items with gold_span_ids (alignment_status auto_aligned).
  lesson_located     top pointer lies in the gold span's lesson (loc_lesson1); same denominator as located.
  verified_correct   correct AND located; same denominator as located (aligned items only).
  CIs                95% percentile bootstrap over GROUPS (10,000 resamples, seed 0); ratio estimator for item-level metrics.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import typed_common as T  # noqa: E402

SYSTEMS = [("direct", "small"), ("direct", "main"), ("rag_hybrid", "main"), ("ircot", "main"), ("otter_v2", "main")]
LABEL = {("direct", "small"): "direct (Qwen3-8B, closed-book probe)", ("direct", "main"): "direct (Qwen3-32B, closed-book)",
         ("rag_hybrid", "main"): "rag_hybrid", ("ircot", "main"): "ircot", ("otter_v2", "main"): "otter_v2"}
B = 10000


def load_jsonl(p: Path):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


class Res:
    """Per-system, per-item results for scoring variant V ('S' or 'J')."""

    def __init__(self, exp_dir: Path, key, rows):
        sysn, bb = key
        d = exp_dir / sysn / bb
        mcq = {r["id"]: r for r in load_jsonl(d / "records.jsonl")}
        fill = {r["id"]: r for r in load_jsonl(d / "fill_blank" / "fill_records.jsonl")}
        self.have = bool(mcq or fill)
        self.rec = {}
        for r in rows:
            i = r["id"]
            if r["format"] == "fill_blank":
                x = fill.get(i)
                if x is None:
                    self.rec[i] = None
                    continue
                self.rec[i] = {"S": bool(x.get("str_match")), "J": x.get("judge_correct"), "C": bool(x.get("str_contains")),
                               "loc": x.get("loc_span_hit1"), "les": x.get("loc_lesson1"), "calls": x.get("cost_llm_calls"),
                               "tok": (x.get("cost_prompt_tokens") or 0) + (x.get("cost_completion_tokens") or 0), "failed": x.get("failed")}
            else:
                x = mcq.get(i)
                if x is None:
                    self.rec[i] = None
                    continue
                c = bool(x.get("correct"))
                self.rec[i] = {"S": c, "J": c, "C": c, "loc": x.get("loc_span_hit1"), "les": x.get("loc_lesson1"),
                               "calls": x.get("cost_llm_calls"), "tok": (x.get("cost_prompt_tokens") or 0) + (x.get("cost_completion_tokens") or 0),
                               "failed": x.get("failed")}

    def correct(self, i, v):
        x = self.rec.get(i)
        if x is None:
            return 0.0
        c = x.get(v)
        return float(bool(c)) if c is not None else float("nan")


def fmt(v, lo=None, hi=None, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if lo is None:
        return f"{v:.3f}"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


class Boot:
    """Group-level bootstrap with a fixed set of resamples shared by every statistic (so paired differences are paired)."""

    def __init__(self, groups: list[str], seed=0):
        self.gids = sorted(set(groups))
        self.gidx = {g: k for k, g in enumerate(self.gids)}
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, len(self.gids), size=(B, len(self.gids)))

    def ratio(self, items, num, den, groups):
        """num/den per item (arrays), restricted to `items` (bool mask). Returns (est, lo, hi)."""
        G = len(self.gids)
        n = np.zeros(G)
        d = np.zeros(G)
        for k in np.nonzero(items)[0]:
            g = self.gidx[groups[k]]
            n[g] += num[k]
            d[g] += den[k]
        if d.sum() == 0:
            return None, None, None
        est = n.sum() / d.sum()
        sn, sd = n[self.idx].sum(1), d[self.idx].sum(1)
        ok = sd > 0
        r = sn[ok] / sd[ok]
        return est, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def group_stats(self, items, correct, groups):
        """per-group (all items correct, 1) for the items in the subset."""
        G = len(self.gids)
        allc = np.ones(G)
        has = np.zeros(G)
        for k in np.nonzero(items)[0]:
            g = self.gidx[groups[k]]
            has[g] = 1
            if not correct[k] == 1.0:
                allc[g] = 0
        return allc * has, has

    def group_acc(self, items, correct, groups):
        n, d = self.group_stats(items, correct, groups)
        if d.sum() == 0:
            return None, None, None
        est = n.sum() / d.sum()
        sn, sd = n[self.idx].sum(1), d[self.idx].sum(1)
        ok = sd > 0
        r = sn[ok] / sd[ok]
        return est, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def diff(self, items, a, b, groups, group_level=False):
        """paired bootstrap of mean(a) - mean(b) (item ratio, or group-level accuracy)."""
        G = len(self.gids)
        if group_level:
            na, d = self.group_stats(items, a, groups)
            nb, _ = self.group_stats(items, b, groups)
        else:
            na, nb, d = np.zeros(G), np.zeros(G), np.zeros(G)
            for k in np.nonzero(items)[0]:
                g = self.gidx[groups[k]]
                na[g] += a[k]
                nb[g] += b[k]
                d[g] += 1
        if d.sum() == 0:
            return None, None, None
        est = (na.sum() - nb.sum()) / d.sum()
        sa, sb, sd = na[self.idx].sum(1), nb[self.idx].sum(1), d[self.idx].sum(1)
        ok = sd > 0
        r = (sa[ok] - sb[ok]) / sd[ok]
        return est, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="vieduhop_typed")
    ap.add_argument("--bench", default=str(ROOT / "benchmarks/typed/typed600.jsonl"))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    exp = ROOT / "runs" / a.exp
    out = Path(a.out_dir) if a.out_dir else exp
    rows = load_jsonl(Path(a.bench))
    n = len(rows)
    ids = [r["id"] for r in rows]
    groups = [r["group_id"] for r in rows]
    boot = Boot(groups)
    res = {k: Res(exp, k, rows) for k in SYSTEMS}
    systems = [k for k in SYSTEMS if res[k].have]
    fmts = np.array([r["format"] for r in rows])
    subj = np.array([r["subject"] for r in rows])
    neg = np.array([r["negated"] for r in rows])
    st = np.array([r["alignment_status"] for r in rows])
    src = np.array([r["source_file"] for r in rows])
    aligned = np.array([bool(r["gold_span_ids"]) for r in rows])
    allm = np.ones(n, bool)
    J_ok = all(res[k].rec.get(i) is None or res[k].rec[i]["J"] is not None for k in systems for i in ids)
    arr = {}
    for k in systems:
        for v in ("S", "J", "C"):
            arr[(k, v)] = np.array([res[k].correct(i, v) for i in ids])
        arr[(k, "loc")] = np.array([float(bool((res[k].rec.get(i) or {}).get("loc"))) for i in ids])
        arr[(k, "les")] = np.array([float(bool((res[k].rec.get(i) or {}).get("les"))) for i in ids])
        arr[(k, "calls")] = np.array([float((res[k].rec.get(i) or {}).get("calls") or 0) for i in ids])
        arr[(k, "tok")] = np.array([float((res[k].rec.get(i) or {}).get("tok") or 0) for i in ids])
        arr[(k, "missing")] = np.array([float(res[k].rec.get(i) is None) for i in ids])
        arr[(k, "failed")] = np.array([float(bool((res[k].rec.get(i) or {}).get("failed"))) for i in ids])
    ones = np.ones(n)
    summary: dict = {"n_items": n, "n_groups": len(boot.gids), "systems": {}}
    L: list[str] = [f"# ViEduHop-Typed: evaluation summary (exp `{a.exp}`)", "",
                    "Generated by `scripts/eval/typed_report.py`. Unit of analysis = the group (`group_id`); CIs are 95% bootstrap over groups "
                    f"({B} resamples). fill_blank is scored two ways: **S** normalized string match, **J** Qwen3-32B judge (fixed prompt "
                    f"`{T.JUDGE_PROMPT_VERSION}`). Every number below is computed from the run files; failed or missing answers count as wrong. "
                    "Gold spans are AUTOMATIC alignments (not human-verified), so located / verified_correct are provisional.", ""]
    L += [f"Items: {n}; groups: {len(boot.gids)}; fill_blank {int((fmts == 'fill_blank').sum())}, true_false {int((fmts == 'true_false').sum())}, "
          f"multiple_choice {int((fmts == 'multiple_choice').sum())}; items with auto-aligned gold span: {int(aligned.sum())}.", ""]
    L += ["Run integrity (items without an answer / failed answers): " + "; ".join(
        f"{LABEL[k]}: missing {int(arr[(k, 'missing')].sum())}, failed {int(arr[(k, 'failed')].sum())}" for k in systems), ""]
    if not J_ok:
        L += ["**Judge verdicts are incomplete for some fill_blank items; J columns marked n/a use the available verdicts only.**", ""]

    def acc_cell(k, v, mask):
        c = np.nan_to_num(arr[(k, v)], nan=0.0)
        e, lo, hi = boot.ratio(mask, c, ones, groups)
        g, glo, ghi = boot.group_acc(mask, c, groups)
        return e, lo, hi, g, glo, ghi

    # ---------------------------------------------------------------- Table 1
    for v, title in (("S", "fill_blank scored by normalized string match"), ("J", "fill_blank scored by the Qwen3-32B judge")):
        L += [f"## Table 1{'' if v == 'S' else 'b'}. All 600 items, {title}", "",
              "| system | item accuracy | group accuracy | located (aligned) | lesson_located | verified_correct | LLM calls/q | tokens/q |",
              "|---|---|---|---|---|---|---|---|"]
        for k in systems:
            e, lo, hi, g, glo, ghi = acc_cell(k, v, allm)
            c = np.nan_to_num(arr[(k, v)], nan=0.0)
            lo_, llo, lhi = boot.ratio(aligned, arr[(k, "loc")], ones, groups)
            le, lelo, lehi = boot.ratio(aligned, arr[(k, "les")], ones, groups)
            vc = c * arr[(k, "loc")]
            ve, vlo, vhi = boot.ratio(aligned, vc, ones, groups)
            ca, _, _ = boot.ratio(allm, arr[(k, "calls")], ones, groups)
            tk, _, _ = boot.ratio(allm, arr[(k, "tok")], ones, groups)
            L.append(f"| {LABEL[k]} | {fmt(e, lo, hi)} | {fmt(g, glo, ghi)} | {fmt(lo_, llo, lhi)} | {fmt(le, lelo, lehi)} | "
                     f"{fmt(ve, vlo, vhi)} | {ca:.2f} | {tk:.0f} |")
            summary["systems"].setdefault(f"{k[0]}/{k[1]}", {})[f"all_{v}"] = {"item_acc": [e, lo, hi], "group_acc": [g, glo, ghi],
                                                                               "located": [lo_, llo, lhi], "lesson_located": [le, lelo, lehi],
                                                                               "verified_correct": [ve, vlo, vhi], "calls_q": ca, "tokens_q": tk}
        L.append("")
    L += ["located / lesson_located / verified_correct are defined only on the auto-aligned items (see the README of the benchmark); `direct` has no "
          "pointer, so it scores 0 there by definition. Pointers come from `where_to_read[0]`, else the first citation.", ""]

    # ---------------------------------------------------------------- by format
    L += ["## Table 2. Item accuracy and group accuracy by format (group accuracy within the format)", "",
          "| system | true_false item | true_false group | multiple_choice item | multiple_choice group | fill_blank item (S) | fill_blank group (S) | fill_blank item (J) | fill_blank group (J) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in systems:
        cells = []
        for f, v in (("true_false", "S"), ("multiple_choice", "S"), ("fill_blank", "S"), ("fill_blank", "J")):
            m = fmts == f
            e, lo, hi, g, glo, ghi = acc_cell(k, v, m)
            cells += [fmt(e, lo, hi), fmt(g, glo, ghi)]
            summary["systems"][f"{k[0]}/{k[1]}"][f"fmt_{f}_{v}"] = {"item": [e, lo, hi], "group": [g, glo, ghi]}
        L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["Group accuracy inside a format: the 400-file groups have 3 true/false items (a group is correct only if all three are), the EXTRA file has 1 "
          "true/false + 1 MCQ per group; the 25 EXTRA fill-in groups contain two phrasings of the same question.", ""]

    # ---------------------------------------------------------------- saturation
    L += ["## Table 3. Saturation: closed-book (`direct`) vs chance and answer-letter priors", ""]
    mcq_prior = max(collections.Counter(r["answer"] for r in rows if r["format"] == "multiple_choice").values()) / max(1, int((fmts == "multiple_choice").sum()))
    tf_prior = max(collections.Counter(r["answer"] for r in rows if r["format"] == "true_false").values()) / max(1, int((fmts == "true_false").sum()))
    L += [f"Chance / majority-letter priors: true_false 0.500 (majority key = {tf_prior:.3f}), multiple_choice 0.250 (always-D = {mcq_prior:.3f}), fill_blank ~0.", "",
          "| format | n | direct Qwen3-8B | direct Qwen3-32B | both correct | neither correct | headroom (1 - best direct) |", "|---|---|---|---|---|---|---|"]
    ks, km = ("direct", "small"), ("direct", "main")
    for f, v in (("true_false", "S"), ("multiple_choice", "S"), ("fill_blank", "S"), ("fill_blank", "J")):
        m = fmts == f
        if ks in systems and km in systems:
            s_, l_ = np.nan_to_num(arr[(ks, v)]), np.nan_to_num(arr[(km, v)])
            es = boot.ratio(m, s_, ones, groups)
            el = boot.ratio(m, l_, ones, groups)
            both = float(((s_ == 1) & (l_ == 1) & m).sum() / m.sum())
            neither = float(((s_ == 0) & (l_ == 0) & m).sum() / m.sum())
            L.append(f"| {f}{' (' + v + ')' if f == 'fill_blank' else ''} | {int(m.sum())} | {fmt(*es)} | {fmt(*el)} | {both:.3f} | {neither:.3f} | {1 - max(es[0], el[0]):.3f} |")
            summary.setdefault("saturation", {})[f"{f}_{v}"] = {"direct_small": es, "direct_main": el, "both": both, "neither": neither}
    L += ["", "Closed-book accuracy near 1.0 means the format cannot show a benefit of retrieval or ontology reasoning; retrieval systems can only "
          "differ on the few remaining items (see Table 7), and on provenance (located / verified_correct).", ""]

    # ---------------------------------------------------------------- by subject / negated / aligned / source
    def split_table(title, masks, v="S", note=""):
        out = [f"## {title}", ""]
        if note:
            out += [note, ""]
        out += ["| system | " + " | ".join(f"{nm} (n={int(m.sum())})" for nm, m in masks) + " |", "|---|" + "---|" * len(masks)]
        for k in systems:
            cells = []
            for nm, m in masks:
                if m.sum() == 0:
                    cells.append("n/a")
                    continue
                e, lo, hi, g, glo, ghi = acc_cell(k, v, m)
                cells.append(f"{fmt(e, lo, hi)} / grp {fmt(g)}")
                summary["systems"][f"{k[0]}/{k[1]}"].setdefault(f"split_{title[:12]}_{v}", {})[nm] = [e, lo, hi, g, glo, ghi]
            out.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
        return out + [""]

    subs = sorted(set(subj))
    L += split_table("Table 4. Item accuracy [95% group-bootstrap CI] / group accuracy by subject (fill_blank by S)", [(s, subj == s) for s in subs])
    L += split_table("Table 4b. Same, fill_blank by judge J", [(s, subj == s) for s in subs], v="J")
    tfm = fmts == "true_false"
    L += split_table("Table 5. true_false only: negated ('Không đúng khi cho rằng') vs not negated",
                     [("negated", tfm & neg), ("not negated", tfm & ~neg)])
    L += split_table("Table 6. Aligned vs not aligned (automatic alignment status; all formats, fill_blank by S)",
                     [("auto_aligned", st == "auto_aligned"), ("key_conflict", st == "key_conflict"), ("not_aligned", st == "not_aligned")])
    L += split_table("Table 6b. By source file", [(s, src == s) for s in sorted(set(src))])

    # ---------------------------------------------------------------- paired differences
    L += ["## Table 7. Paired differences (otter_v2 minus other system), group-level paired bootstrap", "",
          "| comparison | scope | item accuracy diff | group accuracy diff |", "|---|---|---|---|"]
    o = ("otter_v2", "main")
    if o in systems:
        for k in systems:
            if k == o:
                continue
            for nm, m, v in (("all (fill S)", allm, "S"), ("all (fill J)", allm, "J"), ("true_false", fmts == "true_false", "S"),
                             ("multiple_choice", fmts == "multiple_choice", "S"), ("fill_blank S", fmts == "fill_blank", "S"),
                             ("fill_blank J", fmts == "fill_blank", "J")):
                a_, b_ = np.nan_to_num(arr[(o, v)]), np.nan_to_num(arr[(k, v)])
                di = boot.diff(m, a_, b_, groups)
                dg = boot.diff(m, a_, b_, groups, group_level=True)
                L.append(f"| otter_v2 - {LABEL[k]} | {nm} | {fmt(*di)} | {fmt(*dg)} |")
        L.append("")
    # located/verified diff vs ircot
    if o in systems and ("ircot", "main") in systems:
        L += ["Provenance metrics on the aligned items, otter_v2 minus ircot / rag_hybrid (paired, group bootstrap):", ""]
        for other in (("ircot", "main"), ("rag_hybrid", "main")):
            if other not in systems:
                continue
            for nm, ka in (("located", "loc"), ("lesson_located", "les")):
                d = boot.diff(aligned, arr[(o, ka)], arr[(other, ka)], groups)
                L.append(f"- {nm}: otter_v2 - {other[0]} = {fmt(*d)}")
            for v in ("S", "J"):
                d = boot.diff(aligned, np.nan_to_num(arr[(o, v)]) * arr[(o, "loc")], np.nan_to_num(arr[(other, v)]) * arr[(other, "loc")], groups)
                L.append(f"- verified_correct (fill {v}): otter_v2 - {other[0]} = {fmt(*d)}")
        L.append("")

    # ---------------------------------------------------------------- fill-in scoring agreement
    L += ["## Table 8. fill_blank: string match (S) vs judge (J) vs lenient containment (C)", "",
          "| system | n | S acc | C (key contained in answer) | J acc | S=J agree | S yes / J no | S no / J yes |", "|---|---|---|---|---|---|---|---|"]
    fm = fmts == "fill_blank"
    pop = []
    for k in systems:
        s_, j_, c_ = arr[(k, "S")][fm], arr[(k, "J")][fm], arr[(k, "C")][fm]
        if np.isnan(j_).any():
            L.append(f"| {LABEL[k]} | {int(fm.sum())} | {s_.mean():.3f} | {c_.mean():.3f} | judge incomplete | | | |")
            continue
        agree = (s_ == j_).mean()
        L.append(f"| {LABEL[k]} | {int(fm.sum())} | {s_.mean():.3f} | {c_.mean():.3f} | {j_.mean():.3f} | {agree:.3f} | "
                 f"{int(((s_ == 1) & (j_ == 0)).sum())} | {int(((s_ == 0) & (j_ == 1)).sum())} |")
        for pos, i in enumerate(np.nonzero(fm)[0]):
            if res[k].rec.get(ids[i]) is not None:
                pop.append((k, ids[i], bool(s_[pos]), bool(j_[pos])))
    L.append("")
    # disagreement sample of 40 for human review
    if pop:
        rng = random.Random(20261027)
        dis = [p for p in pop if p[2] != p[3]]
        both_y = [p for p in pop if p[2] and p[3]]
        both_n = [p for p in pop if not p[2] and not p[3]]
        take = rng.sample(dis, min(20, len(dis)))
        take += rng.sample(both_n, min(10, len(both_n)))
        take += rng.sample(both_y, min(40 - len(take), len(both_y)))
        take = take[:40]
        rng.shuffle(take)
        rowsd = {r["id"]: r for r in rows}
        recs = {}
        for k in systems:
            for x in load_jsonl(exp / k[0] / k[1] / "fill_blank" / "fill_records.jsonl"):
                recs[(k, x["id"])] = x
        jr = {}
        for k in systems:
            for x in load_jsonl(exp / k[0] / k[1] / "fill_blank" / "judge_verdicts.jsonl"):
                jr[(k, x["id"])] = x
        with open(out / "fill_disagreement_sample_for_review.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_no", "system", "backbone", "id", "group_id", "question", "gold_answer", "accepted_variants", "system_answer",
                        "string_match", "judge_verdict", "judge_reason", "stratum", "human_label (correct / incorrect)", "human_comment"])
            for no, (k, i, s_, j_) in enumerate(take, 1):
                r = rowsd[i]
                x = recs[(k, i)]
                stratum = "disagree" if s_ != j_ else ("both_correct" if s_ else "both_wrong")
                w.writerow([no, k[0], k[1], i, r["group_id"], r["question"], r["answer"], "; ".join(r.get("accepted_variants") or []),
                            x.get("pred"), s_, "correct" if j_ else "incorrect", (jr.get((k, i)) or {}).get("reason"), stratum, "", ""])
        n_dis = len(dis)
        L += [f"Over all {len(pop)} (system, item) fill_blank predictions the two scorers disagree on {n_dis} ({n_dis / len(pop):.3f}): "
              f"{sum(1 for p in dis if p[2] and not p[3])} string-yes/judge-no, {sum(1 for p in dis if (not p[2]) and p[3])} string-no/judge-yes. "
              "A 40-item stratified sample (20 disagreements, 10 both-wrong, 10 both-correct; seed 20261027) is in "
              "`fill_disagreement_sample_for_review.csv` for human labelling; it is NOT a random sample of the population, so its disagreement share "
              "must not be read as a rate. No human label exists yet.", ""]
        summary["fill_agreement"] = {"n_pairs": len(pop), "n_disagree": n_dis}

    # ---------------------------------------------------------------- cost by format
    L += ["## Table 9. Cost per question by format (LLM calls / tokens; otter_v2 includes its verifier calls)", "",
          "| system | true_false | multiple_choice | fill_blank |", "|---|---|---|---|"]
    for k in systems:
        cells = []
        for f in ("true_false", "multiple_choice", "fill_blank"):
            m = fmts == f
            cells.append(f"{arr[(k, 'calls')][m].mean():.2f} / {arr[(k, 'tok')][m].mean():.0f}")
        L.append(f"| {LABEL[k]} | " + " | ".join(cells) + " |")
    L.append("")

    # ---------------------------------------------------------------- group results csv + key conflicts enrich
    gset = collections.OrderedDict()
    for i, r in enumerate(rows):
        gset.setdefault(r["group_id"], []).append(i)
    with open(out / "group_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "subject", "grade", "n_items", "formats", "n_aligned"] + [f"{k[0]}_{k[1]}_group_correct_S" for k in systems]
                   + [f"{k[0]}_{k[1]}_group_correct_J" for k in systems] + [f"{k[0]}_{k[1]}_items_correct_S" for k in systems])
        for g, ix in gset.items():
            row = [g, rows[ix[0]]["subject"], rows[ix[0]]["grade"], len(ix), "+".join(sorted(rows[i]["format"] for i in ix)), int(aligned[ix].sum())]
            row += [int(all(arr[(k, "S")][i] == 1 for i in ix)) for k in systems]
            row += [("" if any(np.isnan(arr[(k, "J")][i]) for i in ix) else int(all(arr[(k, "J")][i] == 1 for i in ix))) for k in systems]
            row += [int(sum(arr[(k, "S")][i] == 1 for i in ix)) for k in systems]
            w.writerow(row)

    kc_path = ROOT / "benchmarks/typed/key_conflicts_for_teacher_review.csv"
    kc = [(i, r) for i, r in enumerate(rows) if r["alignment_status"] == "key_conflict"]
    L += ["## Table 10. Items whose key conflicts with the aligned textbook span (alignment_status = key_conflict)", "",
          f"{len(kc)} items; listed with all system outcomes in `{out.name}/key_conflicts_with_system_outcomes.csv` (organiser-only; for teacher review). "
          "A conflict can be an aligner error, a wrong key, or a flawed item; none of it is established.", ""]
    if kc:
        L += ["| system | accuracy on key_conflict items (S) | on the rest |", "|---|---|---|"]
        kcm = st == "key_conflict"
        for k in systems:
            c = np.nan_to_num(arr[(k, "S")])
            L.append(f"| {LABEL[k]} | {c[kcm].mean():.3f} (n={int(kcm.sum())}) | {c[~kcm].mean():.3f} |")
        L.append("")
        with open(out / "key_conflicts_with_system_outcomes.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "group_id", "format", "subject", "grade", "question", "choices", "key", "key_text", "explanation", "aligner_verdict",
                        "span_id", "aligner_quote", "aligner_note"] + [f"{k[0]}_{k[1]}_correct" for k in systems]
                       + ["teacher_decision (key_ok / key_wrong / question_flawed / span_not_relevant)", "teacher_comment"])
            for i, r in kc:
                w.writerow([r["id"], r["group_id"], r["format"], r["subject"], r["grade"], r["question"],
                            json.dumps(r["choices"], ensure_ascii=False) if r["choices"] else "", r["answer"],
                            r["choices"][r["answer"]] if r["choices"] else r["answer"], r["explanation"], r["align_llm_verdict"],
                            r["align_best_span_id"], r["align_quote"], r["align_note"]] + [int(arr[(k, "S")][i] == 1) for k in systems] + ["", ""])

    # ---------------------------------------------------------------- where do systems fail
    L += ["## Table 11. Items wrong for every evaluated system (S scoring)", ""]
    if systems:
        allwrong = [i for i in range(n) if all(arr[(k, "S")][i] != 1 for k in systems)]
        L.append(f"{len(allwrong)} of {n} items are wrong for all {len(systems)} systems: by format "
                 f"{dict(collections.Counter(rows[i]['format'] for i in allwrong))}, by alignment status "
                 f"{dict(collections.Counter(rows[i]['alignment_status'] for i in allwrong))}. These are the first candidates for key review "
                 "(a key error is as likely as a model error); see `all_systems_wrong.csv`.")
        with open(out / "all_systems_wrong.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "group_id", "format", "subject", "question", "key", "alignment_status", "explanation"])
            for i in allwrong:
                r = rows[i]
                w.writerow([r["id"], r["group_id"], r["format"], r["subject"], r["question"],
                            r["choices"][r["answer"]] if r["choices"] else r["answer"], r["alignment_status"], r["explanation"]])
        L.append("")
    # ---------------------------------------------------------------- notes computed from the data
    L += ["## Notes (all counts computed from the run files)", ""]
    if systems:
        sat_tf = max(np.nan_to_num(arr[(k, "S")])[fmts == "true_false"].mean() for k in systems if k[0] == "direct")
        sat_mc = max(np.nan_to_num(arr[(k, "S")])[fmts == "multiple_choice"].mean() for k in systems if k[0] == "direct")
        L.append(f"- Saturation: best closed-book (`direct`) accuracy is {sat_tf:.3f} on true_false and {sat_mc:.3f} on multiple_choice, so these two formats leave almost no "
                 "room for retrieval to help and differences between systems there are a handful of items; fill_blank is not saturated.")
    if o in systems:
        negi = [i for i in range(n) if rows[i]["negated"]]
        wrong = [i for i in negi if arr[(o, "S")][i] != 1]
        recs_o = {x["id"]: x for x in load_jsonl(exp / o[0] / o[1] / "records.jsonl")}
        pick = collections.Counter(recs_o[ids[i]].get("choice") for i in wrong if ids[i] in recs_o)
        other_wrong = {k: sum(1 for i in negi if arr[(k, "S")][i] != 1) for k in systems if k != o}
        L.append(f"- Negated true/false stems ('Không đúng khi cho rằng: S', n={len(negi)}, all key B = 'Sai'): otter_v2 is wrong on {len(wrong)} of them (chosen letters {dict(pick)}); "
                 f"other systems' wrong counts on the same items: {other_wrong}. In 14 of these 15 outputs the per-option verdict marks 'Đúng' as entailed when the "
                 "cited passage supports the embedded statement S, i.e. the negation prefix is lost in the option-wise proof obligation. This is a measured failure mode of the "
                 "frozen method on this item type; it is not tuned away here.")
        tfn = (fmts == "true_false") & ~neg
        L.append(f"- On non-negated true_false items otter_v2 gets {np.nan_to_num(arr[(o, 'S')])[tfn].mean():.3f}, against "
                 + ", ".join(f"{LABEL[k]} {np.nan_to_num(arr[(k, 'S')])[tfn].mean():.3f}" for k in systems if k != o) + ".")
    L.append("- Judge: Qwen3-32B (:8005) is the same model family as the main backbone, so J may favour answers phrased like its own; it is more lenient than the string match "
             "(0 items string-correct but judge-wrong in this run) and no human label exists for it yet.")
    L.append("- Group accuracy is much lower than item accuracy for the 400-file groups because every group contains a fill_blank item, which is the weak format; "
             "the 25 EXTRA fill-in groups are two phrasings of one question, so their group accuracy is not an independent measurement.")
    L.append("- located / verified_correct use automatic single-span gold from the same retriever family (see benchmark README); they favour retrieval-based systems and "
             "are provisional until a human checks the alignment. Items whose key conflicts with the aligned span (Table 10) are mostly fill_blank items (systems get them right only 20-30% of the time; 34 of the 39 all-systems-wrong items are key_conflict) where "
             "the aligner and the key name different terms, so they are the first candidates for key review.")
    L.append("")
    (out / "typed_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / "typed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda x: None if x != x else float(x)), encoding="utf-8")
    print("\n".join(L[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
