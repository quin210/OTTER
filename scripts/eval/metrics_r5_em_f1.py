"""Additional metrics R@5 / R@10, EM and F1 for ViEduHop (SECONDARY; definitions fixed in docs/METRICS_R5_EM_F1.md before computing).

  .venv/bin/python scripts/eval/metrics_r5_em_f1.py            # regenerates everything (re-run when more systems finish)
  .venv/bin/python scripts/eval/metrics_r5_em_f1.py --no-retrieval   # skip the offline retrieval (uses the cache only)

Outputs: runs/vieduhop_unrev/metrics_r5_em_f1.md, runs/tables/metrics_r5_em_f1.tex, runs/vieduhop_unrev/metrics_r5_em_f1_retrieval.json (cache).
Retrieval is recomputed OFFLINE (bge-m3 query embeddings on :8002 and local indexes, no LLM) with the code the systems use. Scoring code that exists
(otter/evaluation/rag_variants.py, otter/agents/v2.py retrieve(), scripts/eval/typed_common.py, abstain_scores.json) is wrapped, never edited.
Not a pre-registered headline metric: see docs/HEADLINE_METRICS.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

LABEL = ("Additional metrics requested after the pre-registered headline (docs/HEADLINE_METRICS.md); definitions fixed before computing "
         "(docs/METRICS_R5_EM_F1.md). SECONDARY: they do not replace the headline.")
ORDER = ["direct", "rag_bm25", "rag_dense", "rag_hybrid", "rag_bm25+top_k=10", "rag_dense+top_k=10", "rag_hybrid+top_k=10", "ircot",
         "single_agent_tools", "multi_agent_no_workspace", "otter_full", "otter_v2", "hipporag", "lightrag", "minirag", "graphrag"]
GRAPH = ["hipporag", "lightrag", "minirag", "graphrag"]
RAG_MODES = {"rag_bm25": "bm25", "rag_dense": "dense", "rag_hybrid": "hybrid"}
LETTERS = "ABCD"
B = 2000
SEED = 7
NA_AGENT = "n/a (tool-calling agent; no span ranking stored)"
NA_GRAPH = "n/a (no span-level ranking exposed)"


# ============================================================================ generic helpers
def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_runs(exp, bb="main", sub="answers.jsonl"):
    out = {}
    for d in sorted((ROOT / "runs" / exp).glob(f"*/{bb}")):
        f = d / sub
        if f.exists():
            out[d.parent.name] = {r["question_id"]: r for r in jl(f)}
    return out


def sort_systems(names):
    return [s for s in ORDER if s in names] + sorted(s for s in names if s not in ORDER)


class Boot:
    """Bootstrap weight matrices (B x n), item-level or group-level (whole groups resampled), cached; every statistic is a weighted statistic."""
    _cache: dict = {}

    @classmethod
    def weights(cls, n, groups=None):
        key = (n, None if groups is None else tuple(groups))
        if key in cls._cache:
            return cls._cache[key]
        rng = np.random.default_rng(SEED)
        if groups is None:
            W = np.zeros((B, n))
            for b in range(B):
                W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        else:
            ug, gi = np.unique(np.asarray(groups), return_inverse=True)
            G = len(ug)
            cnt = np.zeros((B, G))
            for b in range(B):
                cnt[b] = np.bincount(rng.integers(0, G, G), minlength=G)
            W = cnt[:, gi]
        cls._cache[key] = W
        return W


def ci(vals):
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mean_ci(x, mask=None, groups=None):
    """x: per-item values (full set); mask: subset (bool array); returns (mean, lo, hi, n)."""
    x = np.asarray(x, float)
    mask = np.ones(len(x), bool) if mask is None else np.asarray(mask, bool)
    if mask.sum() == 0:
        return None
    W = Boot.weights(len(x), groups)[:, mask]
    bs = (W @ x[mask]) / np.maximum(W.sum(1), 1e-9)
    lo, hi = ci(bs)
    return float(x[mask].mean()), lo, hi, int(mask.sum())


def macro_f1_ci(gold, pred, mask=None, groups=None, K=4):
    """gold, pred: int arrays (pred -1 = no valid prediction). Macro-F1 over classes 0..K-1, F1=0 when P+R=0."""
    gold = np.asarray(gold)
    pred = np.asarray(pred)
    mask = np.ones(len(gold), bool) if mask is None else np.asarray(mask, bool)
    if mask.sum() == 0:
        return None

    def f(Wm, g, p):
        tot = 0.0
        for c in range(K):
            tp = Wm @ ((g == c) & (p == c)).astype(float)
            pc = Wm @ (p == c).astype(float)
            gc = Wm @ (g == c).astype(float)
            tot = tot + np.where(pc + gc > 0, 2 * tp / np.maximum(pc + gc, 1e-9), 0.0)
        return tot / K

    W = Boot.weights(len(gold), groups)[:, mask]
    bs = f(W, gold[mask], pred[mask])
    point = float(f(np.ones((1, int(mask.sum()))), gold[mask], pred[mask])[0])
    lo, hi = ci(bs)
    return point, lo, hi, int(mask.sum())


def cell(t, nd=3):
    if t is None:
        return "-"
    v, lo, hi = t[:3]
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def pt(t, nd=3):
    return "-" if t is None else f"{t[0]:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def tex_esc(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("†", "$^\\dagger$")


def tex_table(caption, label, header, rows, spec, note=None):
    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\setlength{\\tabcolsep}{3pt}",
         f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    if note:
        L.append(f"\\par\\smallskip{{\\scriptsize {note}}}")
    L.append("\\end{table}")
    return L


# ============================================================================ offline retrieval
def offline_retrieval(items, cache_path, workers=3, enabled=True):
    """Ranked span lists for rag_bm25 / rag_dense / rag_hybrid (top 10) and otter_v2.retrieve() (up to 10), keyed by item id. Cached."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if it["id"] not in cache] if enabled else []
    if todo:
        from otter.agents.runtime import QuestionInput
        from otter.agents.systems import get_system
        from otter.evaluation.rag_variants import retrieve as rag_retrieve
        v2 = get_system("otter_v2", backbone="main")

        def one(it):
            q = QuestionInput.from_item(it)
            query = q.question + " " + " ".join((q.choices or {}).values())
            rec = {m: rag_retrieve(query, mode=mode, subject=q.subject, top_k=10) for m, mode in RAG_MODES.items()}
            spans, _hints, _info = v2.retrieve(q)
            rec["otter_v2"] = [s["span_id"] for s in spans]
            return it["id"], rec

        print(f"offline retrieval: {len(todo)} items ({len(cache)} cached)", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, (iid, rec) in enumerate(ex.map(one, todo), 1):
                cache[iid] = rec
                if k % 100 == 0:
                    print(f"  {k}/{len(todo)}", flush=True)
                    cache_path.write_text(json.dumps(cache))
        cache_path.write_text(json.dumps(cache))
    return cache


def stored_ranking(sysname, row):
    """Ranked list stored in the run itself (used as a cross-check, and the only source for ircot / graph baselines)."""
    tu = (row or {}).get("tool_usage") or {}
    if sysname in RAG_MODES or sysname.startswith("rag_"):
        return tu.get("retrieved_top10")
    if sysname == "ircot":
        return tu.get("retrieved_all")
    if sysname == "otter_v2":
        rb = tu.get("retrieved_by_round")
        return rb[0] if rb else None
    return None


def graph_passages(exp, sysname, bb="main"):
    f = ROOT / "runs" / exp / sysname / bb / "answers.raw.jsonl"
    out = {}
    if not f.exists():
        return None
    for r in jl(f):
        ps = r.get("retrieved_passages")
        if ps:
            ps = sorted(ps, key=lambda p: p.get("rank", 0))
            shown = set(r.get("shown_passage_ranks") or [])
            out[r["question_id"]] = ([list(p.get("span_ids") or []) for p in ps],
                                     [list(p.get("span_ids") or []) for p in ps if p.get("rank") in shown])
        else:
            out[r["question_id"]] = ([], [])
    return out


# ============================================================================ MCQ blocks
def pred_class(row):
    """Chosen option exactly as in headline_analysis.py (abstention / failure / missing -> None)."""
    if row is None or row.get("status") == "abstain":
        return None
    c = row.get("answer_choice")
    return c if c in LETTERS else None


def build_subsets(bench, ids, hard_ids):
    return {
        "all": lambda b: True,
        "book_specific": lambda b: bool(b.get("book_specific")),
        "cross_subject": lambda b: b.get("subset") == "cross_subject",
        "Hard tier": lambda b: b["id"] in hard_ids,
    }


def mcq_block(bench_items, runs, ids, subsets, systems):
    """EM (= accuracy) and macro-F1 (A-D) for each system on the ordered id list, per subset."""
    gold = np.array([LETTERS.index(bench_items[i]["answer"]) for i in ids])
    masks = {k: np.array([bool(f(bench_items[i])) for i in ids]) for k, f in subsets.items()}
    res = {}
    for s in systems:
        rows = [runs[s].get(i) for i in ids]
        preds = [pred_class(r) for r in rows]
        pi = np.array([LETTERS.index(p) if p else -1 for p in preds])
        em = (pi == gold).astype(float)
        res[s] = {"n_failed_or_abstain": int(sum(p is None for p in preds)),
                  "em": {k: mean_ci(em, m) for k, m in masks.items()},
                  "f1": {k: macro_f1_ci(gold, pi, m) for k, m in masks.items()}}
    return res, masks


# ============================================================================ retrieval metrics
def hit_arrays(ranked, gold, k):
    any_k = np.array([float(bool(set(r[:k]) & set(g))) for r, g in zip(ranked, gold)])
    all_k = np.array([float(set(g) <= set(r[:k])) for r, g in zip(ranked, gold)])
    return any_k, all_k


def retrieval_block(ids, bench_items, masks, runs, cache, graph_runs, exp, incomplete=()):
    gold = [bench_items[i]["gold_span_ids"] for i in ids]
    multi = np.array([len(g) >= 2 for g in gold])
    rows = {}
    notes = []
    checks = {}

    def summarize(ranked, shown=None, unit="span", extra=None):
        h5, a5 = hit_arrays(ranked, gold, 5)
        h10, _ = hit_arrays(ranked, gold, 10)
        out = {"unit": unit, "r5": {k: mean_ci(h5, m) for k, m in masks.items()}, "r10": {k: mean_ci(h10, m) for k, m in masks.items()},
               "all5_multi": mean_ci(a5, multi), "n_multi": int(multi.sum())}
        if shown is not None:
            sh = np.array([float(bool(set(s) & set(g))) for s, g in zip(shown, gold)])
            out["shown"] = mean_ci(sh, masks["all"])
            out["shown_size"] = float(np.mean([len(set(s)) for s in shown]))
        if extra:
            out.update(extra)
        return out

    for s in sort_systems(runs):
        base = s.split("+")[0]
        if s == "direct":
            rows[s] = {"na": "n/a (no retrieval)"}
        elif base in RAG_MODES:
            ranked = [cache[i][base] for i in ids if i in cache]
            if len(ranked) != len(ids):
                rows[s] = {"na": "n/a (offline retrieval cache incomplete)"}
                continue
            shown = [(runs[s][i].get("tool_usage") or {}).get("retrieved_span_ids") or [] for i in ids]
            r = summarize(ranked, shown)
            r["same_as"] = base if s != base else None
            rows[s] = r
            stored = [stored_ranking(s, runs[s].get(i)) for i in ids]
            eq10 = float(np.mean([a == (b[:len(a)] if b else None) for a, b in zip(ranked, stored)]))
            eq5 = float(np.mean([a[:5] == (b[:5] if b else None) for a, b in zip(ranked, stored)]))
            checks[s] = f"offline top-10 identical to the stored run list on {eq10:.3f} of items (top-5 identical on {eq5:.3f})"
        elif s == "otter_v2":
            ranked = [cache[i]["otter_v2"] for i in ids if i in cache]
            if len(ranked) != len(ids):
                rows[s] = {"na": "n/a (offline retrieval cache incomplete)"}
                continue
            shown = [(runs[s][i].get("tool_usage") or {}).get("retrieved_span_ids") or [] for i in ids]
            rows[s] = summarize(ranked, shown, extra={"mean_list_len": float(np.mean([len(r) for r in ranked]))})
            stored = [stored_ranking(s, runs[s].get(i)) for i in ids]
            have = [k for k, b in enumerate(stored) if b is not None]
            eq = float(np.mean([ranked[k] == stored[k] for k in have]))
            eq5 = float(np.mean([ranked[k][:5] == stored[k][:5] for k in have]))
            g_have = [gold[k] for k in have]
            r5_off = float(np.mean([bool(set(ranked[k][:5]) & set(gold[k])) for k in have]))
            r5_sto = float(np.mean([bool(set(stored[k][:5]) & set(gold[k])) for k in have]))
            checks[s] = (f"{len(ids) - len(have)} items have no stored first-round list (calculation items delegated to otter_full, route `delegate_calc`; "
                         f"their offline v2 ranking is computed but the run did not use it). On the {len(have)} items with a stored list, the offline first-round list is identical to "
                         f"`retrieved_by_round[0]` on {eq:.3f} (top-5 identical on {eq5:.3f}); R@5 on those items: offline {r5_off:.3f} vs stored {r5_sto:.3f}")
        elif s == "ircot":
            allc = [(runs[s][i].get("tool_usage") or {}).get("retrieved_all") or [] for i in ids]
            shown = [(runs[s][i].get("tool_usage") or {}).get("retrieved_span_ids") or [] for i in ids]
            if not any(allc):
                rows[s] = {"na": "n/a (run does not store the retrieved set)"}
                continue
            r = summarize(allc, shown, unit="first-N of collected (not a ranking)")
            hall = np.array([float(bool(set(a) & set(g))) for a, g in zip(allc, gold)])
            r["collected_all"] = mean_ci(hall, masks["all"])
            r["collected_size"] = float(np.mean([len(set(a)) for a in allc]))
            rows[s] = r
        elif s in GRAPH:
            gp = graph_runs.get(s)
            if not gp or not all(i in gp for i in ids):
                rows[s] = {"na": NA_GRAPH if not gp else "n/a (retrieved contexts missing for some items)"}
                continue
            if not any(gp[i][0] for i in ids):
                rows[s] = {"na": NA_GRAPH}
                continue
            # passage-level: rank cut on the ordered passages; the span set is the union of the passage span ids
            def cut(i, k):
                return [x for p in gp[i][0][:k] for x in p]
            h5 = np.array([float(bool(set(cut(i, 5)) & set(g))) for i, g in zip(ids, gold)])
            h10 = np.array([float(bool(set(cut(i, 10)) & set(g))) for i, g in zip(ids, gold)])
            a5 = np.array([float(set(g) <= set(cut(i, 5))) for i, g in zip(ids, gold)])
            shown = [[x for p in gp[i][1] for x in p] for i in ids]
            sh = np.array([float(bool(set(sv) & set(g))) for sv, g in zip(shown, gold)])
            rows[s] = {"unit": "PASSAGE (top-N passages; each passage = several spans)", "dagger": True,
                       "r5": {k: mean_ci(h5, m) for k, m in masks.items()}, "r10": {k: mean_ci(h10, m) for k, m in masks.items()},
                       "all5_multi": mean_ci(a5, multi), "n_multi": int(multi.sum()), "shown": mean_ci(sh, masks["all"]),
                       "shown_size": float(np.mean([len(set(x)) for x in shown])),
                       "spans_in_top5": float(np.mean([len(set(cut(i, 5))) for i in ids])),
                       "spans_in_top10": float(np.mean([len(set(cut(i, 10))) for i in ids]))}
        else:
            rows[s] = {"na": NA_AGENT}
    for s in incomplete:
        rows[s] = {"na": "n/a (run incomplete; " + (NA_GRAPH if s in GRAPH else NA_AGENT).replace("n/a (", "", 1)}
    for g in GRAPH:
        if g not in runs and g not in incomplete:
            rows.setdefault(g, {"na": "not yet run"})
    return rows, checks


# ============================================================================ Abstain block
def abstain_block(exp):
    out = {}
    for p in sorted((ROOT / "runs" / exp).glob("*/main/abstain_scores.json")):
        s = p.parent.parent.name
        o = json.loads(p.read_text())["overall"]
        recs = jl(p.parent / "abstain_records.jsonl")
        U = [r for r in recs if not r["answerable"] and r["kind"] != "failed"]
        A = [r for r in recs if r["answerable"] and r["kind"] != "failed"]
        items = U + A
        ab = np.array([r["kind"] == "abstain" for r in items], float)
        un = np.array([not r["answerable"] for r in items], float)

        def prf(Wm):
            tp = Wm @ (ab * un)
            fp = Wm @ (ab * (1 - un))
            fn = Wm @ ((1 - ab) * un)
            P = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0.0)
            R = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-9), 0.0)
            F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-9), 0.0)
            return P, R, F
        W = Boot.weights(len(items))
        bsP, bsR, bsF = prf(W)
        P, R, F = (float(x[0]) for x in prf(np.ones((1, len(items)))))
        assert abs(P - o["abstain_precision"]) < 1e-9 and abs(R - o["abstain_recall"]) < 1e-9 and abs(F - o["abstain_f1"]) < 1e-9, (s, P, R, F, o)
        out[s] = {"n_u": o["n_unanswerable"], "n_a": o["n_answerable"], "failed": o["n_failed_unanswerable"] + o["n_failed_answerable"],
                  "P": (o["abstain_precision"],) + ci(bsP), "R": (o["abstain_recall"],) + ci(bsR), "F1": (o["abstain_f1"],) + ci(bsF)}
    return out


# ============================================================================ Typed block
def typed_block():
    import typed_common as T
    items = jl(ROOT / "benchmarks/typed/typed600.jsonl")
    by_id = {r["id"]: r for r in items}
    groups_all = [r["group_id"] for r in items]
    pos = {r["id"]: k for k, r in enumerate(items)}
    base = ROOT / "runs/vieduhop_typed"
    n_fill = sum(r["format"] == "fill_blank" for r in items)
    n_choice = sum(r["format"] != "fill_blank" for r in items)

    def strict(s):
        return " ".join(re.findall(r"[^\W_]+", T.nfc(s or "").lower()))

    def f1_tok(p, g):
        p, g = p.split(), g.split()
        if not p or not g:
            return 0.0
        c = sum((Counter(p) & Counter(g)).values())
        if c == 0:
            return 0.0
        pr, rc = c / len(p), c / len(g)
        return 2 * pr * rc / (pr + rc)

    res, partial = {}, []
    for d in sorted(base.glob("*/*")):
        if not d.is_dir():
            continue
        sysname = f"{d.parent.name} ({d.name})"
        entry = {}
        # --- true_false / multiple_choice
        af = d / "answers.jsonl"
        ans = {r["question_id"]: r for r in jl(af)} if af.exists() else {}
        ids_c = [r["id"] for r in items if r["format"] != "fill_blank"]
        if af.exists() and all(i in ans for i in ids_c):
            for fmt, K, key in (("true_false", 2, "tf"), ("multiple_choice", 4, "mc")):
                ids_f = [i for i in ids_c if by_id[i]["format"] == fmt]
                gold = np.array([ord(by_id[i]["answer"]) - 65 for i in ids_f])
                pr = []
                for i in ids_f:
                    c = pred_class(ans[i])
                    pr.append(ord(c) - 65 if c and ord(c) - 65 < K else -1)
                pr = np.array(pr)
                grp = [by_id[i]["group_id"] for i in ids_f]
                em = (pr == gold).astype(float)
                entry[key] = {"em": mean_ci(em, groups=grp), "f1": macro_f1_ci(gold, pr, groups=grp, K=K), "n": len(ids_f)}
        elif af.exists():
            partial.append(f"{sysname}: true_false+multiple_choice answers {sum(i in ans for i in ids_c)}/{n_choice}")
        # --- fill_blank
        ff = d / "fill_blank" / "answers.jsonl"
        fans = {r["question_id"]: r for r in jl(ff)} if ff.exists() else {}
        ids_f = [r["id"] for r in items if r["format"] == "fill_blank"]
        if ff.exists() and all(i in fans for i in ids_f):
            em, ems, f1 = [], [], []
            for i in ids_f:
                r = by_id[i]
                a = fans[i]
                failed = a.get("status") == "failed" or a.get("error") is not None
                pred = "" if failed else T.clean_prediction(a.get("answer") or "")
                refs = [r["answer"]] + list(r.get("accepted_variants") or [])
                em.append(float(T.string_match(pred, r["answer"], r.get("accepted_variants")) and not failed))
                ems.append(float(bool(strict(pred)) and strict(pred) in {strict(x) for x in refs} and not failed))
                f1.append(max(f1_tok(T.norm_answer(pred), T.norm_answer(x)) for x in refs) if not failed else 0.0)
            grp = [by_id[i]["group_id"] for i in ids_f]
            entry["fill"] = {"em": mean_ci(em, groups=grp), "em_strict": mean_ci(ems, groups=grp), "f1": mean_ci(f1, groups=grp), "n": len(ids_f)}
        elif ff.exists():
            partial.append(f"{sysname}: fill_blank answers {sum(i in fans for i in ids_f)}/{n_fill} (withheld)")
        if entry:
            res[sysname] = entry
    return res, partial, {"n_items": len(items), "n_groups": len(set(groups_all)), "n_fill": n_fill, "n_choice": n_choice}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="vieduhop_unrev")
    ap.add_argument("--typed", action="store_true", default=True)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-md", default=str(ROOT / "runs/vieduhop_unrev/metrics_r5_em_f1.md"))
    ap.add_argument("--out-tex", default=str(ROOT / "runs/tables/metrics_r5_em_f1.tex"))
    a = ap.parse_args()

    bench_all = jl(ROOT / "benchmarks/otter_hard/test.jsonl")
    bench = {r["id"]: r for r in bench_all}
    hard_ids = set(json.load(open(ROOT / "benchmarks/otter_hard/hard_tier_ids.json")))
    runs = load_runs(a.exp)
    all_ids = set(bench)
    complete = [s for s in runs if all_ids <= set(runs[s])]
    incomplete = {s: len(set(runs[s]) & all_ids) for s in runs if s not in complete}
    complete = sort_systems(complete)
    ids = sorted(i for i in all_ids if not bench[i].get("seen_in_development"))
    subsets = build_subsets(bench, ids, hard_ids)

    cache_path = ROOT / "runs/vieduhop_unrev/metrics_r5_em_f1_retrieval.json"
    twist_path = ROOT / "benchmarks/otter_hard/twist.jsonl"
    twist_items = jl(twist_path) if twist_path.exists() else []
    cache = offline_retrieval(bench_all + twist_items, cache_path, workers=a.workers, enabled=not a.no_retrieval)
    graph_runs = {g: graph_passages(a.exp, g) for g in GRAPH if g in runs}

    L = [f"# Additional metrics R@5, EM and F1 for ViEduHop (`{a.exp}`)", "", f"**{LABEL}**", ""]
    L += ["Regenerate: `.venv/bin/python scripts/eval/metrics_r5_em_f1.py` (offline retrieval needs only bge-m3 on :8002; results cached in "
          "`runs/vieduhop_unrev/metrics_r5_em_f1_retrieval.json`). Definitions: `docs/METRICS_R5_EM_F1.md`. 95% bootstrap CIs over items "
          f"({B} resamples, seed {SEED}); Typed: over groups.", ""]
    L += [f"MAIN set: `benchmarks/otter_hard/test.jsonl`, `seen_in_development` excluded, common set of the COMPLETE systems: n = {len(ids)} "
          f"(subsets: " + ", ".join(f"{k} n={sum(bool(f(bench[i])) for i in ids)}" for k, f in subsets.items()) + ").",
          f"Complete systems ({len(complete)}): {', '.join(complete)}.",
          f"Incomplete (answered/{len(all_ids)}; provisional block only): " + (", ".join(f"{s} {n}" for s, n in sorted(incomplete.items())) or "none") + ".",
          "Not yet run: " + (", ".join(g for g in ["hipporag", "minirag", "graphrag", "lightrag", "multi_agent_no_workspace", "otter_full"] if g not in runs) or "none") + ".", ""]

    tex = ["% Additional metrics R@5/EM/F1 for ViEduHop. " + LABEL, "% Generated by scripts/eval/metrics_r5_em_f1.py; do not edit by hand.", ""]

    # ---------------------------------------------------------------- 1. retrieval
    ret_rows, checks = retrieval_block(ids, bench, {k: np.array([bool(f(bench[i])) for i in ids]) for k, f in subsets.items()}, {s: runs[s] for s in complete}, cache, graph_runs, a.exp, incomplete=list(incomplete))
    L += ["## 1. Retrieval: R@5 / R@10 (offline, no LLM)", "",
          "R@5 (any) = at least one gold span in the first 5 retrieved spans; all-gold@5 = every gold span in the first 5, over multi-span items only. "
          "Shown-set recall = any gold span in the spans actually put in the generator prompt (not a rank cut; mean set size in parentheses). "
          "Rows marked † are PASSAGE-level (graph-RAG chunks cover several spans) and are not comparable to span-level rows. "
          "ircot rows use the first N spans of the collected set (insertion order, NOT a relevance ranking).", ""]
    hdr = ["system", "unit", "R@5 all", "R@5 book_specific", "R@5 cross_subject", "R@5 Hard tier", "R@10 all", f"all-gold@5 (multi-span, n={next((r['n_multi'] for r in ret_rows.values() if 'n_multi' in r), 0)})", "shown-set recall (mean size)"]
    body = []
    for s in sort_systems(ret_rows):
        r = ret_rows[s]
        if "na" in r:
            body.append([s, r["na"]] + ["-"] * 7)
            continue
        name = s + (" †" if r.get("dagger") else "")
        unit = r["unit"] + (f"; = {r['same_as']} (same retrieval)" if r.get("same_as") else "")
        if r.get("dagger"):
            unit += f"; mean spans in top-5 passages {r['spans_in_top5']:.1f}, top-10 {r['spans_in_top10']:.1f}"
        body.append([name, unit] + [cell(r["r5"][k]) for k in subsets] + [cell(r["r10"]["all"]), cell(r["all5_multi"]),
                                                                          f"{cell(r['shown'])} ({r['shown_size']:.1f})" if r.get("shown") else "-"])
    L += md_table(hdr, body) + [""]
    for s, c in checks.items():
        L.append(f"* Cross-check {s}: {c}.")
    ir = ret_rows.get("ircot", {})
    if "collected_all" in ir:
        L.append(f"* ircot: recall of the FULL collected set (any gold span in `retrieved_all`, mean size {ir['collected_size']:.1f}): {cell(ir['collected_all'])}.")
    if "otter_v2" in ret_rows and "mean_list_len" in ret_rows["otter_v2"]:
        L.append(f"* otter_v2: mean length of the first-round ranked list {ret_rows['otter_v2']['mean_list_len']:.2f} (max 10); the gap round is not part of this ranking.")
    L += ["* `rag_*+top_k=10` use the retrieval of their base system; only the shown-set recall differs (10 spans shown).",
          "* Typed R@5 is n/a (Typed gold spans were produced by the hybrid retriever family: circular).", ""]
    tx_rows = []
    for s in sort_systems(ret_rows):
        r = ret_rows[s]
        if "na" in r:
            tx_rows.append([tex_esc(s), "\\multicolumn{6}{l}{\\textit{" + tex_esc(r["na"]) + "}}"])
            continue
        tx_rows.append([tex_esc(s + (" †" if r.get("dagger") else ""))] + [pt(r["r5"][k]) for k in subsets] + [pt(r["r10"]["all"]), pt(r["all5_multi"])])
    tex += ["% ---- retrieval"] + tex_table(
        "Retrieval recall on ViEduHop (additional metrics, requested after the pre-registered headline; offline, no LLM). R@$k$: a gold span among the first $k$ retrieved spans; "
        "all-gold@5: every gold span in the top 5, multi-span items only. $^\\dagger$ passage-level, not comparable.",
        "tab:metrics_retrieval", ["System", "R@5 all", "R@5 book-sp.", "R@5 cross-subj.", "R@5 Hard", "R@10 all", "all-gold@5"], tx_rows, "lcccccc",
        note="ircot: first five collected spans = the initial hybrid top-5, so it equals rag\\_hybrid by construction; otter\\_v2: first-round ranking. Additional metrics requested after the pre-registered headline (docs/HEADLINE\\_METRICS.md); definitions fixed before computing. 95\\% CIs are in runs/vieduhop\\_unrev/metrics\\_r5\\_em\\_f1.md.") + [""]

    # ---------------------------------------------------------------- 2. MCQ EM / macro-F1
    mc, masks = mcq_block(bench, {s: runs[s] for s in complete}, ids, subsets, complete)
    L += ["## 2. Answer: EM (= accuracy) and macro-F1 over the options A-D (multiple choice, MAIN set)", "",
          "EM = chosen option equals the key (identical to the headline accuracy; abstention / failure counts as wrong). macro-F1 (A-D) = unweighted mean of the four per-option F1 "
          "of the chosen label (no valid choice = no prediction). Both are computed over ALL items of the (sub)set. Backbone Qwen3-32B-FP8.", ""]
    hdr = ["system", "EM(=acc) all", "EM book_specific", "EM cross_subject", "EM Hard tier", "macro-F1(A-D) all", "macro-F1 book_specific", "macro-F1 cross_subject", "macro-F1 Hard tier", "no valid choice"]
    body = [[s] + [cell(mc[s]["em"][k]) for k in subsets] + [cell(mc[s]["f1"][k]) for k in subsets] + [mc[s]["n_failed_or_abstain"]] for s in complete]
    L += md_table(hdr, body) + [""]
    # consistency with headline_analysis.py's accuracy
    ok_all = []
    for s in complete:
        acc = np.mean([float(runs[s][i].get("answer_choice") == bench[i]["answer"] and runs[s][i].get("status") != "abstain") for i in ids])
        ok_all.append(abs(acc - mc[s]["em"]["all"][0]) < 1e-12)
    L += [f"* EM equals the headline-analysis accuracy formula (`answer_choice == key and status != abstain`) for {sum(ok_all)}/{len(ok_all)} systems.", ""]
    tx_rows = [[tex_esc(s)] + [pt(mc[s]["em"][k]) for k in subsets] + [pt(mc[s]["f1"][k]) for k in subsets] for s in complete]
    tex += ["% ---- EM and macro-F1"] + tex_table(
        "Answer metrics on ViEduHop (additional metrics, requested after the pre-registered headline). EM equals accuracy for multiple choice; macro-F1 is over the option classes A--D. "
        "Common set of complete systems, $n=%d$." % len(ids),
        "tab:metrics_em_f1", ["System"] + ["EM " + k.replace("_", "-") for k in subsets] + ["F1 " + k.replace("_", "-") for k in subsets], tx_rows, "lcccccccc",
        note="Additional metrics requested after the pre-registered headline (docs/HEADLINE\\_METRICS.md); definitions fixed before computing. CIs in the markdown report.") + [""]

    # ---------------------------------------------------------------- 2b. provisional block
    if incomplete:
        L += ["### 2b. PROVISIONAL (system not finished): EM and macro-F1 on the items it has answered", "",
              "Computed on the items answered so far (intersection with the non-seen set); not comparable with section 2. The last two columns give otter_v2 and ircot on the same items.", ""]
        body = []
        for s in sort_systems(incomplete):
            idp = [i for i in ids if i in runs[s]]
            if not idp:
                continue
            sub_all = {"all": lambda b: True}
            m1, _ = mcq_block(bench, runs, idp, sub_all, [s])
            ref = {}
            for rs in ("otter_v2", "ircot"):
                if rs in complete:
                    ref[rs], _ = mcq_block(bench, runs, idp, sub_all, [rs])
            body.append([s, len(idp), cell(m1[s]["em"]["all"]), cell(m1[s]["f1"]["all"]),
                         cell(ref["otter_v2"]["otter_v2"]["em"]["all"]) if "otter_v2" in ref else "-",
                         cell(ref["ircot"]["ircot"]["em"]["all"]) if "ircot" in ref else "-"])
        L += md_table(["system", "n answered", "EM", "macro-F1 (A-D)", "otter_v2 EM same items", "ircot EM same items"], body) + [""]

    # ---------------------------------------------------------------- 3. Abstain F1
    ab = abstain_block("vieduhop_abstain_unrev")
    L += ["## 3. Abstain-task F1 (OTTER-Abstain test, 131 unanswerable + 131 answerable controls; NOT the MCQ macro-F1)", "",
          "Positive class = should abstain; P/R/F1 of unanswerable detection from `abstain_scores.json` (unchanged abstain code), intervals by bootstrap over the 262 items from `abstain_records.jsonl`. "
          "Systems with an abstain run only; the abstain option E is offered in these runs.", ""]
    body = [[s, f"{v['n_u']}/{v['n_a']}", cell(v["P"]), cell(v["R"]), cell(v["F1"]), v["failed"]] for s, v in sort_systems_ab(ab)]
    L += md_table(["system", "n (U/A)", "abstain precision", "abstain recall", "Abstain-F1", "failed"], body) + [""]
    tx_rows = [[tex_esc(s), pt(v["P"]), pt(v["R"]), pt(v["F1"])] for s, v in sort_systems_ab(ab)]
    tex += ["% ---- abstain"] + tex_table(
        "Unanswerable-detection P/R/F1 on OTTER-Abstain test (additional metrics, requested after the pre-registered headline); abstain option E offered.",
        "tab:metrics_abstain_f1", ["System", "P", "R", "Abstain-F1"], tx_rows, "lccc") + [""]

    # ---------------------------------------------------------------- 4. Typed
    ty, partial, tinfo = typed_block()
    L += ["## 4. Typed (benchmarks/typed/typed600.jsonl; CIs over groups)", "",
          f"{tinfo['n_items']} items, {tinfo['n_groups']} groups; true_false and multiple_choice answers: EM = accuracy and macro-F1 over the option classes (T/F: 2 classes; MC: 4 classes); "
          "fill_blank: EM = unchanged `typed_common.string_match` (+ accepted variants), `EM strict` = NFC/lower-case/punctuation only (+ accepted variants), F1 = SQuAD token F1 on the normalised terms. "
          "Each F1 is labelled with its task. A system is shown on a format only when it answered every item of that format.", ""]
    body = []
    for s, e in ty.items():
        body.append([s, cell(e["tf"]["em"]) if "tf" in e else "-", cell(e["tf"]["f1"]) if "tf" in e else "-",
                     cell(e["mc"]["em"]) if "mc" in e else "-", cell(e["mc"]["f1"]) if "mc" in e else "-",
                     cell(e["fill"]["em"]) if "fill" in e else "-", cell(e["fill"]["em_strict"]) if "fill" in e else "-", cell(e["fill"]["f1"]) if "fill" in e else "-"])
    L += md_table(["system (backbone)", "T/F EM(=acc) n=375", "T/F macro-F1 (2 cls)", "MC EM(=acc) n=75", "MC macro-F1 (A-D)", "Fill EM n=150", "Fill EM strict", "Fill F1 (token)"], body) + [""]
    if partial:
        L += ["Partial / withheld: " + "; ".join(partial) + ".", ""]
    tx_rows = []
    for s, e in ty.items():
        tx_rows.append([tex_esc(s), pt(e["tf"]["em"]) if "tf" in e else "--", pt(e["tf"]["f1"]) if "tf" in e else "--", pt(e["mc"]["em"]) if "mc" in e else "--",
                        pt(e["mc"]["f1"]) if "mc" in e else "--", pt(e["fill"]["em"]) if "fill" in e else "--", pt(e["fill"]["f1"]) if "fill" in e else "--"])
    tex += ["% ---- typed"] + tex_table(
        "ViEduHop-Typed (additional metrics, requested after the pre-registered headline). T/F and MC: EM = accuracy, F1 = macro-F1 over option classes; fill-in-the-blank: normalised EM and token F1. "
        "Systems with incomplete fill-in runs are withheld (--).",
        "tab:metrics_typed", ["System", "T/F EM", "T/F F1", "MC EM", "MC F1", "Fill EM", "Fill F1"], tx_rows, "lcccccc") + [""]

    # ---------------------------------------------------------------- 5. Twist
    L += twist_section(a, tex, bench, hard_ids, cache)

    L += ["## Notes and caveats", "",
          "* Everything here is secondary and was requested after the main results; the pre-registered headline (accuracy, located, lesson_located, verified_correct, cost) is unchanged.",
          "* The Hard tier is itself a post hoc subset (benchmarks/otter_hard/hard_tier.README.md).",
          "* R@5 is a retrieval-stage property of the span ranking; the tool-calling agents (single_agent_tools, multi_agent_no_workspace, otter_full) and `direct` have no span ranking and are n/a; "
          "their evidence quality is measured by the headline `located` / `verified_correct`.",
          "* otter_v2's R@5 uses its first-round ranking (up to 10 spans, 2 of which can be ontology-hinted candidates appended after the text hits), before any gap round.",
          "* LightRAG (and other graph baselines when they expose it) are passage-level; do not compare their R@5 directly with span-level rows.", ""]
    Path(a.out_md).write_text("\n".join(L), encoding="utf-8")
    Path(a.out_tex).write_text("\n".join(tex), encoding="utf-8")
    print("\n".join(L))


def sort_systems_ab(ab):
    keys = sorted(ab, key=lambda s: (ORDER.index(s) if s in ORDER else (100 if "+" not in s else 50), s))
    return [(k, ab[k]) for k in keys]


def twist_section(a, tex, bench, hard_ids, cache):
    twist_path = ROOT / "benchmarks/otter_hard/twist.jsonl"
    exp = ROOT / "runs/vieduhop_twist"
    planned = ["direct", "rag_hybrid", "ircot", "otter_v2"]
    L = ["## 5. Twist (benchmarks/otter_hard/twist.jsonl)", ""]
    items = jl(twist_path) if twist_path.exists() else []
    tb = {r["id"]: r for r in items}
    ids = sorted(tb)
    runs = load_runs("vieduhop_twist") if exp.exists() else {}
    comp = [s for s in sort_systems(runs) if set(ids) <= set(runs[s])]
    rows_tex = []
    if not comp:
        L += [f"`runs/vieduhop_twist` {'exists but has no complete system' if exp.exists() else 'does not exist yet'}; the table below is ready and will be filled by re-running this script "
              f"(planned systems: {', '.join(planned)}; n = {len(ids)} items; EM, macro-F1 (A-D), R@5 for rag_*/otter_v2/ircot).", ""]
        L += md_table(["system", "EM(=acc)", "macro-F1 (A-D)", "R@5", "R@10"], [[s, "pending", "pending", "pending" if s != "direct" else "n/a", "pending" if s != "direct" else "n/a"] for s in planned]) + [""]
        rows_tex = [[tex_esc(s), "--", "--", "--" if s != "direct" else "n/a", "--" if s != "direct" else "n/a"] for s in planned]
    else:
        sub = {"all": lambda b: True}
        mc, masks = mcq_block(tb, runs, ids, sub, comp)
        ret, _ = retrieval_block(ids, tb, {"all": np.ones(len(ids), bool)}, {s: runs[s] for s in comp}, cache, {}, "vieduhop_twist")
        body = []
        for s in comp:
            r = ret.get(s, {"na": "n/a"})
            r5 = pt(r["r5"]["all"]) if "r5" in r else "n/a"
            r10 = pt(r["r10"]["all"]) if "r10" in r else "n/a"
            body.append([s, cell(mc[s]["em"]["all"]), cell(mc[s]["f1"]["all"]), cell(r["r5"]["all"]) if "r5" in r else "n/a", cell(r["r10"]["all"]) if "r10" in r else "n/a"])
            rows_tex.append([tex_esc(s), pt(mc[s]["em"]["all"]), pt(mc[s]["f1"]["all"]), r5, r10])
        L += [f"Complete systems on n = {len(ids)} twist items: {', '.join(comp)}.", ""] + md_table(["system", "EM(=acc)", "macro-F1 (A-D)", "R@5", "R@10"], body) + [""]
        missing = [s for s in planned if s not in comp]
        if missing:
            L += ["Not yet complete: " + ", ".join(missing) + ".", ""]
    tex += ["% ---- twist"] + tex_table(
        "ViEduHop-Twist (additional metrics, requested after the pre-registered headline).", "tab:metrics_twist",
        ["System", "EM (=acc)", "macro-F1 (A--D)", "R@5", "R@10"], rows_tex, "lcccc") + [""]
    xexp = ROOT / "runs/vieduhop_twist_abstain"
    if xexp.exists():
        ab = abstain_block("vieduhop_twist_abstain")
        L += ["### Twist-abstain F1 (unanswerable detection)", ""] + md_table(
            ["system", "n (U/A)", "abstain precision", "abstain recall", "Abstain-F1"],
            [[s, f"{v['n_u']}/{v['n_a']}", cell(v["P"]), cell(v["R"]), cell(v["F1"])] for s, v in sort_systems_ab(ab)]) + [""]
    return L


if __name__ == "__main__":
    main()
