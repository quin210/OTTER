"""Aggregate simulation results -> runs/knowledge_maintenance/table_{detection,recall_by_class,alignment,cost,qa,exposure}.{csv,tex}
and table_paper.tex (the single compact table proposed for the paper)."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KM = ROOT / "runs/knowledge_maintenance"


def ms(xs, nd=3, pct=False):
    xs = [x for x in xs if x == x and x is not None]
    if not xs:
        return "-"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.{nd}f}")
    return f(st.mean(xs)) + (f" $\\pm$ {f(st.pstdev(xs))}" if len(xs) > 1 else "")


def write(name, header, rows):
    with open(KM / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[str(c).replace("$\\pm$", "+-") for c in r] for r in rows])
    with open(KM / f"{name}.tex", "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\hline\n" + " & ".join(h.replace("_", "\\_").replace("%", "\\%") for h in header) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(str(x).replace("_", "\\_").replace("%", "\\%") for x in r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def main():
    seeds = sorted(int(p.stem.split("seed")[1]) for p in (KM / "reports").glob("detection_seed*.json"))
    det = [json.loads((KM / "reports" / f"detection_seed{s}.json").read_text()) for s in seeds]
    cost = json.loads((KM / "reports" / "cost.json").read_text())
    # 1. invalidation / detection
    rows = []
    for m, label in (("no_maintenance", "no maintenance"), ("id_exists_check", "span-id existence check"),
                     ("hash_diff_exclude_touched", "hash diff, drop every touched item"), ("ours_direct", "revalidate (ours)")):
        g = lambda k: [d["invalidation"][m][k] for d in det]
        rows.append([label, ms(g("precision"), pct=True), ms(g("recall"), pct=True), ms(g("false_exclusion_rate"), pct=True),
                     ms([d["invalidation"][m]["excluded"] for d in det], nd=0)])
    write("table_invalidation", ["method", "precision %", "recall %", "false-excl. %", "items excluded"], rows)
    dt = [d["detection_touched"] for d in det]
    detect_row = [ms([x["precision"] for x in dt], pct=True), ms([x["recall"] for x in dt], pct=True)]
    # 2. recall by class
    rc = []
    for key, lab in (("orphaned", "all cited spans deleted"), ("stale_number", "edited number in claim"),
                     ("stale_year", "edited year in claim"), ("stale_entity", "edited entity in claim")):
        n = sum(d["recall_by_class"][key]["n"] for d in det)
        e = sum(d["recall_by_class"][key]["excluded"] for d in det)
        rc.append([lab, n, e, f"{100 * e / n:.1f}"])
    write("table_recall_by_class", ["ground-truth class (3 seeds pooled)", "items", "excluded", "recall %"], rc)
    # 3. alignment (ids hidden)
    al = []
    pool = {"unchanged": {}, "edited": {}, "deleted": {}}
    for d in det:
        for k in pool:
            for kk, v in d["alignment"][k].items():
                pool[k][kk] = pool[k].get(kk, 0) + v
    for k, lab in (("unchanged", "unchanged spans"), ("edited", "edited spans"), ("deleted", "deleted spans")):
        tot = sum(pool[k].values())
        al.append([lab, tot] + [f"{kk}: {v} ({100 * v / tot:.1f}%)" for kk, v in sorted(pool[k].items(), key=lambda x: -x[1])])
    with open(KM / "table_alignment.csv", "w", newline="") as f:
        csv.writer(f).writerows([["class", "n", "outcomes..."]] + al)
    # 4. cost
    ext_calls, ver_calls = cost["extraction_calls_full"], cost["verify_calls_full"]
    cr = []
    for s, c in cost["seeds"].items():
        cr.append([s, f"{c['items_rechecked']}/{c['items_total']}", c["ours_llm_calls"], c["ours_tokens"],
                   f"{c['batches_full_revalidation']} (~{c['est_tokens_full_revalidation']:,} tok)",
                   f"{ext_calls}+{ver_calls} ({cost['extraction_tokens_full'] + cost['verify_tokens_full']:,} tok)",
                   f"{c['chunks_touched']}/{cost['n_chunks']}"])
    write("table_cost", ["seed", "items re-checked", "LLM calls (ours)", "tokens (ours)", "calls: full re-verification (est.)",
                         "calls: full rebuild (measured build)", "extraction chunks touched"], cr)
    # 5. QA
    qa = json.loads((KM / "qa" / "qa_scores.json").read_text())
    qr = []
    for sysn, res in qa["dev"].items():
        for g, lab in (("gold_deleted", "gold deleted"), ("edit_relevant", "gold edited (answer-relevant)"), ("control", "control (untouched)")):
            if g in res:
                v = res[g]
                qr.append([sysn, lab, v["n"], f"{100 * v['old_key']:.0f}", f"{100 * v['stale_citation']:.0f}",
                           f"{100 * v['not_supported_status']:.0f}" if "rag_hybrid" not in sysn else "n/a"])
    write("table_qa_dev", ["system", "item group", "n", "old key %", "cites changed span %", "status not 'supported' %"], qr)
    qt = []
    for sysn, res in qa["test_proxy"].items():
        for g, lab in (("gold_deleted", "gold deleted"), ("edit_relevant", "gold edited (answer-relevant)")):
            if g in res:
                v = res[g]
                qt.append([sysn, lab, v["n"], f"{100 * v['old_key']:.0f}", f"{100 * v['stale_citation']:.0f}"])
    write("table_qa_test_proxy", ["system (existing answers, test split)", "item group", "n", "old key %", "cites changed span %"], qt)
    # 6. exposure
    ex = json.loads((KM / "qa" / "exposure.json").read_text())
    er = []
    for k, v in ex.items():
        split, name = k.split("|")
        a = v["all"]
        er.append([split, name, a["q"], a["q_with_bad_hint"], f"{100 * a['q_with_bad_hint'] / a['q']:.1f}", a["bad_hints"], a["hints"]])
    write("table_exposure", ["split", "knowledge base", "questions", "q. with stale hint", "% q.", "stale hints", "hints shown"], er)
    # paper table
    s0 = cost["seeds"]["0"]
    avg = lambda f: st.mean(f(c) for c in cost["seeds"].values())
    ours = det and {m: [d["invalidation"][m] for d in det] for m in ("ours_direct", "hash_diff_exclude_touched", "id_exists_check")}
    m_ = lambda m, k: st.mean(x[k] for x in ours[m])
    with open(KM / "table_paper.tex", "w") as f:
        f.write("% generated by scripts/maintenance/make_tables.py; simulation only (5% spans deleted + 5% edited, 3 seeds)\n")
        f.write("\\begin{tabular}{lrrr}\n\\hline\nmethod & invalidation P / R (\\%) & wrongly dropped (\\%) & LLM calls\\\\\n\\hline\n")
        f.write(f"no maintenance & -- / 0.0 & 0.0 & 0\\\\\n")
        f.write(f"span-id check & {100 * m_('id_exists_check', 'precision'):.1f} / {100 * m_('id_exists_check', 'recall'):.1f} & {100 * m_('id_exists_check', 'false_exclusion_rate'):.1f} & 0\\\\\n")
        f.write(f"hash diff, drop touched & {100 * m_('hash_diff_exclude_touched', 'precision'):.1f} / {100 * m_('hash_diff_exclude_touched', 'recall'):.1f} & {100 * m_('hash_diff_exclude_touched', 'false_exclusion_rate'):.1f} & 0\\\\\n")
        f.write(f"revalidate (ours) & {100 * m_('ours_direct', 'precision'):.1f} / {100 * m_('ours_direct', 'recall'):.1f} & {100 * m_('ours_direct', 'false_exclusion_rate'):.1f} & {avg(lambda c: c['ours_llm_calls']):.0f}\\\\\n")
        f.write(f"full re-verification & -- & -- & {s0['batches_full_revalidation']}\\\\\n")
        f.write(f"full rebuild (extract+verify) & -- & -- & {ext_calls + ver_calls}\\\\\n\\hline\n\\end{{tabular}}\n")
    print("ok", seeds)


if __name__ == "__main__":
    main()
