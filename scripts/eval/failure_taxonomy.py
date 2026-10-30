"""Failure taxonomy of every wrong / ungrounded answer -> <out>/failure_taxonomy.md + failure_taxonomy.csv (+ _counts.csv).
  .venv/bin/python scripts/eval/failure_taxonomy.py --exp core_dev --bench benchmarks/otter_hard/dev.jsonl --out runs/core_dev \
      [--external 'runs/otter_hard_v2/*/main'] [--key-problems ids.txt] [--by hop_count domain trick_type difficulty]
Categories (mutually exclusive, evaluated in this order for WRONG answers; descriptive analysis only, nothing here feeds back into a system):
  c  format / no answer   no parsable A-D choice (incl. 'không xác định được', abstain, failed)
  a  retrieval miss       NO gold span in the system's evidence set
  d  distractor attraction gold span in the evidence set, wrong choice, and a cited NON-gold span from a gold book (near lesson/page)
  b  reading error        gold span in the evidence set, wrong choice, no near-lesson distractor citation
  e  key/question problem item flagged by manual reading (--key-problems file: ids, one per line; listed separately, still counted in a-d)
For CORRECT answers:  grounded = a gold span is cited and every claim is judged supported (if judgments exist); otherwise
  f_nocite (no citation at all), f_nogold (cites only non-gold spans; may still be a valid alternative span), f_unsupported (a gold span is cited but
  >=1 claim is judged not supported by the 8B judge). 'unsupported inference' = an answer was given although no gold span is in the evidence set
  (reported as given / correct among them, the latter = lucky or parametric).
Evidence set: tool_usage.retrieved_top10 / retrieved_all / retrieved_span_ids (rag_*, ircot, otter_v2); for agents (no retrieval log) the union of
  workspace packet source_span_ids and cited spans. So 'a' means different things per family: a true retrieval miss for rag/v2, 'never used the gold span' for agents.
Selective accuracy at coverage: systems with v2_grounding (otter_v2) -> answer only when grounding >= tau."""
import argparse, csv, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.judge import load_judgments  # noqa: E402
from otter.evaluation.metrics import load_answers  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="core_dev"); ap.add_argument("--bench", default="benchmarks/otter_hard/dev.jsonl")
ap.add_argument("--out", default=None); ap.add_argument("--bb", default="main")
ap.add_argument("--external", nargs="*", default=[], help="glob(s) of run dirs holding answers.jsonl (e.g. runs/otter_hard_v2/*/main)")
ap.add_argument("--key-problems", default=None); ap.add_argument("--by", nargs="*", default=["hop_count", "domain", "difficulty", "trick_type"])
ap.add_argument("--systems", default=None, help="comma list (default: all dirs under exp)")
a = ap.parse_args()
bench = {}
for l in open(ROOT / a.bench):
    if l.strip():
        r = json.loads(l); bench[r["id"]] = r
corpus = SpanCorpus.load(book_filter=core_book_filter)
out = Path(a.out) if a.out else ROOT / "runs" / a.exp
keyp = {x.strip().split()[0] for x in open(a.key_problems) if x.strip() and not x.startswith("#")} if a.key_problems else set()
dirs = {d.parent.name: d for d in sorted((ROOT / "runs" / a.exp).glob(f"*/{a.bb}")) if (d / "answers.jsonl").exists()}
for g in a.external:
    for d in sorted(glob.glob(str(ROOT / g) if not g.startswith("/") else g)):
        d = Path(d)
        if (d / "answers.jsonl").exists():
            dirs[f"ext:{d.parent.name}"] = d
_ids = set()
for _d in dirs.values():
    try:
        _ids |= set(json.loads((_d / "run_meta.json").read_text()).get("item_ids") or [])
    except Exception:
        pass
if _ids:
    bench = {k: v for k, v in bench.items() if k in _ids}
if a.systems:
    dirs = {k: v for k, v in dirs.items() if k in a.systems.split(",")}


def evidence(fa):
    tu = fa.tool_usage or {}
    for k in ("retrieved_all", "retrieved_top10", "retrieved_span_ids"):
        if tu.get(k):
            ids = list(tu[k]) if k != "retrieved_top10" else list(tu[k])
            return set(ids) | {c.span_id for cl in fa.claims for c in cl.citations}, k
    ws = tu.get("workspace") or {}
    ids = {s for p in ws.get("packets", []) for s in p.get("source_span_ids", [])}
    ids |= {c.span_id for cl in fa.claims for c in cl.citations} | {c.span_id for c in (fa.where_to_read or [])}
    return ids, ("workspace" if ids else "none")


rows, counts = [], defaultdict(Counter)
sel = {}
for name, d in dirs.items():
    ans = load_answers(d / "answers.jsonl")
    judg = {}
    jf = d / "claim_judgments.small.jsonl"
    if jf.exists():
        judg = load_judgments(jf)
    for qid, it in bench.items():
        fa = ans.get(qid)
        if fa is None:
            continue
        gold = set(it.get("gold_span_ids") or [])
        ch = (fa.answer_choice or "").strip().upper()[:1]
        parsed = ch in "ABCD" and ch != "" and fa.status not in ("abstain", "failed")
        ev, src = evidence(fa)
        cited = [c.span_id for cl in fa.claims for c in cl.citations]
        gold_books = {corpus.get(s).book_id for s in gold if corpus.get(s)}
        near = [s for s in cited if s not in gold and corpus.get(s) and corpus.get(s).book_id in gold_books]
        gold_ev = bool(gold & ev)
        correct = parsed and ch == it["answer"]
        unsup = [j["verdict"] for (q, _), j in judg.items() if q == qid and j["verdict"] != "supported"]
        if correct:
            if not cited: cat = "f_nocite"
            elif not (gold & set(cited)): cat = "f_nogold"
            elif unsup: cat = "f_unsupported"
            else: cat = "grounded"
        else:
            cat = "c_format_noanswer" if not parsed else ("a_retrieval_miss" if not gold_ev else ("d_distractor" if near else "b_reading"))
        rows.append({"system": name, "id": qid, "correct": int(correct), "category": cat, "choice": ch or None, "gold": it["answer"], "gold_in_evidence": int(gold_ev),
                     "evidence_src": src, "answered": int(parsed), "cited_gold": int(bool(gold & set(cited))), "n_unsup_claims": len(unsup),
                     "key_problem_flag": int(qid in keyp), "grounding": (fa.tool_usage or {}).get("v2_grounding"),
                     **{f: it.get(f) for f in a.by}})
        counts[name][cat] += 1
        if parsed and not gold_ev and src != "none":
            counts[name]["unsupported_inference_given"] += 1
            counts[name]["unsupported_inference_correct"] += int(correct)
        counts[name]["n"] += 1; counts[name]["correct"] += int(correct); counts[name]["key_flag_wrong"] += int((not correct) and qid in keyp)
    g = [(r["grounding"], r["correct"], r["answered"]) for r in rows if r["system"] == name and r["grounding"] is not None]
    if g:
        sel[name] = g
cols = ["system", "id", "correct", "category", "choice", "gold", "gold_in_evidence", "evidence_src", "answered", "cited_gold", "n_unsup_claims", "key_problem_flag", "grounding"] + a.by
with open(out / "failure_taxonomy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
CATS = ["grounded", "f_nocite", "f_nogold", "f_unsupported", "a_retrieval_miss", "b_reading", "c_format_noanswer", "d_distractor"]
with open(out / "failure_taxonomy_counts.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["system", "n", "correct"] + CATS + ["unsupported_inference_given", "unsupported_inference_correct", "key_flag_wrong"])
    for s, c in counts.items():
        w.writerow([s, c["n"], c["correct"]] + [c[k] for k in CATS] + [c["unsupported_inference_given"], c["unsupported_inference_correct"], c["key_flag_wrong"]])
L = [f"# Failure taxonomy ({a.exp}, {a.bb}; bench {a.bench}; n items = {len(bench)})", "",
     "Generated by `scripts/eval/failure_taxonomy.py` (definitions in its docstring). Wrong answers: c format/no answer, a no gold span in the evidence set, "
     "d wrong choice + cites a non-gold span of a gold book (near lesson), b reading error (gold in evidence set, wrong choice, no distractor citation). "
     "Correct answers: grounded (gold span cited, all claims judged supported by the 8B judge) vs f_* ungrounded. Evidence set = retrieved list (rag/ircot/v2) or workspace packet sources + citations (agents; "
     "no retrieval log, so 'a' = gold never used). Judge for f_unsupported = Qwen3-8B (not independent for otter_v2).", "",
     "| system | n | correct | grounded | f_nocite | f_nogold | f_unsupp | wrong: a miss | b read | c format | d distr. | UI given | UI correct | key-flag wrong |", "|" + "---|" * 14]
for s, c in counts.items():
    L.append(f"| {s} | {c['n']} | {c['correct']} | " + " | ".join(str(c[k]) for k in CATS) + f" | {c['unsupported_inference_given']} | {c['unsupported_inference_correct']} | {c['key_flag_wrong']} |")
L += ["", "UI = unsupported inference: an answer was given although no gold span is in the system's evidence set (correct ones are lucky / parametric answers). "
      "`direct` has no evidence set, so every answer counts as UI by construction.", ""]
for fld in a.by:
    vals = sorted({str(r[fld]) for r in rows if r.get(fld) is not None})
    if len(vals) < 2 or vals == ["None"]:
        continue
    L += [f"## Wrong answers and ungrounded-correct answers by {fld}", "", "| system | " + " | ".join(f"{fld}={v} (n={len([1 for r in rows if r['system']==list(counts)[0] and str(r.get(fld))==v])})" for v in vals) + " |", "|" + "---|" * (len(vals) + 1)]
    for s in counts:
        cell = []
        for v in vals:
            rs = [r for r in rows if r["system"] == s and str(r.get(fld)) == v]
            wrong = sum(1 for r in rs if not r["correct"]); ung = sum(1 for r in rs if r["correct"] and r["category"] != "grounded")
            cell.append(f"{wrong} wrong / {ung} ungr.")
        L.append(f"| {s} | " + " | ".join(cell) + " |")
    L.append("")
if sel:
    L += ["## Selective accuracy at coverage (systems with a grounding score; answer only when grounding >= tau)", "", "| system | tau | coverage | selective acc | n answered |", "|---|---|---|---|---|"]
    for s, g in sel.items():
        for tau in (0, 1, 2, 3):
            kept = [(c, an) for gr, c, an in g if gr >= tau and an]
            L.append(f"| {s} | {tau} | {len(kept) / len(g):.3f} | {(np.mean([c for c, _ in kept]) if kept else float('nan')):.3f} | {len(kept)} |")
    L.append("")
# wrong-answer id list (for the manual key/question review and the teacher sheet)
wr = defaultdict(list)
for r in rows:
    if not r["correct"]:
        wr[r["id"]].append(f"{r['system']}:{r['choice']}({r['category'][0]})")
L += ["## Items answered wrongly by at least one system (candidates for key / question problems; gold key in brackets)", ""]
for qid, v in sorted(wr.items(), key=lambda kv: -len(kv[1])):
    L.append(f"- {qid} [{bench[qid]['answer']}] ({len(v)} systems): " + ", ".join(v))
if keyp:
    L += ["", "Manually flagged key/question problems: " + ", ".join(sorted(keyp))]
(out / "failure_taxonomy.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:40]))
