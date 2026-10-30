"""Write benchmarks/otter_hard/STATS.md from the benchmark files (+ bias_report.json if present)."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "benchmarks/otter_hard"
rows = [json.loads(l) for l in open(B / "otter_hard_all.jsonl") if l.strip()]
LET = "ABCD"
def tab(title, counter_rows, cols):
    out = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in counter_rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out + [""]
def cb(rs, role):
    v = [r["closed_book"][role] for r in rs if r["closed_book"][role]["n"]]
    return (100 * sum(x["correct"] for x in v) / sum(x["n"] for x in v)) if v else float("nan")
L = [f"# ViEduHop (files: otter_hard): statistics", "", f"Items: {len(rows)} (dev {sum(r['split']=='dev' for r in rows)}, test {sum(r['split']=='test' for r in rows)}). "
     "Computed from `otter_hard_all.jsonl`. Generated items are automatically verified; teacher review coverage is given by `review_status` (see table below).", ""]
import sys
sys.path.insert(0, str(ROOT))
from otter.evaluation.text import tokens
from otter.evaluation.corpus import SpanCorpus, core_book_filter
_corpus = SpanCorpus.load(book_filter=core_book_filter)
def _lens(rs, f):
    v = [f(r) for r in rs]; v = sorted(x for x in v if x is not None)
    return f"{sum(v)/len(v):.1f} / {v[len(v)//2]} / {v[-1]}" if v else "--"
hl = sorted({r.get("hop_levels") or len(r["gold_span_ids"]) for r in rows})
vh = [("hop level", "#samples", "vocab", "question len (mean/med/max syll.)", "answer len", "context len (gold spans)")]
tabh = []
for h in hl:
    rs = [r for r in rows if (r.get("hop_levels") or len(r["gold_span_ids"])) == h]
    voc = len({t for r in rs for t in tokens(r["question"] + " " + " ".join(r["options"].values()))})
    tabh.append((h, len(rs), voc, _lens(rs, lambda r: len(tokens(r["question"]))), _lens(rs, lambda r: len(tokens(r["options"][r["answer"]]))),
                 _lens(rs, lambda r: sum(len(tokens(_corpus.get(s).text)) for s in r["gold_span_ids"] if _corpus.get(s)))))
doms = sorted({r["domain"] for r in rows}); hops = sorted({r["hop_type"] for r in rows}); grades = sorted({r["grade"] for r in rows})
diffs = ["hard", "medium", "easy"]
L += tab("ViHERMES-style length statistics per hop level (syllables)", tabh, list(vh[0]))
L += tab("By subset / origin / review status", [("subset=" + k, v) for k, v in Counter(r.get("subset", "core") for r in rows).items()] +
         [("origin=" + k, v) for k, v in Counter(r.get("origin", "generated") for r in rows).items()] +
         [("review_status=" + k, v) for k, v in Counter(r.get("review_status") for r in rows).items()], ["field", "n"])
L += tab("Bloom-style question type", [(k, v) for k, v in Counter(r.get("bloom") for r in rows).most_common()], ["type", "n"])
L += tab("Pedagogical type of gold spans (LLM label)", [(k, v) for k, v in Counter(t for r in rows for t in (r.get("ped_types") or {}).values()).most_common()], ["type", "n spans"])
cr = [r for r in rows if r.get("subset") == "cross_subject"]
if cr:
    L += tab("OTTER-Cross: subject pairs", [(k, v) for k, v in Counter(" - ".join(r["subjects"]) for r in cr).most_common()], ["pair", "n"])
L += tab("By subject", [(d, sum(r["domain"] == d for r in rows), sum(r["domain"] == d and r["split"] == "dev" for r in rows),
                         sum(r["domain"] == d and r["split"] == "test" for r in rows)) for d in doms] + [("**all**", len(rows), sum(r["split"] == "dev" for r in rows), sum(r["split"] == "test" for r in rows))],
         ["subject", "n", "dev", "test"])
L += tab("By grade", [(g, sum(r["grade"] == g for r in rows)) for g in grades], ["grade", "n"])
L += tab("By hop type", [(h, sum(r["hop_type"] == h for r in rows)) for h in hops], ["hop type", "n"])
L += tab("Subject x hop type", [(d, *[sum(r["domain"] == d and r["hop_type"] == h for r in rows) for h in hops]) for d in doms], ["subject", *hops])
L += tab("Subject x grade", [(d, *[sum(r["domain"] == d and r["grade"] == g for r in rows) for g in grades]) for d in doms], ["subject", *map(str, grades)])
L += tab("By difficulty (closed-book Qwen3-8B probe: hard = 0 correct over option permutations, easy = all correct with mean confidence >= 80, else medium)",
         [(d, sum(r["difficulty"] == d for r in rows), f"{100*sum(r['difficulty']==d for r in rows)/len(rows):.1f}%") for d in diffs], ["difficulty", "n", "share"])
L += tab("Subject x difficulty", [(d, *[sum(r["domain"] == d and r["difficulty"] == x for r in rows) for x in diffs]) for d in doms], ["subject", *diffs])
pos = Counter(r["answer"] for r in rows)
L += tab("Answer position", [(k, pos[k], f"{100*pos[k]/len(rows):.1f}%") for k in LET], ["key", "n", "share"])
ne = [r for r in rows if r["difficulty"] != "easy"]
rowsT = [r for r in rows if r["split"] == "test"]
L += tab("Closed-book accuracy (no passages; mean over 4 option permutations; chance 25%)",
         [("all items", len(rows), f"{cb(rows,'small'):.1f}", f"{cb(rows,'main'):.1f}"),
          ("non-easy (hard+medium)", len(ne), f"{cb(ne,'small'):.1f}", f"{cb(ne,'main'):.1f}"),
          ("test split", len(rowsT), f"{cb(rowsT,'small'):.1f}", f"{cb(rowsT,'main'):.1f}"),
          ("test non-easy", sum(r['difficulty']!='easy' for r in rowsT), f"{cb([r for r in rowsT if r['difficulty']!='easy'],'small'):.1f}", f"{cb([r for r in rowsT if r['difficulty']!='easy'],'main'):.1f}")],
         ["subset", "n", "Qwen3-8B (small) %", "Qwen3-32B-FP8 (main) %"])
L += ["Note: difficulty labels are derived from the small-model probe, so the small-model accuracy on the non-easy subset is low by construction; the main-model number is the independent one.", ""]
L += tab("Closed-book accuracy by subject", [(d, sum(r["domain"] == d for r in rows), f"{cb([r for r in rows if r['domain']==d],'small'):.1f}", f"{cb([r for r in rows if r['domain']==d],'main'):.1f}") for d in doms],
         ["subject", "n", "small %", "main %"])
def ul(r):
    l = {k: len(r["options"][k]) for k in LET}; m = max(l.values())
    return l[r["answer"]] == m and list(l.values()).count(m) == 1
def ratio(r):
    d = [len(r["options"][k]) for k in LET if k != r["answer"]]; return len(r["options"][r["answer"]]) / (sum(d) / 3)
L += ["### Option-length bias", "", f"- Correct option is the unique longest: {100*sum(map(ul, rows))/len(rows):.1f}% (v1: 66.7%; chance 25%)",
      f"- Mean key length / mean distractor length: {sum(map(ratio, rows))/len(rows):.3f}",
      f"- Items whose distractors were rewritten for length matching: {sum(bool(r.get('distractor_rewritten')) for r in rows)}", ""]
bp = B / "bias_report.json"
if bp.exists():
    b = json.loads(bp.read_text())
    L += ["### Answer-only and heuristic baselines (artifact check; chance 25%)", "", "| baseline | accuracy % |", "|---|---|"]
    for k, v in b.items():
        if k.endswith("acc") or k.endswith("acc_non_easy") or k.endswith("rate"):
            L.append(f"| {k} | {100*v:.1f} |")
    L.append("")
ev = [len(r["gold_span_ids"]) for r in rows]
L += ["### Evidence", "", f"- gold spans per item: {dict(Counter(ev))}; distinct gold spans {len({s for r in rows for s in r['gold_span_ids']})}; books {len({b for r in rows for b in r['book_ids']})}",
      f"- cross-book items {sum(len(r['book_ids'])>1 for r in rows)}, cross-lesson items {sum(len(r['lesson_keys'])>1 for r in rows)}",
      f"- seed strategies: {dict(Counter(r['seed_strategy'] for r in rows))}", f"- calculation items with deterministic arithmetic check: {sum(1 for r in rows if r.get('calc') and r['calc'].get('calc_ok'))}/{sum(r['hop_type']=='calculation' for r in rows)}", ""]
import re as _re
_old = (B / "STATS.md").read_text(encoding="utf-8") if (B / "STATS.md").exists() else ""
_m = _re.search(r"<!-- book-specific:begin -->.*?<!-- book-specific:end -->", _old, flags=_re.S)
_nbs = sum(1 for r in rows if r.get("book_specific"))
if _nbs:
    L.append(f"\nBook-specific items merged into this benchmark: {_nbs} of {len(rows)} (details in section 'book-specific pool' below).\n")
if _m:        # section written by scripts/eval/otter_hard_bs_stats.py: keep it when this file is regenerated
    L.append("\n" + _m.group(0) + "\n")
(B / "STATS.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
