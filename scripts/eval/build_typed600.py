"""Build ViEduHop-Typed (600 true/false, fill-in-the-blank and multiple-choice questions) from the two CSVs in data/2_types_questions/.

  .venv/bin/python scripts/eval/build_typed600.py [--out-dir benchmarks/typed] [--allow-missing-alignment]

Inputs are read-only: OTTER_2types_400.csv, OTTER_EXTRA_200.csv and the *_aligned.jsonl files produced by scripts/eval/align_two_types.py.
Outputs (new files): typed600.jsonl, typed600_mcq_runner.jsonl (true_false + multiple_choice in the scripts/eval/run.py schema),
typed600_groups.csv, duplicate_report.md, duplicates.csv, key_conflicts_for_teacher_review.csv, build_stats.json, README.md.
Nothing here is expert- or teacher-validated; alignment is automatic (status 'auto_aligned' is NOT human-verified).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import typed_common as T  # noqa: E402
from otter.evaluation.mcq import SUBJECT_VI_TO_DOMAIN  # noqa: E402
from otter.evaluation.text import char_ngrams, jaccard, norm  # noqa: E402

D = ROOT / "data/2_types_questions"
NEG = re.compile(r"^\s*Không đúng khi cho rằng\s*:\s*", re.I)
FILL_PREFIX = re.compile(r"^\s*Điền khái niệm hoặc giá trị thích hợp\s*:\s*", re.I)
BLANK = re.compile(r"_{2,}")
F400, FEXTRA = "OTTER_2types_400.csv", "OTTER_EXTRA_200.csv"


def sha(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def read_aligned(p: Path) -> dict:
    out = {}
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def true_statement(row: dict) -> str | None:
    """A complete TRUE sentence for the item's fact (None for items whose statement is false)."""
    q, f = row["question"], row["format"]
    if f == "fill_blank":
        return BLANK.sub(row["answer"], FILL_PREFIX.sub("", q))
    if f == "multiple_choice":
        ans = row["choices"][row["answer"]]
        return BLANK.sub(ans, q) if BLANK.search(q) else f"{q} {ans}"
    stem = NEG.sub("", q)
    stem_true = (row["answer"] == "A") != bool(row["negated"])
    return stem if stem_true else None


def build_rows(allow_missing: bool) -> list[dict]:
    al400 = read_aligned(D / "OTTER_2types_400_aligned.jsonl")
    alx = read_aligned(D / "OTTER_EXTRA_200_aligned.jsonl")
    if not allow_missing and (len(al400) != 400 or len(alx) != 200):
        raise SystemExit(f"alignment incomplete: {len(al400)}/400 and {len(alx)}/200 rows; refusing to build")
    rows: list[dict] = []
    for r in csv.DictReader(open(D / F400, encoding="utf-8-sig")):
        fmt = r["type"]
        row = {"id": r["id"], "source_file": F400, "format": fmt, "subject": r["subject"], "grade": int(r["grade"]),
               "topic": r["topic"], "question": r["question"].strip(), "explanation": r["explanation"].strip(),
               "group_id": f"{F400}:{r['pair_id']}", "reasoning_type": r["reasoning_type"],
               "file_fields": {"source_verification": r["source_verification"], "expert_validation": r["expert_validation"],
                               "reference_url": r["reference_url"], "reference_scope": r["reference_scope"]}}
        if fmt == "true_false":
            row["choices"] = {"A": "Đúng", "B": "Sai"}
            row["answer"] = "A" if r["answer"].strip().upper() == "TRUE" else "B"
        else:
            row["choices"] = None
            row["answer"] = r["answer"].strip()
        rows.append(row)
    for r in csv.DictReader(open(D / FEXTRA, encoding="utf-8-sig")):
        qt = r["question_type"]
        fmt = {"fill_in_blank": "fill_blank"}.get(qt, qt)
        row = {"id": r["id"], "source_file": FEXTRA, "format": fmt, "subject": r["subject"], "grade": int(r["grade"]),
               "topic": r["topic"], "question": r["question"].strip(), "explanation": r["explanation"].strip(),
               "group_id": f"{FEXTRA}:{r['fact_group_id']}", "reasoning_type": None,
               "file_fields": {"textbook_citation_status": r["textbook_citation_status"], "review_status": r["review_status"],
                               "reference_url": r["reference_url"], "reference_level": r["reference_level"],
                               "textbook_series": r["textbook_series"], "textbook_lesson": r["textbook_lesson"],
                               "textbook_page": r["textbook_page"], "source_span_id": r["source_span_id"], "origin": r["origin"]}}
        if fmt == "multiple_choice":
            row["choices"] = {k: r[f"option_{k}"] for k in "ABCD"}
            row["answer"] = r["answer"].strip().upper()
            assert row["choices"][row["answer"]] == r["answer_text"], r["id"]
        elif fmt == "true_false":
            row["choices"] = {"A": "Đúng", "B": "Sai"}
            row["answer"] = "A" if r["answer"].strip().lower() in ("đúng", "true") else "B"
        else:
            row["choices"] = None
            row["answer"] = (r["answer"] or r["answer_text"]).strip()
        rows.append(row)
    for row in rows:
        row["domain"] = SUBJECT_VI_TO_DOMAIN[row["subject"]]
        row["negated"] = bool(NEG.match(row["question"])) if row["format"] == "true_false" else False
        if row["format"] == "fill_blank":
            row["accepted_variants"] = T.accepted_variants(row["answer"])
        stmt = true_statement(row)
        row["key_statement_true"] = None
        if row["format"] == "true_false":
            row["key_statement_true"] = (row["answer"] == "A")
        a = (al400 if row["source_file"] == F400 else alx).get(row["id"])
        if a is None:
            row.update(alignment_status="not_run", gold_span_ids=[], candidate_span_ids=[], align_best_span_id=None,
                       align_quote=None, align_quote_found_in_span=None, align_llm_verdict=None,
                       align_consistent_with_key=None, ontology_items=[], align_note=None)
        else:
            st = a["alignment_status"]
            row.update(alignment_status=st,
                       gold_span_ids=[a["best_span_id"]] if (st == "auto_aligned" and a.get("best_span_id")) else [],
                       candidate_span_ids=a.get("candidate_span_ids") or [], align_best_span_id=a.get("best_span_id"),
                       align_quote=a.get("quote"), align_quote_found_in_span=a.get("quote_found_in_span"),
                       align_llm_verdict=a.get("llm_verdict"), align_consistent_with_key=a.get("consistent_with_key"),
                       ontology_items=a.get("ontology_items") or [], align_note=a.get("note"))
        row["alignment_basis"] = ("sentence completed with the correct option (a true statement); distractors are not checked"
                                  if row["format"] == "multiple_choice" else
                                  "key-completed sentence" if row["format"] == "fill_blank" else
                                  "statement with the negation prefix removed; verdict compared with the key")
        row["authorship_user_statement"] = T.AUTHORSHIP_STATEMENT
        row["review_status_user_statement"] = T.PROVENANCE_STATEMENT
        row["provenance_doc"] = "docs/TYPED600_PROVENANCE.md"
        row["subset"] = "typed"
        row["benchmark_name"] = "ViEduHop-Typed"
        row["split"] = "eval_only"
        row["auto_verified"] = False
        row["used_for_tuning"] = False
        row["_true_statement"] = stmt
    return rows


# ------------------------------------------------------------------------------------------------ duplicates
def load_hard_questions() -> list[dict]:
    out, seen = [], set()
    for p in sorted((ROOT / "benchmarks/otter_hard").glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            q = r.get("question")
            if not q or not isinstance(q, str):
                continue
            opts = r.get("options") or {}
            ans = opts.get(r.get("answer"), "") if isinstance(opts, dict) else ""
            key = (r.get("id") or r.get("cand_id"), norm(q))
            if key in seen:
                continue
            seen.add(key)
            out.append({"file": p.name, "id": r.get("id") or r.get("cand_id"), "question": q, "answer_text": ans,
                        "g_q": char_ngrams(q), "g_qa": char_ngrams(q + " " + ans)})
    return out


def dup_analysis(rows: list[dict]) -> dict:
    for r in rows:
        r["_qn"] = norm(FILL_PREFIX.sub("", r["question"]))
        r["_g"] = char_ngrams(FILL_PREFIX.sub("", NEG.sub("", r["question"])))
        r["_gs"] = char_ngrams(r["_true_statement"]) if r["_true_statement"] else None
    n = len(rows)
    pairs = []          # (a, b, kind, sim, same_group)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rows[i], rows[j]
            same_g = a["group_id"] == b["group_id"]
            if a["_qn"] == b["_qn"]:
                pairs.append((a, b, "exact_question", 1.0, same_g))
                continue
            if not same_g and a["_gs"] and b["_gs"]:
                s2 = jaccard(a["_gs"], b["_gs"])
                if s2 >= 0.8:
                    pairs.append((a, b, "same_fact_statement", s2, same_g))
                    continue
            s = jaccard(a["_g"], b["_g"])
            if s >= 0.8:
                pairs.append((a, b, "near_question", s, same_g))
    # group-level fact clustering on true statements of different groups (union-find)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    cross = [(a, b, k, s) for a, b, k, s, sg in pairs if not sg]
    for a, b, k, s in cross:
        if k in ("exact_question", "same_fact_statement"):      # near_question may differ in the key (1945 vs 1967) -> no link
            parent[find(a["group_id"])] = find(b["group_id"])
    clusters = collections.Counter(find(g) for g in groups)
    # against OTTER-Hard files
    hard = load_hard_questions()
    vs_hard = []
    for r in rows:
        best = (0.0, None, None)
        gq = r["_g"]
        gst = r["_gs"]
        for h in hard:
            s = jaccard(gq, h["g_q"])
            if gst:
                s = max(s, jaccard(gst, h["g_qa"]))
            if s > best[0]:
                best = (s, h, norm(FILL_PREFIX.sub("", r["question"])) == norm(h["question"]))
        if best[1] and best[0] >= 0.6:
            vs_hard.append((r, best[1], best[0], best[2]))
    return {"pairs": pairs, "n_groups": len(groups), "n_fact_clusters": len(clusters), "cross_group_pairs": cross,
            "vs_hard": vs_hard, "n_hard_questions": len(hard), "groups": groups}


def write_dup_report(rows: list[dict], dup: dict, out: Path) -> dict:
    pairs = dup["pairs"]
    by_kind = collections.Counter((k, sg) for _, _, k, _, sg in pairs)
    fill_pairs = [(a, b, s) for a, b, k, s, sg in pairs if sg and a["format"] == b["format"] == "fill_blank"]
    exact_fill = [(a, b) for a, b, k, s, sg in pairs if sg and k == "exact_question" and a["format"] == b["format"] == "fill_blank"]
    vh = dup["vs_hard"]
    exact_hard = [x for x in vh if x[3]]
    with open(out / "duplicates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "kind", "similarity", "same_group", "a_id", "a_format", "a_group", "a_text", "b_id", "b_file", "b_text"])
        for a, b, k, s, sg in pairs:
            w.writerow(["typed_vs_typed", k, f"{s:.3f}", sg, a["id"], a["format"], a["group_id"], a["question"], b["id"], b["source_file"], b["question"]])
        for r, h, s, ex in vh:
            w.writerow(["typed_vs_otter_hard", "exact_question" if ex else "near", f"{s:.3f}", "", r["id"], r["format"], r["group_id"],
                        r["question"], h["id"], h["file"], h["question"]])
    cross = [(a, b, k, s) for a, b, k, s, sg in pairs if not sg]
    L = ["# ViEduHop-Typed: duplicate / overlap report", "",
         "Generated by `scripts/eval/build_typed600.py`. Similarity = character 3-gram Jaccard on NFC lower-cased text without punctuation "
         "(`otter.evaluation.text`). Near-duplicate threshold 0.8 between typed items, 0.6 against OTTER-Hard questions "
         "(question text, or the completed true statement against question + correct option). Full pair list: `duplicates.csv`.", "",
         "## 1. Within ViEduHop-Typed", "",
         f"- Items: {len(rows)}; groups: {dup['n_groups']}.",
         f"- Exact duplicate question text (after removing the 'Điền khái niệm hoặc giá trị thích hợp:' prefix): "
         f"{by_kind[('exact_question', True)]} pair(s) inside a group, {by_kind[('exact_question', False)]} pair(s) across groups.",
         f"- Near-duplicate question pairs (Jaccard >= 0.8, not exact): {by_kind[('near_question', True)]} inside a group, "
         f"{by_kind[('near_question', False)]} across groups.",
         f"- Same-fact statements in different groups (true statements, Jaccard >= 0.8): {by_kind[('same_fact_statement', False)]} pair(s).",
         f"- Near-identical fill-in pairs inside a group (the 25 EXTRA fill-in groups): {len(exact_fill)} exact after prefix removal, "
         f"{len(fill_pairs)} exact-or-near in total. These two items test the same fact with the same blank and the same key; "
         "each group therefore counts as ONE distinct question, and group accuracy equals item accuracy up to model noise between the two phrasings.", ""]
    L += ["### Cross-group pairs (potential real duplicates)", ""]
    if cross:
        L += ["| kind | sim | a | b | a text | b text |", "|---|---|---|---|---|---|"]
        for a, b, k, s in cross:
            L.append(f"| {k} | {s:.2f} | {a['id']} ({a['group_id'].split(':')[1]}) | {b['id']} ({b['group_id'].split(':')[1]}) | "
                     f"{a['question'][:90]} | {b['question'][:90]} |")
    else:
        L.append("None found.")
    L += ["", f"Groups merged by cross-group same-fact links: {dup['n_groups']} groups -> {dup['n_fact_clusters']} distinct fact clusters.",
          "The structure is by design: the 400 file has 100 facts x 4 items (3 true/false + 1 fill-in), the EXTRA file has 100 facts x 2 items "
          "(75 MCQ+TF pairs, 25 fill-in pairs). Items inside a group are NOT independent; the group is the unit of analysis.", "",
          "## 2. Against benchmarks/otter_hard/*.jsonl", "",
          f"- OTTER-Hard questions compared: {dup['n_hard_questions']} unique (id, question) rows from all `benchmarks/otter_hard/*.jsonl` files "
          "(test, dev, ablation300, cross, deep, closed-book probe, candidate files).",
          f"- Typed items with a match >= 0.6: {len(vh)}; exact normalized question matches: {len(exact_hard)}.", ""]
    if vh:
        L += ["| typed id | sim | OTTER-Hard id (file) | typed text | OTTER-Hard text |", "|---|---|---|---|---|"]
        for r, h, s, ex in sorted(vh, key=lambda x: -x[2])[:40]:
            L.append(f"| {r['id']} | {s:.2f}{' (exact)' if ex else ''} | {h['id']} ({h['file']}) | {r['question'][:80]} | {h['question'][:80]} |")
        if len(vh) > 40:
            L.append(f"| ... | | | {len(vh) - 40} more in duplicates.csv | |")
    else:
        L.append("No typed item reaches 0.6 similarity with any OTTER-Hard question.")
    # cross-file overlap
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["_qn"]].append(r)
    xf = [v for v in byq.values() if len({x["source_file"] for x in v}) > 1]
    L += ["", "## 3. Overlap between the two source files", ""]
    if xf:
        for v in xf:
            L.append("- exact question match across files: " + " / ".join(f"{x['id']} ({x['source_file']}, {x['group_id'].split(':')[1]}, {x['format']})" for x in v)
                     + f": \"{v[0]['question'][:100]}\"")
    else:
        L.append("No exact question matches across the two files.")
    L.append("Both rows are kept in typed600.jsonl (different ids and groups); they are counted once as one fact in the distinct-fact estimate if the link above exists.")
    (out / "duplicate_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"exact_in_group": by_kind[("exact_question", True)], "exact_across_groups": by_kind[("exact_question", False)],
            "near_in_group": by_kind[("near_question", True)], "near_across_groups": by_kind[("near_question", False)],
            "same_fact_across_groups": by_kind[("same_fact_statement", False)], "extra_fill_exact_pairs": len(exact_fill),
            "vs_hard_ge_0.6": len(vh), "vs_hard_exact": len(exact_hard), "distinct_fact_clusters": dup["n_fact_clusters"],
            "cross_file_exact": len(xf)}


# ------------------------------------------------------------------------------------------------ outputs
def public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def runner_row(r: dict) -> dict:
    """true_false (2 choices) and multiple_choice rows in the scripts/eval/run.py / MCQItem schema (extra keys are allowed)."""
    d = public(r)
    d["options"] = d.pop("choices")
    d["typed_format"] = d["format"]
    return d


def write_groups_csv(rows: list[dict], out: Path) -> list[dict]:
    g = collections.defaultdict(list)
    for r in rows:
        g[r["group_id"]].append(r)
    summ = []
    with open(out / "typed600_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "source_file", "subject", "grade", "topic", "n_items", "formats", "item_ids", "negated_items",
                    "n_auto_aligned", "n_key_conflict", "n_not_aligned", "all_items_aligned", "gold_span_ids_union", "fact_statement"])
        for gid, rs in g.items():
            st = collections.Counter(x["alignment_status"] for x in rs)
            gold = sorted({s for x in rs for s in x["gold_span_ids"]})
            stm = next((x["_true_statement"] for x in rs if x["_true_statement"]), "")
            s = {"group_id": gid, "n": len(rs), "formats": collections.Counter(x["format"] for x in rs), "subject": rs[0]["subject"],
                 "source_file": rs[0]["source_file"], "grade": rs[0]["grade"], "status": st}
            summ.append(s)
            w.writerow([gid, rs[0]["source_file"], rs[0]["subject"], rs[0]["grade"], rs[0]["topic"], len(rs),
                        "+".join(f"{k}:{v}" for k, v in sorted(s["formats"].items())), " ".join(x["id"] for x in rs),
                        sum(x["negated"] for x in rs), st["auto_aligned"], st["key_conflict"], st["not_aligned"] + st["not_run"],
                        st["auto_aligned"] == len(rs), " ".join(gold), stm])
    return summ


def span_info():
    try:
        from otter.evaluation.corpus import SpanCorpus, core_book_filter
        return SpanCorpus.load(book_filter=core_book_filter)
    except Exception as e:
        print("no span corpus:", e)
        return None


def write_key_conflicts(rows: list[dict], out: Path, corpus) -> int:
    kc = [r for r in rows if r["alignment_status"] == "key_conflict"]
    with open(out / "key_conflicts_for_teacher_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "group_id", "format", "subject", "grade", "topic", "question", "choices", "key_in_file", "key_text",
                    "explanation_in_file", "llm_verdict_on_span", "span_id", "span_book", "span_text", "llm_quote", "llm_note",
                    "teacher_decision (key_ok / key_wrong / question_flawed / span_not_relevant)", "teacher_comment"])
        for r in kc:
            sid = r["align_best_span_id"]
            sp = corpus.get(sid) if (corpus is not None and sid) else None
            key_text = r["choices"][r["answer"]] if r["choices"] else r["answer"]
            w.writerow([r["id"], r["group_id"], r["format"], r["subject"], r["grade"], r["topic"], r["question"],
                        json.dumps(r["choices"], ensure_ascii=False) if r["choices"] else "", r["answer"], key_text, r["explanation"],
                        r["align_llm_verdict"], sid, getattr(sp, "book_id", ""), (sp.text[:1200] if sp else ""),
                        r["align_quote"], r["align_note"], "", ""])
    return len(kc)


def counts(rows, key):
    return dict(sorted(collections.Counter(key(r) for r in rows).items(), key=lambda x: str(x[0])))


def table(rows, rk, ck, rname, rfmt=str):
    cols = sorted({ck(r) for r in rows}, key=str)
    L = [f"| {rname} | " + " | ".join(str(c) for c in cols) + " | total |", "|---|" + "---|" * (len(cols) + 1)]
    for rv in sorted({rk(r) for r in rows}, key=str):
        sub = [r for r in rows if rk(r) == rv]
        L.append(f"| {rfmt(rv)} | " + " | ".join(str(sum(1 for r in sub if ck(r) == c)) for c in cols) + f" | {len(sub)} |")
    return L


def write_readme(rows: list[dict], gs: list[dict], dstats: dict, nconf: int, out: Path, hashes: dict) -> None:
    pos_mcq = collections.Counter(r["answer"] for r in rows if r["format"] == "multiple_choice")
    pos_tf = collections.Counter(r["answer"] for r in rows if r["format"] == "true_false")
    al = collections.Counter(r["alignment_status"] for r in rows)
    mc = [r for r in rows if r["format"] == "multiple_choice"]
    keys = {r["choices"][r["answer"]] for r in mc}
    n_dis = sum(1 for r in mc for k, v in r["choices"].items() if k != r["answer"])
    n_dis_key = sum(1 for r in mc for k, v in r["choices"].items() if k != r["answer"] and v in keys)
    sat = ""
    sj = ROOT / "runs/vieduhop_typed/typed_summary.json"
    if sj.exists():
        try:
            sd = json.loads(sj.read_text(encoding="utf-8")).get("saturation", {})
            sat = " Closed-book accuracy of `direct` (Qwen3-8B / Qwen3-32B): " + "; ".join(
                f"{k} {v['direct_small'][0]:.3f} / {v['direct_main'][0]:.3f}" for k, v in sd.items()) + "."
        except Exception:
            sat = ""
    L = ["# ViEduHop-Typed (600 true/false, fill-in-the-blank and multiple-choice questions)", "",
         "Subset `typed`, split `eval_only` (there is no dev/test split; the set is never used for tuning). Built by "
         "`scripts/eval/build_typed600.py` from the two CSV files in `data/2_types_questions/` (read-only inputs). "
         "Evaluation results: `runs/vieduhop_typed/typed_summary.md`. Provenance: `docs/TYPED600_PROVENANCE.md`.", "",
         "## Status in one paragraph", "",
         "The questions were authored by the research group and reviewed by 5 teachers (stated by the user on 27 Oct 2026; review records are "
         "pending). The CSV files themselves still say `source_verification = pending_textbook_alignment`, `review_status = NEEDS_EXPERT_REVIEW`, "
         "`textbook_citation_status = UNVERIFIED` and carry no textbook lesson, page or span. The alignment of items to textbook spans below is "
         "AUTOMATIC (hybrid retrieval + Qwen3-32B reading of the top-6 spans) and is not human-verified. Do not call this set teacher-validated; "
         f"the row field is `review_status_user_statement: \"{T.PROVENANCE_STATEMENT}\"`.", "",
         "The authors further state (relayed by the user, 27 Oct 2026) that the five reviewing teachers teach Hóa học, Sinh học, Lịch sử, Địa lí and Ngữ văn "
         "respectively, each with more than 5 years of teaching (statement by the authors; review records still pending). This matches the five subjects "
         "present in the files (no Vật lí item). No per-item reviewer counts and no applied corrections are claimed.", "",
         "## Construction", "",
         f"- `{F400}` (sha1 {hashes[F400][:10]}): 400 rows = 100 groups (`pair_id`) x 4 items about one fact: 3 true/false (one is sometimes a "
         "'Không đúng khi cho rằng:' negated stem) + 1 fill-in-the-blank. No Vật lí.",
         f"- `{FEXTRA}` (sha1 {hashes[FEXTRA][:10]}): 200 rows = 100 groups (`fact_group_id`) x 2 items about the same fact: 75 x (multiple choice + "
         "true/false) and 25 x (two near-identical fill-in items; one carries the prefix 'Điền khái niệm hoặc giá trị thích hợp:').",
         f"- Group ids are prefixed by the source file (`{F400}:PAIR_001`, `{FEXTRA}:Lịch sử_01`) so they cannot collide.",
         "- true_false -> `choices = {\"A\": \"Đúng\", \"B\": \"Sai\"}`, `answer` = A (TRUE/Đúng) or B (FALSE/Sai). `negated` = stem starts with "
         "'Không đúng khi cho rằng:' (key FALSE means the embedded statement is true). multiple_choice -> A-D as in the CSV. fill_blank -> "
         "`answer` is the gold term, `accepted_variants` lists obvious spelling/transliteration variants (acid/axit, DNA/ADN, ...).",
         "- `gold_span_ids` is filled only when `alignment_status == auto_aligned` (the LLM-chosen span exists among the retrieved candidates, its quote "
         "occurs verbatim in the span, and its verdict agrees with the key). `candidate_span_ids`, the chosen span and the quote are kept separately. "
         "`key_conflict` = a span was chosen but the LLM verdict disagrees with the key (listed in `key_conflicts_for_teacher_review.csv`; may be a "
         "false alarm of the aligner). `not_aligned` = no usable span or quote.",
         "- `auto_verified` is false for every row. `split` is `eval_only`.",
         "- `typed600_mcq_runner.jsonl` is the true_false + multiple_choice part in the schema of `scripts/eval/run.py` (`options`, `domain`, `gold_span_ids`); "
         "the true/false statement is passed unchanged as `question` with options A. Đúng / B. Sai (no wrapper text).", "",
         "## Counts", "", f"Items: {len(rows)}; groups: {len(gs)}.", ""]
    L += table(rows, lambda r: r["format"], lambda r: r["source_file"], "format x source file") + [""]
    L += table(rows, lambda r: r["subject"], lambda r: r["format"], "subject x format") + [""]
    L += table(rows, lambda r: r["grade"], lambda r: r["format"], "grade x format") + [""]
    L += ["Group structure: " + ", ".join(f"{k}: {v}" for k, v in sorted(collections.Counter(
        (g["source_file"], "+".join(f"{a}:{b}" for a, b in sorted(g["formats"].items()))) for g in gs).items(), key=str)) + "", ""]
    L += ["Negated true/false stems: " + str(sum(r["negated"] for r in rows)) + " (all in the 400 file). "
          f"Answer-letter distribution: multiple choice {dict(sorted(pos_mcq.items()))} (key skews to D); true/false {dict(sorted(pos_tf.items()))}.", "",
          "## Alignment status (automatic, not human-verified)", ""]
    L += table(rows, lambda r: r["source_file"], lambda r: r["alignment_status"], "source file x status") + [""]
    L += table(rows, lambda r: r["format"], lambda r: r["alignment_status"], "format x status") + [""]
    L += table(rows, lambda r: r["subject"], lambda r: r["alignment_status"], "subject x status") + [""]
    nfc_ = sum(1 for r in rows if r["alignment_status"] == "key_conflict" and r["format"] == "fill_blank")
    L += [f"`key_conflict` items: {nconf}, of which {nfc_} are fill_blank: there the aligner's extracted term differs from the key, which is often a plausible "
          "alternative completion (e.g. key 'Tư liệu hiện vật' vs aligner 'Sử liệu'), not necessarily a key error. A key conflict on true/false or "
          "multiple-choice items is more informative. All are listed in the organiser-only CSV `key_conflicts_for_teacher_review.csv` "
          "(not placed in annotation/ or teacher packs).", ""]
    L += ["## Evaluation protocol (summary; details in runs/vieduhop_typed/typed_summary.md)", "",
          "Systems: direct (closed-book; Qwen3-32B and a Qwen3-8B probe), rag_hybrid, ircot, otter_v2 (frozen). Main model Qwen3-32B-FP8 on :8005 "
          "(`OTTER_LLM_URL_MAIN`), Qwen3-8B on :8001 (also otter_v2's verifier), bge-m3 on :8002; at most 3 concurrent LLM requests. true_false (2-choice A. Đúng / B. Sai) and "
          "multiple_choice go through `scripts/eval/run.py` on `typed600_mcq_runner.jsonl`; fill_blank goes through `scripts/eval/typed_fill_runner.py` "
          "(same system classes, no choices, one shared instruction prefix). otter_v2 has no native free-text answer output (it returns an empty answer for items without "
          "choices), so a thin subclass in the runner file captures the generator's `answer` field; retrieval, generation, verification and cascade are the frozen code. "
          "fill_blank is scored by normalized string match and by a fixed-prompt Qwen3-32B judge (`runs/vieduhop_typed/judge_prompt_v1.txt`, frozen before inspecting results). "
          "The generation runs were started while the alignment job was still running, from a preliminary runner file with identical system inputs; the runs were then "
          "re-scored against the final gold spans (`run_meta_generation.json` keeps the generation-time metadata). Unit of analysis: group; CIs by group bootstrap.", ""]
    L += ["## Duplicates and overlaps", "",
          f"See `duplicate_report.md` and `duplicates.csv`. Summary: {json.dumps(dstats, ensure_ascii=False)}", ""]
    L += ["## Limitations (read before citing numbers)", "",
          "1. **Citations are unverified.** No item has a textbook lesson, page or span in the source files. `gold_span_ids` come from an automatic "
          "aligner that used the same hybrid retriever family as the systems under test, looked only at the top-6 candidates and records a single best "
          "span. `located` is therefore biased towards retrieval-based systems and under-counts valid alternative spans; it is defined only on "
          "`auto_aligned` items, which are a self-selected (easier-to-retrieve) subset.",
          "2. **Few distinct facts.** 200 groups = about 100 + 100 facts (see the duplicate report for the cluster count). Items inside a group share one fact "
          "and are not independent; report group-level accuracy with group-level bootstrap CIs. The 25 EXTRA fill-in groups are two phrasings of one question.",
          f"3. **MCQ distractors are drawn from unrelated topics** (e.g. 'các phản ứng hóa học' as a distractor for a history stem); they are not hard "
          f"distractors, so multiple choice is easy by elimination. Measured: {n_dis_key} of {n_dis} MCQ distractors are the correct answer of another MCQ item "
          "(the distractor pool is the set of keys of other questions). The key is also skewed towards option D.",
          "4. **True/false and multiple choice saturate in closed book** (they are answerable without the textbook; see Table 3 of "
          "`runs/vieduhop_typed/typed_summary.md`)." + sat + " Fill-in is not saturated under single-key scoring, but its blanks are often under-determined "
          "(several terms fit), so low fill-in accuracy partly measures key ambiguity. The tier tests format coverage and provenance behaviour, not retrieval-dependent reasoning.",
          "5. **Review status.** Teacher review is stated by the authors only; the files contain `expert_validation = TRUE` (400 file) and "
          "`NEEDS_EXPERT_REVIEW` (EXTRA file) at the same time. Keys may contain errors (see `key_conflict`).",
          "6. **No Vật lí**; 5 subjects only. Grades 10-12.",
          "7. Fill-in items are scored by normalized string match and by a Qwen3-32B judge (the judge shares its model family with the main backbone); "
          "they disagree on some items, both are reported.", ""]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "benchmarks/typed"))
    ap.add_argument("--allow-missing-alignment", action="store_true")
    ap.add_argument("--no-dups", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(a.allow_missing_alignment)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) == 600, "ids must be unique and 600"
    with open(out / "typed600.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(public(r), ensure_ascii=False) + "\n")
    with open(out / "typed600_mcq_runner.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["format"] != "fill_blank":
                f.write(json.dumps(runner_row(r), ensure_ascii=False) + "\n")
    hashes = {F400: sha(D / F400), FEXTRA: sha(D / FEXTRA)}
    dstats = {}
    if not a.no_dups:
        dup = dup_analysis(rows)
        dstats = write_dup_report(rows, dup, out)
    gs = write_groups_csv(rows, out)
    nconf = write_key_conflicts(rows, out, span_info())
    write_readme(rows, gs, dstats, nconf, out, hashes)
    stats = {"n": len(rows), "by_format": counts(rows, lambda r: r["format"]), "by_subject": counts(rows, lambda r: r["subject"]),
             "by_grade": counts(rows, lambda r: r["grade"]), "alignment": counts(rows, lambda r: r["alignment_status"]),
             "n_groups": len(gs), "duplicates": dstats, "input_sha1": hashes}
    (out / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
