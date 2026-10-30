"""Experiment B (SYNTHETIC): controlled "second edition" of a few lessons with exact ground truth.

  .venv/bin/python scripts/maintenance/ingest_synthetic.py make   --name s1            # build the edition-2 copy + edit log
  .venv/bin/python scripts/maintenance/ingest_synthetic.py ingest --name s1 --mode llm # ingest into the FULL ontology (read-only)
  .venv/bin/python scripts/maintenance/ingest_synthetic.py eval   --name s1 --mode llm
modes: llm (default pipeline) | lexical (no claim LLM) | nonconfirm (lexical conflicts are not LLM-confirmed)

make   : picks the most number-dense lesson chunks of three core books, copies their spans under a new book id `<book>__ed2`
         (new span ids), applies `simulate_edition.simulate_edition` value swaps (number / year / entity; one value swapped
         everywhere in a span), and records every edit.  Nothing in data/ or runs/corpus is written.
ingest : `ingest-books` of that copy into data/ontology/ontology.sqlite3 (live extraction by `main`).
eval   : item-level ground truth.  A new fact/rule N is a TRUE conflict iff it cites an edited span, states the NEW value, and the
         full ontology holds an item citing the ORIGINAL span whose text contains the OLD value.  Precision = detected conflicts
         that are true; recall = true conflicts that are detected.  Items of unedited spans must not conflict (false-conflict rate).
This is synthetic: the "edition" differs from the original only by the recorded value swaps (no rephrasing, no reordering).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OTTER_RUN_TAG", "knowledge_ingest__synthetic")

from otter.builder.segment import segment_book                                   # noqa: E402
from otter.builder.simulate_edition import find_entities, find_numbers, simulate_edition   # noqa: E402
from otter.builder.ingest import load_spans                                      # noqa: E402
from otter.storage.textutil import norm_key                                      # noqa: E402
from otter.storage.writer import rule_text                                       # noqa: E402

FULL_DB = ROOT / "data/ontology/ontology.sqlite3"
PROV = ROOT / "data/textbooks/ontology.sqlite3"
PARSED = ROOT / "data/textbooks/parsed"
KI = ROOT / "runs/knowledge_ingest"
CACHE = KI / "cache_main"
BOOKS = ["nxbgd_g12_geography_d7554213", "nxbgd_g12_history_b44d76bd", "nxbgd_g11_physics_ce4cc7d0"]
SUFFIX = "__ed2"


def _meta(book: str) -> dict:
    con = sqlite3.connect(f"file:{PROV}?mode=ro", uri=True)
    r = con.execute("SELECT metadata_json FROM books WHERE book_id=?", (book,)).fetchone()
    con.close()
    return json.loads(r[0])


def _rows(book: str) -> dict[str, dict]:
    return {json.loads(l)["span_id"]: json.loads(l) for l in (PARSED / book / "spans.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}


def make(a):
    d = KI / "synthetic" / a.name
    out_root = d / "edition"
    out_root.mkdir(parents=True, exist_ok=True)
    all_edits, all_rows, picked = {}, [], []
    for book in BOOKS:
        raw = _rows(book)
        sections = {}
        for l in (PARSED / book / "sections.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                s = json.loads(l)
                sections[s["section_id"]] = s
        meta = _meta(book)
        spans = load_spans(Path("/nonexistent"), [book], PARSED)[book]
        chunks = segment_book(book, spans, sections, subject=meta.get("subject_normalized") or "other", grade=meta.get("grade"),
                              book_title=meta.get("title") or book)
        dens = []
        for c in [c for c in chunks if c.tokens >= 800]:
            n = sum(len(find_numbers(s.text)) for s in c.spans)
            dens.append((n / max(1, c.tokens), c))
        dens.sort(key=lambda x: -x[0])
        for _, c in dens[a.skip_chunks: a.skip_chunks + a.chunks_per_book]:
            picked.append({"book": book, "chunk_id": c.chunk_id, "n_spans": len(c.spans), "tokens": c.tokens})
            all_rows += [raw[s.span_id] for s in c.spans]
    # one edition book per original book
    by_book = defaultdict(list)
    for r in all_rows:
        by_book[r["book_id"]].append(r)
    for book, rows in by_book.items():
        new_book = book + SUFFIX
        renamed = []
        for r in rows:
            r2 = dict(r)
            r2["span_id"] = r["span_id"].replace(book, new_book, 1)
            r2["book_id"] = new_book
            if r2.get("section_id"):
                r2["section_id"] = r2["section_id"].replace(book, new_book, 1)
            if r2.get("page_span_id"):
                r2["page_span_id"] = r2["page_span_id"].replace(book, new_book, 1)
            r2.pop("text_sha1", None)
            renamed.append(r2)
        new_rows, elog = simulate_edition(renamed, seed=a.seed, delete_frac=0.0, edit_frac=a.edit_frac)
        bd = out_root / new_book
        bd.mkdir(parents=True, exist_ok=True)
        (bd / "spans.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows), encoding="utf-8")
        secs = [json.loads(l) for l in (PARSED / book / "sections.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        used = {r.get("section_id") for r in new_rows}
        keep = []
        for s in secs:
            s2 = json.loads(json.dumps(s).replace(book, new_book))
            keep.append(s2)
        (bd / "sections.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in keep), encoding="utf-8")
        meta = _meta(book)
        meta.update({"book_id": new_book, "synthetic_edition_of": book, "publisher": "SYNTHETIC second edition", "edition": "synthetic-ed2"})
        (bd / "book.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        for sid, e in elog.edited.items():
            all_edits[sid] = {**e, "orig_span_id": sid.replace(new_book, book, 1), "book": new_book}
    (d / "edits.json").write_text(json.dumps({"seed": a.seed, "edit_frac": a.edit_frac, "picked_chunks": picked, "edits": all_edits}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(json.dumps({"books": list(by_book), "spans": len(all_rows), "edits": len(all_edits), "types": dict(Counter(e["type"] for e in all_edits.values())),
                      "chunks": picked}, ensure_ascii=False, indent=1))


def ingest(a):
    from otter.builder.ingest_books import IngestConfig, ingest_books
    d = KI / "synthetic" / a.name
    tag = f"{a.mode}_{a.rules}"
    out = d / f"ingest_{tag}.sqlite3"
    cfg = IngestConfig(old_db=FULL_DB, new_spans=d / "edition", out=out, old_prov=PROV, cache_dir=d / "cache", report_dir=d / f"report_{tag}", claim_rules=a.rules, cross_kind=(a.rules == "v2"),
                       workers=3, verify_workers=3, judge_workers=3, force=True, lexical_only=(a.mode == "lexical"),
                       confirm_conflicts=(a.mode != "nonconfirm"), reason=f"synthetic second edition ({tag})",
                       book_meta={"*": {"publisher": "SYNTHETIC second edition", "edition": "synthetic-ed2"}})
    if a.mode != "llm" and (d / "report_llm").exists():                     # reuse the extraction/verify caches of the first run
        pass
    s = ingest_books(cfg)
    print(json.dumps({k: s[k] for k in ("verdicts", "handling", "cost", "extraction", "verification", "claim_judge")}, ensure_ascii=False, indent=1, default=str))


def _has(tok: str, text: str) -> bool:
    return re.search(r"(?<![\w.,])" + re.escape(tok) + r"(?![\w]|[.,]\d)", text) is not None


def evaluate(a):
    d = KI / "synthetic" / a.name
    edits = json.loads((d / "edits.json").read_text(encoding="utf-8"))["edits"]
    tag = f"{a.mode}_{a.rules}"
    rdir = d / f"report_{tag}"
    stem = f"ingest_{tag}"
    log = [json.loads(l) for l in (rdir / f"{stem}.changelog.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    con = sqlite3.connect(f"file:{FULL_DB}?mode=ro", uri=True)
    texts = {("fact", r[0]): r[1] for r in con.execute("SELECT assertion_id, text FROM facts")}
    texts.update({("rule", r[0]): r[1] for r in con.execute("SELECT rule_id, text FROM rules")})
    cites = defaultdict(list)
    for kind, iid, sid in con.execute("SELECT item_kind, item_id, span_id FROM evidence_links WHERE item_kind IN ('fact','rule')"):
        cites[sid].append((kind, iid))
    gt, rows = {}, []
    for ev in log:
        if ev["item_kind"] not in ("fact", "rule"):
            continue
        spans_ = ev["span_ids"]
        text = ev["new_text"]
        targets, types = set(), set()
        for s in spans_:
            e = edits.get(s)
            if not e:
                continue
            if _has(e["new"], text):
                for k in cites.get(e["orig_span_id"], []):
                    if _has(e["old"], texts[k]):
                        targets.add(k[1])
                        types.add(e["type"])
        rows.append({"kind": ev["item_kind"], "id": ev["new_item_id"], "verdict": ev["verdict"], "matched": ev["matched_old_id"],
                     "cites_edited": any(s in edits for s in spans_), "gt_targets": sorted(targets), "gt_types": sorted(types),
                     "tier": (ev.get("pair") or {}).get("tier"), "text": text[:200]})
    # NB: changelog span_ids are truncated to 12; items with more spans are rare and counted below
    gt_items = [r for r in rows if r["gt_targets"]]
    det = [r for r in rows if r["verdict"] == "conflicts"]
    tp_strict = [r for r in det if r["gt_targets"] and r["matched"] in r["gt_targets"]]
    tp_len = [r for r in det if r["gt_targets"]]
    fp = [r for r in det if not r["gt_targets"]]
    unedited = [r for r in rows if not r["cites_edited"]]
    res = {"mode": a.mode, "claim_rules": a.rules, "n_new_items_fact_rule": len(rows), "n_edits": len(edits), "gt_conflict_items": len(gt_items),
           "detected_conflicts": len(det), "tp_strict": len(tp_strict), "tp_lenient": len(tp_len), "fp": len(fp),
           "precision_strict": round(len(tp_strict) / max(1, len(det)), 4), "precision_lenient": round(len(tp_len) / max(1, len(det)), 4),
           "recall_strict": round(len(tp_strict) / max(1, len(gt_items)), 4), "recall_lenient": round(len(tp_len) / max(1, len(gt_items)), 4),
           "gt_by_type": dict(Counter(t for r in gt_items for t in r["gt_types"])),
           "detected_gt_by_type": dict(Counter(t for r in tp_len for t in r["gt_types"])),
           "verdicts_gt_items": dict(Counter(r["verdict"] for r in gt_items)),
           "unedited_items": len(unedited), "verdicts_unedited": dict(Counter(r["verdict"] for r in unedited)),
           "false_conflicts_unedited": sum(1 for r in unedited if r["verdict"] == "conflicts"),
           "false_conflict_rate_unedited": round(sum(1 for r in unedited if r["verdict"] == "conflicts") / max(1, len(unedited)), 4),
           "cites_edited_not_gt": len([r for r in rows if r["cites_edited"] and not r["gt_targets"]]),
           "verdicts_edited_not_gt": dict(Counter(r["verdict"] for r in rows if r["cites_edited"] and not r["gt_targets"]))}
    (d / f"eval_{tag}.json").write_text(json.dumps({"summary": res, "gt_items_missed": [r for r in gt_items if r["verdict"] != "conflicts"][:40],
                                                       "false_positives": fp[:40]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("httpx", "openai", "httpcore"):
        logging.getLogger(n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["make", "ingest", "eval"])
    ap.add_argument("--name", default="s1")
    ap.add_argument("--mode", default="llm", choices=["llm", "lexical", "nonconfirm"])
    ap.add_argument("--rules", default="v2", choices=["v1", "v2"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edit-frac", type=float, default=0.35)
    ap.add_argument("--skip-chunks", type=int, default=0, help="skip the N most number-dense chunks (held-out set uses 2)")
    ap.add_argument("--chunks-per-book", type=int, default=2)
    a = ap.parse_args()
    {"make": make, "ingest": ingest, "eval": evaluate}[a.stage](a)


if __name__ == "__main__":
    main()
