#!/usr/bin/env python
"""Structure + paragraph spans for every OCR'd book, loaded into the provenance DB.

  .venv/bin/python scripts/ocr/build_spans.py                 # all books with pages.jsonl (only stale ones)
  .venv/bin/python scripts/ocr/build_spans.py --force         # rebuild everything
  .venv/bin/python scripts/ocr/build_spans.py --watch         # poll while the OCR job runs

For each data/textbooks/parsed/<book_id>/pages.jsonl: write sections.jsonl + spans.jsonl
(otter.textbooks.structure.process_book) and replace that book's rows in the WS1 tables
books / sections / spans / spans_fts / pages of data/textbooks/ontology.sqlite3 (WAL; WS2
tables in the same file are never touched). Unmatched local files (``local_<sha10>``) get
book metadata only from their OCR'd cover/title pages (otter.textbooks.inbox), flagged
``metadata_review_needed``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.textbooks.gemini_import import import_book  # noqa: E402
from otter.textbooks.inbox import local_book_record  # noqa: E402
from otter.textbooks.pages import assemble_book_pages, load_jobs  # noqa: E402
from otter.textbooks.provenance import OntologyStore  # noqa: E402
from otter.textbooks.structure import process_book  # noqa: E402

PARSED = ROOT / "data/textbooks/parsed"
REG = ROOT / "data/textbooks/registry/books.jsonl"


def registry() -> dict[str, dict]:
    out = {}
    for line in REG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            b = json.loads(line); out[b["book_id"]] = b
    return out


def stale(d: Path) -> bool:
    p, s = d / "pages.jsonl", d / "spans.jsonl"
    return p.exists() and (not s.exists() or s.stat().st_mtime < p.stat().st_mtime - 1e-3 or not (d / ".spans_loaded").exists())


def build_one(store: OntologyStore, d: Path, reg: dict, jobs: dict) -> dict:
    book_id = d.name
    if book_id in reg:
        book = dict(reg[book_id])
    else:
        job = jobs.get(book_id, {})
        book = local_book_record(book_id, d, job)
        (d / "book.json").write_text(json.dumps(book, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    t0 = time.time()
    n_nodes, n_spans = process_book(d, book.get("title"))
    stats = store.ingest_parsed_book(book, d)
    (d / ".spans_loaded").write_text(json.dumps(stats | {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), encoding="utf-8")
    os.utime(d / "spans.jsonl")
    stats.update({"nodes": n_nodes, "seconds": round(time.time() - t0, 2), "title": book.get("title")})
    return stats


def summarize(store: OntologyStore) -> dict:
    db = store.db
    per = {}
    for bid, in db.execute("SELECT book_id FROM books ORDER BY book_id"):
        lv = dict(db.execute("SELECT level, COUNT(*) FROM sections WHERE book_id=? GROUP BY level", (bid,)).fetchall())
        n_sp = db.execute("SELECT COUNT(*) FROM spans WHERE book_id=?", (bid,)).fetchone()[0]
        n_pg = db.execute("SELECT COUNT(*) FROM pages WHERE book_id=?", (bid,)).fetchone()[0]
        unres = db.execute("SELECT COUNT(*) FROM sections WHERE book_id=? AND link_status='unresolved'", (bid,)).fetchone()[0]
        per[bid] = {"pages": n_pg, "spans": n_sp, "sections_by_level": lv, "unresolved_links": unres}
    tot = {"books": len(per), "pages": sum(v["pages"] for v in per.values()), "spans": sum(v["spans"] for v in per.values())}
    return {"totals": tot, "books": per}


def run(a) -> int:
    reg = registry()
    jobs = {j["book_id"]: j for j in load_jobs(ROOT, "all")}
    wanted = None if a.books == "all" else set(a.books.split(","))
    store = OntologyStore(ROOT / a.db)
    done = 0
    for d in sorted(PARSED.iterdir()):
        if not d.is_dir() or not (d / "pages.jsonl").exists() or (wanted and d.name not in wanted):
            continue
        job = jobs.get(d.name)
        if job and a.gemini_select:
            gs = import_book(d, d.name, job.get("sha256"))
            if gs["imported"] or a.reassemble:
                n = assemble_book_pages(ROOT, job, use_gemini=True)
                print(json.dumps({"reassembled": d.name, "pages": n, "gemini": gs}), flush=True)
        elif job and a.reassemble:
            assemble_book_pages(ROOT, job)
        if not a.force and not stale(d):
            continue
        try:
            st = build_one(store, d, reg, jobs)
            print(json.dumps(st, ensure_ascii=False), flush=True); done += 1
        except Exception as e:  # keep going
            print(f"[error] {d.name}: {type(e).__name__}: {e}", flush=True)
    if done:
        out = ROOT / "runs/ocr/spans_summary.json"; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summarize(store), ensure_ascii=False, indent=1), encoding="utf-8")
    return done


def ocr_running() -> bool:
    pidf = ROOT / "runs/pids/ocr_run.pid"
    try:
        os.kill(int(pidf.read_text().strip()), 0); return True
    except (OSError, ValueError):
        return False


def main():
    if "--allow-corpus-change" not in sys.argv:
        sys.exit("Span corpus v1 is FROZEN (runs/corpus/CORPUS_V1.json, docs/PARALLEL_PLAN.md 'Corpus freeze'): rebuilding "
                 "spans/span ids/DB rows needs coordinator approval. Re-run with --allow-corpus-change only if approved.")
    sys.argv.remove("--allow-corpus-change")
    p = argparse.ArgumentParser()
    p.add_argument("--books", default="all")
    p.add_argument("--db", default="data/textbooks/ontology.sqlite3")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reassemble", action="store_true", help="re-assemble pages.jsonl from ocr_raw (+Gemini) first")
    p.add_argument("--gemini-select", action="store_true", help="let Gemini pages replace Qwen text (off: comparison only)")
    p.add_argument("--watch", action="store_true", help="repeat every --interval s while the OCR job is running")
    p.add_argument("--interval", type=int, default=60)
    a = p.parse_args()
    run(a)
    while a.watch and ocr_running():
        time.sleep(a.interval)
        a.force = a.reassemble = False
        run(a)
    if a.watch:
        run(a)


if __name__ == "__main__":
    main()
