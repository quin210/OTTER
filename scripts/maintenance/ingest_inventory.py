"""Inventory of the 41 OCRed books of corpus v1: which could serve as natural "new books" for `ingest-books`
(other publisher / other edition / not in the 18-book core)?  Read-only; writes runs/knowledge_ingest/natural_inventory.json."""
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
man = json.loads((ROOT / "runs/corpus/CORPUS_V1.json").read_text())
con = sqlite3.connect(f"file:{ROOT / 'data/textbooks/ontology.sqlite3'}?mode=ro", uri=True)
rows = []
for bid, mj in con.execute("SELECT book_id, metadata_json FROM books ORDER BY book_id"):
    m = json.loads(mj)
    fn = m.get("filename_locator") or Path(m.get("local_file") or "").name
    kind = ("core (nxbgd_, in the 18-book evaluation core)" if bid.startswith("nxbgd_") else
            "study-topic booklet (Chuyên đề học tập)" if "Chuyên đề" in (fn or "") else
            "integrated / lower-secondary (KHTN, Lịch sử và Địa lí, Ngữ văn 8-9)")
    rows.append({"book_id": bid, "file": fn, "kind": kind, "subject": m.get("subject_normalized"), "grade": m.get("grade"),
                 "publisher_in_metadata": m.get("publisher"), "series_in_metadata": m.get("series"), "edition_year": m.get("edition_year"),
                 "metadata_review_needed": bool(m.get("metadata_review_needed")), "spans": man["per_book"].get(bid),
                 "sample_copy_by_filename": bool(re.search(r"-mau\.pdf$", fn or ""))})
summary = {"books": len(rows), "core": sum(r["kind"].startswith("core") for r in rows),
           "other": sum(not r["kind"].startswith("core") for r in rows),
           "other_publisher_known": sum(1 for r in rows if not r["kind"].startswith("core") and r["publisher_in_metadata"]),
           "earlier_edition_known": 0,
           "all_files_are_sample_copies": all(r["sample_copy_by_filename"] for r in rows)}
out = ROOT / "runs/knowledge_ingest/natural_inventory.json"
out.write_text(json.dumps({"summary": summary, "books": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
for r in rows:
    if not r["kind"].startswith("core"):
        print(r["book_id"], r["kind"][:22], r["subject"], r["grade"], r["spans"], r["file"])
