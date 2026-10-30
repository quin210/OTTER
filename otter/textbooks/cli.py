from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path

from .acquisition import download_permitted, import_manual
from .catalog import Catalog, Book, normalize_subject
from .parser import parse_book
from .provenance import OntologyStore
from .report import write_report
from .source_adapters.nxbgd_portal import PUBLIC_LISTINGS, bootstrap_catalog, discover_public_listing

ROOT=Path.cwd(); DATA=ROOT/"data/textbooks"; REG=DATA/"registry"; CAT=Catalog(REG)

def discover():
    # Always retain verified observations. Respect robots.txt: if it cannot be
    # read, do not proceed to automated listing traversal.
    books=CAT.load(); existing={b.book_id for b in books}; books += [b for b in bootstrap_catalog() if b.book_id not in existing]
    note="robots.txt could not be obtained in this environment; automated listing traversal was not attempted."
    try:
        import urllib.request
        robots_url="https://taphuan.nxbgd.vn/robots.txt"
        with urllib.request.urlopen(robots_url,timeout=8) as r:
            robots_text=r.read(200_000).decode("utf-8","replace")
        rp=urllib.robotparser.RobotFileParser();rp.set_url(robots_url);rp.parse(robots_text.splitlines())
        if any(not rp.can_fetch("OTTER textbook catalog",url) for url in PUBLIC_LISTINGS.values()): note="robots.txt disallows at least one requested listing path; no listing traversal was performed."
        else:
            # The requested listing paths are public and the catalog adapter only
            # follows visible detail links. Keep requests slow and bounded.
            found=[]
            urls={b.item_page_url for b in books}
            for grade,url in PUBLIC_LISTINGS.items():
                visible=discover_public_listing(url); found.extend(visible)
                for item_url,label in visible:
                    m=re.search(r"\b(10|11|12)\b",label)
                    if not m or int(m.group(1)) != grade or item_url in urls: continue
                    subject=next((s for s in ("Vật lí","Vật lý","Hóa học","Hoá học","Sinh học","Lịch sử","Địa lí","Địa lý","Ngữ văn") if s.casefold() in label.casefold()),None)
                    if not subject: continue
                    volume="tập một" if "tập một" in label.casefold() else "tập hai" if "tập hai" in label.casefold() else None
                    material="specialized_topic" if "chuyên đề học tập" in label.casefold() else "student_textbook"
                    b=Book.observed(title=label,subject=subject,grade=grade,item_url=item_url,evidence_url=item_url,series=None,volume=volume,material_type=material)
                    books.append(b); urls.add(item_url)
                time.sleep(2)
            note=f"Visited {len(PUBLIC_LISTINGS)} public listing pages and observed {len(found)} visible item links. Dynamic pages may require browser rendering; indexed bootstrap observations are retained."
    except Exception as e:
        note=f"robots/listing navigation unavailable ({type(e).__name__}: {e}); no listing traversal performed. Bootstrap observations remain in registry."
    CAT.save(books)
    with (REG/"acquisition_log.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps({"event":"discover","timestamp":datetime.now().astimezone().isoformat(),"observed_items":len(books),"navigation_note":note},ensure_ascii=False)+"\n")
    write_report(CAT,REG,note)
    print(f"Cataloged {len(books)} verified official item pages. {note}")

def main(argv=None):
    p=argparse.ArgumentParser(prog="otter.textbooks"); sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("discover")
    d=sub.add_parser("download-permitted"); d.add_argument("--book-id");d.add_argument("--max-files",type=int,default=1);d.add_argument("--delay-seconds",type=float,default=2)
    i=sub.add_parser("import-manual");i.add_argument("--book-id",required=True);i.add_argument("--file",required=True)
    q=sub.add_parser("parse");q.add_argument("--book-id",required=True)
    sub.add_parser("validate");sub.add_parser("report")
    a=p.parse_args(argv)
    if a.cmd=="discover": discover()
    elif a.cmd=="download-permitted":
        if a.max_files < 1: raise SystemExit("--max-files must be positive")
        candidates=[b for b in CAT.load() if (not a.book_id or b.book_id==a.book_id) and b.official_download_url and b.download_permission_verified]
        for b in candidates[:a.max_files]: download_permitted(CAT,b.book_id,DATA/"raw",REG,delay_seconds=a.delay_seconds)
        if not candidates:
            with (REG/"acquisition_log.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"event":"download_permitted","timestamp":datetime.now().astimezone().isoformat(),"status":"no_eligible_books","reason":"No record had both an observed standard download URL and verified permission; no request made."},ensure_ascii=False)+"\n")
            print("No cataloged books have both an observed standard download URL and verified permission; nothing downloaded.")
    elif a.cmd=="import-manual":
        b=import_manual(CAT,a.book_id,Path(a.file),DATA/"raw",REG);write_report(CAT,REG);print(json.dumps({"book_id":b.book_id,"file":b.local_file,"sha256":b.sha256,"file_type":b.file_type},ensure_ascii=False))
    elif a.cmd=="parse":
        books={b.book_id:b for b in CAT.load()}
        if a.book_id not in books: raise SystemExit("Unknown book_id")
        b=books[a.book_id]
        if not b.local_file: raise SystemExit("Book has no imported/authorized local_file")
        path=Path(b.local_file); sha=b.sha256 or hashlib.sha256(path.read_bytes()).hexdigest()
        pages,sections=parse_book(b.__dict__,path,DATA/"parsed"/b.book_id,sha)
        store=OntologyStore(DATA/"ontology.sqlite3");store.ingest(b.__dict__,DATA/"parsed"/b.book_id/"pages.jsonl",DATA/"parsed"/b.book_id/"sections.jsonl");store.db.close()
        b.page_count=pages; CAT.save(books.values()); write_report(CAT,REG); print(f"Parsed {pages} pages and {sections} sections")
    elif a.cmd in {"validate","report"}:
        text=write_report(CAT,REG)
        print(text if a.cmd=="report" else f"Report written to {REG/'validation_report.md'}")

if __name__=="__main__": main()
