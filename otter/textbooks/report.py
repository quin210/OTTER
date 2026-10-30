from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from .catalog import Catalog
from .provenance import OntologyStore

PRIORITY={"physics":"Vật lí","chemistry":"Hóa học","biology":"Sinh học","history":"Lịch sử","geography":"Địa lí","literature":"Ngữ văn"}

def write_report(catalog: Catalog, registry: Path, network_note: str="") -> str:
    books=catalog.load(); queue=registry/"download_queue.csv"
    store=OntologyStore(registry.parent/"ontology.sqlite3")
    entity_count,relation_count=store.materialize_catalog_graph([b.__dict__ for b in books])
    assertion_count=store.db.execute("SELECT count(*) FROM assertions").fetchone()[0]
    store.db.close()
    if not network_note:
        log=registry/"acquisition_log.jsonl"
        if log.exists():
            for line in reversed(log.read_text(encoding="utf-8").splitlines()):
                try: event=json.loads(line)
                except ValueError: continue
                if event.get("event")=="discover":
                    network_note=event.get("navigation_note",""); break
    parsed_root=registry.parent/"parsed"
    unmatched_file=registry/"unmatched_files.jsonl"
    unmatched_count=sum(1 for line in unmatched_file.read_text(encoding="utf-8").splitlines() if line.strip()) if unmatched_file.exists() else 0
    parsed_page_count=sum(len(p.read_text(encoding="utf-8").splitlines()) for p in parsed_root.glob("*/pages.jsonl")) if parsed_root.exists() else 0
    parsed_section_count=sum(len(p.read_text(encoding="utf-8").splitlines()) for p in parsed_root.glob("*/sections.jsonl")) if parsed_root.exists() else 0
    by_grade=Counter(b.grade for b in books); by_subject=Counter(b.subject_normalized for b in books); by_material=Counter(b.material_type for b in books); by_access=Counter(b.access_status for b in books); by_download=Counter(b.download_status for b in books)
    expected={(g,s) for g in (10,11,12) for s in PRIORITY}
    covered={(b.grade,b.subject_normalized) for b in books}
    missing=sorted(expected-covered)
    user_supplied=sum(b.download_status=="user_supplied" for b in books)
    queued=sum(not b.local_file and b.download_status not in {"downloaded","checksum_verified"} for b in books)
    scanned=sum(1 for p in parsed_root.glob("*/pages.jsonl") for line in p.read_text(encoding="utf-8").splitlines() if '"scanned_pdf_no_text"' in line) if parsed_root.exists() else 0
    lines=["# Textbook discovery and ingestion report", "",f"Generated: {datetime.now(timezone.utc).isoformat()}","", "## Discovery", "",f"Observed official item pages: **{len(books)}**.","The public index exposed official detail-page URLs. Direct portal/robots access through this run's browser and shell was unavailable; listing pagination is therefore incomplete and no absence claims are made.", "",network_note,"", "Grade counts: "+", ".join(f"{k}: {v}" for k,v in sorted(by_grade.items())),"", "Subject counts: "+", ".join(f"{PRIORITY.get(k,k)}: {v}" for k,v in sorted(by_subject.items())),"", "Material type counts: "+", ".join(f"{k}: {v}" for k,v in sorted(by_material.items())),"", "Access status counts: "+", ".join(f"{k}: {v}" for k,v in sorted(by_access.items())),"", "Download status counts: "+", ".join(f"{k}: {v}" for k,v in sorted(by_download.items())),"", "Missing priority grade/subject combinations: "+(", ".join(f"grade {g} {PRIORITY[s]}" for g,s in missing) if missing else "none among the observed entries; listing completeness is unverified."), "", "## Acquisition", "",f"Downloaded by automation: **{sum(b.download_status in {'downloaded','checksum_verified'} for b in books)}**.",f"User-supplied cataloged files: **{user_supplied}** (the filenames indicate sample copies).",f"Unmatched local file assets needing catalog metadata review: **{unmatched_count}**; see `unmatched_files.jsonl`.",f"Still queued for manual/rights action: **{queued}**.","No automated download was performed. Locally supplied copies are not redistributed; their source-use authorization has not been independently checked.","", "## Parsing and ontology integration", "",f"Bibliographic graph: **{entity_count} entities, {relation_count} typed relations** (book/grade/subject/series metadata only). Content assertions: **{assertion_count}**. Parsed source pages: **{parsed_page_count}**; detected sections: **{parsed_section_count}**; scanned pages needing OCR: **{scanned}**. Page locators are stored, but scanned pages with no extractable text do not support ontology assertions. Printed page labels remain unverified unless a reviewer compares them with the rendered book.","", "## Navigation limitation", "",network_note,"",f"Manual acquisition queue: `{queue.as_posix()}`.","The official catalog notice states free for teaching/learning, not commercial; that does not itself establish automated-download permission.",""]
    text="\n".join(lines); (registry/"validation_report.md").write_text(text,encoding="utf-8"); return text
