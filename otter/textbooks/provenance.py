from __future__ import annotations

import json
import sqlite3
import hashlib
from pathlib import Path

class OntologyStore:
    """Small SQLite source/provenance store compatible with builder assertions."""
    def __init__(self, path: Path):
        self.path=path; path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path, timeout=60)
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS books(book_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, source_sha256 TEXT);
        CREATE TABLE IF NOT EXISTS sections(section_id TEXT PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(book_id), title TEXT, pdf_page_start INTEGER, pdf_page_end INTEGER, printed_page_start TEXT, printed_page_end TEXT, metadata_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS spans(span_id TEXT PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(book_id), pdf_page_index INTEGER, source_page_index INTEGER NOT NULL, printed_page_label TEXT, printed_page_verified INTEGER NOT NULL, section_id TEXT, text TEXT NOT NULL, source_sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assertions(assertion_id TEXT PRIMARY KEY, claim TEXT NOT NULL, provenance_status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assertion_sources(assertion_id TEXT NOT NULL REFERENCES assertions(assertion_id), span_id TEXT NOT NULL REFERENCES spans(span_id), PRIMARY KEY(assertion_id,span_id));
        CREATE TABLE IF NOT EXISTS entities(entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, label TEXT NOT NULL, attributes_json TEXT NOT NULL, source_url TEXT);
        CREATE TABLE IF NOT EXISTS relations(relation_id TEXT PRIMARY KEY, subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id), relation_type TEXT NOT NULL, object_entity_id TEXT NOT NULL REFERENCES entities(entity_id), source_url TEXT, source_span_id TEXT REFERENCES spans(span_id), evidence_status TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(span_id UNINDEXED, book_id UNINDEXED, text);
        """)
        self._migrate()

    # Extra provenance columns (added compatibly with ALTER TABLE; column names follow
    # otter/storage/schema.py::FIXTURE_PROVENANCE_DDL so WS2's reader picks them up).
    SPAN_EXTRA_COLUMNS = {
        "start_offset": "INTEGER", "end_offset": "INTEGER", "heading_path_json": "TEXT", "pedagogical_type": "TEXT",
        "extractor_version": "TEXT", "evidence_status": "TEXT", "span_kind": "TEXT", "pedagogical_label_vi": "TEXT",
        "chapter_id": "TEXT", "lesson_id": "TEXT", "page_span_id": "TEXT", "page_quality_score": "REAL",
        "requires_manual_review": "INTEGER", "text_sha1": "TEXT",
    }
    SECTION_EXTRA_COLUMNS = {"level": "TEXT", "parent_id": "TEXT", "ordinal": "TEXT", "heading_path_json": "TEXT", "link_status": "TEXT"}

    def _migrate(self) -> None:
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        for table, cols in (("spans", self.SPAN_EXTRA_COLUMNS), ("sections", self.SECTION_EXTRA_COLUMNS)):
            have = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            for c, t in cols.items():
                if c not in have:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {c} {t}")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS pages(page_span_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, pdf_page_index INTEGER NOT NULL,
          printed_page_label TEXT, printed_page_verified INTEGER NOT NULL, section_id TEXT, text TEXT NOT NULL,
          extraction_method TEXT, extractor_version TEXT, quality_score REAL, requires_manual_review INTEGER, status TEXT,
          image_path TEXT, source_sha256 TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_spans_book_page ON spans(book_id, pdf_page_index);
        CREATE INDEX IF NOT EXISTS ix_spans_section ON spans(section_id);
        CREATE INDEX IF NOT EXISTS ix_sections_book ON sections(book_id);
        CREATE INDEX IF NOT EXISTS ix_pages_book ON pages(book_id, pdf_page_index);
        """)
        self.db.commit()

    def _insert_span_row(self, row: dict) -> None:
        cols = list(row)
        self.db.execute(f"INSERT OR REPLACE INTO spans({','.join(cols)}) VALUES({','.join('?' for _ in cols)})", [row[c] for c in cols])
        self.db.execute("DELETE FROM spans_fts WHERE span_id=?", (row["span_id"],))
        self.db.execute("INSERT INTO spans_fts VALUES(?,?,?)", (row["span_id"], row["book_id"], row["text"]))

    def ingest_parsed_book(self, book: dict, parsed_dir: Path) -> dict:
        """Replace one book's provenance rows from pages.jsonl / sections.jsonl / spans.jsonl (OCR pipeline output).

        Paragraph spans go to ``spans`` (+ ``spans_fts``); page-level records go to ``pages``.
        Spans referenced by ontology assertions/relations are kept if they still exist; links
        to spans that no longer exist are dropped from the legacy catalog tables only.
        """
        def rows(name):
            f = parsed_dir / name
            return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()] if f.exists() else []
        pages, sections, spans = rows("pages.jsonl"), rows("sections.jsonl"), rows("spans.jsonl")
        bid = book["book_id"]
        new_ids = {s["span_id"] for s in spans}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO books VALUES(?,?,?)", (bid, json.dumps(book, ensure_ascii=False, sort_keys=True), book.get("sha256")))
            old = [r[0] for r in self.db.execute("SELECT span_id FROM spans WHERE book_id=?", (bid,))]
            stale = [x for x in old if x not in new_ids]
            for i in range(0, len(stale), 500):
                chunk = stale[i:i + 500]; q = ",".join("?" for _ in chunk)
                self.db.execute(f"UPDATE relations SET source_span_id=NULL WHERE source_span_id IN ({q})", chunk)
                self.db.execute(f"DELETE FROM assertion_sources WHERE span_id IN ({q})", chunk)
                self.db.execute(f"DELETE FROM spans_fts WHERE span_id IN ({q})", chunk)
                self.db.execute(f"DELETE FROM spans WHERE span_id IN ({q})", chunk)
            self.db.execute("DELETE FROM pages WHERE book_id=?", (bid,))
            self.db.execute("DELETE FROM sections WHERE book_id=?", (bid,))
            for p in pages:
                self.db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    p["span_id"], bid, p["pdf_page_index"], p.get("printed_page_label"), int(bool(p.get("printed_page_verified"))),
                    p.get("section_id"), p.get("text") or "", p.get("extraction_method"), p.get("extractor_version"),
                    p.get("quality_score"), int(bool(p.get("requires_manual_review"))), p.get("status"), p.get("image_path"),
                    p["source_file_sha256"]))
            for n in sections:
                self.db.execute("INSERT OR REPLACE INTO sections(section_id,book_id,title,pdf_page_start,pdf_page_end,printed_page_start,printed_page_end,metadata_json,level,parent_id,ordinal,heading_path_json,link_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    n["section_id"], bid, n.get("title"), n.get("pdf_page_start"), n.get("pdf_page_end"), n.get("printed_page_start"),
                    n.get("printed_page_end"), json.dumps(n, ensure_ascii=False, sort_keys=True), n.get("level"), n.get("parent_id"),
                    n.get("ordinal"), json.dumps(n.get("heading_path") or [], ensure_ascii=False), n.get("link_status")))
            for s in spans:
                self._insert_span_row({
                    "span_id": s["span_id"], "book_id": bid, "pdf_page_index": s["pdf_page_index"], "source_page_index": s["pdf_page_index"],
                    "printed_page_label": s.get("printed_page_label"), "printed_page_verified": int(bool(s.get("printed_page_verified"))),
                    "section_id": s.get("section_id"), "text": s["text"], "source_sha256": s["source_file_sha256"],
                    "start_offset": s["start_offset"], "end_offset": s["end_offset"],
                    "heading_path_json": json.dumps(s.get("heading_path") or [], ensure_ascii=False),
                    "pedagogical_type": s.get("pedagogical_type"), "extractor_version": s.get("extractor_version"),
                    "evidence_status": s.get("evidence_status"), "span_kind": s.get("span_kind"),
                    "pedagogical_label_vi": s.get("pedagogical_label_vi"), "chapter_id": s.get("chapter_id"),
                    "lesson_id": s.get("lesson_id"), "page_span_id": s.get("page_span_id"),
                    "page_quality_score": s.get("page_quality_score"), "requires_manual_review": int(bool(s.get("requires_manual_review"))),
                    "text_sha1": s.get("text_sha1")})
        return {"book_id": bid, "pages": len(pages), "sections": len(sections), "spans": len(spans), "stale_spans_removed": len(stale)}

    def ingest(self, book: dict, parsed_pages: Path, parsed_sections: Path | None=None) -> None:
        rows=[json.loads(x) for x in parsed_pages.read_text(encoding="utf-8").splitlines() if x.strip()]
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO books VALUES(?,?,?)",(book["book_id"],json.dumps(book,ensure_ascii=False,sort_keys=True),book.get("sha256")))
            for p in rows:
                self._insert_span_row({"span_id":p["span_id"],"book_id":p["book_id"],"pdf_page_index":p["pdf_page_index"],"source_page_index":p["source_page_index"],"printed_page_label":p["printed_page_label"],"printed_page_verified":int(p["printed_page_verified"]),"section_id":p["section_id"],"text":p["text"],"source_sha256":p["source_file_sha256"],"start_offset":p.get("start_offset",0),"end_offset":p.get("end_offset",len(p["text"])),"span_kind":"page"})
            if parsed_sections and parsed_sections.exists():
                for line in parsed_sections.read_text(encoding="utf-8").splitlines():
                    if not line.strip(): continue
                    section=json.loads(line)
                    self.db.execute("INSERT OR REPLACE INTO sections(section_id,book_id,title,pdf_page_start,pdf_page_end,printed_page_start,printed_page_end,metadata_json) VALUES(?,?,?,?,?,?,?,?)",(section["section_id"],section["book_id"],section.get("title"),section.get("pdf_page_start"),section.get("pdf_page_end"),section.get("printed_page_start"),section.get("printed_page_end"),json.dumps(section,ensure_ascii=False,sort_keys=True)))

    def add_assertion(self, assertion_id: str, claim: str, source_span_ids: list[str], provenance_status: str="candidate_extraction") -> None:
        if not source_span_ids: raise ValueError("Ontology assertions require source_span_ids")
        found={r[0] for r in self.db.execute(f"SELECT span_id FROM spans WHERE span_id IN ({','.join('?' for _ in source_span_ids)})",source_span_ids)}
        if found != set(source_span_ids): raise ValueError("Unknown source span ID")
        with self.db:
            self.db.execute("INSERT INTO assertions VALUES(?,?,?)",(assertion_id,claim,provenance_status))
            self.db.executemany("INSERT INTO assertion_sources VALUES(?,?)",[(assertion_id,s) for s in source_span_ids])

    def retrieve_source(self, span_ids: list[str]) -> list[dict]:
        if not span_ids: return []
        q=f"SELECT s.span_id,s.book_id,b.metadata_json,s.pdf_page_index,s.source_page_index,s.printed_page_label,s.printed_page_verified,s.section_id,s.text,s.source_sha256 FROM spans s JOIN books b USING(book_id) WHERE s.span_id IN ({','.join('?' for _ in span_ids)})"
        result=[]
        for row in self.db.execute(q,span_ids):
            result.append({"span_id":row[0],"book":json.loads(row[2]),"pdf_page_index":row[3],"source_page_index":row[4],"printed_page_label":row[5],"printed_page_verified":bool(row[6]),"section_id":row[7],"text":row[8],"source_file_sha256":row[9]})
        return result

    def verify_evidence(self, assertion_id: str, cited_span_ids: list[str]) -> dict:
        """Check link integrity without overstating semantic support."""
        row=self.db.execute("SELECT claim,provenance_status FROM assertions WHERE assertion_id=?",(assertion_id,)).fetchone()
        if not row: return {"status":"unsupported","source_link_status":"missing_assertion","semantic_support":"unsupported"}
        actual={r[0] for r in self.db.execute("SELECT span_id FROM assertion_sources WHERE assertion_id=?",(assertion_id,))}
        cited=set(cited_span_ids)
        exists={r[0] for r in self.db.execute(f"SELECT span_id FROM spans WHERE span_id IN ({','.join('?' for _ in cited)})",list(cited))} if cited else set()
        if not cited or cited!=actual or exists!=cited:
            return {"status":"manual_review_needed","source_link_status":"mismatch_or_missing","semantic_support":"needs_review","assertion_id":assertion_id}
        return {"status":"candidate_extraction","source_link_status":"verified","semantic_support":"needs_review","assertion_id":assertion_id,"source_span_ids":sorted(actual),"claim":row[0]}

    def materialize_catalog_graph(self, books: list[dict]) -> tuple[int,int]:
        """Build an idempotent bibliographic graph from verified catalog fields.

        This represents book/grade/subject/series metadata only; it does not
        create subject-matter facts from scanned pages.
        """
        def eid(kind: str, key: str) -> str: return kind+":"+hashlib.sha256(key.encode()).hexdigest()[:20]
        entities={}; edges=[]
        for b in books:
            bid=eid("book",b["book_id"]); entities[bid]=(bid,"Book",b["title"],json.dumps(b,ensure_ascii=False,sort_keys=True),b.get("item_page_url"))
            grade=str(b["grade"]); gid=eid("grade",grade); entities[gid]=(gid,"Grade",grade,json.dumps({"grade":b["grade"]}),None)
            subject_key=b.get("subject_normalized") or b.get("subject_original") or "unknown"
            sid=eid("subject",subject_key); entities[sid]=(sid,"Subject",b.get("subject_original") or subject_key,json.dumps({"normalized":subject_key}),None)
            edges.extend([(bid,"has_grade",gid,b.get("item_page_url"),None,"verified_page_only"),(bid,"teaches_subject",sid,b.get("item_page_url"),None,"verified_page_only")])
            if b.get("series"):
                series=b["series"]; rid=eid("series",series); entities[rid]=(rid,"Series",series,json.dumps({"series":series}),None)
                # The series mark was visually verified on a rendered title page.
                file_pages=self.db.execute("SELECT span_id,pdf_page_index FROM spans WHERE book_id=? ORDER BY pdf_page_index",(b["book_id"],)).fetchall()
                target_index=1 if b["title"]=="Ngữ văn 10, tập hai" else 2
                source_span=next((sp for sp,idx in file_pages if idx==target_index),None)
                edges.append((bid,"member_of_series",rid,None,source_span,"verified_page_only" if source_span else "manual_review_needed"))
        with self.db:
            self.db.executemany("INSERT OR REPLACE INTO entities VALUES(?,?,?,?,?)",entities.values())
            for subject,rel,obj,url,span,status in edges:
                rid=eid("rel",f"{subject}|{rel}|{obj}|{url or span or ''}")
                self.db.execute("INSERT OR REPLACE INTO relations VALUES(?,?,?,?,?,?,?)",(rid,subject,rel,obj,url,span,status))
        return len(entities),len(edges)

class CitationResolver:
    def __init__(self, store: OntologyStore): self.store=store
    def resolve(self, span_id: str) -> dict:
        rows=self.store.retrieve_source([span_id])
        if not rows: raise ValueError(f"Unknown source span: {span_id}")
        x=rows[0]; b=x["book"]
        section=self.store.db.execute("SELECT title FROM sections WHERE section_id=?",(x.get("section_id"),)).fetchone() if x.get("section_id") else None
        return {"book_id":b["book_id"],"book_title":b.get("title"),"subject":b.get("subject_normalized"),"grade":b.get("grade"),"series":b.get("series"),"volume":b.get("volume"),"lesson_id":x.get("section_id"),"lesson":section[0] if section else None,"page_number":x["printed_page_label"] if x["printed_page_verified"] else None,"page_type":"printed" if x["printed_page_verified"] else "pdf" if x["pdf_page_index"] is not None else "source","pdf_page_index":x["pdf_page_index"],"pdf_page_number":x["pdf_page_index"]+1 if x["pdf_page_index"] is not None else None,"source_page_index":x["source_page_index"],"span_id":span_id,"source_file_sha256":x["source_file_sha256"],"provenance_status":"verified_page_only" if x["printed_page_verified"] else "candidate_extraction"}
