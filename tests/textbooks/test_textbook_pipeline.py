import json
import tempfile
import unittest
from pathlib import Path

from otter.textbooks.acquisition import import_manual
from otter.textbooks.catalog import Book, Catalog, normalize_subject
from otter.textbooks.parser import parse_book
from otter.textbooks.provenance import CitationResolver, OntologyStore
from otter.textbooks.source_adapters.nxbgd_portal import bootstrap_catalog

class TextbookPipelineTests(unittest.TestCase):
    def test_aliases_and_verified_seed(self):
        self.assertEqual(normalize_subject("Vật lý"),"physics")
        self.assertEqual(normalize_subject("Địa lí"),"geography")
        books=bootstrap_catalog()
        self.assertEqual(len(books),21)
        self.assertTrue(all("/chi-tiet-sach/" in b.item_page_url for b in books))
        self.assertEqual(sum(b.grade==10 and b.subject_normalized=="literature" for b in books),2)

    def test_manual_parse_store_citation_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); reg=Catalog(root/"registry"); book=Book.observed(title="Synthetic Biology 11",subject="Sinh học",grade=11,item_url="https://taphuan.nxbgd.vn/tap-huan/chi-tiet-sach/synthetic-11",evidence_url="https://taphuan.nxbgd.vn/tap-huan/chi-tiet-sach/synthetic-11")
            reg.save([book]); source=root/"lesson.txt"; source.write_text("Bài 1: Tế bào\nTế bào là đơn vị cấu trúc và chức năng của cơ thể sống.\n",encoding="utf-8")
            imported=import_manual(reg,book.book_id,source,root/"raw",root/"registry")
            pages,sections=parse_book(imported.__dict__,Path(imported.local_file),root/"parsed",imported.sha256)
            self.assertEqual((pages,sections),(1,1))
            page=json.loads((root/"parsed"/"pages.jsonl").read_text().splitlines()[0])
            self.assertIsNone(page["printed_page_label"])
            self.assertFalse(page["printed_page_verified"])
            store=OntologyStore(root/"ontology.sqlite3");store.ingest(imported.__dict__,root/"parsed"/"pages.jsonl",root/"parsed"/"sections.jsonl")
            store.add_assertion("a1","Tế bào là đơn vị cấu trúc.",[page["span_id"]])
            verified=store.verify_evidence("a1",[page["span_id"]])
            self.assertEqual(verified["source_link_status"],"verified")
            self.assertEqual(verified["semantic_support"],"needs_review")
            self.assertEqual(verified["status"],"candidate_extraction")
            self.assertEqual(store.verify_evidence("a1",["forged"])["source_link_status"],"mismatch_or_missing")
            citation=CitationResolver(store).resolve(page["span_id"])
            self.assertIsNone(citation["page_number"])
            self.assertEqual(citation["page_type"],"source")
            self.assertEqual(citation["lesson"],"Bài 1: Tế bào")
            self.assertEqual(citation["grade"],11)
            self.assertEqual(len(store.retrieve_source([page["span_id"]])),1)
            with self.assertRaises(ValueError): store.add_assertion("bad","claim",["bogus-span"])
            store.db.close()

    def test_manual_import_requires_catalog_and_valid_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cat=Catalog(root/"registry"); source=root/"bad.pdf";source.write_text("not a PDF")
            with self.assertRaises(ValueError): import_manual(cat,"unknown",source,root/"raw",root/"registry")

    def test_catalog_metadata_graph_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            store=OntologyStore(Path(td)/"ontology.sqlite3")
            book={"book_id":"b1","title":"Book One","grade":10,"subject_original":"Vật lí","subject_normalized":"physics","series":None,"item_page_url":"https://taphuan.nxbgd.vn/tap-huan/chi-tiet-sach/book-one"}
            self.assertEqual(store.materialize_catalog_graph([book]),(3,2))
            self.assertEqual(store.materialize_catalog_graph([book]),(3,2))
            self.assertEqual(store.db.execute("SELECT count(*) FROM relations").fetchone()[0],2)
            self.assertEqual(store.db.execute("SELECT count(*) FROM assertions").fetchone()[0],0)
            store.db.close()

if __name__=="__main__": unittest.main()
