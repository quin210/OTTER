"""DDL for the WS2-owned ontology tables (spec section 3.5).

Table names avoid the legacy `assertions` / `relations` tables that the WS1
provenance module creates in the shared provenance DB: semantic relations live
in `semantic_relations`, textbook assertions in `facts`.

Every item table stores the full pydantic object in `json` (source of truth for
typed access) plus denormalised columns for filtering.
"""
from __future__ import annotations

ONTOLOGY_DDL = """
CREATE TABLE IF NOT EXISTS build_versions(
  build_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, extractor TEXT, verifier TEXT,
  embedding_model TEXT, config_json TEXT, stats_json TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS concepts(
  concept_id TEXT PRIMARY KEY, label TEXT NOT NULL, label_norm TEXT NOT NULL, concept_type TEXT,
  subject TEXT NOT NULL, grade INTEGER, definition TEXT, source_support TEXT, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_concepts_subj_norm ON concepts(subject, label_norm);
CREATE TABLE IF NOT EXISTS concept_aliases(
  concept_id TEXT NOT NULL, alias TEXT NOT NULL, alias_norm TEXT NOT NULL, subject TEXT NOT NULL,
  PRIMARY KEY(concept_id, alias_norm));
CREATE INDEX IF NOT EXISTS ix_alias_norm ON concept_aliases(subject, alias_norm);
CREATE TABLE IF NOT EXISTS semantic_relations(
  relation_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, relation_type TEXT NOT NULL, object_id TEXT NOT NULL,
  status TEXT, subject TEXT, grade INTEGER, source_support TEXT, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_rel_s ON semantic_relations(subject_id);
CREATE INDEX IF NOT EXISTS ix_rel_o ON semantic_relations(object_id);
CREATE TABLE IF NOT EXISTS facts(
  assertion_id TEXT PRIMARY KEY, text TEXT NOT NULL, subject TEXT NOT NULL, grade INTEGER, status TEXT,
  source_support TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fact_concepts(
  assertion_id TEXT NOT NULL, concept_id TEXT NOT NULL, PRIMARY KEY(assertion_id, concept_id));
CREATE INDEX IF NOT EXISTS ix_fc_c ON fact_concepts(concept_id);
CREATE TABLE IF NOT EXISTS rules(
  rule_id TEXT PRIMARY KEY, subject TEXT NOT NULL, grade INTEGER, rule_type TEXT, status TEXT, formula TEXT,
  text TEXT NOT NULL, source_support TEXT, strict_executable INTEGER NOT NULL DEFAULT 0, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS rule_premises(
  rule_id TEXT NOT NULL, ord INTEGER NOT NULL, predicate TEXT NOT NULL, args_json TEXT, text TEXT, negated INTEGER,
  PRIMARY KEY(rule_id, ord));
CREATE TABLE IF NOT EXISTS rule_guards(
  rule_id TEXT NOT NULL, ord INTEGER NOT NULL, predicate TEXT NOT NULL, args_json TEXT, text TEXT, negated INTEGER,
  PRIMARY KEY(rule_id, ord));
CREATE TABLE IF NOT EXISTS rule_conclusions(
  rule_id TEXT NOT NULL, ord INTEGER NOT NULL, predicate TEXT NOT NULL, args_json TEXT, text TEXT, negated INTEGER,
  PRIMARY KEY(rule_id, ord));
CREATE TABLE IF NOT EXISTS rule_exceptions(
  rule_id TEXT NOT NULL, ord INTEGER NOT NULL, predicate TEXT NOT NULL, args_json TEXT, text TEXT, negated INTEGER,
  PRIMARY KEY(rule_id, ord));
CREATE INDEX IF NOT EXISTS ix_rc_pred ON rule_conclusions(predicate);
CREATE INDEX IF NOT EXISTS ix_rp_pred ON rule_premises(predicate);
CREATE TABLE IF NOT EXISTS rule_concepts(
  rule_id TEXT NOT NULL, concept_id TEXT NOT NULL, PRIMARY KEY(rule_id, concept_id));
CREATE INDEX IF NOT EXISTS ix_rcon_c ON rule_concepts(concept_id);
CREATE TABLE IF NOT EXISTS objectives(
  objective_id TEXT PRIMARY KEY, text TEXT NOT NULL, lesson_section_id TEXT, subject TEXT NOT NULL, grade INTEGER,
  json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pedagogical_links(
  link_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, link_type TEXT NOT NULL, target_id TEXT NOT NULL, json TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_pl_s ON pedagogical_links(source_id);
CREATE INDEX IF NOT EXISTS ix_pl_t ON pedagogical_links(target_id);
CREATE TABLE IF NOT EXISTS evidence_links(
  item_kind TEXT NOT NULL, item_id TEXT NOT NULL, span_id TEXT NOT NULL, PRIMARY KEY(item_kind, item_id, span_id));
CREATE INDEX IF NOT EXISTS ix_ev_span ON evidence_links(span_id);
CREATE TABLE IF NOT EXISTS rejected_items(
  item_kind TEXT NOT NULL, item_key TEXT NOT NULL, reason TEXT NOT NULL, chunk_id TEXT, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings(
  kind TEXT NOT NULL, item_id TEXT NOT NULL, model TEXT NOT NULL, subject TEXT, grade INTEGER, dim INTEGER NOT NULL,
  vec BLOB NOT NULL, PRIMARY KEY(kind, item_id, model));
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
  kind UNINDEXED, item_id UNINDEXED, subject UNINDEXED, grade UNINDEXED, text, tokenize="unicode61 remove_diacritics 0");
"""

ONTOLOGY_TABLES = ("build_versions", "concepts", "concept_aliases", "semantic_relations", "facts", "fact_concepts",
                   "rules", "rule_premises", "rule_guards", "rule_conclusions", "rule_exceptions", "rule_concepts",
                   "objectives", "pedagogical_links", "evidence_links", "rejected_items", "embeddings",
                   "knowledge_fts")

# Provenance tables are owned by WS1 (otter/textbooks/provenance.py). This DDL is used ONLY to
# create self-contained synthetic fixture DBs; it is a superset-compatible version of WS1's
# schema (the reader adapts to whatever columns exist, see sqlite_store._span_from_row).
FIXTURE_PROVENANCE_DDL = """
CREATE TABLE IF NOT EXISTS books(book_id TEXT PRIMARY KEY, metadata_json TEXT NOT NULL, source_sha256 TEXT);
CREATE TABLE IF NOT EXISTS sections(section_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, title TEXT,
  pdf_page_start INTEGER, pdf_page_end INTEGER, printed_page_start TEXT, printed_page_end TEXT, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spans(span_id TEXT PRIMARY KEY, book_id TEXT NOT NULL, pdf_page_index INTEGER,
  source_page_index INTEGER NOT NULL, printed_page_label TEXT, printed_page_verified INTEGER NOT NULL, section_id TEXT,
  text TEXT NOT NULL, source_sha256 TEXT NOT NULL, start_offset INTEGER, end_offset INTEGER, heading_path_json TEXT,
  pedagogical_type TEXT, extractor_version TEXT, evidence_status TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(span_id UNINDEXED, book_id UNINDEXED, text);
"""

# Knowledge-maintenance add-on (otter/builder/revalidate.py). These tables exist ONLY in a maintained
# version file (written by `revalidate`); the original build never has them, so old versions behave exactly
# as before. `valid=0` rows are hidden from retrieval/typed access when the store runs with valid_only=True.
MAINTENANCE_DDL = """
CREATE TABLE IF NOT EXISTS item_status(
  item_kind TEXT NOT NULL, item_id TEXT NOT NULL, status TEXT NOT NULL, valid INTEGER NOT NULL,
  support_old TEXT, support_new TEXT, reason TEXT, n_spans INTEGER, n_same INTEGER, n_changed INTEGER, n_gone INTEGER,
  n_unresolved INTEGER, rechecked INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(item_kind, item_id));
CREATE INDEX IF NOT EXISTS ix_item_status_valid ON item_status(valid);
CREATE TABLE IF NOT EXISTS span_alignment(
  old_span_id TEXT PRIMARY KEY, new_span_ids_json TEXT NOT NULL, status TEXT NOT NULL, method TEXT, jaccard REAL,
  cosine REAL, note TEXT);
"""

# Knowledge-ingest add-on (otter/builder/ingest_books.py, `python -m otter.builder ingest-books`). Only present in a file
# written by `ingest-books` (on top of MAINTENANCE_DDL, whose `item_status.valid` hides conflicting items); old builds never
# have these tables. `item_sources` is the provenance ledger: which BOOK (edition, publisher) supports which item, with
# what role. Conflict pairs keep BOTH sides.
INGEST_DDL = """
CREATE TABLE IF NOT EXISTS ingest_runs(
  ingest_id TEXT PRIMARY KEY, build_id TEXT, parent_build_id TEXT, created_at TEXT, books_json TEXT, config_json TEXT,
  summary_json TEXT);
CREATE TABLE IF NOT EXISTS book_sources(
  book_id TEXT PRIMARY KEY, role TEXT NOT NULL, title TEXT, subject TEXT, grade INTEGER, publisher TEXT, series TEXT,
  edition TEXT, edition_year INTEGER, n_spans INTEGER, ingest_id TEXT, origin TEXT, meta_json TEXT);
CREATE TABLE IF NOT EXISTS item_sources(
  item_kind TEXT NOT NULL, item_id TEXT NOT NULL, book_id TEXT NOT NULL, role TEXT NOT NULL, span_ids_json TEXT,
  matched_item_id TEXT, ingest_id TEXT, note TEXT, PRIMARY KEY(item_kind, item_id, book_id, role));
CREATE INDEX IF NOT EXISTS ix_item_sources_book ON item_sources(book_id);
CREATE TABLE IF NOT EXISTS concept_merge_log(
  ingest_id TEXT NOT NULL, new_concept_id TEXT NOT NULL, new_label TEXT, subject TEXT, decision TEXT NOT NULL,
  method TEXT, old_concept_id TEXT, sim REAL, final_concept_id TEXT, evidence_json TEXT,
  PRIMARY KEY(ingest_id, new_concept_id));
CREATE TABLE IF NOT EXISTS item_conflicts(
  conflict_id TEXT PRIMARY KEY, ingest_id TEXT, item_kind TEXT NOT NULL, new_item_id TEXT NOT NULL,
  old_item_id TEXT NOT NULL, verdict TEXT NOT NULL, tier TEXT, slot TEXT, old_value TEXT, new_value TEXT,
  new_book_id TEXT, old_book_ids_json TEXT, handling TEXT, review_status TEXT NOT NULL DEFAULT 'open',
  evidence_json TEXT);
CREATE INDEX IF NOT EXISTS ix_item_conflicts_new ON item_conflicts(item_kind, new_item_id);
"""
INGEST_TABLES = ("ingest_runs", "book_sources", "item_sources", "concept_merge_log", "item_conflicts")
