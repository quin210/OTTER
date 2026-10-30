# otter.storage — ontology store (WS2)

```python
from otter.storage import SqliteOntologyStore
store = SqliteOntologyStore.from_config()          # experiments: data/ontology/ontology.sqlite3 + provenance DB
store = SqliteOntologyStore("fx.sqlite3")          # single-file DB (fixtures)
store = SqliteOntologyStore(onto, provenance_path=prov, embedder=None, use_dense=True)

from otter.storage.fixtures import build_fixture_db, FIXTURE_IDS
build_fixture_db(tmp_path / "fx.sqlite3")          # self-written synthetic Biology + Geography lessons
```
A prebuilt fixture is at `data/ontology/fixture.sqlite3` (`python -m otter.storage.fixtures <path>` to rebuild).

## Layout
* **Ontology DB** (WS2, `paths.ontology_db` = `data/ontology/ontology.sqlite3`): `concepts`, `concept_aliases`,
  `semantic_relations`, `facts` (+`fact_concepts`), `rules`, `rule_premises`, `rule_guards`, `rule_conclusions`,
  `rule_exceptions`, `rule_concepts`, `objectives`, `pedagogical_links`, `evidence_links`, `rejected_items`,
  `embeddings`, `build_versions`, FTS5 `knowledge_fts`. DDL: `otter/storage/schema.py`. Names avoid WS1's legacy
  `assertions`/`relations` tables. Each row keeps the full pydantic object in `json`.
* **Provenance DB** (WS1, `paths.db`): `books`, `sections`, `spans`, `spans_fts` — read-only, ATTACHed as `prov`.
  The span reader adapts to the columns present (`source_sha256`/`source_file_sha256`, optional offsets, etc.).
* Ontology builds go to a temp file and are atomically renamed, so they never touch WS1's file.

## OntologyReader methods (`otter/schemas/store_api.py`) + extras
| method | notes |
|---|---|
| `search_knowledge(q, subject, grade, kinds, top_k)` | kinds ⊆ concept, fact, rule, relation, objective. FTS5 BM25 + bge-m3 dense (RRF, k=60). Hit = `{kind,id,label_or_text,score,subject,source_span_ids, grade, source_support, status, score_components, strict_executable (rules)}`. Grade filter keeps items with NULL grade. |
| `search_spans(q, subject, grade, book_ids, top_k)` | WS1 `spans_fts` BM25 (+ dense if span embeddings were built); `[(SourceSpan, score)]`. |
| `get_concept / get_fact / get_rule` (+ `get_relation`, `get_objective`) | `None` if missing. |
| `rules_concluding(predicate, subject)` | exact predicate match; **check `atom.negated`** on the conclusion. Also `rules_with_premise`. |
| `neighbors(ids, relation_types, max_hops, subject, max_relations=300)` | undirected BFS over typed relations (networkx MultiDiGraph). Accepts concept/rule/fact ids. Returns `{nodes, relations, rules, facts, hops, unknown_ids, invalid_relation_types}`. Subjects never mix (ids are subject-scoped). |
| `get_spans(ids)` / `span_exists(id)` | only existing spans returned → missing ids are invalid citations. |
| `book_metadata(book_id)` / `section(section_id)` | metadata_json merged with columns; never invents fields. |
| `find_concepts(label, subject)`, `facts_about(concept_id)`, `stats()`, `build_info()` | helpers |
| `is_strict_executable(rule)` | **strict-proof policy**: `source_ids_exist` ∧ `source_support=="supported"` ∧ `status=="explicit_rule"` ∧ premises ∧ conclusions ∧ no `possible_missing_guard` / `excluded_from_strict_proof` note. Others stay usable for source-supported evidence chains only. |

## Dense search
bge-m3 at `embedding.base_url` (:8002). If unreachable (probed with 2 s timeout, re-probed every 60 s) or
`OTTER_LLM_MOCK=1` / `OTTER_NO_DENSE=1`, search is FTS/BM25-only. Vectors live in `embeddings`
(`kind`, `item_id`, `model`, float32 blob); only vectors whose `model` equals the embedder's model are used.
`HashEmbedder` is an offline test embedder only.

## IDs
`<subject>:c:<ascii_slug>` concepts, `<subject>:rel:…`, `<subject>:a:…` facts, `<subject>:rule:…`, `<subject>:obj:…`.
Fixture ids: see `FIXTURE_IDS` (valid / incomplete / unsupported / formula rules, missing span id, exercise spans).

## Builder (`otter/builder/`, WS2)
```
python -m otter.builder build --db data/textbooks/ontology.sqlite3 [--books core|<id>|<glob> ...] [--backbone main|small]
       [--verifier small] [--workers 48] [--limit-chunks N] [--no-verify] [--no-embed] [--no-span-embeddings]
python -m otter.builder audit-sample --n 150      # annotation/ontology_audit/ontology_audit.{csv,html} (+ _key.csv)
python -m otter.builder audit-score annotation/ontology_audit/ontology_audit_filled.csv
```
Pipeline: `ingest` (WS1 provenance DB, read-only; paragraph spans `…:uNNN` preferred) → `segment` (lesson-level
chunks 1.5–3k tokens, front matter + exercises/questions skipped) → `extract` (guided-JSON, local span ids S1..Sn,
content-hash cache `data/ontology/cache/extract/`, thread pool) → `normalize` (subject-scoped canonicalisation:
NFC/casefold key, aliases, bge-m3 ≥0.93 with diacritic/number/type guards; predicate canonicalisation) →
`validate` (hard: span existence in DB, dangling ids, domain/range, rule structure, circular rules;
heuristics: `possible_missing_guard`; separate support check = syllable uni/bi-gram overlap + entailment by a
DIFFERENT model, cached in `cache/verify/`) → `persist` (atomic replace of `data/ontology/ontology.sqlite3`,
FTS5, bge-m3 vectors incl. spans, JSONL + networkx `graph.json` in `data/ontology/export/`) → `report`
(`data/ontology/build_report.{json,md}`). Unsupported rules are kept with `excluded_from_strict_proof`.
