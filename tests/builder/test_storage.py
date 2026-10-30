import shutil
import sqlite3
import threading

import pytest

from otter.schemas.store_api import OntologyReader
from otter.storage.embeddings import HashEmbedder
from otter.storage.fixtures import FIXTURE_IDS, build_fixture_db
from otter.storage.schema import ONTOLOGY_TABLES
from otter.storage.sqlite_store import SqliteOntologyStore


@pytest.fixture(scope="module")
def fx_path(tmp_path_factory):
    return build_fixture_db(tmp_path_factory.mktemp("fx") / "fx.sqlite3")


@pytest.fixture()
def store(fx_path):
    return SqliteOntologyStore(fx_path, embedder=HashEmbedder())


def test_protocol_methods_present(store):
    for name in [m for m in dir(OntologyReader) if not m.startswith("_")]:
        assert callable(getattr(store, name)), name


def test_typed_access_and_missing(store):
    c = store.get_concept("biology:c:quang_hop")
    assert c and c.subject == "biology" and c.source_span_ids
    assert store.get_concept("biology:c:does_not_exist") is None
    r = store.get_rule("biology:rule:fx_light_valid")
    assert r and len(r.guards) == 2 and r.validation.source_support == "supported"
    assert store.get_fact("geography:a:fx01").subject == "geography"


def test_bad_span_rejected_and_nullable_page(store):
    assert not store.span_exists(FIXTURE_IDS["missing_span"])
    spans = store.get_spans([FIXTURE_IDS["missing_span"], "fx_geo10_synthetic:p0000:u002"])
    assert [s.span_id for s in spans] == ["fx_geo10_synthetic:p0000:u002"]
    s = spans[0]
    assert s.printed_page_label is None and s.printed_page_verified is False
    page_text = "\n\n".join(x.text for x in store.get_spans([f"fx_geo10_synthetic:p0000:u00{i}" for i in range(7)]))
    assert page_text[s.start_offset:s.end_offset] == s.text


def test_strict_policy(store):
    for rid in FIXTURE_IDS["valid_rules"]:
        assert store.is_strict_executable(store.get_rule(rid)), rid
    for rid in FIXTURE_IDS["incomplete_rules"] + FIXTURE_IDS["unsupported_rules"]:
        assert not store.is_strict_executable(store.get_rule(rid)), rid
    hits = store.search_knowledge("nồng độ carbon dioxide quang hợp", kinds=("rule",), subject="biology")
    flags = {h["id"]: h["strict_executable"] for h in hits}
    assert flags.get("biology:rule:fx_co2_incomplete") is False


def test_search_knowledge_subject_scoped(store):
    hits = store.search_knowledge("nhiệt độ giảm theo độ cao", subject="geography", top_k=5)
    assert hits and all(h["subject"] == "geography" for h in hits)
    assert "geography:rule:fx_lapse_rate" in [h["id"] for h in hits]
    for h in hits:
        assert set(h) >= {"kind", "id", "label_or_text", "score", "subject", "source_span_ids"}
        assert all(store.span_exists(s) for s in h["source_span_ids"])
    assert store.search_knowledge("nhiệt độ giảm theo độ cao", subject="biology", grade=10) == [] or all(
        h["subject"] == "biology" for h in store.search_knowledge("nhiệt độ", subject="biology"))


def test_lexical_only_fallback(fx_path, monkeypatch):
    monkeypatch.setenv("OTTER_NO_DENSE", "1")
    s = SqliteOntologyStore(fx_path)          # default bge-m3 embedder, disabled -> FTS only
    hits = s.search_knowledge("lục lạp", top_k=3)
    assert hits[0]["id"] == "biology:c:luc_lap"
    assert "dense" not in hits[0]["score_components"]


def test_search_spans_filters(store):
    res = store.search_spans("sườn đón gió mưa", subject="geography", top_k=3)
    assert res and res[0][0].span_id == "fx_geo10_synthetic:p0000:u003"
    assert store.search_spans("sườn đón gió mưa", subject="biology", top_k=3) == [] or all(
        s.book_id == "fx_bio11_synthetic" for s, _ in store.search_spans("sườn đón gió", subject="biology"))
    assert all(s.book_id == "fx_bio11_synthetic" for s, _ in
               store.search_spans("quang hợp", book_ids=["fx_bio11_synthetic"], top_k=5))


def test_neighbors_and_rules(store):
    n1 = store.neighbors(["biology:c:quang_hop"], max_hops=1)
    n2 = store.neighbors(["biology:c:quang_hop"], max_hops=2)
    ids1 = {c.concept_id for c in n1["nodes"]}
    ids2 = {c.concept_id for c in n2["nodes"]}
    assert "biology:c:luc_lap" in ids1 and "biology:c:diep_luc" not in ids1 and "biology:c:diep_luc" in ids2
    assert n2["hops"]["biology:c:diep_luc"] == 2
    only = store.neighbors(["biology:c:quang_hop"], relation_types=["occurs_in"], max_hops=1)
    assert {r.relation_type for r in only["relations"]} == {"occurs_in"}
    bad = store.neighbors(["biology:c:quang_hop"], relation_types=["teleports_to"])
    assert bad["invalid_relation_types"] == ["teleports_to"] and bad["relations"] == []
    assert store.neighbors(["nope:c:x"])["unknown_ids"] == ["nope:c:x"]
    # never cross subjects
    assert all(c.subject == "biology" for c in store.neighbors(["biology:c:quang_hop"], max_hops=3)["nodes"])
    concl = store.rules_concluding("photosynthesis_rate_increases", subject="biology")
    assert {r.rule_id for r in concl} >= {"biology:rule:fx_light_valid", "biology:rule:fx_co2_incomplete"}
    assert store.rules_concluding("photosynthesis_rate_increases", subject="geography") == []


def test_metadata(store):
    m = store.book_metadata("fx_bio11_synthetic")
    assert m["subject"] == "biology" and m["grade"] == 11 and m["synthetic"] is True
    assert store.book_metadata("nope") is None
    assert store.section("fx_bio11_synthetic:l02")["title"].startswith("Bài 2")


def test_attached_provenance_layout(fx_path, tmp_path):
    prov, onto = tmp_path / "prov.sqlite3", tmp_path / "onto.sqlite3"
    shutil.copy(fx_path, prov)
    shutil.copy(fx_path, onto)
    with sqlite3.connect(prov) as db:
        for t in ONTOLOGY_TABLES:
            db.execute(f"DROP TABLE IF EXISTS {t}")
    with sqlite3.connect(onto) as db:
        for t in ("books", "sections", "spans", "spans_fts"):
            db.execute(f"DROP TABLE IF EXISTS {t}")
    s = SqliteOntologyStore(onto, provenance_path=prov, embedder=HashEmbedder())
    assert s.span_exists("fx_geo10_synthetic:p0000:u002")
    assert s.search_spans("lũ quét", top_k=1)[0][0].span_id == "fx_geo10_synthetic:p0000:u005"
    assert s.book_metadata("fx_geo10_synthetic")["grade"] == 10
    assert s.search_knowledge("lũ quét", kinds=("rule",))


def test_thread_safety(store):
    errs = []

    def work():
        try:
            for _ in range(20):
                assert store.search_knowledge("quang hợp", top_k=3)
                assert store.neighbors(["geography:c:mua"], max_hops=2)["nodes"]
        except Exception as e:  # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
