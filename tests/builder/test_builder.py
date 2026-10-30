"""Offline end-to-end tests for the WS2 builder (MockLLM, synthetic fixture spans, HashEmbedder)."""
import csv
import json
import re

import pytest

from otter.builder import extract as X
from otter.builder.normalize import _compatible, normalize
from otter.builder.pipeline import BuildConfig, build
from otter.builder.segment import Chunk, segment_book
from otter.llm.interface import MockLLM
from otter.schemas.textbook import SourceSpan
from otter.storage.embeddings import HashEmbedder
from otter.storage.fixtures import FIXTURE_IDS, build_fixture_db
from otter.storage.sqlite_store import SqliteOntologyStore

LINE = re.compile(r"^\[(S\d+)\](?: \([^)]*\))? (.*)$", re.M)


def _ids(prompt: str) -> dict[str, str]:
    """text-snippet -> local id, from the rendered chunk."""
    return {m.group(2): m.group(1) for m in LINE.finditer(prompt)}


def _find(prompt: str, needle: str):
    for text, lid in _ids(prompt).items():
        if needle in text:
            return lid
    return None


def extractor_handler(messages):
    p = messages[-1]["content"]
    out = {"concepts": [], "relations": [], "facts": [], "rules": [], "objectives": []}
    s = _find(p, "Quang hợp là quá trình")
    if s:
        out["concepts"].append({"label": "quang hợp", "concept_type": "Process", "spans": [s],
                                "definition": "quá trình thực vật sử dụng năng lượng ánh sáng để tổng hợp chất hữu cơ "
                                              "từ khí carbon dioxide và nước"})
        out["concepts"].append({"label": "Quang  hợp", "concept_type": "Concept", "spans": [s]})   # same key
        out["concepts"].append({"label": "carbon dioxide", "aliases": ["CO2"], "concept_type": "Substance", "spans": [s]})
        out["facts"].append({"text": "Quang hợp giải phóng khí oxygen.", "concepts": ["quang hợp"], "spans": [s]})
        # invented fact (not in the cited span) -> must end up unsupported
        out["facts"].append({"text": "Cây xương rồng dự trữ nước trong thân để sống ở sa mạc.", "spans": [s]})
        # bad span reference -> rejected
        out["facts"].append({"text": "Quang hợp diễn ra vào ban ngày.", "spans": ["S99"]})
        # dangling concept (label not in the text) -> rejected
        out["relations"].append({"subject": "quang hợp", "relation_type": "involves", "object": "kì lân", "spans": [s]})
    s2 = _find(p, "Lục lạp là bào quan")
    if s2:
        out["concepts"].append({"label": "lục lạp", "concept_type": "Structure", "spans": [s2],
                                "definition": "bào quan thực hiện quang hợp"})
        out["relations"].append({"subject": "quang hợp", "relation_type": "occurs_in", "object": "lục lạp",
                                 "spans": [s2]})
    s3 = _find(p, "nếu cường độ ánh sáng tăng")
    if s3:
        out["rules"].append({
            "rule_type": "conditional", "status": "explicit_rule", "spans": [s3],
            "premises": [{"predicate": "light_intensity_increases", "text": "cường độ ánh sáng tăng"}],
            "guards": [{"predicate": "below_light_saturation", "text": "chưa đạt điểm bão hòa ánh sáng"},
                       {"predicate": "other_factors_non_limiting", "text": "các yếu tố khác không bị giới hạn"}],
            "conclusions": [{"predicate": "photosynthesis_rate_increases", "text": "cường độ quang hợp tăng"}]})
    s4 = _find(p, "nồng độ khí carbon dioxide tăng")
    if s4:   # guard "trong một giới hạn nhất định" deliberately omitted -> possible_missing_guard
        out["rules"].append({
            "rule_type": "formula", "spans": [s4],     # formula without formula text -> coerced to conditional
            "premises": [{"predicate": "Nồng độ CO2 tăng", "text": "nồng độ khí carbon dioxide tăng"}],
            "conclusions": [{"predicate": "photosynthesis_rate_increases", "text": "cường độ quang hợp tăng"}]})
    s5 = _find(p, "Tầng đối lưu")
    if s5:
        out["concepts"].append({"label": "tầng đối lưu", "concept_type": "Structure", "spans": [s5]})
        out["concepts"].append({"label": "quang hợp", "concept_type": "Process", "spans": [s5]})  # other subject
    s6 = _find(p, "Sau bài học này")
    if s6:
        out["objectives"].append({"text": "Nêu được khái niệm quang hợp.", "spans": [s6]})
    return json.dumps(out, ensure_ascii=False)


def verifier_handler(messages):
    p = messages[-1]["content"]
    claims = re.findall(r"^(\d+)\. \(trích dẫn: [^)]*\) (.*)$", p, flags=re.M)
    return json.dumps({"judgments": [{"id": int(i), "label": "unsupported" if "xương rồng" in c else "supported"}
                                     for i, c in claims]}, ensure_ascii=False)


class CountingMock(MockLLM):
    calls = 0

    def chat(self, messages, **kw):
        type(self).calls += 1
        return super().chat(messages, **kw)


@pytest.fixture()
def fx(tmp_path):
    return build_fixture_db(tmp_path / "prov.sqlite3")


def _cfg(tmp_path, fx, **kw):
    base = dict(db=fx, out=tmp_path / "out" / "onto.sqlite3", export_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
                parsed_dir=tmp_path / "none", registry_dir=tmp_path / "none", min_tokens=60, max_tokens=200,
                workers=4, verify_workers=4)
    base.update(kw)
    return BuildConfig(**base)


def _run(tmp_path, fx, **kw):
    ext = CountingMock(handler=extractor_handler, model="mock-extractor")
    ver = MockLLM(handler=verifier_handler, model="mock-verifier")
    return build(_cfg(tmp_path, fx, **kw), extractor_llm=ext, verifier_llm=ver, embedder=HashEmbedder()), ext


def test_end_to_end_build(tmp_path, fx):
    rep, _ = _run(tmp_path, fx)
    out = tmp_path / "out" / "onto.sqlite3"
    store = SqliteOntologyStore(out, provenance_path=fx, embedder=HashEmbedder())
    c = store.get_concept("biology:c:quang_hop")
    assert c is not None and c.definition and c.concept_type == "Process"
    assert "geography:c:quang_hop" in {h["id"] for h in store.search_knowledge("quang hợp", kinds=("concept",))} \
        or store.get_concept("geography:c:quang_hop") is not None          # subject-scoped, never merged
    assert store.get_concept("geography:c:quang_hop").subject == "geography"
    # every stored item cites existing spans only
    for kind, table, pk in (("fact", "facts", "assertion_id"), ("rule", "rules", "rule_id"),
                            ("concept", "concepts", "concept_id")):
        for (iid,) in store.db.execute(f"SELECT {pk} FROM {table}"):
            spans = store._evidence(kind, iid)
            assert spans and all(store.span_exists(s) for s in spans), iid
    # exercises are not sent to the extractor
    ex_spans = set(FIXTURE_IDS["exercise_spans"])
    assert not any(s in ex_spans for s, in store.db.execute("SELECT span_id FROM evidence_links"))
    reasons = {(r["kind"], r["reason"]) for r in map(json.loads, (tmp_path / "out/export/rejected.jsonl").open())}
    assert ("fact", "invalid_span_ref") in reasons and ("relation", "dangling_concept") in reasons
    facts = {f.text: f for f in (store.get_fact(i) for i, in store.db.execute("SELECT assertion_id FROM facts"))}
    assert facts["Cây xương rồng dự trữ nước trong thân để sống ở sa mạc."].validation.source_support == "unsupported"
    assert facts["Quang hợp giải phóng khí oxygen."].validation.source_support == "supported"
    rules = store.rules_concluding("photosynthesis_rate_increases", subject="biology")
    good = [r for r in rules if r.guards]
    bad = [r for r in rules if not r.guards]
    assert good and store.is_strict_executable(good[0])
    assert bad and "possible_missing_guard" in bad[0].validation.notes and not store.is_strict_executable(bad[0])
    assert bad[0].rule_type == "conditional" and "rule_type_coerced_from_formula" in bad[0].validation.notes
    assert bad[0].premises[0].predicate == "nong_do_co2_tang"
    # report + exports
    rj = json.loads((tmp_path / "out/build_report.json").read_text())
    assert rj["totals"]["concept"] >= 4 and rj["totals"]["rule"] == 2
    assert rj["support"]["fact"]["unsupported"] >= 1
    assert rj["verification"]["mode"] == "lexical+llm"
    assert (tmp_path / "out/build_report.md").read_text().startswith("# OTTER ontology build report")
    g = json.loads((tmp_path / "out/export/graph.json").read_text())
    assert g["nodes"]
    assert store.stats()["embeddings"] > 0 and store.build_info()["build_id"] == rep["build_id"]


def test_incremental_cache(tmp_path, fx):
    _run(tmp_path, fx)
    CountingMock.calls = 0
    rep, _ = _run(tmp_path, fx)
    assert CountingMock.calls == 0
    assert rep["extraction"]["cached_chunks"] == rep["extraction"]["chunks"] > 0


def test_verifier_must_differ(tmp_path, fx):
    ext = MockLLM(handler=extractor_handler, model="same-model")
    ver = MockLLM(handler=verifier_handler, model="same-model")
    with pytest.raises(ValueError):
        build(_cfg(tmp_path, fx), extractor_llm=ext, verifier_llm=ver, embedder=HashEmbedder())


def test_lexical_only_without_verifier(tmp_path, fx):
    rep = build(_cfg(tmp_path, fx, no_verify=True), extractor_llm=MockLLM(handler=extractor_handler, model="m"),
                embedder=HashEmbedder())
    assert rep["verification"]["mode"] == "lexical_only"
    assert rep["validation_notes"].get("support_lexical_only", 0) > 0


def test_book_selection(tmp_path, fx):
    rep, _ = _run(tmp_path, fx, books=["fx_geo*"])
    assert set(rep["books"]) == {"fx_geo10_synthetic"}
    assert set(rep["by_subject"]) == {"geography"}


def test_diacritic_guard_and_subject_scope():
    assert not _compatible("bào", "bảo", "Concept", "Concept")
    assert not _compatible("thế kỉ 18", "thế kỉ 19", "Period", "Period")
    sp = {f"b:p0000:u00{i}": SourceSpan(span_id=f"b:p0000:u00{i}", book_id="b", pdf_page_index=0, start_offset=0,
                                        end_offset=10, text=t, source_file_sha256="x")
          for i, t in enumerate(["tế bào là đơn vị", "bảo tồn thiên nhiên"])}
    res = []
    for subj, sid, label in (("biology", "b:p0000:u000", "bào"), ("biology", "b:p0000:u001", "bảo"),
                             ("geography", "b:p0000:u001", "bảo")):
        ch = Chunk(f"c{len(res)}", "b", subj, 10, "t", [], [], [sp[sid]])
        res.append(X.ExtractionResult(ch, X.ExtractionOutput(concepts=[X.XConcept(label=label, spans=["S1"])])))

    class AllSame(HashEmbedder):
        def embed(self, texts):
            import numpy as np
            return np.ones((len(texts), 4), dtype="float32") / 2.0
    onto = normalize(res, sp, embedder=AllSame(), sim_threshold=0.5)
    assert {c.subject for c in onto.concepts.values()} == {"biology", "geography"}
    assert len([c for c in onto.concepts.values() if c.subject == "biology"]) == 2      # bào != bảo


def test_segment_chunks_by_lesson(fx):
    from otter.builder.ingest import load_sections, load_spans
    spans = load_spans(fx, ["fx_bio11_synthetic"])["fx_bio11_synthetic"]
    secs = load_sections(fx, ["fx_bio11_synthetic"])
    chunks = segment_book("fx_bio11_synthetic", spans, secs, subject="biology", grade=11, book_title="t",
                          min_tokens=10, max_tokens=10_000)
    assert len(chunks) == 2 and chunks[0].section_ids == ["fx_bio11_synthetic:l01"]
    assert all(s.pedagogical_type != "exercise" for c in chunks for s in c.spans)
    prompt = X.render_chunk(chunks[0])
    assert "[S1]" in prompt and "fx_bio11_synthetic:p0000" not in prompt


def test_audit_sheet_and_score(tmp_path, fx):
    from otter.builder.audit import make_audit, score_audit
    _run(tmp_path, fx)
    info = make_audit(tmp_path / "out/onto.sqlite3", fx, tmp_path / "audit", n=12, seed=1)
    assert info["n"] >= 6
    rows = list(csv.DictReader((tmp_path / "audit/ontology_audit.csv").open(encoding="utf-8")))
    assert all(r["cited_span_text"] for r in rows) and {"judgment", "notes"} <= set(rows[0])
    html = (tmp_path / "audit/ontology_audit.html").read_text(encoding="utf-8")
    assert "<table" in html and "correct" in html
    for i, r in enumerate(rows):
        r["judgment"] = ["correct", "partial", "incorrect"][i % 3]
    with (tmp_path / "filled.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = score_audit(tmp_path / "filled.csv")
    assert res["overall"]["n_judged"] == len(rows)
    assert 0 <= res["overall"]["precision_strict"] <= res["overall"]["precision_lenient"] <= 1


def test_truncated_output_salvaged():
    raw = '{"concepts":[{"label":"lục lạp","spans":["S1"]},{"label":"diệp'
    assert X.salvage_json(raw) == {"concepts": [{"label": "lục lạp", "spans": ["S1"]}]}
    sch = X.output_schema()
    assert sch["properties"]["concepts"]["maxItems"] == X.TOP_MAX_ITEMS["concepts"]
    assert sch["$defs"]["XConcept"]["properties"]["spans"]["items"]["pattern"] == "^S[0-9]{1,3}$"
    span = SourceSpan(span_id="b:p0000:u000", book_id="b", pdf_page_index=0, start_offset=0, end_offset=7,
                      text="lục lạp", source_file_sha256="x")
    ch = Chunk("c0", "b", "biology", 10, "t", [], [], [span])
    res = X.Extractor(MockLLM(replies=[raw]), None).extract(ch)
    assert res.status == "ok" and [c.label for c in res.output.concepts] == ["lục lạp"]
    assert res.item_errors[-1]["reason"] == "output_truncated_salvaged"


def test_unguided_output_coercion():
    raw = {"concepts": [{"label": "vận tốc", "concept_type": "quantity", "spans": "S1"}],
           "relations": [{"subject": "vận tốc", "relation_type": "is-a", "object": "đại lượng", "spans": ["[S1]"]},
                         {"subject": "a", "relation_type": "teleports", "object": "b", "spans": ["S1"]}],
           "rules": [{"rule_type": "law", "premises": ["vật chuyển động thẳng đều"],
                      "conclusions": [{"predicate": "Vận tốc không đổi", "text": "vận tốc không đổi"}],
                      "formula": "v = s / t", "variables": {"v": "vận tốc", "s": "quãng đường"}, "spans": ["S1"]}],
           "facts": ["not an object"], "objectives": []}
    out, errs = X.parse_output(raw)
    assert out.concepts[0].concept_type == "Quantity" and out.concepts[0].spans == ["S1"]
    assert [r.relation_type for r in out.relations] == ["is_a", "related_to"]
    assert out.relations[1].qualifier == "teleports"
    r = out.rules[0]
    assert r.rule_type == "conditional" and r.premises[0].predicate.startswith("vat_chuyen_dong")
    assert r.conclusions[0].predicate == "van_toc_khong_doi" and {v.symbol for v in r.variables} == {"v", "s"}
    assert [e["reason"] for e in errs] == ["schema_invalid"]
    out2, errs2 = X.parse_output({"rules": [{"premises": [], "conclusions": [{"text": "v = s / t"}], "spans": ["S1"]}]})
    assert not out2.rules and out2.facts[0].text == "v = s / t"
    assert errs2[0]["reason"] == "converted_to_fact_no_premises"


def test_verifier_roles_never_alt(monkeypatch):
    from otter.builder import pipeline as P
    monkeypatch.setattr(P, "endpoint_up", lambda role, config=None: True)
    conf = {"llm": {"main": {"model": "A"}, "small": {"model": "B"}, "alt": {"model": "C"}, "index": {"model": "D"}}}
    assert P.pick_verifier("main", None, conf) == "small"
    assert P.pick_verifier("small", None, conf) == "main"
    assert P.pick_verifier("main", "alt", conf) is None
    conf["llm"]["small"]["model"] = "A"
    assert P.pick_verifier("main", None, conf) is None          # same model -> no verifier
