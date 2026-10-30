"""Knowledge ingestion of NEW books: claim comparison, concept matching, end-to-end ingest (MockLLM, fixture DB, HashEmbedder)."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from otter.builder import claims as C
from otter.builder.ingest_books import IngestConfig, ingest_books
from otter.builder.index_ext import IncrementalBM25
from otter.llm.interface import MockLLM
from otter.schemas.ontology import Fact
from otter.storage.embeddings import HashEmbedder
from otter.storage.fixtures import BIO, GEO, build_fixture_db
from otter.storage.sqlite_store import SqliteOntologyStore
from otter.storage.writer import OntologyWriter

NEWB = "pubB_geo10_synth"


# ------------------------------------------------------------------------------------------- claim comparison
def test_number_parsing_vietnamese_formats():
    assert C.parse_number("1.000") == 1000.0 and C.parse_number("3,14") == 3.14 and C.parse_number("1.250,5") == 1250.5
    assert [v for _, v in C.number_values("từ năm 1930 đến 1945, cao 8.848 m, nặng 0,6 kg")] == [1930.0, 1945.0, 8848.0, 0.6]


def test_identical_text_agrees_without_llm():
    v = C.compare_lexical("Tầng đối lưu là tầng thấp nhất của khí quyển.", "tầng đối lưu là tầng thấp nhất của khí quyển")
    assert v.label == "agrees" and v.tier == "exact"


def test_same_wording_different_number_is_a_pending_conflict():
    v = C.compare_lexical("Cứ lên cao 100 m thì nhiệt độ không khí giảm khoảng 0,6 °C.",
                          "Cứ lên cao 100 m thì nhiệt độ không khí giảm khoảng 0,65 °C.")
    assert v.label == "conflicts" and v.tier == "lexical_pending" and v.slot == "number" and "0,6" in v.old_value
    # no LLM available: the lexical conflict stays, flagged lexical
    assert C.finalize_pending(v, lexical_only=True).label == "conflicts"


def test_same_number_different_format_agrees():
    v = C.compare_lexical("Dân số nước ta năm 2021 khoảng 98.000 nghìn người và đang tăng chậm.",
                          "Năm 2021 dân số nước ta khoảng 98000 nghìn người, đang tăng chậm.")
    assert v.label == "agrees"


def test_different_aspect_is_unrelated_and_numbers_of_other_objects_do_not_conflict():
    v = C.compare_lexical("Sông Cửu Long dài khoảng 4 500 km và đổ ra biển Đông.", "Núi Phan Xi Păng cao 3 143 m ở Hoàng Liên Sơn.")
    assert v.label == "unrelated"


def test_entity_swap_needs_the_llm():
    v = C.compare_lexical("Năm 1789 vua Quang Trung đánh tan quân Thanh ở Ngọc Hồi.",
                          "Năm 1789 vua Gia Long đánh tan quân Thanh ở Ngọc Hồi.")
    assert v.label == "ambiguous" and v.tier == "lexical_pending" and v.slot == "entity"


def test_formula_difference_goes_to_the_judge_not_auto_conflict():
    v = C.compare_lexical("Nếu vật chuyển động đều thì quãng đường tỉ lệ với thời gian. Công thức: s = v t",
                          "Nếu vật chuyển động đều thì quãng đường tỉ lệ với thời gian. Công thức: v = s / t",
                          old_formula="s = v t", new_formula="v = s / t")
    assert v.tier == "lexical_pending" and v.slot == "formula" and v.label == "ambiguous"


def test_v2_time_point_guard_containment_and_negation():
    # statistics of different years are different data points, not a conflict
    v = C.compare_lexical("Sản lượng sữa tươi năm 2010 là 1 296,8 triệu lít.", "Sản lượng sữa tươi năm 2020 là 1 601,6 triệu lít.")
    assert v.label == "unrelated" and v.note == "different_time_point"
    assert C.compare_lexical("Sản lượng sữa tươi năm 2010 là 1 296,8 triệu lít.", "Sản lượng sữa tươi năm 2020 là 1 601,6 triệu lít.",
                             rules="v1").label == "conflicts"                       # v1 behaviour kept for reproducibility
    # same value under another year: a corrected year is a conflict candidate, not a new data point
    v = C.compare_lexical("Sản lượng khai thác năm 2024 chiếm 39,8% tổng sản lượng.", "Sản lượng khai thác năm 2033 chiếm 39,8% tổng sản lượng.")
    assert v.label != "unrelated"
    # a short claim contained in a long one: same slot (judge needed when the number differs)
    long = "Nếu tia hồng ngoại có bước sóng từ 0,76 μm đến 1 mm thì tia hồng ngoại là sóng điện từ không nhìn thấy được"
    v = C.compare_lexical(long, "Tia hồng ngoại có bước sóng từ 0,58 μm đến 1 mm")
    assert v.tier == "lexical_pending"
    # negation flips the claim: never lexical agreement
    v = C.compare_lexical("Cường độ quang hợp tăng khi cường độ ánh sáng tăng trong giới hạn", "Cường độ quang hợp không tăng khi cường độ ánh sáng tăng trong giới hạn")
    assert v.tier == "lexical_pending" and v.slot == "negation"


def test_aggregate_priority_and_unrelated_ignored():
    ag = C.PairVerdict("agrees", "lexical", 0.8)
    cf = C.PairVerdict("conflicts", "llm", 0.9)
    un = C.PairVerdict("unrelated", "lexical", 0.1)
    assert C.aggregate([("o1", un)])[0] == "new"
    assert C.aggregate([("o1", cf), ("o2", ag)])[:2] == ("agrees", "o2")
    assert C.aggregate([("o1", un), ("o2", cf)])[:2] == ("conflicts", "o2")


def test_judge_parses_batches_caches_and_failure_is_ambiguous(tmp_path):
    calls = []

    def handler(messages):
        calls.append(1)
        user = messages[-1]["content"]
        n = len(re.findall(r"^\d+\. A:", user, flags=re.M))
        return json.dumps({"judgments": [{"id": i, "label": "conflict", "slot": "s", "old_value": "a", "new_value": "b"}
                                         for i in range(1, n + 1)]})
    j = C.PairJudge(MockLLM(handler=handler), tmp_path / "jc", batch_size=2, workers=2)
    pairs = [(f"cũ {i}", f"mới {i}") for i in range(5)]
    res = j.judge(pairs)
    assert all(r.label == "conflict" for r in res) and len(calls) == 3
    j2 = C.PairJudge(MockLLM(handler=handler), tmp_path / "jc", batch_size=2)
    j2.judge(pairs)
    assert len(calls) == 3 and j2.stats["cached_batches"] == 3                          # cache hit, no new calls
    pv = C.PairVerdict("ambiguous", "lexical_pending", 0.6)
    assert C.apply_judgment(pv, None).label == "ambiguous"

    class Broken(MockLLM):
        def chat(self, *a, **k):
            raise RuntimeError("down")
    import otter.builder.claims as CC
    orig = CC.time.sleep
    CC.time.sleep = lambda *_: None
    try:
        jb = C.PairJudge(Broken(), None)
        assert jb.judge([("x y z", "x y w")]) == [None] and jb.stats["failed_batches"] == 1
    finally:
        CC.time.sleep = orig


# ------------------------------------------------------------------------------------------- end to end
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verifier_handler(messages):
    user = messages[-1]["content"]
    out = [{"id": int(m.group(1)), "label": "supported"} for m in re.finditer(r"^(\d+)\. \(trích dẫn", user, flags=re.M)]
    return json.dumps({"judgments": out})


TEXTS = [
    ("u000", "definition", "Tầng đối lưu là tầng thấp nhất của khí quyển, nơi tập trung phần lớn hơi nước."),
    ("u001", "law", "Trong tầng đối lưu, cứ lên cao 100 m thì nhiệt độ không khí giảm khoảng 0,65 °C."),
    ("u002", "concept_explanation", "Khí áp là sức nén của không khí lên bề mặt Trái Đất; khí áp giảm khi lên cao."),
    ("u003", "concept_explanation", "Gió mùa là loại gió đổi hướng theo mùa, hình thành do chênh lệch khí áp giữa lục địa và đại dương."),
]


def extractor_handler(messages):
    return json.dumps({
        "objectives": [], "rules": [],
        "facts": [
            {"text": "Tầng đối lưu là tầng thấp nhất của khí quyển.", "concepts": ["tầng đối lưu"], "spans": ["S1"]},
            {"text": "Cứ lên cao 100 m thì nhiệt độ không khí giảm khoảng 0,65 °C.",
             "concepts": ["nhiệt độ không khí", "tầng đối lưu"], "spans": ["S2"]},
            {"text": "Khí áp giảm khi lên cao.", "concepts": ["khí áp"], "spans": ["S3"]},
        ],
        "relations": [{"subject": "gió mùa", "relation_type": "affects", "object": "khí áp", "spans": ["S4"]}],
        "concepts": [
            {"label": "tầng đối lưu", "concept_type": "Structure", "definition": "tầng thấp nhất của khí quyển", "spans": ["S1"]},
            {"label": "nhiệt độ không khí", "concept_type": "Quantity", "spans": ["S2"]},
            {"label": "khí áp", "concept_type": "Quantity", "definition": "sức nén của không khí lên bề mặt Trái Đất", "spans": ["S3"]},
            {"label": "gió mùa", "concept_type": "Phenomenon", "definition": "loại gió đổi hướng theo mùa", "spans": ["S4"]},
        ]}, ensure_ascii=False)


def judge_handler(messages):
    user = messages[-1]["content"]
    out = []
    for m in re.finditer(r'^(\d+)\. A: "(.*?)"\n\s+B: "(.*?)"', user, flags=re.M | re.S):
        a, b = m.group(2), m.group(3)
        na, nb = re.findall(r"\d+,\d+", a), re.findall(r"\d+,\d+", b)
        lab = "conflict" if na and nb and na != nb else "same"
        out.append({"id": int(m.group(1)), "label": lab, "slot": "độ giảm nhiệt", "old_value": ",".join(na), "new_value": ",".join(nb)})
    return json.dumps({"judgments": out})


@pytest.fixture()
def world(tmp_path):
    emb = HashEmbedder()
    old = build_fixture_db(tmp_path / "old.sqlite3", embedder=emb)
    # an OLD fact carrying the lapse-rate number (the fixture has none) so there is something to conflict with
    w = OntologyWriter(old)
    from otter.storage.fixtures import _ok
    f = Fact(assertion_id="geography:a:lapse", text="Cứ lên cao 100 m thì nhiệt độ không khí giảm khoảng 0,6 °C.", subject="geography",
             concept_ids=["geography:c:nhiet_do_khong_khi", "geography:c:tang_doi_luu"], source_span_ids=[f"{GEO}:p0000:u002"],
             validation=_ok())
    w.add_fact(f, grade=10)
    w.add_embeddings("fact", emb.model, [(f.assertion_id, "geography", 10)], emb.embed([f.text]))
    w.close()
    d = tmp_path / "newbooks" / NEWB
    d.mkdir(parents=True)
    rows, off = [], 0
    for u, ped, t in TEXTS:
        sid = f"{NEWB}:p0000:{u}"
        rows.append({"span_id": sid, "book_id": NEWB, "section_id": f"{NEWB}:l01", "heading_path": ["Bài 1. Khí quyển"], "pdf_page_index": 0,
                     "start_offset": off, "end_offset": off + len(t), "text": t, "pedagogical_type": ped,
                     "source_file_sha256": "x" * 8, "extractor_version": "t", "evidence_status": "ocr_unverified"})
        off += len(t) + 2
    (d / "spans.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (d / "sections.jsonl").write_text(json.dumps({"section_id": f"{NEWB}:l01", "book_id": NEWB, "level": "lesson", "title": "Bài 1. Khí quyển",
                                                  "parent_id": None, "heading_path": ["Bài 1. Khí quyển"]}, ensure_ascii=False), encoding="utf-8")
    (d / "book.json").write_text(json.dumps({"book_id": NEWB, "title": "Địa lí 10 (nhà xuất bản B, giả lập)", "subject_normalized": "geography",
                                             "grade": 10}), encoding="utf-8")
    return tmp_path, old, tmp_path / "newbooks"


def run(world, **kw):
    tmp, old, new = world
    ext = kw.pop("extractor_handler", extractor_handler)
    out = tmp / "ver" / kw.pop("name", "new.sqlite3")
    cfg = IngestConfig(old_db=old, new_spans=new, out=out, old_prov=old, cache_dir=tmp / "cache",
                       book_meta={"*": {"publisher": "NXB B (giả lập)", "edition": "2025"}}, **kw)
    summ = ingest_books(cfg, extractor_llm=MockLLM(handler=ext, model="ext-mock"),
                        verifier_llm=MockLLM(handler=verifier_handler, model="ver-mock"),
                        judge_llm=MockLLM(handler=judge_handler, model="judge-mock"), embedder=HashEmbedder(),
                        config={"llm": {"main": {"model": "ext-mock"}, "small": {"model": "ver-mock"}}, "paths": {"db": "x"}})
    return out, summ


def test_ingest_end_to_end(world):
    tmp, old, _ = world
    before = _sha(old)
    out, summ = run(world)
    assert _sha(old) == before and summ["old_db_unchanged"], "the old file must stay byte-identical"
    con = sqlite3.connect(out)
    con.row_factory = sqlite3.Row
    # the extraction was run on the new chunks only
    assert summ["extraction"]["chunks"] == 1 and summ["extraction"]["live_llm_calls"] == 1
    # concept merge decisions are all recorded; exact merges and one new concept
    md = {r["new_label"] if "new_label" in r.keys() else r[2]: r for r in con.execute("SELECT * FROM concept_merge_log")}
    assert md["tầng đối lưu"]["decision"] == "merged" and md["tầng đối lưu"]["method"] == "exact"
    assert md["tầng đối lưu"]["old_concept_id"] == "geography:c:tang_doi_luu"
    assert md["khí áp"]["decision"] == "kept_separate" and md["gió mùa"]["decision"] == "kept_separate"
    # facts: exact text -> agrees (provenance accumulates on the OLD fact), other number -> conflict (hidden), new
    v = summ["verdicts"]["fact"]
    assert v == {"agrees": 1, "conflicts": 1, "new": 1}
    old_fact = json.loads(con.execute("SELECT json FROM facts WHERE assertion_id='geography:a:fx01'").fetchone()[0])
    assert f"{NEWB}:p0000:u000" in old_fact["source_span_ids"] and f"{GEO}:p0000:u001" in old_fact["source_span_ids"]
    cf = con.execute("SELECT * FROM item_conflicts").fetchall()
    assert len(cf) == 1 and cf[0]["old_item_id"] in ("geography:a:lapse", "geography:rule:fx_lapse_rate") and cf[0]["verdict"] == "conflicts" and cf[0]["new_book_id"] == NEWB
    ev = json.loads(cf[0]["evidence_json"])
    assert "0,65" in ev["new_text"] and "0,6 " in ev["old_text"] and ev["new_source"]["publisher"] == "NXB B (giả lập)"
    st = con.execute("SELECT valid, status FROM item_status WHERE item_id=?", (cf[0]["new_item_id"],)).fetchone()
    assert (st["valid"], st["status"]) == (0, "conflict_review")
    # the store API: hidden conflict is invisible, trusted old fact + new facts are visible
    s = SqliteOntologyStore(out, embedder=HashEmbedder())
    assert s.valid_only and s.get_fact(cf[0]["new_item_id"]) is None and s.get_fact("geography:a:lapse") is not None
    assert SqliteOntologyStore(out, valid_only=False).get_fact(cf[0]["new_item_id"]) is not None
    hits = [h["id"] for h in s.search_knowledge("nhiệt độ giảm khi lên cao 100 m", subject="geography", kinds=("fact",), top_k=20)]
    assert cf[0]["new_item_id"] not in hits
    # new items keep their own span links, own provenance
    newfacts = con.execute("SELECT i.item_id FROM item_sources i WHERE i.book_id=? AND i.item_kind='fact' AND i.role IN ('origin','alternative')", (NEWB,)).fetchall()
    assert len(newfacts) == 2                                                          # new + hidden conflict
    for (iid,) in newfacts:
        spans_ = [r[0] for r in con.execute("SELECT span_id FROM evidence_links WHERE item_kind='fact' AND item_id=?", (iid,))]
        assert spans_ and all(x.startswith(NEWB) for x in spans_)
    # provenance: book ledger (both sides) and self-contained spans
    bs = {r["book_id"]: r for r in con.execute("SELECT * FROM book_sources")}
    assert bs[NEWB]["role"] == "ingested" and bs[NEWB]["publisher"] == "NXB B (giả lập)" and bs[GEO]["role"] == "base"
    assert s.span_exists(f"{NEWB}:p0000:u001") and s.span_exists(f"{GEO}:p0000:u002")
    # merged concept: new label kept as alias is not needed (same label); new concepts have the new book
    assert s.get_concept("geography:c:khi_ap") is not None
    assert s.get_concept("geography:c:tang_doi_luu").source_span_ids[-1].startswith(NEWB) or \
        any(x.startswith(NEWB) for x in s.get_concept("geography:c:tang_doi_luu").source_span_ids)
    # version lineage + changelog + review list
    info = s.build_info()
    assert info["parent_build_id"] == "fixture-v1" and info["build_id"] == summ["build_id"]
    log_rows = [json.loads(l) for l in Path(summ["changelog"]).read_text(encoding="utf-8").splitlines()]
    assert {r["op"] for r in log_rows} >= {"added", "provenance_added", "hidden_conflict"}
    rev = [json.loads(l) for l in Path(summ["conflicts_file"]).read_text(encoding="utf-8").splitlines()]
    assert len(rev) == 1 and rev[0]["evidence"]["new_source"]["edition"] == "2025"
    assert Path(str(out.with_suffix("")) + ".summary.json").exists()
    # the claim LLM was used only for the uncertain pair (one number-conflict confirmation)
    assert 1 <= summ["claim_judge"]["pairs"] <= 3 and summ["claim_judge"]["llm_calls"] == 1
    con.close()


def test_llm_agreement_with_different_wording_keeps_a_visible_variant(world):
    def ext(messages):
        d = json.loads(extractor_handler(messages))
        d["facts"] = [{"text": "Trong tầng đối lưu, nhiệt độ không khí giảm khoảng 0,6 °C cứ mỗi 100 m lên cao theo cách trình bày của sách B.",
                       "concepts": ["nhiệt độ không khí", "tầng đối lưu"], "spans": ["S2"]}]
        return json.dumps(d, ensure_ascii=False)
    out, summ = run(world, name="var.sqlite3", extractor_handler=ext)
    assert summ["verdicts"]["fact"] == {"agrees": 1}
    assert summ["handling"].get("add_variant") == 1
    con = sqlite3.connect(out)
    rows = con.execute("SELECT item_id FROM item_sources WHERE item_kind='fact' AND role='origin' AND book_id=?", (NEWB,)).fetchall()
    assert len(rows) == 1                                                              # the variant is its own visible item
    s = SqliteOntologyStore(out)
    assert s.get_fact(rows[0][0]) is not None and con.execute("SELECT COUNT(*) FROM item_conflicts").fetchone()[0] == 0
    old = json.loads(con.execute("SELECT json FROM facts WHERE assertion_id='geography:a:lapse'").fetchone()[0])
    assert any(x.startswith(NEWB) for x in old["source_span_ids"])                       # and the old item gained a second source


def test_short_symbol_label_with_other_type_is_not_merged(world):
    def ext(messages):
        d = json.loads(extractor_handler(messages))
        d["concepts"].append({"label": "khí", "concept_type": "Quantity", "spans": ["S3"]})        # old fixture: nothing called "khí"
        d["concepts"].append({"label": "mưa", "concept_type": "Substance", "spans": ["S3"]})        # exact label of an old Phenomenon, 3 letters
        return json.dumps(d, ensure_ascii=False)
    out, summ = run(world, name="short.sqlite3", extractor_handler=ext)
    con = sqlite3.connect(out)
    r = con.execute("SELECT decision, method FROM concept_merge_log WHERE new_label='mưa'").fetchone()
    assert r[0] == "ambiguous" and "short_label_type_mismatch" in r[1]


def test_include_conflicts_marks_alternative_source(world):
    out, summ = run(world, include_conflicts=True, name="alt.sqlite3")
    con = sqlite3.connect(out)
    row = con.execute("SELECT i.item_id, i.valid, i.status FROM item_status i JOIN item_conflicts c ON c.new_item_id=i.item_id").fetchone()
    assert row[1:] == (1, "alternative_source")
    s = SqliteOntologyStore(out)
    assert s.get_fact(row[0]) is not None
    f = s.get_fact(row[0])
    assert "ingest:alternative_source" in f.validation.notes
    assert con.execute("SELECT handling FROM item_conflicts").fetchone()[0] == "alternative"


def test_lexical_only_uses_no_claim_llm(world):
    out, summ = run(world, lexical_only=True, name="lex.sqlite3")
    assert summ["claim_judge"]["llm_calls"] == 0
    assert summ["verdicts"]["fact"].get("conflicts") == 1                              # same wording, other number: flagged lexically


def test_cross_kind_fact_vs_old_rule(world):
    # an old RULE that states the lapse rate with a number; the new FACT with another number must be compared with it
    tmp, old, _ = world
    from otter.schemas.ontology import Atom, ConditionalRule
    from otter.storage.fixtures import _ok
    w = OntologyWriter(old)
    r = ConditionalRule(rule_id="geography:rule:lapse2", subject="geography",
                        premises=[Atom(predicate="p", args=["geography:c:tang_doi_luu"], text="lên cao 100 m trong tầng đối lưu")],
                        conclusions=[Atom(predicate="q", args=[], text="nhiệt độ không khí giảm khoảng 0,6 °C")],
                        source_span_ids=[f"{GEO}:p0000:u002"], validation=_ok())
    w.add_rule(r, grade=10)
    w.close()
    out, summ = run(world, name="ck.sqlite3", cross_kind=True)
    con = sqlite3.connect(out)
    assert summ["verdicts"]["fact"]["conflicts"] >= 1
    assert con.execute("SELECT COUNT(*) FROM item_conflicts WHERE old_item_id IN ('geography:a:lapse','geography:rule:lapse2')").fetchone()[0] >= 1


def test_no_accumulate_leaves_old_items_byte_identical(world):
    tmp, old, _ = world
    out, summ = run(world, accumulate_provenance=False, merge_aliases=False, name="na.sqlite3")
    a, b = sqlite3.connect(old), sqlite3.connect(out)
    for t, pk in (("facts", "assertion_id"), ("concepts", "concept_id"), ("rules", "rule_id"), ("semantic_relations", "relation_id")):
        old_rows = dict(a.execute(f"SELECT {pk}, json FROM {t}").fetchall())
        new_rows = dict(b.execute(f"SELECT {pk}, json FROM {t}").fetchall())
        assert all(new_rows[k] == v for k, v in old_rows.items())
    assert summ["old_items_touched"] == 0


def test_refuses_overwrite_reingest_and_protected(world):
    tmp, old, new = world
    kw = dict(old_prov=old, cache_dir=tmp / "c")
    with pytest.raises(ValueError):
        ingest_books(IngestConfig(old_db=old, new_spans=new, out=old, **kw))
    out, _ = run(world)
    with pytest.raises(ValueError):                                                    # existing out without --force
        ingest_books(IngestConfig(old_db=old, new_spans=new, out=out, **kw), extractor_llm=MockLLM(handler=extractor_handler))
    with pytest.raises(ValueError):                                                    # a book already in the ontology
        d = tmp / "nb2" / GEO
        d.mkdir(parents=True)
        (d / "spans.jsonl").write_text(json.dumps({"span_id": f"{GEO}:p0000:u001", "book_id": GEO, "pdf_page_index": 0, "start_offset": 0,
                                                   "end_offset": 3, "text": "abc def", "source_file_sha256": "x"}), encoding="utf-8")
        ingest_books(IngestConfig(old_db=old, new_spans=tmp / "nb2", out=tmp / "o2.sqlite3", **kw))
    with pytest.raises(ValueError):
        ingest_books(IngestConfig(old_db=old, new_spans=new, out=tmp / "o3.sqlite3", workers=9, **kw))


def test_dry_run_counts_chunks_and_writes_nothing(world):
    tmp, old, new = world
    out = tmp / "dry" / "n.sqlite3"
    res = ingest_books(IngestConfig(old_db=old, new_spans=new, out=out, old_prov=old, cache_dir=tmp / "cache", dry_run=True),
                       config={"llm": {"main": {"model": "ext-mock"}}, "paths": {"db": "x"}})
    assert res["dry_run"] and res["chunks"] == 1 and res["chunks_to_extract"] == 1 and not out.exists()


def test_second_run_reuses_caches(world):
    out, s1 = run(world, name="r1.sqlite3")
    out2, s2 = run(world, name="r2.sqlite3")
    assert s2["extraction"]["live_llm_calls"] == 0 and s2["extraction"]["cached_chunks"] == 1
    assert s2["claim_judge"]["llm_calls"] == 0 and s2["verification"]["live_llm_calls"] == 0
    assert s2["verdicts"] == s1["verdicts"]


# ------------------------------------------------------------------------------------------- index extension
def test_incremental_bm25_equals_full_rebuild():
    from rank_bm25 import BM25Okapi
    docs = [["a", "b", "c", "a"], ["b", "d"], ["e", "a", "a", "a", "f"], ["b", "c", "d", "e"], ["x"]]
    full = BM25Okapi(docs)
    inc = IncrementalBM25(docs[:2])
    inc.add(docs[2:4])
    inc.add(docs[4:])
    for q in (["a"], ["b", "d"], ["zzz"], ["a", "x", "e"]):
        assert inc.get_scores(q) == pytest.approx(list(full.get_scores(q)))
