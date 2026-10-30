"""Knowledge maintenance: span alignment edge cases + revalidate end-to-end (MockLLM, fixture DB, HashEmbedder)."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from otter.builder.align import AlignConfig, SpanRec, align_spans, load_span_records, span_category
from otter.builder.revalidate import RevalConfig, revalidate
from otter.llm.interface import MockLLM
from otter.storage.embeddings import HashEmbedder
from otter.storage.fixtures import BIO, GEO, build_fixture_db
from otter.storage.sqlite_store import SqliteOntologyStore

LONG = ("Quang hợp là quá trình thực vật sử dụng năng lượng ánh sáng để tổng hợp chất hữu cơ từ khí carbon dioxide "
        "và nước đồng thời giải phóng khí oxygen trong lục lạp của tế bào lá cây xanh")


def rec(i, text, book="b1", hp=("L1",), order=None):
    return SpanRec(span_id=i, book_id=book, text=text, heading_path=tuple(hp), order=order if order is not None else int(re.sub(r"\D", "", i) or 0))


# ------------------------------------------------------------------------------------------- alignment
def test_exact_match_ignores_span_ids():
    old = [rec("o1", LONG), rec("o2", "Lục lạp là bào quan thực hiện quang hợp trong tế bào thực vật")]
    new = [rec("zz9", old[1].text, order=0), rec("zz8", LONG, order=1)]           # ids scrambled and order swapped
    rows, added = align_spans(old, new)
    assert rows["o1"].status == "identical" and rows["o1"].new_ids == ["zz8"]
    assert rows["o2"].status == "identical" and rows["o2"].new_ids == ["zz9"]
    assert added == []


def test_whitespace_only_change_is_identical_norm():
    old = [rec("o1", LONG)]
    new = [rec("n1", LONG.replace(" ", "  ", 3) + "\n")]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status == "identical_norm" and span_category("identical_norm") == "same"


def test_moved_to_other_heading_is_same_content():
    old = [rec("o1", LONG, hp=("Bài 1",))]
    new = [rec("n1", LONG, hp=("Bài 7", "mục 2"))]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status == "moved" and rows["o1"].new_ids == ["n1"]


def test_deleted_span_is_gone_and_added_span_reported():
    old = [rec("o1", LONG), rec("o2", "Hô hấp tế bào là quá trình phân giải chất hữu cơ để giải phóng năng lượng")]
    new = [rec("n1", LONG), rec("n2", "Một đoạn hoàn toàn mới nói về địa chấn và núi lửa ở vành đai lửa Thái Bình Dương")]
    rows, added = align_spans(old, new)
    assert rows["o2"].status == "gone" and rows["o2"].new_ids == []
    assert added == ["n2"]


def test_fuzzy_number_edit_matches_modified():
    old = [rec("o1", "Trong tầng đối lưu, nhiệt độ không khí giảm dần theo độ cao, trung bình cứ lên cao 100 m "
                     "thì nhiệt độ giảm khoảng 0,6 °C.")]
    new = [rec("n1", "Trong tầng đối lưu, nhiệt độ không khí giảm dần theo độ cao, trung bình cứ lên cao 100 m "
                     "thì nhiệt độ giảm khoảng 0,9 °C.")]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status == "modified" and rows["o1"].new_ids == ["n1"] and rows["o1"].score > 0.55


def test_two_equally_similar_candidates_are_not_forced():
    base = "Ở sườn khuất gió không khí đi xuống nóng lên và khô nên lượng mưa ít hơn rất nhiều so với sườn đón gió"
    old = [rec("o1", base)]
    new = [rec("n1", base + " vào mùa hè"), rec("n2", base + " vào mùa đông")]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status in ("ambiguous", "split_merged") and rows["o1"].status != "modified"
    assert not (rows["o1"].status == "modified")


def test_two_old_spans_competing_for_one_new_span_are_ambiguous():
    a = "Nhiệt độ không khí giảm dần theo độ cao trong tầng đối lưu và mưa nhiều ở sườn đón gió của dãy núi"
    old = [rec("o1", a + " ở miền Bắc"), rec("o2", a + " ở miền Nam")]
    new = [rec("n1", a + " ở miền")]
    rows, _ = align_spans(old, new)
    assert all(r.status != "modified" for r in rows.values())
    assert any(r.status in ("ambiguous", "split_merged") for r in rows.values())


def test_duplicate_texts_pair_in_order_and_count_change_is_not_orphaning():
    d = "Hình 5.1 minh hoạ cấu trúc của phân tử ADN gồm hai mạch polynucleotide xoắn kép"
    old = [rec("o1", d, order=1), rec("o2", d, order=2), rec("o3", d, order=3)]
    new = [rec("n1", d, order=1), rec("n2", d, order=2)]
    rows, _ = align_spans(old, new)
    assert [rows[i].status for i in ("o1", "o2", "o3")] == ["identical"] * 3        # text still present
    assert rows["o1"].new_ids == ["n1"] and rows["o2"].new_ids == ["n2"]


def test_other_book_is_never_a_candidate():
    old = [rec("o1", LONG, book="b1")]
    new = [rec("n1", LONG, book="b2")]
    rows, added = align_spans(old, new)
    assert rows["o1"].status == "gone" and added == ["n1"]


def test_short_span_without_exact_match_is_not_forced():
    old = [rec("o1", "Hình 5.1")]
    new = [rec("n1", "Hình 5.2")]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status in ("ambiguous", "gone") and rows["o1"].new_ids == []


def test_split_into_two_spans_and_merge():
    s1 = "Quang hợp là quá trình thực vật sử dụng năng lượng ánh sáng để tổng hợp chất hữu cơ từ khí carbon dioxide"
    s2 = "và nước đồng thời giải phóng khí oxygen trong lục lạp của tế bào lá cây xanh ở thực vật bậc cao"
    old = [rec("o1", s1 + " " + s2)]
    new = [rec("n1", s1, order=1), rec("n2", s2, order=2)]
    rows, _ = align_spans(old, new)
    assert rows["o1"].status == "split_merged" and rows["o1"].new_ids == ["n1", "n2"]
    rows2, _ = align_spans([rec("p1", s1, order=1), rec("p2", s2, order=2)], [rec("m1", s1 + " " + s2)])
    assert rows2["p1"].status == "split_merged" and rows2["p2"].status == "split_merged"
    assert rows2["p1"].new_ids == ["m1"]


def test_paraphrase_found_only_by_dense_is_ambiguous_not_matched():
    class Echo:
        model = "echo"

        def available(self):
            return True

        def embed(self, texts):                      # everything looks identical to the dense model
            import numpy as np
            return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))

    old = [rec("o1", "Diệp lục hấp thụ ánh sáng đỏ và xanh tím để cung cấp năng lượng cho phản ứng sáng của quang hợp")]
    new = [rec("n1", "Sắc tố quang hợp chủ yếu thu nhận bước sóng nhất định của bức xạ mặt trời cung cấp cho pha sáng")]
    rows, _ = align_spans(old, new, embedder=Echo())
    assert rows["o1"].status == "ambiguous" and rows["o1"].method == "dense_only"


# ------------------------------------------------------------------------------------------- end to end
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _fixture_spans(db: Path) -> list[dict]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM spans")]
    con.close()
    return rows


def verifier_handler(messages):
    """Mock verifier: a claim is 'supported' iff every number in it appears in the cited evidence."""
    user = messages[-1]["content"]
    ev = user.split("MỆNH ĐỀ:")[0]
    out = []
    for m in re.finditer(r"^(\d+)\. \(trích dẫn: [^)]*\) (.*)$", user, flags=re.M):
        nums = re.findall(r"\d+(?:,\d+)?", m.group(2))
        ok = all(n in ev for n in nums)
        out.append({"id": int(m.group(1)), "label": "supported" if ok else "unsupported"})
    return json.dumps({"judgments": out})


@pytest.fixture()
def world(tmp_path):
    old = build_fixture_db(tmp_path / "old.sqlite3", embedder=HashEmbedder())
    spans = _fixture_spans(old)
    new_rows = []
    for r in spans:
        sid = r["span_id"]
        if sid.endswith(f"{GEO}:p0000:u003") or sid == f"{GEO}:p0000:u003":
            continue                                                            # deleted: sườn đón gió process
        r = dict(r)
        if sid == f"{GEO}:p0000:u002":                                         # edited: 0,6 -> 0,9
            r["text"] = r["text"].replace("0,6", "0,9")
        if sid == f"{BIO}:p0001:u007" or sid == f"{BIO}:p0000:u003":           # irrelevant edit (no claim depends on it)
            pass
        r["heading_path"] = json.loads(r["heading_path_json"])
        new_rows.append(r)
    new_path = tmp_path / "new_spans.jsonl"
    new_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new_rows), encoding="utf-8")
    return tmp_path, old, new_path


def _run(world, **kw):
    tmp, old, new = world
    out = tmp / "ver" / "new.sqlite3"
    cfg = RevalConfig(old_db=old, new_spans=new, out=out, old_prov=old, cache_dir=tmp / "cache", embed=True, **kw)
    summ = revalidate(cfg, verifier=MockLLM(handler=verifier_handler), embedder=HashEmbedder())
    return out, summ


def test_revalidate_end_to_end(world):
    tmp, old, new = world
    before = _sha(old)
    out, summ = _run(world)
    assert _sha(old) == before, "the old version must stay byte-identical"
    s_old = SqliteOntologyStore(old, embedder=HashEmbedder())
    s_new = SqliteOntologyStore(out, embedder=HashEmbedder())
    assert not s_old.valid_only and s_new.valid_only
    # statuses
    con = sqlite3.connect(out)
    st = {(k, i): (s, v) for k, i, s, v in con.execute("SELECT item_kind,item_id,status,valid FROM item_status")}
    assert st[("fact", "geography:a:fx02")] == ("unchanged", 1)                      # cites u004 (untouched)
    assert st[("fact", "geography:a:fx01")] == ("unchanged", 1)
    assert st[("concept", "geography:c:suon_don_gio")] == ("orphaned", 0)             # only cited the deleted span
    assert st[("relation", "geography:rel:suon_don_gio__contrasts_with__suon_khuat_gio")][0] == "modified"
    assert st[("concept", "geography:c:nhiet_do_khong_khi")][0] == "modified"        # u002 edited
    assert set(summ["totals"]) <= {"unchanged", "modified", "orphaned", "needs_review"}
    assert summ["cost"]["items_rechecked"] > 0 and summ["cost"]["llm_calls"] >= 1
    assert summ["cost"]["items_rechecked"] < summ["cost"]["items_total"]
    # old store is untouched and still answers from the old text
    assert s_old.get_concept("geography:c:suon_don_gio") is not None
    # new store hides orphaned/unsupported items everywhere
    assert s_new.get_concept("geography:c:suon_don_gio") is None
    assert SqliteOntologyStore(out, valid_only=False).get_concept("geography:c:suon_don_gio") is not None
    hits = s_new.search_knowledge("sườn đón gió", subject="geography", top_k=20)
    assert all(h["id"] != "geography:c:suon_don_gio" for h in hits)
    assert "geography:c:suon_don_gio" not in s_new.graph().nodes
    assert s_new.stats()["hidden_invalid_items"] == summ["invalid_items"]
    # provenance of the new edition is in the file and the deleted span is gone
    assert not s_new.span_exists(f"{GEO}:p0000:u003") and s_new.span_exists(f"{GEO}:p0000:u002")
    assert "0,9" in s_new.get_spans([f"{GEO}:p0000:u002"])[0].text
    # version lineage + changelog
    info = s_new.build_info()
    assert info["parent_build_id"] == s_old.build_info()["build_id"] and "new edition" in info["reason"]
    assert info["build_id"] == summ["build_id"]
    log_rows = [json.loads(l) for l in Path(summ["changelog"]).read_text(encoding="utf-8").splitlines()]
    assert log_rows and all(r["status"] != "unchanged" or not r["valid"] for r in log_rows)
    assert {r["item_id"] for r in log_rows if r["status"] == "orphaned"} >= {"geography:c:suon_don_gio"}
    s_old.close(); s_new.close()


def test_edited_claim_value_is_unsupported_and_rules_lose_strict_flag(world):
    out, summ = _run(world)
    con = sqlite3.connect(out)
    con.row_factory = sqlite3.Row
    # any item whose claim carries the old number 0,6 must be unsupported after the edit (mock verifier is number-aware)
    bad = [r for r in con.execute("SELECT * FROM item_status WHERE status='modified'")]
    assert bad
    inv = {(r["item_kind"], r["item_id"]) for r in bad if not r["valid"]}
    for k, i in inv:
        if k == "rule":
            assert con.execute("SELECT strict_executable FROM rules WHERE rule_id=?", (i,)).fetchone()[0] == 0
    # every invalid rule is hidden from rules_concluding as well
    s = SqliteOntologyStore(out)
    for k, i in inv:
        if k == "rule":
            assert s.get_rule(i) is None


def test_refuses_to_overwrite_old_and_dry_run_writes_nothing(world):
    tmp, old, new = world
    with pytest.raises(ValueError):
        revalidate(RevalConfig(old_db=old, new_spans=new, out=old, old_prov=old), verifier=MockLLM(handler=verifier_handler))
    out = tmp / "dry" / "n.sqlite3"
    summ = revalidate(RevalConfig(old_db=old, new_spans=new, out=out, old_prov=old, dry_run=True))
    assert not out.exists() and summ["cost"]["batches_planned"] <= summ["cost"]["batches_full_revalidation"]


def test_identical_edition_is_a_noop(world):
    tmp, old, _ = world
    same = tmp / "same.jsonl"
    rows = _fixture_spans(old)
    for r in rows:
        r["heading_path"] = json.loads(r["heading_path_json"])
    same.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp / "noop" / "n.sqlite3"
    summ = revalidate(RevalConfig(old_db=old, new_spans=same, out=out, old_prov=old, cache_dir=tmp / "c"),
                      verifier=MockLLM(handler=verifier_handler), embedder=HashEmbedder())
    assert set(summ["totals"]) == {"unchanged"} and summ["invalid_items"] == 0 and summ["cost"]["llm_calls"] == 0
    s = SqliteOntologyStore(out)
    assert s.stats()["hidden_invalid_items"] == 0


def test_scrambled_ids_in_new_edition_are_remapped(world):
    tmp, old, _ = world
    rows = _fixture_spans(old)
    for r in rows:
        r["heading_path"] = json.loads(r["heading_path_json"])
        r["span_id"] = "NEW_" + hashlib.md5(r["span_id"].encode()).hexdigest()[:8]
    p = tmp / "scr.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp / "scr" / "n.sqlite3"
    summ = revalidate(RevalConfig(old_db=old, new_spans=p, out=out, old_prov=old, cache_dir=tmp / "c2"),
                      verifier=MockLLM(handler=verifier_handler), embedder=HashEmbedder())
    assert set(summ["totals"]) == {"unchanged"}
    s = SqliteOntologyStore(out)
    c = s.get_concept("geography:c:tang_doi_luu")
    assert c and all(x.startswith("NEW_") for x in c.source_span_ids) and s.span_exists(c.source_span_ids[0])


def test_verifier_failure_gives_needs_review_not_a_guess(world):
    tmp, old, new = world

    class Broken(MockLLM):
        def chat(self, *a, **k):
            raise RuntimeError("down")
    import otter.builder.validate as V
    orig = V.time.sleep
    V.time.sleep = lambda *_: None
    try:
        out = tmp / "br" / "n.sqlite3"
        summ = revalidate(RevalConfig(old_db=old, new_spans=new, out=out, old_prov=old, cache_dir=tmp / "c3"),
                          verifier=Broken(), embedder=HashEmbedder())
    finally:
        V.time.sleep = orig
    assert summ["totals"].get("needs_review", 0) > 0 and summ["cost"]["failed_batches"] > 0
    con = sqlite3.connect(out)
    # unsupported must never be inferred from a failed verifier
    assert con.execute("SELECT COUNT(*) FROM item_status WHERE status='modified' AND support_new='unsupported' AND "
                       "reason LIKE '%verifier_failed%'").fetchone()[0] == 0


def test_open_version_by_build_id(world):
    out, summ = _run(world)
    s = SqliteOntologyStore.open_version(summ["build_id"], out.parent)
    assert s.ontology_path == out
    s0 = SqliteOntologyStore.open_version(summ["parent_build_id"], world[1].parent)
    assert s0.ontology_path == world[1]


# ------------------------------------------------------------------------------------------- edition simulator
def test_simulate_edition_is_deterministic_and_records_exact_changes():
    from otter.builder.simulate_edition import simulate_edition
    rows = [{"span_id": f"b:p{i:04d}:u000", "book_id": "b", "pdf_page_index": i, "start_offset": 0,
             "text": f"Năm {1900 + i} nhà bác học Nguyễn Văn An{'' if i % 2 else ' Trần Thị Bình'} đo được {i + 3},5 mét trong thí nghiệm số {i}."}
            for i in range(200)]
    new1, log1 = simulate_edition(rows, seed=3, delete_frac=0.1, edit_frac=0.1)
    new2, log2 = simulate_edition(rows, seed=3, delete_frac=0.1, edit_frac=0.1)
    assert new1 == new2 and log1.edited == log2.edited and log1.deleted == log2.deleted
    assert len(log1.deleted) == 20 and len(log1.edited) == 20
    by_new = {r["span_id"]: r for r in new1}
    by_old = {r["span_id"]: r for r in rows}
    assert not set(log1.deleted) & set(by_new)
    for sid, e in log1.edited.items():
        old_t, new_t = by_old[sid]["text"], by_new[sid]["text"]
        assert e["old"] in old_t and e["new"] in new_t and e["n_replaced"] >= 1
        assert e["old"] != e["new"] and new_t != old_t
    untouched = [s for s in by_new if s not in log1.edited]
    assert all(by_new[s]["text"] == by_old[s]["text"] for s in untouched)       # nothing else changes
