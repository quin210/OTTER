"""Synthetic fixture DB for offline tests (WS3 tools, WS5 runner, WS2 builder).

All lesson text below is SELF-WRITTEN for unit tests (not copied from any textbook). Book ids are
prefixed `fx_` and marked `synthetic: true` in metadata so they can never be confused with real
sources. The fixture contains, per subject:

* spans with exact offsets into a synthetic page text, sections, book metadata (no page labels);
* concepts, typed relations (incl. multi-hop paths), facts, learning objectives, pedagogical links;
* VALID rules (strict-executable), INTENTIONALLY INCOMPLETE rules (a guard stated in the span was
  not captured -> `possible_missing_guard`, partially_supported, not strict) and an UNSUPPORTED rule
  (cited span does not state it -> excluded from strict proof), and a formula rule with variables.

Use: `from otter.storage.fixtures import build_fixture_db; build_fixture_db(tmp_path / "fx.sqlite3")`
then `SqliteOntologyStore(path)`.  `FIXTURE_IDS` lists ids that tests can rely on.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

from otter.schemas.ontology import (Atom, Concept, ConditionalRule, Fact, LearningObjective, PedagogicalLink,
                                    SemanticRelation, Validation)
from otter.storage.embeddings import Embedder, HashEmbedder
from otter.storage.schema import FIXTURE_PROVENANCE_DDL
from otter.storage.textutil import slugify
from otter.storage.writer import OntologyWriter, concept_text, rule_text

BIO = "fx_bio11_synthetic"
GEO = "fx_geo10_synthetic"

# (local key, pedagogical_type, text)   -- self-written synthetic content
LESSONS = {
    BIO: {
        "title": "Sinh học 11 (tài liệu tổng hợp dùng cho kiểm thử)", "subject": "biology", "grade": 11,
        "sections": [
            ("l01", "Bài 1. Quang hợp ở thực vật (tổng hợp)", [
                ("u000", "objective", "Sau bài học này, em sẽ: nêu được khái niệm quang hợp; giải thích được "
                                      "ảnh hưởng của cường độ ánh sáng đến cường độ quang hợp."),
                ("u001", "definition", "Quang hợp là quá trình thực vật sử dụng năng lượng ánh sáng để tổng hợp "
                                       "chất hữu cơ từ khí carbon dioxide và nước, đồng thời giải phóng khí oxygen."),
                ("u002", "concept_explanation", "Lục lạp là bào quan thực hiện quang hợp. Trong lục lạp có diệp lục, "
                                                "sắc tố hấp thụ năng lượng ánh sáng."),
                ("u003", "principle", "Khi các yếu tố khác không bị giới hạn, nếu cường độ ánh sáng tăng nhưng chưa "
                                      "đạt điểm bão hòa ánh sáng thì cường độ quang hợp tăng."),
                ("u004", "principle", "Khi cường độ ánh sáng vượt quá điểm bão hòa ánh sáng, cường độ quang hợp "
                                      "không tăng thêm."),
                ("u005", "condition", "Trong một giới hạn nhất định, khi nồng độ khí carbon dioxide tăng thì cường "
                                      "độ quang hợp tăng."),
                ("u006", "exercise", "Câu hỏi: Vì sao người trồng rau trong nhà kính có thể bổ sung ánh sáng nhân "
                                     "tạo vào những ngày ít nắng?"),
            ]),
            ("l02", "Bài 2. Hô hấp tế bào (tổng hợp)", [
                ("u007", "definition", "Hô hấp tế bào là quá trình phân giải chất hữu cơ để giải phóng năng lượng "
                                       "cho tế bào; quá trình này diễn ra chủ yếu ở ti thể."),
                ("u008", "concept_explanation", "Chất hữu cơ được tạo ra trong quang hợp là nguyên liệu cho hô hấp "
                                                "tế bào."),
            ]),
        ],
    },
    GEO: {
        "title": "Địa lí 10 (tài liệu tổng hợp dùng cho kiểm thử)", "subject": "geography", "grade": 10,
        "sections": [
            ("l01", "Bài 1. Nhiệt độ không khí và mưa (tổng hợp)", [
                ("u000", "objective", "Sau bài học này, em sẽ: trình bày được sự thay đổi nhiệt độ không khí theo "
                                      "độ cao; giải thích được sự hình thành mưa ở sườn đón gió."),
                ("u001", "concept_explanation", "Tầng đối lưu là tầng thấp nhất của khí quyển."),
                ("u002", "law", "Trong tầng đối lưu, nhiệt độ không khí giảm dần theo độ cao, trung bình cứ lên cao "
                                "100 m thì nhiệt độ giảm khoảng 0,6 °C."),
                ("u003", "process", "Khi gió ẩm từ biển thổi tới gặp sườn núi, không khí bị đẩy lên cao, nhiệt độ "
                                    "giảm, hơi nước ngưng tụ và gây mưa ở sườn đón gió."),
                ("u004", "process", "Ở sườn khuất gió, không khí đi xuống, nóng lên và khô nên lượng mưa ít."),
                ("u005", "condition", "Ở vùng đồi núi dốc, khi lớp phủ thực vật bị suy giảm, mưa lớn trong thời gian "
                                      "ngắn làm tăng dòng chảy mặt và nguy cơ lũ quét."),
                ("u006", "exercise", "Câu hỏi: Một ngọn núi cao 2 000 m, nhiệt độ ở chân núi là 25 °C. Hãy ước tính "
                                     "nhiệt độ ở đỉnh núi."),
            ]),
        ],
    },
}


def _sid(book: str, page: int, u: str) -> str:
    return f"{book}:p{page:04d}:{u}"


def _ok(notes: Optional[list[str]] = None, support: str = "supported", score: float = 0.9) -> Validation:
    return Validation(schema_valid=True, source_ids_exist=True, source_support=support, support_score=score,
                      human_verified=None, notes=["fixture"] + (notes or []))


def _a(pred: str, text: str, args: Optional[list[str]] = None, negated: bool = False) -> Atom:
    return Atom(predicate=pred, args=args or [], text=text, negated=negated)


def _write_provenance(db: sqlite3.Connection) -> dict[str, str]:
    """Create books/sections/spans/spans_fts; return {local_key: span_id}."""
    db.executescript(FIXTURE_PROVENANCE_DDL)
    ids: dict[str, str] = {}
    for book_id, b in LESSONS.items():
        pages = []
        page_texts = []
        for p, (lkey, title, units) in enumerate(b["sections"]):
            text, offs = "", []
            for u, ped, t in units:
                if text:
                    text += "\n\n"
                offs.append((u, ped, len(text), len(text) + len(t), t))
                text += t
            pages.append((p, lkey, title, offs))
            page_texts.append(text)
        sha = hashlib.sha256("\f".join(page_texts).encode("utf-8")).hexdigest()
        meta = {"book_id": book_id, "title": b["title"], "subject": b["subject"], "subject_normalized": b["subject"],
                "grade": b["grade"], "publisher": None, "edition_year": None, "series": None,
                "synthetic": True, "sha256": sha,
                "license_notes": "Self-written synthetic fixture for unit tests; not a real textbook."}
        db.execute("INSERT OR REPLACE INTO books VALUES(?,?,?)", (book_id, json.dumps(meta, ensure_ascii=False), sha))
        for p, lkey, title, offs in pages:
            sec = f"{book_id}:{lkey}"
            smeta = {"section_id": sec, "book_id": book_id, "level": "lesson", "title": title, "parent_id": None,
                     "pdf_page_start": p, "pdf_page_end": p, "heading_path": [title]}
            db.execute("INSERT OR REPLACE INTO sections VALUES(?,?,?,?,?,?,?,?)",
                       (sec, book_id, title, p, p, None, None, json.dumps(smeta, ensure_ascii=False)))
            for u, ped, s, e, t in offs:
                sid = _sid(book_id, p, u)
                ids[f"{book_id}:{u}"] = sid
                db.execute("INSERT OR REPLACE INTO spans VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (sid, book_id, p, p, None, 0, sec, t, sha, s, e, json.dumps([title], ensure_ascii=False),
                            ped, "fixture-v1", "extracted"))
                db.execute("DELETE FROM spans_fts WHERE span_id=?", (sid,))
                db.execute("INSERT INTO spans_fts VALUES(?,?,?)", (sid, book_id, t))
    db.commit()
    return ids


def _cid(subject: str, label: str) -> str:
    return f"{subject}:c:{slugify(label)}"


def build_fixture_db(path: str | Path, *, embedder: Optional[Embedder] = None,
                     with_embeddings: bool = True) -> Path:
    """Create (overwrite) a self-contained fixture DB (provenance + ontology tables) at `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    w = OntologyWriter(path)
    S = _write_provenance(w.db)

    def sp(book: str, *us: str) -> list[str]:
        return [S[f"{book}:{u}"] for u in us]

    concepts: list[Concept] = []

    def C(subject, grade, label, ctype, book, units, definition=None, aliases=None) -> str:
        c = Concept(concept_id=_cid(subject, label), label=label, aliases=aliases or [], concept_type=ctype,
                    subject=subject, grade=grade, definition=definition, source_span_ids=sp(book, *units),
                    validation=_ok())
        concepts.append(c)
        return c.concept_id

    # ---------------- biology
    b, g = "biology", 11
    qh = C(b, g, "quang hợp", "Process", BIO, ["u001"], "quá trình thực vật sử dụng năng lượng ánh sáng để tổng hợp "
           "chất hữu cơ từ khí carbon dioxide và nước, đồng thời giải phóng khí oxygen")
    ll = C(b, g, "lục lạp", "Structure", BIO, ["u002"], "bào quan thực hiện quang hợp")
    dl = C(b, g, "diệp lục", "Substance", BIO, ["u002"], "sắc tố hấp thụ năng lượng ánh sáng", aliases=["chất diệp lục"])
    clh = C(b, g, "chất hữu cơ", "Substance", BIO, ["u001", "u008"])
    co2 = C(b, g, "carbon dioxide", "Substance", BIO, ["u001", "u005"], aliases=["khí carbon dioxide", "CO2"])
    o2 = C(b, g, "oxygen", "Substance", BIO, ["u001"], aliases=["khí oxygen"])
    cdas = C(b, g, "cường độ ánh sáng", "Quantity", BIO, ["u003"])
    cdqh = C(b, g, "cường độ quang hợp", "Quantity", BIO, ["u003"])
    dbh = C(b, g, "điểm bão hòa ánh sáng", "State", BIO, ["u003", "u004"])
    hh = C(b, g, "hô hấp tế bào", "Process", BIO, ["u007"], "quá trình phân giải chất hữu cơ để giải phóng năng lượng "
           "cho tế bào")
    tt = C(b, g, "ti thể", "Structure", BIO, ["u007"])
    # ---------------- geography
    G, gg = "geography", 10
    tdl = C(G, gg, "tầng đối lưu", "Structure", GEO, ["u001"], "tầng thấp nhất của khí quyển")
    kq = C(G, gg, "khí quyển", "Entity", GEO, ["u001"])
    ndkk = C(G, gg, "nhiệt độ không khí", "Quantity", GEO, ["u002"])
    dc = C(G, gg, "độ cao", "Quantity", GEO, ["u002"])
    sdg = C(G, gg, "sườn đón gió", "Place", GEO, ["u003"])
    skg = C(G, gg, "sườn khuất gió", "Place", GEO, ["u004"])
    gia = C(G, gg, "gió ẩm", "Phenomenon", GEO, ["u003"])
    mua = C(G, gg, "mưa", "Phenomenon", GEO, ["u003", "u004"], aliases=["lượng mưa"])
    lq = C(G, gg, "lũ quét", "Phenomenon", GEO, ["u005"])
    dcm = C(G, gg, "dòng chảy mặt", "Phenomenon", GEO, ["u005"])
    lptv = C(G, gg, "lớp phủ thực vật", "Entity", GEO, ["u005"])
    for c in concepts:
        w.add_concept(c)

    labels = {c.concept_id: c.label for c in concepts}

    def R(subject, grade, s, rtype, o, book, units, status="explicit_fact", support="supported", qualifier=None):
        rid = f"{subject}:rel:{slugify(labels[s])}__{rtype}__{slugify(labels[o])}"
        r = SemanticRelation(relation_id=rid, subject_id=s, relation_type=rtype, object_id=o, status=status,
                             qualifier=qualifier, source_span_ids=sp(book, *units),
                             validation=_ok(support=support, score=0.9 if support == "supported" else 0.4))
        w.add_relation(r, subject=subject, grade=grade, text=f"{labels[s]} {rtype} {labels[o]}")
        return rid

    R(b, g, qh, "occurs_in", ll, BIO, ["u002"])
    R(b, g, dl, "part_of", ll, BIO, ["u002"])
    R(b, g, qh, "involves", co2, BIO, ["u001"])
    R(b, g, qh, "involves", clh, BIO, ["u001"])
    R(b, g, cdas, "affects", cdqh, BIO, ["u003"])
    R(b, g, hh, "occurs_in", tt, BIO, ["u007"])
    R(b, g, hh, "involves", clh, BIO, ["u007", "u008"])
    R(b, g, hh, "contrasts_with", qh, BIO, ["u007"], status="hypothesized_relation", support="partially_supported")
    R(G, gg, tdl, "part_of", kq, GEO, ["u001"])
    R(G, gg, dc, "affects", ndkk, GEO, ["u002"], qualifier="trong tầng đối lưu")
    R(G, gg, gia, "causes", mua, GEO, ["u003"], qualifier="ở sườn đón gió")
    R(G, gg, mua, "occurs_in", sdg, GEO, ["u003"])
    R(G, gg, dcm, "causes", lq, GEO, ["u005"])
    R(G, gg, sdg, "contrasts_with", skg, GEO, ["u003", "u004"], status="derived_conclusion")

    def F(subject, grade, key, text, cids, book, units, status="explicit_fact", support="supported"):
        f = Fact(assertion_id=f"{subject}:a:{key}", text=text, subject=subject, concept_ids=cids, status=status,
                 source_span_ids=sp(book, *units), validation=_ok(support=support))
        w.add_fact(f, grade=grade)
        return f.assertion_id

    F(b, g, "fx01", "Quang hợp giải phóng khí oxygen.", [qh, o2], BIO, ["u001"])
    F(b, g, "fx02", "Diệp lục là sắc tố hấp thụ năng lượng ánh sáng.", [dl], BIO, ["u002"])
    F(b, g, "fx03", "Chất hữu cơ được tạo ra trong quang hợp là nguyên liệu cho hô hấp tế bào.", [clh, qh, hh],
      BIO, ["u008"])
    F(G, gg, "fx01", "Tầng đối lưu là tầng thấp nhất của khí quyển.", [tdl, kq], GEO, ["u001"])
    F(G, gg, "fx02", "Ở sườn khuất gió, lượng mưa ít.", [skg, mua], GEO, ["u004"])

    def RU(subject, grade, key, rtype, prem, concl, guards=(), exceptions=(), book=None, units=(), notes=None,
           support="supported", status="explicit_rule", formula=None, variables=None, cids=(), objs=()):
        r = ConditionalRule(rule_id=f"{subject}:rule:{key}", subject=subject, rule_type=rtype, premises=list(prem),
                            guards=list(guards), conclusions=list(concl), exceptions=list(exceptions), formula=formula,
                            variables=variables or {}, learning_objective_ids=list(objs), status=status,
                            source_span_ids=sp(book, *units),
                            validation=_ok(notes=notes, support=support,
                                           score={"supported": 0.9, "partially_supported": 0.5}.get(support, 0.1)))
        w.add_rule(r, grade=grade, concept_ids=cids)
        return r.rule_id

    obj_bio = f"{b}:obj:fx01"
    obj_geo = f"{G}:obj:fx01"
    # VALID: all premises + guards captured from u003
    RU(b, g, "fx_light_valid", "conditional",
       [_a("light_intensity_increases", "cường độ ánh sáng tăng", [cdas])],
       [_a("photosynthesis_rate_increases", "cường độ quang hợp tăng", [cdqh])],
       guards=[_a("below_light_saturation", "cường độ ánh sáng chưa đạt điểm bão hòa ánh sáng", [dbh]),
               _a("other_factors_non_limiting", "các yếu tố khác không bị giới hạn")],
       book=BIO, units=["u003"], cids=[cdas, cdqh, qh], objs=[obj_bio])
    # VALID: saturation -> no further increase (negated conclusion explicitly stated in text)
    RU(b, g, "fx_light_saturation", "conditional",
       [_a("light_above_saturation", "cường độ ánh sáng vượt quá điểm bão hòa ánh sáng", [cdas, dbh])],
       [_a("photosynthesis_rate_increases", "cường độ quang hợp tăng thêm", [cdqh], negated=True)],
       book=BIO, units=["u004"], cids=[cdqh], objs=[obj_bio])
    # INCOMPLETE: guard "trong một giới hạn nhất định" (u005) was not captured
    RU(b, g, "fx_co2_incomplete", "conditional",
       [_a("co2_concentration_increases", "nồng độ khí carbon dioxide tăng", [co2])],
       [_a("photosynthesis_rate_increases", "cường độ quang hợp tăng", [cdqh])],
       book=BIO, units=["u005"], notes=["possible_missing_guard"], support="partially_supported", cids=[co2, cdqh])
    # UNSUPPORTED: cited span u001 does not state this
    RU(b, g, "fx_water_unsupported", "causal",
       [_a("plant_watered_heavily", "cây được tưới nhiều nước")],
       [_a("photosynthesis_rate_increases", "cường độ quang hợp luôn tăng", [cdqh])],
       book=BIO, units=["u001"], notes=["excluded_from_strict_proof"], support="unsupported",
       status="hypothesized_relation")
    # VALID chain: photosynthesis product -> respiration input
    RU(b, g, "fx_resp_input", "causal",
       [_a("organic_matter_produced", "chất hữu cơ được tạo ra trong quang hợp", [clh, qh])],
       [_a("respiration_substrate_available", "có nguyên liệu cho hô hấp tế bào", [hh])],
       book=BIO, units=["u008"], cids=[clh, hh])
    # VALID formula rule
    RU(G, gg, "fx_lapse_rate", "formula",
       [_a("in_troposphere", "ở trong tầng đối lưu", [tdl]),
        _a("altitude_increases", "lên cao thêm một độ cao Δh", [dc])],
       [_a("air_temperature_decreases", "nhiệt độ không khí giảm khoảng 0,6 °C mỗi 100 m", [ndkk])],
       formula="delta_T = -0.6 * delta_h / 100",
       variables={"delta_T": f"{ndkk} (°C)", "delta_h": f"{dc} (m)"},
       book=GEO, units=["u002"], cids=[ndkk, dc, tdl], objs=[obj_geo])
    # VALID orographic rain: two premises
    RU(G, gg, "fx_orographic_rain", "causal",
       [_a("moist_wind_from_sea", "gió ẩm từ biển thổi tới", [gia]),
        _a("air_forced_up_slope", "gặp sườn núi, không khí bị đẩy lên cao")],
       [_a("rain_on_windward_slope", "mưa ở sườn đón gió", [mua, sdg])],
       book=GEO, units=["u003"], cids=[gia, mua, sdg], objs=[obj_geo])
    RU(G, gg, "fx_leeward_dry", "causal",
       [_a("on_leeward_slope", "ở sườn khuất gió, không khí đi xuống", [skg])],
       [_a("little_rain", "lượng mưa ít", [mua])],
       book=GEO, units=["u004"], cids=[skg, mua])
    # VALID flash flood with both guards
    RU(G, gg, "fx_flood_valid", "conditional",
       [_a("heavy_short_rain", "mưa lớn trong thời gian ngắn", [mua])],
       [_a("flash_flood_risk_increases", "tăng dòng chảy mặt và nguy cơ lũ quét", [dcm, lq])],
       guards=[_a("steep_hilly_area", "ở vùng đồi núi dốc"),
               _a("vegetation_cover_reduced", "lớp phủ thực vật bị suy giảm", [lptv])],
       book=GEO, units=["u005"], cids=[mua, lq, dcm, lptv])
    # INCOMPLETE: same conclusion, guard "vùng đồi núi dốc" missing
    RU(G, gg, "fx_flood_incomplete", "conditional",
       [_a("heavy_short_rain", "mưa lớn trong thời gian ngắn", [mua])],
       [_a("flash_flood_risk_increases", "tăng nguy cơ lũ quét", [lq])],
       guards=[_a("vegetation_cover_reduced", "lớp phủ thực vật bị suy giảm", [lptv])],
       book=GEO, units=["u005"], notes=["possible_missing_guard"], support="partially_supported", cids=[mua, lq])

    for oid, subject, grade, book, text in (
            (obj_bio, b, g, BIO, "Nêu được khái niệm quang hợp; giải thích được ảnh hưởng của cường độ ánh sáng đến "
                                 "cường độ quang hợp."),
            (obj_geo, G, gg, GEO, "Trình bày được sự thay đổi nhiệt độ không khí theo độ cao; giải thích được sự hình "
                                  "thành mưa ở sườn đón gió.")):
        w.add_objective(LearningObjective(objective_id=oid, text=text, lesson_section_id=f"{book}:l01",
                                          subject=subject, grade=grade, source_span_ids=sp(book, "u000")))
    for src, lt, tgt, book in ((qh, "teaches", obj_bio, BIO), (f"{b}:rule:fx_light_valid", "teaches", obj_bio, BIO),
                               (f"{G}:rule:fx_lapse_rate", "teaches", obj_geo, GEO), (ndkk, "teaches", obj_geo, GEO)):
        w.add_link(PedagogicalLink(link_id=f"link:{slugify(src)}:{lt}:{slugify(tgt)}", source_id=src, link_type=lt,
                                   target_id=tgt, source_span_ids=sp(book, "u000")))

    if with_embeddings:
        emb = embedder or HashEmbedder()
        for kind, table, pk, textf in (
                ("concept", "concepts", "concept_id", lambda d: concept_text(Concept.model_validate(d))),
                ("fact", "facts", "assertion_id", lambda d: d["text"]),
                ("rule", "rules", "rule_id", lambda d: rule_text(ConditionalRule.model_validate(d)))):
            rows = [json.loads(r[0]) for r in w.db.execute(f"SELECT json FROM {table} ORDER BY {pk}")]
            if not rows:
                continue
            vecs = emb.embed([textf(d) for d in rows])
            grades = {r[0]: r[1] for r in w.db.execute(f"SELECT {pk}, grade FROM {table}")}
            w.add_embeddings(kind, emb.model, [(d[pk], d["subject"], grades.get(d[pk])) for d in rows], vecs)
        rows = w.db.execute("SELECT span_id, book_id, text FROM spans ORDER BY span_id").fetchall()
        vecs = emb.embed([r[2] for r in rows])
        w.add_embeddings("span", emb.model, [(r[0], r[1], None) for r in rows], vecs)
    w.add_build_version("fixture-v1", extractor="fixture", verifier="fixture",
                        embedding_model=(embedder or HashEmbedder()).model if with_embeddings else "",
                        notes="synthetic fixture")
    w.close()
    return path


FIXTURE_IDS = {
    "books": [BIO, GEO],
    "valid_rules": ["biology:rule:fx_light_valid", "biology:rule:fx_light_saturation", "biology:rule:fx_resp_input",
                    "geography:rule:fx_lapse_rate", "geography:rule:fx_orographic_rain",
                    "geography:rule:fx_leeward_dry", "geography:rule:fx_flood_valid"],
    "incomplete_rules": ["biology:rule:fx_co2_incomplete", "geography:rule:fx_flood_incomplete"],
    "unsupported_rules": ["biology:rule:fx_water_unsupported"],
    "formula_rules": ["geography:rule:fx_lapse_rate"],
    "concepts": ["biology:c:quang_hop", "biology:c:luc_lap", "biology:c:ho_hap_te_bao",
                 "geography:c:tang_doi_luu", "geography:c:mua"],
    "exercise_spans": [f"{BIO}:p0000:u006", f"{GEO}:p0000:u006"],
    "missing_span": f"{BIO}:p0099:u999",
}


if __name__ == "__main__":  # python -m otter.storage.fixtures [path]
    import sys
    out = build_fixture_db(sys.argv[1] if len(sys.argv) > 1 else "data/ontology/fixture.sqlite3")
    print(out)
