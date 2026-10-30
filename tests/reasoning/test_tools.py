"""Spec §10 items 1-4 (+ tool contracts) on the synthetic fixture DB, fully offline."""
from __future__ import annotations

import pytest

from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.proof_engine import KnowledgeBase, ProofEngine, parse_literal
from otter.schemas.textbook import SourceSpan
from otter.tools import ToolContext, ToolRegistry

from ws3_helpers import S


def reg_for(store, subject="biology", **kw):
    return ToolRegistry(ToolContext(store=store, subject=subject, **kw), max_total_calls=kw.pop("budget", 100)
                        if "budget" in kw else 100)


# ---------------------------------------------------------------- 1. no fabricated page
def test_missing_page_is_null(store):
    cites, invalid = CitationResolver(store).resolve_many([S["bio_luclap"], S["geo_lapse"]])
    assert not invalid and len(cites) == 2
    for c in cites:
        assert c.page_number is None                       # fixture spans have no verified printed page
        assert c.book_title and "kiểm thử" in c.book_title  # real metadata from the DB, not invented
        assert c.pdf_page_number == 1


def test_unverified_printed_label_not_used_as_page():
    span = SourceSpan(span_id="b:p0003:u001", book_id="b", pdf_page_index=3, printed_page_label="17",
                      printed_page_verified=False, start_offset=0, end_offset=4, text="abcd", source_file_sha256="x")

    class St:
        def get_spans(self, ids):
            return [span] if "b:p0003:u001" in ids else []

        def book_metadata(self, b):
            return {"title": "T", "grade": 10}

        def section(self, s):
            return None

    c = CitationResolver(St()).resolve("b:p0003:u001")
    assert c.page_number is None and c.pdf_page_number == 4
    span.printed_page_verified = True
    assert CitationResolver(St()).resolve("b:p0003:u001").page_number == "17"


# ---------------------------------------------------------------- 2. bad span rejected
def test_unknown_span_cannot_be_citation_verified(store, ids):
    reg = reg_for(store)
    tc = reg.call("verify_evidence", {"claim": "Lục lạp là bào quan thực hiện quang hợp",
                                      "span_ids": [S["bio_luclap"], ids["missing_span"]]})
    r = tc.result
    assert r["structural"]["status"] == "fail" and ids["missing_span"] in r["structural"]["unknown_ids"]
    assert r["status"] == "undetermined" and r["verified_span_ids"] == []
    rs = reg.call("retrieve_source", {"span_ids": [ids["missing_span"]]})
    assert rs.status == "invalid_span_ids" and rs.result["spans"] == []
    cites, invalid = CitationResolver(store).resolve_many([ids["missing_span"]])
    assert cites == [] and invalid == [ids["missing_span"]]


def test_verify_evidence_layers_reported_separately(store):
    reg = reg_for(store)
    r = reg.call("verify_evidence", {"claim": "Lục lạp là bào quan thực hiện quang hợp",
                                     "span_ids": [S["bio_luclap"]]}).result
    assert r["structural"]["status"] == "pass"
    assert r["source_link"]["status"] == "unlinked"          # direct span citation, no assertion/rule link
    assert r["semantic_support_is_heuristic"] is True and r["status"] == "supported"
    r2 = reg.call("verify_evidence", {"claim": "x", "rule_ids": ["biology:rule:fx_light_valid"],
                                      "span_ids": [S["bio_light"]]}).result
    assert r2["source_link"]["status"] == "pass"


def test_judge_quote_must_occur_in_span(store):
    from otter.llm.interface import MockLLM
    import json
    fake = json.dumps({"verdicts": [{"id": "claim", "verdict": "supported", "span_id": S["bio_luclap"],
                                     "quote": "Ti thể là bào quan thực hiện quang hợp"}]})
    reg = ToolRegistry(ToolContext(store=store, subject="biology", judge_llm=MockLLM(replies=[fake])))
    r = reg.call("verify_evidence", {"claim": "Ti thể thực hiện quang hợp", "span_ids": [S["bio_luclap"]]}).result
    assert r["status"] == "undetermined"                      # fabricated quote is rejected


# ---------------------------------------------------------------- 3. guard semantics
def test_rule_entailed_only_with_all_premises_and_guards(store):
    reg = reg_for(store)
    rid = "biology:rule:fx_light_valid"
    base = [{"predicate": "light_intensity_increases", "evidence_ids": [S["bio_light"]]}]
    r = reg.call("execute_rule", {"rule_id": rid, "facts": base}).result
    assert r["status"] == "undetermined" and len(r["missing"]) == 2 and r["conclusions"] == []
    r = reg.call("execute_rule", {"rule_id": rid, "facts": base + [
        {"predicate": "below_light_saturation", "evidence_ids": [S["bio_light"]]},
        {"predicate": "other_factors_non_limiting", "evidence_ids": [S["bio_light"]]}]}).result
    assert r["status"] == "entailed" and r["conclusions"][0]["predicate"] == "photosynthesis_rate_increases"
    assert r["conclusions"][0]["value"] is True and r["premise_support"] == "evidenced"


def test_failed_guard_blocks_and_never_negates(store):
    reg = reg_for(store)
    r = reg.call("execute_rule", {"rule_id": "biology:rule:fx_light_valid", "facts": [
        {"predicate": "light_intensity_increases"},
        {"predicate": "below_light_saturation", "value": False},
        {"predicate": "other_factors_non_limiting"}]}).result
    assert r["status"] == "blocked" and r["conclusions"] == []
    assert "guard" in r["reason"]
    # goal mode: failed antecedent of the only applicable rule does NOT prove "not conclusion"
    r = reg.call("execute_rule", {"goal": "geography:rain_on_windward_slope" if False else "rain_on_windward_slope",
                                  "facts": [{"predicate": "moist_wind_from_sea", "value": False}]})
    assert r.status in ("undetermined", "blocked") and r.status != "contradicted"


def test_proof_engine_backward_blocked_is_undetermined():
    from otter.schemas.ontology import Atom, ConditionalRule, Validation
    rule = ConditionalRule(rule_id="r1", subject="biology", premises=[Atom(predicate="p")],
                           guards=[Atom(predicate="g")], conclusions=[Atom(predicate="c")], source_span_ids=["s"],
                           validation=Validation(source_ids_exist=True, source_support="supported"))
    eng = ProofEngine(lambda pred: [rule] if pred == "c" else [], subject="biology")
    kb = KnowledgeBase()
    kb.add(*parse_literal("p"))
    kb.add(*parse_literal("not g"))
    res = eng.prove("c", kb)
    assert res.status == "undetermined"                       # blocked rule != evidence of "not c"
    kb2 = KnowledgeBase()
    kb2.add(*parse_literal("p"))
    kb2.add(*parse_literal("g"))
    assert eng.prove("c", kb2).status == "entailed"


def test_incomplete_and_unsupported_rules_not_strictly_executable(store, ids):
    reg = reg_for(store)
    for rid in ids["incomplete_rules"][:1] + ids["unsupported_rules"]:
        r = reg.call("execute_rule", {"rule_id": rid, "facts": [
            {"predicate": "co2_concentration_increases"}, {"predicate": "plant_watered_heavily"}]}).result
        assert r["status"] == "undetermined" and r["executable"] is False


def test_cross_subject_rule_blocked(store):
    reg = reg_for(store, subject="biology")
    r = reg.call("execute_rule", {"rule_id": "geography:rule:fx_leeward_dry",
                                  "facts": [{"predicate": "on_leeward_slope"}]}).result
    assert r["status"] == "blocked" and "cross_subject" in r["reason"]


# ---------------------------------------------------------------- 4. safe arithmetic
@pytest.mark.parametrize("expr,vars_,unit,expected", [
    ("2 + 3 * 4", None, None, 14.0),
    ("(2 + 3) * 4 / 5", None, None, 4.0),
    ("2^10", None, None, 1024.0),
    ("sqrt(16) + abs(-2)", None, None, 6.0),
    ("F / m", {"F": "6 N", "m": "2 kg"}, "m/s^2", 3.0),
    ("d / t", {"d": "36 km", "t": "1 h"}, "m/s", 10.0),
])
def test_calculator_basic(store, expr, vars_, unit, expected):
    r = reg_for(store).call("calculate", {"expression": expr, "variables": vars_, "target_unit": unit}).result
    assert r["status"] == "ok" and r["value"] == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')", "().__class__.__bases__", "open('x')", "lambda: 1",
    "[1,2,3]", "eval('1+1')", "x.y", "9**9**9", "'a' * 3",
])
def test_calculator_blocks_arbitrary_python(store, expr):
    r = reg_for(store).call("calculate", {"expression": expr}).result
    assert r["status"] in ("error", "unknown") and r["value"] is None


def test_calculator_dimension_mismatch_is_unknown(store):
    r = reg_for(store).call("calculate", {"expression": "a + b", "variables": {"a": "2 m", "b": "3 s"}}).result
    assert r["status"] == "unknown"
    r = reg_for(store).call("calculate", {"expression": "a", "variables": {"a": "2 furlong"}}).result
    assert r["status"] == "unknown"


def test_calculator_formula_from_rule(store):
    reg = reg_for(store, subject="geography")
    r = reg.call("calculate", {"rule_id": "geography:rule:fx_lapse_rate", "variables": {"delta_h": "2000 m"}}).result
    assert r["status"] == "ok" and r["value"] == pytest.approx(-12.0)
    assert r["formula_source_span_ids"] == [S["geo_lapse"]]


# ---------------------------------------------------------------- registry contracts
def test_registry_budget_disabled_and_unknown(store):
    reg = ToolRegistry(ToolContext(store=store, disabled=frozenset({"execute_rule"})), max_total_calls=2)
    assert reg.call("execute_rule", {"rule_id": "x"}).status == "disabled"
    assert reg.call("nope", {}).status == "error"
    reg.call("calculate", {"expression": "1+1"})
    reg.call("calculate", {"expression": "1+1"})
    assert reg.call("calculate", {"expression": "1+1"}).status == "budget_exceeded"
    assert reg.usage()["calls"] == 2


def test_search_and_expand_return_ids(store):
    reg = reg_for(store)
    r = reg.call("search_knowledge", {"query": "cường độ ánh sáng quang hợp"}).result
    assert r["status"] == "ok" and all(h["id"] and h["subject"] == "biology" for h in r["hits"])
    e = reg.call("expand_graph", {"node_ids": ["biology:c:quang_hop"], "relation_types": ["bogus_rel"]}).result
    assert e["status"] == "error" and e["reason"] == "invalid_relation_types"
    e = reg.call("expand_graph", {"node_ids": ["biology:c:quang_hop"]}).result
    assert e["status"] == "ok" and e["nodes"]


def test_judge_multi_hop_evidence_each_quote_verified(store):
    import json
    from otter.llm.interface import MockLLM
    rep = json.dumps({"verdicts": [{"id": "claim", "verdict": "supported", "span_id": S["geo_rain"],
                                    "quote": "không khí bị đẩy lên cao, nhiệt độ giảm",
                                    "more": [{"span_id": S["geo_lapse"], "quote": "nhiệt độ không khí giảm dần theo độ cao"},
                                             {"span_id": S["geo_tdl"], "quote": "câu không có trong đoạn này"}]}]})
    reg = ToolRegistry(ToolContext(store=store, subject="geography", judge_llm=MockLLM(replies=[rep])))
    r = reg.call("verify_evidence", {"claim": "Lên cao nhiệt độ giảm nên hơi nước ngưng tụ",
                                     "span_ids": [S["geo_rain"], S["geo_lapse"], S["geo_tdl"]]}).result
    assert r["status"] == "supported" and r["evidence_span_ids"] == [S["geo_rain"], S["geo_lapse"]]
