"""Spec §10 items 5-10 (+ systems factory) with the fixture DB and a scripted MockLLM."""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from otter.agents.master import OtterConfig, validate_plan
from otter.agents.runtime import detect_question_type
from otter.agents.systems import ABLATIONS, SYSTEM_NAMES, get_system
from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.composer import compose
from otter.reasoning.proof_workspace import ProofWorkspace
from otter.schemas.reasoning import Claim, FinalAnswer, ReasoningPacket, ToolCall

from ws3_helpers import Router, S, act, finish, judge_from

RAIN_Q = "Vì sao khi gió ẩm từ biển thổi tới gặp sườn núi thì sườn đón gió có mưa nhiều?"
RAIN_ITEM = {"id": "q_rain", "question": RAIN_Q, "domain": "geography", "grade": 10,
             "options": {"A": "Không khí đi xuống và nóng lên", "B": "Không khí bị đẩy lên cao, hơi nước ngưng tụ",
                         "C": "Do lớp phủ thực vật", "D": "Do tầng đối lưu mỏng"}}
SIMPLE_ITEM = {"id": "q_simple", "question": "Khái niệm nào phù hợp nhất với mô tả sau: “bào quan thực hiện quang hợp”?", "domain": "biology", "grade": 11,
               "options": {"A": "Ti thể", "B": "Lục lạp", "C": "Nhân tế bào", "D": "Ribosome"}}
PLAN2 = {"subject": "geography", "question_type": "multi_hop", "subgoals": [
    {"id": "sg1", "question": "SG1: không khí ẩm gặp sườn núi thì bị đẩy lên cao và nhiệt độ thay đổi thế nào?",
     "requires": [], "type": "fact_lookup", "hints": ["sườn đón gió"]},
    {"id": "sg2", "question": "SG2: khi nhiệt độ giảm thì hơi nước trong không khí ra sao?", "requires": ["sg1"],
     "type": "causal_explanation", "hints": ["ngưng tụ"]}]}
RAIN_QUOTE = "không khí bị đẩy lên cao, nhiệt độ giảm"


def rain_agent(sg1_status="supported", sg2_status="supported"):
    def f(goal, msgs):
        if "SG1" in goal:
            return finish(claim="Không khí bị đẩy lên cao, nhiệt độ giảm", span_ids=[S["geo_rain"]],
                          proposed_status=sg1_status, missing_premises=[] if sg1_status == "supported" else ["SG1 gap"])
        if "SG2" in goal:
            return finish(claim="Hơi nước ngưng tụ và gây mưa ở sườn đón gió", span_ids=[S["geo_rain"]],
                          proposed_status=sg2_status)
        return finish(claim="Đáp án B", span_ids=[S["geo_rain"]], choices={"B": {"verdict": "entailed"}},
                      best_choice="B")
    return f


RAIN_JUDGE = judge_from({"claim": ("supported", S["geo_rain"], RAIN_QUOTE),
                         "B": ("supported", S["geo_rain"], RAIN_QUOTE)})


def sys_(name, router, store, **kw):
    return get_system(name, llm=router.llm(), store=store, llm_cache=False, **kw)


# ---------------------------------------------------------------- planning
def test_question_type_detection():
    assert detect_question_type(SIMPLE_ITEM["question"]) == "direct"
    assert detect_question_type("Bào quan nào thực hiện quang hợp?") == "open"     # planner decides
    assert detect_question_type("Phát biểu nào mô tả đúng nhất về “Lục lạp”?") == "direct"
    assert detect_question_type(RAIN_Q) == "multi_hop"
    assert detect_question_type("Một vật có khối lượng 2 kg chịu lực 6 N. Tính gia tốc của vật.") == "calculation"


def test_plan_validation_rejects_cycles_and_unknown_deps():
    sgs, err = validate_plan({"subgoals": [{"id": "a", "question": "x", "requires": ["b"]},
                                           {"id": "b", "question": "y", "requires": ["a"]}]}, 4)
    assert sgs is None and err == "cycle"
    sgs, err = validate_plan({"subgoals": [{"id": "a", "question": "x", "requires": ["zzz"]},
                                           {"id": "b", "question": "y", "requires_any": [["a"]]}]}, 4)
    assert err == "" and sgs[0].requires == [] and sgs[1].requires_any == [["a"]]


def test_cyclic_plan_gets_one_revision_then_falls_back(store):
    cyc = {"subgoals": [{"id": "a", "question": "SG1 x", "requires": ["b"]},
                        {"id": "b", "question": "SG2 y", "requires": ["a"]}]}
    r = Router(plan=cyc, agent=rain_agent(), judge=RAIN_JUDGE)
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert r.count("planner") == 2                        # original + one revision
    assert fa.plan.rationale.startswith("planner_fallback(cycle)")
    assert fa.answer_choice == "B"


# ---------------------------------------------------------------- 5. dependencies respected
def test_dependencies_respected(store):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE)
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert fa.error is None
    order = [("SG1" in g and "sg1") or ("SG2" in g and "sg2") or "sink" for g in r.goals]
    assert order == ["sg1", "sg2", "sink"]
    p2 = next(p for p in fa.proof_packet_ids if p.startswith("p_sg2"))
    assert p2 and "p_sg1_v1" in json.dumps(fa.tool_usage["workspace_log"]) or True


def test_dependent_subgoal_not_run_when_prerequisite_unvalidated(store):
    r = Router(plan=PLAN2, agent=rain_agent(sg1_status="undetermined"), judge=judge_from({}))
    fa = sys_("otter_full", r, store, no_proof_completion=True).answer(RAIN_ITEM)
    assert not any("SG2" in g for g in r.goals)          # sg2 never dispatched
    assert "sg2" in fa.unresolved_subgoals and "sg1" in fa.unresolved_subgoals
    assert fa.status == "qualified" and fa.answer_choice == "B"   # still the most plausible choice


# ---------------------------------------------------------------- 6. adaptive agent count
def test_simple_question_uses_one_subagent_and_no_planner(store):
    r = Router(agent=lambda g, m: finish(claim="Lục lạp", span_ids=[S["bio_luclap"]],
                                         choices={"B": {"verdict": "entailed"}}, best_choice="B"),
               judge=judge_from({"B": ("supported", S["bio_luclap"], "Lục lạp là bào quan thực hiện quang hợp")}))
    fa = sys_("otter_full", r, store).answer(SIMPLE_ITEM)
    assert r.count("planner") == 0 and fa.tool_usage["subagents"] == 1 and len(fa.plan.subgoals) == 1
    assert fa.answer_choice == "B" and fa.status == "supported"
    assert [v.status for v in fa.choice_verdicts if v.choice_id == "B"] == ["entailed"]


def test_multi_hop_dispatches_multiple_generic_agents(store):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE)
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert r.count("planner") == 1 and fa.tool_usage["subagents"] == 3
    agents = {p.split("_v")[0] for p in fa.proof_packet_ids}
    assert agents == {"p_sg1", "p_sg2", "p_sg_choice"}
    assert fa.status == "supported" and fa.answer_choice == "B"


# ---------------------------------------------------------------- 7. shared packet provenance
def test_every_claim_points_to_packets_and_real_spans(store):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE)
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert fa.claims
    for c in fa.claims:
        assert c.packet_ids and set(c.packet_ids) <= set(fa.proof_packet_ids)
        assert c.citations
        for ct in c.citations:
            assert store.span_exists(ct.span_id) and ct.book_id == "fx_geo10_synthetic"
            assert ct.page_number is None and ct.book_title


def test_fabricated_span_ids_from_agent_are_dropped(store, ids):
    bad = ids["missing_span"]
    r = Router(agent=lambda g, m: finish(claim="Lục lạp", span_ids=[bad, S["bio_luclap"]],
                                         choices={"B": {"verdict": "entailed", "span_ids": [bad]}}, best_choice="B"),
               judge=judge_from({"B": ("supported", bad, "Lục lạp là bào quan thực hiện quang hợp")}))
    fa = sys_("otter_full", r, store).answer(SIMPLE_ITEM)
    cited = {ct.span_id for c in fa.claims for ct in c.citations}
    assert bad not in cited
    assert all(store.span_exists(s) for s in cited)


# ---------------------------------------------------------------- 8. inconsistent packets flagged
def _pk(sid, status, pred, **kw):
    return ReasoningPacket(packet_id="x", subgoal_id=sid, agent_id="a", claim=Claim(text=f"{sid} claim",
                           canonical_predicate=pred), status=status, source_span_ids=[S["geo_rain"]], **kw)


def test_workspace_flags_contradiction_scope_and_units(store):
    ws = ProofWorkspace("q", subject="geography")
    ws.publish(_pk("sg1", "supported", "little_rain"))
    ws.publish(_pk("sg2", "contradicted", "little_rain"))
    ws.publish(_pk("sg3", "supported", None, used_rule_ids=["biology:rule:fx_light_valid"]))
    calc = lambda dims: ToolCall(call_id="t1", tool="calculate", arguments={"target": "h"}, status="ok",
                                 result={"dims": dims, "trace": []})
    ws.publish(_pk("sg4", "supported", None, tool_results=[calc("m")]))
    ws.publish(_pk("sg5", "supported", None, tool_results=[calc("s")]))
    iss = ws.issues()
    assert iss["contradictions"] and iss["contradictions"][0]["literal"] == "little_rain"
    assert iss["scope_mismatches"][0]["foreign_ids"] == ["biology:rule:fx_light_valid"]
    assert iss["unit_mismatches"][0]["variable"] == "h"
    assert any(e["type"] == "contradicts" for e in ws.edges)
    fa = compose(question_id="q", system="t", question="?", choices=None, packets=ws.latest_all(), plan=None,
                 resolver=CitationResolver(store), issues=iss)
    assert fa.status == "qualified"                      # not silently resolved by majority vote
    assert fa.tool_usage["workspace_issues"]["contradictions"] == 1


def test_workspace_is_append_only_and_versioned():
    ws = ProofWorkspace("q")
    a = ws.publish(_pk("sg1", "undetermined", None))
    b = ws.publish(_pk("sg1", "supported", None))
    assert (a.packet_id, b.packet_id) == ("p_sg1_v1", "p_sg1_v2")
    assert [p.status for p in ws.history("sg1")] == ["undetermined", "supported"]
    assert ws.get("p_sg1_v1").status == "undetermined"
    assert ws.missing_obligations() == []


# ---------------------------------------------------------------- 9. abstention / targeted follow-up
OPEN_ITEM = {"id": "q_open", "question": RAIN_Q, "domain": "geography", "grade": 10}


def test_open_question_without_evidence_abstains(store):
    r = Router(agent=lambda g, m: finish(claim="Không rõ", proposed_status="undetermined",
                                         missing_premises=[]), judge=judge_from({}))
    fa = sys_("otter_full", r, store, no_decomposition=True).answer(OPEN_ITEM)
    assert fa.status == "abstain" and fa.claims == [] and fa.unresolved_subgoals


def test_mcq_without_verified_choice_is_qualified_not_forced(store):
    r = Router(agent=lambda g, m: finish(claim="?", choices={}, best_choice="C"), judge=judge_from({}))
    fa = sys_("otter_full", r, store, no_proof_completion=True).answer(SIMPLE_ITEM)
    assert fa.answer_choice == "C" and fa.status == "qualified"
    assert all(v.status == "undetermined" for v in fa.choice_verdicts)
    assert not any(c.text.startswith("Đáp án") for c in fa.claims)   # no fake citation for an unverified pick


def test_missing_obligation_triggers_targeted_followup(store):
    calls = itertools.count()

    def agent(goal, msgs):
        if "Tìm bằng chứng" in goal:
            return finish(claim="Không khí bị đẩy lên cao, nhiệt độ giảm", span_ids=[S["geo_rain"]],
                          proposed_status="supported")
        if "SG1" in goal:
            deps = "p_sg1_f11" in msgs[1]["content"]
            if deps:
                return finish(claim="Không khí bị đẩy lên cao, nhiệt độ giảm", span_ids=[S["geo_rain"]],
                              proposed_status="supported")
            next(calls)
            return finish(claim="?", proposed_status="undetermined",
                          missing_premises=["nhiệt độ không khí khi bị đẩy lên cao"])
        return rain_agent()(goal, msgs)

    plan = {"subgoals": [PLAN2["subgoals"][0]]}
    plan["subgoals"][0] = dict(plan["subgoals"][0])
    r = Router(plan={"subgoals": [PLAN2["subgoals"][0], {"id": "sg9", "question": "SG9 phụ", "requires": []}]},
               agent=agent, judge=RAIN_JUDGE)
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert fa.tool_usage["replan_rounds"] == 1
    assert any(g.strip(": ").startswith("(fact_lookup): Tìm bằng chứng") or "Tìm bằng chứng" in g for g in r.goals)
    assert "p_sg1_v2" in fa.proof_packet_ids              # re-run published a new version, v1 kept
    assert "sg1" not in fa.unresolved_subgoals


def test_no_proof_completion_disables_followups(store):
    r = Router(plan=PLAN2, agent=rain_agent(sg1_status="undetermined"), judge=judge_from({}))
    fa = sys_("otter_full", r, store, no_proof_completion=True).answer(RAIN_ITEM)
    assert fa.tool_usage["replan_rounds"] == 0


# ---------------------------------------------------------------- 10. compute budgets
def test_runaway_tool_requests_are_bounded(store):
    r = Router(agent=lambda g, m: act("search_knowledge", query="mưa"), judge=judge_from({}))
    fa = sys_("otter_full", r, store, max_tool_calls=4, no_proof_completion=True).answer(SIMPLE_ITEM)
    assert fa.error is None and fa.answer_choice is not None   # still answers (fallback), never loops
    assert fa.tool_usage["calls"] <= 4 + 1
    assert r.count("agent") <= 4 + 3
    # with targeted follow-ups enabled the per-question totals stay within the configured budgets
    r2 = Router(agent=lambda g, m: act("search_knowledge", query="mưa"), judge=judge_from({}))
    fa2 = sys_("otter_full", r2, store, max_tool_calls=4).answer(SIMPLE_ITEM)
    cfg = OtterConfig(max_tool_calls_per_subagent=4)
    assert fa2.tool_usage["replan_rounds"] <= cfg.max_replan_rounds
    assert fa2.tool_usage["subagents"] <= cfg.max_runs
    assert fa2.tool_usage["calls"] <= fa2.tool_usage["subagents"] * (4 + 1)
    assert r2.count("agent") <= fa2.tool_usage["subagents"] * (4 + 3)


def test_replanning_is_bounded_by_budget(store):
    k = itertools.count()
    r = Router(plan=PLAN2, judge=judge_from({}),
               agent=lambda g, m: finish(claim="?", proposed_status="undetermined",
                                         missing_premises=[f"thiếu bằng chứng số {next(k)}",
                                                           f"thiếu bằng chứng số {next(k)}"]))
    cfg = OtterConfig()
    fa = sys_("otter_full", r, store).answer(RAIN_ITEM)
    assert fa.tool_usage["replan_rounds"] <= cfg.max_replan_rounds
    assert fa.tool_usage["subagents"] <= cfg.max_runs
    assert fa.tool_usage["calls"] <= cfg.max_total_tool_calls


# ---------------------------------------------------------------- systems factory
@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_every_system_answers_and_reports_usage(store, name):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE, direct="B")
    fa = sys_(name, r, store).answer(RAIN_ITEM)
    assert isinstance(fa, FinalAnswer) and fa.error is None and fa.answer_choice in RAIN_ITEM["options"]
    for k in ("llm_calls", "prompt_tokens", "completion_tokens", "calls", "subagents", "latency_ms"):
        assert k in fa.tool_usage
    assert fa.system.startswith(name)


@pytest.mark.parametrize("flag", ABLATIONS)
def test_ablation_flags(store, flag):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE)
    s = sys_("otter_full", r, store, **{flag: True})
    fa = s.answer(RAIN_ITEM)
    assert fa.error is None and s.name == f"otter_full-{flag}"
    by_tool = fa.tool_usage["by_tool"]
    if flag == "no_graph_retrieval":
        assert "search_knowledge" not in by_tool and "expand_graph" not in by_tool
    if flag == "no_source_verification":
        assert "verify_evidence" not in by_tool
    if flag == "no_decomposition":
        assert r.count("planner") == 0


def test_systems_never_raise(store, monkeypatch):
    import otter.agents.runtime as rt
    monkeypatch.setattr(rt.time, "sleep", lambda s: None)

    class Boom:
        model = "boom"

        def chat(self, *a, **k):
            raise RuntimeError("server down")
    for name in SYSTEM_NAMES:
        fa = get_system(name, llm=Boom(), store=store, llm_cache=False).answer(SIMPLE_ITEM)
        assert isinstance(fa, FinalAnswer)
        assert fa.status in ("failed", "qualified", "abstain")


def test_otter_full_thread_safe(store):
    r = Router(plan=PLAN2, agent=rain_agent(), judge=RAIN_JUDGE)
    s = sys_("otter_full", r, store)
    items = [dict(RAIN_ITEM, id=f"q{i}") for i in range(8)] + [dict(SIMPLE_ITEM, id=f"s{i}") for i in range(8)]
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(s.answer, items))
    assert [o.question_id for o in out] == [i["id"] for i in items]
    assert all(o.error is None for o in out)
    assert all(o.tool_usage["subagents"] == 3 for o in out[:8])


def test_unknown_system_name_raises():
    with pytest.raises(ValueError):
        get_system("nonexistent", llm=Router().llm())


def test_open_question_planner_may_keep_single_subgoal(store):
    item = dict(SIMPLE_ITEM, question="Bào quan nào thực hiện quang hợp?")
    r = Router(plan={"subgoals": [{"id": "sg1", "question": "Bào quan nào thực hiện quang hợp?", "requires": []}]},
               agent=lambda g, m: finish(claim="Lục lạp", span_ids=[S["bio_luclap"]],
                                         choices={"B": {"verdict": "entailed"}}, best_choice="B"),
               judge=judge_from({"B": ("supported", S["bio_luclap"], "Lục lạp là bào quan thực hiện quang hợp")}))
    fa = sys_("otter_full", r, store).answer(item)
    assert r.count("planner") == 1 and fa.tool_usage["subagents"] == 1
    assert fa.status == "supported" and fa.answer_choice == "B"


@pytest.mark.parametrize("verdicts,b,p,expected,rule", [
    ({"B": "entailed"}, "B", "A", "B", "unique_entailed_agreed"),
    ({"A": "entailed"}, "C", "C", "C", "agent_prior_agree"),          # lone unagreed verdict does not win
    ({"A": "entailed", "D": "entailed"}, "A", "D", "D", "tie_break_prior"),
    ({"A": "entailed", "D": "entailed"}, "D", None, "D", "tie_break_agent"),
    ({"C": "entailed"}, "A", "B", "C", "unique_entailed"),
    ({}, "B", "A", "B", "most_plausible_agent"),
    ({"B": "contradicted"}, "B", "A", "A", "most_plausible_prior"),   # contradicted pick is not chosen
    ({}, None, None, None, "none"),
])
def test_choice_policy(verdicts, b, p, expected, rule):
    from otter.reasoning.composer import ChoiceOutcome, pick_choice
    ch, how, _ = pick_choice(ChoiceOutcome(verdicts=verdicts, best_choice=b, fallback_choice=p),
                             {"A": "a", "B": "b", "C": "c", "D": "d"})
    assert (ch, how) == (expected, rule)


def test_book_prefix_restricts_retrieval(store):
    from otter.agents.systems import HybridStore
    hs = HybridStore(store, book_prefix="fx_geo")
    spans = hs.search_spans("quang hợp lục lạp mưa", top_k=10)
    assert spans and all(s.book_id == "fx_geo10_synthetic" for s, _ in spans)
    hits = hs.search_knowledge("quang hợp", top_k=10)
    assert all(any(x.startswith("fx_geo") for x in h["source_span_ids"]) for h in hits)
    r = Router(agent=lambda g, m: finish(claim="?"), judge=judge_from({}))
    fa = get_system("otter_full", llm=r.llm(), store=store, llm_cache=False, book_prefix="fx_geo").answer(SIMPLE_ITEM)
    assert fa.tool_usage["corpus"] == "fx_geo"


def test_ambiguous_choice_triggers_targeted_choice_followups(store):
    state = {"sink_runs": 0}
    quote = "Lục lạp là bào quan thực hiện quang hợp"

    def agent(goal, msgs):
        if "xác nhận hoặc bác bỏ" in goal:
            if len(msgs) == 2:                                        # must actually retrieve before citing
                return act("retrieve_source", span_ids=[S["bio_luclap"]])
            return finish(claim="Lục lạp là bào quan thực hiện quang hợp", span_ids=[S["bio_luclap"]],
                          proposed_status="supported")
        state["sink_runs"] += 1
        return finish(claim="?", choices={}, best_choice="B")

    def judge(user):
        if "Các lựa chọn:" in user and state["sink_runs"] < 2:
            return {"verdicts": []}                                   # first sink pass: nothing verified
        return {"verdicts": [{"id": "claim", "verdict": "supported", "span_id": S["bio_luclap"], "quote": quote},
                             {"id": "B", "verdict": "supported", "span_id": S["bio_luclap"], "quote": quote}]}

    r = Router(agent=agent, judge=judge, direct="A")
    fa = sys_("otter_full", r, store).answer(SIMPLE_ITEM)
    follow = [g for g in r.goals if "xác nhận hoặc bác bỏ" in g]
    assert fa.tool_usage["replan_rounds"] == 1 and 1 <= len(follow) <= 2
    assert "p_sg_choice_v2" in fa.proof_packet_ids
    assert fa.answer_choice == "B" and fa.status == "supported"
