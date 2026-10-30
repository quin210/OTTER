"""otter_v2 offline tests (MockLLM): option obligations, verified citation, cascade router, abstention, ablations."""
from __future__ import annotations

import json

from otter.agents.systems import get_system
from otter.agents.v2 import ABSTAIN_TEXT, quote_in_span
from otter.llm.interface import MockLLM
from tests.reasoning.ws3_helpers import S

ITEM = {"id": "q1", "question": "Bào quan nào thực hiện quang hợp?", "domain": "biology", "grade": 11,
        "options": {"A": "Ti thể", "B": "Lục lạp", "C": "Ribosome", "D": "Lizoxom"}}


class Gen:
    """Routes by system prompt: generator vs verifier; counts calls."""

    def __init__(self, gen_reply, ver_reply=None):
        self.gen_reply, self.ver_reply = gen_reply, ver_reply or {"claims": [{"id": "c1", "verdict": "supported"}],
                                                                 "answer_supported": "yes"}
        self.gen_calls = self.ver_calls = 0

    def gen(self, msgs):
        self.gen_calls += 1
        r = self.gen_reply(self.gen_calls, msgs) if callable(self.gen_reply) else self.gen_reply
        return json.dumps(r, ensure_ascii=False)

    def ver(self, msgs):
        self.ver_calls += 1
        return json.dumps(self.ver_reply, ensure_ascii=False)


def mk(g: Gen, store, **flags):
    return get_system("otter_v2", llm=MockLLM(handler=g.gen), verifier_llm=MockLLM(handler=g.ver), store=store,
                      llm_cache=False, **flags)


def good_reply(span=None, verdict="entailed"):
    span = span or S["bio_luclap"]
    return {"options": {"A": "refuted", "B": verdict, "C": "unsupported", "D": "unsupported"}, "answer": "B",
            "claims": [{"text": "Lục lạp thực hiện quang hợp", "span_ids": [span], "quote": "x"}], "missing": ""}


def test_quote_in_span():
    assert quote_in_span("Lục lạp là bào quan", "Lục lạp là bào quan quang hợp ở thực vật.")
    assert not quote_in_span("Ti thể sản xuất ATP", "Lục lạp là bào quan quang hợp.")


def test_single_round_two_calls_and_real_citation(store):
    g = Gen(good_reply())
    fa = mk(g, store).answer(ITEM)
    assert fa.error is None and fa.answer_choice == "B"
    assert g.gen_calls == 1 and g.ver_calls == 1
    assert fa.tool_usage["llm_calls"] == 2 and fa.tool_usage["v2_rounds"] == 1
    assert all(c.span_id for cl in fa.claims for c in cl.citations)


def test_fabricated_span_ids_dropped(store):
    g = Gen(good_reply(span="fake:p0:u9"))
    fa = mk(g, store).answer(ITEM)
    assert fa.claims == [] and fa.tool_usage["v2_grounding"] == 0


def test_cascade_runs_gap_round_only_when_not_entailed(store):
    weak = {"options": {"A": "unsupported", "B": "unsupported", "C": "unsupported", "D": "unsupported"},
            "answer": "B", "claims": [], "missing": "lục lạp quang hợp"}
    g = Gen(lambda n, m: weak if n == 1 else good_reply())
    fa = mk(g, store).answer(ITEM)
    assert g.gen_calls == 2 and fa.tool_usage["v2_rounds"] == 2
    g2 = Gen(good_reply())
    mk(g2, store).answer(ITEM)
    assert g2.gen_calls == 1
    g3 = Gen(good_reply())
    mk(g3, store, no_gap_round=True).answer(ITEM)
    assert g3.gen_calls == 1


def test_verifier_unsupported_claim_dropped(store):
    g = Gen(good_reply(), {"claims": [{"id": "c1", "verdict": "unsupported"}], "answer_supported": "no"})
    fa = mk(g, store).answer(ITEM)
    assert fa.claims == [] and fa.status == "qualified" and fa.answer_choice == "B"


def test_abstention_threshold_and_ablation(store):
    g = Gen(good_reply(), {"claims": [{"id": "c1", "verdict": "unsupported"}], "answer_supported": "no"})
    fa = mk(g, store, abstention=True, abstain_below=2).answer(ITEM)
    assert fa.status == "abstain" and fa.answer_choice is None and fa.answer == ABSTAIN_TEXT
    fa2 = mk(g, store, abstention=True, abstain_below=2, no_abstention=True).answer(ITEM)
    assert fa2.status != "abstain"


def test_explicit_option_e_is_abstain(store):
    item = dict(ITEM, options={**ITEM["options"], "E": "Không đủ căn cứ trong sách"})
    g = Gen({"options": {}, "answer": "E", "claims": [], "missing": "x"})
    fa = mk(g, store).answer(item)
    assert fa.status == "abstain" and fa.answer_choice == "E"


def test_no_verified_citation_skips_verifier(store):
    g = Gen(good_reply())
    mk(g, store, no_verified_citation=True).answer(ITEM)
    assert g.ver_calls == 0


def test_never_raises(store):
    def boom(m):
        raise RuntimeError("down")
    s = get_system("otter_v2", llm=MockLLM(handler=boom), verifier_llm=MockLLM(handler=boom), store=store, llm_cache=False)
    fa = s.answer(ITEM)
    assert fa.status == "failed" and fa.error


def test_where_to_read_resolved(store):
    fa = mk(Gen(good_reply()), store).answer(ITEM)
    assert fa.where_to_read and fa.where_to_read[0].span_id == S["bio_luclap"]
    assert fa.where_to_read[0].quote and fa.tool_usage["v2_located"] in ("verified", "unverified")
    fa2 = mk(Gen({"options": {}, "answer": "B", "claims": [], "missing": ""}), store).answer(ITEM)
    assert fa2.tool_usage["v2_located"] in ("unverified", "none")


def test_frozen_yaml_is_single_source(tmp_path):
    from otter.agents.v2 import V2Config, load_frozen_config
    p = tmp_path / "c.yaml"
    p.write_text("retrieval: {top_k_question: 7, max_spans: 9}\nrouter: {cascade: true, gap_level: 3}\n"
                 "claims: {max_claims: 2}\noperating_points: {calibrated_abstain: {abstention: true, abstain_below: 2}}\n")
    c = load_frozen_config(p)
    assert (c.top_k_question, c.max_spans, c.gap_level, c.max_claims, c.abstain_below) == (7, 9, 3, 2, 2)
    assert c.abstention is False                       # answer-always stays default
    d, y = V2Config(), load_frozen_config()           # repo yaml agrees with documented frozen values
    assert (y.gap_level, y.top_k_question, y.top_k_option, y.max_spans) == (2, 6, 3, 10)


def test_cli_flags_override_yaml(store):
    g = Gen(good_reply())
    s = mk(g, store, gap_level=3)
    assert s.cfg.gap_level == 3
    assert mk(g, store).cfg.gap_level == 2


def test_router_escalates_on_verifier_rejection(store):
    """v2.1: generator says `entailed` but the cross-model verifier rejects -> one gap round (ablation: verify_trigger=false)."""
    rej = {"claims": [{"id": "c1", "verdict": "unsupported"}], "answer_supported": "no"}
    g = Gen(good_reply(), rej)
    fa = mk(g, store).answer(ITEM)
    assert fa.tool_usage["v2_escalation"] == "verifier_reject"
    g2 = Gen(good_reply(), rej)
    fa2 = mk(g2, store, verify_trigger=False).answer(ITEM)
    assert g2.gen_calls == 1 and fa2.tool_usage["v2_escalation"] is None


def test_router_escalates_on_conflicting_entailed(store):
    conflict = dict(good_reply(), options={"A": "refuted", "B": "entailed", "C": "unsupported", "D": "entailed"})
    g = Gen(conflict)
    fa = mk(g, store).answer(ITEM)
    assert fa.tool_usage["v2_escalation"] == "conflicting_entailed"
    g2 = Gen(conflict)
    mk(g2, store, no_gap_round=True).answer(ITEM)
    assert g2.gen_calls == 1
    g3 = Gen(conflict)
    mk(g3, store, no_router=True).answer(ITEM)
    assert g3.gen_calls == 2                      # no_router: always two rounds


def test_gap_round_searches_only_with_gap_queries(store):
    """v2.1: the gap round must not re-run the first-round queries (they dominated the fused top-k and nothing new was added)."""
    s = mk(Gen(good_reply()), store)
    q = __import__("otter.agents.runtime", fromlist=["QuestionInput"]).QuestionInput.from_item(ITEM)
    seen = []
    orig = s._search
    s._search = lambda query, qq, k: (seen.append(query), orig(query, qq, k))[1]
    s.retrieve(q)
    n_first = len(seen)
    seen.clear()
    s.retrieve(q, extra_queries=["lục lạp quang hợp"], only_extra=True)
    assert n_first >= 3 and seen == ["lục lạp quang hợp"]
