"""Deterministic proof composer (spec 5.4 / 5.5).

Decides *which* claims are supported purely from workspace packets + system checks:
  * only `supported` packets with >=1 resolvable source span become cited answer claims;
  * every claim keeps its packet ids and citations resolved from real metadata (CitationResolver);
  * workspace issues (contradictions, scope / unit mismatches, cycles) are reported and downgrade
    the status to `qualified` -- never resolved by majority vote;
  * MCQ: per-choice verdicts entailed / contradicted / undetermined; a unique entailed choice gives
    `supported`; otherwise the most plausible choice is still returned with status `qualified`.
The LLM is used (optionally, by the caller) only to verbalize the supported content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from otter.reasoning.citation_resolver import CitationResolver
from otter.schemas.reasoning import AnswerClaim, ChoiceVerdict, Citation, FinalAnswer, QuestionPlan, ReasoningPacket


@dataclass
class ChoiceOutcome:
    verdicts: dict[str, str] = field(default_factory=dict)
    spans: dict[str, list[str]] = field(default_factory=dict)
    best_choice: Optional[str] = None             # agent's most plausible choice
    packet_id: Optional[str] = None
    fallback_choice: Optional[str] = None         # Master's closed-book prior: tie-breaker / fallback only


def resolve_citations(resolver: CitationResolver, span_ids: list[str], *, allow_unverified: bool = False
                      ) -> tuple[list[Citation], list[str]]:
    cites, invalid = resolver.resolve_many(span_ids, with_quote=True, quote_chars=240)
    if allow_unverified and invalid:          # ablation only: keep the agent's raw ids, explicitly unknown book
        cites += [Citation(book_id="unknown", span_id=s) for s in invalid]
    return cites, invalid


def pick_choice(outcome: ChoiceOutcome, choices: dict[str, str]) -> tuple[Optional[str], str, list[str]]:
    """Deterministic MCQ decision. Returns (choice, rule, notes).

    E = choices entailed by verified evidence (quote-checked judge), b = subagent's most plausible
    choice, p = Master's closed-book prior (`fallback_choice`; tie-breaker only).
      unique_entailed_agreed  |E|=1 and E agrees with b (or p)            -> may be `supported`
      agent_prior_agree       b == p, overriding a lone unagreed verdict  -> qualified
      tie_break_prior/agent   |E|>1: prefer p, then b, inside E           -> qualified
      unique_entailed         |E|=1, nobody agrees                        -> qualified
      most_plausible          E empty: b (if not contradicted), else p    -> qualified
    """
    v = outcome.verdicts
    E = [k for k in choices if v.get(k) == "entailed"]
    contra = {k for k in choices if v.get(k) == "contradicted"}
    b = outcome.best_choice if outcome.best_choice in choices else None
    p = outcome.fallback_choice if outcome.fallback_choice in choices else None
    notes: list[str] = []
    if len(E) > 1:
        notes.append(f"several_choices_entailed:{','.join(E)}")
    if len(E) == 1 and E[0] in (b, p):
        return E[0], "unique_entailed_agreed" if E[0] == b else "unique_entailed_prior_agreed", notes
    if b and b == p and b not in contra:
        if E:
            notes.append(f"evidence_verdict_overridden:{','.join(E)}")
        return b, "agent_prior_agree", notes
    if len(E) > 1:
        if p in E:
            return p, "tie_break_prior", notes
        if b in E:
            return b, "tie_break_agent", notes
        return E[0], "first_entailed", notes
    if len(E) == 1:
        return E[0], "unique_entailed", notes
    for cand, rule in ((b, "most_plausible_agent"), (p, "most_plausible_prior")):
        if cand and cand not in contra:
            return cand, rule, notes
    open_ = [k for k in choices if k not in contra]
    if len(open_) == 1 and len(choices) > 1:
        return open_[0], "elimination", notes
    if b or p:
        notes.append("chosen_choice_was_contradicted")
        return b or p, "most_plausible_contradicted", notes
    return None, "none", notes


def compose(*, question_id: str, system: str, question: str, choices: Optional[dict[str, str]],
            packets: list[ReasoningPacket], plan: Optional[QuestionPlan], resolver: CitationResolver,
            issues: Optional[dict[str, list]] = None, choice: Optional[ChoiceOutcome] = None,
            blocked_subgoals: Optional[list[str]] = None, verbalize: Optional[Callable[..., str]] = None,
            allow_unverified_citations: bool = False) -> FinalAnswer:
    issues = issues or {}
    blocked_subgoals = blocked_subgoals or []
    claims: list[AnswerClaim] = []
    unresolved: list[str] = []
    invalid_all: list[str] = []
    k = 0
    choice_pid = choice.packet_id if choice else None
    for p in packets:
        if p.packet_id == choice_pid:
            continue
        if p.status in ("supported", "contradicted"):
            cites, invalid = resolve_citations(resolver, p.source_span_ids, allow_unverified=allow_unverified_citations)
            invalid_all += invalid
            if cites:
                k += 1
                text = p.claim.text if p.status == "supported" else f"Bằng chứng mâu thuẫn với: {p.claim.text}"
                claims.append(AnswerClaim(claim_id=f"c{k}", text=text, packet_ids=[p.packet_id], citations=cites))
                continue
            unresolved.append(p.subgoal_id)            # "supported" without any real span -> not citable
        else:
            unresolved.append(p.subgoal_id)
    for sg in blocked_subgoals:
        if sg not in unresolved:
            unresolved.append(sg)

    n_conflicts = len(issues.get("contradictions", [])) + len(issues.get("scope_mismatches", [])) + \
        len(issues.get("unit_mismatches", [])) + len(issues.get("dependency_cycles", []))
    notes: list[str] = []
    answer_choice = None
    verdict_list: list[ChoiceVerdict] = []
    if choices:
        choice = choice or ChoiceOutcome()
        answer_choice, how, cnotes = pick_choice(choice, choices)
        notes += cnotes
        for key in choices:
            verdict_list.append(ChoiceVerdict(choice_id=key, status=choice.verdicts.get(key, "undetermined"),
                                              packet_ids=[choice.packet_id] if choice.packet_id else []))
        if answer_choice is not None:
            sp = choice.spans.get(answer_choice, []) if choice.verdicts.get(answer_choice) == "entailed" else []
            cites, invalid = resolve_citations(resolver, sp, allow_unverified=allow_unverified_citations)
            invalid_all += invalid
            if cites:
                k += 1
                claims.insert(0, AnswerClaim(claim_id=f"c{k}", text=f"Đáp án {answer_choice}: {choices[answer_choice]}",
                                             packet_ids=[choice.packet_id] if choice.packet_id else [],
                                             citations=cites))
        supported = (how == "unique_entailed_agreed" and n_conflicts == 0 and not blocked_subgoals
                     and any(c.text.startswith(f"Đáp án {answer_choice}") for c in claims))
        status = "supported" if supported else ("qualified" if answer_choice else "abstain")
        if answer_choice is None:
            answer = "Chưa đủ bằng chứng để chọn đáp án."
        else:
            body = [c.text for c in claims if not c.text.startswith("Đáp án ")][:3]
            answer = f"Đáp án {answer_choice}. {choices[answer_choice]}"
            if body:
                answer += " Căn cứ: " + " ".join(b.rstrip(".") + "." for b in body)
            if status == "qualified":
                answer += (" (Lưu ý: đáp án là phương án hợp lý nhất; chưa được kiểm chứng đầy đủ bằng nguồn sách "
                           "giáo khoa.)")
    else:
        supported_claims = [c.text for c in claims if not c.text.startswith("Bằng chứng mâu thuẫn")]
        if not supported_claims:
            status = "abstain"
            answer = "Chưa đủ bằng chứng từ sách giáo khoa để trả lời câu hỏi này."
        else:
            status = "supported" if not unresolved and n_conflicts == 0 else "qualified"
            unresolved_q = [p.claim.text for p in packets if p.subgoal_id in unresolved]
            answer = None
            if verbalize is not None:
                try:
                    answer = verbalize(question, supported_claims, unresolved_q, status)
                except Exception:
                    answer = None
            if not answer:
                answer = " ".join(s.rstrip(".") + "." for s in supported_claims)
                if unresolved_q:
                    answer += " Một số điểm chưa đủ bằng chứng: " + "; ".join(unresolved_q[:3]) + "."
    if n_conflicts:
        notes.append(f"workspace_issues:{n_conflicts}")
        if status == "supported":
            status = "qualified"
    fa = FinalAnswer(question_id=question_id, system=system, answer=answer, answer_choice=answer_choice,
                     status=status, choice_verdicts=verdict_list, claims=claims, unresolved_subgoals=unresolved,
                     proof_packet_ids=[p.packet_id for p in packets], plan=plan)
    fa.tool_usage["composer_notes"] = notes
    if choices:
        fa.tool_usage["choice_rule"] = how
    fa.tool_usage["invalid_span_ids"] = sorted(set(invalid_all))
    fa.tool_usage["workspace_issues"] = {k2: len(v) for k2, v in issues.items() if v}
    return fa
