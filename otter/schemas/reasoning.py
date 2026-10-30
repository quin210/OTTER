"""Online reasoning contracts: subgoal DAG, tool results, reasoning packets,
proof workspace records and the final answer (spec sections 4-5)."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PacketStatus = Literal["supported", "contradicted", "undetermined", "failed"]
InferenceStatus = Literal["entailed", "contradicted", "undetermined", "blocked"]
CheckStatus = Literal["pass", "fail", "needs_review", "not_applicable"]
ClaimType = Literal["fact_lookup", "definition", "condition_check", "rule_application", "calculation",
                    "comparison", "causal_explanation", "choice_evaluation", "other"]


class Subgoal(BaseModel):
    subgoal_id: str
    question: str
    requires: list[str] = Field(default_factory=list)       # AND-dependencies (subgoal ids)
    requires_any: list[list[str]] = Field(default_factory=list)  # OR-groups
    expected_claim_type: ClaimType = "other"
    ontology_hints: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "blocked", "failed"] = "pending"


class QuestionPlan(BaseModel):
    question_id: str
    subject: Optional[str] = None
    question_type: str = "direct"                            # direct | multi_hop | conditional | calculation | choice_check
    subgoals: list[Subgoal]
    rationale: str = ""


class ToolCall(BaseModel):
    call_id: str
    tool: Literal["search_knowledge", "expand_graph", "retrieve_source", "execute_rule", "calculate", "verify_evidence"]
    arguments: dict[str, Any]
    status: str                                              # tool-specific status (ok/entailed/blocked/unknown/...)
    result: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None


class Claim(BaseModel):
    text: str
    canonical_predicate: Optional[str] = None
    arguments: list[str] = Field(default_factory=list)


class PacketValidation(BaseModel):
    schema_: CheckStatus = Field("pass", alias="schema")
    rule_execution: CheckStatus = "not_applicable"
    calculation: CheckStatus = "not_applicable"
    source_ids_exist: CheckStatus = "needs_review"
    semantic_support: CheckStatus = "needs_review"          # heuristic / LLM-assisted, disclosed as such

    model_config = {"populate_by_name": True}


class ReasoningPacket(BaseModel):
    packet_id: str                                           # "p_<subgoal>_v<k>"
    version: int = 1
    subgoal_id: str
    agent_id: str
    claim: Claim
    status: PacketStatus
    proof_mode: Literal["formal", "source_supported", "mixed", "none"] = "source_supported"
    used_assertion_ids: list[str] = Field(default_factory=list)
    used_rule_ids: list[str] = Field(default_factory=list)
    source_span_ids: list[str] = Field(default_factory=list)
    dependent_packet_ids: list[str] = Field(default_factory=list)
    tool_results: list[ToolCall] = Field(default_factory=list)
    missing_premises: list[str] = Field(default_factory=list)  # proof obligations the agent could not close
    counterevidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    validation: PacketValidation = Field(default_factory=PacketValidation)
    created_at: Optional[str] = None


class Citation(BaseModel):
    book_id: str
    book_title: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[int] = None
    chapter_id: Optional[str] = None
    lesson_id: Optional[str] = None
    lesson_title: Optional[str] = None
    page_number: Optional[str] = None                        # printed page only when verified; else null
    pdf_page_number: Optional[int] = None
    span_id: str
    quote: Optional[str] = None


class AnswerClaim(BaseModel):
    claim_id: str
    text: str
    packet_ids: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class ChoiceVerdict(BaseModel):
    choice_id: str
    status: InferenceStatus
    packet_ids: list[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    question_id: str
    system: str                                              # baseline / variant name
    answer: str
    answer_choice: Optional[str] = None
    status: Literal["supported", "qualified", "abstain", "failed"] = "supported"
    choice_verdicts: list[ChoiceVerdict] = Field(default_factory=list)
    claims: list[AnswerClaim] = Field(default_factory=list)
    # optional (otter_v2): decisive passage(s) to read: resolved book/grade/lesson/page/span + short quote
    where_to_read: list[Citation] = Field(default_factory=list)
    unresolved_subgoals: list[str] = Field(default_factory=list)
    proof_packet_ids: list[str] = Field(default_factory=list)
    plan: Optional[QuestionPlan] = None
    tool_usage: dict[str, Any] = Field(default_factory=lambda: {"calls": 0, "latency_ms": None, "prompt_tokens": 0,
                                                                   "completion_tokens": 0, "llm_calls": 0, "subagents": 0})
    error: Optional[str] = None
