"""Master agent (spec 5.1, 5.4, 5.5).

1. subject / question-type detection (deterministic; planner LLM only for non-simple items);
2. adaptive Question Dependency DAG: 1 subgoal for simple items, AND/OR acyclic DAG otherwise
   (for MCQ a sink `choice_evaluation` node depending on every other subgoal is appended);
3. dispatch of ready subgoals (all AND-deps / one per OR-group validated) to homogeneous
   Subagents, concurrently, bounded by `max_subagents` (thread pool = semaphore);
4. targeted re-planning only for *specific* missing proof obligations, within budgets
   (`max_replan_rounds`, `max_depth`, a hard cap on subagent runs and on tool calls);
5. deterministic composition (otter.reasoning.composer) + optional LLM verbalization.
"""
from __future__ import annotations

import itertools
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from otter.agents import prompts as P
from otter.agents.runtime import (LRU, Meter, MeteredLLM, QuestionInput, detect_question_type, extract_letter,
                                  normalize_subject, safe_json)
from otter.agents.subagent import CHOICE_NODE_TYPE, Subagent, SubagentResult
from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.composer import ChoiceOutcome, compose
from otter.reasoning.proof_workspace import ProofWorkspace
from otter.reasoning.text import normalize, truncate
from otter.schemas.reasoning import FinalAnswer, QuestionPlan, Subgoal
from otter.tools.base import ToolContext
from otter.tools.registry import ToolRegistry

VALID_TYPES = {"fact_lookup", "definition", "condition_check", "rule_application", "calculation", "comparison",
               "causal_explanation", "choice_evaluation", "other"}
SINK_ID = "sg_choice"


@dataclass
class OtterConfig:
    max_subagents: int = 4                 # concurrent subagents per question
    max_depth: int = 3                     # DAG depth incl. follow-up subgoals; also proof-engine depth
    max_replan_rounds: int = 2
    max_tool_calls_per_subagent: int = 8
    max_subgoals: int = 4                  # planner subgoals (excluding the MCQ sink)
    max_question_seconds: float = 900.0
    decomposition: bool = True             # ablation no_decomposition -> False
    graph_retrieval: bool = True           # no_graph_retrieval -> disable search_knowledge / expand_graph
    source_verification: bool = True       # no_source_verification -> no verify_evidence, raw citations
    proof_completion: bool = True          # no_proof_completion -> no targeted follow-up subgoals
    rule_execution: bool = True            # no_rule_execution -> disable execute_rule
    workspace: bool = True                 # False -> multi_agent_no_workspace (independent packets, flat agg.)
    warm_start: bool = True
    always_plan: bool = False              # force the planner LLM even for simple items
    choice_prior: bool = True              # closed-book prior choice: MCQ tie-breaker / fallback only
    keep_transcripts: bool = False
    keep_trace: bool = True                # compact proof-workspace snapshot in FinalAnswer.tool_usage["workspace"]

    @property
    def disabled(self) -> frozenset:
        d = set()
        if not self.graph_retrieval:
            d |= {"search_knowledge", "expand_graph"}
        if not self.rule_execution:
            d.add("execute_rule")
        if not self.source_verification:
            d.add("verify_evidence")
        return frozenset(d)

    @property
    def max_runs(self) -> int:
        return self.max_subgoals + 1 + self.max_subagents * self.max_replan_rounds

    @property
    def max_total_tool_calls(self) -> int:
        return self.max_runs * (self.max_tool_calls_per_subagent + 1)


# --------------------------------------------------------------------------- planning
def validate_plan(raw: Any, max_subgoals: int) -> tuple[Optional[list[Subgoal]], str]:
    """Parse planner JSON into subgoals; returns (subgoals | None, error)."""
    if not isinstance(raw, dict):
        return None, "plan_not_object"
    items = raw.get("subgoals")
    if not isinstance(items, list) or not items:
        return None, "no_subgoals"
    sgs: list[Subgoal] = []
    ids: list[str] = []
    for i, it in enumerate(items[:max_subgoals]):
        if not isinstance(it, dict) or not str(it.get("question") or "").strip():
            continue
        sid = re.sub(r"[^\w\-]", "_", str(it.get("id") or f"sg{i + 1}"))[:20] or f"sg{i + 1}"
        if sid in ids or sid == SINK_ID:
            sid = f"sg{i + 1}_{len(ids)}"
        ids.append(sid)
        t = str(it.get("type") or it.get("expected_claim_type") or "other")
        hints = it.get("hints") or it.get("ontology_hints") or []
        sgs.append(Subgoal(subgoal_id=sid, question=str(it["question"]).strip()[:400],
                           requires=[str(x) for x in (it.get("requires") or []) if isinstance(x, (str, int))],
                           requires_any=[[str(y) for y in g] for g in (it.get("requires_any") or [])
                                         if isinstance(g, list) and g],
                           expected_claim_type=t if t in VALID_TYPES and t != CHOICE_NODE_TYPE else "other",
                           ontology_hints=[str(h)[:60] for h in hints if isinstance(h, (str, int))][:5]))
    if not sgs:
        return None, "no_valid_subgoals"
    known = {s.subgoal_id for s in sgs}
    for s in sgs:                      # drop references to unknown ids (do not invent dependencies)
        s.requires = [r for r in s.requires if r in known and r != s.subgoal_id]
        s.requires_any = [[r for r in g if r in known and r != s.subgoal_id] for g in s.requires_any]
        s.requires_any = [g for g in s.requires_any if g]
    g = nx.DiGraph()
    g.add_nodes_from(known)
    for s in sgs:
        for r in s.requires + [x for grp in s.requires_any for x in grp]:
            g.add_edge(r, s.subgoal_id)
    if not nx.is_directed_acyclic_graph(g):
        return None, "cycle"
    return sgs, ""


@dataclass
class _Node:
    sg: Subgoal
    state: str = "pending"                 # pending | running | done | blocked
    followups: list[str] = field(default_factory=list)
    depth: int = 0
    is_sink: bool = False
    runs: int = 0
    result: Optional[SubagentResult] = None
    replanned: bool = False


class MasterAgent:
    def __init__(self, llm: Any, store: Any, *, config: Optional[OtterConfig] = None, system_name: str = "otter_full",
                 llm_cache: Optional[LRU] = None, resolver: Optional[CitationResolver] = None):
        self.llm = llm
        self.store = store
        self.cfg = config or OtterConfig()
        self.name = system_name
        self.llm_cache = llm_cache
        self.resolver = resolver or CitationResolver(store)

    # ------------------------------------------------------------------ entry
    def answer(self, item: dict[str, Any]) -> FinalAnswer:
        t0 = time.perf_counter()
        meter = Meter()
        qid = str(item.get("id") or "")
        try:
            fa = self._answer(item, meter, t0)
        except Exception as e:                        # never raise
            import traceback
            fa = FinalAnswer(question_id=qid, system=self.name, answer="", status="failed",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")
        fa.tool_usage.update(meter.as_dict())
        fa.tool_usage["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return fa

    def _answer(self, item: dict[str, Any], meter: Meter, t0: float) -> FinalAnswer:
        cfg = self.cfg
        q = QuestionInput.from_item(item)
        llm = MeteredLLM(self.llm, meter, cache=self.llm_cache)
        ws = ProofWorkspace(q.qid, subject=q.subject)
        ctx = ToolContext(store=self.store, subject=q.subject, grade=q.grade, judge_llm=llm, workspace=ws,
                          disabled=cfg.disabled, max_hops=cfg.max_depth,
                          extra={"citation_resolver": self.resolver})
        reg = ToolRegistry(ctx, max_total_calls=cfg.max_total_tool_calls)
        qtype = detect_question_type(q.question, q.choices)
        plan, plan_note = self._plan(q, qtype, llm)
        nodes = self._nodes(plan, q)
        agent_ids = itertools.count(1)
        lock = threading.Lock()
        rounds_used = 0
        asked: set[str] = set()
        runs = 0
        deadline = t0 + cfg.max_question_seconds

        prior_ex = ThreadPoolExecutor(max_workers=1) if (q.choices and cfg.choice_prior and cfg.workspace) else None
        prior_fu = prior_ex.submit(self._direct_choice, q, llm) if prior_ex else None

        def deps_for(n: _Node) -> list[str]:
            if not cfg.workspace:
                return []
            return n.sg.requires + [x for g in n.sg.requires_any for x in g] + n.followups

        def latest_status(sid: str) -> Optional[str]:
            p = ws.latest(sid)
            return p.status if p else None

        def terminal(sid: str) -> bool:
            return nodes[sid].state in ("done", "blocked")

        def dep_payload(n: _Node) -> list[dict[str, Any]]:
            out = []
            for d in deps_for(n):
                p = ws.latest(d)
                if p is None:
                    continue
                if p.status != "supported" and not n.is_sink:
                    continue
                ev = []
                if p.source_span_ids:
                    for sp in self.store.get_spans(p.source_span_ids[:3]):
                        ev.append((sp.span_id, truncate(sp.text, 260)))
                out.append({"packet_id": p.packet_id, "status": p.status, "claim": p.claim.text, "evidence": ev})
            return out

        def readiness(n: _Node) -> str:
            """ready | wait | blocked | skip (follow-ups gave nothing new -> keep current packet)."""
            ds = deps_for(n)
            if any(not terminal(d) for d in ds):
                return "wait"
            if n.followups and n.result is not None:
                if not any(latest_status(f) == "supported" for f in n.followups):
                    return "skip"
                return "ready"
            if n.is_sink or not cfg.workspace:
                return "ready"
            and_ok = all(latest_status(d) == "supported" for d in n.sg.requires)
            or_ok = all(any(latest_status(x) == "supported" for x in g) for g in n.sg.requires_any)
            return "ready" if and_ok and or_ok else "blocked"

        def run_node(n: _Node) -> SubagentResult:
            aid = f"subagent_{next(agent_ids)}"
            budget = cfg.max_tool_calls_per_subagent
            sa = Subagent(aid, llm, reg, max_tool_calls=budget, verify=cfg.source_verification,
                          warm_start=cfg.warm_start)
            notes = []
            if n.is_sink:
                unresolved = [d for d in deps_for(n) if latest_status(d) != "supported"]
                if unresolved:
                    notes.append("Một số mục tiêu con chưa được kiểm chứng: " + ", ".join(unresolved) + ".")
            return sa.run(n.sg, q, dep_payload(n), notes)

        def maybe_replan(n: _Node, res: SubagentResult) -> bool:
            nonlocal rounds_used
            p = res.packet
            if not (cfg.proof_completion and cfg.workspace) or rounds_used >= cfg.max_replan_rounds:
                return False
            if n.depth + 1 >= cfg.max_depth or n.replanned or time.perf_counter() > deadline:
                return False
            if p.status == "supported" or p.status == "failed":
                return False
            ent = [k for k, v in res.choice_verdicts.items() if v == "entailed"] if n.is_sink else []
            if n.is_sink and len(ent) == 1 and ent[0] == res.best_choice:
                return False                         # verified and agreed: nothing to complete
            # only when the corpus actually returned material (otherwise follow-ups cannot help)
            if not any(c.tool in ("retrieve_source", "search_knowledge") and c.status in ("ok", "partial")
                       for c in p.tool_results):
                return False
            obligations = []
            for ob in p.missing_premises:
                key = normalize(ob)
                if len(key) < 6 or len(key) > 300 or key == normalize(n.sg.question) or key in asked:
                    continue
                obligations.append(ob)
                if len(obligations) >= 2:
                    break
            if n.is_sink and q.choices and not obligations:
                # specific obligation per contender choice: confirm or refute it from the textbook
                prior = prior_fu.result() if (prior_fu is not None and prior_fu.done()) else None
                contra = {k for k, v in res.choice_verdicts.items() if v == "contradicted"}
                cands = [k for k in dict.fromkeys([res.best_choice, prior] + ent) if k in q.choices and k not in contra]
                for k in cands[:2]:
                    ob = f"với câu hỏi “{q.question}”, lựa chọn “{q.choices[k]}” có phải là đáp án đúng không"
                    if normalize(ob) not in asked:
                        obligations.append(ob)
            if not obligations or runs + len(obligations) + 1 > cfg.max_runs:
                return False
            rounds_used += 1
            n.replanned = True
            ctype = "condition_check" if n.sg.expected_claim_type in ("condition_check", "rule_application") \
                else "fact_lookup"
            for j, ob in enumerate(obligations, 1):
                asked.add(normalize(ob))
                fid = f"{n.sg.subgoal_id}_f{rounds_used}{j}"
                nodes[fid] = _Node(sg=Subgoal(subgoal_id=fid, question=f"Tìm bằng chứng trong sách giáo khoa xác nhận "
                                                                       f"hoặc bác bỏ: {ob}",
                                              expected_claim_type=ctype, ontology_hints=[ob[:60]]),
                                   depth=n.depth + 1)
                n.followups.append(fid)
            n.state = "pending"
            ws.log.append({"event": "replan", "subgoal_id": n.sg.subgoal_id, "followups": list(n.followups),
                           "obligations": obligations})
            return True

        blocked: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, cfg.max_subagents)) as ex:
            running: dict[Future, _Node] = {}
            while True:
                with lock:
                    for n in list(nodes.values()):
                        if n.state != "pending" or n in running.values():
                            continue
                        r = readiness(n)
                        if r == "blocked":
                            n.state = "blocked"
                            n.sg.status = "blocked"
                            blocked.append(n.sg.subgoal_id)
                            ws.log.append({"event": "blocked", "subgoal_id": n.sg.subgoal_id})
                        elif r == "skip":
                            n.state = "done"
                        elif r == "ready":
                            over = runs >= cfg.max_runs or time.perf_counter() > deadline
                            if over and not (n.is_sink and n.result is None):
                                n.state = "done" if n.result is not None else "blocked"
                                if n.result is None:
                                    blocked.append(n.sg.subgoal_id)
                                continue
                            n.state = "running"
                            n.runs += 1
                            runs += 1
                            running[ex.submit(run_node, n)] = n
                if not running:
                    # anything still pending can never become ready -> blocked (no silent success)
                    pend = [n for n in nodes.values() if n.state == "pending"]
                    if not pend:
                        break
                    for n in pend:
                        n.state = "done" if n.result is not None else "blocked"
                        if n.result is None:
                            blocked.append(n.sg.subgoal_id)
                    continue
                done_set, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fu in done_set:
                    n = running.pop(fu)
                    res = fu.result()
                    pub = ws.publish(res.packet)
                    res.packet = pub
                    for d in pub.dependent_packet_ids:
                        dp = ws.get(d)
                        if dp is not None and dp.status == "supported":
                            ws.link(d, "supports", pub.packet_id)
                    with lock:
                        n.result = res
                        n.state = "done"
                        n.sg.status = "done" if pub.status != "failed" else "failed"
                        maybe_replan(n, res)

        # ---------------------------------------------------------------- compose
        packets = ws.latest_all()
        issues = ws.issues()
        choice = None
        fallback_used = False
        if q.choices:
            sink = nodes.get(SINK_ID)
            if cfg.workspace and sink is not None and sink.result is not None:
                r = sink.result
                choice = ChoiceOutcome(verdicts=dict(r.choice_verdicts), spans=dict(r.choice_spans),
                                       best_choice=r.best_choice, packet_id=r.packet.packet_id)
            elif not cfg.workspace:
                choice = ChoiceOutcome(best_choice=self._flat_aggregate(q, packets, llm))
            else:
                choice = ChoiceOutcome()
            if prior_fu is not None:
                choice.fallback_choice = prior_fu.result()
                prior_ex.shutdown(wait=False)
            elif not choice.best_choice and not any(v == "entailed" for v in choice.verdicts.values()):
                choice.fallback_choice = self._direct_choice(q, llm)
                fallback_used = True
        verbalize = None
        if not q.choices:
            def verbalize(question, supported, unresolved, status):
                obj, _ = safe_json(llm, [{"role": "system", "content": P.VERBALIZE_SYSTEM},
                                         {"role": "user", "content": P.verbalize_user(question, supported, unresolved,
                                                                                      status)}], max_tokens=500, retry_msg=P.JSON_RETRY)
                return str(obj.get("answer")).strip() if isinstance(obj, dict) and obj.get("answer") else None
        qplan = QuestionPlan(question_id=q.qid, subject=q.subject, question_type=qtype,
                             subgoals=[n.sg for n in nodes.values()], rationale=plan_note)
        fa = compose(question_id=q.qid, system=self.name, question=q.question, choices=q.choices, packets=packets,
                     plan=qplan, resolver=self.resolver, issues=issues, choice=choice,
                     blocked_subgoals=sorted(set(blocked)), verbalize=verbalize,
                     allow_unverified_citations=not cfg.source_verification)
        usage = reg.usage()
        fa.tool_usage.update({
            "calls": usage["calls"], "by_tool": usage["by_tool"], "subagents": runs,
            "subgoals": len(nodes), "replan_rounds": rounds_used, "question_type": qtype, "subject": q.subject,
            "fallback_direct_choice": fallback_used, "prior_choice": choice.fallback_choice if choice else None,
            "workspace_edges": len(ws.edges), "workspace_log": ws.log[-30:],
        })
        if cfg.keep_trace:
            fa.tool_usage["workspace"] = ws.snapshot()
        if cfg.keep_transcripts:
            fa.tool_usage["transcripts"] = {n.sg.subgoal_id: n.result.transcript for n in nodes.values()
                                           if n.result is not None}
        return fa

    # ------------------------------------------------------------------ helpers
    def _plan(self, q: QuestionInput, qtype: str, llm: Any) -> tuple[list[Subgoal], str]:
        cfg = self.cfg
        simple = Subgoal(subgoal_id="sg1", question=q.question,
                         expected_claim_type="fact_lookup" if qtype == "direct" else "other")
        if not cfg.decomposition or (qtype == "direct" and not cfg.always_plan):
            return [simple], f"single_subgoal({'no_decomposition' if not cfg.decomposition else qtype})"
        msgs = [{"role": "system", "content": P.PLANNER_SYSTEM},
                {"role": "user", "content": P.planner_user(q.question, q.choices, q.subject, q.grade, cfg.max_subgoals)}]
        obj, _ = safe_json(llm, msgs, max_tokens=700, retry_msg=P.INVALID_ACTION)
        sgs, err = validate_plan(obj, cfg.max_subgoals)
        if err == "cycle":                              # one plan revision
            msgs += [{"role": "assistant", "content": P.to_json(obj, 3000)},
                     {"role": "user", "content": "Kế hoạch có chu trình phụ thuộc. Hãy sửa lại thành DAG không chu "
                                                 "trình và trả về đúng một đối tượng JSON."}]
            obj, _ = safe_json(llm, msgs, max_tokens=700)
            sgs, err = validate_plan(obj, cfg.max_subgoals)
        if sgs is None:
            return [simple], f"planner_fallback({err})"
        if isinstance(obj, dict):
            subj = normalize_subject(obj.get("subject"))
            if q.subject is None and subj:
                q.subject = subj
        return sgs, str(obj.get("rationale", ""))[:300] if isinstance(obj, dict) else ""

    def _nodes(self, sgs: list[Subgoal], q: QuestionInput) -> dict[str, _Node]:
        nodes: dict[str, _Node] = {}
        g = nx.DiGraph()
        for s in sgs:
            g.add_node(s.subgoal_id)
            for r in s.requires + [x for grp in s.requires_any for x in grp]:
                g.add_edge(r, s.subgoal_id)
        depth = {n: 0 for n in g}
        for n in nx.topological_sort(g):
            for m in g.successors(n):
                depth[m] = max(depth[m], depth[n] + 1)
        for s in sgs:
            nodes[s.subgoal_id] = _Node(sg=s, depth=depth.get(s.subgoal_id, 0))
        if q.choices and self.cfg.workspace:
            if len(sgs) == 1 and not sgs[0].requires:
                # simple item: the single subgoal IS the choice evaluation (one subagent)
                s = sgs[0]
                nodes = {SINK_ID: _Node(sg=Subgoal(subgoal_id=SINK_ID, question=s.question,
                                                   expected_claim_type=CHOICE_NODE_TYPE,
                                                   ontology_hints=s.ontology_hints), is_sink=True)}
            else:
                nodes[SINK_ID] = _Node(sg=Subgoal(subgoal_id=SINK_ID, question="Đánh giá từng lựa chọn và chọn đáp án "
                                                  "đúng dựa trên các kết quả đã kiểm chứng.",
                                                  requires=[s.subgoal_id for s in sgs],
                                                  expected_claim_type=CHOICE_NODE_TYPE),
                                       is_sink=True, depth=max(depth.values(), default=0) + 1)
        return nodes

    def _flat_aggregate(self, q: QuestionInput, packets: list, llm: Any) -> Optional[str]:
        findings = []
        for p in packets:
            ev = ""
            if p.source_span_ids:
                sp = self.store.get_spans(p.source_span_ids[:1])
                if sp:
                    ev = f" [{sp[0].span_id}] {truncate(sp[0].text, 250)}"
            findings.append(f"({p.status}) {p.claim.text}{ev}")
        obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                   {"role": "user", "content": P.flat_aggregate_user(q.question, q.choices, q.subject,
                                                                                     q.grade, findings)}],
                             max_tokens=400, retry_msg=P.JSON_RETRY)
        return extract_letter(obj.get("answer") if isinstance(obj, dict) else raw, q.choices)

    def _direct_choice(self, q: QuestionInput, llm: Any) -> Optional[str]:
        try:
            obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                       {"role": "user", "content": P.direct_user(q.question, q.choices, q.subject,
                                                                                 q.grade)}], max_tokens=300, retry_msg=P.JSON_RETRY)
            return extract_letter(obj.get("answer") if isinstance(obj, dict) else raw, q.choices)
        except Exception:
            return None
