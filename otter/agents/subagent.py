"""Homogeneous Subagent (spec 5.2): same prompt + same ToolRegistry for every instance.

The tool loop is plain JSON prompting (one JSON action per turn), so it works without native
tool calling. After `finish`, deterministic system checks (not the LLM) decide the packet
status: an LLM-proposed `supported` survives only if `verify_evidence` (structural + source-link
+ quote-verified semantic support) or a strict `execute_rule` entailment backs it. The agent can
never self-assert `verified`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from otter.agents import prompts as P
from otter.agents.runtime import QuestionInput, extract_letter
from otter.llm.interface import parse_json_loose
from otter.reasoning.text import truncate
from otter.schemas.reasoning import Claim, PacketValidation, ReasoningPacket, Subgoal, ToolCall
from otter.tools.registry import ToolRegistry

CHOICE_NODE_TYPE = "choice_evaluation"
_ID_KEYS = ("id", "span_id", "rule_id", "assertion_id", "concept_id", "relation_id", "subject_id", "object_id")


@dataclass
class SubagentResult:
    packet: ReasoningPacket
    choice_verdicts: dict[str, str] = field(default_factory=dict)       # choice -> entailed|contradicted|undetermined
    choice_spans: dict[str, list[str]] = field(default_factory=dict)    # choice -> verified evidence span ids
    agent_verdicts: dict[str, str] = field(default_factory=dict)
    best_choice: Optional[str] = None
    turns: int = 0
    transcript: list[dict[str, str]] = field(default_factory=list)


def _collect_ids(x: Any, out: set[str]) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            if k in _ID_KEYS and isinstance(v, str):
                out.add(v)
            elif k in ("source_span_ids", "linked_span_ids", "seed_ids", "used_rule_ids", "concept_ids",
                       "evidence_span_ids", "formula_source_span_ids", "verified_span_ids") and isinstance(v, list):
                out.update(str(i) for i in v if isinstance(i, str))
            else:
                _collect_ids(v, out)
    elif isinstance(x, list):
        for v in x:
            _collect_ids(v, out)


def _ids(x: Any) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        return [str(i) for i in x if isinstance(i, (str, int)) and str(i)]
    return []


def compact_result(tool: str, res: dict[str, Any]) -> dict[str, Any]:
    """Shrink a tool result for the LLM context (full results stay in the packet's tool_results)."""
    r = {k: v for k, v in res.items() if k not in ("trace", "score_components", "sub_proofs", "kb_conflicts",
                                                     "parse_errors", "notes", "ignored_arguments")}
    if tool == "search_knowledge":
        hits = []
        for h in res.get("hits", [])[:8]:
            v = {"kind": h.get("kind"), "id": h.get("id")}
            if h.get("kind") == "concept":
                v["label"] = h.get("label")
                if h.get("definition"):
                    v["definition"] = truncate(h["definition"], 200)
            elif h.get("kind") == "fact":
                v["text"] = truncate(h.get("text", ""), 250)
            elif h.get("kind") == "rule":
                v["if"] = [a.get("text") or a.get("predicate") for a in h.get("premises", [])]
                v["guards"] = [a.get("text") or a.get("predicate") for a in h.get("guards", [])]
                v["then"] = [("KHÔNG " if a.get("negated") else "") + (a.get("text") or a.get("predicate"))
                             for a in h.get("conclusions", [])]
                v["predicates"] = {"premises": [a["predicate"] for a in h.get("premises", [])],
                                   "guards": [a["predicate"] for a in h.get("guards", [])]}
                v["source_support"] = h.get("source_support")
                if h.get("formula"):
                    v["formula"] = h["formula"]
            v["source_span_ids"] = h.get("source_span_ids", [])[:3]
            hits.append(v)
        r = {"status": res.get("status"), "hits": hits}
    elif tool == "retrieve_source":
        r = {"status": res.get("status"), "missing_span_ids": res.get("missing_span_ids", []),
             "spans": [{"span_id": s["span_id"], "heading": " > ".join(s.get("heading_path", [])[-2:]),
                        "text": truncate(s.get("text", ""), 500)} for s in res.get("spans", [])[:6]]}
    elif tool == "expand_graph":
        r = {"status": res.get("status"), "invalid_ids": res.get("invalid_ids", []),
             "nodes": [{"id": n["id"], "label": n.get("label")} for n in res.get("nodes", [])[:15]],
             "relations": [{"s": x["subject_id"], "rel": x["relation_type"], "o": x["object_id"],
                            "span": (x.get("source_span_ids") or [None])[0]} for x in res.get("relations", [])[:20]],
             "rules": [{"id": x["id"], "then": [c.get("text") for c in x.get("conclusions", [])]}
                       for x in res.get("rules", [])[:8]]}
    return r


def _parse_action(text: str) -> tuple[Optional[str], dict[str, Any]]:
    try:
        obj = parse_json_loose(text)
    except ValueError:
        return None, {}
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else {}
    if not isinstance(obj, dict):
        return None, {}
    action = obj.get("action") or obj.get("tool") or obj.get("name")
    args = obj.get("arguments") or obj.get("args") or obj.get("parameters") or obj.get("action_input") or {}
    if action is None and ("claim" in obj or "choices" in obj or "best_choice" in obj):
        action, args = "finish", obj
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"query": args}
    return (str(action).strip() if action else None), (args if isinstance(args, dict) else {})


class Subagent:
    def __init__(self, agent_id: str, llm: Any, registry: ToolRegistry, *, max_tool_calls: int = 8,
                 verify: bool = True, warm_start: bool = True, max_tokens: int = 700):
        self.agent_id = agent_id
        self.llm = llm
        self.reg = registry
        self.max_tool_calls = max_tool_calls
        self.verify = verify
        self.warm_start = warm_start
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ public
    def run(self, sg: Subgoal, q: QuestionInput, deps: list[dict[str, Any]], notes: Optional[list[str]] = None
            ) -> SubagentResult:
        calls: list[ToolCall] = []
        seen: set[str] = set()
        is_choice = sg.expected_claim_type == CHOICE_NODE_TYPE and bool(q.choices)
        dep_spans = [sid for d in deps for sid, _ in d.get("evidence", [])]
        seen.update(dep_spans)                       # evidence of verified dependency packets may be cited
        try:
            args, turns, transcript = self._loop(sg, q, deps, notes or [], is_choice, calls, seen)
            return self._finalize(sg, q, deps, args, calls, seen, is_choice, turns, transcript, dep_spans)
        except Exception as e:                      # never raise: emit a failed packet
            pk = ReasoningPacket(packet_id=f"p_{sg.subgoal_id}", subgoal_id=sg.subgoal_id, agent_id=self.agent_id,
                                 claim=Claim(text=sg.question), status="failed", proof_mode="none",
                                 tool_results=calls, limitations=[f"subagent_error: {type(e).__name__}: {e}"[:300]],
                                 dependent_packet_ids=[d["packet_id"] for d in deps],
                                 validation=PacketValidation(source_ids_exist="not_applicable",
                                                             semantic_support="not_applicable"))
            return SubagentResult(packet=pk)

    # ------------------------------------------------------------------ tool loop
    def _call(self, tool: str, args: dict[str, Any], calls: list[ToolCall], seen: set[str]) -> ToolCall:
        tc = self.reg.call(tool, args, agent_id=self.agent_id)
        calls.append(tc)
        _collect_ids(tc.result, seen)
        return tc

    def _loop(self, sg, q, deps, notes, is_choice, calls, seen):
        enabled = list(self.reg.enabled)
        system = P.subagent_system(enabled, self.max_tool_calls)
        user = P.subagent_user(sg.question, sg.expected_claim_type, q.question, q.choices, q.subject, q.grade,
                               deps, sg.ontology_hints, is_choice, notes)
        used = 0
        warm: list[str] = []
        if self.warm_start:
            query = sg.question if not is_choice else q.question
            hint = " ".join(sg.ontology_hints[:4])
            if "search_knowledge" in enabled:
                tc = self._call("search_knowledge", {"query": f"{query} {hint}".strip(), "top_k": 6}, calls, seen)
                used += 1
                warm.append(P.observation_msg(tc.call_id, tc.tool, P.to_json(compact_result(tc.tool, tc.result), 2500),
                                              self.max_tool_calls - used))
            if "retrieve_source" in enabled:
                tq = f"{query} {' '.join((q.choices or {}).values())}" if is_choice else f"{query} {hint}"
                tc = self._call("retrieve_source", {"query": tq.strip(), "top_k": 5}, calls, seen)
                used += 1
                warm.append(P.observation_msg(tc.call_id, tc.tool, P.to_json(compact_result(tc.tool, tc.result), 3200),
                                              self.max_tool_calls - used))
        if warm:
            user += "\n\nKết quả tra cứu ban đầu (hệ thống đã gọi sẵn):\n" + "\n".join(warm)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        invalid = 0
        turns = 0
        max_turns = self.max_tool_calls - used + 2
        while turns < max_turns:
            turns += 1
            r = self.llm.chat(msgs, temperature=0.0, max_tokens=self.max_tokens)
            action, args = _parse_action(r.text)
            msgs.append({"role": "assistant", "content": _clip_assistant(r.text)})
            if action == "finish":
                return args, turns, msgs
            if action is None or action not in enabled:
                invalid += 1
                if invalid > 2:
                    break
                msg = P.INVALID_ACTION if action is None else (
                    f"Công cụ '{action}' không tồn tại hoặc bị tắt. Công cụ hợp lệ: {', '.join(enabled)} hoặc finish.")
                msgs.append({"role": "user", "content": msg})
                continue
            if used >= self.max_tool_calls or (self.reg.budget_left() == 0):
                msgs.append({"role": "user", "content": P.FORCE_FINISH})
                continue
            tc = self._call(action, args, calls, seen)
            used += 1
            msgs.append({"role": "user", "content": P.observation_msg(
                tc.call_id, tc.tool, P.to_json(compact_result(tc.tool, tc.result), 3000), self.max_tool_calls - used)})
        # forced finish
        msgs.append({"role": "user", "content": P.FORCE_FINISH})
        r = self.llm.chat(msgs, temperature=0.0, max_tokens=self.max_tokens)
        action, args = _parse_action(r.text)
        msgs.append({"role": "assistant", "content": _clip_assistant(r.text)})
        return (args if action == "finish" or args else {}), turns + 1, msgs

    # ------------------------------------------------------------------ deterministic checks
    def _finalize(self, sg, q, deps, args, calls, seen, is_choice, turns, transcript, dep_spans=()) -> SubagentResult:
        limitations: list[str] = []
        claim_text = str(args.get("claim") or "").strip() or sg.question

        def keep(ids: list[str], kind: str) -> list[str]:
            ok = [i for i in ids if i in seen]
            dropped = [i for i in ids if i not in seen]
            if dropped:
                limitations.append(f"dropped_unseen_{kind}_ids:{','.join(dropped[:5])}")
            return list(dict.fromkeys(ok))

        span_ids = keep(_ids(args.get("span_ids")), "span")
        a_ids = keep(_ids(args.get("assertion_ids")), "assertion")
        r_ids = keep(_ids(args.get("rule_ids")), "rule")
        # LLMs often put ids in the wrong list: re-type by what the store says they are
        store = self.reg.ctx.store
        for lst in (a_ids, r_ids):
            for i in list(lst):
                if store.span_exists(i):
                    lst.remove(i)
                    if i not in span_ids:
                        span_ids.append(i)
        for i in list(a_ids):
            if store.get_rule(i) is not None:
                a_ids.remove(i)
                r_ids.append(i)
            elif store.get_fact(i) is None:
                a_ids.remove(i)                    # concept ids etc. are not assertions
        for i in list(r_ids):
            if store.get_rule(i) is None:
                r_ids.remove(i)
        missing = [str(m).strip() for m in (args.get("missing_premises") or []) if str(m).strip()][:4]
        proposed = str(args.get("proposed_status") or "").strip().lower()

        # rule / calculation checks from the agent's own tool calls
        rule_calls = [c for c in calls if c.tool == "execute_rule"]
        entailed_rules = [c for c in rule_calls if c.status == "entailed"]
        rule_check = "not_applicable"
        rule_spans: list[str] = []
        if rule_calls:
            rule_check = "pass" if entailed_rules else ("fail" if any(c.status == "blocked" for c in rule_calls)
                                                        else "needs_review")
            for c in entailed_rules:
                rule_spans += c.result.get("source_span_ids", [])
                r_ids += [r for r in c.result.get("used_rule_ids", []) if r not in r_ids]
                if c.result.get("premise_support") == "agent_asserted":
                    limitations.append(f"rule_premises_agent_asserted:{c.result.get('rule_id') or c.result.get('goal')}")
            for c in rule_calls:
                if c.status == "blocked":
                    limitations.append(f"rule_blocked:{c.result.get('rule_id')}:{c.result.get('reason')}")
                missing += [m for m in c.result.get("missing", []) if m not in missing][:2]
        calc_calls = [c for c in calls if c.tool == "calculate"]
        calc_check = "not_applicable"
        if calc_calls:
            calc_check = "pass" if calc_calls[-1].status == "ok" else ("fail" if calc_calls[-1].status == "error"
                                                                       else "needs_review")
        retrieved = []
        for c in calls:
            if c.tool == "retrieve_source":
                retrieved += [s["span_id"] for s in c.result.get("spans", [])]
        retrieved = list(dict.fromkeys(retrieved))

        verdicts: dict[str, str] = {}
        agent_verdicts: dict[str, str] = {}
        choice_spans: dict[str, list[str]] = {}
        best = None
        if is_choice:
            raw = args.get("choices") if isinstance(args.get("choices"), dict) else {}
            for k in q.choices or {}:
                v = raw.get(k)
                vv = (v.get("verdict") if isinstance(v, dict) else v) or "undetermined"
                vv = str(vv).strip().lower()
                agent_verdicts[k] = vv if vv in ("entailed", "contradicted") else "undetermined"
                if isinstance(v, dict):
                    span_ids += [s for s in keep(_ids(v.get("span_ids")), "span") if s not in span_ids]
            best = extract_letter(args.get("best_choice") or args.get("answer"), q.choices)

        validation = PacketValidation(rule_execution=rule_check, calculation=calc_check)
        verify_call = None
        evidence_pool = span_ids or retrieved[:6]
        if not span_ids and retrieved:
            limitations.append("no_spans_cited_by_agent;verifier_used_retrieved_spans")
        status = "undetermined"
        final_spans: list[str] = []
        if self.verify and "verify_evidence" in self.reg.enabled and (evidence_pool or a_ids or r_ids or rule_spans or dep_spans):
            pool = list(dict.fromkeys(evidence_pool + rule_spans + list(dep_spans)))
            if is_choice:
                verify_call = self._call("verify_evidence", {
                    "claims": [{"id": k, "text": t} for k, t in (q.choices or {}).items()], "question": q.question,
                    "span_ids": pool, "assertion_ids": a_ids, "rule_ids": r_ids}, calls, seen)
            else:
                verify_call = self._call("verify_evidence", {
                    "claim": claim_text, "span_ids": pool, "assertion_ids": a_ids, "rule_ids": r_ids}, calls, seen)
            vr = verify_call.result
            st = vr.get("structural", {}).get("status")
            validation.source_ids_exist = "pass" if st == "pass" else ("fail" if st == "fail" else "needs_review")
            if is_choice:
                for c in vr.get("claims", []):
                    k = c["id"]
                    judge = c.get("status")
                    av = agent_verdicts.get(k, "undetermined")
                    if judge == "supported" and av != "contradicted":
                        verdicts[k] = "entailed"
                        choice_spans[k] = c.get("evidence_span_ids", [])
                    elif judge == "contradicted" and av != "entailed":
                        verdicts[k] = "contradicted"
                        choice_spans[k] = c.get("evidence_span_ids", [])
                    else:
                        verdicts[k] = "undetermined"
                ent = [k for k, v in verdicts.items() if v == "entailed"]
                validation.semantic_support = "pass" if len(ent) == 1 else "needs_review"
                if len(ent) == 1:
                    status = "supported"
                    best_k = ent[0]
                    final_spans = choice_spans.get(best_k, [])
                    claim_text = f"Đáp án {best_k}: {q.choices[best_k]}"
                    if best and best != best_k:
                        limitations.append(f"agent_best_choice_{best}_differs_from_verified_{best_k}")
                else:
                    if len(ent) > 1:
                        limitations.append(f"several_choices_supported:{','.join(ent)}")
                    final_spans = [s for k in (ent or ([best] if best else [])) for s in choice_spans.get(k, [])]
            else:
                sem = vr.get("semantic_support")
                vstat = vr.get("status")
                validation.semantic_support = ("pass" if vstat == "supported" else
                                               "fail" if vstat == "contradicted" else "needs_review")
                rule_ok = bool(entailed_rules) and st == "pass" and not any(
                    c.result.get("premise_support") == "agent_asserted" for c in entailed_rules)
                if proposed == "contradicted":
                    status = "contradicted" if vstat == "contradicted" else "undetermined"
                elif vstat == "supported" or rule_ok:
                    status = "supported" if proposed != "undetermined" or rule_ok else "undetermined"
                    if proposed == "undetermined" and not rule_ok:
                        limitations.append("verifier_supported_but_agent_undetermined")
                elif proposed == "supported":
                    limitations.append(f"agent_claim_not_verified(semantic={sem})")
                final_spans = list(dict.fromkeys(vr.get("evidence_span_ids", []) + (rule_spans if rule_ok else [])))
                if status == "contradicted":
                    final_spans = vr.get("evidence_span_ids", [])
        elif not self.verify:
            # ablation: no source verification -> agent-proposed statuses are taken at face value
            validation.source_ids_exist = "needs_review"
            validation.semantic_support = "needs_review"
            limitations.append("source_verification_disabled")
            if is_choice:
                verdicts = dict(agent_verdicts)
                ent = [k for k, v in verdicts.items() if v == "entailed"]
                if best and (not ent or best in ent):
                    status = "supported" if ent == [best] else "undetermined"
                elif len(ent) == 1:
                    status, best = "supported", ent[0]
                if best:
                    claim_text = f"Đáp án {best}: {q.choices[best]}"
            else:
                status = proposed if proposed in ("supported", "contradicted") else "undetermined"
            final_spans = list(dict.fromkeys(_ids(args.get("span_ids")) + rule_spans))   # unchecked ids
        else:
            limitations.append("no_evidence_retrieved")
            validation.source_ids_exist = "not_applicable"
            if is_choice:
                verdicts = {k: "undetermined" for k in q.choices or {}}

        if status == "supported" and not is_choice:
            missing = []
        # NB: no auto-filled obligations -- only agent/rule-specified gaps drive targeted re-planning
        if entailed_rules and final_spans and any(s not in rule_spans for s in final_spans):
            mode = "mixed"
        elif entailed_rules and status == "supported":
            mode = "formal"
        elif status in ("supported", "contradicted"):
            mode = "source_supported"
        else:
            mode = "none"
        pk = ReasoningPacket(
            packet_id=f"p_{sg.subgoal_id}", subgoal_id=sg.subgoal_id, agent_id=self.agent_id,
            claim=Claim(text=claim_text, canonical_predicate=(str(args["canonical_predicate"]).strip()
                                                              if args.get("canonical_predicate") else None)),
            status=status, proof_mode=mode, used_assertion_ids=a_ids, used_rule_ids=list(dict.fromkeys(r_ids)),
            source_span_ids=list(dict.fromkeys(final_spans)), dependent_packet_ids=[d["packet_id"] for d in deps],
            tool_results=calls, missing_premises=missing,
            counterevidence_ids=(final_spans if status == "contradicted" else []),
            limitations=limitations, validation=validation)
        return SubagentResult(packet=pk, choice_verdicts=verdicts, choice_spans=choice_spans,
                              agent_verdicts=agent_verdicts, best_choice=best, turns=turns, transcript=transcript)


def _clip_assistant(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    return text[:1500]
