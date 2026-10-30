"""`execute_rule`: strict rule application / goal proving with guard semantics (spec 6).

entailed     -- every premise AND guard entailed by explicit literals (or recursively proved);
blocked      -- a premise/guard is contradicted, an exception applies or the rule is applied out of
                its subject scope. NOT evidence for the negated conclusion;
undetermined -- some premise/guard unproven, or the rule is not validated for strict execution;
contradicted -- (goal mode only) a validated rule entails the negation of the goal.
"""
from __future__ import annotations

from typing import Any, Optional

from otter.reasoning.proof_engine import (GroundAtom, KnowledgeBase, ProofEngine, evaluate_rule, parse_literal,
                                          rule_executable)


def id_exists(store: Any, x: str) -> bool:
    try:
        return bool(store.span_exists(x) or store.get_fact(x) or store.get_rule(x) or store.get_concept(x))
    except Exception:
        return False


def _literals(facts: Any, guards: Any) -> list[Any]:
    lits: list[Any] = []
    for src in (facts, guards):
        if not src:
            continue
        if isinstance(src, dict):                 # {"pred(a)": true, "guard_x": false}
            for k, v in src.items():
                lits.append({"predicate": k, "value": v})
        elif isinstance(src, (list, tuple)):
            lits.extend(src)
        else:
            lits.append(src)
    return lits


def execute_rule(ctx: Any, rule_id: Optional[str] = None, goal: Optional[Any] = None, facts: Any = None,
                 premises: Any = None, guards: Any = None, backward: bool = True) -> dict[str, Any]:
    if not rule_id and not goal:
        return {"status": "error", "reason": "need_rule_id_or_goal"}
    kb = KnowledgeBase()
    unsupported_literals, parse_errors = [], []
    for lit in _literals(facts, None) + _literals(premises, None) + _literals(None, guards):
        try:
            atom, val = parse_literal(lit)
        except ValueError as e:
            parse_errors.append(str(e))
            continue
        ev = [e for e in (lit.get("evidence_ids", []) if isinstance(lit, dict) else []) if id_exists(ctx.store, e)]
        if not ev:
            unsupported_literals.append(("" if val else "not ") + str(atom))
        kb.add(atom, val, ev, origin="agent")
    engine = ProofEngine(lambda p: ctx.store.rules_concluding(p, subject=ctx.subject), subject=ctx.subject,
                         max_depth=ctx.max_hops, executable_support=ctx.executable_support)
    base = {"premise_support": "evidenced" if not unsupported_literals else "agent_asserted",
            "agent_asserted_literals": unsupported_literals, "parse_errors": parse_errors,
            "kb_conflicts": kb.conflicts}
    if rule_id:
        rule = ctx.store.get_rule(rule_id)
        if rule is None:
            return {"status": "error", "reason": "unknown_rule_id", "rule_id": rule_id, **base}
        ok, why = rule_executable(rule, ctx.executable_support)
        ev = evaluate_rule(rule, kb, subject=ctx.subject, executable_support=ctx.executable_support)
        sub_proofs = []
        if ev.status == "undetermined" and ok and backward and ev.missing:
            for m in ev.missing:                     # try to close open obligations with other rules
                pr = engine.prove(m, kb)
                sub_proofs.append(pr.to_dict())
                if pr.status in ("entailed", "contradicted"):
                    atom, val = parse_literal(m)
                    kb.add(atom, val if pr.status == "entailed" else not val, pr.source_span_ids, origin="backward")
            ev = evaluate_rule(rule, kb, subject=ctx.subject, executable_support=ctx.executable_support)
        return {"status": ev.status, "rule_id": rule_id, "executable": ok, "executable_reason": why,
                "reason": ev.reason, "premise_status": ev.premise_status, "guard_status": ev.guard_status,
                "exception_status": ev.exception_status, "conclusions": ev.conclusions, "missing": ev.missing,
                "failed": ev.failed, "bindings": ev.bindings, "source_span_ids": ev.source_span_ids,
                "used_rule_ids": [rule_id] + [r for p in sub_proofs for r in p.get("used_rule_ids", [])],
                "sub_proofs": sub_proofs, **base}
    try:
        pr = engine.prove(goal, kb)
    except ValueError as e:
        return {"status": "error", "reason": f"bad_goal: {e}", **base}
    return {"status": pr.status, "goal": pr.goal, "reason": pr.reason, "missing": pr.missing,
            "used_rule_ids": pr.used_rule_ids, "source_span_ids": pr.source_span_ids, "trace": pr.trace, **base}
