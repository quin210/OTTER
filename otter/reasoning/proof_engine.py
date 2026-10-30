"""Scoped forward / backward chaining over validated ontology rules (spec section 6).

Semantics (open world, no closed-world assumption):
  * an atom is ``entailed`` only if it is explicitly known true (or derived by a rule whose
    premises AND guards are all entailed);
  * ``contradicted`` only if its negation is explicitly known / derived;
  * ``undetermined`` otherwise -- absence of evidence is never evidence of falsity;
  * a rule whose premise or guard is contradicted, or whose exception applies, or which is
    applied outside its subject scope, is ``blocked``: it licenses nothing, in particular NOT
    the negation of its conclusion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

from otter.schemas.ontology import Atom, ConditionalRule

DEFAULT_EXECUTABLE_SUPPORT = frozenset({"supported"})


@dataclass(frozen=True)
class GroundAtom:
    predicate: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.predicate}({','.join(self.args)})" if self.args else self.predicate


_ATOM_RE = re.compile(r"^\s*(not\s+|¬|!|~)?\s*([\w\-\.:]+)\s*(?:\((.*)\))?\s*$", re.UNICODE)


def parse_literal(x: Any) -> tuple[GroundAtom, bool]:
    """Parse an atom given as Atom / dict / string into (atom, truth_value).

    Accepted: Atom(..., negated=..), {"predicate", "args", "negated"|"value"}, "pred(a,b)",
    "not pred(a)", "¬pred". Raises ValueError on garbage.
    """
    if isinstance(x, Atom):
        return GroundAtom(x.predicate, tuple(x.args)), not x.negated
    if isinstance(x, dict):
        pred = x.get("predicate") or x.get("atom") or x.get("name")
        if not pred:
            raise ValueError(f"literal without predicate: {x!r}")
        if isinstance(pred, str) and "(" in pred and not x.get("args"):
            atom, val = parse_literal(pred)
        else:
            atom, val = GroundAtom(str(pred), tuple(str(a) for a in (x.get("args") or []))), True
        if x.get("negated"):
            val = not val
        if "value" in x and x["value"] is not None:
            v = x["value"]
            v = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "1", "yes", "entailed", "đúng")
            val = val if v else not val
        return atom, val
    if isinstance(x, str):
        m = _ATOM_RE.match(x)
        if not m:
            raise ValueError(f"cannot parse literal: {x!r}")
        neg, pred, args = m.groups()
        a = tuple(s.strip() for s in args.split(",") if s.strip()) if args else ()
        return GroundAtom(pred, a), not bool(neg)
    raise ValueError(f"cannot parse literal: {x!r}")


class KnowledgeBase:
    """Explicit literal store: atom -> truth value, with the evidence ids that justify it."""

    def __init__(self) -> None:
        self.values: dict[GroundAtom, bool] = {}
        self.evidence: dict[GroundAtom, list[str]] = {}
        self.conflicts: list[dict[str, Any]] = []

    @classmethod
    def from_literals(cls, literals: Iterable[Any], evidence: Optional[dict[str, list[str]]] = None) -> "KnowledgeBase":
        kb = cls()
        for lit in literals or []:
            ev = lit.get("evidence_ids", []) if isinstance(lit, dict) else []
            atom, val = parse_literal(lit)
            kb.add(atom, val, ev, origin="given")
        return kb

    def add(self, atom: GroundAtom, value: bool, evidence_ids: Iterable[str] = (), origin: str = "given") -> bool:
        """Add a literal. Returns False (and records a conflict) if the opposite is already known."""
        if atom in self.values and self.values[atom] != value:
            self.conflicts.append({"atom": str(atom), "existing": self.values[atom], "new": value, "origin": origin})
            return False
        self.values[atom] = value
        self.evidence.setdefault(atom, []).extend(e for e in evidence_ids if e not in self.evidence.get(atom, []))
        return True

    def lookup(self, atom: GroundAtom) -> Optional[GroundAtom]:
        """Exact literal, else a *propositional* literal (same predicate, no args) given by an agent:
        agents rarely reproduce concept-id arguments, so `pred` stands for `pred(<rule's constants>)`.
        Variables are never matched this way."""
        if atom in self.values:
            return atom
        if atom.args and not any(is_var(a) for a in atom.args):
            prop = GroundAtom(atom.predicate, ())
            if prop in self.values:
                return prop
        return None

    def status(self, atom: GroundAtom, positive: bool = True) -> str:
        key = self.lookup(atom)
        if key is None:
            return "undetermined"
        return "entailed" if self.values[key] == positive else "contradicted"


# ---------------------------------------------------------------------------
# rule helpers

def is_var(s: str) -> bool:
    return isinstance(s, str) and s.startswith("?")


def ground(a: Atom, b: dict[str, str]) -> GroundAtom:
    return GroundAtom(a.predicate, tuple(b.get(x, x) for x in a.args))


def unify(pattern: Atom, atom: GroundAtom, b: dict[str, str]) -> Optional[dict[str, str]]:
    if pattern.predicate != atom.predicate or len(pattern.args) != len(atom.args):
        return None
    out = dict(b)
    for p, v in zip(pattern.args, atom.args):
        if is_var(p):
            if p in out and out[p] != v:
                return None
            out[p] = v
        elif p != v:
            return None
    return out


def rule_executable(rule: ConditionalRule, executable_support: Iterable[str] = DEFAULT_EXECUTABLE_SUPPORT) -> tuple[bool, str]:
    """Strict proof execution only uses rules that are schema-valid, linked to existing spans
    and validated as source-supported (spec 3.4: unsupported candidates are excluded).
    Mirrors `otter.storage.writer.is_strict_executable` (WS2 policy)."""
    v = rule.validation
    if not v.schema_valid:
        return False, "rule_schema_invalid"
    if not rule.source_span_ids or not v.source_ids_exist:
        return False, "rule_without_existing_source_span"
    if v.source_support not in set(executable_support):
        return False, f"rule_source_support_{v.source_support}"
    if rule.status != "explicit_rule":
        return False, f"rule_status_{rule.status}"
    if not rule.premises or not rule.conclusions:
        return False, "rule_without_premises_or_conclusions"
    if "possible_missing_guard" in v.notes or "excluded_from_strict_proof" in v.notes:
        return False, "rule_flagged_by_validator"
    return True, "ok"


def _bindings(atoms: list[Atom], kb: KnowledgeBase, b: dict[str, str]) -> Iterator[dict[str, str]]:
    """All maximal variable bindings making each atom match some known literal (or skip it)."""
    if not atoms:
        yield b
        return
    head, rest = atoms[0], atoms[1:]
    matched = False
    if any(is_var(x) and x not in b for x in head.args):
        for known in list(kb.values):
            nb = unify(head, known, b)
            if nb is not None:
                matched = True
                yield from _bindings(rest, kb, nb)
    if not matched:
        yield from _bindings(rest, kb, b)


@dataclass
class RuleEvaluation:
    rule_id: str
    status: str                                      # entailed | blocked | undetermined
    reason: str = ""
    bindings: dict[str, str] = field(default_factory=dict)
    premise_status: dict[str, str] = field(default_factory=dict)
    guard_status: dict[str, str] = field(default_factory=dict)
    exception_status: dict[str, str] = field(default_factory=dict)
    conclusions: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def evaluate_rule(rule: ConditionalRule, kb: KnowledgeBase, *, subject: Optional[str] = None,
                  executable_support: Iterable[str] = DEFAULT_EXECUTABLE_SUPPORT,
                  allow_cross_subject: bool = False) -> RuleEvaluation:
    """Evaluate one rule against explicit literals only (no recursion)."""
    ev = RuleEvaluation(rule_id=rule.rule_id, status="undetermined", source_span_ids=list(rule.source_span_ids))
    ok, why = rule_executable(rule, executable_support)
    if not ok:
        ev.reason = why
        return ev
    if subject and rule.subject != subject and not allow_cross_subject:
        ev.status, ev.reason = "blocked", f"cross_subject_application({rule.subject}!={subject})"
        return ev
    conds = list(rule.premises) + list(rule.guards)
    best: Optional[tuple[int, dict[str, str]]] = None
    for b in _bindings(conds, kb, {}):
        score = sum(kb.status(ground(a, b), not a.negated) == "entailed" for a in conds)
        if best is None or score > best[0]:
            best = (score, b)
    b = best[1] if best else {}
    ev.bindings = b
    for group, target in ((rule.premises, ev.premise_status), (rule.guards, ev.guard_status)):
        for a in group:
            g = ground(a, b)
            st = kb.status(g, not a.negated)
            label = ("not " if a.negated else "") + str(g)
            target[label] = st
            if st == "contradicted":
                ev.failed.append(label)
            elif st == "undetermined":
                ev.missing.append(label)
    for a in rule.exceptions:
        g = ground(a, b)
        st = kb.status(g, not a.negated)
        ev.exception_status[("not " if a.negated else "") + str(g)] = st
    if ev.failed:
        kind = "guard" if any(k in ev.guard_status for k in ev.failed) else "premise"
        ev.status, ev.reason = "blocked", f"{kind}_not_satisfied"
        return ev
    if any(st == "entailed" for st in ev.exception_status.values()):
        ev.status, ev.reason = "blocked", "exception_applies"
        return ev
    if ev.missing:
        ev.status, ev.reason = "undetermined", "unproven_premises_or_guards"
        return ev
    unbound = [x for c in rule.conclusions for x in c.args if is_var(x) and x not in b]
    if unbound:
        ev.status, ev.reason = "undetermined", f"unbound_conclusion_variables:{sorted(set(unbound))}"
        return ev
    ev.status, ev.reason = "entailed", "all_premises_and_guards_entailed"
    ev.conclusions = [{"atom": str(ground(c, b)), "predicate": c.predicate, "args": list(ground(c, b).args),
                       "value": not c.negated, "text": c.text} for c in rule.conclusions]
    return ev


# ---------------------------------------------------------------------------

@dataclass
class ProofResult:
    goal: str
    status: str                                      # entailed | contradicted | undetermined | blocked
    trace: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    used_rule_ids: list[str] = field(default_factory=list)
    source_span_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ProofEngine:
    """Forward and backward chaining with depth / cycle / step limits."""

    def __init__(self, rules_for: Callable[[str], list[ConditionalRule]], *, subject: Optional[str] = None,
                 max_depth: int = 3, max_steps: int = 200,
                 executable_support: Iterable[str] = DEFAULT_EXECUTABLE_SUPPORT) -> None:
        self.rules_for = rules_for                    # predicate -> rules concluding it (either polarity)
        self.subject = subject
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.executable_support = frozenset(executable_support)
        self._steps = 0

    # -- forward -----------------------------------------------------------
    def forward_chain(self, kb: KnowledgeBase, rules: list[ConditionalRule], *, max_rounds: Optional[int] = None) -> dict[str, Any]:
        rounds = max_rounds or self.max_depth
        derived: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        for r in range(rounds):
            new = False
            for rule in rules:
                ev = evaluate_rule(rule, kb, subject=self.subject, executable_support=self.executable_support)
                trace.append({"round": r, **ev.to_dict()})
                if ev.status != "entailed":
                    continue
                for c in ev.conclusions:
                    atom = GroundAtom(c["predicate"], tuple(c["args"]))
                    if atom in kb.values:
                        if kb.values[atom] != c["value"]:
                            kb.add(atom, c["value"], [rule.rule_id], origin=rule.rule_id)  # records conflict
                        continue
                    kb.add(atom, c["value"], [rule.rule_id, *rule.source_span_ids], origin=rule.rule_id)
                    derived.append({"atom": c["atom"], "value": c["value"], "rule_id": rule.rule_id, "round": r,
                                    "source_span_ids": list(rule.source_span_ids)})
                    new = True
            if not new:
                break
        return {"derived": derived, "trace": trace, "conflicts": list(kb.conflicts)}

    # -- backward ----------------------------------------------------------
    def prove(self, goal: Any, kb: KnowledgeBase) -> ProofResult:
        atom, val = parse_literal(goal)
        self._steps = 0
        return self._prove(atom, val, kb, 0, frozenset())

    def _prove(self, atom: GroundAtom, positive: bool, kb: KnowledgeBase, depth: int, visited: frozenset) -> ProofResult:
        label = ("not " if not positive else "") + str(atom)
        self._steps += 1
        known = kb.status(atom, positive)
        if known != "undetermined":
            return ProofResult(goal=label, status=known, evidence_ids=list(kb.evidence.get(kb.lookup(atom) or atom, [])),
                               trace=[{"step": "known_literal", "atom": label, "status": known}])
        if depth >= self.max_depth:
            return ProofResult(goal=label, status="undetermined", missing=[label], reason="depth_limit")
        if atom in visited:
            return ProofResult(goal=label, status="undetermined", missing=[label], reason="cycle")
        if self._steps > self.max_steps:
            return ProofResult(goal=label, status="undetermined", missing=[label], reason="step_limit")
        visited = visited | {atom}
        trace: list[dict[str, Any]] = []
        missing: list[str] = []
        any_blocked = False
        for rule in self.rules_for(atom.predicate):
            for concl in rule.conclusions:
                b = unify(concl, atom, {})
                if b is None:
                    continue
                supports = (not concl.negated) == positive    # rule concludes the goal (else its negation)
                ok, why = rule_executable(rule, self.executable_support)
                if not ok:
                    trace.append({"step": "rule_skipped", "rule_id": rule.rule_id, "reason": why})
                    continue
                if self.subject and rule.subject != self.subject:
                    trace.append({"step": "rule_blocked", "rule_id": rule.rule_id, "reason": "cross_subject_application"})
                    any_blocked = True
                    continue
                sub_results: list[ProofResult] = []
                status = "entailed"
                for kind, group in (("premise", rule.premises), ("guard", rule.guards)):
                    for a in group:
                        g = ground(a, b)
                        if any(is_var(x) for x in g.args):
                            sr = ProofResult(goal=str(g), status="undetermined", missing=[str(g)], reason="unbound_variable")
                        else:
                            sr = self._prove(g, not a.negated, kb, depth + 1, visited)
                        sr.trace.insert(0, {"step": f"{kind}_of", "rule_id": rule.rule_id})
                        sub_results.append(sr)
                        if sr.status in ("contradicted", "blocked") and status != "blocked":
                            status = "blocked"
                        elif sr.status == "undetermined" and status == "entailed":
                            status = "undetermined"
                exc_applies = False
                for a in rule.exceptions:
                    if kb.status(ground(a, b), not a.negated) == "entailed":
                        exc_applies = True
                if exc_applies:
                    status = "blocked"
                step = {"step": "apply_rule", "rule_id": rule.rule_id, "bindings": b, "status": status,
                        "concludes": "goal" if supports else "negation_of_goal",
                        "sub": [s.to_dict() for s in sub_results], "source_span_ids": list(rule.source_span_ids)}
                trace.append(step)
                if status == "entailed":
                    spans = list(rule.source_span_ids) + [x for s in sub_results for x in s.source_span_ids]
                    used = [rule.rule_id] + [x for s in sub_results for x in s.used_rule_ids]
                    ev = [x for s in sub_results for x in s.evidence_ids]
                    return ProofResult(goal=label, status="entailed" if supports else "contradicted", trace=trace,
                                       used_rule_ids=_uniq(used), source_span_ids=_uniq(spans), evidence_ids=_uniq(ev),
                                       reason="rule_entailed" if supports else "rule_entails_negation")
                if status == "blocked":
                    any_blocked = True
                elif supports:
                    missing.extend(m for s in sub_results for m in (s.missing if s.status == "undetermined" else []))
        if not trace:
            return ProofResult(goal=label, status="undetermined", missing=[label], reason="no_rule_concludes_goal")
        # A blocked rule does NOT prove the negation: overall the goal stays undetermined.
        return ProofResult(goal=label, status="undetermined", trace=trace, missing=_uniq(missing) or [label],
                           reason="all_candidate_rules_blocked" if any_blocked and not missing else "unproven_obligations")


def _uniq(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out
