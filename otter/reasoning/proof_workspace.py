"""Shared Proof Workspace (spec 5.4): append-only, versioned reasoning packets + typed edges.

* `publish` never overwrites: a new packet for the same subgoal gets version k+1;
* edges: supports | contradicts | depends_on | uses_rule | cites_span;
* `issues()` reports contradictions (opposite verified statuses of the same literal, counter-
  evidence), scope mismatches (cross-subject ids), unit mismatches (same variable, different
  dimensions), dependency cycles and duplicate evidence. Nothing is resolved by majority vote.
* `missing_obligations()` lists specific unproven premises for targeted re-planning.
"""
from __future__ import annotations

import datetime as _dt
import re
import threading
from collections import defaultdict
from typing import Any, Iterable, Optional

from otter.reasoning.proof_engine import parse_literal
from otter.schemas.reasoning import ReasoningPacket

EDGE_TYPES = ("supports", "contradicts", "depends_on", "uses_rule", "cites_span")
SUBJECT_PREFIXES = ("physics", "chemistry", "biology", "history", "geography", "literature", "other")


def _safe(s: str) -> str:
    return re.sub(r"[^\w#\-]+", "_", s)


class ProofWorkspace:
    def __init__(self, question_id: str = "", subject: Optional[str] = None):
        self.question_id = question_id
        self.subject = subject
        self._lock = threading.RLock()
        self._packets: list[ReasoningPacket] = []
        self._by_id: dict[str, ReasoningPacket] = {}
        self._by_subgoal: dict[str, list[str]] = defaultdict(list)
        self.edges: list[dict[str, str]] = []
        self.log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ writes
    def publish(self, packet: ReasoningPacket) -> ReasoningPacket:
        with self._lock:
            p = packet.model_copy(deep=True)
            k = len(self._by_subgoal[p.subgoal_id]) + 1
            p.version = k
            p.packet_id = f"p_{_safe(p.subgoal_id)}_v{k}"
            p.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")
            self._packets.append(p)
            self._by_id[p.packet_id] = p
            self._by_subgoal[p.subgoal_id].append(p.packet_id)
            for d in p.dependent_packet_ids:
                self._edge(p.packet_id, "depends_on", d)
            for r in p.used_rule_ids:
                self._edge(p.packet_id, "uses_rule", r)
            for s in p.source_span_ids:
                self._edge(p.packet_id, "cites_span", s)
            self.log.append({"event": "publish", "packet_id": p.packet_id, "agent": p.agent_id, "status": p.status,
                             "at": p.created_at})
            return p.model_copy(deep=True)

    def link(self, src: str, edge_type: str, dst: str) -> None:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"invalid edge type {edge_type}")
        with self._lock:
            self._edge(src, edge_type, dst)

    def _edge(self, src: str, t: str, dst: str) -> None:
        e = {"src": src, "type": t, "dst": dst}
        if e not in self.edges:
            self.edges.append(e)

    # ------------------------------------------------------------------ reads
    def get(self, packet_id: str) -> Optional[ReasoningPacket]:
        with self._lock:
            p = self._by_id.get(packet_id)
            return p.model_copy(deep=True) if p else None

    def latest(self, subgoal_id: str) -> Optional[ReasoningPacket]:
        with self._lock:
            ids = self._by_subgoal.get(subgoal_id)
            return self._by_id[ids[-1]].model_copy(deep=True) if ids else None

    def history(self, subgoal_id: str) -> list[ReasoningPacket]:
        with self._lock:
            return [self._by_id[i].model_copy(deep=True) for i in self._by_subgoal.get(subgoal_id, [])]

    def packets(self) -> list[ReasoningPacket]:
        with self._lock:
            return [p.model_copy(deep=True) for p in self._packets]

    def latest_all(self) -> list[ReasoningPacket]:
        with self._lock:
            return [self._by_id[ids[-1]].model_copy(deep=True) for ids in self._by_subgoal.values() if ids]

    def subgoal_ids(self) -> list[str]:
        with self._lock:
            return list(self._by_subgoal)

    # ------------------------------------------------------------------ analysis
    def issues(self) -> dict[str, list[dict[str, Any]]]:
        latest = self.latest_all()
        out: dict[str, list[dict[str, Any]]] = {"contradictions": [], "scope_mismatches": [], "unit_mismatches": [],
                                                "dependency_cycles": [], "duplicate_evidence": []}
        # 1) same literal, opposite verified statuses (or explicitly negated predicate)
        lits: dict[Any, list[tuple[str, bool]]] = defaultdict(list)
        for p in latest:
            if p.status not in ("supported", "contradicted") or not p.claim.canonical_predicate:
                continue
            try:
                atom, val = parse_literal(p.claim.canonical_predicate if not p.claim.arguments else
                                          {"predicate": p.claim.canonical_predicate, "args": p.claim.arguments})
            except ValueError:
                continue
            truth = val if p.status == "supported" else (not val)
            lits[atom].append((p.packet_id, truth))
        for atom, xs in lits.items():
            if len({t for _, t in xs}) > 1:
                pos = [i for i, t in xs if t]
                neg = [i for i, t in xs if not t]
                out["contradictions"].append({"kind": "opposite_literal", "literal": str(atom), "true_in": pos,
                                              "false_in": neg})
                for a in pos:
                    for b in neg:
                        self.link(a, "contradicts", b)
        # 2) explicit counter-evidence against what another packet used
        for p in latest:
            if not p.counterevidence_ids:
                continue
            for q in latest:
                if q.packet_id == p.packet_id or q.status != "supported":
                    continue
                used = set(q.used_assertion_ids) | set(q.source_span_ids) | set(q.used_rule_ids)
                hit = sorted(used & set(p.counterevidence_ids))
                if hit:
                    out["contradictions"].append({"kind": "counterevidence", "packet": q.packet_id,
                                                  "counter_packet": p.packet_id, "ids": hit})
                    self.link(p.packet_id, "contradicts", q.packet_id)
        # 3) scope: ids from another subject than the question scope
        if self.subject:
            for p in latest:
                foreign = sorted({i for i in (p.used_rule_ids + p.used_assertion_ids)
                                  if i.split(":", 1)[0] in SUBJECT_PREFIXES and i.split(":", 1)[0] != self.subject})
                if foreign:
                    out["scope_mismatches"].append({"packet": p.packet_id, "foreign_ids": foreign,
                                                    "expected_subject": self.subject})
        # 4) units: same variable with different dimensions across calculations
        dims: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for p in latest:
            for tc in p.tool_results:
                if tc.tool != "calculate" or tc.status != "ok":
                    continue
                for line in tc.result.get("trace", []):
                    m = re.match(r"^(\w+) = .*\(SI: \S+ (\S+)\)$", line)
                    if m:
                        dims[m.group(1)].add((m.group(2), p.packet_id))
                tgt = tc.arguments.get("target")
                if tgt and tc.result.get("dims"):
                    dims[str(tgt)].add((tc.result["dims"], p.packet_id))
        for var, xs in dims.items():
            if len({d for d, _ in xs}) > 1:
                out["unit_mismatches"].append({"variable": var, "dims": sorted({d for d, _ in xs}),
                                               "packets": sorted({i for _, i in xs})})
        # 5) dependency cycles
        adj = defaultdict(list)
        for e in self.edges:
            if e["type"] == "depends_on":
                adj[e["src"]].append(e["dst"])
        out["dependency_cycles"] = _cycles(adj)
        # 6) the same span cited by several packets for different claims (informational)
        cites = defaultdict(set)
        for p in latest:
            for s in p.source_span_ids:
                cites[s].add(p.packet_id)
        out["duplicate_evidence"] = [{"span_id": s, "packets": sorted(ps)} for s, ps in cites.items() if len(ps) > 2]
        return out

    def missing_obligations(self, subgoal_ids: Optional[Iterable[str]] = None) -> list[dict[str, str]]:
        wanted = set(subgoal_ids) if subgoal_ids is not None else None
        out = []
        for p in self.latest_all():
            if wanted is not None and p.subgoal_id not in wanted:
                continue
            if p.status in ("undetermined", "failed"):
                obligations = p.missing_premises or [p.claim.text]
                for o in obligations:
                    if o:
                        out.append({"subgoal_id": p.subgoal_id, "packet_id": p.packet_id, "obligation": o})
        return out

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"packets": [{"packet_id": p.packet_id, "subgoal_id": p.subgoal_id, "agent_id": p.agent_id,
                                 "status": p.status, "claim": p.claim.text[:300], "proof_mode": p.proof_mode,
                                 "source_span_ids": p.source_span_ids, "used_rule_ids": p.used_rule_ids,
                                 "used_assertion_ids": p.used_assertion_ids,
                                 "dependent_packet_ids": p.dependent_packet_ids, "missing_premises": p.missing_premises,
                                 "validation": p.validation.model_dump(by_alias=True),
                                 "tools": [f"{t.tool}:{t.status}" for t in p.tool_results],
                                 "limitations": p.limitations[:5]} for p in self._packets],
                    "edges": list(self.edges)}


def _cycles(adj: dict[str, list[str]]) -> list[list[str]]:
    seen, stack, found = set(), [], []
    on = set()

    def dfs(u: str) -> None:
        seen.add(u)
        stack.append(u)
        on.add(u)
        for v in adj.get(u, []):
            if v in on:
                found.append(stack[stack.index(v):] + [v])
            elif v not in seen:
                dfs(v)
        stack.pop()
        on.discard(u)

    for n in list(adj):
        if n not in seen:
            dfs(n)
    return found
