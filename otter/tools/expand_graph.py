"""`expand_graph`: typed neighbourhood (concepts, relations, related rules incl. guards)."""
from __future__ import annotations

from typing import Any, Optional, get_args

from otter.schemas.ontology import RelationType
from otter.tools.base import concept_view, relation_view, rule_view

VALID_RELATIONS = frozenset(get_args(RelationType))


def expand_graph(ctx: Any, node_ids: list[str], relation_types: Optional[list[str]] = None,
                 max_hops: int = 1) -> dict[str, Any]:
    if isinstance(node_ids, str):
        node_ids = [node_ids]
    node_ids = [str(x) for x in (node_ids or []) if x]
    if not node_ids:
        return {"status": "error", "reason": "no_node_ids"}
    bad = [r for r in (relation_types or []) if r not in VALID_RELATIONS]
    if bad:
        return {"status": "error", "reason": "invalid_relation_types", "invalid": bad,
                "valid_relation_types": sorted(VALID_RELATIONS)}
    hops = max(1, min(int(max_hops or 1), ctx.max_hops))
    known = [n for n in node_ids if ctx.store.get_concept(n) or ctx.store.get_rule(n) or ctx.store.get_fact(n)]
    unknown = [n for n in node_ids if n not in known]
    if not known:
        return {"status": "error", "reason": "unknown_node_ids", "invalid_ids": unknown}
    nb = ctx.store.neighbors(known, relation_types=relation_types or None, max_hops=hops, subject=ctx.subject)
    nodes = [concept_view(c) for c in nb.get("nodes", [])][:30]
    rels = [relation_view(r) for r in nb.get("relations", []) if r.relation_type in VALID_RELATIONS][:40]
    rules = [rule_view(r) for r in nb.get("rules", [])][:15]
    status = "ok" if (nodes or rels or rules) else "no_results"
    return {"status": status, "seed_ids": known, "invalid_ids": unknown, "max_hops": hops,
            "nodes": nodes, "relations": rels, "rules": rules}
