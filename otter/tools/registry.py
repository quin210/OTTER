"""ToolRegistry: the single, shared tool API used by every (homogeneous) Subagent.

* every call returns a `ToolCall` record (id, arguments, status, structured result, latency);
* tools never raise: exceptions become status="error";
* disabled tools (ablations) return status="disabled";
* a global per-question call budget guarantees termination (thread-safe counter).
"""
from __future__ import annotations

import inspect
import itertools
import threading
import time
from collections import Counter
from typing import Any, Callable, Optional

from otter.schemas.reasoning import ToolCall
from otter.tools.base import ToolContext
from otter.tools.calculate import calculate
from otter.tools.execute_rule import execute_rule
from otter.tools.expand_graph import expand_graph
from otter.tools.retrieve_source import retrieve_source
from otter.tools.search_knowledge import search_knowledge
from otter.tools.verify_evidence import verify_evidence

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_knowledge": search_knowledge,
    "expand_graph": expand_graph,
    "retrieve_source": retrieve_source,
    "execute_rule": execute_rule,
    "calculate": calculate,
    "verify_evidence": verify_evidence,
}
TOOL_NAMES = tuple(TOOLS)


class ToolRegistry:
    def __init__(self, ctx: ToolContext, *, max_total_calls: Optional[int] = None):
        self.ctx = ctx
        self.max_total_calls = max_total_calls
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.calls: list[ToolCall] = []
        self.counts: Counter = Counter()
        self._params = {n: set(inspect.signature(f).parameters) - {"ctx"} for n, f in TOOLS.items()}

    @property
    def enabled(self) -> list[str]:
        return [t for t in TOOL_NAMES if t not in self.ctx.disabled]

    def budget_left(self) -> Optional[int]:
        if self.max_total_calls is None:
            return None
        with self._lock:
            return max(0, self.max_total_calls - len(self.calls))

    def call(self, name: str, arguments: Optional[dict[str, Any]] = None, *, agent_id: str = "") -> ToolCall:
        args = dict(arguments or {})
        with self._lock:
            cid = f"t{next(self._ids):03d}"
            over = self.max_total_calls is not None and len(self.calls) >= self.max_total_calls
        t0 = time.perf_counter()
        if name not in TOOLS:
            status, result = "error", {"status": "error", "reason": f"unknown_tool:{name}", "valid_tools": self.enabled}
        elif name in self.ctx.disabled:
            status, result = "disabled", {"status": "disabled", "reason": "tool disabled in this configuration"}
        elif over:
            status, result = "budget_exceeded", {"status": "budget_exceeded", "reason": "per-question tool budget exhausted"}
        else:
            dropped = [k for k in args if k not in self._params[name]]
            clean = {k: v for k, v in args.items() if k in self._params[name]}
            try:
                result = TOOLS[name](self.ctx, **clean)
            except TypeError as e:
                result = {"status": "error", "reason": f"bad_arguments: {e}"}
            except Exception as e:                       # never propagate tool failures
                result = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
            if dropped:
                result["ignored_arguments"] = dropped
            status = str(result.get("status", "ok"))
        lat = round((time.perf_counter() - t0) * 1000, 2)
        if name in TOOLS:
            tc = ToolCall(call_id=cid, tool=name, arguments=args, status=status, result=result, latency_ms=lat)
        else:  # keep the record faithful (unknown tool name) without failing schema validation
            tc = ToolCall.model_construct(call_id=cid, tool=name, arguments=args, status=status, result=result,
                                          latency_ms=lat)
        with self._lock:
            if status not in ("disabled", "budget_exceeded") and name in TOOLS:
                self.calls.append(tc)
                self.counts[name] += 1
        return tc

    def usage(self) -> dict[str, Any]:
        with self._lock:
            return {"calls": len(self.calls), "by_tool": dict(self.counts)}
