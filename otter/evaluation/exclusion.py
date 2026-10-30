"""Run-time hook for OTTER-Abstain `excluded_span_ids` (evaluation-runner side; no system code is modified).

Semantics: for an item with a non-empty `excluded_span_ids`, those spans must be invisible to the system's retrieval for that
question only (the corpus, index and ontology stay shared). The runner sets the exclusion in a context variable around
`system.answer(item)` (`excluded_spans(...)`) and, once per run, wraps the system's span store with `ExcludingStore`
(`install_exclusion(system)`), which

* drops excluded spans from `search_spans` results (over-fetching by len(excluded) so top-k stays full),
* drops excluded spans from `get_spans`,
* drops ontology hits (`search_knowledge`, ...) whose only evidence spans are excluded,
* passes every other call through unchanged. With no exclusion active it is a transparent pass-through, so
  benchmarks without `excluded_span_ids` produce exactly the same outputs as before.

Supported systems: `text_rag` / `rag_*` (otter.evaluation.rag_variants reads `current_exclusion()` natively; the older WS3 `TextRAGSystem` via system.store) and Master-based systems (`system.master.store`: single_agent_tools,
multi_agent_no_workspace, otter_full). Known limit: ontology assertions *derived* from an excluded span (concepts/facts built
over several spans) may still be reachable through graph neighbours; `search_knowledge` hits are filtered by their
`source_span_ids` only. `direct` has no retrieval. External/baseline adapters (WS4) must filter their own retrieved spans
with `current_exclusion()` or the `excluded_span_ids` key in the item they receive (see benchmarks/abstain/README.md).
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Iterable

_EXCL: contextvars.ContextVar[frozenset] = contextvars.ContextVar("otter_excluded_spans", default=frozenset())


def current_exclusion() -> frozenset:
    return _EXCL.get()


@contextlib.contextmanager
def excluded_spans(ids: Iterable[str] | None):
    tok = _EXCL.set(frozenset(ids or ()))
    try:
        yield
    finally:
        _EXCL.reset(tok)


def _sid(x: Any) -> str | None:
    if isinstance(x, (tuple, list)) and x:
        x = x[0]
    if isinstance(x, dict):
        return x.get("span_id")
    return getattr(x, "span_id", None)


class ExcludingStore:
    def __init__(self, store: Any):
        self._inner = store

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr) or name not in ("search_spans", "get_spans", "search_knowledge"):
            return attr

        def call(*a, **kw):
            ex = _EXCL.get()
            if not ex:
                return attr(*a, **kw)
            if name == "search_spans":
                kw = dict(kw)
                if "top_k" in kw:
                    kw["top_k"] = int(kw["top_k"]) + len(ex)
                res = attr(*a, **kw)
                out = [h for h in res if _sid(h) not in ex]
                return out[: int(kw["top_k"]) - len(ex)] if "top_k" in kw else out
            if name == "get_spans":
                return [s for s in attr(*a, **kw) if _sid(s) not in ex]
            res = attr(*a, **kw)                                   # search_knowledge
            keep = []
            for h in res:
                srcs = (h.get("source_span_ids") if isinstance(h, dict) else None) or []
                if srcs and all(s in ex for s in srcs):
                    continue
                keep.append(h)
            return keep
        return call


# modules whose systems read `current_exclusion()` themselves (no store wrapping needed)
NATIVE_MODULES = ("otter.evaluation.rag_variants",)


def install_exclusion(system: Any) -> bool:
    """Make `system` honour the per-item exclusion; returns True if it does (native or via a wrapped store)."""
    if type(system).__module__ in NATIVE_MODULES:
        return True
    done = False
    calc = getattr(system, "calc_system", None)           # otter_v2 delegates calculation items to an otter_full instance
    for holder in (system, getattr(system, "master", None), calc, getattr(calc, "master", None)):
        st = getattr(holder, "store", None) if holder is not None else None
        if st is not None and not isinstance(st, ExcludingStore):
            holder.store = ExcludingStore(st)
            done = True
        elif isinstance(st, ExcludingStore):
            done = True
    return done
