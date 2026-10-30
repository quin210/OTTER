"""WS3 QA systems behind one factory: `get_system(name, backbone="main", **flags) -> QASystem`.

  direct                    closed-book single LLM call (no retrieval, no citations)
  text_rag                  hybrid BM25 + bge-m3 span retrieval, one LLM call, cites span ids
  single_agent_tools        one Subagent with the same six tools, equalised tool budget, no DAG/workspace
  multi_agent_no_workspace  planner decomposition + independent packets + flat LLM aggregation
  otter_full                Master + homogeneous Subagents + Shared Proof Workspace + targeted re-planning

Ablation flags (any system using the Master): no_decomposition, no_graph_retrieval,
no_source_verification, no_proof_completion, no_rule_execution (+ no_workspace: otter_full with independent
packets and flat aggregation, i.e. the multi_agent_no_workspace behaviour).
Other keyword overrides: store=, llm=, db_path=, provenance_path=, max_subagents=, max_tool_calls=,
max_replan_rounds=, max_subgoals=, tool_budget= (single agent), keep_transcripts=, llm_cache=.

Every `answer()` is thread-safe and never raises (status="failed" + error instead). tool_usage
always carries llm_calls / prompt_tokens / completion_tokens / tokens / calls / subagents / latency_ms.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import Any, Optional

import numpy as np

from otter.agents import prompts as P
from otter.agents.master import MasterAgent, OtterConfig
from otter.agents.runtime import LRU, CachedStore, Meter, MeteredLLM, QuestionInput, extract_letter, safe_json
from otter.llm.interface import LLM, get_llm, load_config
from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.text import truncate
from otter.schemas.reasoning import AnswerClaim, FinalAnswer

log = logging.getLogger("otter.systems")

SYSTEM_NAMES = ("direct", "text_rag", "single_agent_tools", "multi_agent_no_workspace", "otter_full", "otter_v2")
ALIASES = {"single_ontology": "single_agent_tools", "multi_flat": "multi_agent_no_workspace", "otter": "otter_full",
           "rag": "text_rag", "closed_book": "direct"}
ABLATIONS = ("no_decomposition", "no_graph_retrieval", "no_source_verification", "no_proof_completion",
             "no_rule_execution", "no_workspace")

# Evaluation corpus "v1-core": the 18 core textbooks (book_id prefix nxbgd_, 16,716 spans). Override with
# get_system(..., book_prefix=None|"<prefix>") or env OTTER_BOOK_PREFIX ("" = all books).
DEFAULT_BOOK_PREFIX = os.environ.get("OTTER_BOOK_PREFIX", "nxbgd_")
CORPUS_NAME = os.environ.get("OTTER_CORPUS_NAME", "v1-core")

_STORES: dict[tuple, Any] = {}
_STORES_LOCK = threading.Lock()


# --------------------------------------------------------------------------- store with hybrid span search
class HybridStore(CachedStore):
    """Cached store whose `search_spans` is guaranteed hybrid: if the DB holds bge-m3 span vectors the
    store's own BM25+dense RRF is used; otherwise BM25 candidates are fused (RRF) with an on-the-fly
    bge-m3 ranking of those candidates (span vectors cached in memory). Offline -> BM25 only."""

    def __init__(self, store: Any, *, rerank_pool: int = 40, book_prefix: Optional[str] = DEFAULT_BOOK_PREFIX, **kw):
        super().__init__(store, **kw)
        self._pool = rerank_pool
        self._vecs = LRU(200000)
        self._native: Optional[bool] = None
        self._emb = None
        self._emb_lock = threading.Lock()
        self.book_prefix = book_prefix or None
        self._allowed: Optional[list[str]] = None

    # ---- evaluation corpus restriction (book_id prefix, e.g. the 18 core "nxbgd_" textbooks) ----
    def allowed_books(self) -> Optional[list[str]]:
        if not self.book_prefix:
            return None
        if self._allowed is None:
            try:
                rows = self._store.db.execute("SELECT book_id FROM books").fetchall()
                ids = [r[0] for r in rows]
            except Exception:
                ids = []
            self._allowed = sorted(b for b in ids if b.startswith(self.book_prefix))
        return self._allowed

    def _in_corpus(self, span_id: str) -> bool:
        return not self.book_prefix or str(span_id).startswith(self.book_prefix)

    def search_knowledge(self, query: str, *, subject=None, grade=None, kinds=("concept", "fact", "rule"), top_k=10):
        base = super().__getattr__("search_knowledge")
        hits = base(query, subject=subject, grade=grade, kinds=kinds, top_k=top_k * 3)
        if self.book_prefix:
            hits = [h for h in hits if any(self._in_corpus(s) for s in (h.get("source_span_ids") or []))]
        # Ontology precision (WS2 spot-check): rules ~73%, facts ~80%, relations ~50%, concepts ~55%.
        # Prefer supported facts/rules; down-weight unsupported items and bare concepts.
        kind_w = {"rule": 1.15, "fact": 1.1, "relation": 0.9, "concept": 0.75}

        def adj(h):
            w = kind_w.get(h.get("kind"), 1.0)
            sup = h.get("source_support")
            if sup in ("unsupported", "contradicted"):
                w *= 0.4
            elif sup not in ("supported", None):
                w *= 0.8
            return float(h.get("score") or 0.0) * w
        hits = sorted(hits, key=adj, reverse=True)
        return hits[:top_k]

    def _native_dense(self) -> bool:
        if self._native is None:
            try:
                emb = self._embedder()
                model = getattr(emb, "model", None)
                row = self._store.db.execute("SELECT COUNT(*) FROM embeddings WHERE kind='span' AND model=?",
                                             (model,)).fetchone()
                self._native = bool(row and row[0] > 0)
            except Exception:
                self._native = False
        return self._native

    def _embedder(self):
        with self._emb_lock:
            if self._emb is None:
                try:
                    from otter.storage.embeddings import Bgem3Embedder
                    self._emb = Bgem3Embedder.from_config()
                except Exception:
                    self._emb = False
            return self._emb or None

    def search_spans(self, query: str, *, subject=None, grade=None, book_ids=None, top_k: int = 10):
        base = super().__getattr__("search_spans")
        allowed = self.allowed_books()
        if allowed is not None:
            book_ids = [b for b in (book_ids or allowed) if b in set(allowed)]
            if not book_ids:
                return []
        if self._native_dense():
            return base(query, subject=subject, grade=grade, book_ids=book_ids, top_k=top_k)
        cands = base(query, subject=subject, grade=grade, book_ids=book_ids, top_k=max(top_k, self._pool))
        emb = self._embedder()
        if not cands or emb is None or not emb.available():
            return cands[:top_k]
        try:
            todo = [s for s, _ in cands if not self._vecs.get(s.span_id)[0]]
            if todo:
                m = emb.embed([s.text[:2000] for s in todo])
                for s, v in zip(todo, m):
                    self._vecs.put(s.span_id, np.asarray(v, dtype=np.float32))
            qv = np.asarray(emb.embed([query])[0], dtype=np.float32)
            sims = []
            for s, _ in cands:
                v = self._vecs.get(s.span_id)[1]
                sims.append(float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9)))
            dense_rank = {cands[i][0].span_id: r for r, i in enumerate(np.argsort(-np.array(sims)))}
            fused = [(s, 1.0 / (60 + i + 1) + 1.0 / (60 + dense_rank[s.span_id] + 1)) for i, (s, _) in enumerate(cands)]
            fused.sort(key=lambda x: -x[1])
            return fused[:top_k]
        except Exception as e:
            log.debug("dense rerank failed: %s", e)
            return cands[:top_k]


def default_store(db_path: Optional[str] = None, provenance_path: Optional[str] = None,
                  book_prefix: Optional[str] = DEFAULT_BOOK_PREFIX) -> Any:
    key = (db_path, provenance_path, book_prefix)
    with _STORES_LOCK:
        if key not in _STORES:
            from otter.storage import SqliteOntologyStore
            if db_path:
                raw = SqliteOntologyStore(db_path, provenance_path=provenance_path)
            else:
                raw = SqliteOntologyStore.from_config()
            _STORES[key] = HybridStore(raw, book_prefix=book_prefix)
        return _STORES[key]


def _wrap_store(store: Any, book_prefix: Optional[str] = None) -> Any:
    return store if isinstance(store, CachedStore) else HybridStore(store, book_prefix=book_prefix)


# --------------------------------------------------------------------------- base
class _Base:
    name = "base"

    def __init__(self, llm: LLM, *, name: str, llm_cache: Optional[LRU] = None):
        self.llm = llm
        self.name = name
        self.llm_cache = llm_cache

    def answer(self, item: dict[str, Any]) -> FinalAnswer:
        t0 = time.perf_counter()
        meter = Meter()
        qid = str(item.get("id") or "")
        try:
            fa = self._answer(item, MeteredLLM(self.llm, meter, cache=self.llm_cache))
        except Exception as e:
            fa = FinalAnswer(question_id=qid, system=self.name, answer="", status="failed",
                             error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}")
        fa.tool_usage.update(meter.as_dict())
        fa.tool_usage.setdefault("subagents", 0)
        fa.tool_usage["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        fa.tool_usage["corpus"] = getattr(self, "corpus", None)
        return fa

    def _answer(self, item, llm) -> FinalAnswer:  # pragma: no cover
        raise NotImplementedError


class DirectSystem(_Base):
    def _answer(self, item, llm):
        q = QuestionInput.from_item(item)
        obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                   {"role": "user", "content": P.direct_user(q.question, q.choices, q.subject, q.grade)}],
                             max_tokens=400, retry_msg=P.JSON_RETRY)
        ans = obj.get("answer") if isinstance(obj, dict) else None
        letter = extract_letter(ans if ans is not None else raw, q.choices)
        if q.choices:
            text = f"Đáp án {letter}. {q.choices[letter]}" if letter else "Không xác định được đáp án."
            expl = obj.get("explanation") if isinstance(obj, dict) else None
            if letter and expl:
                text += f" {truncate(str(expl), 400)}"
        else:
            text = str(ans or raw).strip()
        fa = FinalAnswer(question_id=q.qid, system=self.name, answer=text, answer_choice=letter,
                         status="qualified" if (letter or not q.choices) else "abstain")
        fa.tool_usage["calls"] = 0
        return fa


class TextRAGSystem(_Base):
    def __init__(self, llm, store, *, top_k: int = 6, **kw):
        super().__init__(llm, **kw)
        self.store = store
        self.top_k = top_k
        self.resolver = CitationResolver(store)

    def _answer(self, item, llm):
        q = QuestionInput.from_item(item)
        query = q.question + " " + " ".join((q.choices or {}).values())
        hits = self.store.search_spans(query, subject=q.subject, grade=q.grade, top_k=self.top_k)
        spans = [{"span_id": s.span_id, "text": truncate(s.text, 700)} for s, _ in hits]
        obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                   {"role": "user", "content": P.rag_user(q.question, q.choices, q.subject, q.grade,
                                                                          spans)}], max_tokens=500, retry_msg=P.JSON_RETRY)
        ans = obj.get("answer") if isinstance(obj, dict) else None
        retrieved = [s["span_id"] for s in spans]
        cited = [str(s) for s in (obj.get("cited_span_ids") or [])] if isinstance(obj, dict) else []
        valid = [s for s in cited if s in retrieved]       # only ids that were actually shown
        letter = extract_letter(ans if ans is not None else raw, q.choices)
        claims = []
        if valid and (letter or not q.choices):
            cites, _ = self.resolver.resolve_many(valid, with_quote=True, quote_chars=240)
            if cites:
                text = f"Đáp án {letter}: {q.choices[letter]}" if q.choices else str(ans)
                claims.append(AnswerClaim(claim_id="c1", text=text, citations=cites))
        if q.choices:
            text = f"Đáp án {letter}. {q.choices[letter]}" if letter else "Không xác định được đáp án."
            if letter and isinstance(obj, dict) and obj.get("explanation"):
                text += f" {truncate(str(obj['explanation']), 400)}"
        else:
            text = str(ans or raw).strip()
        fa = FinalAnswer(question_id=q.qid, system=self.name, answer=text, answer_choice=letter, claims=claims,
                         status="qualified" if (letter or not q.choices) else "abstain")
        fa.tool_usage.update({"calls": 1, "by_tool": {"retrieve_source": 1}, "retrieved_span_ids": retrieved,
                              "cited_invalid_span_ids": [s for s in cited if s not in retrieved]})
        return fa


class MasterSystem:
    """QASystem adapter around MasterAgent (single_agent_tools / multi_agent_no_workspace / otter_full)."""

    def __init__(self, llm: LLM, store: Any, config: OtterConfig, *, name: str, llm_cache: Optional[LRU] = None):
        self.name = name
        self.config = config
        self.master = MasterAgent(llm, store, config=config, system_name=name, llm_cache=llm_cache,
                                  resolver=CitationResolver(store))

    def answer(self, item: dict[str, Any]) -> FinalAnswer:
        fa = self.master.answer(item)
        fa.tool_usage["corpus"] = getattr(self, "corpus", None)
        return fa


# --------------------------------------------------------------------------- factory
def _budgets() -> dict[str, int]:
    try:
        return dict(load_config().get("budgets", {}))
    except Exception:
        return {}


def get_system(name: str, backbone: str = "main", **flags: Any):
    """Build a QASystem. Unknown *flags* are ignored (logged); unknown *names* raise ValueError."""
    base = ALIASES.get(name, name)
    if base not in SYSTEM_NAMES:
        raise ValueError(f"unknown WS3 system {name!r}; expected one of {SYSTEM_NAMES}")
    llm = flags.pop("llm", None) or get_llm(backbone)
    cache = flags.pop("llm_cache", None)
    if cache is None:
        cache = LRU(50000)
    elif cache is False:
        cache = None
    store = flags.pop("store", None)
    db_path = flags.pop("db_path", None)
    prov = flags.pop("provenance_path", None)
    abl = {a: bool(flags.pop(a, False)) for a in ABLATIONS}
    suffix = "".join(f"-{a}" for a, on in abl.items() if on)
    sys_name = flags.pop("system_name", None) or (base + suffix)

    if base == "direct":
        flags.pop("book_prefix", None)
        flags.pop("corpus_name", None)
        _warn_unused(flags)
        d = DirectSystem(llm, name=sys_name, llm_cache=cache)
        d.corpus = "closed_book"
        return d
    prefix = flags.pop("book_prefix", DEFAULT_BOOK_PREFIX if store is None else None)
    corpus = flags.pop("corpus_name", CORPUS_NAME if prefix == DEFAULT_BOOK_PREFIX else (prefix or "all_books"))
    store = _wrap_store(store, prefix) if store is not None else default_store(db_path, prov, prefix)
    if base == "text_rag":
        s = TextRAGSystem(llm, store, top_k=int(flags.pop("top_k", 6)), name=sys_name, llm_cache=cache)
        s.corpus = corpus
        _warn_unused(flags)
        return s

    if base == "otter_v2":
        from otter.agents.v2 import build_otter_v2
        calc_flags = {k: flags.pop(k) for k in list(flags) if k in ("max_subagents", "max_tool_calls")}

        def _calc():
            return get_system("otter_full", backbone, llm=llm, llm_cache=cache or False, store=store,
                              system_name="otter_v2_calc", **calc_flags)
        s = build_otter_v2(llm, store, flags, name=sys_name, cache=cache, calc_factory=_calc)
        s.corpus = corpus
        _warn_unused(flags)
        return s

    b = _budgets()
    cfg = OtterConfig(
        max_subagents=int(flags.pop("max_subagents", b.get("max_subagents", 4))),
        max_depth=int(flags.pop("max_depth", b.get("max_depth", 3))),
        max_replan_rounds=int(flags.pop("max_replan_rounds", b.get("max_replan_rounds", 2))),
        max_tool_calls_per_subagent=int(flags.pop("max_tool_calls", b.get("max_tool_calls_per_subagent", 8))),
        max_subgoals=int(flags.pop("max_subgoals", 4)),
        max_question_seconds=float(flags.pop("max_question_seconds", 900)),
        decomposition=not abl["no_decomposition"], graph_retrieval=not abl["no_graph_retrieval"],
        source_verification=not abl["no_source_verification"], proof_completion=not abl["no_proof_completion"],
        rule_execution=not abl["no_rule_execution"], warm_start=bool(flags.pop("warm_start", True)),
        always_plan=bool(flags.pop("always_plan", False)), keep_transcripts=bool(flags.pop("keep_transcripts", False)),
        keep_trace=bool(flags.pop("keep_trace", True)),
    )
    if base == "single_agent_tools":
        # equalised budget: one agent gets the tool calls OTTER would typically spend on ~2 subagents
        cfg.max_tool_calls_per_subagent = int(flags.pop("tool_budget", 2 * cfg.max_tool_calls_per_subagent))
        cfg.decomposition = False
        cfg.proof_completion = False
        cfg.max_subagents = 1
        cfg.max_replan_rounds = 0
    elif base == "multi_agent_no_workspace" or abl["no_workspace"]:
        cfg.workspace = False
        cfg.proof_completion = False
        cfg.always_plan = cfg.always_plan or bool(flags.pop("always_plan_flat", False))
    _warn_unused(flags)
    m = MasterSystem(llm, store, cfg, name=sys_name, llm_cache=cache)
    m.corpus = corpus
    return m


def _warn_unused(flags: dict) -> None:
    if flags:
        log.warning("get_system: ignoring unknown flags %s", sorted(flags))
