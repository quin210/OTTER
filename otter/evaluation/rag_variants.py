"""Classic RAG baselines as explicit systems: rag_bm25 (BM25 over spans), rag_dense (bge-m3 cosine over spans),
rag_hybrid (reciprocal-rank fusion k=60 of BM25 + dense). `text_rag` is an alias of rag_hybrid.

Identical for all three: v1-core span index (16,716 `nxbgd_` spans, bge-m3 vectors from the ontology DB `embeddings` table),
query = question + options, subject filter only (no grade filter, because OTTER-Hard items span grades), top-k spans
(default 5; sensitivity: --flag top_k=10) shown with the shared WS3 RAG answer prompt, citations = model-cited ids that were
actually retrieved. The ordered top-10 retrieval list is stored in tool_usage["retrieved_top10"] for recall@k / MRR.
Per-question span exclusion (OTTER-Abstain) is honoured via otter.evaluation.exclusion.current_exclusion().
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from otter.agents import prompts as P
from otter.agents.runtime import MeteredLLM, QuestionInput, extract_letter, safe_json
from otter.agents.systems import _Base
from otter.evaluation.corpus import SpanCorpus, core_book_filter
from otter.evaluation.exclusion import current_exclusion
from otter.llm.interface import get_llm
from otter.reasoning.citation_resolver import CitationResolver
from otter.reasoning.text import truncate
from otter.schemas.reasoning import AnswerClaim, FinalAnswer

ROOT = Path(__file__).resolve().parents[2]
MODES = {"rag_bm25": "bm25", "rag_dense": "dense", "rag_hybrid": "hybrid", "text_rag": "hybrid"}
_LOCK = threading.Lock()
_INDEX: dict[str, Any] = {}


def get_index() -> SpanCorpus:
    """Shared in-process index: BM25 (bigram syllables) + bge-m3 span vectors from the ontology DB."""
    with _LOCK:
        if "corpus" not in _INDEX:
            corpus = SpanCorpus.load(book_filter=core_book_filter)
            db = ROOT / "data/ontology/ontology.sqlite3"
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
                vec = {i: np.frombuffer(b, dtype=np.float32) for i, b in
                       c.execute("SELECT item_id, vec FROM embeddings WHERE kind='span' AND model='BAAI/bge-m3'")}
            dim = len(next(iter(vec.values())))
            mat = np.zeros((len(corpus.spans), dim), dtype=np.float32)
            miss = 0
            for k, s in enumerate(corpus.spans):
                v = vec.get(s.span_id)
                if v is None:
                    miss += 1
                else:
                    mat[k] = v / (np.linalg.norm(v) or 1.0)
            from otter.evaluation.embeddings import get_embedder
            emb = get_embedder()
            corpus.attach_embedder(emb)
            corpus._dense = mat                     # attach_embedder resets it; install precomputed vectors
            corpus._missing_dense = miss
            _INDEX["corpus"] = corpus
        return _INDEX["corpus"]


def _norm_q(corpus: SpanCorpus, q: str) -> np.ndarray:
    v = np.asarray(corpus._embedder.embed([q])[0], dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def retrieve(query: str, *, mode: str, subject: Optional[str], top_k: int = 10, exclude: frozenset = frozenset()) -> list[str]:
    corpus = get_index()
    mask = corpus._mask(subject, None, None, False)
    ranks = []
    if mode in ("bm25", "hybrid"):
        s = np.where(mask, corpus.bm25_scores(query), -np.inf)
        ranks.append(s)
    if mode in ("dense", "hybrid"):
        d = corpus._dense @ _norm_q(corpus, query)
        ranks.append(np.where(mask, d, -np.inf))
    if exclude:
        ex = np.array([s.span_id in exclude for s in corpus.spans], dtype=bool)
    else:
        ex = None
    fused = np.zeros(len(corpus.spans))
    for s in ranks:
        if ex is not None:
            s = np.where(ex, -np.inf, s)
        order = np.argsort(-s)
        r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
        fused += 1.0 / (60 + r)
    fused = np.where(mask, fused, -np.inf)
    if ex is not None:
        fused = np.where(ex, -np.inf, fused)
    order = np.argsort(-fused)[:top_k]
    return [corpus.spans[i].span_id for i in order if np.isfinite(fused[i])]


class RagVariantSystem(_Base):
    def __init__(self, llm, *, mode: str, top_k: int = 5, name: str, llm_cache=None, store=None):
        super().__init__(llm, name=name, llm_cache=llm_cache)
        self.mode, self.top_k = mode, int(top_k)
        self.corpus = "v1-core"
        self.resolver = CitationResolver(store) if store is not None else None

    def _answer(self, item, llm):
        q = QuestionInput.from_item(item)
        query = q.question + " " + " ".join((q.choices or {}).values())
        top10 = retrieve(query, mode=self.mode, subject=q.subject, top_k=max(10, self.top_k), exclude=current_exclusion())
        corpus = get_index()
        shown = top10[: self.top_k]
        spans = [{"span_id": i, "text": truncate(corpus.get(i).text, 700)} for i in shown]
        obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                   {"role": "user", "content": P.rag_user(q.question, q.choices, q.subject, q.grade, spans)}],
                             max_tokens=500, retry_msg=P.JSON_RETRY)
        ans = obj.get("answer") if isinstance(obj, dict) else None
        cited = [str(s) for s in (obj.get("cited_span_ids") or [])] if isinstance(obj, dict) else []
        valid = [s for s in cited if s in shown]
        letter = extract_letter(ans if ans is not None else raw, q.choices)
        claims = []
        if valid and (letter or not q.choices) and self.resolver is not None:
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
        fa.tool_usage.update({"calls": 1, "by_tool": {"retrieve_source": 1}, "retrieved_span_ids": shown,
                              "retrieved_top10": top10, "retrieval_mode": self.mode, "top_k": self.top_k,
                              "cited_invalid_span_ids": [s for s in cited if s not in shown]})
        return fa


def get_system(name: str, backbone: str = "main", **flags: Any) -> RagVariantSystem:
    from otter.agents.systems import DEFAULT_BOOK_PREFIX, default_store
    from otter.agents.runtime import LRU
    llm = flags.pop("llm", None) or get_llm(backbone)
    return RagVariantSystem(llm, mode=MODES[name], top_k=int(flags.pop("top_k", 5)), name=name, llm_cache=LRU(50000),
                            store=default_store(None, None, DEFAULT_BOOK_PREFIX))


# --------------------------------------------------------------------------- IRCoT (Trivedi et al., ACL 2023)
_COT_SYS = "Bạn là học sinh giỏi suy luận từng bước dựa trên sách giáo khoa. Chỉ viết ĐÚNG MỘT câu suy luận tiếp theo."
_COT_USER = """{scope}
Các đoạn trích đã thu thập:
{ev}

Câu hỏi: {question}
{choices}
Các bước suy luận đến nay: {cot}

Viết MỘT câu suy luận tiếp theo (ngắn gọn, dựa vào các đoạn trích, nêu rõ thông tin còn cần tra cứu). Nếu đã đủ thông tin để chọn đáp án, viết câu bắt đầu bằng "Vậy đáp án là"."""


class IRCoTSystem(RagVariantSystem):
    """Interleaving retrieval with chain-of-thought: <=4 reasoning steps; each step retrieves top-5 hybrid spans using the
    newest CoT sentence as the query; final answer with the shared rag_* MCQ prompt over all collected spans."""

    def __init__(self, llm, *, max_steps: int = 4, per_step: int = 5, max_shown: int = 12, **kw):
        super().__init__(llm, mode="hybrid", top_k=per_step, **kw)
        self.max_steps, self.per_step, self.max_shown = max_steps, per_step, max_shown

    def _answer(self, item, llm):
        q = QuestionInput.from_item(item)
        corpus = get_index()
        ex = current_exclusion()
        collected = retrieve(q.question + " " + " ".join((q.choices or {}).values()), mode="hybrid", subject=q.subject,
                             top_k=self.per_step, exclude=ex)
        cot: list[str] = []
        steps = 0
        for _ in range(self.max_steps):
            ev = "\n\n".join(f"[{i}] {truncate(corpus.get(i).text, 500)}" for i in collected[-self.max_shown:])
            ch = ("Các lựa chọn:\n" + "\n".join(f"{k}. {v}" for k, v in q.choices.items())) if q.choices else ""
            r = llm.chat([{"role": "system", "content": _COT_SYS},
                          {"role": "user", "content": _COT_USER.format(scope=P._scope(q.subject, q.grade), ev=ev or "(chưa có)",
                                                                        question=q.question, choices=ch, cot=" ".join(cot) or "(chưa có)")}],
                         temperature=0.0, max_tokens=120)
            sent = (r.text or "").strip().split("\n")[0].strip()
            if not sent:
                break
            cot.append(sent)
            steps += 1
            if "đáp án là" in sent.lower():
                break
            new = retrieve(sent, mode="hybrid", subject=q.subject, top_k=self.per_step, exclude=ex)
            collected += [s for s in new if s not in collected]
        self_top = collected
        shown = collected[-self.max_shown:] if len(collected) > self.max_shown else collected
        spans = [{"span_id": i, "text": truncate(corpus.get(i).text, 700)} for i in shown]
        obj, raw = safe_json(llm, [{"role": "system", "content": P.DIRECT_SYSTEM},
                                   {"role": "user", "content": P.rag_user(q.question, q.choices, q.subject, q.grade, spans)}],
                             max_tokens=500, retry_msg=P.JSON_RETRY)
        ans = obj.get("answer") if isinstance(obj, dict) else None
        cited = [str(s) for s in (obj.get("cited_span_ids") or [])] if isinstance(obj, dict) else []
        valid = [s for s in cited if s in shown]
        letter = extract_letter(ans if ans is not None else raw, q.choices)
        claims = []
        if valid and (letter or not q.choices) and self.resolver is not None:
            cites, _ = self.resolver.resolve_many(valid, with_quote=True, quote_chars=240)
            if cites:
                claims.append(AnswerClaim(claim_id="c1", text=f"Đáp án {letter}: {q.choices[letter]}" if q.choices else str(ans), citations=cites))
        text = (f"Đáp án {letter}. {q.choices[letter]}" if letter else "Không xác định được đáp án.") if q.choices else str(ans or raw).strip()
        if q.choices and letter and isinstance(obj, dict) and obj.get("explanation"):
            text += f" {truncate(str(obj['explanation']), 400)}"
        fa = FinalAnswer(question_id=q.qid, system=self.name, answer=text, answer_choice=letter, claims=claims,
                         status="qualified" if (letter or not q.choices) else "abstain")
        fa.tool_usage.update({"calls": 1 + steps, "by_tool": {"retrieve_source": 1 + steps}, "retrieved_span_ids": shown,
                              "retrieved_top10": self_top[:10], "retrieved_all": self_top, "retrieval_mode": "ircot",
                              "cot_steps": cot, "top_k": self.per_step,
                              "cited_invalid_span_ids": [s for s in cited if s not in shown]})
        return fa


def get_ircot(backbone: str = "main", **flags: Any) -> IRCoTSystem:
    from otter.agents.systems import DEFAULT_BOOK_PREFIX, default_store
    from otter.agents.runtime import LRU
    llm = flags.pop("llm", None) or get_llm(backbone)
    return IRCoTSystem(llm, max_steps=int(flags.pop("max_steps", 4)), name="ircot", llm_cache=LRU(50000),
                       store=default_store(None, None, DEFAULT_BOOK_PREFIX))
