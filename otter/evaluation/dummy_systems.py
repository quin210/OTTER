"""Systems used to test the evaluation pipeline (not paper baselines).

* `dummy:first`  — always answers A.
* `dummy:random` — deterministic pseudo-random choice per question id.
* `dummy:oracle` — gold answer + gold spans as citations (needs `bench=`); upper bound / test only.
* `dummy:abstain` — always abstains.
* `eval_direct`  — closed-book prompt through `get_llm(backbone)`; used for pipeline sanity checks
  before WS3's `direct` system is available. Answers carry no citations.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from otter.llm.interface import LLM, UsageMeter, get_llm, parse_json_loose
from otter.schemas.reasoning import AnswerClaim, Citation, FinalAnswer

LETTERS = ("A", "B", "C", "D")


class FirstSystem:
    name = "dummy_first"

    def answer(self, item: dict[str, Any]) -> FinalAnswer:
        return FinalAnswer(question_id=item["id"], system=self.name, answer="A", answer_choice="A")


class RandomSystem:
    name = "dummy_random"

    def answer(self, item):
        c = LETTERS[int(hashlib.md5(item["id"].encode()).hexdigest(), 16) % 4]
        return FinalAnswer(question_id=item["id"], system=self.name, answer=c, answer_choice=c)


class AbstainSystem:
    name = "dummy_abstain"

    def answer(self, item):
        return FinalAnswer(question_id=item["id"], system=self.name, answer="Không đủ bằng chứng.", status="abstain")


class OracleSystem:
    name = "dummy_oracle"

    def __init__(self, bench: dict, corpus=None):
        self.bench = bench
        self.corpus = corpus

    def answer(self, item):
        g = self.bench[item["id"]]
        cites = []
        for sid in g.gold_span_ids:
            s = self.corpus.get(sid) if self.corpus else None
            cites.append(Citation(book_id=s.book_id if s else "unknown", span_id=sid,
                                  book_title=s.book_title if s else None, subject=s.domain if s else None,
                                  grade=s.grade if s else None))
        claims = [AnswerClaim(claim_id="c1", text=g.answer_text, citations=cites)] if cites else []
        return FinalAnswer(question_id=item["id"], system=self.name, answer=g.answer, answer_choice=g.answer,
                           claims=claims)


DIRECT_PROMPT_VERSION = "eval_direct_v1"
_SYS = ("Bạn là trợ lý trả lời câu hỏi trắc nghiệm theo chương trình sách giáo khoa Việt Nam. "
        "Chỉ trả về JSON hợp lệ.")


def format_mcq(item: dict[str, Any]) -> str:
    opts = "\n".join(f"{k}. {item['options'][k]}" for k in LETTERS if k in item["options"])
    head = f"Môn: {item.get('subject') or item.get('domain') or ''}; lớp {item.get('grade') or '?'}\n"
    return f"{head}Câu hỏi: {item['question']}\n{opts}"


class EvalDirect:
    """Closed-book single call. JSON {"answer": "A|B|C|D", "confidence": 0-100}."""
    name = "eval_direct"
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": list(LETTERS)},
                                                "confidence": {"type": "integer", "minimum": 0, "maximum": 100}},
              "required": ["answer", "confidence"]}

    def __init__(self, backbone: str = "small", llm: Optional[LLM] = None):
        self.llm = llm or get_llm(backbone)
        self.backbone = backbone

    def answer(self, item):
        meter = UsageMeter()
        msgs = [{"role": "system", "content": _SYS},
                {"role": "user", "content": format_mcq(item) +
                 '\n\nChọn một đáp án đúng nhất. Trả về JSON: {"answer": "<A|B|C|D>", "confidence": <0-100>}'}]
        try:
            r = self.llm.chat(msgs, json_schema=self.schema, max_tokens=64)
            meter.add(r)
            obj = parse_json_loose(r.text)
            c = str(obj.get("answer", "")).strip().lstrip("(").upper()[:1]
            c = c if c in LETTERS else None
            return FinalAnswer(question_id=item["id"], system=self.name, answer=c or r.text[:200], answer_choice=c,
                               status="supported" if c else "failed",
                               tool_usage={"calls": 0, "latency_ms": r.latency_ms, "prompt_tokens": meter.prompt_tokens,
                                           "completion_tokens": meter.completion_tokens, "llm_calls": meter.llm_calls,
                                           "subagents": 0, "confidence": obj.get("confidence")})
        except Exception as e:  # never raise (QASystem contract)
            return FinalAnswer(question_id=item["id"], system=self.name, answer="", status="failed",
                               error=f"{type(e).__name__}: {e}")


class GoldContext:
    """Oracle-evidence reference: the LLM reads the item's *gold spans* (from the benchmark) and answers.
    Upper bound for retrieval; never a baseline. Citations = gold spans (trivially valid)."""
    name = "gold_context"
    schema = EvalDirect.schema

    def __init__(self, bench: dict, corpus, backbone: str = "main", llm: Optional[LLM] = None):
        self.bench, self.corpus = bench, corpus
        self.llm = llm or get_llm(backbone)

    def answer(self, item):
        g = self.bench[item["id"]]
        spans = [self.corpus.get(s) for s in g.gold_span_ids] if self.corpus is not None else []
        spans = [s for s in spans if s is not None]
        ctx = "\n\n".join(f"[{i + 1}] {s.text[:2500]}" for i, s in enumerate(spans)) or "(không có)"
        msgs = [{"role": "system", "content": _SYS},
                {"role": "user", "content": f"Đoạn trích sách giáo khoa:\n{ctx}\n\n" + format_mcq(item) +
                 '\n\nDựa vào đoạn trích, chọn một đáp án đúng nhất. Trả về JSON: {"answer": "<A|B|C|D>", "confidence": <0-100>}'}]
        try:
            r = self.llm.chat(msgs, json_schema=self.schema, max_tokens=64)
            obj = parse_json_loose(r.text)
            c = str(obj.get("answer", "")).strip().lstrip("(").upper()[:1]
            c = c if c in LETTERS else None
            cites = [Citation(book_id=s.book_id, span_id=s.span_id, book_title=s.book_title, subject=s.domain,
                              grade=s.grade) for s in spans]
            claims = [AnswerClaim(claim_id="c1", text=g.options.get(c, "") if c else "", citations=cites)] if cites and c else []
            return FinalAnswer(question_id=item["id"], system=self.name, answer=c or r.text[:200], answer_choice=c,
                               status="supported" if c else "failed", claims=claims,
                               tool_usage={"calls": 0, "latency_ms": r.latency_ms, "prompt_tokens": r.prompt_tokens,
                                           "completion_tokens": r.completion_tokens, "llm_calls": 1, "subagents": 0})
        except Exception as e:
            return FinalAnswer(question_id=item["id"], system=self.name, answer="", status="failed",
                               error=f"{type(e).__name__}: {e}")


def get_dummy(name: str, *, backbone: str = "small", bench: Optional[dict] = None, corpus=None, llm=None):
    if name == "first":
        return FirstSystem()
    if name == "random":
        return RandomSystem()
    if name == "abstain":
        return AbstainSystem()
    if name == "oracle":
        assert bench is not None, "oracle needs the bench"
        return OracleSystem(bench, corpus)
    if name == "eval_direct":
        return EvalDirect(backbone=backbone, llm=llm)
    if name == "gold_context":
        assert bench is not None, "gold_context needs the bench"
        return GoldContext(bench, corpus, backbone=backbone, llm=llm)
    raise KeyError(f"unknown dummy system {name!r}")
