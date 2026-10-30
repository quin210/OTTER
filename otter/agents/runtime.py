"""Runtime plumbing shared by all WS3 systems: metered + cached LLM, cached read-only store,
question input normalisation and cheap deterministic question analysis.

Everything here is thread-safe: one system instance serves many concurrent questions
(runner workers) and each question runs several subagents concurrently.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from otter.llm.interface import LLM, LLMResponse, parse_json_loose

SUBJECTS = ("physics", "chemistry", "biology", "history", "geography", "literature")
SUBJECT_FROM_VI = {
    "vật lí": "physics", "vật lý": "physics", "hóa học": "chemistry", "hoá học": "chemistry",
    "sinh học": "biology", "lịch sử": "history", "địa lí": "geography", "địa lý": "geography",
    "ngữ văn": "literature", "văn": "literature",
}
_SUBJECT_KEYWORDS = {
    "physics": "lực gia tốc vận tốc newton động năng thế năng công suất điện áp dòng điện khối lượng quán tính chuyển động",
    "chemistry": "nguyên tử electron phân tử liên kết phản ứng oxi hóa mol dung dịch acid base ion hạt nhân proton",
    "biology": "tế bào enzyme quang hợp hô hấp gen protein màng sinh vật lục lạp ti thể nhiễm sắc thể",
    "history": "triều đại cách mạng chiến tranh văn minh đế quốc vương triều khởi nghĩa thế kỉ lịch sử",
    "geography": "khí hậu địa hình khí quyển sông ngòi dân số thạch quyển gió mưa nhiệt độ đất đai bản đồ",
    "literature": "nhân vật tác phẩm thơ truyện tác giả văn bản nghệ thuật ẩn dụ hoán dụ người kể chuyện",
}


# --------------------------------------------------------------------------- question input
@dataclass
class QuestionInput:
    qid: str
    question: str
    choices: Optional[dict[str, str]] = None
    subject: Optional[str] = None            # normalised english subject (ontology scope)
    grade: Optional[int] = None

    @classmethod
    def from_item(cls, item: dict[str, Any]) -> "QuestionInput":
        choices = item.get("options") or item.get("choices") or None
        if isinstance(choices, list):            # [{"id","text"}] form from the spec
            choices = {str(c.get("id")): str(c.get("text", "")) for c in choices if isinstance(c, dict)}
        if choices:
            choices = {str(k): str(v) for k, v in choices.items() if v is not None}
        subject = normalize_subject(item.get("domain")) or normalize_subject(item.get("subject"))
        q = str(item.get("question") or "")
        if subject is None:
            subject = guess_subject(q + " " + " ".join((choices or {}).values()))
        grade = item.get("grade")
        try:
            grade = int(grade) if grade is not None and str(grade).strip() != "" else None
        except (TypeError, ValueError):
            grade = None
        return cls(qid=str(item.get("id") or item.get("question_id") or ""), question=q, choices=choices or None,
                   subject=subject, grade=grade)


def normalize_subject(s: Any) -> Optional[str]:
    if not s:
        return None
    t = str(s).strip().lower()
    if t in SUBJECTS:
        return t
    return SUBJECT_FROM_VI.get(t)


def guess_subject(text: str) -> Optional[str]:
    t = (text or "").lower()
    scores = {s: sum(1 for w in kw.split() if w in t) for s, kw in _SUBJECT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


_CALC = re.compile(r"(\btính\b|bao nhiêu|\d+[\.,]?\d*\s*(m/s|km|kg|°c|oc|n\b|j\b|w\b|mol|g\b|%|cm|m\b|giờ|phút|s\b))", re.I)
_COND = re.compile(r"(\bnếu\b|\bkhi\b|trong điều kiện|giả sử|với điều kiện|trường hợp)", re.I)
_MULTI = re.compile(r"(vì sao|tại sao|giải thích|dẫn đến|hệ quả|hậu quả|nguyên nhân|kết quả là|ảnh hưởng|so sánh|"
                    r"điểm giống|điểm khác|mối quan hệ|trước tiên|sau đó|tiếp theo|đồng thời|cả hai|"
                    r"từ đó|nhờ đó|do đó|góp phần|tạo điều kiện|gián tiếp)", re.I)


_RECOG = re.compile(r"(khái niệm nào phù hợp|phát biểu nào mô tả đúng nhất về|thuật ngữ nào|được gọi là gì|"
                    r"là gì\s*\??\s*$|nghĩa là gì|định nghĩa)", re.I)


def detect_question_type(question: str, choices: Optional[dict[str, str]] = None) -> str:
    """Cheap deterministic analysis used only to skip the planner LLM on clear recognition items.

    direct      -- recognition / definition templates (one subgoal, no planner call)
    calculation -- numeric quantities to compute
    multi_hop   -- explicit causal / comparative / sequential cues
    conditional -- explicit condition / scenario
    open        -- anything else: the planner LLM decides between 1 subgoal and a DAG
    """
    q = question or ""
    n_numbers = len(re.findall(r"\d+(?:[\.,]\d+)?", q))
    if n_numbers >= 2 and _CALC.search(q):
        return "calculation"
    if _RECOG.search(q) and len(q) < 260:
        return "direct"
    if _MULTI.search(q):
        return "multi_hop"
    if _COND.search(q):
        return "conditional"
    return "open"


# --------------------------------------------------------------------------- LRU cache
class LRU:
    def __init__(self, maxsize: int = 4096, ttl: Optional[float] = None):
        self.maxsize, self.ttl = maxsize, ttl
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> tuple[bool, Any]:
        with self._lock:
            if key in self._d:
                t, v = self._d[key]
                if self.ttl is None or time.time() - t < self.ttl:
                    self._d.move_to_end(key)
                    self.hits += 1
                    return True, v
                del self._d[key]
            self.misses += 1
            return False, None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.time(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


class CachedStore:
    """Read-only caching proxy over an `OntologyReader` (runtime never mutates the store).

    Search results are cached with a TTL because WS1/WS2 add spans/ontology progressively.
    """
    CACHED = {"search_knowledge", "search_spans", "get_concept", "get_fact", "get_rule", "rules_concluding",
              "neighbors", "get_spans", "span_exists", "book_metadata", "section"}
    DEFAULTS = {"search_knowledge": list, "get_concept": None, "get_fact": None, "get_rule": None,
                "rules_concluding": list, "neighbors": lambda: {"nodes": [], "relations": [], "rules": []}}

    def __init__(self, store: Any, maxsize: int = 20000, ttl: float = 900.0):
        self._store = store
        self._cache = LRU(maxsize, ttl)

    @property
    def inner(self) -> Any:
        return self._store

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._store, name)
        if name not in self.CACHED or not callable(attr):
            return attr

        def call(*a, **kw):
            try:
                return attr(*a, **kw)
            except sqlite3.OperationalError:      # e.g. ontology tables not built yet -> behave as empty
                if name not in self.DEFAULTS:
                    raise
                d = self.DEFAULTS[name]
                return d() if callable(d) else d

        def wrapped(*a, **kw):
            try:
                key = (name, json.dumps([a, kw], sort_keys=True, ensure_ascii=False, default=str))
            except (TypeError, ValueError):
                return call(*a, **kw)
            hit, v = self._cache.get(key)
            if hit:
                return v
            v = call(*a, **kw)
            self._cache.put(key, v)
            return v
        return wrapped


# --------------------------------------------------------------------------- metered LLM
@dataclass
class Meter:
    """Thread-safe per-question usage meter (same fields as `otter.llm.UsageMeter`)."""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    llm_latency_ms: float = 0.0
    errors: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResponse, cached: bool = False) -> None:
        with self._lock:
            if cached:
                self.cache_hits += 1
                return
            self.llm_calls += 1
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens
            self.llm_latency_ms += r.latency_ms or 0.0

    def error(self) -> None:
        with self._lock:
            self.errors += 1

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {"llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens, "tokens": self.prompt_tokens + self.completion_tokens,
                    "llm_cache_hits": self.cache_hits, "llm_latency_ms": round(self.llm_latency_ms, 1),
                    "llm_errors": self.errors}


class MeteredLLM(LLM):
    """Wraps a backbone LLM: per-question metering, optional shared response cache (temperature 0
    only, so cached replies are what the deterministic decode would return anyway), and retries."""

    def __init__(self, llm: LLM, meter: Meter, cache: Optional[LRU] = None, retries: int = 2):
        self.llm, self.meter, self.cache, self.retries = llm, meter, cache, retries
        self.model = getattr(llm, "model", "")
        self._dead: Optional[Exception] = None        # per-question circuit breaker


    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_schema=None, stop=None):
        key = None
        if self.cache is not None and temperature == 0:
            key = hashlib.sha1(json.dumps([self.model, messages, max_tokens, json_schema, stop], ensure_ascii=False,
                                          sort_keys=True).encode()).hexdigest()
            hit, v = self.cache.get(key)
            if hit:
                self.meter.add(v, cached=True)
                return v
        if self._dead is not None:
            raise RuntimeError(f"llm_unavailable (circuit open): {self._dead}")
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                r = self.llm.chat(messages, temperature=temperature, max_tokens=max_tokens, json_schema=json_schema,
                                  stop=stop)
                self.meter.add(r)
                if key is not None:
                    self.cache.put(key, r)
                return r
            except Exception as e:                      # transient server errors: retry with backoff
                last = e
                self.meter.error()
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        self._dead = last
        raise last  # type: ignore[misc]

    def chat_json(self, messages, schema=None, **kw):
        r = self.chat(messages, json_schema=schema, **kw)
        return parse_json_loose(r.text), r


def safe_json(llm: LLM, messages: list[dict], *, max_tokens: int = 600, retry_msg: Optional[str] = None
              ) -> tuple[Optional[Any], str]:
    """One chat call parsed as JSON; on a parse failure one repair turn. Returns (obj | None, raw_text)."""
    r = llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    try:
        return parse_json_loose(r.text), r.text
    except ValueError:
        pass
    if retry_msg is None:
        return None, r.text
    msgs = messages + [{"role": "assistant", "content": r.text[:2000]}, {"role": "user", "content": retry_msg}]
    r2 = llm.chat(msgs, temperature=0.0, max_tokens=max_tokens)
    try:
        return parse_json_loose(r2.text), r2.text
    except ValueError:
        return None, r2.text


def extract_letter(x: Any, choices: Optional[dict[str, str]]) -> Optional[str]:
    """Normalise an LLM answer field to a choice key ('B', 'b.', 'Đáp án B', option text)."""
    if not choices or x is None:
        return None
    s = str(x).strip()
    keys = list(choices)
    if s in choices:
        return s
    m = re.match(r"^\(?([A-Za-z])[\.\):]?(\s|$)", s)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    m = re.search(r"(?:đáp án|answer|lựa chọn)\s*[:\-]?\s*([A-Za-z])\b", s, re.I)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    low = s.lower()
    for k in keys:
        if choices[k].strip().lower() == low:
            return k
    return None
