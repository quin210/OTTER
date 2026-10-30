"""LLM provider abstraction. All modules call `get_llm(role)`; tests use MockLLM.

Endpoints are OpenAI-compatible vLLM servers on localhost (see configs/default.yaml
and docs/PARALLEL_PLAN.md). No external APIs are used.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""


@dataclass
class UsageMeter:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, r: LLMResponse) -> None:
        self.llm_calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens


class LLM:
    model: str = ""

    def chat(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024,
             json_schema: Optional[dict] = None, stop: Optional[list[str]] = None) -> LLMResponse:
        raise NotImplementedError

    def chat_json(self, messages: list[dict], schema: Optional[dict] = None, **kw) -> tuple[Any, LLMResponse]:
        """Chat and parse a JSON object; uses guided decoding when a schema is given."""
        r = self.chat(messages, json_schema=schema, **kw)
        return parse_json_loose(r.text), r


def parse_json_loose(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")


class OpenAICompatLLM(LLM):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: float = 600,
                 enable_thinking: bool = False):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.enable_thinking = enable_thinking

    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_schema=None, stop=None):
        extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        # Server-side guided decoding (xgrammar) made vLLM's engine loop CPU-bound with dozens of
        # concurrent schema-constrained requests (~1.6 s/iteration on Qwen3-32B). By default the
        # schema goes into the prompt and the reply is parsed loosely; OTTER_GUIDED_JSON=1 restores it.
        guided = json_schema is not None and os.environ.get("OTTER_GUIDED_JSON") == "1"
        if guided:
            extra["guided_json"] = json_schema
        elif json_schema is not None:
            messages = _with_schema_instruction(messages, json_schema)
        t = time.perf_counter()
        r = self.client.chat.completions.create(model=self.model, messages=messages, temperature=temperature,
                                                max_tokens=max_tokens, stop=stop, extra_body=extra)
        u = r.usage
        text = r.choices[0].message.content or ""
        raw_text = text
        if json_schema is not None and not guided:
            try:  # hand callers clean JSON text, as guided decoding would
                text = json.dumps(parse_json_loose(text), ensure_ascii=False)
            except ValueError:
                pass
        resp = LLMResponse(text=text, model=self.model,
                           prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                           completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                           latency_ms=(time.perf_counter() - t) * 1000)
        _log_call({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": self.model, "url": str(self.client.base_url),
                   "temperature": temperature, "max_tokens": max_tokens, "schema": bool(json_schema),
                   "messages": messages, "raw_response": raw_text, "text": text,
                   "finish_reason": r.choices[0].finish_reason, "prompt_tokens": resp.prompt_tokens,
                   "completion_tokens": resp.completion_tokens, "latency_ms": round(resp.latency_ms, 1)})
        return resp


def _with_schema_instruction(messages: list[dict], schema: dict) -> list[dict]:
    note = ("\n\nChỉ trả lời bằng MỘT đối tượng JSON hợp lệ (không giải thích, không markdown) "
            "tuân theo JSON Schema sau:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = m["content"] + note
            return out
    return out + [{"role": "user", "content": note.strip()}]


@dataclass
class MockLLM(LLM):
    """Deterministic offline LLM: `handler(messages) -> str`, or scripted replies in order."""
    handler: Optional[Callable[[list[dict]], str]] = None
    replies: list[str] = field(default_factory=list)
    model: str = "mock"

    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_schema=None, stop=None):
        if self.handler is not None:
            text = self.handler(messages)
        elif self.replies:
            text = self.replies.pop(0)
        else:
            text = "{}"
        return LLMResponse(text=text, model=self.model, prompt_tokens=sum(len(m["content"]) // 4 for m in messages),
                           completion_tokens=len(text) // 4)


_LOG_LOCK = threading.Lock()


def _log_call(entry: dict) -> None:
    """Append the exact prompt/response of one LLM call to runs/llm_calls/<run tag>.<pid>.jsonl.

    On by default so every final number can be audited; set OTTER_LLM_LOG=0 to disable,
    OTTER_LLM_LOG_DIR to relocate, OTTER_RUN_TAG (set by scripts/eval/run.py) to group by run.
    """
    if os.environ.get("OTTER_LLM_LOG", "1") == "0":
        return
    try:
        d = Path(os.environ.get("OTTER_LLM_LOG_DIR") or CONFIG_PATH.parents[1] / "runs" / "llm_calls")
        d.mkdir(parents=True, exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9_.+=-]+", "__", os.environ.get("OTTER_RUN_TAG", "untagged"))
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _LOG_LOCK, open(d / f"{tag}.{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # logging must never break a run


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_llm(role: str = "main", config: Optional[dict] = None) -> LLM:
    """role in configs/default.yaml `llm` section (main, small, ...). OTTER_LLM_MOCK=1 forces MockLLM."""
    if os.environ.get("OTTER_LLM_MOCK") == "1":
        return MockLLM()
    cfg = (config or load_config())["llm"][role]
    base_url = os.environ.get(f"OTTER_LLM_URL_{role.upper()}") or cfg["base_url"]   # route one process to a replica of the same model
    return OpenAICompatLLM(base_url=base_url, model=cfg["model"],
                           enable_thinking=cfg.get("enable_thinking", False))
