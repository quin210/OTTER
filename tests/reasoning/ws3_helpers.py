"""Shared WS3 test helpers: fixture span ids + a scripted MockLLM router."""
from __future__ import annotations

import json
import threading
from typing import Callable, Optional

from otter.llm.interface import MockLLM

BIO = "fx_bio11_synthetic"
GEO = "fx_geo10_synthetic"
S = {  # fixture span ids (self-written synthetic text, see otter/storage/fixtures.py)
    "bio_def": f"{BIO}:p0000:u001", "bio_luclap": f"{BIO}:p0000:u002", "bio_light": f"{BIO}:p0000:u003",
    "bio_sat": f"{BIO}:p0000:u004", "geo_lapse": f"{GEO}:p0000:u002", "geo_rain": f"{GEO}:p0000:u003",
    "geo_flood": f"{GEO}:p0000:u005", "geo_tdl": f"{GEO}:p0000:u001",
}


def finish(**kw) -> str:
    return json.dumps({"thought": "done", "action": "finish", "arguments": kw}, ensure_ascii=False)


def act(tool: str, **kw) -> str:
    return json.dumps({"thought": "call", "action": tool, "arguments": kw}, ensure_ascii=False)


class Router:
    """Routes MockLLM calls by prompt role and records them (thread-safe).

    agent(goal_line, messages) -> reply text; judge(user_msg) -> {"verdicts": [...]}.
    """

    def __init__(self, *, plan: Optional[dict] = None, agent: Optional[Callable[[str, list], str]] = None,
                 judge: Optional[Callable[[str], dict]] = None, direct: str = "A"):
        self.plan, self.agent, self.judge, self.direct = plan, agent, judge, direct
        self.calls: list[str] = []
        self.goals: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, msgs: list[dict]) -> str:
        sys_ = msgs[0]["content"]
        goal = ""
        if "Tác tử Chủ" in sys_:
            role, out = "planner", json.dumps(self.plan or {"subgoals": []}, ensure_ascii=False)
        elif "Tác tử con" in sys_:
            u = msgs[1]["content"]
            goal = u.split("Mục tiêu con được giao", 1)[1].split("\n", 1)[0] if "Mục tiêu con được giao" in u else ""
            role, out = "agent", (self.agent(goal, msgs) if self.agent else finish(claim="?"))
        elif "bộ kiểm tra bằng chứng" in sys_:
            role = "judge"
            out = json.dumps(self.judge(msgs[-1]["content"]) if self.judge else {"verdicts": []}, ensure_ascii=False)
        elif "diễn đạt câu trả lời" in sys_:
            role, out = "verbalize", json.dumps({"answer": "Câu trả lời đã kiểm chứng."}, ensure_ascii=False)
        else:
            role, out = "direct", json.dumps({"answer": self.direct}, ensure_ascii=False)
        with self.lock:
            self.calls.append(role)
            if goal and len(msgs) == 2:
                self.goals.append(goal)
        return out

    def llm(self) -> MockLLM:
        return MockLLM(handler=self)

    def count(self, role: str) -> int:
        with self.lock:
            return self.calls.count(role)


def judge_from(table: dict) -> Callable[[str], dict]:
    """table: claim id -> (verdict, span_id, quote); a claim id is only answered if it appears in the prompt."""
    def f(user: str) -> dict:
        return {"verdicts": [{"id": k, "verdict": v, "span_id": s, "quote": q} for k, (v, s, q) in table.items()
                             if f"\n{k}. " in user or user.startswith(f"{k}. ")]}
    return f
