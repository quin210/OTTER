"""LLM judge for claim-to-span support (heuristic; disclosed in the paper).

For each answer claim we show the judge the claim and the *text of the spans it cites* (resolved
from the corpus; never from the system's own quote field) and ask for supported / partial /
unsupported. Claims without any resolvable citation are labelled `no_citation` without a call.
The judge must be a different model than the evaluated system's backbone (enforced by the CLI).
"""
from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from otter.llm.interface import LLM, parse_json_loose
from otter.schemas.reasoning import FinalAnswer

JUDGE_PROMPT_VERSION = "claim_support_v1"
VERDICTS = ("supported", "partial", "unsupported")
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)},
                                           "reason": {"type": "string"}}, "required": ["verdict", "reason"]}

_PROMPT = """Bạn là người kiểm định bằng chứng. Hãy đánh giá xem MỆNH ĐỀ có được các ĐOẠN TRÍCH sách giáo khoa \
bên dưới hỗ trợ hay không. Chỉ dựa vào nội dung các đoạn trích, không dùng kiến thức bên ngoài.
- "supported": mọi thông tin quan trọng trong mệnh đề được các đoạn trích nêu trực tiếp hoặc suy ra hiển nhiên.
- "partial": một phần mệnh đề được hỗ trợ, phần còn lại không có trong đoạn trích.
- "unsupported": đoạn trích không hỗ trợ hoặc mâu thuẫn với mệnh đề.

MỆNH ĐỀ: {claim}

ĐOẠN TRÍCH:
{spans}

Trả về JSON: {{"verdict": "supported|partial|unsupported", "reason": "<ngắn gọn>"}}"""


def judge_answer(fa: FinalAnswer, corpus, llm: LLM, *, max_span_chars: int = 2500) -> list[dict[str, Any]]:
    out = []
    for cl in fa.claims:
        spans = [corpus.get(c.span_id) for c in cl.citations]
        spans = [s for s in spans if s is not None]
        base = {"question_id": fa.question_id, "claim_id": cl.claim_id, "claim": cl.text,
                "span_ids": [c.span_id for c in cl.citations], "judge_model": llm.model,
                "prompt_version": JUDGE_PROMPT_VERSION}
        if not spans:
            out.append({**base, "verdict": "no_citation", "reason": "no resolvable citation"})
            continue
        txt = "\n\n".join(f"[{s.span_id}] {s.text[:max_span_chars]}" for s in spans)
        try:
            from otter.evaluation.hard_gen import chat_json
            obj, r = chat_json(llm, [{"role": "user", "content": _PROMPT.format(claim=cl.text, spans=txt)}], SCHEMA,
                               max_tokens=256, validate=lambda o: str(o.get("verdict", "")).strip().lower() in VERDICTS)
            v = str(obj.get("verdict", "")).strip().lower()
            out.append({**base, "verdict": v if v in VERDICTS else "unsupported", "reason": obj.get("reason", ""),
                        "raw_verdict": v})
        except Exception as e:
            out.append({**base, "verdict": "error", "reason": f"{type(e).__name__}: {e}"})
    return out


def judge_run(answers: dict[str, FinalAnswer], corpus, llm: LLM, out_path: Path, *, workers: int = 8) -> Path:
    """Resumable: skips question ids already present in out_path."""
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["verdict"] != "error":
                    done.add(r["question_id"])
    lock = threading.Lock()

    def one(fa):
        rows = judge_answer(fa, corpus, llm)
        if not rows:
            rows = []
        with lock, open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    todo = [fa for q, fa in answers.items() if q not in done and fa.claims]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, todo))
    out_path.touch()
    return out_path


def load_judgments(path: Path) -> dict[tuple[str, str], dict]:
    res: dict[tuple[str, str], dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["verdict"] != "error":
                res[(r["question_id"], r["claim_id"])] = r
    return res


def audit_sample(judgments: dict, n: int, seed: int = 0) -> list[dict]:
    """Stratified (by verdict) sample of judged claims for the human audit sheet."""
    rng = random.Random(seed)
    by_v: dict[str, list] = {}
    for r in judgments.values():
        if r["verdict"] in VERDICTS:
            by_v.setdefault(r["verdict"], []).append(r)
    total = sum(len(v) for v in by_v.values()) or 1
    out = []
    for v, rows in sorted(by_v.items()):
        k = max(1, round(n * len(rows) / total))
        out += rng.sample(rows, min(k, len(rows)))
    rng.shuffle(out)
    return out[:n]


def agreement(human: list[dict], judgments: dict) -> Optional[dict]:
    """human rows: {"question_id","claim_id","human_verdict"} -> accuracy + Cohen's kappa."""
    pairs = [(h["human_verdict"], judgments[(h["question_id"], h["claim_id"])]["verdict"]) for h in human
             if h.get("human_verdict") in VERDICTS and (h["question_id"], h["claim_id"]) in judgments]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == v for a, _ in pairs) / n) * (sum(b == v for _, b in pairs) / n) for v in VERDICTS)
    return {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0}
