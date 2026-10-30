"""Fill-in-the-blank runner for ViEduHop-Typed (the MCQ runner scripts/eval/run.py cannot score free-text answers).

  .venv/bin/python scripts/eval/typed_fill_runner.py --system direct|rag_hybrid|ircot|otter_v2 --backbone main|small \
      [--bench benchmarks/typed/typed600.jsonl] [--exp vieduhop_typed] [--workers 3]
  .venv/bin/python scripts/eval/typed_fill_runner.py --stage judge --system ... [--judge-url http://127.0.0.1:8005/v1]

It imports (never edits) the same systems as run.py: DirectSystem, rag_variants (rag_hybrid, ircot) and otter_v2. The fill_blank question
is wrapped by the SAME instruction for every system (typed_common.FILL_INSTRUCTION). Items are fed through otter.evaluation.runner.run_system
with MCQItem(options={}) so that the system receives no choices (QuestionInput.choices=None -> free-text branch of every system).

otter_v2 has a free-text branch (OtterV2.gen_user without choices) but returns FinalAnswer.answer == "" and discards the generator's
`answer` field. `V2FreeText` below is a thin subclass living in this file: it records the generator's raw reply through a recording proxy,
re-parses its `answer`, and puts it into FinalAnswer.answer after the unchanged frozen pipeline has run. Retrieval, generation,
verification, cascade and citation code are the frozen ones. The chosen generation round replicates the frozen rule (second round
kept iff its grounding >= first round's), checked against tool_usage['v2_grounding'].

Outputs under runs/<exp>/<system>/<backbone>/fill_blank/: answers.jsonl, run_meta.json, fill_records.jsonl (string-match score, location,
cost; judge verdicts are added by --stage judge into judge_verdicts.jsonl).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import typed_common as T  # noqa: E402

from otter.evaluation.mcq import MCQItem  # noqa: E402
from otter.evaluation.runner import base_meta, run_system  # noqa: E402
from otter.evaluation.metrics import load_answers  # noqa: E402
from otter.evaluation.text import sha1  # noqa: E402

log = logging.getLogger("typed.fill")


# ------------------------------------------------------------------------------------------------ otter_v2 free-text shim
def make_v2_free_text(system):
    from otter.agents.runtime import parse_json_loose
    from otter.agents.v2 import OtterV2

    class _Rec:
        def __init__(self, llm):
            self.llm, self.last = llm, None

        def chat(self, *a, **k):
            r = self.llm.chat(*a, **k)
            self.last = r.text
            return r

    class V2FreeText(OtterV2):
        _tl = threading.local()

        def _generate(self, llm, q, spans, hints):
            rec = _Rec(llm)
            g = super()._generate(rec, q, spans, hints)
            ans = ""
            try:
                obj = parse_json_loose(rec.last or "")
                if isinstance(obj, dict):
                    ans = str(obj.get("answer") or "").strip()
            except Exception:
                pass
            g["free_answer"] = ans
            return g

        def _grounded(self, q, g, level, ver):
            keep, grounding = super()._grounded(q, g, level, ver)
            self._tl.gens.append((g, grounding))
            return keep, grounding

        def _answer(self, q, llm, vllm):
            self._tl.gens = []
            fa = super()._answer(q, llm, vllm)
            gens = self._tl.gens
            if gens and fa.status != "failed":
                chosen = gens[1] if (len(gens) > 1 and gens[1][1] >= gens[0][1]) else gens[0]
                if fa.tool_usage.get("v2_grounding") != chosen[1]:        # replication of the frozen choice rule disagrees
                    fa.tool_usage["shim_choice_mismatch"] = True
                fa.answer = chosen[0].get("free_answer") or ""
                fa.tool_usage["shim_free_text_answer"] = True
            return fa

    system.__class__ = V2FreeText
    return system


def build_system(name: str, backbone: str):
    from otter.evaluation.runner import resolve_system
    if name == "otter_v2":
        s = resolve_system("otter_v2", backbone=backbone, flags={}, bench_index={}, corpus=None)
        return make_v2_free_text(s)
    return resolve_system(name, backbone=backbone, flags={}, bench_index={}, corpus=None)


# ------------------------------------------------------------------------------------------------ data
def load_fill(bench: Path, limit: int | None = None):
    rows = [json.loads(l) for l in open(bench, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["format"] == "fill_blank"]
    if limit:
        rows = rows[:limit]
    items = [MCQItem(id=r["id"], question=T.fill_question(r["question"]), options={}, answer=None, subject=r["subject"],
                     domain=r["domain"], grade=r["grade"]) for r in rows]
    return rows, items


def llm_json(url: str, model: str, system: str, user: str, max_tokens: int = 200) -> str:
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["direct", "rag_hybrid", "ircot", "otter_v2"])
    ap.add_argument("--backbone", default="main", choices=["main", "small"])
    ap.add_argument("--bench", default=str(ROOT / "benchmarks/typed/typed600.jsonl"))
    ap.add_argument("--exp", default="vieduhop_typed")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stage", default="run", choices=["run", "score", "judge"])
    ap.add_argument("--judge-url", default="http://127.0.0.1:8005/v1")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.workers > 3:
        raise SystemExit("cap: at most 3 concurrent LLM requests")
    rows, items = load_fill(Path(a.bench), a.limit)
    out = Path(a.out) if a.out else ROOT / "runs" / a.exp / a.system / a.backbone / "fill_blank"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    os.environ["OTTER_RUN_TAG"] = f"{a.exp}/{a.system}/{a.backbone}/fill_blank"
    if a.stage == "run":
        meta = base_meta(a.system, a.backbone, Path(a.bench), {})
        try:
            from otter.evaluation.run_provenance import code_fingerprint
            meta.update(code_fingerprint())
        except Exception as e:
            log.warning("code fingerprint failed: %s", e)
        meta.update(system=a.system, exp=a.exp, runner="scripts/eval/typed_fill_runner.py", fill_instruction=T.FILL_INSTRUCTION,
                    item_ids=[it.id for it in items], otter_llm_url_main=os.environ.get("OTTER_LLM_URL_MAIN"))
        system = build_system(a.system, a.backbone)
        run_system(system, items, out, workers=a.workers, meta=meta)
    if a.stage in ("run", "score"):
        write_records(out, rows, a.system)
    if a.stage == "judge":
        run_judge(out, rows, a.judge_url, a.judge_model, a.workers)
        write_records(out, rows, a.system)
    return 0


def write_records(out: Path, rows: list[dict], system: str) -> None:
    from otter.evaluation.corpus import SpanCorpus, core_book_filter
    from otter.evaluation.source_location import score as loc_score
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    ans = load_answers(out / "answers.jsonl")
    judge = {}
    jp = out / "judge_verdicts.jsonl"
    if jp.exists():
        for l in open(jp, encoding="utf-8"):
            if l.strip():
                j = json.loads(l)
                judge[j["id"]] = j
    recs = []
    for r in rows:
        fa = ans.get(r["id"])
        rec = {"id": r["id"], "group_id": r["group_id"], "subject": r["subject"], "grade": r["grade"], "format": "fill_blank",
               "source_file": r["source_file"], "negated": False, "alignment_status": r["alignment_status"],
               "aligned": bool(r["gold_span_ids"]), "gold": r["answer"], "missing": fa is None}
        if fa is None:
            rec.update(pred=None, failed=True, str_match=False, str_contains=False)
        else:
            pred = T.clean_prediction(fa.answer)
            tu = fa.tool_usage or {}
            rec.update(pred=pred, raw_answer=fa.answer, failed=fa.status == "failed" or fa.error is not None, status=fa.status,
                       str_match=T.string_match(pred, r["answer"], r.get("accepted_variants")),
                       str_contains=T.string_contains(pred, r["answer"], r.get("accepted_variants")),
                       n_citations=sum(len(c.citations) for c in fa.claims),
                       cost_llm_calls=tu.get("llm_calls"), cost_prompt_tokens=tu.get("prompt_tokens"),
                       cost_completion_tokens=tu.get("completion_tokens"), cost_latency_ms=tu.get("latency_ms") or tu.get("wall_latency_ms"),
                       shim_choice_mismatch=tu.get("shim_choice_mismatch"))
            if r["gold_span_ids"]:
                rec.update(loc_score(r["gold_span_ids"], fa, corpus))
        j = judge.get(r["id"])
        if j:
            rec["judge_correct"] = j["verdict"] == "correct"
            rec["judge_reason"] = j.get("reason")
        recs.append(rec)
    with open(out / "fill_records.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False, default=str) + "\n")
    n = len(recs)
    log.info("%s: n=%d string_match=%.3f contains=%.3f judge=%s", system, n, sum(x["str_match"] for x in recs) / n,
             sum(x["str_contains"] for x in recs) / n,
             (sum(x.get("judge_correct", False) for x in recs) / n) if judge else "n/a")


def run_judge(out: Path, rows: list[dict], url: str, model: str, workers: int) -> None:
    import concurrent.futures as cf
    from otter.agents.runtime import parse_json_loose
    ans = load_answers(out / "answers.jsonl")
    jp = out / "judge_verdicts.jsonl"
    done = set()
    if jp.exists():
        done = {j["id"] for j in (json.loads(l) for l in open(jp, encoding="utf-8") if l.strip()) if j["verdict"] != "error"}
    todo = [r for r in rows if r["id"] in ans and r["id"] not in done]
    lock = threading.Lock()

    def one(r):
        fa = ans[r["id"]]
        pred = T.clean_prediction(fa.answer)
        if not pred:
            v = {"verdict": "incorrect", "reason": "empty answer", "judge_called": False}
        else:
            try:
                txt = llm_json(url, model, T.JUDGE_SYSTEM, T.judge_user(r["question"], r["answer"], r.get("accepted_variants") or [], pred))
                o = parse_json_loose(txt)
                verdict = str(o.get("verdict", "")).strip().lower()
                v = {"verdict": "correct" if verdict == "correct" else "incorrect", "reason": str(o.get("reason", ""))[:200],
                     "judge_called": True, "raw": txt[:300] if verdict not in ("correct", "incorrect") else None}
            except Exception as e:
                v = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"[:200], "judge_called": True}
        v.update(id=r["id"], prediction=pred, judge_prompt_version=T.JUDGE_PROMPT_VERSION, judge_model=model)
        with lock, open(jp, "a", encoding="utf-8") as f:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    raise SystemExit(main())
