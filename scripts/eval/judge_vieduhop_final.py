"""FINAL LLM-as-a-Judge pass (rubric judge_v2, docs/LLM_JUDGE.md) for the ViEduHop final runs. Idempotent and resumable.

  .venv/bin/python scripts/eval/judge_vieduhop_final.py --exp vieduhop_unrev --judge qwen3-32b --url http://127.0.0.1:8005/v1 --workers 4
  .venv/bin/python scripts/eval/judge_vieduhop_final.py --exp vieduhop_abstain_unrev --judge qwen3-8b --workers 2

What it does differently from llm_judge_answers.py (which it reuses: same prompts, rules, cache key, summaries):
  * only COMPLETE runs are judged (every benchmark item has an answer row; --allow-incomplete overrides, never for the final numbers);
  * items flagged seen_in_development are excluded (they are never judged, so they cannot enter any number);
  * a fixed item order (SHA-1 of the item id, independent of system) is processed in stages of --stage items across ALL complete
    systems, so that at any moment every system has judged the same prefix of items (a random subset of the benchmark);
    --max-items caps the prefix;
  * output: <run>/answer_judgments.<judge>.jsonl  (per-item verdicts: scores, raw scores, rules, reason, prompt_sha1, judge_version,
    judge_model) and <run>/answer_judgments.<judge>.meta.json (judge version, prompt template sha1, served model, endpoint, judged ids).
    answers.jsonl is never touched. Re-running skips everything already judged (cache key = judge_version + id + prompt sha1).
  * --url routes the role to another replica of the same model (OTTER_LLM_URL_<ROLE>), e.g. the 2nd Qwen3-32B-FP8 on :8005.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["OTTER_GUIDED_JSON"] = "0"

import otter.evaluation.answer_judge as aj  # noqa: E402
from otter.evaluation.answer_judge import JUDGE_VERSION, JUDGES, judge_flags, judge_run, load_judgments  # noqa: E402
from otter.evaluation.metrics import load_answers  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402

log = logging.getLogger("otter.judge.final")
BENCH = {"vieduhop_unrev": "benchmarks/otter_hard/test.jsonl", "vieduhop_abstain_unrev": "benchmarks/abstain/test.jsonl"}


def template_sha1() -> str:
    return hashlib.sha1((aj.PROMPT_TEMPLATE + aj.SPECIAL_UNANSWERABLE + aj.SPECIAL_ANSWERABLE_ABSTAIN).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--backbone", default="main")
    ap.add_argument("--bench", default=None)
    ap.add_argument("--judge", default="qwen3-32b", choices=list(JUDGES))
    ap.add_argument("--url", default=None, help="override the endpoint of the judge role (same model replica)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--systems", default=None, help="comma-separated run directory names (default: all complete ones)")
    ap.add_argument("--stage", type=int, default=200, help="items per stage, processed across all systems before the next stage")
    ap.add_argument("--max-items", type=int, default=None, help="cap on the item prefix (smoke tests / time budget)")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--status", action="store_true", help="only print which runs are complete / how many judged")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    role = JUDGES[a.judge]["role"]
    if a.url:
        os.environ[f"OTTER_LLM_URL_{role.upper()}"] = a.url
    bench_path = ROOT / (a.bench or BENCH[a.exp])
    bench = {}
    for l in open(bench_path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            bench[r["id"]] = r
    unseen = sorted((i for i, r in bench.items() if not r.get("seen_in_development")),
                    key=lambda i: hashlib.sha1(("judge_final|" + i).encode()).hexdigest())
    if a.max_items:
        unseen = unseen[: a.max_items]
    log.info("bench %s: %d items, %d judged-eligible (seen_in_development excluded)", bench_path.name, len(bench), len(unseen))
    dirs = sorted((ROOT / "runs" / a.exp).glob(f"*/{a.backbone}"))
    if a.systems:
        keep = set(a.systems.split(","))
        dirs = [d for d in dirs if d.parent.name in keep]
    ready = []
    for d in dirs:
        if not (d / "answers.jsonl").exists():
            continue
        answers = load_answers(d / "answers.jsonl")
        missing = [i for i in bench if i not in answers]
        done = len(load_judgments(d / f"answer_judgments.{a.judge}.jsonl"))
        if missing and not a.allow_incomplete:
            log.info("SKIP incomplete %-45s answers=%d/%d", d.parent.name, len(answers), len(bench))
            continue
        log.info("ready      %-45s answers=%d/%d judged_so_far=%d", d.parent.name, len(answers), len(bench), done)
        ready.append((d, answers))
    if a.status or not ready:
        return 0
    corpus_mod = __import__("otter.evaluation.corpus", fromlist=["SpanCorpus"])
    corpus = corpus_mod.SpanCorpus.load(book_filter=None)
    llm = get_llm(role)
    log.info("judge %s model=%s url=%s workers=%d", a.judge, llm.model, os.environ.get(f"OTTER_LLM_URL_{role.upper()}") or load_config()["llm"][role]["base_url"], a.workers)
    stages = list(range(a.stage, len(unseen), a.stage)) + [len(unseen)]
    for n in stages:
        ids = unseen[:n]
        items = [bench[i] for i in ids]
        for d, answers in ready:
            meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
            sysname = meta.get("system") or d.parent.name
            bbm = meta.get("backbone_model")
            os.environ["OTTER_RUN_TAG"] = f"{JUDGE_VERSION}/{a.exp}/{d.parent.name}/{d.name}/{a.judge}"
            out = d / f"answer_judgments.{a.judge}.jsonl"
            t0 = dt.datetime.now()
            st = judge_run(items, answers, corpus, llm, out, judge_id=a.judge, workers=a.workers)
            rows = load_judgments(out)
            summ = aj.summarize_judgments(rows, ids, n_boot=2000)
            fl = judge_flags(sysname, bbm, a.judge)
            (d / f"answer_judgments.{a.judge}.meta.json").write_text(json.dumps({
                "judge_version": JUDGE_VERSION, "prompt_template_sha1": template_sha1(), "judge": a.judge, "judge_role": role,
                "judge_model_served": llm.model, "judge_endpoint": os.environ.get(f"OTTER_LLM_URL_{role.upper()}") or load_config()["llm"][role]["base_url"],
                "judge_checkpoint": {"qwen3-32b": "Qwen/Qwen3-32B-FP8", "qwen3-8b": "Qwen/Qwen3-8B"}[a.judge],
                "temperature": 0.0, "thinking": "off", "guided_decoding": False, "system": sysname, "system_backbone_model": bbm, **fl,
                "bench": str(bench_path.relative_to(ROOT)), "seen_in_development_excluded": True,
                "item_order": "sha1('judge_final|'+id) ascending; judged prefix = first n_prefix ids of the non-seen items",
                "n_prefix": n, "n_eligible": len(unseen), "stats_last_call": st, "summary": summ,
                "judged_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}, indent=1, ensure_ascii=False), encoding="utf-8")
            log.info("[%s n=%d] %-40s %s | judge=%.3f strict=%.3f judged=%d/%d (%.0fs)", a.judge, n, d.parent.name, st,
                     summ.get("judge", float("nan")), summ.get("strict", float("nan")), summ.get("n_judged", 0), summ.get("n_items", 0),
                     (dt.datetime.now() - t0).total_seconds())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
