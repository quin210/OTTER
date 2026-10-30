"""LLM-as-a-Judge (rubric judge_v2, docs/LLM_JUDGE.md) for the final answers of run directories. Resumable.

  # one experiment, both local judges (run the two judges as two processes in parallel: they live on different servers)
  .venv/bin/python scripts/eval/llm_judge_answers.py --exp core_dev --backbone main --judge qwen3-32b --workers 8
  .venv/bin/python scripts/eval/llm_judge_answers.py --exp core_dev --backbone main --judge qwen3-8b  --workers 8
  # explicit directories (e.g. external graph-RAG answers); bench is read from run_meta.json or given with --bench
  .venv/bin/python scripts/eval/llm_judge_answers.py --run-dir runs/otter_hard_v2/hipporag/main --bench benchmarks/otter_hard/test.jsonl
  # print one prompt, no model call
  .venv/bin/python scripts/eval/llm_judge_answers.py --run-dir runs/core_dev/otter_full/main --dry-run

Output per run dir: answer_judgments.<judge>.jsonl (+ answer_judgments.<judge>.meta.json). Raw judge calls are logged by
otter.llm to runs/llm_calls/judge_v2__<exp>__<system>__<backbone>__<judge>.<pid>.jsonl. No guided decoding (the schema is
in the prompt and the reply is parsed loosely; one retry). :8004 is shared: keep --workers <= 8.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["OTTER_GUIDED_JSON"] = "0"                    # schema in the prompt, loose parsing

from otter.evaluation.answer_judge import (JUDGE_VERSION, JUDGES, build_prompt, judge_flags, judge_run,  # noqa: E402
                                           load_judgments, summarize_judgments)
from otter.evaluation.metrics import load_answers  # noqa: E402
from otter.llm.interface import get_llm  # noqa: E402

log = logging.getLogger("otter.judge")
BENCH_BY_EXP = {"otter_hard_v2": "benchmarks/otter_hard/test.jsonl", "OTTER_MCQS_300": "tests/textbooks/OTTER_MCQS_300.jsonl",
                "abstain": "benchmarks/abstain/test.jsonl"}


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            rows[r["id"]] = r
    return rows


def resolve_bench(run: Path, meta: dict, override: str | None) -> Path:
    cands = [override, meta.get("bench"), BENCH_BY_EXP.get(run.parent.parent.name)]
    for c in cands:
        if c:
            p = Path(c)
            p = p if p.is_absolute() else ROOT / p
            if p.exists():
                return p
    raise SystemExit(f"{run}: cannot determine the benchmark file (pass --bench)")


def run_dirs(a) -> list[Path]:
    dirs = [Path(d) for d in a.run_dir]
    for e in a.exp:
        ep = Path(e) if Path(e).exists() else ROOT / "runs" / e
        for bb in a.backbone:
            dirs += sorted(d for d in ep.glob(f"*/{bb}") if (d / "answers.jsonl").exists())
    if a.systems:
        keep = set(a.systems.split(","))
        dirs = [d for d in dirs if d.parent.name in keep or d.parent.name.split("+")[0] in keep]
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", default=[], help="experiment under runs/ (repeatable)")
    ap.add_argument("--backbone", action="append", default=None, help="main|small (repeatable; default main)")
    ap.add_argument("--run-dir", action="append", default=[])
    ap.add_argument("--systems", default=None, help="comma-separated system names to keep")
    ap.add_argument("--bench", default=None, help="override the benchmark file")
    ap.add_argument("--judge", default="qwen3-32b", choices=list(JUDGES) + ["both"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="judge at most N new items per run dir (smoke tests)")
    ap.add_argument("--pairs", default=None, help="calibration sample.jsonl: judge only the listed (exp, system, qid) pairs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.backbone = a.backbone or ["main"]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dirs = run_dirs(a)
    if not dirs:
        raise SystemExit("no run directories found")
    from otter.evaluation.corpus import SpanCorpus
    corpus = SpanCorpus.load(book_filter=None)               # all 41 books: a cited span outside v1-core must still resolve
    log.info("corpus: %d spans; %d run dirs", len(corpus), len(dirs))
    judges = list(JUDGES) if a.judge == "both" else [a.judge]
    pairs = {}
    if a.pairs:
        for l in open(a.pairs):
            r = json.loads(l)
            pairs.setdefault((r["exp"], r["system"]), set()).add(r["qid"])
    for jid in judges:
        llm = get_llm(JUDGES[jid]["role"])
        for d in dirs:
            meta = json.loads((d / "run_meta.json").read_text()) if (d / "run_meta.json").exists() else {}
            try:
                bench = load_rows(resolve_bench(d, meta, a.bench))
            except SystemExit as e:
                log.warning("skip %s: %s", d, e)
                continue
            answers = load_answers(d / "answers.jsonl")
            ids = meta.get("item_ids") or [q for q in answers if q in bench]
            items = [bench[i] for i in ids if i in bench]
            if a.pairs:
                want = pairs.get((meta.get("exp") or d.parent.parent.name, d.parent.name), set())
                items = [it for it in items if it["id"] in want]
                if not items:
                    continue
            sysname = meta.get("system") or d.parent.name
            bbm = meta.get("backbone_model")
            if bbm is None:
                from otter.llm.interface import load_config
                bbm = load_config()["llm"].get("small" if d.name.startswith("small") else "main", {}).get("model")
            if a.dry_run:
                it = next((x for x in items if x["id"] in answers), None)
                print(build_prompt(it, answers[it["id"]], corpus)[0][0]["content"])
                return 0
            os.environ["OTTER_RUN_TAG"] = f"{JUDGE_VERSION}/{meta.get('exp') or d.parent.parent.name}/{sysname}/{d.name}/{jid}"
            out = d / f"answer_judgments.{jid}.jsonl"
            t0 = dt.datetime.now()
            st = judge_run(items, answers, corpus, llm, out, judge_id=jid, workers=a.workers, limit=a.limit)
            summ = summarize_judgments(load_judgments(out), [it["id"] for it in items], n_boot=2000)
            fl = judge_flags(sysname, bbm, jid)
            (d / f"answer_judgments.{jid}.meta.json").write_text(json.dumps({
                "judge_version": JUDGE_VERSION, "judge": jid, "judge_role": JUDGES[jid]["role"], "judge_model_served": llm.model,
                "judge_checkpoint": {"qwen3-32b": "Qwen/Qwen3-32B-FP8", "qwen3-8b": "Qwen/Qwen3-8B"}[jid],
                "temperature": 0.0, "thinking": "off", "guided_decoding": False, "system_backbone_model": bbm, **fl,
                "n_items": len(items), "stats_last_call": st, "summary": summ,
                "judged_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}, indent=1, ensure_ascii=False), encoding="utf-8")
            log.info("%s [%s]: %s | judge=%.3f strict=%.3f n=%d/%d %s (%.0fs)", d, jid, st, summ.get("judge", float("nan")),
                     summ.get("strict", float("nan")), summ.get("n_judged", 0), summ.get("n_items", 0),
                     "SELF-JUDGE" if fl["self_judge"] else "", (dt.datetime.now() - t0).total_seconds())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
