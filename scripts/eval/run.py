"""Run one system on one benchmark (resumable) and score it.

Examples
  .venv/bin/python scripts/eval/run.py --system eval_direct --backbone small \
      --bench tests/textbooks/OTTER_MCQS_300.jsonl --exp mcq300_dev --workers 16
  .venv/bin/python scripts/eval/run.py --system otter_full --backbone main --bench benchmarks/otter_hard/test.jsonl \
      --exp hard --flag no_workspace=true
  .venv/bin/python scripts/eval/run.py --system external:runs/ext/hipporag2/answers.jsonl --system-name hipporag2 \
      --backbone main --bench benchmarks/otter_hard/test.jsonl --exp hard
Output: runs/<exp>/<system>/<backbone>/{answers.jsonl, run_meta.json, scores.json, records.jsonl}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation.mcq import load_mcq  # noqa: E402
from otter.evaluation.metrics import evaluate, load_answers, write_json  # noqa: E402
from otter.evaluation.runner import base_meta, import_external, resolve_system, run_system, system_slug  # noqa: E402


def parse_flags(kvs: list[str]) -> dict:
    out = {}
    for kv in kvs or []:
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_corpus_if_any():
    try:
        from otter.evaluation.corpus import SpanCorpus, core_book_filter
        c = SpanCorpus.load(book_filter=None if os.environ.get("OTTER_EVAL_ALL_BOOKS") == "1" else core_book_filter)
        return c if len(c) and any(s.text for s in c.spans) else None
    except Exception as e:  # corpus is optional for scoring
        logging.getLogger("otter.eval").warning("no span corpus for citation checks: %s", e)
        return None


def score_dir(out: Path, items, corpus) -> dict:
    answers = load_answers(out / "answers.jsonl")
    judg = None
    jp = sorted(out.glob("claim_judgments*.jsonl"))
    if jp:
        from otter.evaluation.judge import load_judgments
        judg = load_judgments(jp[-1])
    res = evaluate(items, answers, corpus=corpus, judgments=judg)
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in res.pop("records"):
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    write_json(res, out / "scores.json")
    if any(not it.answerable for it in items):               # OTTER-Abstain bench: abstention metrics
        from otter.evaluation.abstain_metrics import evaluate_abstain
        ab = evaluate_abstain(items, answers)
        recs = ab.pop("records")
        with open(out / "abstain_records.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_json(ab, out / "abstain_scores.json")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--system-name", default=None, help="directory name for the system (default: derived)")
    ap.add_argument("--backbone", default="main", choices=["main", "small"])
    ap.add_argument("--bench", default=str(ROOT / "tests/textbooks/OTTER_MCQS_300.jsonl"))
    ap.add_argument("--exp", default=None, help="experiment name (default: bench file stem)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default=None, help="keep only items with this `split` value")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="file with one question id per line")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--flag", action="append", default=[], help="ablation flag k=v passed to get_system")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--abstain-option", action="store_true",
                    help="OTTER-Abstain: add the explicit fifth option E 'Không đủ căn cứ trong sách' to every question")
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args()

    items = load_mcq(a.bench)
    if a.split:
        items = [it for it in items if getattr(it, "split", None) == a.split]
    if a.ids:
        keep = {x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()}
        items = [it for it in items if it.id in keep]
    if a.limit:
        items = items[: a.limit]
    flags = parse_flags(a.flag)
    sysname = a.system_name or system_slug(a.system)
    if flags:
        sysname += "+" + ",".join(f"{k}={v}" for k, v in sorted(flags.items()))
    exp = a.exp or Path(a.bench).stem
    out = Path(a.out) if a.out else ROOT / "runs" / exp / sysname / a.backbone
    log_dir = Path(os.environ.get("OTTER_EVAL_LOG_DIR", ROOT / "runs/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(log_dir / f"eval_{exp}_{sysname}_{a.backbone}.log".replace("/", "_"))])
    os.environ["OTTER_RUN_TAG"] = f"{exp}/{sysname}/{a.backbone}"      # groups raw LLM logs under runs/llm_calls/
    meta = base_meta(a.system, a.backbone, Path(a.bench), flags)
    try:
        from otter.evaluation.run_provenance import code_fingerprint
        meta.update(code_fingerprint())
    except Exception as e:                                                # never block a run on provenance
        logging.warning("code fingerprint failed: %s", e)
    meta.update(system=sysname, exp=exp, split=a.split, limit=a.limit, item_ids=[it.id for it in items])
    if not a.system.startswith("external:") and not os.environ.get("OTTER_NO_RUNDIR_GUARD"):
        from otter.evaluation.run_provenance import guard_rundir     # never mix code versions / overwrite: old dir -> runs/<exp>_prev_<stamp>/
        g = guard_rundir(out)
        meta["method_sha256"] = g["method_sha256"]
        if g["moved_to"]:
            logging.warning("existing results were produced by a different method version; moved to %s", g["moved_to"])
    corpus = load_corpus_if_any()
    if a.system.startswith("external:"):
        meta = import_external(Path(a.system.split(":", 1)[1]), items, out, meta=meta)
    else:
        bench_index = {it.id: it for it in items}
        system = resolve_system(a.system, backbone=a.backbone, flags=flags, bench_index=bench_index, corpus=corpus)
        if any(it.excluded_span_ids for it in items):       # OTTER-Abstain: per-item hidden spans (otter/evaluation/exclusion.py)
            from otter.evaluation.exclusion import install_exclusion
            meta["exclusion_hook"] = install_exclusion(system)
        meta["abstain_option"] = a.abstain_option
        meta = run_system(system, items, out, workers=a.workers, retry_failed=a.retry_failed, meta=meta,
                          abstain_option=a.abstain_option)
    logging.info("run meta: %s", {k: meta.get(k) for k in ("n_items", "n_answers", "n_failed", "n_missing")})
    if not a.no_score:
        res = score_dir(out, items, corpus)
        o = res["overall"]
        logging.info("accuracy %.4f [%.4f, %.4f] n=%d coverage=%s failed=%d", o["accuracy"], o["acc_ci_lo"],
                     o["acc_ci_hi"], o["n"], o["coverage"], o["failed"])
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
