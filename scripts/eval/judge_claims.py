"""Claim-support judging for a run directory + human audit sheet.

  .venv/bin/python scripts/eval/judge_claims.py --run-dir runs/hard/otter_full/main --judge-role small --audit 60
  .venv/bin/python scripts/eval/judge_claims.py --run-dir ... --human annotation/claim_audit/<name>_filled.csv
The judge role must differ from the run's backbone (pass --allow-same-model to override; then disclose).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from otter.evaluation.corpus import SpanCorpus, core_book_filter  # noqa: E402
from otter.evaluation.judge import agreement, audit_sample, judge_run, load_judgments  # noqa: E402
from otter.evaluation.metrics import load_answers  # noqa: E402
from otter.evaluation.sheets import read_filled_csv, span_block, write_sheet  # noqa: E402
from otter.llm.interface import get_llm, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--judge-role", default=None, choices=["main", "small"],
                    help="default: the other backbone (main-backbone runs -> small judge, small-backbone runs -> main)")
    ap.add_argument("--allow-same-model", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--audit", type=int, default=0, help="write a human audit sheet with N claims")
    ap.add_argument("--human", default=None, help="filled audit CSV -> agreement stats")
    a = ap.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "run_meta.json").read_text())
    cfg = load_config()["llm"]
    role = a.judge_role or ("small" if meta.get("backbone") == "main" else "main")
    jm, bm = cfg[role]["model"], meta.get("backbone_model")
    if jm == bm and not a.allow_same_model:
        raise SystemExit(f"judge model {jm} equals system backbone; choose another --judge-role")
    corpus = SpanCorpus.load(book_filter=core_book_filter)
    out = run / f"claim_judgments.{role}.jsonl"
    if not a.human:
        from otter.evaluation.judge import JUDGE_PROMPT_VERSION
        import datetime as _dt
        CKPT = {"main": "Qwen/Qwen3-32B-FP8 (served as Qwen/Qwen3-32B)", "small": "Qwen/Qwen3-8B"}
        sysn = str(meta.get("system", ""))
        (run / f"claim_judgments.{role}.meta.json").write_text(json.dumps({
            "judge_role": role, "judge_model_served": jm, "judge_checkpoint": CKPT.get(role), "judge_base_url": cfg[role].get("base_url"),
            "prompt_version": JUDGE_PROMPT_VERSION, "temperature": 0.0, "thinking": "off", "system_backbone": bm,
            "independent_of_generator": jm != bm,
            "independent_of_v2_verifier": not (sysn.startswith("otter_v2") and role == "small"),     # v2 verifier = role small (Qwen3-8B)
            "judged_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
        judge_run(load_answers(run / "answers.jsonl"), corpus, get_llm(role), out, workers=a.workers)
        print(f"judgments -> {out}")
    j = load_judgments(out)
    if a.audit:
        rows = []
        for r in audit_sample(j, a.audit):
            rid = f"{r['question_id']}::{r['claim_id']}"
            rows.append({"id": rid, "question_id": r["question_id"], "claim_id": r["claim_id"], "claim": r["claim"],
                         "span_ids": r["span_ids"], "judge_verdict_hidden": "",
                         "__blocks": [{"kind": "text", "label": "Claim", "value": r["claim"]},
                                      span_block("Cited spans", [corpus.get(s) for s in r["span_ids"]])]})
        name = f"{meta.get('exp')}_{meta.get('system')}_{meta.get('backbone')}".replace("/", "_")
        fields = [{"name": "human_verdict", "label": "Support", "type": "choice", "options": ["supported", "partial", "unsupported"]},
                  {"name": "notes", "label": "Notes", "type": "text", "required": False}]
        csv_p, html_p = write_sheet(rows, context_cols=["id", "question_id", "claim_id", "claim", "span_ids"], fields=fields,
                                    out_dir=ROOT / "annotation/claim_audit", name=name,
                                    title=f"Claim-support audit · {name}",
                                    instructions="Judge only from the cited span texts. The LLM judge verdict is hidden "
                                                 "to avoid anchoring. supported = all key content stated or trivially implied; "
                                                 "partial = part supported; unsupported = not supported or contradicted.")
        print(f"audit sheet -> {csv_p}, {html_p}")
    if a.human:
        h = [{"question_id": r["id"].split("::")[0], "claim_id": r["id"].split("::")[1], "human_verdict": r.get("human_verdict")}
             for r in read_filled_csv(Path(a.human))]
        print(json.dumps(agreement(h, j), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
