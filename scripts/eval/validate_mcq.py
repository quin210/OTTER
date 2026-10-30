"""Validate an MCQ benchmark file and write a Markdown + JSON report.

.venv/bin/python scripts/eval/validate_mcq.py --bench tests/textbooks/OTTER_MCQS_300.jsonl \
    --out benchmarks/reports/mcq300_validation.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from otter.evaluation.mcq import load_mcq, render_validation_md, validate_mcq  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="tests/textbooks/OTTER_MCQS_300.jsonl")
    ap.add_argument("--out", default="benchmarks/reports/mcq300_validation.md")
    ap.add_argument("--title", default="OTTER_MCQS_300 validation report")
    ap.add_argument("--near-dup", type=float, default=0.85)
    a = ap.parse_args()
    items = load_mcq(a.bench)
    rep = validate_mcq(items, near_dup_threshold=a.near_dup)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_validation_md(rep, title=a.title, source=a.bench), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{rep['n_items']} items, {rep['n_errors']} errors, {len(rep['near_duplicates'])} near-dup pairs -> {out}")
    return 1 if rep["n_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
