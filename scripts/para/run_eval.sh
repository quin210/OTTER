#!/usr/bin/env bash
# ViEduHop-Para evaluation (post hoc). Same flags as scripts/eval/run_final_unreviewed.sh (none beyond --backbone main), one system at a time, 3 workers => <=3 concurrent LLM requests.
set -u
cd "$(dirname "$0")/../.."
BENCH=benchmarks/otter_hard/para.jsonl; EXP=vieduhop_para
for s in direct rag_hybrid ircot otter_v2; do
  OTTER_RUN_TAG="$EXP-$s-" .venv/bin/python scripts/eval/run.py --system $s --backbone main --bench $BENCH --exp $EXP --workers 3 > runs/logs/para_$s.log 2>&1 || echo "FAILED $s" >> runs/vieduhop_para/failed.log
done
echo EVALDONE >> runs/vieduhop_para/failed.log
