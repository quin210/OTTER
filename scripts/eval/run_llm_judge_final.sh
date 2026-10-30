#!/usr/bin/env bash
# One command for the FINAL LLM-as-a-Judge pass (rubric judge_v2, docs/LLM_JUDGE.md) over finished experiment directories.
#   scripts/eval/run_llm_judge_final.sh EXP[,EXP2,...] ["main small"] [extra run dirs ...]
# e.g. after the freeze:
#   scripts/eval/run_llm_judge_final.sh hard_v2_reviewed,mcq300_v2,abstain_v2 "main small" runs/otter_hard_v2/hipporag/main
# Both local judges run in parallel (Qwen3-32B-FP8 on :8004 and Qwen3-8B on :8001, 8 workers each; :8004 is shared, keep it at 8).
# Resumable (cache key = judge_version + item + prompt hash). Writes <run>/answer_judgments.<judge>.jsonl, then rebuilds the tables.
# Raw judge calls are logged to runs/llm_calls/judge_v2__*. SKIP_TABLES=1 skips the table rebuild. JUDGE_WORKERS overrides 8.
set -u
cd "$(dirname "$0")/../.."
EXPS=${1:?experiments (comma separated, dirs under runs/)}
BBS=${2:-"main small"}
shift 2 2>/dev/null || shift $#
EXTRA=("$@")
PY=.venv/bin/python
W=${JUDGE_WORKERS:-8}
ARGS=(); IFS=, read -ra EL <<< "$EXPS"
for e in "${EL[@]}"; do ARGS+=(--exp "$e"); done
for b in $BBS; do ARGS+=(--backbone "$b"); done
for d in "${EXTRA[@]}"; do ARGS+=(--run-dir "$d"); done
mkdir -p runs/logs
for J in qwen3-32b qwen3-8b; do
  $PY scripts/eval/llm_judge_answers.py "${ARGS[@]}" --judge $J --workers $W > runs/logs/judge_v2_final_$J.log 2>&1 &
done
wait
grep -h "judge=" runs/logs/judge_v2_final_qwen3-*.log | sed 's/^.*INFO //' || true
[ "${SKIP_TABLES:-0}" = 1 ] && exit 0
for e in "${EL[@]}"; do
  $PY scripts/eval/aggregate.py --exp "$e" --ref otter_full || true
  [ -f "runs/$e/results.md" ] || true
done
$PY scripts/eval/abstain_summary.py "${EL[@]}" 2>/dev/null | tail -n +1 || true
