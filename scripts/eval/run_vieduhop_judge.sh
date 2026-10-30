#!/usr/bin/env bash
# Driver for the final ViEduHop LLM-judge passes (resumable: every pass re-detects the complete runs and skips cached verdicts).
#   JUDGE=qwen3-32b URL=http://127.0.0.1:8005/v1 WORKERS=4 PLAN="hard:100 hard:200 abstain:100 hard:300 abstain:262" scripts/eval/run_vieduhop_judge.sh
# PLAN items are <hard|abstain>:<item prefix size>. Optional SYSTEMS_HARD / SYSTEMS_ABSTAIN restrict the run directories.
set -u
cd "$(dirname "$0")/../.."
J=${JUDGE:?}; W=${WORKERS:-2}; PLAN=${PLAN:?}
URL_ARG=(); [ -n "${URL:-}" ] && URL_ARG=(--url "$URL")
for p in $PLAN; do
  kind=${p%%:*}; m=${p##*:}
  exp=vieduhop_unrev; sysarg=(); [ -n "${SYSTEMS_HARD:-}" ] && sysarg=(--systems "$SYSTEMS_HARD")
  if [ "$kind" = abstain ]; then exp=vieduhop_abstain_unrev; sysarg=(); [ -n "${SYSTEMS_ABSTAIN:-}" ] && sysarg=(--systems "$SYSTEMS_ABSTAIN"); fi
  echo "$(date +%T) pass $kind max_items=$m judge=$J"
  .venv/bin/python scripts/eval/judge_vieduhop_final.py --exp "$exp" --judge "$J" "${URL_ARG[@]}" "${sysarg[@]}" --workers "$W" --stage 50 --max-items "$m" 2>&1 | grep -v "HTTP Request"
done
echo "$(date +%T) plan finished"
