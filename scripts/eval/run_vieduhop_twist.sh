#!/usr/bin/env bash
# ViEduHop-Twist evaluation (27 Oct 2026). Same systems/flags as scripts/eval/run_final_unreviewed.sh; at most 4 LLM requests of this script at any time
# (one worker per process, 4 processes). Backbone = Qwen3-32B-FP8; URL per process via OTTER_LLM_URL_MAIN (default: config :8004; set URL_A/URL_B to pick replicas).
#   MCQ:     exp vieduhop_twist          benchmarks/otter_hard/twist.jsonl           direct, rag_hybrid, ircot, otter_v2
#   Abstain: exp vieduhop_twist_abstain  benchmarks/abstain/twist_abstain.jsonl      --abstain-option; direct, rag_hybrid, ircot, otter_v2, otter_v2(abstention=true abstain_below=3), otter_full
# Resumable. Method files are not touched (otter_v2 frozen: otter/agents/v2.py, configs/otter_v2.yaml).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
BENCH=benchmarks/otter_hard/twist.jsonl; AB=benchmarks/abstain/twist_abstain.jsonl
EXP=${EXP:-vieduhop_twist}; XEXP=${XEXP:-vieduhop_twist_abstain}; TAU=${TAU:-3}
URL_A=${URL_A:-http://127.0.0.1:8004/v1}; URL_B=${URL_B:-http://127.0.0.1:8005/v1}
mkdir -p runs/logs
r() { # url exp bench system workers [flags...]
  local url=$1 exp=$2 bench=$3 sys=$4 w=$5; shift 5
  local fl=(); for f in "$@"; do case "$f" in --abstain-option) fl+=("$f");; *=*) fl+=(--flag "$f");; *) fl+=(--flag "$f=true");; esac; done
  local tag="$exp-$sys-$(echo "$*" | tr ' =,' '___')"
  OTTER_LLM_URL_MAIN=$url OTTER_RUN_TAG="$tag" $PY scripts/eval/run.py --system "$sys" --backbone main --bench "$bench" --exp "$exp" --workers "$w" "${fl[@]}" \
     > "runs/logs/twist_${tag}.log" 2>&1 || echo "FAILED $sys $*" >> runs/logs/twist_failed.log
}
PHASE=${PHASE:-all}
if [ "$PHASE" = mcq ] || [ "$PHASE" = all ]; then
  ( r $URL_A $EXP $BENCH direct 1; r $URL_A $EXP $BENCH rag_hybrid 1 ) &
  ( r $URL_B $EXP $BENCH ircot 1 ) &
  ( r $URL_A $EXP $BENCH otter_v2 1 ) &
  wait
fi
if [ "$PHASE" = abstain ] || [ "$PHASE" = all ]; then
  ( r $URL_A $XEXP $AB direct 1 --abstain-option; r $URL_A $XEXP $AB rag_hybrid 1 --abstain-option; r $URL_A $XEXP $AB otter_full 1 --abstain-option ) &
  ( r $URL_B $XEXP $AB ircot 1 --abstain-option ) &
  ( r $URL_A $XEXP $AB otter_v2 1 --abstain-option ) &
  ( r $URL_B $XEXP $AB otter_v2 1 --abstain-option abstention=true abstain_below=$TAU ) &
  wait
fi
echo "ALLDONE $(date)" >> runs/logs/twist_failed.log
