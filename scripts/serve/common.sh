#!/usr/bin/env bash
# Shared helpers for OTTER vLLM launchers (WS1 owns these). Usage: source common.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VL_PY=${VL_PY:-python}
# vl ships transformers 5.5 which breaks vLLM 0.11.2 tokenizer init; overlay transformers 4.57.3 + hf_hub 0.36
# (installed with pip --target, outside the vl env) via PYTHONPATH.
export PYTHONPATH=${OTTER_ENVS:-$HOME/envs}/otter-vl-overlay${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$ROOT/runs/logs" "$ROOT/runs/pids"
launch() {  # launch <name> <gpus> <args...>
  local name=$1 gpus=$2; shift 2
  local pidf="$ROOT/runs/pids/$name.pid"
  if [[ -f $pidf ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$pidf"))"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpus VLLM_WORKER_MULTIPROC_METHOD=spawn \
    nohup "$VL_PY" -m vllm.entrypoints.openai.api_server "$@" > "$ROOT/runs/logs/$name.log" 2>&1 &
  echo $! > "$pidf"; echo "started $name pid $! on GPU $gpus -> runs/logs/$name.log"
}
