#!/usr/bin/env bash
# GPU3: Qwen3-8B dev/small backbone on :8001 (shares GPU3 with bge-m3 :8002).
source "$(dirname "$0")/common.sh"
launch qwen3_8b_8001 "${GPU:-3}" --model Qwen/Qwen3-8B --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.86 \
  --max-num-seqs 48 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-prefix-caching --disable-log-requests
