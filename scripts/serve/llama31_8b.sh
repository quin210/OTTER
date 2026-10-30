#!/usr/bin/env bash
# Phase B: meta-llama/Llama-3.1-8B-Instruct cross-family backbone on GPU2 -> :8003.
source "$(dirname "$0")/common.sh"
launch llama31_8b_8003 "${GPU:-2}" --model meta-llama/Llama-3.1-8B-Instruct --served-model-name meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 8003 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --max-num-seqs 128 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser llama3_json \
  --disable-log-requests --seed 0
