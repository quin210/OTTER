#!/usr/bin/env bash
# Phase B: second Qwen3-32B-FP8 replica, single GPU (TP=1) on GPU2 -> :8005, served as "Qwen/Qwen3-32B".
# Dedicated to WS4 baseline graph indexing (config role `index`), so WS4 doesn't starve `main` (:8004).
source "$(dirname "$0")/common.sh"
launch qwen3_32b_index_8005 "${GPU:-2}" --model Qwen/Qwen3-32B-FP8 --served-model-name Qwen/Qwen3-32B \
  --host 127.0.0.1 --port 8005 --tensor-parallel-size 1 --max-model-len 16384 \
  --gpu-memory-utilization 0.95 --max-num-seqs 24 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --disable-log-requests --seed 0
