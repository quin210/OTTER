#!/usr/bin/env bash
# Phase B: main backbone on GPUs 0,1 (TP=2) -> :8004. Checkpoint: official Qwen/Qwen3-32B-FP8 (block-wise FP8 weights),
# served under the name "Qwen/Qwen3-32B" so configs stay unchanged (coordinator decision 26 Oct ~00:55).
source "$(dirname "$0")/common.sh"
launch qwen3_32b_8004 "${GPUS:-0,1}" --model Qwen/Qwen3-32B-FP8 --served-model-name Qwen/Qwen3-32B \
  --host 127.0.0.1 --port 8004 --tensor-parallel-size 2 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --max-num-seqs 64 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --disable-log-requests --seed 0
