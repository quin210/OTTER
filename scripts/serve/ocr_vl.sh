#!/usr/bin/env bash
# Phase A: Qwen3-VL-8B-Instruct OCR servers, one per GPU. Usage: ocr_vl.sh [gpu ...] (default 0 1 2) -> port 8100+gpu
source "$(dirname "$0")/common.sh"
for g in "${@:-0 1 2}"; do for gpu in $g; do
  launch "ocr_vl_gpu$gpu" "$gpu" --model Qwen/Qwen3-VL-8B-Instruct --served-model-name Qwen/Qwen3-VL-8B-Instruct \
    --host 127.0.0.1 --port $((8100+gpu)) --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 --limit-mm-per-prompt '{"image":1,"video":0}' --disable-log-requests --seed 0
done; done
