#!/usr/bin/env bash
# GPU3: BAAI/bge-m3 dense embeddings (OpenAI /v1/embeddings) on :8002.
source "$(dirname "$0")/common.sh"
launch bge_m3_8002 "${GPU:-3}" --model BAAI/bge-m3 --served-model-name BAAI/bge-m3 \
  --host 127.0.0.1 --port 8002 --runner pooling --max-model-len 8192 --gpu-memory-utilization 0.10 \
  --max-num-seqs 128 --disable-log-requests
