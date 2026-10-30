#!/usr/bin/env bash
# Stop ONLY the Phase-A Qwen3-VL OCR servers (pids from runs/pids/ocr_vl_gpu{0,1,2}.pid).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for g in 0 1 2; do f="$ROOT/runs/pids/ocr_vl_gpu$g.pid"; [[ -f $f ]] || continue
  p=$(cat "$f"); if kill -0 "$p" 2>/dev/null; then kill "$p" && echo "stopped ocr_vl_gpu$g ($p)"; fi; done
