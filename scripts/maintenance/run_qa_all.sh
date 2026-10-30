#!/bin/bash
# resumable: run_system skips answered items
cd "$(dirname "$0")/../.."
for arm in rag_new otter_stale_kb otter_new; do
  .venv/bin/python scripts/maintenance/run_qa.py run --arm $arm --workers 8
done
