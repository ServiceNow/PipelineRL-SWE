#!/usr/bin/env bash
# NEW_PATH 4.A.59: one CPU eai job per pool (analysis/cost_headroom/pinned_rerun.sh). Usage: SUBMIT=1 bash ... lcb bcb cc apps omni mmlupro aime fresh
set -euo pipefail
for P in "$@"; do
  if [[ ${SUBMIT:-0} != 1 ]]; then echo "dry run: bash analysis/cost_headroom/pinned_rerun.sh $P"; continue; fi
  make job JOB_NAME="pinrerun_${P}_$(date -u +%H%M%S)" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=96 SNAPSHOT=1 \
    COMMAND="bash analysis/cost_headroom/pinned_rerun.sh $P" 2>&1 | grep -E "^[0-9a-f]{8}-|rror" | cut -c1-80 || true
  sleep 35
done
