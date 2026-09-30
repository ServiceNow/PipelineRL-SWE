#!/usr/bin/env bash
# Frozen-feature head comparison: LCB and Omni, no new generations.
# Commit and push first; SNAPSHOT=1 uses that exact code revision.
set -euo pipefail
MLP_JOB_NAME=${MLP_JOB_NAME:-mlp_heads_$(date -u +%Y%m%d_%H%M%S)}
MLP_OUT=${MLP_OUT:-/mnt/llmd/results/exps/aristides/reason/${MLP_JOB_NAME}}
make job JOB_NAME="${MLP_JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="python -u analysis/cost_headroom/mlp_heads.py --out ${MLP_OUT}"
echo "Submitted ${MLP_JOB_NAME}; results: ${MLP_OUT}"
