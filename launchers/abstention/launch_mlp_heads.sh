#!/usr/bin/env bash
# Frozen-feature head comparison: LCB and Omni, no new generations.
# Commit and push first; SNAPSHOT=1 uses that exact code revision.
set -euo pipefail
MLP_JOB_NAME=${MLP_JOB_NAME:-mlp_heads_$(date -u +%Y%m%d_%H%M%S)}
MLP_OUT=${MLP_OUT:-/mnt/llmd/results/exps/aristides/reason/${MLP_JOB_NAME}}
for MLP_POOL in LCB Omni; do
  for MLP_TASK in success cost; do
    for MLP_CONFIG in 0 1; do
      for MLP_SEED in 0 1 2; do
        make job JOB_NAME="${MLP_JOB_NAME}_${MLP_POOL}_${MLP_TASK}_${MLP_CONFIG}_${MLP_SEED}" \
          ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
          GPU=1 GPU_MEM=16 CPU=8 CPU_MEM=32 SNAPSHOT=1 \
          COMMAND="python -u analysis/cost_headroom/mlp_heads.py --out ${MLP_OUT} --pool ${MLP_POOL} --task ${MLP_TASK} --config-index ${MLP_CONFIG} --seed ${MLP_SEED}"
        sleep 15
      done
    done
  done
done
make job JOB_NAME="${MLP_JOB_NAME}_aggregate" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  GPU=0 CPU=8 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="python -u analysis/cost_headroom/mlp_heads.py --out ${MLP_OUT} --aggregate"
echo "Submitted independent runs and aggregation; results: ${MLP_OUT}"
