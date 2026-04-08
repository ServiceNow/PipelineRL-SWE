#!/usr/bin/env bash
set -euo pipefail

EXPERT_BASE_URL=${EXPERT_BASE_URL:-http://localhost:8380}
MODEL_NAME=${MODEL_NAME:-openai/gpt-oss-120b}
REASONING_LEVELS=${REASONING_LEVELS:-"low medium high"}
SUBSAMPLE=${SUBSAMPLE:-32}
TEST_MAX_SAMPLES=${TEST_MAX_SAMPLES:-500}

for EFFORT in ${REASONING_LEVELS}; do
  TIMESTAMP=$(date +%s)
  JOB_NAME=gpt_oss_reasoning_smoke_${EFFORT}
  OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP}

  make job \
    JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
    ENV=pipeline-rl \
    CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT=1 \
    NPROC=1 \
    COMMAND="cd /home/toolkit/PipelineRL-SWE; mkdir -p ${OUTPUT_DIR}; set -o pipefail; { python -m pipelinerl.swe.scripts.run_expert_repair_eval --config-dir conf --config-name swe \
      output_dir=${OUTPUT_DIR} \
      wandb.use_wandb=false \
      expert_eval.base_url=${EXPERT_BASE_URL} \
      expert_eval.model_name=${MODEL_NAME} \
      expert_eval.parameters.max_tokens=15000 \
      expert_eval.parameters.temperature=0.7 \
      expert_eval.parameters.reasoning_effort=${EFFORT} \
      expert_eval.subsample=${SUBSAMPLE} \
      dataset_loader_params.test_max_samples=${TEST_MAX_SAMPLES}; } 2>&1 | tee -a ${OUTPUT_DIR}/launch.out"
done
