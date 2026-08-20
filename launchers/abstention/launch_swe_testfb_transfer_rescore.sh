#!/usr/bin/env bash
# Re-score the existing SWE-Smith-trained test-feedback checkpoints on
# SWE-bench Verified while preserving their training-time feature schema.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
NO_COT_RUN=${NO_COT_RUN:-/mnt/llmd/results/exps/aristides/reason/cot_abstention_qwen3_emb8b_lora_r32_no_cot_testfb_full_route3_10epoch_1786226769}
NO_COT_CHECKPOINT=${NO_COT_CHECKPOINT:-${NO_COT_RUN}/checkpoints/epoch_0007}
NO_COT_TRAJECTORIES=${NO_COT_TRAJECTORIES:-/mnt/llmd/results/exps/aristides/reason/verified_instruct_trajectories_for_testfb/with_testfb/trajectories_eval.jsonl}
COT_RUN=${COT_RUN:-/mnt/llmd/results/exps/aristides/reason/cot_abstention_qwen3_emb8b_lora_r32_cot_testfb_full_route3_10epoch_1786475425}
COT_CHECKPOINT=${COT_CHECKPOINT:-${COT_RUN}/checkpoints/epoch_0009}
COT_TRAJECTORIES=${COT_TRAJECTORIES:-/mnt/llmd/results/exps/aristides/reason/verified_cot_trajectories_with_testfb/trajectories_eval.jsonl}
VERIFIED_LABELS=${VERIFIED_LABELS:-/mnt/llmd/results/exps/aristides/reason/verified_real_label_eval_1785965509/predictions/predictions_route_3.results.jsonl}
SNAPSHOT=${SNAPSHOT:-1}

JOB_NAME=${JOB_NAME:-swe_testfb_transfer_rescore_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}
mkdir -p "${OUTPUT_DIR}"

RUNNER="${OUTPUT_DIR}/run_rescore.sh"
cat > "${RUNNER}" << SCRIPT_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

python pipelinerl/swe/scripts/offline_router/score_cot_abstention_predictor.py \\
  --checkpoint-dir '${NO_COT_CHECKPOINT}' \\
  --train-config '${NO_COT_RUN}/train_config.json' \\
  --trajectories '${NO_COT_TRAJECTORIES}' \\
  --real-labels-jsonl '${VERIFIED_LABELS}' \\
  --include-test-feedback \\
  --test-feedback-format full \\
  --output-path '${OUTPUT_DIR}/verified_no_cot_testfb_predictions.jsonl' \\
  2>&1 | tee '${OUTPUT_DIR}/score_no_cot.log'

python pipelinerl/swe/scripts/offline_router/score_cot_abstention_predictor.py \\
  --checkpoint-dir '${COT_CHECKPOINT}' \\
  --train-config '${COT_RUN}/train_config.json' \\
  --trajectories '${COT_TRAJECTORIES}' \\
  --real-labels-jsonl '${VERIFIED_LABELS}' \\
  --include-test-feedback \\
  --test-feedback-format full \\
  --output-path '${OUTPUT_DIR}/verified_cot_testfb_predictions.jsonl' \\
  2>&1 | tee '${OUTPUT_DIR}/score_cot.log'
SCRIPT_EOF
chmod +x "${RUNNER}"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" \
  ENV=pipeline-rl \
  CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" \
  NPROC=1 \
  GPU=1 \
  GPU_MEM=80 \
  CPU=16 \
  CPU_MEM=128 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
