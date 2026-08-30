#!/usr/bin/env bash
# Prepared (opt-in) 551 -> 341 chronological LCB MDP experiment.
#
# This launcher is deliberately inert unless SUBMIT=1.  It fits the router on
# every problem before 2024-10-01, uses the first half of the later temporal
# block for calibration, and evaluates on the remaining later date groups.
# Route horizons are asymmetric by design: four scout draws, ten expert draws.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

LCB_COLLECTION_DIR=${LCB_COLLECTION_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}
EXPERT_EVAL_DIR=${EXPERT_EVAL_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_multidraw_experts_1787502237}
EXPERT_TRAIN_DIR=${EXPERT_TRAIN_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_multidraw_experts_train_1787866632}
SCOUT_EVAL_DIR=${SCOUT_EVAL_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_multidraw_scout_1787547502}
SCOUT_TRAIN_DIR=${SCOUT_TRAIN_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_multidraw_scout_train_1787866682}

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SEED=${SEED:-17}
# Seed for sampling reachable failure histories. Defaults to 0 to reproduce every
# artifact built before 2026-08-30. Set equal to SEED for a multi-seed run: holding
# it fixed measures only optimizer variance, not variance of the whole pipeline.
DATASET_SEED=${DATASET_SEED:-0}
# Rolling-origin folds. Set both to evaluate on contiguous date blocks instead of the
# single 551/341 split: three folds give ~534 test problems against 171, which is what
# the wide CIs on the current frontier need.
FOLD_MANIFEST=${FOLD_MANIFEST:-}
FOLD_INDEX=${FOLD_INDEX:-}
NPROC=${NPROC:-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-2e-5}
NUM_ORDERINGS=${NUM_ORDERINGS:-5}
HISTORIES_PER_PROBLEM=${HISTORIES_PER_PROBLEM:-10}
START_PROTOCOL=${START_PROTOCOL:-scout_first}
STATE_LAYOUT=${STATE_LAYOUT:-problem_first}
POS_WEIGHT=${POS_WEIGHT:-none}
STATE_FEATURE_MODE=${STATE_FEATURE_MODE:-text_only}
STATE_FEATURE_HIDDEN_SIZE=${STATE_FEATURE_HIDDEN_SIZE:-64}
TEMPORAL_CALIBRATION_FRACTION=${TEMPORAL_CALIBRATION_FRACTION:-0.5}
ROUTE_DRAW_COUNTS=${ROUTE_DRAW_COUNTS:-scout=4,oss20=10,oss120=10}
SNAPSHOT=${SNAPSHOT:-1}
JOB_NAME=${JOB_NAME:-lcb_mdp_temporal_551_341${FOLD_INDEX:+_fold${FOLD_INDEX}}_${STATE_LAYOUT}_seed${SEED}_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}}

if [[ -n "${FOLD_MANIFEST}" ]]; then
  SPLIT_BANNER="  split: rolling-origin fold ${FOLD_INDEX} from $(basename "${FOLD_MANIFEST}")
  (contiguous date blocks; train precedes calibration precedes test)"
else
  SPLIT_BANNER="  train: 551 problems through 2024-09-28
  calibration: earliest half of later 341 problems
  test: latest half of later 341 problems"
fi

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
${SPLIT_BANNER}
  state layout: ${STATE_LAYOUT}; state features: ${STATE_FEATURE_MODE}; pos weight: ${POS_WEIGHT}
  train seed: ${SEED}; dataset-history seed: ${DATASET_SEED}
  fold: ${FOLD_INDEX:-none (single 551/341 split)}
  route draws: ${ROUTE_DRAW_COUNTS}

Submit explicitly with: SUBMIT=1 bash ${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341.sh
EOF
  exit 0
fi

TENSORS_DIR=${OUTPUT_DIR}/tensors_v3
DATASET_DIR=${OUTPUT_DIR}/reachable_dataset
MODEL_DIR=${OUTPUT_DIR}/model
REPLAY_DIR=${OUTPUT_DIR}/replay

if [[ "${NPROC}" -gt 1 ]]; then
  TRAIN_LAUNCH="python -m accelerate.commands.launch --multi_gpu --mixed_precision bf16 --num_processes ${NPROC} --config_file conf/accelerate/base_mp.yaml"
else
  TRAIN_LAUNCH="python"
fi

# No `cd` into the live tree: SNAPSHOT=1 already sets --workdir and PYTHONPATH to the
# snapshot, and this job chains build -> train -> replay, so its final stage starts
# hours after launch. Prefixing with `cd ${REPO_ROOT}` made that stage load the live
# entry script, which is how two jobs died on 2026-08-28.
COMMAND="python pipelinerl/swe/scripts/livecodebench/build_mdp_tensors_v2.py \
  --collection-dir ${EXPERT_EVAL_DIR} --collection-dir ${EXPERT_TRAIN_DIR} \
  --collection-dir ${SCOUT_EVAL_DIR} --collection-dir ${SCOUT_TRAIN_DIR} \
  --source-collection-dir ${LCB_COLLECTION_DIR} --output-dir ${TENSORS_DIR} \
  --split-mode source_temporal --temporal-calibration-fraction ${TEMPORAL_CALIBRATION_FRACTION} \
  ${FOLD_MANIFEST:+--fold-manifest ${FOLD_MANIFEST} --fold-index ${FOLD_INDEX}} \
  --route-draw-counts ${ROUTE_DRAW_COUNTS} && \
python pipelinerl/swe/scripts/livecodebench/build_mdp_reachable_dataset.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${DATASET_DIR} \
  --histories-per-problem ${HISTORIES_PER_PROBLEM} --start-protocol ${START_PROTOCOL} \
  --state-layout ${STATE_LAYOUT} --seed ${DATASET_SEED} && \
${TRAIN_LAUNCH} pipelinerl/swe/scripts/livecodebench/train_mdp_reachable_policy.py \
  --dataset-dir ${DATASET_DIR} --output-dir ${MODEL_DIR} --seed ${SEED} \
  --num-epochs ${NUM_EPOCHS} --lr ${LR} --max-seq-length 8192 --pos-weight ${POS_WEIGHT} \
  --state-feature-mode ${STATE_FEATURE_MODE} --state-feature-hidden-size ${STATE_FEATURE_HIDDEN_SIZE} && \
python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \
  --tensors-dir ${TENSORS_DIR} --output-dir ${REPLAY_DIR} --sequential-model-dir ${MODEL_DIR} \
  --num-orderings ${NUM_ORDERINGS} --start-protocol ${START_PROTOCOL} \
  --state-layout ${STATE_LAYOUT} --cost-mode usd"

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC="${NPROC}" GPU=1 GPU_MEM=80 CPU=16 CPU_MEM=128 \
  COMMAND="${COMMAND}"

echo "Job: ${JOB_NAME}"
echo "Output: ${OUTPUT_DIR}"
