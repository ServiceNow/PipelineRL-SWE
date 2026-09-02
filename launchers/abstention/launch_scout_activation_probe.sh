#!/usr/bin/env bash
# Scout activation probe for the abstention gate.
#
# The gate is the last live lever -- oracle stopping is ~63% of headroom -- and its response
# to predictor quality is convex: flat below AUC 0.85, +2.4pt at 0.90. Every affordable
# channel tried here sits at 0.70-0.76 once conditioned on the decision being live:
# the `nothing` head 0.7335, post-scout abstention 0.7629, cheap observables 0.7491,
# failure-mode taxonomy 0.7030, token entropy at or below chance, verbalized confidence
# 0.5700, kNN retrieval 0.6690.
#
# Activations are the one channel never tried. arXiv 2602.09924 shows pre-generation probes
# predict a model's OWN success well enough to route a pool; the harder question here is
# whether a 4B model's internal state predicts POOL SOLVABILITY -- will anything in the
# portfolio solve this at any depth -- which is what abstention actually asks. Because
# failures are ~89% shared across this pool, that is largely a difficulty question, and
# difficulty is what activations plausibly encode better than surface text.
#
# One GPU job, no collection, no API spend. Extracts last-token hidden states at eight
# depths, before generation and after the scout's completed attempt, then fits linear probes
# and reports AUC both unconditionally and conditional on the scout having failed. Only the
# conditional number counts, and the bar is 0.90.
#
# Labels are de-contaminated with the truncation audit: 24.2% of "impossible" problems fall
# at a 32k cap, so those are relabelled solvable before any probe is fit. Without that the
# probe would be trained to predict a collection artifact.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
TENSORS_DIR=${TENSORS_DIR:-$R/lcb_mdp_temporal_551_341_prepared_v1/tensors_v3}
AUDIT_DIR=${AUDIT_DIR:-$R/lcb_truncation_audit_1788302327}
SCOUT_GLOB_EVAL=${SCOUT_GLOB_EVAL:-$R/lcb_multidraw_scout_1787547502/scout_eval_d0.jsonl}
SCOUT_GLOB_TRAIN=${SCOUT_GLOB_TRAIN:-$R/lcb_multidraw_scout_train_1787866682/scout_train_d0.jsonl}
MAX_LEN=${MAX_LEN:-8192}
LIMIT=${LIMIT:-0}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}

JOB_NAME=${JOB_NAME:-scout_activation_probe_${TIMESTAMP}}
OUTPUT_DIR=${OUTPUT_DIR:-$R/${JOB_NAME}}
mkdir -p "${OUTPUT_DIR}"
ACT="${OUTPUT_DIR}/activations.npz"

RUNNER="${OUTPUT_DIR}/run_probe.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
  echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
  echo "echo '=== extracting activations (GPU) ==='"
  echo "python pipelinerl/swe/scripts/livecodebench/scout_activation_probe.py --phase extract \\"
  echo "  --model '${MODEL}' --activations '${ACT}' --max-len ${MAX_LEN} --limit ${LIMIT} \\"
  echo "  --scout-draw '${SCOUT_GLOB_EVAL}' --scout-draw '${SCOUT_GLOB_TRAIN}'"
  echo "echo '=== fitting probes ==='"
  echo "python pipelinerl/swe/scripts/livecodebench/scout_activation_probe.py --phase probe \\"
  echo "  --activations '${ACT}' --tensors-dir '${TENSORS_DIR}' --audit-dir '${AUDIT_DIR}' \\"
  echo "  | tee '${OUTPUT_DIR}/probe_report.txt'"
} > "${RUNNER}"
chmod +x "${RUNNER}"

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job:     ${JOB_NAME}
  model:   ${MODEL}
  labels:  pool solvability from ${TENSORS_DIR}, de-contaminated with ${AUDIT_DIR}
  output:  ${OUTPUT_DIR}/probe_report.txt
  cost:    one GPU, no API spend

Submit with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

make -C "${REPO_ROOT}" job \
  JOB_NAME="${JOB_NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=24 CPU=8 CPU_MEM=48 \
  COMMAND="bash ${RUNNER}"

echo "Job:        ${JOB_NAME}"
echo "Report:     ${OUTPUT_DIR}/probe_report.txt"
