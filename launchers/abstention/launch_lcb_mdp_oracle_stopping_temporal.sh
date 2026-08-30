#!/usr/bin/env bash
# Independent replay-only upper-bound diagnostic on the newest temporal counts_last model.
#
# This deliberately gives the policy one bit of future information at every decision:
# whether any still-unseen stored draw succeeds. It replaces only stopping; route beliefs,
# ordering, costs, R, and Bellman H=2 are unchanged. The result is diagnostic-only.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${ARTIFACT_DIR}/tensors_v3}
REPLAY_TAG=${REPLAY_TAG:-replay_oracle_stop_h2_v1}
JOB_NAME=${JOB_NAME:-lcb_mdp_temporal_counts_last_oracle_stop_h2_${TIMESTAMP}}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  artifact: ${ARTIFACT_DIR}
  output: ${ARTIFACT_DIR}/${REPLAY_TAG}
  policy: sequential_decay Bellman H=2
  intervention: perfect any-remaining-success stopping only (diagnostic leakage)

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

# The diagnostic is committed, so use a hermetic repository snapshot for the job.
ARTIFACT_DIR="${ARTIFACT_DIR}" \
TENSORS_DIR="${TENSORS_DIR}" \
STATE_LAYOUT=counts_last \
START_PROTOCOL=scout_first \
NUM_ORDERINGS=5 \
REPLAY_TAG="${REPLAY_TAG}" \
JOB_NAME="${JOB_NAME}" \
SNAPSHOT=1 \
EXTRA_REPLAY_ARGS="--bellman-horizons 2 --oracle-stopping-family sequential_decay --oracle-stopping-horizon 2 --bootstrap-samples ${BOOTSTRAP_SAMPLES}" \
bash "${SCRIPT_DIR}/launch_lcb_mdp_replay.sh"
