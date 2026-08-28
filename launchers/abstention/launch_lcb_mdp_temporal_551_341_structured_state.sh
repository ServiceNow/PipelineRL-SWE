#!/usr/bin/env bash
# Opt-in test of direct numeric routing-state features on the temporal LCB split.
# Text context remains counts_last; only the policy head additionally receives
# the normalized 11-dimensional structured_v1 state vector.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
JOB_NAME=${JOB_NAME:-lcb_mdp_temporal_551_341_counts_last_structured_v1_${TIMESTAMP}}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  comparison: counts_last text-only versus counts_last + structured_v1 state features
  features: per-route failed/remaining fractions, total failure fraction, latest-route one-hot
  split: chronological 551 train / 170 calibration / 171 test

Submit explicitly with: SUBMIT=1 bash ${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341_structured_state.sh
EOF
  exit 0
fi

SUBMIT=1 TIMESTAMP="${TIMESTAMP}" STATE_LAYOUT=counts_last \
  STATE_FEATURE_MODE=structured_v1 STATE_FEATURE_HIDDEN_SIZE=64 \
  JOB_NAME="${JOB_NAME}" \
  bash "${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341.sh"
