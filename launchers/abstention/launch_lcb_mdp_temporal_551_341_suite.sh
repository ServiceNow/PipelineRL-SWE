#!/usr/bin/env bash
# Opt-in submission wrapper for the first temporal representation comparison.
# It deliberately waits 30 seconds between scheduler submissions.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted. This suite will submit, 30 seconds apart:
  1. temporal 551->341, problem_first, unweighted BCE
  2. temporal 551->341, counts_last, unweighted BCE

Submit explicitly with: SUBMIT=1 bash ${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341_suite.sh
EOF
  exit 0
fi

SUBMIT=1 TIMESTAMP="${TIMESTAMP}a" STATE_LAYOUT=problem_first POS_WEIGHT=none \
  JOB_NAME="lcb_mdp_temporal_551_341_problem_first_unweighted_${TIMESTAMP}" \
  bash "${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341.sh"

sleep 30

SUBMIT=1 TIMESTAMP="${TIMESTAMP}b" STATE_LAYOUT=counts_last POS_WEIGHT=none \
  JOB_NAME="lcb_mdp_temporal_551_341_counts_last_unweighted_${TIMESTAMP}" \
  bash "${SCRIPT_DIR}/launch_lcb_mdp_temporal_551_341.sh"
