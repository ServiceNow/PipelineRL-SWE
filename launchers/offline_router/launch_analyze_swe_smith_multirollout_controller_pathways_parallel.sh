#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
SUBMIT_SLEEP_SECS=${SUBMIT_SLEEP_SECS:-10}
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-analyze_swe_smith_multirollout}

echo "Launching controller pathway analysis jobs with TIMESTAMP=${TIMESTAMP}"

TIMESTAMP="${TIMESTAMP}" \
JOB_NAME="${JOB_NAME_PREFIX}_controller_ceiling" \
bash "${SCRIPT_DIR}/launch_analyze_swe_smith_multirollout_controller_ceiling.sh"

sleep "${SUBMIT_SLEEP_SECS}"

TIMESTAMP="${TIMESTAMP}" \
JOB_NAME="${JOB_NAME_PREFIX}_learned_controller_cv" \
bash "${SCRIPT_DIR}/launch_analyze_swe_smith_multirollout_learned_controller_cv.sh"

sleep "${SUBMIT_SLEEP_SECS}"

TIMESTAMP="${TIMESTAMP}" \
JOB_NAME="${JOB_NAME_PREFIX}_verifier_calibration" \
bash "${SCRIPT_DIR}/launch_analyze_swe_smith_multirollout_verifier_calibration.sh"

sleep "${SUBMIT_SLEEP_SECS}"

TIMESTAMP="${TIMESTAMP}" \
JOB_NAME="${JOB_NAME_PREFIX}_policy_sim" \
bash "${SCRIPT_DIR}/launch_analyze_swe_smith_multirollout_policy_sim.sh"

echo "Submitted all controller pathway analysis jobs."
