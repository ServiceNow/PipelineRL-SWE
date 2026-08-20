#!/usr/bin/env bash
# Submit matched input-only, post-scout, and post-scout+test-feedback models.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${LCB_COLLECTION_DIR:?Set LCB_COLLECTION_DIR to a corrected LCB collection}"

TIMESTAMP=${TIMESTAMP:-$(date +%s)}

TIMESTAMP="${TIMESTAMP}" INPUT_ONLY=true INCLUDE_THINKING=false \
  INCLUDE_TEST_FEEDBACK=false \
  bash "${SCRIPT_DIR}/launch_lcb_abstention_train.sh"

TIMESTAMP="${TIMESTAMP}" INPUT_ONLY=false INCLUDE_THINKING=false \
  INCLUDE_TEST_FEEDBACK=false \
  bash "${SCRIPT_DIR}/launch_lcb_abstention_train.sh"

TIMESTAMP="${TIMESTAMP}" INPUT_ONLY=false INCLUDE_THINKING=false \
  INCLUDE_TEST_FEEDBACK=true TEST_FEEDBACK_FORMAT=full \
  bash "${SCRIPT_DIR}/launch_lcb_abstention_train.sh"
