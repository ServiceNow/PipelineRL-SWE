#!/usr/bin/env bash
# Train the CoT-stripped ablation predictor: Thinking-2507 trajectories,
# but thinking traces EXCLUDED from predictor input.
#
# This is the cleanest isolation of the CoT signal:
#   - same scout model (Thinking-2507), same patches, same labels
#   - only difference from the CoT predictor: include_thinking=False
#
# Comparing this cell vs. the full CoT predictor (launch_cot_abstention_predictor.sh
# with INCLUDE_THINKING=true) isolates exactly how much the reasoning trace adds.
#
# Optional env vars:
#   TRAJECTORIES_DIR  -- default: cot_trajectories_1785341592_fixed
#   NUM_EPOCHS        -- default: 10
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TRAJECTORIES_DIR=${TRAJECTORIES_DIR:-/mnt/llmd/results/exps/aristides/reason/cot_trajectories_1785341592_fixed}

INCLUDE_THINKING=false \
TRAJECTORIES_DIR="${TRAJECTORIES_DIR}" \
NUM_EPOCHS=${NUM_EPOCHS:-10} \
  bash "${SCRIPT_DIR}/launch_cot_abstention_predictor.sh"
