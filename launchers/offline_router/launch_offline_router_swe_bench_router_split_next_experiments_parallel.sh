#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

BASE_TIMESTAMP=${TIMESTAMP:-$(date +%s)}

# Submit sequentially to avoid local snapshot-directory collisions. The remote
# jobs still run concurrently once each submission succeeds.
TIMESTAMP=$((BASE_TIMESTAMP + 1)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_full_5epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 2)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_delta_aux_seq_mse_w20_full_5epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 3)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_restricted_delta_aux_seq_mse_w20_overfit512_10epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 4)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_delta_bin_20bucket_overfit512_10epoch.sh"
