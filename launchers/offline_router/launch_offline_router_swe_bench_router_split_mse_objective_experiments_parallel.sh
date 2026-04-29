#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_TIMESTAMP=${TIMESTAMP:-$(date +%s)}

TIMESTAMP=$((BASE_TIMESTAMP + 1)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_mse_seq_w20_full_5epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 2)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_ce_mse_seq_w20_full_5epoch.sh"
TIMESTAMP=$((BASE_TIMESTAMP + 3)) bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_router_split_bin_20bucket_mse_delta_seq_w20_full_5epoch.sh"
