#!/usr/bin/env bash
set -euo pipefail

BASE_TIMESTAMP=${TIMESTAMP:-$(date +%s)}

TIMESTAMP=$((BASE_TIMESTAMP + 1)) bash launch_offline_router_swe_bench_router_split_bin_20bucket_delta_aux_seq_mse_w20_train_only.sh &
TIMESTAMP=$((BASE_TIMESTAMP + 2)) bash launch_offline_router_swe_bench_router_split_bin_20bucket_delta_aux_seq_huber02_w10_train_only.sh &
TIMESTAMP=$((BASE_TIMESTAMP + 3)) bash launch_offline_router_swe_bench_router_split_bin_20bucket_delta_aux_seq_mse_w20_overfit512_10epoch.sh &
TIMESTAMP=$((BASE_TIMESTAMP + 4)) bash launch_offline_router_swe_bench_router_split_bin_20bucket_overfit512_10epoch.sh &

wait
