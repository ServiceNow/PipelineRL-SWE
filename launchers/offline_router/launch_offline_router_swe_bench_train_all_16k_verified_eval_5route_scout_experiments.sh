#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "${SCRIPT_DIR}/launch_offline_router_swe_bench_train_all_16k_verified_eval_5route_scout_conditioned_routing_experiments.sh"
