#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TIMESTAMP=${TIMESTAMP:-$(date +%s)}

# Submit sequentially to avoid snapshot directory races; the scheduler can run the jobs in parallel.
TIMESTAMP="${TIMESTAMP}" bash "${SCRIPT_DIR}/launch_swebench_train_collect_gpt_oss_20b.sh"
TIMESTAMP="${TIMESTAMP}" bash "${SCRIPT_DIR}/launch_swebench_train_collect_qwen3_coder_30b.sh"
TIMESTAMP="${TIMESTAMP}" bash "${SCRIPT_DIR}/launch_swebench_train_collect_qwen3_4b_scout.sh"
