#!/usr/bin/env bash
# Source this file to expose the pinned official LiveCodeBench evaluator.
set -euo pipefail

LCB_RUNNER_COMMIT=${LCB_RUNNER_COMMIT:-28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24}
LCB_RUNNER_CACHE=${LCB_RUNNER_CACHE:-/mnt/llmd/results/tools/livecodebench}
LCB_RUNNER_DIR=${LCB_RUNNER_DIR:-${LCB_RUNNER_CACHE}/${LCB_RUNNER_COMMIT}}

if [[ ! -d "${LCB_RUNNER_DIR}/lcb_runner/evaluation" ]]; then
  mkdir -p "${LCB_RUNNER_CACHE}"
  exec 9>"${LCB_RUNNER_CACHE}/.install.lock"
  flock 9
  if [[ ! -d "${LCB_RUNNER_DIR}/lcb_runner/evaluation" ]]; then
    tmp_dir="${LCB_RUNNER_DIR}.tmp.$$"
    rm -rf "${tmp_dir}"
    git clone --filter=blob:none https://github.com/LiveCodeBench/LiveCodeBench.git "${tmp_dir}"
    git -C "${tmp_dir}" checkout --detach "${LCB_RUNNER_COMMIT}"
    rm -rf "${LCB_RUNNER_DIR}"
    mv "${tmp_dir}" "${LCB_RUNNER_DIR}"
  fi
  flock -u 9
  exec 9>&-
fi

export LCB_RUNNER_COMMIT
export LCB_RUNNER_DIR
export PYTHONPATH="${LCB_RUNNER_DIR}:${PYTHONPATH:-}"

python -c 'from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness' \
  >/dev/null
echo "Using official LiveCodeBench runner ${LCB_RUNNER_COMMIT} from ${LCB_RUNNER_DIR}"
