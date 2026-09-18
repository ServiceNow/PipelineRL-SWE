#!/usr/bin/env bash
# History-conditioned probe: re-prefill the scout on the problem PLUS the attempts that already
# failed (any route), and read beliefs from that. Prompts are built by build_history_prompts.py
# into ${D}; one eai job per shard, submitted in parallel (never chained). The prompt-only
# baseline is extracted in the same jobs so the comparison is matched.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
D=${D:-/mnt/llmd/results/exps/aristides/reason/history_probe}
VARIANT=${VARIANT:-count}
SHARDS=${SHARDS:-4}
SNAPSHOT=${SNAPSHOT:-1}
S=Qwen/Qwen3-4B-Instruct-2507
if [[ "${SUBMIT}" != "1" ]]; then
  echo "Prepared, not submitted: ${SHARDS} jobs over ${D}/${VARIANT}_shard*.jsonl"
  echo "Submit with:  SUBMIT=1 bash ${BASH_SOURCE[0]}"; exit 0
fi
for i in $(seq 0 $((SHARDS-1))); do
  c="python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${S} --route-label hist_${VARIANT}_${i} --prompts-file ${D}/${VARIANT}_shard${i}.jsonl --activations ${D}/act_${VARIANT}_shard${i}.npz --max-len 8192"
  make -C "${REPO_ROOT}" job \
    JOB_NAME="histprobe_${VARIANT}${i}_${TIMESTAMP}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=80 CPU=8 CPU_MEM=64 \
    COMMAND="export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 && export PYTHONPATH=/mnt/llmd/results/exps/aristides/envs/accel:\${PYTHONPATH:-} && ${c}" \
    && echo "  submitted histprobe_${VARIANT}${i}_${TIMESTAMP}" || echo "  FAILED to submit shard ${i}"
  [[ $i -lt $((SHARDS-1)) ]] && sleep 30
done
