#!/usr/bin/env bash
# History-conditioned probe: re-prefill the scout on the problem PLUS the attempts that already
# failed (any route), and read beliefs from that. Prompts are built by build_history_prompts.py
# into ${D}; one eai job per shard, submitted in parallel (never chained). The prompt-only
# baseline is extracted in the same jobs so the comparison is matched.
set -euo pipefail
# ENVIRONMENT IS LOAD-BEARING. pool_activation_probe.py needs vllm-env (transformers 4.57);
# pipeline-rl ships 4.51.1, which rejects the dtype kwarg this script passes, and the job dies
# instantly with "Failed to import transformers.models.qwen3.modeling_qwen3". That is what made
# eight extraction jobs FAIL while looking, from the job list alone, like a GPU queue problem.
# launch_activation_extract.sh had this right; this launcher did not.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
D=${D:-/mnt/llmd/results/exps/aristides/reason/history_probe}
VARIANT=${VARIANT:-count}
SHARDS=${SHARDS:-4}
SNAPSHOT=${SNAPSHOT:-1}
# Optional question appended after the history (SUFFIX=path), so the last-token readout has
# something to judge; TAG names the output.
SUFFIX=${SUFFIX:-}
TAG=${TAG:-${VARIANT}}
S=Qwen/Qwen3-4B-Instruct-2507
if [[ "${SUBMIT}" != "1" ]]; then
  echo "Prepared, not submitted: ${SHARDS} jobs over ${D}/${VARIANT}_shard*.jsonl"
  echo "Submit with:  SUBMIT=1 bash ${BASH_SOURCE[0]}"; exit 0
fi
SHARD_IDS=${SHARD_IDS:-$(seq 0 $((SHARDS-1)))}
for i in ${SHARD_IDS}; do
  c="/home/toolkit/.conda/envs/vllm-env/bin/python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${S} --route-label hist_${VARIANT}_${i} --prompts-file ${D}/${VARIANT}_shard${i}.jsonl --activations ${D}/act_${TAG}_shard${i}.npz --max-len 8192${SUFFIX:+ --user-suffix-file ${SUFFIX}}"
  make -C "${REPO_ROOT}" job \
    JOB_NAME="histprobe_${TAG}${i}_${TIMESTAMP}" ENV=vllm-env CONDA_EXE=/opt/conda/bin/conda \
    SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=${GPU_MEM:-40} CPU=8 CPU_MEM=64 \
    COMMAND="export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 && export HF_HUB_OFFLINE=1 && export PYTHONPATH=/mnt/llmd/results/exps/aristides/envs/accel:\${PYTHONPATH:-} && ${c}" \
    > /tmp/claude-13011/-home-toolkit-PipelineRL-SWE/29f3ed3b-1f85-424a-8576-97a9148bdc53/scratchpad/histprobe_submit_${TAG}${i}.log 2>&1 && echo "  submitted histprobe_${TAG}${i}_${TIMESTAMP}" || echo "  FAILED to submit shard ${i}"
  sleep 30
done
