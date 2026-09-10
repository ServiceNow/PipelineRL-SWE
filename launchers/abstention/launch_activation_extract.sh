#!/usr/bin/env bash
# Submit one scout activation extraction as a GPU eai job.
#
#   launch_activation_extract.sh <name-stem> <prompts.jsonl> <max-len>
#
# Extraction is a GPU job and must NOT run on the interactive node: it holds the local
# GPU for the length of the run and blocks anything else that needs it.
#
# The vllm-env interpreter and the accel PYTHONPATH shim are both load-bearing. pipeline-rl's
# transformers (4.51.1) rejects the `dtype` kwarg this script passes, and vllm-env has no
# `accelerate` of its own -- the shim supplies it. HF_HUB_OFFLINE avoids the expired-token
# path that silently failed three earlier extraction attempts.
set -euo pipefail

STEM=$1; PROMPTS=$2; MAXLEN=$3
R=/mnt/llmd/results/exps/aristides/reason
NAME="$(echo "${STEM}" | tr "A-Z" "a-z" | sed "s/[^a-z0-9]\+/_/g; s/^_//; s/_$//")_${RANDOM}${RANDOM}"
DIR="${R}/${NAME}"
mkdir -p "${DIR}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export HF_HUB_DISABLE_IMPLICIT_TOKEN=1"
  echo "export HF_HUB_OFFLINE=1"
  echo "export PYTHONPATH=/mnt/llmd/results/exps/aristides/envs/accel:\${PYTHONPATH:-}"
  echo "cd /home/toolkit/PipelineRL-SWE"
  echo "/home/toolkit/.conda/envs/vllm-env/bin/python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py \\"
  echo "  --phase extract --model 'Qwen/Qwen3-4B-Instruct-2507' --route-label scout \\"
  echo "  --prompts-file '${PROMPTS}' \\"
  echo "  --activations '${DIR}/scout.npz' \\"
  echo "  --max-len ${MAXLEN}"
} > "${DIR}/run.sh"
chmod +x "${DIR}/run.sh"

make job JOB_NAME="${NAME}" ENV=vllm-env CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=40 CPU=8 CPU_MEM=64 SNAPSHOT=0 \
  COMMAND="bash ${DIR}/run.sh"
echo "submitted ${NAME} -> ${DIR}"
