#!/usr/bin/env bash
# Submit one replay_mdp_full_execution.py sweep as a CPU-only eai job.
#
#   launch_replay.sh <name-stem> <args to replay_mdp_full_execution.py...>
#
# The output dir is derived from the job name, so every submission gets its own
# directory and no two runs can land on the same name.
set -euo pipefail

STEM=$1; shift
R=/mnt/llmd/results/exps/aristides/reason
NAME="$(echo "${STEM}" | tr "A-Z" "a-z")_${RANDOM}${RANDOM}"
DIR="${R}/${NAME}"
mkdir -p "${DIR}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "python pipelinerl/swe/scripts/livecodebench/replay_mdp_full_execution.py \\"
  echo "  $* \\"
  echo "  --output-dir ${DIR}/replay 2>&1 | tail -6"
} > "${DIR}/run.sh"
chmod +x "${DIR}/run.sh"

make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=32 SNAPSHOT=0 \
  COMMAND="bash ${DIR}/run.sh"
echo "submitted ${NAME} -> ${DIR}"
