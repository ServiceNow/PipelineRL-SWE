#!/usr/bin/env bash
# Collect the peer pool on the FULL evaluation split, for the cross-model transfer table.
#
# The screens that motivated this ran 50 problems each and, by construction, none of them fell in
# the evaluation split -- so the transfer AUCs had to come from an internal split at n=50. This
# collects every peer on the same 171 problems the rest of the paper is evaluated on, which is
# what turns that result from suggestive into a primary table.
#
# These are API-only models. That is the point rather than a limitation: their weights are not
# available, so same-model selective prediction (logits, entropy, an auxiliary head) is simply not
# runnable on them. A shared latent read from one small model we DO host is the only option, and
# each peer costs a two-parameter response curve on top.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
R=/mnt/llmd/results/exps/aristides/reason
SRC=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448
IDS=${IDS:-$R/lcb_test171_ids.txt}
BASE=${BASE:-$R/peer_transfer_$(date +%s)}
KEY=/home/toolkit/.secrets/openrouter_api_key
CONCURRENCY=${CONCURRENCY:-6}
SUBMIT=${SUBMIT:-0}
PEERS=${PEERS:-"glm5:z-ai/glm-5 deepseek:deepseek/deepseek-v4-flash kimi:moonshotai/kimi-k2.5 minimax:minimax/minimax-m2.7 qmax:qwen/qwen3-max"}

mkdir -p "${BASE}"
for SPEC in ${PEERS}; do
  LABEL=${SPEC%%:*}; MODEL=${SPEC#*:}
  RUN="${BASE}/run_${LABEL}.sh"
  cat > "${RUN}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
export OPENROUTER_API_KEY=\$(cat ${KEY})
python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \\
  --source-collection-dir '${SRC}' --output-dir '${BASE}' \\
  --route-label '${LABEL}' --model '${MODEL}' --api-key-file '${KEY}' \\
  --splits 'train,eval' --problem-ids-file '${IDS}' \\
  --max-tokens 65536 --temperature 0.2 --concurrency ${CONCURRENCY} \\
  --eval-timeout 10 --gen-timeout 1800 --max-invalid-frac 0.20 \\
  --output-suffix _t171
SCRIPT
  chmod +x "${RUN}"
  if [[ "${SUBMIT}" == "1" ]]; then
    make -C "${REPO_ROOT}" job JOB_NAME="peer_${LABEL}_$RANDOM" ENV=pipeline-rl \
      CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=0 GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=32 \
      COMMAND="bash ${RUN}" >/dev/null \
      && echo "submitted ${LABEL} (${MODEL})" || { echo "FAILED ${LABEL}"; exit 1; }
  fi
done
echo "Output: ${BASE}"
