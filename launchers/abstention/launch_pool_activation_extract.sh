#!/usr/bin/env bash
# Extract pre-generation activations from EVERY pool member, not just the scout.
#
# This runs the baseline the project has never been able to run. arXiv 2602.09924 states that
# its routing rule "requires training separate probes for each model in the pool", and does
# not charge for those forward passes. Our experts were API-served, so their hidden states
# were unreachable and every cross-model number came from probing the 4B scout instead.
# Self-hosting the experts fills the diagonal of the transfer matrix -- their method, run
# faithfully -- against our scout row.
#
# Environment: vllm-env, for transformers 4.57 (pipeline-rl is on 4.51, which has no gpt_oss,
# and upgrading it would break vllm 0.8.5 underneath the training loop). Only torch,
# transformers and numpy are needed here because the prompts are frozen to a file by
# `--phase prompts` under pipeline-rl first, so both routes provably embed identical text.
#
# Sizing: gpt-oss ships MXFP4, but without triton_kernels transformers dequantizes to bf16.
# 20b -> ~42GB, one card. 120b -> ~234GB, four. Extraction is 892 forward passes, so the
# large allocation is short-lived.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason

TIMESTAMP=${TIMESTAMP:-$(date +%s)}
PROMPTS=${PROMPTS:-$R/pool_probe_prompts.jsonl}
BASE=${BASE:-$R/pool_activations_${TIMESTAMP}}
MAX_LEN=${MAX_LEN:-8192}
SNAPSHOT=${SNAPSHOT:-1}
SUBMIT=${SUBMIT:-0}
ENVNAME=${ENVNAME:-vllm-env}
mkdir -p "${BASE}"

# label:model:gpus
ROUTES=${ROUTES:-"scout:Qwen/Qwen3-4B-Instruct-2507:1 oss20:openai/gpt-oss-20b:1 oss120:openai/gpt-oss-120b:4"}

for SPEC in ${ROUTES}; do
  LABEL=${SPEC%%:*}; REST=${SPEC#*:}; MODEL=${REST%:*}; GPUS=${REST##*:}
  RUNNER="${BASE}/run_${LABEL}.sh"
  {
    echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
    echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
    echo "python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract \\"
    echo "  --model '${MODEL}' --route-label ${LABEL} \\"
    echo "  --prompts-file '${PROMPTS}' \\"
    echo "  --activations '${BASE}/${LABEL}.npz' --max-len ${MAX_LEN}"
  } > "${RUNNER}"
  chmod +x "${RUNNER}"
  if [[ "${SUBMIT}" == "1" ]]; then
    make -C "${REPO_ROOT}" job JOB_NAME="pool_act_${LABEL}_${TIMESTAMP}" ENV="${ENVNAME}" \
      CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU="${GPUS}" GPU_MEM=80 \
      CPU=8 CPU_MEM=64 COMMAND="bash ${RUNNER}" >/dev/null 2>&1 \
      && echo "submitted ${LABEL} (${MODEL}, ${GPUS} GPU)"
    sleep 30
  fi
done

cat <<EOM

Activations land in ${BASE}/{scout,oss20,oss120}.npz

Then fill the transfer matrix under pipeline-rl (needs scikit-learn):
  python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase matrix \\
    --activations-file scout=${BASE}/scout.npz \\
    --activations-file oss20=${BASE}/oss20.npz \\
    --activations-file oss120=${BASE}/oss120.npz \\
    --tensors-dir $R/lcb_mdp_temporal_551_341_prepared_v1/tensors_v3 \\
    --audit-dir $R/lcb_truncation_audit_1788302327
EOM
[[ "${SUBMIT}" == "1" ]] || echo "Prepared but not submitted. Submit with SUBMIT=1."
