#!/usr/bin/env bash
# Encoder x prompt scan for the probe. Both axes are FREE at inference: every candidate is one
# prefill of a small model, and changing the prompt costs the identical forward pass.
#
# 3b-lii measured that scaling the probe does NOT help -- the 4B scout beats gpt-oss-20b and -120b
# on average at 1/42 the price -- so the open question is not size but FAMILY, SPECIALISATION and
# PROMPT. This scan varies all three and leaves the cost case untouched.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
OUT=${OUT:-/mnt/llmd/results/exps/aristides/reason/probe_scan_${TIMESTAMP}}
PROMPTS=${PROMPTS:-/mnt/llmd/results/exps/aristides/reason/pool_probe_prompts.jsonl}
SNAPSHOT=${SNAPSHOT:-1}

# model | short tag | family/specialisation
MODELS=(
  "Qwen/Qwen3-0.6B|q06|general, 6.7x smaller than the scout"
  "Qwen/Qwen3-1.7B|q17|general"
  "Qwen/Qwen3-4B-Instruct-2507|q4i|CURRENT SCOUT, baseline"
  "Qwen/Qwen3-4B-Thinking-2507|q4t|same size, reasoning-tuned"
  "Qwen/Qwen2.5-Coder-1.5B-Instruct|c15|code-specialised"
  "Qwen/Qwen2.5-Coder-3B-Instruct|c3|code-specialised"
  "google/gemma-2-2b-it|g2|different family, general"
  "meta-llama/Llama-3.2-1B-Instruct|l1|different family, general, tiny"
)
# Prompt variants, all on the CURRENT scout so the two axes stay separable. The text is written
# to FILES: prompt text has newlines and shell metacharacters, and passing it inline through a job
# COMMAND is a quoting trap (it broke the first submit on an unquoted "?").
PDIR=${PDIR:-/mnt/llmd/results/exps/aristides/reason/probe_prompts}
mkdir -p "${PDIR}"
cat > "${PDIR}/judge_sys.txt" <<'TXT'
You are an expert at assessing programming problems. Do not solve the problem. Judge how difficult
it is, and how likely a strong model is to solve it on a single attempt.
TXT
cat > "${PDIR}/plain_sys.txt" <<'TXT'
You are a helpful assistant.
TXT
cat > "${PDIR}/difficulty_suffix.txt" <<'TXT'

How difficult is this problem, and how likely is a strong model to solve it in one attempt?
TXT

CMDS=()
for spec in "${MODELS[@]}"; do
  IFS='|' read -r M TAG _ <<< "$spec"
  CMDS+=("python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${M} --route-label ${TAG} --prompts-file ${PROMPTS} --activations ${OUT}/enc_${TAG}.npz --max-len 8192")
done
S=Qwen/Qwen3-4B-Instruct-2507
CMDS+=("python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${S} --route-label pjudge --prompts-file ${PROMPTS} --activations ${OUT}/prompt_judge.npz --max-len 8192 --system-prompt-file ${PDIR}/judge_sys.txt")
CMDS+=("python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${S} --route-label pplain --prompts-file ${PROMPTS} --activations ${OUT}/prompt_plain.npz --max-len 8192 --system-prompt-file ${PDIR}/plain_sys.txt")
CMDS+=("python pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract --model ${S} --route-label psuffix --prompts-file ${PROMPTS} --activations ${OUT}/prompt_suffix.npz --max-len 8192 --user-suffix-file ${PDIR}/difficulty_suffix.txt")

if [[ "${SUBMIT}" != "1" ]]; then
  echo "Prepared but not submitted. ${#CMDS[@]} extractions -> ${OUT}"
  printf '  %s\n' "${MODELS[@]}"
  echo "  + 3 prompt variants on the current scout (judge / plain / suffix)"
  echo; echo "Submit with:  SUBMIT=1 bash ${BASH_SOURCE[0]}"
  exit 0
fi

JOINED=$(printf ' && %s' "${CMDS[@]}"); JOINED=${JOINED:4}
make -C "${REPO_ROOT}" job \
  JOB_NAME="probe_scan_${TIMESTAMP}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  SNAPSHOT="${SNAPSHOT}" NPROC=1 GPU=1 GPU_MEM=80 CPU=8 CPU_MEM=64 \
  COMMAND="export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 && export PYTHONPATH=/mnt/llmd/results/exps/aristides/envs/accel:\${PYTHONPATH:-} && mkdir -p ${OUT} && ${JOINED}"
echo "Job: probe_scan_${TIMESTAMP}"
echo "Output: ${OUT}"
