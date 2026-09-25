#!/usr/bin/env bash
# Submit the 137M history-conditioned route-success fine-tune as an EAI job.
set -euo pipefail

R=/mnt/llmd/results/exps/aristides/reason
NAME="jina137_history_ft_$(date -u +%Y%m%d_%H%M%S)_${RANDOM}"
DIR="${R}/${NAME}"
mkdir -p "${DIR}"

cat > "${DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /home/toolkit/PipelineRL-SWE
# The jina model and its remote-code dependency live in the HOME HF cache, not the shared
# /transformers_cache mount; the first run died on exactly that ("not a valid model identifier").
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python pipelinerl/swe/scripts/livecodebench/finetune_encoder_head.py --prompts-dir ${R}/pool_v2_hist_prompts_5r --variant code --tensors-dir ${R}/pool_v2_tensors_5rung --batch 4 --epochs 3 --seed 0 --out ${DIR}/encoder_head.pt 2>&1 | tee ${DIR}/train.log
EOF
chmod +x "${DIR}/run.sh"

make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  GPU=1 GPU_MEM=0 CPU=8 CPU_MEM=64 SNAPSHOT=0 \
  COMMAND="bash ${DIR}/run.sh"
echo "submitted ${NAME} -> ${DIR}"
