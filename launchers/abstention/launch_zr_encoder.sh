#!/usr/bin/env bash
# ZeroRouter stage-2 encoder (fine-tuned DistilBERT + 11 linguistic features) on LCB / Omni / MMLU-Pro (NEW_PATH 4.A.33).
# Writes <pool>/zr_encoder_D{1,5}.npz for offline routing evaluation. GPU eai job; commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/zr_encoder; NAME="zr_encoder_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/home/toolkit/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python analysis/cost_headroom/zr_encoder.py > ${O}/enc.log 2>&1
echo EXIT \$? >> ${O}/enc.log
EOS
chmod +x ${O}/run.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}"
