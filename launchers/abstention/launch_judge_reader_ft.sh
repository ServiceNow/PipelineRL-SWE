#!/usr/bin/env bash
# FrugalGPT-style per-tier answer scorer in the code setting (NEW_PATH 4.A.24): fine-tune the 137M jina-code encoder, one per
# LCB tier, on problem + attempt code -> correct?, vs the frozen 4B judge probe on the same text. GPU eai job; commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/judge_reader_ft; NAME="judge_ft137m_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/home/toolkit/.cache/huggingface
python analysis/cost_headroom/finetune_judge_reader.py > ${O}/ft.log 2>&1
echo EXIT \$? >> ${O}/ft.log
EOS
chmod +x ${O}/run.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}"
