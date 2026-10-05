#!/usr/bin/env bash
# Generic CPU analysis job: runs one analysis/cost_headroom script on eai and writes <name>.log next to it on /mnt.
# Usage: SUBMIT=1 bash launchers/abstention/launch_cpu_analysis.sh pin_preview.py [args...]. Commit + push first.
set -euo pipefail
S=$1; shift; NAME=${S%.py}; O=/mnt/llmd/results/exps/aristides/reason/cpu_analysis; mkdir -p $O
cat > $O/run_${NAME}.sh <<EOF
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE/analysis/cost_headroom
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python -u $S $* > $O/${NAME}.log 2>&1
echo EXIT \$? >> $O/${NAME}.log
cp -f ${NAME}.json $O/ 2>/dev/null || true
EOF
chmod +x $O/run_${NAME}.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat $O/run_${NAME}.sh; exit 0; fi
make job JOB_NAME="ana_${NAME}_$(date -u +%H%M%S)" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash $O/run_${NAME}.sh"
echo "log: $O/${NAME}.log"
