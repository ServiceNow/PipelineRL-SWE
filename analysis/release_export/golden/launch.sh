#!/usr/bin/env bash
# One eai job per golden command (internal). <src>: a code snapshot (repo layout); <out>: output root; <tag>: job-name tag.
# Lines after the "wait" line read outputs of earlier ones: launch them with PHASE=2 once the first phase is done.
# Usage: [SUBMIT=1] [PHASE=1|2] [ONLY="i j"] bash launch.sh <src> <out> <tag> [commands file]
set -uo pipefail
SRC=$1; OUT=$2; TAG=$3; L=${4:-$(cd "$(dirname "$0")" && pwd)/commands.txt}
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; mkdir -p "$OUT/logs" "$OUT/cmds"
ENV="export SCOUT_WORK=/home/toolkit/scout-before-you-route/work SCOUT_OUT=$OUT GOUT=$OUT NTHREADS=16 FT=/mnt/llmd/results/exps/aristides/release_check_20261123/golden_ft HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
i=0; phase=1
while IFS= read -r c; do
  i=$((i + 1)); [[ $c == wait ]] && { phase=2; continue; }
  [[ $phase == ${PHASE:-1} ]] || continue; [[ -n ${ONLY:-} && " $ONLY " != *" $i "* ]] && continue
  g=0; m=0; [[ $c == *"# GPU"* ]] && { g=1; m=40; }
  c=${c%%   # GPU}; c=${c//python /$PY }; c="${c%% && *} && mkdir -p \"\${OUT_DIR:-.}\" && ${c#* && }"
  printf '%s\n' "$ENV && cd $SRC/analysis && ( $c ) > $OUT/logs/$i.log 2>&1; echo EXIT \$? >> $OUT/logs/$i.log" > "$OUT/cmds/$i.sh"
  [[ ${SUBMIT:-0} == 1 ]] || { echo "[dry] gold_${TAG}_$i g=$g: ${c:0:120}"; continue; }
  echo -n "gold_${TAG}_$i "
  make -C /home/toolkit/PipelineRL-SWE job JOB_NAME="gold_${TAG}_$i" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=0 GPU=$g GPU_MEM=$m \
    CPU=16 CPU_MEM=128 COMMAND="bash $OUT/cmds/$i.sh" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true
done < "$L"
