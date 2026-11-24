#!/usr/bin/env bash
# Small exact-match regression set for the release-repo restructure (internal). Runs commands.txt (CPU lines, or GPU lines
# with GPU=1) from <repo>/analysis, writing under <out>. FT = fixed fine-tuned predictions (copied from the baseline run);
# SCOUT_WORK = the input tree (shared by every run).
# Usage: [GPU=1] [JOBS=8] bash golden.sh <repo> <out> [commands file]
set -uo pipefail
REPO=$1; export GOUT=$2; L=${3:-$(cd "$(dirname "$0")" && pwd)/commands.txt}
export SCOUT_WORK=${SCOUT_WORK:-/home/toolkit/scout-before-you-route/work} SCOUT_OUT=$GOUT NTHREADS=${NTHREADS:-4}
export FT=/mnt/llmd/results/exps/aristides/release_check_20261123/golden_ft
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
mkdir -p "$GOUT/logs"; cd "$REPO/analysis"; i=0
while IFS= read -r c; do
  i=$((i + 1)); [[ $c == wait ]] && { wait; continue; }   # a "wait" line: later commands read earlier outputs
  if [[ ${GPU:-0} == 1 ]]; then [[ $c == *"# GPU"* ]] || continue; else [[ $c == *"# GPU"* ]] && continue; fi
  c=${c%%   # GPU}; c=${c//python /$PY }
  c="${c%% && *} && mkdir -p \"\${OUT_DIR:-.}\" && ${c#* && }"
  ( bash -c "$c" > "$GOUT/logs/$i.log" 2>&1; echo "[$i] exit $?" ) &
  while (( $(jobs -rp | wc -l) >= ${JOBS:-8} )); do wait -n; done
done < "$L"
wait; echo GOLDEN DONE
