#!/usr/bin/env bash
# Track B patch-aware test pilot (NEW_PATH.md 4.B.4): does a writer that sees the candidate patch unlock
# per-candidate test-writer selection? CPU-only eai job, open models only:
#   generate  1992 scripts (166 instances x {correct, wrong} candidate x {oss20, dsv4f, qcoder30} x 2 draws), OpenRouter
#   execute   every script on base / C / W / gold in Daytona, concurrency 3 (org disk cap; <= 10 sandboxes total)
#   analyze   split-draw ceiling vs every fixed writer + same-family false accepts -> analysis.log
# Resumable: each step skips rows already written. Estimated ~$3 total (<= $4.5), ~5 h (dsv4f writing ~1 h at 32 concurrent, execution ~3-4 h).
# Commit and push first: SNAPSHOT=1 runs the committed tree.
set -euo pipefail

R=/mnt/llmd/results/exps/aristides/reason
O=${R}/swe_patchaware_pilot
NAME="patchaware_pilot_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${O}"

cat > "${O}/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/home/toolkit/.cache/huggingface HF_DATASETS_CACHE=/home/toolkit/.cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
S=pipelinerl/swe/scripts/offline_router/swe_patchaware_pilot.py
python \${S} generate --concurrency 32 > ${O}/generate.log 2>&1
python \${S} execute --concurrency 3 > ${O}/execute.log 2>&1
python \${S} execute --concurrency 3 >> ${O}/execute.log 2>&1   # retry rows that errored (sandbox create etc.)
python \${S} analyze > ${O}/analysis.log 2>&1
echo ALL DONE >> ${O}/analysis.log
EOF
chmod +x "${O}/run.sh"

if [ "${SUBMIT:-0}" != "1" ]; then
  echo "dry run (set SUBMIT=1 to submit): would submit ${NAME}, command:"
  cat "${O}/run.sh"
  exit 0
fi

make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda \
  GPU=0 GPU_MEM=0 CPU=4 CPU_MEM=16 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O} (generate.log, execute.log, analysis.log)"
