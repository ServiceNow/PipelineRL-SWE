#!/usr/bin/env bash
# Fresh-set GPU pieces (NEW_PATH 4.A.43): Intern-Decision inference on fresh Omni / MMLU-Pro with the trained adapters, and
# MiniLM + jina-code text embeddings of the expanded pools. No API spend. Commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; O=${R}/intern_finetune_20261001; T=$(date -u +%H%M%S)
for POOL in Omni MMLU-Pro; do
  L=${POOL,,}; L=${L//-/_}
  CMD="bash analysis/cost_headroom/run_intern_fresh.sh ${POOL} ${O}"
  if [[ ${SUBMIT:-0} == 1 ]]; then make job JOB_NAME="intern_fresh_${L}_${T}" CONDA=0 GPU=1 GPU_MEM=48 CPU=8 CPU_MEM=64 SNAPSHOT=1 COMMAND="${CMD}"; else echo "$CMD"; fi
done
CMD="D=\${TMPDIR:-/tmp}/st_deps; python -m pip install -q --no-deps --target \$D sentence-transformers==3.4.1 && PYTHONPATH=\$D python analysis/cost_headroom/fresh_embeddings.py > ${R}/expanded_eval_20261001/embeddings.log 2>&1"
if [[ ${SUBMIT:-0} == 1 ]]; then make job JOB_NAME="fresh_embed_${T}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=0 CPU=8 CPU_MEM=64 SNAPSHOT=1 COMMAND="${CMD}"; else echo "$CMD"; fi
