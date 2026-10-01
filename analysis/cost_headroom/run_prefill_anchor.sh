#!/usr/bin/env bash
set -euo pipefail
ANCHOR_DATASET=$1
ANCHOR_ROOT=/mnt/llmd/results/exps/aristides/reason/prefill_anchor_verified_20261001
mkdir -p "${ANCHOR_ROOT}"
exec > >(tee -a "${ANCHOR_ROOT}/${ANCHOR_DATASET}.log") 2>&1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python - "${ANCHOR_DATASET}" "${ANCHOR_ROOT}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
label, out = sys.argv[1], Path(sys.argv[2])
root = Path('/mnt/llmd/results/exps/aristides/reason')
source = root / f'{label}_probe_prompts.jsonl'
rows = [json.loads(line) for line in source.read_text().splitlines()]
rng = np.random.default_rng(20261001)
selected = sorted(rng.choice(len(rows), 32, replace=False))
(out/f'{label}_prompts.jsonl').write_text(''.join(json.dumps(dict(problem_id=rows[i]['problem_id'], prompt=rows[i]['prompt']))+'\n' for i in selected))
PY
ANCHOR_MODEL=Qwen/Qwen3-4B-Instruct-2507
if [[ ${ANCHOR_DATASET} == omni500 ]]; then ANCHOR_MODEL=Qwen/Qwen3-4B-Thinking-2507; fi
export PYTHONPATH="/mnt/llmd/results/exps/aristides/envs/accel:${PYTHONPATH:-}"
/home/toolkit/.conda/envs/vllm-env/bin/python -u pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py \
  --phase extract --model "${ANCHOR_MODEL}" --route-label "${ANCHOR_DATASET}_anchor" \
  --prompts-file "${ANCHOR_ROOT}/${ANCHOR_DATASET}_prompts.jsonl" \
  --activations "${ANCHOR_ROOT}/${ANCHOR_DATASET}.npz" --max-len 8192 \
  --system-prompt "You are a helpful assistant."
