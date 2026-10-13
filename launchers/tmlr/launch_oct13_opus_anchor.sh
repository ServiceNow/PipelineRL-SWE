#!/usr/bin/env bash
# 2026-10-13 (user-approved, paid: ~$10-30, guard $30 billed): Claude Opus 5.5 as a frontier reference on LiveCodeBench. 100 test
# problems of the manifest split (random sample, seed 0), one sample each, reasoning effort high, through the same collector,
# prompts and grader as the five routes (collect_lcb_expert.py). Anthropic thinking needs temperature 1. Output: $O/opus55_{train,eval}_d0.jsonl
# (the collector's own train/eval file names; the ids are all manifest TEST problems). Usage: SUBMIT=1 bash launch_oct13_opus_anchor.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; O=$R/lcb_opus55_anchor_20261013; mkdir -p $O
SRC=$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448
/home/toolkit/.conda/envs/pipeline-rl/bin/python3 - "$O/problem_ids.txt" <<'EOF'
import json, sys, numpy as np
t = json.load(open("/mnt/llmd/results/exps/aristides/reason_pinned/pool_v2_tensors_5rung/split_manifest.json"))["test_problem_ids"]
ids = sorted(np.random.default_rng(0).choice(sorted(map(str, t)), 100, replace=False))
open(sys.argv[1], "w").write("\n".join(ids) + "\n"); print(len(ids), "test ids ->", sys.argv[1])
EOF
cat > $O/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh
python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py --source-collection-dir $SRC --output-dir $O --route-label opus55 \
  --model anthropic/claude-opus-5.5 --reasoning-effort high --temperature 1.0 --splits train,eval --problem-ids-file $O/problem_ids.txt \
  --max-tokens 64000 --gen-timeout 3600 --max-invalid-frac 0.10 --output-suffix _d0 --api-key-file /home/toolkit/.secrets/openrouter_api_key \
  --concurrency 8 --budget-usd 30 > $O/log.txt 2>&1
echo EXIT \$? >> $O/log.txt
EOS
chmod +x $O/run.sh
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run: $O/run.sh"; exit 0; }
make job JOB_NAME=opus55_lcb_anchor ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 SNAPSHOT=1 COMMAND="bash $O/run.sh" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true
