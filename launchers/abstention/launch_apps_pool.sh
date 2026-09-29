#!/usr/bin/env bash
# APPS full pool (NEW_PATH 4.A.13 pre-registered GAIN call; 4.A.35): ONE draw per route (one-shot claims only; the pre-registration
# assumed Omni's 4/3/3/2/2 -- deviation reported). oss20lo d0 exists from the screen (apps_pool/oss20lo_{train,eval}.jsonl).
# Est. spend ~$11 (oss120hi ~$8). Execution-graded locally by collect_cc_expert. GPU not needed; eai CPU job. Commit + push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/apps_pool; NAME="apps_pool_$(date -u +%Y%m%d_%H%M%S)"
cat > ${O}/run_full.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
F=${O}
for s in train eval; do [ -f \$F/oss20lo_\${s}_d0.jsonl ] || cp \$F/oss20lo_\${s}.jsonl \$F/oss20lo_\${s}_d0.jsonl; done
C="-m pipelinerl.swe.scripts.codecontests.collect_cc_expert --tasks-file ${R}/apps_tasks.jsonl --output-dir \$F --concurrency 40 --grade-concurrency 4 --output-suffix _d0"
python \$C --route-label oss20md --model openai/gpt-oss-20b --reasoning-effort medium > \$F/log_oss20md_d0.txt 2>&1 &
python \$C --route-label dsv4f --model deepseek/deepseek-v4-flash --temperature 0.7 --top-p 0.95 --max-tokens 128000 --reasoning-enabled --require-parameters > \$F/log_dsv4f_d0.txt 2>&1 &
python \$C --route-label oss120md --model openai/gpt-oss-120b --reasoning-effort medium --max-tokens 115264 > \$F/log_oss120md_d0.txt 2>&1 &
python \$C --route-label oss120hi --model openai/gpt-oss-120b --reasoning-effort high --max-tokens 115264 > \$F/log_oss120hi_d0.txt 2>&1 &
wait
echo ALL DONE > \$F/full_done.txt
EOS
chmod +x ${O}/run_full.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run_full.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run_full.sh"
echo "submitted ${NAME} -> ${O}"
