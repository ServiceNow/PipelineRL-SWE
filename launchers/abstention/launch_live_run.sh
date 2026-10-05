#!/usr/bin/env bash
# Live run (NEW_PATH 4.A.52). STEP=extract: GPU job, Qwen3-4B-Instruct-2507 prefill of the 1,000 frozen live MMLU-Pro prompts
# (same system prompt / layers / 8192 cap as the fresh features). STEP=route: CPU job, frozen readouts -> per-target routes -> live calls
# (guard $8). Commit + push first. Prepare the sample once with analysis/cost_headroom/live_20261005/prepare_live.py.
set -euo pipefail
O=/mnt/llmd/results/exps/aristides/reason/live_run_20261005; STEP=${STEP:-extract}
cat > ${O}/run_${STEP}.sh <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_DATASETS_OFFLINE=1
if [ "${STEP}" = extract ]; then
  export PYTHONPATH="/mnt/llmd/results/exps/aristides/envs/accel:\${PYTHONPATH:-}"
  /home/toolkit/.conda/envs/vllm-env/bin/python -u pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract \
    --model Qwen/Qwen3-4B-Instruct-2507 --route-label mmlupro_live --prompts-file ${O}/prompts_extract.jsonl --activations ${O}/prefill.npz \
    --max-len 8192 --system-prompt "You are a helpful assistant." > ${O}/extract.log 2>&1
  echo EXIT \$? >> ${O}/extract.log
else
  python -u analysis/cost_headroom/live_20261005/live_route.py --budget-usd \${BUDGET:-8} > ${O}/route.log 2>&1
  echo EXIT \$? >> ${O}/route.log
fi
EOF
chmod +x ${O}/run_${STEP}.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run_${STEP}.sh; exit 0; fi
if [ "${STEP}" = extract ]; then RES="GPU=1 GPU_MEM=48 CPU=8 CPU_MEM=64"; else RES="GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=64"; fi
make job JOB_NAME="live_${STEP}_$(date -u +%H%M%S)" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda ${RES} SNAPSHOT=1 COMMAND="bash ${O}/run_${STEP}.sh"
