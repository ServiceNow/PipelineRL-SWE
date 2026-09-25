#!/usr/bin/env bash
# Test-writer smoke test (NEW_PATH.md 2.7) — local Qwen3-4B leg (GPU eai job).
# qwen3-4b is not on OpenRouter; the pool serves it locally. This job spins a vLLM server
# (same pattern as lcb_corrected_temporal_*/run_collect.sh) and generates the qwen4b suites
# against it. ~341 short calls, minutes of GPU time. Run in parallel with
# launch_testwriter_smoke.sh (the OpenRouter leg). Resumable on rerun.
set -euo pipefail

R=/mnt/llmd/results/exps/aristides/reason
NAME="testwriter_smoke_qwen4b_$(date -u +%Y%m%d_%H%M%S)_${RANDOM}"
DIR="${R}/${NAME}"
mkdir -p "${DIR}"

cat > "${DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
/home/toolkit/.conda/envs/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
  --model 'Qwen/Qwen3-4B-Instruct-2507' \
  --served-model-name 'Qwen/Qwen3-4B-Instruct-2507' \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  > ${DIR}/vllm_server.log 2>&1 &
VLLM_PID=\$!
trap 'kill \${VLLM_PID} 2>/dev/null || true' EXIT

ready=false
for _ in \$(seq 1 120); do
  if curl -s http://localhost:8000/v1/models | grep -q 'Qwen3-4B'; then ready=true; break; fi
  sleep 5
done
if [ "\$ready" != "true" ]; then
  echo "vLLM failed to become ready" >&2; tail -50 ${DIR}/vllm_server.log >&2; exit 1
fi

/home/toolkit/.conda/envs/pipeline-rl/bin/python pipelinerl/swe/scripts/livecodebench/generate_test_suites.py \
  --pool-dir ${R}/pool_v2_tensors_5rung \
  --out-dir ${R}/testwriter_smoke_lcb \
  --splits test \
  --writers qwen4b \
  --base-url http://localhost:8000 \
  --api-key-file ${DIR}/local_key \
  --concurrency 8 \
  2>&1 | tee ${DIR}/gen.log
EOF
chmod +x "${DIR}/run.sh"
echo local > "${DIR}/local_key"

if [ "${SUBMIT:-0}" != "1" ]; then
  echo "dry run (set SUBMIT=1 to submit): would submit ${NAME}, command:"
  cat "${DIR}/run.sh"
  exit 0
fi

make job JOB_NAME="${NAME}" ENV=vllm-env CONDA_EXE=/opt/conda/bin/conda \
  GPU=1 GPU_MEM=16 CPU=4 CPU_MEM=16 SNAPSHOT=0 \
  COMMAND="bash ${DIR}/run.sh"
echo "submitted ${NAME} -> qwen4b suites -> ${R}/testwriter_smoke_lcb"