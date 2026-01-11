set -euo pipefail

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

: "${EXTERNAL_VLLM_URL:?EXTERNAL_VLLM_URL is required for the external expert}"
: "${EXTERNAL_VLLM_TOKEN:?EXTERNAL_VLLM_TOKEN is required for the external expert}"

JOB_NAME=multi_expert_perf_head_ext_logging
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME} 
make job JOB_NAME=${JOB_NAME} ENV=prl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 NPROC=8 COMMAND="cd /home/toolkit/PipelineRL-SWE; python -m pipelinerl.launch --config-dir conf --config-name swe output_dir=${OUTPUT_DIR} wandb.wandb_workspace_root=/mnt/llmd/results/exps wandb.wandb_project_name=prl finetune.seq_parallel=4 finetune.seq_length=50000 swe.enable_expert_reward=true world.expert_llm.enabled=true swe.expert_models.1.base_url=\"${EXTERNAL_VLLM_URL}\" swe.expert_models.1.api_key=\"${EXTERNAL_VLLM_TOKEN}\""
