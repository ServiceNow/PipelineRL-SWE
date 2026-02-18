TIMESTAMP=$(date +%s)
JOB_NAME=multi_expert_multi_env_new_${TIMESTAMP}
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME}_${TIMESTAMP} 
make multi-replica-job REPLICAS=2 JOB_NAME=${JOB_NAME}_${TIMESTAMP} ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 NPROC=8 COMMAND="cd /home/toolkit/PipelineRL-SWE; PIPELINERL_FINETUNE_QUIET_NON_MAIN=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TORCH_NCCL_DUMP_ON_TIMEOUT=1 python -m pipelinerl.launch --config-dir conf --config-name swe output_dir=${OUTPUT_DIR} wandb.wandb_workspace_root=/mnt/llmd/results/exps wandb.wandb_project_name=prl finetune.seq_length=16000 swe.enable_expert_reward=true"
