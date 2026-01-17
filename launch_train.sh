TIMESTAMP=$(date +%s)
JOB_NAME=multi_expert
OUTPUT_DIR=/mnt/llmd/results/exps/aristides/reason/${JOB_NAME} 
make multi-replica-job REPLICAS=2 JOB_NAME=${JOB_NAME}_${TIMESTAMP} ENV=prl CONDA_EXE=/opt/conda/bin/conda SNAPSHOT=1 NPROC=8 COMMAND="cd /home/toolkit/PipelineRL-SWE; python -m pipelinerl.launch --config-dir conf --config-name swe output_dir=${OUTPUT_DIR} wandb.wandb_workspace_root=/mnt/llmd/results/exps wandb.wandb_project_name=prl finetune.seq_parallel=4 finetune.seq_length=50000 swe.enable_expert_reward=true"