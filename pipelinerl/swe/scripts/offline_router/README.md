# Offline Router

This directory contains the offline router-supervision pathway. It is intentionally separate from the online PPO / actor / preprocess / finetune pipeline.

## 1) Collect a static supervision dataset

The collector writes Parquet shards to:

- `output_dir/train/*.parquet`
- `output_dir/eval/*.parquet`
- `output_dir/metadata.json`
- `output_dir/collection_config.json`

It reuses:

- prompt building from `pipelinerl.swe.scripts.repair_eval_utils.build_repair_messages`
- OpenAI-compatible chat calls from `pipelinerl.swe.scripts.repair_eval_utils.chat_completion`
- reward computation from `pipelinerl.swe.utils.repair_utils.calculate_precise_reward`

The plain collector entrypoint assumes the primary model and expert endpoints are already running and reachable over OpenAI-compatible HTTP APIs.

If you want one-node orchestration that starts the endpoints first and then collects, use:

- `pipelinerl.swe.scripts.offline_router.run_collection_job`
- or the wrapper `launch_offline_router_collect.sh`

Each route can now specify:

- `model_path`: checkpoint or HF path to load
- `served_model_name`: API name exposed by vLLM
- `model_name`: name sent by the collector in chat-completion requests

In the common case, `model_name` should equal `served_model_name`.

Example:

```bash
conda run --no-capture-output -n pipeline-rl python -m pipelinerl.swe.scripts.offline_router.collect_router_dataset \
  output_dir=/mnt/llmd/results/offline_router_collect_example \
  offline_router.primary_model.base_url=http://127.0.0.1:8000
```

Convenience launcher:

```bash
bash launch_offline_router_collect.sh
```

`launch_offline_router_collect.sh` reserves a single 8-GPU node by default and starts:

- primary-model vLLM on GPU `0`
- Devstral on GPU `1`
- GPT-OSS on GPUs `2,3,4,5`

Then it waits for `/health` on each endpoint before starting collection.

Notes:

- Experts default to `world.expert_llms[*].port` if `offline_router.expert_base_urls` is not set.
- Route order is fixed as `primary_model`, then experts sorted by `expert_rank`.
- Resume is shard-based: existing `problem_id`s already present in Parquet shards are skipped.

## 2) Train the router offline

The trainer reads the collected Parquet dataset and optimizes only the routing objective on the completion-last representation of `prompt_text + primary_output_text`.

Modes:

- `offline_router.train.mode=frozen_backbone`
  - train `performance_value_head` only
- `offline_router.train.mode=full_backbone`
  - train `pretrained_model + performance_value_head`
  - `value_head` stays frozen

Example:

```bash
conda run --no-capture-output -n pipeline-rl python -m pipelinerl.swe.scripts.offline_router.train_router_offline \
  output_dir=/mnt/llmd/results/offline_router_train_example \
  model_path=/mnt/llmd/results/exps/aristides/reason/some_run/finetune/current \
  offline_router.train.dataset_dir=/mnt/llmd/results/offline_router_collect_example \
  offline_router.train.mode=frozen_backbone
```

Convenience launcher:

```bash
DATASET_DIR=/mnt/llmd/results/offline_router_collect_example \
MODEL_PATH=/mnt/llmd/results/exps/aristides/reason/some_run/finetune/current \
bash launch_offline_router_train.sh
```

Outputs:

- `summary.json`
- `route_metrics.csv`
- `pairwise_metrics.csv`
- `eval_predictions.jsonl`
- `checkpoints/best/`
- `checkpoints/last/`

If `wandb.use_wandb=true`, the trainer also logs epoch losses and per-route / pairwise eval metrics to W&B.

`route_metrics.csv` includes per-route:

- `pearson`
- `spearman`
- `r2`
- `mse`
- `rmse`
- `mae`
- true/pred mean/std

`pairwise_metrics.csv` includes:

- delta Pearson
- delta Spearman
- delta MAE
- sign accuracy
- ROC-AUC
