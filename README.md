# Pipeline RL

[![Github](https://img.shields.io/badge/HF%20Blog%20Post-0000)](https://huggingface.co/blog/ServiceNow/pipelinerl/)

A scalable asynchronous reinforcement learning implementation with in-flight weight updates. Designed to maximize GPU utilization while staying as on-policy as possible.

<p align="center">
    <img src="assets/figure1.jpg" alt="Pipeline-RL Architecture" width="600">
</p>

## Why PipelineRL?

PipelineRL tackles the classic trade-off between **inference throughput** (large batches on many GPUs) and **on-policy data freshness** by performing _inflight weight updates_. After each optimizer step, updated weights are broadcast to the inference servers without halting sampling. This keeps batch sizes optimal and data near on-policy, yielding fast, stable RL for large language models.

<p align="center">
    <img src="assets/losses.png" alt="Pipeline-RL Effectiveness" width="600">
</p>

- In experiments on 7B and 32B models (batch size 4096, lr=1e-6, max tokens=8192), PipelineRL matches or exceeds Open-Reasoner-Zero on AIME-2024 and MATH-500.
- Uses a simplified GRPO algorithm: no value network, no trust-region clamping, no KL or entropy bonuses by default (though KL support is available).





## Setup

Clone the repository and change the directory to `pipelinerl`
```bash
git clone git@github.com:ServiceNow/PipelineRL.git
cd PipelineRL
```

Create the environments with dependencies.
```bash
conda create -n pipeline-rl -y python=3.11
conda run --no-capture-output -n pipeline-rl pip install torch==2.6.0 
conda run --no-capture-output -n pipeline-rl pip install -e . --no-build-isolation
```

By default Pipeline-RL will use the file system as the medium for streaming the generated data to the trainer processes. This works on one node, but the files can get quite large. To use Redis instead you will need to install the Redis server in the same conda environment:
```bash
conda install redis-server==7.4.0 -c conda-forge 
```

## Run experiments

First, activate the conda environment:
```bash
conda activate pipeline-rl
```

Single node with 8 H100 GPUs:

```bash
python -m pipelinerl.launch output_dir=results/base1
```

If you only have 4 H100 GPUs:
```bash
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1 
```

To use Redis instead of the filesystem for data streaming:
```
python -m pipelinerl.launch streams=redis output_dir=results/base1
```


## Architecture and pipeline stages

<p align="center">
    <img src="assets/structure.jpg" alt="Pipeline-RL Structure" width="600">
</p>

PipelineRL is organized as a modular, Hydra-driven pipeline with 6 core components driving 3 main stages of the RL training: actor, verifier and trainer. Below is a code-grounded mapping of each component:

### 1. Orchestrator
- File: `pipelinerl/launch.py`
- Entrypoint: `@hydra.main(...) def main(cfg)`
- Responsibilities:
  - Parse & validate the Hydra config, initalize directories, set up logging and streams backend.
  - Build a **WorldMap** (in `pipelinerl/world.py`) for rank-aware job & GPU placement:
    - Reads environment variables `WORLD_SIZE`, `RANK`, and `MASTER_ADDR` to determine cluster topology.
    - Computes `gpus_per_llm` from tensor/pipeline parallel settings and allocates each node’s GPUs into actor, preprocessor, and trainer pools based on `cfg.world.*_fraction`.
  - Creates Job entries for all roles: `actor_llm`, `preprocessor_llm`, `actor`, `preprocessor`, `verifier`, and `finetune`.
  - Launch subprocesses through `launch_jobs(...)`, which invokes:
    - `run_ref_llm` → Reference LLM servers for KL penalties.
    - `run_actor_llm` → Actor LLM servers for policy sampling.
    - `run_actor` → Actor processes generating raw rollouts.
    - `run_preprocess` → Preprocessor workers computing advantages & reference log-probs.
    - `run_finetune` → Trainer workers updating weights via Accelerate, DeepSpeed, or FSDP.
    - `run_verifier` → Optional verifier servers for final reward checks.

### 2. Inference servers
- **Reference LLMs**: spawned by `run_ref_llm` (in `launch.py`), running `vllm.entrypoints.openai.api_server` to serve reference log-probs.
- **Actor LLMs**: launched via `run_actor_llm` → `pipelinerl/entrypoints/llm.py` → `pipelinerl/run_llm.py`:
  - Subclasses vLLM’s `Worker` to add:
    - `init_actor_update_group(...)` for NCCL process-group setup.
    - `receive_weight_update(request)` to pause inference, broadcast new weights via NCCL, and reload model parameters.
  - Exposes HTTP endpoints:
    - `POST /v1/chat/completion` for sampling.
    - `POST /receive_weight_update` for weight updates.

### 3. Actor processes
- Entrypoint: `pipelinerl/entrypoints/actor.py`
- Setup & initialization:
  - Load train/test datasets via `load_datasets`.
  - Wait for inference servers (`wait_for_inference_servers`) and optional verifier (`wait_for_verifier`).
  - Initialize `TrainerState(exp_path)`, start listening for weight updates, and block until the first model version arrives.
- Rollout scheduling (`ActorLoop` & `rollout_maker_entrypoint`):
  - `ActorLoop` creates `problem_queue` and `result_queue`, then spawns multiple worker processes (via `mp.Process`) to run `rollout_maker_entrypoint`.
  - Each worker process:
      - Sets up a uvloop-based asyncio event loop.
      - Listens for weight‐update broadcasts via `TrainerState` to get model version.
      - Calls `schedule_rollouts(cfg, attempts, problem_queue, result_queue, trainer_state, llms, name)`, which:
          * Pulls problems from `problem_queue` (random sampling for training, sequential for testing).
          * For each GRPO group, issues exactly `cfg.attempts` concurrent HTTP calls to Actor LLM servers (`generate_math_rollout`).
          * Collects `RolloutResult` objects (texts, log-probs, rewards, latencies) and pushes the full batch into `result_queue` once all attempts complete.
- Writing and stats (`ActorLoop.run`):
  - On each generator step:
      * Update allowed outstanding groups if a new `propagated_weight_version` arrived.
      * Refill `problem_queue` up to the lag-controlled limit (`cfg.finetune.max_lag` / `cfg.attempts`).
      * Read one batch of `RolloutResult` from `result_queue`.
      * Write each sample dict to the `actor` stream.
      * Aggregate prompt/output token counts, rewards, and success metrics via a sliding window (`SlidingWindowAggregator`) and write stats to the `stats` stream and WANDB.
  - Training loops run indefinitely; test loops stop after one pass.
- Evaluation & backpressure:
  - `run_actor_loop` can pause training scheduling to run a one-shot test loop (`is_training=False`), based on `cfg.eval_every_n_versions`.
  - Scheduling backpressure is controlled via `cfg.finetune.max_lag` and `cfg.finetune.weight_update_interval`, ensuring on-policy data freshness.

### 4. Preprocessor
- Entrypoint: `pipelinerl/entrypoints/preprocess.py`
- Workflow:
  - `run_dataset_loader` (thread) reads raw actor traces in chunks from the input stream.
  - `ProcessPoolExecutor` workers run `process_chunk(...)`, which:
    - Tokenizes and preprocesses sequences.
    - Optionally attaches reference log-probs.
  - Writes processed micro-batches to `StreamRangeSpec(topic=cfg.preprocess.output)`.

### 5. Trainer (fine-tuner)
- Entrypoint: `pipelinerl/entrypoints/finetune.py`
- Loop structure:
  - Creates the input stream to consume preprocessed batches.
  - Background threads:
    1. `run_sample_loader` reads JSON micro-batches from the input stream into a local queue.
    2. `run_fixed_batch_data_loader` or `run_dynamic_batch_size_data_loader` collates samples into PyTorch tensors.
  - Main training loop:
    - Pull a batch → call `rl_step(...)` (in `pipelinerl/finetune/rl/utils.py`) to compute policy-gradient (+ KL penalty if configured) → `optimizer.step()` → `lr_scheduler.step()`.
    - On rank 0, use `WeightUpdateManager.send_weight_update(version)` to gather model parameters, send `WeightUpdateRequest` to Actor LLMs (HTTP), broadcast tensors via NCCL, and write a `WeightUpdateSuccess` message to the update stream.

### 6. Verifier
- Entrypoint: `pipelinerl/entrypoints/verifier.py`
- Serves a FastAPI app with:
  - `POST / `: checks model outputs (math or countdown puzzles) via `math_verify` or `countdown_utils`.
  - `GET /health`: readiness probe.

### Streams backend
- Defined in `pipelinerl/streams.py`.
- Implements `SingleStreamSpec` and `StreamRangeSpec` for file-system or Redis-based queues.
- `write_to_streams(...)` and `read_stream(...)` provide a JSON-line protocol for inter-process messaging.
- Available backends:
  - File system: default.
  - Redis: requires Redis server.

### Streams & Queues
- `problem_queue` (multiprocessing.Queue): produced by `ActorLoop.run` to hold raw problems; consumed by rollout worker processes in `rollout_maker_entrypoint` via `schedule_rollouts`.
- `result_queue` (multiprocessing.Queue): produced by rollout workers (lists of `RolloutResult`); consumed by `ActorLoop.run` to publish completed rollouts.
- `actor` stream (SingleStreamSpec(topic="actor")): file- or Redis-backed stream. Produced by `ActorLoop.run` writing each sample dict; consumed by the Preprocessor stage (configured via `cfg.preprocess.input`).
- `training_data` stream (StreamRangeSpec(topic="training_data")): File- or Redis-backed stream used to transfer processed training micro-batches from the Preprocessor to the Trainer. Configured via `cfg.preprocess.output` and `cfg.finetune.input` (defaulting to "training_data") in `conf/base.yaml`. Written in `pipelinerl/run_preprocess.py` and consumed in `pipelinerl/run_finetune.py`.
- `actor_test` and `stats_test` streams: analogous streams used for evaluation loops (test samples and test metrics).
- `stats` stream (SingleStreamSpec(topic="stats")): produced by `ActorLoop.publish_stats` with sliding-window metrics; consumed by external monitoring (e.g. WANDB, logging viewers).
