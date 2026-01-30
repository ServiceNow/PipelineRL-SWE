# Changes Overview (PipelineRL vs PipelineRL-SWE)

This is a high-level summary of all repo differences between `/home/toolkit/PipelineRL` (original) and `/home/toolkit/PipelineRL-SWE` (current). It is derived from a full recursive diff. I’ve excluded `.git/` metadata from the discussion below.

## Config changes
- `conf/base.yaml`
  - `actor.llm_max_rollouts` reduced (64 → 32).
  - `actor.shared_memory_entry_size` increased (10,000,000 → 50,000,000).
  - Added `agent.max_prompt_length`.
  - Added `world.fixed_gpus` block (`actor/preprocessor/finetune/expert` with null defaults).
- `conf/finetune/base.yaml`
  - `learning_rate` reduced (1e-6 → 5e-7).
  - Added `rl.performance_value_loss_coef` default (0.0).
- `conf/finetune/actor_critic.yaml`, `conf/finetune/ppo.yaml`
  - Added `rl.performance_value_loss_coef`.
- New config files in SWE only:
  - `conf/localization.yaml`
  - `conf/swe.yaml`

## New “expert LLM” support and pipeline integration
- `pipelinerl/launch.py`
  - Added `run_expert_llm(...)` and `expert_llm` job kind.
  - `launch_jobs` now supports `expert_llm` and uses a guarded `_extend` helper for process lists.
- `pipelinerl/world.py`
  - Adds expert GPU reservation and job placement (`expert_llms` / `expert_llm`).
  - Adds `get_expert_llm_url(s)` helpers.
  - Changes GPU placement order (see “GPU placement” section below).
- `pipelinerl/actor.py` and `pipelinerl/rollouts.py`
  - Rollout flow now accepts optional `expert_llms`.
  - Captures `expert_reward` and `performance_targets` in rollout results.

## Performance-value head + targets (multi-head value modeling)
- `pipelinerl/finetune/value_model.py`
  - Adds a new `PerformanceValueHead` and `performance_value` output in model output.
  - Loads/saves `performance_value_head` weights (`performance_value_head.pt`).
  - `AutoModelForCausalLMWithValueHead` now supports `performance_value_dim`.
- `pipelinerl/finetune/rl/__init__.py`
  - Adds `performance_value_loss_coef` config.
  - Adds performance head training loss using last prompt token mask.
  - Adds `performance_targets` to RL data columns.
  - Updates group-id handling and adds debug logging for group variance.
- `pipelinerl/finetune/data.py`
  - Adds `performance_targets` and `expert_reward` propagation.
  - Supports variable `performance_value_dim` and padding for nested targets.
- `pipelinerl/finetune/types.py`
  - Adds `performance_targets` field and validator.
- `pipelinerl/finetune/utils.py`
  - Adds `performance_value_dim` to sentinel batch/example creation.
- `pipelinerl/finetune/checkpoints.py`
  - Loads value head with `performance_value_dim` (for performance head).
- `pipelinerl/finetune_loop.py`
  - Uses `get_performance_value_dim(cfg)` and passes it into model loading.

## Actor/metrics enhancements
- `pipelinerl/actor.py`
  - Adds abstention stats based on self-eval predictions and configurable threshold.
  - Tracks `repair_expert_reward` metrics.
  - Supports expert LLMs for rollouts and waits for their servers.
  - Dataset loader now allows separate `test_dataset_path` override.

## Networking / observability changes
- `pipelinerl/torch_utils.py`
  - Much more detailed logging for `StatelessProcessGroup` init: env vars, DNS resolution, local IPs.
  - Warns if resolved address doesn’t match local interfaces; logs bind failures explicitly.
- `pipelinerl/utils.py`
  - `wait_for_inference_servers` now uses timeouts and logs DNS/connectivity diagnostics on failure.
- `pipelinerl/vllm0.py`, `pipelinerl/vllm1.py`
  - Adds hostname logging in `init_actor_update_group`.

## GPU placement / scheduling changes
- `pipelinerl/world.py`
  - Original: reverse node ordering, inference placement first, finetune last.
  - Current: ascending node ordering, **finetune placed first**, inference later.
  - Adds expert GPU reservation which affects available GPU pool.
  - Adds `fixed_gpus` handling (all stages + expert) with validation.

## Miscellaneous code changes
- `pipelinerl/finetune/lora.py`
  - Import path changed to `from peft import get_peft_model`.
- New module in SWE only:
  - `pipelinerl/config_utils.py`
- New directories / artifacts in SWE only:
  - `pipelinerl/swe/`, `pipelinerl/tokenizers/`, `results/`, `wandb/`, `.vscode/`
  - `stream_to_dataframe.py`, `tapedata.sqlite`, `vllm_oss120b.yaml`, `pipelinerl.egg-info/`

## Launch/finetune environment handling (non-TCPStore-specific)
- `pipelinerl/launch.py`
  - Adds `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` injection for finetune env.
  - Adds extra logging for finetune placement.

## Notes about TCP/port issues
- The only differences that *directly* alter the TCPStore endpoint or its binding are in `launch.py` (which we already corrected) and in environment variables. Most other changes are placement/logging or expert/performance-value functionality.

---
If you want a per-file, low-level diff summary (or to include `.git/` metadata), tell me and I can generate an expanded version.
