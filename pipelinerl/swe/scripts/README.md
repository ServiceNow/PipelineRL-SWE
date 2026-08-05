# pipelinerl/swe/scripts

Python scripts for SWE-bench data collection, router training, eval, and analysis.
Companion bash launchers live in `launchers/`.

## Directory map

| Directory | Purpose |
|-----------|---------|
| `offline_router/` | **Main workspace.** Offline router training pipeline: collect traces, materialize datasets, train Qwen embedding routers / CoT abstention predictors, eval via Daytona. See [`offline_router/README.md`](offline_router/README.md). |
| `new/` | Runtime trace analysis: score live actor traces with the policy value head, run ridge probes, analyze routing utility. See [`new/README.md`](new/README.md). |
| `openrouter_sweep/` | OpenRouter diversity sweep (15 models × SWE-Smith): collect, eval, analyze cross-model complementarity. |

## Root-level scripts

These are shared utilities and older standalone evaluators used by scripts in the subdirectories.

| Script | Purpose |
|--------|---------|
| `repair_eval_utils.py` | **Shared utility library** (prompt building, OpenAI-compatible HTTP calls, reward computation). Imported by most collection scripts in `offline_router/`. |
| `run_actor_repair_eval.py` | Run the primary actor (policy) on a SWE eval set asynchronously. Produces `repair_output` predictions. |
| `run_expert_repair_eval.py` | Run an expert model on a SWE eval set. Used to obtain expert route predictions. |
| `run_value_handoff.sh` | Shell wrapper for the value-handoff eval run. |
| `analyze_actor_self_eval.py` | Analyze pass@k and reward distributions from actor self-eval outputs. |
| `analyze_handoff.py` | Analyze actor→expert handoff decisions from router trace data. |
| `compare_self_eval_runs.py` | Compare two self-eval runs (e.g., before vs. after fine-tune). |
| `test.py` | Ad-hoc testing/debugging script. |

## Quick-start pointers

- **Train an abstention predictor** (predict scout success from CoT trace):
  → `launchers/abstention/launch_cot_abstention_predictor.sh`
  → `offline_router/train_cot_abstention_predictor.py`

- **Eval SWE-Smith with Daytona** (real pass/fail labels):
  → `offline_router/run_swesmith_eval_daytona.py`

- **Eval SWE-bench Verified with Daytona** (official benchmark):
  → `offline_router/run_swebench_eval_daytona.py`
  → `launchers/abstention/launch_verified_real_label_eval.sh`

- **Train a multi-route router** (SWE-Smith real labels, Qwen3-Embedding-8B + LoRA):
  → `launchers/offline_router/swe_smith_real/launch_offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch.sh`
  → `offline_router/train_qwen_embedding_cascade_baseline.py`
