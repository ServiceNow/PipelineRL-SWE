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
- or the wrapper `launchers/offline_router/launch_offline_router_collect.sh`

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
bash launchers/offline_router/launch_offline_router_collect.sh
```

For the SWE-smith in-distribution control, use the one-node launcher from the
repo root:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id.sh
```

It collects router-train traces from `/mnt/llmd/data/swe_smith/ds_train`, router
eval traces from `/mnt/llmd/data/swe_smith/ds_test`, writes
`collect/route_distribution_summary.json`, then trains the text reward-vector
LoRA router with random train sampling. Override `TRAIN_MAX_SAMPLES`,
`EVAL_MAX_SAMPLES`, `MAX_TRAIN_ROWS`, `MAX_EVAL_ROWS`, or `RUN_TRAIN=0` for a
collection-only run.

If collection already finished and only router training needs to be retried, run:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_train_only.sh
```

For the 3-way generalization setup, first materialize official SWE-bench train
into the local format expected by collection, if it is not already present:

```bash
bash prepare_swe_bench_train_local.sh
```

The preparation wrapper reconstructs touched-file contents from GitHub raw-file
URLs by default, samples `MAX_NORMALIZED_ROWS=6000` rows before reconstruction,
and applies `MAX_TOTAL_TOKENS=16000` to avoid overlong collection prompts.
Override these variables if a larger router-train set is needed. Set
`GOLD_FILE_SOURCE=git` to use repo clones instead of raw-file fetches.

Then collect router-train traces from SWE-bench train and router-test traces
from the existing SWE-bench Lite dataset:

```bash
bash launchers/offline_router/launch_offline_router_swe_bench_router_split_collect.sh
```

This launcher is collection-only. It writes
`collect/route_distribution_summary.json` and stops before router training.
Override `TRAIN_DATASET_PATH`, `EVAL_DATASET_PATH`, `TRAIN_MAX_SAMPLES`, or
`EVAL_MAX_SAMPLES` as needed.

For the independent scalar text ablation, which trains one numeric reward example
per `(problem, route)` and merges scalar predictions back into route vectors for
the same metrics, run:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_scalar_train_only.sh
```

For the independent reward-bin ablation, which trains one bin-label example per
`(problem, route)` and evaluates by softmaxing over all bin labels to compute an
expected reward, run:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_bin_train_only.sh
```

For a matched 20-interval reward-grid comparison, use the `20bucket` launchers.
These put all three text modes on the same `0.00, 0.05, ..., 1.00` training
target grid:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_vector_20bucket_train_only.sh
bash launchers/offline_router/launch_offline_router_swe_smith_id_scalar_20bucket_train_only.sh
bash launchers/offline_router/launch_offline_router_swe_smith_id_bin_20bucket_train_only.sh
```

For the reverse-label diagnostic, keep the same 21 reward values but assign them
to letters in descending order (`A=1.00, ..., U=0.00`):

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_bin_reverse_20bucket_train_only.sh
```

To estimate forward-bin seed variability, run three sequential `bin20` trainings
inside one job. Override `ROUTER_SEEDS` for a different seed list:

```bash
bash launchers/offline_router/launch_offline_router_swe_smith_id_bin_20bucket_multiseed_train_only.sh
```

To inspect whether numeric reward strings are single tokenizer tokens:

```bash
conda run --no-capture-output -n pipeline-rl python -m pipelinerl.swe.scripts.offline_router.probe_reward_tokenization \
  --model-path /mnt/llmd/results/exps/aristides/reason/swe_smith_policy_conditioned_no_devstral_1773812579/finetune/current \
  --output-json router_analysis/reward_tokenization_policy_checkpoint.json
```

`launchers/offline_router/launch_offline_router_collect.sh` reserves a single 8-GPU node by default and starts:

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
bash launchers/offline_router/launch_offline_router_train.sh
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

## 3) Compare Router Calibration

To compare completed offline-router training runs, use:

```bash
conda run --no-capture-output -n pipeline-rl python -m pipelinerl.swe.scripts.offline_router.report_router_calibration \
  --run vector=/path/to/vector_train_dir \
  --run scalar=/path/to/scalar_train_dir \
  --run bin=/path/to/bin_train_dir \
  --output-dir router_analysis/offline_router_calibration
```

The report writes:

- `run_summary.csv`
- `route_metrics.csv`
- `pairwise_metrics.csv`
- `collapse_stats.csv`
- `reliability_buckets.csv`
- `decision_summary.csv`
- `calibration_report.json`

---

## 4) OpenRouter Diversity Sweep (Set 1)

Test cross-family model complementarity: how correlated are solve patterns across 15 OpenRouter models? Low correlation = more routing opportunity.

```bash
# Step 1: collect predictions from 15 models via OpenRouter API
export OPENROUTER_API_KEY=sk-...
bash launchers/openrouter_sweep/launch_collect.sh

# Step 2: run Daytona eval on all model predictions (one job per model)
PREDICTIONS_DIR=/mnt/llmd/results/.../openrouter_sweep_collect_XYZ \
bash launchers/openrouter_sweep/launch_daytona.sh

# Step 3: analyze correlations and routing headroom
python pipelinerl/swe/scripts/openrouter_sweep/analyze_openrouter_sweep.py \
  --daytona-log-dir logs/run_evaluation \
  --run-id-prefix or_sweep \
  --output-dir /mnt/.../analysis
```

Output: `phi_correlation_matrix.png`, `routing_headroom_matrix.png`, `resolve_rates.png`, `sweep_summary.json`.

## 5) Autoregressive CoT Verifier (Set 2)

Hypothesis: a Thinking-capable small model (4B) can learn to predict patch success by reasoning over (problem + task CoT + patch), outperforming our embedding-based approach.

```bash
# Step 1: collect CoT trajectories with Qwen3-4B-Thinking-2507
bash launchers/abstention/launch_cot_collection.sh

# Step 2: run Daytona eval on predictions_train.jsonl
# (standard Daytona job — see run_swesmith_eval_daytona.py)

# Step 3: train the verifier
TRAJECTORIES_DIR=/mnt/.../cot_trajectories_XYZ \
TRAIN_DAYTONA_REPORT=logs/run_evaluation/<run_id>/report.json \
bash launchers/offline_router/analysis/launch_autoregressive_verifier_train.sh
```

Key scripts:
| Script | Purpose |
|---|---|
| `collect_cot_trajectories.py` | Run Qwen3-4B-Thinking via local vLLM; extracts `<think>` blocks into `thinking_text` column |
| `train_autoregressive_verifier.py` | SFT fine-tune a causal LM on (problem + task thinking + patch) → Yes/No |

---

## 7) Dataset & Label Reference

> **Rule**: proxy labels must be avoided or explicitly flagged. To detect: check `route_rewards`
> unique values — binary `{0.0, 1.0}` means real Daytona pass/fail; continuous floats mean proxy
> reward-model scores. The column name alone is NOT reliable.

### SWE-Smith traces

| Dataset | Scout model | CoT? | Label type | Success rates (per route) | Path |
|---------|-------------|------|------------|--------------------------|------|
| `instruct_patches_trajectories_1785884242` | Qwen3-4B-**Instruct**-2507 | No | **REAL** (join with 4-route parquet, route 0) | — (trace only) | `.../instruct_patches_trajectories_1785884242/` |
| `cot_trajectories_1785341592` (and siblings) | Qwen3-4B-**Thinking** | Yes | No labels embedded — join with 4-route parquet | — (trace only) | `.../cot_trajectories_1785341592/` |
| **4-route real-label parquet** | Routes 0–3 (see below) | — | **REAL** (Daytona binary 0/1) | [25%, 40%, 40%, 46%] | `.../offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/{train,eval}/` |

4-route parquet route index → model:
- Route 0: `Qwen/Qwen3-4B-Instruct-2507` (scout/instruct)
- Route 1: `openai/gpt-oss-20b`
- Route 2: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Route 3: `openai/gpt-oss-120b`

### SWE-bench Verified traces

| Dataset | Scout model | CoT? | Label type | Notes | Path |
|---------|-------------|------|------------|-------|------|
| `verified_zeroshot_nocot` | Qwen3-4B-**Instruct**-2507 | No | **PROXY** (~10%) in `labels_verified.parquet` | Real labels pending (Daytona route-0 job) | `.../verified_zeroshot_nocot/` |
| 5-route collection parquet | Routes 0–4 (see below) | — | **PROXY** (continuous reward model) | Rates ~10–32% vs real 41–53% — do not use as ground truth | `.../offline_router_swe_bench_...5route_.../collect/eval/` |
| **Verified Daytona eval** (`verified_real_label_eval_1785965509`) | Routes 1–3 | — | **REAL** | Route 0 (instruct) pending | `logs/run_evaluation/predictions_route_{1,2,3}_1785965509/` |

5-route Verified parquet route index → model:
- Route 0: `Qwen/Qwen3-4B-Instruct-2507`
- Route 1: `openai/gpt-oss-20b`
- Route 2: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Route 3: `openai/gpt-oss-120b`
- Route 4: `google/gemini-3-flash-preview`

Real Daytona results (routes 1–3): 151 / 153 / 194 resolved out of 369.

### MATH traces

| Dataset | Model | CoT? | Label type | Notes | Path |
|---------|-------|------|------------|-------|------|
| `math_cot_trajectories_1785795546` | Qwen3-4B-**Thinking** | Yes | **REAL** (`scout_correct`, deterministic grading) | 65.6% correct on eval (328/500) | `.../math_cot_trajectories_1785795546/` |

### Trained abstention predictors

| Run | Train domain | Scout traces | Label | Best AUC |
|-----|-------------|-------------|-------|----------|
| `cot_route0_1785909531` | MATH | 4B-Thinking CoT | `scout_correct` (real) | **0.9533** |
| `no_cot_input_only_route0_1785944036` | MATH | 4B-Thinking CoT | `scout_correct` (real) | **0.8855** |
| `cot_1785364718` | SWE-Smith | 4B-Instruct (route_outputs[0]) | route-3 real | 0.7246 |
| `no_cot_1785518862` | SWE-Smith | 4B-Instruct (route_outputs[0]) | route-3 real | 0.7135 |
| `no_cot_route3_1785884243` | SWE-Smith | 4B-Instruct (route_outputs[0]) | route-3 real | 0.6970 |

High AUCs (~0.95) are on MATH, not SWE-Smith. SWE-Smith peaks at ~0.72. Do not conflate.
The SWE-Smith predictors labelled "cot" use Instruct-4B traces (from route_outputs[0] in the parquet),
not Thinking/CoT traces — the CoT traces exist in `cot_trajectories_*` but abstention predictors
have not yet been trained on them.

---

## 6) CoT Abstention Predictor (Set 3 — current main track)

### Operating mode

The **4B scout output is never submitted as a final answer**. The router makes a binary decision for each instance:

- **Call oss-120b** — if P(oss-120b succeeds) is above threshold → pay for the strong model, submit its patch
- **Abstain completely** — if P(oss-120b succeeds) is below threshold → submit nothing (empty patch)

The 4B scout runs cheaply first; its output is used only as input features for the routing decision. The predictor label is always **oss-120b success** (route 3), not scout success.

### Predictor inputs

Two input modes:
- **Post-primary (PS)**: input = problem + scout patch (+ thinking trace if CoT scout). Gives the model's attempt as a signal for task tractability.
- **Input-only (IO)**: input = problem statement only. Isolates task-difficulty signal; harder but more robust baseline.

```bash
# Step 1: collect scout CoT trajectories on SWE-Smith
bash launchers/abstention/launch_cot_collection.sh

# Step 2: eval those trajectories to get real pass/fail labels
bash launchers/abstention/launch_verified_real_label_eval.sh  # for SWE-bench Verified
# or use the existing SWE-Smith real-label collection at:
# /mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_*/

# Step 3: train the predictor
bash launchers/abstention/launch_cot_abstention_predictor.sh         # post-primary (uses scout trace)
INPUT_ONLY=true bash launchers/abstention/launch_cot_abstention_predictor.sh  # input-only

# MATH domain input-only baseline:
MATH_OUTPUT_DIR=/mnt/... bash launchers/abstention/launch_math_input_only_abstention_predictor.sh

# Transfer experiment (train on SWE-Smith 4B-Instruct, score zero-shot on Verified):
# Traces already collected; no vLLM needed.
bash launchers/abstention/launch_swe_smith_instruct_abstention_transfer.sh

# Step 4: score predictions (zero-shot on a held-out domain)
# --parquet-dir for proxy/parquet labels; --real-labels-jsonl for Daytona .results.jsonl
python pipelinerl/swe/scripts/offline_router/score_cot_abstention_predictor.py \
  --checkpoint-dir /mnt/.../checkpoints/epoch_NNNN \
  --train-config   /mnt/.../train_config.json \
  --trajectories   /mnt/.../trajectories.jsonl \
  --real-labels-jsonl /mnt/.../predictions_route_3.results.jsonl \
  --output-path    /mnt/.../predictions.jsonl

# Step 5: analyze abstention curves
python -m pipelinerl.swe.scripts.offline_router.analyze_cot_verifier_abstention \
  --scores-jsonl /mnt/.../scores.jsonl
```

MATH results (Qwen3-Embedding-8B, LoRA r32, 10 epochs):
- Post-primary AUC: **0.953** (learns whether scout got it right from trace)
- Input-only AUC: **~0.885** (task-difficulty signal only)

---

## Script Reference

### Data Collection
| Script | Purpose |
|---|---|
| `run_collection_job.py` | Hydra launcher — main entry point for collection runs |
| `collect_router_dataset.py` | Core collection loop: runs models via HTTP, writes sharded parquet |
| `collect_cot_trajectories.py` | Collect scout CoT thinking traces + patches from SWE-Smith via local vLLM |
| `collect_math_cot_trajectories.py` | Same but for MATH domain trajectories |
| `collect_verified_cot_trajectories.py` | Collect scout traces on SWE-bench Verified instances |
| `collect_self_assessment.py` | Collect scout self-assessments (does the model think it succeeded?) |
| `collect_model_discovery_candidates.py` | Collect outputs for model-discovery (deciding which models to include) |
| `collect_openrouter_expert_from_existing.py` | Re-run an OpenRouter expert on already-collected instances |
| `recompute_openrouter_expert_rewards.py` | Recompute rewards for OpenRouter outputs after reward fn change |
| `reparse_model_discovery_outputs.py` | Re-parse raw model-discovery outputs after format change |

### Dataset Building & Export
| Script | Purpose |
|---|---|
| `materialize_real_label_router_dataset.py` | **Primary dataset builder.** Merges real eval labels into collection parquet, splits train/eval |
| `materialize_multirollout_verifier_dataset.py` | Multi-rollout variant (multiple 4B attempts per instance) |
| `materialize_active_real_label_verifier_subset.py` | Active-learning subset: pick maximally uncertain instances |
| `materialize_trace_cost_router_dataset.py` | Variant with trace-level cost features |
| `augment_router_dataset_with_file_context.py` | Add repo file context to an existing dataset |
| `build_router_collection_from_discovery.py` | Assemble a route collection from model-discovery candidates |
| `splice_route_collections.py` | Merge two route collections side-by-side (add a new expert route) |
| `splice_router_collections.py` | Merge two router datasets with different instance sets |
| `export_real_label_subset_from_router_collection.py` | Export a named subset with real labels |
| `export_swe_smith_real_eval_package.py` | Package the 286-instance SWE-smith eval set for eval runners |
| `export_swe_smith_multirollout_real_eval_package.py` | Same for the 150-instance multirollout eval set |
| `repackage_swe_smith_eval_with_loose_patches.py` | Repackage predictions with loosened patch matching |
| `repair_swe_smith_eval_patch_hashes.py` | Fix fake `index 0000000..1111111` lines in LLM patches with real git hashes |
| `filter_swe_smith_bugged_context_dataset.py` | Filter local `swe_smith_bugged_context` HF dataset to relevant subset |
| `validate_swe_smith_bugged_context.py` | Sanity-check bugged_context instances against SWE-smith-py |
| `sample_instance_ids_from_pool.py` | Sample N instance IDs from a pool |
| `sample_real_label_instance_ids.py` | Sample from instances that have real eval labels |
| `summarize_collected_dataset.py` | Print summary statistics for a collected dataset directory |
| `export_parquet_ids.py` | Export instance IDs from a parquet dataset to a text file |
| `extract_instruct_patches_as_trajectories.py` | Extract patches from instruct-mode outputs and format as trajectories |
| `extract_verified_route_predictions.py` | Extract route predictions from SWE-bench Verified evaluation outputs |

### Evaluation
| Script | Purpose |
|---|---|
| `run_swesmith_eval_daytona.py` | **SWE-Smith eval runner.** Daytona sandboxes, ~7¢/200 instances. |
| `run_swebench_eval_daytona.py` | **SWE-bench Verified eval runner.** Images from `ghcr.io/epoch-research/swe-bench.eval.x86_64.*`. |
| `run_claude_api_eval.py` | Run Claude API (Haiku/Sonnet/Opus) inference on eval set. Resume-safe. |
| `score_qwen_embedding_cascade_verifier.py` | Run a trained Qwen embedding verifier, save per-instance scores |
| `score_cot_abstention_predictor.py` | Score a trained CoT abstention predictor on held-out instances |
| `score_autoregressive_verifier.py` | Score a trained autoregressive verifier (causal LM) |
| `score_autoreg_verifier.py` | Alternate scoring entrypoint for autoregressive verifier |

### Training
| Script | Purpose |
|---|---|
| `train_cot_abstention_predictor.py` | **Abstention predictor training.** Qwen3-Embedding-8B + LoRA, BCE head. `--input-only` flag for problem-only mode, `--include-thinking` for scout trace. |
| `train_qwen_embedding_cascade_baseline.py` | Multi-route router training. LoRA fine-tune, BCE reward head. Supports `post_primary` (PS) and `input_only` (IO) modes. |
| `train_qwen_embedding_router_baseline.py` | Frozen-encoder variant (trained head only, no LoRA) |
| `train_qwen_embedding_state_policy.py` | State-policy encoder (value-function framing) |
| `train_qwen_embedding_listwise_verifier.py` | Listwise ranking loss variant |
| `train_modernbert_router_baseline.py` | ModernBERT baseline (comparison model) |
| `train_autoregressive_verifier.py` | SFT fine-tune a causal LM on (problem + thinking + patch) → Yes/No |
| `train_router_offline.py` | Older offline router training script (pre-Qwen embedding era) |

### Reporting & Analysis
| Script | Purpose |
|---|---|
| `calibrate_real_router_from_predictions.py` | Platt/isotonic calibration of raw router scores against real labels |
| `sweep_threshold_utility.py` | Sweep abstention threshold, report utility at each operating point |
| `report_cost_aware_cascade_utility.py` | Routing utility accounting for per-route costs |
| `report_weighted_cascade_utility.py` | Weighted quality×cost utility report |
| `report_direct_router_weighted_cost_utility.py` | Direct-routing utility (no cascade) |
| `report_router_calibration.py` | Calibration curves: predicted vs actual success probability |
| `report_delta_bucket_calibration.py` | Calibration by predicted Δ-reward bucket |
| `report_swe_smith_package_proxy_similarity.py` | Proxy vs real label similarity in eval package |
| `simulate_active_verifier_labeling.py` | Simulate active-learning labeling curves |
| `probe_reward_tokenization.py` | Debug reward tokenization edge cases |
| `analyze_multirollout_controller_pathways.py` | Controller decision pathways in multirollout setting |
| `analyze_multirollout_marginal_value_bins.py` | Marginal value of additional rollouts by difficulty band |
| `analyze_multirollout_verifier_scores.py` | Verifier score distributions across routes |
| `analyze_verifier_calibration_diagnostics.py` | Detailed calibration diagnostics for trained verifier |
| `analyze_cot_verifier_abstention.py` | Abstention curves for CoT predictor: precision/recall vs. threshold, cost vs. quality trade-off |
