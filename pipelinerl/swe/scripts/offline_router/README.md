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
| `verified_zeroshot_nocot` | Qwen3-4B-**Instruct**-2507 | No | **PROXY** (~10%) in `labels_verified.parquet` | Use real Daytona labels instead | `.../verified_zeroshot_nocot/` |
| 5-route collection parquet | Routes 0–4 (see below) | — | **PROXY** (continuous reward model) | Rates ~10–32% vs real 41–53% — do not use as ground truth | `.../offline_router_swe_bench_...5route_.../collect/eval/` |
| **Verified Daytona eval** (`verified_real_label_eval_1785965509`) | Routes 1–3 | — | **REAL** | Summary JSONLs reconstructed from per-instance log reports (old summaries were 0/369 due to pre-fix bug) | `.../verified_real_label_eval_1785965509/predictions/predictions_route_{1,2,3}.results.jsonl` |
| **Verified Daytona eval** (`verified_real_label_eval_1786046449`) | Route 0 (instruct) | — | **REAL** | 89/369 resolved | `.../verified_real_label_eval_1786046449/predictions/predictions_route_0.results.jsonl` |

5-route Verified parquet route index → model:
- Route 0: `Qwen/Qwen3-4B-Instruct-2507`
- Route 1: `openai/gpt-oss-20b`
- Route 2: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- Route 3: `openai/gpt-oss-120b`
- Route 4: `google/gemini-3-flash-preview`

Real Daytona results: route 0 = 89/369 (24%), routes 1/2/3 = 151/153/194 resolved out of 369.

### MATH traces

| Dataset | Model | CoT? | Label type | Notes | Path |
|---------|-------|------|------------|-------|------|
| `math_cot_trajectories_1785795546` | Qwen3-4B-**Thinking** | Yes | **REAL** (`scout_correct`, deterministic grading) | 65.6% correct on eval (328/500) | `.../math_cot_trajectories_1785795546/` |

### Trained abstention predictors

| Run | Train domain | Scout traces | Label | In-dist AUC | Cross-domain AUC (→Verified, route-3) |
|-----|-------------|-------------|-------|-------------|---------------------------------------|
| `cot_route0_1785909531` | MATH | 4B-Thinking CoT | `scout_correct` (real) | **0.9533** | 0.553 |
| `no_cot_input_only_route0_1785944036` | MATH | 4B-Thinking IO | `scout_correct` (real) | **0.8855** | 0.547 |
| `cot_1785364718` | SWE-Smith | 4B-Instruct, post-primary | route-3 real | 0.7246 | 0.550 |
| `no_cot_1785518862` | SWE-Smith | 4B-Instruct, post-primary | route-3 real | 0.7135 | 0.523 |
| `no_cot_route3_1785884243` | SWE-Smith | 4B-Instruct, post-primary | route-3 real | 0.6970 | **0.637** |
| `swe_smith_instruct_to_verified_transfer_..._1786049784` | SWE-Smith | 4B-Instruct, post-primary | route-3 real | — | **0.636** |
| `swe_smith_instruct_to_verified_transfer_..._input_only_1786126252` | SWE-Smith | 4B-Instruct, IO | route-3 real | — | 0.578 |
| `cot_abstention_..._cot_fixed_route3_1786133032` | SWE-Smith | 4B-**Thinking** CoT, post-primary | route-3 real | **0.749** | *not yet scored* |

Baselines on Verified (route-3 labels, for context):
- Scout patch length (negative, post-scout): **0.614** — shorter scout patch → oss-120b more likely to succeed
- Issue description length (inverted, pre-routing): **0.548** — shorter issue desc → oss-120b more likely to succeed (raw AUC 0.452, inverted sign)
- Gold patch length (inverted, oracle/informational only): **0.686** — longer gold patch → oss-120b more likely to succeed (raw 0.314; gold not available pre-routing)
- Random: ~0.500

Notes:
- High in-dist AUC (~0.95) is MATH-only; SWE-Smith peaks at ~0.72. Do not conflate.
- Cross-domain transfer (SWE-Smith → Verified) peaks at ~0.637 — barely above the scout patch-length heuristic (0.614), which itself supports the value of scouting.
- IO predictor (0.578) is below post-primary (0.637), confirming the scout's patch content adds signal beyond problem statement alone.
- The SWE-Smith predictors labelled "cot" use Instruct-4B traces (from route_outputs[0] in the parquet), not Thinking/CoT traces.

### Opus 5 SWE-Smith collection

Claude Opus 5 outputs on 286 SWE-Smith eval instances collected via OpenRouter:
- **Path**: `.../swe_smith_collect_anthropic_claude_opus_5_openrouter_1786126553/collect/eval/`
- **Proxy success rate**: 85.3% (reward-model score, not real Daytona — inflate substantially)
- **Real resolve rate (Daytona)**: **165/286 = 57.7%** (`opus_swe_smith_daytona_eval_1786133340`, concurrency=8)
- **Launcher**: `bash launchers/abstention/launch_opus_swe_smith_daytona_eval.sh`

Model comparison on 286 SWE-Smith eval instances:
| Model | Resolve rate |
|-------|-------------|
| Qwen3-4B-Instruct (scout) | ~25% |
| openai/gpt-oss-120b | ~47.2% |
| Claude Opus 5 | **57.7%** |

The collection eval split uses `ds_train` as the dataset path (the 286 source eval problems were sampled from `ds_train`, not `ds_test`).

---

## Research Direction & Framing

### Core abstention problem

The router makes a binary **abstain vs. call oracle** decision per instance. The oracle (`openai/gpt-oss-120b`) is expensive; the 4B scout runs cheaply first. The predictor label is always **oracle success**, not scout success. The scout's output is a feature, never a submitted answer.

### Why the scout should help

The scout's attempt encodes information about task difficulty that isn't available from the problem statement alone:
- What approach the model took
- Where it got stuck or fell back on heuristics
- Whether it produced a coherent partial solution

This is a richer difficulty signal than, e.g., issue description length. The challenge is extracting it with a predictor that generalises.

### Two experimental domains

**SWE-Smith / SWE-bench Verified (agentic)**
- Scout generates a patch; no way to verify it cheaply at decision time
- Predictor sees: problem + scout patch + optional thinking trace
- Scout failure is implicit (we never know if the patch is correct before running Daytona)
- Current best: 0.637 cross-domain AUC (SWE-Smith train → Verified eval), barely above patch-length heuristic (0.614)

**LiveCodeBench (code generation, see Section 8)**
- Scout generates Python code and **executes it locally** against test cases
- Scout pass/fail is a hard binary signal — no ambiguity
- Abstention is **conditional**: the cascade already handles `scout passes → done`; the predictor operates only on scout-failed instances, asking "can the oracle succeed where the scout failed?"
- Features: scout code, failure mode (wrong answer / TLE / runtime error / empty), thinking trace about why it failed
- This is a strictly cleaner version of the abstention problem: better features, cleaner labels, explicit failure signal

### Target ablation (2×2 × 2 domains)

Two binary factors crossed across two domains:

| | No test feedback | Test feedback |
|---|---|---|
| **No CoT** | problem + patch only | problem + patch + test failure signal |
| **CoT** | problem + patch + thinking trace | problem + patch + thinking trace + test failure signal |

Applied to: **LCB** (in-domain) and **SWE-Smith → SWE-Bench Verified** (cross-domain transfer).

- Factor A (CoT) isolates whether the scout's reasoning trace adds signal beyond the raw attempt
- Factor B (test feedback) isolates whether knowing which tests failed adds signal beyond the patch text
- Cross-domain comparison shows whether each signal transfers across repos/domains

### Experiment status tracker

#### Currently running / queued

| Job ID | Purpose | Unblocks |
|--------|---------|---------|
| `scout_daytona_eval_rerun_1786136775` | Daytona eval on Instruct-4B scout patches (1146 train + 286 eval) | All SWE test-feedback cells |
| `lcb_collect_qwen_qwen3_4b_thinking_2507_1786213696` | LCB collection with Thinking-4B scout + oracle | All LCB cells |
| `verified_abstention_eval_1786214481` | Cross-domain scoring: CoT predictor (epoch 7) on SWE-bench Verified | CoT predictor cross-domain AUC |

#### SWE-Smith → Verified (6 cells)

| Cell | Scout | CoT | Test FB | Run ID | In-dist AUC | Cross-domain AUC |
|------|-------|-----|---------|--------|-------------|-----------------|
| Instruct-4B, no-CoT, no tests | Instruct-2507 | ✗ | ✗ | `no_cot_route3_1785884243` | 0.697 | **0.637** |
| Thinking-4B, CoT, no tests | Thinking-2507 | ✓ | ✗ | `cot_abstention_..._cot_fixed_route3_1786133032` | **0.749** | *scoring now* |
| Thinking-4B, no-CoT stripped, no tests | Thinking-2507 | ✗ | ✗ | — pending training — | — | — |
| Instruct-4B, no-CoT, with tests | Instruct-2507 | ✗ | ✓ | — blocked on scout Daytona — | — | — |
| Thinking-4B, CoT, with tests | Thinking-2507 | ✓ | ✓ | — blocked on scout Daytona — | — | — |
| Thinking-4B, no-CoT stripped, with tests | Thinking-2507 | ✗ | ✓ | — blocked on scout Daytona — | — | — |

The "Thinking-4B, no-CoT stripped" cell uses the **same** `cot_trajectories_1785341592_fixed` trajectories as the CoT cell but excludes `--include-thinking` from the training command. This is the cleanest ablation for the CoT signal (same model, same patches, only predictor input differs).

The "Instruct-4B" cells are a separate condition (different model, different patch quality) that isolates model quality independent of the predictor. They share the `instruct_patches_trajectories_1785884242` trajectory dataset.

#### LCB (4 core cells + 1 optional reference)

| Cell | Scout | CoT | Test FB | Status |
|------|-------|-----|---------|--------|
| Thinking-4B, no-CoT stripped, no tests | Thinking-2507 | ✗ | ✗ | — blocked on LCB collection — |
| Thinking-4B, CoT, no tests | Thinking-2507 | ✓ | ✗ | — blocked on LCB collection — |
| Thinking-4B, no-CoT stripped, with tests | Thinking-2507 | ✗ | ✓ | — blocked on LCB collection — |
| Thinking-4B, CoT, with tests | Thinking-2507 | ✓ | ✓ | — blocked on LCB collection — |
| Instruct-4B, no-CoT, no tests | Instruct-2507 | ✗ | ✗ | needs separate LCB collection |
| Instruct-4B, no-CoT, with tests | Instruct-2507 | ✗ | ✓ | needs separate LCB collection |

The stripped / CoT split reuses the same `lcb_collect_qwen_qwen3_4b_thinking_2507_1786213696` collection — the only difference is `--include-thinking` in the training command. Test-feedback cells add the execution failure signal (error type + failing test output) to the predictor input.

Instruct-4B is a **first-class LCB condition**, not just a reference. On SWE-B, Instruct-4B outperforms Thinking-4B at actual patch generation — the thinking model is worse on agentic coding despite its reasoning capability. On competitive programming (LCB), instruct may again be the stronger scout: direct code synthesis without a thinking scratchpad can be more reliable for well-specified algorithmic problems. Comparing the two on LCB directly tests this and may reveal a different optimal scout for each domain.

#### Other model pairs (future)

The current setup fixes the scout/oracle pair at `Qwen3-4B / gpt-oss-120b`. The abstention predictor framing is model-agnostic, so other pairs are worth trying:

- **Different scout sizes**: 7B, 14B scouts — larger scouts produce better patches but cost more; the predictor needs to recalibrate to each scout's error distribution
- **Different oracle models**: Claude Opus 5 (57.7% on SWE-Smith eval), stronger future models — the label distribution and difficulty cutoff shift per oracle
- **Different scout families**: a non-Qwen scout (e.g. a fine-tuned code model) may produce qualitatively different failure signals, testing generality of the predictor architecture
- **Asymmetric pairs**: small instruct scout → large thinking oracle; or small thinking scout → large instruct oracle

These are out of scope for the current ablation but are the natural next step once the 2×2 grid is complete.

#### Infrastructure still needed

1. **`augment_trajectories_with_test_feedback.py`** — joins per-instance Daytona `report.json` into trajectory JSONL as a `test_feedback` field (FAIL_TO_PASS names, pass/fail counts, error type, message excerpt)
2. **`--include-test-feedback` flag in `train_cot_abstention_predictor.py`** — appends test feedback text to predictor input prompt
3. **Launcher for SWE test-feedback training runs** — wraps augmentation + training for all three test-feedback cells above

### Unifying story

Scout reasoning traces are an approximation of execution feedback when you cannot run the code (SWE agentic tasks). LCB proves the concept with clean execution signals; SWE tests how well reasoning traces substitute for them. If the predictor works in both regimes, that validates the reasoning-as-signal hypothesis.

### CoT trace extraction bug (fixed Aug 2026)

`Qwen3-4B-Thinking-2507` was collected with the opening `<think>` tag stripped during collection. The full output (thinking + solution) was stored in `patch_text`. Fixed dataset at `cot_trajectories_1785341592_fixed/` — extraction:

```python
if "</think>" in full:
    thinking, answer = full.split("</think>", 1)
    row["thinking_text"] = thinking.strip()
    row["patch_text"] = answer.strip()
```

All prior CoT predictors labelled "cot" used Instruct traces (route_outputs[0] from the 4-route parquet), not Thinking traces. The first real CoT predictor using properly extracted thinking is `cot_abstention_qwen3_emb8b_lora_r32_cot_fixed_route3_10epoch_1786133032` (in progress).

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

## 8) LiveCodeBench Domain (Set 4 — in progress)

### Motivation

LCB allows us to prove the scout abstention framing with clean execution signals, before (or alongside) the harder SWE case where we lack execution feedback at decision time. Multiple domains strengthen the argument that the predictor is learning genuine difficulty signal rather than overfitting to SWE-specific artefacts.

### Problem formulation

**Stage 1 — cascade filter**: scout (4B) generates Python code and executes it against private test cases. If it passes → done, no oracle needed. This handles the easy problems cheaply.

**Stage 2 — conditional abstention**: for scout-failed instances, apply the abstention predictor: given the problem, the scout's code, and its failure mode, predict P(oracle succeeds). If above threshold → call oracle. If below → abstain. This avoids wasting oracle tokens on problems that are genuinely unsolvable.

The predictor's job is therefore narrow: **discriminate between "oracle can fix this" and "oracle will also fail"** among the hardest instances.

### Why failure mode matters

The nature of the scout's failure is highly predictive:
- **Wrong answer**: model understood the problem, got logic wrong — oracle likely fixes it
- **Runtime error** (IndexError, etc.): probably a small bug — oracle likely fixes it
- **TLE**: wrong complexity class — oracle may or may not have a better algorithm
- **Empty output / syntax error**: model was lost — uncertain; depends on problem difficulty

### Dataset

HuggingFace: `livecodebench/code_generation_lite`. Problems from competitive programming contests (LeetCode, AtCoder, Codeforces) with dates from 2023-09-01 onwards. Each problem includes pre-written test cases (stdin/stdout format, `testtype="stdin"`). Private test cases are base64+zlib compressed JSON. No live environment needed during generation or evaluation.

### Collection pipeline

```bash
# Phase 1+2: collect scout + oracle trajectories (500 problems, scout=4B, oracle=120B)
bash launchers/abstention/launch_lcb_collection.sh

# Phase 3: train abstention predictor on scout-failed subset
LCB_COLLECTION_DIR=/mnt/.../lcb_collect_openai_gpt_oss_4b_TIMESTAMP \
  bash launchers/abstention/launch_lcb_abstention_train.sh
```

The collection script runs both scout and oracle via OpenRouter, evaluates locally using `subprocess.run(python_code, input=stdin)`, and outputs:
- `trajectories_train.jsonl`, `trajectories_eval.jsonl` — scout outputs with thinking traces
- `train/labels.parquet`, `eval/labels.parquet` — oracle success labels at route index 3

Active run: `lcb_collect_qwen_qwen3_4b_thinking_2507_1786213696` (Thinking-4B scout, oracle in progress). Note: earlier run `lcb_collect_openai_gpt_oss_4b_1786134110` used a non-existent model ID (`openai/gpt-oss-4b`) and produced empty scout trajectories — discard it.

### Key scripts

| Script | Purpose |
|---|---|
| `pipelinerl/swe/scripts/livecodebench/collect_lcb_trajectories.py` | Two-phase async collection: scout + oracle via OpenRouter, local code execution eval |
| `launchers/abstention/launch_lcb_collection.sh` | EAI job: 500 problems, scout=gpt-oss-4b, oracle=gpt-oss-120b, concurrency=16 |
| `launchers/abstention/launch_lcb_abstention_train.sh` | Train abstention predictor on LCB output; mirrors SWE-Smith setup |

### Assumption

This experiment assumes access to unit tests at decision time. For the purposes of this project, we treat this as given and note it explicitly. The LCB finding validates the approach under the best-case signal condition; SWE tests the approach when only reasoning traces are available (worst-case signal).

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
