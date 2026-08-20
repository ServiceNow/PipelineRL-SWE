# Scout-Signal Routing Launchers

The research design has two independent axes:

- **Decision regime**: stronger-model success prediction/abstention versus full model routing
- **Dataset**: SWE versus LiveCodeBench

| Dataset | Success prediction / abstention | Full routing |
|---------|---------------------------------|--------------|
| **SWE** | Predict gpt-oss-120b success from scout evidence; train on SWE-Smith and transfer to SWE-Bench Verified | Previously attempted and unsuccessful |
| **LCB** | Old near-perfect result is invalid; corrected official temporal rerun is the current validity gate | Proposed scout-first direct-jump policy, conditional on the corrected gate |

In the prediction regime, the label is the success of another, stronger model, not scout success.
In the full-routing regime, the policy keeps the scout answer or jumps directly to a selected
mid/high tier, skipping unnecessary stages of a sequential cascade. Its headline evaluation is
final correctness versus realized inference cost.

Historically, the apparently near-perfect LCB prediction result motivated retrying full routing on
LCB after it had failed on SWE-Bench-style tasks. Because the old LCB evaluator and split were
invalid, the corrected LCB prediction experiment must first re-establish that signal.

## Label types

| Type | Description | How produced |
|------|-------------|--------------|
| **Real** | Binary pass/fail from actually running the test suite in a Daytona sandbox | `launch_verified_real_label_eval.sh` (SWE-bench Verified) or the `swe_smith_real/` collection in `offline_router/` |
| **Proxy** | Fast approximation via a learned reward/verifier model | `offline_router/swe_smith_proxy/` scripts |

The scripts in this directory use **real labels** exclusively (either from SWE-bench Verified Daytona
eval or from the SWE-Smith real-label dataset at
`/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_*/`).

## Scripts

### Data collection

| Script | What it does | Labels |
|--------|-------------|--------|
| `launch_cot_collection.sh` | Collects CoT thinking traces + patches from the scout model on SWE-Smith | Real (uses existing real-label runs) |
| `launch_math_cot_collection.sh` | Same but for MATH domain trajectories | Real |
| `launch_swe_smith_collect_opus_openrouter.sh` | Runs Opus 5 (`anthropic/claude-opus-5`) on SWE-Smith eval-286 via OpenRouter; "call stronger model instead of abstaining" baseline | Real (test suite eval) |
| `launch_daytona_cot_predictions.sh` | Runs a trained CoT model to generate patch predictions via Daytona | — |
| `launch_lcb_corrected_collection.sh` | Collects a fresh temporal LCB split; public tests become router feedback and the official full suite supplies labels | Real |
| `launch_lcb_corrected_repair.sh` | Serially regrades saved corrected-LCB outputs, retries only failed oracle calls, and validates the repaired collection | Real |

### Training

| Script | What it does |
|--------|-------------|
| `launch_cot_abstention_predictor.sh` | Main launcher: trains Qwen3-Embedding-8B + LoRA to predict success from scout thinking trace + patch. Key env vars: `INPUT_ONLY`, `INCLUDE_THINKING`, `LABEL_ROUTE_IDX`, `LORA_R`, `NUM_EPOCHS` |
| `launch_math_input_only_abstention_predictor.sh` | Input-only baseline on MATH: uses problem statement only (no trace/patch); sets `INPUT_ONLY=true`, `INCLUDE_THINKING=false` |
| `launch_instruct_no_cot_predictor.sh` | Trains an instruct model (no chain-of-thought) as predictor |
| `launch_lcb_corrected_ablation_suite.sh` | Submits matched input-only, post-scout, and post-scout+public-feedback LCB predictors | Real |

### Evaluation

| Script | What it does |
|--------|-------------|
| `launch_verified_abstention_eval.sh` | Evaluates an abstention predictor on SWE-bench Verified |
| `launch_verified_real_label_eval.sh` | Runs the SWE-bench Verified Daytona sandbox eval to produce real pass/fail labels. Images: `ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest`. Concurrency capped at 8 (Daytona 10-CPU limit). |
| `launch_swe_testfb_transfer_rescore.sh` | Correctly re-scores existing SWE-Smith test-feedback checkpoints on Verified with their training-time feedback schema |

## Key data paths

### SWE-Smith
- **4-route real-label parquet (canonical label source)**: `/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/{train,eval}/`
  - Routes: [4B-Instruct, gpt-oss-20b, Qwen3-Coder-30B, gpt-oss-120b]
  - Labels: REAL (binary `{0.0,1.0}` Daytona pass/fail). Solve rates: [25%, 40%, 40%, 46%]
- **Instruct-4B trajectories**: `/mnt/llmd/results/exps/aristides/reason/instruct_patches_trajectories_1785884242/` (1146 train + 286 eval)
- **CoT-Thinking-4B trajectories**: `/mnt/llmd/results/exps/aristides/reason/cot_trajectories_1785341592/`
- **Trajectories with test feedback**: `.../instruct_patches_trajectories_1785884242_with_testfb/`
- **SWE-Smith eval dataset**: `/mnt/llmd/data/swe_smith_bugged_context/ds_test` (286 instances, `swe_smith_test_bugged_context`)

### SWE-Bench Verified
- **Verified dataset (all_16k, 369 instances)**: `/mnt/llmd/data/swebench_verified/all_16k/ds`
- **Verified dataset (full 500 instances)**: `/mnt/llmd/data/swebench_verified/full/ds`
- **Scout trajectories (369, with test feedback)**: `/mnt/llmd/results/exps/aristides/reason/verified_scout_daytona_eval_1786402548/trajectories_with_testfb/`
- **Oracle labels (134/369 real Daytona)**: `logs/run_evaluation/verified_oracle_eval_1786405908/`
- **131 missing instances oracle eval (in progress)**: `logs/run_evaluation/verified_expand_oracle_eval_fixed_1787014995/`
- **5-route Verified collection (PROXY labels — do not use for ground truth)**: `/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_*/collect/`

### LCB (LiveCodeBench)
- **Old collection is invalid for paper results**: the local evaluator forced function-call tasks to fail, decoded private tests incorrectly, and used a random split.
- **Corrected protocol**: `release_v6`, `2024-10-01` temporal cutoff, official runner commit `28fef95`, public tests for router feedback, full public+private suite for labels.
- **Pinned dataset source**: `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505`. The collector downloads the official `test*.jsonl` files directly because `datasets>=4` removed Hub dataset-script execution.
- **Corrected collection launcher**: `launch_lcb_corrected_collection.sh`
- **Matched predictor suite**: `launch_lcb_corrected_ablation_suite.sh`
- **Current corrected outputs**: `lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448`
  contains all 892 model outputs but is quarantined until `launch_lcb_corrected_repair.sh`
  finishes and validation passes. Its first grading pass overlapped fork-based official evaluator
  calls and recorded infrastructure errors as failures.

### MATH
- **CoT trajectories**: `/mnt/llmd/results/exps/aristides/reason/math_cot_trajectories_1785795546/` (1462 train + 493 eval)
  - Labels: REAL (`scout_correct` field, symbolic math grading). 65.6% correct on eval.

## Label hygiene — CRITICAL

**NEVER use `route_successes` or `route_rewards` from collection parquets as ground truth for SWE tasks.**
These are proxy metrics from the RL training pipeline. Detection:
- `route_rewards` unique values binary `{0.0, 1.0}` → real Daytona labels ✅
- `route_rewards` continuous floats → proxy labels ❌ (the Verified 5-route collection parquet has this)
- Oracle solve rate ~10% on Verified → red flag for proxy labels (real rate is ~36%)

**Exception**: corrected LCB `route_successes` are real labels from the pinned official evaluator. Do not use labels from the superseded collection.

## Model names — exact identifiers only

- **Scout (instruct)**: `Qwen/Qwen3-4B-Instruct-2507`
- **Scout (CoT)**: `Qwen/Qwen3-4B-Thinking-2507`
- **Oracle**: `openai/gpt-oss-120b`
- **Embedding predictor**: `Qwen/Qwen3-Embedding-8B`

Never invent model IDs — `openai/gpt-oss-4b` does not exist on OpenRouter and caused an entire collection run to fail.

## EAI job gotchas

- **Always commit + push before relaunching** — `SNAPSHOT=1` captures the repo at submission time.
- **Space submissions**: wait at least a few seconds between `eai job submit` calls. Multi-job suite launchers default to a 5-second delay.
- **GPU count**: `_GPU = NPROC * GPU`. Set only `NPROC` for multi-GPU DDP. Do NOT set `GPU=N, NPROC=1` (requests N GPUs but uses 1).
- **Verified predictor context length**: set `MAX_MODEL_LEN=32768` (not 16384 — Verified instances are longer than SWE-Smith; 16384 caused 369/369 scoring failures).
- **Oracle patch format**: 120b sometimes returns markdown with explanation + \`\`\`diff fence rather than raw git diff. `convert_text_to_patches.py` only handles search/replace blocks — add fence extraction before running Daytona eval.
- **Public Hugging Face models**: cached login tokens can expire and cause public model downloads to return 401. Set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`; the corrected LCB launcher does this automatically.
- **LCB evaluator concurrency**: the official checker forks a process and creates a
  `multiprocessing.Manager` per call. Do not overlap checker calls from threads; the corrected
  collector serializes grading and validates that no runner errors entered the labels.

### Monitoring jobs with `eai`

```bash
# Find a job by name and show its current run state.
eai job ls -N <job-name> --state all \
  --fields id,state,runs.exitCode,runs.runTime,stateInfo

# Read current logs, or follow them until completion.
eai job log <job-id> --tail 100
eai job log -f <job-id>

# Inspect full job metadata or execute a diagnostic command in a running job.
eai job info <job-id>
eai job exec <job-id> -- /bin/bash -lc '<command>'

# Stop a job that is still alive.
eai job kill <job-id>
```

Some launchers redirect detailed application output to files under their result directory,
so an empty `eai job log` does not prove that no work is happening. Check the launcher's
`collect.log`, `train.log`, `score.log`, or service log and verify that output row counts and
file modification times are advancing.

## Security note

`DAYTONA_API_KEY` and `OPENROUTER_API_KEY` are always read from `.env` at runtime — never
hardcoded, never passed as EAI job command-line arguments (which would appear in logs).
