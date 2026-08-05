# Abstention Launchers

Scripts for the **abstention/cascade** research track: predict whether the scout model will succeed
on a given task, and either abstain (call a stronger model) or proceed with the scout.

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

### Training

| Script | What it does |
|--------|-------------|
| `launch_cot_abstention_predictor.sh` | Main launcher: trains Qwen3-Embedding-8B + LoRA to predict success from scout thinking trace + patch. Key env vars: `INPUT_ONLY`, `INCLUDE_THINKING`, `LABEL_ROUTE_IDX`, `LORA_R`, `NUM_EPOCHS` |
| `launch_math_input_only_abstention_predictor.sh` | Input-only baseline on MATH: uses problem statement only (no trace/patch); sets `INPUT_ONLY=true`, `INCLUDE_THINKING=false` |
| `launch_instruct_no_cot_predictor.sh` | Trains an instruct model (no chain-of-thought) as predictor |

### Evaluation

| Script | What it does |
|--------|-------------|
| `launch_verified_abstention_eval.sh` | Evaluates an abstention predictor on SWE-bench Verified |
| `launch_verified_real_label_eval.sh` | Runs the SWE-bench Verified Daytona sandbox eval to produce real pass/fail labels. Images: `ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest`. Concurrency capped at 8 (Daytona 10-CPU limit). |

## Key data paths

- **SWE-Smith real-label collect dir**: `/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_*/collect/`
- **SWE-Smith eval dataset**: `/mnt/llmd/data/swe_smith_bugged_context/ds_test` (286 instances, dataset name `swe_smith_test_bugged_context`)
- **MATH trajectories**: set via `MATH_OUTPUT_DIR` env var when launching `launch_math_input_only_abstention_predictor.sh`
- **Parquet outputs**: each collection job writes `train/` and `eval/` parquet dirs alongside `trajectories_train.jsonl` / `trajectories_eval.jsonl`

## Security note

`DAYTONA_API_KEY` and `OPENROUTER_API_KEY` are always read from `.env` at runtime — never
hardcoded, never passed as EAI job command-line arguments (which would appear in logs).
