# Offline Router Launchers

Launch from the repo root unless noted otherwise.

## Expanded SWE-bench Train Dataset

Build a local SWE-bench train dataset from all official train rows, without the
old 6k pre-reconstruction sampling cap:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_prepare_swe_bench_train_all_16k.sh
```

Defaults write the dataset to `/mnt/llmd/data/swebench/all_16k/ds_train` and the
job log to `/mnt/llmd/results/exps/aristides/reason/prepare_swe_bench_train_all_16k_${TIMESTAMP}`.
Override `MAX_TOTAL_TOKENS`, `DATASET_ROOT`, or `SINGLE_OUTPUT_PATH` to create
larger-context variants such as `24k` or `32k`.

Build the matching SWE-bench Lite eval set from all 300 official test rows with
the same 16k token filter:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_prepare_swe_bench_lite_all_16k.sh
```

Defaults write this eval dataset to `/mnt/llmd/data/swebench_lite/all_16k/ds`.

Build the SWE-bench Verified eval set from all 500 official test rows with the
same 16k token filter:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_prepare_swe_bench_verified_all_16k.sh
```

Defaults write this eval dataset to `/mnt/llmd/data/swebench_verified/all_16k/ds`.

## Current SWE-bench Router-Split Experiments

Submit the four next-step jobs sequentially locally. Remote jobs run concurrently
after each submission succeeds:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_offline_router_swe_bench_router_split_next_experiments_parallel.sh
```

The wrapper launches:

- `bin_20bucket_full_5epoch`: plain direct reward-bin baseline, 5 epochs.
- `bin_20bucket_delta_aux_seq_mse_w20_full_5epoch`: matched full-data delta-aux run, 5 epochs.
- `bin_20bucket_restricted_delta_aux_seq_mse_w20_overfit512_10epoch`: same sequential delta aux, but restricted CE over reward-bin tokens, 512-row overfit gate.
- `delta_bin_20bucket_overfit512_10epoch`: direct signed-delta bin prediction over `[-1, 1]`, 512-row overfit gate.

Each script accepts the usual overrides, including `TIMESTAMP`, `SOURCE_RUN_DIR`,
`COLLECT_OUTPUT_DIR`, `OUTPUT_ROOT`, `TRAIN_OUTPUT_DIR`, `TRAIN_NPROC`,
`MAX_SEQ_LENGTH`, `MAX_TRAIN_ROWS`, `NUM_EPOCHS`, and `TRAIN_EXTRA_ARGS`.

## MSE Objective Sweep

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_offline_router_swe_bench_router_split_mse_objective_experiments_parallel.sh
```

The wrapper launches:

- `bin_20bucket_mse_seq_w20_full_5epoch`: expected reward-bin value trained with MSE only.
- `bin_20bucket_ce_mse_seq_w20_full_5epoch`: regular CE plus expected-value MSE.
- `bin_20bucket_mse_delta_seq_w20_full_5epoch`: expected-value MSE plus delta MSE.

## ModernBERT Encoder Baseline

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_offline_router_swe_bench_router_split_modernbert_large_mse_full_5epoch.sh
```

This trains `answerdotai/ModernBERT-large` as a two-output reward regressor over the
same collected router dataset. It uses the original repair prompt plus the primary
attempt as input, predicts both route rewards directly, and writes the usual
`route_metrics.csv`, `pairwise_metrics.csv`, `utility_vs_baselines.csv`, and
`summary.json`.

Direct routing classifier baseline on the same old SWE-bench router split
collection:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_offline_router_swe_bench_router_split_modernbert_large_classifier_12epoch.sh
```

This uses `answerdotai/ModernBERT-large` with cross-entropy over the better
route, defaults to 12 epochs, and keeps the existing 178-row eval subset for
apples-to-apples comparison with prior ModernBERT and text-mode runs.
