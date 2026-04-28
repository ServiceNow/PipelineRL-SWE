# Offline Router Launchers

Launch from the repo root unless noted otherwise.

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
