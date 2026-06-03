# Corrected SWE-Smith Real-Label Flow

SWE-Smith dataset rows contain a bug-introducing patch. For repair collection, solvers must see the repository after that bug patch has been applied. The older `/mnt/llmd/data/swe_smith` local dataset exposed clean `base_commit` file contents, so model patches were generated for the wrong source state.

The corrected dataset preparation does this per row:

1. reconstruct clean touched-file contents from `base_commit`;
2. apply the SWE-Smith bug patch to produce bugged repair context;
3. store clean target files as `reference_file_contents`;
4. store bugged solver input files as `repair_file_contents`;
5. store the original bug-introducing patch as `bug_introducing_patch`;
6. store the inverse bugged-to-clean patch as `repair_target_patch`;
7. keep compatibility aliases: `clean_file_contents`, `bug_patch`, `fix_patch`, `gold_file_contents`, and `patch`.

That keeps existing collection code working: solvers are prompted with `file_contents`, loaded from `repair_file_contents`, and proxy reward compares generated edits to `patch`, which is a compatibility alias for `repair_target_patch`. SWE-Smith real eval packages still materialize raw HuggingFace task rows for the harness, so evaluation still uses the official bug branch/tests.

## Commands

Prepare the fixed local dataset:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_prepare_swe_smith_bugged_context.sh
```

Validate the prepared dataset before collecting traces:

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python -m pipelinerl.swe.scripts.offline_router.validate_swe_smith_bugged_context \
  --dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_train \
  --limit 200
/home/toolkit/.conda/envs/pipeline-rl/bin/python -m pipelinerl.swe.scripts.offline_router.validate_swe_smith_bugged_context \
  --dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_test \
  --limit 200
```

Collect the four-route traces after validation passes:

```bash
TIMESTAMP=$(date +%s) bash launchers/offline_router/launch_swe_smith_real_label_trace_collect_parallel.sh
```

The real-label trace launcher now defaults to `/mnt/llmd/data/swe_smith_bugged_context`. Override `SWE_SMITH_DATA_ROOT`, `TRAIN_DATASET_PATH`, or `EVAL_DATASET_PATH` only if intentionally using another corrected dataset.

## Sanity Expectations

For a corrected dataset:

- `bug_introducing_patch` applies to `reference_file_contents`;
- `repair_target_patch` applies to prompted `repair_file_contents`;
- `bug_introducing_patch` should not apply again to prompted `repair_file_contents`.

If these fail, do not launch GPU collection or AWS evaluation.
