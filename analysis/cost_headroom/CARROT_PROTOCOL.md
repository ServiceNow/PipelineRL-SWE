# CARROT-KNN-SBERT comparison

Reference: https://github.com/somerstep/CARROT
Inspected upstream revision: `3e6acff6aecf4cbcb8f31a118d04c799c2ea1655`.
The implementation follows `carrot/train_and_infer.py`'s
`carrot-knn-sbert` / SPROUT branch and the encoder in `carrot/data_utils.py`.

## Estimators and data

- Local `sentence-transformers/all-MiniLM-L12-v2` embeddings, with the model's
  default truncation and pooling. No embedding or generation API calls.
- Separate multi-output uniform-weight cosine kNN regressors for success and
  **raw output tokens**, without log transforms, smearing, or mean matching.
- Five-fold unshuffled training CV chooses k by mean multi-output R2, from
  powers of two 2 through 512 that fit every training fold. This follows the
  upstream regression selection rule. Separate k values for the two targets.
- All valid draws are averaged within each problem/route. This adapts the
  upstream binary outcome labels to our repeated-draw expected-success target.
- Original LCB, Omni, and MMLU-Pro training/test splits. Calibration labels do
  not enter k selection. No refitting on held-out expansion problems.
- Input and output prices match the existing paper's recorded rates. Costs
  combine exact recorded mean input length with predicted output length.
- Successful evaluation requires at least one valid draw per problem/route;
  the script rejects missing labels instead of silently changing the sample.

This is **CARROT-KNN-SBERT**, not the OpenAI `text-embedding-3-small` variant
or the fine-tuned RoBERTa variant. The local SBERT variant is included upstream.
The other variants are not covered by this experiment.

## Comparisons

1. Our success + our cost versus our success + CARROT cost.
2. Our full router versus CARROT success + CARROT cost.
3. CARROT success + CARROT cost versus CARROT success + constant mean training
   output length (exact query input cost is retained).
4. CARROT success + our cost versus CARROT success + CARROT cost.
5. Our full router versus our success + median training output length.
6. Our success + CARROT cost versus our success + median training output length.

Each pair uses 12 accuracy targets over the interior 5–95% of its shared test
accuracy band, with convex-hull interpolation. Report **direct cost savings**:
`1 - exp(mean(log(cost_left / cost_right)))`. This differs from subtracting two
savings summaries against median pricing and from the main paper's three-arm
bands that also include the oracle. Do not copy these numbers into the main
paper's existing table as if the metrics were identical.

One thousand paired problem bootstrap resamples (seed 0) keep every route and
its averaged draws together. Report the plug-in effect and percentile interval;
save bootstrap values. Bands are recomputed per resample. Intervals condition
on fitted predictors and exclude training/CV-selection uncertainty. Evaluation
frontiers permit mixtures selected using test outcomes; they are descriptive,
not a deployment policy chosen in advance. Encoder overhead is excluded.

## Reproduction and outputs

```bash
CARROT_PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3 LOCAL=1 bash launchers/abstention/launch_carrot_compare.sh
# For snapshot cluster submission:
SUBMIT=1 bash launchers/abstention/launch_carrot_compare.sh
```

Launcher uses three CPU snapshot jobs with eight CPUs and 16 GB RAM each.
It installs `sentence-transformers==3.4.1` without dependencies in each job's
isolated `/tmp/carrot_python_deps`; other dependencies use `pipeline-rl`.
CPU inference disables the unused optional DeepSpeed import, which otherwise
initializes Triton and fails on this environment's CPU workers.

Default output: `/mnt/llmd/results/exps/aristides/reason/carrot_compare_20261001/`.
Each pool saves a text/ID-checked embedding cache, prediction arrays, selected
neighbor counts/CV scores, per-route prediction diagnostics, and `results.json`.
The setup smoke run uses five resamples solely to check execution; report only
the full 1,000-resample jobs. Expanded evaluation remains pending completion of
collections, feature extraction, and problem-weight handling.

## Execution record

Initial three cluster submissions exited during startup with no usable scheduler
logs. The full experiment was run locally using the verified environment and
all 1,000 resamples. The runner now persists startup and runtime logs under
`logs/` and the launcher also supports `LOCAL=1`. Local results are the primary
artifacts; no successful remote execution is claimed.
