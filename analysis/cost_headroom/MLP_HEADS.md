# Frozen-feature MLP head comparison (2026-09-30)

Authorized experiment: LCB and Omni first; four arms isolate success-head,
cost-head, and combined changes. This compares heads on our rich features;
it is not a reproduction of the prefill-router paper's PCA/layer-selection
pipeline. No API collection or encoder inference is needed.

- Features: same frozen Qwen3-4B activations per dataset, concatenated mean and
  last-token readouts across eight stored layers. Standardization fits train only.
- Linear control: rebuild rich logistic heads with the existing script
  (`--rich --select-C`, calibration Platt scaling), and plain RidgeCV cost heads
  with the existing log-output retransformation and train-mean correction.
  Audit against published prediction files separately; report any differences.
- MLP: one shared hidden layer, independent output units for the five routes.
  Separate networks for success and cost. Width 64 or 128, GELU, dropout .2,
  AdamW (learning rate 1e-4, weight decay .1), up to 500 full-batch epochs,
  early stopping after 50 epochs without improvement.
- Seeds: 0, 1, 2. Choose width by mean calibration loss across all three seeds;
  select each seed's epoch on calibration. Average all three seeds for the
  primary comparison; also report individual seeds. No test-based selection.
- Success training: binomial BCE over every valid draw. Match the existing
  first-draw Platt calibration. Cost training: per-route train-standardized
  log mean output tokens, equal weighting of observed problem/route pairs.
- Arms: linear, MLP success + linear cost, linear success + MLP cost, MLP both.
- Evaluation: same market prices, value grid, realized outcomes, and paired
  problem bootstrap. All four arms use the same shared accuracy band on each
  bootstrap resample. Report direct cost savings versus linear.
- Deployable evaluation: mixtures and operating points selected on calibration,
  then applied once to test. Report achieved accuracy and cost with paired
  differences, rather than claiming equal test accuracy.
- Limitations: test frontier comparisons are descriptive; bootstrap intervals
  condition on fitted predictors and calibration selection. Seeds are reported
  separately. Encoder overhead is common; learned output length excludes it.

Launch after committing and pushing:

```bash
bash launchers/abstention/launch_mlp_heads.sh
```

The launcher uses independent eai jobs for each dataset/head/width/seed
(24 small GPU jobs, one GPU and eight CPUs each), separated by 15 seconds.
A CPU aggregation job waits for completion markers before selecting settings
and computing comparisons. All jobs use the same code snapshot.
Each dataset writes selections, prediction arrays, paired comparisons, and
calibration-selected operating points under the reported run directory.

## Results (completed 2026-09-30, approximately 14:00 Eastern)

All 24 training jobs and CPU aggregation succeeded. Run directory:
`/mnt/llmd/results/exps/aristides/reason/mlp_heads_20260930_175029`.
Compact results: `analysis/cost_headroom/mlp_heads_results.json`; full per-problem
results and prediction arrays remain in the run directory. Training revision:
`e617f42`; aggregation revision: `2777db9` (launcher-only resource correction).

Additional cost saved versus the reconstructed linear router at matched test
accuracy, with paired 95% bootstrap intervals (negative means more expensive):

| Replacement | LCB | Omni |
| --- | --- | --- |
| Success head only | -1.5% [-6.1, +2.1] | -0.03% [-7.0, +8.1] |
| Cost head only | -0.7% [-5.0, +2.7] | +5.5% [-2.4, +12.3] |
| Both heads | -1.9% [-7.3, +2.4] | +7.1% [-1.6, +15.4] |

- Shared accuracy band: LCB 53.7–87.4%; Omni 52.6–73.1%. All 500 paired
  bootstrap resamples had valid overlap. Percentages are direct relative cost
  savings against linear, not percentage-point changes in the older headline
  savings against median-length pricing.
- Width 64 selected for both heads on both datasets. Success epochs: LCB
  20–23, Omni 4. Cost epochs: LCB 21/27/81, Omni 10/37/27.
- Individual-seed cost effects on LCB were all negative (-8.5, -2.9, -3.3%);
  on Omni all positive (+4.7, +2.5, +6.2%). Ensemble averaging improves LCB
  substantially relative to individual cost MLPs, but does not beat ridge.
- Baseline reconstruction audit: published LCB predictions save 0.13% versus
  reconstructed linear (CI -0.18 to +0.27); Omni routes/costs match at the
  point estimate. Maximum probability discrepancies are .0016 and .0028;
  cost discrepancies are below .000004 cents. This does not explain the result.
- Mean route dollar-cost R² decreases from .511 to .421 on LCB, and .462 to
  .354 on Omni. Mean binomial log-loss changes from .387 to .391 on LCB and
  .449 to .446 on Omni. Prediction quality does not directly order routing gains.
- Calibration-selected operating points show no consistent dominance. For
  example, at Omni's 70% calibration target, linear achieves 69.55% test accuracy
  at .09250 cents; MLP cost achieves 70.49% at .09259 cents. Its paired accuracy
  gain is +0.94 points [.28, 1.72], while the cost difference is uncertain.
  At the 65% target it is more accurate and more expensive. These are not
  equal-test-accuracy savings estimates, and multiple targets were examined.
- Conclusion: keep the linear heads as the primary method. No significant
  averaged routing improvement from this small MLP search; Omni's cost-head
  effect is promising but uncertain. This does not rule out stronger nonlinear
  predictors or the paper's PCA/selected-layer pipeline. That pipeline has not
  been tested here.
