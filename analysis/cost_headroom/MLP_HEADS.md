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

The launcher uses one eai GPU, eight CPUs, 64 GB CPU RAM, and a code snapshot.
Each dataset writes selections, prediction arrays, paired comparisons, and
calibration-selected operating points under the reported run directory.
