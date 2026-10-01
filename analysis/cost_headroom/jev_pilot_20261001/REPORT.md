# Jev pilot results

Completed 300/300 calls with no errors. Reported API spend: **$0.016132**. Resolved model: `typesafe/jev-1.13-20260917`.

Frozen sample: 100 test problems each from LCB, Omni, and MMLU-Pro. No prompt tuning, probability calibration, or new target-model generations. Training-only aggregate route accuracies were included in the prompt. This is a preliminary test of a prompted predictor, not a strictly no-label comparison.

## Routing with the same cost estimates

Both success predictors use the same per-route median TRAIN output length and exact query input pricing. Values below are **direct target-generation cost savings from Jev relative to our success heads**, at matched accuracy. Positive favors Jev; negative favors our heads. Conditional paired problem-bootstrap 95% intervals; 1,000 resamples.

| Dataset | Jev savings vs. our success heads | Jev savings vs. training base-rate routing |
| --- | ---: | ---: |
| LCB | +13.6% [-6.3, +27.5] | +11.2% [-3.1, +26.2] |
| Omni | +22.9% [-6.9, +39.2] | +31.6% [+14.9, +47.1] |
| MMLU-Pro | -15.0% [-44.9, +2.3] | -14.2% [-42.8, +10.6] |

None of the Jev-versus-our-head intervals excludes zero. Jev improves on the constant training-base-rate router on Omni in this pilot; the LCB and MMLU contrasts are inconclusive. Do not interpret these small-sample intervals as equivalence.

## Prediction quality

Expected per-draw Brier scores (lower is better), with equal problem/route weighting:

| Dataset | Our heads | Jev | Training base rates | Jev mean probability / observed correctness |
| --- | ---: | ---: | ---: | ---: |
| LCB | 0.135 | 0.162 | 0.174 | 0.601 / 0.740 |
| Omni | 0.152 | 0.205 | 0.224 | 0.516 / 0.644 |
| MMLU-Pro | 0.165 | 0.172 | 0.179 | 0.648 / 0.751 |

Our heads have lower Brier-score and log-loss point estimates on all three subsets. Jev underpredicts mean correctness on each subset. Better probability scoring does not ensure better routing: route-relative probabilities and their interaction with costs determine the chosen route. No probability-metric significance test is claimed.

## Limits and next steps

These are descriptive matched-accuracy frontiers, with convex-hull mixtures selected using evaluation outcomes. Intervals condition on fitted/prompted predictions and omit training variability. Every problem keeps all its routes and valid generation draws together. Each comparison uses its own shared accuracy band. Jev API overhead is reported separately and excluded from generation-spend ratios.

The prompt has been exposed to this pilot. Any subsequent prompt changes should be selected using training/calibration examples and evaluated on fresh held-out problems. Completing the original test set adds 491 questions, but its 300 pilot questions are exploratory evidence. A calibrated Jev predictor would require additional predictions on calibration data, without using test outcomes to fit the mapping. Cost-bucket prediction remains untested.

Raw responses, frozen requests, complete bootstrap values, and the resolved provider version are saved beside this report. See [protocol](README.md).
