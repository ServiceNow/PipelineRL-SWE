# Calibration-selected fixed-policy evaluation

Select V by maximizing OUR router’s mean validation utility, V × expected correctness − realized mean target-generation cost. Freeze V before test. Route each test problem once, with no randomized mixtures. Compare learned costs with median training-length costs while keeping our success predictions fixed. Positive savings and positive accuracy difference favor learned costs. Paired problem bootstrap 95% intervals, 2,000 draws.

| Dataset | Validation-selected V ($/correct) | Learned-cost accuracy | Median-cost accuracy | Cost savings | Accuracy difference |
|---|---:|---:|---:|---:|---:|
| LCB | 1 | 0.892 | 0.892 | +3.0% [+0.4, +6.0] | +0.0pp [-0.2, +0.2] |
| Omni | 1 | 0.737 | 0.733 | +0.8% [-0.4, +3.1] | +0.3pp [+0.0, +1.0] |
| MMLU-Pro | 1 | 0.807 | 0.807 | +0.0% [+0.0, +0.0] | +0.0pp [+0.0, +0.0] |

This is a fixed-policy check on the historical split, whose test outcomes have informed earlier exploratory analyses. It complements, but does not replace, fresh expanded-test evaluation. Paired intervals keep problems intact and condition on fitted predictors and selected validation V. Predictor overhead excluded.
