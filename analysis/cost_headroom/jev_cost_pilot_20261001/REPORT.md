# Jev cost-bucket pilot

Our success predictions are fixed in all arms. Five training-quantile output-length buckets per route; arithmetic training bucket means weighted by Jev probabilities. Input cost and recorded route prices retained. Same 100 held-out problems per dataset as the success pilot; all valid stored generation draws. No new target generations.

Direct generation cost savings at matched accuracy (paired problem bootstrap, 1,000 draws):

| Dataset | Jev vs median | Jev vs training mean | Jev vs ours |
|---|---|---|---|
| LCB | +5.8% [-11.0, +16.2] | +4.7% [-4.4, +11.8] | -59.6% [-84.3, -31.7] |
| Omni | -0.8% [-7.6, +5.8] | +1.0% [-4.6, +5.9] | -25.6% [-46.6, -5.8] |
| MMLU-Pro | -12.6% [-39.4, +10.2] | -9.3% [-30.6, +6.0] | -22.1% [-58.6, +7.3] |

Collection: 300/300 calls; recorded API spend $0.042466284.

Exploratory reuse of the success-pilot test subset; cost prompt and bins frozen before cost calls, with no tuning or calibration on test outcomes. Intervals condition on fitted predictors and use pair-specific shared accuracy bands. API prediction overhead excluded from generation savings. Bucket expectations are bounded by training bucket means and may miss extreme tails. This tests a prompted, untrained Jev predictor with training aggregate priors; it is not a trained cost head. Raw responses are retained locally in responses.jsonl; manifest and derived predictions/results are saved.

Our cost head vs Jev (same success predictions):

| Dataset | Our generation cost savings vs Jev |
|---|---|
| LCB | 37.4% [24.1, 45.7] |
| Omni | 20.4% [5.5, 31.8] |
| MMLU-Pro | 18.1% [-7.8, 36.9] |

Jev shows no clear gain over either constant-length baseline on these subsets. Our cost head outperforms Jev on LCB and Omni under this conditional paired analysis; MMLU-Pro is inconclusive. This does not rule out trained or calibrated Jev variants.
