# Intern-Decision-4B success pilot

Same frozen 100 problems per dataset and original Jev success questions/training aggregate priors. Published inference with pinned weights and default temperature; no prompt tuning, domain calibration, or target-model generations.

Direct generation cost savings at matched accuracy; 1,000 paired problem bootstrap draws. Positive favors Intern.

| Dataset | Intern vs ours, our costs fixed | Intern vs Jev, our costs fixed | Intern vs ours, median costs fixed |
|---|---|---|---|
| LCB | -25.8% [-45.0, +0.2] | +3.5% [-7.7, +15.3] | +18.8% [-1.6, +30.9] |
| Omni | +1.1% [-16.3, +14.1] | -0.3% [-7.4, +5.2] | +6.5% [-17.0, +25.2] |
| MMLU-Pro | -10.1% [-34.9, +12.4] | -0.8% [-14.7, +10.4] | -5.3% [-29.7, +20.0] |

Conditional exploratory replay of previously examined test problems. All routes and valid generation draws remain clustered by problem. Convex-hull frontiers use evaluation outcomes, not a separately selected deployment policy. Pair-specific accuracy bands; no training uncertainty or multiplicity correction. Predictor inference overhead excluded from generation spending. No API spending. All probability metrics, derived predictions, contrasts and bootstrap samples in results.json.
