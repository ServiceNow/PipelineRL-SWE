# Jev success + our prefill cost replay

No new API calls. Same 100 held-out problems per dataset. Prompts, probabilities and trained heads unchanged.

## Primary comparison

Direct generation cost savings from **Jev success + our costs** relative to **our success + our costs**, at matched accuracy. Positive favors Jev success. Paired problem-bootstrap 95% intervals (1,000 draws).

| Dataset | Hybrid savings vs our full router |
|---|---|
| LCB | -25.9% [-49.7, -3.3] |
| Omni | +0.8% [-17.1, +15.8] |
| MMLU-Pro | -9.2% [-34.6, +15.3] |

The hybrid is worse on LCB under the conditional paired analysis: its estimated generation spend is 25.9% higher, with the interval excluding zero. Omni and MMLU-Pro are inconclusive. No dataset establishes a hybrid improvement over our full router. The earlier LCB/Omni point-estimate advantage for Jev success with median costs does not persist with our learned costs.

## Cost-head effect with Jev success held fixed

Our learned costs compared with median training output length, keeping Jev success predictions fixed:

| Dataset | Learned-cost savings |
|---|---|
| LCB | +15.1% [-3.7, +30.1] |
| Omni | -0.2% [-16.6, +17.9] |
| MMLU-Pro | +12.3% [-6.1, +35.3] |

## Interpretation limits

Exploratory replay of an already analyzed pilot subset. All routes and stored valid generation draws remain clustered by problem. The frontiers use evaluation outcomes to choose convex-hull mixtures; they are descriptive and do not establish an independently selected deployment policy. Intervals condition on existing predictions and omit training uncertainty. Each contrast uses its own shared accuracy band. These are generation costs; Jev inference and encoder overhead are excluded. No calibration, prompt tuning, or paper edits. Full contrasts, success predictions and bootstrap samples are in results.json.
