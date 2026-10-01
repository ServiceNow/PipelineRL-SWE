# Jev preliminary success-prediction comparison

Authorized by the user after a cost estimate, with a $1 cumulative spend guard.
100 original held-out test problems per dataset (LCB, Omni, MMLU-Pro), sampled
without labels using seed 20261001. No new target-model generations.

Pinned model: `typesafe/jev-1.13`; record the resolved model in each response.
Endpoint: `https://openrouter.ai/api/alpha/decisions`.
Documentation: https://openrouter.ai/blog/tutorials/how-to-use-jev/
Pricing checked October 1: $0.042/M input tokens, free output. Estimate around
$0.02 for 300 calls with 1,000 instruction tokens; exact payloads may differ.

`manifest.json` freezes actual request payloads, sample IDs and a SHA256 before
any paid request. Every request asks five independent correctness `noul`
questions. State contains the problem, grading rule, model/effort descriptions,
and **training-only aggregate accuracy priors**. No gold answers, generated
answers, individual labeled examples, or held-out outcomes are sent. This is
an untrained prompted predictor with aggregate task-specific priors, rather
than a strictly no-label zero-shot comparison. No prompt tuning on this pilot.

The user requested success prediction with costs held constant. All routing
arms therefore use the **same median valid TRAIN output length for each route**,
plus the exact recorded query input cost, matching the paper's reference rule.
Compare our success predictions, Jev success predictions, and training base
rates. Never substitute learned query-output costs in this first experiment.

Prediction metrics are expected per-draw Brier score and log loss, averaged
with equal problem/route weight using all valid stored draws. Routing uses
paired, descriptive matched-accuracy frontier comparisons over pair-specific
shared bands with 1,000 problem bootstrap samples. Each problem's route/draw
results stay together. No prompt-selection or predictor-fit uncertainty is
included. Small pilot subsets are exploratory and not paper-ready evidence.

The $1 guard tracks response usage.cost, and conservatively charges unknown
billing at the maximum 64k input-token price. Concurrent calls reserve that
maximum before dispatch. Three attempts for transient failures, resume skips
successful calls, stop after ten errors. Usage cost and resolved models are
saved; credential contents are never logged. Ambiguous API billing may differ
from observed spending; the guard is not an account-level hard spending cap.

```bash
/home/toolkit/.conda/envs/pipeline-rl/bin/python3 analysis/cost_headroom/jev_pilot.py --prepare-only
# One paid API/schema check, then resume the frozen remainder:
/home/toolkit/.conda/envs/pipeline-rl/bin/python3 analysis/cost_headroom/jev_pilot.py --limit 1
/home/toolkit/.conda/envs/pipeline-rl/bin/python3 analysis/cost_headroom/jev_pilot.py
```

Outputs: append-only `responses.jsonl`, `collection_status.json`, `results.json`.
No cost-bucket questions in the initial pilot. A later cost experiment could
use training-defined length bins and map predicted bin probabilities to
training-derived representative lengths; tail treatment and calibration
require separate evaluation.
