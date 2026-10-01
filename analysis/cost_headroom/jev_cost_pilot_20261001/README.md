# Frozen Jev cost pilot

Authorized cost follow-up to the success pilot. Exactly the same 100 test problems per dataset, selected in the original success manifest. One request per problem, five route-specific choice questions. $1 cumulative budget guard, concurrency eight, resumable collector. No target-model generation calls.

Each route has five bins defined by training quantiles of per-problem mean valid-draw output tokens; duplicate boundaries are collapsed. Training bucket arithmetic means, bin frequencies, and route mean supplied as aggregate priors. No gold answers, individual labeled examples, or test outcomes in API state. Expected output tokens equal probability-weighted training bucket means; add exactly known input cost at recorded pool prices. This predicts total output including reasoning tokens.

Primary replay fixes our success predictions across Jev, our prefill cost head, training median, and training mean. Direct matched-accuracy savings on pair-specific shared frontier bands, 1,000 paired problem bootstraps with all valid draws kept together. Also report per-route raw-token R² and aggregate MAE. Prompt frozen before cost calls, no test calibration or tuning. Pilot remains exploratory because the same test subset has already been analyzed for success. API prediction overhead is recorded separately.

Run: `/home/toolkit/.conda/envs/pipeline-rl/bin/python3 analysis/cost_headroom/jev_cost_pilot.py`. `--limit 1` checks one paid response without changing the full manifest. `--analyze-only` replays stored responses. Raw responses are kept locally; manifest and derived token predictions/results are versioned.
