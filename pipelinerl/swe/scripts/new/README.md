# Offline Router Analysis (New)

This workflow uses runtime `router_trace` streams produced by the actor, then optionally scores
those traces with the policy performance head, and finally evaluates utility-based routing.

## 1) Runtime traces

Enable in config:

```yaml
swe:
  router_trace:
    enabled: true
    include_outputs: true
    include_policy_token_ids: false
    include_file_contents: false
```

Actor writes:

- `streams/router_trace/...` (train)
- `streams/router_trace_test/...` (test)

## 2) Score traces with performance head

```bash
python -m pipelinerl.swe.scripts.new.score_router_traces \
  --input-glob "/path/to/run/streams/router_trace_test/*/*/0.jsonl" \
  --model-path "/path/to/run/finetune/current" \
  --output-jsonl "/path/to/run/router_trace_test_scored.jsonl"
```

## 3) Utility-based router simulation

Decision rule:

- route to `argmax_i(pred_quality_i - lambda * cost_i)`
- abstain if best utility < `tau`

```bash
python -m pipelinerl.swe.scripts.new.analyze_router_traces \
  --input-glob "/path/to/run/router_trace_test_scored.jsonl" \
  --output-dir "/path/to/run/router_analysis" \
  --score-key policy_value_prompt_last_all \
  --lambda-start 0.0 --lambda-stop 0.10 --lambda-step 0.01 \
  --tau-start 0.0 --tau-stop 1.0 --tau-step 0.05
```

Optional cost calibration (e.g., USD per 1K tokens):

```bash
  --policy-cost-per-1k 0.002 \
  --expert-costs-per-1k "0.004,0.012"
```

Optional regret baseline for "oracle gap captured":

```bash
  --regret-baseline-route policy
```

Note: oracle-vs-predictor regret is evaluated in ranking-only mode (tau/abstention off).

## Outputs

Tables:

- `operating_points.csv` (lambda/tau sweep: quality, cost, utility, coverage)
- `frontier_points.csv` (Pareto frontier points)
- `oracle_operating_points.csv` (oracle decisions on realized utility for same lambda/tau sweep)
- `oracle_frontier_points.csv` (oracle Pareto frontier points)
- `oracle_predictor_regret.csv` (oracle-vs-predictor utility regret by lambda/tau)
- `best_expert_identity.csv` (how often each expert is the realized best expert)
- `summary.json` (includes AUC-QC)

Graphs (if matplotlib is installed):

- `cost_vs_quality_scatter.png`
- `pareto_frontier_auc.png`
- `oracle_pareto_frontier_auc.png`
- `pareto_frontier_combined.png` (actual vs baseline-mu vs oracle on one axes)
- `coverage_heatmap_lambda_tau.png`
- `utility_heatmap_lambda_tau.png`
- `mean_regret_vs_lambda.png`
- `p95_regret_vs_lambda.png`
- `pred_vs_realized/pred_vs_realized_<idx>_<route>.png` (wider plot, title includes `R^2`)
- `pred_vs_realized/pred_vs_realized_stats.csv` (`R^2` + per-route summary stats)
- `routing_pies/lambda_<lambda>/routing_lambda_<lambda>_tau_<tau>.png`

Baseline comparator folder:

- `baseline/` repeats the lambda/tau sweep using constant predictions per route (`mu` mean realized reward)
- Includes `operating_points.csv`, `frontier_points.csv`, `summary.json`, `route_mean_rewards.csv`, and matching charts/pies
