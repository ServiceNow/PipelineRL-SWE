# Offline Router Analysis (New)

This workflow uses runtime `router_trace` streams produced by the actor, then optionally scores
those traces with the policy performance head, and finally sweeps routing thresholds.

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

The actor writes:

- `streams/router_trace/...` (train)
- `streams/router_trace_test/...` (test)

## 2) Score traces with performance head

```bash
python -m pipelinerl.swe.scripts.new.score_router_traces \
  --input-glob "/path/to/run/streams/router_trace_test/*/*/0.jsonl" \
  --model-path "/path/to/run/finetune/current" \
  --output-jsonl "/path/to/run/router_trace_test_scored.jsonl"
```

## 3) Analyze routing tradeoffs (cost/performance/regret)

```bash
python -m pipelinerl.swe.scripts.new.analyze_router_traces \
  --input-glob "/path/to/run/router_trace_test_scored.jsonl" \
  --output-dir "/path/to/run/router_analysis" \
  --score-key policy_value_prompt_last_all \
  --threshold-start 0.0 --threshold-stop 1.0 --threshold-step 0.05
```

Optional cost calibration (e.g. USD per 1K tokens):

```bash
  --policy-cost-per-1k 0.002 \
  --expert-costs-per-1k "0.004,0.012"
```

Outputs:

- `threshold_metrics.csv` (main table: cost, performance, regret, route shares)
- `baseline_metrics.csv` (policy-only / oracle / best-single-expert baselines)
- `summary.json`
- `threshold_metrics.png`
- `pies/routing_pie_*.png`
