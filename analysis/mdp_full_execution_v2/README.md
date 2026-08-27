# Full-execution MDP figures (protocol v2)

These figures replace the public/private-verifier frontier as the primary LCB
allocation protocol. They were generated from saved model outputs—no new model
calls—using:

- artifact root: `/mnt/llmd/results/exps/aristides/reason/mdp_full_execution_v2_1787679948`
- 341 problems × 3 models × 10 draws
- canonical split: 170 train / 85 calibration / 86 test problems
- mandatory scout first
- router entry only after full-execution failure
- full execution used for both routing verdict and final correctness
- realized prompt + completion tokens used for reported cost

The current plots contain count-based adaptive policies and precisely named
fixed baselines. The first corrected learned run
(`lcb_mdp_full_execution_seed17_1787680221`) achieved held-out prediction AUCs
of about 0.79--0.86 but did not beat counts or the cascade at matched cost. Its
learned curve is diagnostic-only: audit found that full histories caused
context truncation and that the abstention head described only the next draws.

The current builder instead uses problem + explicit per-route failure/remaining
counts + only the latest failed attempt. Its abstention target is positive only
when no successful valid draw remains in any route. Rebuilding the saved LCB
tensors changed 2,092 such labels and produced zero over-8,192-token states in a
1,200-state sample (sample maximum 4,363). One gated diagnostic retraining is
still required before any learned curve is added to the paper figures.


## Learned-policy diagnostics

`replay_mdp_full_execution.py` writes three complementary artifacts:

- `replay_results.json`: aggregate frontier and fixed-baseline metrics;
- `episode_traces.jsonl`: per-problem/order/policy outcomes plus compact decision traces
  (state hash, available routes, probabilities, utility/cost scores, chosen action, and result);
- `diagnostics.json`: route-choice/pass counts, mean predictions, and paired 95% bootstrap
  intervals. Bootstrap resampling is clustered by problem, preserving the repeated draw
  orderings within each problem.

The trace output intentionally excludes problem text and generated code.

Regenerate with:

```bash
python pipelinerl/swe/scripts/livecodebench/generate_mdp_full_execution_figures.py \
  --replay-results /mnt/llmd/results/exps/aristides/reason/mdp_full_execution_v2_1787679948/replay_counts_usd/replay_results.json \
  --output-dir analysis/mdp_full_execution_v2
```
