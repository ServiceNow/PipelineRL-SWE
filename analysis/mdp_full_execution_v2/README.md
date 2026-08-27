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
1,200-state sample (sample maximum 4,363).

The gated latest-attempt run `lcb_mdp_latest_attempt_seed17_1787808515`
completed but did not beat counts or the cascade. Its replay diagnostics are in
`replay_diagnostics_v1/` under that artifact root. The learned policy barely
reduced scout success probability after repeated failures (mean 0.176 after one
scout failure versus 0.163 after nine), so probability/cost routing exhausted
cheap scout draws before escalating. This is a negative learned-policy result;
do not add its curve as a headline improvement.


A predeclared no-retraining hybrid replay now supplies the missing structural
update by multiplying each learned probability by the count prior
`2 / (2 + route failures)`. Results are in `replay_hybrid_v1/` under the same
artifact root. At budget 0.07306, this hybrid matched counts at 74.42%
correctness while lowering mean cost from 0.04008 to 0.03669 (8.45%; paired
95% CI for the cost delta [-0.00639, -0.00077]). Its abstaining high-budget
point achieved 74.19% at 0.02870, statistically indistinguishable in
correctness from the neighboring 74.42%-at-0.04008 count point, with a 28.4%
cost reduction (cost-delta CI [-0.01841, -0.00517]). The hybrid adds useful
Pareto points but remains worse than counts at the smallest budget.

These are provisional results from one trained seed and five replay orderings.
They support a factorized policy--reliable count decay plus a learned semantic
residual--but require 20-ordering and multi-training-seed replication before use
as a headline claim. The very low scout rates diagnosed here are conditional
on prior mandatory-scout failure: marginal saved-draw scout success is 28.4%,
whereas next-scout success after one held-out failure is only 3.15%.


## Learned-policy diagnostics

`replay_mdp_full_execution.py` writes three complementary artifacts:

- `replay_results.json`: aggregate frontier and fixed-baseline metrics;
- `episode_traces.jsonl`: per-problem/order/policy outcomes plus compact decision traces
  (state hash, available routes, probabilities, utility/cost scores, chosen action, and result);
- `diagnostics.json`: route-choice/pass counts, mean predictions, and paired 95% bootstrap
  intervals. Bootstrap resampling is clustered by problem, preserving the repeated draw
  orderings within each problem.
- `hybrid_frontier_comparisons.json` in the hybrid replay directory: paired comparisons between
  count, pure learned, count-decayed learned, and count-decayed learned-with-abstention frontiers.

The trace output intentionally excludes problem text and generated code.

Regenerate with:

```bash
python pipelinerl/swe/scripts/livecodebench/generate_mdp_full_execution_figures.py \
  --replay-results /mnt/llmd/results/exps/aristides/reason/mdp_full_execution_v2_1787679948/replay_counts_usd/replay_results.json \
  --output-dir analysis/mdp_full_execution_v2
```
