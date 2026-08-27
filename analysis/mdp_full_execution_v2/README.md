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


## Marginal-value stopping cannot abstain early (negative result)

`replay_early_stop_dev_v1/` adds a `_value_stop` variant that replaces the
probability threshold `p_any <= tau` with a marginal-value rule
`max_m p_m / cost_m <= T`, selected on calibration under the same retained-
correctness constraint. It was introduced to make the policy quit earlier: the
`_abstain` policies were observed abstaining only after 12--20 failures.

It does not achieve that. Across both budgets audited, **0% of `_value_stop`
abstentions occur while the scout still has draws remaining**, versus 96.2% for
the `tau` rule at budget 0.01667. The `_value_stop` policy always exhausts all
ten scout draws first; its abstention depth (median 11 at low budget, 17 at
high budget) is set by scout exhaustion, not by a decision to stop.

The cause is an expressiveness limit of the rule class, not a bad threshold.
Dividing by cost makes the statistic scout-dominated, and the two routes do not
overlap in it. On the unconstrained decay policy at budget 0.07306:

| statistic | value |
|---|---:|
| scout `p/cost` while available, minimum over 2,373 decisions | 24.03 |
| oss120 `p/cost`, maximum over 3,752 decisions | 20.58 |
| overlap | none |

Scout's floor strictly exceeds oss120's ceiling, so **any `T` large enough to
stop a single scout re-roll forbids 100% of oss120 calls** (and 32.9% of
oss20 calls). The rule cannot express "stop re-rolling the cheap route but do
escalate to the expensive one" at any threshold. Route costs span 53x
(scout 0.00055, oss20 0.00396, oss120 0.02973) while the decayed scout
probability stays above 1.6% for its whole ten-draw run.

**The defect is in the action rule, not the stopping rule.** Note that
`max_m p_m/cost_m <= T` is *exactly equivalent* to `max_m (p_m*R - cost_m) <= 0`
for `R = 1/T`, so `_value_stop` already applies the myopic-optimal stopping
criterion. Replacing the stopping statistic changes nothing. What fails is that
the action rule `argmax_m p_m/cost_m` is **scale-invariant and therefore
independent of `R`**: it prefers the cheapest route at every value of a correct
answer, so it grinds the scout to exhaustion no matter how `T` is set. A single
`T` is then asked to do two conflicting jobs -- permit oss120 (`T <= 20.58`) and
stop scout re-rolls (`T > 24.03`) -- along a trajectory it cannot reshape.

**This does not make the ratio rule wrong.** Under a hard per-query budget it is the
standard greedy knapsack rule and is exactly the RoR baseline's allocation
(arXiv 2607.08665, "estimated marginal correctness per unit cost"). What it lacks
is a stopping theory, not a correct ordering: RoR stops only when the budget is
exhausted. The budget-swept `counts` family is therefore kept on by default as the
RoR-faithful cell, and the value sweep is offered as the Lagrangian dual of the same
problem rather than as a correction to it. See `PAPER_PLAN.md` for the primal/dual
table and the caveats.

Using the value difference for *selection* as well dissolves the conflict,
because one scalar `R` then controls escalation and stopping coherently.
Measured over all 3,799 decisions at budget 0.07306:

| `R` ($/correct) | `argmax p/cost` | `argmax (p*R - cost)` | disagree | myopic stop |
|---:|---|---|---:|---:|
| 0.02 | scout 58% oss20 39% oss120 3% | oss20 62% scout 38% | 28.4% | 54.2% |
| 0.05 | *(unchanged)* | oss20 95% scout 5% | 55.5% | 27.3% |
| 0.10 | *(unchanged)* | oss20 90% oss120 10% | 68.1% | 12.8% |
| 0.25 | *(unchanged)* | oss120 68% oss20 32% | 88.7% | 0.0% |
| 1.00 | *(unchanged)* | oss120 99% oss20 1% | 95.9% | 0.0% |

The ratio rule's route mix is constant down the whole column; the value rule
sweeps from scout to oss120 as `R` rises while its stopping rate falls from 54%
to 0%. The `tau` rule outperforms on early stopping because `p_any` is
cost-blind and therefore encodes "this problem looks hopeless" rather than
"this next dollar is inefficient" -- a crude proxy for small `R`.

Finite-horizon planning is a separate, later correction and will push abstention
*later*, not earlier: with `Q_m = p_m*R - cost_m + (1-p_m)*V(s')` and
`V >= 0`, the continuation term only ever makes continuing more attractive.

`_value_stop` is nevertheless **not** globally worse, and should not be
reported as a failed policy. It contributes 15 Pareto points versus 12 for
`tau`, and `sequential_decay_value_stop` owns the entire mid-cost region
(0.01038--0.01714 at 61.9--67.4% correctness) where no `tau` policy appears --
reaching 67.44% at 0.01714 against `sequential_decay_abstain`'s 68.37% at
0.02084. It is dominated at the headline operating point: 74.19% at 0.03072
versus `sequential_decay_abstain`'s 74.19% at 0.02870 for the same correctness.

The honest summary is that `_value_stop` is a useful frontier contributor for a
different reason than intended -- "run the cheap route to exhaustion, then quit
before escalating" -- and is not evidence that marginal-value stopping works.
Early abstention remains unsolved.

## Free-start protocol is currently vacuous (2026-08-27, development-only)

`replay_free_start_dev_v1/` removes the mandatory scout, exposing
{abstain, scout, oss20, oss120} at the root, on all 341 problems including those
a mandatory scout would have solved. The dataset builder gained a matching
`--start-protocol free_start` mode (14,844 reachable examples, all problems
present at depth 0).

The replay changes almost nothing: **104 of 108 (budget, policy) cells are
bit-identical to the scout-first replay.** The four differences are all at the
degenerate smallest budget 0.00055, where root abstention becomes affordable
(34.42% at 0.00038--0.00045 versus 34.88% at 0.00058).

The reason is the same cost-normalized action rule. **Every policy chooses scout
at the root 100% of the time at every budget above the smallest**, because
`prior/cost` is 432.8 for scout against 96.0 for oss20 and 21.5 for oss120. The
free-start policy simply re-imposes the mandatory scout on itself. This is not
an artifact of the model being untrained on root states: the `counts` family
uses train priors rather than the model and behaves identically.

So the free-start ablation cannot currently answer its scientific question
("should the scout be called at all?") -- it is evaluated under a rule that is
structurally incapable of declining. It is blocked on the same action-rule fix,
and should be re-run once selection uses `argmax_m (p_m*R - cost_m)`. Treat the
present free-start numbers as a protocol check, not a result.

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
