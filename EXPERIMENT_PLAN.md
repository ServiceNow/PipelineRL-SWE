# Experiment plan: what varies, what is held fixed, and what each run answers
*Written 2026-09-23, before the pool_v2 data lands, so the arms are pre-registered rather than chosen
after seeing results. Supersedes the ad-hoc arm lists in THREADS_IN_PROGRESS.*

## 0. The one rule that shapes everything

On the old pool the `content_*` **envelope over 18 variants** beat the Zero Router by +5 to +9pt,
while a single pre-registered variant beat it by **+0.0pt**. Most of that is grid coverage, not
cherry-picking -- but we cannot tell the two apart after the fact, and neither can a reviewer.

**So: ONE pre-registered arm per belief source, with the full grid, decided before the run.**
Everything else is an ablation reported separately and never folded into the headline number.

Four more rules, each of which was learned by getting it wrong in this project:
* **Compare at equal SPEND, never at equal R.** Two policies that choose their own spend land at
  different frontier points; utility at a common R credits whichever one spent more (3b-lxxxvii).
* **Convex hulls for every policy, baselines included.** A fixed plan can randomise between two
  plans to hit any intermediate cost; the step envelope strands it in the gaps and inflates our gap
  by roughly 2x.
* **Matched estimator, grid and split per arm.** Twice caused a wrong retraction.
* **Per-provider hygiene first**: truncation, tool-call and reasoning-presence rates per rung,
  before any pass@1 is compared to any other (3b-lxxxix).

## 1. The dimensions

### A. Belief source -- what predicts p (THE MAIN AXIS)
| # | arm | what it adds | status |
|---|---|---|---|
| A0 | pool prior + count decay, no features | the RoR v1 baseline | exists |
| A1 | prefill probe + count decay | today's deployed rule | exists |
| A2 | prefill probe + analytic success-count posterior (with replacement) | principled decay | exists (3b-lxxvi) |
| A3 | probe re-read on the latest failed attempt | reads the failure | exists (3b-lxxiv) |
| A4 | probe re-read on the whole trajectory, NO decay | reads the history | exists (3b-lxxxii) |
| A5 | spike-and-slab Beta head (distribution over q) | tail resolution | exists (3b-lxxxiii) |
| A6 | oracle: empirical q from held-out draws | upper reference, not a method | new, cheap |
| A7a | frozen sentence embedding (MiniLM / e5) + the SAME head | "is the 4B prefill doing anything?" | **to build** |
| A7b | TF-IDF + the same head | the floor: is any of this semantic? | **to build** |
| A7c | *optional* fine-tuned small encoder | a stronger baseline that COSTS training | only if A7a is close |

**A7 has to be protocol-matched or it answers nothing.** Same frozen-encoder-plus-linear-head
recipe, same training data, same calibration, same grid -- vary ONLY the encoder. Then the
comparison is clean and has two axes, which must both be reported:

* **Accuracy.** If a 22M sentence encoder matches the 4B prefill under the identical frozen
  protocol, the prefill does not earn its keep, because it is also ~100x more expensive per query
  (~0.006c or ~0.3 GPU-seconds against ~0.0001c). The paper's framing would have to become "one
  very cheap embedding cuts your API bill", which is a *better* result, just not the one we assumed.
* **Training cost, which is where the asymmetry lives.** Ours is a linear head on frozen features:
  minutes on CPU, and it is refit per R and per pool (0a/E1), so that cost RECURS. A baseline that
  only matches us after fine-tuning its encoder (A7c) pays a much larger recurring cost for the same
  deployment, and that has to be stated rather than hidden in a pass@1 column. A7a and A7b, being
  frozen, are as cheap to train as we are -- so if either ties us on accuracy, we lose outright.

The hypothesis worth testing, and the reason to expect the 4B to win: predicting whether a
*reasoning model will solve* a problem may need more than surface semantics. A7 is what turns that
from an assumption into a measurement.

### B. Decision machinery -- how beliefs become actions
| # | arm | note |
|---|---|---|
| B0 | best FIXED plan (Zero Router), convex hull | the baseline to beat |
| B1 | myopic Lagrangian greedy: draw argmax p*R - c, abstain when all < 0 | deployed rule |
| B2 | knapsack by marginal p/c, parameterised by BUDGET not price | measured: +3pt at tight budgets, -3pt at loose ones with 2-3-draw beliefs (`measure_knapsack_policy.py`) |
| B3 | h-step lookahead (h=2 and h=H) | monotone-case result says it only reorders routes; verify it |
| B4 | agreement-gated escalation | non-abstaining baseline |
| B5 | budget-aware best-of-K | non-abstaining baseline |

### C. Constraint form
C0 price only (sweep R) | C1 per-episode cap only | C2 two-constraint B x R with linked cap B = k*R.
Accounting: realised spend, not expected.

### D. Pool
D0 LCB (892 problems, 5 rungs, temporal split) | D1 BigCodeBench (~1,050 tasks, same rungs, random
split at LCB's 62/38 ratio). **D1 matters because mean q is ~0.28 there against ~0.80 on LCB** --
the band where stopping and escalation decisions are actually live.

### E. Belief training distribution
E0 i.i.d. states | E1 the policy's own visited states (DAgger-style). Extraction is R-independent;
only the heads depend on R, so E1 costs a head refit per R, not a re-extraction.

## 2. What actually gets run (not the cross product)

**Run 1 -- headline.** A0..A5 + A7 x {D0, D1}, machinery fixed at B1, constraint C2, 5 seeds.
One pre-registered arm each. Output: frontier + savings-at-fixed-accuracy. ~14 replay sweeps.
A6 (oracle) alongside as the ceiling.

**Run 2 -- machinery ablation.** Winner of Run 1 x {B0..B5} on D0, then the winner of that on D1.
Answers "does the knapsack/budget parameterisation beat the price threshold once beliefs are good",
which is open: B2's deficit at loose budgets was a belief-quality artifact of 2-3-draw estimates.

**Run 3 -- constraint ablation.** Winner x {C0, C1, C2} on both pools.

**Run 4 -- training distribution.** Winner x {E0, E1}, both pools.

**Run 5 -- whether vs which.** The crossing experiment (stop source x route source) on both pools.
**Re-run from scratch**: the old "whether, not which" result came from a 3-rung pool at a 7x price
spread where switching was barely possible, and the decorrelation table (3b-lxxxviii) says switching
beats resampling by 20-50pt conditional on a failure.

**Run 6 -- belief quality curve.** Gain vs number of draws the belief is estimated from, with our
probe's position marked on it. This is the figure that answers "is the modest effect real or an
artifact of thin data", and it needs the full-depth pool (3b-lxxxvii).

## 3. Order, and what blocks what
1. LCB collection completes -> hygiene check -> tensors + split manifest.
2. Prefill extraction on pool_v2 (GPU): prompt-only, plus the reachable-state lattice for A3/A4.
3. BCB collection (~$9) -> same hygiene -> tensors. Can run during step 2.
4. Run 1, then 5 and 6 (they answer the framing questions), then 2, 3, 4.
5. Paper rewrite on market-price accounting.

## 4. Open decisions
* Which A-arm is pre-registered as "our method" for the headline. Current best guess A5, but A4 is
  simpler and the choice must be made on CALIBRATION on the train/cal split, not on the frontier.
* Whether BCB gets the same draw counts as LCB (its per-draw cost is ~0.3x, so depth is cheap).
* Whether to add gpt-oss-20b-high as a 6th rung (dominated on single-draw cost/accuracy, but the
  decorrelation table says a dominated rung can still be the best thing to call after a failure).
