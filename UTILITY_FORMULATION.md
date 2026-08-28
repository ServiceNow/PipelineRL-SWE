# Utility-Based Routing and Abstention

This note formalizes the sequential inference problem used in the current full-execution
experiments, explains how the utility-controlled policy differs from the RoR-style
probability-to-cost rule, and records the limits of the implementation. It is intended to be the
mathematical source of truth for the paper. In particular, the current policy is a **myopic
utility policy**. It is related to a Lagrangian relaxation of budget-constrained routing, but it is
not yet the exact solution of the full finite-horizon MDP.

## 1. Setting

For one problem, let the state at decision time be

\[
s = (x, h, n, k),
\]

where \(x\) is the problem, \(h\) is the history of verified failed attempts, \(n_m\) is the
number of failures observed from route \(m\), and \(k_m\) is the number of draws remaining for
route \(m\).

An action is either another draw from an available route \(m \in \mathcal M(s)\), or the explicit
stop/reject action \(\bot\). A route call has expected cost \(c_m(s)>0\). The policy estimates

\[
p_m(s)=\Pr(\text{the next draw from route }m\text{ passes full execution}\mid s).
\]

Success terminates the episode. A failed attempt produces a new state \(s'\) with another observed
failure, an updated history, and one fewer available draw for the selected route.

Let \(R>0\) be the deployment value of a verified correct answer, measured in the same units as
cost. Let \(d\geq0\) be the cost of rejecting or returning no answer. The current experiments use
US dollars and initially set \(d=0\).

## 2. The RoR-style budget and ratio rule

The budget-constrained objective can be written schematically as

\[
\max_\pi \Pr_\pi(\text{eventual success})
\quad\text{subject to}\quad
\mathbb E_\pi[C]\leq B,
\]

where \(B\) is an inference budget and \(C\) is total spend. The RoR-style baseline maintains a
count-based success belief for each route. In our implementation, after \(n_m\) observed failures,

\[
\widehat p_m(n_m)=\frac{\alpha\,\bar p_m}{\alpha+n_m},
\]

where \(\bar p_m\) is the route's training prior and \(\alpha\) is a pseudo-count. Because an
episode stops on its first success, only failure updates are needed inside a continuing episode.

The next route is chosen greedily by marginal success per unit cost:

\[
m^*_{\mathrm{ratio}}=\arg\max_{m\in\mathcal M(s)}
\frac{\widehat p_m(n_m)}{c_m(s)}.
\]

The episode continues until a draw succeeds, no draw is affordable, no draw remains, or the hard
budget is exhausted.

This ratio is a natural marginal-allocation heuristic. It should not be described as an exact
optimizer for every finite, indivisible, history-dependent stochastic routing problem. Its relevant
limitation here is narrower: **the unthresholded ratio argmax only ranks available calls. It does
not compare the best call with doing nothing.** If every available route has nonnegative predicted
success, one route always has the largest ratio, however poor its absolute value may be.

Consequently, the standard ratio-and-budget policy can stop because it succeeded or ran out of
budget, but it cannot selectively say, "another affordable call is not worth its cost." This is
why count-only RoR continues spending on queries that its budget still permits even when the
remaining success probability is extremely small.

## 3. Our current myopic utility rule

We assign the immediate net value

\[
Q_{\mathrm{myopic}}(s,m)=p_m(s)R-c_m(s)
\]

to another route call, and \(Q(s,\bot)=-d\) to stopping. The decision is

\[
a^*(s)=\arg\max\left(\{-d\}\cup
\{p_m(s)R-c_m(s):m\in\mathcal M(s)\}\right).
\]

Thus the policy abstains exactly when

\[
\max_m[p_m(s)R-c_m(s)]\leq-d.
\]

With zero reject cost, this reduces to

\[
\max_m[p_m(s)R-c_m(s)]\leq0.
\]

This rule has three useful properties:

1. **Stopping is an ordinary action.** No independently tuned abstention threshold is required.
2. **The decision has an absolute scale.** A route must repay its cost in expected correctness
   value, not merely have a better ratio than the other routes.
3. **One deployment parameter controls the frontier.** Increasing \(R\) makes correctness more
   valuable, causing more escalation and less abstention; decreasing \(R\) favors cheap routes and
   earlier rejection.

### Ratio threshold versus utility

For a single route and \(d=0\), continuing under utility is equivalent to

\[
\frac{p_m(s)}{c_m(s)}>\frac{1}{R}.
\]

Therefore a ratio *can* support abstention if it is augmented with an absolute threshold. The
problem is not the mathematical ratio itself; it is the ratio-only policy that always takes its
argmax while budget remains. Introducing the threshold \(1/R\) is precisely introducing the
missing value-of-correctness comparison.

The route rankings can also differ. Ratio routing chooses according to \(p_m/c_m\), independently
of \(R\). Utility routing compares \(Rp_m-c_m\), so the preferred route can change with the value of
a correct answer. For example:

| route | \(p_m\) | \(c_m\) | \(p_m/c_m\) |
|---|---:|---:|---:|
| cheap | 0.10 | 0.01 | 10 |
| expensive | 0.50 | 0.10 | 5 |

The ratio rule always prefers the cheap route. At \(R=0.15\), the utilities are \(0.005\) and
\(-0.025\), so utility also chooses the cheap route. At \(R=1\), they are \(0.09\) and \(0.40\),
so utility chooses the expensive route. This is the behavior seen in the free-start replay: the
root action moves from reject, to scout, to larger solvers as \(R\) increases.

## 4. Relationship to a Lagrangian relaxation

A Lagrangian relaxation of the budget-constrained objective is

\[
\max_\pi\left[
\Pr_\pi(\text{success})-\lambda\,\mathbb E_\pi[C]
\right],\qquad\lambda>0.
\]

Multiplying by \(R=1/\lambda\) gives the equivalent policy-level objective

\[
\max_\pi\left[
R\Pr_\pi(\text{success})-\mathbb E_\pi[C]
\right].
\]

This explains why sweeping \(R\) traces a correctness-cost frontier and why a zero-value stopping
action arises naturally. It is appropriate to describe the **objective** as a Lagrangian or
utility form of cost-constrained routing.

It is not appropriate to claim that the currently implemented one-step decision rule is the exact
Lagrangian solution of the sequential MDP. That requires continuation value.

## 5. The full finite-horizon MDP

The Bellman equation is

\[
V(s)=\max\left\{
-d,
\max_{m\in\mathcal M(s)}\left[
-c_m(s)+p_m(s)R+(1-p_m(s))\,\mathbb E[V(s')\mid s,m,\text{failure}]
\right]
\right\}.
\]

The current implementation drops the continuation term and uses

\[
V_{\mathrm{myopic}}(s)=\max\left\{-d,\max_m[-c_m(s)+p_m(s)R]\right\}.
\]

Because future value is normally nonnegative relative to rejection, omitting continuation can make
the policy stop too early or choose the wrong route when a failed attempt would still leave useful
options. Conversely, inaccurate probabilities can also make it continue too long. The planned
continuation-value ablation should compare:

1. the current myopic rule;
2. structural finite-horizon continuation computed from empirical route/depth hazards; and
3. optionally, a learned semantic residual on top of that structural value.

## 6. Beliefs are separate from the decision rule

The formulation and the probability model are independent experimental axes:

| beliefs | budget + ratio | utility + stop |
|---|---|---|
| count-based | RoR-style baseline | formulation-only ablation |
| query-conditioned | learned ratio ablation | proposed method |

This 2-by-2 comparison is essential. The current development result shows that utility plus
count-only beliefs does not produce the cheap matched-accuracy point; the saving appears only when
the policy combines an explicit stop action with query-conditioned beliefs. The scientific claim
is therefore not "subtracting cost fixes routing." It is that semantic beliefs can identify which
failure histories are exhausted, and the utility action space lets the policy act on that
information.

The beliefs must nevertheless be calibrated. Route selection compares probabilities across models
with different costs, and stopping depends on their absolute scale. Good AUC alone is insufficient.
Calibration should be fit without final-test access and should condition on route-specific failure
history, not merely total failure depth.

## 7. What the present replay establishes

On the current 86-problem development test split with five stored-draw orderings, the raw learned
utility policy reaches the collected-pool ceiling of 81.40% at an average solver-call cost of
\$0.06690. The extended RoR-style count-and-ratio baseline reaches the same correctness at
\$0.13793, a 51.5% reduction in recorded solver cost.

At that operating point, every abstained episode belongs to a problem for which none of the 30
stored route/draw outcomes succeeds. This is evidence of selective exhaustion detection within the
offline replay pool. It does **not** prove that those problems are intrinsically unsolvable or that
fresh generations would never succeed.

These numbers are development evidence, not the final paper estimate. The operating point was
identified from a repeatedly inspected split; the router is miscalibrated; the result uses one
training seed; and router inference, verifier execution, and reject costs are not yet included.

## 8. Required paper terminology and reporting

Use the following language consistently:

- **RoR-style count-and-ratio baseline**, not an exact primal optimizer;
- **utility-form or Lagrangian objective**, when discussing the policy-level objective;
- **myopic utility policy**, for the currently implemented \(pR-c\) decision rule;
- **continuation-aware utility policy**, only after the Bellman continuation term is implemented;
- **pool-unsolved problem**, for a problem with no success among the stored draws; and
- **solver-call cost**, until router inference and execution costs are included.

The final paper should report both correctness at matched total cost and total cost at matched
correctness, with paired problem-clustered intervals. It should also show sensitivity to \(R\),
reject cost \(d\), router cost, execution cost, and the number of available draws.
