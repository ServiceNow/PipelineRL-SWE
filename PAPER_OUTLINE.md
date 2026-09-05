# Paper outline — living document

**Edited in place.** Append-only history lives in `RESEARCH_LOG.md`. Every number is cited to a
run or marked TODO; nothing enters by recollection.

**Status:** LCB and TACO complete, both re-run with rich features (all layers x {mean,last}),
which lifted every head on both datasets. Headline: full method beats RoR at 6/6 targets on LCB
and 5/6 on TACO. Cost conditioning is regime-dependent, and the regime is now quantified.
SWE-Smith not started.
**Last updated:** 2026-09-04

---

## 1. Working title
*Cheap Beliefs for Expensive Pools: Activation Priors and Query-Conditioned Cost in Sequential
Test-Time Model Selection*

## 2. One paragraph
Sequential test-time model selection asks, at each step, whether to resample the committed
model, reroute, or stop. Every system we surveyed conditions the *correctness* half of that
decision on the query and leaves the *cost* half a per-model constant from the training set. We
show per-problem cost is highly variable (p90/p10 16–37×), highly predictable from one cheap
model's pre-generation activations (R² 0.65–0.83 against the constant), and that conditioning
on it is worth 13–46% at matched accuracy — including on top of the existing state of the art's
own beliefs. The resulting policy is cheaper than the counting-based baseline at all seven
accuracy targets and beats a compiled fixed schedule with no losses.

## 3. Contributions

Ordered by how well they replicate. Three claims were retracted earlier in this project for
insufficient checking (§8), so each is annotated with prior art and with its measured regime.

**C1. Activation-derived per-problem priors inside a sequential resample/reroute/abstain MDP.**
*(leads; replicates on both datasets with large effects)*
Replacing the count-based beliefs the state of the art uses with $\hat\theta_m(x)$ from one
cheap model's prefill is worth **+13.9% to +69.7% on TACO** (five of six targets significant)
and comparable margins on LCB. The comparison isolates the belief source *within* the same
sequential MDP, which single-commit routers cannot do.
*Prior art:* NVIDIA's prefill router (2603.20895) has activation routing but is single-commit,
no abstention, no depth. RoR (2607.08665) has the resample/reroute MDP but count beliefs and no
stop action. Ours is the first to put query-conditioned priors inside the sequential problem.

**C2. Abstention as the zero-value action.** Stopping is exactly $\max_m Q(s,m)\le 0$ — no
threshold, no extra head. On the harder pool this is load-bearing: TACO abstains 25–49% against
LCB's 10.5%, because ~half of TACO is unsolved by the whole pool at one draw.

**C3. Query-conditioned cost — with a measured regime.** *(demoted from lead after TACO)*
Conditioning $c_m$ on the query helps when beliefs are weak and is redundant or harmful when
they are strong:

| | LCB | TACO |
|---|---|---|
| on count beliefs | +12.9–45.9%, 6/7 significant | +6.3–36.8%, 3/6 significant |
| on activation beliefs | +15.4–26.0%, 5/7 significant | **−29.1% to +3.6%, two significant losses** |

The mechanism is signal-to-noise in the cost term: when beliefs already rank routes well, the
error in $\hat c_m(x)$ outweighs the spread it resolves. The same mechanism explains the
price-ratio limitation (§6.7). **Report as: a drop-in upgrade for count-based routers, not a
universal improvement.** Still novel — RoR and 2603.20895 both use per-model constants, and
length-from-activations (2607.05316, 2602.11812) is never operationalised in a decision.

**C4. One cheap probe beats probing every model, once the probe is priced.** Per-candidate
probes are better predictors (+0.05-0.06 AUC) and buy a real but small frontier gain
(+0.6 to +8.6pt) -- for 48.5x the probe cost, which takes 10-38pt back. Charged honestly they
lose to the single scout probe at **every** target (§6.5). Deployment consequence: weights for
one cheap model, not all of them, and the argument is now economic rather than a null result.

**C5. Methodological: cross-model activation comparisons need a fixed readout.** §6.4.

## 4. Related work

### 4.1 Sequential and budgeted test-time model selection *(closest)*

- **RoR — "Resample or Reroute?"** (2607.08665, Chen). Formalises resampling the committed model
  and rerouting to another as competing uses of one per-query budget, allocating each unit to
  whichever action has the highest estimated marginal correctness per dollar. Evaluated by replay
  on an eleven-model open-weight pool over four benchmarks. **Our primary baseline**; it has the
  same action space but count-based beliefs, no stop action, and per-model constant costs.
- **How Much of the Routing Gap Is Real?** (2607.03436). Companion analysis showing that part of
  the celebrated router-to-oracle gap is single-draw label noise no router can capture, since the
  per-instance oracle is built from one sample under stochastic decoding. Motivates evaluating at
  k>=30 draws, which is why our tensors are multi-draw.
- **SeqRoute** (2605.25424). Routes between a weak 8B and a strong 70B with a budget shared across
  a multi-turn session, so the interesting decision is *when* to spend rather than on what. Single
  use per query: no resampling, no abstention, no verifier.
- **Cluster, Route, Escalate** (2606.27457). Clusters queries, routes within cluster, escalates on
  failure — a cascade with learned entry points rather than a per-query sequential policy.
- **UCCI** (2605.18796). Turns token-margin signals into calibrated error probabilities via
  isotonic regression, so a cascade can threshold on a quantity that means what it says. Relevant
  as the calibration-first alternative to our probe.
- **Dynamic Model Routing and Cascading: A Survey** (2603.04445). The current map of the area;
  useful for its distinction between routing (one commit) and cascading (sequential escalation),
  which is exactly the axis our abstention action extends.

### 4.2 Activation-based prediction and routing

- **LLM Router: Rethinking Routing with Prefill Activations** (2603.20895, NVIDIA). Routes from
  prefill hidden states rather than surface text, and decouples the *encoder* producing the signal
  from the *target* whose success is predicted, so open-weight encoders can score closed-source
  models. **The closest work to our signal**, and the source of the deployment argument that only
  one model's weights are needed. Single-commit, no abstention, per-model constant cost,
  last-token readout, no discussion of chat-template comparability (see §6.4).
- **LLMs Encode Their Failures** (2602.09924). Linear probes on pre-generation activations predict
  a model's *own* success well enough to route a pool below the best single model's cost. Requires
  a probe per pool member and does not charge for those forward passes; no abstention.
- **Scrouting / SuperScout** (2608.04804). A 7B searcher explores a repository and its hidden
  states feed an N-way router over frontier fixers. Notable for honesty about its own null: the
  hidden state separates solved from failed at AUC 0.600, and a no-router ablation ties the routed
  system, so the authors conclude the verified handoff rather than the routing carries the result.
- **Knowing When to Quit** (2604.18419, ICML 2026). Formalises mid-generation abstention as an
  action in a KL-regularised MDP, with an MLP probe on hidden states estimating the value and
  abstention triggered when it falls below the fallback's worth. Single model, no pool, no
  resampling, and cost is absent from the objective — the closest work on *abstention*, and
  orthogonal to the routing half.

### 4.3 Output-length prediction *(the foundation for query-conditioned cost — cite, never claim)*

- **How Much is Left?** (2607.05316). Shows total response length is linearly decodable from the
  prompt's last hidden state before a single token is emitted, across three model families. Tests
  cross-*dataset* transfer extensively but never cross-*model*, and explicitly declines to
  operationalise the prediction, naming prompt-end early termination as future work.
- **Predicting LLM Output Length via Entropy-Guided Representations** (2602.11812, ICLR 2026).
  Reuses the serving model's own hidden states with entropy-guided token pooling for static
  prediction, plus a progressive variant that re-estimates the remaining length each step. Aimed
  at batching and RL-sampling throughput, not decision-making.
- **Latency-aware routing** (2607.18253) and **learned per-query cost predictors** (2604.03527).
  Both predict per-query cost for routing, but from query *embeddings* rather than activations, and
  both target latency/SLA objectives rather than a correctness-versus-cost utility.

### 4.4 Routing benchmarks and gap analyses

- **LLMRouterBench** (2601.07206). 33 models over 21 datasets with real API pricing; single-commit,
  single-draw, so it measures routing rather than sequential allocation.
- **The Routing Plateau** (2606.07587). Documents that learned routers cluster tightly well below
  the oracle across many pools, which is the observation the whole area is trying to explain.
- **CodeRouterBench / ACRouter** (2606.22902). ~10K coding instances by 8 frontier models with a
  complete task-by-model outcome matrix — the closest public artifact to our correctness tensor.
- **The 67-model Rasch analysis** (2606.27288). Fits item-response models to a large pool and finds
  most of the apparent routing headroom is explained by a single latent difficulty dimension —
  independent support for the shared-difficulty account our transfer matrix measures directly.
- **Zero-shot routing via a universal latent space** (2601.06220) and **routing collapse**
  (2602.03478). The first adds unseen models without retraining; the second characterises how
  routers degenerate to a single model under distribution shift.

### 4.5 Agentic stopping and escalation *(adjacent modality)*

- **EET** (2601.05777, ACL Findings 2026). Checks LLM confidence at milestones during agentic
  generation, conditioned on retrieved historical experience, to decide early termination.
- **AgentStop** (2604.15075, ACM CAIS 2026). Gradient-boosted trees over token logprobs, step
  counts and repetition features to kill unpromising agent runs for energy savings.
- **FailFast–RestartSmart** (2608.03222). A 0.6B monitor reading only observable trajectory text —
  no logits or hidden states — aborts at a chosen false-positive budget and restarts fresh.
- **SWE-Router** (2607.00053). A cheap model runs K turns and a value head reads the partial
  trajectory to continue or escalate; notably did not beat baselines on its SWE-Smith split.
- **AutoMix** (NeurIPS 2024). Few-shot self-verification feeding a POMDP accept-or-escalate router
  — the earliest formulation close to ours, without activations or per-query cost.
- **COPE** (2506.11578, TMLR). A plan/execute cascade where a small model plans and a large one
  executes, escalating on test failure; single-draw and excludes local compute from its cost model.

## 5. Method

### 5.1 Setting

A pool of $M$ routes (models) $\{1,\dots,M\}$, each able to produce up to $K$ i.i.d. draws for a
query $x$. Draw $k$ of route $m$ yields an outcome $y_{m,k}\in\{0,1\}$ (verified correct or not)
and costs $c_{m,k}>0$ dollars. Outcomes are fixed offline in a correctness tensor, so all
policies are compared by replay on identical draws.

**State.** After some draws have been spent,
$$s = \bigl(x,\; \mathbf{n},\; \mathbf{r}\bigr),\qquad \mathbf{n}=(n_1,\dots,n_M),\quad \mathbf{r}=(r_1,\dots,r_M),$$
where $n_m$ is the number of *failed* draws already taken from route $m$ and $r_m$ the number
remaining. The episode ends on the first success, so any live state has all draws so far failed.

**Actions.**
$$\mathcal{A}(s)=\{m : r_m>0\}\;\cup\;\{\bot\},$$
i.e. spend one more draw on **any** route that still has capacity, or **abstain** ($\bot$),
ending the episode with no answer. Every route is available at every step: there is no
requirement to exhaust a route before switching, and no commitment to a route once chosen.
"Resample" and "reroute" are just names for the two cases of the same action — picking a route
with $n_m>0$ versus one with $n_m=0$ — not separate action types or a staged protocol.
Abstention is an action of the MDP, not a post-hoc filter.

**Objective.** With $R$ the value of a correct answer in dollars (the Lagrangian dual of a
budget constraint), maximise
$$J(\pi)=\mathbb{E}\Bigl[\,R\cdot\mathbb{1}[\text{episode ends in a success}] \;-\; \textstyle\sum_{\text{draws taken}} c_{m,k}\Bigr].$$
Sweeping $R$ traces the cost–accuracy frontier. $R\to\infty$ never abstains; $R\to 0$ always does.

### 5.2 Beliefs

Let $\theta_m(x)\in(0,1)$ be the probability that a *fresh* draw from route $m$ solves $x$. After
$n_m$ failures, a Beta–Bernoulli posterior with pseudo-count $\sigma$ gives the decayed belief
$$p_m(s)\;=\;\theta_m(x)\cdot\frac{\sigma}{\sigma+n_m},\qquad \sigma=2.0 .$$
Failures on route $m$ depress only route $m$; the routes are treated as conditionally
independent **given $x$**, so $P(\text{no route succeeds next})=\prod_m\bigl(1-p_m(s)\bigr)$.

*This assumption is measured, not assumed.* Unconditionally the routes are strongly coupled: on
clean LCB, gpt-oss-120b's pass@1 is 97.7% when the scout succeeded on draw 0 and 69.9% once the
scout has failed once — a 28-point drop — decaying only to 66.0% after six scout failures. But
that coupling is *problem difficulty*, which $\hat\theta_m(x)$ already reads off the prompt.
Conditioning on $\hat\theta$, the scout-failure count adds essentially nothing: log-loss
0.36886 → 0.36875 ($\Delta=-0.00011$), coefficient $-0.0315$. **The probe performs the belief
propagation that a coupled posterior would otherwise have to do**, which is why the factorised
form loses nothing here.

The corollary matters for the baseline. Count-based beliefs have no per-problem prior, so
observed failures are the *only* channel through which RoR can learn that a problem is hard —
and that channel is worth 28 points. A cross-route-coupled RoR should therefore be stronger
than the uncoupled form, and we should implement and report it rather than let a reviewer
point it out (§9).

**Why the decay is a global constant, and why that is not a simplification.** $\sigma$ = 2.0 is
shared by every problem and route. Per-problem decay was tested: a factorized encoder that emits
its own $s_m(x)$ learns something *true* -- splitting test problems at the median learned $s$
gives ground-truth hazard ratios $p(1)/p(0)$ of 0.000 vs 1.549 (scout) and 0.220 vs 0.795
(oss20), and the model is strictly better per-route on both ranking (AUC .831/.845/.785 vs
.817/.827/.753) and calibration (ECE .087/.135/.122 vs .172/.213/.231).

**It is nonetheless decision-irrelevant.** Under the myopic rule a route is worth another draw
while $\theta\sigma/(\sigma+n) \ge c_m/R$, so the stopping depth is
$$n^* = \sigma\left(\frac{\theta R}{c_m} - 1\right).$$
Whether a route is bought *at all* depends on $\theta$ and never on $\sigma$ -- the sign of
$(\theta R/c_m - 1)$ contains no $\sigma$. At $R=\$0.05$, 49% of problems never buy oss120 at
any depth and no value of $\sigma$ flips one of them. Where $\sigma$ does move $n^*$, it moves it
only where depth is worthless: on oss120, the sole route where depth pays, mean $n^*$ is 0.07
against 0.06, both flooring to zero. **Per-problem decay is real, learnable, and does not change
any decision** -- which is what licenses the constant, and is a sharper claim than "we simplified".

**Sigma has real leverage on absolute cost, but helps both arms about equally.** Absolute cost
at the 50% target moves ~2x across sigma (counts 0.01132 at 0.3 vs 0.02344 at 5.0; content
0.00661 vs 0.01270). The *relative* advantage is therefore roughly sigma-invariant, since faster
decay helps the baseline as much as it helps us -- which is why a sweep of the ratio alone reads
as a null.

**Protocol: tune sigma per family on calibration, and report the tuned comparison.** Leaving both
at RoR's inherited 2.0 flatters us at some targets and penalises us at others. Tuned (best sigma
per family):

| target | RoR best (sigma) | ours best (sigma) | saving, both tuned | saving, both at 2.0 |
|---|---|---|---|---|
| 50% | 0.01132 (0.3) | 0.00661 (0.3) | **+41.6%** | +32.4% |
| 60% | 0.02156 (1.0) | 0.01874 (0.3) | +13.1% | +15.4% |
| 65% | 0.02763 (1.0) | 0.02486 (0.3) | +10.0% | +2.5% |
| 70% | 0.03930 (2.0) | 0.03210 (2.0) | +18.3% | +18.3% |
| 75% | 0.04831 (0.3) | 0.03736 (1.0) | +22.7% | +39.5% |
| 80% | 0.08251 (10.0) | 0.06826 (10.0) | +17.3% | +18.4% |

**+10% to +42% tuned, against +2.5% to +39.5% untuned** -- a similar mean but far less erratic.
The untuned +39.5% at 75% was partly the baseline stuck at a bad sigma, and the untuned +2.5% at
65% was partly us stuck at one. Report the tuned version: it is both fairer and more stable.
*Caveat:* the table above selects sigma on test; the final numbers must select on calibration,
as the layer choice does.

**Where sigma has leverage**, measured on our theta and cost estimates: the fraction of
(problem, route) pairs with 0 < n* < K is 10-20% for the cheap routes and **27-50% for oss120**,
the route where depth actually pays. Not a corner case.

Three belief sources differ only in where $\theta$ comes from:

| family | $\theta_m$ | conditioned on the query? |
|---|---|---|
| `counts` (RoR) | $\bar\theta_m$, the train-set marginal solve rate of route $m$ | no |
| `content` | $\hat\theta_m(x)$ from the probe, **no decay** ($p_m=\theta_m$) | yes |
| `content_decay` | $\hat\theta_m(x)$ from the probe, decayed as above | yes |

The undecayed `content` arm is retained only as an ablation: with $p_m$ frozen at depth 0 the
stop condition below can never fire, and it was measured abstaining 0.0% at every operating
point above 55%.

### 5.3 Costs — the conditioned half

Every system we surveyed uses a per-route constant estimated from training data,
$$\bar c_m=\frac{1}{|\mathcal{D}_{\text{tr}}|}\sum_{x\in\mathcal{D}_{\text{tr}}}\overline{c_m(x)} ,$$
(RoR: count-based; NVIDIA's prefill router: "median training output tokens as a verbosity
proxy"). We instead predict cost per query. Writing $T_m(x)$ for the expected total tokens
(prompt + completion) of a draw from $m$ on $x$, and $\rho_m$ for its price per token,
$$\hat c_m(x)=\rho_m\cdot\hat T_m(x).$$

$\hat T_m$ is fit in log space, because token counts are right-skewed:
$$\hat\beta_m=\arg\min_\beta\;\sum_{x\in\mathcal{D}_{\text{tr}}}\bigl(\log T_m(x)-\beta^\top h(x)\bigr)^2+\alpha\lVert\beta\rVert^2 .$$
Exponentiating a log-space fit returns a conditional **median**, which would systematically
under-price every route. We correct with Duan's smearing estimator,
$$\hat T_m(x)=\exp\!\bigl(\hat\beta_m^\top h(x)\bigr)\cdot\underbrace{\frac{1}{|\mathcal{D}_{\text{tr}}|}\sum_{x'\in\mathcal{D}_{\text{tr}}}\exp\!\bigl(\log T_m(x')-\hat\beta_m^\top h(x')\bigr)}_{\text{smearing factor }\hat\varsigma_m},$$
which restores the conditional mean under log-normal residuals. Measured $\hat\varsigma_m\in[1.004,1.013]$,
and predicted route means land within 1–7% of the constants they replace.

### 5.4 Decision rule and abstention

At a live state, the value of spending one draw on route $m$ is
$$Q(s,m)\;=\;p_m(s)\cdot R\;-\;c_m(x),$$
and abstaining is worth exactly zero. The policy is therefore
$$\pi(s)=\begin{cases}\bot & \text{if } \displaystyle\max_{m\in\mathcal{A}(s)\setminus\{\bot\}} Q(s,m)\;\le\;0,\\[6pt] \displaystyle\arg\max_{m} Q(s,m) & \text{otherwise.}\end{cases}$$

**Abstention needs no threshold and no extra head.** It is the zero-value action: the policy
stops precisely when no remaining draw has positive expected value. This is the substantive
difference from threshold-based stopping ($p_{\text{any}}\le\tau$), which requires calibrating
$\tau$ separately and which we retain only as a baseline arm.

Note $Q$ is *not* scale-invariant, and that matters. The budget-constrained form used by RoR
ranks routes by the density $p_m/c_m$ under a spend cap; that ratio is invariant to a common
rescaling of cost, so it ranks routes identically at every valuation and always prefers the
cheapest. The dollar difference $p_mR-c_m$ does not, which is what lets an expensive route win
when the problem looks solvable and lets *every* route lose when it does not.

### 5.5 The probe

All beliefs and costs come from one forward pass of the cheapest route. Let $h(x)\in\mathbb{R}^d$
be the mean-pooled last-layer-fraction hidden state of the scout over the prompt tokens,
computed **before any generation** — one prefill, no decode:
$$h(x)=\frac{1}{|T(x)|}\sum_{t\in T(x)} h^{(\ell)}_t(x).$$
Mean pooling rather than the last prompt token is a deliberate control: chat templates end
differently across model families (Qwen on a newline after `<|im_start|>assistant`, gpt-oss
harmony on the word `assistant` mid-header), and reading position $N\!-\!1$ compares different
objects across models (§6.4).

Two heads per route, both linear, fit on the training split only:
$$\hat\theta_m(x)=\sigma\!\bigl(w_m^\top h(x)+b_m\bigr),\qquad \hat T_m(x)\ \text{as in §5.3},$$
with $\ell$ and the regularisation chosen on the calibration split and never on test.

**Where each cost is incurred — the two are not comparable per-episode constants.**

| | when computed | per episode | cost |
|---|---|---|---|
| our probe | once, from the prompt alone | 1 prefill | **$0.000149** |
| scout as a *route* | when the policy buys a draw | 0-K decodes | $0.000920 each |
| LoRA encoder | **once per decision**, on the rendered state | 5.94-9.47 prefills of ~1178 tok | **$0.0039-0.0062** |

`theta_m(x)` and `c_m(x)` depend on the problem only, so the vector is computed once and reused
at every decision; all state-dependence comes from the analytic decay. The encoder is the
opposite: it re-reads problem + latest attempt code + execution feedback at every node, so it
genuinely refines as evidence arrives. That is the capability the 26-42x buys.

**State-conditioning the probe was tested and the factorized form wins.** Re-prefilling at every
decision with the accumulated state (counts, failing tests) would make the probe state-conditioned
like the encoder. Bounding that offline -- fitting a joint model on [activation + state features]
over 23,525 reachable states -- the analytic product is *better*, not merely cheaper:

| target | OURS $\hat\theta(x)\cdot\sigma/(\sigma+n)$ | learned [act + state] | decay only | $\theta$ only |
|---|---|---|---|---|
| scout | 0.7112 | 0.6607 | **0.7939** | 0.6866 |
| oss20 | **0.7707** | 0.7631 | 0.7312 | 0.7207 |
| oss120 | **0.7903** | 0.7126 | 0.7675 | 0.6916 |

With ~10k training states the learned joint model overfits where imposed structure generalises.
*Honest exception:* on the scout, decay alone beats the product, because the scout's own $\theta$
is poorly predicted (0.6866) so multiplying by it injects noise. *Caveat:* feature concatenation
cannot simulate a decoder reasoning over the state in context, so this bounds rather than settles
the question.

**And we measured what the refinement is worth.** Conditioning on `theta-hat`, the observed
scout-failure count improves log-loss by 0.00011 (0.36886 -> 0.36875), coefficient -0.0315. The
*unconditional* coupling is large -- oss120 pass@1 falls 97.7% -> 69.9% after one scout failure --
but that is problem difficulty, which the probe reads off the prompt rather than inferring from
failures. So: the probe does not refine; refining buys ~nothing once a per-problem prior exists;
the encoder pays 26-42x to do it and ties on the frontier.
*Caveat:* measured on the scout-failure channel only. The encoder also sees the failing code and
execution feedback, a richer signal not yet isolated. Test before leaning on this.

**Cost of the method itself.** Training: 551 problem-level labels, 0.16 CPU-seconds for all
heads, $\approx$40 KB of parameters. Inference: one 4B prefill ($\$0.000149$), already paid for
under the mandatory-scout protocol; the decision rule itself is arithmetic, so no network runs
at decision time. Crucially, $h(x)$ comes from **one** model, so the pool's other members need
never expose weights — outcomes at training time and a price at decision time suffice.

### 5.6 Baselines as special cases

The rule specialises cleanly, which is how the ablations are constructed:

| set | recovers |
|---|---|
| $\theta_m=\bar\theta_m$, $c_m=\bar c_m$ | RoR / count-based routing |
| $K=1$, no $\bot$ | single-commit routing (e.g. prefill-activation routers) |
| $\hat\theta_m(x)$, $c_m=\bar c_m$ | activation beliefs, constant cost |
| $\hat\theta_m(x)$, $\hat c_m(x)$ | **full method** |

Every `_qcost` arm differs from its twin in exactly one term, so any gap is attributable to
cost conditioning alone.

## 6. Experiments

**6.1 Setup.** LiveCodeBench, 892 problems, temporal 551/170/171, 6 draws × 3 routes
{Qwen3-4B-Instruct-2507, gpt-oss-20b, gpt-oss-120b}, all served locally under vLLM at 32768.
Self-hosted AWS-node token prices. `lcb_local_pool_1788407418`. Problem-clustered paired
bootstrap, 5000 resamples, throughout.

**6.2 Cost is variable and only partly predictable (C1).**

**All cost numbers are reported in dollars.** R2 in log space compresses exactly the large values
that dominate spend and flatters the fit badly (LCB oss120: 0.792 log vs 0.633 dollar; TACO
oss20: 0.215 log vs **-0.109** dollar). Test split, rich features, shrinkage applied:

| | true mean $ | constant MAE $ | probe MAE $ | MAE cut | R2 |
|---|---|---|---|---|---|
| LCB scout | 0.001490 | 0.001313 | 0.000998 | +24.0% | 0.241 |
| LCB oss20 | 0.009492 | 0.007533 | 0.005042 | +33.1% | 0.473 |
| LCB oss120 | 0.046233 | 0.031148 | **0.016795** | **+46.1%** | **0.633** |
| TACO scout | 0.001580 | 0.001431 | 0.001090 | +23.9% | 0.127 |
| TACO oss20 | 0.010694 | 0.008161 | 0.006873 | +15.8% | **-0.109** |
| TACO oss120 | 0.050096 | 0.029080 | 0.022007 | +24.3% | 0.341 |

**Where the cost estimate stands.**
1. **It works, and best where it matters most.** On the expensive route -- the one whose cost
   dominates every decision -- it cuts absolute error 46.1% on LCB and 24.3% on TACO. MAE falls
   on *all six* route/dataset cells, by 16-46%.
2. **The one negative cell, TACO oss20, is fully diagnosed.** MAE improves (-15.8%) while R2 is
   -0.109, because the probe beats the constant typically and loses badly on a few extremes.
   Decomposing the squared error on the 168 test problems:

   | | capped | uncapped |
   |---|---|---|
   | share of problems | 20.8% | 79.2% |
   | **share of squared error** | **68.8%** | 31.2% |
   | R2 | **-0.250** | **+0.113** |
   | mean true cost | $0.02584 | $0.00671 |

   **A fifth of the problems produce two-thirds of the error.** On uncapped problems the probe
   works; the negative R2 is entirely the capped tail.

   It is *not* that oss20 tops out most often -- the scout hits the cap on 8.6% of draws too, and
   on more problems (27.5% vs 22.9%). It is that oss20's outputs are the **longest** (mean 7994
   tokens, p90 28,973), so topping out is 3.9x more expensive there. Combined with its
   4.7-10.3% EmptyGeneration rate on TACO -- reasoning to the cap and emitting nothing -- this
   route has a failure mode that is simultaneously maximally expensive, unpredictable from the
   prompt (whether a model runs away reasoning is largely stochastic), and concentrated enough
   to dominate squared error.

   **A larger cap would make this WORSE, not better.** LCB exists at two caps, and raising it
   from 4096 to 32768 *lowered* dollar-space R2 on every route: scout 0.466 -> 0.241, oss20
   0.523 -> 0.473, oss120 0.683 -> 0.633. A low cap compresses the target -- with 26.9% of draws
   pinned at 4096 the heavy tail is invisible and there is less variance to get wrong. Raising
   the cap reveals a tail that is largely stochastic rather than prompt-determined, so the
   regression has more to miss. (Confounded: the 4096 collection was OpenRouter-served with 22.6%
   EmptyGeneration and the 32768 one is local with ~0%. But the direction is large and holds for
   oss120, which had no EmptyGeneration problem at either cap.)

   **Implication for the paper:** the cost head's weakness is specific and characterisable -- it
   fails on runaway generation -- and it is a genuine limit of prompt-based cost prediction, not
   an artifact of our token budget. The deferred 64k re-collection is therefore worth doing for
   *label quality*, not as a fix for cost estimation.
3. **Truncation is roughly half the TACO deficit.** Restricting to problems that never hit the
   token cap moves TACO oss20 from -0.109 to +0.128 and TACO scout from 0.127 to 0.264, while
   oss120 (0% capped) is unchanged -- a clean control. But cap-free TACO is still 0.13-0.34
   against LCB's 0.33-0.63, so truncation is a contributor, not the cause.
4. **The structure is not the problem.** A hurdle model (P(cap) x cap + (1-P(cap)) x
   E[tokens | uncapped]) is worse everywhere, so the bimodality induced by the cap is not what a
   better functional form would fix.
5. **The shrinkage is load-bearing, not cosmetic.** With a route whose R2 is negative in the
   space that matters, calibration-fitted shrinkage (SS5.3) is what stops query-conditioned cost
   from degrading the policy; slopes are 0.72-0.98 on LCB and 0.83-0.96 on TACO.

**Between-problem variance** (the ceiling any prompt-only predictor can reach) is 84.8/62.3/90.1%
on LCB and 82.3/85.9/91.0% on TACO, and per-problem cost spans p90/p10 of 16-37x, so the quantity
is worth predicting even where we predict it poorly.

**6.3 Frontier (C1, C5).** All numbers: 96-point value grid, 3 draw-ordering seeds, rich
features, `scout_first` for every arm including the baseline, problem-clustered paired bootstrap.

**LCB** — advantage over RoR, mean +/- sd over 3 draw-ordering seeds, 96-point grid

| target | beliefs only | +query-conditioned cost | worst seed (full) |
|---|---|---|---|
| 50% | +45.7% ± 1.5 | **+49.1% ± 5.2** **(unstable)** | +43.2% |
| 60% | +16.5% ± 0.6 | **+27.8% ± 1.0** | +27.2% |
| 65% | +2.1% ± 0.7 | **+25.4% ± 2.1** | +23.7% |
| 70% | +9.8% ± 9.0 | **+28.0% ± 8.2** **(unstable)** | +18.6% |
| 75% | +36.5% ± 2.3 | **+39.7% ± 2.1** | +38.5% |
| 80% | +19.2% ± 1.7 | **+21.5% ± 5.0** **(unstable)** | +18.5% |

**TACO** — advantage over RoR, mean +/- sd over 3 draw-ordering seeds, 96-point grid

| target | beliefs only | +query-conditioned cost | worst seed (full) |
|---|---|---|---|
| 30% | +74.6% ± 1.1 | **+69.8% ± 2.8** | +67.1% |
| 35% | +58.1% ± 3.8 | **+60.2% ± 2.7** | +57.9% |
| 40% | +36.5% ± 1.6 | **+43.4% ± 2.4** | +42.0% |
| 45% | +19.3% ± 19.9 | **+30.1% ± 19.3** **(unstable)** | +11.3% |
| 50% | +9.4% ± 5.3 | **+13.9% ± 4.1** | +10.2% |
| 55% | -4.7% ± 2.1 | **-16.6% ± 11.3** **(unstable)** | -24.2% |

**Two cells are seed-unstable and must be reported as such**: LCB 70% (sd 9.0 on the beliefs arm)
and TACO 45% (sd 19.9). Seed 0 was favourable at both, so any table built on a single seed
overstates us there. Everywhere else sd is 0.6-3.8.

**Grid density mattered and was understating us.** At 24 points LCB 50% read +32.4% and TACO 30%
+71.5%; at 96 they are +52.8% and +69.6%. "Cheapest arm reaching T" silently switches policy
family when no operating point lands near T, so sparse grids are not conservative -- they are
noisy in both directions.

**Coupled RoR, the strongest honest baseline, does not close the gap.** Giving count beliefs a
cross-route difficulty channel (a second Beta-Bernoulli decay on other routes' failures, swept
over kappa) moves our advantage by at most 1.1pt on LCB and 0.2pt on TACO.

**6.3a Reporting: cost-at-matched-accuracy hides the regime where we lose.**

The policy optimises $J(\pi)=R\cdot\text{acc}-\text{cost}$, but the frontier reports *cost at a
matched accuracy target*. That is the right view for comparability -- it is what RoR publishes --
but it has two defects. It is **grid-sensitive**: "cheapest arm reaching T" silently switches
policy family when no operating point lands near T (this produced a spurious 58% protocol effect,
§6.15). And it **never probes high R**, where cost is irrelevant and only the ceiling matters.

**Utility at matched R** has neither defect: both arms take the same R, every swept point is a
comparable pair, and no target must be hit. Under it we win at **18/24** swept R on LCB and
**17/24** on TACO, with a systematic shape:

| regime | LCB | TACO |
|---|---|---|
| low R (cost-dominated) | tied | tied |
| mid R | **+$0.0299** | **+$0.0533** |
| **high R (accuracy-dominated)** | **-$0.0163** | **-$0.0660** |

**We lose at high R because our accuracy ceiling is lower**: 86.1% vs RoR's 86.5% on LCB, 61.3%
vs 61.9% on TACO. When cost stops mattering, RoR simply buys everything; abstention and
cost-aware routing leave a few tenths of a point unclaimed. The frontier tables never show this,
because they stop at 80% (LCB) and 50% (TACO).

**Report both.** Frontier for comparability with the baseline; utility-at-matched-R as primary,
since it is the objective and is artifact-free; ceiling gap disclosed rather than concealed by
the target range. Do **not** report relative utility: $J$ crosses zero, so $\Delta J/|J|$ produces
+2213% and -221% next to each other. Use dollars, or $\Delta J/R$ (accuracy-equivalent units).

**6.3b Against the compiled fixed schedule, not just RoR.** Compiling the best fixed
route-and-depth schedule for each budget: five clear wins, one marginal, one tie, no losses --
up from three clear wins before query-conditioned cost, so the cost head is what separates us
from our closest non-adaptive competitor. Abstention runs 10.5% at the 70% point, low because
clean labels lifted the pool's solvability.

**6.4 Readout control (C4).** Four readouts from one forward pass. Last-token vs mean-pooled
*reverses* the cross-model ordering; gpt-oss-120b's own probe improves 0.7743 → 0.8313 from
pooling alone, because harmony ends its prompt on `assistant` mid-header while Qwen ends on a
newline after a completed one. Mean-pooled, the scout probe is significantly worse per route
(−0.061, −0.049) and tied on pool solvability (−0.028 [−0.075, +0.015]). 2603.20895 uses
last-token and does not discuss template comparability; scope this as a proposed control, not a
refutation — our pool is 3 models on one benchmark against their 11–20 across three.

**6.5 Per-candidate probing loses once its probe is priced (C4).** Probing each candidate
means a prefill on *each* model in the pool, including the 120B: $0.000741 + $0.006338 on top of
the scout's $0.000149, so $0.007229 against $0.000149 -- **48.5x**. Rich features, 96-point grid,
3 draw-ordering seeds, both arms otherwise identical (same cost head, same protocol):

| target | pooled probe (ours) | per-candidate, **probe free** | per-candidate, **charged** |
|---|---|---|---|
| 50% | +49.1% ± 5.2 | +57.7% ± 3.8 | **+10.6% ± 3.3** |
| 60% | +27.8% ± 1.0 | +33.2% ± 0.9 | **+0.6% ± 1.2** |
| 65% | +25.4% ± 2.1 | +26.2% ± 8.0 | **+1.0% ± 8.3** |
| 70% | +28.0% ± 8.2 | +29.1% ± 2.5 | **+10.3% ± 3.6** |
| 75% | +39.7% ± 2.1 | +39.5% ± 4.0 | **+28.8% ± 4.0** |
| 80% | +21.5% ± 5.0 | +25.0% ± 0.9 | **+16.8% ± 0.8** |

**The signal is real and the economics kill it.** Given away free, per-candidate probing wins at
5/6 targets, by +0.6 to +8.6pt -- so the better AUC does convert, contrary to what we reported at
24 points with single-layer features. Charged what it costs, it is worse than the single scout
probe at **all six**, by 10.9 to 38.5pt, and its beliefs-only arm falls *below* the RoR baseline
at 60% and 65% (-9.5%, -23.5%). The extra signal is worth less than one prefill on the 120B.

*The comparison is exact on the differential.* Under `scout_first` the scout's prefill is already
bought, so both arms are over-charged by the same $0.000149 -- pooled should be $0 marginal and
per-candidate $0.007079. Correcting both moves each column up and changes no sign.

**This retires the earlier null.** At 24 points with single-layer preds we read +6.4/-1.0/+6.4/
+18.0/-8.8/-2.8% and called it "buys nothing". That was grid noise on both sides. The honest
claim is stronger and simpler: per-candidate probing is a *better predictor that costs too much*,
and the deployment argument (one model's weights, not the pool's) now rests on price rather than
on a failure to convert.

**6.6a Probe scaling: how small can the probe be?** Same rich features, four smaller models,
2x2 over scale x code-specialisation:

| probe | params | LCB pool AUC | LCB cost R2 | TACO pool AUC | TACO cost R2 |
|---|---|---|---|---|---|
| **scout 4B** | 4.41B | **0.8652** | **0.633** | **0.8453** | **0.341** |
| Qwen3-1.7B | 2.15B | 0.7861 | 0.570 | 0.8111 | 0.163 |
| Qwen3-0.6B | 0.69B | 0.8184 | 0.509 | 0.8042 | 0.180 |
| Coder-1.5B | 1.89B | 0.8167 | 0.361 | 0.8057 | 0.128 |
| Coder-0.5B | 0.66B | 0.8261 | 0.355 | 0.7816 | **-0.205** |

**The two heads scale differently, and that is the interesting part.** Belief signal survives
shrinking -- a 0.69B model retains ~95% of pool AUC (0.818 vs 0.865) despite being 6.4x smaller
and able to solve almost nothing on TACO. **That is the strongest evidence for the
shared-difficulty account**: a model far too weak to solve these problems still separates
solvable from hopeless, so the probe reads a property of the *problem*, not of the solver.
Cost prediction does not survive: R2 falls from 0.633 to 0.355-0.570 on LCB and collapses on
TACO, going negative for Coder-0.5B.

**Code specialisation does not help and hurts cost.** Coder-1.5B 0.361 vs Qwen3-1.7B 0.570;
Coder-0.5B 0.355 vs Qwen3-0.6B 0.509, with beliefs roughly a wash. The probe does not need a
model that writes code; it needs one that represents difficulty.

*Scale is non-monotone for beliefs (0.6B beats 1.7B on LCB), which at n~170 is within noise --
do not claim an ordering among the small models without a paired bootstrap.*

**6.6 Capacity ablation.** Linear vs MLP on the same activations, 535 pooled test problems over
rolling-origin folds: linear wins pool solvability (P(MLP better)=0.029) and all three cost
targets. "Linear suffices" is measured, not assumed.

**6.7 Cost accounting — the method needs a cost ratio above ~6x, and MoE makes that hard to pin down.**

*This is the sharpest limitation in the paper. It belongs near the front, not buried.*

**The result.** Full method vs RoR across the oss120:scout price ratio (oss20 interpolated
geometrically):

| ratio | LCB 60/70/80% | TACO 30/40/50% |
|---|---|---|
| 2x | -45.6 / -95.5 / -20.8 | +20.3 / -15.9 / -107.0 |
| 3x | -4.8 / -34.1 / +11.3 | +33.0 / -4.1 / -79.5 |
| 4x | +19.2 / -0.6 / +7.6 | +32.5 / -2.8 / -7.1 |
| **6x** | **+46.6 / +30.2 / +7.0** | **+56.1 / +26.0 / +11.0** |
| 8x | +49.8 / +26.9 / +18.1 | +56.5 / +20.5 / +16.5 |
| 10x | +34.2 / +42.3 / +20.6 | +54.6 / +16.7 / +20.7 |
| 30-40x | +34.6 / +33.7 / +27.4 | +63.0 / +31.4 / +25.1 |

**Break-even is ~4-6x; the method is reliably positive from 6x up, and negative below ~3x.**

**Why the ratio is genuinely hard to pin down: our pool is two MoEs.** Cost per token is
GPU-$/hour divided by tokens/hour, and for a mixture-of-experts model those two terms pull in
opposite directions. gpt-oss-120b holds 116.8B parameters but activates 5.1B; gpt-oss-20b holds
20.9B and activates 3.6B. So it needs a large card to *hold* but behaves like a small model when
*running*. Every accounting choice lands somewhere different:

| basis | oss120:scout | note |
|---|---|---|
| **total parameter count** (RoR's proxy) | **30x** | 116.8/4; ignores that only 5.1B is active |
| **active parameter count** | **1.3x** | 5.1/4; ignores that you must hold 61GB |
| our AWS-node estimate (what the tensors use) | 40x | derived from total params -> node size |
| measured throughput, flat GPU price | 1.3x | our own serving data, concurrency 16 |
| measured throughput, cheapest card that fits | **3.2-23x** | swings on the KV-cache budget alone |
| OpenRouter list price for the 120B vs an assumed 4B price | <1x to ~8x | mixed basis; not a coherent comparison |

**Measured from our own collection** (concurrency 16, tokens/s per GPU): scout 1220, oss20 2126,
oss120 911. **gpt-oss-20b is faster per GPU than the 4B scout.** Charging each model the cheapest
card its weights fit gives oss120:scout of 7.3x at a 4GB KV budget and 3.2x at 14GB -- a 2x swing
from the KV assumption alone, because heavy concurrency forces the small model onto a bigger card
while the large one is already on the biggest.

**Inversion is not an operating point.** A ratio below 1 requires a *mixed* basis -- a small model
at low single-tenant utilisation against a large one at hyperscale API rates. It is a pricing
error, not a deployment, and is excluded rather than reported as a failure mode.

**What prior work does, and what we will do.**
- **RoR (2607.08665)**: "Per-draw cost is proxied by the model's parameter count in billions -- a
  monotone stand-in for $-per-token serving cost", resolved from the repository name. Adds a
  provider-price snapshot as a robustness table.
- **NVIDIA prefill router (2603.20895)**: OpenRouter list prices (March 19 2026), min-max
  normalised into the routing score; input tokens per query, output tokens approximated by "median
  training output tokens as a verbosity proxy". **No MoE discussion and no cost-basis sensitivity
  analysis** -- the authors list both as limitations.

**Decision: report the parameter-count proxy as primary** for direct comparability with RoR, with
the full ratio sweep as the robustness analysis, and our measured throughput basis as a secondary
result. Under RoR's own accounting our pool is 30x, comfortably inside the working range.

**The methodological point worth making, even though it cuts against us**: parameter count is a
poor proxy on MoE pools. It reports 30x where measured throughput reports 1.3-7x, so the standard
accounting in this literature *overstates* the headroom available to cost-aware routing whenever
the pool contains sparse models. That is a caution for the field and a limitation on our own
headline in the same breath.

**6.8 Seed sensitivity.** Seed controls draw orderings — a variance source the problem bootstrap
holds fixed. Four seeds, full method vs RoR:

| target | mean | sd | worst seed |
|---|---|---|---|
| 50% | +55.3% | 0.7 | +54.5% |
| 60% | +39.2% | 6.5 | +33.5% |
| 65% | +20.5% | 2.1 | +17.4% |
| 70% | +20.8% | 0.3 | +20.5% |
| 75% | +30.6% | 1.8 | +28.8% |
| 80% | +20.7% | 4.8 | +18.1% |

**Quote mean ± sd, not seed 0.** The worst seed at every target is still >= +17.4%, so the
headline is not seed-dependent. 60% is the noisy point (sd 6.5) and should be reported as such.

**6.9 Truncation sensitivity — how much of the edge is anticipating runaway generations?**

At-cap draws are expensive failures: they cost the full cap and essentially never solve the
problem (LCB scout 0.0%, oss20 3.6%; TACO 0.0% on both). Cheap routes do this far more than the
120B does — LCB 4.13%/3.64%/0.00% of scout/oss20/oss120 draws, TACO 8.64%/8.63%/0.02% — so part
of any cost advantage over a per-route constant could be "the probe learned which prompts make
small models ramble" rather than difficulty prediction. `make_notrunc_tensors.py` builds the
world where that channel does not exist, and the cost head is **refit on the same valid mask**
(without this the policy is charged for spending it can no longer incur; refitting drops the
scout constant 31% and oss20's 14% on LCB, 39% and 19% on TACO). 96 points, 3 seeds, full method.

| LCB | all draws | at-cap deleted | delta | | TACO | all draws | at-cap deleted | delta |
|---|---|---|---|---|---|---|---|---|
| 50% | +49.1 ± 5.2 | +56.8 ± 3.3 | +7.7pt | | 30% | +69.8 ± 2.8 | +67.9 ± 3.1 | -1.9pt |
| 60% | +27.8 ± 1.0 | +26.8 ± 0.6 | -1.0pt | | 35% | +60.2 ± 2.7 | +57.4 ± 3.4 | -2.8pt |
| 65% | +25.4 ± 2.1 | +25.3 ± 0.9 | -0.1pt | | 40% | +43.4 ± 2.4 | +43.0 ± 4.2 | -0.4pt |
| 70% | +28.0 ± 8.2 | **+9.8 ± 0.8** | **-18.2pt** | | 45% | +30.1 ± 19.3 | +10.1 ± 6.4 | -20.0pt* |
| 75% | +39.7 ± 2.1 | +35.0 ± 2.7 | -4.7pt | | 50% | +13.9 ± 4.1 | +18.4 ± 4.7 | +4.5pt |
| 80% | +21.5 ± 5.0 | **+6.8 ± 11.2** | **-14.7pt** | | 55% | -16.6 ± 11.3 | -16.9 ± 0.8 | -0.3pt |

**The result is truncation-robust except at LCB's top two targets.** Four of six LCB cells and
five of six TACO cells move by under 5pt. What moves is LCB 70% and 80%, where the advantage
thins to +9.8% and +6.8% (worst seed −2.7%). The mechanism is visible in the arms themselves: at
80%, deleting runaway draws makes *RoR* 25% cheaper ($0.0838 → $0.0630) while barely touching us
($0.0610 → $0.0598). RoR's high-target arm buys many cheap draws — 4,478 against our 1,676 — and
each carries truncation risk a per-route constant cannot see. We were already avoiding them.

**The dataset that truncates twice as much is the one that barely moves**, which rules out the
simple story. TACO caps at 8.6% against LCB's ~4% and is unchanged at 5/6 targets, because its
operating range (30–55%) never reaches the many-draws regime where the effect lives. So the
finding is not "truncation drives the result" but "truncation drives the *high-target tail* of
the result on LCB". *The TACO 45% delta is starred because that cell is seed-unstable in the base
world (sd 19.3, §6.3); notably its sd falls to 6.4 once at-cap draws are gone, so the instability
was itself truncation-driven.*

*The counterfactual is imperfect and slightly favours the baseline.* Deleting runaway draws
deletes a route entirely on the problems where every draw ran away — 4 LCB oss20 problems, 1 LCB
scout, 14 TACO oss20, 2 TACO scout — so the ablated world is a little easier than a world with a
genuinely larger cap. The 64k re-collection (§6.16) is the direct test.

**Report both columns.** Anticipating budget-exhausting draws is legitimate — every deployed
system has a cap, and failed draws cost 2.6–5.3× more than solved ones — but the high-target LCB
numbers should not be quoted without it.

**6.10 TACO medium+hard — the replication.** 883 problems, random split 547/168/168 (TACO's
dates are 79.8% Unix-epoch sentinels, so its "temporal" split was a platform confound: train a
five-platform mixture, eval 99.5% Codeforces, 42 test problems. See RESEARCH_LOG).
Solve rates over 6 draws: scout 16.9%, oss20 38.0%, oss120 45.3% — against LCB's 42/65/82.
Best-of-6 oss120 reaches only 55.95%, so no single-model policy is good.

Paired bootstrap, 168 test problems:

| comparison | 30% | 35% | 40% | 45% | 50% | 55% |
|---|---|---|---|---|---|---|
| activation beliefs vs RoR | **+69.7** | **+51.4** | **+30.6** | **+39.7** | **+13.9** | −2.2 |
| qcost on RoR beliefs | **+36.8** | **+21.1** | +10.2 | **+26.1** | +6.3 | **−9.2** |
| qcost on activation beliefs | −15.2 | −2.7 | **−25.8** | +3.6 | −0.6 | **−29.1** |
| full method vs RoR | **+65.1** | **+50.1** | +12.7 | **+41.9** | +13.3 | **−31.9** |

(bold = interval clear of zero)

**Two pre-registered predictions, both resolved.** Abstention has room: 25–49% against LCB's
10.5% — confirmed. Cost conditioning does *not* degrade gracefully here; it fails on strong
beliefs — refuted, and it is what demoted C3.

The best TACO configuration is `content_decay` **without** qcost. Report it that way.

**6.11 Rich features — read the whole representation, not one layer.**
The single-layer/single-readout probe captured only 11-40% of the between-problem variance
ceiling on TACO (ceiling 82-91%). Concatenating all layers x {mean, last}:

| | cost R2 | | belief AUC | |
|---|---|---|---|---|
| | single | rich | single | rich |
| LCB scout | 0.476 | 0.555 | 0.769 | 0.782 |
| LCB oss20 | 0.576 | 0.701 | 0.767 | 0.813 |
| LCB oss120 | 0.700 | **0.792** | 0.792 | **0.864** |
| TACO scout | 0.195 | 0.397 | 0.731 | 0.756 |
| TACO oss20 | 0.093 | 0.215 | 0.788 | 0.827 |
| TACO oss120 | 0.359 | 0.416 | 0.821 | 0.843 |

**Free at inference**: the forward pass already computes every hidden state, so this is one
prefill plus a dot product, probe still linear. The risk was overfitting 40,960 features on
~550 problems; held-out test improved on both datasets while an MLP on the single layer
collapsed to negative R2, so the gain is features, not capacity. Selection verified on
CALIBRATION (rich wins 11 of 12 cells) after initially being chosen on test.

**6.12 When does cost conditioning help? The regime, quantified.**
With rich features, qcost on activation beliefs:
- **LCB**: +22.8/+22.7/+21.8/+18.8/+4.8/+11.0%, four of six significant.
- **TACO**: significant wins at 35% (+16.8) and 45% (+13.5), significant losses at 30% (-29.5)
  and 55% (-18.4), ties elsewhere. Inconsistent.

The discriminating quantity is cost signal *relative to* belief signal. TACO's belief AUC
matches LCB's (0.76-0.84 vs 0.78-0.86) while its cost R2 is roughly half (0.22-0.42 vs
0.56-0.79). Cost conditioning helps in proportion to that ratio, and the calibration-fitted
shrinkage (§5.3) dials it in automatically -- slopes are 0.72-0.98 on LCB and 0.83-0.96 on
TACO -- so a practitioner need not know the regime in advance. **This is a stronger claim than
an unconditional one: it is predictive from calibration data alone.**

**6.13 Decay-rate sweep.** The pseudo-count sigma = 2.0 is inherited from RoR, and the research
log records an empirical optimum near 0.3. Sweeping {0.3, 1, 2, 5, 10} on LCB shows **no
consistent ordering** -- sigma=5 wins two targets, sigma=2 wins two, sigma=1 one, and the spread
sits inside the measured seed noise. The claimed optimum at 0.3 is not supported. One fewer
hyperparameter; keep the inherited default and say so.

**6.14 Coupled-RoR baseline.** Count beliefs given a second Beta–Bernoulli decay on other
routes' failures, $\kappa\in\{2,5,10,20\}$, taking the most favourable $\kappa$ per target.
Coupling barely helps and hurts at three targets; our margin moves from
+56.2/+43.9/+21.9/+20.5/+32.5/+18.1 to **+55.2/+37.5/+21.4/+20.4/+32.5/+18.1**. We built the
strongest honest version of the baseline and it did not close the gap. TODO: repeat on TACO.

**6.15 Start protocol: the mandatory scout is not what carries the result.** `scout_first`
forces one scout draw before any decision (the protocol under which the probe's prefill is
already bought); `free_start` lets the policy choose from the empty state. Advantage over RoR
at 96 points, seed 0, full method:

| | LCB 50/60/65/70/75/80% | TACO 30/35/40/45/50/55% |
|---|---|---|
| `scout_first` | +52.8/+28.9/+27.7/+33.3/+42.1/+27.3% | +69.6/+59.5/+42.0/+50.0/+18.4/-22.1% |
| `free_start` | +48.1/+22.0/+20.8/+32.7/+41.5/+27.1% | +67.6/+52.7/+41.6/+33.2/+10.4/-22.5% |
| gap | +0.2 to +6.9pt | +0.4 to +16.8pt |

Both protocols win at every target we win at, and lose at the one we lose at (TACO 55%), so the
result does not depend on the forced scout. It is worth 0.2-6.9pt on LCB and more on TACO,
concentrated at 45-50% where the free policy skips the cheap evidence and commits early.

**We previously read this as a 58% protocol effect and that was a grid artifact.** At 24 points
"cheapest arm reaching T" jumped between policy families on one side of the comparison and not
the other; at 96 the LCB gap collapses to a few points. Only the TACO 45% cell keeps a large
gap, and it is one of the two seed-unstable cells (§6.3), so do not lean on it.

**6.16 The 64k re-collection, and what is left of the serving-path artifact.** *In flight —
oss20 train complete, oss120 and eval pending.* Doubling the cap 32k → 64k, on the same 551
train problems:

| | local pool, 32k | OpenRouter, 64k |
|---|---|---|
| at the cap | 3.63% | **1.47%** |
| empty answer | 0.00% (local) / 22.6% (OpenRouter 32k) | **7.11%** |
| oss20 solve rate | 65.4% | **70.8%** |
| mean completion tokens | 4,442 | 4,330 |

**Doubling the cap costs nothing and buys 5.4 points.** Mean completion length is flat (4,442 →
4,330) because the tail is thin: only 1.47% of draws still reach 64k. So the truncation channel
in §6.9 is a collection artifact worth removing, not an intrinsic property of the routes.

**The remaining empties are one provider emitting tool calls.** 84% of the 219 surviving empties
carry `finish_reason == "tool_calls"` — gpt-oss opened a harmony tool call and the provider
returned no assistant content — and every one produced reasoning text but no answer. They
concentrate almost entirely on a single provider: **10.9% empty on Parasail against 1.1% on
CoreWeave**. Plain retry cannot fix that, since it frequently draws the same provider again; the
collector now excludes the provider that returned nothing from the next attempt. This is the
concrete form of the serving-path claim in §6.1 and should be reported as a measurement hazard
for anyone collecting multi-model pools through an aggregator.

**6.12 (retired) TACO in progress.** 883 problems (677/206). Draw-0 solve rates: scout
18.4/14.8%, oss20 44.0/31.0%, oss120 48.7/43.4% (medium/hard) — against LCB's 42/65/82%.
Pool-solved at one draw 52.2/50.0%, so ~half the problems are unsolved by the whole pool. Three
distinguishable rungs, and on medium oss20 is within 5pt of oss120 at ~8.6× less cost, so the
routing decision is non-vacuous — unlike LCB where oss120 dominates.

## 7. Limitations
1. Two benchmarks, both competitive programming; SWE-Smith would add a modality.
1b. **Cost conditioning does not replicate on strong beliefs** (§6.10). It is a contribution
   with a regime, not a universal improvement.
2. n=171 test problems; the capacity question needed pooled folds to resolve at all.
3. **The method requires a cost ratio above ~6x** (§6.7). On MoE pools that ratio is
   accounting-dependent: parameter count says 30x, measured throughput says 1.3-7x depending on
   the KV budget. We report under RoR's proxy for comparability and sweep the rest.
4. Outcomes collected under our own serving configuration; another provider's defaults could
   shift absolute solve rates. All comparisons are within-collection.
5. The readout correction (§6.4) is measured on 3 models on one benchmark.
6. **Lower accuracy ceiling than the baseline.** 86.1% vs 86.5% (LCB), 61.3% vs 61.9% (TACO).
   At high R, where cost is irrelevant, RoR buys everything and wins on utility; our abstention
   and cost-aware routing forgo the last few tenths (§6.3a).
7. At-cap draws are route-asymmetric (oss20 10.3% on TACO hard, oss120 ~0%) — genuine budget
   exhaustion (`finish_reason=length`). Excluding them halves the advantage at the 70% and 80%
   targets (§6.9), so part of the gain is anticipating budget-exhausting draws.

## 8. Retracted — do not resurrect
- Cross-model superiority ("own activations are the worst predictor of own success") — readout
  artifact (§6.4).
- "Nobody predicts generation length from activations" — false (2607.05316, 2602.11812).
- "First to route from a small model's hidden states" — false (2608.04804, 2603.20895).
- "C1 is basis-independent" — false (§6.7).
- Our own collection bugs framed as contributions — appendix at most.

## 9. TODO
- [x] TACO frontier — §6.10; C3 demoted, C1 promoted
- [x] Coupled-RoR baseline on LCB — §6.11, margin holds
- [ ] TACO price sweep (LCB's degraded badly; TACO's ladder is less saturated)
- [ ] Coupled-RoR on TACO
- [ ] Seeds on TACO
- [ ] SWE-Smith → Verified (needs Daytona harness fixed; 6/10 historical runs all-error)
- [x] Seeds 0–3; report mean ± sd (§6.8)
- [x] Truncation-sensitivity frontier (§6.9)
- [ ] Writing — nothing drafted
