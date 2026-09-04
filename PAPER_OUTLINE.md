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

**C4. One cheap probe matches probing every model — on the frontier.** Per-candidate probes are
significantly better predictors (+0.05–0.06 AUC) and buy **nothing** on cost-at-accuracy (§6.5).
Deployment consequence: weights for one cheap model, not all of them.

**C5. Methodological: cross-model activation comparisons need a fixed readout.** §6.4.

## 4. Related work
**Sequential/budgeted selection.** RoR (2607.08665) — primary baseline. Routing-gap companion
(2607.03436). SeqRoute (2605.25424). Cluster-Route-Escalate (2606.27457). UCCI (2605.18796).
Survey (2603.04445).
**Activation-based prediction/routing.** NVIDIA prefill router (2603.20895) — closest.
LLMs Encode Their Failures (2602.09924). Scrouting (2608.04804) — hidden-state AUC 0.600, null
routing ablation. Knowing When to Quit (2604.18419).
**Output-length prediction (C1's foundation).** How Much is Left? (2607.05316). Entropy-Guided
Representations (2602.11812). Latency-aware routing (2607.18253). Learned per-query cost
predictors (2604.03527).
**Benchmarks/gap analyses.** LLMRouterBench (2601.07206); Routing Plateau (2606.07587);
CodeRouterBench/ACRouter (2606.22902); 67-model Rasch (2606.27288); universal latent space
(2601.06220); routing collapse (2602.03478).
**Agentic stopping.** EET (2601.05777); AgentStop (2604.15075); FailFast-RestartSmart
(2608.03222); SWE-Router (2607.00053); AutoMix; COPE (2506.11578).

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

**6.2 Cost is variable and predictable (C1).** Between-problem variance share 84.8/62.3/90.1%;
per-problem p90/p10 37×/16×/17×; failed draws cost 5.3×/2.6×/3.0× more than solved. Probe vs
constant: R² 0.65/0.80/0.83, MAE cut 45–65%. Beats rescaling the scout's *observed* length
(0.796 vs 0.588; 0.834 vs 0.629) and subsumes it (+0.002/+0.008 combined) while needing only a
prefill. Not collinear with predicted success (partial ρ 0.46–0.75).

**6.3 Frontier (C1, C5).** `lcb_replay_qcost_1788470886`.
Cost conditioning in isolation — RoR beliefs: +45.9/+18.8/+35.1/+5.8/+13.3/+26.7/+12.9%
(6/7 significant). Activation beliefs: +19.7/+15.4/+26.0/+15.9/+16.7/−5.8/+0.0% (5/7).
**Full method vs RoR (rich features): +47.8/+34.6/+23.8/+33.7/+42.4/+27.4%, all six
significant.** On TACO: +63.0/+63.0/+31.4/+54.3/+25.1%, five significant, -21.5% at 55%.
vs compiled fixed schedule: five clear wins, one marginal, one tie, no losses (up from three
clear wins pre-qcost — cost conditioning is what separated us from our closest competitor).
Abstention 10.5% at the 70% point, low because clean labels lifted the pool.

**6.4 Readout control (C4).** Four readouts from one forward pass. Last-token vs mean-pooled
*reverses* the cross-model ordering; gpt-oss-120b's own probe improves 0.7743 → 0.8313 from
pooling alone, because harmony ends its prompt on `assistant` mid-header while Qwen ends on a
newline after a completed one. Mean-pooled, the scout probe is significantly worse per route
(−0.061, −0.049) and tied on pool solvability (−0.028 [−0.075, +0.015]). 2603.20895 uses
last-token and does not discuss template comparability; scope this as a proposed control, not a
refutation — our pool is 3 models on one benchmark against their 11–20 across three.

**6.5 Per-candidate probing buys no frontier advantage (C3).** +6.4/−1.0/+6.4/+18.0/−8.8/−2.8%
across six targets; signs alternate, mean ≈ +3%, all inside typical intervals at this n. The
better predictor converts into nothing — the convex-gate argument playing out. TODO: CI on the
+18% at 70%.

**6.6 Capacity ablation.** Linear vs MLP on the same activations, 535 pooled test problems over
rolling-origin folds: linear wins pool solvability (P(MLP better)=0.029) and all three cost
targets. "Linear suffices" is measured, not assumed.

**6.7 Price-ratio sensitivity — the method needs a real cost ladder.**
Full method vs RoR as the oss120:scout ratio varies: 40× → +43.9/+20.5/+18.1%; 10× →
+34.2/+42.3/+20.6%; 4× → +19.2/−0.6/+7.6%; **0.6× (inverted) → −40.0/−49.3/−13.8%.**
Cost conditioning alone degrades harder (+35.1% at 40× to −160% at 0.6×). **The prediction that
C1 would be basis-independent was wrong**: what matters is cost *signal* vs cost *prediction
error*, and when routes cost alike a noisy c injects variance into a near-constant term.
State as a limitation. Two honest mitigations: the inverted regime is degenerate for every
method (optimal policy is "always call the big model"; absolute costs are pennies), and 4–40×
brackets plausible self-hosted deployments while 0.6× is a hyperscale-API artifact.
*Pre-registered:* TACO should degrade far more gracefully, since oss120 solves 43–49% there
against 81.9% on LCB, so no single-model policy is good.

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

**6.9 Truncation sensitivity.** Excluding the 416 at-cap draws (2.59% of cells,
`tensors_v3_notrunc`, no problem left with zero valid draws):

| target | all draws | at-cap excluded | delta |
|---|---|---|---|
| 50% | +56.2% | +54.2% | -1.9pt |
| 60% | +43.9% | +45.9% | +2.0pt |
| 65% | +21.9% | +20.8% | -1.1pt |
| 70% | +20.5% | **+11.4%** | **-9.1pt** |
| 75% | +32.5% | +32.7% | +0.3pt |
| 80% | +18.1% | **+10.1%** | **-8.0pt** |

The advantage survives everywhere but roughly **halves at 70% and 80%**. This is mechanism, not
noise: at-cap draws are expensive failures, the cost model predicts they will be long and steers
away, and a per-route constant cannot. A meaningful part of the edge at high accuracy targets is
therefore *anticipating budget-exhausting draws* -- legitimate, since every deployed system has a
cap and failed draws cost 2.6-5.3x more than solved ones, but it must be stated rather than left
for a reviewer to find. Report both columns.

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

**6.13 Coupled-RoR baseline (LCB).** Count beliefs given a second Beta–Bernoulli decay on other
routes' failures, $\kappa\in\{2,5,10,20\}$, taking the most favourable $\kappa$ per target.
Coupling barely helps and hurts at three targets; our margin moves from
+56.2/+43.9/+21.9/+20.5/+32.5/+18.1 to **+55.2/+37.5/+21.4/+20.4/+32.5/+18.1**. We built the
strongest honest version of the baseline and it did not close the gap. TODO: repeat on TACO.

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
3. Method requires a real cost ladder (§6.7); harmful under price inversion.
4. Outcomes collected under our own serving configuration; another provider's defaults could
   shift absolute solve rates. All comparisons are within-collection.
5. The readout correction (§6.4) is measured on 3 models on one benchmark.
6. At-cap draws are route-asymmetric (oss20 10.3% on TACO hard, oss120 ~0%) — genuine budget
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
- [ ] CI on the +18% per-candidate point at 70%
- [ ] Writing — nothing drafted
