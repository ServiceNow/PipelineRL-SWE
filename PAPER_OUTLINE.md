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

**C3b. The discipline that makes conditioning safe: shrink every conditioned quantity toward
the unconditional baseline it replaces, fitted on held-out data.** This is the difference between
a method that is *sometimes* better and one that is *consistently* better, and we can measure it
because we shipped one head with the property and one without.

Each head added **alone** to RoR's count beliefs, hull frontier, worst and best case over targets:

| head | LCB worst | LCB best | TACO worst | TACO best |
|---|---|---|---|---|
| cost $\hat c_m(x)$ — **has** calibration shrinkage | **+9.5%** | +30.4% | -31.6% | +5.6% |
| belief $\hat\theta_m(x)$ — raw logistic, **no** shrinkage | -9.1% | +41.7% | **-118.3%** | +43.7% |

The head carrying the guarantee never hurts at all on LiveCodeBench and has 3.7x less downside on
TACO. **Every catastrophic number in this paper came from the head that lacked it.** The
mechanism is precise: a 40,960-feature logistic head fit on ~550 problems emits next-draw
probabilities of $10^{-5}$–$10^{-6}$ for routes that solve the problem, the utility rule reads
$p\cdot R - c < 0$, and the policy abstains on winnable problems (§6.10a).

Platt scaling on the held-out calibration split fixes it with the same guarantee the cost head
has: **slope $\to 0$ collapses the prediction to the per-route constant, which is exactly what
RoR uses, so the floor of the conditioned method is the baseline's performance.** Fitted slopes
are 0.32–0.49 on *both* datasets, so the raw head was over-confident everywhere and LiveCodeBench
merely had enough headroom to absorb it.

*Generalisable claim, and the one a reader should take away even if they never use activations:*
any router that replaces a per-model constant with a per-query prediction must fit the shrinkage
on held-out data, or it inherits an unbounded downside for a bounded upside.

**C4. One cheap probe beats probing every model, once the probe is priced.** Per-candidate
probes are better predictors (+0.05-0.06 AUC) and buy a real but small frontier gain
(+0.6 to +8.6pt) -- for 48.5x the probe cost, which takes 10-38pt back. Charged honestly they
lose to the single scout probe at **every** target (§6.5). Deployment consequence: weights for
one cheap model, not all of them, and the argument is now economic rather than a null result.

**C5. Methodological: cross-model activation comparisons need a fixed readout.** §6.4.

## 3b. The contribution this reframes into: cross-model selective prediction

Everything the method does is downstream of one quantity -- **problem difficulty, a latent shared
across the pool**. The belief head reads it (SS6.9h shows the cost head is the same head in
disguise), the value flows through the give-up decision rather than route choice (SS6.9i), and the
advantage scales with how many problems nothing solves (SS6.9k). That is **selective prediction**,
not routing, and it should be positioned as such.

**What makes it new is the cross-model direction.** Selective prediction and learning-to-defer
estimate a confidence for *the model that will answer*, from that model's own signals -- logits,
entropy, an auxiliary head. Every baseline in that literature is same-model. We estimate the
**shared latent once, from one cheap model's prefill**, and transfer it to models whose weights we
never touch.

**The exploitable consequence: adding a model to the pool costs tens of labels, not hundreds.**
If difficulty is shared, a new route needs only its own 2-parameter response curve on the latent,
not a fresh high-dimensional probe. Holding out gpt-oss-120b entirely, fitting the latent from the
*other* routes' labels, then fitting only `P(new model solves | latent)` on N labelled problems:

| N labels for the new model | shared latent (2 params) | its own probe (40,960 features) |
|---|---|---|
| **10** | **0.7634** | 0.6507 |
| **25** | **0.7898** | 0.6834 |
| 50 | 0.7898 | 0.7263 |
| 200 | 0.7898 | 0.7512 |
| 550 (all) | 0.7898 | 0.7755 |

*(LCB, test AUC for gpt-oss-120b, 30 resamples per N. TACO is sharper still: 0.8126 from 10 labels,
against 0.7916 for a full probe at 200.)*

**25 labels through the shared latent beat 550 labels of a dedicated probe**, and the latent
version saturates almost immediately because there are only two parameters to fit. This is the
practical form of the claim: pools change monthly, and re-fitting a router per new model is the
real deployment cost. Here it is a few dozen labelled problems.

**It also fixes our baseline story.** The right comparison is not only RoR but **same-model
selective prediction**, and we already have it: the per-candidate probes of SS6.5 *are* each model
predicting its own success from its own activations, which is the standard setup. The 4B scout's
probe beats gpt-oss-20b's own probe at predicting gpt-oss-20b (SS6.9h), and per-candidate probing
loses once priced (SS6.5). Re-frame those two results as the selective-prediction baseline
comparison rather than as ablations.

### 3b-ii The latent transfers to models from other labs, at ~25 labels each

Fitting the difficulty latent from **only** scout / gpt-oss-20b / gpt-oss-120b, then applying it
to five models it has never seen, each from a different lab, with N labelled problems and a
two-parameter response curve:

| model | solve rate | latent, N=10 | latent, N=25 | its own probe, N=25 |
|---|---|---|---|---|
| DeepSeek | 80% | 0.788 | **0.818** | 0.645 |
| GLM-5 | 44% | 0.824 | **0.833** | 0.815 |
| Kimi | 80% | 0.757 | **0.771** | 0.626 |
| MiniMax | 50% | 0.670 | **0.741** | 0.711 |
| Qwen-Max | 66% | 0.779 | **0.818** | 0.596 |

**25 labels through the shared latent beat a dedicated probe on the same 25 at four of five
models**, by up to 0.22 AUC. The latent is not an artifact of the gpt-oss family: it transfers
across architectures and vendors.

**Two limitations, one of them serious.** (a) These screens are 50 problems each with **zero
overlap with the main test split**, so the AUCs come from an internal split at n=50 and are noisy.
(b) **Three of the five peers had corrupted labels.** The pool screen recorded 42% empty outputs
for z-ai/glm-5, 18% for minimax and 10% for kimi -- answers written to the `reasoning` channel with
`content` left blank, the same artifact that cost gpt-oss 17 points of solve rate. Their rows above
therefore treat provider artifacts as wrong answers. **deepseek and qwen3-max had 0% empty and are
clean**, and both score 0.818, which is reassuring but is two models, not five.

The collector now recovers an answer from the reasoning channel when it carries a fenced code
block, and all five peers are being re-collected on the real 171-problem test split. **Do not use
the table above for anything but motivation until that lands.**

### 3b-iii On the MDP framing: setting, not contribution

State it as inherited. The evidence that the sequential machinery is *not* where the value lives:

- Bellman lookahead at $h=4$ and $h=6$ (full depth) changes accuracy and abstention by **exactly
  0.000** at every operating point on both datasets (§6.9c). It is worth keeping only as a cost
  saving.
- The decay is a two-parameter Beta posterior whose constant was inherited unfitted from RoR by
  everyone, us included, until §6.9d.
- The value flows through the stop/go decision, not the resample-vs-reroute structure the MDP
  exists to express (§6.9i).

What the setting *does* earn: abstention is only meaningful sequentially (you give up after
evidence, under a budget), and RoR's protocol is required for comparability. **Present the MDP as
the evaluation setting we inherit, and put the contribution on the cross-model latent.** Claiming
MDP machinery as a contribution while full-depth lookahead is a no-op is the kind of claim a
reviewer will test and we will lose.

### 3b-iv Literature check: the shared latent is Item Response Theory, and we must say so

**The framing is not new; the instantiation is.** Decomposing correctness into a shared *item
difficulty* plus a per-model *ability* is Item Response Theory, and there is an active LLM
literature applying it: **IrtNet** (arXiv 2510.00844) already does IRT-based **routing**;
contextual multidimensional IRT (2608.22295) predicts on unseen questions; adaptive testing
(2511.04689) and Ai2's fluid benchmarking use it to cut evaluation cost. **Cite these and drop any
claim that a shared difficulty latent is novel.** It is not.

**What survives contact with IrtNet, the closest work.** It predicts difficulty from 768-d
*sentence-transformer embeddings* of the query; it does **no** abstention, **no** cost-aware
selection, and gives **no** sample-complexity analysis for adding a model. Our differentiators are
therefore: (i) difficulty read from a **solver model's prefill activations**, (ii) a
**cost-constrained sequential setting with a give-up action**, and (iii) the **pool-extension label
efficiency** (~25 labels per new model).

**The baseline this implies, which we had never run.** If a 768-d sentence embedding suffices, then
"activations" is not the contribution and "any query representation" is. Tested against the
cheapest possible representations:

| representation | LCB pool AUC | TACO pool AUC |
|---|---|---|
| **scout activations (40,960-d)** | **0.8629** | **0.8814** |
| TF-IDF + SVD (256-d, no model at all) | 0.7642 | 0.7204 |
| **problem statement length (1 number)** | 0.7110 | 0.6248 |

Activations win by **+0.10 to +0.16 AUC**, so the claim holds. **But statement length alone reaches
0.711 on LiveCodeBench**, so a substantial share of "difficulty prediction" is "long problems are
hard", and the paper must report that rather than let a reader assume the activation is doing all
the work. *Still missing: a true sentence-transformer baseline (none is cached here); TF-IDF is a
weaker proxy and IrtNet's representation may sit between these rows.*

### 3b-v Single-shot routing, in the literature's own units: it works, and it is the weak half

The routing literature evaluates one-shot model selection. Reported that way, on the test split:

| policy | accuracy | cost | accuracy per $ |
|---|---|---|---|
| LCB, always gpt-oss-120b | 68.9% | $0.0393 | 17.5 |
| **LCB, routed by the probe** | 67.7% | $0.0343 | **19.8** |
| LCB, oracle (cheapest that solves) | 72.5% | $0.0290 | 24.9 |
| TACO, always gpt-oss-120b | 47.6% | $0.0429 | 11.1 |
| **TACO, routed by the probe** | 48.2% | $0.0390 | **12.4** |
| TACO, oracle | 54.8% | $0.0320 | 17.1 |

**Routing works and is worth about +12-13% cost-efficiency** over always calling the big model --
but that is **22-31% of the oracle's available gain**, against the abstention channel's +43.7%
(§6.9f). Routing is the weak half by a factor of three or more, now measured in the units the
routing papers use rather than only through our decomposition.

**Two diagnostics that explain why.**
- *Density routing degenerates in one shot.* Selecting by $\hat\theta/c$ scores 33.5% on LCB: it
  picks the scout almost always. Cost-sensitivity needs a resample-or-give-up option to recover
  from a bad cheap draw; without one it is a trap. This is the single-shot analogue of §6.3a.
- *The pool has a dominant model.* On solvable problems the probe routes to a winner 93.4% of the
  time against always-120b's 95.0% (LCB). It is not misrouting; there is simply rarely a cheaper
  route that also works, because gpt-oss-120b dominates gpt-oss-20b almost everywhere.

**Consequence for the claim.** Routing gains are bounded by pool complementarity, which is a
property of the pool and not of the router. **Report the oracle row** so the reader sees the
ceiling, and be explicit that this pool has little routing headroom — which is also the argument
for evaluating on the five API peers, whose strengths are far less nested.

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

**6.3a Reporting: the frontier artifact is fixable, so fix it rather than report around it.**

The policy optimises $J(\pi)=R\cdot\text{acc}-\text{cost}$, but the frontier reports *cost at a
matched accuracy target*. That is the comparable view — it is what RoR publishes — and it has one
defect: **"cheapest arm reaching T" evaluates a family at a single swept operating point.** When
no point lands near T the comparison silently switches to a different arm, and the number moves
for reasons that have nothing to do with the policy. On the 64k pool this made three targets
swing with the draw-ordering seed (sd 10–12) while their neighbours held at sd 3–4.

**The fix is convexification, not a different metric.** Operating points can be mixed by flipping
a coin per episode, so the achievable set is the *convex hull* of a family's (cost, accuracy)
points — the standard construction for a constrained MDP, and the same one that makes randomised
tests admissible in Neyman–Pearson. Comparing hulls compares the policy classes; comparing single
grid points compares the grids. Units and interpretation are unchanged.

| target | cheapest grid arm | **randomised (hull) frontier** |
|---|---|---|
| 50% | +56.0% ± 1.8 | **+42.2% ± 1.6** |
| 60% | +32.0% ± 4.2 | **+12.9% ± 3.1** |
| 65% | +8.8% ± **12.0** | **+11.3% ± 2.9** |
| 70% | +21.3% ± 4.4 | **+17.6% ± 3.7** |
| 75% | +6.3% ± **10.1** | **+15.5% ± 5.6** |
| 80% | +6.8% ± 7.5 | **+2.6% ± 6.2** |

Seed sd falls 2–4× at exactly the unstable cells. **It also cuts the flattering numbers** (50%:
+56.0 → +42.2; 60%: +32.0 → +12.9), because mixtures are available to the baseline too — the
correction is fair in both directions, which is why it is trustworthy rather than convenient.

**Under it the method is a strict improvement.** Sweeping 401 accuracy levels from 45% to 84.5%,
the mean advantage over RoR is negative at **zero** of them; the worst point anywhere is
**+1.72%**, at 79.3% accuracy. There is no regime where count-based routing beats us.
`hull_frontier.py --scan` regenerates this.

**Report the hull frontier as primary, with utility at matched R as the objective-level check.**
The frontier is what the field reports and is interpretable in dollars saved; the hull makes it
sound. Utility at matched R (§6.3d) agrees — 84/96 swept R, unanimous over 8 seeds. Do **not**
report *relative* utility: $J$ crosses zero, so $\Delta J/|J|$ puts +2213% next to −221%. Use
dollars, or $\Delta J/R$ (accuracy-equivalent units).

**6.3b Against the compiled fixed schedule, not just RoR.** Compiling the best fixed
route-and-depth schedule for each budget: five clear wins, one marginal, one tie, no losses --
up from three clear wins before query-conditioned cost, so the cost head is what separates us
from our closest non-adaptive competitor. Abstention runs 10.5% at the 70% point, low because
clean labels lifted the pool's solvability.

**6.3b-bis Three comparisons, and which one answers which question.**

A "policy family" spans four arm types: the RoR-faithful budget sweep (`counts`), our give-up
extension (`_abstain`), and our utility rule (`_value`, `_value_frozen`). Which arms each side may
use changes the number by tens of points, so every frontier claim must say which comparison it is.
All rows: hull frontier, % cost saved at matched accuracy, LCB 64k/8 seeds and TACO 32k/3 seeds.

| | LCB 50/60/70/80/84% | TACO 35/45/50/55/60% |
|---|---|---|
| **Q1** ours (any arm) vs **RoR as published** (`counts`) | **+46.8/+18.6/+21.2/+4.2/+9.6** | +49.4/+36.3/+24.8/-1.2/-42.8 |
| **Q2** both families, all arm types | +42.2/+12.9/+17.6/+2.6/+9.5 | +44.6/+30.1/+14.5/-8.4/-49.1 |
| **Q3** both pinned to the utility arm | +42.1/+4.5/+8.0/**-8.1**/+9.8 | +44.6/+13.8/-41.8/-94.7/-105.1 |

**Q1 is the deployment claim** — what a practitioner gets by switching from the published method.
On LiveCodeBench we win at **every** target. **Q2 is the fair scientific comparison**: it hands
RoR our utility rule and abstention, so it isolates the belief and cost source while granting both
sides the same formulation freedom. **Q3 is the ablation** that shows what the activations alone
buy, and it is where the 80% dip lives.

The ordering Q1 > Q2 > Q3 at every target is the expected one, and it is the decomposition: the
gap Q1-Q2 is what our *formulation* is worth to the baseline, and Q3 is what the *probe* is worth
once formulation is equalised. Read together: **the utility rule is a small reliable win, the
activation prior is the large volatile one.**

*Two errors this replaces, both from mismatched arm sets.* Comparing our best arm against the
baseline's best arm while calling the latter "RoR" credited RoR with our abstention, and on TACO
inverted the sign badly enough to produce a wrong mechanism (§6.10a). Restricting *us* to the
single `_value` arm while leaving RoR its full sweep made the same mistake in reverse and produced
a spurious -6.4% at LCB 80%. Neither is a defensible pairing; always state the arm sets.

**6.3b-bis-2 Attribution: what each contribution is worth.**

The frontier picks the cheapest arm *of a family* reaching T, and a family contains four arm
types: budget-swept (`counts`), our give-up extension (`_abstain`), and our utility rule
(`_value`, `_value_frozen`). So "RoR" was free to answer with `counts_value` — **RoR's beliefs
inside our decision rule, abstention included** — while our family answered at the same target
with a non-abstaining arm. That is not a comparison of belief sources, and on TACO it inverted
the sign and led us to a wrong mechanism (§6.10a).

**Always report the three-way decomposition, matched arm type on both sides**, against the
RoR-faithful `counts` (the code's own label: `"" is the RoR-faithful cell (no give-up arm)`):

| LCB target | rule only | + activation beliefs | full method |
|---|---|---|---|
| 50% | +8.1 ± 1 | **+46.4 ± 2** | **+46.7 ± 2** |
| 60% | +6.6 ± 1 | +13.9 ± 3 | +10.8 ± 4 |
| 65% | +5.8 ± 0 | +11.3 ± 3 | +7.4 ± 5 |
| 70% | +4.3 ± 1 | +1.5 ± 3 | +12.0 ± 4 |
| 75% | -0.2 ± 2 | -2.7 ± 2 | +4.7 ± 2 |
| 80% | +1.6 ± 1 | -7.4 ± 5 | -6.4 ± 5 |
| 84% | -0.7 ± 1 | +4.4 ± 2 | **+9.2 ± 1** |

**The utility rule is a small, reliable win; the activation prior is the large and volatile
term.** The rule alone gives +4.3 to +8.1% over the first four LCB targets and +4.2 to +12.1% at
*every* TACO target (§6.10a) — modest but never harmful. The prior swings from +46% to -7% on LCB
and +39% to -105% on TACO, tracking headroom to the pool ceiling.

**This changes the emphasis between contributions.** C2 (abstention as the zero-value action) is
cheaper to justify and more robust than C1 (activation priors) — it needs no probe, no per-problem
prediction, and it never lost on either dataset. C1 buys far more where there is headroom. The
paper should say that plainly rather than fold them into one number.

*Consequence for the tables above:* §6.3/§6.3a family-level numbers hand the baseline our rule,
which makes them **conservative** at targets where our arm also abstains, and **misleading** where
it does not. Keep them as the end-to-end "best configuration of each method" view, and lead the
attribution claims with this table.

**6.3c Why the advantage compresses near the ceiling — two value regimes, not a defect.**

Routing information is worth something only in proportion to what the policy is allowed *not* to
buy. Advantage against headroom (hull, 8 seeds, ceiling 84.8%):

| target | % of ceiling | advantage | what our arm does |
|---|---|---|---|
| 50% | 59% | **+42.2%** | abstains 49%, 2.05 attempts |
| 60% | 71% | +12.9% | abstains 26%, 3.26 attempts |
| 70% | 82% | +17.6% | abstains 0%, 4.65 attempts |
| 78% | 92% | +5.0% | abstains 0%, 5.21 attempts |
| 80% | 94% | +2.6% | abstains 0%, 5.36 attempts |
| **84%** | **99%** | **+9.5%** | **abstains 8.3%, 4.09 attempts** |

**Allocation regime (50–70%).** Slack in the target lets the policy skip problems the probe says
are hopeless; we reach 50% on 2.05 attempts where RoR needs 4.07.

**Saturation trough (~78–82%).** The target forces near-total spend, no allocation decision
survives, and both policies converge on "draw the 120B until it works". +2.6% is what is left when
there is almost nothing to decide. This is structural and should be predicted, not explained away.

**Abstention regime (at the ceiling).** The win *returns* — non-monotonically — because reaching
84.09% means identifying the ~15% of problems nothing in the pool solves and refusing to pay for
them. Ours spends $0.134 at 8.3% abstention and 4.09 attempts; RoR spends $0.147 grinding every
problem at 7.54 attempts, for the same accuracy. **C2 earns its place here specifically**, and the
non-monotonicity is the evidence that abstention and allocation are separate mechanisms rather
than one effect measured twice.

**6.3d Utility at matched R — the objective-level check.**

The frontier, even convexified, still never probes high $R$, where cost is irrelevant and only
the ceiling matters. Utility at matched $R$ closes that: both arms take the same $R$, every swept
point is a comparable pair, and no target must be hit. On the **64k pool over 8 seeds we win
84/96 swept $R$, every one of them unanimous across all 8 seeds**, including 19/19 at high $R$
($R>\$1$). The 12 losses are the 12 *lowest* $R$ ($\$0.0006$–$\$0.0018$), where $\Delta J$ is
$-\$0.00015$ to $-\$0.00030$ against a charged probe of $\$0.000149$ — i.e. **the losses are the
probe fee and nothing else**, at values of a correct answer too small to justify buying any
prediction. That is the honest floor of the method.

On the older 32k pool the same metric gave 73/96 and *lost* at high $R$ ($-\$0.030$ at $R=6.16$);
that reversed once truncation was fixed, along with the ceiling gap (§6.3c). The shape below is
from the 32k pool and is retained only for the regime story:

| regime | LCB | TACO |
|---|---|---|
| low R (cost-dominated) | tied | tied |
| mid R | **+$0.0299** | **+$0.0533** |
| **high R (accuracy-dominated)** | **-$0.0163** | **-$0.0660** |

On the 32k pool this looked structural — we lost at high R because our ceiling was lower (86.1%
vs 86.5% on LCB, 61.3% vs 61.9% on TACO), which read as abstention leaving a few tenths of a point
unclaimed. **It was truncation.** On the re-collected pool all three families reach *exactly* the
same ceiling, 84.80%, and high R flips from a $0.030 loss to winning 19/19: truncated draws had
been denying the abstaining policy the attempts it needed. TACO has not yet been re-collected at
64k, so its row above still stands.

*(Reporting guidance now lives in §6.3a: hull frontier primary, this as the objective-level
check.)*

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

| probe | params | LCB pool AUC | LCB cost $R^2$ (oss120 / oss20) | TACO pool AUC | TACO cost $R^2$ (oss120 / oss20) |
|---|---|---|---|---|---|
| **scout 4B** | 4.41B | **0.8652** | **+0.492 / +0.370** | **0.8453** | **+0.262 / -0.027** |
| Qwen3-1.7B | 2.15B | 0.7861 | +0.450 / +0.258 | 0.8111 | +0.150 / -0.119 |
| Qwen3-0.6B | 0.69B | 0.8184 | +0.426 / +0.264 | 0.8042 | +0.170 / -0.149 |
| Coder-1.5B | 1.89B | 0.8167 | +0.285 / +0.159 | 0.8057 | +0.077 / -0.270 |
| Coder-0.5B | 0.66B | 0.8261 | +0.328 / +0.148 | 0.7816 | +0.001 / **-0.299** |

*Cost columns regenerated in **dollar space** on the fully re-collected pools; the AUC columns are
unchanged by calibration, since Platt scaling is monotone. The earlier log-space cost figures
(0.633 down to 0.355) overstated every cell -- see §6.9-cost.*

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

**6.7-REGEN Price ratio: the threshold is ~10x, not ~6x.** Regenerated on the fully re-collected
pool with calibrated beliefs and the dollar-space cost head. The scout stays at $0.278/M; only the
120B's price moves. Hull frontier, seed 0.

| gpt-oss-120b price | ratio to scout | 50% | 65% | 75% | 80% |
|---|---|---|---|---|---|
| $0.834/M | 3x | -36.8% | **-118.9%** | -63.1% | -33.5% |
| $1.668/M | 6x | +10.5% | -44.3% | -22.4% | -10.7% |
| $2.780/M | 10x | +33.5% | -0.5% | -1.3% | +2.6% |
| $6.950/M | 25x | **+46.6%** | **+14.5%** | **+11.9%** | **+7.7%** |
| $11.13/M (list) | 40x | +45.8% | +12.9% | +12.1% | +5.5% |

**This moves the threshold up and the previous claim was too generous.** The outline previously
said "above ~6x"; on clean data with calibrated heads, **6x is clearly negative at three of four
targets** and 10x is only break-even above the 50% target. A solid win needs **~25x**, at which
point the result is stable through list price. Below ~10x the method is actively harmful, and at
3x catastrophically so (-118.9%).

*Why this matters for the paper's honesty.* The spread the method exploits is basis-dependent
(SS6.7), and MoE active-vs-total parameter accounting can move a pool by more than the 6x-to-25x
gap that separates "harmful" from "clearly useful". **State the threshold in the units the operator
actually pays and let them check their own pool**, rather than asserting the method transfers.

**6.14-REGEN Coupled RoR, the strongest honest baseline, on clean data.** Giving count beliefs a
cross-route difficulty channel and comparing against *that* rather than plain counts:

| | 50% | 65% | 75% | 80% |
|---|---|---|---|---|
| vs plain `counts` | +45.8% | +12.9% | +12.1% | +5.5% |
| **vs `counts_coupled`** | **+51.7%** | **+21.2%** | **+12.6%** | **+7.1%** |

Our margin is *larger* against the coupled baseline, not smaller: letting the baseline infer
difficulty from cross-route failures makes it spend more, not less. The earlier reading (coupling
moves our advantage by at most 1.1pt) was measured against the wrong arm.

**6.15-REGEN Start protocol.** `free_start` +50.8/+13.6/+10.8/+4.9% against `scout_first`
+45.8/+12.9/+12.1/+5.5% at 50/65/75/80%. Conclusion unchanged: the mandatory scout does not carry
the result, and the two protocols agree within a few points.

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

**6.9-HEADLINE-2 Both datasets are strict improvements, under configurations chosen on
calibration.**

| | strict-improvement scan | worst point |
|---|---|---|
| **LCB, full method** (beliefs + query-conditioned cost) | **0 of 401 negative** | **+5.07%** |
| LCB, beliefs + constant costs | 0 of 401 negative | +2.90% |
| **TACO, beliefs + constant costs** | **0 of 401 negative** | **+1.00%** |
| TACO, full method | 70 of 401 negative | -11.83% |

**LiveCodeBench wants the cost head; TACO wants it dropped -- and the choice is made for us.** The
cost head's dollar-space shrinkage (SS6.9-cost) collapses to the per-route constant when the
calibration split shows no dollar signal, which is exactly TACO's situation ($R^2\approx 0$ against
LiveCodeBench's 0.37). **No dataset-specific tuning is required**; the precondition is measurable
before deployment. That is a stronger claim than either benchmark alone: the cost component is
*gated on a measurable property of the pool*, and the gate is part of the method.

TACO at fractions of its 58.9% ceiling, beliefs + constant costs, 3 seeds:
**+35.0 / +30.6 / +11.2 / +12.9 / +5.1%** at 55/65/75/85/95%.

**One tension to disclose rather than resolve.** On utility at matched $R$ the *full* method is
better on TACO (64/96 against 58/96 for beliefs-only) while being worse on the frontier. So the
cost head raises mean utility and adds downside at particular accuracy targets. Report both; do
not select the flattering metric per dataset.

**How TACO should be presented.** Not as a second benchmark row -- it is the only pool with enough
unsolvable mass (38-40%) to run the dose-response experiment of SS6.9k, which is a stronger result
than a frontier table. Lead with the sweep, report the frontier over the reachable range, and state
that targets above ~95% of ceiling are a regime where even oracle cost loses (SS6.9g) and which
nobody deploys in.

**6.9-HEADLINE Final results on the fully re-collected pools.** All three routes at a 65,536-token
cap, calibrated belief head, cost head fitted and shrunk in dollar space, both methods on their own
MLE-fitted decay, hull frontier, against RoR as published (`counts`).

**LiveCodeBench (5 draw-ordering seeds):**

| target | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| advantage | **+45.8% ± 1.4** | **+17.7% ± 2.2** | **+18.7% ± 3.9** | **+8.0% ± 3.5** | **+8.6% ± 2.5** |

Scanning 401 accuracy levels from 45% to 84.5%: **negative at none, worst point +5.07%.** Utility
at matched $R$: **79/96** swept values. This is a strict improvement with a floor five points above
the baseline.

**TACO (3 seeds, matched split, oss20/oss120 re-collected; the scout is still 32k):**

| target | 35% | 45% | 50% | 55% |
|---|---|---|---|---|
| advantage | **+34.4% ± 2.8** | **+7.2% ± 4.3** | **+18.0% ± 3.7** | -7.3% ± 10.2 |

Utility **64/96**. The 60% target is unreachable on this pool's test split. So TACO is a win at
three of four reachable targets and on utility, but not a strict improvement -- consistent with
SS6.9l: it is strong on the abstention axis and weak on the cost axis, and the re-collection does
not change that because TACO's cost tail is intrinsic rather than a cap artifact (SS6.9m).

**Raising the cap further will not help.** Draw lengths follow an approximate power law -- each
doubling of the cap roughly halves the survivors (LCB 17.7/10.4/5.1/2.0% above 8k/16k/32k/64k;
TACO 31.8/19.1/9.9/4.0%) -- so there is no cap at which the distribution becomes well behaved.
Worse, each increase makes spend *more* tail-dominated: p90/p50 rises (LCB 9.50 -> 10.82) and the
top decile of draws grows from 56% to **62%** of all spend on LCB, 41% to **50%** on TACO. **Over
half of all money goes to a tenth of the draws.** That is the strongest motivation for
query-conditioned cost and simultaneously the reason it is hard.

**6.9a The 64k pool changes the headline, and the two metrics disagree — informatively.**
Rebuilding the whole chain on the re-collected pool (oss20 and oss120 at 64k; the scout is still
the local 32k collection, and fully local 64k runs for all three routes are in flight), 96 points,
**8 draw-ordering seeds**:

| target | 32k pool (3 seeds) | 64k pool (8 seeds) | beliefs only, 64k |
|---|---|---|---|
| 50% | +49.1 ± 5.2 | **+56.0 ± 1.8** | +51.8 ± 2.6 |
| 60% | +27.8 ± 1.0 | **+32.0 ± 4.2** | +26.4 ± 5.0 |
| 65% | +25.4 ± 2.1 | +8.8 ± 12.0 *(ns)* | +9.2 ± 10.3 |
| 70% | +28.0 ± 8.2 | **+21.3 ± 4.4** | +3.0 ± 4.3 |
| 75% | +39.7 ± 2.1 | +6.3 ± 10.1 *(ns)* | +0.9 ± 7.4 |
| 80% | +21.5 ± 5.0 | +6.8 ± 7.5 *(ns)* | **-6.2 ± 10.4** |

**The truncation ablation predicted this.** §6.9's counterfactual forecast LCB 80% at +6.8%; the
real re-collection gives +6.8%. Deleting runaway draws synthetically and raising the cap for real
agree to a tenth of a point, which is the strongest evidence the ablation was measuring the right
thing.

**On the frontier the win narrows to the cost-sensitive regime**, and 65/75/80% become
indistinguishable from the baseline across seeds. But 70% stays solid (+21.3 ± 4.4) while both its
*neighbours* collapse — a target-local instability, which is the signature of the grid artifact in
§6.3a rather than of a real accuracy-dependent effect.

**On the primary metric the result gets better, not worse.**

| | 32k pool | 64k pool |
|---|---|---|
| swept R where we win | 73/96 | **84/96** |
| unanimous across seeds | — | **84/84** |
| high R (R > $1) | **loses** (-$0.030 at R=6.16) | **wins 19/19** |
| accuracy ceiling vs RoR | 86.1% vs 86.5% | **84.80% vs 84.80%** |

Every one of the 84 wins is unanimous over all 8 seeds. **The 12 losses are the probe fee and
nothing else**: they are the 12 lowest-R points ($0.0006–$0.0018), where R is too small for the
policy to buy anything, and $\Delta J$ is $-0.00015$ to $-0.00030$ against a charged probe of
$0.000149. That is the honest floor of the method — below a certain value of a correct answer,
paying for a prediction is strictly wasted — and it is worth stating as such.

**So report utility at matched R as primary and say why.** The two metrics move in *opposite*
directions on cleaner data: utility improves and stabilises (73→84/96, unanimous), while the
frontier destabilises at three targets. They cannot both be tracking the underlying quality of
the policy. The frontier's "cheapest arm reaching T" switches policy families discontinuously
(§6.3a), and the 64k pool, by moving many operating points slightly, moves which family wins each
target — noise the matched-R comparison is immune to by construction.

*Serving path is not the explanation.* The 32k→64k comparison also changes the serving path
(local vLLM → OpenRouter), so we measured that channel on its own: oss120 never truncates, and
its paired per-problem solve rate moves **-1.12pt, 95% CI [-1.97, -0.27]** over 892 problems.
Detectable, but far too small to move a frontier point from +21.5% to +6.8% -- and the truncation
ablation (§6.9), which changes *no* serving path at all, forecast +6.8% at that target. Truncation
explains the change; the serving path does not. The one remaining hole is the scout, still 32k
here and truncating 4.13% of draws; a local 64k scout collection is running to close it.

**6.9b Calibrating the belief head: the fix, and what it costs.** Platt scaling on the held-out
calibration split (§C3b), 96 points, 3 matched seeds, against pure RoR (`counts`):

| target | LCB raw | **LCB calibrated** | TACO raw | **TACO calibrated** |
|---|---|---|---|---|
| 50% / 35% | +41.7 ± 1.8 | +41.7 ± 1.9 | +44.6 ± 2.8 | +38.1 ± 3.3 |
| 60% / 40% | +13.7 ± 4.8 | +11.8 ± 2.7 | +37.5 ± 2.5 | +32.6 ± 2.8 |
| 65% / 45% | +12.3 ± 2.2 | +10.7 ± 1.1 | +30.1 ± 3.1 | +27.6 ± 3.9 |
| 70% / 50% | +18.2 ± 1.9 | +16.3 ± 1.6 | +14.5 ± 4.0 | **+17.7 ± 1.0** |
| 75% / 55% | +20.2 ± 2.0 | +14.1 ± 0.9 | -8.4 ± 2.0 | -7.2 ± 2.8 |
| 80% / 60% | +1.5 ± 6.9 | **+10.5 ± 3.3** | -49.1 ± 19.5 | **-27.2 ± 9.9** |
| 84% | +9.6 ± 2.2 | **+11.5 ± 4.0** | — | — |

**On LiveCodeBench this makes the method a strict improvement with margin.** Scanning 401
accuracy levels from 45% to 84.5%: raw beliefs are negative at 4 of them (worst -0.44%);
**calibrated beliefs are negative at none, with a worst point of +7.59%.** The floor moves from
"occasionally a hair below the baseline" to "never less than 7.6% cheaper".

**It buys the floor by giving up the peak, exactly as compression should.** The 75% target falls
20.2 -> 14.1, and seed sd falls almost everywhere (80%: 6.9 -> 3.3; 75%: 2.0 -> 0.9). That is the
trade to state explicitly: **calibration converts peak advantage into worst-case guarantee**, and
for a router that is the right direction, because the failure mode of an uncalibrated router is
unbounded while its upside is not.

**It helps TACO most where TACO was worst but does not rescue it.** The 60% target improves
-49.1 -> -27.2 with sd halved, and 50% actually flips to *better* than raw (+14.5 -> +17.7). But
the scan is still negative at 96 of 401 levels (worst -62.9 -> -52.3). So over-confidence was a
*contributor* to the TACO failure, not its sole cause, and §6.10a's headroom account still stands
for the residue. **TACO is also still the 32k pool**; its re-collection is running.

**6.9c Failures are evidence, and the decay is the posterior — but the frontier rewards being
wrong about it.**

$\hat\theta_m(x)$ is an *estimate*, so a failure is genuine new evidence that the problem is
harder than the probe thought. Modelling the residual uncertainty as $\theta\mid\hat\theta \sim
\mathrm{Beta}$ with mean $\hat\theta$ and concentration $\kappa$ gives the posterior after $n$
failures as exactly $\hat\theta_m(x)\,\kappa/(\kappa+n)$ — the decay's own functional form, with
$\sigma=\kappa=$ **how much uncertainty the probe leaves**, a quantity to fit rather than a knob.
A perfect probe gives $\kappa=\infty$ and no decay; a useless one gives small $\kappa$, which is RoR.

Maximum-likelihood $\kappa$ on the calibration split: **0.54/0.79/0.64** (TACO) and
**0.50/1.00/1.35** (LCB) for scout/oss20/oss120 — all *below* the inherited $\sigma=2$, i.e.
failures are **more** informative than the default assumes. Held-out on test, $\sigma=2$
over-predicts after failures on nearly every route (TACO oss120 at $n{=}1$: observed 0.086,
$\sigma{=}2$ says 0.251); fitted $\kappa$ is closer but still optimistic.

**Our beliefs are better calibrated than RoR's at 18 of 18 route × failure-count cells, on both
datasets** (total absolute error: TACO 0.620 vs 1.024, LCB 1.093 vs 2.434). RoR believes oss20
has a 15.2% chance after three failures where the observed rate is **0.0%**.

**And that is exactly why RoR wins the near-ceiling frontier.** Being over-optimistic after
failures makes it keep grinding, and grinding is what an accuracy target near the pool ceiling
demands. Our correctly pessimistic beliefs say stop — right for utility, wrong for
cost-at-matched-accuracy in that regime. **The metric rewards a miscalibrated policy**, which is
the sharpest argument yet for reporting utility at matched $R$ alongside it (§6.3d). Testable and
being tested: raising $\sigma$ for our arms only should *improve* the near-ceiling frontier while
degrading utility.

*Myopia is not the cause, and this is now settled at full depth.* Bellman lookahead at $h=4$ and
$h=6$ changes accuracy and abstention by **exactly 0.000** at every operating point on both
datasets — it never rescues a give-up, because $V\ge 0$ can only make the policy abstain *less*
and it never does. It is, however, a free cost saving: $-\$0.0070$ per episode on TACO and
$-\$0.0006$ on LCB at identical accuracy. Take the lookahead for the cost, not for the decisions.

**6.9d The decay's functional form is right; its constant was never fitted by anyone.**

A natural objection: failing a problem the probe called *easy* should update harder than failing
one it already called *hard*, and $\hat\theta\,\kappa/(\kappa+n)$ applies the same *proportional*
decay to both. Tested three ways.

**Analytically the objection dissolves.** The absolute drop after one failure is
$\hat\theta-\hat\theta\kappa/(\kappa+1)=\hat\theta/(\kappa+1)$ — **proportional to
$\hat\theta$**. At $\kappa{=}1$, a problem predicted 0.8 drops 0.40; one predicted 0.1 drops 0.05,
eight times less. The decision rule compares $p$ with $c/R$ in absolute units, so the Beta
posterior already encodes surprisal in the units the policy consumes. Proportional invariance is
the feature, not the bug.

**Empirically the extra parameter is not supported.** Fitting $\kappa$ separately above and below
the median $\hat\theta$ splits 3–3 on which half decays slower, and held-out it wins 2 of 6 route
cells by margins in the fourth decimal. Held-out log-loss of the sequential prediction the policy
acts on:

| | $\sigma{=}2$ | $\kappa$ fitted | $\kappa(\hat\theta)$ |
|---|---|---|---|
| TACO scout / oss20 / oss120 | 0.1696 / 0.3214 / 0.3169 | **0.1608** / 0.3177 / **0.2730** | 0.1609 / 0.3176 / 0.2759 |
| LCB scout / oss20 / oss120 | 0.2175 / 0.4328 / 0.4639 | 0.1855 / **0.4101** / **0.4450** | **0.1825** / 0.4103 / 0.4516 |

**But $\sigma{=}2$ loses all six cells**, and that is the finding. The constant is inherited from
RoR and has never been fitted — by them or, until now, by us. Fitted on calibration by maximum
likelihood it is **0.5–1.35** for our beliefs and **0.25–1.09** for RoR's, and ours is larger at
**6 of 6** route × dataset cells, exactly as the posterior account predicts: a better prior leaves
less for a failure to teach. *Both* methods must be re-run with their own fitted constant, or
fitting only ours would rig the comparison.

**6.9n RESOLVED: difficulty and cost are monotonically related on LiveCodeBench and
hump-shaped on TACO.**

| pool solve rate | LCB mean cost | TACO mean cost |
|---|---|---|
| never solves | 20,666 | **11,416** |
| (0, 0.2] | 22,308 | **20,229** |
| (0.2, 0.4] | 10,468 | 15,796 |
| (0.4, 0.6] | 6,482 | 12,139 |
| (0.6, 0.8] | 2,761 | 5,553 |
| always solves | 553 | 2,818 |
| **between-stratum spread** | **40.3x** | **7.2x** |
| **share of cost variance between strata** | **46.4%** | **12.8%** |

**On LiveCodeBench cost falls monotonically with solve rate over a 40x range**, so a difficulty
signal converts directly into a cost estimate. **On TACO the relation is hump-shaped**: the
never-solved problems are *cheap* (mean 11,416, median 3,526) because the model recognises they are
beyond it and bails, while the *almost*-never-solved are the most expensive (20,229) because it
grinds and fails. Cost rises then falls with difficulty, and a monotone difficulty signal cannot
express that. Only **12.8%** of TACO's cost variance is between difficulty strata at all, against
**46.4%** on LiveCodeBench.

This is the full explanation of SS6.9h: our cost head is a difficulty head, difficulty maps to cost
on one benchmark and not the other, and the head inherits exactly that.

**Two competing explanations were tested and rejected**, both worth recording because both are the
natural first guesses:
- *"TACO is harder / our mixture is too hard."* Reweighting LiveCodeBench's test set to TACO's
  exact difficulty histogram (40% never-solve, 5% always-solve) leaves its cost $R^2$ at
  **+0.338 +- 0.027**, against its unweighted +0.370 and TACO's -0.027. The difficulty mixture
  explains almost none of the gap.
- *"Restriction of range."* Rejected earlier on a bad test (a 5-95 percentile *range* that spanned
  0.00-0.94, i.e. no restriction). The distribution-matched test above is the correct one and also
  rejects it.

**Predicted and confirmed fix direction.** If TACO needs a hump and LiveCodeBench a line, a
non-monotone learner should help TACO and not LiveCodeBench. Gradient boosting does exactly that:
TACO oss20 **0.005 -> 0.135**, TACO oss120 0.238 -> 0.325, while losing on both scout rows
(SS6.9h). Select the functional form per route on calibration.

*Caveat on the within-stratum numbers.* Within a narrow difficulty band the head is worse than
that band's own mean ($R^2$ negative on both datasets). That is not evidence the head is useless:
the policy never sees stratum means, only a global per-route constant, and against that the head is
clearly better (+0.370 on LCB, driving the +47% cost-only advantage). The correct statement is
that **the head is useful and all of its usefulness is difficulty-mediated.**

**6.9o Prescription for cost prediction, and why it may not be worth cashing.**

The hump (SS6.9n) says a monotone predictor cannot work on TACO, which yields a concrete recipe:

| | linear on activations | non-monotone $f(\hat{\text{difficulty}})$ | blended on calibration |
|---|---|---|---|
| LCB | +0.370 | +0.283 | **+0.385** |
| TACO | **-0.027** | -0.104 | **+0.093** |

1. **Two-stage.** Predict difficulty from activations (which works: pool AUC 0.85), then map
   predicted difficulty to cost through a *non-monotone* binned function, and blend with the linear
   head on calibration. Takes TACO from actively harmful to useful.
2. **Choose the functional form per route on calibration.** Gradient boosting independently reaches
   0.135 on TACO oss20 and 0.325 on TACO oss120 while losing on both scout rows -- the signature of
   "TACO needs a hump, LiveCodeBench needs a line".
3. **Stack the free scout-draw features** (SS6.9h): TACO oss20 0.005 -> 0.187.

**But we have already run the experiment that tests whether this pays.** Stacking improved TACO
oss20's dollar $R^2$ **37-fold** (0.005 -> 0.187) and the policy did not move: utility 68 -> 67/96,
frontier slightly worse. **Better cost $R^2$ did not become better decisions.** The reason is
SS6.9i: the decision compares cost *ratios* between routes, and those are near-unpredictable
(negative $R^2$ in 4 of 6 route pairs); every fix above targets the *level*.

**So the honest status is:** the recipe improves the estimate, is well-motivated by a measured
mechanism, and has no demonstrated effect on the policy. Combined with the oracle bound (SS6.9g:
perfect cost is worth +13pt on TACO at mid targets and nothing near the ceiling), **cost estimation
is the wrong place to spend further effort.** The stop/go channel (SS6.9i) and the unsolvable-
fraction axis (SS6.9k) are where the method's value actually lives.

*A grading artifact was checked and is negligible.* 53 of the cheap never-solved failures carry
`module 'tmp_sol' has no attribute 'Sol...'`, a call-format mismatch where the harness wants a
`Solution` class and the model writes a stdin script. It affects only **9 of 883** problems (1.0%,
those 53 errors being 9 problems x 6 draws); removing them leaves the between-stratum spread at
7.2x, unchanged. The hump is real, not a harness artifact.

**6.9m Why cost is predictable on LiveCodeBench and not on TACO. Three explanations tested and
rejected; what remains.**

Both are competitive-programming benchmarks, so the difference demands an account.

**Rejected 1 -- restriction of range.** TACO is medium+hard only, so its difficulty spread could be
truncated. It is not the cause: restricting LiveCodeBench to TACO's difficulty band moves
difficulty->cost $R^2$ only 0.453 -> 0.393, against TACO's 0.110.

**Rejected 2 -- label noise.** Split-half reliability of the per-problem cost label is **0.86-0.96
everywhere**, including the failing cell (TACO solvable, 0.907). The signal is real and we are
failing to extract it.

**Rejected 3 -- truncation bimodality.** A solvable TACO problem mixes short successes with
failures that hit the cap 18.9% of the time (LCB: 5.9%), which should make its mean cost
tail-dominated. Plausible, and false: the 64k re-collection halves the at-cap rate (13.8% -> 6.4%)
and predictability does *not* recover ($R^2$ -0.013 -> -0.092, p90/p50 4.7x -> 4.9x). **The heavy
tail is intrinsic, not an artifact of the cap.**

**Where the failure actually lives.** Not TACO as a whole -- TACO's *unsolvable* problems are the
best-predicted cell we have ($R^2$ +0.408, better than any LiveCodeBench cell). It is TACO's
**solvable** problems in **dollar space**: $R^2$ +0.279 in logs, -0.013 in dollars, label
reliability 0.907, p90/p50 4.9x. The predictor ranks these problems correctly and cannot locate
the magnitude of the tail, and the mean of a heavy-tailed variable is set by the tail.

**The two benchmarks are not the same cost regime, despite the same task genre.** On LiveCodeBench
the model usually succeeds quickly (median 1,374 tokens on solvable problems) and rambles only on
the rare hopeless ones -- so "stuck" is a distinct, uniformly expensive mode (median 13,948,
sd/mean 0.78) that a probe can flag. TACO medium+hard is pre-filtered to remove easy problems, so
the model sits near its competence limit throughout: solvable problems cost 2.6x more at the median
and fail into a runaway 18.9% of the time. **Filtering a benchmark for difficulty changes the cost
distribution's shape, not just its level**, and that is what breaks the cost head.

*This is where we stop.* The residual explanation -- the prompt does not encode which problems will
run away (SS6.9h) -- is consistent with every measurement but is not further falsifiable without
new features. The oracle bound (SS6.9g) says closing it is worth +13pt on TACO, so it is a real
target for future work rather than a defect to re-fit around.

**6.9l Two independent value axes, which is why TACO is both the best and the worst case.**

An apparent contradiction: SS6.9k shows the advantage *rises* with the fraction of problems nothing
solves, and TACO has 2.5x more of them than LiveCodeBench (38.1% vs 15.2% of the test split) --
yet TACO is where we lose. Normalising for ceiling does **not** resolve it; at matched fractions
of each pool's own ceiling TACO is still worse everywhere (+32.8/+14.0/+11.0/+3.3/-8.5 against
+50.9/+36.2/+17.6/+17.8/+7.4). Decomposing does:

| at matched % of ceiling | 55% | 65% | 75% | 85% |
|---|---|---|---|---|
| beliefs only, LCB | +45.7% | +35.6% | +15.1% | +8.2% |
| beliefs only, TACO | +34.4% | +23.5% | +6.7% | +6.1% |
| **cost only, LCB** | **+47.4%** | **+38.0%** | **+18.6%** | **+17.1%** |
| **cost only, TACO** | **+17.2%** | **+12.0%** | **+0.5%** | **-8.7%** |

**The belief half transfers; the cost half does not.** TACO retains 70-75% of LiveCodeBench's
belief-side value and only ~35% of its cost-side value, and that 18-30pt cost gap is most of the
cross-dataset difference.

**So the method has two independent value sources, and a benchmark can be strong on one and weak
on the other:**

1. **Abstention value** scales with the fraction of problems nothing in the pool solves (SS6.9k,
   measured by intervention). TACO is the *strong* case.
2. **Cost-conditioning value** scales with how much cost is predictable from the prompt, which is
   bounded by how much of cost is difficulty (SS6.9h: 15-50% on LCB, 7-15% on TACO). TACO is the
   *weak* case.

TACO is high on (1) and low on (2); LiveCodeBench is the reverse. They net out against TACO, and
because the cost failure bites hardest at high targets -- exactly where TACO's reported range sits
-- one component failing looked like the whole dataset failing.

**Deployment consequence: the two halves are separately shippable.** On a TACO-like pool, ship the
belief head and keep the baseline's constant costs; that configuration is positive at 4 of 5 TACO
targets (SS6.9g, lambda=0). Ship both only where cost is predictable. **Do not present the method
as a single indivisible contribution** -- the evidence says it is two, with different scope
conditions and different failure modes.

**6.9k The scope condition, measured by intervention: advantage scales with what you can skip.**

If query-conditioning pays through the stop/go decision (SS6.9i), its value must scale with how
many problems *nothing in the pool solves* -- with none, there is nothing to skip. LiveCodeBench
sits at 8.3% and TACO at 40%, but they differ in a dozen other ways. So hold TACO fixed and vary
**only the mix**, dropping problems (never duplicating) and preserving split proportions.
Targets are expressed as a fraction of each pool's own ceiling, so the comparison is like-for-like:

| unsolvable | pool ceiling | 50% of ceiling | 60% | 70% | 80% |
|---|---|---|---|---|---|
| 10% | 89.6% | +26.6% | +16.9% | -3.0% | -6.2% |
| 20% | 80.0% | +23.9% | +19.5% | -6.1% | -6.4% |
| 30% | 70.3% | +26.3% | +14.9% | -2.7% | **+4.0%** |
| 40% | 60.0% | +38.6% | +24.0% | **+12.0%** | **+13.3%** |
| 50% | 50.0% | **+50.2%** | **+31.0%** | **+17.2%** | **+9.1%** |

**The advantage rises with the unsolvable fraction in every column**, and the high-target columns
**flip sign** -- from -6.2% at 10% unsolvable to +9.1% at 50%. This is an intervention on one
variable in one dataset, so it is not confounded by benchmark identity, and it is the cleanest
evidence in the paper that the mechanism is selective prediction rather than routing.

**It also predicts where the method should be deployed**: pools whose members frequently *all*
fail -- agentic SWE, hard theorem proving, long-horizon tasks -- not pools where nearly everything
is eventually solvable. And it reframes the TACO result: TACO is not a failure case, it is the
*high-value* end of this axis whose targets we happened to evaluate too close to its ceiling.

**6.9j Why cost is NOT state-dependent, and why that quietly favours us.**

Tempting error, tested and retracted. Expected next-draw cost *appears* to rise steeply with
failures (LCB oss20 5,670 -> 17,398 tokens after three, 3.07x), which looks like a missing mirror
of the belief decay. **It is not.** Draws are i.i.d. samples from one model on one problem, so
within a problem they are exchangeable and per-draw cost is constant by construction. Measured,
the within-problem ratio of draw 3 to draw 0 is **1.00 / 1.00 / 1.00 / 1.00 / 1.06 / 0.93** across
the six route-dataset cells, and the marginal rise is *entirely* selection: problems still alive
at depth 3 had draw-0 costs 1.10-2.90x the average.

**So the asymmetry in the rule is correct.** Beliefs decay because a failure is evidence about an
unknown $\theta$; costs do not, because sampling from a distribution does not change it.

**But the selection is real and it biases the baseline specifically.** The states a sequential
policy visits are not a random sample of problems -- they are the expensive, hard ones. A
per-route constant is fitted over all problems and therefore **under-prices continuation by up to
2.9x exactly where the policy is still deciding**, while a per-problem head already knows this
problem is expensive. That is a reason to prefer query-conditioned cost we had not identified:
**the baseline's cost model is biased in the sequential setting, not merely imprecise.** Needs its
own ablation before being claimed.

**6.9i The unifying result: query-conditioning pays in the stop/go decision, not in arm selection.**

Two independent decompositions land in the same place.

**Beliefs help through abstention, not routing** (SS6.9f): activation priors are worth up to
**+43.7%** when the give-up action is available and **~0%** when it is disabled, on both datasets.

**Costs help through the absolute level, not the ratio.** The utility rule uses absolute $c_m$ to
decide *whether any action is worth taking* and relative $c_m$ to decide *which*. We predict the
level reasonably ($R^2$ up to 0.49) and the ratio not at all:

| pair | sd(log cost ratio) TACO / LCB | ratio $R^2$ TACO / LCB |
|---|---|---|
| oss20 -> oss120 (**within family**) | **0.564 / 0.616** | -0.287 / -0.102 |
| scout -> oss20 (cross family) | 1.426 / 1.264 | -0.038 / -0.022 |
| scout -> oss120 (cross family) | 1.373 / 1.142 | +0.129 / +0.050 |

**The ratio is negatively predictable in 4 of 6 cells** -- our conditioned estimate is worse than a
constant at exactly the quantity route selection compares. So both halves of the method contribute
through the same channel, and neither improves arm selection.

**This single claim explains every result in SS6.9**: why per-candidate probing bought nothing
(C4), why the budget-swept family shows no benefit, why the cost head's errors hurt TACO
disproportionately, and why the near-ceiling regime -- where there is no stop/go decision left to
make -- is where we lose.

**A deployment rule falls out of it.** Within-family cost ratios are **2.3x more stable** than
cross-family ones (sd 0.56-0.62 against 1.14-1.43), because models of a family are verbose on the
same problems (r = 0.90-0.93, SS6.9h). So **a within-family pool can route on per-route constants
and be nearly right; a cross-family pool has far more ratio spread, which is where
query-conditioned cost would pay -- and precisely where prompt-only estimation fails.** That is a
testable prediction for anyone assembling a heterogeneous pool.

**6.9h Why cost prediction is hard, and it is not the shared difficulty factor.**

The natural worry: both heads read one activation, so if the probe finds difficulty, the cost head
should inherit it. It does — completely — and that is the problem.

**How much of per-problem cost is difficulty at all?** Regressing cost on the *true* per-problem
pass rate (the best any pure difficulty signal could do):

| | difficulty -> cost, dollar $R^2$ | our cost head, dollar $R^2$ |
|---|---|---|
| LCB oss120 | 0.501 | **0.492** |
| LCB oss20 | 0.388 | **0.370** |
| LCB scout | 0.152 | 0.162 |
| TACO oss120 | 0.154 | 0.238 |
| TACO oss20 | 0.133 | 0.005 |
| TACO scout | 0.070 | 0.128 |

**Our cost head is a difficulty head in disguise, and it is already saturated.** On LiveCodeBench
it lands within 0.02 of the difficulty-implied ceiling at both large routes. So the shared factor
is not what we are failing to extract — we have all of it.

**The gap is verbosity, which is orthogonal to difficulty.** On TACO, difficulty explains only
7-15% of cost variance against LiveCodeBench's 15-50%, so **85-93% of TACO's cost variance is how
much the model rambles**. That is a genuinely different target from the one the belief head solves.

**Verbosity is strongly shared across models — and invisible in the prompt.** Log-length
correlates **r = 0.929** between gpt-oss-20b and gpt-oss-120b on LiveCodeBench (0.898 on TACO), and
a single factor explains **78-86%** of all cross-route variance, so "this problem makes models
ramble" is a real property of the problem. The ceiling it implies is log-$R^2$ **0.88-0.93**.

But no prompt probe gets near it, **including each model's own**:

| target route | probe = scout | probe = oss20 | probe = oss120 |
|---|---|---|---|
| scout | **+0.162** | +0.076 | +0.100 |
| oss20 | **+0.370** | +0.302 | +0.270 |
| oss120 | **+0.492** | +0.440 | +0.419 |

gpt-oss-20b's own activations predict its own cost *worse* than the 4B scout's do. So the shared
verbosity factor is **not encoded in the prompt representation** — how long a model will ramble is
settled during decoding, not before it. **This bounds the research direction**: prompt-only cost
prediction has a hard ceiling far below the problem-level signal, and closing the gap requires
partial generation (decode $k$ tokens, then extrapolate), which costs money and changes the
method's economics. It also re-confirms C4 from an independent angle: per-candidate probing loses
on cost as well as on beliefs.

**The labels are clean, so the headroom is real.** Split-half reliability of the per-problem cost
label gives a dollar-space ceiling of $R^2$ **0.82-0.96** against our 0.03-0.58. This is not a
noise floor; it is a modelling gap.

**Two concrete leads, both measured.**
1. **Non-linearity, per route.** Gradient boosting on a 1/16 feature subsample beats ridge exactly
   where ridge fails worst — TACO oss20 **0.005 -> 0.135**, TACO oss120 **0.238 -> 0.325** — while
   losing on the scout rows. Select the model per route on calibration, the same discipline as the
   log/direct target-space choice.
2. ~~**Cost is outcome-conditional, so make it state-dependent.**~~ **Retracted — see 6.9j.**
   Failed draws do cost 2.4x-10.6x more than solved ones, but draws within a problem are
   exchangeable, so there is no within-problem depth effect to model (measured ratio 0.93-1.06).
   The apparent rise is selection across problems, which a per-problem cost head already absorbs.

**3. Stack the free post-decoding signal — the largest cost-side gain we have.** Under
`scout_first` the scout's generation is already bought, so its **realized length and outcome are
free at decision time**. Fitted as a separate view and blended on the calibration split:

| | activation only | free scout signal | **stacked** |
|---|---|---|---|
| TACO oss20 | +0.005 | +0.106 | **+0.187** |
| TACO oss120 | +0.238 | +0.103 | **+0.306** |
| LCB oss20 | +0.370 | +0.181 | **+0.392** |
| LCB oss120 | +0.492 | +0.271 | **+0.522** |

Blend weights are large on both views (0.35-0.85 activation, 0.34-0.74 free), so they are
complementary rather than redundant, and the gain is biggest exactly where the activation head
fails worst — **TACO oss20, 0.005 → 0.187, the cell responsible for that dataset's frontier
collapse.**

*Two caveats.* The features exist only after a scout draw, so they are unavailable at the root
under `free_start`; and they make the cost estimate state-dependent, which the replay does not yet
model. **Correction to an earlier reading in this section:** we first measured this as worth only
+0.01-0.03 and discarded it. That was an artifact of putting 2 features into a ridge beside 40,960
under one shared penalty, where they are crushed. Fit the views separately and stack.

*Dead end, recorded so it is not retried.* Prompt length alone: $R^2$ 0.007-0.045 on TACO,
negative on LCB scout.

**6.9g An oracle cost head: how much is on the table, and what no cost head can fix.**

Blending the per-route constant with perfect per-problem foresight,
$c_\lambda(x)=(1-\lambda)\bar c+\lambda\,c_{\text{true}}(x)$, converts "is our cost head the
problem?" into a measurable requirement. Hull frontier against pure RoR, seed 0, calibrated
beliefs, fitted decays. *Diagnostic only — the oracle end reads test outcomes.*

| cost head | TACO 35/45/50/55/60% | LCB 50/65/75/80/84% |
|---|---|---|
| constant (what RoR spends) | +34.1/+6.2/+8.2/+2.1/-5.5 | +41.3/+15.5/+9.2/-0.3/+3.8 |
| 25% oracle | +45.9/+13.6/+13.9/+6.4/+0.6 | — |
| 50% oracle | +52.4/+16.1/+15.3/+5.1/-9.4 | — |
| **oracle** | **+53.5/+19.3/+22.3/+3.8/-7.3** | **+76.2/+53.2/+42.8/+28.4/+9.6** |
| *our fitted head* | *+27.5/+9.8/+10.3/-5.6/-21.8* | *+46.9/+15.0/+9.1/+7.6/+10.8* |

**1. The idea is strongly validated on both datasets.** Perfect cost nearly doubles the
LiveCodeBench advantage (+41.3 to +76.2 at the 50% target, -0.3 to +28.4 at 80%) and adds 13-14
points on TACO at 45-50%. Query-conditioned cost is not a marginal trick; we are capturing a small
fraction of what it is worth.

**2. Our TACO head is net-negative, not merely weak.** It sits *below the constant* at four of
five targets while the oracle sits far above. We are injecting noise, and the shrinkage that
exists to prevent exactly that was fitted in log space (§6.9-cost). With the guardrail working,
the floor is the constant row — which is positive at four of five TACO targets.

**3. The near-ceiling failure is not a cost problem.** Across the sweep the 60% target reads
-5.5 / +0.6 / -9.4 / -14.0 / -7.3%: **non-monotonic, scattering +-7% with no trend**, on absolute
costs of $0.061-0.070 against a fixed $0.06138 baseline. Cost-head quality has *no systematic
effect* there, against a clear monotone trend at 50% (+8.2 -> +22.3). No cost head, however
perfect, rescues that regime; it is the headroom effect (§6.10a), now cleanly separated from
estimator quality. *Do not read the individual near-ceiling cells — they are single-seed and the
hull there is set by a handful of extreme operating points.*

**Research direction with a number attached.** The gap between our head and the oracle is
**+26pt** on LiveCodeBench at 50% and **+12pt** on TACO at 50%. That is a concrete target for
future cost estimators, stated in the units the policy spends rather than in log-space $R^2$.

*A natural alternative was tested and lost.* Under `scout_first` the scout's realized generation
length is free at decision time, and predicting other routes' lengths from it is the obvious
baseline. The prompt activation beats it (r = 0.563 vs 0.455 to oss20 length on TACO; 0.796 vs
0.626 on LCB) and adding the realized length on top of the activation buys only +0.01-0.03 $R^2$.
Worth reporting: **activations before generation beat observed generation length**, and cost
nothing extra.

**6.9f What the activation prior is actually for: it decides WHETHER to play, not WHICH arm.**

Decomposing against `counts_value` (RoR's beliefs inside our rule, so abstention is available to
both sides), fair fitted decays, hull frontier, 3 seeds:

| | LCB 50/65/75/80/84% | TACO 35/45/50/55/60% |
|---|---|---|
| **beliefs only** | **+43.7/+14.5/+0.9/+5.2/+5.9** | +32.5/+2.1/-19.1/+4.0/-19.6 |
| cost only | +44.2/+22.0/+6.7/+6.3/+10.0 | +12.1/-10.6/-25.3/-3.3/-14.0 |
| both | +46.9/+15.0/+9.1/+7.6/+10.8 | +27.4/-9.9/-24.5/-5.1/-23.0 |

But in the **budget-swept family, where the give-up action is disabled**, the same beliefs are
worth essentially nothing on *either* dataset:

| | LCB 50/65/75/80% | TACO 35/45/50/55/60% |
|---|---|---|
| beliefs only, no abstention | -3.4/-2.4/-4.4/+2.4 | -1.3/-12.8/+1.0/-1.4/-4.0 |

**So the activation prior's entire value flows through abstention.** With the give-up action it is
worth up to +43.7%; without it, zero. That is not a weakness, it is the mechanism, and it is what
we should claim: a prompt-only probe answers *"is this problem worth attempting at all"* — a
per-problem question — and adds nothing to *"which route should I try"*, which a per-route constant
already answers well. **C1 and C2 are therefore not separable contributions**: the prior is the
information and abstention is the channel through which it pays.

*Consequence for the paper.* Do not report "activation beliefs improve routing"; the measurement
does not support it. Report that they improve **selective prediction under a cost budget**, and
show both rows above so the reader sees why.

**6.9d-bis Fitting the decay for BOTH methods costs us more than it gains. Report this.**

$\sigma{=}2$ is unfitted for *everyone*, so fitting only our $\kappa$ would rig the comparison.
Giving each method its own maximum-likelihood constant from the calibration split (RoR 0.40 / ours
0.66 on TACO; 0.71 / 0.95 on LCB), 96 points, 3 seeds:

| setting | LCB utility | TACO utility |
|---|---|---|
| both at the inherited $\sigma{=}2$ | 84/96 | 57/96 |
| **only ours fitted** (rigged) | 84/96 | **67/96** |
| **both fitted** (fair) | **79/96** | **46/96** |

**The baseline gains more from fitting than we do.** On TACO our advantage goes from 57/96 to
46/96 — from a modest win to roughly a tie — and on LCB from 84 to 79. So a real part of what
looked like our contribution was **our machinery tuned against the baseline's inherited constant**.
Under the fair setting the LiveCodeBench *frontier* actually strengthens
(+46.9/+18.6/+19.2/+10.9/+8.0 at 50/60/70/80/84%), so the effect is not uniform, but the
utility-level claim on TACO does not survive.

*This is the single most important methodological point in the paper's evaluation.* An unfitted
hyperparameter in the baseline is a silent advantage for whoever tunes theirs, and $\sigma{=}2$
propagated from RoR into every comparison in this literature. **Any router paper reusing RoR's
pseudo-count should fit it for both arms before claiming a win.**

**6.9e The frontier rewards miscalibration, demonstrated by intervention.** Raising $\sigma$ for
our arms alone makes our beliefs deliberately worse — more optimistic after failures — and the two
metrics move in opposite directions, monotonically (TACO, calibrated beliefs, seed 0):

| $\sigma$ | frontier @ 60% | utility (swept $R$ won) |
|---|---|---|
| 2 | -31.6% | **57/96** |
| 5 | -21.9% | 41/96 |
| 10 | -21.8% | 25/96 |
| 20 | -21.1% | 17/96 |
| 50 | -21.3% | **14/96** |

Same probe, same costs, one constant. Degrading the beliefs **improves the near-ceiling frontier
by 10 points and destroys utility**, 57/96 to 14/96. This is an intervention, not a correlation,
and it is the paper's strongest evidence that **cost-at-matched-accuracy near the pool ceiling
prefers the policy with worse beliefs** — because reaching a near-ceiling target requires grinding,
and only a policy that wrongly believes the next draw might work will grind. Report utility at
matched $R$ as primary (§6.3d) and disclose this; do not tune to the frontier.

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

**6.10a TACO is not a strict improvement, and the reason sharpens the claim.**
*(All TACO numbers are still the 32k pool; a 64k re-collection is running, and on LCB that
re-collection removed the entire negative tail — treat this section as provisional.)*

Hull frontier, 3 seeds, against RoR:

| target | % of pool ceiling | advantage |
|---|---|---|
| 30% | 50% | **+46.9% ± 2.6** |
| 40% | 67% | **+37.5% ± 2.5** |
| 45% | 75% | **+30.1% ± 3.1** |
| 50% | 83% | **+14.5% ± 4.0** |
| 55% | 92% | **-8.4% ± 2.0** |
| 60% | 100% | **-49.1% ± 19.5** |

Over 401 levels from 28% to 61.5%, the mean advantage is negative at **95** of them — unlike LCB,
where it is negative at zero. **The crossover is at ~87% of the pool ceiling.**

**The mechanism inverts the intuition that a larger unsolvable set should favour abstention.**
TACO's pool solves only **60.0%** of problems against LiveCodeBench's **91.7%**, so the 55-60%
targets sit at 92-100% of everything the pool can do. At the 60% target:

| | cost | accuracy | abstention | attempts |
|---|---|---|---|---|
| RoR | $0.0621 | 60.48% | **39.52%** | 6.86 |
| ours | $0.0865 | 60.00% | **0.00%** | 7.71 |

**Correction — the arm labelled "RoR" here is not RoR.** It is `counts_value`: RoR's count
beliefs inside *our* utility rule, which gets abstention free as the zero-value action. The
RoR-faithful arm (`counts`, budget-swept, no give-up) does not abstain at all. Meanwhile our
family had fallen back to `content_decay_qcost`, which also does not abstain. So the row above
compares *our rule with their beliefs* against *our beliefs without our rule*, and the earlier
reading of it — that count-decay "quits natively" — was wrong. Attribution needs matched arms.

**Matched-arm decomposition against the RoR-faithful `counts` arm** (hull, same arm type on
both sides, so only the belief/cost source differs):

| target | % of pool ceiling | rule only | + activation beliefs | full method |
|---|---|---|---|---|
| 40% | 67% | +8.7 ± 1 | **+39.4 ± 3** | **+42.5 ± 2** |
| 45% | 75% | +8.9 ± 1 | +26.2 ± 3 | +21.5 ± 3 |
| 50% | 83% | +12.1 ± 1 | +4.2 ± 2 | -24.6 ± 7 |
| 55% | 92% | +6.6 ± 0 | -61.3 ± 5 | -81.8 ± 1 |
| 60% | 100% | +4.2 ± 0 | **-104.5 ± 11** | -96.5 ± 7 |

**The rule is steadily positive and the beliefs are what swing.** `counts_value` — RoR beliefs in
our utility rule — beats RoR-faithful by +4 to +12% at *every* TACO target, and by +1.6 to +8.1%
at every LCB target. Everything volatile lives in the activation prior: +39% where there is
headroom, -105% at the ceiling.

**So state the scope as headroom, and attribute it to the prior rather than to abstention.**
Near the pool ceiling the policy must attempt nearly everything, so a *prompt-only* prior is
actively harmful in both directions: called hopeless, a solvable problem is skipped and cannot be
recovered; called promising, an unsolvable one absorbs attempts. Count-decay conditions only on
**observed** failures, so it carries no prior error into that regime. LCB's targets sit at 55-87%
of its 91.7% pool ceiling and we win throughout; TACO's 55-60% targets sit at 92-100% of its 60.0%
ceiling and we lose badly. The falsifiable rule for the paper: **the activation prior needs
roughly 15% headroom to the pool ceiling; below that, use counts.** This is the same
signal-to-noise boundary as C3, now measured on the belief head too.

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

**TACO's high-target loss: two mechanisms proposed, both disconfirmed.** Do not re-tell either
without new evidence.

1. *"Count-decay quits natively; our prior delays the quit."* Wrong on the premise — the
   RoR-faithful arm abstains 0.00% at all 17 operating points. The quitting arm was `counts_value`,
   which is our own rule (§6.3b-bis).
2. *"We abstain on problems we could have solved."* Measured and false. At the TACO 55% target,
   matched utility arms discard almost identical fractions of pool-solvable problems: **ours 6.8%
   of all episodes, RoR 6.9%**. Abstention quality is not the difference.
3. *"Our prior escalates to the 120B, and escalation is bad value on TACO"* (2.5 accuracy points
   per extra dollar against LiveCodeBench's 7.1). The route mix is real — at the 55% target we buy
   22% oss120 where RoR buys 0% — but the causal claim fails its own test: making oss120 *cheaper*
   should have flipped the loss and instead **made it worse** (-81.8% at the 50% target under
   `--prices oss120=0.834`, against -10.7% at list price), non-monotonically across the whole
   sweep. Escalation share is a symptom, not the cause.

**Status: the mechanism for TACO's high-target loss is unknown.** TACO is still the 32k pool,
whose 8.6% truncation rate is twice LiveCodeBench's, and on LiveCodeBench the 64k re-collection
removed that dataset's entire negative tail. Wait for the TACO re-collection before proposing a
fourth explanation.


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
