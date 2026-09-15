# Prior art — the papers this project must position against

**Living document.** Every entry was checked by reading the paper, not from recollection. Four
separate times this project claimed novelty for something already published (§8 of
`PAPER_OUTLINE.md`); this file exists so that stops happening. **Before writing any novelty claim,
check it here first.**

Format: what they do / what they do NOT do / what that leaves us.

---

## 0. RoR (2607.08665) — the paper was REWRITTEN between v1 and v3. Cite the version.

Our description was **accurate for v1**. arXiv 2607.08665 has since been replaced by a
substantially different paper under a changed title, and anyone pulling the default (latest)
version will not find the method we benchmark against. **Always cite `arXiv:2607.08665v1`.**

| | **v1** (2026-07-09) — *what we reproduce* | **v3** (2026-09-02) — *what arXiv serves today* |
|---|---|---|
| title | "Resample or Reroute? **Budget-Aware Test-Time Model Selection for Large Language Models**" | "Resample or Reroute? **Recoverable Stopping Debt Without Identified Action Selection**" |
| pool | **eleven-model** open-weight pool | **two** primary (Qwen2.5-7B/14B) + 2 post-hoc |
| benchmarks | **four**, of differing difficulty | MBPP+ (152 queries), LCB, BigCodeBench |
| method | *"allocate each unit of budget between resampling and rerouting so that expected correctness is maximized"*; *"an online resample-or-reroute (RoR) allocation policy driven by estimated marginal correctness per unit cost"* | **no allocation framework**; disclaims the greedy rule as *"a transparent heuristic, not a proof of horizon-optimal control"* |
| claim | **positive** — favourable cost-quality Pareto front against single-route, one-commit-router, budget-aware best-of-K, cascade and random-allocation baselines | **negative** — *"current evidence does not identify when to resample rather than reroute"* |

**The author's own words (arXiv comments field), which settle what happened:**

> *"v3: substantive methodological reconstruction of v1-v2. The budget-allocation, learned-router,
> cost-oracle and broad Pareto claims are not retained; replaced by a three-gate identification
> framework with fresh MBPP+, LiveCodeBench and preregistered BigCodeBench evidence.
> v2: corrected Phi-4 cost (14.7B)"*

**Read this carefully — "not retained" is not "was wrong".** The author does not claim the v1
numbers were false, and nothing in v3 identifies an error in them. What changed is the *evidentiary
bar*: v3 adopts preregistered support gates, exact-fold exchangeable references and query-cluster
intervals, and the broad v1 claims do not survive **that** standard. v1 asked "does budget-aware
allocation beat naive baselines?"; v3 asks "can you identify, per instance, which of resample or
reroute is right?" **Both answers can be true at once** — a greedy allocation can beat naive
baselines on average while the per-instance action remains unidentified.

**And we have independent evidence the v1 policy does work, mildly.** Our own replay of it beats a
fixed `gpt-oss-120b` by **+8.1%** at that model's own accuracy (§3b-xxxvii anchor). So the
v1-style policy is not an artefact; it is a real but modest gain, which is exactly what v1 reported.

**The genuine risk this creates for us, and it is not a novelty risk.** The author abandoned
budget-allocation and Pareto claims *after raising his own evidentiary standard*. A reviewer who
reads v3 may hold **us** to that bar — preregistration, exchangeable references, explicit support
gates. Our strongest answer is the seed unanimity already in hand (5/5 at every LCB target, 3/3 on
TACO, 19/19 targets across four pools), and our weakest points are the single-split results. **Fix
the single-split results before submission; that is where this bites.**

**Who owns the sequential formulation — the decisive split, and it favours us.**

v1 claims the *framing* and a *greedy* rule, and is explicit about both:
*"no previous work treats resampling the committed model and rerouting to an alternative model as
competing uses of a single per-query cost budget"*, and the method is
*"an online resample-or-reroute (RoR) allocation policy driven by estimated marginal correctness per
unit cost"*. A greedy index on marginal value per dollar — which is exactly what our `counts` arm
implements.

**v3 then disclaims horizon-optimality outright:** *"The greedy rule is a transparent heuristic, not
a proof of horizon-optimal control."*

| component | whose |
|---|---|
| resample-vs-reroute as competing uses of one budget | **his (v1)** — cite it, it is the framing |
| greedy allocation by marginal correctness per unit cost | **his (v1)** — this is our `counts` arm |
| **backward induction over the failure-count lattice; horizon-optimal control** | **ours** — and v3 says in terms that he does *not* have it |
| **abstention as the zero action of the Lagrangian** | ours in this setting; the *idea* of abstention-under-budget is ROI-Reasoning's (§4b) |
| global budget solved in the dual | neither — textbook constrained-MDP / Neyman-Pearson |
| **per-problem beliefs from one cheap prefill** | **ours** |

So the answer to "was the MDP his?" is **no**. He has the *framing* and a *greedy heuristic on it*,
and explicitly says the heuristic is not horizon-optimal control. **The Bellman lattice is the gap
he names and does not fill.** And v3 fills it no better — it abandons allocation entirely for an
identification framework. That makes §3b-lv (the formalisation) and the horizon sweep a direct
contribution against a gap the prior author states himself, which is the strongest possible form of
this argument.

**A practical warning about reading this line of work.** All three versions are extremely dense and
use heavy bespoke terminology ("two-sided FIT action support", "recoverable stopping debt",
"exact-fold exchangeable reference"). This project has already been burned twice by paraphrase — the
eleven-model/four-benchmark description that turned out to be v1-specific, and an in-session claim
that the author had "retracted" v1 when the comments field says only "not retained". **Quote this
paper verbatim, always with a version number, and never summarise it from memory.**

**Also note what is *no longer* claimed by anyone:** the learned-router and cost-oracle results are
withdrawn too. That does not widen our *novelty* space — v1 remains published and citable prior art
regardless of later withdrawal — but it does mean **there is no standing positive result in the
literature for budget-aware resample-vs-reroute allocation.** Ours would be the first that survives.

**Unchanged across versions, and these are what our baseline relies on:** no give-up/abstain
action, count-based beliefs, per-model constant costs proxied by parameter count in billions.

**What this means for us — and it is favourable, not awkward.**

1. **Our `counts` arm is a faithful reproduction of v1's published policy.** The label "RoR as
   published" is correct *provided the version is stated*. Say so explicitly in the paper.
2. **The author has since weakened the positive claim to a negative one.** v3's headline is that
   the resample-vs-reroute choice is **not identified** by available evidence. Our contribution —
   a per-problem belief that makes the choice identifiable, and a give-up action that neither
   version has — is therefore a **direct answer to an open problem the original author now states
   outright**, rather than an incremental gain over a standing positive result. That is a stronger
   position for §1 of the paper.
3. **Do not cite v1's Pareto-front result as if it stands.** Cite v1 for the *formulation and the
   policy we compare against*; cite v3 for the *identification problem we address*.
4. Reviewers pulling the bare arXiv ID will see v3. **Anticipate it in the related-work text** in
   one sentence, or the mismatch will read as carelessness.

*Verified 2026-09-15 by fetching both v1 and v3 abstracts, plus the v3 HTML full text. A human read
of v1's method section is still required before submission to confirm the `counts` arm matches its
allocation rule in detail.*

---

## 0b. THE CLOSEST WORK TO OUR MDP IS NOT IN THE LLM LITERATURE — and we cite none of it

**Gap found 2026-09-15.** Every entry in this file is a 2024–2026 LLM paper. Our formulation
(§3b-lv) — a budget-constrained sequential search over noisy alternatives, relaxed by a Lagrange
multiplier, solved by backward induction, with the achievable set taken as a convex hull — is
**textbook decision theory**. A reviewer from that side will recognise it immediately, and finding
it uncited reads far worse than citing it and claiming less.

**Be honest about what this means: our mathematical contribution is close to zero.** The formulation
is known; what is ours is the *belief source* (one cheap prefill), the *cross-model* application, and
the empirical result. Position it that way deliberately rather than being corrected into it.

| classical work | why it is the closest | what it covers of ours |
|---|---|---|
| **Weitzman (1979), "Optimal Search for the Best Alternative", *Econometrica* 47(3):641–654** — *Pandora's box* ✅VERIFIED | $n$ alternatives, each with a known reward distribution and an **inspection cost**; open sequentially, stop when what you hold beats the reservation value of every unopened box | **This is routing-with-resampling.** A route is a box; a draw is an opening. Weitzman's reservation-value index is the ancestor of our $p_mR - c_m$ rule, and his optimality proof is the reason a myopic index can be right at all |
| **Altman (1999), *Constrained Markov Decision Processes*, Chapman & Hall/CRC** ✅VERIFIED | Lagrangian relaxation of a constrained MDP; the optimal constrained policy is a **randomised mixture of deterministic policies**, and the achievable set is a convex hull | **This is our entire frontier construction.** "Mixtures count, so compare hulls" is his theorem, not our idea. **Most important missing citation.** |
| **Gittins index; Whittle (1988) restless bandits** | decoupling a budget-coupled multi-armed problem by a common multiplier, giving a per-arm index | **This is the $R$ sweep.** Per-problem decoupling under a common price is the Whittle relaxation exactly |
| **Badanidiyuru, Kleinberg & Slivkins, "Bandits with Knapsacks"** | online decisions under hard budget constraints, with regret guarantees | the online version of our allocation; we solve offline in the dual, they solve online |
| **Golovin & Krause, adaptive submodularity** | adaptive stochastic optimisation under a budget, with greedy guarantees | when greedy *is* near-optimal here — relevant to RoR v1's greedy rule and to why our lookahead helps |
| **Gergatsouli et al., "Weitzman's Rule for Pandora's Box with Correlations", NeurIPS 2023** ✅VERIFIED | Pandora's box where box values are **correlated** | **exactly our cross-route $\rho$ problem** (§3b-xliv item 5): independence over-values "try another route", and this literature has the machinery |
| Wald, sequential analysis / SPRT | optimal stopping of a sampling process | the stopping half of the rule |

**Where this leaves the contribution claim, stated honestly:**

1. **Not ours:** the Lagrangian relaxation, the convex-hull frontier, the index-policy form, backward
   induction. All classical. Cite Altman and Weitzman explicitly in §3.
2. **Not ours:** resample-vs-reroute as competing uses of one budget (RoR v1, §0).
3. **Ours:** per-problem beliefs from one cheap prefill of a model that never generates; the
   cross-model cost head; and the empirical finding that this is what makes the sequential rule
   work (19/19, §3b-xlvi) while count-based beliefs do not.
4. **Ours, more weakly:** applying horizon-optimal backward induction here where the prior work used
   a greedy index and explicitly disclaimed optimality — but note this is *applying* known
   machinery, not inventing it.

**Verified 2026-09-15.** Weitzman: *Econometrica* 47(3):641–654, 1979 — confirmed, and the survey
literature notes his index rule *"was later recognized to be a special case of Gittins's optimal
algorithm for Bayesian bandits"*, so **Weitzman and Gittins are one lineage, not two**; cite them
together. Altman: Chapman & Hall/CRC 1999, ISBN 9780849303821 — confirmed, and the load-bearing
claim is verified verbatim: **"deterministic policies are not optimal for constrained MDPs"**, with
optimal *mixed* stationary-deterministic policies obtained through a "mixing policy" that samples a
policy and then commits to it. **That is our convex-hull frontier, as a theorem from 1999.**
Correlated variant: Gergatsouli et al., *"Weitzman's Rule for Pandora's Box with Correlations"*,
NeurIPS 2023 — confirmed.

*Still unverified and lower priority:* Badanidiyuru/Kleinberg/Slivkins (Bandits with Knapsacks) and
Golovin & Krause (adaptive submodularity). Check before citing.

## 0c. "Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades" (2605.06350)

Dylan Bouchard, **7 May 2026 — two months BEFORE RoR v1**. Found 2026-09-15 and **not previously in
this file, though it is a closer antecedent to §3b-lv than anything that was.**

*They do:* a decision-theoretic treatment of cascades grounded in *"constrained optimization and
duality"*, with *"budget- and quality-constrained formulations"*, cost-quality frontiers over model
pools, and multi-stage $k$-model cascades with switching points. The decision rule is Lagrangian:
**"a single shadow price equalizes marginal quality-per-cost across stage boundaries."** Validated
across five benchmarks and eight models.

*They do NOT:* **resample** the same model (deterministic threshold cascades that escalate to a
*different* model only); have an **abstain/give-up** action; or solve an **MDP** — it is Lagrangian
duality over threshold cascades, not backward induction over a state lattice.

**What this costs us, and it is worth knowing now rather than in review.** The **shadow-price
framing for LLM routing is Bouchard's**, published before Chen and before us. §3b-lv derives R as
the dual of a global budget as though that were setup rather than prior art; it is prior art. **Cite
it in §3 alongside Altman.** Between Bouchard (duality for cascades), Chen v1 (resample-vs-reroute
under one budget) and Altman/Weitzman (the machinery), **we own none of the formulation.**

**What it leaves us, which is the same list the rest of this file converges on:**

| | Bouchard | Chen v1 | Chen v3 | ours |
|---|---|---|---|---|
| shadow price / duality | **yes** | yes | dropped | yes |
| resample the same model | no | **yes** | yes | yes |
| abstain / give up | no | no | no | **yes** |
| MDP, backward induction, horizon-optimal | no | no (greedy; v3 disclaims) | no | **yes** |
| per-problem beliefs from a non-generating probe | no (confidence) | no (counts) | no | **yes** |
| cross-model per-query cost prediction | no | no (parameter count) | no | **yes** |

**The strategic read.** This area is crowded and moving fast — Bouchard May, Chen July, ModelSwitch
and several adaptive-self-consistency papers in the same window. Nobody owns the setting, and the
formulation was never going to be our differentiator. **The belief source is, and that is exactly
what §3b-xlvi measures (19/19) and §3b-xlix isolates against length and TF-IDF.** Lead with it.

*Verification note: read via automated extraction of the abstract page. The shadow-price quote
should be confirmed against the PDF before citing, and the paper read properly for whether any
resampling variant appears in the body.*

## 0d. Rejecting a verifier's ACCEPTANCE — explored, and we must position against it

Checked 2026-09-15 after asking whether the policy should be allowed to distrust a positive
verification signal. **It is not virgin territory.**

**What RoR does, verbatim from v1's method section:** *"The main experiment assumes a reliable
verifier: a policy stops as soon as a drawn answer is verified correct (early stopping)."* So v1
**cannot** reject an acceptance. It handles imperfection two other ways: a **parametric degradation**
(*"with quality q, final selection succeeds with probability q·1{any drawn sample correct}"* — a
**false-reject** model applied at final selection, not false accepts mid-episode), and a real
agreement/base-test verifier whose *"measured base-verifier false-accept rate is only **1.0%**"*, at
which *"RoR under this real verifier nearly matches its perfect-verifier ceiling (0.897 vs 0.897)."*

**⚠ Our weak verifier is 11x more adversarial than theirs** — 11.01% false accepts against 1.0%. At
1% the oracle assumption is nearly free; at 11% our ceiling collapses 84.8% → 62.1% (§3b-lx). **We
are not testing the same regime and must say so**, rather than presenting our collapse as a
contradiction of their result.

**Prior work that DOES model distrusting an acceptance:**

| work | what it does | what it does not |
|---|---|---|
| **"Belief-Guided Inference Control for LLM Services via Verifiable Observations"** (2604.27536, Yuan, Lin, Chen, Xu, Yang, Ngai; 30 Apr 2026) | a **POMDP** whose latent state is *response reliability*, verifiable observations aggregated into a **belief state**, explicitly a *"budgeted sequential decision problem"* deciding *"whether the default low-cost response is sufficiently reliable or whether additional computation should be allocated"* | binary default-vs-escalate; **no resampling** of the same model; **no abstain action**; no per-problem prefill belief |
| **AutoMix** (self-verification + a **meta-verifier**, POMDP framing) | the meta-verifier decides whether to *trust* the self-verification — literally rejecting an acceptance | binary routing; no budget sweep; no abstain ⚠ *recalled, NOT verified — check before citing* |

**Consequence for us, stated plainly.** The POMDP-with-noisy-verifier framing is **taken** (April 2026,
two months before RoR v1). Our optional-accept arm is not a new formulation; it is that formulation
applied to a **multi-route pool with resampling and an abstain action**, with $q_m = P(\text{correct}
\mid \text{weak PASS on route } m)$ measured per route (0.725 / 0.939 / 0.951 — a scout acceptance is
far less trustworthy than a gpt-oss-120b one).

**What remains ours in this regime, if the arm works:** resampling *and* rerouting *and* abstaining
under a noisy verifier, with the accept/continue decision driven by a per-problem prefill belief.
None of the three above has that combination. **But claim the combination, not the idea.**

## 1. Prefill-activation router — "LLM Router: Rethinking Routing with Prefill Activations" (2603.20895)

**The closest work, and the one we have overclaimed against twice.**

*They do:*
- Route from **prefill activations**, before generation.
- **Encoder-Target Decoupling**: one model's hidden states predict a *different* model's success.
  Verbatim: *"open-weight encoders can serve as strong predictors of closed-source target
  performance, and in several cases, hidden states of a different model outperform the target
  model's own hidden states."* Attributed to encoder representational geometry (effective
  dimensionality, isotropy, Fisher separability).
- **Transfer to closed-source targets** whose weights are unavailable (Claude Opus, GPT-5, Gemini).
- **SharedTrunkNet**: PCA-reduced features from all K targets concatenated, one multi-output MLP
  predicting all candidates jointly.
- Evaluate on LiveCodeBench among others.

*They do NOT:*
- Report any **label-efficiency / few-shot** result for adding a new target. Verbatim: *"All
  experiments train jointly on the full datasets; there is no analysis of how many labeled examples
  are needed to extend the router to a novel target."*
- Have a **give-up / abstention** action — routing is single-commit, `argmax_k s_k,q`.
- **Resample or reroute** sequentially. Verbatim: *"We instead commit to one model before
  generation, and ask whether richer signals can raise that decision's accuracy without
  multi-stage fallback."*
- Predict **cost per query**. They use *"each model's median training output tokens serving as a
  verbosity proxy"*, because *"output length is unavailable before generation."*

*Leaves us:* label efficiency for pool extension (their explicit gap, and our 2-parameter response
curve on a shared latent is the architecture that makes it askable — their joint MLP is not);
abstention; the sequential rule; per-query cross-model cost prediction.
**Does NOT leave us:** cross-model prediction itself, or transfer to weights we never touch.

---

## 2. "The LLM Already Knows: Estimating LLM-Perceived Question Difficulty via Hidden
Representations" (2509.12886)

*They do:* estimate difficulty from **the target LLM's own initial hidden state, with no generated
tokens**; model token generation as a Markov chain with a value function. Spend it on adaptive
Self-Consistency / Best-of-N / Self-Refine for inference efficiency.

*They do NOT:* cross-model transfer; abstention; multi-model pools; cost prediction.

*Leaves us:* everything cross-model and everything budget-constrained. **Does NOT leave us:**
"difficulty is readable from a prefill hidden state before generation" — that is theirs.

---

## 3. "Predictive Scheduling for Efficient Inference-Time Reasoning" (2602.01237)

**The nearest prior work for the greedy-knapsack allocator, and we should compare against it.**

*They do:* an **MLP on intermediate hidden states** estimates each query's difficulty / optimal
reasoning length **before any generation**; a **greedy algorithm allocates a fixed total token
budget across problems in a batch**. GSM8K: **+7.9pp over uniform budgeting at identical token
cost, closing >50% of the oracle gap.**

*They do NOT:* use a *different* model's states (hidden states come from the target,
DeepSeek-R1-Distill-Qwen-1.5B); route across a **pool** of different models (one model, varying
budget); abstain; predict dollar cost.

*Leaves us:* cross-model, multi-model pool, abstention, cost. **Does NOT leave us:** "activations
-> difficulty -> greedy budget allocation across problems" as a pipeline.

---

## 4. Greedy Knapsack / predict-then-optimize allocation

A **named baseline**, not a contribution. *"The Greedy Knapsack baseline uses predicted difficulty
signals to guide a post-hoc greedy selection of problems under a fixed token budget, decoupling
prediction and execution."* Other work gives a **concave knapsack with formal optimality
guarantees** and an exact marginal-greedy procedure — strictly stronger than sort-and-fill.

**RETRACTED (2026-09-10): "it beats our MDP above 0.30x".** That claim rested on a greedy arm that
(i) picked its route **best-of-3 on the test split**, (ii) was scored as a **single point against
the MDP's convex hull**, so mixtures were available to us and not to it, and (iii) drew to
exhaustion on each problem rather than at a planned depth — so it was not the named baseline at
all. **The correctly-specified version is the one-shot knapsack of §3b-xiv** (each problem gets one
`(model, depth)` plan, value $1-(1-\theta)^n$, cost $n\hat c$, common multiplier), **and we beat it
by +1.1 to +20.2pt.**

## 4b. ROI-Reasoning (2601.03822) — RE-READ 2026-09-14; the earlier entry over-conceded

*"ROI-Reasoning: Rational Optimization for Inference via Pre-Computation Meta-Cognition"*, Zhao, Qi,
Sun. Re-read from the arXiv HTML because this project had been conceding "abstention-under-budget"
to it wholesale, and that concession was load-bearing for our own attribution claims.

*They do:*
- Budget **multiple tasks under a strict GLOBAL token constraint**, named as an Ordered Stochastic
  Multiple-Choice Knapsack (OS-MCKP). Confirmed global, not per-query.
- **Solve-or-skip.** The skip is a **trained output token**: *"the model learns to output a
  standardized `\boxed{NA}` token as its final answer"* when *"expected cost outweighs potential
  benefit"*, taught by Refusal Learning — if no sample is correct the target label becomes
  `<predicted_level>Level-3</predicted_level> \boxed{NA}`.
- **Two stages of fine-tuning on the target model**: Meta-Cognitive Fine-Tuning, then
  Rationality-Aware RL (Dr. GRPO) for *"sequential decision making under a hard token budget"*.
- Difficulty as a **4-level ordinal tag**, and it is a proxy for **cost, not success**:
  *"The discrete level k serves as a coarse proxy for expected computational cost... Level-0
  denotes short solutions (e.g., within 256 tokens), Level-1 corresponds to 256-512 tokens."*

*They do NOT — and three of these were wrong or missing in the previous entry:*
- **Solve a knapsack.** They name OS-MCKP and then optimise a policy by RL; the group-relative
  advantage acts as an implicit shadow price. **"Greedy Knapsack" (predict-then-optimize) is one of
  THEIR baselines**, not their method — so the knapsack solve is prior art they position against.
- **Predict a success probability, or a continuous cost.** *"No probability output; ROI is implicit
  in the learned policy, not explicitly predicted."* There is no $p$, no $c$ in continuous units,
  and no explicit price — so the rule $\arg\max_m(p_mR-c_m)$ with a zero-crossing give-up **does not
  appear in this paper**.
- Route among **multiple models** (one base model, e.g. Qwen2.5-1.5B-Instruct; effort/length only),
  **resample** at inference (single pass, *"generation is terminated once the token limit is
  reached"*), or work **without training the target** (two fine-tuning stages are required).

*Corrected attribution — the previous entry said "the knapsack formulation, or abstention-under-
budget, both of which are theirs", and that is too generous:*
- **Theirs, cite it:** the *goal* — budgeted inference over many tasks under a global constraint
  where skipping is an available action — and solve-or-skip as a trained model behaviour.
- **Neither of ours:** the Lagrangian relaxation itself. $\max(0,\max_m(p_mR-c_m))$ is the textbook
  dual of a constrained allocation (the same convexification `hull_frontier.py` already cites for
  randomised tests in Neyman-Pearson). Do not concede it to them; do not claim it.
- **Left to us, and now wider than we thought:** explicit calibrated per-problem success *and* cost
  prediction driving the decision (they have neither); cross-model pricing over a heterogeneous
  pool from one cheap prefill; resampling as part of the allocation; and **requiring no training of
  any pool model** against their two mandatory fine-tuning stages.

---

## 5. IRT for LLM evaluation

- **JE-IRT (2509.22888)** — joint embedding of models and questions; question-embedding norm *is*
  difficulty; generalises to new models and new benchmarks.
- **Contextual multidimensional IRT (2608.22295)** — predicts performance on **unseen questions**
  from question content, latent capability profiles per model.
- **IrtNet (2510.00844)** — sentence embeddings, difficulty + discrimination, used for routing.

*Leaves us:* nothing about the *existence* of a shared difficulty scalar — that is 70-year-old
psychometrics with an active LLM line. Our angle is where the scalar is read from (a cheap prefill),
what it is spent on (abstention under a budget), and when it fails (§3b-xxi).

---

## 6. Own-model response-length prediction for serving

**EGTP (ICLR 2026)**, **TRAIL**, Piotrowski et al. — predict a model's **own** output length from
its **own** activations, for KV-cache reservation and SRPT scheduling (1.66-2.01x mean latency).

*Leaves us:* **cross-model** cost prediction — one cheap prefill pricing *other* models inside a
budget-constrained decision. That specific combination remains unclaimed.

---

## 6b. Ensemble deferral — "Semantic Agreement Enables Efficient Open-Ended LLM Cascades" (2509.21837)

*They do:* use *"n lightweight ensemble models"* (heterogeneous 8B/3B/1B mixes) that **generate
responses**, defer to a larger target on **semantic disagreement**, and report matching target
quality at **40% of cost with 60% lower latency**.

*They do NOT:* use internals. The signal is *"mean pairwise similarity between $y^{(i)}$ and all
other $y^{(j)}$"* over **complete responses**, explicitly text-only, working with *"black-box APIs"*
and requiring *"no access to internals"*.

*Leaves us:* the prefill version — n forward passes, **zero generated tokens**. Our measurement
says that gap is real: one scout prefill (AUC 0.768 LCB / 0.819 TACO) **beats six generations from
that same model** (0.789 at 47x the cost on LCB; never reached on TACO). A generation is one
Bernoulli($\theta$) sample; a prefill is a continuous read on $\theta$.

**Note, however:** we tested cross-family prefill ensembling on our own pool and it **gains
nothing** — best single encoder 0.859, every 2- and 3-encoder combination 0.851-0.858 under
concatenation, mean, rank and stacked blending. So the committee idea is theirs *and* does not
appear to transfer to prefill. See also §1: the prefill-router found **larger** encoders strictly
better, which is the same conclusion from the other direction.

## 7. RouterBench (2403.12031)

Their **Zero Router** is the non-decreasing convex hull of the individual LLMs — the correct
non-adaptive baseline, and what we now compare against. Metric is **AIQ**, mean quality over the
shared cost domain. Their own **KNN and MLP routers "generally do not significantly outperform the
Zero Router"**, winning on MMLU/Winogrande and losing on ARC-Challenge/MBPP. Our +3.91% weighted
lands in that regime; **do not claim to beat their routers** without their per-dataset AIQ values.
