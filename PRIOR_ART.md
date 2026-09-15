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
