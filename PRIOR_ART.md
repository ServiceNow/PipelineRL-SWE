# Prior art — the papers this project must position against

**Living document.** Every entry was checked by reading the paper, not from recollection. Four
separate times this project claimed novelty for something already published (§8 of
`PAPER_OUTLINE.md`); this file exists so that stops happening. **Before writing any novelty claim,
check it here first.**

Format: what they do / what they do NOT do / what that leaves us.

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

## 4b. ROI-Reasoning (2601.03822) — the knapsack formalisation, checked

*They do:* formalise budget allocation as an **Ordered Stochastic Multiple-Choice Knapsack**
(OS-MCKP): *"each problem $x_i$ corresponds to a class, while different actions (e.g., solving,
partially attempting, or abstaining) correspond to items within that class"*, with an explicit
ordering constraint because *"the reward and computational cost of an action are not known at
decision time"*. They **have an abstain action** — `\boxed{NA}`, trained by "Refusal Learning" for
"low-ROI problems when the expected cost outweighs the potential benefit".

*They do NOT:* allocate over a **pool of different models** (the knapsack is over reasoning-effort
levels for **one** model, e.g. Qwen2.5-1.5B-Instruct); use a **separate cheap model** (predictions
are the target's own `<predicted_level>` meta-cognitive tags); price **multiple models from one
forward pass**; or **resample** (one generation trajectory per problem).

*Leaves us:* **cross-model pricing — one cheap prefill valuing an entire heterogeneous pool** — and
resampling as part of the allocation. **Does NOT leave us:** the knapsack formulation, or
abstention-under-budget, both of which are theirs.

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
