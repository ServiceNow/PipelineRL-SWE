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
prediction and execution."* ROI-Reasoning (2601.03822) formalises it as an Ordered Stochastic
Multiple-Choice Knapsack Problem; other work gives a **concave knapsack with formal optimality
guarantees** and an exact marginal-greedy procedure — strictly stronger than sort-and-fill.

*Consequence:* our §3b-xxvii greedy ranker **is this baseline**. It must be reported as a baseline,
including that it **beats our MDP on LiveCodeBench above 0.30x** (+3.8pt at 0.50x, +2.4pt at 1.00x).

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

## 7. RouterBench (2403.12031)

Their **Zero Router** is the non-decreasing convex hull of the individual LLMs — the correct
non-adaptive baseline, and what we now compare against. Metric is **AIQ**, mean quality over the
shared cost domain. Their own **KNN and MLP routers "generally do not significantly outperform the
Zero Router"**, winning on MMLU/Winogrande and losing on ARC-Challenge/MBPP. Our +3.91% weighted
lands in that regime; **do not claim to beat their routers** without their per-dataset AIQ values.
