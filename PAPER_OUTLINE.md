# Paper outline — living document

**Edited in place, not appended to.** The append-only history lives in `RESEARCH_LOG.md`.
Every number here is either measured and cited to a run, or marked TODO. Nothing enters by
recollection.

**Status:** LCB complete on clean labels, C1 demonstrated end to end. TACO collecting.
SWE-Smith not started.
**Last updated:** 2026-09-03

---

## 1. Working title

*Cheap Beliefs for Expensive Pools: Query-Conditioned Cost and Abstention in Sequential
Test-Time Model Selection*

## 2. The claim, in one paragraph

Sequential test-time model selection asks, at each step, whether to resample the model you
committed to, reroute to a different one, or stop. Existing systems condition the *correctness*
half of that decision on the query and leave the *cost* half a per-model constant estimated
from the training set. We show that per-problem cost is both highly variable (p90/p10 of 16-37x)
and highly predictable from a single cheap model's pre-generation activations (R2 0.65-0.83
against the constant), that conditioning it sharpens the stop decision, and that the resulting
policy beats the counting-based state of the art and a compiled fixed schedule on cost at
matched accuracy.

## 3. Contributions, honestly scoped

Each is annotated with what prior work already owns, because three claims have already been
retracted this project for insufficient checking.

**C1. Query-conditioned cost in the decision rule.** *(strongest novel piece; DEMONSTRATED)*
The utility rule `argmax_m (p_m(x)*R - c_m)` is conditioned only in the numerator by every
system we found. RoR (2607.08665) uses count-based costs; NVIDIA's prefill router (2603.20895)
explicitly uses "each model's median training output tokens as a verbosity proxy." We predict
`c_m(x)` from the scout's pre-generation activations and put it in the rule.
- *Prior art that must be cited, not claimed:* length IS linearly decodable from a prompt's last
  hidden state (2607.05316) and hidden states are already used for length prediction in serving
  (2602.11812). **Neither operationalises it in a routing or abstention decision** — 2607.05316
  explicitly names prompt-end early termination as an idea it does not pursue.
- *Ours:* cross-model (a 4B predicting a 120B's length), inside the rule, and **measured to
  help**: swapping the constant for c_m(x) and changing nothing else saves 12.9-45.9% on count
  beliefs (6/7 targets significant) and 15.4-26.0% on activation beliefs (5/7). Because it helps
  on count beliefs too, it is an orthogonal improvement any router can adopt.

**C2. Abstention and resampling depth over a routed pool.**
RoR has resample-vs-reroute but count beliefs and no stop action. 2603.20895 has activation
routing but is single-commit with no abstention. "Knowing When to Quit" (2604.18419, ICML 2026)
has abstention as an MDP action driven by an MLP probe on hidden states — but for a **single
model**, with no pool, no resampling, and no cost in the objective.
- *Ours:* the combination — a pool, depth, an analytic belief decay, and abstention falling out
  of the same utility rule rather than a separate thresholded classifier.

**C3. A methodological correction: cross-model activation comparisons need a fixed readout.**
2603.20895 reports that "foreign encoders consistently outperform a target model's own internal
states," using a last-token readout, and does not discuss chat-template comparability. We show
that on our pool this direction is **an artifact of readout position**: gpt-oss harmony ends its
prompt on the word `assistant` mid-header while Qwen ends on a newline after a completed one,
and switching to mean-pooling reverses the ordering (see §6.4).
- *Scope honestly:* our pool is 3 models on one benchmark; theirs is 11-20 across three. This is
  a caution and a proposed control, **not** a refutation of their result.

**C4. Empirical: the policy wins on clean labels.** The full method is cheaper than RoR at
**all seven** accuracy targets, +18.1% to +56.2%, every one significant (P >= 0.996), and beats
a compiled fixed schedule with three clear wins and no losses (§6.3).

**C5. Data-hygiene findings that are ours to own but are not contributions.** Appendix only.

## 4. Related work

### 4.1 Sequential / budgeted test-time model selection (closest)
- **RoR — "Resample or Reroute?"** (2607.08665, Chen). Formalises resample-vs-reroute as
  competing uses of one per-query budget; greedy marginal-correctness-per-cost and a UCB
  variant; 11-model open-weight pool, four benchmarks; verifier-gated ablation. **Our primary
  baseline.** No abstention; count-based beliefs; cost not query-conditioned.
- **How Much of the Routing Gap Is Real?** (2607.03436). Companion analysis: k>=30 draws, peer
  models, shows part of the router-oracle gap is single-draw label noise.
- **SeqRoute** (2605.25424). Binary weak/strong, single-use per query, no resampling, no
  abstention, budget global across a multi-turn session.
- **Cluster, Route, Escalate** (2606.27457); **UCCI** (2605.18796) calibrated uncertainty for
  cost-optimal cascade routing; **Dynamic Model Routing and Cascading survey** (2603.04445).

### 4.2 Activation-based prediction and routing
- **LLM Router: Rethinking Routing with Prefill Activations** (2603.20895, NVIDIA). Prefill
  activations, encoder-target decoupling so open-weight encoders predict closed-source targets,
  Fisher separability for layer choice, SharedTrunkNet. Single-commit, no abstention, per-model
  constant cost, last-token readout. **Closest to our signal; see C3.**
- **LLMs Encode Their Failures** (2602.09924). Pre-generation probes for a model's OWN success;
  pool routing needs a probe per candidate; probe forward passes not charged; no abstention.
- **Scrouting / SuperScout** (2608.04804). 7B searcher's hidden states feed an N-way fixer
  router on SWE-bench Pro. Hidden-state AUC 0.600; the no-router ablation ties the routed system
  ("the handoff carries the result and routing collapses to cost allocation"); single-shot.
- **Knowing When to Quit** (2604.18419, ICML 2026). Abstention as an MDP action, MLP probe on
  hidden states, value thresholding. Single model, no pool, no resampling, no cost.

### 4.3 Output-length prediction (the C1 foundation — cite, do not claim)
- **How Much is Left?** (2607.05316). Total response length linearly decodable from the prompt's
  last hidden state before any output. Same-model only; cross-*dataset* transfer tested,
  cross-*model* not; never operationalised in a decision.
- **Predicting LLM Output Length via Entropy-Guided Representations** (2602.11812, ICLR 2026).
  Entropy-guided token pooling + progressive per-step prediction, for serving throughput and RL
  sampling efficiency.
- Latency-aware routing with output-length predictors (2607.18253); learned per-query cost
  predictors over query embeddings (2604.03527).

### 4.4 Routing benchmarks and gap analyses
LLMRouterBench (2601.07206); The Routing Plateau (2606.07587); CodeRouterBench / ACRouter
(2606.22902); the 67-model Rasch analysis (2606.27288); zero-shot routing via a universal latent
space (2601.06220); routing collapse (2602.03478).

### 4.5 Agentic stopping and escalation
EET (2601.05777, ACL Findings 2026); AgentStop (2604.15075, ACM CAIS 2026); FailFast-RestartSmart
(2608.03222); Atropos (2604.15075); SWE-Router (2607.00053); AutoMix; COPE (2506.11578, TMLR).

## 5. Method

### 5.1 Setting
Pool of M routes, K draws each. At each step: resample route m, reroute, or stop. Myopic
utility `argmax_m (p_m(s)*R - c_m(s))`, abstention as the zero-value action (no separate
threshold). Belief decay `p_m(n) = theta_m(x) * s/(s+n)`.

### 5.2 The cheap probe
One prefill of a 4B scout. Ridge/logistic heads on frozen mean-pooled activations give
`theta_m(x)` and `c_m(x)` for every route, including routes whose weights we never touch.
Training: 551 problem-level labels, 0.16 CPU-seconds, ~40KB of weights.

### 5.3 Why this is deployable where per-candidate probing is not
Probing a model requires its weights; no API exposes hidden states. Per-candidate probing needs
every pool member resident. We need weights for one cheap model. *(Note: 2603.20895's
encoder-target decoupling makes the same argument — cite it rather than claim it.)*

## 6. Experiments and results

### 6.1 Setup
LiveCodeBench, 892 problems, temporal split 551/170/171, six draws per route, routes
{Qwen3-4B-Instruct-2507, gpt-oss-20b, gpt-oss-120b}, all served locally under vLLM at a 32768
cap. Costs from AWS-node token prices. `lcb_local_pool_1788407418`.

### 6.2 Cost is variable and predictable  *(C1)*
Between-problem share of length variance 84.8/62.3/90.1%; per-problem cost p90/p10 = 37x/16x/17x;
failed draws cost 5.3x/2.6x/3.0x more than solved. Probe vs the per-route constant: R2
0.65/0.80/0.83, MAE cut 45-65%. Beats rescaling the scout's *observed* length (R2 0.796 vs 0.588,
0.834 vs 0.629) and subsumes it (+0.002/+0.008 when combined) — while needing only a prefill.
Not collinear with predicted success: partial rho 0.46-0.75.

### 6.3 Frontier  *(C1, C4)*  — `lcb_replay_qcost_1788470886`
Problem-clustered paired bootstrap, 171 test problems, 5000 resamples.

**Cost conditioning in isolation** (same beliefs, same decay, only c_m changes):

| target | RoR beliefs: const -> qcost | activation beliefs: const -> qcost |
|---|---|---|
| 50% | +45.9% [+30.6, +62.1] | +19.7% [+1.5, +40.0] |
| 55% | +18.8% [+4.3, +33.8] | +15.4% [-0.5, +34.1] |
| 60% | +35.1% [+21.5, +48.8] | +26.0% [+1.4, +52.7] |
| 65% | +5.8% [-8.4, +20.9] | +15.9% [+2.0, +31.3] |
| 70% | +13.3% [+1.4, +26.8] | +16.7% [+1.8, +32.7] |
| 75% | +26.7% [+14.5, +40.3] | -5.8% [-19.2, +6.9] |
| 80% | +12.9% [+1.0, +25.4] | +0.0% [-11.3, +11.3] |

**Full method vs RoR**: +56.2 / +29.6 / +43.9 / +21.9 / +20.5 / +32.5 / +18.1%, all seven
significant, every P >= 0.996.

vs compiled fixed schedule (pre-qcost arm): three clear wins, two marginal, two ties, no losses.
Abstention 10.5% at the 70% operating point -- low because clean labels lifted the pool
(oss120 81.9% per draw), which is why the harder pools matter for the stopping story.

### 6.4 Readout control  *(C3)*
Four readouts from one forward pass. Last-token vs mean-pooled reverses the cross-model
ordering; gpt-oss-120b's own probe improves 0.7743 -> 0.8313 from pooling alone. Mean-pooled,
the scout probe is significantly worse per route (-0.061, -0.049) and tied on pool solvability
(-0.028 [-0.075, +0.015]).

### 6.5 Capacity ablation
Linear vs MLP on the same activations, 535 pooled test problems over rolling-origin folds:
linear wins pool solvability (P(MLP better)=0.029) and all three cost targets. "Linear
suffices" is measured, not assumed.

### 6.6 TODO
- [ ] TACO medium+hard (883 problems) — collecting
- [ ] SWE-Smith -> SWE-bench Verified cross-dataset transfer
- [x] Query-conditioned cost wired into the utility rule and measured — §6.3
- [ ] Per-candidate frontier (does the AUC deficit cost anything downstream?) — replays done,
      not yet analysed
- [ ] Re-run the fixed-schedule comparison against the qcost arm (currently pre-qcost)
- [ ] Both cost bases (self-hosted node vs API list) — cheap, not run
- [ ] 8B LoRA retrained on clean labels — optional
- [ ] Seeds/CV on the frontier

## 7. Limitations
1. One benchmark until TACO and SWE land.
2. n=171 test problems; the capacity question needed folds to resolve at all.
3. Outcomes collected under our own serving configuration; another provider's defaults could
   shift absolute solve rates. All comparisons are within-collection.
4. Cost basis is self-hosted node pricing while the deployment story is API-served experts —
   report both.
5. The readout correction (C3) is measured on a 3-model pool on one benchmark.

## 8. Retracted during this project — do not resurrect
- Cross-model superiority ("own activations are the worst predictor of own success") — readout
  artifact, §6.4.
- "Nobody predicts generation length from activations" — false (2607.05316, 2602.11812).
- "First to route from a small model's hidden states" — false (2608.04804, 2603.20895).
- Our own collection bugs framed as contributions — they are appendix material at most.
