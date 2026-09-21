# Paper outline — living document

**Edited in place.** Append-only history lives in `RESEARCH_LOG.md`. In-flight work is tracked in
`THREADS_IN_PROGRESS.md`. Every number is cited to a run or marked TODO; nothing enters by
recollection.

**Status (2026-09-08).** The probe **clears the difficulty-prediction baselines** the literature
demands, under matched pipelines: +0.095 / +0.139 AUC over TF-IDF on solvability and a win at every
one of six routes on cost (§3b-xx). §6.9k's scope law is **falsified** (§3b-xix): abstention is
worth *most* on the pool with the *least* unsolvable mass, because the mechanism is knapsack
reallocation against a binding budget, not declining hopeless problems — so RouterBench flips from
falsification target to expected strength. **Blocking:** rewrite §6.9k; re-check every "TACO is
interesting because it has unsolvable mass" claim; and check why a plain `RidgeCV` on activations
beats our committed calibrated probe at pool-solvability on both benchmarks (§3b-xx).

LiveCodeBench and TACO complete on fully re-collected 64k pools, with calibrated
belief heads and a dollar-space cost head. **The belief head is a strict improvement over RoR at
every accuracy level on both benchmarks** (floors +2.90% / +0.93%), isolated the clean way — RoR's
constant costs on both sides, so only the belief source differs. Added 2026-09-08: the belief head
and the cost head are **substitutes, not complements** — each alone buys ~+25pt at a 0.25x budget,
and stacking them buys nothing (§3b-xii). Both are readouts of one prefill, and which one a pool
supports is predictable in advance from cost R2. A five-model API peer pool is
collected for the cross-model transfer result. Seven downstream applications were tested; six
failed for one identifiable reason (§7b), and that reason is now the paper's second contribution.
**Target: TMLR** — the result is a careful, heavily-ablated characterisation of what a cheap
prompt-only probe can and cannot do, which is the kind of thoroughness TMLR rewards and which a
venue optimising for novelty would push us to overclaim.
**Last updated:** 2026-09-07

---

## 1. Working title
*Whether, Not Which: Cross-Model Selective Prediction from One Cheap Prefill*
(a 2026-09-08 retitle was reverted: it rested on a rigged TF-IDF comparison, §3b-xx CORRECTION)

## 2. One paragraph
Given a pool of language models and a cost budget, a policy must decide at each step whether to
spend another draw, on which model, or to give up. A single scalar per problem — a **shared, model-independent difficulty
estimate**, read from one cheap model's prefill before any generation — is enough to drive all
three decisions, and we characterise exactly what it can and cannot do. It supports the
**whether** decision or the **which** decision, and *which of the two* is set by a measurable
property of the pool. On pools with heavy unsolvable mass the model x problem interaction is
unlearnable and the shared scalar carries everything: six of seven applications needing the
interaction fail. On a 91.4%-contested pool the decomposition **inverts** -- the shared scalar is
worse than not routing at all (-0.60% AIQ) while the interaction is worth +4.52%. Used for *whether*, it closes **65-73%** of the oracle gap in selective
prediction on three benchmarks including out-of-domain SWE-bench Verified, and buys **+26pt of
accuracy at a quarter of the budget** of calling the largest model on everything. We show the gain
is a **knapsack over problems** — order by difficulty, and the budget decides where the line goes —
and we introduce the control that separates real information from the decision rule's mere response
to per-problem dispersion, which no routing paper we surveyed runs. Finally we show the scalar can
be read from one cheap model's prefill and transferred to models whose weights are never available
at ~25 labels each, and we benchmark it against the deployable text baselines the difficulty-prediction literature
implies, which it clears on every cell under matched pipelines.

## 3. Contributions

Ordered by how well they replicate. Several claims were retracted during this project (§8), and
every contribution is annotated with prior art and measured regime.

**C0 (framing). The object is a shared difficulty scalar, and the paper is about what it is for.**
Difficulty estimation is established — IRT for LLM evaluation, and same-model prefill difficulty
(2509.12886) — so the *existence* of a per-problem difficulty scalar is not our claim. Ours are
(i) that one cheap model's prefill reads it **better than the deployable text baselines**
(§3b-xx), (ii) that it transfers **cross-model** including to weights we never touch (C3), and
(iii) the decision theory and validation methodology below, which hold for any difficulty signal
and are the transferable part.

**C1. A per-problem difficulty prior, inside a sequential cost-constrained rule, strictly beats
count-based beliefs.** Isolating the belief source with RoR's constant costs on *both* sides:
**0 of 401 accuracy levels negative on LiveCodeBench (floor +2.90%) and 0 of 401 on TACO (floor
+0.93%)**; in budget units, **+26.5pt / +25.8pt at a 0.25x budget**. A TF-IDF belief arm remains
worth reporting for completeness, but §3b-xx removes the concern that motivated it: TF-IDF is a
substantially weaker difficulty signal on both pools.

**C1b. Belief and cost heads are substitutes, not complements.** Each alone buys ~+25pt at a 0.25x
budget; stacking is worth -2.13% to -17.20%. On LCB they are literally one scalar (PC1 = 71.4%,
corr -0.52..-0.71); on TACO they are near-orthogonal (PC1 = 43.7%). Rank-1 collapse *improves* LCB
belief AUC at every route (§3b-xiii).

**C2. The value flows through abstention, and the mechanism is a knapsack.** With the give-up
action the same beliefs are worth up to **+43.7%**; without it, ~0%. Stopping is exactly
$\max_m Q(s,m)\le 0$. And abstention at a binding budget is **reallocation** — skip expensive
problems to afford cheap ones — not declining hopeless ones (§3b-xix).

**C2b (methodological, and we think the most portable). The shuffled-prediction control.** A
constant belief and constant cost make every problem's utility identical, so the baseline
*structurally cannot* partially abstain (measured: 0.0% abstention). Any per-problem dispersion
unlocks it, informative or not. Permuting predictions across problems while preserving marginals
separates the two: **92% (LCB) / 69% (TACO) of the belief head's gain is information**, against
**49% / 31% for the cost head**. No routing paper we surveyed runs this control, and without it
"conditioned beats constant" is not a claim about prediction quality (§3b-xv).

**C3. Cross-model transfer: the ordering is free, the probabilities cost ~25 labels.** AUC is
invariant to a positive affine transform, so a 2-parameter link cannot change a ranking — the
ordering transfers with **no labels** (mean peer AUC 0.789 over five API models from five labs).
Labels buy calibration: Brier **0.1946 at N=25 against 0.2434 for the base rate**, saturating at
N≈25. On SWE-bench Verified the base rate wins at every N and the gap plateaus, and the
discriminator is the probe's own AUC (0.79 vs 0.64) (§3b-xviii, §3b-xvii).

**C4. What a prompt representation carries is set by pool structure, and both regimes are
measured.** On our pools (28-41% pool-unsolvable, 52-55% contested) it carries the **shared
difficulty factor** and not the model x problem **interaction**: seven applications tested, the six
needing the interaction all fail for one cause (§7b). On RouterBench (3.8% unsolvable, 91.4%
contested) the decomposition inverts: the shared scalar is **worse than not routing** (-0.60% AIQ)
and the interaction carries the entire gain (+4.52%) (§3b-xxi). **Same probe, opposite
decomposition, and contested mass predicts which** -- so this is a scope law with two measured
regimes, not a flat negative result. *(Supersedes the earlier universal claim; that version was
written before RouterBench and was true only of the pools it was measured on.)*

**C5. Selective prediction, the cleanest form of the result.** One prefill, one ordering, one line:
gap to oracle closed **64.9% (LCB) / 72.7% (TACO) / 43.8% (SWE-bench Verified)**, cutting wasted
spend at 50% coverage by **62% / 54% / 43%**. No MDP, no cost model, no budget parameter (§3b-xvi).

**C6. Methodological corrections that apply beyond this method.** Held-out calibration for every
conditioned quantity (shrink toward the constant it replaces); fit in the space the policy acts in
(dollars, not logs); the randomised convex-hull frontier; and the attribution rule that an arm
changing two things belongs in no single-component cell (§8 R0).

*Prior art:* the prefill-activation router (2603.20895) is single-commit, uses **median training
output tokens** for cost, and has no give-up action; RoR (2607.08665) has the resample/reroute MDP
with count beliefs and no stop action; "The LLM Already Knows" (2509.12886) reads difficulty from
the target's *own* prefill for adaptive decoding; IRT work (JE-IRT 2509.22888, IrtNet 2510.00844,
contextual MIRT 2608.22295) models shared difficulty from response patterns or sentence embeddings,
without abstention or cost-awareness. Own-model length prediction for *scheduling* is a separate
established line (EGTP, ICLR 2026; TRAIL). **Cross-model per-query cost prediction — one cheap
prefill pricing other models inside a budget-constrained decision — is the piece none of them do.**

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

### 3b-vi The sharpest result: a shared latent supports *whether*, never *which*

Tested on a deliberately complementary pool -- five API models from five labs (deepseek-v4-flash,
glm-5, kimi-k2.5, minimax-m2.7, qwen3-max), 171 test problems, response curves fitted on the
calibration split, the latent never having seen a peer label.

**The pool has abundant routing headroom.** Best single model 71.9%; union 85.4%; **52.6% of
problems contested**. Our cascade pool by comparison: best 68.9%, union 72.5%, +3.6pt. So the peer
pool offers **+13.5pt** against the cascade's +3.6pt.

**No probe variant reaches any of it.**

| router | accuracy | cost | acc/$ | distinct models used |
|---|---|---|---|---|
| 1-D shared latent (2 params/model) | 70.2% | $0.0169 | **41.4** | **1 of 5** |
| latent + 8 shared PCA dims | 70.8% | $0.0226 | 31.4 | 5 of 5 |
| per-model probe on activations | 64.9% | $0.0284 | 22.9 | 5 of 5 |
| always kimi (best single) | 71.9% | $0.0321 | 22.4 | — |
| **oracle cheapest-that-solves** | **85.4%** | $0.0179 | **47.7** | — |

**Why the 1-D latent collapses, and it is structural rather than a fitting failure.** All five
response curves are monotone in the *same* scalar with similar slopes (2.11-3.12), so their
**ordering is nearly constant across problems**: one model dominates the whole latent range and
`argmax` picks it 171/171 times. A one-dimensional latent can express *how hard is this problem*
-- which is a threshold, i.e. abstention -- but cannot express *which model suits this problem*,
which requires the ordering to change. **The 1-D router's apparently strong acc/$ is entirely
"always deepseek", reproducible from calibration-set means with no probe at all.**

**And adding dimensions does not fix it.** Eight shared PCA directions, or a full per-model probe,
do start using all five models -- and perform *worse* (31.4 and 22.9 against 41.4). The extra
dimensions add variance, not signal. **The failure is not dimensionality; the information is not in
the prompt representation.**

**This is the paper's cleanest claim, and it is half positive and half negative:**
> A shared difficulty latent read from one cheap prefill transfers across models and labs at ~25
> labels each, and it supports **selective prediction** -- whether a problem is worth attempting --
> which is worth up to +43.7% on the frontier. It does **not** support **model selection**. We
> demonstrate this on a pool with 52.6% contested problems and +13.5pt of available routing gain,
> where every probe variant we tried captures none of it.

That is a more useful contribution than a routing win would have been: it tells practitioners which
half of the router to build, and it is falsifiable by anyone who finds a representation that does
route.

### 3b-vii The unification: shared factors are predictable, interaction terms are not

Cost-based routing was tested as an alternative to success-based routing, on the peer pool. It
fails the same way, and decomposing why unifies every negative result in the paper.

**Per-peer cost is partly predictable** from the scout's activations -- dollar $R^2$ of 0.390
(minimax), 0.246 (glm-5), 0.158 (qmax), 0.144 (kimi), 0.032 (deepseek). **The cost *ratio* between
peers is not: 0 of 10 pairs positive.** Routing on predicted cost scores 35.3 acc/\$ against 41.4
for ignoring cost entirely, and 27.5 for picking the predicted-cheapest.

**The structure genuinely exists.** The peer log-cost matrix is **not** rank-1: the first singular
component explains **72.1%**, leaving **27.9% real per-problem x per-model interaction**. Model
verbosity constants span **14x** (qwen3-max 793 mean output tokens, glm-5 11,269), and pairwise
log-ratios have sd/|mean| of 0.6-7.8, so the cheaper model genuinely changes from problem to
problem. There is something to route on; we cannot see it.

| component | what it is | predictable from a prefill? |
|---|---|---|
| rank-1 problem factor | "this problem is long / hard" | **yes** -- the shared latent |
| rank-1 model factor | "glm-5 is verbose, qwen3-max is terse" | **yes** -- a constant from calibration means, no probe |
| **the 27.9% interaction** | "*this* model rambles on *this* problem" | **no** -- ratio $R^2 \le 0$ on all 10 pairs |

**Success decomposes identically:**
- **Success** = shared difficulty (predictable, so **abstention works**) + model x problem
  interaction (invisible, so **routing fails**).
- **Cost** = shared length (predictable, so the **cost level works**) + model verbosity constant (a
  constant) + model x problem interaction (invisible, so the **cost ratio fails**).

**One mechanism explains six symptoms**: per-candidate probing buys nothing (§6.5); extra PCA
dimensions make routing worse (§3b-vi); cost ratios are negatively predictable (§6.9i); the router
collapses to one model (§3b-vi); TACO's cost head fails where the shared component is weakest
(§6.9n); and beliefs pay only through the give-up action (§6.9f). **A prompt representation carries
shared factors and not interaction terms**, and every routing decision -- success- or cost-based --
needs an interaction term.

*This is the sentence the paper should be built around.* It is a positive claim about what
prompt-only prediction can do, a sharp negative about what it cannot, and it is falsifiable: find a
representation that predicts the interaction term, and routing becomes possible.

### 3b-viii CORRECTION: encoder-target decoupling is theirs, not ours

**Two sections of this outline overclaimed novelty and are corrected here.** The prefill-router
paper (2603.20895) states as its own contribution:

> *"Encoder-Target Decoupling: We show that open-weight encoders can serve as strong predictors of
> closed-source target performance"* and **"hidden states of a different model outperform the
> target model's own hidden states."**

So **one model's activations predicting another model's success is their claim**, and our
measurement that own-model access is worth about zero (-0.024, -0.011 AUC; -0.027, -0.019 on the
differential) is a **replication of their finding on a different pool**, not a discovery. An
earlier draft of this section placed them in a "per-model, not shared" cell of a novelty table.
That was wrong, and it was wrong twice — the reframe felt compelling and I did not check the paper
before writing it down.

**Replication is still worth reporting**, and ours is stronger evidence than a table row: a 4.41B
probe that solves 41.5% predicts gpt-oss-120b's per-problem success within 0.05 AUC of gpt-oss-120b's
own activations and ties it on pool solvability, and a **0.69B** probe retains 95% of pool AUC while
solving almost nothing (§6.6a). Report it as **independent confirmation of encoder-target
decoupling on a sequential pool**, and cite them for the claim.

**What is actually left to us.** They explicitly do **not** address, and we do:

| | 2603.20895 | this work |
|---|---|---|
| abstention / give-up action | no | **yes — the entire value channel (§6.9f, +43.7%)** |
| resampling the same model | no | **yes** |
| sequential decisions under a cost budget | no | **yes** |
| label-efficient addition of a new model | no | **yes — ~25 labels (§3b-ii)** |
| routing gains | **+45.58% gap closed** | **negative on 3 pools (§7b)** |

**Their evaluation:** MMLU-Pro, Humanity's Last Exam, **LiveCodeBench**, LLMRouterBench; pools of
11 (frontier), 20 (7-9B), and 9 (mixed-tier) models; 85/15 splits.

**The live tension worth resolving before submission.** They close 45.58% of the
strongest-to-oracle gap; we get **-33%** on our LiveCodeBench cascade. We overlap on the benchmark,
so the difference is the **pool**: ours is a 3-model cascade where gpt-oss-120b dominates and the
whole oracle gap is **3.6pt**, against their 9-20 model pools chosen for complementary strengths.
That is the pool-structure account (§3b-vi) and it predicts our routing result should improve on
their pools. **`LLMRouterBench` is cached locally** (`datasets--withmartian--routerbench`), so this
is directly testable and should be tested rather than argued.

### 3b-x The two papers sit at opposite ends of one measurable axis

Pool structure, computed identically across every pool we have access to:

| pool | best single | oracle | headroom | **contested** | **pool-unsolvable** |
|---|---|---|---|---|---|
| **RouterBench (11 models)** — their main testbed | 84.3% | 96.1% | +11.8pt | **91.4%** | **3.9%** |
| API peers (5 labs) | 71.9% | 85.4% | +13.5pt | 52.6% | 14.6% |
| SWE-bench Verified (5 routes) | 62.3% | 71.3% | +9.0pt | 55.0% | **28.7%** |
| LiveCodeBench cascade (3) | 68.9% | 72.5% | +3.6pt | 46.1% | 8.3% |
| TACO cascade (3) | — | 60.0% | — | — | **40.0%** |

**RouterBench is 91.4% contested with 3.9% unsolvable** — maximally favourable to routing and
useless for abstention. Our pools are 28.7-40% unsolvable at roughly half the contest rate:
hostile to routing, rich for abstention. **Their +45.58% gap closure and our -33% are the same
scope law read from opposite ends**, not a contradiction, and the axis is measurable before
deployment.

**This is the honest positioning.** They opened the question of predicting one model's success from
another's activations and answered the routing half on routing-favourable pools. We answer the
**selective-prediction half** on pools that frequently fail outright, add the sequential
cost-budgeted setting with a give-up action, and supply **the measurement that says which half
applies to a given pool** (§6.9k). Their paper existing makes ours easier to position, not harder.

**The falsifiable test, on their benchmark, with data already cached.** RouterBench has 3.9%
unsolvable, so our scope law predicts **abstention should be worth almost nothing there**. If it is
worth something, §6.9k is wrong. Running our method on RouterBench is therefore a test of our own
central claim on the opponent's home ground, and it needs only an activation extraction over its
prompts.

### 3b-xi What the prefill-router paper leaves open, in its own numbers

Having corrected the overclaim (§3b-viii), three concrete openings remain, all sourced from their
paper rather than asserted.

**1. They report the gap they cannot act on.** Their Table 1 gives "all incorrect" rates — queries
**no model in the pool solves** — of **14.22% (frontier), 8.21% (small), 12.81% (mixed)**. Their
router "always selects a model; there is no mention of refusing to answer or deferring". So on
their own pools, roughly one query in ten buys a model that cannot succeed. **That is our
motivation stated in their measurements**, and it is the channel worth +43.7% here (§6.9f).

**2. We disagree with their mechanism, and have evidence.** Their §7 ("Foundations for prefill
signals") attributes cross-model transfer to the **encoder's representational geometry** — high
dimensionality, isotropy, Fisher separability — concluding that larger open-weight encoders have
better geometry for separating correct from incorrect.

Our probe-scaling result points elsewhere: a **0.69B** probe retains **95%** of pool-solvability AUC
(0.818 against the 4B's 0.865 on LiveCodeBench; 0.804 against 0.845 on TACO) **while solving almost
nothing on TACO itself**. A geometry-of-the-encoder account predicts strong scale dependence; we
measure a 0.047 AUC drop across a 6.4x size gap, from a model that cannot do the task. That favours
the signal being **problem-level difficulty**, which any competent encoder reads, over a property
of the encoder's representation space.
*State it as a competing hypothesis with an ablation, not a refutation:* their geometric measures
and our scale sweep are compatible with a weak scale effect on top of a dominant shared factor, and
distinguishing them properly needs their d_eff/anisotropy/Fisher measurements run on our probe
ladder. **That is a concrete, cheap experiment and it should be in the paper.**

**3. They do not measure the axis that decides which method applies.** Their pools sit at 8-14%
all-incorrect and RouterBench at 3.9%; ours at 28.7-40% (§3b-x). Neither their paper nor
RouterBench measures advantage *as a function of* that fraction — which is exactly the controlled
sweep of §6.9k.

### 3b-xii Report the frontier in budget units — it is a knapsack, and R is the multiplier

The utility rule $\arg\max_m (p_m R - c_m)$ with a **common** $R$ across problems is the
Lagrangian relaxation of a global budget constraint, with $R$ the multiplier. **Sweeping $R$ is
the knapsack.** RoR's density rule is the same greedy applied *within* a problem; **abstention is
what extends it across problems**, because giving up on one frees budget for another.

Re-expressing the frontier in the units a practitioner actually has — a budget, as a multiple of
calling the largest model on everything:

| budget | LCB: RoR<br>as published | LCB: RoR<br>+ our cost head | LCB: ours | TACO: RoR<br>as published | TACO: RoR<br>+ our cost head | TACO: ours |
|---|---|---|---|---|---|---|
| **0.25x** | 27.6% | **53.6%** | 52.8% | 17.9% | **42.4%** | 41.0% |
| 0.50x | 58.7% | 55.3% | **62.2%** | 45.7% | 48.2% | **50.1%** |
| 0.75x | 65.6% | 65.4% | **68.3%** | 52.3% | 50.2% | **52.0%** |
| 1.00x | 66.3% | 71.9% | **72.9%** | 54.3% | **54.0%** | 53.7% |
| 1.50x | 75.2% | 76.4% | **77.4%** | 56.7% | 55.1% | **56.2%** |
| 2.00x | 79.0% | 79.5% | **79.7%** | 57.5% | **57.3%** | 56.2% |
| 3.00x | 82.5% | **83.0%** | 80.2% | 58.3% | 57.8% | **58.6%** |

**CORRECTION-2 (2026-09-08, supersedes both earlier versions of this section).** The first
version reported only RoR-as-published vs our full arm. A first correction then claimed the gain
was "almost entirely the cost head" — **that claim was itself wrong** and is withdrawn. It compared
our full arm against `counts_qcost`, which subtracts our *own* cost head from the baseline side;
that measures **redundancy between our two heads**, not attribution against RoR. The correct
attribution holds the other component fixed on both sides. Doing that:

**Accuracy at a fixed budget (family hulls, multiples of always-oss120):**

| budget | RoR | + our beliefs | + our cost head | + both | bel | cost | both |
|---|---|---|---|---|---|---|---|
| **0.25x** LCB | 27.6% | **54.2%** | 53.6% | 52.8% | **+26.5** | +25.9 | +25.1 |
| 0.50x | 58.7% | 59.4% | 55.3% | **62.2%** | +0.7 | -3.4 | +3.5 |
| 1.00x | 66.3% | 70.1% | 71.9% | **72.9%** | +3.7 | +5.6 | +6.6 |
| 2.00x | 79.0% | 79.5% | 79.5% | **79.7%** | +0.5 | +0.5 | +0.7 |
| **0.25x** TACO | 17.9% | **43.7%** | 42.4% | 41.0% | **+25.8** | +24.5 | +23.1 |
| 0.50x | 45.7% | 47.6% | 48.2% | **50.1%** | +1.9 | +2.5 | +4.4 |
| 1.00x | 54.3% | **54.7%** | 54.0% | 53.7% | +0.4 | -0.3 | -0.6 |
| 2.00x | 57.5% | 56.7% | 57.3% | 56.2% | -0.8 | -0.3 | -1.4 |

**The +25pt is real and it belongs to the beliefs.** At the tight budget the belief head alone is
the *best* arm on both benchmarks. C1 stands.

**Strict-improvement scan, every pairing, over the paper's own ranges** (LCB 45-84.5%, TACO
28-61.5%; 401 levels; negative-count / floor):

| held fixed on both sides | LCB | TACO |
|---|---|---|
| costs = RoR constant; **beliefs alone** | **0/401, +2.90%** | **0/401, +0.93%** |
| beliefs = RoR counts; **cost head alone** | **0/401, +3.88%** | 159/401, -21.63% |
| both changed (full arm vs RoR) | **0/401, +5.07%** | 63/401, -11.83% |
| *redundancy:* beliefs added on top of cost head | 28/401, -2.13% | 24/401, -3.94% |
| *redundancy:* cost head added on top of beliefs | 155/401, -4.06% | 200/401, -17.20% |

**The two heads are substitutes, not complements.** Each alone buys ~+25pt at the tight budget and
a strict improvement on LCB; adding the second on top of the first buys nothing and often costs.
They are two readouts of **one** prefill, so this is what redundancy should look like — and it is a
result, not a disappointment: *you only need to pay for the probe once, and you can spend it on
whichever head your pool supports.*

**Which head a pool supports is measurable in advance.** Per-query cost $R^2$ (test half):

| route | RoR constant | prompt length | prefill activations |
|---|---|---|---|
| LCB scout / oss20 / oss120 | -0.148 / -0.146 / -0.221 | -0.103 / 0.013 / 0.021 | **0.051 / 0.356 / 0.586** |
| TACO scout / oss20 / oss120 | -0.000 / -0.004 / -0.001 | -0.028 / -0.073 / 0.028 | **0.166 / 0.081 / 0.309** |

*(Corrected 2026-09-08: an earlier version of this table reported 0.457/0.675/0.784 and
0.294/0.303/0.415 from a random half-split of all 892 problems, which put ~275 training problems
into the "test" half. The numbers above use the manifest test split, n=171/168.)*

TACO's cost is harder to predict on the routes that matter (oss20 0.081 against LCB's 0.356) and that is exactly
where the cost head goes negative on the frontier (-21.63% floor). The gate of SS6.9-cost is the
right mechanism; this is the number that sets it. **Note also that prompt length is worthless on
both pools and actively negative on TACO — the cost signal is in the activations, not in a free
proxy.** That is the cleanest single defence of paying for a prefill at all.

### 3b-xiii Why the two heads are redundant, and when one scalar is enough

C1b said the belief head and the cost head are substitutes. This is the mechanism. Correlation
between the belief logit and the log cost, per route, and the PCA of the six predictions the probe
emits (3 belief logits + 3 log costs):

| | corr(logit $\theta$, log $c$) scout / oss20 / oss120 | PC1 | PC1+PC2 |
|---|---|---|---|
| LiveCodeBench | -0.599 / -0.521 / -0.710 | **71.4%** | 81.5% |
| TACO | -0.132 / -0.165 / -0.059 | 43.7% | 78.6% |

**On LiveCodeBench the two heads are reading one scalar** — hard problems cost more and solve less,
and a single component carries 71.4% of everything the probe emits. That is *why* they are
substitutes, and it is the shared-difficulty-factor claim (C4) showing up inside our own
predictions. **On TACO they are nearly orthogonal**: difficulty and expense are separate
properties, PC1 carries only 43.7%, and the pool needs two dimensions.

**Collapsing to one scalar is not merely lossless on LCB — it is an improvement.** Rank-1
reconstruction of the six predictions (fitted on train, scored on the manifest test split):

| | belief AUC, 2 heads | belief AUC, **1 scalar** | cost $R^2$, 2 heads | cost $R^2$, 1 scalar |
|---|---|---|---|---|
| LCB scout | 0.830 | **0.872** | 0.051 | -0.023 |
| LCB oss20 | 0.759 | **0.786** | 0.356 | **0.392** |
| LCB oss120 | 0.768 | **0.803** | 0.586 | 0.554 |
| TACO scout | 0.736 | **0.772** | 0.166 | 0.052 |
| TACO oss20 | **0.833** | 0.758 | 0.081 | 0.042 |
| TACO oss120 | **0.819** | 0.726 | 0.309 | 0.049 |

**On LCB the single scalar beats both independent heads on belief AUC at every route** (+0.027 to
+0.042) at no real cost to the cost head — the rank-1 projection denoises two separately-fit
heads by forcing them through the structure the data actually has. **On TACO it does the opposite**
where the pool is genuinely 2-D (oss20 0.833 -> 0.758). PC1's variance share predicts which regime
you are in, and it is measurable before any of this is deployed.

*Design consequence:* the probe should emit **one number** on pools like LCB, with per-route
2-parameter links to beliefs and costs — the same 2-parameter response curve the transfer result
already uses for new models (§3b-ii). That unifies the belief head, the cost head, and cross-model
transfer into one object, and removes the duplication of feeding the same 40,960 activations into
two independently-penalised ridge problems. *Not yet run end-to-end through the policy;* the table
above is a predictor-level result and the frontier version is the obvious next experiment.

### 3b-xviii C3 re-scored on Brier: the claim survives on LiveCodeBench and dies on SWE-V

§3b-xvii showed the AUC table could not measure label efficiency. Re-running the five-peer transfer
with the link fitted on N **calibration** problems and scored on the 171 **test** problems (20
resamples), against each peer's own base rate estimated from the same N labels:

| peer | solve | AUC | Brier N=10 | **N=25** | N=50 | N=170 | base N=25 | base N=170 |
|---|---|---|---|---|---|---|---|---|
| deepseek | 0.702 | 0.746 | 0.2247 | 0.1902 | 0.1852 | 0.1812 | 0.2192 | 0.2201 |
| glm-5 | 0.620 | 0.812 | 0.2064 | 0.1986 | 0.1899 | 0.1822 | 0.2559 | 0.2476 |
| kimi | 0.719 | 0.790 | 0.1943 | 0.1794 | 0.1658 | 0.1629 | 0.2134 | 0.2067 |
| minimax | 0.544 | 0.811 | 0.2572 | 0.1986 | 0.1876 | 0.1815 | 0.2668 | 0.2535 |
| qwen3-max | 0.520 | 0.787 | 0.2441 | 0.2062 | 0.1998 | 0.1925 | 0.2616 | 0.2527 |
| **mean** | | **0.789** | 0.2253 | **0.1946** | 0.1857 | 0.1801 | **0.2434** | 0.2361 |
| mean, glm-5 dropped | | 0.783 | 0.2301 | 0.1936 | 0.1846 | 0.1795 | 0.2403 | 0.2333 |

**The latent beats the new model's base rate for every peer at every N**, by ~24% of Brier at
N=25, and it saturates at N≈25 (0.1946 -> 0.1857 -> 0.1801 for 25 -> 50 -> 170). **So "~25 labels"
was right after all** — the original table simply could not see it, because AUC is blind to what
the labels buy. The claim is now supported by the metric that matches the use: a utility rule
consumes probabilities.

**The result also survives the label-quality problem.** glm-5 still has 25.7% empty and 38.6%
truncated outputs (down from 42% but not fixed), and dropping it moves the mean by 0.001. The
conclusion does not rest on the corrupted row.

**And it does not transfer to SWE-bench Verified** (§3b-xvii): there the base rate wins at every N
up to all 219 available, with reliability (0.0389) exceeding resolution (0.0254) — the latent's
miscalibration costs more than its separation buys, and more labels do not close the gap. The
discriminating quantity is the latent's own strength: **peer AUC 0.746-0.812 on LiveCodeBench
against 0.616-0.670 on SWE-V.** Pool extension works where the probe is strong and fails where it
is weak, which is a boundary condition rather than a contradiction, and it is measurable before
committing to a pool.

**Restated C3.** *The latent's **ordering** transfers to unseen models for free — a 2-parameter
link cannot change a ranking, so no labels are needed for it. Its **probabilities** transfer at
about 25 labels on a pool where the probe reaches AUC ~0.79, and do not transfer at all on one
where it reaches ~0.64.* Both halves are new relative to the AUC-only claim, and the second half
is the one a deployment depends on.

### 3b-xx The baseline battery the difficulty-prediction literature demands — and we pass it

Difficulty estimation is an established field we had not benchmarked against: IRT for LLM
evaluation (JE-IRT 2509.22888; contextual multidimensional IRT 2608.22295; IrtNet 2510.00844), and
same-model prefill difficulty estimation (**"The LLM Already Knows" 2509.12886**, which reads
difficulty from *the target LLM's own initial hidden state* with no generation, and spends it on
adaptive Self-Consistency / Best-of-N). We are cross-model where they are same-model, and
budget-constrained where they are efficiency-oriented. **Their existence demands cheap baselines,
and this section runs them.**

**Matched pipelines.** Identical target (pool-solvability), identical estimator (`RidgeCV`),
identical alpha grid (25 values, 1e-2..1e6), identical manifest split. Pool-solvability AUC:

| benchmark | n | *(ref)* human easy/med/hard | prompt length | TF-IDF | **prefill activations** |
|---|---|---|---|---|---|
| LiveCodeBench | 171 | *0.755* | 0.711 | 0.729 | **0.824** |
| TACO | 168 | *0.434* | 0.375 | 0.746 | **0.885** |

Per-query cost $R^2$, same matched protocol:

| benchmark | route | TF-IDF | **prefill activations** | delta |
|---|---|---|---|---|
| LCB | scout | -0.177 | -0.061 | +0.116 |
| LCB | oss20 | -0.103 | **0.208** | +0.311 |
| LCB | oss120 | -0.001 | **0.414** | +0.415 |
| TACO | scout | 0.008 | **0.132** | +0.124 |
| TACO | oss20 | -0.001 | **0.186** | +0.186 |
| TACO | oss120 | 0.123 | **0.395** | +0.272 |

**Activations beat TF-IDF on solvability by +0.095 / +0.139 AUC and on cost at every one of six
routes**, by +0.12 to +0.42 $R^2$, with TF-IDF at or below zero on four of six. The
difficulty-prediction baselines are real baselines and the probe clears them.

**The human easy/medium/hard column is a reference, not a baseline** — it is benchmark metadata
that does not exist for an arbitrary user query. It is reported only to show how much signal a
coarse human judgment carries (a lot on LCB at 0.755, none on TACO at 0.434), and notably the probe
beats it on both.

**TF-IDF is nonetheless the right baseline to keep**, because it is genuinely deployable: fit the
vectoriser and ridge offline, then at test time only `transform` the prompt — the same input the
probe prefills, with no GPU forward pass. That it loses by this margin is the result.

**The LiveCodeBench comparison is a temporal one**, which strengthens it: train 2023-09-02..
2024-09-28, calibration 2024-10-05..2025-01-04, test 2025-01-11..2025-04-06, so 0.824 was measured
on genuinely future contests. *(TACO's split is not temporal — dates overlap and many are missing —
so its +0.139 is not shift-tested.)*

**A finding about our own pipeline, worth acting on.** A plain `RidgeCV` on activations
(0.824 / 0.885) **beats the committed, calibrated probe** (0.759 / 0.838) at pool-solvability on
both benchmarks. The committed chain may be over-engineered or mis-regularised for this target.
Worth a direct check before submission — it is free accuracy if it replicates.

**CORRECTION (2026-09-08).** The first version of this section reported TF-IDF 0.750/0.759 against
activations 0.745 on LCB and concluded "a bag of words matches the 40,960-dimensional probe",
which triggered a full reframe of the paper (retired title, rewritten abstract, a new C0 saying the
representation claim was measurably false). **That comparison was rigged by construction and the
conclusion was wrong.** The TF-IDF arm was given `RidgeCV` over 13 alphas while the activation arm
got a single hand-picked `alpha=1000`; cross-validation selects **21544**, so the probe was
under-regularised by more than an order of magnitude. With matched pipelines the ordering reverses
on every cell. The reframe has been reverted. **Rule, now on the same footing as §8 R0: a
representation comparison must give every arm the same estimator, the same hyperparameter search,
and the same split — a hand-set penalty on one arm is not a baseline, it is a handicap.**

### 3b-xxi RouterBench, with our actual probe: +3.91% AIQ, and it is all interaction

**Protocol.** RouterBench (36,497 prompts, 11 models, 3.8% pool-unsolvable, 91.4% contested).
Scout activations extracted for every prompt (Qwen3-4B-Instruct-2507, `--max-len 8192`, the same
config as our LCB/TACO probes). Kernel ridge on 21,898 x 20,480, alpha selected on a held-out
slice (lands at 1e5, interior to a 1e2-1e6 grid). **Their baseline** (Zero Router = non-decreasing
convex hull of the individual LLMs) and **their metric** (AIQ, mean quality over the shared cost
domain). Training pooled, evaluation per dataset, MMLU subsets aggregated as they aggregate them.

| dataset | n | Zero | TF-IDF | **ACTIV** | shuffled | shared-only | give-up off |
|---|---|---|---|---|---|---|---|
| mmlu | 5596 | 0.7200 | 0.7344 | **0.7472** | 0.6932 | 0.7354 | 0.7459 |
| hellaswag | 3984 | 0.7717 | 0.8151 | **0.8244** | 0.7023 | 0.7795 | 0.8238 |
| grade-school-math | 3011 | 0.8752 | 0.8817 | **0.8957** | 0.8192 | 0.8304 | 0.8955 |
| arc-challenge | 583 | 0.8961 | 0.8978 | **0.9133** | 0.8720 | 0.9149 | 0.9127 |
| winogrande | 516 | 0.7249 | 0.7035 | 0.7246 | 0.7191 | 0.7296 | 0.7247 |
| chinese_zodiac | 164 | 0.5775 | 0.5983 | 0.5842 | 0.4458 | 0.4080 | 0.5827 |
| mbpp | 162 | 0.6591 | 0.6561 | 0.6531 | 0.5693 | 0.5545 | 0.6522 |
| consensus_summary | 156 | 0.9021 | 0.9087 | 0.8777 | 0.8164 | 0.7949 | 0.8777 |

**Weighted AIQ against the Zero Router (n=14,172):**

| arm | delta |
|---|---|
| TF-IDF (our text baseline) | +2.43% |
| **prefill activations (ours)** | **+3.91%** |
| shuffled control | **-6.03%** |
| shared scalar only | **-0.60%** |
| ours, give-up action disabled | +3.82% |

**Decomposition:** information (ours - shuffled) **+9.94%**; interaction (ours - shared scalar)
**+4.52%**; abstention (ours - give-up-off) **+0.10%**.

**Three findings, and the first two are the paper's.**

1. **The probe beats the text baseline on their benchmark too** — +3.91% against +2.43%, and
   pool-solvability AUC 0.777 against 0.744, with per-model AUC higher on all 11. Consistent with
   §3b-xx.
2. **A shared difficulty scalar is worse than not routing at all here (-0.60%).** All of the gain
   is the model x problem interaction. **This is the sharpest available statement of C4's scope
   condition:** on our pools the interaction is unlearnable and the shared scalar carries
   everything; on a 91.4%-contested pool the shared scalar is worthless and the interaction carries
   everything. Same probe, opposite decomposition, and pool structure predicts which.
3. **Abstention contributes +0.10% — nothing.** Confirmed with the real probe, not just TF-IDF.
   The give-up action is a step function here because predicted $p$ is high everywhere. So the
   two-channel story holds: unsolvable mass -> abstention channel; contested mass -> routing
   channel.

**Positioning.** RouterBench reports that its own KNN and MLP routers "generally do not
significantly outperform the Zero Router". +3.91% weighted is a modest, honest win in that regime;
we should not claim to beat their routers without their per-dataset AIQ values. We lose on
consensus_summary (-2.71%) and mbpp (-0.92%), both small.

**Everything in this section before 2026-09-10 was TF-IDF, not our method, and three separate
errors were corrected on the way here** — all in the flattering direction, all caught by controls
rather than by inspection:
- *mixtures disabled*: the budget lookup returned the best hull vertex rather than interpolating,
  which penalises the baseline most. Inflated the frontier gain to "+5.9..+15.0pt".
- *global cost constants in a per-dataset evaluation*: RouterBench's per-dataset costs range from
  0.21x (winogrande) to 2.20x (gsm8k) of the global mean, so gpt-4 was mispriced ~6x and never
  selected. Produced a spurious -17.44% on winogrande.
- *train/test overlap*: a per-dataset split drawn from a fresh RNG while the router had been
  trained on the global 60% split, so ~60% of each evaluation set was in training. Produced a
  spurious +14.96% on winogrande.

**One control that came out clean and is worth reporting:** an oracle-dataset router — *told* which
benchmark each prompt came from, using that benchmark's per-model base rates and no per-prompt
signal — is worth **-0.07%**. So none of the gain is benchmark identification; it is per-prompt
prediction, as claimed.

### 3b-xxvi Selecting the belief head on AUC is the wrong objective — measured, not argued

§3b-xxii found the shipped head under-regularised and selecting C on calibration AUC worth
+0.024 AUC. Regenerating both belief heads with `--select-C` and re-running all eight replays
(5 LCB seeds, 3 TACO; everything else held fixed), bootstrap CIs over seeds:

| | arm | shipped C | **C selected on AUC** | delta |
|---|---|---|---|---|
| LCB | beliefs only | +2.90% [+0.73,+3.44] | **+4.03% [+0.98,+4.41]** | **+1.13pp** |
| LCB | full method | +5.07% [+1.04,+7.71] | +4.31% [+0.22,+5.33] | -0.76pp |
| **TACO** | beliefs only | +0.93% [-4.75,+2.81] P=0.514 | **-3.41% [-11.22,-1.09] P=0.000** | **-4.34pp** |
| TACO | full method | -11.83% P=0.000 | -16.57% P=0.000 | -4.74pp |

**A head with strictly better ranking makes the policy worse on three of four arms.** The mechanism
is the one §3b-xv identified: AUC is invariant to any monotone transform, so it scores only the
*order* of the predictions, while the rule $\arg\max_m(p_mR-c_m)$ consumes their *values*.
Selecting C on AUC picked 16-160x stronger regularisation, which compressed predictions toward the
base rate and destroyed the dispersion the decision rule was exploiting. **Better ranking, less
spread, worse policy.**

This is C3b one level up. We had already fixed "fit the cost head in dollar space, not log space,
because the policy spends dollars" — and then selected the belief head in *AUC space* when the
policy acts in *probability space*. **The general rule: select every component on the objective the
policy optimises, not on a proxy that is invariant to the thing the policy uses.**

**Consequence for TACO, and it settles a question left open in §3b-xxiii.** TACO's
strict-improvement claim is dead under *both* heads: P(floor>0) = 0.514 with the shipped head (a
coin flip) and 0.000 with the AUC-selected one. **Report TACO as a win on the budget axis and on
utility at matched R, and drop the strict-improvement language for it entirely.** LiveCodeBench's
claim survives both (P = 0.998 / 0.995).

**Open, and the right next experiment:** sweep C and select on the *calibration frontier* rather
than calibration AUC, then report test once. Design: ~6 values of C spanning the shipped 0.003125
down to the AUC-selected 0.0000188, choose on one seed's calibration frontier, then run the
remaining seeds at the chosen value (6 + 8 jobs rather than 48). **The AUC-vs-frontier divergence
measured above is the evidence that this matters**, and it is a better contribution than the
+0.024 AUC it replaces.

### 3b-xxxvi Whose budget is global? RoR's is per-query; the global-budget-with-abstention paper is ROI-Reasoning

**A conflation to avoid (made once in session, caught immediately).** Two prior baselines have
different budget formulations and they must not be merged:

| baseline | budget | ranking statistic | abstains? |
|---|---|---|---|
| **RoR** (2607.08665) | **per-query** — *"competing uses of one per-query budget"* | density $p/c$ | **no** |
| **ROI-Reasoning** (2601.03822) | **global batch** (OS-MCKP over problems) | ROI | **yes** — theirs |
| **ours** (`_value` arms) | global, solved in the dual at price $R$ | surplus $pR-c$ | yes |

So our `counts` arm, which caps spend per episode, is a **faithful** reimplementation of RoR on the
budget axis. There is no formulation error there. The error would be to describe RoR as a batch
knapsack — that is ROI-Reasoning, a different paper, and `PRIOR_ART.md` §4b already records that
**the knapsack formulation and abstention-under-budget are both theirs, not ours**.

**Where abstention comes from, stated correctly.** Under a global budget $B$, the Lagrangian
relaxation of $\max\sum_i \mathbb{E}[\text{correct}_i]$ s.t. $\sum_i \mathbb{E}[\text{cost}_i]\le B$
decouples per problem; with $R = 1/\lambda$ the per-problem rule is
$\max(0,\ \max_m (p_m R - c_m))$. The zero action is in the feasible set, so **abstention is
complementary slackness** — the marginal dollar buys more accuracy elsewhere in the batch. RoR
cannot express this because it ranks by **density** $p/c$, a ratio of positive quantities with no
zero to cross; we rank by **surplus**, which has a sign. Stated in our own code at
`replay_mdp_full_execution.py:703`: *"the density p/c ... never crosses zero, so it cannot stop,
which is why the budget-swept baseline has no abstention."*

**The uncomfortable consequence, which is the point of this section.** Our `_value` arms run with
`unconstrained_budget` (`:1312`, `:1706`), so the cap never binds and $R$ alone traces the frontier:
**our arm is a global-budget method and our primary baseline is a per-query-budget method.** A
global budget is strictly more powerful — it can move money from a doomed problem to a solvable one;
a per-query cap cannot. So part of every margin reported against `counts` is *formulation*, and
formulation-with-abstention is **ROI-Reasoning's contribution, not ours**.

**Therefore `counts_value` — count beliefs under our global dual — is the baseline that isolates
what is actually ours.** The 2x2:

| | per-query cap (RoR's knob) | global dual $R$ |
|---|---|---|
| count beliefs | `counts` — **RoR as published** | `counts_value` — **the honest baseline** |
| our beliefs | `content_decay_qcost` | `content_decay_qcost_value` — **ours** |

Read it this way: the **row** difference is representation (ours), the **column** difference is
formulation (ROI-Reasoning's). If the margin vs `counts_value` is small, the win is formulation and
the contribution claim must shrink to cross-model pricing from one prefill. Reporting only
`counts` would claim the column as ours. **Report the full grid.**

### 3b-lx The belief claim SURVIVES a weak verifier but roughly halves — and the ceiling collapses

Policy stops on `weak_verifier_outcome` (11.01% false accepts) and is still **scored on the truth**,
so a false accept ends the episode with a wrong answer shipped. Ours vs `counts_value`, formulation
held fixed:

| verifier | 50% | 60% | 70% | 80% | 84% | top accuracy reachable |
|---|---|---|---|---|---|---|
| **oracle** (what every prior result assumed) | +40.8% | +19.2% | +12.0% | +2.3% | +6.6% | **84.8%** |
| **weak** (11% false accepts) | **+19.6%** | **+10.5%** | unreachable | unreachable | unreachable | **62.1%** |

*(beliefs + cost head: +45.8/+15.7/+15.2/+5.5/+10.5 oracle, +13.3/+16.2 weak.)*

**Three things, and all of them must be in the paper.**

1. **The claim survives.** At every target the weak verifier can reach, activation beliefs still beat
   count beliefs — **+19.6% and +10.5%**. The result is not an artefact of assuming an oracle.
2. **It roughly halves.** +40.8 → +19.6 and +19.2 → +10.5. **We reproduce RoR v1's own ablation
   exactly** — *"gains are verifier-gated, shrinking as verifier quality degrades"* — which makes
   this corroboration of the prior work rather than a weakness of ours.
3. **The ceiling collapses from 84.8% to 62.1%**, and this is the bigger deal. With an 11%
   false-accept rate the policy *cannot* reach the high-accuracy regime at any price, because
   episodes terminate on a lie. **Every result we report at the 70/80/84% targets is conditional on
   a verifier far better than the weak one.**

**Reporting rule.** Report both verifiers side by side, and state the top reachable accuracy for
each. Do **not** report 70/80/84% targets without saying they require a near-oracle verifier. The
honest headline is *"activation beliefs cut cost at matched accuracy by +19.6%/+10.5% under a
realistic verifier and +40.8%/+19.2% under a perfect one"* — which is still the strongest claim in
this document that survives every objection raised against it.

**This also settles §3b-lvii properly.** Under the oracle verifier a wrong answer converts to a
decline for free, so disclosure is worthless; under the weak verifier 11% of accepted answers are
wrong and you ship believing you won, so disclosure is worth something again. **Verifier quality is
the dial that controls whether the disclosure argument is available at all** — state it as a
function, not as a yes or no.

### 3b-lxi Does the MDP earn its keep? The SEQUENTIAL structure does; the deep lookahead does not

Our own results are awkward for a paper that leads with "MDP": the exact solve does not beat h=2
(§3b-lviii), h=2 barely beats myopic, and the headline arm **is** myopic (h=1). So the question has
to be asked directly: what part of the machinery is actually load-bearing?

**Test: same beliefs, single-commit versus sequential** (vs `counts_value`, LCB seed 0):

| arm | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| **single-commit**, our beliefs | +15.1% | **−10.0%** | *unreachable* | *unreachable* | *unreachable* |
| **sequential**, our beliefs | **+40.8%** | **+19.2%** | **+12.0%** | +2.3% | +6.6% |

**The sequential structure is what earns its keep, on three counts.** It **more than doubles** the
tight-budget gain (+15.1 → +40.8); it **turns a loss into a gain** at 60% (−10.0 → +19.2); and most
importantly it **extends the reachable range** — single-commit routing on this pool cannot reach the
70% target *at any price*, while the sequential policy reaches 84.8%. Resampling is what buys the
high-accuracy regime; no amount of better single-commit routing gets there.

**So justify the three components we actually use, not "the MDP":**

| component | evidence it pays | verdict |
|---|---|---|
| **sequential resample/reroute** | +15.1→+40.8 at 50%, −10.0→+19.2 at 60%, and 70%+ unreachable without it | **load-bearing** |
| **stop action (zero-value abstention)** | `counts` cannot discriminate at entry; the interaction term is +27.5pt at the tight target (§3b-xxxvii) | **load-bearing** |
| global price rather than a per-episode cap | +5.1%/+5.8% at tight targets, ~0 or negative at loose (§3b-xxxvii) | **marginal** |
| **deep lookahead (h>2)** | h=4/6/18 identical to each other and no better than h=2; 84% *degrades* 3.9→2.9 | **does NOT pay** |

**Write it this way and the awkwardness disappears.** We are not claiming exact dynamic programming
helps — we measured that it does not, and we explain why (§3b-lviii: the transition model, not the
search depth, is the binding constraint). We are claiming that **a sequential policy with a stop
action, driven by per-problem beliefs, beats a single-commit router with the same beliefs** — which
is exactly what the table shows, and which is the claim RoR v1 made and v3 withdrew.

### 3b-lviii The EXACT solve does not beat h=2 — the transition model caps the lookahead, not the depth

§3b-lv predicted that raising the horizon should help, since truncation under-values continuation
and makes stopping too attractive. Run on LCB, matched grid, against the **myopic** arm:

| horizon | 50% | 60% | 70% | 80% | 84% | lattice nodes |
|---|---|---|---|---|---|---|
| **h=2** | −8.8% | −5.4% | **+1.3%** | **+4.2%** | +3.9% | 4 |
| h=4 | −8.8% | −6.1% | −0.7% | +2.2% | **+4.5%** | 20 |
| h=6 | −8.8% | −6.1% | −0.7% | +1.6% | +3.6% | 56 |
| **h=18 (exact, nothing truncated)** | −8.8% | −6.1% | −0.7% | +1.6% | **+2.9%** | 342 |

**The prediction was half right and the interesting half is wrong.** Lookahead *does* help at the
loose end — h=2 is +4.2%/+3.9% at the 80/84% targets against myopic — but **going deeper than 2 adds
nothing and at 84% actively hurts** (3.9 → 4.5 → 3.6 → **2.9**). h=4, h=6 and h=18 are essentially
identical at the tight end, so the solve has converged; the differences are not search error.

**Diagnosis, pre-registered in §3b-lv and now supported.** At $h \ge 2$ the lattice reuses the root
belief at every node, asserting failing route $m$ says nothing about route $m'$ — which is false.
Independence *over*-values continuation while truncation *under*-values it, and the two partially
cancel at $h{=}2$. Deeper search removes the truncation error while **compounding** the transition
error, so the net is flat-to-negative. **The value of the lookahead is capped by the quality of the
transition model, not by the depth of the search.**

**This is the same shape as the belief finding, and the two should be reported together.** More
exact optimisation over a wrong model buys nothing; the binding constraint is the model. Use
**h=2**, and say why — not because deeper is expensive (342 nodes is free) but because deeper is
*not better* until the transition model is fixed.

### 3b-lix The null ladder: random allocation BEATS the count-based greedy rule

All on one budget grid, relative to `counts` (the RoR v1 policy), positive = cheaper at that accuracy:

| arm | information available | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|---|
| **random allocation** | **none at all** | **+3.5%** | **+4.2%** | **+6.6%** | **+4.4%** | **+1.4%** |
| budget-aware best-of-K | pool prior, one route | −22.0% | −13.7% | +11.5% | −11.7% | — |
| `counts` (RoR v1) | pool prior, may reroute | — | — | — | — | — |
| **ours** | **per-problem** | **+48.6%** | **+20.6%** | **+16.7%** | **+5.3%** | **+7.5%** |

**Choosing routes uniformly at random is cheaper than the count-based greedy rule at every target.**
That is a real result about the baseline, not a bug: the density rule $p_m/c_m$ is scale-invariant in
cost, so with a pool prior it systematically prefers the cheapest route and over-buys scout draws,
while random selection diversifies. A rule with *no information* beats a rule with *pool-level
information used badly*.

**Three consequences.**

1. **Report random allocation.** RoR v1 names it as a baseline and it is the one that exposes this.
2. **"We beat RoR" is the weaker claim; "we beat random" is the load-bearing one.** Ours is +46.7%
   over random at the 50% target and positive at all five. Lead with that comparison, or a reader
   who runs the null themselves will find the embarrassment first.
3. **It sharpens the per-problem claim.** Pool-level beliefs are not merely *less* useful than
   per-problem ones — plugged into a density rule they are *worse than nothing*. That is exactly
   why `counts` cannot discriminate at entry (§3b-liv) and why the interaction term is large.

*Single seed, one pool. Replicate before it goes in the paper — this is a strong claim about a
published baseline and it needs seeds.*

### 3b-lvi Probe scan: code-specialisation HURTS, the solving prompt hurts, and the free scalars are redundant

Eleven candidates, one protocol (same manifest split, kernel-ridge head, penalty per route on
calibration, scored on test). Mean test AUC over the three routes, `mean+last` readout unless
stated. **Absolute levels are not the deployed head's** (binarised label); the ranking is the claim.

**Encoder axis — the user's hypothesis was right: general beats code-specialised.**

| encoder | params | specialisation | mean AUC |
|---|---|---|---|
| **Qwen3-4B-Instruct** (current scout) | 4B | general | **0.8673** |
| Qwen3-4B-Thinking | 4B | reasoning | 0.8577 (**0.8729** on `content_last`) |
| **Qwen3-0.6B** | **0.6B** | general | **0.8528** |
| Qwen3-1.7B | 1.7B | general | 0.8513 |
| Qwen2.5-**Coder**-1.5B | 1.5B | **code** | 0.8422 |
| Qwen2.5-**Coder**-3B | 3B | **code** | 0.8399 |
| Llama-3.2-1B-Instruct | 1B | general, other family | 0.8362 |

**A 0.6B general model beats a 3B code-specialised one** (0.8528 vs 0.8399). Code-specialisation is
*negative* at matched scale: both Coder models sit below every general Qwen3 including the one
5x smaller. The plausible mechanism is that a code-tuned model's representation collapses toward
"what code do I emit", discarding the difficulty signal we need. **And Qwen3-0.6B is within 0.0145
AUC of the 4B scout at roughly a sixth of the prefill cost** — a further cost reduction available if
the policy gap is as small as the AUC gap.

**Prompt axis — the SOLVING prompt was hurting, and the judge prompt is not the fix.**

| system prompt on the 4B scout | mean AUC | vs current |
|---|---|---|
| "expert competitive programmer … output only Python code" (**current**) | 0.8673 | — |
| "You are an expert at assessing … do not solve it" (**judge**) | 0.8692 | +0.0019 |
| solving prompt + difficulty question appended (**suffix**) | 0.8716 | +0.0043 |
| **"You are a helpful assistant." (plain)** | **0.8765** | **+0.0092** |

**Removing task framing beats both solving and judging framing.** The purpose-built judge prompt is
worth almost nothing (+0.0019) while simply *deleting* the competitive-programmer instruction is
worth five times as much. The prompt was never ablated and it was the wrong one — though the
magnitude is small, and **§3b-lvii must confirm it reaches the policy before it is claimed**.

**Readout matters, and concatenation can hurt.** `last` and `content_last` beat `mean` almost
everywhere, and for Qwen3-4B-Thinking `content_last` alone (0.8729) beats `mean+last` (0.8577).
Concatenating a weak readout with a strong one dilutes it. **Select the readout on calibration.**

**The free scalars are a null result — report it.** Prompt NLL, next-token entropy and next-token
max log-prob score 0.61–0.78 *alone*, so they carry real signal, but concatenating them onto the
activations moves the mean AUC by **≤0.0002 on every candidate**. The activations already contain
everything these three scalars know. *A free signal still has to be non-redundant, and this one
is not.*

*Gemma-2-2b failed to extract:* its chat template rejects a system role (`jinja2 TemplateError:
System role not supported`). Cross-family evidence therefore rests on Llama-3.2-1B alone, which is
the weakest candidate in the scan — so "Qwen is the better family" is **not** established here.

### 3b-lv The method, stated properly — and why we were truncating at h=2 for no reason

**Setup.** Problems $i = 1..N$. Routes $m = 1..M$, route $m$ having $K_m$ available draws at cost
$c_m$ each. Within an episode the state is the failure-count vector
$\mathbf{n} = (n_1,\dots,n_M)$; $p_m(\mathbf{n})$ is the probability the next draw on route $m$
succeeds given $\mathbf{n}$. Actions: buy one draw on some route, or stop.

**The global problem.** Maximise expected solves under one batch budget:

$$\max \sum_i \mathbb{E}[\mathbb{1}\{\text{solved}_i\}] \quad\text{s.t.}\quad \sum_i \mathbb{E}[\text{cost}_i] \le B.$$

**Lagrangian relaxation.** With multiplier $\lambda \ge 0$ the objective decouples across problems;
dividing by $\lambda$ and writing $R = 1/\lambda$ — the **dollar value of a correct answer** — each
problem independently solves $\max\ \mathbb{E}[\mathbb{1}\{\text{solved}\}]\,R - \mathbb{E}[\text{cost}]$.
Sweeping $R$ traces the frontier; this is the standard convexification, the same one that makes
randomised tests admissible in Neyman–Pearson.

**The per-problem MDP.** Backward induction over the failure-count lattice:

$$V(\mathbf{n}) \;=\; \max\Big(\,\underbrace{0}_{\text{stop}},\ \max_m\ \big[\,p_m(\mathbf{n})\,R \;-\; c_m \;+\; (1-p_m(\mathbf{n}))\,V(\mathbf{n}+\mathbf{e}_m)\,\big]\Big)$$

with $V \equiv 0$ once every route is exhausted. **The zero in the outer max is abstention** — it is
not an added mechanism but the null action's value, and it fires exactly under complementary
slackness (§3b-xxxvi).

**Beliefs.** $p_m(\mathbf{n}) = \theta_m \cdot \sigma/(\sigma + n_m)$, a Beta–Bernoulli decay.
The **baseline** sets $\theta_m = \pi_m$, the pool-level route prior — *identical for every
problem*, which is why count beliefs cannot discriminate at $\mathbf{n} = \mathbf{0}$ (§3b-liv).
**Ours** sets $\theta_m$ from one prefill of the cheapest model. §3b-liii closed the question of
learning $\sigma$ per problem: it works, and the prefill beats it.

**Horizon.** Truncating the recursion at depth $h$ sets $V = 0$ beyond depth $h$:

| $h$ | what it is | lattice nodes per decision |
|---|---|---|
| 1 | **the myopic rule** $\max(0,\max_m[p_mR - c_m])$ — verified byte-identical | 1 |
| 2 | what every result in this document used | 4 |
| 4 | | 20 |
| 6 | | 56 |
| **18** | $\sum_m K_m$ — **nothing truncated, the exact solve** | **342** |

**We have been truncating at $h=2$ for no reason.** The exact solve is 342 nodes per decision on
this pool — 85x the $h=2$ cost and still negligible. There was never a computational barrier; $h=2$
was inherited and never revisited.

**What the truncation does, and the prediction it makes.** Setting $V = 0$ beyond depth $h$
*underestimates* $V(\mathbf{n})$, because a real continuation has non-negative value. Under-valuing
continuation makes the stop action (value exactly 0) relatively more attractive, so **a truncated
policy gives up too early**. Raising $h$ should therefore help most where you want to keep buying —
loose budgets — and matter least at tight ones.

**That is exactly what was measured.** $h{=}2$ against the myopic arm: **+10.4%/+13.6% at the 80/84%
targets** and *worse* at 50% (§3b-liii, §3b-xliv). The loose end is where §3b-l says all the
remaining headroom is.

**And it predicts an interaction with the cross-route correction.** At $h \ge 2$ the lattice reuses
the root belief at every node, asserting that failing route $m$ says nothing about route $m'$
(§3b-xliv item 5). That assumption *over*-values continuation, while truncation *under*-values it —
opposite signs. So the two partially cancel at $h{=}2$, and **the independence error should grow
with $h$**: the deeper the lookahead, the more compounding an unrealistic transition model does.
**If the exact solve underperforms $h{=}2$ at tight budgets, that is the diagnosis, not a failure of
exact dynamic programming.** Sweep launched over $h \in \{2,4,6,18\} \times \rho \in \{0, 0.1, 0.25\}$
to test it.

### 3b-liii §3b-xxxi CLOSED: learned sigma works, and our probe beats it

The comparison §3b-xxxi said had never existed — learned per-problem $\sigma$ against our probe, on
the 64k pool, **in one replay** so seed, draw orderings and budget grid cannot confound the belief
source. Factorized scorer trained on the 64k reachable dataset; control is `counts_value`.

| arm | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| **learned $\sigma$** (factorized scorer) | +12.8% | +4.2% | +4.8% | −0.3% | +7.3% |
| **our probe, beliefs only** | **+40.8%** | **+19.2%** | **+12.0%** | +2.3% | +6.6% |
| our probe + cost head | +45.8% | +15.7% | +15.2% | +5.5% | +10.5% |
| learned $\sigma$, Bellman h2 | +11.2% | +2.8% | +5.9% | −0.0% | +9.7% |
| **ours, Bellman h2** | +41.0% | +15.4% | **+16.7%** | **+10.4%** | **+13.6%** |

Head to head (positive = ours cheaper): beliefs-only **+32.1/+15.7/+7.6/+2.6/−0.8%**, full arm
**+37.9/+12.1/+11.0/+5.8/+3.4%**.

**Answer: learned $\sigma$ works — it beats count beliefs at four of five targets — but our probe
beats it at four of five.** The open question is closed: *it works, and it does not beat ours.* This
also retires the "predict $\sigma$ per problem" direction: a learned decay is a one-parameter
summary of what the prefill already carries in full.

**A second finding, and it matters more.** `ours + Bellman h2` is **+10.4%/+13.6%** at the 80/84%
targets against the myopic arm's +5.5%/+10.5%. h2 is worse at the tight end and **better at the
loose end** — which is exactly the regime §3b-l identified as holding all the remaining headroom and
where no other variant moved anything. **The lookahead is the one structural lever that acts where
we are weakest.** Select the horizon per operating point on calibration, as with the cost head.

### 3b-liv Entry vs continue, done correctly: at tight budgets most of the value is the decision to START

Re-run under `--start-protocol free_start`, where the policy may decline before buying anything.
(The first attempt was confounded: under `scout_first` the scout draw is mandatory, so
`failures.sum() == 0` never occurs in the decision loop and the entry arm never fired — §3b-xlix.)

| the per-problem prior is allowed to act... | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| **only at ENTRY** (depth 0), counts after | **+25.0%** | +5.1% | +0.6% | −1.3% | +1.0% |
| **only on CONTINUE**, counts at entry | **+40.8%** | **+16.8%** | **+10.8%** | +4.3% | +5.0% |
| everywhere (full) | +43.3% | +20.6% | +11.9% | +1.6% | +6.5% |

**This answers "is it just a better difficulty estimate plugged into RoR's machinery?" — and the
answer is *it depends on the budget*, which is itself the finding.**

- **At loose budgets, yes.** Entry contributes ~0 and the whole margin is CONTINUE: the prior is
  being tracked through the episode, which *is* the same machinery with a better input.
- **At the tight budget, no.** Entry alone is worth **+25.0% of the full +43.3%** — 58% of the
  margin — at a decision point where count beliefs are **structurally incapable** of discriminating,
  since $s\pi/(s+n)$ at $n=0$ is identical for every problem. That is not a better estimate in the
  same machinery; it is discrimination where the baseline has none.

The two halves are **sub-additive** (25.0 + 40.8 ≫ 43.3), as every pair of our components has been.

**Consistent with three earlier measurements:** `free_start` helps our arm and helps RoR by *exactly*
0.0% (§3b-xliv); `counts_value` abstains at 41.6% and still costs more (§3b-xxxvii); and the
representation x formulation interaction is +27.5pt at the tight target and ~0 at the loose one.
**All four are the same mechanism seen from different angles.**

### 3b-lii The smallest model in the pool is also the BEST encoder — and success has almost no label noise

Two measurements that together say belief quality is worth attacking and say how.

**1. There is essentially no irreducible label noise on success.** Between-problem share of the
variance in observed per-problem rates (ICC), with binomial sampling variance removed:

| route | ICC | draw noise |
|---|---|---|
| scout | **0.975** | 2.5% |
| oss20 | **0.917** | 8.3% |
| oss120 | **0.924** | 7.6% |

Unlike **cost**, whose ICC is 0.841 and whose ceiling is therefore materially below $R^2=1$
(§3b-xxix), success is nearly deterministic per problem: 92–98% of the spread is real
between-problem signal. **So the oracle-belief headroom of §3b-l (+65.5%/+73.4% at the loose
targets, of which we capture 4%/9%) is genuinely reachable, not a mirage of draw noise.** Belief
quality is the right thing to attack.

**2. Scaling the probe does not help. The 4B scout is the best encoder we have.** Identical probe
pipeline, identical split and penalty selection, only the model whose prefill is read changes; AUC
on the held-out test split:

| encoder | prefill cost | AUC → scout | AUC → oss20 | AUC → oss120 | mean |
|---|---|---|---|---|---|
| **4B scout** | **\$0.000149** | **0.8703** | **0.9066** | 0.8261 | **0.8677** |
| gpt-oss-20b | \$0.000709 (4.8x) | 0.8429 | 0.8851 | 0.8256 | 0.8512 |
| gpt-oss-120b | \$0.006345 (**42.6x**) | 0.8601 | 0.8895 | **0.8366** | 0.8620 |

**The cheapest encoder is the best on average, and 42x cheaper than the dearest.** It is also a
clean instance of the **encoder-target decoupling** the prefill-router paper reports (`PRIOR_ART.md`
§1): the scout's hidden states predict `gpt-oss-20b`'s success **better than gpt-oss-20b's own
hidden states do** (0.9066 against 0.8851). Only for `gpt-oss-120b` does the target's own prefill
win, and by 0.011 AUC for 42x the price.

**This strengthens the cost case on quality grounds rather than merely on price.** The usual
objection — "you used a small probe to make the economics work; a bigger probe would predict
better" — is false here and measurably so. **Report this table.**

*Caveat on comparability:* these AUCs use a binarised label (rate > 0.5) and a kernel-ridge head
with the penalty selected on calibration, so they are **not** the deployed head's numbers (which
predicts the rate and scores 0.7685 on oss120). The protocol is identical across the three
encoders, so the **relative** ranking is sound; the absolute level is not the deployed probe's.

### 3b-l Oracle bounds: the loose end is where everything is left, and it is a BELIEF problem

Two oracle arms on the current setup (LCB, matched grid), each replacing exactly one component and
holding the rest fixed. Both are diagnostic upper bounds, never deployable.

| oracle component | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| perfect **stopping** | *n/a* | *n/a* | *n/a* | **+44.0%** | **+65.0%** |
| perfect **routing** | *n/a* | *n/a* | **+43.1%** | **+27.6%** | +8.3% |
| perfect **both** | *n/a* | *n/a* | *n/a* | +51.0% | +69.4% |
| perfect **beliefs** (§3b-xlix) | +75.4% | +71.8% | +68.7% | **+65.5%** | **+73.4%** |

*The tight-target cells are unreadable and are marked n/a rather than reported.* The oracle arms
attach to the **frozen-$R$** policy, which has only 4–5 operating points from the retention grid, so
its hull cannot span the tight end; the raw numbers there (−445%, −524%) are an artefact of grid
coverage, not a finding. **Only the 70–84% columns are interpretable.**

**This overturns the working hypothesis from §3b-xliv.** That section argued the stopping channel
was nearly closed because four variants all bought ~+10% at the 50% target and faded by 80%. The
oracle says the opposite about the *other* end: **perfect stopping is worth +44% to +65% at the
80–84% targets**, and every variant we ran moves that regime by ~0. So stopping is not closed — it
is **untouched where it matters most**, and the variants were all competing over the tight end
because that is where a *myopic* improvement can act.

**And the loose end is a belief problem before it is a policy problem.** Perfect beliefs are worth
**+65.5%/+73.4%** at 80/84% while we capture **4%/9%** of that (§3b-xlix). Perfect routing is worth
only +8.3% at 84%. So the binding constraint at loose budgets is *knowing which problems are
hopeless*, not *choosing among routes* — and no structural variant can manufacture that.

**Direction this sets.** Stop tuning the give-up rule at the tight end, where four knobs already
overlap. The open problem is **belief quality at high coverage**: at 84% we recover 9% of the
available belief headroom and 0% of the stopping headroom.

### 3b-li Quantile cost head: a small, consistent win for the CONSERVATIVE direction only

Re-run after the void first attempt (§3b-xliv item 6). Each $q$ against a **matched mean arm** built
by the identical pipeline, so the contrast is the functional and nothing else.

| $q$ | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| 0.10 | −4.9% | −7.2% | +2.8% | +0.6% | +1.5% |
| 0.25 | −2.5% | −6.5% | +3.5% | +2.1% | +0.4% |
| 0.50 (median) | −0.2% | −2.4% | +1.2% | +1.8% | +1.0% |
| 0.75 | −1.2% | −3.9% | +3.7% | +1.6% | +1.4% |
| **0.90** | **+3.8%** | **+1.2%** | **+1.5%** | **+5.5%** | **+0.8%** |

**Only $q=0.90$ is positive at every target**, and the optimistic tail ($q=0.10$, 0.25) is clearly
negative at the two tight ones. That is the sign the asymmetry argument predicted — under a cap,
under-pricing can exhaust the episode while over-pricing only forgoes a buy — but the magnitude is
small (≈+2.6% mean across targets). **Worth reporting as a cheap, directional finding; not worth
building on.** Single seed; the +5.5% at 80% should be seed-replicated before being quoted.

### 3b-xlviii CONTAMINATION: `content_preds_RANK1.jsonl` has test-split AUC 0.998 and must not be used

Found while running the belief-source ladder, where the RANK1 arm returned +71.8/+67.2/+64.6/
+31.5/+20.9% — better than our own arm at every target and within a few points of the **oracle**.
That is impossible on its face: a rank-1 projection of a set of predictions can only *lose*
information, never gain it. Checked directly against the truth on the manifest **test** split:

| preds file | AUC (oss120) | corr with true per-problem rate | MAE vs truth |
|---|---|---|---|
| `content_preds_rich_cal.jsonl` (ours) | 0.7685 | 0.5315 | 0.2531 |
| **`content_preds_RANK1.jsonl`** | **0.9980** | **0.9061** | **0.1472** |
| `content_preds_KNN.jsonl` | 0.7450 | 0.4928 | 0.2706 |
| `content_preds_ORACLE.jsonl` (true rate) | 1.0000 | 1.0000 | 0.0006 |

**A rank-1 collapse of a 0.769-AUC predictor cannot have AUC 0.998.** The file was built with test
labels in scope. **Any number computed from it is contaminated.**

**Consequences.**
1. The RANK1 row of the belief-source ladder is **void** and excluded.
2. **§3b-xiii's claim that "collapsing to one scalar is not merely lossless on LCB — it is an
   improvement" must be re-verified**, since its rank-1 reconstruction table may have been computed
   from this artefact. Until re-fitted train-only, treat that claim as unsupported.
3. Every `*_ORACLE*` and `*RANK1*` file in the prepared directories should be assumed diagnostic
   and never fed to a reported arm. The oracle files are *labelled* as such and are used correctly
   as upper bounds; RANK1 is not labelled and was being read as a method arm.

### 3b-xlix What the belief head is actually doing — it is the activations, not merely a per-problem prior

The 19/19 claim compares against `counts_value`, which has **no per-problem prior at all**, so it
cannot separate "per-problem beliefs help" from "these beliefs help". The ladder separates them.
Only the belief source varies: constant costs on both sides, **no cost head anywhere**, same
formulation, grid, seed and split. TF-IDF and length heads are fitted with the **same** protocol as
the activation head — same manifest split, penalty selected per route on calibration by the same
criterion — so this cannot repeat the rigged comparison of §3b-xx.

| belief source | needs a forward pass? | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|---|
| `counts` — no per-problem prior | no | — | — | — | — | — |
| **problem LENGTH** (chars + words) | **no** | **+2.7%** | **−2.8%** | **−2.2%** | +5.3% | +4.7% |
| **TF-IDF** of the statement | no | +12.2% | **−0.3%** | +11.8% | +1.2% | +0.3% |
| kNN on activations | yes | +26.8% | +8.8% | +7.6% | +0.8% | +3.3% |
| **activations (ours)** | yes | **+40.8%** | **+19.2%** | **+12.0%** | **+2.3%** | **+6.6%** |
| *ORACLE beliefs* | — | *+75.4%* | *+71.8%* | *+68.7%* | *+65.5%* | *+73.4%* |

**Three things this settles.**

1. **It is not "any per-problem prior".** Problem length — the most trivial per-problem signal that
   exists — buys essentially nothing and is *negative* at two targets. The policy does not simply
   benefit from having some per-problem number to condition on.
2. **It is the activations specifically, not the text.** A properly-fitted TF-IDF head on the same
   statements recovers at most a third of the gain and is **negative at 60%**. Our arm beats it at
   every target. Since the probe is one prefill of a model that never generates, this is the
   representation claim in its strongest available form.
3. **The channel is far from exhausted, and the gap is worst where we are weakest.** Against the
   oracle belief we capture **54% / 27% / 17% / 4% / 9%** of the available cost saving at the five
   targets. At the loose end we recover almost none of it — which is exactly the regime no variant
   has moved (§3b-xliv), and says the loose-budget problem is *belief quality*, not policy
   structure.

**Entry vs continue: the first run was confounded and is being redone.** Under `scout_first` the
scout draw is *mandatory*, so `failures.sum() == 0` never occurs inside the decision loop and the
"entry only" arm never fires — it returned −1.0/−0.7/−0.3/−0.2/−0.1%, i.e. count beliefs everywhere,
exactly as that diagnosis predicts. **Under this protocol there is no entry decision to measure.**
Re-running under `--start-protocol free_start`, where the policy may decline before buying anything.

### 3b-xlvi THE CLEAN CLAIM: the belief head alone is positive on 19 of 19 targets across four pools

Everything conditional in this document came from reporting the **bundled** arm. Isolate the belief
head — formulation held at the global dual, **costs held constant and identical on both sides** —
and the result is not conditional at all:

| pool | | | | | |
|---|---|---|---|---|---|
| **LiveCodeBench** (5 seeds) | 50% **+40.5±1.0** | 60% **+19.6±2.4** | 70% **+16.5±2.6** | 80% **+4.6±1.6** | 84% **+5.4±2.1** |
| *sign test* | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **TACO** (3 seeds) | 35% **+33.6±1.1** | 40% **+29.3±0.2** | 45% **+2.2±2.0** | 50% **+7.3±1.9** | 55% **+1.1±0.8** |
| *sign test* | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 |
| **RouterBench** (14,599 test) | 60% **+35.2** | 70% **+76.9** | 75% **+75.1** | 80% **+50.8** | 84% **+22.9** |
| **SWE-bench Verified** | 30% **+19.5** | 40% **+9.5** | 50% **+3.4** | 55% **+1.2** | — |

**19 of 19 targets positive, across four pools and two task families** (competitive programming and
software engineering), and unanimous across seeds at 17 of the 19 where seeds exist. The SWE-V row
uses my deliberately weak probe reconstruction (§3b-xxxviii); the recorded probe is stronger.

**This is the headline. Lead with it.** One sentence: *replacing count-based beliefs with beliefs
read from one cheap prefill, changing nothing else, reduces cost at matched accuracy at every
accuracy target on every pool we have.*

**Why it read as a pile of conditionals until now.** The reported arm bundled the belief head with
the **cost head**, and the cost head is genuinely pool-dependent: a large win on LCB (+39.4±2.9 at
the tight target) and **actively harmful on TACO above 40%** (−21.6% at 45%). §3b-xiii predicted
exactly this — on TACO the heads are nearly orthogonal (PC1 43.7% against LCB's 71.4%) and TACO's
cost $R^2$ is the weakest we measure. **So the cost head is a second contribution with a documented
failure mode, not a caveat on the first.** Report them separately and select on calibration.

### 3b-xlvii TACO's pool is NOT degenerate — the problem is the price of the top rung

A standing hypothesis (and an intuition worth testing before acting on it) was that TACO fails
because `oss20` and `oss120` perform alike, i.e. a degenerate two-rung pool. **Measured, it is
false:**

| pool | scout | oss20 | oss120 | rungs on the hull | switching price of the top rung |
|---|---|---|---|---|---|
| LCB | 41.5% / \$0.00116 | 68.1% / \$0.00889 | 80.9% / \$0.03237 | **3 of 3** | **\$0.18** per unit accuracy |
| TACO | 16.3% / \$0.00161 | 38.3% / \$0.01464 | 45.6% / \$0.05005 | **3 of 3** | **\$0.49** per unit accuracy |

All three TACO routes sit on the hull and the `oss20`→`oss120` gap is a real 7.3pp. The pool is
structurally fine. What differs is **price**: TACO's top rung costs **2.7x more per unit of
accuracy** than LCB's. So at loose budgets on TACO the expensive rung *must* be bought, and
mis-pricing it is directly expensive — which is precisely why the **cost** head hurts there and the
**belief** head does not.

**Do not rebuild the TACO pool.** The principled form of that move is a **pre-registered
admissibility criterion on pool structure** — e.g. "≥3 rungs on the cost/accuracy hull with
switching prices inside the operating range" — decided *without reference to our method's
performance*. TACO **passes** that criterion, so excluding it would be selection on the outcome.
And it is unnecessary: the belief-head claim survives TACO at all five targets.

### 3b-xlv TACO on the matched grid: the headline arm's cost claim RETRACTS above 40%, but the representation claim survives

3 seeds, matched geometric budget grid, strict `counts` baseline (mean ± sd, sign test across seeds):

| contrast | 35% | 40% | 45% | 50% | 55% |
|---|---|---|---|---|---|
| ours vs **RoR as published** | +35.5±2.9 (3/3) | +20.2±2.3 (3/3) | **−5.7±4.0 (0/3)** | **−12.5±4.6 (0/3)** | **−31.8±5.1 (0/3)** |
| ours vs `counts_value` | +34.4±2.3 | +18.9±1.8 | −7.5±3.3 | +8.0±1.3 | −21.6±3.6 |
| **our beliefs only** vs `counts_value` | **+33.6±1.1** | **+29.3±0.2** | **+2.2±2.0** | **+7.3±1.9** | **+1.1±0.8** |
| our **cost head** only vs `counts_value` | +26.3±1.6 | +16.9±0.6 | **−21.6±4.5** | +8.0±0.5 | −5.0±3.3 |
| formulation alone | +1.6±1.0 | +1.7±1.0 | +1.7±1.0 | −22.4±6.4 | −8.5±6.1 |

**RETRACT §3b-xxxiv's TACO row.** It recorded +45.7/+42.5/+21.4/+13.1/+9.1 against "RoR as
published". On the matched grid the same contrast is **+35.5/+20.2/−5.7/−12.5/−31.8**, losing at
three of five targets with 0/3 seeds. The old numbers were the **linear** budget grid, which
under-samples RoR by 3 points below \$0.02 (§3b-xxxv). *The TACO cost claim holds only at the two
tightest targets.*

**But the representation claim survives intact, and the diagnosis is clean.** Our **belief head
alone** is **positive at all five targets** (+33.6/+29.3/+2.2/+7.3/+1.1). What breaks the full arm
is the **cost head**: −21.6% at 45% and −5.0% at 55%. That is exactly what §3b-xiii predicted for
this pool — on TACO the two heads are nearly orthogonal (PC1 carries only **43.7%** against LCB's
71.4%), difficulty and expense are separate properties, and TACO's cost $R^2$ is the weakest we
measure. **On TACO the cost head is actively harmful above 40% and the belief head is not.**

**What this implies for the paper, and it is not a fudge.** The deployable arm is not fixed across
pools: whether to include the cost head is a **per-pool decision that must be made on calibration**,
exactly as `--select-alpha` and the target-space choice already are. Reporting one arm everywhere
either wastes the belief head's TACO win or imports the cost head's TACO loss. **Report the
factorization on every pool, select the arm on calibration, and say so.** What must never be done is
selecting it on test, which is what quoting the better of the two post hoc would amount to.

### 3b-xliv Six method variants, run. Two work, two are marginal, one is mixed, one was void

LCB 64k, seed 0, matched geometric grid, each against **its own run's** control so nothing is
confounded across replays. Cost at matched accuracy vs `content_decay_qcost_value` unless stated.

| variant | 50% | 60% | 70% | 80% | 84% | verdict |
|---|---|---|---|---|---|---|
| **two-constraint (cap x price)** | **+4.1%** | **+7.1%** | **+11.4%** | **+2.4%** | — | **works** |
| **posterior over $k$** (exact Bayes, no $\sigma$) | **+11.0%** | **+5.7%** | **+5.9%** | +0.9% | −6.5% | **works, tight/mid only** |
| `free_start` (ours) | +9.2% | +2.2% | −1.4% | −2.2% | −1.4% | marginal, tight only |
| winner's-curse shrink ($\lambda{=}0.25$) | +12.1% | +1.2% | +0.8% | +2.1% | +0.3% | marginal, tight only |
| cross-route $\rho{=}0.25$ *(vs its own h2 control)* | +8.0% | +2.0% | −0.7% | −3.6% | −2.9% | mixed |
| quantile cost head | — | — | — | — | — | **VOID, see below** |

**1. The two-constraint policy is the best new result.** Sweeping a per-episode cap *and* the
global price jointly (12 caps x 96 prices) beats the price alone at **all four reachable targets**,
and the cap *alone* is catastrophic (−61.7% at 50%). So neither knob is right on its own and the
product grid genuinely contains both edges — as predicted from the anomaly that motivated it. It
inherits the cap's ceiling and cannot reach 84%. **This is neither RoR's cap nor the pure
Lagrangian, so it is a policy-class contribution rather than a representation one.**

**2. The posterior-over-$k$ head works where the bimodality argument says it should.** +11.0/+5.7/
+5.9% at the tight and middle targets, and **−6.5% at 84%** — at maximum $R$ nothing is declined, so
the sharper posterior has nothing to act on and only its extra estimation noise remains.

**3. `free_start` is a mechanism confirmation more than a win.** +9.2%/+2.2% at the tight targets
for our arm and **exactly +0.0% at every target for RoR**. Count beliefs are identical for every
problem at entry, so RoR cannot use the freedom to skip the scout *selectively* — it can only enter
or not, which the budget sweep already covered. **Only the arm with per-problem beliefs can spend a
free start.** Third independent instance of the selective-vs-indiscriminate mechanism. Within
`free_start`, ours vs RoR is +53.3/+22.3/+15.5/+3.2/+6.3%, slightly better at the tight end than
the +48.6/+20.6/+16.7/+5.3/+7.5% under `scout_first`.

**4. The winner's curse is real but small.** +12.1% at the 50% target and ~0 elsewhere, and
$\lambda{=}1$ turns negative at 70% (−3.9%). The selection bias exists and biting it earlier helps
exactly where the give-up is most active, but it does not explain the 63–86% waste.

**5. Cross-route $\rho$ partially repairs h2 rather than improving on myopic.** Against its own
h2 control it is **+6.9–8.0% at 50%** and negative from 70% up. h2 alone is −8.9% against myopic at
50%; $\rho$ recovers most of that. So the independence assumption *is* costing the lookahead at
tight budgets, which was the hypothesis — but the net effect against the myopic headline is ~zero.

**6. The quantile experiment was VOID and has been re-run.** All five $q$ produced **byte-identical
frontiers**. Cause: the pipeline applies a dollar-space linear recalibration $a + b\cdot\text{pred}$
fitted on calibration *after* the smearing step, and a linear recalibration **absorbs any constant
multiplicative shift exactly** — so the quantile was silently undone. Fixed by re-applying the
quantile/mean ratio *after* the recalibration, so the shrinkage corrects the shape and the quantile
sets the level (ratios now 0.47–0.85 at $q{=}0.25$ against 1.24–1.94 at $q{=}0.90$). *Lesson for
this codebase: any downstream affine recalibration will annihilate an upstream level knob. Check
that variants differ before reading their results.*

### 3b-lxxxv ⚠ METHODS: a serving artifact, not a model, cost gpt-oss-20b twelve points (2026-09-21)

**Symptom.** In the pool pilot gpt-oss-20b at high reasoning effort measured **78.5%** pass@1
against 80.5% at medium, which reads as "high effort on a small model is a trap". It is not. 17.5%
of its draws never closed an answer turn: 7.0% hit the 65k token cap and **10.5% returned
`finish_reason == "tool_calls"`** -- the model opened a harmony tool call although no tools were
offered. `extract_code` then scraped a fragment of in-progress reasoning and graded *that*: those
draws pass 32% against 90% for draws that stop normally. On draws that stop normally the rung is
**90.3%** (an upper bound -- dropping truncated draws selects failures out of the sample).

**Cause: two OpenRouter endpoints, not the model and not the prompt.** Pinned to Parasail the leak
reproduces 3/3; unpinned it was 4/6 live on 2026-09-21. It is fixed by **neither** `tool_choice:
"none"` **nor** a system line telling the model it has no tools (3/3 still tool_calls with both).
In the pilot's 2,300 gpt-oss draws every one of the 31 tool_calls came from Parasail (4.3% of its
draws) or AkashML (3.5%); **every other provider was at exactly 0.0%**. With
`provider.ignore = [Parasail, AkashML]` the live rate is 0/6.

**Fix** (dc96a74, 7d3f0da): `--ignore-providers` defaults to those two endpoints, and a `tool_calls`
finish now triggers the provider-excluding retry even when `content` is non-empty -- the existing
retry only fired on *empty* content, which is why these rows sailed through.

**Why this is in the outline and not just the commit log.** This is the *third* time a serving
artifact has been mistaken for a capability difference in this project, always on gpt-oss: empty
content at the 4096 cap (43.1% of oss20 eval draws), the provider-concentrated empties on the 64k
re-collection, and now tool_calls with non-empty content. **A per-provider breakdown of
`finish_reason` is a required check on any new collection**, before any rung is compared to another
-- it takes one pass over the rows and it has moved a rung by 12 points. gpt-oss-120b is clean
(0/200 tool_calls, 0/200 truncated, p90 23,868 tokens against the 65,536 budget), so its 87.5% and
its position as the rescue rung (§3b-lxxxiv) stand.

### 3b-lxxxiv ⭐ THE FRONTIER RUNG IS NOT NEEDED: gpt-oss-120b-high beats Claude as the rescue rung

**Question.** The recollection plan (THREADS §0) budgeted ~$75/pool for claude-sonnet-5 and
claude-opus-5 on the argument that the method needs a wide price spread, so the pool should straddle
open-weights → frontier. Does the result rest on them?

**Measured** (pool pilot, 100 LCB problems, recommended sampling, market prices 2026-09):

| rung | pass@1 | c/draw | coverage over its draws |
|---|---|---|---|
| gpt-oss-20b low | 67.9% (6 draws) | 0.011 | 84 → 90% over 6 |
| gpt-oss-20b medium | 83.6% (8 draws) | 0.034 | 85 → 94% over 8 |
| deepseek-v4-flash | 83.0% | 0.048 | 82 → 92% over 2 |
| gpt-oss-120b medium | 86.8% (5 draws) | 0.103 | 86 → 91% over 5 |
| gpt-oss-120b high | 87.5% | 0.652 | 88 → 91% over 2 |
| **claude-sonnet-5** | **88.3%** | **2.406** | single draw |
| **claude-opus-5** | **94.0%** | **6.033** | 94 → 94% over 2 |

1. **Neither Claude model adds any coverage.** Union over the open ladder at one draw each is
   **95.7%**; adding sonnet-5, opus-5, qwen3-235b-thinking, glm-5, kimi-k2.5 or gemini-3-flash moves
   it by **+0.0pt each**. Opus solves **0** problems no open rung solves and *misses* 2 that an open
   rung gets; Sonnet solves 0 and misses 7. This is the nested-pool result (§3b-lxxx) extending all
   the way to the frontier: ten models from seven labs, one difficulty ladder, no specialists.
2. **The open top rung is the better escalation target.** Of the 12 problems both 20b rungs fail at
   draw 0, **gpt-oss-120b high rescues 8**, opus-5 7, qwen3-235b 6, deepseek 5, sonnet-5 5. On the 6
   problems 20b-medium fails on all 8 of its draws: 120b-high 2, opus-5 2, sonnet-5 1, everyone else
   0. So the rung the policy escalates to is the open one, at **1/9 of opus's price**.
3. **Opus is dominated by cheap depth.** One opus draw is 6.03c for 94%; the open ladder reaches
   95.7% for well under 1c of blind draws. A policy with Claude in the pool would almost never call
   it, so the money would buy a rung that never fires.
4. **The spread is already there without it.** 0.011c → 0.652c is **59x**, against **7x** for the
   old scout/20b/120b pool at market prices — which is what made that pool's give-up decision cheap
   and the fixed cascade hard to beat (§3b-lxxiii). The price-spread requirement is met by the open
   ladder alone; it was the *old* pool, not the absence of Claude, that compressed it.

**Sonnet vs Opus, for the record:** 88.3% vs 94.0% pass@1, 2.41c vs 6.03c per draw — 5.7 points for
2.5x the price, and Sonnet is a worse rescue rung than gpt-oss-120b high on both hard sets.

**Decision.** The default pool is **open rungs only** (~$32 for 892 LCB problems, launcher
`launch_pool_recollect.sh`), and the frontier is a later top-up. *Caveats, so the top-up is a
decision and not an afterthought:* the hard sets are n=12 and n=6, so ±1–2 problems is noise; and
LCB is competitive programming, where gpt-oss is strongly tuned. **Trigger for buying Claude:** the
open ladder tops out — the fixed cascade sits on our frontier at the expensive end, or the policy
spends its whole cap on 120b-high and still cannot reach the accuracy target — or a library-heavy
pool (BigCodeBench) shows frontier models rescuing problems the ladder cannot.

### 3b-lxxxiii ⭐ THE DESIGN THIS POINTS TO: predict a DISTRIBUTION over a route's success rate

**The diagnosis (LCB seed 0, deep-state replay).** A probe that reads the whole trajectory and runs
with NO decay is **+16.4 / +16.7% at the 80 / 84% targets and -59.8 / -18.1% at 50 / 60%** against
today's beliefs. Depth-bucketed calibration does not fix the tight end (tried: -59.8 unchanged), and
depth-0 predictor quality is fine (Brier 0.104 vs 0.097 scout; the deep head is BETTER for the 120B,
0.101 vs 0.105). The cause is the output shape:

> **the learned head has no tail.** Among states with history, 5.2% of decay beliefs fall below 2%
> but only **0.2%** of the deep head's do; its depth-0 5th percentile is 0.058 (scout) against the
> decay's 0.042, and its spread is narrower everywhere.

With a scout draw at ~0.07c and a small R, the give-up test fires only below ~0.3-2%. A logistic
head + Platt cannot reach there, so at tight budgets it can never decline cheaply and keeps buying;
at loose budgets the threshold is high and its better ranking wins. The same lens explains why the
**success-count posterior** is our best tight-budget belief (+23.5% at 50%, 5/5 seeds): mass on
"never solves" drives its probability to ~0 by construction.

**So the head should emit a distribution over the per-draw success rate q_m, conditioned on the
state, not a scalar probability.** One object then supplies: tail resolution (mass at q~0 => belief
~0), fluke-vs-never (is the mass bimodal?), optimal DEPTH (P(no success in n draws) = sum_g w_g
(1-q_g)^n in closed form, no lattice), and no hand-written decay (the history conditions the
distribution). It merges the two things that have worked: reading failures and the count posterior.

**Parameterisations, cheapest first.**
1. *Spike-and-slab Beta* (recommended start): 3 outputs per route -- logit pi_0 (P(never)), log a,
   log b -- so p = (1-pi_0) a/(a+b). Exact conjugate update for draws the text did not include,
   closed-form depth values, tiny parameter count (3 x d per route).
2. *Histogram over a q grid* (generalises today's posterior-over-k): G bins, softmax; G=11-21.
   Most flexible, but G x d parameters per route -- with 551 training problems, needs low-rank or
   PCA features.
3. *Posterior over k in K draws* (today's head) is the special case with grid j/K; its replay form
   must be the with-replacement one (3b-lxxvi).

**Training loss = marginal likelihood of the observed draws**, not Brier on a point estimate: for a
route with s successes and f failures among the draws NOT in the state's history,
L = log sum_g w_g q_g^s (1-q_g)^f. This rewards tail mass when everything fails, handles unequal
draw counts (needed for the asymmetric-depth pool), and is the quantity the policy consumes.
**Calibration**: shrink the predicted histogram toward the pool-level one, coefficient on calibration.
**Metrics**: report tail quality (log-loss on all-fail episodes, calibration of P(k=0)) alongside
AUC/Brier -- AUC cannot see the tail, and the policy lives there at tight budgets.

**First attempt, measured (LCB seed 0, one ordering; `gridmatch/deepstate/lcb_s0_dist`).** A
spike-and-slab Beta head on PCA-256 features, fitted by marginal likelihood and shrunk toward the
pool prior (lambda = 0.40 on calibration marginal LL), with NO decay, vs today's beliefs:
**-111.0 / -40.2 / +7.0 / +13.8 / +21.7 / +6.0%** at 50-84%. Against the logistic reader in the same
setup (-59.8 / -18.1 / -10.5 / -4.8 / +16.4 / +16.7) it is clearly better where reading pays (70-80%)
and clearly worse at the tight end. It does have more tail (1.8% of beliefs below 2% vs 0.4% for the
logistic head, still under the decay's 5.2%).

*Why the tight end got worse, measured:* at depth 0 on test problems its RANKING matches today's
probe (AUC 0.765/0.774/0.825 vs 0.787/0.768/0.823) but its LEVELS are worse (Brier 0.120/0.122/0.113
vs 0.097/0.100/0.105). The stop test compares p*R with c, so it is level-sensitive, and both the
marginal-likelihood objective and the shrinkage optimise the whole distribution rather than its mean.

**Three requirements the policy places on a belief head, which we kept trading off:** level
calibration at ENTRY (tight budgets), tail resolution DEEP in an episode (give-up after failures),
and ranking everywhere. The design that satisfies all three: distributional output at FULL feature
dimension (no PCA), its mean affinely recalibrated per route on calibration, and any shrinkage chosen
on the POLICY objective rather than marginal likelihood -- trained on the policy's visited states.

**Build it on the recollected pool** (asymmetric draw depth makes the distribution identifiable and
depth decisions meaningful), with beliefs trained on the policy's own visited states (THREADS 0a).

### 3b-lxxxii ▶ RUNNING: can the probe learn the decay itself? (deep-history probe, 2026-09-19)

**Why.** Every belief update we use is hand-written: prompt prior × count decay $\kappa/(\kappa+n)$,
or the posterior over success counts. The history probe (§3b-lxxiv) reads only the *latest* failed
attempt, was trained on at most two failures (never two on the same model), and still leans on the
decay for every failure it did not read and for every repeat. The decay is the weakest part of the
method (§3b-lxxvii), and a small LLM reading the episode should be able to learn it.

**Design.** Prompt = problem + one-line trajectory summary ("So far the small 4B model has failed
3 times, the medium 20B model once, and the large 120B model has not been tried") + the most recent
failed attempt's code + the question suffix; last-token readout. 7,878 examples: ~12 sampled
histories per problem, 1–6 failures deep, scout first, **including repeats on one model** (up to 6).
Screen: the probe with **no decay at all** vs prompt probe + decay, broken down by how many times the
target model has already failed (0, 1, 2, 3+). Files `history_probe/deep_*`, activations
`act_deepjudge_shard*` (eai `histprobe_deepjudge*`).

**Why it matters (novelty).** If it works, the method becomes "a cheap LLM reads the episode and
outputs the belief": the scout's prefill is a *learned belief state* for the whole episode
(amortized inference), with no posterior formula. Prefill routers read only the prompt; RoR uses
counts; the abort probes read a model's *own* trajectory, not a pool's.

**Success criteria, in order.** (1) Screen: no-decay deep probe ≥ probe + decay at every depth,
especially same-model failures 2 and 3+. (2) Policy: beats D (§3b-lxxiv) at matched cost; needs a
prefill for every reachable test state (failure counts in text capped at a few; ~15–20k prefills,
sharded GPU job). (3) Holds on the new pool at recommended temperatures and on BigCodeBench.
If (1) fails: the structural prior carries something ~550 training problems cannot teach, and the
fallback is an LLM-read prior + a proper (with-replacement) posterior update.

### 3b-lxxxi Datasets: replace TACO with BigCodeBench (and CodeContests); what the literature uses

**What the closest work evaluates on.** RoR v3: MBPP+ (152 queries), LiveCodeBench, BigCodeBench.
Prefill-router: LiveCodeBench among others. RouterBench: MMLU, HellaSwag, GSM8K, ARC-Challenge,
Winogrande, MBPP, MT-Bench. Agent-as-a-Router: CodeRouterBench. SWE-Router / Scrouting: SWE-bench
(Scrouting: SWE-bench Pro). **Nobody in our related work uses TACO.**

**BigCodeBench** (HF card): 1,140 tasks, each in Complete (docstring) and Instruct (NL) prompt
styles; 77 stdlib + 62 third-party libraries, 7 domains, ~5.6 unit tests/task; v0.1.0–v0.1.4;
released June 2024 (no temporal split → random split; contamination risk). A 60/20/20 split gives
~230 test tasks (LCB has 171). Grader needs its library environment; `bigcodebench` 0.2.5 is on
PyPI and reachable from the dev box. First step: run every reference solution in our environment and
drop tasks that do not pass (environment flakiness must not look like model failure). Use Instruct.

**CodeContests**: stdin/stdout like TACO → rides the existing TACO conversion and grading path;
recognisable (AlphaCode); 2022 (contamination risk); official test split only 165 → sample from
train, own split.

**CodeRouterBench** (HF `Lance1573/CodeRouterBench`, MIT): ~10K tasks repackaged from HumanEval,
MBPP and BigCodeBench (+176 OOD), 8 frontier API models (claude-opus-4-6, claude-sonnet-4-6,
gpt-5.4, glm-5, kimi-k2.5, MiniMax-M2.7, Qwen3-Max, qwen3.5-plus), **one result per (task, model)**,
prices included. Not a main pool (single draw; mostly easy inputs). **Useful as a free, larger
"which peer" test**: one scout-prefill extraction over its prompts re-tests "not which peer" on
thousands of tasks vs our 171.

**LiveCodeBench size.** release_v6 ≈ 1,050 problems (May 2023–Apr 2025; the card documents up to
v5 = 880). We use 892 (from 2023-09): 551/170/**171 test**. Seeds reshuffle draws, not problems, so
all LCB numbers rest on 171 problems → **rolling-origin evaluation** (train before a cutoff, test the
next block, slide, pool; 535 pooled test problems when done for the capacity ablation) keeps the
temporal guarantee and triples test size.

**Plan.** Main pools: LiveCodeBench + BigCodeBench (+ CodeContests replacing TACO), all collected
once with the pilot-chosen rungs at recommended temperatures, multi-draw, our own outcomes (inputs
and tests only from the benchmarks).

### 3b-lxxx Pool pilot: which rungs, and does a bigger pool leave more margin? (2026-09-18/19)

100 LCB problems (50 per split), 2 draws per rung, recommended sampling (Qwen3 Instruct T=0.7/top-p
0.8; Qwen3 Thinking T=0.6/0.95; gpt-oss T=1.0), one prompt/extraction/grading path for every rung
(the 4B models served by vLLM inside their job). Launcher `launchers/abstention/launch_pool_pilot.sh`,
data `pool_pilot_lcb/`. Cost per draw = (prompt + completion tokens) × our serving-cost estimates;
the two Qwen MoE prices are **assumed** (scaled by total parameters from the gpt-oss estimates).

| rung | pass@1 | ¢/draw | verdict |
|---|---|---|---|
| Qwen3-4B-Instruct (scout) | 47.5% | 0.064 | keep |
| gpt-oss-20b low | 69.0% | 0.17 | keep |
| **Qwen3-4B-Thinking** | **79.0%** | 0.37 | keep — nearly 20B-medium at 70% of its price |
| gpt-oss-20b medium | 82.0% | 0.51 | keep |
| gpt-oss-120b medium | 85.0% | 2.66 | keep |
| gpt-oss-120b high | 87.5% | 12.6 | keep (top) |
| gpt-oss-120b low | 81.0% | 1.47 | drop — dominated by 20B medium |
| Qwen3-30B-A3B-Thinking | 80.5% | ~1.9* | drop — dominated |
| gpt-oss-20b high | 78.5% | 2.86 | drop — 7% truncated at 65k, 4.5% empty |
| Qwen3-235B-A22B-Thinking | 87.0% | ~31* | drop — 120B-high matches it cheaper |

Price spread ~200× (target ≥50×). Top steps buy little (+3 for 5×, +2.5 for 5×): the "sometimes"
regime. At recommended temperatures everything is much stronger: 94% of these problems are solved
by some rung within 2 draws.

**Headroom at matched accuracy** (perfect foresight vs the best fixed plan — every single rung,
best-of-2, every cheapest-first cascade):

| target | 3 rungs (scout, 20B-med, 120B-med) | 6 rungs |
|---|---|---|
| 70% | 81% | 77% |
| 85% | 73% | 78% |
| 88% | 68% | 77% |
| 90% | 61% | 73% |
| 92% | unreachable | 64% |

The bigger pool keeps headroom high exactly at the top, where our margins were thinnest, and extends
reach. Caveat: 100 problems × 2 draws is noisy; half the problems are from the easier training
period. **Recommended pool: the 6 kept rungs** (+ Claude only for the "Claude for everything" framing,
§3b-lxxix).

### 3b-lxxix Framing idea: "$ saved vs Claude for everything" (parked)

Deployment pitch: the default is a frontier model on every query; our machinery redirects instances
to cheaper models. In the MDP: Claude is the top route and the give-up action becomes "send it to
Claude"; the prefill's difficulty signal then answers "skip the cheap rungs on this one". Works only
if Claude is **clearly** more accurate than our best cheap route — on the old 4k-cap LCB setup Opus 5
got 43% vs gpt-oss-120b's 39%; on today's pool gpt-oss-120b is ~68–70% on test. If Claude is only
marginally better, "Claude for everything" is a strawman; always also report vs the Zero Router
*including* Claude and vs cascades ending in Claude. Keep an explicit give-up only for problems
predicted hopeless even for Claude. Next step if pursued: add Opus 5 (`anthropic/claude-opus-5`) and
Sonnet 5 to the pilot (100 × 2 draws) to measure accuracy and cost.

### 3b-lxxviii Resampling is essential for accuracy but selective; why we rarely resample the 120B

LCB test, % solved: scout 29.2 / 31.0 / 35.7 (1 draw / best of 2 / best of 6); 20B 56.1 / 64.3 /
**74.9**; 120B 69.6 / 75.4 / **83.6**; pool 73.1 (one draw each) → **84.8** (all draws). Resampling
buys ~12 points of what the pool can reach — the whole high-accuracy regime (single-commit routing
cannot reach 70%). But per problem the 120B **always** solves 48.0%, **sometimes** 35.7%, **never**
16.4%: resampling pays only on the "sometimes" third. The value of resampling hinges on telling a
fluke from a never after a failure — the right framing is not "resample or reroute?" but "after
*this* failure, is resampling a recovery or a waste?".

**We rarely resample the 120B at the Figure-1 point, correctly.** After a first 120B failure, ours
takes another 120B draw 13% of the time at 68.8% (R = \$0.21), 54% at 80% (R = \$0.45), 86% at 84%
(R = \$0.79); RoR v1 7% / 32% / 83%. A 120B draw costs ~4.5¢ and succeeds ~31% after one failure, so
at the price implied by a 69% target (R ≈ 8–21¢) a second draw is break-even at best.

### 3b-lxxvii Why the decay does not work: bimodality and independence (measured)

All 892 LCB problems; P(next draw succeeds | first n draws on the same model failed):

| scout | real | count decay |
|---|---|---|
| n = 0 | 42.0% | 42.0% |
| n = 1 | **6.6%** | 20.5% (3× too optimistic) |
| n = 2 | 2.1% | 13.5% (6×) |
| n = 5 | 2.0% | 6.7% |

(120B: 80% → **32%** real vs 39% believed after one failure; the shape is a sharp drop then a
plateau, which no single $\kappa$ fits.) After one scout failure 85% of the remaining problems are
ones the scout **never** solves: problems are bimodal, and a single Beta prior (mean + concentration)
cannot express "one failure ⇒ probably never". **Cross-model:** where the 120B's first draw failed the
scout succeeds 6.7% vs 42% overall; independence keeps the scout's belief unchanged, so the policy
buys cheap "lottery tickets" after the strongest model failed.

**Consequence seen in the trajectories.** With draws at ~0.07¢ the give-up needs beliefs below
~0.3–0.8% to fire; hyperbolic decay from ~30% takes dozens of failures. So at the Figure-1 point our
cheapest configuration stops mainly via the **cap** (25% of problems end at the cap after ~4.5 draws,
5.7% by choice); the price-only arm at the same accuracy gives up by choice on all 30.4% of its
unsolved problems but only after **6.5 draws** on average and costs 3.72¢ vs 2.80¢. The stopping rule
is optimal given its beliefs (§3b-lxxii monotone case); the beliefs after failures are wrong, and the
cap patches that from outside. Failure-reading (§3b-lxxiv) and the success-count posterior
(§3b-lxxvi) both attack exactly this.

### 3b-lxxvi Posterior over success counts: helps, but the replay version had a finite-pool edge

The head predicts P(k = j | x), j = 0…6 successes among a model's 6 stored draws; after n failures
P(k=j | n) ∝ P(k=j)·C(6−j, n)/C(6, n) and p_next = Σ P(k=j|n)·j/(6−n). A full distribution can put
mass on "never" and "always" separately; one failure rules out "always" instantly (example: prior
never 40 / always 40 / coin-flip 20 → 50% → **12%** after one failure; decay 50% → 24%).

**Matched (constant costs, one seed; `gridmatch/voi/v`)** vs the Beta decay: +16.8 / +0.8 / +8.9 /
+4.0 / −2.1% at 50–84% myopic; with 2-step lookahead +16.8 / +0.1 / +9.5 / +8.5 / +2.1% — and with a
better belief model the lookahead starts to help at the loose end (it did not over the decay).

**⚠ Finite-pool artifact.** That update is exact Bayes for sampling *without replacement* from the 6
stored draws — replay-only knowledge. With k=1 of 6 and 5 failures seen it "knows" the last draw
succeeds; its belief even **rises** with repeated failures (2.9% → 3.2% in the example, against 2.4% →
1.2% with replacement). Not test leakage (prior from the prompt, trained on train), but an advantage
this arm has in replay and not in deployment, largest deep in an episode — where its 70–80% gains came
from. **Fixed:** `--posterior-with-replacement` (p = Σ π_j q_j(1−q_j)ⁿ / Σ π_j(1−q_j)ⁿ, q_j = j/6),
one helper for the per-step belief and the lattice. Rerun running (`gridmatch/postwr`, 5 seeds). The
count decay has no such artifact (never references the pool size).

### 3b-lxxv Whether × which: the value is almost all in knowing when to stop

Crossed arms (constant costs): the stop decision taken on one belief source, the route choice on the
other (`--whether-which-arms`; LCB 5 seeds, TACO 3), saving vs count beliefs for both:

| prefill used for… | LCB 50 | 60 | 70 | 75 | 80 | 84% |
|---|---|---|---|---|---|---|
| only **whether** (routes by counts) | +40.9 | +22.6 | +16.7 | +9.1 | +2.1 | +1.6 |
| only **which** (stop by counts) | −10.5 | −6.3 | −8.9 | −12.9 | +0.4 | +0.3 |
| both | +40.5 | +19.6 | +16.5 | +6.4 | +4.6 | +4.8 |

Prefill route choice *added to* prefill give-up: LCB −0.6 / −3.9 / −0.2 / −3.1 / **+2.5 / +3.2** (5/5
at 80–84%); TACO **+13.5 / +10.3** at 35/40% (3/3), −8 to −9 at 45–55% (0/3). So "which tier"
pays only in narrow regimes. **Title consequence:** "Which Tier, Not Which Peer" (current tex title)
overclaims; "Whether, Not Which" was accurate; rhyming alternative "Pick Your Fights, Not Your
Knights". The earlier 7–13% of route-choice value without a give-up action was mostly route choice
doing the give-up's job.

### 3b-lxxiv Failure-reading is a clear win on LCB (history probe; §3b-lxxxii takes it further)

**Mechanism.** After a failure the scout prefills the problem + the failed attempt + the question
*"given the problem and any failed attempts above, will another attempt solve this problem? Answer
yes or no."*; we read the **last token** and fit the same per-model linear heads (labels: each
model's success rate on its remaining draws, excluding the draws in the history). Arms: A = prompt
probe + count decay; B = history probe (reads the latest failure; decay for failures it did not
read; re-prefill charged in full, median \$0.00029); C = A recalibrated on which models failed (no
content); D = C + B.

**What carries it (predictor level, 120B after a failure, AUC):** prompt + decay 0.741; + which models
failed with the question 0.750; **+ the failed code with the question 0.800**; + code + "passed k of N
tests" 0.798. Without the question, mean + last readout, the failed code *hurts* the policy (−12 /
−18% at 50 / 60%). ⇒ **the failed program read as an answer to a yes/no question carries it; test
counts add nothing** (no special verifier needed).

**Policy, LCB, 5 seeds, constant costs (code only):** D vs A **+9.3 / +10.1 / +14.3 / +21.4 / +16.7 /
+8.4%** at 50–84%, 5/5 everywhere; D vs counts +54.3 / +28.7 / +27.2 / +25.7 / +17.5 / +14.5.
Trajectory-only history (which models failed, no code): D vs A −10.6 … +5.4 — not the source.

**Full method with failure-reading** (cost head, cap × price, 5 seeds) vs baselines:
agreement +31.9 / +36.8 / +26.2 / +16.3 / **+0.2** / +9.5 (was −11.4 at 80%); RoR v1 +30.5 / +34.4 /
+27.1 / +19.1 / **+8.0** / +12.2; Zero Router +32 to +40%. At gpt-oss-120b's accuracy **55% cheaper
than always calling it** (was 49%). vs the paper's current method, same seeds: −7.9 at 50% (0/5),
+7.4 to +11.1 at 60–84% (5/5). Caveats: LCB only; T = 0.2 makes a failed program unusually predictive
of the same model's next draws (recollection at recommended temperatures will tell).

**TACO failed first, from a bug in C/D, not in the reading.** C alone −48 to −72% at 35–45% while D
vs C was +9 to +25%: the recalibration was fitted on examples with ≤1 failure per model, **linear in
the failure count**, and the replay extrapolated it to 5–6; several TACO weights were positive, so
beliefs *rose* with every failure and the policy never stopped (LCB's weights happened to be
negative). **Fix:** fit and apply on 0/1 "this model has failed"; repeats stay with the decay
(`history_probe_eval.py`, replay `indicator` flag). Rerun `gridmatch/hist3` (LCB + TACO, constant and
full method, TACO full method for the first time) — running.

### 3b-lxxiii Linked cap B = k·R: the fair one-knob version of the two-constraint family

The (cap × price) family has 16 × 96 = 1,536 operating points and its hull is taken on test, vs ~96
per baseline (violates matched comparisons). Linked cap: "never spend more on one problem than k
times what its answer is worth" — one 96-point price sweep. k chosen on the **calibration** split
(min mean log cost over the common accuracy range, averaged over seeds; `select_linked_k.py`):
**k = 0.5 on LCB** (+15.7% vs no cap over the range), **k = 0.35 on TACO** (+4.5%).

Test (5 / 3 seeds): the 16× grid bought only **2–6%**. LCB linked vs agreement +34.5 / +30.3 / +15.0 /
+3.9 / **−11.0** / −3.0; vs RoR v1 +33.1 / +27.6 / +16.1 / +7.1 / −2.3 / +0.1; vs Zero Router +28 to
+35%. TACO strong at 35–45% (+37 to +50 vs agreement), losing at 50–52%. The high-accuracy loss is a
belief problem, fixed by failure-reading (§3b-lxxiv); **the fair headline needs the linked-cap
version of the failure-reading method (not yet run).**

### 3b-lxxii The one-step rule stops exactly where the optimal policy stops (monotone case)

Under the belief model a failure lowers only the failed model's belief and costs are fixed, so
$Q^{\pi_\bot}(\mathbf n',m)\le Q^{\pi_\bot}(\mathbf n,m)$ for every reachable $\mathbf n'$. If
$\max_m Q^{\pi_\bot}(\mathbf n,m)\le0$ then $V^*=0$ at every reachable state (backward induction) and
the optimal policy stops; if some $Q^{\pi_\bot}>0$ then $V^*\ge Q^*\ge Q^{\pi_\bot}>0$ and it continues.
So the myopic rule's stop set equals the optimal stop set (Chow–Robbins–Siegmund monotone case);
lookahead can only change route order. Explains why deeper lookahead never helped over the decay.
Terminology now in the tex: our rule is greedy on $Q^{\pi_\bot}=p_mR-c_m$ (the action value of
always-abstain), i.e. one step of policy improvement / value iteration from $V_0\equiv0$; $h=H=18$ is
exact $Q^*$; "optimal at its own spend" holds exactly for the depth-$H$ policy only. **Note:**
beliefs that can *rise* after a failure (history reading, finite-pool posterior) break monotonicity,
so lookahead may matter again with them.

### 3b-lxxi Cap-grid truncation in the two-constraint family (bug, fixed) and the corrected numbers

The two-constraint family swept caps only up to the expected-cost exhaustion point, while the
baselines' grid had been extended to the realised ceiling (§3b-lxvii) — our family stopped at 79.6% on
LCB and the price arm alone represented us above it. One shared ceiling now (`_top`), 16 caps
(`gridmatch/fixedacct2`). LCB vs agreement +36.9 / +31.5 / +16.8 / +9.3 / −11.4 / +2.3; the loss band
78–83.5% narrows but remains (the price-only arm is weak there). **Same-model cost update** across
seeds (`costupd2`): LCB ours +4.7 / +8.0 / +6.9 / +3.9 / +5.0 (5/5) → vs agreement at 80% −11.4 → −5.7;
**TACO −5 to −6% at 45–52% (0/3)** — pool-dependent like the cost head. **Probe penalty chosen on
calibration AUC** (deployed activations, 3 seeds, beliefs only vs counts): +51.0 / +26.4 / +19.1 /
+8.0 / +4.4 / +7.7 vs fixed-C +40.8 / +21.2 / +15.5 / +6.6 / +4.6 / +5.2 — a free +3–10 points, not yet
in the method. Per-pool configuration chosen on calibration is the principled way to use these.

### 3b-lxx TACO: the scout → 20B → 120B cascade touches our frontier at one point

Frontier vs the Zero Router (`overleaf/figures/fig_frontier_zero_router.pdf`, seed 0): 31% cheaper at
gpt-oss-120b's accuracy on LCB, 22% on TACO. Across 3 TACO seeds we never lose to the Zero Router:
+27 / +18 / +28 / +18 / **+6** / +22 / +41 / +61% at 40 / 45 / 48 / 50 / **52** / 53 / 54 / 55%
(3/3 each). The dip is where the fixed cascade lands (52.1–53.8% at ~4.3¢): with a verifier, "try
each tier once, cheapest first" is a crude adaptive policy and on a 3-rung pool it is near right at
one budget. It cannot resample, so the Zero Router tops out at 55.4%. Belief-only arm dips to +5% too
(not the cost head). Motivates the bigger pool (§3b-lxxx).

### 3b-lxix Trajectory figures: when each method stops, and where it spends

`--dump-trajectories` (per-episode route sequence, outcome, abstention, spend) on LCB seed 0; each
method at its cheapest single operating point reaching gpt-oss-120b's accuracy (68.8%): ours 2.80¢,
RoR v1 3.13¢, random 3.41¢ (agreement-gating dropped from these figures). Figures `traj_giveup`,
`traj_flow` (weighted flow graph), `traj_draws` in `overleaf/figures/`.

- Our unsolved problems mostly end after **2 draws** (scout + one bigger model); RoR v1 ends most after
  **7+ draws**, mostly on the 20B, when the cap binds. Share of spend on unsolved problems: ours 64%,
  RoR v1 70%, random 66%.
- 120B usage: ours 28.4% of problems, first 120B draw at **draw 2** (median); RoR v1 **11.9%, draw 9**
  (density rule: the 120B costs 40× the scout, so $p/c$ is tiny until cheap beliefs decay); random
  39.4%, draw 2 but blind.
- At this operating point ours stops mostly via the cap (5.7% by choice, 25% at the cap) — §3b-lxxvii.

### 3b-lxviii Bibliography verified against arXiv; paper rewritten (2026-09-18)

33 of 50 bib entries had `{TODO-verify}` authors (rendering "TODO-verify (2026)"); AutoMix's author
list was wholly wrong; Gergatsouli had two invented co-authors; several titles belonged to other
papers (e.g. `rasch2026` is a co-failure-ceiling paper, `perquerycost2026` an explainable-routing
paper that does not predict per-query cost; `coderouterbench2026` is Agent-as-a-Router). Every arXiv
entry now checked against the arXiv API; EET could not be found (dropped); EGTP was a duplicate. tex
rewritten findings-first with baselines that never abstain, realised accounting, 14/14 isolation, the
monotone-case result and precise value-function terminology; numbers still from `fixedacct` (pending
list in a comment block at the top of `overleaf/tmlr.tex`).

### 3b-lxvii Matched accounting, all seeds: we beat agreement-gating 5/5 at every LCB target it can reach

Every capped arm now checks its cap against **realised** spend, as agreement-gating always did (the
asymmetry that had us at −17.4% against it at 70% on seed 0). "Ours" is the union of the price arm
and the two-constraint arm — both are our method. Mean ± sd across seeds; positive = ours cheaper.

**LiveCodeBench, 5 seeds**

| ours vs | 50% | 60% | 70% | 75% |
|---|---|---|---|---|
| **agreement-gating** | **+35.7±3.6** | **+28.4±2.3** | **+18.4±5.2** | **+10.7±5.1** |
| count beliefs + cap (RoR v1) | +34.4±4.1 | +25.3±3.5 | +19.6±5.3 | +13.3±2.8 |
| random allocation | +30.4±5.3 | +12.9±3.8 | +24.0±6.3 | +16.0±3.1 |
| *seeds positive vs agreement* | *5/5* | *5/5* | *5/5* | *5/5* |

**The seed-0 agreement result replicates and strengthens**: +18.4±5.2 at 70% on average, not +9.6%,
and positive on every seed at every target agreement can reach.

**TACO, 3 seeds**

| ours vs | 35% | 40% | 45% | 50% | 55% |
|---|---|---|---|---|---|
| **agreement-gating** | **+45.9±4.5** | **+48.8±1.3** | **+39.1±1.7** | +0.0±5.7 | −4.4±10.6 |
| count beliefs + cap (RoR v1) | +43.6±4.2 | +40.9±1.8 | +25.6±1.8 | −2.0±6.0 | +10.7±1.5 |
| random allocation | +44.1±0.8 | +41.0±1.6 | +28.6±2.9 | −4.5±11.4 | +17.8±3.7 |

**Strong at the tight targets (3/3 seeds, +39 to +49%), a tie at 50%, noisy at 55%.**

**The 50% tie is NOT the cost head.** §3b-xlv found the cost head harmful on TACO above 40%, so the
obvious suspicion. Checked: the belief-only price arm is **also** behind agreement at 50%
(−8.3±6.9) while beating it at 55% (+6.2±8.7); the cost head is what costs us at 55% (−15.2±9.7).
So TACO at 50% is a genuine near-tie with agreement, not a component defect.

**This partly reverses §3b-xlv's TACO retraction — but attribute it correctly.** Against counts the
full method is now +43.6/+40.9/+25.6/−2.0/+10.7, against the retracted +35.5/+20.2/−5.7/−12.5/−31.8.
Two things changed at once: the accounting fix (applied to **both** arms) **and** "ours" now includes
the two-constraint arm, a method component that did not exist when §3b-xlv was written. It is not
"the accounting fix rescued TACO."

**Top-end numbers (80%, 84%) withheld pending a re-run.** Under realised accounting the budget grid
still topped out at exhaustion in *expected* costs, so every budget-swept arm was truncated (counts
reached 79.6% instead of 84.8%) and the comparisons at 80%+ came out `n/a` for an artefactual reason.
Grid fixed; all 8 seeds re-running. The 50–75% columns above sit well below that ceiling and are
unaffected.

### 3b-lxvi ⚠ RouterBench is WEAKER than the headline numbers suggest — the probe is largely learning task identity

**Do not quote the +22.9% to +76.9% cost savings.** They are computed on a **pooled** evaluation and
are inflated by task identification, not difficulty prediction.

**The evidence.** Variance decomposition of the probe's own predictions on the held-out split, by
dataset:

| | between-dataset share of variance |
|---|---|
| **the probe's predictions** | **0.681** |
| **true success** | **0.154** |

The probe puts **68% of its variance on a dimension carrying 15% of the real variance.** Training is
**pooled over ~85 datasets** (HellaSwag, GSM8K, 60+ MMLU subjects, MBPP, Chinese riddles…), so
"which dataset is this" is by far the cheapest signal to learn, and a pooled evaluation pays for it:
route GSM8K to the math-strong model, HellaSwag elsewhere. Real routing, but **task** routing.

**Per-dataset evaluation removes it and the gain collapses:** 12.67pp mean vertical gain pooled →
~2–5pp per dataset → **+3.91% AIQ**, which `PRIOR_ART.md` §7 already places in the regime where
RouterBench's *own* KNN and MLP routers *"generally do not significantly outperform the Zero
Router"*. Per-dataset we win on 5 of 6 headline sets and tie/lose on winogrande.

**Report AIQ (+3.91%) as the RouterBench result, and nothing else.** State the decomposition as a
limitation rather than letting a reader find it.

**Two things this does NOT undermine.** (i) The shuffled control still drops hard on every dataset
(MMLU 0.7472→0.6932, HellaSwag 0.8244→0.7023), so the probe carries real per-problem information —
the question is only how much of it is task-level. (ii) **LCB and TACO are single-task pools**, so
the dataset-identity shortcut was never available there and their results cannot be contaminated
this way.

**The fix, if we want RouterBench to carry weight:** train per-dataset, or residualise dataset
identity out of the target, forcing the probe onto the within-task difficulty that the truth is 85%
made of. Either outcome is informative. *Not run.*

### 3b-lxii Baselines vs controls: `counts_value` is NOT a baseline, and the price cannot be ablated of abstention

**Two structural facts that reorganise the paper's comparison section.**

**1. `counts_value` abstains up to 73.2%** (41.6% at the 60% target), so it is count beliefs plus
**our** give-up mechanism. Putting it in a baseline table hands the baseline our contribution and
then reports a modest margin — backwards. It is the **control** for the belief ablation (formulation
held fixed, only the belief source differs) and belongs in the ablation section. Arms with **zero**
abstention are the baselines: `counts` (RoR v1), random allocation, budget-aware best-of-K,
agreement-gating, and the fixed-model hull.

**2. A price-swept policy cannot be ablated of abstention.** Removing the stop action from
`counts_value` collapses all 35 of its zero-abstention operating points to a **single point** —
\$0.17275–\$0.17324, accuracy 84.8%–84.8%. Ours likewise (9 points, \$0.1686–\$0.1698, 84.8%).
**The price controls spend only through the stop decision**; with no stop, every $R$ produces the
same behaviour — buy everything, hit the ceiling.

| budget control | stop action | result |
|---|---|---|
| **cap**, no price | none | a curve — **this is `counts` / RoR v1** |
| **price**, no stop | none | **degenerate: one point** |
| **price** + stop | zero-value action | a curve — **ours** |

**So the cap and abstention are alternative mechanisms for budget control, not independent
features.** This is why RoR *needs* a cap: its density rule $p/c$ never crosses zero, so it has no
stop, so it has no other way to regulate spend (§3b-xxxvi). **Put this table in the paper** — it
pre-empts the "ablate abstention" request by showing the ablation is not constructible, and it
reframes the formulation axis as *"is spend better controlled by refusing to pay past a cap, or by
declining problems that are not worth it?"*

### 3b-lxiii The per-episode cap does NOT bound realised spend — a third of RoR's episodes exceed it

The feasibility check is `spent_budget + cost_est[mi] > budget`, where `spent_budget` accumulates
**estimated** cost while money is charged at **realised** cost. Measured on RoR's own arm:

| cap $B$ | max realised | ratio | episodes over $B$ |
|---|---|---|---|
| \$0.00084 | \$0.01868 | **22.3x** | 31.5% |
| \$0.01508 | \$0.16670 | 11.1x | 38.9% |
| \$0.03643 | \$0.35019 | 9.6x | 35.0% |

**Retract any claim that the cap is a hard per-episode guarantee** — including one made in session.
It bounds *planned* spend only.

**Per-episode spend distribution at ~70% accuracy** (the comparison the mean hides):

| arm | acc | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| `counts` (cap) | 71.5% | \$0.04900 | \$0.01197 | \$0.20690 | \$0.31182 | \$0.35019 |
| **ours** (price) | 69.6% | **\$0.03723** | **\$0.00947** | **\$0.16892** | \$0.31515 | \$0.55408 |

**We are cheaper at the mean, median and p95, and worse only in the extreme tail.** Report the
distribution, not the mean: it costs nothing, it answers the obvious deployment question, and the
finding is favourable. *(Indicative — from an older run whose traces survived, not the matched grid.)*

**`--cap-on-realised` checks the money actually spent**, bounding realised spend to $B$ plus at most
one draw's overshoot. Its **tightness depends on predicting the next draw's cost**, which a
per-problem cost head does and constant per-route costs cannot — so a genuinely binding cap is
something only our cost head enables. *In flight.*

### 3b-lxiv Optional accept: letting the policy distrust a verifier recovers 6.3 points of ceiling

*(Weak-verifier regime; kept out of the headline per §3b-lvii but recorded.)*

Forcing a stop on every weak PASS is not a decision problem — it is noise the policy cannot act on.
Letting a PASS **bank** a candidate worth $R\,q_m$ and continue:

| regime | 50% | 60% | top accuracy |
|---|---|---|---|
| oracle verifier | +40.8% | +19.2% | 84.8% |
| weak, **forced** accept | +19.6% | +10.5% | **62.1%** |
| weak, **optional** accept | +19.1% | +2.4% (+20.4% with cost head) | **68.4%** |

**6.3 points of ceiling recovered**, about 29% of the 21.5 available to the pool's 83.6%. Partial —
and the reason is our own thesis unapplied: $q_m$ is a **pool-level** constant (0.725 / 0.939 /
0.951), so the policy knows *"a scout PASS is 72% reliable in general"* but not *"**this** one is
not."*

**Per-problem $q$ is predictable, and verifier failure is systematic rather than random.** ICC of
per-problem $q$ — the between-problem variance share with binomial noise removed — is **0.964 /
0.856 / 0.882** across the three routes. **Verifier failure is a problem property, not draw noise.**
The mechanism is visible in the construction: `weak_verifier_outcome = public_resolved`
(`build_mdp_tensors_v2.py:206`), so a false accept means *the public tests do not cover what the
hidden tests catch* — fixed across every draw on that problem. A ridge head on the same prefill
predicts it at test correlation **0.622 / 0.436 / 0.355**. *Policy run in flight.*

**Boundary to state:** this holds for *public-test* verifiers. A sampling verifier — an LLM judge,
or self-consistency agreement as RoR v1 used — would have far lower ICC and correspondingly less
predictable $q$. Claim it for systematic verifiers, not for all weak ones.

### 3b-lxv Figures: the frontier is the wrong headline plot

Two presentation findings worth carrying into the writing.

**1. A frontier plot cannot show this result.** Two policies over the same pool trace nearly the
same curve *by construction*, and a 20% cost saving is a small horizontal shift on any axis. **Log-x
makes it strictly worse** — it linearises the concavity, so the arms render as near-parallel
diagonals and the saving reads as a uniform offset. Linear-x at least shows the diminishing returns
and renders the saving as a visible lens. **Lead with cost-saved-vs-accuracy; demote the frontier to
a supporting panel.**

**2. The five-target table hides a trough.** Plotted continuously, the advantage over `counts` is
**strongly non-monotone**: +11.5% at 78%, **+5.3% at 80%**, **+0.6% at 81.2%**, +1.6% at 82%, +7.5%
at 84%. The quoted targets straddle the trough. Nothing dishonest happened — but **report the curve,
with the quoted targets marked on it**, or a reviewer who plots it finds the trough first.
Provisional diagnosis is hull-vertex placement (the §3b-xxxiii mechanism), **not yet verified** —
and that diagnosis was wrong once already.

Figures are generated by `pipelinerl/swe/scripts/livecodebench/make_paper_figures.py` into
`analysis/paper_figures/` as SVG; every number is read from replay outputs except the two
single-split pools, which are marked in the source.

### 3b-lvii DISCLOSURE LARGELY RETRACTED for execution-verified domains — and what replaces it is cleaner

**The objection, and it is correct.** In a domain with a cheap verifier, a delivered wrong answer
and an announced decline are *the same object*: you run the unit tests, they fail, and you now know.
Converting one into the other costs a test execution. §3b-xliii's headline — "we announce 100% of
our failures and RoR announces none" — **does not survive this** on LiveCodeBench, TACO or
SWE-bench Verified, because in all three the policy's own success signal *is* test execution. RoR
finishes its budget knowing every draw it made failed; it can say so for free.

**Scope of the retraction.**

| pool | cheap verifier at decision time? | is the disclosure claim available? |
|---|---|---|
| LiveCodeBench | yes — tests | **no** |
| TACO | yes — tests | **no** |
| SWE-bench Verified | yes — tests | **no** |
| RouterBench | **no** — MMLU/ARC-style graded answers | yes, but it is the one pool with no resampling |

So the disclosure framing survives only where we have the *least* interesting policy, and must be
dropped from the headline. **What was true and stays true:** RoR has no stop action, so it *spends*
its whole cap before discovering the failure, while we stop early. That is a **cost** difference,
which the cost claim already measures — not a separate deployability axis.

**The good news, and it repairs §3b-xl.** If the verifier converts wrong answers to declines, the
honest accounting charges **every failure equally** rather than charging only *our* abstentions.
And at matched accuracy the failure rate is equal **by construction**, so this adds a near-identical
constant to both arms:

| charge per failure | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| \$0 (as reported) | +48.6% | +20.6% | +16.7% | +5.3% | +7.5% |
| \$0.02638 (= a full gpt-oss-120b call) | **+26.0%** | **+14.1%** | **+14.2%** | **+5.0%** | **+7.3%** |
| \$0.05277 (2x) | +18.5% | +10.7% | +12.3% | +4.7% | +7.1% |
| \$0.26383 (**10x**) | **+6.0%** | **+6.0%** | **+6.0%** | +3.4% | +5.9% |

**The advantage cannot be priced away.** Adding a constant to both numerator and denominator drags
the ratio toward 1, so the percentage shrinks — but the sign cannot flip, and even at **ten times**
the dearest route's price per failure we remain +3.4% to +6.0% at every target. §3b-xl's alarming
−38.5% came entirely from an **asymmetric** charge: pricing our give-up while letting RoR's failures
go free. Under a verifier that is simply the wrong accounting.

**Lesson for the paper.** Do not claim a deployability axis that a domain's own verifier collapses.
State instead: *in verifier-rich domains the benefit is purely cost, and it survives charging every
failure at ten times the dearest route.* Reserve the disclosure argument for verifier-poor settings
and mark it explicitly as an extrapolation we have not measured.

### 3b-xliii Failure disclosure: at matched accuracy we convert ~100% of failures from shipped wrong answers into announced declines

**Why this is a pure substitution question.** At matched accuracy the failure *rate* is equal by
construction — both arms fail on the same share of problems. The arms differ only in **what kind of
failure** they produce. So the quantity to report is the **disclosure rate**, abstentions / failures.

| accuracy | RoR wrong | RoR declines | RoR disclosure | ours wrong | ours declines | **ours disclosure** |
|---|---|---|---|---|---|---|
| 35.0% | 65.0% | 0.0% | 0.0% | 0.0% | 64.9% | **100.0%** |
| 65.0% | 35.1% | 0.0% | 0.0% | 0.0% | 35.9% | **100.0%** |
| 75.0% | 24.4% | 0.0% | 0.0% | 0.0% | 25.0% | **100.0%** |
| 80.0% | 19.2% | 0.0% | 0.0% | 1.8% | 18.5% | **91.3%** |
| 84.8% | 15.2% | 0.0% | 11.7% | 3.5% | — | 23.1% |

**Below 80% accuracy the substitution is one-for-one and total:** RoR's wrong-answer rate and our
decline rate agree to within a percentage point at every target, and we announce **100%** of our
failures while RoR announces none — *while also costing 5–49% less*. The advantage collapses at the
very top (23.1% at 84.8%), where we too are shipping wrong answers; the claim is therefore a
**budget-constrained-regime** claim, which is the regime the paper is about.

**The exchange rate, which is how to defend it.** Charge $W$ per wrong answer shipped and $A$ per
decline announced, fold both into the cost axis and rebuild the hulls (linear in the rates, so
mixtures carry the chord of their charges):

| $W$ | $A$ | 50% | 60% | 70% | 80% |
|---|---|---|---|---|---|
| \$0 | \$0 (as reported) | +48.6% | +20.6% | +16.7% | +5.3% |
| \$0 | \$0.0264 | −38.5% | −25.7% | −0.9% | +0.4% |
| \$0.0264 | \$0.0264 (equal) | **+26.0%** | **+14.1%** | **+14.2%** | **+5.0%** |
| \$0.0528 | \$0.0264 (2x) | **+49.5%** | **+34.7%** | **+25.4%** | **+9.5%** |

**Break-even: a wrong answer need only cost 0.44x a decline at the 50% target, 0.55x at 60%, 0.05x
at 70%, and at 80% we win even if wrong answers are free.** So:

> Under any accounting in which shipping a wrong answer costs at least **0.55x** what announcing a
> decline costs, our advantage holds at every target measured — and if a wrong answer costs twice a
> decline, the advantage *grows* to +9.5% to +49.5%.

That is a weak premise. In every deployment we can think of, a silently wrong patch costs *more*
than a flagged one — review time, and false confidence — so $W > A$, i.e. $W/A > 1 \gg 0.55$.

**Attribution update (ROI-Reasoning re-read, 2026-09-14).** Earlier sections of this document
attribute "the formulation column" and "abstention-under-a-global-budget" to ROI-Reasoning. A
re-read shows that **over-conceded**: they never predict a success probability or a continuous cost
(*"No probability output; ROI is implicit in the learned policy"*; difficulty is a 4-level ordinal
tag proxying **cost**), they do not solve a knapsack (they name OS-MCKP and train a policy by
Dr. GRPO — *Greedy Knapsack is one of THEIR baselines*), and their skip is a fine-tuned output token
`\boxed{NA}`, not a computed zero-crossing. So the rule $\max(0,\max_m(p_mR-c_m))$ is **not in their
paper**; it is the textbook Lagrangian dual of a constrained allocation. **Cite them for the goal —
budgeted inference over many tasks with skipping available — and cite standard constrained-MDP
material for the rule. Claim neither.** See `PRIOR_ART.md` §4b. What this does *not* change is the
empirical finding below, which is about our own arms.

**Disclosure belongs to the policy CLASS, not to our representation — check before claiming
it.** `counts_value` (count beliefs + the give-up action on a global price) also discloses **100%**
of its failures at 64.9% and 75.6% accuracy. So the 100%-vs-0% gap is the value of *having* a
give-up action, which `PRIOR_ART.md` §4b records as **ROI-Reasoning's**, not ours. Stated correctly:

| | vs **RoR as published** | vs **`counts_value`** |
|---|---|---|
| disclosure at ~65% / ~75% | **100% vs 0%** | **100% vs 100% — ties** |
| cost at that accuracy | \$0.02609 vs \$0.03179 / \$0.05303 vs \$0.06566 | \$0.02609 vs \$0.03023 / \$0.05303 vs \$0.06511 |
| our advantage | disclosure **and** −18/−19% cost | **cost only, −13.7% / −18.6%** |

So: *"we announce every failure and RoR announces none"* is true, large, and a fair thing to say
about the prior method — but it is **not evidence for the contribution**, because any policy in this
class gets it. **What is ours is delivering the same disclosure for 14–19% less money.** Do not let
the deployability argument smuggle the formulation back in as ours; that is the error §3b-xxxvi
exists to prevent.

*One narrow place where disclosure is genuinely ours:* at the loose end `counts_value` **cannot
sustain abstention at all** — it discloses 0.0% at 84.8% and cannot reach the 80% target, while we
still disclose 91.3% at 79.8% and 23.1% at 84.8%. Better beliefs let the give-up keep firing where
count beliefs force it off.

**This is the strongest defensible form of the claim, and it repairs §3b-xl.** Charging for
abstention alone made the tight-budget win vanish (−38.5% at 50%), because it prices our mechanism
and gives RoR's failures a free pass. Charging *both* failure modes — which any honest accounting
must — restores it at **+26.0% / +14.1% / +14.2% / +5.0%** even at a 1:1 price, and that version
survives the objection rather than dodging it. **Lead with the symmetric accounting, not with free
abstention.**

### 3b-xlii Wasted spend: everyone wastes two thirds of it; what differs is what the waste buys

Dollars spent on episodes that produced no correct answer, from per-episode traces (LCB seed 0,
matched grid). This is the quantity neither cost-at-matched-accuracy nor conditional accuracy
exposes.

| policy | accuracy | total | **wasted** | % of spend | on declines | on wrong answers |
|---|---|---|---|---|---|---|
| RoR as published | 74.2% | \$0.05860 | \$0.04162 | 71.0% | \$0 | **\$0.04162** |
| `counts_value` | 74.2% | \$0.05596 | \$0.04162 | 74.4% | \$0.04162 | \$0 |
| **ours** | 75.0% | \$0.05303 | **\$0.03489** | 65.8% | \$0.03489 | \$0 |
| RoR as published | 64.9% | \$0.03179 | \$0.02416 | 76.0% | \$0 | \$0.02416 |
| **ours** | 64.1% | \$0.02609 | **\$0.01809** | 69.3% | \$0.01809 | \$0 |
| RoR as published | 35.7% | \$0.01212 | \$0.01163 | **95.9%** | \$0 | \$0.01163 |
| **ours** | 34.7% | \$0.00305 | \$0.00264 | 86.5% | \$0.00264 | \$0 |

**Correct the rhetoric.** This document (and a claim made aloud in session) framed the difference as
*"RoR spends the money and then fails; we decline before spending."* Half right. We also spend
before declining — every dollar of our waste sits on episodes we eventually abstained on, meaning we
bought draws, they failed, and only then did we give up. What is true is narrower and should be
stated that way:

1. **Waste is the norm, not the exception, for every arm** — 63.5% to 95.9% of all spend, rising as
   the budget tightens. No policy in this class is efficient in the sense a reader might assume.
2. **At matched accuracy we waste 16–18% fewer dollars** (\$0.03489 vs \$0.04162 at ~74%;
   \$0.09494 vs \$0.11634 at 84.8%). Real, and the same order as the headline cost gap rather than
   a separate larger effect.
3. **The kind of waste is categorically different.** RoR's waste is 100% on *wrong answers it
   delivered*; ours is ~100% on *declines it announced*. Same money, different product — and that
   is a deployment argument, not a cost argument.
4. At the top operating point `counts_value` wastes **exactly** what `counts` does (\$0.11634),
   because at maximum $R$ it never abstains and degenerates to `counts`. A useful sanity check that
   the two arms coincide where they should.

*Report this as a panel with the three-state decomposition (§3b-xxxix), not as a headline.* It does
not enlarge the cost claim; it characterises it.

### 3b-xli The "representation" row bundles two of our components — split it

`counts_value` -> `content_decay_qcost_value` changes **three** things: the belief prior
(counts -> activations), the Beta-Bernoulli decay, and the dollar-space cost head. Calling all of
that "the representation" is the same class of error as attributing the formulation column to us.
Every single-factor cell already exists; the formulation is held at the global dual throughout.

**LCB seed 0, matched grid, cost saved vs `counts_value`:**

| swap | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|
| `counts_qcost_value` — **our cost head only** | **+40.5%** | +8.1% | +9.5% | +4.1% | +8.5% |
| `content_decay_value` — **our beliefs only**, decay matched | **+40.8%** | **+19.2%** | **+12.0%** | +2.3% | +6.6% |
| `content_decay_qcost_value` — **both** (ours) | +45.8% | +15.7% | +15.2% | +5.5% | +10.5% |
| *`content_value` — beliefs, decay REMOVED* | *−18.1%* | *−29.0%* | *+1.1%* | *+6.8%* | *+18.2%* |

**Both components are individually large, and they are strongly sub-additive.** At the 50% target
+40.5% and +40.8% compose to +45.8%, not the +64.8% independence would predict. They are partly
redundant by construction: both exist to stop money going to expensive hopeless problems, one by
seeing the problem is hard, the other by seeing the route is dear. Same pattern §3b-xx found on the
strict-improvement floors (+2.90 / +3.88 / +5.07).

**`content_value` is a structurally broken cell — do not read it as "beliefs alone".** Without the
decay the belief never falls, so $p_mR - c_m$ never crosses zero and **the give-up action is
disabled**; the arm was measured abstaining 0.0% at every point above 55%. The code states this at
`:545`. The valid single-factor belief swap is `content_decay_value`, which applies the same
analytic $s/(s+n)$ the count family already uses, so only the *prior* differs.

**Relabel §3b-xxxvii accordingly.** Its "representation" row is **our two activation heads together**
(beliefs + cost head), not one thing. Both are ours — `PRIOR_ART.md` §1 records that cross-model
per-query *cost* prediction is the piece no prior work does — but the paper must report them split,
because a reader who assumes "representation = the belief head" would over-credit the belief head by
roughly a factor of two at the tight targets.

### 3b-xxxviii Only two of the four pools were ever conflated — and the reason is structural

**The formulation column is worth exactly nothing wherever there is no resampling.** With constant
beliefs $p_m$ and constant per-route costs $c_m$, $\arg\max_m(p_m R - c_m)$ picks the **same model
for every problem**, and the give-up is all-or-nothing. So the "global dual + count beliefs" arm
traces precisely the fixed-model hull — RouterBench's own **Zero Router**. The dual has nothing to
allocate *between* until some problems can consume more draws than others.

Measured, not argued:

| pool | resample depth | `base rate + global R` vs fixed-model hull |
|---|---|---|
| RouterBench (11 models, 14,599 test) | none | **−0.1% to +0.2%** |
| SWE-bench Verified (5 routes, 369) | none | **+0.0%** at every target |

**Consequence: SWE-V and RouterBench were never conflated.** Their published baselines were already
the constant-belief arm — i.e. the `counts_value` cell, the honest one. Confirmation: the SWE-V
"RoR cost" row recorded in §3b-xxxiii reads \$0.00260 / \$0.00347 / \$0.00433 / \$0.00477 at the
30/40/50/55% targets, and an independently rebuilt Zero Router reproduces those four numbers
**exactly**. So **+11.7% to +37.4% on SWE-V and the RouterBench gains are representation-only as
they stand** and need no correction.

| pool | has a per-query-cap arm? | was the baseline conflated? |
|---|---|---|
| LiveCodeBench | yes | **yes** — corrected in §3b-xxxvii |
| TACO | yes | **yes** — rerun in flight |
| SWE-bench Verified | no | no — already representation-only |
| RouterBench | no | no — already representation-only |

*Do not read my SWE-V reconstruction as a revision of the numbers.* Rebuilding the probe from the
stored activations (5-fold ridge, `content_last`, hand-set $\alpha=3000$, route AUCs 0.628–0.718)
gives a **weaker** predictor than the recorded one and hence smaller margins (+1.2% to +19.5%). That
is my probe being worse, not the decomposition biting — exactly the matched-comparison trap that has
already caused two wrong retractions here. **The recorded SWE-V numbers stand;** the reconstruction
establishes only what the *baseline* is.

**Why this matters for the claim.** Formulation buys something only on the two resampling pools, and
even there the matched grid cuts it to **+5.1% / +5.8%** at tight targets and **negative** at loose
ones (§3b-xxxvii). On the two single-commit pools it buys **zero**. So across all four pools the
cost advantage is overwhelmingly **representation**, which is the part that is ours.

### 3b-xxxix Accuracy on covered instances: informative at the loose end, degenerate at the tight end — and the reason is the whole difference between the two policies

Abstention always yields a failure in our accounting, so every episode ends in one of **three**
states, not two: solved, abstained, or **covered-and-wrong**, where
$\text{covered-and-wrong} = 1 - \text{solved} - \text{abstained}$. The third state is where
conditional accuracy lives.

**Our arm** (`_value`, cap unbinding, so $R$ alone drives it):

| cost | solved | abstained | covered-and-wrong | acc on covered | mean attempts |
|---|---|---|---|---|---|
| \$0.14951 | 84.8% | 3.5% | **11.7%** | 87.9% | 4.65 |
| \$0.05303 | 75.0% | 25.0% | **0.0%** | 100.0% | 3.57 |
| \$0.02609 | 64.1% | 35.9% | **0.0%** | 100.0% | 3.12 |
| \$0.00638 | 44.9% | 55.1% | **0.0%** | 100.0% | 1.97 |

**RoR as published** (per-query cap *binds*, no give-up action):

| cost | solved | abstained | covered-and-wrong | acc on covered |
|---|---|---|---|---|
| \$0.16660 | 84.8% | 0.0% | 15.2% | 84.8% |
| \$0.05860 | 74.2% | 0.0% | **25.8%** | 74.2% |
| \$0.03179 | 64.9% | 0.0% | **35.1%** | 64.9% |
| \$0.01212 | 35.7% | 0.0% | **64.3%** | 35.7% |

**Two different failure mechanisms, and this is the cleanest statement of the difference.**
Our value arms run with `unconstrained_budget`, so a per-task budget can never run out; the only
route to covered-and-wrong is **exhausting draws** (6 per route × 3 routes), which happens only at
high $R$ where the surplus never crosses zero. At low $R$ the policy gives up long before
exhaustion, so covered-and-wrong is **exactly zero** and conditional accuracy saturates at 100%.
RoR has no give-up at all, so **every** one of its failures is covered-and-wrong, and its
conditional accuracy is identically its overall accuracy. Its per-query cap binds and it burns the
cap before failing.

> **RoR spends the money and then fails. We decline before spending.** Both end with no correct
> answer; only one paid for it.

**Reporting rule.** Conditional accuracy is real, not an artefact — but it is *informative only at
the loose end* (87.9% vs 84.8%) and *trivially 100% at the tight end*, where we buy it by declining.
Quoting it as a headline would claim ~100% accuracy against RoR's 35.7%, which is true and
misleading. **Report overall accuracy with abstentions counted as failures**, and carry the
three-state decomposition as a diagnostic panel — it is the panel that shows the mechanism.

**The derived quantity worth adding: wasted spend** — dollars spent on episodes that produced no
correct answer. It is the thing RoR does and we do not, and neither cost-at-matched-accuracy nor
conditional accuracy exposes it directly. Needs per-episode traces (`capture_trace`), which the
current runs delete; one rerun on one config would give it.

### 3b-xl Charging for abstention: the fragile half of the margin is the half that is not ours

Abstention is free in our accounting, so part of the tight-budget advantage may be nothing but
declining work. No rerun is needed to charge it — a point's priced cost is
$\text{realised spend} + A \cdot \text{abstention rate}$, and the hulls rebuild on the priced axis.
This prices declining as a **pure cost penalty with no accuracy credit**: you pay to dispose of the
problem and are not credited with solving it. (Escalation that also *solves* the problem is a
different policy class and a separate experiment.)

**Against RoR, which never abstains, the tight-budget win does not survive a serious price.**

| price of one abstention | 50% | 60% | 70% | 80% |
|---|---|---|---|---|
| \$0 (as reported) | +48.6% | +20.6% | +16.7% | +5.3% |
| \$0.00500 | +32.1% | +11.8% | +13.3% | +4.3% |
| \$0.02638 (= one gpt-oss-120b call) | **−38.5%** | **−25.7%** | −0.9% | +0.4% |

Break-even: **0.56×** the dearest route at the 50% target, **0.45×** at 60%, 0.95× at 70%, 1.07× at
80%. So if disposing of a declined problem costs even half a gpt-oss-120b call, the tight-budget
advantage over RoR is gone. **This belongs in the paper.**

**Against `counts_value`, which also abstains and is also charged, the advantage is robust to any
price.**

| price of one abstention | 50% | 60% | 70% | 80% |
|---|---|---|---|---|
| \$0 | +45.8% | +15.7% | +15.2% | +5.5% |
| \$0.02638 (= one gpt-oss-120b call) | **+23.9%** | **+10.6%** | **+12.9%** | **+6.0%** |
| \$0.05277 (2x) | +16.9% | +7.9% | +11.2% | **+6.4%** |

It even *grows* at the 80% target as the price rises, because our abstentions are better targeted.

**Read the two tables together — they fall exactly on the attribution line.** The part of the margin
that collapses under an abstention price is the part that comes from *having* a give-up action at
all, which is **ROI-Reasoning's** contribution and which RoR simply lacks. The part that is
**ours** — better beliefs, hence better-targeted abstention — survives a price of twice the dearest
route in the pool. That is the strongest form of the claim, and it is the one to lead with:

> Against a baseline that abstains on the same budget with count-based beliefs, activation beliefs
> cut cost at matched accuracy by **+6.0% to +23.9%** *even when every abstention is charged the
> price of a full gpt-oss-120b call.*

*Caveat:* LCB seed 0 only; the same sweep should be run on every seed and on TACO.

### 3b-xxxvii The 2x2, run: how much is representation and how much is formulation

LCB pool64k, seed 0, **matched 96-point geometric budget grid and 96-point $R$ sweep** (§3b-xxxv).
Cost at matched accuracy; positive = ours cheaper.

| contrast | isolates | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|---|
| **ours vs `counts_value`** | **representation**, formulation held at global dual | **+45.8%** | **+15.7%** | **+15.2%** | **+5.5%** | **+10.5%** |
| **`content_decay_qcost` vs `counts`** | **representation**, formulation held at RoR's per-query cap | **+16.9%** | **+6.7%** | **+18.9%** | **+5.8%** | n/a |
| `counts_value` vs `counts` | formulation alone | +5.1% | +5.8% | +1.8% | **−0.2%** | **−3.3%** |
| ours vs `counts` | everything — *what was reported before this section* | +48.6% | +20.6% | +16.7% | +5.3% | +7.5% |

**The matched grid moved the attribution in our favour.** Formulation alone was +16.7%/+13.9% at
the two tight targets under the sparse grid; under the matched grid it is **+5.1%/+5.8%**, and it is
**negative** at 80% and 84%. So most of what looked like "the global budget is worth a lot when money
is tight" was **RoR's missing grid points**, not the formulation. Representation is worth
**+5.8% to +18.9%** inside RoR's own formulation and **+5.5% to +45.8%** inside ours.

**The complementarity is larger, not smaller:**

| target | repr. alone | form. alone | if independent | actual | **interaction** |
|---|---|---|---|---|---|
| 50% | 16.9% | 5.1% | 21.1% | **48.6%** | **+27.5pt** |
| 60% | 6.7% | 5.8% | 12.1% | 20.6% | **+8.5pt** |
| 70% | 18.9% | 1.8% | 20.3% | 16.7% | −3.6pt |
| 80% | 5.8% | −0.2% | 5.6% | 5.3% | −0.3pt |

**Why.** Count beliefs are $\hat p_m = s\,\pi_m/(s + n_m)$, so at $n_m = 0$ **every problem has the
identical belief vector** — the route prior. The code says it: *"Count beliefs have no per-problem
prior, so observed failures are their ONLY channel for learning that a problem is hard"* (`:595`).
Hence `counts_value` **cannot abstain selectively at entry**: with no failures observed the surplus
$p_mR - c_m$ is the same number on every problem and the give-up is all-or-nothing. Its only route
to discrimination is to **pay for failures first**. Visible in the arm — it abstains at **41.6%** at
the 50% target and *still* costs more than ours.

So: **abstention is only worth having if you can tell which problems to abstain on before paying.**
The global dual supplies the give-up action (ROI-Reasoning's); per-problem priors from activations
make it *selective* (ours). Neither alone gets the tight-budget number — and under the matched grid
the formulation alone gets almost none of it.

**The anchor to lead with.** At the accuracy `gpt-oss-120b` reaches when called on every problem
(68.77%, \$0.04519):

| policy | cost | vs RoR | vs calling the model on everything |
|---|---|---|---|
| RoR as published | \$0.04152 | — | +8.1% |
| `counts_value` | \$0.04065 | +2.1% | +10.1% |
| **`content_decay_qcost`** (ours, their cap) | **\$0.03412** | **+17.8%** | **+24.5%** |
| **ours** | **\$0.03412** | **+17.8%** | **+24.5%** |

At this operating point the two of ours are **indistinguishable to five decimals**, so the entire
+17.8% is the representation and none of it is the formulation. That makes it the cleanest
attributable headline available.

*Caveats:* single seed. The 2x2 has **not** been run on TACO / SWE-V / RouterBench, whose published
numbers are still the conflated `ours vs counts` row on a linear grid.

### 3b-xxxv The two arms were not swept under the same spacing law

**The defect.** The budget-swept (RoR) arm's grid was 12 linear points from the cheapest route to
$3\times$ the dearest, plus 5 more to full exhaustion — 17 points at a ~\$0.012 pitch. The value
arm gets **96 geometric** points. On the LCB pool that is:

| | points below \$0.02 | points below \$0.05 |
|---|---|---|
| linear budget grid (17 pts) | **3** | 7 |
| matched geometric grid (96 pts) | **55** | 71 |

Every tight-budget claim reads off inside the region the linear grid samples three times.

**Why it is not obviously fatal, and why it still has to be fixed.** Mixtures make the baseline
frontier continuous, so a sparse grid does not credit RoR with something unreachable — the chord
between two swept points is an achievable randomised policy. The exposure runs the **other way**:
greedy allocation under a cap is not concave in $B$, so a denser grid can surface points *above*
the chord and shrink our margin exactly where it is largest. Untested until now, and the first
thing a reviewer pulls on.

**Fix.** `--budget-grid geometric --budget-grid-points N` sweeps the budget arm under the value
arm's spacing law. `linear` remains the default so previously recorded numbers stay reproducible.
This is the matched-comparison rule that has already produced two wrong retractions in this line.

**Result of the matched sweep (LCB, seed 0).** The defect was real and it cut where predicted —
the tight end — but it cut the **formulation** effect far harder than ours.

| target | | RoR cost, linear 17 | RoR cost, matched 96 | ours vs RoR, linear | ours vs RoR, **matched** |
|---|---|---|---|---|---|
| 50% | | \$0.01728 | **\$0.01515** | +54.9% | **+48.6%** |
| 60% | | \$0.02499 | **\$0.02282** | +27.5% | **+20.6%** |
| 70% | | \$0.04494 | \$0.04494 | +16.7% | +16.7% |
| 80% | | \$0.09273 | \$0.09162 | +6.4% | +5.3% |
| 84% | | \$0.15041 | \$0.15041 | +7.5% | +7.5% |

RoR gains most at the tight end (12.3% cheaper at the 50% target, 8.7% at 60%) and nothing at 70%
and above, where its linear grid was already dense enough. **Our margin against it falls by 6.3 and
6.9 points at the two tight targets and is unchanged elsewhere.** The claim survives; it is smaller.

The matched grid also helps `content_decay_qcost` — our beliefs under RoR's cap — from 11 hull
vertices to 15 and from \$0.01359 to \$0.01260 at the 50% target. Both budget-swept arms were
under-sampled; fixing it is not a one-sided concession.

### 3b-xxxiv Which RoR are we beating? Both, and they must be reported separately

**A labelling error running through this document.** `hull()` pools a *policy family*, and the
`counts` family contains `counts_value`, `counts_abstain` and `counts_value_frozen` — arms that use
**our** zero-value give-up rule. Seven of the nine vertices of the LCB `counts` hull come from those
arms, abstaining at **18.7%–41.6%**. So every "vs RoR as published" number in this document was
actually measured against **RoR augmented with our give-up action**, which is strictly stronger
than the published method: RoR has no stop action at all (*"stop-only is a descriptive reference,
not a member of A"*, `PRIOR_ART.md` §1).

**Both baselines, separately** (cost saved at matched accuracy, seeds as stated):

| | target | **vs RoR as published** | vs RoR + our give-up |
|---|---|---|---|
| LCB (5 seeds) | 50% | **+54.5%** | +45.8% |
| | 60% | **+28.6%** | +17.7% |
| | 70% | +20.3% | +18.7% |
| | 80% | +9.2% | +8.0% |
| | 84% | +8.8% | +8.6% |
| TACO (3 seeds) | 35% | **+45.7%** | +33.6% |
| | 40% | **+42.5%** | +29.3% |
| | 45% | **+21.4%** | +4.7% |
| | 50% | +13.1% | +12.8% |
| | 55% | +9.1% | +8.6% |

**The gap is largest exactly where abstention does its work** — tight budgets — which is the
mechanism check passing: +28.6% against +17.7% at LCB 60%, +42.5% against +29.3% at TACO 40%.

**And it dissolves the TACO 45% "dip".** Against published RoR that target is **+21.4%**, not +4.7%.
The dip was our own give-up action being handed to the baseline at that accuracy, not hull-vertex
placement as §3b-xxxiii claims. *Correct that section: the vertex-placement explanation is right
about the mechanism in general and wrong about this instance.*

**Reporting rule.** Lead with **RoR as published** — it is the actual prior method, and a reader
comparing against the literature expects the literature's policy. Carry **RoR + give-up** alongside
as a steelman, since the result survives it and reporting only the published arm invites the fair
objection that we withheld our own mechanism from the baseline. **Never report one as if it were
the other**, which is what this document did until now.

*Scope of the correction:* every "vs RoR" figure in §3b-xii, §3b-xv, §3b-xxi..xxxiii and in the
summary tables is the augmented arm and needs relabelling. The strict-improvement floors are
likewise measured against the augmented hull, so they are conservative rather than wrong.

### 3b-xxxiii The cost claim holds on all three benchmarks — and what nearly hid it on SWE-V

**Cost at matched accuracy is what the routing literature reports and what a practitioner asks.**
It is also weaker than "strictly better at every level", which is the claim that died on TACO. The
weaker claim holds everywhere we have looked.

**LiveCodeBench** (5 seeds, myopic vs `counts`): strict-improvement floor **+2.90%**, rising to
**+4.86%** with Bellman h2, P(floor>0) = 1.000. Cost to match always-gpt-oss-120b: **69.8%** of its
price, where RoR needs 86.1%.

**TACO** (3 seeds), cost saved against RoR at fixed accuracy targets:

| target | advantage | 95% CI | P(win) |
|---|---|---|---|
| 35% | **+33.6%** | [+32.2, +34.9] | **1.000** |
| 40% | **+29.3%** | [+29.1, +29.7] | **1.000** |
| 45% | +4.7% | [+2.2, +7.7] | **1.000** |
| 50% | **+12.8%** | [+9.5, +14.9] | **1.000** |
| 55% | +8.6% | [+1.2, +13.6] | **1.000** |

*The 45% dip is hull-vertex placement, not a weakness:* RoR happens to have a vertex at exactly
$0.01388 / 45.0% while ours sit at 43.7% and 48.6%, so at that target RoR reads a vertex and we
interpolate along a chord. The neighbourhood is smooth (+29.3 -> +4.7 -> +12.9 across 40/45/50%).
**Report a curve or an average over targets, never a single target.**

**SWE-bench Verified** (369 problems, single-draw so single-commit + abstain, out-of-fold probe):

| target | RoR cost | ours | advantage |
|---|---|---|---|
| 30% | $0.00260 | $0.00163 | **+37.4%** |
| 40% | $0.00347 | $0.00253 | **+27.0%** |
| 50% | $0.00433 | $0.00362 | **+16.4%** |
| 55% | $0.00477 | $0.00421 | **+11.7%** |

**RETRACTED within the hour: "the cost claim fails on SWE-V".** A first version of this table showed
±4% noise and I built a "switching price too cheap to skip" story on top of it. **The cause was a
bug of mine:** I passed *actual realised per-problem costs* into the decision rule for both arms.
With constant beliefs, RoR's utility $p R - c_i$ then varies only through $c_i$, so it was
attempting the genuinely cheapest problems first — per-problem cost information no deployed policy
has. The LCB/TACO replays correctly decide on **constant per-route estimates** and charge realised
costs; once SWE-V does the same, the advantage appears. **Any new frontier harness must decide on
estimates and charge realisations, and the tell is a constant-belief arm producing a many-vertex
hull — it should be nearly all-or-nothing.**

**The probe is doing real work on SWE-V**, which the broken table obscured: ranking by predicted
gemini-success, the attempted set solves **90.4% at 20% coverage** against 62.3% overall (+28.1pt
lift), decaying monotonically to parity at full coverage.

**Two caveats.** SWE-V is single-draw, so it is a *different policy class* (single-commit routing
with abstention, not the resample MDP) and should be reported as such rather than blended into one
table. And it is one split with out-of-fold predictions, not seed-replicated like LCB's five —
bootstrap it before the draft.

**Pool-structure context, which still stands.** The switching price of the top rung differs by
orders of magnitude across pools: LCB $0.38 (inside the operating range), TACO $5.99 (never worth
buying — every TACO number is effectively a two-rung pool), SWE-V $0.0085 (always worth buying).
That shapes *which channel* pays — routing versus abstention — but it does not determine whether
the cost claim holds, which it does in all three.

### 3b-xxxii Bellman lookahead: helps LiveCodeBench's floor, hurts TACO, does nothing for budgets

The one *structural* change tried, as opposed to the five predictor improvements of §3b-xxx. An
exact Bellman solve over the failure-count lattice plans depth like a knapsack but **re-solves at
every state** rather than committing — the resample-aware allocation the one-shot arm (§3b-xiv)
cannot express. H=1 reproduces the myopic rule exactly; larger H credits the continuation value the
myopic rule drops.

| | floor vs `counts` | 95% CI | P(>0) | seeds negative |
|---|---|---|---|---|
| LCB myopic (5 seeds) | +2.90% | [+0.73, +3.44] | 0.998 | 2/5 |
| **LCB bellman h2** | **+4.86%** | **[+1.82, +6.31]** | **1.000** | **1/5** |
| TACO myopic (3 seeds) | +0.93% | [-4.75, +2.81] | 0.502 | 1/3 |
| TACO bellman h2 | **-6.38%** | [-18.22, +2.40] | 0.247 | 2/3 |

**LiveCodeBench's strict-improvement claim gets stronger** — the floor rises by two thirds, the
interval moves up and away from zero, and the share of individually-negative seeds halves.
**TACO's gets worse**, from a coin flip to clearly negative.

**It does not move budget-units performance at all**: -0.8 / +0.0 / +0.6 / -0.1 pt at
0.25x/0.50x/1.0x/2.0x on LCB, averaged over 5 seeds.

**Horizon ordering (seed 0, the only seed run with 2/4/8):** h2 +3.46% > h4 +1.36% > h8 +0.94%.
Deeper lookahead is worse, consistent with belief error compounding across the lattice — a 2-step
solve recovers the continuation value, an 8-step solve trusts imperfect beliefs eight moves out.

**A seed-0 result that did not replicate.** Seed 0 alone showed **+4.4pt at 1.0x budget**; across
five seeds that is **+0.6pt**. This is the sixth intervention this session whose single-seed result
overstated its replication. **Treat every single-seed number in this document as a hypothesis.**

**Where this leaves the two positives.** Of seven interventions tried, the five that improved a
predictor metric all failed (§3b-xxx), adaptive-R fired its kill criterion, and Bellman h2 — the
only one that changed the *policy structure* — is the only one that improved anything that
survived replication. That asymmetry is the most robust thing in this section: **at our prediction
quality, policy structure is where the remaining value is, and better inputs are not.**

### 3b-xxxi What the tight-budget win is actually made of, and what the knapsack framing settles

**The +26pt is information, not RoR's structural limit.** LCB @ 0.25x, seed 0, decomposed with the
shuffled-prediction control:

| | accuracy | attributable to |
|---|---|---|
| RoR as published | 26.8% | — |
| + per-problem **dispersion** only (predictions shuffled, marginals kept) | 28.8% | **+2.0pt** — fixing RoR's structural inability to allocate selectively |
| + our actual per-problem **information** | 52.7% | **+24.0pt** — the signal |

So the concern that we only win because count-based beliefs *cannot* vary per problem is
measurable, and it accounts for **2 of the 26 points**. The rest is the prediction being right.

**But the headline framing should change anyway.** RoR-as-published is the weakest reasonable
policy in that regime. The honest statement is not "we beat RoR by 26 points" but: **a sequential
budget policy with no per-problem prior leaves ~24 points on the table at tight budgets, and one
cheap prefill recovers them.** RoR is the demonstration that the gap exists, not the measure of our
contribution.

**Both our rule and the knapsack are already global; the difference is adaptivity.** Our `_value`
arms and the one-shot arm of §3b-xiv both solve a **global** budget in the **dual** — each problem
takes the plan maximising $(1-(1-\theta_m)^n)R - n\hat c_m$ under a **common** $R$, swept to trace
the frontier. `hull()` pools budget-swept and $R$-swept arms, so the reported frontier already
contains globally-coupled policies. *(An earlier note in `THREADS_IN_PROGRESS.md` describing the
replay as "per-problem budget" was wrong.)*

**Which makes the one-shot comparison a clean adaptivity ablation.** Same global formulation, same
predictions, same multiplier — the only difference is that the MDP re-reads
$\theta\sigma/(\sigma+n)$ after each observed failure. It wins by **+1.1 to +20.2pt**. *Adaptivity
is worth that much, holding the allocation formulation fixed* — and a knapsack cannot have it,
because committing a plan up front is what a knapsack is.

**So "make the knapsack resample-aware" resolves to one specific defect.** Depth is already part of
the item, and the budget is already global. What is stale is $R$ itself: it is computed from
**predicted** costs (cost $R^2$ = 0.545, not 1.0), realised spend drifts from plan, and nothing
corrects the drift. **Dual descent — updating $R$ against realised spend as the batch progresses —
is the missing piece**, requires no change to the per-problem rule, and is the next experiment
(`THREADS_IN_PROGRESS.md` §1).

**Prior art, checked (`PRIOR_ART.md` §4b, §6b).** ROI-Reasoning formalises this allocation as an
Ordered Stochastic Multiple-Choice Knapsack **and has an abstain action**, so neither the knapsack
nor abstention-under-budget is ours — but it is **single-model**, uses the target's **own**
meta-cognitive predictions, and does not resample. Semantic-agreement cascades ensemble small
models for deferral but require **full generations**. **What survives as ours in this area: one
cheap prefill pricing an entire heterogeneous pool, which is what makes a cross-model knapsack
affordable at all.**

### 3b-xxx Selecting components on a predictor metric degrades the policy — four cases and one fix

**The pattern.** Four interventions, each improving its own predictor metric and each making the
frontier worse:

| intervention | predictor gain | frontier effect |
|---|---|---|
| belief penalty $C$ selected on AUC | +0.024 AUC | **-4.34pp** TACO floor (P(floor>0) 0.514 -> 0.000) |
| cost penalty $\alpha$ selected on $R^2$ | 4/6 cells; TACO oss20 -0.033 -> +0.187 | **-3.2pt** LCB 0.25x |
| cross-route coupling | +0.024 AUC from observed scout outcome | **-23.8pt** LCB 0.25x |
| isotonic recalibration (ISO) | perfect calibration by construction | -0.4 to -1.9pt |

**The cause.** AUC is invariant to any monotone transform, $R^2$ and calibration error are
invariant to affine rescaling of the residual structure the rule uses — but
$\arg\max_m(p_mR-c_m)$ consumes prediction *values*, and abstention fires on $\max_m Q\le0$.
Better-regularised predictors are more **compressed**; the rule needs **spread**. Optimising
accuracy is paid for in dispersion.

**The fix, and its limits.** Selecting $\alpha$ on the *calibration frontier* (`--eval-split
calibration`, test touched once), LCB, 5 seeds:

| budget | delta vs shipped | sd | seeds positive |
|---|---|---|---|
| 0.25x | **+3.88pt** | 0.73 | **5/5** |
| 0.50x | **+2.34pt** | 1.06 | **5/5** |
| 1.00x | **-2.60pt** | 1.92 | 1/5 |
| 2.00x | +0.26pt | 0.59 | 3/5 |

**Frontier-selection does not dominate — it trades regimes.** The tight-budget gain is real and
consistent (5/5 seeds, sd 0.73) and it costs 2.6pt at 1.0x, equally consistently. Reporting a mean
over budgets (+0.97pt) hides that, and an earlier single-seed read of "+1.3pt" did exactly that.

**The selection is also noisy at these sample sizes.** TACO (n=168 calibration) picks
$\alpha=10^6$ where LCB (n=170) picks $10^4$ — good, in that the criterion does per-pool work
rather than finding a property of the ridge geometry. But TACO's curve is non-monotone
(46.3 / 48.1 / 44.7 / 46.3 / 48.2 across $10^2..10^6$), with the two best values tied within 0.1pt
at opposite ends of the grid. **Taking an argmax over a curve like that is not a reliable
procedure**, and the honest reading is that the frontier criterion reliably rejects *extreme*
regularisation while being unable to discriminate within an order of magnitude.

**What survives as a claim.** *In a cost-constrained sequential policy, component hyperparameters
must not be selected on standard predictor metrics, because those metrics are invariant to the
prediction spread the decision rule consumes.* Four independent demonstrations support the
negative; the positive prescription (select on the policy objective) avoids the failure mode but
buys a regime trade rather than a uniform win, and needs more calibration data than we have to be
executed cleanly.

### 3b-xxix Cost prediction is the largest measured lever, and its ceiling is not 1.0

**Oracle arms bound each channel** (LCB seed 0, everything else fixed):

| budget | ours | +per-tercile sigma | +oracle sigma | +oracle cost |
|---|---|---|---|---|
| 0.25x | 52.7% | 53.9% | 53.7% | **60.7%** |
| 1.00x | 67.4% | **71.3%** | **75.8%** | **77.2%** |
| 2.00x | 80.0% | 79.3% | 80.9% | 79.8% |

Per-tercile sigma is worth **+3.9pt at 1.0x and ~0 at 0.25x** — exactly the falsifiable prediction
(sigma governs continue decisions, and at 0.25x $n^*$ floors at 0.34). Oracle sigma bounds that
channel at **+8.4pt**, so three constants capture ~46% of it. **Oracle cost is the largest single
lever at +9.8pt (1.0x) and +8.0pt (0.25x).**

**What predicts cost, measured:**

| predictor | LCB $R^2$ | TACO $R^2$ |
|---|---|---|
| constant (RoR) | -0.221 | -0.001 |
| prompt length | 0.017 | 0.018 |
| scout's *realised output length* | 0.254 | 0.141 |
| **prefill activations** | **0.414** | **0.395** |
| activations + scout output length | 0.417 | 0.396 |

The scout's realised output length is **15x better than prompt length** — an earlier dismissal of
"length" conflated the two — but adds **+0.003** on top of activations. **The prefill already knows
how verbose the answer will be, better than generating it and measuring**, which is the
prefill-beats-generation result again on the cost side.

**The ceiling is the between-problem variance share, not 1.0.** Decomposing per-draw cost:

| route | ICC (= max achievable per-problem $R^2$) | we reach | captured |
|---|---|---|---|
| LCB oss120 | **0.841** | 0.414 | **49%** |
| TACO oss120 | **0.795** | 0.395 | **50%** |
| LCB oss20 | 0.671 | 0.208 | 31% |

16-20% of cost variance is irreducible draw noise. **We capture half of what is predictable**, on a
plain regression target, with three known mechanical deficiencies: the cost head is a linear ridge
on pooled mean/last (never given the rich multi-layer treatment the belief head got), fitted in log
space with smearing (the space C3b says is wrong), under a single hard-coded alpha (the pattern that
cost +0.024 AUC on the belief side). **Fix those before concluding anything about representation
limits.**

**Correction to §3b-xxviii.** That section concluded "the representation is the bottleneck,
+7.8-12.4pt", measured with sigma and the cost head at their broken values. Fixing sigma recovers
+3.9pt and oracle cost bounds another +9.8pt, so a meaningful share of what was attributed to the
representation is machinery we had mis-specified.

**Correction on the learned belief (2026-09-10).** A claim that "the learned sigma / factorized
scorer was never replayed" was **wrong**. It was replayed on the 32k pool
(`lcb_mdp_factorized_seed17_*`) with full Bellman-lookahead and q-stop arms, and beat `counts` by
**+5.3pt at both 0.25x and 1.0x**. What is true is narrower: that run carries no `content_decay`
arm, so the factorized belief has never been compared head-to-head against our probe, and it has
never been re-run on the 64k pool. **The open question is whether it beats ours, not whether it
works.**

### 3b-xxviii RETRACTION of §3b-xxvii, and the correct decomposition: it is the representation

§3b-xxvii argued the probe is a good ranker and a poor estimator, predicting that perfect
calibration of our own ordering would close the high-budget gap. **The test refutes it.** LCB seed
0, everything identical except the belief input, where `ISO` is our predictions isotonically
recalibrated against truth (monotone, so our ranking is preserved *exactly*) and `RANK1` is a
one-factor oracle with perfect shared difficulty and zero interaction:

| budget | RoR | ours | ISO | RANK1 | ORACLE |
|---|---|---|---|---|---|
| 0.25x | 26.8% | 52.7% | 55.2% | 65.7% | 68.4% |
| 0.50x | 58.4% | 62.6% | 60.7% | 73.1% | 74.7% |
| 1.00x | 64.9% | 67.4% | 67.0% | 74.9% | 84.8% |
| 2.00x | 78.5% | 80.0% | 79.1% | 81.8% | 84.8% |

| budget | **calibration** (ours->ISO) | **representation** (ISO->RANK1) | **interaction** (RANK1->ORACLE) |
|---|---|---|---|
| 0.25x | +2.5 | **+10.5** | +2.7 |
| 0.50x | **-1.9** | **+12.4** | +1.6 |
| 1.00x | **-0.4** | **+7.8** | +9.9 |
| 2.00x | -0.9 | +2.7 | +3.0 |

**Perfect calibration buys nothing and is negative at three of four budgets. The representation is
the bottleneck, worth +7.8 to +12.4pt.** Our *ordering* is the weak link, not our probabilities.
The prediction §3b-xxvii made — "ours->ISO large at 1.0x" — came back at -0.4.

**So the honest diagnosis is the plain one: the probe does not carry enough per-problem
information.** Not a ceiling on the channel (oracle reaches 84.8%), not miscalibration (ISO is
flat), and not primarily the interaction (worth +2.7/+1.6/+9.9/+3.0, second everywhere except
1.0x). *Caveat: RANK1 is fitted to truth including test, so +10.5 is a ceiling for a perfect
shared-difficulty probe, not what a realistically better one would deliver. One seed.*

**This also downgrades the per-tercile sigma experiment** (§6.x correction): sigma governs continue
decisions, which sit downstream of an ordering that is already wrong. Run it for completeness, but
the prior on it mattering should now be low.

**Where the remaining effort should go — cheap evidence, not better readouts.** The measured
alternative is not a richer probe of the same prefill but *buying a little evidence*: RoR's own
failure channel is worth 28 points (§6.x), and one observed scout failure moves oss120 pass@1 from
97.7% to 69.9%. That is a far larger per-problem signal than our prefill carries, and it is already
priced in our budget framework (the scout costs 0.05x of an oss120 call). **§3b-xxix prices the
options.**

### 3b-xxvii The probe is a good ranker and a poor estimator — **RETRACTED, see §3b-xxviii**

**The observation.** Replacing $\hat\theta(x)$ with the true per-problem solve rate and changing
nothing else (counts, decay, cost head, sequential rule all identical), LCB seed 0:

| budget | RoR | ours | oracle $\theta$ | ours gain | oracle gain | **we capture** |
|---|---|---|---|---|---|---|
| 0.25x | 26.8% | 52.7% | 68.4% | +26.0 | +41.6 | **63%** |
| 0.50x | 58.4% | 62.6% | 74.7% | +4.2 | +16.4 | 26% |
| **1.00x** | 64.9% | 67.4% | **84.8%** | +2.5 | +19.9 | **13%** |
| 2.00x | 78.5% | 80.0% | 84.8% | +1.5 | +6.3 | 24% |

**The high-budget flatness is a prediction failure, not a structural ceiling.** A perfect belief
gains +19.9pt at 1.0x and reaches the pool ceiling; we capture 13% of it.

**The explanation, and it unifies four separate results.** Tight budgets bind on the *extensive*
margin — which problems to attempt at all — which needs only an **ordering**; a misranking near
the cutoff swaps two problems of similar value and costs almost nothing. High budgets bind on the
*intensive* margin — how deep, on which route — which compares $p\cdot R$ against $c$ for a
specific marginal draw and needs to know whether $p$ is 0.15 or 0.35. **A ranking cannot answer
that; only a calibrated value can.**

Under that reading, four findings we had treated separately are one finding:

| finding | restated |
|---|---|
| C3: ordering transfers free, calibration costs ~25 labels | ranking is cheap, values are expensive |
| §3b-xxvi: selecting C on AUC helped ranking, hurt the policy | AUC scores order; the policy consumes values |
| SWE-V pool extension fails, reliability 0.0389 > resolution 0.0254 | a calibration failure, not a discrimination one |
| we capture 63% of headroom at 0.25x, 13% at 1.0x | ordering suffices at the extensive margin only |

**The decomposition that tests it** (in flight): `ISO` = our predictions isotonically recalibrated
against truth — monotone, so our ranking is preserved *exactly* and only calibration changes;
`RANK1` = a one-factor oracle with perfect shared difficulty and zero interaction. Then
ours -> ISO is what perfect calibration of our own ordering buys, ISO -> RANK1 is what a better
representation buys, and RANK1 -> ORACLE is the interaction, which C4 says no prompt-only probe
reaches. **Prediction: ours -> ISO is large at 1.0x and small at 0.25x.** If ISO barely moves and
RANK1 does, the thesis is wrong and our ranking, not our calibration, is the weak link.

**Why our existing calibration does not already fix this.** Platt scaling fits **two parameters per
route** on a **global** monotone curve. It can remove average over-confidence and nothing else:
- it cannot fix miscalibration that varies **across the range**, and the range that matters is the
  low-$p$ tail where the stop decision bites and where we have the fewest calibration examples;
- the calibration split is **~170 problems**, so the tail is fit on tens of points;
- post-hoc calibration is monotone by construction, so it **cannot add resolution**. Under the
  Murphy decomposition (Brier = reliability - resolution + uncertainty) it drives reliability toward
  zero and leaves resolution untouched. Where resolution is low — SWE-V, 0.0254 — perfect
  calibration just makes the predictor *confidently average*;
- **we calibrate the wrong conditional.** We calibrate $P(\text{solve next draw}\mid x, m)$ at depth
  0, but the policy consumes it after $n$ observed failures, where the belief is
  $\hat\theta\cdot\sigma/(\sigma+n)$ with a **single global $\sigma$**. If the true decay shape
  varies per problem, no amount of calibrating the depth-0 prior repairs it. *This is the specific,
  testable one, and it is the natural next experiment: fit $\sigma$ per problem (or per difficulty
  bin) and re-run.*

### 3b-xxiii Bootstrap CIs on the strict-improvement floors — and TACO does not survive

The floor is a **minimum over 401 correlated levels**, the most downward-biased statistic in the
paper, and it had never carried an interval. Cluster bootstrap over seeds, 4000 resamples, on the
statistic `hull_frontier.py` actually reports (min over levels of the seed-**averaged** advantage):

| arm | floor | 95% CI | **P(floor > 0)** | seeds with a negative floor |
|---|---|---|---|---|
| LCB, full method | +5.07% | [+1.04, +7.71] | **0.999** | 2 of 5 |
| LCB, beliefs only | +2.90% | [+0.73, +3.44] | **0.997** | 2 of 5 |
| **TACO, beliefs only** | **+0.93%** | **[-4.75, +2.81]** | **0.511** | 1 of 3 |

**LiveCodeBench's strict-improvement claim survives; TACO's does not.** P(floor>0) on TACO is
**0.511** — a coin flip. "Negative at 0 of 401 levels on TACO" must be dropped, or restated as
"the seed-averaged curve was non-negative on the three seeds we ran, with an interval spanning
zero." With three seeds this is a statement about sample size as much as effect size, and the
honest fix is more seeds, not a softer verb.

**A second finding that must be reported whichever way TACO resolves.** The floor is computed on
the seed-*averaged* curve, and averaging smooths away each seed's worst level. Computed per seed
instead, **2 of 5 LCB seeds and 1 of 3 TACO seeds have a negative floor**, and the mean per-seed
floor drops to +1.69% / +0.27% / -0.65%. **So "strict improvement" is a property of the averaged
curve, not of a single run** — a reviewer running one seed would not reproduce it. Say so.

### 3b-xxiv SWE-bench Verified at 16k: truncation was real but is not the whole deficit

39.6% of SWE-V prompts exceed 8192 tokens (11.7% exceed 16384), so the confound flagged in
§3b-xvi was severe. Re-extracted at `--max-len 16384`, out-of-fold on the same 369 problems:

| max-len | pool-solvability AUC | AURC | gap to oracle closed |
|---|---|---|---|
| 8192 | 0.691 | 0.181 | 44.0% |
| **16384** | **0.730** | **0.173** | **47.3%** |

**Truncation cost ~0.04 AUC and ~3pt of gap closure — real, and not enough.** SWE-V at 16k
(0.730) is still far below LiveCodeBench (0.824) and TACO (0.885). **The domain boundary is
genuine**, not an artifact: repo-level software engineering is harder for a prompt-only probe than
competitive programming, and the earlier caveat can now be withdrawn rather than carried.

### 3b-xxv The cost head on RouterBench: it pays only where cost is already nearly deterministic

Fitting the conditioned cost head on RouterBench activations and swapping it for the constant:

| dataset | n | cost $R^2$: const / length / **activations** | AIQ const-c | AIQ qcost | delta |
|---|---|---|---|---|---|
| mmlu | 5596 | -0.000 / 0.975 / **0.987** | +3.64% | **+4.63%** | **+0.99%** |
| hellaswag | 3984 | -0.000 / 0.972 / **0.977** | +6.74% | **+7.06%** | +0.31% |
| grade-school-math | 3011 | -0.000 / 0.231 / **0.381** | +2.01% | +1.67% | **-0.34%** |
| mbpp | 162 | -0.005 / 0.045 / **0.251** | -0.94% | -1.38% | **-0.44%** |

**This came out backwards from the prediction, which is the interesting part.** We expected the
cost head to pay on the generative sets (gsm8k, mbpp) where output length is genuinely
unpredictable, and to be pointless on multiple-choice where prompt length already determines cost.
The opposite happened: **it helps only where cost prediction is near-perfect ($R^2\approx0.98$) and
hurts where prediction is merely good ($R^2$ 0.25-0.38).**

That is C3b's discipline restated as a threshold rather than a principle: **a conditioned quantity
must be nearly exact to beat the constant it replaces, because the policy consumes it as a price
and mis-pricing mis-routes.** Weighted over these four sets the cost head is worth **+0.45%** —
marginal, and consistent with C1b (the heads are substitutes, and the belief head is the stronger
one). *Recommendation: report RouterBench with beliefs only, and use this table as the evidence for
the gate rather than as a result.*

### 3b-xxii The shipped belief head is under-regularised — free accuracy

`activation_content_preds.py` uses a hard-coded `C = 0.05/(dim//2560) = 0.003125`. Selecting C on
the **calibration** split (which the script already loads for Platt) and scoring on test:

| route | shipped | C* on cal | gain |
|---|---|---|---|
| LCB scout | 0.840 | 0.00030 | -0.006 |
| LCB oss20 | 0.769 | 0.00001 | +0.022 |
| **LCB oss120** | 0.758 | 0.00010 | **+0.043** |
| TACO scout | 0.743 | 0.00003 | +0.015 |
| TACO oss20 | 0.800 | 0.00003 | +0.040 |
| TACO oss120 | 0.818 | 0.00010 | +0.032 |

**Mean +0.024 AUC, +0.043 on gpt-oss-120b** — the expensive route the policy's decisions hinge on.
**But see §3b-xxvi: this AUC gain makes the *policy* worse on three of four arms**, because AUC
scores order and the policy consumes values. The fix is to select C on the frontier, not on AUC.
Selected C is 10-300x smaller than shipped, so the head is badly under-regularised; the effect is
near-zero only on the scout, where classes are balanced. Separately, **noisy-OR beats max** for
pool-solvability (0.770 vs 0.759 LCB, 0.844 vs 0.838 TACO), which improves §3b-xvi for free. Both
are one-line changes and every downstream number should be regenerated after them.

*(An earlier claim that a plain ridge beat the committed chain by 0.065 was mostly a target
mismatch — the shipped head predicts pass@1 on draw 0, and it was scored on pool-solvability over
six draws. The residual above is the real part.)*

### 3b-xix The scope law (SS6.9k) is FALSIFIED, by an oracle, before we ran the probe

SS6.9k claims the advantage scales with the fraction of problems **nothing** in the pool solves, and
line 439 commits us: *"if abstention is worth something [on RouterBench], SS6.9k is wrong."*

The test needs no probe. Give an **oracle** the per-problem choice over all models, once with a skip
action and once without, sweep the multiplier, and read the gap. If oracle abstention buys nothing,
no predictor could. Budgets are expressed as a fraction of the spend at which the no-skip oracle
frontier saturates, because at looser budgets the oracle simply picks the cheapest model that works
and nothing binds:

| benchmark | pool-unsolvable | 0.05x | 0.10x | **0.20x** | 0.40x | 0.70x |
|---|---|---|---|---|---|---|
| **RouterBench** | **3.9%** | +33.1pt | +47.8pt | **+67.0pt** | +5.5pt | +0.1pt |
| LiveCodeBench | 8.2% | +60.2pt | +69.9pt | +7.9pt | +1.3pt | -0.1pt |
| TACO | 39.5% | +33.3pt | +41.0pt | +7.0pt | +2.7pt | +0.6pt |

**With an ORACLE, abstention is worth the most on the pool with the least unsolvable mass** —
RouterBench gains +67.0pt from a skip action at a 0.20x budget, more than TACO at 39.5%.

**PARTIAL RETRACTION (2026-09-09).** This section originally concluded "the law is backwards".
That was too strong: an oracle measures what abstention makes *available*, not what a real
predictor can *realise*. Running the method on RouterBench with actual (TF-IDF) beliefs, the
give-up action is worth **exactly 0.0pt at every budget** — predicted $p$ is high everywhere (1st
percentile of $\max_m p_m$ is 0.401), so the skip is a step function from 100% to 0.6% abstention
between two adjacent values of $R$ and never has a useful partial regime. **§6.9k is right about
realised abstention value and wrong about the ceiling.** Restated: *the fraction of the abstention
ceiling a predictor can capture scales with unsolvable mass, even though the ceiling itself does
not.*

**What the mechanism actually is.** Abstention at a binding budget is not primarily about declining
*hopeless* problems; it is **reallocation** — skipping expensive problems to afford more cheap ones.
Pool-unsolvable mass is the extreme case (infinite cost per unit of value) and therefore sets the
*ceiling*, but the *driver* is cost dispersion against a binding budget. This is the same finding as
the shuffled-prediction control (SS3b-xv): dispersion is what unlocks partial abstention, and
information is what aims it. The user's framing is the right one — **order the problems and draw a
line; the budget decides where the line goes.**

**Consequences.** (1) SS6.9k must be rewritten: the advantage scales with *budget tightness and cost
dispersion*, and pool-unsolvable fraction bounds the ceiling rather than predicting the gain.
(2) Every "TACO is the interesting pool because it has unsolvable mass" claim needs re-checking.
(3) **RouterBench is no longer a falsification target but a plausible strength**, which inverts the
plan: run our method there expecting it to work, and the 91.4%-contested structure that made us
predict failure is irrelevant to the abstention channel.
(4) The claim in SS3b that the result "should be positioned as selective prediction, not routing"
survives and is strengthened — it is selective prediction under a budget.

*Caveat:* oracle beliefs upper-bound every arm; this says what is available, not what our probe
captures. LCB here is 8.2% unsolvable over all 892 problems with complete data, against 15.2% on
the 171-problem test split used in SS3b-xvi.

### 3b-xvi Selective prediction, reported as a risk-coverage curve at last

The paper's title claims selective prediction and the outline never reported the curve. Score every
problem by predicted pool-solvability from the one scout prefill, attempt the top $c$ fraction, and
measure **risk** = the share of attempted problems that *nothing in the pool solves* (pure wasted
spend). LCB/TACO use the committed probe on the manifest test split; SWE-bench Verified uses
out-of-fold ridge on the scout activations (5-fold, n=369).

| benchmark | n | pool-unsolvable | AUC | AURC | random | oracle | **gap closed** |
|---|---|---|---|---|---|---|---|
| LiveCodeBench | 171 | 15.2% | 0.759 | 0.062 | 0.152 | 0.013 | **64.9%** |
| TACO | 168 | 41.1% | 0.838 | 0.185 | 0.411 | 0.100 | **72.7%** |
| SWE-bench Verified | 369 | 28.7% | 0.703 | 0.182 | 0.287 | 0.046 | **43.8%** |

Wasted spend cut at fixed coverage:

| benchmark | @50% coverage | @70% | @90% |
|---|---|---|---|
| LiveCodeBench | 5.8% vs 15.2% (**-62%**) | 9.2% (-40%) | 11.0% (-27%) |
| TACO | 19.0% vs 41.1% (**-54%**) | 24.6% (-40%) | 35.1% (-15%) |
| SWE-bench Verified | 16.3% vs 28.7% (**-43%**) | 21.3% (-26%) | 25.3% (-12%) |

**This is the cleanest statement of the contribution and it holds on all three pools, including
the out-of-domain one.** It needs no MDP, no cost model and no budget: one cheap prefill, one
ranking, and you halve the money spent on problems nothing can solve. SWE-bench Verified is the
weakest of the three (43.8% against 64.9/72.7%), consistent with its lower probe AUC — *and still
subject to the `--max-len 8192` truncation confound, so it is a floor, not a domain boundary.*

### 3b-xvii Pool extension on SWE-bench Verified — and a measurement error in C3

**AUC cannot measure label efficiency for a 2-parameter link.** AUC is invariant to any positive
affine transform, so $\sigma(az+b)$ induces exactly the ranking $z$ does, for every $(a,b)$. Held-out
AUC for the latent arm on SWE-V is therefore *identical* at N=10, 25, 50 and 100 (e.g. oss20 0.653
at all four). **Any movement with N in an AUC-based label-efficiency table is the fit recovering the
wrong *sign* at small N, not the response curve being learned.** §3b-ii's "~25 labels" is an
AUC table, so its N=10 -> N=25 improvement is a sign-recovery effect and the claim must be restated:
**the ranking transfers essentially for free; labels buy calibration, which AUC cannot see.**

Measuring it properly, with Brier on 150 held-out problems, 12 resamples, the latent fitted on the
four routes that are *not* the held-out one:

| | N=10 | N=25 | N=50 | N=100 |
|---|---|---|---|---|
| latent + 2-param link | 0.2981 | 0.2579 | 0.2455 | 0.2401 |
| the new model's own probe | 0.2772 | 0.2703 | 0.2660 | 0.2607 |
| **the new model's base rate alone** | **0.2543** | **0.2436** | **0.2375** | **0.2350** |

**On SWE-bench Verified the base rate wins at every N.** The latent ranks better than a dedicated
probe (AUC 0.616-0.670 against 0.517-0.629) but is *worse calibrated than simply knowing how often
the new model succeeds*. So pool extension does not transfer to this domain: the ordering does, the
probabilities do not, and the probabilities are what a utility rule consumes.

*Two caveats.* The own-probe Brier uses clipped ridge output rather than a fitted link, which
flatters the other two rows; the latent-vs-base-rate comparison is the fair one and it is the one
that fails. And the 8192-token truncation applies here too.

**Consequence for C3.** The five-peer LiveCodeBench result may still stand — different pool,
different domain — but it must be re-scored on Brier before it is claimed, and the "~25 labels"
phrasing has to go in favour of "ranking transfers with no labels; calibration needs N and may not
be worth it." Marked as blocking in §THREADS.

### 3b-xv The shuffled-prediction control, and what TACO's cost head is actually doing

**The null that RoR cannot provide.** With RoR's constant per-route belief *and* constant
per-route cost, the utility $p_mR-c_m$ is **identical for every problem**, so the rule can only
attempt-everything or attempt-nothing: measured, the winning `counts` arm at TACO's 0.25x budget
abstains **0.0%** and spreads 4.95 thin attempts. *Any* per-problem dispersion unlocks partial
abstention, whether or not the dispersion is informative. So "conditioned beats constant" conflates
two things, and the literature has no baseline that separates them.

**The control.** Permute the predicted vectors across problem ids. Marginal distribution is
preserved exactly; per-problem pairing is destroyed (residual corr -0.042 to +0.048). Re-run the
full replay. Anything the shuffled arm still buys is the decision rule's response to dispersion;
the rest is information. Seed 0, accuracy at the 0.25x budget:

| | head | RoR | **shuffled** | real | dispersion | information | % from information |
|---|---|---|---|---|---|---|---|
| LCB | beliefs | 26.8% | 28.8% | 52.7% | +2.0pt | **+23.9pt** | **92%** |
| LCB | cost | 26.8% | 40.4% | 53.3% | +13.6pt | +12.9pt | 49% |
| TACO | beliefs | 19.6% | 27.1% | 43.7% | +7.5pt | **+16.6pt** | **69%** |
| TACO | cost | 19.6% | 35.5% | 42.6% | +15.9pt | +7.1pt | 31% |

**The belief head survives the null on both benchmarks** — 92% and 69% of its gain is destroyed by
shuffling, so it is carrying real per-problem information. That is C1, and it is now defended
against the strongest null available rather than against a baseline that structurally cannot
abstain.

**The cost head does not, and TACO is where it fails hardest.** Only 31% of TACO's cost-head gain
survives shuffling. This resolves what looked like a contradiction: TACO's per-query cost $R^2$ is
0.081-0.309 and its cost head predicts *success* at **chance** (AUC 0.477-0.597, against
0.750-0.779 on LCB), yet the cost head still appeared to buy +23pt at a tight budget. It was
mostly buying dispersion. The +21.63% frontier floor against it at high targets is the same fact
seen from the other end: when you must attempt nearly everything, triage value vanishes and only
the price errors remain.

**So TACO never needed cost prediction, and the published TACO arm correctly drops it.** Beliefs
alone and beliefs+cost both reach 43.7% at 0.25x. The gate of SS6.9-cost is doing its job; this
section supplies the mechanism the gate was missing.

**Reporting consequence — this is the baseline the paper should add.** Not the prefill router
(priced out at 49x, SS3b-xii) and not a second selective-prediction arm, but the *shuffled-prediction
control*, reported beside every conditioned arm. It is the only comparison that separates "we
predicted something useful" from "we gave a degenerate decision rule something to vary on", and no
routing paper we surveyed runs it.

*Caveat:* one replay seed per cell (5 draw orderings within it). The belief effects are 16-24pt and
safe at this resolution; the dispersion split for the cost head should be re-run at 3 seeds before
publication.

### 3b-xiv The sequential machinery is **not** extraneous — we tried to remove it

If the probe emits a static per-problem scalar, an obvious simplification is to drop the MDP: pick
each problem's whole plan (route $m$, depth $n$) up front from the probe alone, take
$\arg\max_{m,n}(1-(1-\hat\theta_m)^n)R - n\hat c_m$, skip when that maximum is $\le 0$, and sweep
$R$. Same Lagrangian, same information, no belief updating during execution. Scored on the same
test split, 5 draw orderings, 80 values of $R$:

| budget | LCB RoR | LCB MDP | LCB **one-shot** | TACO RoR | TACO MDP | TACO **one-shot** |
|---|---|---|---|---|---|---|
| 0.25x | 27.6% | **54.2%** | 39.2% | 17.9% | **43.7%** | 35.7% |
| 0.50x | 58.7% | **59.4%** | 39.2% | 45.7% | **47.6%** | 47.0% |
| 1.00x | 66.3% | **70.1%** | 69.0% | 54.3% | **54.7%** | 50.0% |
| 2.00x | 79.0% | **79.5%** | 74.3% | 57.5% | **56.7%** | 54.2% |

**The sequential rule wins everywhere** (-1.1 to -20.2pt on LCB, -0.6 to -8.0pt on TACO). One-shot
still beats RoR at the tight budget (+11.5 / +17.8pt), so the probe carries real information
without any sequencing — but *observing that a draw failed* is worth points the prompt cannot
supply. This is the direct answer to "is the MDP machinery extraneous": **no**, and the claim in
§6.x that the probe "does not refine" is about the *probe*, not about the *policy*. Both are true:
the probe is static, and acting on realised outcomes still pays.

**Cost-normalising the prefill-router comparison.** A full prefill-routing setup needs one prefill
per candidate, not one total. On our pool that is **$0.007259/problem (LCB), $0.007841 (TACO) —
49x our single scout probe** ($0.000149 / $0.000160), or **0.16x of one always-oss120 budget unit
spent before generating a token**. At the 0.25x budget where routing matters most, the all-model
probe alone consumes **64% of the entire budget**. The budget framing prices a probe, and
per-candidate probing does not survive it. Report this rather than treating the prefill router as
un-runnable.

**Reporting recommendation:** lead with this table *including the middle column*. Cost-at-matched-accuracy is what the routing
literature reports and belongs in the paper for comparability, but "how many problems do I solve
for my budget" is the question an operator has, and it is where the method looks strongest and its
regime is clearest.

*(Superseded: an earlier attempt at this as a separate "workload allocation" experiment used a
value-per-dollar route selector that always chose the scout, so the allocation did not scale with
the budget at all -- it read +90% at 0.25x and -45% at 1.0x purely as an artifact. The frontier
already is the allocator; no separate experiment is needed.)*

## 4. Related work

### 4.1 Sequential and budgeted test-time model selection *(closest)*

- **RoR — "Resample or Reroute? Budget-Aware Test-Time Model Selection"** (**arXiv:2607.08665v1**,
  Teng-Ruei Chen). **Cite the version: v3 (2026-09-02) is a substantially different paper under a
  changed title — see `PRIOR_ART.md` §0.** v1 is what our `counts` arm reproduces: an eleven-model
  pool over four benchmarks, with *"an online resample-or-reroute (RoR) allocation policy driven by
  estimated marginal correctness per unit cost"*, reporting a favourable cost-quality Pareto front.
  v3 instead reports that It formalises resample-vs-reroute as competing uses of a second call after a
  fallible verifier accepts a candidate, and establishes that recoverable stopping debt exists
  (+2.59pp on MBPP+). It is a **negative-result** paper: *"current evidence does not identify when
  to resample rather than reroute."* It has **two** primary models (Qwen2.5-7B/14B), not eleven —
  that was RouterBench conflated in — and **no budget-allocation framework**; it disclaims the
  greedy rule as *"a transparent heuristic, not a proof of horizon-optimal control."*
  **Therefore our `counts` arm is our own construction and must not be labelled "RoR as
  published".** Rename it *count-belief greedy allocation under a per-query cap*. Correct in the old
  entry: no stop action, count-based beliefs, per-model constant costs by parameter count.
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
the ENTRY decision** -- which is what licenses the constant *in the low-budget regime*, and is a
sharper claim than "we simplified".

**CORRECTION (2026-09-10). The original sentence read "does not change any decision" and that is
false.** At $n=0$ the decay factor is $\sigma/(\sigma+0)=1$ exactly, so $\sigma$ cancels from the
entry condition $\theta R \ge c$ -- that part is right. But $\sigma$ is precisely the factor
multiplying *observed failures*, so it governs every **continue** decision, and continue decisions
are the whole sequential half of the method. The measurement that licensed the constant was taken
at $R=\$0.05$, where $n^*$ floors at zero and nothing about depth matters at all. Generalising from
a regime where a quantity is inert to "the quantity is inert" is the §8 R0 error again.

**Depth is not inert at the budgets where we underperform.** Mean $n^*$ on gpt-oss-120b, the only
route where depth pays:

| budget | 0.25x | 1.00x | 2.00x | 3.00x |
|---|---|---|---|---|
| $n^*$ (oss120) | 0.34 | **2.88** | **4.40** | **5.52** |

**And $\sigma$ is strongly heterogeneous.** Realised $P(\text{success on next draw}\mid n$ prior
failures$)$ on oss120, pooled over random orderings, split by predicted-difficulty tercile:

| tercile | $n{=}0$ | $n{=}1$ | $n{=}2$ | $n{=}3$ | **implied $\sigma$** |
|---|---|---|---|---|---|
| hard | 0.532 | 0.248 | 0.136 | 0.091 | **0.69** |
| middle | 0.912 | 0.538 | 0.261 | 0.127 | **0.80** |
| easy | 0.982 | 0.840 | 0.750 | 0.692 | **6.45** |

A ~9x spread against our single global $\sigma=0.95$. The constant is about right for hard and
middle problems and **~7x too small for easy ones**: after two failures on an easy problem we have
cut the belief by more than half when the data says a failure there is mostly draw noise. $\sigma$
encodes *how much of the residual uncertainty is problem difficulty rather than draw noise*, and
that ratio is exactly what varies with difficulty.

**This is now a candidate explanation for the 13% capture rate at 1.0x budget** (§3b-xxvii), and it
is directly testable: swap the global $\sigma$ for a per-tercile $\sigma$ and re-run. If $\sigma$
is the culprit the gap should close at 1.0x-2.0x and not at 0.25x, where $n^*$ floors.

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

## 7b. Applications tested and rejected — do not retry without a new representation

Every application that needs more than the **shared scalar** fails. Recorded with numbers so the
search is not repeated.

| application | result | why it fails |
|---|---|---|
| single-shot routing, success-based | +12-13% acc/$, 22-31% of oracle | needs the model x problem interaction |
| routing on a *complementary* pool (5 labs, 52.6% contested, +13.5pt available) | router collapses to one model, 171/171 picks | 1-D latent gives a constant model ordering |
| multidimensional latent (8 PCA dims) | **worse** (31.4 vs 41.4 acc/$) | added variance, no interaction signal |
| per-model probe on activations | **worse still** (22.9 acc/$) | same |
| cost-based routing | 35.3 acc/$ vs 41.4 for ignoring cost | cost *ratio* $R^2 \le 0$ on all 10 peer pairs |
| adaptive evaluation (Fisher-information item selection) | **10% worse than random** | 2-param model with global discrimination is misspecified |
| difficulty-stratified evaluation (reweighted mean) | ~14% lower error, ~26% fewer calls | modest, real, but not a contribution on its own |

**What survives is one thing, and it is genuinely not in the literature:**

> **Cross-model selective prediction.** Read difficulty from one cheap prefill *before anything
> runs*; transfer it to models whose weights you never touch at **~25 labels each** (ties a
> 170-label dedicated probe); use it for the **whether-to-attempt** decision under a cost budget
> (worth up to **+43.7%**); and it works on **API-only models**, where same-model confidence
> methods (logits, entropy, auxiliary heads) cannot run at all.

IRT derives difficulty from response patterns; IrtNet from sentence embeddings, with no abstention
and no cost-awareness; prefill-activation routers do single-commit routing with no give-up action.
None of them do this.

## 8. Retracted — do not resurrect

**R0 (2026-09-08) — a retraction that was itself wrong, kept as a worked example.**
For a few hours this document claimed C1 was retracted because "the cost head carries essentially
all of the gain." **That was incorrect and is withdrawn. C1 stands** (floors +2.90% / +0.93%,
+26.5pt / +25.8pt at 0.25x budget; §3b-xii).

*The mistake:* to test whether our **beliefs** beat RoR's, I compared our full arm
(`content_decay_qcost`) against `counts_qcost`. But `counts_qcost` is RoR's beliefs **plus our own
cost head** — it subtracts our contribution from the *baseline* side. That comparison answers "do
the beliefs add anything **on top of our cost head**?", which is a redundancy question. The answer
is no, and I read that as "the beliefs do nothing." What it actually means is that the two heads
are **substitutes** (C1b) — a result, not a failure.

*The rule this yields:* **to attribute a gain to component X, hold every other component fixed on
both sides.** `content_decay` vs `counts` (beliefs differ, both constant costs) is the belief test;
`counts_qcost` vs `counts` (costs differ, both count beliefs) is the cost test. An arm that changes
two things belongs in neither cell. The five-row table in §3b-xii is the shape every future
component claim in this project must take.

*What was genuinely wrong in the original, and is fixed:* (i) the C1 floors +6.28%/+4.37% were
from a pre-recollection run and are now +2.90%/+0.93%; (ii) the arm was not named, and it differs
per dataset (LCB uses the cost head, TACO does not); (iii) the first budget table omitted the
single-component columns, which is what let the misreading happen at all.

*What the false alarm did surface, and is worth keeping:* the cost-vs-success $R^2$ split, the
finding that prompt length is worthless as a cost proxy (negative on TACO), and the 49x
cost-normalisation of per-candidate prefill routing.

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
