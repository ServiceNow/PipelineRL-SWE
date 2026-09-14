# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-10.
Companion to `PAPER_OUTLINE.md` (the paper as it stands), `PRIOR_ART.md` (what we position
against), `RESEARCH_LOG.md` (append-only history). This file answers: *what is running, why, and
what does it decide?*

---

## -1. RUNNING RIGHT NOW (2026-09-14 19:0x) — what each job decides

All local jobs are CPU-only replays on the LCB 64k pool with the matched geometric budget grid
(§3b-xxxv), seed 0 unless stated. Output root `/mnt/llmd/results/exps/aristides/reason/gridmatch/`.
Launchers are `run_*.sh` in that directory; every one deletes its `episode_traces.jsonl` on exit
(they run 0.7–5 GB each).

| job | dir | state | question it decides |
|---|---|---|---|
| **learned sigma head-to-head** | eai `lcb_fact64k_h2h_seed17_1789410753` | **QUEUING** (dataset stage done, 138 MB) | Does a **learned per-problem $\sigma$** beat our probe? §3b-xxxi: this comparison has never existed — never on the 64k pool, never with a `content_decay` arm in the same replay. One replay, so seed / draw orderings / budget grid cannot confound the belief source. Launcher `launchers/abstention/launch_lcb_factorized_64k_headtohead.sh`. |
| **winner's-curse shrink** | `shrink/shrink` | running ~9 min | Is the give-up firing **too late** because $\max_m(\hat p_mR-\hat c_m)$ is selection-biased upward? Sweeps $\lambda \in \{0.25,0.5,0.75,1.0\}$ in $\max - \lambda(\max-\text{mean}) \le 0$. Predicts the measured 63–86% waste. Routing untouched. |
| **free_start refresh** | `shrink/freestart` | running ~9 min | How much does the **mandatory scout generation** cost us? `scout_first` pays a full scout draw (\$0.00084, 5.6x the probe) on 100% of episodes. Known good on the *old* pool (+50.8/+13.6/+10.8/+4.9%); never re-run here. It is the only way "we decline before spending" becomes literally true. |
| **cross-route rho** | `rho/rho` | running ~3 min | The Bellman lattice reuses the root belief at every node — its own docstring says this "asserts that failing route m says nothing about route m'". False, so "try another route" is over-valued. Sweeps $\rho\in\{0.1,0.25,0.5,0.75\}$ discounting a route by $(1-\rho)^{\text{failures elsewhere}}$. h=2 only. |

**Finished, awaiting analysis:**

| job | dir | what it tests |
|---|---|---|
| two-constraint policy | `capped/{content_decay_qcost,counts}` | Cap **and** price swept jointly (12 caps x 96 prices). Motivated by the anomaly that the capped arm wins at 70–80% yet cannot reach 84%. Given to **both** arms so a win is not just a richer policy class for us. |
| quantile cost head | `quantile/q{mean,0.25,0.50,0.75,0.90}` | Is $\mathbb{E}[c]$ the wrong functional? Duan's smearing rescales $\exp(\text{log-fit})$ from the conditional median to the mean; this swaps in the $q$-th quantile of the same residual law. Includes a **matched mean arm** so the comparison is not against a differently-fitted stored file. |
| posterior-over-$k$ head | `post/s{0,1,2}` | Exact Bayes on the success count instead of Beta-Bernoulli. $k$ is strongly bimodal (49%/34% at 0/6 for scout), which $\theta\sigma/(\sigma+n)$ cannot represent. **No $\sigma$ at all.** 3 seeds. |
| matched-grid 2x2 | `runs/lcb_s{0..4}`, `runs/taco_s{0..2}` | LCB analysed (§3b-xxxvii). **TACO not yet analysed** — free, no compute. |

**Ideas considered and dropped, with the reason:**

- **Monotone beliefs across routes** ($p_\text{scout}\le p_\text{oss20}\le p_\text{oss120}$). Measured
  first: the binary complementarity is tiny (scout-solves-and-oss120-does-not 0.3%, oss20 0.7%), but
  **per-problem pass-rate monotonicity is violated on 16.5% of problems**. Baking it in would be
  wrong one problem in six.
- **Joint (p, c) head.** Substantially done: §3b-xiii measures corr −0.60/−0.52/−0.71 on LCB with
  **PC1 carrying 71.4%**, and the rank-1 collapse is already built and is an *improvement*. What is
  left is the joint *uncertainty* (variance of the surplus), which folds into the shrink and rho
  arms rather than being a fourth head.

**Two bugs found while checking what the probe is charged (fixed, `ae9bbbe`):**
`content_post` and `content_commit` were never charged `--probe-cost-usd` (my new arm would have run
with a free probe); and the **fixed-model reference points** were charged the probe off a *stale*
`family` variable from an earlier loop, so the single-model diamonds could be inflated by \$probe
depending on loop order.

---

## 0. Headline as it now stands (2026-09-14, evening)

### Read this first: two things found on 2026-09-14 evening

**1. `content_preds_RANK1.jsonl` is CONTAMINATED — do not use it.** Test-split AUC **0.9980** and
correlation **0.9061** with the true per-problem rate, against 0.7685 / 0.5315 for our honest probe.
A rank-1 projection can only lose information, so the file was built with test labels in scope. It
surfaced because the belief ladder's RANK1 arm beat our own at every target and came within a few
points of the oracle. **§3b-xiii's "collapsing to one scalar is an improvement" may rest on this
artefact and is unsupported until re-fitted train-only.** Unlike the `*_ORACLE*` files it is not
labelled diagnostic, so it was being read as a method arm.

**2. The 19/19 claim is about the ACTIVATIONS, not merely about having a per-problem prior**
(§3b-xlix). Belief-source ladder, only the belief source varying, constant costs both sides, no cost
head anywhere, TF-IDF and length fitted by the **same** protocol as the activation head:

| belief source | forward pass? | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|---|
| problem LENGTH | no | +2.7% | −2.8% | −2.2% | +5.3% | +4.7% |
| TF-IDF of statement | no | +12.2% | −0.3% | +11.8% | +1.2% | +0.3% |
| kNN on activations | yes | +26.8% | +8.8% | +7.6% | +0.8% | +3.3% |
| **activations (ours)** | yes | **+40.8%** | **+19.2%** | **+12.0%** | **+2.3%** | **+6.6%** |
| *ORACLE beliefs* | — | *+75.4%* | *+71.8%* | *+68.7%* | *+65.5%* | *+73.4%* |

Length buys nothing and is negative at two targets; a properly-fitted TF-IDF head recovers at most a
third and is negative at 60%. **But we capture only 54/27/17/4/9% of the oracle belief channel.**

**3. The direction this sets, which reverses an earlier hypothesis.** §3b-xliv argued the stopping
channel was nearly closed. Oracle arms on the current setup say otherwise at the *other* end:
**perfect stopping is worth +44.0%/+65.0% at the 80/84% targets** and every variant we ran moves
that regime by ~0. Combined with perfect beliefs being worth +65.5%/+73.4% there while perfect
*routing* is worth only +8.3% at 84%: **the loose end is a BELIEF-QUALITY problem, and it is where
all the headroom is.** Stop tuning the give-up rule at the tight end where four knobs already
overlap. *(Tight-target oracle cells are unreadable — the oracle arms attach to the frozen-$R$
policy with 4–5 points, so its hull cannot span the tight end; reported as n/a, not as numbers.)*

**4. Entry-vs-continue was confounded and is being redone.** Under `scout_first` the scout draw is
mandatory, so `failures.sum()==0` never occurs in the decision loop and the "entry only" arm never
fires — it returned ≈0%, exactly as that diagnosis predicts. Re-running under `free_start`.

**5. Quantile cost head (§3b-li):** only $q=0.90$ is positive at every target (+3.8/+1.2/+1.5/+5.5/
+0.8); the optimistic tail is negative at the tight targets. Right sign for the asymmetry argument,
small magnitude, single seed. Report as directional; do not build on it.

---

### The clean claim — lead with this

**Replacing count-based beliefs with beliefs read from one cheap prefill, changing nothing else,
cuts cost at matched accuracy at every accuracy target on every pool we have.**

Belief head isolated: formulation held at the global dual, **costs constant and identical on both
sides**, so only the belief source differs.

| pool | | | | | |
|---|---|---|---|---|---|
| **LiveCodeBench** (5 seeds) | 50% **+40.5±1.0** | 60% **+19.6±2.4** | 70% **+16.5±2.6** | 80% **+4.6±1.6** | 84% **+5.4±2.1** |
| *sign test* | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| **TACO** (3 seeds) | 35% **+33.6±1.1** | 40% **+29.3±0.2** | 45% **+2.2±2.0** | 50% **+7.3±1.9** | 55% **+1.1±0.8** |
| *sign test* | 3/3 | 3/3 | 2/3 | 3/3 | 2/3 |
| **RouterBench** (14,599 test) | 60% **+35.2** | 70% **+76.9** | 75% **+75.1** | 80% **+50.8** | 84% **+22.9** |
| **SWE-bench Verified** | 30% **+19.5** | 40% **+9.5** | 50% **+3.4** | 55% **+1.2** | |

**19/19 targets positive**, four pools, two task families, unanimous across seeds at 17/19.
(§3b-xlvi. SWE-V row uses a deliberately weak probe reconstruction; the recorded probe is stronger.)

### The paper shape

1. **Representation claim** — the table above. Unconditional. The headline.
2. **Formulation contribution** — the **two-constraint policy** (per-episode cap *and* global price
   swept jointly): **+4.1/+7.1/+11.4/+2.4%** over the price alone at all four reachable targets,
   while the cap *alone* is **−61.7%** at 50%. Neither RoR's cap nor the pure Lagrangian. Open:
   it inherits the cap's ceiling and cannot reach 84%.
3. **Cost head** — a second representation contribution **with a documented failure mode**: a large
   win on LCB (+39.4±2.9 tight) and **actively harmful on TACO above 40%** (−21.6% at 45%). Report
   separately from the belief head and select on **calibration**, never post hoc on test.

### Why this looked like a pile of conditionals until now

Every reported arm **bundled** the belief head with the cost head. The belief head is universal; the
cost head is pool-dependent, exactly as §3b-xiii predicted (on TACO the two heads are near-orthogonal
— PC1 43.7% against LCB's 71.4% — and TACO's cost $R^2$ is the weakest we measure).

### Two retractions from today

- **TACO's bundled cost claim above 40%** (§3b-xlv). §3b-xxxiv recorded +45.7/+42.5/+21.4/+13.1/+9.1
  vs RoR; on the matched grid it is **+35.5/+20.2/−5.7/−12.5/−31.8**, losing at three of five
  targets with 0/3 seeds. The old numbers used the **linear** budget grid.
- **"TACO fails because `oss20` ≈ `oss120`"** (§3b-xlvii) — **false**. All three routes sit on the
  hull and the gap is a real 7.3pp. The difference is **price**: TACO's top rung costs **\$0.49 per
  unit accuracy against LCB's \$0.18**, 2.7x dearer, so the expensive rung must be bought at loose
  budgets and mis-pricing it is directly costly. **Do not rebuild the TACO pool** — it passes any
  pre-registered structural criterion, so dropping it would be selection on the outcome, and the
  belief claim survives it anyway.

### Method variants, all six run (§3b-xliv)

| variant | 50% | 60% | 70% | 80% | 84% | verdict |
|---|---|---|---|---|---|---|
| two-constraint (cap x price) | +4.1 | +7.1 | **+11.4** | +2.4 | — | **works** |
| posterior over $k$ (exact Bayes, no $\sigma$) | +11.0 | +5.7 | +5.9 | +0.9 | −6.5 | **works, tight/mid** |
| `free_start` (ours) | +9.2 | +2.2 | −1.4 | −2.2 | −1.4 | marginal |
| winner's-curse shrink | +12.1 | +1.2 | +0.8 | +2.1 | +0.3 | marginal |
| cross-route $\rho$ (vs its h2 control) | +8.0 | +2.0 | −0.7 | −3.6 | −2.9 | mixed |
| quantile cost head | — | — | — | — | — | void, re-running |

**The pattern that matters:** four of five peak at the 50% target around +8 to +12% and fade or
reverse by 80%. That is where abstention is active (50.9% abstain at 50%, 6.8% at 84%), so they are
all the *same* intervention — improving the give-up decision — and will stack **sub-additively**, as
the belief and cost heads did. **Stop adding stopping tweaks.** The two-constraint policy is the
exception: it peaks in the middle and is about the feasible set, not stopping. *An oracle-stopping
run is in flight to bound what is left in that channel.*

`free_start` is a **mechanism confirmation**: +9.2/+2.2 for our arm and **exactly +0.0% at every
target for RoR**, because count beliefs are identical at entry so RoR cannot skip the scout
*selectively*. Third independent instance of the selective-vs-indiscriminate mechanism.

### Where a clean win could still live

- **Not more stopping tweaks** — four knobs, one channel, redundant.
- **The loose end is untouched.** Our margin at 80–84% is +3.2 to +7.2% and nothing has moved it. At
  high $R$ nothing is declined, so it is pure routing and depth; oracle cost is the largest lever
  there (+9.8pt at 1.0x).
- **Richer policy classes** — the two-constraint result says the class was leaving money on the
  table independent of predictions. Fixing its 84% ceiling would be a real contribution.

---

## 1. Backlog thread: the stale-multiplier fix (adaptive R) — CLOSED, kill criterion fired

**Result: -16.5 to -26.5pt against static R.** The Lagrangian optimum is a single multiplier; dual
descent is for online settings where the price must be learned, and we tune R offline on a
calibration sweep, so adaptation only adds noise around a value that was already right. The
receding-horizon and index-policy escalations are therefore **not worth building**.

**The setup, stated correctly.** Both our sequential MDP (`_value` arms) and the one-shot knapsack
(§3b-xiv) solve a **global** budget in the **dual**: pick each problem's plan under a common
multiplier $R$, sweep $R$ to trace the frontier. This was previously described in this file as
"per-problem budget" — that was wrong, `hull()` pools budget-swept and $R$-swept arms and the
latter are the Lagrangian.

**So the difference between them is adaptivity alone**, and it is worth a lot: identical global
formulation, identical predictions, and the MDP beats the one-shot knapsack by **+1.1 to +20.2pt**
purely by re-reading $\theta\sigma/(\sigma+n)$ after each observed failure.

**What is still static in both: $R$ itself.**
- $R$ is chosen from **predicted** costs, and our cost $R^2$ is 0.545 (LCB oss120), not 1.0.
- Realised spend therefore drifts from plan, and **nothing corrects the drift** — overspend early
  and the policy keeps buying at the original price signal for the rest of the batch.

**The fix.** Dual descent: update $R$ against realised spend as the batch progresses. Standard
online-knapsack correction, no change to the per-problem rule, just a running multiplier.

**Kill criterion.** If adaptive $R$ gains nothing, global coupling is not the missing ingredient
and the more elaborate versions below are not worth building either.

**If it does pay, in order:** receding-horizon knapsack (re-solve over remaining problems/budget
after each draw); index policy with option value (our myopic $p R - c$ is the Whittle/Gittins index
with option value set to zero, which is why it under-invests in problems that pay off on draw 3);
hybrid split (knapsack for the extensive margin, sequential rule for the intensive one).

---

## 2. The finding that should shape everything else

**Selecting a component on its own predictor metric degrades the policy.** Four independent cases
(§3b-xxx):

| intervention | predictor gain | frontier effect |
|---|---|---|
| belief $C$ on AUC | +0.024 AUC | -4.34pp TACO floor |
| cost $\alpha$ on $R^2$ | 4/6 cells | -3.2pt LCB 0.25x |
| cross-route coupling | +0.024 AUC | -23.8pt LCB 0.25x |
| isotonic recalibration | perfect by construction | -0.4..-1.9pt |

Cause: AUC is invariant to monotone transforms; the rule $\arg\max_m(p_mR-c_m)$ consumes **values**
and abstention fires on $\max_m Q\le0$. Better-regularised predictors are more **compressed**; the
rule needs **spread**.

**Fix and its limits.** Frontier-selected $\alpha$ (`--eval-split calibration`, test touched once),
LCB 5 seeds: **+3.88pt at 0.25x (5/5 seeds), +2.34pt at 0.50x (5/5), but -2.60pt at 1.00x (1/5)**.
It **trades regimes rather than dominating**. Selection is noisy at n~170: TACO picks $10^6$ where
LCB picks $10^4$ (good — per-pool), but TACO's curve is non-monotone with its two best values tied
at opposite ends of the grid. **The criterion reliably rejects extreme regularisation and cannot
discriminate within an order of magnitude.**

---

## 3. Blocking before any draft

1. **Re-run the greedy-knapsack comparison properly.** The arm reported earlier picked its route
   **best-of-3 on test**, was a **single point against the MDP's hull**, and drew to exhaustion per
   problem rather than at fixed depth. **The "a published baseline beats us above 0.30x" claim was
   retracted on those grounds.** The correctly-specified version is the one-shot knapsack of
   §3b-xiv, which we win.
2. **More TACO seeds.** Three is why its floor CI spans zero; strict improvement is dead at 3 seeds
   under both belief heads (P=0.514 shipped, 0.000 selected).
3. **Report the -2.60pt at 1.0x in the table**, not a footnote, alongside the +3.88pt.
4. **Independent replication** of the headline numbers. See §5.

---

## 4. Not running, deliberately

| | why |
|---|---|
| prefill ensembling / tiny-model committee | cross-family ensembling gains **nothing**: best single encoder 0.859, every 2- and 3-encoder combination 0.851-0.858 across concatenation, mean, rank and stacked blending. Encoders read the same shared factor (C4), so there is no error diversity to exploit |
| better readouts of the same prefill | §3b-xxviii: the bottleneck is information, not extraction |
| cost head from generation signals | scout's realised output length is 15x better than prompt length (0.254 vs 0.017) but adds **+0.003** over activations |
| cost head on RouterBench | 78% multiple-choice, prompt length already gives cost $R^2$ 0.97 |
| SWE-V pool extension | base rate wins at every N to 219; a calibration failure, not a fixable one |

---

## 5. Reliability note

**Six errors this session, each caught by a control or by the user rather than by inspection**, and
each in the flattering direction: the `counts_qcost` attribution; a hand-set ridge penalty on one
arm of a representation comparison; disabled mixtures on RouterBench; global cost constants in a
per-dataset evaluation; train/test overlap from mismatched splits; running the TF-IDF baseline as
if it were our method. Plus three claims stated too broadly and retracted (the scope law, the
ranker-vs-estimator thesis, "learned sigma was never replayed"), and a `content_decay_coupled`
dispatch bug that silently ran the arm as `counts`.

**Consequence:** every headline number needs reproduction by someone else before submission. The
common failure mode is *measuring in one regime and generalising*, so the check should specifically
re-run each claim at a budget/seed/pool it was not measured at.
