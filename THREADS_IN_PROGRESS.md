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

## 0. Headline as it now stands (2026-09-14)

**Two defects found and fixed in the same pass.** (1) RoR (2607.08665) budgets **per query**; our
`_value` arms run with `unconstrained_budget`, so they are a **global**-budget method — and
abstention-under-a-global-budget is **ROI-Reasoning's** (2601.03822), recorded in `PRIOR_ART.md` §4b
as not ours. (2) The budget arm was swept on 17 **linear** points against the value arm's 96
**geometric** ones — 3 points below \$0.02 against 55.

**The 2x2, on the matched grid** (LCB pool64k, seed 0). Row = representation (**ours**), column =
formulation (**theirs**):

| contrast | isolates | 50% | 60% | 70% | 80% | 84% |
|---|---|---|---|---|---|---|
| ours vs `counts_value` | **representation**, global dual held | **+45.8%** | **+15.7%** | **+15.2%** | **+5.5%** | **+10.5%** |
| `content_decay_qcost` vs `counts` | **representation**, RoR's own cap held | **+16.9%** | **+6.7%** | **+18.9%** | **+5.8%** | n/a |
| `counts_value` vs `counts` | formulation alone | +5.1% | +5.8% | +1.8% | −0.2% | −3.3% |
| ours vs `counts` | everything | +48.6% | +20.6% | +16.7% | +5.3% | +7.5% |

**The matched grid cost us 6.3 and 6.9 points at the two tight targets** (54.9→48.6, 27.5→20.6) and
nothing at 70% and above. **But it moved the attribution in our favour:** formulation alone fell
from +16.7%/+13.9% to **+5.1%/+5.8%**, and is negative at 80% and 84%. Most of what looked like "the
global budget is worth a lot when money is tight" was RoR's missing grid points.

**Headline to lead with — anchored, and fully attributable:**

> At the accuracy `gpt-oss-120b` reaches when called on every problem (68.77%), the router costs
> **\$0.0341 against \$0.0452** — **24.5% less than calling that model on everything**, and **17.8%
> less than RoR**. At this operating point our arm under RoR's *own* per-query formulation lands at
> the same \$0.0341, so the entire margin is the representation and none of it is the formulation.

**Mechanism (structural, from the code not a fit).** Count beliefs are $s\pi_m/(s+n_m)$: at $n_m=0$
**every problem carries the identical belief vector**, so `counts_value` **cannot abstain
selectively at entry** — it must buy failures to discriminate. It abstains at **41.6%** at the 50%
target and still costs more than us. *Abstention only pays if you know what to abstain on before
spending.* Hence the interaction: **+27.5pt** over independent components at 50%, +8.5pt at 60%.

**Graphs:** https://claude.ai/code/artifact/23d6da83-6b81-46d8-bf0e-749cf68af553

**Status of the three other pools:** the 2x2 has **not** been run on TACO / SWE-V / RouterBench, and
their numbers are still the conflated row on the old linear grid. **No paper headline can be fixed
until that decomposition is run on all four** — and on the tight targets it should be expected to
come down, as LCB's did.

**Blocking before a draft:** the 2x2 + matched grid on the other three pools; seeds (this is n=1);
bootstrap SWE-V (one split only); more TACO seeds; independent replication (§5).

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
