# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-10.
Companion to `PAPER_OUTLINE.md` (the paper as it stands), `PRIOR_ART.md` (what we position
against), `RESEARCH_LOG.md` (append-only history). This file answers: *what is running, why, and
what does it decide?*

---

## 1. Next thread: the stale-multiplier fix (adaptive R)

**Nothing is running. This is what to do next.**

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
