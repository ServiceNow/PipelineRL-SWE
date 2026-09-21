# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-19 00:35 UTC.
Companion to `PAPER_OUTLINE.md` (findings, newest in §3b-lxviii – §3b-lxxxii), `PRIOR_ART.md`,
`RESEARCH_LOG.md`. This file answers: *what is running, why, and what does it decide?*
Replay outputs live under `/mnt/llmd/results/exps/aristides/reason/gridmatch/<run>/`; launchers are
`run_*.sh` there; every replay deletes its `episode_traces.jsonl` on exit.

---

## -1. RUNNING RIGHT NOW — what each job decides

| thread | where | state | decides |
|---|---|---|---|
| **Failure-reading, TACO fix** | `gridmatch/hist3` (`run_hist3.sh`) | 8/16 done (LCB + TACO constant-cost done; full-method runs going) | Does failure-reading survive on TACO once the C/D recalibration uses 0/1 "model has failed" instead of the count (§3b-lxxiv)? Also TACO's full method vs baselines for the first time. |
| **With-replacement posterior** | `gridmatch/postwr` (`run_postwr.sh`) | 0/5 (started ~00:05) | Does the success-count posterior still beat the Beta decay once its replay-only finite-pool edge is removed (§3b-lxxvi)? LCB, 5 seeds, constant costs, h = 1 and 2. |
| **Deep-history probe** | `history_probe/act_deepjudge_shard*` | **extraction done**; screen not yet run | Can the probe learn the decay itself (§3b-lxxxii)? Next: `history_probe_eval.py --variant deep --act-tag deepjudge --readouts last`, then a per-depth breakdown (target model already failed 0/1/2/3+) of the no-decay probe vs prompt probe + decay. |

## 0. Where things stand (2026-09-19)

**Method in its best current form (LCB):** prompt probe + **failure-reading** (the scout re-prefills
problem + failed code + a yes/no question, last-token readout) + 0/1 "model has failed"
recalibration, with the per-query cost head and cap × price. vs agreement +32 / +37 / +26 / +16 /
+0.2 / +10 and vs RoR v1 +31 / +34 / +27 / +19 / +8 / +12 at 50–84% (5/5); 55% cheaper than always
calling gpt-oss-120b at its accuracy. TACO pending the fix. **Fair headline** needs the linked-cap
(one-sweep) version of this method — not yet run.

**Findings that shape the paper (details in PAPER_OUTLINE):** the value is almost all *whether*, not
*which* (§3b-lxxv; the current tex title "Which Tier, Not Which Peer" overclaims); the count decay is
wrong in shape and ignores cross-model evidence (§3b-lxxvii); resampling is selective — the question
is fluke vs never (§3b-lxxviii); the one-step rule's stopping is optimal under its beliefs (§3b-lxxii);
a 3-rung pool lets "try each tier once" compete (§3b-lxx).

## 0a. THE PLAN: belief models trained on the policy's own visited states (2026-09-21)

**The problem it fixes.** Our belief heads are trained on a distribution of states we chose, not the
one the policy visits. The deep-history probe was fitted on ~12 histories per problem sampled
uniformly over depths 1-6; a tight-budget policy almost never passes depth 1, a loose-budget one
lives deep. One pooled calibration therefore splits the difference: measured on LCB seed 0, the
no-decay deep reader is **+18.8 / +12.8% at the 80 / 84% targets and -56.5 / -18.1% at 50 / 60%**
against today's beliefs. Per-depth calibration is a patch for this; training on the policy's own
visitation is the principled version (standard covariate-shift / DAgger-style correction), and it
subsumes the patch.

**The key timing fact: R does NOT enter at extraction time.** A state's prefill depends only on its
text (problem + trajectory summary + the attempt that failed last), so one extraction of the
reachable lattice serves every price R. Only the linear heads depend on R, and only through WHICH
states get visited -- and they fit in seconds. So per-R (or per-budget-regime) beliefs are cheap,
and an operator can refit for their own budget in seconds.

**Loop, per pool:**
1. Extract the reachable state lattice once (build_deep_state_prompts.py; 30.7k states for LCB test,
   seed 0 x 1 ordering; train/cal states as well for fitting).
2. Start from today's belief (prompt probe + decay).
3. For each R on a coarse grid spanning the frontier: replay on TRAIN problems, record visited
   states, refit the head and its calibration on that visitation (CAL visitation for calibration),
   iterate 1-2 times.
4. Evaluate on test ONCE per R; the frontier is the hull over R.

**Rules.** Visitation for fitting comes from train/cal problems only -- never test. Report the
matched one-sweep (linked-cap) version. Keep a structural prior as a fallback: at tight budgets the
rule needs probabilities in the 0.3-2% range, where a learned head is weakest and the decay (or the
success-count posterior) behaves sensibly; prior + learned correction may beat either alone.

**Do this on the RECOLLECTED pool, not this one** (see 0b): at T=0.2 draws are near-copies, so depth
and resampling decisions are being tuned in a regime that will not exist at recommended temperatures.

**Also queued for the new pool:** a proper sentence-embedding baseline (the IRT / ZeroRouter line
uses embeddings; our only text baseline today is TF-IDF).

## 0b. Next, in order (nothing launched)

1. **RECOLLECTION FIRST** (was 3): everything below is better measured on the new pool.: 6-rung pool from the pilot (scout-Instruct, gpt-oss-20b low, Qwen3-4B-Thinking,
   gpt-oss-20b medium, gpt-oss-120b medium, gpt-oss-120b high) at recommended temperatures, multi-draw,
   on **LiveCodeBench + BigCodeBench (+ CodeContests replacing TACO)**. Needs: BigCodeBench grader
   (`bigcodebench` 0.2.5 on PyPI; validate every reference solution first), CodeContests conversion.
   Then re-extract prompt + history activations on the new pool, and run the visitation-trained
   belief loop of §0a there. Collect ASYMMETRIC draw depth (~16 on the cheap rungs, ~8 middle, 4 on
   the 120B) so per-problem depth can matter -- the one thing a fixed plan cannot do.
2. **Linked-cap (one-sweep) headline** for whichever belief model wins on the new pool.
3. **Rolling-origin LCB evaluation** to lift the 171-problem test set (~3×).
5. **Paper**: refresh numbers (fixedacct2 / linked / failure-reading), decide title (recommend back to
   "Whether, Not Which" or "Pick Your Fights, Not Your Knights"), add trajectory + Zero Router figures,
   the decay diagnosis, whether × which. Not compiled locally (no TeX on the box).

## 0c. Parked

- "$ saved vs Claude for everything" framing (§3b-lxxix): add Opus 5 / Sonnet 5 to the pilot first.
- CodeRouterBench as a free, larger "which peer" test (one scout-prefill extraction).
- Weak-verifier thread: per-problem verifier trust lifts the reachable ceiling 68.4% → 72.6% (one
  seed, `gridmatch/optaccept/oa_perproblem`).
- Calibration-selected probe penalty (+3–10 pts) and the same-model cost update (LCB +, TACO −): fold
  in via per-pool configuration chosen on calibration.
- Agreement-gating dropped from the trajectory figures; still in the paper's tables.

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
