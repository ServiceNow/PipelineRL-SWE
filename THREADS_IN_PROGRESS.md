# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-10.
Companion to `PAPER_OUTLINE.md` (the paper as it stands), `PRIOR_ART.md` (what we must position
against), `RESEARCH_LOG.md` (append-only history). This file answers one question: *what is
running, why, and what does it decide?*

---

## 1. Jobs in flight

| job | what it is | decides | ETA |
|---|---|---|---|
| `iso_lcb_s0_2134522074` | our theta-hat, isotonically recalibrated against truth | **is calibration the bottleneck?** | ~5 min |
| `rank1_lcb_s0_3045219557` | one-factor oracle: perfect *shared* difficulty, zero interaction | ceiling for any prompt-only probe | ~5 min |
| `orc_lcb_s1`, `orc_taco_s0/s1` | oracle per-(problem,route) theta | does seed 0's headroom hold up | done/near |

**These four settle the central open question** (§2 below). Nothing else is running.

---

## 2. The open question: is our probe a ranker or an estimator?

**Thesis (§3b-xxvii).** The probe orders problems well and estimates their probabilities badly.
Tight budgets need only an *ordering* (which problems to attempt — errors near the cutoff swap
similar-value problems and are nearly free). High budgets need *calibrated values* (whether a 4th
draw at p=0.15 beats one at p=0.35), which a ranking cannot supply.

**Evidence it already explains:** C3 (ordering transfers free, calibration costs ~25 labels);
§3b-xxvi (selecting C on AUC improved ranking and *hurt* the policy); the SWE-V pool-extension
failure (reliability 0.0389 > resolution 0.0254); and the low-vs-high budget split itself
(we capture 63% of oracle headroom at 0.25x, **13% at 1.0x**).

**The decomposition in flight**, at each budget:

| gap | measures | if large |
|---|---|---|
| ours -> ISO | perfect calibration of *our own* ranking | fixable without a better probe |
| ISO -> RANK1 | perfect shared difficulty | our ranking is weak; better representations pay |
| RANK1 -> ORACLE | model x problem interaction | unreachable by any prompt-only probe (C4) |

**Prediction:** ours->ISO is large at 1.0x and small at 0.25x. If instead ISO barely moves and
RANK1 does, the thesis is wrong and ranking is the weak link.

---

## 3. Blocking before any draft

1. **Add Greedy Knapsack as a first-class baseline** (`PRIOR_ART.md` §4). It is a *named baseline*,
   not our method, and it **beats our MDP on LCB above 0.30x** (+3.8pt at 0.50x, +2.4pt at 1.00x).
   Run it with the route chosen on calibration, not best-of-3 on test.
2. **Cite Predictive Scheduling (2602.01237)** as nearest prior work for the allocator, and the
   prefill router (2603.20895) for cross-model. Reviewers in this area will know both.
3. **More TACO seeds.** Three is why its floor CI spans zero; the strict-improvement claim is dead
   at 3 seeds either way (P=0.514 shipped, 0.000 selected).
4. **Select C on the frontier, not on AUC** (§3b-xxvi). The AUC-selected head is worse on 3 of 4
   arms.
5. **Regenerate every downstream number** after 4.

## 4. Not running, deliberately

| | why |
|---|---|
| cost head on RouterBench | 78% of it is multiple-choice where prompt length gives cost R2 0.97; measured +0.45% weighted (§3b-xxv) |
| more RouterBench work | +3.91% AIQ lands in the same regime as their own KNN/MLP routers; not a differentiator |
| SWE-V pool extension | base rate wins at every N up to all 219, and it is a calibration failure, not a fixable one |
| glm-5 recollection | 25.7% empty / 38.6% truncated; would firm up C3, but C3 is now our weakest claim |

## 5. Known-stale

- Every §6.x frontier number predates the `--select-C` regeneration and the mixture/cost-constant
  corrections. Do not quote §6.x without checking against §3b-xxi..xxvii.
- SWE-V numbers at `--max-len 8192` are superseded by the 16k re-extraction (§3b-xxiv).
