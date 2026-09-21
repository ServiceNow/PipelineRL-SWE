# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-19 00:35 UTC.
Companion to `PAPER_OUTLINE.md` (findings, newest in §3b-lxviii – §3b-lxxxii), `PRIOR_ART.md`,
`RESEARCH_LOG.md`. This file answers: *what is running, why, and what does it decide?*
Replay outputs live under `/mnt/llmd/results/exps/aristides/reason/gridmatch/<run>/`; launchers are
`run_*.sh` there; every replay deletes its `episode_traces.jsonl` on exit.

---

## -1. RUNNING RIGHT NOW (2026-09-21 19:35 ET)

Full recollection still ON HOLD. Two things land tonight, and both change the rung table.

| thread | state | ETA (ET) | decides |
|---|---|---|---|
| `oss20hi` / `oss120hi` @ the 110k cap | 4 jobs | ~19:45 | Whether either rung was only dominated because the 65,536 cap truncated 12.2% / 4.1% of its eval draws |
| `dsv4f` d10-d15 eval, reasoning ON | 6 jobs, train done | ~20:15-20:45 | **The decisive one.** If DeepSeek holds ~88% on eval at ~0.1c it dominates gpt-oss-120b-high (83.8% at 0.929c) and the $16 line item goes |

### The pool table is likely to change shape, not just entries
Tonight's measurements (PAPER_OUTLINE 3b-lxxxvi/lxxxvii/lxxxviii) point away from the current
"16 draws of the cheapest rung" plan:
* **Switching beats resampling** by 20-50pt conditional on a failure, so a rung can earn its place
  by decorrelating even when it is dominated on the single-draw frontier. Breadth over depth.
* **Per-problem allocation wins at every budget when compared at equal SPEND** (+8.3pt at R=1c to
  +2.1pt at 30c) -- and needs only a **9x** price spread, not the 698x a frontier rung would buy.
* **More draws sharpen q** but with diminishing returns: 2 -> 8 halves the over-estimate, 8 -> 10
  barely moves it. So depth past ~8 on any one rung is hard to justify.
* LCB ships difficulty labels (892/892 matched; 265 easy / 312 medium / 315 hard) and 30% of the
  pool is easy problems where 93.8% of draws succeed and nothing is decided. Collecting fewer draws
  on the easy third is the cheapest way to buy information per dollar -- but note the hard subset
  did NOT show more headroom, and that measurement is noise-limited at 2-3 draws.

**Do not rewrite the rung table until the DeepSeek eval lands.** Then: recompute the conditional
switching table on post-fix draws, rerun the whether-vs-which crossing, and only then set draws
per rung.

## 0. THE PLAN (2026-09-21, supersedes everything below): recollect, then compare belief models

**Why.** Every belief improvement we have is LiveCodeBench-only, and this pool cannot arbitrate:
T=0.2 makes a model's draws near-copies (so depth and resampling decisions are tuned in a regime
that will not exist at recommended temperatures), 6 draws caps how much depth can vary, 3 rungs let
"try each tier once" be near-optimal, and TACO is too quirky to referee (PAPER_OUTLINE 3b-lxxxi).
The pilot also showed the old pool is NESTED (Jaccard 0.90-0.98, zero unique problems across seven
labs), so there is no specialisation to route on -- only difficulty and depth.

### 1. Collect one proper pool (the decisive experiment)

| rung | market price ($/M in, out) | pilot pass@1 | c/draw | draws | cost over 892 |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct (PROBE ONLY, local prefill, no decode) | -- | 47.5% | ~0.006c or free | n/a | -- |
| gpt-oss-20b low | 0.03 / 0.13 | 67.9% | 0.011 | 16 | $1.57 |
| gpt-oss-20b medium | 0.03 / 0.13 | 83.6% | 0.034 | 12 | $3.64 |
| deepseek-v4-flash | 0.055 / 0.111 | 83.0% | 0.048 | 8 | $3.43 |
| gpt-oss-120b medium | 0.15 / 0.60 | 86.8% | 0.103 | 6 | $5.51 |
| gpt-oss-120b high | 0.15 / 0.60 | 87.5% | 0.652 | 3 | $17.45 |
| | | | | | **~$32** |

**No frontier rung, and this is a result, not a concession (PAPER_OUTLINE 3b-lxxxiv).** Measured on
the pilot: claude-opus-5 (94.0%, 6.03c) and claude-sonnet-5 (88.3%, 2.41c) each add **+0.0pt** to
the union over the open ladder (95.7%) and solve **zero** problems no open rung solves, while
gpt-oss-120b high rescues **8/12** of the problems both 20b rungs fail against opus's 7/12 at a
ninth of the price. The open ladder's own price spread is 59x (old pool: 7x at market prices).
Claude is a **later top-up**, triggered only if the ladder tops out (the fixed cascade sits on our
frontier at the expensive end, or the policy exhausts its cap on 120b-high below target) or if
BigCodeBench shows the frontier rescuing what the ladder cannot. Cost of that top-up, if it fires:
$21.5 sonnet + $53.8 opus at one draw each over 892 problems.

* Sampling: each card's recommended settings (Qwen 0.7/0.8 with top-k 20 / min-p 0 -- **now
  supported, `--top-k` / `--min-p`, commit c1ce23b**; gpt-oss T=1.0 with effort; DeepSeek 0.7/0.95).
* Depth is asymmetric on purpose: deep where a draw costs ~0.01-0.05c, rationed at the top. Pilot
  saturation: 20b-low 84->90% over 6, 20b-medium 85->94% over 8, 120b-medium 86->91% over 5, so the
  counts above buy episodes deeper than coverage needs -- deliberately, since the policy needs
  states to visit, not just coverage.
* Pools: **LiveCodeBench first** (temporal split, 551/341), then **BigCodeBench** (1,140 tasks,
  random split, needs its own grader: validate every reference solution and drop flaky tasks).
  CodeContests only if that grader work comes free.
* Launch: `SUBMIT=1 bash launchers/abstention/launch_pool_recollect.sh` -- 45 eai jobs, one per
  (route, draw), both splits inside each job, resumable (a rerun reuses complete rows).

### 2. Re-extract probe features on the new pool
Prompt-only, plus the reachable-state lattice (build_deep_state_prompts.py) for history-conditioned
beliefs. Extraction is R-independent; only the heads depend on the price (0a).

### 3. Belief bake-off, matched on the new data -- "how do these compare when multi-sampling is real"
count decay (today) | analytic posterior over success counts (with-replacement, 3b-lxxvi) |
latest-failure reading (3b-lxxiv) | deep-history reading with no decay (3b-lxxxii) | distributional
spike-and-slab head at FULL dimension with its mean recalibrated (3b-lxxxiii). Each fitted on the
policy's own visited states (0a), any shrinkage chosen on the POLICY objective. Judge on: entry-level
calibration, tail quality (log-loss on all-fail episodes), ranking, and the policy frontier.

### 4. Policy evaluation
Against agreement-gating, RoR v1, random allocation, budget-aware best-of-K and the Zero Router
(Claude only if the top-up fires), with the linked-cap single sweep so grids match (3b-lxxiii), realised
accounting, 5 seeds. New diagnostics this pool finally makes meaningful: per-problem depth vs the
best fixed best-of-K (the thing a fixed plan cannot do), and share of spend on unsolvable problems
(ours 32.3% vs RoR v1 35.1% today).

### 5. Paper
Rewrite on **market-price accounting** with the local-probe framing ("one small local prefill cuts
your API bill"): the old node estimates priced gpt-oss-120b at $11.13/M against a market $0.60/M.
Keep the pool-independent findings: whether-not-which, the nested-pool result, the decay diagnosis,
the three requirements on a belief head, and the evaluation practice.

### Accounting rules for the new numbers
* Pool: market API prices. Probe: state BOTH ~0.3 GPU-seconds of a small local card AND ~0.006c at a
  rented 4B endpoint. No more self-hosting estimates for models nobody self-hosts: an 80GB card is
  $25k+, so gpt-oss-120b is an API model for our readers, not a local one.
* The method needs a wide price spread. Measured, the open rungs give 59x (0.011c -> 0.652c per
  draw), so they suffice; it was the OLD three-route pool that was compressed (7x at market
  prices). Superseded: the earlier rule that the pool must straddle open-weights -> frontier.

## 0a-bis. THE BELIEF MODEL SHOULD EMIT A DISTRIBUTION (2026-09-21) -- see PAPER_OUTLINE 3b-lxxxiii

Measured: a no-decay deep-history probe is +16.4/+16.7% at 80/84% and -59.8/-18.1% at 50/60%.
Depth-bucketed calibration does not fix it; depth-0 accuracy is fine. Cause: **no tail** -- 5.2% of
decay beliefs are below 2%, only 0.2% of the learned head's are, while the give-up test at tight
budgets fires below ~0.3-2%. Fix: predict a distribution over the per-draw success rate (start:
spike-and-slab Beta, 3 outputs per route), trained by marginal likelihood of the observed draws,
shrunk toward the pool prior on calibration. Gives tail resolution, fluke-vs-never, closed-form
depth values, and removes the decay. Build on the recollected pool with visitation-trained fitting.

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
