# Per-query cost for routing reasoning models: when it pays, why, and how cheaply

Paper outline (bullets, not prose). Status 2026-09-30. Target: TMLR (analysis paper) + a 4-page workshop cut.
Numbers come from `NEW_PATH.md` §4.A.4–4.A.36 (section in brackets). Figures: `analysis/figures/*.png`
(regenerate with `python analysis/figures/make_figures.py`).

---

## Abstract (bullets)
- Routers pick a model per query by trading predicted success against cost; almost all treat cost as a per-model constant.
- For reasoning models, output length (hence cost) varies 10–90× across queries within one model.
- Measurement: perfect per-query cost knowledge is worth 20–65% of cost at matched accuracy on reasoning pools, ~10% on chat.
- Method: one frozen 4B prefill, linear read-outs for success and length per model; saves 22–36% vs the median-length rule
  on LCB, Omni-MATH, MMLU-Pro (the last two pre-registered).
- Why: hard problems are long for every model; the gain appears when that difficulty is legible from the prompt; on some
  tasks length tracks work required rather than difficulty.
- Against shared-latent routing (ZeroRouter), reproduced on a realistic 5-model pool with its configuration tuned on
  calibration: +17.6 pt on LCB (CI excludes 0), +9.3 on MMLU-Pro (n.s.), tie on Omni; its success predictions are fine, its
  difficulty-derived pricing is the weak part.
- New models onboard from 5–10 labelled examples; per-request budgets and single-submission cascades as deployment cases.

## 1. Introduction
- **Setting**
  - Routing rule: argmax_m p_m(x)·V − c_m(x).
  - Prior work: c_m(x) = price × median/mean length, "q-independent" (SWE-Router), "less critical" (Dekoninck et al.).
  - Reasonable for chat models: few dozen output tokens; cost input-dominated (RouterBench: output 17–42% of the bill).
  - Some routers do predict per-query cost (MixLLM, CARROT, GraphRouter, Route-To-Reason, ZeroRouter); none ablates it
    against the constant, and CARROT reports only marginal gains on chat benchmarks.
- **The shift**
  - Reasoning models: output length varies 10–90× across problems within one model/setting; 81–96% of that variance is
    between problems.
- **Questions**
  - Does per-query cost matter? When? What does it take to capture it? How cheaply can a new model be priced?
- **Contributions**
  1. Headroom measurement across 9 one-shot pools + 2 agentic sets; a principle for when per-query cost matters (Fig. 1).
  2. A single-prefill router that prices each query; wins on code / math / knowledge; pre-registered out-of-sample calls;
     baselines incl. a faithful ZeroRouter reproduction (Tables 2–4, Fig. 4, Fig. 6).
  3. Mechanism: legible difficulty drives length; level vs differences; convex capture; R² ≠ routing value; work required
     ≠ difficulty (Figs. 2, 3, 9).
  4. Onboarding a new model from 5–10 examples against the shared structure (Fig. 5).
  5. Deployment regimes: per-request budgets, abstention, single-submission cascades (Figs. 7, 8).
  6. Scope: agents, caps, verification.

## 2. Related work
- **Constant or median cost:** 2603.20895 (prefill activations, median length — our reference rule); IRT-Router 2506.01048;
  C3PO; 2602.09924 (prefill success probes, average cost); SWE-Router ("q-independent").
- **Per-query cost predictors:** MixLLM 2502.18482; CARROT 2502.03261 (kNN / RoBERTa; marginal over constant on RouterBench /
  SPROUT); GraphRouter 2410.03834; Route-To-Reason 2505.19435 (text-embedding MLP predicts reasoning-model length);
  2509.09782; Dekoninck 2410.10347 ("less critical").
- **Shared latents / cold start:** ZeroRouter 2601.06220 (IRT latent from ~200 models, cost = per-bin mean length,
  ~200-anchor onboarding) — closest prior art, credited for the shared-latent framing; SCOPE 2601.22323; 2607.18253.
- **Budget control:** R2-Router 2602.02823 (prompt-instructed length budgets); TALE-style self-estimates.
- **Length / difficulty prediction:** EGTP / PLP 2602.11812; TRAIL; OUTLETS 2609.01068.
- **Abort and reroute (agents):** SWE-Router 2607.00053; Fail-Fast 2608.03222; TACIT-Switch; EarlyEval 2609.02783.
- **Cascades with a learned answer check:** FrugalGPT (per-tier DistilBERT scorer); AutoMix; 2504.19391; cascade routing
  (Dekoninck); Cheap Verifiers, Large Blind Spots 2609.01345.
- **Correctness probes:** Openia 2501.12934; AutoProbe 2510.02934; 2512.07404; 2606.14530; HSRM 2608.30841; 2512.22245.
- **Positioning bullets**
  - Per-query cost prediction exists; what is missing: a measurement of its value vs a constant, when it pays, cost read
    from an LLM prefill, whether difficulty-only pricing suffices, low-shot onboarding, the router-vs-cascade result.

## 3. Setting and protocol
- **3.1 Pools** (Table 1; pool statistics table to build for Appendix A)
  - LCB (892; 441/110/341), Omni-MATH-500 (500), MMLU-Pro (1000), CodeContests (700), TACO, BigCodeBench, SWE-Smith;
    APPS (1000, pending).
  - Routes: gpt-oss-20b low / medium, deepseek-v4-flash, gpt-oss-120b medium / high; 2–4 draws per route (APPS: 1).
  - OpenRouter market prices ($/M in/out): 0.018/0.09, 0.047/0.094, 0.15/0.6.
  - Contrast sets: RouterBench (11 chat models); SWE-rebench July 2026 (13 agents × 5 runs); nebius SWE-agent (3387 tasks).
- **3.2 Router**
  - Frozen Qwen3-4B-Instruct-2507 prefill (Thinking on Omni, fixed by pre-registration); rich activations.
  - Success head per route → p_m(x); cost head per route (plain RidgeCV → log output tokens) → c_m(x).
  - Shared structure: level(x) = mean predicted log length; difficulty(x) = mean success logit.
- **3.3 Reference and metrics**
  - Reference: input + median train output (2603.20895).
  - Headroom = oracle per-problem cost vs reference at matched accuracy; gain = ours vs reference; capture = gain / headroom.
  - Test frontiers over V, convex hull, cost ratio over the shared accuracy band; paired bootstrap over test problems.
  - Deployable variant: operating point chosen on calibration, applied once to test.
  - No abstention in one-shot results unless stated (§9.2).
- **3.4 Pre-registration rule** (committed before data)
  - Screen: gpt-oss-20b-low × 1 + the prefill; probe 5-fold CV log-length R² ≥ .50 → GAIN (≥ 10%, CI > 0); ≤ .35 → NO GAIN.
  - Headroom ≥ 15% predicted for any reasoning pool.

## 4. How much is per-query cost worth? (headroom)
- **Figure 1:** `analysis/figures/fig1_headroom.png`
- **Table 1 (headroom, market prices)** [4.A.4, 4.A.7, 4.A.9, 4.A.14, 4.A.19]

  | Pool | Headroom |
  |---|---|
  | MMLU-Pro | 65.1% [55.2, 71.2] |
  | LCB | 47.3% [39.7, 52.9] |
  | Omni | 44.2% [31.8, 53.5] |
  | TACO | 36.1% [24.9, 46.0] |
  | CodeContests | 21.8% |
  | BCB | 20.2% |
  | SWE-Smith | 14.2% (n.s.) |
  | RouterBench (chat) | 10.5% [7.4, 12.8] |
  | Agentic, 7 open models (~10× price range) | 25.7% [6.3, 40.6] |
  | Agentic, all 13 (~190× price range) | 8.5% (n.s.) |

- **Principle:** per-query cost pays when within-model variation is large relative to price gaps.
  - Reasoning: high within-model variation; chat: low; mixed-price agents: dominated by price gaps.
  - Agentic variance shares: model 74 / task 15 / interaction 6 / run 4; within-model ICC .69–.92.

## 5. One prefill prices the pool (main results)
- **Table 2 (gain vs reference at matched accuracy; deployable cost ratios)** [4.A.8, 4.A.9, 4.A.12, 4.A.19]

  | Pool | Gain | Deployable | Note |
  |---|---|---|---|
  | LCB | 35.6% [27.1, 41.9] | 0.63 / 0.67 / 0.73 at 70 / 75 / 80% | |
  | Omni | 21.9% [8.9, 33.5] | 0.70 / 0.73 at 60 / 65%; same cost +2–3 pt at 70–75% | pre-registered |
  | MMLU-Pro | 30.7% [16.3, 43.2] | 0.39–0.58 at 57–72% | pre-registered |
  | APPS | pending | | pre-registered GAIN |

- **Table 3 (baselines; ours minus baseline, paired where bracketed)** [4.A.6, 4.A.20, 4.A.26]

  | Baseline | LCB | Omni | MMLU-Pro |
  |---|---|---|---|
  | Mean-per-model constant | +34.3 | +23.3 | +29.5 |
  | Single best model | +39.3 | +42.0 | +25.3 |
  | MixLLM-style | +21.5 | +9.4 [−4.4, 24.2] | +26.7 [12.4, 40.2] |
  | Prompt-feature GBM | +23.3 | +15.2 [−1.4, 34.1] | +20.9 [3.8, 33.3] |
  | Cost from the success head | +1.5 [−1.3, 4.3] | +0.8 [−7.2, 9.0] | +21.5 [8.5, 30.3] |
  | ZeroRouter-style bins on our difficulty (K=5 / 10) | +3.7 [0.1, 7.4] / +1.8 | −0.1 / +3.5 | +19.6 [7.8, 32.7] / +19.1 |

  - Unbracketed rows are differences of separately reported gains → re-run paired (see the plan at the end).
  - RouterBench (their benchmark): probe 8.5% vs MixLLM-style 8.1% vs GBM 7.0% of 10.5% — low headroom, predictors converge.
- **Pre-registration (Figure 4):** `analysis/figures/fig4_preregistration.png`
  - Confirmed GAIN: Omni (screen .67 → 21.9%), MMLU-Pro (.65 → 30.7%).
  - Pending GAIN: APPS .61 (running), K&K .61, SuperGPQA .58. Pending NO GAIN: AIME .13, BBEH .28. No call: OlympiadBench .48.
  - Excluded: ZebraLogic (public solutions redacted; grader passed placeholders) — report openly.

## 6. Why it works, and when
- **6.1 Hard is long for everyone** [4.A.10]
  - Between-problem variance share 81–96%; within a problem-route, failed runs are not longer than solved ones (×0.83–1.25).
  - Length is a property of the problem (ICC .85–.95), not of failing.
- **6.2 The rule's two inputs** (Table 5) [4.A.15]

  | Pool | Difficulty → length | Prefill reads difficulty | Gain |
  |---|---|---|---|
  | LCB | .56 | .46 | yes |
  | Omni | .44 | .47 | yes |
  | CodeContests | .43 | .23 | no: illegible |
  | TACO | .08 | .32 | no: not difficulty-driven |
  | BCB | .10 | .09 | no: not difficulty-driven |

  - Metadata-only head (tier + platform) gets 24.7% of LCB's 46% headroom.
- **6.3 Convex capture (Figure 2):** `analysis/figures/fig2_capture_curve.png` [4.A.4, 4.A.8]
  - Synthetic calibrated heads: ≈ 0 below R² ~.3, most headroom only above ~.5.
  - Mechanism: a decision flips only when the cost error is smaller than the utility margin.
  - Caption caveat: real-head x-values are the head's mean test R²; the synthetic axis is log-length R² (check axes match).
- **6.4 Level vs differences (Figure 3):** `analysis/figures/fig3_level_vs_differences.png` [4.A.17]
  - True level alone: LCB 35.6, Omni 21.0, CC 12.7, BCB 9.8, TACO −1.9; differences alone: 36.5 / 41.1 / 19.3 / 15.2 / 30.0.
  - Our savings come mostly through the level (LCB 33.3 of 35.6; Omni 18.6 of 21.0).
- **6.5 Success and cost: one latent or two?** [4.A.20, 4.A.26, 4.A.28]
  - Cost from the success head ties the dedicated cost read on LCB / Omni; loses 19–22 pt on MMLU-Pro.
  - MMLU-Pro (Figure 9): `analysis/figures/fig9_mmlupro_length_drivers.png`
    - Even TRUE difficulty explains little of length (R² .19 vs .46–.51 on Omni / LCB).
    - Subject closes half the routing gap (9.2% → 22.3%, probe 30.7%); source / option count add nothing.
    - Remainder is item-level WORK REQUIRED (multi-quantity calculations and multi-part "explain" questions long; recall
      and one-formula plug-ins short); probe's extra signal correlates .64 with the true residual.
  - Coding, whole datasets (no subsetting): true difficulty explains 8–10% of length on TACO / BCB, the probe 38–53%;
    CodeContests is difficulty-driven (.41 vs .44). Routing gains there n.s. (flat ladders). APPS pending.
  - Line for the paper: price by work required, not by difficulty; they coincide on LCB / Omni, come apart on MMLU-Pro.
- **6.6 The model-specific remainder is unpredictable** [4.A.11, 4.A.16, 4.A.17]
  - Nulls on between-route differences: prefixes (hand-crafted, 4B-read); Thinking / Base / own-model prefills;
    fine-tuned 137M readers (single, joint); entropy scalars; self-estimated budgets (Spearman .12); prompted probing;
    rating reader (6.4k labels); low-rank / kNN / pooled heads; selective use; dollar calibration.
  - Justifies one offset per model in onboarding.
- **6.7 R² is not routing value** [4.A.15]
  - LCB prefix: R² .70–.86 → gain 26.1% (free) / 8.9% (charged) vs 35.6% without; R² gains sit on easy problems.

## 7. Comparison with shared-latent routing (ZeroRouter)
- **Setup** [4.A.30, 4.A.32–4.A.35]
  - Their stage 1: D-dim 2PL IRT (MAP instead of SVI) on the pool's own train outcomes; D = 1, 2, 5, 20.
  - Their stage 2: our 4B reader and, separately, their own fine-tuned DistilBERT + 11 linguistic features.
  - Their pricing: s = αᵀb → K bins → per-model mean length (K = 5 / 10 / 20; unstated in the paper).
  - Population variant: + N Open LLM Leaderboard models (their data source), N = 0 … 196.
- **Headline figure (Figure 0):** `analysis/figures/fig0_headline_vs_zerorouter.png` [4.A.36]
  - (a) cost saved averaged over the accuracy range, ZeroRouter tuned on calibration, paired CIs.
  - (b) ZeroRouter cost / our cost at each accuracy with a 95% paired band: LCB 1.06–1.41 (54–87%), MMLU-Pro 0.98–1.23
    (57–81%), Omni 1.30 → 0.79 (53–73%). Our edge is in the low-to-mid accuracy range; it vanishes at the top.
- **Figure 6:** `analysis/figures/fig6_zerorouter_component_swap.png`
- **Table 4 (cost saved vs reference)**

  | Pool | Ours | ZeroRouter (4B reader, D=1 / 5, best K) | ZeroRouter (own DistilBERT) | Their success + our cost | Our success + their pricing |
  |---|---|---|---|---|---|
  | LCB | 36.0 | 9–10 / 19 | ≤ 13 | 27–36 | 4–17 |
  | Omni | 22.7 | 20–23 / 12–19 | ≤ 13 | 19–25 | 10–17 |
  | MMLU-Pro | 30.9 | 10–12 / 8–22 | ≤ 12 | 31–36 | 2–7 |

- **Findings**
  - DEFINITIVE (ZeroRouter's configuration chosen on calibration, paired) [4.A.36]: LCB +17.6 [+9.8, +24.9]; MMLU-Pro +9.3
    [−2.6, +21.5] (n.s.); Omni +3.0 [−10.7, +16.6] (tie). Against its default configuration the gaps were larger (LCB
    +17 to +27, MMLU-Pro +21 to +25) — report the tuned numbers.
  - The whole gap is the pricing (component swap); their success model is usable.
  - Dimension buys nothing at pool size; their own encoder is no better (latent barely readable from text: R² ≤ .26).
  - Population (5–53 leaderboard models, preview): no improvement (log-loss .59–.64 vs ours .519; pricing 1–12%); full
    196-model curve running. Those models are unlike a reasoning pool.
- **Differentiation bullets**
  1. Cost read directly, not derived from success parameters (they lose even where length tracks difficulty).
  2. Work required ≠ difficulty (MMLU-Pro; coding at the prediction level).
  3. No model population needed; a deployment pool has ~5 models; nobody fits 200 models on their own workload.
  4. Onboarding from 5–10 examples vs ~200 anchors (§8).
  5. Budgets need the per-query length distribution, not a per-bin mean (§9.1).

## 8. Onboarding new models
- **Method:** cost = shared level + 1 offset; success = logistic in the shared difficulty (2 params); fitted on k examples.
- **Figure 5:** `analysis/figures/fig5_onboarding.png` (ours vs ZeroRouter-style vs naive, k = 5 … 200)
- **8.1 Hold out each route, re-add it** [4.A.18]: cost only, k=5: LCB 34.5 vs 36.0 full (96%); Omni 26.5 vs 28.7.
- **8.2 Remove a useful model, re-add it** [4.A.21, corrected 4.A.24]; gain (max reachable accuracy)

  | Pool | Full | Without dsv4f | Onboard k=10 | Naive k=10 |
  |---|---|---|---|---|
  | LCB | 36.0 (89.3) | 4.1 (88.2) | 25.0 (88.2) | −19.3 (88.6) |
  | Omni | 28.7 (74.3) | −25.0 (69.4) | 19.1 (73.6) | −14.9 (73.9) |
  | MMLU-Pro | 30.9 (82.1) | −22.1 (73.4) | 29.0 (80.7) | −16.1 (80.9) |

  - Read "without" with its max accuracy (its gain covers only reachable accuracies).
- **8.3 Pool without deepseek; remove and re-add gpt-oss-120b** [4.A.24]

  | Pool | Full | Without gpt-oss-120b | Onboard k=10 / 50 | Naive k=10 |
  |---|---|---|---|---|
  | LCB | 21.2 (88.2) | max 67.0 | 8.2 (86.7) / 12.2 (88.6) | −45.1 (86.0) |
  | Omni | 18.9 (69.4) | max 62.7 | 10.5 (66.0) / 11.6 (66.1) | −7.8 (64.5) |
  | MMLU-Pro | 13.3 (73.4) | max 62.4 | 15.7 (72.0) / 17.4 (72.6) | −20.2 (71.5) |

- **8.4 Head-to-head with ZeroRouter's onboarding** [4.A.31, 4.A.33]
  - Ours − ZeroRouter: LCB +10 [5, 19] (k=10) / +6 [4, 8] (k=50); MMLU-Pro +8–10 / +7; Omni ≈ 0.
  - By model: ours much better on deepseek-v4-flash (+9 to +31); worse on the cheapest route (−3 to −19).
- **8.5 Predicting onboarding loss** [4.A.18 #2.2]: tracks each model's model-specific cost-variance share.
- **8.6 A genuinely new family** [4.A.19 #2.3]: GLM-4.7-flash, Nemotron-3-super, MiniMax-M2.5 on MMLU-Pro — all dominated;
  onboarded k=10: 30.0% vs 30.9% base; naive −12.3% (protection, not gain).
- **Caveats:** cheapest route better with naive success onboarding; removing oss20md improves Omni / MMLU-Pro; no test yet
  where a genuinely new model lowers cost.

## 9. Deployment regimes
- **9.1 Per-request budgets ("at most $X per query")** [4.A.29]
  - **Figure 7:** `analysis/figures/fig7_per_query_budget.png`
  - Hard cap (max_tokens enforces X; overrun fails): +5.7 pt (LCB, 0.20¢), +7.5 (Omni, 0.054¢), +5.5 (MMLU-Pro, 0.017¢) over
    the constant rule, CIs > 0.
  - Constant rule needs up to 2.4–2.9× the budget in the middle band; 1.1–1.3× averaged over budgets.
  - Soft cap (no truncation) at matched 10% violation rate: +2–5 pt, smaller overshoots (directional).
  - Method note: log-normal length model loses to constant on MMLU-Pro; use empirical residuals (tails matter).
- **9.2 Abstention** [4.A.23]
  - When wrong answers cost nothing, skipping adds nothing (0.1 / −7.0 n.s. / 0.0%): the cheapest route is nearly free.
  - Penalised errors (λ > 0) out of scope.
- **9.3 Single-submission cascades (no verifier)** [4.A.25, 4.A.27]
  - **Figure 8:** `analysis/figures/fig8_cascades.png`
  - Cost vs our router at matched accuracy (LCB): FrugalGPT-style 137M judge −25.1% [−45, −5]; 4B judge −48.9%; PERFECT
    judge −18.6% [−39, −1]; router + judge +0.7 / +2.1%.
  - A cascade pays for the cheap attempt on every problem; the prefill skips it (the "shortcut").
  - Frozen 4B judge beats the fine-tuned 137M on AUC on every tier (.91 vs .87 cheap code; .84–.86 vs .76–.80 strong code).
- **9.4 With a free verifier** [4.A.2, 4.A.22]
  - Fixed cascade ≈ planning; the prefill buys +3–4 pt at 80–85% on LCB.
  - A perfect shared latent would beat the cascade by 40–45% (stopping rule, not reordering); ours reads it at R² .42 vs
    ~.75 break-even.

## 10. Scope: agents, caps, verification
- **Agents** [4.A.14, 4.A.15]: issue-text prefill → per-task log cost R² .29 (3387 tasks, still rising); partial trajectory
  read by the 4B: own .28, cross-model .16; below the ~.5 threshold → no routing claim.
- **Caps** [4.A.18]: self-chosen one-shot token caps neutral; agentic per-task step caps −22.3% vs a global cap.
- **Verification** [4.B]: a cheap cross-model test writer is 33–42% cheaper than self-verification; per-instance writer
  choice has no value (pre-registered NO GO). Appendix.

## 11. Limitations
- Every confirmed pre-registered call is a GAIN call (AIME / BBEH not yet run).
- All pools use the same five models (gpt-oss 20b / 120b × two efforts + deepseek-v4-flash).
- APPS: one draw per model (deviation from the pre-registered design); pending.
- Work ≠ difficulty reaches routing on one pool (MMLU-Pro); coding evidence at the prediction level only.
- ZeroRouter reproduction fills unstated details (K, 11 features, stage-2 loss, MAP vs SVI); no released code; leaderboard
  population unlike a reasoning pool.
- Small test sets (Omni 150, LCB 341); offline replay only; market prices change (principle is price-relative).
- Onboarding pools share a family; the other families tried were dominated.
- Cascade comparison LCB only.
- Agentic predictability below threshold.

## 12. Conclusion (bullets)
- Per-query cost matters for reasoning models (20–65% headroom) and is predictable where difficulty — or work — is legible.
- One frozen prefill captures it; difficulty-derived pricing does not.
- A 5–10 example onboarding recipe and a single-prefill "shortcut" that beats cascades.

## Appendices (planned)
- **A.** Pool statistics (n, draws, accuracy ladder, output p90/p10, prices).
- **B.** Table of nulls (§6.6).
- **C.** ZeroRouter reproduction details: choices, sensitivity (K, D, seeds, encoder, population).
- **D.** Protocol details: hulls, bands, paired bootstrap, deployable selection.
- **E.** Agentic and verification details.

---

## Figures (current files)

| # | File | Content |
|---|---|---|
| 0 | `analysis/figures/fig0_headline_vs_zerorouter.png` | Headline: ours vs tuned ZeroRouter — averaged (a) and by accuracy (b) |
| 1 | `analysis/figures/fig1_headroom.png` | Headroom by pool type |
| 2 | `analysis/figures/fig2_capture_curve.png` | Capture vs predictor R², synthetic + real heads |
| 3 | `analysis/figures/fig3_level_vs_differences.png` | Level-only vs differences-only headroom |
| 4 | `analysis/figures/fig4_preregistration.png` | Pre-registration scorecard |
| 5 | `analysis/figures/fig5_onboarding.png` | Onboarding vs k: ours / ZeroRouter-style / naive |
| 6 | `analysis/figures/fig6_zerorouter_component_swap.png` | ZeroRouter component swap |
| 7 | `analysis/figures/fig7_per_query_budget.png` | Accuracy vs per-request budget |
| 8 | `analysis/figures/fig8_cascades.png` | Router vs single-submission cascades |
| 9 | `analysis/figures/fig9_mmlupro_length_drivers.png` | What explains MMLU-Pro length |

- To add: population-size curve (N = 0 … 196) once the job finishes; APPS rows; pool statistics table.

## Claims ledger (planning)

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Headroom large for reasoning (20–65%), small for chat (~10%) | [4.A.4, 4.A.7, 4.A.14, 4.A.19] | Solid |
| C2 | Headroom when within-model variation is large relative to price gaps | [4.A.14] | Solid (3 regimes); interpretation |
| C3 | One 4B prefill captures most headroom where difficulty is legible (LCB 35.6, Omni 21.9, MMLU-Pro 30.7) | [4.A.6, 4.A.9, 4.A.19] | Solid; 2 pre-registered |
| C4 | Pre-registered screen predicts gain | 2/2 confirmed; 6 pending | Solid but thin, one-sided |
| C5 | Beats literature cost predictors | [4.A.6, 4.A.20] | Solid (LCB, MMLU-Pro) / directional (Omni) |
| C6 | Most value is the shared level; recoverable from the success head on LCB / Omni | [4.A.17, 4.A.20] | Solid |
| C7 | Dedicated cost read adds +21.5 pt over cost-from-success on MMLU-Pro | [4.A.20] | Solid, one pool |
| C8 | Model-specific remainder unpredictable | [4.A.11, 4.A.16, 4.A.17] | Solid negative |
| C9 | R² ≠ routing value | [4.A.15] | Solid |
| C10 | Onboarding from 5–10 examples restores a removed model's accuracy while saving vs the reference | [4.A.18, 4.A.21, 4.A.24] | Solid (3 pools × 2 variants) |
| C11 | Dominated newcomers priced out from ~10 examples | [4.A.19 #2.3] | Solid, protective |
| C12 | Router beats single-submission cascades incl. a perfect judge | [4.A.25, 4.A.27] | Solid on LCB |
| C13 | Difficulty-only pricing ties where length is difficulty, loses ~19 pt where not | [4.A.20, 4.A.26, 4.A.28] | Solid on 3 pools |
| C14 | Per-request budgets: +5–7.5 pt at fixed budget in the middle band | [4.A.29] | Solid (hard cap) |
| C15 | Faithful ZeroRouter on a 5-model pool, its config tuned on calibration: we win +17.6 on LCB (CI > 0), +9.3 on MMLU-Pro (n.s.), tie on Omni; the gap is its pricing | [4.A.32–4.A.36] | Solid on LCB; directional on MMLU-Pro; population curve running |
| C16 | Coding length readable beyond difficulty (TACO / BCB) | whole-dataset diagnostic | Solid at prediction level |

## TMLR readiness and remaining plan (assessment, 2026-09-30)
- **Criteria:** claims supported by accurate, convincing evidence; interest to part of the audience (not novelty).
- **Interest:** yes — practical question nobody has answered; falsifiable counterpoint to ZeroRouter.
- **Evidence already strong:** headroom + principle; 3 wins across domains (2 pre-registered); broad baselines incl. faithful
  reproduction; mechanism with supporting nulls; one protocol.
- **Dataset coverage**
  - Code: LCB (win), APPS (pending, pre-registered GAIN); math: Omni (win, pre-registered); knowledge: MMLU-Pro (win,
    pre-registered; work ≠ difficulty); code negatives explained by the rule: CodeContests, TACO, BCB; chat contrast:
    RouterBench; agentic contrast; predicted NO GAIN still to run: AIME, BBEH.
- **Reviewer risks and fixes**

  | Risk | Fix | Must / nice | Cost |
  |---|---|---|---|
  | Pre-registration confirmed only on the GAIN side | Run AIME (predicted NO GAIN) | Must | ~$15–20 |
  | APPS pending | Finish; report either way | Must | running (~$11) |
  | One model family | One pool with a second family (Qwen3 / GLM open reasoning models) | Strongly recommended | ~$15–30 |
  | Unpaired baseline rows; tables from older scripts | Re-run every table from one script per table, all paired | Must | free |
  | Deployable numbers only for 3 pools | Extend to all headline pools | Must | free |
  | ZeroRouter details unstated | Document choices; sensitivity done | Must (writing) | free |
  | Offline replay only | Limitation; optional small live run | Nice | small |
  | Omni test set small | State; CIs shown | Nice | — |
  | Missing RouteLLM / CARROT / GraphRouter | CARROT ≈ our MixLLM-style / kNN (say so) or add RouteLLM | Nice | free |
  | Cascade comparison LCB only | 137M scorers on Omni / MMLU-Pro | Nice | GPU job |
  | ZeroRouter win significant on LCB only (MMLU-Pro n.s., test n=300) | Larger MMLU-Pro test set (more problems, one draw per model) | Nice | ~$5–10 |

- **Verdict:** with APPS, AIME, a second family and a clean re-run of all tables → solid TMLR submission; without the second
  family and two-sided pre-registration → likely a revision request. ~1–2 days compute + scripting, ~$40–60, plus writing.
- **Older open items:** onboarding baselines (k-shot own heads; cost-only / success-only ablations); a positive new-model test
  (~$2–5); final figures.

## Workshop cut (4 pages)
- Title: "The Price Isn't Constant: One Prefill Prices a Pool of Reasoning Models."
- Content: intro (½ page); headroom (Fig. 1); main result + baselines + pre-registration (Tables 2–3, Fig. 4); why (Table 5,
  Fig. 3); ZeroRouter component swap (Fig. 6); onboarding in one paragraph; one line on cascades; limitations.
- Drop: agents, caps, verification, nulls table (one sentence each).

## Title options
- Per-Query Cost for Routing Reasoning Models: When It Pays, Why, and How Cheaply
- The Price Isn't Constant: Per-Query Cost for Routing Reasoning Models
- Price by Work, Not Difficulty: Per-Query Cost in Reasoning-Model Routing
