# Paper outline: one prefill, a shared difficulty latent, and per-query cost in reasoning-model routing

Status: draft outline, 2026-09-29. All numbers come from `NEW_PATH.md` §4.A.4–4.A.21 (section given in brackets).
Target: a TMLR analysis paper, plus a 4-page workshop cut (at the end of this file).
Track B (verification) enters only as one measurement subsection.

---

## 0. Thesis

**A single cheap prefill (Qwen3-4B, frozen, linear read-outs) extracts a shared per-problem difficulty latent.
That latent does three jobs across a whole pool of reasoning models:**

1. **Route:** who can solve it.
2. **Price:** how many tokens each model will spend on it. This is worth 20–30% of cost at matched accuracy, a gain the
   routing literature leaves unclaimed by treating cost as a per-model constant.
3. **Onboard:** a new model is added from ~5–10 labelled examples by fitting 1–3 parameters against the latent.

The latent is "shared" in a measurable sense. For reasoning models, most variation in output length is between
problems rather than between models (81–96% of variance). Hard problems are long for everyone, and the
model-specific remainder is essentially unpredictable before generation.

What makes this a paper and not a trick:
- **When:** the rule for when it pays was stated and pre-registered, then confirmed on 2 new datasets.
- **Why:** difficulty drives length, and the prefill can read that difficulty.
- **Where not:** non-reasoning pools, mixed-price agentic pools, and pools where difficulty is illegible.

## 1. Claims ledger

Status key: **Solid** means CI excludes 0 and the protocol is matched. **Directional** means the point estimate
favours us but the CI touches 0. **Pending** means not yet run.

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Per-query cost headroom is large for reasoning pools (20–65%) and small for non-reasoning pools (~10%) | 7 reasoning pools, RouterBench, agentic [4.A.4, 4.A.7, 4.A.14, 4.A.19] | Solid |
| C2 | Headroom appears when within-model cost variation is large relative to between-model price gaps | Reasoning vs RouterBench; agentic open-only 25.7% vs all-13 8.5% [4.A.14] | Solid (3 regimes); the principle is an interpretation |
| C3 | One 4B prefill captures most of that headroom where difficulty is legible: LCB 35.6%, Omni 21.9%, MMLU-Pro 30.7% vs the paper rule | [4.A.6, 4.A.9, 4.A.19] | Solid; Omni and MMLU-Pro were pre-registered |
| C4 | A pre-registered screen (one cheap model × one prefill, R² ≥ .50) predicts whether a new dataset yields a gain | 2/2 confirmed; 5 calls pending [4.A.9, 4.A.12, 4.A.13, 4.A.19] | Solid but thin (n=2) |
| C5 | It beats literature cost predictors (MixLLM-style, prompt-GBM) | LCB and MMLU-Pro significant; Omni directional [4.A.6, 4.A.20] | Solid / Directional |
| C6 | Most of the value is the shared level ("hard is long for everyone"); on LCB and Omni it is recoverable from the success head alone | Level-only 10–36% [4.A.17]; from-success ties LCB/Omni [4.A.20] | Solid. This supports the single-latent framing |
| C7 | Length is not just difficulty on MMLU-Pro: a dedicated cost read adds +21.5 pt over cost-from-success | [4.A.20] | Solid on one pool |
| C8 | The model-specific part of cost resists every pre-generation predictor tried | ~12 nulls [4.A.11, 4.A.16, 4.A.17] | Solid as a negative |
| C9 | Higher cost-prediction R² does not imply better routing | LCB prefix: R² .70–.86 but gain 34→26% [4.A.15] | Solid |
| C10 | Onboarding: a new model's cost from 5 examples gives 96% of a trained head; cost + success from 5–10 examples recovers most of a removed model's value; naive onboarding is worse than leaving the model out | [4.A.18 #2, #2.1, 4.A.21] | Solid (offline, 3 pools) |
| C11 | New families that are dominated by the pool are priced out correctly from ~10 examples (onboarding protects the router) | GLM / Nemotron / MiniMax on MMLU-Pro [4.A.19 #2.3] | Solid, but protective, not a gain |

## 2. Section-by-section outline

### §1 Introduction
- **Setup:**
  - Routers choose argmax_m p_m(x)·V − c_m(x).
  - Nearly all prior work sets c_m(x) to a per-model constant: a price × median length, "q-independent" (SWE-Router), or
    "less critical" (Dekoninck et al.).
  - That was reasonable for chat models, where output is a few dozen tokens and cost is input-dominated (RouterBench:
    output is 17–42% of the bill).
- **The shift:** a reasoning model's output length varies 10–90× across problems within one model and setting, and
  81–96% of that variance is between problems.
- **Our claim (the thesis above):** one cheap prefill, one latent, three uses.
- **Contributions:**
  1. A headroom measurement across 9 one-shot pools and 2 agentic sets, with a principle for when per-query cost matters.
  2. A single-prefill router that prices each query, with two pre-registered out-of-sample confirmations and baselines.
  3. An account of why it works (legible difficulty drives length) and where it cannot: the model-specific remainder is
     unpredictable, and R² is not routing value.
  4. Onboarding a new model against the latent from ~5–10 examples.
  5. Scope results: agentic step caps backfire; cross-model verification as a measurement.

### §2 Setting and protocol
- **Pools (one-shot):**
  - LCB (892; 441/110/341), Omni-MATH-500 (500), MMLU-Pro (1000), CodeContests (700), TACO, BigCodeBench, SWE-Smith
    reasoning.
  - Routes: gpt-oss-20b low/med, deepseek-v4-flash, gpt-oss-120b med/high.
  - 2–4 draws per route.
  - OpenRouter market prices ($/M in/out): 0.018/0.09, 0.047/0.094, 0.15/0.6.
- **Contrast pools:** RouterBench (11 chat models); SWE-rebench July 2026 (13 standalone agents × 5 runs); nebius
  SWE-agent (3387 tasks × ~20 runs).
- **Prefill:**
  - Qwen3-4B-Instruct-2507 (Thinking on Omni, fixed by pre-registration).
  - Frozen; rich activation features.
  - Success head: per-route probe → p_m(x).
  - Cost head: per-route plain RidgeCV → log output tokens. Cost = input×price_in + exp(pred)×price_out.
- **Shared latent, operationally:**
  - Level(x) = mean over routes of the predicted log length.
  - Difficulty(x) = mean over routes of the success logit.
  - Both are linear reads of the same activations.
- **Reference rule:** input + median train output (arXiv 2603.20895).
  - Headroom = oracle per-problem cost vs the reference, at matched accuracy.
  - Gain = learned vs the reference.
  - Capture = gain / headroom.
- **Evaluation:**
  - Test frontiers over V, then the convex hull.
  - Cost ratio averaged over the accuracy band shared by all arms.
  - Paired bootstrap over test problems, with identical resamples per arm.
  - Deployable variant: operating point chosen on calibration and applied once to test.
- **Pre-registration rule (committed before data):**
  - Screen = gpt-oss-20b-low × 1 plus the prefill.
  - Probe 5-fold CV log-output R² ≥ .50 → GAIN (≥ 10%, CI > 0); ≤ .35 → NO GAIN; otherwise no call.
  - Headroom ≥ 15% predicted for any reasoning pool.

### §3 How much is per-query cost worth? (headroom)
Table 1: headroom [95% CI], market prices [4.A.4, 4.A.7, 4.A.9, 4.A.14, 4.A.19].

| Pool | Headroom |
|---|---|
| MMLU-Pro | 65.1% [55.2, 71.2] |
| LCB | 47.3% [39.7, 52.9] |
| Omni | 44.2% [31.8, 53.5] |
| TACO | 36.1% [24.9, 46.0] |
| LCB, gpt-oss only | 30.8% |
| CodeContests | 21.8% |
| BCB | 20.2% |
| SWE-Smith | 14.2% (n.s.) |
| RouterBench (chat) | 10.5% [7.4, 12.8] |
| Agentic SWE-rebench, 7 open models (~10× price range) | 25.7% [6.3, 40.6] |
| Agentic SWE-rebench, all 13 (~190× price range) | 8.5% (n.s.) |

- **Principle (C2):** per-query cost pays when within-model variation is large relative to price gaps. Reasoning
  models are high on both counts; chat is low on within-model variation; mixed-price agent pools are dominated by
  price gaps.
- Agentic variance shares: model 74 / task 15 / interaction 6 / run 4. Within-model ICC is .69–.92, so per-task
  cost is stable.

### §4 One prefill prices the pool (main result)
- Table 2: gain vs the reference at matched accuracy [95% CI] and deployable cost ratios.

  | Pool | Gain | Deployable | Note |
  |---|---|---|---|
  | LCB | 35.6% [27.1, 41.9] | 0.63 / 0.67 / 0.73 at 70 / 75 / 80% | |
  | Omni | 21.9% [8.9, 33.5] | 0.70 / 0.73 at 60 / 65%; same cost +2–3 pt accuracy at 70–75% | pre-registered |
  | MMLU-Pro | 30.7% [16.3, 43.2] | 0.39–0.58 at 57–72% | pre-registered |

- **Table 3: baselines**, ours minus baseline, paired [4.A.20, 4.A.6]:

  | Baseline | LCB | Omni | MMLU-Pro |
  |---|---|---|---|
  | Mean-per-model constant | +34.3 | +23.3 | +29.5 |
  | Single best model | +39.3 | +42.0 | +25.3 |
  | MixLLM-style (embeddings → MLP / RF / kNN) | +21.5 | +9.4 [−4.4, 24.2] | +26.7 [12.4, 40.2] |
  | Prompt-feature GBM | +23.3 | +15.2 [−1.4, 34.1] | +20.9 [3.8, 33.3] |
  | Cost from the success head | +1.5 [−1.3, 4.3] | +0.8 [−7.2, 9.0] | +21.5 [8.5, 30.3] |

  - Rows without brackets are point differences of separately reported gains. The baselines' own gains vs the
    reference: mean-constant 1.3 / −1.4 / 1.2; single best model −3.7 / −20.1 / +5.4; LCB MixLLM-style 14.1 [5.3, 21.0];
    LCB GBM 12.3 [0.7, 21.4].
  - **TODO:** paired CIs for those rows (decompose.py now saves the draws).
- **Pre-registration scorecard (Fig. 4):**
  - Confirmed GAIN: Omni, MMLU-Pro.
  - Pending GAIN: APPS .61, K&K .61, SuperGPQA .58.
  - Pending NO GAIN: AIME .13, BBEH .28.
  - No call: OlympiadBench .48.
  - Excluded: ZebraLogic (the public solutions are redacted; the grader passed placeholders).
  - Report the exclusion openly.
- **RouterBench (their benchmark):** probe 8.5% vs MixLLM-style 8.1% vs GBM 7.0% of 10.5%. We're at least as good,
  but there's little headroom to separate predictors.

### §5 Why it works, and when (the shared latent)
- **5.1 Hard is long for everyone.**
  - Between-problem variance share is 81–96%.
  - Within a problem-route, failed runs are not longer than solved ones (×0.83–1.25), so length is a property of
    the problem (ICC .85–.95), not of failing [4.A.10].
- **5.2 The rule's two inputs.** Non-circular link and legibility (Table 4) [4.A.15]:

  | Pool | Difficulty → length | Prefill reads difficulty | Gain |
  |---|---|---|---|
  | LCB | .56 | .46 | yes |
  | Omni | .44 | .47 | yes |
  | CodeContests | .43 | .23 | no: illegible |
  | TACO | .08 | .32 | no: not difficulty-driven |
  | BCB | .10 | .09 | no: not difficulty-driven |

  - External labels (LCB tier, Omni rating) agree.
  - A metadata-only head (tier + platform) gets 24.7% of LCB's 46% headroom.
- **5.3 The capture curve is convex (Fig. 2).**
  - Synthetic calibrated heads at controlled R²: ≈ 0 below ~.3, most headroom only above ~.5.
  - Real heads sit on the curve when plotted against dollar R²; dollar calibration adds nothing [4.A.4, 4.A.8].
  - Mechanism: a decision flips only when the cost error is smaller than the utility margin between routes.
- **5.4 Level vs differences (Fig. 3)** [4.A.17]:

  | Pool | True level alone | True differences alone | Full |
  |---|---|---|---|
  | LCB | 35.6 | 36.5 | 46 |
  | Omni | 21.0 | 41.1 | 44 |
  | CodeContests | 12.7 | 19.3 | 22 |
  | BCB | 9.8 | 15.2 | 20 |
  | TACO | −1.9 | 30.0 | 36 |

  - Our savings come mostly through the level: LCB 33.3 of 35.6, Omni 18.6 of 21.0.
  - This is the single-latent result: the latent already carries most of what is capturable.
- **5.5 Success and cost are (mostly) one latent.**
  - Cost inferred from the success head ties the dedicated cost read on LCB and Omni.
  - It loses by 21.5 pt on MMLU-Pro, where the from-success log-length R² is .14–.36 vs the probe's .20–.66 [4.A.20].
  - Framing: one representation, two linear read-outs. Usually one direction suffices; on some tasks
    (MMLU-Pro: subject-driven verbosity?) length needs its own direction.
  - Honest statement: routers that already have a prefill success head get much of this for free. Our contribution
    is showing it, measuring it, and saying when a separate read-out is needed.
- **5.6 The model-specific remainder is unpredictable** (C8). Table of nulls on between-route differences:
  - Prefixes (hand-crafted and 4B-read)
  - Thinking / Base / own-model prefills
  - Fine-tuned 137M readers (single and joint)
  - Entropy scalars
  - Self-estimated budgets (difference Spearman .12)
  - Prompted probing
  - A rating reader trained on 6.4k free labels
  - Low-rank / kNN / pooled heads
  - Selective use
  - Dollar calibration

  The prompt-side ceiling appears to be real. This is also why the latent is the right abstraction: it holds what
  is predictable.
- **5.7 R² is not routing value (C9).**
  - LCB prefix: R² .70–.86 → gain 26.1% (free) / 8.9% (charged) vs 35.6% without it.
  - The R² gains sit on easy problems that finish inside the prefix; on hard, decision-relevant problems the key
    routes get worse [4.A.15].
  - Takeaway: evaluate cost predictors by routing value, not R².

### §6 Onboarding a new model against the latent
- **Method:**
  - Cost: shared level + a per-model offset, 1 parameter.
  - Success: logistic in the shared difficulty, 2 parameters.
  - Both are fitted on k labelled problems; the prefill and all other heads are untouched.
- **6.1 Hold out each route, then re-add it** [4.A.18].
  - Cost only, k=5: LCB 34.5 vs 36.0 full (96%); Omni 26.5 vs 28.7.
  - Cost + success, k=5: LCB 29.5, Omni 24.0; naive (median + base rate) −4.7 / ~15.
- **6.2 Remove a genuinely useful model, then add it back (Table 5)** [4.A.21]:

  | Pool | Full | Without dsv4f | Onboard k=5 / 10 / 50 | Naive k=5 / 10 / 50 |
  |---|---|---|---|---|
  | LCB | 36.0 | 4.1 | 23.2 / 25.0 / 28.8 | −16.3 / −19.3 / −22.0 |
  | Omni | 28.7 | −25.0 | 18.9 / 19.1 / 23.5 | — / −14.9 / −11.6 |
  | MMLU-Pro | 30.9 | −22.1 | 16.6 / 29.0 / 29.9 | −16.0 / −16.1 / −13.8 |

  - Naive onboarding is worse than not adding the model.
  - dsv4f is the hardest case (largest model-specific share).
- **6.3 Predicting onboarding loss:** the loss tracks each model's model-specific share of cost variance
  (.14–.22 → 4–14 pt; ≤ .07 → 0–4 pt) [4.A.18 #2.2]. How far a model deviates from the latent says how cheaply it can
  be added.
- **6.4 A genuinely new family** (GLM-4.7-flash, Nemotron-3-super, MiniMax-M2.5 on MMLU-Pro) [4.A.19 #2.3].
  - All three are dominated by the pool.
  - Onboarded at k=10: 30.0% vs 30.9% base; naive: −12.3%.
  - The latent protects the router from newcomers that look good on average.
- **Caveats to state:**
  - For the cheapest route (oss20lo), naive success onboarding beats the 2-parameter fit.
  - Removing oss20md improves the router on Omni and MMLU-Pro (its heads mislead).
  - No test yet in which a genuinely new model lowers cost.

### §7 Scope: agents, caps, verification
- **Agentic predictability** [4.A.14, 4.A.15]:
  - Issue-text prefill → per-task mean log cost R² .29 (nebius, 3387 tasks), still rising with more tasks. It is ~0 on
    SWE-rebench's 111 tasks (data-starved).
  - A partial trajectory read by the 4B: own-run .28, cross-model .16.
  - All are below the ~.5 threshold, so no routing claim.
- **Caps closed** [4.A.18]:
  - One-shot per-query token caps are neutral.
  - Agentic per-task step caps are −22.3% [−25.5, −19.7] vs one global cap. In agents, long means failing, so a
    better length estimate grants doomed runs more budget.
- **Verification measurement** [4.B]:
  - A cheap cross-model test writer (dsv4f) is 33–42% cheaper than self-verification at matched accuracy.
  - Tests pay only above route-once accuracy (~52%).
  - Per-instance writer choice has no demonstrated value (pre-registered NO GO).
  - One paragraph or an appendix.

### §8 Related work
Group by what each assumes about cost.

| Group | Papers |
|---|---|
| Constant or median cost | 2603.20895; IRT-Router 2506.01048; C3PO; 2602.09924; SWE-Router ("q-independent") |
| Per-query cost predictors on chat benchmarks | MixLLM 2502.18482; 2509.09782; Dekoninck 2410.10347 ("less critical") |
| Budget control | R2-Router 2602.02823 (prompt-instructed budgets, own bench); TALE-style self-estimates |
| Length / difficulty prediction | EGTP / PLP 2602.11812; TRAIL; OUTLETS 2609.01068 |
| Abort and reroute | SWE-Router 2607.00053; Fail-Fast 2608.03222; TACIT-Switch; EarlyEval 2609.02783 |
| Cold start / shared latents | IRT-Router; universal latent space 2601.06220; SCOPE 2601.22323; 2607.18253 |

**To do:** read 2601.06220 and IRT-Router's cold-start closely. They are the closest to §6.

### §9 Limitations (write plainly)
- Two confirmed pre-registrations; 5 decisive calls pending.
- The dedicated-cost-read advantage over from-success rests on one pool (MMLU-Pro).
- Small test sets (Omni 150, LCB 341). Deployable CIs are wide.
- Pools lean on gpt-oss and dsv4f. The new-family test added only dominated models.
- Everything is offline replay on stored draws; there are no live deployment numbers.
- Market prices change. The principle (C2) is price-relative, which helps.
- Agentic predictability is below threshold. The agentic headroom exists, but it is not captured.

## 3. Figures and tables
1. **Fig. 1:** headroom bars, grouped reasoning / chat / agentic-open / agentic-mixed, with CIs (C1, C2).
2. **Fig. 2:** capture curves (synthetic, gain vs dollar R²) with real heads overlaid; LCB, Omni, MMLU-Pro above the
   knee, CC / TACO / BCB below.
3. **Fig. 3:** level vs differences stacked bars per pool (true components vs what our predictor captures).
4. **Fig. 4:** pre-registration scorecard: screen R² on x, realised gain on y, thresholds .35 / .50 drawn, calls
   marked before/after.
5. **Fig. 5:** onboarding curves vs k (onboard vs naive vs full vs without), 3 pools.
6. **Fig. 6:** the R² ≠ routing panel: R² rises, gain falls (LCB prefix), split by easy vs hard problems.
7. Tables 1–5 as above; an appendix table of nulls; an appendix table of all pool statistics (n, draws, accuracy
   ladder, output p90/p10, prices).

## 4. Before submission (ranked; spend needs sign-off)
1. **Run the pending pre-registered pools.** A mix of GAIN and NO GAIN calls matters more than more GAINs.
   - SuperGPQA or APPS (GAIN) plus AIME (NO GAIN): ~$15–20.
   - BBEH: ~$30+, optional.
2. **Onboarding baselines:** k-shot own heads (train the new model's probes on k), cost-only and success-only
   ablations, and IRT-Router-style cold start. Offline; free.
3. **Re-run the LCB baselines under the paired bootstrap with saved draws** (Table 3 consistency). Free.
4. **A positive new-model test:** a newcomer that is cheaper or better than part of the pool, so onboarding can show
   a gain, not just protection. This needs choosing a model; ~$2–5.
5. **Decide the MMLU-Pro story** (why length ≠ difficulty there: subject-driven verbosity?). Offline analysis.
6. Figures, and a clean re-run of every table from one script per table.

## 5. Workshop cut (4 pages)
- **Title:** "The Price Isn't Constant: One Prefill Prices a Pool of Reasoning Models."
- **Content:**
  - §1 intro (½ page)
  - Headroom (Fig. 1)
  - Main result + baselines + pre-registration (Table 2 / 3, Fig. 4)
  - Why: legible difficulty + level vs differences (Table 4, Fig. 3)
  - Onboarding in one paragraph (Table 5 condensed)
  - Limitations
- **Drop:** agentic, caps, verification, the nulls table (one sentence each).

## Title options
- One Prefill Prices the Pool: Difficulty-Driven Per-Query Cost in Reasoning-Model Routing
- The Price Isn't Constant: Per-Query Cost for Routing Reasoning Models
- Hard Is Long for Everyone: A Shared Difficulty Latent for Routing, Pricing and Onboarding LLMs
