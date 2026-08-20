# Paper Plan: Scout-Gated LLM Routing via Reasoning Traces

> Working title: **"Reasoning as Signal: Using Scout Traces to Route Hard Instances to Stronger Models"**

---

## Core Idea

When you're deciding whether to call an expensive large model, the obvious approach is to estimate difficulty from the problem statement alone. We argue there's a much richer signal available at almost no cost: let a cheap 4B model attempt the task first, and use its reasoning trace to predict whether the expensive model will succeed.

The scout's trace encodes *what it tried*, *where it got stuck*, and *how confident it was* — information that no amount of problem description analysis can reveal. This transforms difficulty estimation from a pure prediction problem into something closer to execution feedback.

The surprising result: a predictor trained on scout traces can route instances so well that it **exceeds the "always call the stronger model" ceiling** — not by improving either model, but by correctly partitioning which instances each model handles best.

---

## Problem Setting

### The cascade routing decision

Given a task instance, we have two models available:
- **Scout** (`Qwen3-4B`, cheap): generates a patch/solution, may or may not succeed
- **Oracle** (`gpt-oss-120b`, expensive): higher capability, rarely fails on tractable instances

The router makes a **binary decision per instance**:
- **Call oracle** → pay for the strong model, submit its output
- **Abstain** → submit nothing (or escalate to an even stronger model, e.g. Opus 5)

The scout's output is **never submitted as a final answer** — it runs solely to generate routing features. The label for predictor training is always oracle success, not scout success.

### Why abstention beats always-oracle

At ~47% oracle resolve rate (oss-120b on SWE-Smith), a substantial fraction of oracle calls are wasted — the problem was either unsolvable or unsolvable within that model's capability. An abstention predictor that correctly identifies those cases saves compute without losing resolved instances.

### The Opus tier

When the router would abstain, a third option exists: escalate to Opus 5 (57.9% resolve rate on the same eval set). This turns the binary decision into a **3-tier cascade**:

```
Scout attempt → predictor scores P(oracle succeeds)
├── P ≥ threshold   → call oss-120b
└── P < threshold   → call Opus 5   (instead of submitting nothing)
```

---

## The Routing Signal: What the Scout's Trace Reveals

### Problem-statement-only baseline (input-only)

The predictor sees only the problem description. This estimates task difficulty from surface features: problem length, issue complexity, codebase mentions. AUC 0.682 in-domain.

### Post-primary: scout patch as signal

The predictor also sees what the scout produced. A coherent partial solution is evidence the task is tractable; an incoherent or empty patch is evidence it's not. AUC 0.694 (no-CoT) in-domain.

### CoT: scout reasoning trace as signal

The predictor sees the scout's full `<think>...</think>` block — the internal monologue about what the problem requires, what approach to take, where it might fail. This is the richest signal: you can see exactly where the model got confused or resorted to guessing.

AUC **0.749** in-domain (Thinking-4B scout). The advantage over input-only concentrates in the **high-abstention regime** — when you're making the hardest calls about which instances to skip entirely.

| % abstained from oracle | CoT predictor | Input-only | Gap |
|------------------------|---------------|------------|-----|
| 20% | 43.2% | 42.8% | +0.4pp |
| 30% | 41.4% | 38.6% | +2.8pp |
| 40% | 38.2% | 35.1% | +3.2pp |
| 50% | 34.7% | 30.5% | **+4.2pp** |

At low abstention (call almost everyone), both predictors behave similarly — the oracle handles most tractable instances regardless. The scout trace becomes critical when you're asking "is this specific failure mode fixable by a stronger model?"

### Test execution feedback as signal

In domains where you can run the scout's code, the failure mode (wrong answer / TLE / runtime error / syntax error) is a direct oracle for difficulty class:
- **Wrong answer**: model understood the problem, made a logic error — oracle likely fixes it
- **Runtime error**: small bug — oracle likely fixes it
- **TLE**: wrong algorithm class — oracle may or may not have a better approach
- **Empty / syntax error**: model was lost — hard cases, uncertain

This is the gold-standard signal. Our hypothesis: CoT traces approximate this when execution is unavailable.

---

## Two-Domain Experimental Setup

### Domain 1: SWE-Smith → SWE-Bench Verified (agentic, no execution)

- **Task**: given a GitHub issue and repo context, produce a patch
- **Scout**: `Qwen3-4B-Instruct-2507` or `Qwen3-4B-Thinking-2507`
- **Oracle**: `gpt-oss-120b`
- **Label**: does the oracle's patch pass Daytona sandbox eval?
- **Train/eval**: 1146 SWE-Smith training instances → transfer eval on 369 SWE-Bench Verified instances (cross-domain)
- **Execution feedback**: available via Daytona sandbox eval on scout patches — FAIL_TO_PASS test names, pass/fail counts, error types per instance. This is an additional feature source on top of the patch text, unlike LCB where execution feedback replaces it as the primary signal.

Resolve rates: scout ~25%, oss-120b ~47%, Opus 5 ~58%

### Domain 2: LiveCodeBench (code gen, with execution)

- **Task**: competitive programming problems (LeetCode / AtCoder / Codeforces)
- **Scout**: `Qwen3-4B-Instruct-2507` or `Qwen3-4B-Thinking-2507`
- **Oracle**: `gpt-oss-120b`
- **Execution**: the pinned official LiveCodeBench runner handles both stdin/stdout and function-call problems
- **Routing feedback**: only public tests are exposed to the predictor in the primary experiment
- **Final scoring and labels**: scout and oracle correctness use the full public+private suite
- **Stage 1 — cascade**: if the scout passes the full suite → done, no oracle needed
- **Stage 2 — conditional abstention**: among scout-failed instances, predict P(oracle succeeds) from deployable features

LCB is a cleaner version of the same problem: better features, harder labels, explicit failure signal. It proves the concept under best-case conditions; SWE tests whether reasoning traces can substitute when execution is unavailable.

---

## The 2×2 Ablation

Two binary factors crossed across both domains:

|  | No test feedback | Test feedback |
|--|-----------------|---------------|
| **No CoT** | problem + patch | problem + patch + test failure signal |
| **CoT** | problem + patch + thinking trace | problem + patch + thinking trace + test results |

### What each cell isolates

- **No-CoT, no-tests** (baseline): how well can patch content alone predict oracle success?
- **CoT, no-tests**: does the reasoning trace add signal beyond the raw patch?
- **No-CoT, with-tests**: does execution feedback add signal beyond the patch?
- **CoT, with-tests**: are reasoning traces and execution feedback complementary or redundant?

The cleanest ablation for the CoT signal: Thinking-4B-stripped (model thinks, but traces excluded from predictor input) vs. Thinking-4B-with-CoT — same model, same patches, only what the predictor sees differs.

### Model quality control

Two "no-CoT" conditions are not equivalent:
- `Qwen3-4B-Instruct-2507`: no thinking capability; direct patch generation
- `Qwen3-4B-Thinking-2507` (stripped): model thinks internally, traces excluded from predictor input

Instruct-4B performs better on SWE-B agentic tasks despite lacking reasoning traces. The instruct vs. thinking-stripped comparison isolates model quality independent of the predictor.

---

## Results (Preliminary)

### Predictor AUC — SWE-Smith in-domain

Verified by inspecting `train_config.json` and `run_train.sh` for each run.

| Predictor | Scout model | CoT | Test FB format | AUC (in-domain) | AUC (→ Verified) | Run ID |
|-----------|------------|-----|----------------|-----------------|------------------|--------|
| Input-only | Instruct-2507 | ✗ | — | 0.682 | 0.578 | `nocot_input_only_1786126252` |
| No-CoT | **Instruct-2507** | ✗ | — | 0.697 | **0.637** | `no_cot_route3_1785884243` |
| No-CoT + tests (names only, no labels) | Instruct-2507 | ✗ | all names, no labels | 0.697 | — | `no_cot_testfb_names_only_route3_1786226829` |
| CoT stripped | **Thinking-2507** | ✗ | — | 0.717 | *not run* | `no_cot_route3_1786218095` |
| CoT | **Thinking-2507** | ✓ | — | 0.749 | **0.624** | `cot_fixed_route3_1786133032` |
| No-CoT + tests (count only) | Instruct-2507 | ✗ | N/M fraction | 0.762 | *not run* | `no_cot_testfb_count_only_route3_1786226851` |
| No-CoT + tests (old format, 8 names) | Instruct-2507 | ✗ | 8 fail names + count | 0.779 *(avg 2 runs)* | *not run* | `no_cot_testfb_route3_1786221930/1786223095` |
| No-CoT + tests (full labeled list) | Instruct-2507 | ✗ | FAILED/PASSED per test | **0.780** | **0.683** | `no_cot_testfb_full_route3_1786226769` |
| CoT + tests | Thinking-2507 | ✓ | full | **0.781** | **0.673** | `cot_testfb_full_route3_1786475425` |

**Key result (2026-08-08)**: Test feedback beats CoT by +3pp in-domain (0.780 vs 0.749), using Instruct-2507 with no thinking traces at all.

**Corrected cross-domain finding (2026-08-20)**: test feedback transfers. No-CoT + feedback reaches 0.6829 AUC versus 0.637 without feedback (+0.0459); CoT + feedback reaches 0.6728 versus 0.624 (+0.0488). Feedback is the transferable signal, while CoT still does not improve over the matched no-CoT feedback model (0.6728 vs. 0.6829).

**Test feedback format ablation — completed (2026-08-08)**:

| Format | AUC | Δ vs. no-testfb |
|--------|-----|-----------------|
| Full labeled list (FAILED/PASSED per test + totals) | **0.780** | +0.083 |
| Count only (N/M fraction) | 0.762 | +0.065 |
| Names only (all names, no labels) | 0.697 | **+0.000** |
| No test feedback (baseline) | 0.697 | — |

Key finding: **test names alone add zero signal**. The entire gain comes from PASS/FAIL labels. Count-only (N/M) captures 78% of the gain; full per-test labels add another +0.018. The signal is "how many tests did the scout break?" not "which specific tests broke?"

**LCB validity correction (2026-08-20)**:

The previously reported LCB AUCs and solve rates are **not valid paper results**. The old wrapper had three coupled problems:

- It returned failure for every function-call problem, making platform/task format a shortcut to the label.
- It decoded private tests with the wrong serialization format, so `--use-private-tc` did not actually provide the intended full-suite grading.
- It used a random train/eval split, allowing benchmark-era and platform composition shortcuts.

This invalidates the old 0.951–0.977 AUC table, the associated capture curve, and the interpretation that LCB execution feedback saturated routing. Those numbers remain useful only as debugging evidence for the wrapper failure and must not appear as headline results.

The corrected protocol uses official LCB grading for both stdin and function-call tasks, a pinned `release_v6`, a `2024-10-01` temporal split, public tests as routing feedback, and public+private tests for final scout/oracle correctness. The corrected experiment will determine whether LCB is a genuine controlled positive setting or whether the old performance was entirely artifact-driven.

Key takeaways that remain supported:
- Scout patch text improves SWE-Smith in-domain AUC over input-only, but the gain is modest.
- CoT improves SWE-Smith in-domain prediction but does not transfer as well as the no-CoT predictor.
- SWE-Smith test-feedback models are strong in-domain and transfer better when scored with their required feedback feature: 0.6829 no-CoT and 0.6728 CoT on Verified.
- Difficulty/model success appears much harder to infer for SWE-Bench-style agentic tasks than for short-form code generation.
- No LCB routing conclusion is considered established until the corrected temporal experiment completes.

### Routing policy — 285 SWE-Smith eval instances

| Policy | Resolve rate |
|--------|-------------|
| Always oss-120b | 47.4% |
| Random upgrade X% → Opus | ≤ 57.9% (linear ceiling) |
| Always Opus 5 | 57.9% |
| CoT predictor + Opus fallback (optimal threshold) | **62.1%** |

The CoT predictor routing 57% of instances to Opus and 43% to oss-120b **exceeds the always-Opus ceiling by +4.2pp**. A random upgrade policy is provably bounded by that ceiling — the only way to break through it is to correctly partition which instances each model handles best. The predictor does this.

Graph: `router_analysis/opus_fallback_policy_cot_vs_inputonly.png`

---

## Narrative Arc for the Paper

### 1. Hook: the ceiling problem

Everyone building LLM systems knows the cost-quality tradeoff. The standard solution — cascade from cheap to expensive — still leaves you calling the expensive model on instances it was never going to fix. The conventional wisdom for deciding when to escalate: look at the problem. We argue: look at what the cheap model *did*.

### 2. The insight: traces as difficulty signals

A thinking model's attempt is not just a (possibly wrong) answer. It's a structured record of what the problem requires, what approach the model took, and where it failed. This is high-bandwidth information about instance difficulty that no amount of problem description analysis can extract.

### 3. Proof of concept: LCB

In LiveCodeBench, execution feedback is clean and verifiable. Wrong answer vs. TLE vs. runtime error may provide meaningful routing signals. The corrected temporal experiment tests this hypothesis under controlled conditions; the previous LCB evidence is invalid.

### 4. SWE-B agentic tasks — with and without execution feedback

On SWE-B we have two signal regimes. With Daytona eval on scout patches, we get structured test execution feedback (which tests failed, error types) — comparable in spirit to LCB. Without it, the only signal is the scout's patch and reasoning trace. The 2×2 ablation tests both regimes, letting us measure how much execution feedback adds over CoT traces on SWE-B, and how the signals compare to their LCB counterparts.

### 5. The ceiling-breaking result

By combining oss-120b and Opus 5 with the predictor as a router, we achieve 62.1% — above the 57.9% always-Opus ceiling. This is impossible without a predictor that genuinely discriminates difficulty. We compare against the random upgrade baseline (linear interpolation, ceiling at 57.9%) to show the gap is real.

### 6. Ablation: what each signal contributes

The 2×2 table shows CoT traces and execution feedback are both informative, and the cross-domain comparison shows which signal generalizes. The key question for the paper: are they complementary, or does one subsume the other?

---

## Next Experiment Sequence (updated 2026-08-20)

The immediate goal is to determine whether there is a defensible paper core before spending compute on more router variants. Generic “difficulty” is not the precise target: the predictor estimates **model-specific success**. SWE-Bench-style outcomes also depend on repository localization, environment behavior, patch validity, and model-specific solution paths, so weak transfer there is plausible rather than surprising.

### Round 1 — run in parallel

#### A. Correct SWE test-feedback transfer scoring — completed

Re-score the two existing SWE-Smith-trained full-feedback checkpoints on SWE-Bench Verified:

| Predictor | Checkpoint | Verified trajectories |
|-----------|------------|-----------------------|
| No-CoT + full test feedback | `...no_cot_testfb_full.../epoch_0007` | Instruct trajectories with Daytona feedback |
| CoT + full test feedback | `...cot_testfb_full.../epoch_0009` | Thinking trajectories with Daytona feedback |

The scorer must reconstruct the exact training feature schema from `train_config.json` and include the stored test feedback. It must fail rather than silently score when feedback is absent. Compare against the valid no-feedback transfer baselines: 0.637 no-CoT and 0.624 CoT.

**Result**: the prior failure was a scoring artifact. Correct feature-matched scoring gives 0.6829 AUC for no-CoT + feedback and 0.6728 for CoT + feedback. Both beat their no-feedback transfer baselines; CoT adds no benefit once feedback is present.

#### B. Collect corrected LCB data

Collect one fresh matched dataset with:

- Dataset: official LiveCodeBench `release_v6`, problems on/after `2023-09-01`
- Split: train before `2024-10-01`, eval on/after `2024-10-01`
- Scout: `Qwen/Qwen3-4B-Instruct-2507`, local vLLM, temperature 0
- Oracle: `openai/gpt-oss-120b`, temperature 0
- Evaluator: pinned official runner commit `28fef95`
- Routing feedback: public tests only
- Final scout/oracle correctness: full public+private suite
- Task formats: stdin and function-call, both graded officially
- Resumption: only rows carrying the matching evaluator commit and feedback-suite marker are reusable

This collection replaces all old LCB labels and trajectories. Old oracle rows cannot be repaired because they stored only booleans rather than oracle code.

### Round 2 — matched LCB ablations, run in parallel after collection

Train the same predictor architecture on the same temporal split and oracle labels:

| Condition | Predictor input | Purpose |
|-----------|-----------------|---------|
| Input-only | problem | benchmark-structure/difficulty baseline |
| Post-scout | problem + scout code | value of the attempted solution |
| Post-scout + public test feedback | problem + scout code + public test outcome | incremental value of deployable execution feedback |

Run one seed first as a validity gate. If the corrected result is nontrivial, repeat all three conditions with at least three seeds and report mean, standard deviation, and paired bootstrap confidence intervals.

A full public+private feedback condition is permitted only as a clearly labeled diagnostic upper bound. It is not a deployable primary result and must not be compared as though private evaluation outcomes were available to the router.

### Round 3 — analysis and paper decision

For both domains, report more than pooled AUC:

- Paired bootstrap confidence intervals for AUC differences
- Per-platform and macro-averaged LCB AUC
- A platform+difficulty metadata baseline to quantify benchmark shortcuts
- Capture-at-routing-budget and cost/coverage curves
- Calibration/Brier score if probabilities are used as routing thresholds
- Scout/oracle overlap and achievable routing ceiling
- Fixed thresholds selected on validation data, never optimized on the reported eval set

### Decision gates

1. **SWE feedback transfers — passed**: corrected test-feedback scoring beats both matched no-feedback transfer baselines. Retain SWE feedback as a positive cross-domain result; do not claim an incremental CoT gain.
2. **SWE feedback does not transfer, corrected LCB works**: frame the paper as a controlled positive result plus a boundary finding. Execution-grounded routing works in code generation; model-specific success in agentic software repair is much harder to infer and transfer.
3. **Corrected LCB input-only remains high but post-scout adds little**: the result is benchmark difficulty prediction, not scout-trace routing. Narrow the claim accordingly.
4. **Corrected LCB also collapses**: stop the multi-tier cascade work and do not pursue the current routing claim without a new signal or task.
5. **Only after a positive gate**: run Thinking-4B/CoT ablations, multiple seeds, and then consider the 20B/30B multi-tier cascade.

### Explicitly deferred

- Additional SWE router architectures or hyperparameter sweeps
- LCB Thinking-4B/CoT collection
- 20B/30B cascade tiers
- Opus expansion
- Private-test-feedback headline experiments

These do not answer the immediate validity question and should not run before rounds 1–3 establish a credible signal.

---

## Open Questions / Risks

1. **Can any scout-derived signal transfer on SWE?** CoT transferred worse than no-CoT (0.624 vs. 0.637). Correct test-feedback rescoring is the remaining high-value check; another failure supports a domain-boundary result rather than more SWE hyperparameter search.

2. **Are test feedback and CoT complementary on SWE?** This remains unknown. The old LCB run cannot answer it, and Daytona reports provide a coarser signal than full code-generation tests.

3. **Instruct vs. thinking on LCB**: Instruct-4B outperforms Thinking-4B on SWE-B agentic tasks. On LCB competitive programming, this may flip. The comparison is a clean experiment but the result is uncertain.

4. **62.1% ceiling-breaking: noise?** 285 instances is small. The CoT predictor at optimal threshold beating always-Opus is a strong result — worth checking confidence intervals.

5. **Other model pairs**: the 4B/120B pairing is one point on a large space. Results may not transfer to other small/large combinations.

---

---

## Exploratory Track: LCB Scout-First Cascade Routing (2026-08-19)

> **Status**: paused pending the corrected LCB validity gate. The old 100-problem solve rates and 0.95+ AUC results came from the broken evaluator and are not evidence for this extension.

### The idea

The original cascade problem: naive cascade (4B → 20B → 30B → 120b → Opus) pays cumulative
intermediate costs for hard problems, incurring all tiers before reaching the right one.

**Scout-first routing**: run the scout (4B) first, then use its output + test feedback to
**jump directly to the correct tier**, skipping all intermediate model calls.

Two bonuses stack:
1. **Free wins**: if scout passes tests → keep its output, pay nothing extra (~34% of problems)
2. **Rich routing signal**: scout failure mode (which tests failed, error type) is a much
   stronger difficulty signal than problem text alone → can confidently skip intermediate tiers

This is a proper multi-class routing problem, not binary abstention. The predictor maps
(problem + scout trajectory + test feedback) → {keep-scout | escalate-to-20B | escalate-to-30B | escalate-to-120b | abstain}.

### Why LCB is ideal

- Test evaluation is instant and free (subprocess, stdin/stdout)
- Scout (4B) already solves 34% of LCB problems — meaningful free-win rate
- Remaining 66% have a clear failure signal available immediately
- Input-only AUC already 0.9508 → predictor has very high information about difficulty tiers

### Solve rates and tier overlap (100-problem eval)

| Model | Solve rate | Marginal vs. 4B |
|-------|-----------|-----------------|
| Scout (4B) | 34% | — |
| gpt-oss-120b | 39% | +5pp |
| Claude Opus 5 | 43% | +9pp |

Per-problem: 29 solved by all, 14 solved by 120b/Opus but not 4B, 52 unsolvable by any.
Opus deferred from collection (5-problem "Opus-only" tier too thin for training signal, $25/MTok).

### Data plan

- **Dataset**: 892 usable LCB problems (post-2023-09-01 cutoff)
- **Temporal split**: 2024-10-01 cutoff → 551 train / 341 eval
  - 341 eval problems is much more statistically meaningful than original 100
- **Old collection**: cannot supply corrected labels; old oracle rows did not retain code and old scout grading was invalid
- **New collection**: re-collect the full corrected split with official grading
- **Test feedback**: public tests only for the primary router; full-suite grading only for final correctness

### Collection plan

First complete the corrected 4B/120B binary experiment above. Do not collect intermediate tiers until the corrected temporal analysis shows routing utility beyond input-only. If that gate passes, collect 20B and 30B outputs on the exact same problem IDs and retain all generated code so every tier can be regraded.

### Relationship to abstention track

The abstention track (binary: route to 120b vs. fall back to Opus 5) remains the primary
contribution. The cascade routing exploration either:
- Strengthens the LCB section with a richer multi-tier result
- Or becomes a separate contribution showing scout-first routing generalizes the abstention idea

---

## Related Work to Position Against

- **LLM cascades / routing**: FrugalGPT, routing networks, model selection papers — they route based on problem features, not scout output
- **Confidence-based abstention**: calibrated LLM confidence for selective prediction — we use a separate predictor rather than the oracle's own confidence
- **Difficulty estimation**: predicting instance hardness from problem features — we show scout traces dominate problem features
- **Process reward models / verifiers**: similar in spirit but trained to verify answers, not route between models; our predictor doesn't see the oracle's output at all
- **Adaptive compute**: early-exit networks, speculative decoding — related family but different mechanism
