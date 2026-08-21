# Paper Plan: Scout-Gated LLM Routing via Reasoning Traces

> Working title: **"Reasoning as Signal: Using Scout Traces to Route Hard Instances to Stronger Models"**

---

## Core Idea

When you're deciding whether to call an expensive large model, the obvious approach is to estimate difficulty from the problem statement alone. We argue there's a much richer signal available at almost no cost: let a cheap 4B model attempt the task first, and use its reasoning trace to predict whether the expensive model will succeed.

The scout's trace encodes *what it tried*, *where it got stuck*, and *how confident it was* — information that no amount of problem description analysis can reveal. This transforms difficulty estimation from a pure prediction problem into something closer to execution feedback.

The surprising result: a predictor trained on scout traces can route instances so well that it **exceeds the "always call the stronger model" ceiling** — not by improving either model, but by correctly partitioning which instances each model handles best.

---

## Experimental Axes

The project crosses two independent axes: the **decision regime** and the **dataset**. SWE is not
synonymous with abstention, and LCB is not synonymous with full routing.

### Axis 1: decision regime

#### A. Stronger-model success prediction / abstention

Run a cheap scout, then use the problem plus its trajectory, output, and optional test feedback to
predict whether a particular different, stronger model will succeed:

    P(stronger model succeeds | problem, scout evidence)

The target is the stronger model's success, not scout success. The prediction can decide whether
calling that stronger model is worth its cost, with low scores leading to abstention or a separately
specified fallback. The scout acts as a probe; this regime does not inherently decide among the
scout and several model tiers.

#### B. Full routing / cascade shortcutting

Run the scout once, then choose the final model action:

    Scout attempt -> router
    |-- keep scout answer
    |-- jump directly to a mid-tier model
    +-- jump directly to a high-tier model

The purpose is to shortcut a conventional sequential ladder such as
4B -> 20B -> 30B -> 120B. If the scout evidence indicates that an instance is very hard, route
directly to the appropriate high tier instead of paying for every intermediate attempt. The
headline result is the end-to-end frontier of **final correctness versus realized inference cost**,
not a single success-prediction AUC.

### Axis 2: dataset

Both regimes can be tested on either dataset:

| Dataset | Stronger-model success prediction / abstention | Full routing |
|---------|------------------------------------------------|--------------|
| **SWE** | Main completed line: predict gpt-oss-120b success from SWE-Smith scout evidence and test transfer on SWE-Bench Verified | Tried previously on SWE-Bench-style tasks and failed; tier selection appears substantially harder in this domain |
| **LCB** | Corrected temporal result: scout attempt predicts 120B success (AUC 0.769 vs. 0.552 input-only); public-test feedback has no demonstrated incremental gain in the first seed | Proposed next line after replication: keep the scout or jump directly to a mid/high tier |

### Experimental history and current hypothesis

The order matters. We first applied the abstention-prediction regime to LCB and observed
near-perfect apparent performance. That suggested LCB might supply the clean difficulty and
execution signals that were missing when full routing failed on SWE-Bench-style tasks, motivating a
new full-routing experiment on LCB.

The old LCB result cannot support that conclusion because its evaluator was broken and its split
leaked benchmark structure. The corrected LCB abstention experiment is therefore a **replication
and signal-validity gate**. The first corrected seed preserves a large scout-attempt signal beyond
problem-only features, but does not show an incremental public-test-feedback gain. Replicate before
collecting intermediate tiers; if the trace result holds, proceed to full routing on LCB.

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
- **Candidate tiers**: scout plus one or more mid-tier and high-tier models; the corrected initial
  collection currently includes `gpt-oss-120b` as the first stronger-model label
- **Execution**: the pinned official LiveCodeBench runner handles both stdin/stdout and function-call problems
- **Router observation**: problem + scout trajectory/code + public-test feedback only
- **Router action**: keep the scout or jump directly to a selected stronger tier
- **Final scoring**: execute the selected answer on the full public+private suite
- **Cost accounting**: charge the scout plus only the selected stronger tier, if any

The corrected 4B/120B LCB collection first supports the stronger-model success-prediction cell.
If that validity gate passes, the same problem IDs and scout evidence become the foundation for the
separate full-routing cell after intermediate-tier outputs are collected.

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

The corrected protocol uses official LCB grading for both stdin and function-call tasks, a pinned `release_v6`, a `2024-10-01` temporal split, public tests as routing feedback, and public+private tests for final scout/oracle correctness. The serial regrade removes evaluator concurrency failures. Two held-out oracle API timeouts are quarantined rather than counted as oracle failures, leaving 339 valid eval instances.

**Corrected temporal LCB result (2026-08-21, one seed)**:

| Predictor input | Final eval AUC | Interpretation |
|-----------------|----------------|----------------|
| Problem only | 0.5521 | Near chance; little useful signal from description alone |
| Problem + scout patch (no CoT) | **0.7693** | Large, credible scout-attempt signal |
| Problem + scout patch (no CoT) + public test feedback | 0.7346 | No demonstrated incremental feedback gain in this seed |

The previous 0.951-0.977 AUCs were higher because the target and split were invalid: function-call tasks were systematically labeled failed, private tests were decoded incorrectly, and random splitting enabled benchmark-era/platform shortcuts. The corrected collection also exposed a separate fork-safety issue in concurrent official-runner calls; serial regrading repaired the saved generations. These changes remove both label-format shortcuts and split leakage, so the new AUC is the relevant number. It is lower, but still strongly above the input-only baseline.

The feedback model reached a transient epoch-2 AUC of 0.7765, but selecting it using the held-out eval set would be post hoc. The reported final 10-epoch value is 0.7346. Multiple seeds, a validation-selected checkpoint policy, and paired confidence intervals are required before claiming a small difference between post-scout and post-scout-plus-feedback.

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
- Dataset revision: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`; load the official
  `test*.jsonl` files directly because `datasets>=4` no longer executes the repository's
  legacy loading script
- Split: train before `2024-10-01`, eval on/after `2024-10-01`
- Scout: `Qwen/Qwen3-4B-Instruct-2507`, local vLLM, temperature 0
- Oracle: `openai/gpt-oss-120b`, temperature 0
- Evaluator: pinned official runner commit `28fef95`
- Routing feedback: public tests only
- Final scout/oracle correctness: full public+private suite
- Task formats: stdin and function-call, both graded officially
- Resumption: only rows carrying the matching evaluator commit and feedback-suite marker are reusable

This collection replaces all old LCB labels and trajectories. Old oracle rows cannot be repaired because they stored only booleans rather than oracle code.

The first complete corrected collection job
(`lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448`) retained all generated code but
its grading pass was invalid: concurrent calls to the fork-based official evaluator produced
infrastructure errors that were recorded as failures. Quarantine its labels until the serial
regrade/retry job completes and strict validation passes. The saved generations can be repaired
without recollecting the scout or successful oracle outputs.

### Round 2 — LCB signal-validity gate, run in parallel after collection

Train the same predictor architecture on the same temporal split and 120B-success labels:

| Condition | Predictor input | Purpose |
|-----------|-----------------|---------|
| Input-only | problem | benchmark-structure/difficulty baseline |
| Post-scout | problem + scout code | value of the attempted solution |
| Post-scout + public test feedback | problem + scout code + public test outcome | incremental value of deployable execution feedback |

**Completed, one seed (2026-08-21)**: all three conditions used 551 temporal-train instances and 339 valid temporal-eval instances. Final AUCs are 0.5521 input-only, 0.7693 post-scout, and 0.7346 post-scout plus public feedback. The corrected validity gate provisionally passes for scout-attempt signal, but not for an incremental test-feedback claim.

Next: repeat all three conditions with at least three seeds, use a validation-selected checkpoint policy, and report mean, standard deviation, and paired bootstrap confidence intervals.

This round does **not** constitute the actual-routing result. It only establishes whether the scout
trajectory and public-test feedback add information beyond problem-only difficulty prediction. A
positive gate justifies collecting outcomes from the intermediate tiers needed to evaluate the
shortcut policy.

A full public+private feedback condition is permitted only as a clearly labeled diagnostic upper bound. It is not a deployable primary result and must not be compared as though private evaluation outcomes were available to the router.

### Round 3 — collect tiers and evaluate actual routing

After a positive signal-validity gate, collect outputs and full-suite correctness for the selected
mid-tier and high-tier models on exactly the same LCB problem IDs. Retain generated code from every
tier so labels can be regraded. Train or derive a router that selects among `keep scout` and direct
jumps to the available stronger tiers.

Primary LCB comparisons:
- Always scout, always each stronger tier, and random routing at matched cost
- Problem-only routing versus scout-trajectory routing versus scout+public-feedback routing
- Conventional sequential escalation versus direct scout-to-selected-tier jumps
- Oracle model selection computed from the collected per-instance tier outcomes as an upper bound

Report the correctness-cost Pareto frontier, expected inference cost at fixed correctness, and
correctness at fixed budget. AUCs are supporting diagnostics, not the actual-routing headline.

### Round 4 — analysis and paper decision

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
3. **Corrected LCB signal gate -- provisional pass**: post-scout reaches 0.7693 versus 0.5521 input-only on the first temporal seed. Replicate before treating this as a paper result; public feedback is not yet a positive result.
4. **Corrected LCB input-only remains high but post-scout adds little**: the result is benchmark difficulty prediction, not scout-trace routing. Narrow the claim accordingly.
5. **Corrected LCB also collapses on replication**: stop the multi-tier cascade work and do not pursue the current routing claim without a new signal or task.
6. **Only after replicated positive gate**: collect the chosen mid/high tiers and evaluate direct-jump routing.
   Thinking-4B/CoT ablations follow once the policy definition is stable.

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

## LCB Scout-First Cascade Routing (2026-08-19)

> **Status**: actual-routing track, pending replication of the corrected LCB signal-validity result.
> The first repaired temporal seed gives post-scout AUC 0.7693 versus 0.5521 input-only. The old
> 100-problem solve rates and 0.95+ AUC results came from the broken evaluator and are not evidence
> for this track.

### The idea

The original cascade problem: naive cascade (4B → 20B → 30B → 120b → Opus) pays cumulative
intermediate costs for hard problems, incurring all tiers before reaching the right one.

**Scout-first routing**: run the scout (4B) first, then use its output + test feedback to
**jump directly to the correct tier**, skipping all intermediate model calls.

Two bonuses stack:
1. **Free wins**: if the router keeps a correct scout output, pay nothing beyond the scout
2. **Rich routing signal**: scout failure mode (which tests failed, error type) is a much
   stronger difficulty signal than problem text alone → can confidently skip intermediate tiers

This is a proper multi-class routing problem, not binary abstention. The predictor maps
(problem + scout trajectory + test feedback) → {keep-scout | escalate-to-20B | escalate-to-30B | escalate-to-120b | abstain}.

### Why LCB is ideal

- Test evaluation is instant and free (subprocess, stdin/stdout)
- Public tests provide immediate, deployable feedback on the scout attempt
- Competitive-programming tasks offer explicit correctness and relatively cheap per-tier evaluation
- The old run appeared to show unusually strong difficulty signal, which motivated this experiment;
  corrected grading must establish whether that signal was real

### Corrected binary anchor (temporal eval)

The previous 100-problem multi-tier solve-rate table is invalid and must not be used. The repaired
4B/120B collection has 339 valid held-out instances after quarantining two unresolved 120B API
calls:

| Model | Solve rate |
|-------|------------|
| Scout (Qwen3-4B-Instruct-2507) | 93 / 339 = 27.4% |
| Oracle (gpt-oss-120b) | 224 / 339 = 66.1% |

These are only a binary anchor, not routing outcomes. No corrected 20B, 30B, or Opus outcomes have
been collected, so multi-tier overlap and any cascade frontier remain unmeasured.

### Data plan

- **Dataset**: 892 usable LCB problems (post-2023-09-01 cutoff)
- **Temporal split**: 2024-10-01 cutoff → 551 train / 341 eval
  - 341 eval problems is much more statistically meaningful than original 100
- **Old collection**: cannot supply corrected labels; old oracle rows did not retain code and old scout grading was invalid
- **New collection**: re-collect the full corrected split with official grading
- **Test feedback**: public tests only for the primary router; full-suite grading only for final correctness

### Collection plan

First complete the corrected 4B/120B binary experiment above. Do not collect intermediate tiers until the corrected temporal analysis shows routing utility beyond input-only. If that gate passes, collect 20B and 30B outputs on the exact same problem IDs and retain all generated code so every tier can be regraded.

### Position in the experiment matrix

This is the LCB/full-routing cell. It was proposed because the earlier LCB/abstention cell appeared
near-perfect, while the SWE/full-routing cell had failed. The corrected 4B/120B LCB predictor is an
intermediate replication and signal-validity check, not the final policy experiment. Its job is to
determine whether the observation that motivated LCB full routing survives correct grading and a
temporal split.

---

## Related Work to Position Against

- **LLM cascades / routing**: FrugalGPT, routing networks, model selection papers — they route based on problem features, not scout output
- **Confidence-based abstention**: calibrated LLM confidence for selective prediction — we use a separate predictor rather than the oracle's own confidence
- **Difficulty estimation**: predicting instance hardness from problem features — we show scout traces dominate problem features
- **Process reward models / verifiers**: similar in spirit but trained to verify answers, not route between models; our predictor doesn't see the oracle's output at all
- **Adaptive compute**: early-exit networks, speculative decoding — related family but different mechanism
