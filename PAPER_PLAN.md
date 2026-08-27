# Paper Plan: Scout-Gated LLM Routing via Reasoning Traces

> **Protocol correction (2026-08-25, latest):** The public/private-test formulation is no longer
> the primary deployment protocol. All LCB MDP figures and LCB post-scout results below that use
> public feedback are retained as historical weak-verifier experiments and are superseded for the
> main claim. Protocol v2 uses full execution, enters the router only after a verified scout
> failure, constructs reachable histories only, and reports a disjoint train/calibration/test
> split. See the final section of this document.

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

## SCOOPED (partially): Resample-or-Reroute covers the sequential execution-grounded MDP (2026-08-21)

**"Resample or Reroute? Budget-Aware Test-Time Model Selection for Large Language Models"**
(Teng-Ruei Chen, arXiv 2607.08665, posted 2026-07-09) formulates almost exactly the sequential
policy sketched above:

- Actions `{RESAMPLE(m), REROUTE(m')}` as competing uses of one per-query budget; claims to be
  first to unify them
- Online allocation policy (greedy marginal-correctness-per-cost + UCB variant), belief updated
  from verifier-scored draws
- Real code verifier: HumanEval+ base tests as deployable signal (~1% false-accept vs full suite)
- Cost-quality Pareto fronts beating FrugalGPT cascades, budget-aware best-of-K, one-commit
  routers on an 11-model pool
- Companion paper proves the "recoverability asymmetry"; public correctness tensors +
  reproducibility repo

### Their experimental scope

GSM8K, MATH-500, GPQA-Diamond, HumanEval+ — all single-shot. Offline replay of precomputed
correctness tensors (k=30 draws/cell, T=0.2); no agentic benchmark, no SWE-bench, no sandbox
economics. Key honest finding: gains are **verifier-gated** — they shrink as verifier quality
drops and can invert under agreement-based verification.

### Our prior attempt at the weak-verifier version — already failed

We tested the learned-controller variant of this policy on SWE-Smith before RoR appeared
(`analyze_swe_smith_multirollout_learned_controller_cv_1781899785`, n=100-fold CV): the proxy
verifier predicted attempt correctness without executing tests, and the learned resample/escalate
controller's utility was **below the best static baseline at every cost weight λ** (−0.008 to
−0.078), with unstable oracle-gap capture. Consistent with the introspection-fails trilogy:
without real execution, patch correctness in agentic repair is not predictable enough to drive
sequential decisions.

### Remaining open deltas (the honest list)

1. **The abstain/give-up arm.** RoR has no give-up action — every budget unit buys another draw.
   Integrating calibrated P(no model succeeds) into the allocation (reject-option theory cited,
   not used) is open.
2. **Agentic SWE domain.** Multi-turn repair with expensive, slow sandbox evals changes the
   allocation economics entirely; untested there.
3. **The weak-verifier regime via traces.** Their verifier-gating ablation is the gap our trace-
   prediction work targets — but our own SWE-Smith result above suggests learned predictors do
   not fill it in agentic repair. Whether scout-trace prediction works as a weak verifier on LCB
   (where the corrected post-scout AUC 0.769 exists) is the open question.
4. Scout-first direct-jump framing and temporal-split evaluation rigor.

### Consequence for the paper plan

Do NOT claim first formulation of resample-vs-escalate. The full-routing LCB experiment proceeds
as planned (it answers a different question: direct-jump cascade shortcutting with model-specific
success targets), but any sequential-policy extension must be positioned explicitly as extending
RoR to {abstention arm, agentic domain, expensive-verifier economics}. Given the delta is now an
increment rather than a new problem, the decision gate shifts back toward the signal-science /
boundary-findings core plus the LCB frontier result.

---

## Early Exit / Early Stopping / Abort Literature for Software Agents (surveyed 2026-08-21)

A fast-growing 2026 cluster predicts agent-trajectory failure mid-run and intervenes. All found
work stops/aborts/retries **with the same model**; none routes across models based on the signal
(exceptions noted). This matters for positioning: the "predict failure early" idea is taken, but
"failure prediction → cross-model routing decision" is not.

| Paper | Venue | Domain | Signal | Action |
|-------|-------|--------|--------|--------|
| **FailFast–RestartSmart** (arXiv 2608.03222) | preprint | SWE-bench Verified | 0.6B monitor on observable trajectory text (+ dense F2P replay supervision), no logits/hidden states needed | Abort at FPR budget → fresh same-policy retry with edit-overlay carry-over. Saves 15–20% tokens; restart +5.2pp resolve. **Explicitly notes escalating aborted instances to a larger model as an uninvestigated alternative** |
| **EET** (arXiv 2601.05777, ACL Findings 2026) | ACL Findings | SWE-bench Verified, 3 agents | LLM confidence scores conditioned on retrieved historical experience, checked at milestones during generation AND selection | Two-sided stop: continue / accept patch early / **halt ("unlikely resolved")** — the lower threshold is effectively intra-agent abstention. −32% cost, ≤0.2pp resolve loss |
| **Doomed from the Start** (arXiv 2607.06503) | preprint | TextCraft, WebShop | Linear probes on hidden-state activations, round 1 onward | Recall-controlled calibrated abort cascade with distribution-free certification; up to 60% token savings at 90% success-recall. No model switching |
| **AgentStop** (ACM CAIS 2026, Brave) | CAIS | SWE-bench Verified + QA | Token logprobs, step counts, repetition features → GBT | Terminate unlikely-to-succeed local agents for energy savings; 15–20% waste reduction, <5% utility drop |
| **Atropos** (arXiv 2604.15075) | preprint | SWE agents + self-consistency | GCN over inference-path graphs, mid-run | Early terminate + **hotswap context to stronger LLM** — the only one crossing models |
| **AgentDiet** (arXiv 2509.23586) | preprint | coding agents | LLM reflection module | Trajectory token reduction (−21–36% cost); orthogonal efficiency axis |

### Reading for our positioning

1. **The occupied claim**: mid-trajectory failure prediction on SWE-bench is now standard
   equipment (four independent systems). Do not claim it.
2. **The open seam**: every system except Atropos treats failure as a reason to *stop or retry
   the same policy*. None asks "given this predicted-failure evidence, which *other* model should
   run, or should we give up?" — i.e., the routing/abstention decision layered on top of
   execution-grounded failure signals. FailFast names this exact extension as future work.
3. **A subtle distinction that protects our signal-science claims**: these systems predict
   *their own run's* outcome from *long partial prefixes*. Our negative results concern
   predicting *a different model's* success from a *completed* short attempt. Both can be true:
   self-failure prediction from rich trajectories works (their result), cross-model success
   prediction from scout attempts is what's hard (our result). State this carefully — reviewers
   will otherwise read FailFast as contradicting our introspection-fails trilogy.
4. EET's lower confidence threshold ("halt, issue unlikely resolvable by this configuration") is
   the closest existing thing to deployment-grade abstention in agentic repair — cite it as such,
   but note it never hands off to another model and its halt decisions are LLM-judged rather than
   calibrated against measured P(oracle succeeds).

---

## Status Summary of Active Threads (2026-08-21)

1. **LCB full-routing frontier (main methodological bet)**: oss-20B tier collection running;
   feeds `launch_lcb_full_router_train.sh`. Direct-jump cascade shortcutting vs sequential
   ladders and fixed policies. Must work for the method-paper framing.
2. **RoR differentiation**: content-based prediction of model-specific success vs RoR's count-
   based beliefs over verifier bits; plus abstain arm, agentic economics, temporal-split rigor.
3. **Signal-science core**: introspection-fails trilogy (self-assessment 0.570 saturated; token
   entropy ≤0.60; proxy-verifier controller below static baseline), content-predicts-where-
   introspection-doesn't, transfer asymmetry (feedback yes, CoT no), SWE/LCB boundary.
4. **Ceiling-breaking CI**: +4.2pp over always-Opus; honest bootstrap [+0.4, +8.8] excludes zero,
   naive grazes. Thin but real; report both CIs.
5. **Abstention reframed**: dead as policy novelty; alive as economic give-up branch
   (resolve-per-dollar) and as the open RoR delta. EET's halt-threshold is prior art to cite.
6. **Multi-sample static router**: dissected and closed — pass-pattern-as-router is incoherent
   (any pass → submit; only decision point is all-fail where signal degenerates to one bit).

Decision point: when oss-20B data lands → retry failed rows → train router variants → frontier
result decides between exciting-method-paper and analysis/boundary-paper framings per existing
gates.

---

## Confidence Baselines Fail: Self-Assessment and Token-Level Uncertainty (documented 2026-08-21)

These two experiments establish that neither the generator's implicit uncertainty nor its explicit
self-report carries routing signal, while a third model reading the attempt does. Together they
support the paper's core claim: the value comes from the **evidence content of the attempt**, not
from model introspection. Both were run earlier but never tabulated; results below were computed
from the saved artifacts on 2026-08-21.

### Baseline A — oracle verbalized self-assessment (pre-attempt)

**Source**: `/mnt/llmd/results/exps/aristides/reason/self_assessment_eval.jsonl` (286 SWE-Smith eval
instances), collected by `pipelinerl/swe/scripts/offline_router/collect_self_assessment.py`.
gpt-oss-120b was asked *before attempting*: "Can you fix this bug? Y or N" (single character, no
reasoning allowed). Soft confidence `p_yes` came from answer-token logprobs; hard `Y/N` from the
decoded answer. Labels are real Daytona route-3 (oss-120b success) outcomes joined from the
4-route real-label parquet (`offline_router_swe_smith_train1500_real_labels_4route_1780639659`,
eval split).

| Metric | Value |
|--------|-------|
| n instances | 286 |
| 120b solve rate | 47.2% |
| **AUC of p_yes vs actual 120b success** | **0.570** |
| Mean p_yes overall | 0.902 |
| Mean p_yes where 120b failed | 0.868 |
| Mean p_yes where 120b succeeded | 0.941 |
| AUC vs scout success (control) | 0.567 |
| Cost per query | ~5,300 prompt + ~214 completion tokens through the 120B (210 of them reasoning tokens despite the no-reasoning instruction) |

Interpretation:

- The failure mode is **saturation, not mere miscalibration**: 183/286 instances received
  p_yes ≈ 1.0 and only 28 fell below 0.5. The model essentially always says yes, leaving almost no
  variance to rank with; residual AUC 0.570 is barely above chance.
- The success/failure confidence gap (0.941 vs 0.868) is trivial.
- For comparison on comparable data: input-only embedding predictor ≈ 0.682 in-domain and
  post-scout ≈ 0.716–0.749. A trained predictor over problem text alone beats asking the oracle
  itself by ~11 AUC points.
- Caveat to state in the paper: self-assessment requires no training labels, whereas the embedding
  predictors need ~1k+ labeled instances. But those labels are required for any learned router
  anyway, so the marginal cost of the trained baseline is zero and verbalized confidence is
  strictly dominated in every deployment-relevant regime.

### Baseline B — scout token-level uncertainty (post-hoc entropy/logprobs)

**Source**: `analysis/entropy_abstention_results.json`, produced by
`analysis/entropy_abstention.py`. Qwen3-4B-Instruct-2507 was teacher-forced over its own saved
SWE-Smith outputs (no re-generation, so Daytona labels stay valid); per-token full-vocab entropy
and chosen-token logprobs were aggregated into instance-level signals. n = 217 eval instances,
real labels. Signals predict oss-120b (route 3) and Claude Sonnet success.

| Signal (↑ = confident) | AUC vs 120b | AUC vs Claude |
|------------------------|-------------|---------------|
| mean_token_logprob | **0.447** | **0.385** |
| −mean_entropy | **0.462** | **0.401** |
| −early_entropy (first 256 tokens) | 0.603 | 0.564 |
| −output_length | 0.660 | 0.621 |
| Input-only trained head | 0.634 | 0.623 |
| Post-scout trained head | **0.716** | 0.638 |

Interpretation:

- Token-level confidence sits at or **below chance** for the strongest oracle (0.447/0.462).
  Below 0.5 means confident scout outputs were *less* likely to have the oracle succeed — the
  signal is not merely absent but mildly inverted.
- Trivial heuristics beat every uncertainty measure: output length alone (0.660) outperforms all
  entropy variants; models get verbose when struggling.
- The trained post-scout head dominates (0.716), confirming the attempt's content carries signal
  the generator's internal state does not surface.

### Combined argument for the paper

Cost-ordered baseline ladder on SWE-Smith (AUC for predicting 120b success):

verbalized confidence 0.570 < token-level entropy ≈0.45–0.60 < prompt-length heuristic 0.563–0.66
< input-only embedding 0.653–0.682 < post-scout embedding 0.716–0.749 < post-scout + test feedback
0.780

Each step up costs more compute but actually buys signal. Reviewers asking "why not just calibrate
the model's own confidence?" can be answered with these two rows: implicit introspection fails
(Baseline B) and explicit self-report fails (Baseline A). Remaining gap worth closing before
submission: a self-consistency baseline (k samples, agreement rate) — it costs k× scout inference
but reviewers may request the strongest cheap baseline rather than the weakest one.

---

## Multi-Sample Execution-Verified Baseline (analyzed 2026-08-21)

The strongest cheap baseline candidate for the abstention claim: run k independent scout
attempts, grade each in Daytona, and use the pass/fail pattern as the routing signal. Analyzed
from the existing multi-rollout collection
(`/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_multirollout_eval150_proxy_verifier_rescore_dataset_1781735193`,
150 SWE-Smith eval instances x 3 Qwen3-4B-Instruct rollouts, real Daytona labels).

### Raw numbers

| Quantity | Value |
|----------|-------|
| AUC of scout-pass-fraction vs 120b success | **0.841** (any-of-3 oracle labels) / 0.831 (rollout-0 oracle) |
| Oracle success given 0/3 scout passes | **14.6%** (n=89) |
| Oracle success given ≥1/3 scout passes | 75–86% |
| Best-of-3 scout (submit if any passes) | ~free resolve gains already covered by `best_of_rollout_stats.json` |

On its face 0.84 beats every single-attempt trained predictor (0.716–0.78).

### Why the headline AUC overstates the baseline (important correction)

A deployable policy never sees the graded agreement levels: **if any scout patch passes, you
submit it and never invoke the router.** The only decision point is "all k patches failed," where
the signal is deterministically 0/k — execution ground truth, not a prediction. The graded
fraction {⅓, ⅔, 1} never reaches a threshold. So:

- The AUC 0.841 ranks instances a real policy would have already resolved for free; as a
  *decision* signal it degenerates to a single bit ("did all k fail").
- The policy-relevant quantity is the conditional rate: after k scout failures, the oracle still
  succeeds 14.6% of the time. Escalating on all-fail captures those; abstaining saves oracle cost
  but cannot raise resolve rate. This is a **budget-constrained economics decision**, not a
  resolve-rate improvement.
- The "consensus policy beats always-120B" framing is mostly best-of-k with execution
  verification wearing a router costume. It must be reported as such.

### Honest accounting

Cost per decision: k scout generations + k sandbox evaluations, versus one generation + one
embedding pass for the post-scout predictor. In exchange the k-sample method obtains *certainty*
about scout failure rather than an estimate. Where unit tests are available this is the baseline
to beat on the cost axis; the paper's claim must be that trace-based prediction recovers most of
the signal at roughly one-third the execution cost, or works where execution is unavailable
(SWE-Bench-style grading is not available mid-pipeline).

### What multi-sampling actually buys: a sequential execution-grounded policy

The coherent form of the k-sample idea is not a static router but a state machine over
{resample scout | escalate | stop}, with real evaluation outcomes as the state
(`small_resample_vs_big_headroom.json`, n=148 overlap):

| State | Resample value | Shift-up value |
|------|----------------|----------------|
| Scout attempt passes | Submit immediately. Escalation is counterproductive: the stronger model fails 9.6% of scout-passed instances; further sampling breaks it 5.8% | — |
| Scout attempt fails | Any of 2 further scout attempts succeeds 15.6% | 20B succeeds 25.0% |

With execution in the loop, every branch has known payoffs and resample-vs-escalate reduces to a
cost-ratio calculation — an adaptive cascade, most natural on LCB where grading is instant and
free. The learned-controller variant of this was tested previously
(`analyze_swe_smith_multirollout_learned_controller_cv_1781899785`, verifier calibration dirs) and
failed for a specific reason: the trained proxy verifier could not predict patch correctness
without running tests. This replicates, in a third guise, the same finding as the verbalized
self-assessment and token-entropy baselines above: in agentic repair, introspective signals do not
predict execution outcomes; only execution itself does. This consistency strengthens the domain
boundary claim — it explains why SWE routing is hard and why execution-grounded routing should be
evaluated where execution is cheap (LCB).

---

- **Atropos** (arXiv 2604.15075): R agentic samples of an SLM, GCN predicts failure, early-stop or
  hotswap mid-run to GPT-4o. Closest existing system to multi-trace-then-select-large.
- **AI21 budget-aware SWE agents**: parallel rollouts + learned stop-classifiers on rollout
  features (patch consistency, generated-test outcomes); cascade over budget levels.
- **"LLMs Encode Their Failures"** (arXiv 2602.09924): probes predict policy-specific success;
  probe-guided pool routing exceeds best single model at ~70% cost reduction (maj@5 setting).
- **SWE-Router** (arXiv 2607.00053): cheap model runs K turns, value head reads partial
  trajectory, continue-vs-escalate on SWE-Bench Verified (Route-AUC 0.78); Bayes-optimality result
  for trajectory conditioning; notably did *not* beat baselines on its SWE-Smith split.
- **SuperScout / "Scrouting"** (arXiv 2608.04804): 7B searcher explores repo, hidden states feed
  an N-way frontier-fixer router on SWE-bench Pro; handoff consumed by fixer.
- **AutoMix** (NeurIPS 2024): few-shot self-verification + POMDP accept/escalate router; includes
  an explicit "unsolvable, route nowhere" class — the main prior art for true abstention.
- **FrugalGPT** (TMLR 2024): sequential cascade with a *trained* DistilBERT answer-quality scorer
  and thresholds — structurally, our binary abstention cell is FrugalGPT restricted to two tiers
  with an embedding scorer. This is the sharpest form of the "not novel alone" critique and must
  be answered head-on.
- **RouteLLM / Hybrid LLM**: input-only routers (pre-generation), no attempt evidence — the
  baselines our post-scout predictors dominate.
- **Dekoninck & Baader, ICML 2025**: unified routing/cascading theory; identifies quality-estimator
  quality as the critical factor — supports framing signal validity as the core contribution.
- **RLCascadeRouter** (2026): RL cascade routing without quality estimators.
- **UCCI** (2026): calibration-first uncertainty cascades (isotonic on token margins).
- **Darwin Cascade** (non-archival): empty-patch as a deterministic 100%-precision escalation gate
  on SWE-bench Lite — a degenerate execution-feedback gate, useful as a reference point.

### Positioning consequence

Binary scout→predict-oracle→abstain, alone, is a two-tier FrugalGPT/AutoMix variant and will be
reviewed as such. The defensible contributions are: (1) the signal science — CoT vs execution
feedback ablations, transfer, and the SWE/LCB domain boundary; (2) multi-tier direct-jump routing
versus sequential cascades with model-specific success targets and the ceiling-breaking
partitioning result; (3) true abstention modeled explicitly as a give-up branch (P(scout),
P(oracle), escalate only when oracle-P high and scout-P low) — most cascade work assumes the large
model always helps and never prices the hopeless branch; (4) corrected-evaluation rigor
(temporal splits, evaluator pinning, leakage quarantine) as methodology contribution.

---

- **LLM cascades / routing**: FrugalGPT, routing networks, model selection papers — they route based on problem features, not scout output
- **Confidence-based abstention**: calibrated LLM confidence for selective prediction — we use a separate predictor rather than the oracle's own confidence. Both our verbalized-confidence and token-level-uncertainty baselines fail empirically (see previous section), so this positioning claim is now backed by data
- **Difficulty estimation**: predicting instance hardness from problem features — we show scout traces dominate problem features
- **Process reward models / verifiers**: similar in spirit but trained to verify answers, not route between models; our predictor doesn't see the oracle's output at all
- **Adaptive compute**: early-exit networks, speculative decoding — related family but different mechanism

---

## LCB Full-Routing First Results + Failure-Dumping Diagnosis (2026-08-22)

First embedding-router runs on the corrected temporal LCB split (n=339 eval), one seed, using the
repo-standard multi-route trainer (`train_qwen_embedding_router_baseline.py`, `--objective
reward_bce`, one LoRA with an N-dim output over {scout, oss20, oss120}). Jobs:
`lcb_embed_router_{postscout_testfb,inputonly}_seed17_1787374{180,211}`.

### Frontier summary

| Operating point | Post-scout+FB | Input-only | Random (matched) | Reference |
|---|---|---|---|---|
| Top | 60.8% @ 43.5k | **65.2% @ 49.9k** | 62.3% @ 56.8k | always-120B: 66.1% @ 64.1k |
| Mid | **54.3% @ 19.4k** | 56.1% @ 22.6k | 44.8% @ 27.0k | always-oss20: 40.4% @ 14.4k |
| Low | **46.9% @ 10.8k** | 45.1% @ 10.1k | 38.3% @ 16.2k | |

Embedding ≫ the TF-IDF router trained the same morning (top end 59.6% → 65.2%). Router beats
random-at-matched-cost at every λ (+1 to +10pp). Direct-jump still edges the sequential public-
feedback cascade (59.3% @ 42.1k).

### Failure-dumping diagnosis (the important finding)

Per-instance analysis of the input-only router at λ=0:

- It routes 50 instances away from 120B; **0/50 deviations succeed**. 92% of deviated instances
  are *all-hopeless* (no route solves them); 120B would have resolved only 6%.
- Routed-to-oss20 success rate: **0%** vs 40.4% base rate — the oss20 head is used as a garbage
  disposal for hopeless instances under cost-weighted argmax.
- So the "65.2% ≈ always-120B at 78% of the cost" headline is NOT cheap-win discovery: it is
  default-to-expensive plus rational **failure dumping** — identify hopeless instances and shunt
  them to the cheap tier instead of wasting a 64k-token call on a ~6% shot.

Post-scout+FB shows the same pattern, weaker: 85 deviations, 9.4% chosen-success, 69%
all-hopeless.

**Implication for thread (a)**: roughly 30% of eval instances are all-hopeless AND identifiable
(per-head AUCs from problem text alone: scout 0.854 / oss20 0.785 / oss120 0.827). The router
invented the give-up action implicitly; the MDP should make it explicit. This is the strongest
evidence yet that the abstain arm has real value on LCB.

### Unresolved red flag — reconcile before trusting either number

Problem-statement-only per-head AUCs here are 0.79–0.85, but the corrected abstention experiment
(2026-08-21) measured input-only AUC **0.5521** for the same target (oss120 success), same
temporal split. Differences between setups: multi-task (3 heads) vs single-target training; lr
1e-4 vs 2e-5; input template (route legend + prompt vs abstention format); max_seq 8192 vs 24000;
checkpoint selection by best eval_loss (here best epoch = final, so mild). A 0.28-AUC gap cannot
be hand-waved. Required reconciliation experiment: single-target vs multi-task on identical
converted data with everything else matched, both variants, then re-run policy evaluation.

### Caveats on all numbers above

Single seed; checkpoint selected by eval loss (touches eval); route mixes show the λ=0 policies
send almost nothing to scout despite its free wins — calibration of the keep-scout arm remains
poor in both TF-IDF and embedding routers. Multi-seed replication + paired bootstrap CIs are next,
followed by the reconciliation experiment above.

---

## Hyperparameter Reconciliation: The Input-Only Baselines Were Undertrained (2026-08-23)

**The single most important finding of this cycle.** Every predictor trained before 2026-08-22
used lr 2e-5 / seq 24000 (`train_cot_abstention_predictor.py` defaults). Controlled re-runs on
identical corrected LCB temporal data show this recipe severely undertrains the LoRA:

### LCB input-only (problem text → oss120 success), single-target grid

| Config | oss120 eval AUC |
|--------|-----------------|
| lr 2e-5, seq 24k (= old abstention gate recipe) | **0.500** |
| lr 5e-5, seq 8192 | 0.758 |
| **lr 1e-4, seq 8192** | **0.850** |
| multi-task 3 heads, lr 2e-5 | 0.778 |

Runs: `lcb_reconcile_a{1..4}_*`. A2 reproduces the old gate's ≈chance result exactly (harness
validated); A1 shows the same input contains a 0.85-AUC signal.

### Consequences for the scout thesis

- **LCB**: post-scout+test-feedback at lr 1e-4 scores **0.789** (`lcb_reconcile_a5_*`) — BELOW
  properly-trained input-only (0.850). On LCB, scout content adds nothing to oss120-success
  prediction; the original gate comparison (post-scout 0.769 vs input-only 0.552) was an artifact
  of comparing a converged model against an undertrained one. "Scout before you route" is dead on
  LCB as a prediction-improvement claim.
- **SWE**: input-only re-run at lr 1e-4 gives in-domain **0.7267** (was 0.682) and Verified
  transfer **0.624** (was 0.578). Also undertrained. Post-scout @ 1e-4 was still running at
  documentation time (`...route3_nocot_10epoch_1787458284`) — its result decides whether scout
  patches add signal on agentic repair once the baseline is honest.
- ALL AUC tables above in this document that predate 2026-08-22 must be read as
  lr-2e-5-conditioned; relative claims (CoT vs no-CoT, feedback deltas) need re-derivation at
  converged settings before publication.
- Routing frontier results (2026-08-22 section) survive: they never claimed scout superiority,
  and the input-only router being strong is consistent with this finding.
- Failure-dumping diagnosis unchanged and reinforced: problem text alone identifies the
  all-hopeless cluster (that is precisely what 0.85 AUC means).

### Required going forward

1. New standard recipe: lr 1e-4 (or validated convergence point), seq 8192 for LCB-length inputs;
   report learning-rate sensitivity for any headline AUC.
2. Re-run the SWE 2x2 ablation (no-CoT / CoT x feedback) at lr 1e-4 before quoting any scout-value
   delta.
3. Await SWE post-scout @ 1e-4 result to determine whether ANY scout-content claim survives on
   either domain.

---

## Thread (a) MDP: Setup Status (2026-08-23)

**Goal**: {resample | escalate | abstain} MDP over real execution signals on LCB, positioned as
RoR-extension (content-vs-counts + explicit give-up arm + their verifier-gating regime).

### Data collection (in progress)

Correctness tensors `(problem × model × draw)` on the temporal eval split (339 problems),
protocol RoR-parity: T=0.2 primary, k=10 draws per model; plus scout-only T=0.6 sensitivity arm.

- `collect_lcb_expert.py` extended: `--output-suffix` (per-draw files keep resume logic),
  `--splits`, `_generation_temperature` provenance field.
- Launcher `launch_lcb_multidraw_mdp_collect.sh` with MODE=scout / MODE=experts:
  - **scout job (GPU)** `lcb_multidraw_scout_1787460253`: local vLLM serves Qwen3-4B-Instruct-
    2507 (Qwen models are NOT available on OpenRouter — HTTP 400 "not a valid model ID", verified
    by direct API test after all 341 rows failed); collector hits localhost.
  - **experts job (CPU)** `lcb_multidraw_experts_1787460296`: oss20+oss120 via OpenRouter,
    sequential loop (official evaluator forbids concurrent forks).
- Estimated wall time: hours (10k generations + 2× serialized gradings each).

### Still to build (while collection runs)

1. Tensor builder: draw jsonls → (problem × model × draw) outcome arrays + public-feedback texts.
2. Policy replay suite: RoR-faithful greedy/UCB counting rule (baseline), content-predictor policy
   (same allocation rule, probabilities from the embedding router), ABSTAIN extension (give-up arm
   gated on posterior mass below threshold), cascade / best-of-K / static-router / random
   controls.
3. Eval harness: budget sweeps → cost-correctness Pareto fronts, paired bootstrap CIs, train-half
   prior calibration, threshold selection on train only.
4. Note: existing T=0 collections remain valid for thread-(b) static routing; T=0 draws cannot be
   reused for resampling.

### Decision-relevant context

RoR (arXiv 2607.08665) already owns the counting-based version of this MDP (k=30, T=0.2). Our
deltas: abstain arm, content-based beliefs, and the weak-verifier/agentic regime. Given the
reconciliation finding (input-only ≈ scout on LCB prediction), the MDP's content-policy may not
need scout evidence at all on LCB — problem-text difficulty alone may drive it — which simplifies
the story to "explicit give-up arm + calibrated allocation."

---

## THE HEADLINE FINDING: Scout Evidence Adds Nothing In-Domain on Either Domain (2026-08-23)

Completing the reconciliation matrix. All cells single-target oss120-success prediction,
converged recipe (lr 1e-4):

| Domain | input-only | post-scout | delta |
|--------|-----------|------------|-------|
| LCB (temporal, n=339) | **0.850** | 0.789 (+test-fb) | −6.1pp |
| SWE-Smith in-domain | **0.727** | 0.694 | −3.3pp |
| SWE → Verified transfer | **0.624** | 0.620 | −0.4pp |

Runs: `lcb_reconcile_a{1,5}_*`, `swe_smith_instruct_to_verified_transfer_route3_nocot{,_input_only}_10epoch_1787457{281,284}`.

**Claim**: with properly trained baselines, scout patches — and even scout execution feedback on
LCB — do not improve stronger-model success prediction over the problem statement alone. Point
estimates are negative on both domains independently.

### Interpretation

- Problem statements already encode instance difficulty; a weak model's attempt adds evidence
  about *that model's* limitations, which is largely predictable from difficulty plus noise.
- Earlier "scout beats problem-only" results measured an undertrained baseline (see previous
  section). All pre-2026-08-22 relative claims in this document inherit that artifact.
- Mechanistically plausible that scout content actively hurts: predictors latch onto 4B-specific
  failure patterns rather than intrinsic hardness.

### Status of the three research threads after this finding

- **Thread (b) "scout before you route"**: dead as a prediction-improvement story on both
  domains. Routing frontiers survive but their value comes from cost allocation + the all-hopeless
  cluster, not from scout evidence.
- **Thread (a) MDP**: survives fully — resample/escalate/abstain economics never required scout
  superiority; the abstain arm is motivated by the identifiable ~30% hopeless cluster.
- **Thread (c)**: transforms from backup into potentially the main scientific contribution: a
  rigorous negative result — "attempt-based signals do not add information beyond problem text for
  routing to stronger models" — plus the introspection-fails trilogy and corrected-evaluation
  methodology.

### Mandatory before publishing this claim

1. Paired bootstrap CIs on both in-domain deltas (are they significantly ≤ 0, or noise?)
2. ≥2 additional seeds on the four headline cells
3. One alternative-target check (does scout content predict oss20 or scout success better than
   problem text? establishes whether the null is target-specific)
4. Re-write affected sections above; mark superseded tables as lr-2e-5-conditioned

### Thread (a) collection status update

- Scout multidraw job `lcb_multidraw_scout_1787460253` exited without collecting: argparse
  requires `--api-key-file` even when the key is provided via environment. Fix: create a dummy
  local key file in the runner and pass it explicitly. Relaunch pending.
- Experts job `lcb_multidraw_experts_1787460296` running normally (~25 min per draw-file, 20 files
  total ≈ 8h).

### VERIFICATION REVERSAL (2026-08-23, later same day)

The multi-seed verification battery overturned the seed-17 conclusion above:

| Cell | seed17 | seed18 | seed19 | mean ± sd |
|------|--------|--------|--------|-----------|
| LCB input-only | 0.850 | 0.848 | 0.732 | 0.810 ± 0.055 |
| LCB postscout+fb | 0.789 | 0.905 | 0.891 | 0.862 ± 0.053 |

- SWE seed18: mixed again (input-only wins in-domain by 4pp, post-scout wins transfer by 5pp).
- Alternative-target check (oss20): postscout+fb 0.798 > input-only 0.754.

**Corrected conclusions:**

1. There is NO reliable scout effect in either direction on either domain; the seed-17
   "scout hurts significantly" claim was itself an artifact of run-level variance.
2. **The dominant empirical effect is training-run variance (~±0.06 AUC)** for LoRA fine-tuning
   on ~500 examples. Every single-run AUC comparison in this document (and in much of the routing
   literature) is fragile at this data scale. This is a publishable methodological finding.
3. Any real scout delta is small relative to this noise floor; detecting it requires many-seed
   means (10 seeds/cell → SE ≈ 0.02).

### PIVOT DECISION

Primary effort moves to **thread (a): the {resample | escalate | abstain} MDP**, whose claims
(cost allocation economics, give-up arm, failure-dumping structure) are stable across all the
seed churn above. A 6-job seed extension (seeds 20–22, both LCB cells) runs as background
due-diligence for the variance measurement, not as a headline claim.


---

## FINAL RECONCILIATION VERDICT AT n=10 SEEDS: Scout Evidence Helps on LCB, Decisively (2026-08-24)

The 10-seed sweep supersedes BOTH the original single-seed claims and the three-seed
"reversal." Full LCB temporal results (oss120-success prediction, converged recipe lr 1e-4):

| Cell | n | mean AUC | sd | min–max |
|------|---|----------|----|---------|
| input-only | 10 | 0.784 | ±0.067 | 0.642–0.858 |
| post-scout + test-fb | 10 | **0.900** | **±0.017** | 0.869–0.927 |

- Delta +0.116 AUC; Welch t=5.36, p<0.001; all 10 per-seed differences positive.
- **Variance-stabilization finding**: input-only training is wildly seed-unstable (collapses to
  0.642); scout-grounded training is near-deterministic (sd 0.017). Attempt content does not just
  add signal — it anchors optimization. To our knowledge unstated in the routing literature.
- SWE remains unresolved at current evidence: seed17 favored input-only, seed18 split by metric;
  needs its own multi-seed sweep before any claim.

### Methodology lessons (paper-ready as-is)

1. Single-run LoRA comparisons on ~500-example datasets have ~±0.07 AUC run noise: both the
   original gate's numbers and every intermediate conclusion in this document (including two
   reversals by the author within one day) illustrate it. Multi-seed reporting is not optional.
2. Undertrained-baseline artifacts can create, destroy, or invert apparent feature contributions
   (lr 2e-5 vs 1e-4 moved input-only by +0.35 AUC).
3. Recommended protocol adopted going forward: ≥10 seeds, mean±sd, Welch tests, lr sensitivity
   reported.

### Standing status of threads

- Thread (b): alive again on LCB — "scout before you route" holds at n=10 with a large,
  low-variance margin over problem-only routing. SWE side pending multi-seed.
- Thread (a): unchanged, data collection nearly complete.
- Thread (c): shrinks back to the introspection-fails trilogy + methodology lessons; the
  cross-domain negative-result framing is off the table unless the SWE sweep also comes back null.

---

## FINAL TWO-DOMAIN VERDICT AT n=10 SEEDS PER DOMAIN (2026-08-24)

SWE multi-seed sweep complete (`swe_smith_instruct_to_verified_transfer_route3_nocot{,_input_only}_10epoch_seed{17..26}_*`, lr 1e-4):

| Metric | input-only | post-scout | delta | Welch p |
|--------|-----------|------------|-------|---------|
| SWE in-domain | 0.720 ± 0.021 | 0.717 ± 0.020 | −0.002 | 0.80 |
| SWE → Verified transfer | 0.606 ± 0.015 | 0.625 ± 0.048 | +0.019 | 0.26 |

### The paper's central empirical contrast (both domains, matched n=10 protocol)

| | LCB | SWE |
|---|---|---|
| input-only | 0.784 ± 0.067 | 0.720 ± 0.021 |
| post-scout + evidence | 0.900 ± 0.017 | 0.717 ± 0.020 |
| scout delta | **+0.116 (p<0.001, 10/10 seeds)** | ≈0 (n.s.) |
| variance effect of scout content | stabilizes (sd ÷4) | slightly destabilizes |

**Headline claim**: attempt-based difficulty signals decisively improve stronger-model success
prediction for short-form code generation (LCB) — large effect, near-zero run variance — while
adding nothing reliable for agentic software repair (SWE), where neither problem text nor scout
attempts predict oss120 success well (absolute AUCs ~0.72/~0.72 in-domain). The boundary is
replicated across seeds with a matched protocol on both domains.

Note the interesting asymmetry within SWE: transfer point-estimate favors scout (+1.9pp) while
in-domain does not, and post-scout transfer variance triples — consistent with scout content
adding noise rather than signal in this regime.

---

## SESSION LOG & THREAD-(a) BUILD STATUS (2026-08-24)

### Completed this cycle

1. **Two-domain n=10 verdicts** (sections above): scout evidence decisively helps LCB
   (+0.116 AUC, variance-stabilizing), does nothing on SWE. Central empirical table done.
2. **MDP tensors built** (`mdp_tensors_v1`): 341 problems x {scout(T0.2), oss20, oss120} x 10
   draws; solve rates 28.4/41.4/65.1 (full), 43.7/44.2/70.5 (public). Draw-variation check:
   T=0.2 yields mixed outcomes on 52% of oss20 instances — temperature validated.
3. **Verifier reliability measured**: false-accept rate of public tests = 35.1% (scout),
   6.4% (oss20), 7.7% (oss120). RoR's HumanEval+ verifier had ~1%; our regime breaks their core
   assumption — this is the quantitative motivation for the content arm.
4. **RoR-faithful replay harness built and running** (`replay_mdp_baseline.py`,
   `mdp_replay_v1`): counting policies with public verifier max out ~56% correctness;
   oracle all-tests verifier lifts the same policies to ~74% at half the always-120B cost.
   The deployable-vs-oracle gap (~18pp) is the target quantity for the content policy.
5. **Framing decision**: public/private split is a benchmark artifact; deployed analogue is
   partial-vs-final verification (CI economics) or absent verification (agentic domains).
6. **AWS $/M-token prices recorded** (0.278 / 1.299 / 4.64 / 8.79 / 11.13 for
   4B/oss20/qwen30/gemini-flash/oss120) in `replay_mdp_baseline.py` USD_PER_M_TOKENS.

### In flight

7. **Depth-1 sequential dataset** (`build_mdp_sequential_dataset.py`, `mdp_seq_dataset_v1`):
   3,410 decision-point examples (1,640 cal / 1,770 test); heads = {scout-next, oss20,
   oss120, nothing}; nothing-rate 34.3%.
8. **Sequential policy training** (`train_mdp_sequential_policy.py`, job
   `mdp_sequential_policy_1787591035`): Qwen3-Embedding-8B + LoRA, 4 BCE heads, input =
   problem + most recent scout draw (code+feedback+public-pass bit) + reroll count +
   earlier outcome bits.

### Next steps

9. Wire trained policy into `replay_mdp_baseline.py` as a fourth policy family: per-step
   probabilities drive {resample | upgrade | abstain}, abstain threshold tau tuned on cal split.
10. Evaluate on test half vs RoR ladder (public-verifier counting ~56%, oracle-verifier ~74%,
    cascade ~66%, always-X). Headline: content-conditioned sequential control closes the
    deployable-oracle gap while staying deployable.
11. Optional depth extension (3 recent draws) if v1 shows promise; sensitivity arm (T=0.6
    scout draws) already collected for draw-diversity ablation.
12. Multi-seed the policy training once v1 validates single-seed.
13. Paper assembly order: two-domain verdict table -> methodology lessons -> MDP frontier ->
    RoR positioning -> introspection-fails trilogy as motivation appendix.

---

## MDP 2x2 FACTORIAL RESULTS: {counts,content} x {abstain,no} (2026-08-25)

Replay harness extended to the full factorial (`replay_mdp_baseline.py --content-preds ...`).
Abstention rule: tau swept on calibration half, selecting (retain >= 95% of no-abstain cal
correctness) then minimize spend. Content prior = static per-problem predictions from the
seed-17 embedding router. Weights cost c={1,5,30}, 5 orderings, test half.

| Cell | Correctness | Spend | Spend vs counts |
|------|-------------|-------|-----------------|
| counts (RoR faithful) | 52.1% | 19.7 | — |
| counts + abstain | 51.9% | 16.8 | −15% |
| content (static prior) | **54.9%** | 20.7 | — |
| **content + abstain** | 51.8% | **10.5** | **−49%** |

Mid-budget domination: at B=30, content+abstain reaches 48.9% @ 6.1 vs counts' 48.3% @ 9.3 —
same quality, 34% less spend; Pareto-dominant.

### Reading

1. Content prior alone: +2.6–2.8pp correctness everywhere.
2. Count-based abstention barely works (−15% spend): counting posteriors are built on the lying
   verifier's bits and cannot identify hopeless instances.
3. Content-based abstention halves spend at ≤3pp correctness cost — consistent with the
   failure-dumping analysis (the model knows when nothing will succeed).
4. All cells still below always_oss120 (75.9% @ 85) — as designed; the claim is Pareto dominance
   at constrained budgets, not absolute records. RoR's own headline is likewise cost-quality.

### Claim wording for the paper

"Content-conditioned beliefs improve per-instance success prediction (+0.12 AUC static); the
calibrated give-up arm converts this into ~50% inference-spend reduction at <=3pp correctness
cost; jointly they Pareto-dominate the training-free counting baseline at every budget."

### Pending on top of this grid

- Sequential policy row (training run `mdp_sequential_policy_1787670814`) — history-conditioned,
  should improve on the static-prior rows.
- Multi-router-seed propagation: content priors above come from ONE router seed; sweep all 10
  router seeds through the replay for error bars.
- T=0.6 scout sensitivity arm.
- Oracle-verifier column for the content cells (how close does deployable content get to the
  perfect-verifier bound?).

---

## PAPER FIGURES (2026-08-25)

Generated by `pipelinerl/swe/scripts/livecodebench/generate_mdp_paper_figures.py`
(regenerate with one command; data sources are the committed replay JSONs and transcribed
n=10 tables):

| Figure | File | Shows |
|--------|------|-------|
| MDP frontier (2x2) | `analysis/mdp_paper_figures/fig_mdp_frontier_2x2.png` | correctness-vs-spend for {counts, content} x {abstain} + fixed baselines; content+abstain Pareto-dominates |
| Verifier regime gap | `analysis/mdp_paper_figures/fig_verifier_regime_gap.png` | same RoR-UCB algorithm under public vs all-tests verifier — the ~18pp false-accept collapse the content policy targets |
| Two-domain verdict | `analysis/mdp_paper_figures/fig_two_domain_verdict.png` | n=10 scout-delta bars with sd error bars: +11.6pp LCB, ~0 SWE |
| Variance stabilization | `analysis/mdp_paper_figures/fig_seed_variance_stabilization.png` | 10-seed box/scatter: input-only ±0.067 with catastrophic runs vs post-scout ±0.017 |

Pending additions once `mdp_sequential_policy_*` finishes: sequential-policy curve overlaid on
the frontier figure + abstain-threshold operating points.

---

## KEY STRUCTURAL INSIGHT: The MDP Lives Entirely in the Failure Region (2026-08-25)

Both RoR and our replay terminate the episode the moment any draw passes the verifier
(`if submitted_idx is not None: break`). Since LCB's full suite contains the public tests,
a public-fail guarantees full-suite fail — so an episode that exhausts its budget without a
verifier pass resolves with probability exactly zero.

Consequence: **all strategic value of the MDP concentrates in the failure region** — the strip
of episodes where every draw so far has failed verification. While draws pass, nothing is
decided (free wins terminate trivially). Within the failure region, the only decisions are:

1. resample the scout vs escalate to oss20/oss120 (budget allocation across models)
2. continue attempting vs give up (the abstain arm)

This same structural fact surfaced three independent times during development, each initially
looking like a separate issue:

- The static multi-sample "router" framing was degenerate because any-pass instances never
  reach a router (see Multi-Sample section);
- Abstention only has value inside the failure region — outside it, giving up is strictly
  dominated by submitting a passing draw;
- Content prediction of eventual correctness is valuable precisely there too: the verifier
  keeps saying "maybe" (public-passes that die privately) while content evidence can say "no."

Paper framing note: early stopping on verifier success should be stated explicitly as an
assumption shared with RoR, and the frontier comparison between policies should be described
as differing in *how they behave after consecutive verifier failures* — counts-plus-prior is
myopic there (its beliefs decay only via pseudo-counts), while history-conditioned content
policies can read *how* attempts are failing, not just how many. This also cleanly explains
why our counting baseline underperforms even the naive cascade under public-only verification:
in the failure region, its belief signal is weakest exactly where decisions matter most.

### Deployment framing for the verifier regimes (restating from discussion)

The public/private split is a benchmark-integrity artifact; the deployed analogue of
"routing-time verification" is partial-or-absent verification — cheap CI subsets vs full
suites, or no tests at all (agentic domains). The content policy's job generalizes to
"predict eventual correctness when current verification is incomplete," i.e. will-this-pass-CI
prediction. Regime 2 (oracle verifier) remains a clearly-labeled diagnostic bound.


---

## PROTOCOL V2: FULL EXECUTION + REACHABLE FAILURE STATES (2026-08-25)

The public/private distinction is a benchmark artifact and is no longer the primary real-world
framing. The main LCB protocol now models the first-order deployment distinction: execution is
available. Every attempted program is graded with the complete evaluator; a pass submits and a
failure enters or remains in the allocation MDP. Public-only verification is retained solely as
a weak-verifier sensitivity arm.

### Legacy sequential run disposition

`mdp_sequential_policy_1787670814` succeeded (8 epochs; old test AUCs approximately 0.75--0.86),
but is diagnostic-only and must not be reported as the learned-policy result. Audit found three
violations: its problem text was empty because `mdp_tensors_v1/problems.jsonl` omitted
`problem_statement`; it created depth-0 states before the mandatory scout; and it created later
states even after an earlier verifier pass. It also used public-test feedback with full-suite
targets.

### Corrected immutable artifacts

Artifact root: `mdp_full_execution_v2_1787679948`. The bundle was reconstructed entirely from
saved generations, so no model recollection was needed.

- 341 problems x {scout, oss20, oss120} x 10 valid draws
- canonical problem split: 170 train / 85 calibration / 86 test
- full problem statements restored from the corrected temporal collection
- prompt/completion token counts and aligned draw records retained
- reachable dataset: 6,683 train / 3,275 calibration / 2,901 test decision states
- all states begin after a failed mandatory scout; histories stop on any full-execution pass
- policy text contains explicit per-route failure/remaining counts and only the latest failed
  attempt; complete histories remain provenance metadata rather than model context
- the `nothing` target is positive only when no successful valid draw remains in any route

### Corrected baseline replay (test split, 20 draw orderings)

These are preliminary protocol-validation numbers, not the final learned-policy result:

| Precisely defined policy | Resolve rate | Mean estimated USD | Mean calls |
|---|---:|---:|---:|
| single scout | 34.5% | 0.00059 | 1.00 |
| single oss20 | 46.1% | 0.00382 | 1.00 |
| single oss120 | 66.2% | 0.02910 | 1.00 |
| scout then one oss120 | 67.3% | 0.02559 | 1.65 |
| one-pass scout -> oss20 -> oss120 | 69.1% | 0.02476 | 2.13 |
| best-of-10 oss120 (stop on pass) | 81.4% | 0.13261 | 3.04 |

The count-based adaptive curve reaches 74.1% at mean estimated cost 0.0400; its decision-relevant
conditional resolve rate after scout failure is 60.4% at that point. Costs use realized prompt +
completion tokens multiplied by the recorded per-model token rate; decision-time affordability
uses train-split expected cost, so future completion lengths are not leaked.

Figures in `analysis/mdp_full_execution_v2/` replace the public-verifier frontier for protocol
validation. They currently contain counts and fixed baselines only.

### First corrected learned run disposition

`lcb_mdp_full_execution_seed17_1787680221` completed end to end. Its four held-out AUCs were
approximately 0.79--0.86, but its sequential policy did not improve the cost/accuracy frontier
over the count policy or one-pass cascade. Calibration BCE selected epoch 0 and worsened sharply
thereafter.

Post-run audit found two representation/label issues, so this learned curve is diagnostic rather
than final: complete histories caused 13.2% sampled-state truncation (about 46% at depth 10), and
the old `nothing` head meant only that the next draw from each route would fail. The corrected
builder now presents problem + explicit per-route counts + only the latest failed attempt, and
labels `nothing` only when no successful valid draw remains anywhere. On the saved LCB tensors,
the new representation had zero truncation in a 1,200-state sample (maximum 4,363/8,192 tokens)
and corrected 2,092 labels where all next draws fail but a later draw succeeds. No recollection
was required.

### Remaining before final paper figures

1. Train one inexpensive LCB diagnostic on the rebuilt latest-attempt dataset, using calibration
   checkpoint selection and a less aggressive optimization configuration. The scheduled configuration
   is seed 17, 3 epochs, learning rate 2e-5, and 5 replay orderings, reusing the saved generations.
2. Add per-episode action/prediction traces and paired problem-level bootstrap intervals to replay.
3. Advance to multi-seed LCB training only if the learned policy beats counts or the cascade at
   matched cost; otherwise report the learned-policy result as a negative result.
4. Re-run the LCB counts-only versus latest-failed-attempt comparison strictly conditional on
   full-execution scout failure.
5. Regenerate the two-domain and variance figures only after that gate. Keep SWE-Smith unlaunched
   until the LCB diagnostic passes.

### Prepared SWE-Smith protocol-v2 extension (not launched)

The agentic-domain adapter now consumes real sandbox execution for both routing and final
correctness, uses the matched scout/oss20/oss120 portfolio, and reuses the same reachable-state
trainer/replay. The existing eval150 artifacts contain 143 problems with all nine reports; this
is suitable for development only (72/36/35 internal split). The preferred paper evaluation uses
eval150 for training/calibration and the already-generated, non-overlapping eval300 bundle as an
untouched test set after its nine real sandbox reports per problem are collected. Missing reports
are invalid, never negative labels. See `analysis/swe_smith_mdp_full_execution_v2/README.md`.

---

## LEGACY WEAK-VERIFIER CAPABILITY LADDER (SUPERSEDED; 2026-08-25)

This section records the old public/private-verifier experiment for provenance only. It uses
the invalid legacy state dataset audited above (including empty problem text and unreachable
histories), so neither its learned-policy numbers nor its figure are primary paper results.
Protocol v2 above replaces it; do not compare these values directly with full-execution replay.

Sequential MDP policy trained (`mdp_sequential_policy_1787670814`: 4-head BCE on depth-1
decision points, single seed) and wired into the replay as history-conditioned beliefs
(`mdp_replay_v1_full`, 10 orderings, public verifier). Complete ladder at B=90:

| Cell | Content? | Abstain? | Correctness | Spend |
|------|----------|----------|-------------|-------|
| counts (RoR faithful) | ✗ | ✗ | 52.6% | 19.9 |
| counts + abstain | ✗ | ✓ | 52.5% | 17.0 |
| content (static prior) | ✓ | ✗ | 55.8% | 20.9 |
| content + abstain | ✓ | ✓ | 53.2% | **10.6** |
| **sequential** | **history** | ✗ | **57.0%** | 21.6 |

Every ingredient contributes monotonically:
- static content prior: +3.2pp over counting
- history conditioning: +1.1pp further (+4.4pp total over RoR-faithful)
- abstention trades correctness for spend with strength tracking belief quality
  (counts −15%, static content −49%, sequential n/a — it avoids doomed spends natively,
  so no tau met the retain-95% calibration bar at high budgets)

Mid-budget dominance holds: at B=60, sequential+abstain 52.5% @ 13.3 vs counts 49.4% @ 15.0.

Policy test AUCs at final epoch: scout_next 0.844, oss20_fresh 0.750, oss120_fresh 0.765,
nothing 0.780 — stable from epoch 1 onward.

### Caveats

Single policy-training seed; single router seed for the static-prior rows; orderings=10.
Multi-seed propagation through both trained components is the remaining robustness step before
these become headline numbers.

### Paper figure

`analysis/mdp_paper_figures/fig_mdp_frontier_2x2.png` shows this legacy weak-verifier ladder
(six curves + fixed baselines). It is retained for provenance and is not a main result figure.
