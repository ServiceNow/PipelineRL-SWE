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
| **LCB** | Historical binary-prediction line; useful as signal evidence, but superseded as the primary deployment protocol | Protocol-v2 provisional result: count-decayed learned routing plus abstention adds useful full-execution Pareto points; replicate on a larger untouched temporal test set |

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

### Latest-attempt gated rerun and decision diagnostics

`lcb_mdp_latest_attempt_seed17_1787808515` completed end to end (seed 17, 3 epochs,
learning rate 2e-5, 5 replay orderings). Calibration selected epoch 1. Test AUCs were
0.693 scout-next, 0.834 oss20-next, 0.852 oss120-next, and 0.862 nothing-remaining.
Despite this predictive signal, the learned policy did not improve the frontier:

- at budget 0.00861, learned minus counts correctness was -9.77 points, paired
  problem-bootstrap 95% CI [-16.05, -3.95];
- at budgets 0.06501 and 0.07306, the deltas were -5.12 points
  (CI [-10.00, -0.93]) and -5.81 points (CI [-10.47, -1.86]);
- at the highest budgets both reached 74.65%, but learned cost was 0.04220 versus
  0.04164 for counts;
- nearest the cascade's cost, learned was -2.56 points (CI [-8.14, +2.56]) and used
  7.17 more attempts/problem-ordering (CI [+5.78, +8.61]).

The replay-only diagnostic artifact is
`/mnt/llmd/results/exps/aristides/reason/lcb_mdp_latest_attempt_seed17_1787808515/replay_diagnostics_v1`.
It contains 20,640 adaptive and 3,440 fixed per-episode rows, action/probability traces, and
5,000-draw paired bootstraps clustered over the 86 test problems.

The failure mechanism is structural. Across 2,498 unique learned states, mean predicted scout
success changed only from 0.176 after one scout failure to 0.163 after nine. Its mean
probability/cost score therefore stayed around 285--319, versus roughly 53--58 for oss20.
The learned policy exhausted scout, then oss20, and delayed oss120: at budget 0.06501 counts
made 144 oss120 choices while learned made zero. The learned `p(any remaining)` never fell
below 0.252, so the calibration-selected abstention threshold 0.05 could never fire.

### Count-decayed learned hybrid replay (provisional positive)

> **Mechanism correction (2026-08-27).** The framing below -- "structural Bayesian update plus a
> learned semantic residual" -- is not what the decay is doing. The calibration audit later in this
> file shows the learned scout probability is **9.28x overconfident**, and the `2/(2+failures)`
> factor shrinks it ~3.5x by depth 5, cutting ECE from 0.1341 to 0.0382. The decay is functionally a
> **rare-event recalibrator**. The empirical result stands; the explanation does not. Do not write
> the paper around the structural-prior story until recalibration has separated the two effects.

The predeclared no-retraining hybrid multiplies each learned route-success probability by the
same failure-count prior used by the count policy, `pseudo_count / (pseudo_count + failures)`
with pseudo-count 2. This retains RoR's structural update after failures while allowing the
model's problem- and attempt-specific prediction to act as a residual. Its abstention probability
is recomputed from these decayed route probabilities.

The replay artifact is
`/mnt/llmd/results/exps/aristides/reason/lcb_mdp_latest_attempt_seed17_1787808515/replay_hybrid_v1`.
It uses the same test split and saved predictions, five draw orderings, and 5,000 paired bootstrap
draws clustered over the 86 test problems. The hybrid passes the provisional decision gate:

- at budget 0.07306, the non-abstaining hybrid exactly matched counts at 74.42% correctness while
  reducing mean cost from 0.04008 to 0.03669 (8.45%); paired cost delta 95% CI
  [-0.00639, -0.00077], with P(delta > 0) = 0.0032;
- at the high-budget abstaining point, the hybrid reached 74.19% correctness at 0.02870 cost,
  versus 74.42% at 0.04008 for the neighboring count point. The correctness difference was
  -0.23 points, CI [-1.16, +0.47], while cost fell by 28.4%, CI [-0.01841, -0.00517];
- at the middle abstaining point, the hybrid reached 73.95% at 0.02820 versus counts' 73.72% at
  0.03690. The correctness difference was +0.23 points, CI [-0.47, +1.16], while cost fell by
  23.6%, CI [-0.01475, -0.00319];
- the hybrid remains materially worse at the smallest budget (47.91% versus 57.44%), so it is an
  addition to the Pareto frontier rather than a uniformly dominant replacement for counts.

#### How much of the gain comes from abstention?

At the useful high-budget points, the hybrid abstains on 26--27% of episodes that enter the router
after mandatory scout failure. Since 65.1% of all episodes enter the router, this is 17--18% of all
episodes. The same-budget decomposition is:

| Budget | Policy | Correctness | Mean realized cost |
|---:|---|---:|---:|
| 0.07306 | counts, no abstention | 74.42% | 0.04008 |
| 0.07306 | hybrid, no abstention | 74.42% | 0.03669 |
| 0.07306 | hybrid + abstention | 73.95% | 0.02820 |
| 0.08112 | counts, no abstention | 74.65% | 0.04164 |
| 0.08112 | hybrid + abstention | 74.19% | 0.02870 |

Thus the 8.45% result isolates better route selection at identical observed correctness, while the
joint routing-plus-abstention policy saves 29.6--31.1% relative to counts without abstention for an
observed 0.47-point correctness reduction. That reduction is statistically indistinguishable from
zero in the present paired sample. Count abstention alone is weaker: at budget 0.07306 it reaches
73.72% at cost 0.03690, versus the hybrid's 73.95% at 0.02820. The learned component's clearest
current contribution is therefore identifying hopeless residual cases, while count decay supplies
the reliable sequential route update.

The apparently tiny scout rates are conditional rates in the hard tail after mandatory scout
failure, not the marginal scout capability: scout solves 28.4% of all saved draws and the
mandatory-scout replay solves 34.9% of test episodes. Among held-out reachable states, empirical
next-scout success drops from 3.15% after one scout failure to 0.81%, 0.19%, and 0.30% after two,
three, and four failures. The model substantially overpredicts these values (15.6%, 12.8%, 11.7%,
and 11.4%), explaining why pure learned routing keeps drawing scout. The hybrid corrects this
calibration/structural failure without retraining.

This is a one-model-seed, five-ordering result and is therefore provisional, not yet a headline
paper claim. It supports the factorized policy design: count decay supplies the reliable
sequential update, and learned semantic predictions supply the residual routing signal.

#### Paper value relative to the earlier binary-abstention result

If replicated, this is a stronger primary result than the earlier pure stronger-model-success
prediction/AUC line. It is closer to deployment, evaluates the complete correctness--cost outcome,
chooses repeatedly among three model tiers plus give-up, and exposes a general design principle:
pure learned probabilities can fail under repeated evidence, whereas a structural count update
combined with a learned semantic residual can route and stop effectively. The pure-abstention
experiments remain valuable motivation and representation evidence, but binary AUC alone does not
show that the predictor improves an end-to-end allocation policy.

The current evidence strength is nevertheless lower than the conceptual strength: there is one
trained seed and only 86 independent test problems. Five replay orderings produce 430 correlated
episodes, not 430 independent problems; one test problem moves observed accuracy by 1.16 points,
so this split cannot resolve a 0.47-point accuracy difference. Because the current test frontier has
now informed policy development, treat it as development evidence and reserve a newly collected,
chronologically later problem set for a single locked confirmation.

### Marginal-value stopping does not produce early abstention (2026-08-27, development-only)

Motivated by the observation that `_abstain` policies quit only after 12--20 failures, the replay
gained a `_value_stop` variant replacing the probability threshold `p_any <= tau` with a
marginal-value rule `max_m p_m / cost_m <= T`, with `T` selected on calibration under the same
retained-correctness constraint. Artifact: `replay_early_stop_dev_v1/` under
`lcb_mdp_latest_attempt_seed17_1787808515`. It does not do what it was built for.

- **0% of `_value_stop` abstentions occur while the scout still has draws left**, at both audited
  budgets, versus **96.2%** for the `tau` rule at budget 0.01667. Its abstention depth is set by
  scout exhaustion (10 draws), not by a stopping decision.
- **The route statistics do not overlap.** On the unconstrained decay policy at budget 0.07306,
  scout's `p/cost` while available has minimum **24.03** across 2,373 decisions, while oss120's
  `p/cost` has maximum **20.58** across 3,752 decisions. Any `T` that stops a single scout re-roll
  therefore forbids **100%** of oss120 calls and 32.9% of oss20 calls. This is an expressiveness
  limit of the rule class, not a tuning failure: no threshold expresses "stop re-rolling the cheap
  route but escalate to the expensive one." Route costs span 53x.
- **The defect is the action rule, not the stopping rule.** `max_m p_m/cost_m <= T` is *exactly
  equivalent* to `max_m (p_m*R - cost_m) <= 0` for `R = 1/T`, so `_value_stop` already uses the
  myopic-optimal stopping criterion; re-parameterizing it changes nothing. The failure is that
  `argmax_m p_m/cost_m` is scale-invariant and hence **independent of `R`** -- it prefers the
  cheapest route at every valuation, so it grinds the scout to exhaustion however `T` is set. One
  `T` is then asked to both permit oss120 (`T <= 20.58`) and stop scout (`T > 24.03`) along a
  trajectory it cannot reshape. `tau` does better at early stopping because `p_any` is cost-blind,
  encoding "hopeless problem" -- a crude proxy for small `R`.
- **But `_value_stop` is not globally worse and must not be written up as a failed policy.** It
  contributes 15 Pareto points versus 12 for `tau`, and `sequential_decay_value_stop` owns the
  entire mid-cost band (0.01038--0.01714 at 61.9--67.4% correctness) where no `tau` policy appears,
  reaching 67.44% at 0.01714 against `sequential_decay_abstain`'s 68.37% at 0.02084. It is
  dominated only at the headline point: 74.19% at 0.03072 versus `_abstain`'s 74.19% at 0.02870.

Reading: `_value_stop` earns its frontier points for a different reason than intended -- "run the
cheap route to exhaustion, then quit before escalating" -- so **early abstention remains unsolved**.
All numbers above are development-only on the 86-problem test split and are not confirmation
results.

**The prescribed fix is to select actions by value difference and sweep the valuation.** Replace
`argmax_m p_m/cost_m` with `argmax_m (p_m*R - cost_m)`, stop when that maximum is `<= 0`, and
generate the frontier by sweeping the single scalar `R` (the Lagrange multiplier on correctness)
rather than by a budget grid crossed with a separately calibrated `tau`/`T`. Measured over all
3,799 decisions at budget 0.07306, the ratio rule's route mix is constant at scout 58% / oss20 39%
/ oss120 3% for *every* `R`, while the value rule sweeps scout 38% (R=0.02) to oss120 99% (R=1.00)
as its stopping rate falls from 54.2% to 0%; the two rules disagree on 28.4%--95.9% of decisions.
This removes the meaningless "retain 95% of calibration correctness" threshold-selection heuristic
and makes escalation and stopping share one knob. Finite-horizon planning
(`Q_m = p_m*R - cost_m + (1-p_m)*V(s')`) stays on the list but is **demoted and re-labelled**: since
`V >= 0`, the continuation term only makes continuing more attractive, so it pushes abstention
later, not earlier. It corrects a different error and should follow the `R`-sweep, not precede it.

### RoR is the primal; our utility formulation is its Lagrangian dual (2026-08-27)

Checked against the paper itself (arXiv 2607.08665, *"Resample or Reroute? Budget-Aware Test-Time
Model Selection for Large Language Models"*) rather than second-hand notes. RoR maximizes expected
correctness **subject to a per-query cost budget**, allocating by *estimated marginal correctness
per unit cost*, with beliefs from empirical success counts over an eleven-model pool. Confirmed: no
abstain/give-up action; budget exhaustion is the only stopping condition. Correction to an earlier
note in this file: the "T=0.2" in the RoR-parity protocol is the **sampling temperature**, not a
decision threshold.

This means two things we had wrong.

1. **The budget cap `B` is RoR's formulation, not incidental machinery.** Deleting it would turn
   the primary baseline into a strawman. The budget-swept `counts` family (greedy `p/cost`, no
   give-up arm) is the RoR-faithful cell and stays on by default.
2. **The `argmax_m p_m/cost_m` ratio rule is not a defect.** Under a hard budget it is the standard
   greedy knapsack rule, i.e. the correct primal solution. The earlier entry above framing it as
   the bug is withdrawn; what it lacks is a stopping theory, not a correct ordering.

Our formulation is the **Lagrangian dual** of the same problem. Maximizing expected correctness
subject to `sum(cost) <= B` relaxes to maximizing `sum(p) - lambda*sum(cost)`, i.e. acting on
`p_m*R - cost_m` with `R = 1/lambda`. That is the contribution statement:

| | RoR (primal) | ours (dual) |
|---|---|---|
| knob | per-query budget `B` | value of a correct answer `R` |
| action rule | `argmax p/cost` (greedy knapsack) | `argmax (p*R - cost)` |
| stopping | budget exhaustion only | `max_m (p_m*R - cost_m) <= 0` |
| abstention | none | falls out as the zero-utility action |

What the dual buys, stated as claims we must defend:

- **No arbitrary per-query cap.** `B` is a per-episode allowance applied identically to every
  problem regardless of promise, which works against the adaptive allocation the paper argues for.
  `R` spends more on tractable problems and nothing on hopeless ones by construction.
- **Abstention is derived, not tuned.** This removes the `tau` grid and the "retain 95% of
  calibration correctness" selection heuristic, which had no decision-theoretic justification.
  It directly addresses the give-up arm listed above as the open RoR delta.
- **One interpretable knob.** `R` is dollars per correct answer -- a deployment parameter a reader
  can reason about, unlike `tau = 0.2`.

Honest caveats to carry:

- Sweeping `R` is a linear scalarization, so it recovers only the **convex hull** of the
  correctness--cost frontier; a budget sweep can reach non-convex points that no `R` attains.
  Mixing two policies convexifies, but the limitation should be stated.
- The dual is only as good as `p`. RoR's count-based beliefs are empirically grounded per episode;
  ours are learned and **9.28x overconfident on the cheap route** (see calibration entry). The
  formulation advantage can be swamped entirely by belief quality, so calibration is a
  precondition for this claim, not a refinement of it.

### Free-start protocol is currently vacuous (2026-08-27, development-only)

`replay_free_start_dev_v1/` removes the mandatory scout, exposing {abstain, scout, oss20, oss120} at
the root across all 341 problems including those a mandatory scout would have solved; the builder
gained a matching `--start-protocol free_start` mode (14,844 reachable examples, all problems at
depth 0). **104 of 108 (budget, policy) cells are bit-identical to scout-first.** The four
differences sit at the degenerate smallest budget 0.00055, where root abstention becomes affordable
(34.42% at 0.00038--0.00045 versus 34.88% at 0.00058).

Same root cause: **every policy picks scout at the root 100% of the time at every budget above the
smallest**, because `prior/cost` is 432.8 (scout) versus 96.0 (oss20) and 21.5 (oss120). The
free-start policy re-imposes the mandatory scout on itself. This is not an out-of-distribution
artifact from the model never seeing root states -- the `counts` family uses train priors, not the
model, and behaves identically. The ablation therefore cannot yet answer "should the scout be called
at all?", because it is evaluated under a rule structurally incapable of declining. It is blocked on
the same action-rule fix and must be re-run after it. Treat these numbers as a protocol check, not
a result.

### Calibration audit: the binding constraint (2026-08-27)

Measured on `model/test_predictions.jsonl` -- the reachable-dataset test split (2,901 states, 61
problems, depths 1--10), which has balanced route/depth coverage and no policy-induced selection
bias, unlike replay traces.

| head | base rate | mean pred | pred/actual | ECE | AUC |
|---|---:|---:|---:|---:|---:|
| `nothing` | 0.552 | 0.597 | 1.08x | 0.178 | 0.862 |
| `oss120_fresh` | 0.270 | 0.263 | **0.97x** | 0.107 | 0.852 |
| `oss20_fresh` | 0.090 | 0.180 | 2.00x | 0.103 | 0.834 |
| `scout_next` | 0.015 | 0.134 | **9.28x** | 0.120 | 0.693 |

Four findings.

1. **Good ranking, bad probabilities.** AUCs 0.69--0.86 match the training log exactly. This is not
   an undertrained model; it is a calibration failure.
2. **Miscalibration is monotone in base rate**, and base rate is inversely ordered with cost. The
   cheapest route is the rarest event and therefore the most inflated, so **miscalibration
   systematically manufactures the cheap-route preference** that drives all the scout-grinding
   pathologies recorded above. This is an artifact, not a property of the problem.
3. **It is overfitting, not undertraining.** Train loss falls monotonically (0.566 / 0.366 / 0.277)
   while calibration loss bottoms at epoch 1 (0.448 / 0.396 / 0.415), and *every* calibration AUC
   peaks at **epoch 0** and declines thereafter. More epochs strictly hurt. Data-limited at 141
   training problems, not compute-limited. Minor: the checkpoint is selected on calibration loss
   (epoch 1) while AUC peaks at epoch 0 -- if recalibration handles probabilities, select on AUC.
4. **Not distribution shift.** Train/test base rates agree within 6% (scout 0.0148 vs 0.0145) with
   identical depth composition. The model predicts 0.134 for scout when even its *own training*
   base rate is 0.0148, so it is not regressing to the training marginal either. Classic
   extreme-imbalance failure of a 1.5%-positive head.

Depth structure the model misses entirely. Empirical scout hazard by failure depth is
`0.041, 0.025, 0.013, 0.011, 0.008, 0, 0, 0, 0, 0` -- **exactly zero positives across 1,097 states
at depths 6--10** -- while the model predicts ~0.11 throughout. Conversely at depth 1 it
*underestimates* oss120 (0.337 predicted vs 0.541 actual). At the single most important decision the
relative scout-vs-oss120 comparison is distorted by roughly 10x.

This retroactively explains the count-decay hybrid: multiplying by `2/(2+failures)` shrinks scout's
inflated number ~3.5x by depth 5, cutting ECE from 0.1341 to 0.0382 on policy traces. **The
"structural Bayesian prior" is functionally a rare-event recalibrator.** The mechanism described in
the hybrid entry above is wrong and should be restated.

Consequence: calibration is a **precondition**, not a refinement. Every downstream comparison
(utility formulation, continuation value, planning) is evaluated on probabilities that are 9x wrong
on the cheap route, so a negative result would be unattributable. Fix order: post-hoc per-route,
depth-conditioned recalibration fit on calibration only (free, replay-only); then `pos_weight`/focal
loss on the rare heads at the next retrain; then more unique problems. Not more epochs.

### Headroom and baselines: what routing can and cannot buy (2026-08-27)

Oracle analysis on the 86-problem replay test split, from the saved tensors.

| solvable within 10 draws by | share |
|---|---:|
| scout | 47.7% |
| oss20 | 67.4% |
| oss120 | 81.4% |
| **any route** | **81.4%** |
| none (hopeless) | 18.6% |

**oss120's coverage exactly equals any-route coverage.** Scout and oss20 contribute *zero*
exclusive coverage -- every problem they solve, oss120 also solves. There is **no capability
complementarity** in this pool: routing can only ever buy money, never accuracy. The paper cannot
claim model specialization, and must answer "why not just use the big model?" with the cost figure.
The 18.6% hopeless share is the theoretical ceiling on what abstention can save.

Fixed baselines (first-valid-draw semantics, USD):

| policy | correctness | cost |
|---|---:|---:|
| `single_scout` | 34.88% | $0.00058 |
| `single_oss20` | 46.74% | $0.00379 |
| `best_of_10_scout` | 47.67% | $0.00493 |
| `scout_then_oss20` | 53.95% | $0.00379 |
| `single_oss120` | 66.98% | $0.02900 |
| `scout_then_oss120` | 68.37% | $0.02541 |
| `single_pass_cascade` | 70.00% | $0.02421 |
| `best_of_10_oss120` | **81.40%** | $0.13376 |
| `sequential_decay_abstain` (ours) | 73.95% | $0.02820 |
| `counts` (best adaptive) | 74.65% | $0.04164 |

The adaptive policy strictly dominates `single_oss120` and captures 91% of achievable accuracy at
21% of the cost of `best_of_10_oss120`. That is the honest headline framing.

### Re-rolling the large model is the largest untouched lever (2026-08-27)

`single_oss120` 66.98% -> `best_of_10_oss120` 81.40% is **+14.42 points** from re-rolling the
expensive model alone, and it reaches the 81.4% ceiling. Our policies never use it:

| policy | mean oss120 calls/episode | distribution |
|---|---:|---|
| `counts` | 0.33 | 0x: 67%, 1x: 33% |
| `sequential_decay_abstain` | 0.19 | 0x: 81%, 1x: 19% |

**Never twice, in any episode.** The cause is structural: the budget grid is
`linspace(min_cost, 3*max_cost, 12)`, topping out at $0.089 -- three oss120 calls -- while
`best_of_10_oss120` costs $0.13376. **The experiment's cost ceiling sits below the region where the
highest-value strategy lives**, so we have never tested it. The value sweep has no ceiling and fixes
this automatically (its cap is $1.342 against a maximum possible spend of $0.342, so it cannot
bind). **Fixed 2026-08-27:** the budget grid now runs to full-exhaustion spend `sum(expected_cost *
K)` = $0.34241 in 17 points, preserving the original 12 so recorded numbers stay comparable and
adding $0.140/$0.191/$0.241/$0.292/$0.342. Both formulations now reach the same ceiling; capping the
RoR cell at $0.089 while sweeping ours to $0.342 would have made the baseline a strawman at the high
end.

Note for positioning: re-rolling large models differentiates us from *cascades*
(`single_pass_cascade` is one shot per tier and structurally cannot), but **not from RoR**, whose
whole premise is resample-vs-reroute. Consistent with the standing instruction above not to claim
first formulation of resample-vs-escalate.

### Data pool audit: 892 problems collected, 341 used (2026-08-27)

The source collection (`lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448`, pinned
`release_v6`) already holds **892 problems: 551 train + 341 eval**, temporally split. The MDP
experiment uses **only the 341 eval problems**, which it then re-splits 170/85/86.

Worse, that re-split is **random, not temporal**: `split_manifest.json` records
`{"method": "sorted_ids_numpy_permutation", "seed": 0}`. The chronological rigor the plan claims is
not currently implemented.

Both are fixed by one collection job: run oss20/oss120 multidraw on the 551 already-collected
problems. No pool decisions, no new dataset design, purely generation + Daytona execution.

- Unique problems: 341 -> 892 (2.6x).
- Enables a genuine temporal design: train on the earlier 551, hold out the later 341 as a
  chronological test set touched once.
- Test problems 86 -> 341, so one problem moves accuracy **0.29pt instead of 1.16pt** -- which is
  what makes the ~0.5pt effects we keep measuring resolvable at all.

Collection design, from the hazard data: **asymmetric draws.** Scout draws 6--10 are provably dead
(0 successes in 1,097 states), while oss120 re-rolls are worth +14.42 points. Cut scout to ~4 draws
and keep oss120 at 10. This is the long pole (multi-day) and has zero dependency on any replay-side
work, so it launches first and everything else proceeds in parallel.

### Planned: structural continuation value, and the experiment grid (2026-08-27)

The value rule as implemented is **myopic**: `Q_m = p_m*R - cost_m`, dropping `(1-p_m)*V(s')`. On
real priors at R=$0.25 that flips the first decision -- myopic picks oss120 (0.130), one-step picks
scout (0.0592 + 0.761*0.130 = 0.158) -- so the myopic dual **over-escalates**. Note also that since
`V >= 0`, a continuation term pushes abstention *later*, not earlier; it corrects a different error
than the late-abstention pathology.

Failure is deterministic in this MDP (a failed route leads to exactly one successor), so lookahead
branches only on route choice: ~3^N model calls per decision for depth N. Full DP over the
(counts, last_route) lattice is ~1.7M forward passes.

**Preferred design -- structural continuation, semantic decision point:**

```
Q_m = p_m^model(s)*R - cost_m + (1 - p_m^model(s)) * V_struct(s'_m)
```

- `p^model`: instance-specific, evaluated only at the current state -- **one forward pass per
  decision, same cost as today**.
- `V_struct`: full-horizon continuation from *empirical per-route, per-depth hazards*, solved by
  backward induction over the 1,331-state count lattice **once, offline, in milliseconds**.

This gives the exact horizon with no truncation parameter, and encodes the structural collapse the
model completely misses (scout hazard is literally 0 from depth 6). It is the same
structure-plus-semantics factorization the count-decay hybrid stumbled onto, applied to the *value
function* rather than as a multiplicative fudge on the probability -- a principled version of the
thing that already worked. **Hazards must be fit on train only**; fitting them on the evaluation
split would reintroduce exactly the threshold-on-test problem the dual formulation removed.

Depth-1/depth-2 model-based lookahead is then an **ablation, not the method**: its job is to test
whether semantic continuation adds anything over structural. Full DP only if that says yes.

**Experiment grid.** The value sweep gives a clean 2x2 isolating formulation from belief source,
which is what a reviewer needs to see before the dual can be claimed as the contribution:

| | primal (budget + `p/cost`) | dual (`R` + `p*R - cost`) |
|---|---|---|
| count beliefs | `counts` <- RoR-faithful | `counts_value` |
| learned beliefs | `sequential_decay` | `sequential_decay_value` |

All cross-formulation comparisons are **cost-matched with problem-clustered paired bootstrap CIs**,
matching by nearest mean realized cost (as the existing cascade comparison already does). The
headline test is RoR-primal `counts` against `sequential_decay_value`. This pairing is not yet in
`paired_comparisons` -- the two sweeps share no knob -- but both write full `episode_traces.jsonl`,
so it is computable post hoc without re-running.

### RoR read from the paper: what they do, and where our regime sits (2026-08-27)

Read from the PDF, not second-hand. Supersedes guesses elsewhere in this file.

**Belief update (their Eq. 2).** For model `m` on query `i`, after `n_im` draws of which `w_im`
verified correct:

```
p_hat_im = (s * p_bar_m + w_im) / (s + n_im)
```

`p_bar_m` is "model m's offline accuracy, **calibrated once on the train split**" -- a *scalar per
model*. `s` is the pseudo-count. **RoR never conditions on the problem text.** The only per-query
signal is the observed pass/fail counts inside that query.

Two consequences:

- **Our `counts` baseline is exactly RoR.** `pseudo_count * prior[m] / (pseudo_count + failures[m])`
  equals Eq. 2 when `w = 0`, which always holds in the failure region because a success terminates.
  The reproduction is faithful.
- **RoR calibrates its prior on train; we do not calibrate ours.** The current comparison is
  therefore rigged *in RoR's favour*. Calibrating our beliefs is a fairness requirement, not an
  optimization.

**Their Table I positioning** (reproduced) draws our gap for us:

| family | chooses among models | repeated draws | per-query budget | imperfect verifier |
|---|:-:|:-:|:-:|:-:|
| Learned routers | yes | -- | -- | -- |
| Cascades | yes (sequential) | -- | cost gate | confidence gate |
| Best-of-N / self-consistency | -- | yes | fixed N | selector needed |
| Reject option | abstain/accept | -- | reject cost | confidence |
| **RoR** | yes | yes | yes | yes (parametric) |

Query-conditioning sits in the *learned routers* row (no draws, no budget); abstention sits in the
*reject option* row (no draws, no budget). Neither is in the RoR row. Our cell is the union.

**Their pool: 11 open-weight models across eight pretraining lineages** -- Mistral; Qwen 2.5 at
7B/14B/32B plus a Qwen-based DeepSeek-R1 distill; Phi-4; OLMo-2; Yi-1.5; Granite-3.3; Gemma-2;
Llama-3.1. Note this is **cross-lineage**, not a size ladder. Ours (Qwen3-4B, gpt-oss-20b,
gpt-oss-120b) is essentially two lineages and mostly a capability ladder, which is the likely source
of our zero-complementarity finding.

**Their per-benchmark regimes (Table V, matched mid budget, parameter-size cost proxy):**

| benchmark | regime | RoR | best baseline | note |
|---|---|---:|---:|---|
| GSM8K | saturated | 0.993 @ 9.6 | 0.992 cascade @ 12.3 | margin is cost only |
| MATH-500 | intermediate | 0.877 @ 26.5 | 0.850 best-of-K | +2.7 pts |
| **GPQA-Diamond** | **hard, heterogeneous** | **0.892 @ 26.6** | 0.644 best-of-K | **+24.8 pts** |
| HumanEval+ | code | 0.962 @ 21.5 | 0.952 cascade | undirected baselines converge |

**This is the finding that matters most for us.** GPQA-Diamond is where "the pool's specialists
genuinely differ" and rerouting pays (+24.8). On **code**, their own honest caveat states: with a
near-perfect execution verifier and a high union ceiling (0.988 oracle), "the two undirected
budget-scalable baselines converge toward RoR" -- cascade 0.952, random 0.952--0.960 -- "because
when almost every query is solvable by *some* draw and verification is reliable, spreading" budget
works nearly as well as directing it. And on saturated benchmarks they state plainly that "the
advantage comes from **stopping early on cheap models** rather than from finding a specialist."

So: **we are working in the regime their own paper flags as least favourable to cross-model
selection.** Our LCB setting is full-execution verified with an 81.4% union ceiling and zero
specialist advantage -- the code/saturated regime, not the GPQA regime. That explains the
difficulty, and it also dictates the framing.

**Precision on "stopping" -- two different mechanisms, do not conflate them.** RoR's Algorithm 1 has
exactly two loop exits: a verifier-confirmed correct draw (line 9, which they label "early
stopping"), or budget exhaustion (line 3). On failure, line 13 still *returns the best candidate* --
RoR never refuses to answer. So their "stopping early on cheap models" means **terminating on a
win** obtained cheaply, not giving up.

| stop when | RoR | ours |
|---|:-:|:-:|
| you have won (success termination) | yes (line 9) | yes |
| you cannot win (give-up) | **no** | yes |

The two cover **disjoint** problem sets: success termination avoids overspending on the 81.4% that
are solvable; abstention avoids spending on the 18.6% that are not. Our contribution is therefore
*orthogonal* to their mechanism, not a better version of it.

**Reframing (recommended).** Do not claim specialist routing gains; the pool cannot deliver them and
we should say so. Claim instead: *in the verified-code regime where selection gains provably vanish,
what remains is cost efficiency -- RoR captures the solvable half by terminating on success, and the
unsolvable half is untouched because the budget-constrained ratio formulation cannot express
giving up.* Our proof that the ratio statistic
admits no viable threshold (scout floor 24.03 vs oss120 ceiling 20.58, disjoint) and that the
Lagrangian dual yields stopping for free at `Q <= 0` is then directly on-target rather than
incidental. Our zero-complementarity measurement becomes a *supporting* finding, not an
embarrassment.

Also note a parallel worth citing: at their lowest GPQA budgets, single-commit baselines *beat* RoR
(0.55--0.57 vs 0.35) until mean spend reaches ~15. Our own low-budget loss to `counts` has the same
shape, so low-budget underperformance is a known regime effect, not necessarily a defect -- but
ours is currently caused by miscalibration, which theirs is not.

### Reject cost: abstention is currently free, and that is not defensible (2026-08-27)

In the present replay, abstaining ends the episode with `correct = False` and **no additional
charge** -- the policy simply stops paying and takes the loss. That makes abstention free, and a
large part of the reported ~30% saving comes from declining 17--18% of episodes at zero penalty.

In deployment a refused query does not vanish; something else absorbs it (human escalation, a
frontier API call, a lost request). The reject-option literature -- their Table I row -- models this
as a **reject cost** `d`: being wrong costs 1, rejecting costs `d < 1`, and the optimal rule rejects
when confidence falls below `1 - d`.

The utility formulation expresses this in one line: abstaining has value `-d` rather than `0`, so
the stopping test becomes `max_m (p_m*R - cost_m) <= -d` and the policy continues while any action
beats the fallback. `d` is either fixed from deployment economics or swept as a second axis. This is
another argument for the dual: the primal has nowhere to put `d`.

**Required before claiming the abstention result.** Report the frontier at `d = 0` (current), at
`d` equal to one oss120 call, and at `d` equal to a plausible human-escalation cost. If the gain
survives only at `d = 0`, say so. A reviewer will otherwise observe that we bought a 30% saving by
refusing to answer, and the result will not survive.

### First utility-formulation frontier (2026-08-27, development-only, PARTIAL COMPARISON)

Artifact: `lcb_mdp_latest_attempt_seed17_1787808515/replay_ror_vs_value_{scout_first,free_start}_v1`.
Produced with the **miscalibrated** model (scout 9.28x over) on the **old `problem_first`** layout,
so it is a lower bound on what the corrected model should reach.

Best correctness at or below each cost, scout-first:

| cost | `counts` (RoR primal) | `counts_value` | `sequential_decay` | `sequential_decay_value` |
|---:|---|---|---|---|
| $0.0100 | 60.93% | 60.93% | 47.91% | **61.86%** |
| $0.0200 | **65.12%** | 65.12% | 65.12% | 64.65% |
| $0.0300 | 68.60% | 65.12% | 68.60% | **70.93%** |
| $0.0417 | 74.65% @ .0416 | 74.65% | 74.42% | **75.81% @ .0305** |

- The dual with learned beliefs wins at three of four matched costs; at $0.0417 it is **+1.16 points
  at 27% lower cost**.
- It reaches the **81.40% union ceiling at $0.1226**, against `best_of_10_oss120` 81.40% @ $0.13376
  -- same accuracy, **8.3% cheaper** -- and 80.93% @ $0.09757.
- Abstention falls smoothly out of `R` with no threshold: 65.1% of episodes at R=$0.0012 down to
  0.0% at R>=$1.32, where correctness pins at the ceiling.

**Two caveats that must travel with these numbers.**

1. **The comparison above $0.0417 is not valid.** This run predates the budget-grid extension, so the
   primal families were swept only to $0.0417 while the dual reached $0.143. Nothing in the
   high-cost region is a fair RoR comparison until the extended-grid re-run lands.
2. **`counts_value` matches `counts` exactly wherever both were swept** (60.93/60.93, 65.12/65.12,
   74.65/74.65). So the formulation *alone* buys nothing in that region. The win comes from learned
   beliefs **plus** a formulation that can use them -- which is the more defensible claim, and the
   one the 2x2 grid is designed to support.

**Free-start becomes informative only under the dual.** Under the ratio rule every policy chose
scout at the root 100% of the time at every budget (recorded above as the vacuous-ablation entry).
Under the value rule the root action sweeps with `R`:

| R ($/correct) | root action |
|---:|---|
| 0.0012 | ABSTAIN 100% |
| 0.0055 | scout 70%, ABSTAIN 30% |
| 0.0264 | scout 100% |
| 0.1260 | oss20 100% |
| 0.6021 | oss120 85%, oss20 15% |
| 2.8770 | oss120 100% |

"Should the scout be called at all?" is now answerable, and the answer is a function of what a
correct answer is worth. This is the clearest single demonstration that the dual expresses
decisions the primal structurally cannot, and it is independent of the frontier numbers.

### The advantage is on COST at matched accuracy, and abstention is the whole mechanism (2026-08-27)

The accuracy framing understates the result and is also the fragile one (the primal was
under-swept). The cost framing at *matched* accuracy is stronger and rests on a mechanism RoR
structurally lacks. Every policy reaching the 81.40% union ceiling, by cost:

| policy | knob | cost | abstain |
|---|---:|---:|---:|
| **`sequential_value`** | R=0.1863 | **$0.06852** | **13.5%** |
| `sequential_decay_value` | R=0.8902 | $0.12261 | 13.0% |
| `best_of_10_oss120` | -- | $0.13376 | n/a |
| every non-abstaining policy (incl. `counts_value`) | -- | $0.1429--$0.1437 | 0.0% |

- **48.8% cheaper than `best_of_10_oss120`** and **52% cheaper than the best non-abstaining
  policy**, at identical ceiling accuracy.
- Ablating only the give-up arm within one policy: `sequential_decay_value` costs $0.12261 with
  13.0% abstention and $0.14285 with 0%, i.e. **14.2% cheaper at identical accuracy purely from
  abstention**.
- **No policy without a give-up arm finds a cheap ceiling point**, including `counts_value` -- the
  dual running on RoR's count beliefs. So this is not the formulation alone: it requires learned,
  query-conditioned beliefs *and* an action space that can stop.

**The abstentions are provably all on unsolvable problems.** The policy reaches *exactly* the
ceiling while declining 13.5% of episodes; abstaining on even one solvable problem would have put
correctness below 81.40%. So it identified 13.5 of the 18.6 percentage points of hopeless problems
(73% of them) and quit before spending. This is forced by the arithmetic, not inferred from the
frontier, and it is the `nothing` head's 0.855 within-depth AUC doing the work.

**This is also where the reject-cost objection is weakest.** On a hopeless problem RoR exhausts its
budget and returns its best wrong candidate (their Algorithm 1, line 13); we spend little and return
nothing. Both are scored incorrect, so the correctness axis is a wash and the saving survives for
any reject cost below the grinding cost. For code specifically, a silently-wrong patch that will be
reviewed or merged is plausibly worse than an explicit refusal. Still report the sweep.

**Open puzzle before claiming any of this.** `sequential_value` (raw learned, no count decay) finds
the ceiling at $0.06852 while `sequential_decay_value` needs $0.12261 -- the decay makes it too
conservative here, inverting the earlier frontier result where decay won. Understand this before
choosing which policy to headline; it may be another symptom of the miscalibration, since the decay
was compensating for inflated scout probabilities that matter less once R is large.

Caveats unchanged: development-only, 86 test problems, one seed, five orderings, miscalibrated
model on the old `problem_first` layout, reject cost priced at zero.

### FAIR RoR comparison, extended grid (2026-08-28) -- the cost result holds

`replay_extgrid_scout_first_v2`, budget grid swept to full exhaustion ($0.34241). This supersedes
the "comparison above $0.0417 is not valid" caveat: RoR now reaches the ceiling.

Cheapest route to the 81.40% union ceiling, per family:

| policy | cost | abstain |
|---|---:|---:|
| **`sequential_value`** (dual + learned beliefs) | **$0.06690** | 13.7% |
| `sequential_decay_value` | $0.12256 | 12.8% |
| `best_of_10_oss120` | $0.13376 | -- |
| **`counts`** (RoR primal, extended grid) | **$0.13793** | 0% |
| `counts_value` (dual + count beliefs) | $0.14343 | 0% |

- **51.5% cheaper than a properly-swept RoR at identical accuracy.**
- At matched cost (~$0.066): RoR 78.37% @ $0.06611 vs ours 81.40% @ $0.06690, **+3.03 points**.
- RoR's extended frontier: 74.65% @ $0.04164 -> 78.37% @ $0.06611 -> 80.93% @ $0.11818 ->
  81.40% @ $0.13793. It does improve substantially once allowed to spend, as predicted; the
  advantage is not an artifact of under-sweeping the baseline.

**The 2x2 is now clean and rests on a fairly-swept baseline:** primal+counts $0.13793 ->
dual+counts $0.14343 (no better) -> dual+learned $0.06690. The formulation alone buys nothing; the
result requires query-conditioned beliefs *and* an action space that can stop. `counts_value`
cannot abstain at all, because count beliefs never say "hopeless": `p_m*R - cost_m` stays positive
for some route at any R large enough to reach the ceiling. That is the structural reason RoR cannot
reach the cheap point, and it is not a tuning gap.

### Job 3 negative result: counts_last + pos_weight made the model worse (2026-08-28)

`lcb_mdp_latest_attempt_seed17_1787873573`. **Two variables were changed at once, which was an
error** -- they should have been separate parallel jobs, and the confound cannot be resolved from
this run.

Calibration-split AUC at the selected epoch (the honest comparison, and the checkpoint criterion):

| head | old | new | delta |
|---|---:|---:|---:|
| scout_next | 0.7223 | 0.5742 | **-0.1481** |
| oss20_fresh | 0.7530 | 0.7465 | -0.0065 |
| oss120_fresh | 0.7408 | 0.7359 | -0.0049 |
| nothing | 0.7066 | 0.7028 | -0.0038 |

Worse on every head. The test split shows scout at +0.1358 (0.693 -> 0.829) but that contradicts
calibration by a similar magnitude in the opposite direction, which is the signature of noise on a
1.5%-positive target rather than a real gain -- do not report the test-split scout number.

**The stated success test failed.** Scout prediction by failure depth went 0.480 -> 0.405 against a
truth of 0.041 -> 0.000; the old model went 0.180 -> 0.107. The relative dynamic range is unchanged,
so the model still does not learn the decay, and the read-out-position hypothesis is not confirmed.
Calibration also degraded (scout 9.28x -> 29.89x over), which is expected from `pos_weight` --
it rebalances gradients, it does not calibrate -- so the ratio is not the metric to judge it by.

Next, as **separate** jobs: (a) `counts_last` alone, (b) `pos_weight` alone, (c) explicit numeric
count features concatenated to the pooled embedding before the head, which is the stronger version
of the hypothesis and does not depend on the encoder noticing two digits of text.

### Runtime reference and job hygiene (2026-08-27)

Measured on the 86-problem test split, 5 orderings, one GPU:

| run shape | duration |
|---|---:|
| budget sweep only, 12 points | 13m 40s |
| 12 budgets + 24-point value sweep, scout_first | 52m 15s |
| same, free_start | 55m 37s |
| both protocols chained in one process | **1h 48m** |

The value sweep dominates: it escalates into states the budget sweep never visits, so the scorer's
state-key cache misses. With the 17-point extended grid expect ~60--65 min per protocol.

**Protocols must be launched as separate parallel jobs**, never chained -- chaining converted two
~55 minute runs into a 1h48m serial wait. `launch_lcb_mdp_replay.sh` runs exactly one protocol over
an existing artifact for this reason. It requires `STATE_LAYOUT` explicitly: artifacts built before
2026-08-27 were trained on `problem_first`, and replaying them under the new `counts_last` default
would feed the scorer state text it has never seen.

### Revised work order (2026-08-27, supersedes the ordering in the list below)

1. **Launch oss20/oss120 multidraw on the 551 held-out problems.** Long pole, multi-day, zero
   dependencies. Asymmetric draws: ~4 scout, 10 oss120.
2. **Retire the budget cap as the frontier knob for our method** while keeping the budget-swept
   `counts` cell for RoR parity, and extend its grid past $0.134. Hours.
3. **Post-hoc recalibration** (per-route, depth-conditioned, calibration split only), then re-run
   the frontier. Everything downstream is unattributable until this lands.
4. **Structural continuation value**; then depth-1/2 lookahead as an ablation.
5. **On new data**: retrain with `pos_weight`, select on AUC, evaluate once on the 341 temporal test.
6. **Train a `content` (problem-text-only) predictor** over the reachable dataset and supply it via
   `--content-preds`. The `content` family is currently **never evaluated** in any run -- the flag
   has never been passed -- so the cleanest test of query-conditioning is absent from the grid.
7. **Multi-seed (5--10) and 20 replay orderings**, propagating every seed through the full frontier
   rather than reporting per-head AUC.
8. **Report the reject-cost sweep** (see the reject-cost entry above) before claiming the abstention
   result.

Fairness checklist for the headline RoR comparison, all required together: extended budget grid to
$0.342 (done); our beliefs calibrated as theirs are (open); multi-seed (open); cost-matched pairing
with problem-clustered bootstrap (computable post hoc); reject cost reported (open).

Open question worth deciding explicitly rather than drifting: whether the re-roll dimension earns
its place at all. If the decisions that matter all occur in the first ~3 actions, this is an
escalation problem dressed as a 30-step MDP, and the simpler framing may be the more defensible
paper.

### Remaining before final paper figures

1. **Freeze and stabilize the current result.** Repeat the existing hybrid with 20 replay
   orderings and paired problem-clustered intervals. This reduces ordering noise but does not
   increase the independent sample size or turn the current split into a confirmation set.
2. **Run replay-only development ablations on train/calibration only.** Compare route/depth
   calibration, route-specific failure decay, prediction of any remaining success per route, and
   finite-horizon rather than greedy probability/cost control. Predeclare one final policy before
   touching the new confirmation set.
3. **Scale unique problems before draws.** Audit the available pinned/newer LCB pool and collect
   the largest defensible chronological expansion, targeting at least 1,000 and preferably several
   thousand unique problems if available. Preserve 10 draws per route where affordable, but
   prioritize problem diversity: additional draws improve within-problem transition estimates,
   whereas additional problems determine confidence intervals and generalization.
4. **Use a strict temporal confirmation design.** Earlier problems train, a later block selects
   checkpoints/calibration/decay/planning settings, and the newest block is touched once. Preserve
   contest/platform grouping where needed and report both micro and macro results.
5. **Measure training variance.** Run at least five, preferably ten, router seeds because the
   earlier binary experiments found large LoRA seed variance. Propagate every seed through the
   frozen replay and report paired frontier uncertainty, not only per-head AUC.
6. **Complete the baseline and ablation grid.** Include fixed cascades, counts, counts + abstain,
   input-only semantic residual, static content, pure learned sequential, count-decayed hybrid,
   calibrated hybrid, and planning hybrid. Separate the value of semantic routing, structural
   updating, planning, and abstention.
7. **Test domain generality after locking the method.** Activate SWE-Smith eval150 only for
   development, first measuring marginal and post-failure conditional route rates; retain eval300
   as untouched test. If possible, enlarge SWE-Smith training/development with new independent
   tasks rather than consuming eval300 during iteration.
8. **Report deployment sensitivities.** Recompute frontiers under token cost, latency/attempt
   count, and nonzero execution cost, plus model-price sensitivity. Regenerate paper figures only
   from full-execution results; retain the pure learned failure as an informative ablation and keep
   all legacy public/private figures superseded.

## PLAN TO SUBMISSION

This is the operative roadmap from the current development result to a submission-ready TMLR
paper, superseding shorter work-order lists above where they conflict. The target claim is:

> Query-conditioned beliefs about whether further verified attempts remain worthwhile enable
> substantially cheaper inference at matched correctness than count-based rerouting, because the
> policy can stop selectively on exhausted problems.

The math and terminology are maintained in [`UTILITY_FORMULATION.md`](UTILITY_FORMULATION.md).

### Phase 1 -- make the pipeline trustworthy

1. Extend `build_mdp_tensors_v2.py` to ingest both `*_train_d*.jsonl` and `*_eval_d*.jsonl` and
   matching source metadata. It currently cannot consume the running 551-problem train collection.
2. Remove the duplicated expert loops from `launch_lcb_multidraw_mdp_collect.sh`.
3. Add integrity tests for disjoint problem splits, full-execution labels, cost/draw alignment,
   pass termination, free-start coverage, deterministic replay, and invalid-draw handling.
4. Standardize terminology: `pool-unsolved`, not intrinsically unsolvable; `myopic utility`, not an
   exact solved dual MDP; and `solver-call cost` until all overhead is priced.
5. Add router inference cost and latency, verifier/execution cost, and reject cost.

**Gate:** one documented command rebuilds tensors, trains, replays, validates invariants, and
regenerates tables without manual artifact manipulation.

### Phase 2 -- finish collection and freeze the data protocol

1. Complete and audit the ongoing 551-problem expert collection. The four-draw scout collection is
   complete; ten-draw expert collection is the current long pole.
2. Treat the repeatedly inspected 341-problem collection as development data, not a final test.
3. Collect a newer chronological LCB block that remains sealed until the method is frozen.
4. Assign immutable roles: earlier problems for training, the current 341 for development and
   calibration, and the newest block for one-shot confirmation.
5. Prefer independent problems over unnecessary extra draws; provisionally retain four to five
   scout draws and ten expert draws per problem.
6. Commit a versioned manifest containing IDs, dates, platforms, dataset revision, models,
   sampling parameters, artifacts, and split roles.

**Gate:** final confirmation IDs and analysis protocol are committed before outcomes are examined.

### Phase 3 -- lock the method on development data

1. Separate the confounded job-3 changes: `counts_last` without balanced `pos_weight`, then
   balanced `pos_weight` under the old layout.
2. Add explicit numeric execution-state features rather than relying only on prompt text counts.
3. Calibrate without final-test access using route-specific failure counts or state features, not
   only total failure depth.
4. Compare raw, per-route Platt/isotonic, empirical route-depth hazard, and
   learned-plus-structural calibration.
5. Compare the current one-step `pR-c` rule against structural finite-horizon continuation. Add a
   learned continuation residual only if the structural comparison leaves a clear gap.
6. Freeze one primary and one fallback policy, including checkpoint, calibration, state layout,
   and the value-of-correctness selection rule.

**Gate:** no policy-design decision changes after final confirmation begins.

### Phase 4 -- complete baselines and ablations

| component | required conditions |
|---|---|
| fixed allocation | single routes, one-pass cascade, best-of-K |
| RoR-style | count beliefs + hard budget + probability/cost ratio |
| formulation-only | count beliefs + utility stop |
| problem-only semantics | problem text without attempt history |
| sequential semantics | problem + failure state + latest failed attempt |
| calibration | raw versus frozen calibrated beliefs |
| planning | myopic versus continuation-aware utility |
| stopping | reject arm versus forced exhaustion |
| initial action | mandatory scout versus free start over all problems |

The core factorial comparison is count versus query-conditioned beliefs crossed with
budget-and-ratio versus utility-and-stop. Isolate whether failed-attempt content helps beyond
problem difficulty, planning helps beyond one-step utility, and abstention causes the saving.

**Gate:** every component named in the main claim has a direct ablation.

### Phase 5 -- freeze evaluation and statistics

Freeze checkpointing, calibration, `R` selection, replay orderings, cost model, reject costs,
primary metrics, matching tolerances, and baseline implementations before confirmation.

1. Run at least five router seeds, preferably ten.
2. Use at least twenty replay orderings while keeping the problem as the independent unit.
3. Report paired problem-clustered intervals for cost at matched correctness, correctness at matched
   cost, abstention rate, and false abstention on pool-solvable problems.
4. Use a predeclared frontier interpolation/envelope rule, not a post-hoc test row that happens to
   equal the finite-pool ceiling.
5. Propagate training variance through the complete frontier rather than reporting only head AUC.

**Gate:** the headline comparison has a problem-level uncertainty interval.

### Phase 6 -- stress-test deployment assumptions

Sweep router cost and latency, verifier/execution cost, reject cost, value of correctness `R`, model
prices, draw counts, temperature, imperfect verification, and mandatory-scout versus free-start.
Present a robust deployment region, not one favorable operating point. If the saving survives only
with free rejection or omitted router cost, make that part of the claim.

### Phase 7 -- establish external validity

After locking LCB, activate SWE-Smith protocol v2. First verify meaningful marginal pass rates,
post-failure success, complementarity, and reroll decay. Use eval150 for development and preserve
non-overlapping eval300 as test after collecting its complete real sandbox reports. Run the frozen
method and baseline grid without redesigning around SWE-Smith test outcomes. A negative result can
still establish a useful boundary condition.

### Phase 8 -- paper and reproducibility package

Organize the paper around the stopping problem, the limitation of unthresholded count-and-ratio
allocation, query-conditioned failure beliefs, utility and continuation, full-execution replay,
temporal LCB confirmation, mechanism ablations, external validity, and limitations.

Primary outputs: correctness-cost frontier with uncertainty; both matched frontier comparisons;
the beliefs-by-formulation 2-by-2; calibration and hazards; abstention timing and false abstention;
mandatory-scout versus free-start; deployment-cost sensitivity; and cross-domain results.

Release manifests, provenance, environment/model versions, one-command pipeline entry points,
frozen configs and seeds, aggregate outputs, figure scripts, and redistribution limitations.

### Immediate execution order

1. Finish and validate the 551-problem collection.
2. Fix train-split ingestion and collection-launch duplication while it runs.
3. Launch the two clean state-layout/positive-weight diagnostics on existing development data.
4. Add total-cost accounting and route-specific calibration.
5. Run continuation and the complete development ablation grid.
6. Freeze the method and confirmation protocol.
7. Collect and open the new chronological confirmation block exactly once.
8. Run SWE-Smith only after LCB is locked.
9. Regenerate the paper from frozen artifacts and complete the reproducibility audit.

### Temporal 551-to-341 protocol setup (2026-08-28)

We will proceed pragmatically without waiting for a hypothetical new LCB release. The primary next
experiment is the existing chronological split: fit on all 551 problems dated through 2024-09-28,
calibrate on the earliest 170 problems of the later 341 block (through 2025-01-04), and report on
the remaining 171 problems dated 2025-01-11 onward. This is development evidence, not a claim of
researcher-blind final confirmation: the later 341 collection has already informed exploratory work.

The input artifacts and split are now materialized at
`/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_prepared_v1/`: `tensors_v3`
contains all 892 problems with the 551/170/171 source-temporal assignment, and both
`reachable_dataset_problem_first` and `reachable_dataset_counts_last` have been built and
validated. `rolling_temporal_folds_v1.json` partitions all 892 problems into five contiguous,
equal-date-preserving blocks (177/180/179/174/182 problems) and defines three rolling-origin
train/calibration/test folds. These are preparation artifacts only; no corresponding jobs have
been submitted yet.

The tensor builder now supports combined train/eval collections and explicit asymmetric horizons:
four scout draws and ten draws for each expert. The four-scout horizon is intentional; available
train data show later scout draws are unproductive, and absent draws are represented as unavailable
rather than failures. The first opt-in suite compares `problem_first` and `counts_last`, both with
unweighted BCE, under this same temporal protocol. Its submission wrapper waits 30 seconds between
job launches.

Method development is still authorized on this temporal setup. The next controlled improvement is
explicit numeric state features (per-route failures, remaining draws, total failures, and latest
route) concatenated to the policy head, followed by route-specific calibration and structural
continuation value. The prompt-position `counts_last` result does not settle whether structured
count features are useful. After a stable method is selected, report rolling chronological folds
across the 892 problems and reserve SWE-Smith eval300 as the external confirmation rather than
waiting for new LCB data.

#### Structured-state feature test (prepared, not submitted)

The next controlled row holds the chronological split, data construction, route horizons, seed,
three-epoch schedule, unweighted BCE, latest failed code, aggregate full-execution feedback, and
`counts_last` text prompt fixed. It changes only the policy head: alongside the normalized Qwen
embedding, a small 11-to-64 feature encoder receives per-route failed fractions, per-route
remaining-draw fractions, a total-failure fraction, and a latest-route one-hot (none/scout/oss20/
oss120). The fused representation feeds the same four success/none heads. Features are normalized
by the valid draw capacity available to that problem and route; no hidden test inputs, individual
test identifiers, or extra verifier information are added. The dataset serializes the vector and
its named schema, training checkpoints it with the head, and replay reconstructs the same vector at
every decision. The opt-in launcher is
`launchers/abstention/launch_lcb_mdp_temporal_551_341_structured_state.sh`. Compare its calibrated
frontier, abstention timing, depth hazards, and paired problem-level intervals directly against the
contemporaneous `counts_last` text-only run. A gain would show that the router needs direct numeric
state access; no gain would rule out this simple representation change before more complex
continuation modeling.

#### Fair reporting of overall and conditional savings

Overall realized spend and correctness over every test problem and replay ordering are the primary
deployment metrics. They include the mandatory scout, scout-success episodes, router-entry
episodes, escalations, re-rolls, and abstentions. The post-scout-failure analysis is a legitimate
secondary mechanism analysis because the initial scout is fixed and mandatory for every policy; the
conditioning event is therefore not selected by the learned router. It must report the number of
router-entry episodes, conditional correctness, conditional realized spend, and problem-clustered
uncertainty intervals, and it must use the same correctness-matching/envelope rule as the overall
frontier. It should not replace the overall metric or be presented as an unconditional deployment
saving.

A direct readout from the completed exploratory random-341 `counts_last` replay illustrates the
expected decomposition (86 test problems, five orderings, one training seed; not temporal
confirmation). At utility value `R=0.126`, the learned value-stop policy reached 80.23% overall
correctness versus 80.00% for the no-abstention count/RoR-style policy, while reducing overall
realized spend from $0.0980 to $0.0561 (42.8%). Among scout-failure router entries, conditional
correctness was 69.64% versus 69.29% and conditional spend was $0.1504 versus $0.0860 (42.8%).
At `R=0.186`, both policies reached 81.40% overall correctness; overall spend was $0.1379
versus $0.0821 (40.5%), and conditional spend was $0.2117 versus $0.1259 (40.5%), with equal
71.43% conditional correctness. These are exploratory effect sizes only; the temporal runs must
re-estimate them with paired problem-level intervals. The earlier 8--10% shorthand should not be
used as a general summary of this replay, and the retired weak-verifier ~50% result remains
non-primary.

### Free-start is a null result -- do not spend jobs on it (2026-08-28)

`replay_extgrid_free_start_v2` vs `replay_extgrid_scout_first_v2`, both on the extended budget grid,
random-341 development split, all policies given the same start protocol.

| cost | ours SF | ours FS | RoR SF | RoR FS |
|---:|---:|---:|---:|---:|
| $0.005 | 55.12% | 55.12% | 34.88% | 34.88% |
| $0.010 | 61.86% | 61.86% | 60.93% | 60.93% |
| $0.020 | 64.65% | 64.65% | 65.12% | 65.12% |
| $0.030 | 70.93% | 70.93% | 68.60% | 68.60% |
| $0.045 | 77.44% | 77.44% | 74.65% | 74.65% |
| $0.067 | **81.40%** | 79.30% | 78.37% | 78.37% |
| $0.138 | 81.40% | 81.40% | 81.40% | 81.40% |

Free-start is **identical to scout-first at every operating point except $0.067, where it is 2.1
points worse**. At the ceiling: ours $0.06861 free-start vs $0.06690 scout-first; RoR is
bit-identical at $0.13793 under both.

Mechanism: at the root the scout costs $0.00055 and solves 47.7% of problems outright, so any `R`
large enough to take any action at all takes that one. The mandatory scout was never a binding
constraint -- it is what the optimal policy does anyway. This reproduces, under the utility rule,
the earlier finding that the ratio rule also chose scout at the root 100% of the time.

**Decision: do not launch free-start training jobs.** The ablation is well-posed, already answered,
and the answer is null. Report it as a one-line negative ("removing the mandatory scout changes
neither frontier") rather than treating it as an open question. Note also that the earlier
"free-start is vacuous" entry above is now superseded in its diagnosis: it is not vacuous because
the *rule* cannot decline the scout, it is null because declining the scout is not worth doing.

### Router inference cost: sized, and the headline survives (2026-08-28)

Addresses the audit finding that reported dollar costs cover solver calls only, while our policy
invokes an 8B embedding encoder repeatedly and RoR's count update is free.

Measured at the winning operating point (`sequential_value`, R=0.1863, 430 episodes): **6.42 router
calls per episode** against 7.28 solver attempts. At ~3,000 tokens per rendered state:

| router price | added $/episode | ours | vs RoR $0.13793 |
|---|---:|---:|---:|
| $0.02 /M | $0.00038 | $0.06728 | 51.2% cheaper |
| $0.05 /M | $0.00096 | $0.06786 | 50.8% cheaper |
| $0.10 /M | $0.00192 | $0.06882 | 50.1% cheaper |
| $0.278 /M (scout's own price -- an overestimate) | $0.00535 | $0.07225 | **47.6% cheaper** |

So the headline degrades from 51.5% to at worst 47.6%, and $0.278/M is a deliberate overestimate
since the encoder does a single forward pass with no generation. **The conclusion is unaffected.**

**But it must be included at the cheap end.** At operating points around $0.005 total spend, router
cost at $0.10/M is **38% of total spend**, so the low-cost region of our frontier is materially
overstated without it. Add router cost as a configurable term and report the frontier with it on;
do not quietly restrict the claim to the ceiling where it happens not to matter.

### Conditional vs overall savings coincide here; the 8% vs 51% gap is abstention (2026-08-28)

Computed directly from `replay_extgrid_scout_first_v2` traces at the matched 81.40% ceiling
(430 episodes, `sequential_value` R=0.1863 vs `counts` at full-exhaustion budget):

| | overall | conditional (reached router) |
|---|---:|---:|
| `counts` (RoR) | $0.13793 | $0.21171 |
| `sequential_value` (ours) | $0.06690 | $0.10263 |
| saving | **51.5%** | **51.5%** |

**There is no dilution.** Scout-resolved episodes are 34.9% of episodes but only **0.05% of total
spend** ($0.00055 against a $0.138 average), so conditioning on router entry cannot move the
percentage. Reporting both remains honest but is uninformative in this setup -- say so rather than
implying the conditional number is a stronger effect.

**The 8-10% vs 40-51% confusion is not conditional-vs-overall, it is with/without abstention.**
Three distinct comparisons have been in circulation:

| figure | what it compares |
|---|---|
| 8.45% | count-decay hybrid: `sequential_decay` vs `counts`, **both without abstention**, budget $0.07306, matched 74.42% |
| 40-43% | the `counts_last` replay, conditional and overall |
| 51.5% | ours **with** abstention vs RoR, matched at the 81.40% ceiling |

8.45% is what better routing buys with the give-up arm off; 51.5% is with it on. This agrees with
the direct ablation (abstention off within one policy costs 14.2% at identical accuracy) and with
the fact that no policy lacking a give-up arm reaches a cheap ceiling point. Do not present 8-10%
and 51.5% as competing estimates of the same quantity.

### Tier 1 baseline-fairness pass (2026-08-28, implemented)

Three changes, all replay-only, aimed at things that could **shrink the 51.5% headline**. Done
before quoting that number again, on the principle that we already found one instance of the
harness handicapping RoR (the budget grid capped at $0.089, below `best_of_10_oss120` at $0.13376).

**1. Freeze `R` on calibration (`--retention-grid`).** The headline R=0.1863 was selected by reading
the *test* frontier for the cheapest point at exactly ceiling accuracy. That is post-hoc selection
and is the most likely reason the number is optimistic. For each correctness-retention target the
cheapest R meeting it **on calibration** is now frozen, and the test split is touched once per
target, emitted as `<family>_value_frozen` with `selection: calibration_only` plus the calibration
correctness and ceiling for provenance. Expect the frozen number to be worse than 51.5%; that is
the point.

**2. Pseudo-count sweep for RoR (`--pseudo-count-grid`).** RoR's update is
`p_hat = (s*p_bar_m + w)/(s + n)`; in our failure region `w=0`, so the belief is `s*p_bar/(s+n)` and
`s` directly sets how fast the baseline escalates:

| s | belief after 5 failures |
|---:|---|
| 0.5 | 0.09x prior (escalates fast) |
| 2 (our default) | 0.29x prior |
| 10 | 0.67x prior (keeps re-rolling) |

RoR reports insensitivity to `s` over an order of magnitude, but on **their** pool (11 models, k=30,
four benchmarks). Our regime differs enough -- 3 models, 53x cost spread, k=10, near-perfect
verifier -- that the claim should be verified rather than inherited. Emitted as `counts_s{value}`.
**Report RoR at its best s.**

**3. RoR's UCB variant (`--ucb`).** RoR ships greedy *and* UCB (exploration bonus
`sqrt(2 ln(t+1)/(n_m+1))/c_m`). We only implemented greedy. Their Table V suggests greedy is the
stronger arm on three of four benchmarks, so this is expected to be a null check -- but it converts
an assumption into evidence. Emitted as `counts_ucb`.

Supporting change: `--decay-pseudo-count` separates our `sequential_decay` variant's pseudo-count
from RoR's, so sweeping `s` for the baseline does not drag our own policy along with it.

Tests added: UCB explores more routes than greedy; decay pseudo-count is independent of the RoR
pseudo-count; greedy is bit-identical when exploration is off. 25 tests pass.

**Not in Tier 1, with reasons.** Reject cost `d` is omitted deliberately: on a pool-unsolved problem
both policies end up incorrect -- we return nothing, RoR returns its best wrong candidate (their
Alg. 1 line 13) -- so a symmetric accounting charges `d` to both and it cancels. It only fails to
cancel if refusing is worse than silently answering wrong, which for code is the opposite of true.
State that in one sentence rather than sweeping it. Router inference cost is computed separately
(6.42 calls/episode; 51.5% -> at worst 47.6%) and does not need to be a policy arm, but **must be
reported at the cheap end too**, where it is 38% of total spend.

### Tier 1 results: RoR is not under-tuned (2026-08-28)

`replay_tier1_counts_cpu_v1` -- count families only, so no scorer and no GPU. Cheapest cost to reach
the 81.40% ceiling:

| RoR variant | cost @ ceiling |
|---|---:|
| s=0.5 | **$0.13733** (best) |
| s=1 | $0.13740 |
| s=2 (our default) | $0.13793 |
| s=5 / 10 / 20 | $0.13849 |
| UCB | $0.13849 |

**Both fairness checks come back null.**

1. **Pseudo-count sensitivity: none.** The full sweep spans 0.8%, and every `s` gives identical
   74.65% at a $0.045 budget. RoR's own insensitivity claim replicates on our pool despite the
   regime difference (3 models vs 11, 53x cost spread, k=10 vs 30). Our `s=2` default was not
   handicapping the baseline.
2. **UCB is not RoR's stronger arm here** ($0.13849 vs greedy $0.13793), consistent with their
   Table V. Comparing against greedy was already the harder comparison; that is now measured
   rather than assumed.

**Headline restated against RoR at its best setting: $0.06690 vs $0.13733 = 51.3% cheaper**
(was 51.5% at s=2). Immaterial change -- report the 51.3% figure and cite the sweep.

Remaining Tier 1 item is the one expected to bite: freezing `R` on calibration for *our* policy
(`sequential*_value_frozen`), which needs the scorer and is running on the local GPU. The
count-family half (`counts_value_frozen`) is already in this artifact.

Note on execution: this arm needs **no GPU and no eai job** -- dropping `--sequential-model-dir`
collapses `families` to `["counts"]` and takes the scorer out of the path entirely. Prefer that for
any baseline-only sweep.

### Tier 1 result: frozen-R survives, but the retention target is a new knob (2026-08-28)

`replay_tier1_fairness_v1`. R selected on CALIBRATION only, test touched once per target.

| policy | retention target | R* (calibration) | cal corr | TEST corr | TEST cost |
|---|---:|---:|---:|---:|---:|
| **`sequential_value_frozen`** | **0.95** | **0.1863** | 76.00% | **81.40%** | **$0.06690** |
| `sequential_value_frozen` | 0.98 / 0.99 / 1.00 | 0.4073 | 77.65% | 81.40% | $0.14001 |
| `sequential_decay_value_frozen` | 0.95 | 0.2755 | 74.12% | 79.30% | $0.05754 |
| `sequential_decay_value_frozen` | 1.00 | 1.3162 | 77.65% | 81.40% | $0.14275 |
| `counts_value_frozen` | 0.99 / 1.00 | 0.2755 | 77.65% | 81.40% | $0.14343 |

**The post-hoc selection concern is resolved.** Calibration independently selects R=0.1863 -- the
same value previously read off the test frontier -- giving 81.40% at $0.06690. Against RoR at its
best `s` ($0.13733) that is **51.3% cheaper with no test-set selection anywhere in the pipeline**.
All three Tier 1 threats (post-hoc R, pseudo-count tuning, UCB variant) are now cleared.

**But the retention target is itself a selection knob, and it matters a lot.** At targets 0.98--1.00
the frozen R overshoots to 0.4073 and costs $0.14001 for the same 81.40% -- no better than RoR.
Cause: the calibration ceiling is 77.65% while the test ceiling is 81.40%, so a strict
retention target on calibration purchases spending the test split does not need. **Report all four
targets.** Quoting only 0.95 moves the selection up one level rather than removing it. The honest
headline is "at a predeclared 0.95 retention target", and the target must be declared before the
final temporal evaluation, not chosen from this table.

**Do not quote a 58.1% figure.** An automated summary produced it by comparing
`sequential_decay_value_frozen` at 79.30% against RoR at 81.40%; it is cheaper only because it is
less accurate. Matched-accuracy comparisons only.

### Launcher hazard: running jobs execute the LIVE tree but import from a frozen snapshot (2026-08-28)

Two of the three temporal representation jobs died, and **not** because of the eai 502 outage
(that was hours later, at ~22:50; these failed at 19:16--19:24). Real cause, from
`eai job log`, identical for both:

```
ModuleNotFoundError: No module named 'pipelinerl.swe.scripts.livecodebench.structured_state'
```

**Mechanism.** The launcher sets `--workdir` and `PYTHONPATH` to a frozen snapshot
(`~/snapshots2/<sha>`), but `COMMAND` begins with `cd /home/toolkit/PipelineRL-SWE`. So a job runs
the **live working tree's script file** while resolving `import pipelinerl.*` against its
**launch-time snapshot**. Editing the live tree therefore breaks jobs that are already running, as
soon as a script gains an import the older snapshot lacks.

| time | event |
|---|---|
| 17:54 | `problem_first`, `counts_last` launched; snapshot predates `structured_state.py` |
| ~18:04 | `structured_state.py` added to the live tree; `structured` launched |
| 19:16--19:24 | the first two reach replay, run the *live* `replay_mdp_full_execution.py` (which now imports `structured_state`) against their *old* snapshot -> ImportError |
| 20:16 | `structured` succeeds -- its snapshot contains the module |

Both jobs had already completed training, so only the replay stage was lost; the saved models were
re-replayed directly rather than retrained.

**Implications.**

- A long multi-stage job is exposed to every edit made to the live tree while it runs. New modules
  imported by any stage are the specific trigger.
- Fix options: make `COMMAND` `cd` into the snapshot rather than the live tree, so a job is fully
  hermetic; or, until then, avoid adding new modules to the live tree while multi-stage jobs are in
  flight.
- **Check job state explicitly rather than inferring it from missing output.** These looked like
  "still running" for hours because the output directory simply never appeared. `eai job ls` showed
  FAILED/CANCELLED immediately.

### Uncertainty intervals and the router-cost regime boundary (2026-08-28)

Problem-clustered bootstrap (10,000 resamples over the 86 development problems) on
`replay_tier1_fairness_v1`.

**Headline now has an interval.** At matched 81.40% ceiling accuracy, ours $0.06690 vs RoR $0.13793:

- **cost saving 51.5%, 95% CI [33.2%, 65.9%]**; P(saving>0)=100%, P(>25%)=99.7%, P(>40%)=89.8%.
- correctness delta +0.00 pts, CI [+0.00, +0.00]. The zero width is correct, not a bug: both
  policies sit at the ceiling, so every resample has them solving the identical problem set. It is
  a genuinely matched comparison.

**The one place we trailed is noise.** At 68.60% correctness the apparent -1.0% is -0.2% with
95% CI [-0.8%, +0.3%] -- indistinguishable from zero. So there is no operating point where the
baseline significantly beats us.

**Router cost is negligible; an earlier "regime boundary" entry here was wrong and is retracted.**
Two errors inflated it ~8x: the per-state token count was *assumed* at 3,000 when the measured mean
is **1,321** (median 918, p90 2,602), and the price was *assumed* at $0.10/M without calculation.

Calculated properly: a single Qwen3-Embedding-8B forward pass over 1,321 tokens is
`2 x 8e9 x 1321 = 21.1 TFLOP`, ~52.8 ms on an H100 at 400 TFLOP/s effective, ~$0.0278/M tokens at
$2.50/GPU-hour. At 6.42 router calls/episode that is **$0.000236/episode = 0.4% of our $0.06690
ceiling point**, and ~4% at the cheapest operating points -- not the 38% recorded earlier.

| our correctness | raw | +8B router | +0.6B router | +$0.13/M (embedding-3-large) |
|---:|---:|---:|---:|---:|
| 52.56% | +15.2% | +12.0% | +14.9% | +0.5% |
| 61.63% | +29.3% | +27.6% | +29.1% | +21.4% |
| 68.60% | -0.2% | -1.5% | -0.3% | -6.2% |
| 74.42% | +18.9% | +18.2% | +18.8% | +15.7% |
| 78.14% | +20.1% | +19.7% | +20.1% | +18.2% |
| **81.40%** | **+51.5%** | **+51.3%** | **+51.5%** | **+50.7%** |

Savings stay positive across the whole frontier under **every** pricing basis, including $0.13/M
which is higher than the figure that produced the spurious boundary. The only negative cell is the
68.60% point already shown to be noise (-0.2%, CI [-0.8%, +0.3%]).

Report the 8B compute basis as primary (it matches how the solver costs are constructed -- actual
resource use) and the commercial-embedding-API basis as a sensitivity row.

**Smaller encoder (0.6B) is a latency argument, not an economic one.** At $0.0021/M the router
becomes 0.0% of spend, but 8B is already 0.4%, so cost is not the motivation. The real gain is
latency: 3.96 ms vs 52.8 ms per call, i.e. ~24 ms vs ~340 ms added per episode. Worth an ablation
for the deployment story; not urgent, and it must not be justified on cost grounds.

**Process note.** Two invented numbers reached this document today: a "58.1% saving" from an
automated summary comparing mismatched accuracies, and a "$0.10/M" router price that was assumed
rather than derived. Both produced findings that survived one review pass. Derive cost parameters
before recording conclusions that depend on them.

**Gap to close:** the `counts_s*` and `counts_ucb` arms run with `capture_trace=False`, so they
write no episode traces and cannot be bootstrapped. Comparisons against RoR at its best `s` are
currently point estimates only. Enable traces for those arms before the final tables.

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

---

## Exact Bellman solve over the failure-count lattice (2026-08-29)

### Motivation

Every routing arm to date has used a **myopic** rule: `argmax_m p_m·R − c_m`, with abstention as
the zero-value action. Codex's audit flagged this correctly — calling it a "dual" is too strong,
because it drops the continuation term `(1 − p_m)·V(s')`. Consequences: the policy cannot
distinguish "scout is cheap and oss120 is still in reserve" from "this is the last affordable
shot," and it does not know draws are finite.

This is also the most likely explanation for the sharpest negative result in the temporal
analysis: at matched `R`, the learned scorer is significantly **cheaper** at 15–18 of 24 `R`
values but significantly **more accurate at 0 of 24**. All measured gain sits on the cost axis.
Ordering is exactly what a myopic rule throws away.

### The solve is exact, not learned

The earlier framing of "train with the full Bellman" (fitted-Q, or policy gradient against the
replay simulator) was wrong, and it was the reason the direction was previously deprioritized as
data-starved at 551 problems. **The MDP is small enough to solve outright:**

- Draws are exchangeable given the problem, so per-route failure counts are a sufficient
  statistic. Full history adds tokens, not information.
- With `n_total ≤ 10` over three routes the lattice is `C(13,3) = 286` states.
- The transition model is fully determined by beliefs the scorer already emits.
- Budget and capacity feasibility are deterministic functions of the count vector (per-route
  costs are constants), so neither enters the state.

```
V(k) = max( 0,  max_m [ p_m(k)·R − c_m + (1 − p_m(k))·V(k + e_m) ] )
p_m(k) = pbar_m · s / (s + n_m + k_m)
```

Backward induction, descending on every axis. No bootstrapping, no value regression, and no
hindsight bias from an oracle target — the three failure modes that made the learned variants
unattractive. Future beliefs are extrapolated with the same Beta–Bernoulli posterior the myopic
decay arm already applies, so belief models are identical between arms and the comparison
isolates the continuation value alone.

`--bellman-horizons H1,H2,...` adds `<family>_bellman_h<H>_value` arms beside the myopic
`<family>_value` arms, plus calibration-frozen variants and paired bootstraps at matched `R`.

**H=1 truncates every successor to zero and therefore reproduces the myopic rule exactly.**
Verified two ways: as a unit test, and end-to-end on the real temporal tensors — 0 mismatches in
correctness or cost across all 24 `R` values.

### Why RoR did not do this

Not tractability. Their 11-model pool gives `C(21,11) ≈ 353k` states — 265× ours, still very
solvable — and the hard budget adds no dimension for the reason above.

Partly the knapsack framing. Maximizing `1 − ∏(1−p_m)` subject to `Σc_m ≤ B` is a knapsack in
log space (`max Σ log(1/(1−p_m))`), for which greedy-by-density is the textbook heuristic; RoR's
`p/c` is its small-`p` linearization. Framed as set selection under a budget, the adaptive
structure disappears from view.

**But the substantive reason is that it would not have paid.** The continuation value only
separates actions when routes differ in how much *query-specific* option value they preserve.
Under a global scalar `p̄_m` every problem looks identical to the policy, the optimal ordering is
constant across queries, and greedy-by-density is already near-optimal.

Measured on the counts family (RoR's own belief model), temporal split, 2 orderings:

| target | myopic | h=2 | h=4 |
|--------|--------|-----|-----|
| 65.0% | $0.04358 | $0.04513 | $0.04529 |
| 70.0% | $0.07470 | $0.07576 | $0.07623 |
| 73.0% | $0.16475 | $0.16250 | $0.16302 |

±1–3%, both directions. **Given RoR's belief model, RoR left nothing on the table.**

### Claim under test

> The exact Bellman solve is worth doing only once the prior is query-conditioned. The learned
> prior and the sequential solve are complementary; neither is worth much alone.

This is currently a hypothesis with one supporting leg (the counts family above). It requires the
`sequential_decay_bellman_h*` arms to confirm. **Pre-registered failure condition:** if those also
come back flat, the continuation value is null for this problem class regardless of the prior, and
the correct write-up is a negative result — greedy-by-density is near-optimal for adaptive routing
— not a method contribution.

### Status

Implemented in `replay_mdp_full_execution.py`; 37 tests pass (from 25). Launched on all three
temporal layouts with `--bellman-horizons 2,4,8`, reusing existing models and `tensors_v3`; no
retraining and no new collection.

### Consequence for the log-basis work

If the policy applies `s·p̄_m(x)/(s + n_m)` analytically inside the DP, the network never has to
represent the decay internally. The "why can't the model learn the hazard" thread (and the
proposed `log(s/(s+n))` feature basis) therefore drops from method-critical to diagnostic. It also
explains why `structured` did not beat `counts_last`: those features asked the network to learn
something that can simply be computed.

---

## Temporal Bellman readout and oracle-stopping headroom check (2026-08-30)

The three H=2/4/8 temporal replays completed. Bellman continuation is **not** a large new source of
gain under the current analytic future-belief model. On `counts_last`, H=2 is mildly helpful at a
few matched-R points, while deeper H=4/8 is generally no better. For example, at R=0.1882,
`sequential_decay` moves from 71.46% @ $0.09907 to H=2 at 71.70% @ $0.09481; the paired cost delta
is significant but small (95% CI [-$0.00625, -$0.00224]) and the correctness interval includes
zero. At the 73.10% ceiling, H=2 saves 1.8% and H=8 is effectively myopic.

The calibration-frozen 95%-retention temporal point is the current conservative summary:

| policy | correctness | mean realized cost |
|---|---:|---:|
| RoR-style `counts` frontier | 70.41% | $0.07930 |
| stronger `counts_value_frozen` | 70.41% | $0.07762 |
| `sequential_decay_bellman_h2_value_frozen` | 70.41% | $0.07057 |

Thus the newest saving is **11.0% versus RoR-style counts, or 9.1% versus the stronger count+utility
baseline**. The older 27% development point was a real dominance observation, but it used the old
random/development protocol, miscalibrated `problem_first` model, and exploratory operating-point
selection. Retain it only as superseded development evidence. The 51.5% random-split result likewise
remains valid for its protocol but has not transferred to the strict later temporal block.

### One-variable oracle diagnostic

Before changing the loss, architecture, decay model, or portfolio, measure whether perfect stopping
knowledge has material headroom. The diagnostic replaces **only** stop/continue with the leaked bit
`any still-unseen stored draw succeeds`. If false, abstain immediately; if true, continue with the
unchanged `sequential_decay` Bellman-H=2 route beliefs, ranking, costs, capacities, and
calibration-frozen R. It is an upper-bound mechanism analysis, never a deployable policy or headline
method.

Implemented behind `--oracle-stopping-family` / `--oracle-stopping-horizon`, with problem-clustered
paired bootstraps and explicit `diagnostic_only` metadata. Tests verify both immediate stopping on a
doomed state and overriding a false value-stop until a stored success is reached (39 replay tests
pass). The independent opt-in launcher is:

```bash
SUBMIT=1 bash launchers/abstention/launch_lcb_mdp_oracle_stopping_temporal.sh

---

## Learned one-step transitions in the Bellman solve (2026-08-30)

### What the analytic DP was assuming

The exact Bellman solve calls the scorer **once**, at the current state, then extrapolates every
future belief as `pbar_m · s/(s + n_m)`. That carries two assumptions:

1. beliefs decay at rate `s/(s+n)`;
2. a failure on route `m` says **nothing** about route `m′`.

The second is almost certainly false — a hard problem is hard for every route — and the analytic
form structurally cannot express it. The trained model can: it sees the whole count vector and
emits all four heads, so it is free to lower `oss120` after three scout failures. **We never asked
it.** The model was consulted only about the state we are in, never about the states we plan
through.

This is the concrete mechanism behind the standing diagnosis. Two independent observations point at
the transition model rather than the recursion: `h4`/`h8` are no better than `h2` (deeper lookahead
compounds transition error), and lookahead makes the *counts* family monotonically **worse**
($0.07762 → $0.07854 → $0.07930 → $0.07930), because optimizing harder against a bad belief model
moves you further in the wrong direction.

### The arm

`--learned-transition-horizons H` queries the scorer at each depth-1 successor instead of
extrapolating. At `H=2` that is *every node the recursion touches*, so the arm is the **exact
Bellman optimum under the model's own beliefs**, with no analytic extrapolation anywhere.

Only the source of the raw belief changes; the decay is applied identically at every node either
way. The paired bootstrap references the **same horizon with analytic transitions**, so the delta
isolates the transition model and nothing else.

**No leakage.** Successor counts are known exactly at decision time, but the content of an attempt
not yet made is not, so the latest observed attempt's text is carried forward unchanged. Using the
real future draw's code would leak an outcome the policy cannot see in deployment. A test asserts
the root's successor queries contain no draw code at all.

Requires no retraining. 44 tests pass (from 39).

### Why this before Codex's retraining program

Codex's recommended sequence (train an explicit remaining-solvability head `q(s)`, add
depth-balanced sampling, paired adjacent-state consistency, action-conditioned transition loss,
remove the analytic double-decay) is well motivated and probably right eventually. But every step
of it costs a training run, and all of it is premised on the transition model being the bottleneck.

This arm tests that premise for the price of one replay. If learned transitions move the frontier,
the premise holds and the retraining program is justified. If they do not, the premise was wrong and
we have avoided an entire retraining sequence chasing it.

### Launcher hermeticity fixed (long-standing hazard, recorded 2026-08-28 as `a2e27d2`)

With `SNAPSHOT=1` the Makefile already sets both `--workdir` and `PYTHONPATH` to
`/home/toolkit/snapshots2/<revision>`, so jobs are hermetic **by default**. Our launcher silently
undid this by prefixing `COMMAND` with `cd ${REPO_ROOT}` (the live tree): Python then loaded the
**entry script** from the live tree — it becomes `sys.path[0]` — while `from pipelinerl...` still
resolved against the snapshot. That mixed resolution is what killed two jobs on 2026-08-28.

`launch_lcb_mdp_replay.sh` no longer `cd`s. Edits made to the live tree while a job runs can no
longer reach it. The same `cd ${REPO_ROOT}` pattern remains inside `COMMAND` in several other
launchers (`launch_lcb_mdp_full_execution.sh`, `launch_lcb_mdp_sequential_train.sh`,
`launch_lcb_mdp_temporal_551_341.sh`, `launch_lcb_embedding_router_train.sh`, and the collection
launchers) and should get the same treatment.

### The `nothing` head is currently dead weight

Confirmed by inspection: `p_any` is read at exactly one place, the `tau_abstain` threshold. The
value and Bellman arms stop via `max_m(p_m·R − c_m) ≤ 0`, so **they never touch the `nothing` head
at all.** It was trained for the original threshold-abstention design and became vestigial when the
utility formulation replaced it.

It is not redundant in principle, though. The per-route heads answer "will the next draw of route
`m` succeed?"; `nothing` answers "will *any* remaining draw of *any* route succeed?" — a statement
about the whole remaining budget, not one step. Those come apart: every next-draw probability can
sit below threshold while the problem is still solvable within the remaining draws.

The Bellman state value `V(0)` is the utility-theoretic analogue of that same quantity, so we now
have two independent estimates of it and can cross-check them.

**But it is currently our worst head on the temporal split** — scout_next 0.8166, oss20_fresh
0.8265, oss120_fresh 0.7529, **nothing 0.7335**. If the oracle diagnostic confirms stopping is the
dominant lever, then the head governing stopping being the weakest is precisely the problem, and
Codex's proposal to train `q(s)` explicitly becomes the priority. (The 0.862 figure for this head
is from the *development* model, not the temporal one.)

### Headline correction

Codex reported the temporal saving as 9.1% / 11.0%. That is accurate for **one calibration-frozen
point at 70.41%**, which sits at the weakest part of the curve — and it is not comparable to the
myopic frozen arm, which lands at 67.37%. The matched-accuracy frontier bootstrap (171 problems,
1000 resamples, both frontiers rebuilt inside each resample) gives:

| target | RoR | ours (h2) | saving | 95% CI | P(>0) |
|---|---:|---:|---:|---|---:|
| 60.0% | $0.03396 | $0.02441 | **+28.1%** | [+7.6, +36.8] | 1.00 |
| 65.0% | $0.04877 | $0.03972 | **+18.6%** | [+1.8, +33.2] | 0.98 |
| 68.0% | $0.06570 | $0.05443 | +17.2% | [−3.0, +29.3] | 0.96 |
| 70.0% | $0.07699 | $0.06783 | +11.9% | [−9.4, +27.9] | 0.89 |
| 73.0% | $0.16584 | $0.14124 | +14.8% | [−9.2, +26.7] | 0.90 |

So the 27% was **not invalid** — it was a different operating point. The frontier is the comparable
number because it is what RoR reports.

The genuinely negative finding is a different one: **Bellman ≈ myopic.** The myopic frontier gives
+28.8 / +18.6 / +17.3 / +9.6 / +13.1 at the same targets. `h2` leads slightly at the top but well
inside the CIs. The pre-registered failure condition is essentially met — *the continuation value is
not a method contribution on its own*, and the savings versus RoR come from the learned prior plus
abstention. Whether learned transitions change that is what these jobs test.

---

## The model already encodes cross-route failure information (2026-08-30)

First **positive mechanistic** evidence for the transition story. Everything prior to this was an
argument from absence — a list of nulls plus a structural claim about what the analytic form cannot
express. This measures the signal directly.

Within-problem contrasts on the `sequential_value` arm, whose recorded `p_success_next` is raw model
output with no analytic decay applied. The target route's own failure count is held at 0, isolating
the cross-route effect.

| belief | driver | mean Δ | median Δ | \|Δ\|>0.05 | n |
|---|---|---:|---:|---:|---:|
| p[oss20] | oss120 failures | **−0.0568** | −0.0497 | 49.3% | 71 |
| p[oss120] | oss20 failures | **−0.0515** | −0.0416 | 41.7% | 96 |
| p[oss120] | scout failures | −0.0175 | −0.0129 | 6.6% | 121 |
| p[oss20] | scout failures | +0.0134 | +0.0001 | 16.5% | 121 |
| p[scout] | **own** failures | −0.0041 | | | 121 |
| p[oss20] | **own** failures | −0.0334 | | | 96 |
| p[oss120] | **own** failures | −0.0504 | | | 71 |

**Cross-route sensitivity is as large as own-route sensitivity.** oss120's belief falls 0.0515 when
*oss20* fails versus 0.0504 when *oss120 itself* fails. The analytic extrapolation
`p_m · s/(s + n_m)` decays on own-route counts only, so it models the second and discards the first:
roughly **half the belief movement the model actually produces is thrown away** before the DP sees
it. That is the quantity `--learned-transition-horizons` recovers.

The asymmetry is interpretable. Scout failures carry almost no cross-route information (−0.0175,
+0.0134) — scout is weak, so its failure is barely evidence. Expert failures are strongly
informative about other experts. A global scalar prior cannot express this, and RoR's belief update
has no channel for it: their `p̂_im` updates route `m` from route `m`'s own observations only.

### Correction to the standing "the model cannot learn the decay" diagnosis

That diagnosis was measured on the **scout** head and then generalized to the model. The bottom rows
reproduce it exactly — scout own-route sensitivity is −0.0041, flat. But the **expert** heads move an
order of magnitude more, and expert decisions are where the money is. The correct statement is
narrower: *the model is depth-insensitive for the cheapest route, and responsive for the experts.*

This also predicted the wrong thing earlier in the day. The reasoning was: learned transitions query
states differing from the root only in depth; the model is depth-insensitive; therefore
`raw(successor) ≈ raw(root)` and the arm is approximately a no-op. The first and third steps hold;
the second is false for exactly the routes that matter.

### Caveats

Beliefs **move** cross-route; this does not show the movement is well **calibrated**. The sign is
right (failures lower other routes' estimates), the magnitude is unverified. n is 71–121 problems
per contrast, and deeper driver-failure states within a problem are reached only along particular
trajectories, so some selection remains.

### Consequence for priority

Raises the expected value of the running `learned_trans` jobs and drops the `q(s)` / depth-balanced
training program to second place, pending their result. The `q(s)` argument is untouched by this
finding — it rests on the `nothing` head being the weakest (0.733) and on 43% of training gradient
coming from depths 1–2, which is a separate defect.

---

## Overnight batch: 13 jobs, three nulls and one dose-response (2026-08-31)

### 1. Oracle decomposition on the real learned prior

`sequential_decay` Bellman H=2, frozen R at 0.95 retention, temporal test (n=171):

| arm | correctness | mean cost | saving |
|---|---:|---:|---:|
| frozen baseline | 70.41% | $0.07057 | — |
| + oracle routing | 70.99% | $0.06845 | +3.0% |
| + oracle stopping | 73.10% | $0.02573 | **+63.5%** |
| + oracle both | 73.10% | $0.02337 | +66.9% |

Reproduces the counts-family CPU result on the real prior. **Routing is solved; stopping holds
essentially all remaining headroom**, and the two barely interact.

### 2. q(s) stopping fails, and the failure localizes the defect

| threshold | correctness | cost | abstention |
|---|---:|---:|---:|
| q ≤ 0.30 | 73.10% | $0.17140 | **0.0%** |
| q = 0.40 | 70.99% | $0.10522 | 14.6% |
| q = 0.50 | 69.24% | $0.08920 | 18.6% |
| value-rule baseline | 70.41% | $0.07057 | 29.6% |

**The learned q(s) never drops below 0.30**, so every threshold at or below that abstains on
nothing. Where it does abstain it is *strictly dominated* by the existing value stop — less accurate
and more expensive. Current beliefs recover **none** of the 63.5%.

The `nothing` head is therefore not merely unused, it is **worse than the rule we already have**. It
has usable rank-ordering (AUC 0.733) but a **compressed range**: it never becomes confident a problem
is doomed, though 83.8% of depth-10 states are. That is the concrete, localized defect.

Caveat: the grid topped out at 0.50, reaching 18.6% abstention against the baseline's 29.6%, so a
higher threshold is not formally excluded. Accuracy is already falling as the threshold rises
(70.99 → 69.24), so the trajectory argues against it.

### 3. Learned one-step transitions: null

| layout | mean ΔCost | mean ΔAcc | significant of 24 R |
|---|---:|---:|---|
| counts_last | +$0.00010 | −0.01pt | cost 1, acc 0 |
| problem_first | +$0.00020 | +0.04pt | cost 0, acc 0 |
| structured | −$0.00003 | +0.05pt | cost 2, acc 1 |

**Correction to the cross-route finding's interpretation.** That measurement compared states spanning
**≥2 accumulated failures** — an aggregate — and was then treated as a per-step effect when
predicting this arm would work. At H=2 the DP propagates exactly one step, and that marginal is too
small to flip decisions. The signal is real; its per-step magnitude is not enough.

### 4. Seed variance (5 seeds, counts_last, whole-pipeline seeding)

| seed | nothing AUC | frozen@0.95 |
|---|---:|---|
| 1 | 0.7329 | 69.01% / $0.06085 |
| 2 | 0.7313 | 67.95% / $0.05128 |
| 3 | 0.7454 | 67.72% / $0.04886 |
| 4 | 0.6648 | 66.67% / $0.04460 |
| 17 | 0.7335 | 67.37% / $0.04951 |

Accuracy sd 0.86pt; cost sd $0.00602 (**11.8% CV**), 36% min-to-max spread. Seed 17 — every previously
reported number — sits essentially at the mean, so nothing was cherry-picked. But an 11.8% cost CV
against an 18–28% claimed saving is material and must accompany the headline.

### 5. Rolling folds: savings scale with training-set size

Matched-accuracy savings versus RoR, point estimates:

| fold | train n | savings across the overlap |
|---|---:|---|
| 0 | 177 | +6.0, +0.7, +0.0, +4.6, −12.2 → **~null** |
| 1 | 357 | +12.9, +18.0, +12.2, −11.8, −6.3 → **mixed** |
| 2 | 536 | +23.5, +20.0, +12.0, +24.5, +7.3 → **clearly positive** |
| (551/341) | 551 | +18–28% |

The strict-temporal result **does not replicate uniformly**, but the pattern is not noise: savings
increase monotonically with training-set size. The rolling-origin design confounds *time period* with
*training-set size*, and this says the second dominates.

**This reframes the programme: we are data-limited, not method-limited.** It is consistent with
`best_calibration_epoch: 0` — the model overfits before completing one pass on 551 problems — and it
supplies a dose-response curve rather than a hunch.

### What survives

Every method addition attempted is null: Bellman ≈ myopic, learned transitions ≈ analytic, q-stopping
worse than the existing rule. The contribution remains **learned query-conditioned prior plus
abstention beats RoR**, now with the added, actionable result that the effect scales with training
data.

The planning machinery is not the bottleneck. The Bellman solve is exactly optimal *given the belief
model*; it produced nothing because the myopic rule was already near-optimal under those beliefs.
Three independent results say the beliefs are the binding constraint: lookahead makes the
count-prior family monotonically worse, learned transitions change nothing, and q(s) cannot separate
doomed from salvageable states at all.

---

## Data ceiling, dataset options, and the decay-rate question (2026-08-31)

### LiveCodeBench is exhausted

`livecodebench/code_generation_lite` holds six files, `test.jsonl`–`test6.jsonl`, **last modified
2025-06-05** — over a year stale. The full `code_generation` repo is worse (one file, frozen
2024-06). Coverage is 2023-05 → 2025-04, **1055 problems**. There is no v7 to wait for.

Exact accounting of what we use:

| | count |
|---|---:|
| available in release_v6 | 1055 |
| excluded by `--min-date 2023-09-01` in the collection script | 163 |
| eligible | 892 |
| **actually used** | **892** |

Nothing is lost to validity gates or collection failures. The 163 are excluded by a **date flag**,
nothing more, and they are the *earliest* problems — so they would go straight into training with
zero leakage risk and the temporal design would get strictly cleaner.

But re-reading the dose-response with spacing in mind, the curve is already flattening:

```
177 -> 357 (+180 problems):   ~0% -> ~14%    +14 pts
357 -> 536 (+179 problems):  ~14% -> ~17%     +3 pts
```

So 551 → 714 is +30% data on the flat part. Expect a few points, not a rescue. Worth collecting as
hygiene, not as a fix. (Held loosely: each fold has a different test set and one seed.)

### TACO as the scaling option

**[TACO](https://huggingface.co/datasets/BAAI/TACO): 26,443 problems** with hidden tests, compiled
from CodeContests + APPS plus other competitive sites. Same genre as LCB, **25× our data**.

The contamination objection applies to the *models* (TACO is 2023, our models are 2025-era), not to
the *router*. That suggests the design:

> **Train the router on TACO, evaluate on held-out LCB.**

If routing signal learned from contaminated problems transfers to clean ones, that is simultaneously
a direct test of the data-scaling hypothesis and a useful standalone claim: **router training data
need not be contamination-free — only evaluation does.** That would make a large stock of old
benchmarks usable as router training data.

**Gate before committing to a full collection**: if the scout solves most of TACO from memorization,
the difficulty distribution collapses and there is no routing problem left to learn. Measurable with
a few hundred scout draws.

### SWE-Smith launched

`launch_swe_smith_mdp_full_execution.sh` was fully prepared and never run; trace root and report root
both exist on disk (150 instances, all three routes, multiple rollouts). Launched as
`swe_smith_mdp_full_execution_seed17_1788198503`.

n=150 sits *below* the 177 where fold savings were ~0%, so expect it to be underpowered as a
headline. That is not what it is for: it tests whether the **decomposition** holds cross-domain —
does routing stay worth ~3% and stopping ~60% in agentic SWE? That structural claim does not need
large n, and it is the claim the reframed paper rests on.

### The decay rate: we match RoR exactly, and their sensitivity analysis may be masking heterogeneity

Confirmed from the RoR paper ([arXiv 2607.08665](https://arxiv.org/abs/2607.08665)):

| | prior `p̄` | decay rate `s` |
|---|---|---|
| **RoR** | global scalar, calibrated once on train (50% of queries) | **hand-set, s = 2** |
| **ours (to date)** | `p̄_m(x)`, learned per query | **hand-set, s = 2.0** |
| **factorized (running)** | `θ_m(x)`, learned per query | **`s_m(x)`, learned per query** |

RoR sweeps `s ∈ {0.5, 2, 8}` and reports ≤1.7pt change, concluding the policy "is insensitive to s
over an order of magnitude." We swept `s ∈ {0.5, 1, 2, 5, 10, 20}` and found a 0.8% spread — a clean
replication of their finding.

**But a global sweep cannot detect heterogeneity.** If persistence varies across problems — some
hopeless after one failure, others merely needing another draw — then every *global* value is a bad
compromise and the sweep looks exactly this flat. RoR's flat curve is therefore fully consistent with
strong per-query variation in `s`, and their sensitivity analysis cannot distinguish the two cases.

This is a sharper motivation for the factorized model than the one recorded at launch, and it gives
a claim with teeth: *global sensitivity analyses are not evidence a parameter does not matter.* We
can demonstrate the masking directly, since we reproduced their flat curve on our own pool.

### Launch note

Both new pipeline jobs first died in accelerate startup: a bare `accelerate launch` reads the user
default config, which names `/home/toolkit/icrl/configs/ds_config.json`, absent from the job image.
`launch_lcb_mdp_temporal_551_341.sh` already passed
`--config_file conf/accelerate/base_mp.yaml`, which is why the seed and fold jobs were unaffected.
Both launchers now match that invocation.

---

## Belief diagnosis, two retractions, and the ladder finding (2026-08-31)

### A: q(s) stopping — the defect was calibration, not representation

Per-(head, depth) Platt scaling on the calibration split turned a degenerate arm into a working one.
Uncalibrated, every threshold ≤0.30 abstained on **nothing**, because `q(s)` never fell below 0.30.
Calibrated, the arm sweeps 63.27% @ $0.026 to 73.10% @ $0.172 with abstention 0–37%.

So the `nothing` head was never broken: its ranking was usable (AUC 0.733), its **range** was
compressed by BCE against a base rate swinging 0.170 → 0.838 with 43% of the data at depths 1–2. We
were one step from concluding it could not represent doom, which would have been wrong.

**But it does not beat the value rule.** At matched accuracy, calibrated q-stopping costs ≈$0.0534
against the baseline's $0.05227 — about 2% worse. And it recovers **none** of the oracle's headroom:
at 73.10% it costs $0.17235 against the oracle's $0.02573; for the oracle's money it gets 10 points
less accuracy.

Conclusion: the stopping headroom is real but unreachable from current beliefs, and the binding
constraint is **belief accuracy, not calibration or how beliefs are consumed.**

### Learned one-step transitions: null, and a retraction

| layout | mean ΔCost | mean ΔAcc | significant of 24 R |
|---|---:|---:|---|
| counts_last | +$0.00010 | −0.01pt | cost 1, acc 0 |
| problem_first | +$0.00020 | +0.04pt | cost 0, acc 0 |
| structured | −$0.00003 | +0.05pt | cost 2, acc 1 |

**Retraction 1.** The cross-route measurement that motivated this arm compared states spanning **≥2
accumulated failures** — an aggregate — and was then treated as a per-step effect. H=2 propagates one
step, and that marginal is too small to move decisions.

### Seeds and folds

Five whole-pipeline seeds: accuracy sd 0.86pt, cost sd $0.00602 (**11.8% CV**), 36% min-to-max
spread. Seed 17 — the source of every reported number — sits at the mean, so nothing was
cherry-picked, but an 11.8% CV against an 18–28% claimed saving must accompany the headline.

**Retraction 2.** The claim that savings scale with training-set size does not hold. Held-out AUC is
flat and non-monotone across folds:

```
train n     scout    oss20   oss120  nothing
    177    0.5665   0.7901   0.7405   0.6772
    357    0.8124   0.7385   0.6989   0.6079
    536    0.7184   0.7872   0.7374   0.7053
    551    0.8166   0.8265   0.7529   0.7335
```

3× the data buys oss120 **+0.012 AUC**. The fold-to-fold savings differences were far more likely
**test-set differences** than a data-size effect. The dose-response recorded earlier is withdrawn.

### The model is roughly a difficulty score

A **3-way human difficulty label** beats our fine-tuned 8B on two of three routes:

| route | difficulty-only AUC | our model |
|---|---:|---:|
| scout | **0.8407** | 0.8166 |
| oss20 | 0.7945 | **0.8265** |
| oss120 | **0.7992** | 0.7529 |

Predicted heads are near-identical (Spearman 0.93–0.98) while true per-problem rates are not
(0.38–0.78).

### But that is largely correct behaviour — the ladder

LCB success patterns over 892 problems, thresholding per-problem rate at 0.5:

```
[1,1,1] 271   [0,1,1] 193   [0,0,1] 147   [0,0,0] 199    ladder-consistent = 810 (90.8%)
[1,0,1]  71   [1,0,0]   6   [0,1,0]   4   [1,1,0]   1    non-ladder        =  82 ( 9.2%)
```

**90.8% of problems are monotone in capability** (`scout ⊂ oss20 ⊂ oss120`). Under monotonicity, a
difficulty score plus per-route thresholds is the *correct* model — a rank-1 structure. So the
"collapse" is appropriate, not a defect.

**Retraction 3.** True ρ = 0.39 (scout↔oss120) was read as abundant route-specific signal. Scout
solves 30% and oss120 59%; two binaries with base rates that far apart have a correlation ceiling
well below 1 even under perfect nesting. Much of that 0.39 is a base-rate artifact. The honest
measure is the non-ladder fraction, **9.2%** — which is exactly why oracle routing is worth only 3%.

The causal story inverts: not "the model collapsed, so headroom is low", but **"the routes are nearly
a capability ladder, headroom is genuinely low, and the model learned that correctly."**

### Why the anti-collapse objectives already failed on SWE-Smith

`analysis/oss20_vs_oss120_per_route_success_accuracy.csv`:

```
route_classifier_cheapest_success_r32   OSS120 0.4819   OSS20 0.5134   <- chance
route_classifier_cheapest_success_r64   OSS120 0.4771   OSS20 0.4902   <- chance
joint_outcome_oversample_r32            OSS120 0.5751   OSS20 0.6288
plain per-route BCE (best, r128)        OSS120 0.7677   OSS20 0.7320   <- wins
```

Two causes, both of which also apply to LCB:

1. **Degenerate labels.** On SWE-Smith 62.6% of instances are all-fail (44%) or all-pass (19%);
   "cheapest successful route" is undefined on the former and constant on the latter. The learned
   solution is "say cheapest", visible in the mean predictions (OSS20 0.587, OSS120 0.413).
2. **Supervision loss.** Per-route BCE gives 4 labels per instance; the joint classifier gives 1 — a
   4× reduction on ~1500 instances.

LCB is somewhat better but not different in kind: decision-relevant fraction 47.3% versus SWE-Smith's
37.4%.

**Do not build the cross-route ranking objective.** It has been tried, it failed for reasons that
transfer, and the structure it targets is 9% of problems.

### Where this leaves the contribution

The pool is the confound. `scout` = Qwen3-4B, `oss20` = gpt-oss-20B, `oss120` = gpt-oss-120B — three
models, two families, monotone in size. A near-pure capability ladder **by construction**. RoR used 11
models across 8 lineages precisely to obtain diversity. So "routing is one-dimensional" is a claim
about *our pool*, not about routing, and must not be generalised.

That makes the honest next experiment: **measure routing headroom as a function of pool diversity.**
Qwen3-Coder-30B already appears in the SWE-Smith collection — a different family at a middling size.
Adding a genuinely off-ladder model converts "our pool happens to be a ladder" into "here is how pool
composition determines available routing headroom", which is a contribution with a knob rather than
an artifact.

---

## Complementarity is the constraint, and the decay constant is wrong by 5x (2026-08-31)

### The 15-model OpenRouter sweep answers the pool-diversity question

`openrouter_sweep_collect_1785216501` (14 Daytona-graded models, 9 families, 143 common SWE-Smith
instances) was already collected and analysed, and it settles the question directly.

**Size-matched, our capability-ladder pool is *more* complementary than diverse ones:**

| 3-model subsets of the diverse pool | ladder-consistency |
|---|---|
| mean | 95.4% |
| median | 95.8% |
| range | [88.1, 100.0] |
| **our LCB 3-model pool** | **90.8%** |
| diverse subsets at or above ours | 95.9% |

Deliberate family diversity produces *more* ladder structure, not less: weaker models are dominated
rather than complementary.

**And the headline number:**

```
                      observed union   best single   if independent
14 models, 9 families      44.8%          40.6%          98.8%
LCB, our 3 models          77.7%          76.5%          93.2%

realised complementarity   SWE-Smith 11.2% (mean over 3-model subsets)
                           LCB        7.4%
                        -> models are ~89% REDUNDANT relative to independence
```

Fourteen models across nine families beat the best single model by **4.2 points**; independence
predicts 98.8% union. Two independent domains agree at ~7-11%.

**LLM failures are overwhelmingly shared.** This retires the pool-diversity hypothesis, explains why
oracle routing is worth 3% while oracle stopping is worth 63.5%, and converts every null recorded
today into evidence for a single thesis rather than a list of failures.

Caveats to state, not bury: n=143 common instances; sweep models are mid-tier (2.8-40.6% solve), so a
frontier pool may differ; 3-model subset statistics are combinatorially dependent.

### The decay constant is mis-specified by ~5x, in RoR and in us

Measured on LCB, P(oss120 succeeds next) after n **own-route** failures, as a ratio to baseline:

```
model                     ratio at n=0..4                            SSE
true                      1.0    0.2715 0.1037 0.0759 0.0501          -
Beta-Bernoulli s=2.0      1.0    0.6667 0.5000 0.4000 0.3333       0.4986   <- RoR default AND ours
Beta-Bernoulli s=0.5      1.0    0.3333 0.2000 0.1429 0.1111       0.0213   <- our sweep FLOOR
Beta-Bernoulli s=0.37     1.0    0.2701 0.1561 0.1098 0.0847       0.0051   <- best
two-component mixture     0.99   0.2897 0.0274 0.0021 0.0002       0.0142
```

`s = 2.0` fits 100x worse than `s ≈ 0.37`, and 0.37 lies **below our sweep floor of 0.5**. That is why
`counts_s0.5` was the best RoR arm in every table today — pinned at the grid boundary, the classic
sign the optimum is outside the grid, which was never followed up.

Our `sequential_decay` arms default `decay_pseudo_count` to the same 2.0, so **our own method has been
under-decaying too** — over-continuing on doomed problems, which is exactly the stopping failure the
oracle puts at 63.5%.

A two-component solvable/hopeless mixture was also fitted and is *worse* than a well-tuned
Beta-Bernoulli (SSE 0.0142 vs 0.0051), so the functional form is fine; only the constant is wrong.

### Cross-route belief updating: rejected before building it

P(oss120 succeeds next) conditioned on **scout** failures: 0.7552, 0.6137, 0.5992, 0.5868, 0.5795.

The first scout failure is worth −0.14, and it then **saturates at ~−0.01 per additional failure**.
Under `scout_first` every reachable state has at least one scout failure, so that evidence is
*constant across every state the router observes* and carries no discriminative signal. Own-route
evidence is 4x stronger on the first observation (−0.55). Do not build cross-route updating.

### Consequence for the method

The chain is now: failures are shared (89% redundant) -> routing is worth 3%, stopping 63.5% ->
stopping is governed by the belief-decay rate -> that rate is a hand-set global constant, wrong by
5x, and a global sweep cannot fix per-problem heterogeneity -> **learn it per problem**, which is
exactly the running factorized experiment.

---

## "Scout before you route": the probe result, and what it does and does not explain (2026-08-31)

### The probe claim, and the confound check it survives

| predicting oss120 success, LCB corrected temporal | AUC |
|---|---:|
| problem text alone | **0.5521** |
| after one scout draw | **0.7693** |

Verified directly from `eval_predictions.jsonl` in both runs: **identical instance sets** (339 ids,
set-equal), **identical labels**, same 66.1% positive rate. The comparison is clean — post-scout is
not measured on an easier surviving subset.

Cost of the probe: scout `$0.000553` against oss120 `$0.029725` = **1.86% of one expert call.**

This is a direct challenge to the dominant paradigm. RouteLLM/RouterBench-style routers predict from
**query text**, which here is near-chance. Running the cheap model and reading its attempt is what
carries the signal.

The mechanism comes from our own measurements: failures are ~89% redundant, so a *cheap* probe is
informative about *expensive* routes. Under independence a scout failure would say nothing about
oss120. Shared failure is what makes probing work — measurement (1) explains measurement (2).

**Status: one seed.** Three-seed replication launched (`SEED` was never plumbed through
`launch_lcb_abstention_train.sh`; the trainer accepted `--seed` but the launcher dropped it). Nothing
should be built on this until it replicates.

### Does +0.22 AUC drive the gains over RoR? Only partly, and here is the accounting

Two reasons the AUC edge does not convert proportionally.

**1. Ranking is not thresholding.** At comparable abstention rates:

| | abstains | accuracy | cost |
|---|---:|---:|---:|
| oracle stopping | 26.9% | 73.10% | $0.02573 |
| ours (calibrated q) | 30.1% | 68.89% | $0.05529 |

The oracle abstains on exactly the doomed problems and loses nothing. We abstain on a similar
fraction, overlap, and pay 4.2 accuracy points. AUC 0.77 is good *ranking*; abstention is a
*threshold* decision, which overlap punishes far more.

**2. RoR's budget is already a crude stopping rule.** Algorithm 1 exits on budget exhaustion, so on a
doomed problem RoR spends exactly `B` and halts. Our advantage is not "we stop and they never do" but
"we stop earlier and selectively, they stop at `B` unconditionally" — which is bounded by `B`.

### Savings versus budget: non-monotone, peaking mid-range

```
  budget B  RoR acc   RoR cost   our cost   saving
   0.00621   47.25%    0.00555    0.00443   +20.1%
   0.02962   57.54%    0.02628    0.01795   +31.7%
   0.03548   57.89%    0.02873    0.01869   +35.0%   <- peak
   0.05304   57.89%    0.02873    0.01869   +35.0%
   0.10177   70.41%    0.07930    0.07057   +11.0%
   0.13880   71.46%    0.09350    0.09031    +3.4%   <- trough
   0.24989   73.10%    0.17323    0.14742   +14.9%
```

**Retraction 4.** The earlier claim that larger `B` widens our margin is wrong. At `B = $0.25` RoR's
*realised* cost is only $0.17323: above ~$0.17 the budget stops binding because RoR runs out of draws
(10 per route) before it runs out of money, so its waste is capped by draw availability, not `B`.

What actually drives the pattern is **granularity, not budget size**. RoR is steppy and we are smooth:
budgets 0.03548–0.05304 give *identical* RoR results (57.89% @ $0.02873) because its next escalation
step is not yet affordable, while a continuous value rule interpolates through that plateau. The
trough at 0.1388 is where a RoR step happens to land efficiently.

That is a better claim than "bigger budget helps us" — it is a mechanism, and mid budgets are the
deployment-relevant regime.

Caveat: computed with the mis-set `s = 2.0` decay; the corrected sweep may flatten the peak if a
better-tuned RoR fills its own plateaus.

### The contribution, restated

> **Scout before you route.** Query-text routing is near-chance on competitive programming
> (AUC 0.5521). Because model failures are ~89% redundant, a single draw from the cheapest model —
> 1.86% of an expert call — raises solvability prediction to AUC 0.7693. And because pools are
> capability ladders (90.8% consistent), that information is worth **3% for selecting a model** and
> **63.5% for deciding whether to continue at all**.

Three measurements, one causal chain, an actionable rule, and no dependence on beating RoR.

### What SeqRoute contributes, and what it does not

[SeqRoute](https://arxiv.org/html/2605.25424): binary action space (weak 8B / strong 70B),
**single-use per query — no resampling**, **no abstention**, **no verifier or early stopping**.
Budget is global across a *multi-turn session with unknown future turns*; the contribution is delayed
gratification.

|  | resampling | abstention | verifier | budget |
|---|---|---|---|---|
| SeqRoute | no | no | no | global, within a session |
| RoR | yes | no | yes | per query |
| ours | yes | yes | yes | per query |

The unoccupied cell is **batch budget + verifier + resampling + abstention**. The importable lesson is
the *setting*: budget should be global and the decision is allocation across queries. That also
dissolves the mid-budget artifact — under a batch budget you allocate continuously across queries, so
RoR's step-size plateaus stop being what our advantage is made of.

Formulation: **knapsack with purchasable information.** N queries, one budget, unknown success
probabilities, and the option to pay ~2% to sharpen the estimate of any query before allocating the
rest. Every measurement above is a parameter of it.

### Dropped

Factorized/learned-decay (B), learned transitions, deeper Bellman horizons, cross-route belief
updating, and anti-collapse objectives. All null, confounded, or already refuted on SWE-Smith, and
none on the critical path.

---

## Single-draw benchmarks inflate measured complementarity ~2.6x (2026-08-31)

### The measurement

Same models, same problems, same metric; only the estimator changes.

| draws used | complementarity | union | best single |
|---:|---:|---:|---:|
| 1 | **19.3%** | 78.8% | 75.5% |
| 3 | 10.5% | 79.0% | 77.3% |
| 10 | **7.4%** | 77.7% | 76.5% |

(`k=2` and `k=4` are omitted: the "majority of k draws" rule is ill-defined at even k and
depresses union and best-single together, so those rows are not comparable.)

### Why it matters

**Every routing benchmark is single-draw** — RouterBench, CodeRouterBench, RouterEval all record one
response per (query, model). So every published complementarity and routing-headroom figure built on
them inherits this inflation, and **it cannot be corrected from those datasets**, because the repeat
draws do not exist in them.

The mechanism: with one draw you cannot separate *"model A reliably solves this"* from *"model A got
lucky here"*. Both present as specialisation. A measurable fraction of reported complementarity is
therefore sampling noise wearing a specialisation costume.

**Falsifiable consequence:** a router that exploits single-draw complementarity is partly exploiting
noise, so its gains should shrink on redeployment — on a fresh draw the apparent specialist may
simply fail. That is a concrete account of why routers underdeliver in practice.

### It reframes the disagreement with the literature productively

The earlier framing — "routing headroom is small, contra the field" — is combative and partly wrong.
The better claim is:

> Routing headroom is measured on single-draw benchmarks and is inflated ~2.6x by sampling noise.
> The denoised headroom is what a deployed router can actually capture.

We are not contradicting their numbers; we are supplying a correction factor they had no way to
compute. **We have multi-draw data and the benchmarks do not.** That is uniquely ours.

Consequence for CodeRouterBench: its 85.9%/93.3% figures are upper bounds. Denoised, the peer pool
plausibly lands in the 30s — still far above our ladder pool's 7.4%, so the pool-structure effect
survives the correction.

### Retracted along the way

The claim that claude-opus's 35.0% on CodeRouterBench's LCB tasks was a harness artifact. Our own
independent pipeline puts claude-opus-5 at **43.0%** (n=100) on LiveCodeBench, and oss-120b at 59.1%.
Two independent harnesses agree an Opus-class model underperforms mid-tier Qwen models on these
tasks, so the score matrix is measuring capability, not extraction failure. The price/capability
inversion is real.

### The cheap peer pool

Searching all subsets on the 1,869 LCB-sourced CodeRouterBench tasks:

| k | comp | gain | union | pool $ | spread | models |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 93.3% | +19.6pt | 81.0% | $0.0030 | 2.9x | Qwen3-Max(61%) qwen3.5-plus(54%) |
| 3 | 84.3% | +24.0pt | 85.3% | $0.0087 | 7.3x | + glm-5(42%) |

Our current pool costs **$0.0342** to run once and reaches union 77.7%. The 3-model peer pool costs
**$0.0087** — 4x cheaper — reaches union 85.3%, and is several times more complementary even after
denoising. The expensive models are the ones to drop: claude-opus costs 39x Qwen3-Max and scores
worse on these tasks.

---

## Plan: swap to a peer pool, gated on denoised complementarity (2026-08-31)

### Why a pool swap rather than more method work

Every method addition tried today is null: Bellman ≈ myopic, learned transitions ≈ analytic,
q-stopping ≈ value rule, factorized decay (invalid as run, relaunched), cross-route updating refuted
before building, anti-collapse objectives already at chance on SWE-Smith.

The common explanation is structural rather than methodological: our pool is a capability ladder
(90.8% consistent), so route ordering lies on a one-dimensional difficulty manifold, the continuation
term cannot differentiate actions, and the DP degenerates to "climb, then stop". **Planning has no
content when there is nothing to plan over.**

A peer pool is the one condition under which the planning machinery could pay, and it is the single
alternative explanation not yet ruled out.

### The probe gets better, not worse

|  | probe model | solve | cost rank | capability rank |
|---|---|---:|---|---|
| ladder pool (current) | Qwen3-4B | 27.8% | cheapest | **weakest** |
| peer pool (proposed) | Qwen3-Max | 61.3% | cheapest | **strongest** |

On the peer pool the natural probe is simultaneously cheapest and best. It costs $0.0008, carries
essentially no opportunity cost — you would run it regardless — and its *failure* is the event that
triggers routing among the complementary remainder, which is where the +24 points sit. "Scout before
you route" becomes *probe with your best cheap model and route only its failures*, which is a
stronger claim than probing with a weak 4B.

### Design

```
pool     Qwen3-Max ($0.0008, 61.3%)  qwen3.5-plus ($0.0023, 54.4%)  glm-5 ($0.0056, 42.3%)
tasks    ~600 LCB-sourced (statements and grader already in hand)
draws    10 each  =~ 18k generations  =~ $55
```

Pool cost $0.0087 per full sweep against our current $0.0342 — **4x cheaper**, union 85.3% against
77.7%.

Everything already built is pool-agnostic: replay, oracle arms, Bellman horizons, q-stopping,
paired bootstrap, matched-accuracy frontier. Only the tensors change.

### Staged gates, each conditioning the next

1. **Denoised complementarity.** Does CodeRouterBench's 85.9% survive multi-draw estimation? Our
   correction factor predicts the 30s; it could land near our 7.4%. **If it does, stop** — the
   pool-structure story is dead and the ladder paper is what we have.
2. **Oracle decomposition.** Does routing 3% / stopping 63.5% shift? This is the load-bearing claim
   and the peer pool is its hardest test.
3. **Probe value.** Is best-and-cheapest-as-probe worth its ~2%?
4. **Bellman.** With problem-dependent route ordering finally present, does planning pay?

Gate 1 is cheap and can kill the line, which is the property today's experiments lacked.

### Pre-registered failure condition

If gate 1 passes and gates 2-4 come back null again, the conclusion is that **belief quality — not
pool structure and not planning — was the binding constraint throughout**, and the paper is the
measurement work: complementarity, the single-draw inflation correction, the oracle decomposition,
and the probe result. Estimated at ~40% likely.

### Checks before spending

- Confirm Qwen3-Max, qwen3.5-plus and glm-5 are available on OpenRouter under usable model ids.
- Confirm CodeRouterBench's LCB task ids map onto LiveCodeBench problems our grader accepts.

---

## What CodeRouterBench does and does not report, and why 2.6x is a floor (2026-08-31)

### The source

CodeRouterBench is the artifact released with **Agent-as-a-Router / ACRouter**
([arXiv 2606.22902](https://arxiv.org/abs/2606.22902), Zhou et al.): ~10K task instances, 8 frontier
LLMs, a complete task-by-model outcome matrix, used for regret-based router comparison.

### The gap

Checked against the paper directly:

- **No decoding configuration is reported** — no temperature, no top_p.
- Scoring is **pass@1**, one generation per (task, model).
- **Variance is never discussed.** All routing comparisons treat a fixed outcome matrix `O` as
  ground truth.

So the critique does not depend on knowing what they did, which is the strong form:

> A benchmark whose entire artifact is a per-(task, model) outcome matrix reports neither its
> generation configuration nor any variance estimate. Routing headroom measured on it cannot be
> separated from sampling noise — by construction, since no repeat draws exist.

This is a statement about what the data can support, not an accusation about their protocol.

### Our 2.6x is a lower bound, not an estimate

Our multi-draw LCB collection runs at **temperature 0.2** (`TEMP_PRIMARY`), and the corrected
baseline collection at **temperature 0.0** — literally greedy. So the 19.3% -> 7.4% inflation was
measured under *near-deterministic* decoding. At the temperatures model cards recommend (Qwen ~0.7;
DeepSeek explicitly warns greedy degrades reasoning models through repetition) the effect can only be
larger.

That closes the "but they may have used greedy" objection empirically rather than by argument.

### Retraction 5: the price/capability inversion is greedy-conditional

Earlier this session the claim that claude-opus's 35.0% was a harness artifact was retracted on the
grounds that our own pipeline independently gives claude-opus-5 **43.0%** on LCB. But **our number is
also greedy** (temperature 0.0). Two greedy harnesses agreeing tells us about greedy decoding, not
about capability.

Several of CodeRouterBench's eight models are reasoning-capable, and greedy decoding is known to
degrade such models. So a third explanation is live and untested: **both harnesses may be measuring
degraded behaviour rather than capability.**

Consequently the price/capability inversion — Qwen3-Max at 61.3% beating claude-opus at 35.0% while
costing 39x less — is established **only under greedy decoding**. That is materially weaker than the
claim made earlier, which treated it as a property of the models.

It does not affect the peer-pool gate: gate 1 collects at temperature 0.2 with four draws, so it
measures the pool under our own known configuration either way.

---

## Status after the corrected-s sweep, and the open threads (2026-08-31)

### The method contribution is real; the earlier "too thin" framing was wrong

Two components, both architecturally distinct from RoR rather than tuning:

- **Abstention.** RoR's Algorithm 1 exits only on verified success or budget exhaustion; it
  *structurally cannot* give up. Largest measured effect we have: abstention on/off was 8.45% ->
  51.5% on the development split.
- **Learned query-conditioned prior.** RoR's `p̄_m` is a global scalar, so its per-query
  discrimination is AUC 0.5 **by construction**. Ours is ~0.75.

**And the margin survives correct baseline tuning**, which was the check most likely to sink it.

| accuracy | best-tuned RoR | ours | saving |
|---:|---:|---:|---:|
| 60% | $0.02509 (s=0.3) | $0.01911 | **+23.8%** |
| 65% | $0.04677 (s=0.3) | $0.03611 | **+22.8%** |
| 70% | $0.07228 (s=0.3) | $0.06946 | +3.9% |
| 73% | $0.16370 (s=0.3) | $0.13123 | **+19.8%** |

The empirical optimum is **s = 0.3**, close to the 0.37 predicted from the decay fit, and well below
RoR's published default of 2.0 — which was costing them ~26% at the cheap end.

Components are not individually novel (learned per-query scoring is the routing literature;
abstention exists in deferral and cascade work). What is plausibly unoccupied is the combination:
**sequential multi-draw resample/reroute with a learned per-query prior and a real give-up action.**
RoR is the nearest neighbour and lacks the third.

The concern is **robustness, not originality**: 11.8% seed CV, folds that do not replicate uniformly,
one pool, one domain, and a 70% operating point that nearly vanishes.

### Measurement and method are one paper, not two

They were being separated artificially. The measurement *explains* the method:

> **Method:** sequential routing with a learned prior and abstention, +20-24% over correctly-tuned RoR.
> **Explanation:** it works through *stopping*, not selection — perfect routing is worth 3% and
> perfect stopping 63.5%, because model failures are ~89% redundant.

The measurement makes the method credible rather than lucky; the method gives the measurement teeth.

### Frozen probe: LoRA earns its keep

| | scout | oss20 | oss120 | nothing |
|---|---:|---:|---:|---:|
| frozen encoder, head only | 0.7900 | 0.7069 | 0.6606 | 0.6614 |
| LoRA baseline | **0.8166** | **0.8265** | **0.7529** | **0.7335** |

LoRA wins on all four heads, so the `best_epoch: 0` overfitting is **not** LoRA memorising — the
frozen features genuinely carry less. This removes "simplify the method" as an option and rules out
one explanation for the belief-quality ceiling.

### Open thread 1: does more data reduce seed variance?

Our 11.8% cost CV is **pure training variance** — all five seeds share one test set, so none of it is
test-set noise. That is the component more data could fix.

Evidence points both ways:
- **For:** `best_calibration_epoch: 0` everywhere is the signature of overfitting before one pass,
  i.e. a model under-constrained by its data, which is exactly when seed variance is large.
- **Against:** held-out AUC is flat in training size (177 -> 551 buys oss120 **+0.012 AUC**). A model
  whose mean does not move with data is not obviously variance-limited by data either.

**Gate before spending on TACO:** train 3 seeds on **half the data (275 problems)** with the same
split and test set, and compare the spread against the 5 seeds at 551.

```
cost CV at 275 >> 11.8%  -> variance shrinks with data -> TACO helps
cost CV at 275 ~= 11.8%  -> variance is capacity- or task-driven -> TACO will not fix it
```

Three jobs, no collection, no API spend.

### Open thread 2: TACO sizing

Budget is available, so the full MDP is viable. Two separable price tags:

- **Abstention predictor**: needs one draw per problem. Cheap; TACO could expand its training set a
  lot for little money.
- **Full sequential MDP**: needs multiple draws per route per problem, and cost scales with draws.

Draw-count evidence already collected argues for shrinking the budget rather than defaulting to 10:
**scout draws 5-10 are worth 0.59 points**, **oss20 draws 2-10 are worth 0.00 points** (fully
subsumed, though they retain cost value at $0.004 against $0.030), and only **oss120 1->10 is worth
6.16 points and still climbing**. An asymmetric allocation — few scout and mid-tier draws, more on
the top tier — is the efficient shape.

### Open thread 3: clean-B, still unread after three attempts

The learned-decay experiment has been blocked three times by scaffolding rather than by its
hypothesis: (1) a bare `accelerate launch` picking up a stale deepspeed config, (2) double-decay plus
problem-text-only input, (3) the double-decay guard firing unconditionally because `families` always
contained `sequential_decay`. Now relaunched as `lcb_mdp_factorized_seed17_1788211558`. **There is
still no clean read on whether a learned per-problem decay beats the hand-set constant.**

---

## 2026-09-01: three gates resolved

### Gate 1 answered: seed variance does NOT shrink with data

Three seeds trained on **half the problems (275)**, same split, same test set, scored on the same
`sequential_decay_value_frozen @ retention 0.95` point as the five full-data seeds.

| training set | n seeds | accuracy | cost | **cost CV** | min-max spread |
|---|---:|---|---|---:|---:|
| 551 problems | 5 | 67.74% (sd 0.86pt) | $0.05102 (sd $0.00602) | **11.8%** | 36% |
| 275 problems | 3 | 68.85% (sd 0.55pt) | $0.05865 (sd $0.00450) | **7.7%** | 16% |

The pre-registered gate was `CV at 275 >> 11.8% -> TACO helps`. We got **7.7%**, i.e. the point
estimate moves the *wrong* way. With n=3 against n=5 the variance ratio (1.79, df 2 and 4) is
nowhere near significant, so the honest claim is not "less data is better" — it is that **there is
no evidence seed variance is data-limited**, and the direction of the point estimate rules out the
optimistic reading.

Half-data accuracy is also *higher* than full-data (68.85 vs 67.74; every half seed beats the full
mean). Taken with the already-recorded flat AUC-vs-training-size curve (177 -> 551 buys +0.012),
the consistent picture is that **551 problems is already past saturation for this recipe**. The
11.8% CV is capacity- or task-driven, not sample-driven.

**Consequence for TACO:** the "more problems -> tighter router" argument is dead. TACO is still
defensible as a *second dataset* — external validity, does the method transfer off LiveCodeBench —
but it should not be sold, or budgeted, as a variance fix. That is a much smaller collection than
the full MDP at 10 draws.

### Gate 2 answered: the decay is learnable and per-problem, and it buys nothing

`lcb_mdp_factorized_seed17_1788211558` — fourth attempt, first clean read.

Per-head test AUC, factorized (learned `theta_m` and `s_m` from problem text) vs the LoRA baseline:

| head | factorized | LoRA baseline |
|---|---:|---:|
| scout_next | **0.8312** | 0.8166 |
| oss20_fresh | **0.8454** | 0.8265 |
| oss120_fresh | **0.7852** | 0.7529 |
| nothing | 0.7199 | **0.7335** |

The parameterization **improves every per-route head** and **loses on `nothing`** — the stopping
head. The policy frontier follows the stopping head, not the routing heads:

| matched accuracy | baseline (hand-set decay) | factorized (learned decay) | saving |
|---:|---:|---:|---:|
| 36.80% | $0.00235 | $0.00255 | **-8.7%** |
| 45.87% | $0.00410 | $0.00410 | -0.1% |
| 54.95% | $0.01064 | $0.01369 | **-28.6%** |
| 64.02% | $0.03445 | $0.03321 | +3.6% |
| 73.10% | $0.15019 | $0.16149 | **-7.5%** |

Learned per-problem decay does not beat a single hand-set constant. (The baseline frontier is built
from more families, which favours it; even granting that, there is no coherent gain.)

**But the recovered parameters are the interesting part.** Inverting `p_m(n) = theta_m*s_m/(s_m+n)`
from consecutive-count prediction pairs on the test split:

| route | learned s (median) | IQR | implied theta_m |
|---|---:|---|---:|
| scout | 0.564 | [0.27, 0.88] | 0.428 |
| oss20 | 0.682 | [0.20, 1.08] | 0.386 |
| oss120 | 0.702 | [0.19, 0.99] | 0.542 |

Three *independent* estimates of the decay now agree that it is far sharper than RoR assumes:

- RoR's default (and our hand-set value): **s = 2.0**
- s that makes RoR optimal on our data, by grid search: **s = 0.3**
- s learned end-to-end from problem text alone: **median 0.56-0.70**

And the wide IQR shows the model does learn genuine *per-problem* variation — answering the earlier
"would per-problem s be a contribution?" question empirically: **it is learnable, it corroborates
the sharp-decay finding, and conditioning the policy on it is worth nothing.**

This is now the third independent confirmation of the same thing. Oracle decomposition put routing
at +3.0% of headroom and stopping at +63.5%. Cross-route belief updating saturated after the first
failure. And now a strictly better belief model — better on all three routing heads — produces no
better policy, because it is slightly worse on the one head that governs stopping. **Belief
refinement is saturated; only stopping carries value.**

### Gate 3 (peer pool): complementarity survives resampling in a PEER pool

`lcb_peerpool_gate1_eval_1788210194`, 341 held-out LCB problems, temp 0.2.

**One model is void.** `qwen/qwen3.5-plus-20260420` returned **empty output on 336/341 problems**
with median `completion_tokens` 4098 against a 4096 cap — a thinking model that spent its entire
budget reasoning and emitted nothing visible. Its 1.6% pass@1 is a harness bug, not a capability
measurement, and its complementarity with everything is exactly 0.0%. It must be re-collected with a
larger cap and reasoning-field capture before it means anything. The valid reading is glm5 + qmax.

| model | draws | pass@1 per draw | mean |
|---|---:|---|---:|
| qwen/qwen3-max | 4 | .469 .457 .466 .419 | 0.453 |
| glm5 | 3 | .443 .405 .405 | 0.417 |
| q35p (void) | 3 | .012 .026 .009 | 0.016 |

Realized complementarity, glm5 + qmax:

| draws per model | best single | observed union | independent union | realized comp |
|---:|---:|---:|---:|---:|
| 1 | 0.4692 | 0.5601 | 0.7077 | **38.7%** |
| 2 | 0.5103 | 0.5982 | 0.7631 | ~35% |
| 3 | 0.5455 | 0.6334 | 0.7955 | **36.3%** |

(k=1 across the three individual draws: 38.1 / 32.1 / 33.5%, mean 34.6% — so the k=1 to k=3 movement
is within single-draw noise.)

**This is flat in the number of draws.** Our ladder (scout/oss20/oss120) collapses 19.3% -> 10.5% ->
7.4% over the same sweep. The peer pool does not collapse at all.

**This forces a correction to the single-draw-inflation claim.** The earlier framing —
"CodeRouterBench's complementarity is inflated because it takes one draw per cell" — is too strong.
Inflation is a property of **ladders**, not of benchmarking with one draw. In a ladder the weak
model's successes are close to a subset of the strong model's, so resampling the strong model
recovers most of what the weak one uniquely solved, and apparent complementarity evaporates. Between
**peers of similar strength** the disjoint region is real and resampling does not recover it.
CodeRouterBench pools eight frontier peers, so its numbers are in the regime where the critique does
*not* bite. The defensible version of the finding is the conditional one: *complementarity measured
at one draw is only meaningful when the pool is not a ladder, and every cost-routing pool is a
ladder by construction.* We cannot test CRB's specific 93.3% qmax+q35p figure until q35p is
re-collected.

**Net:** the peer-pool line lives (36% >> 7.4%, and stable), but the "CRB complementarity is fake"
paper idea does not survive in its strong form.

### Why the better learned decay buys nothing — mechanism (2026-09-01)

**Correction first.** The initial reading blamed the `nothing` head. That was wrong: `value_stop`
fires on `max(action_values) <= 0`, computed from per-route `p_m`, costs and `R` alone — the
`nothing` head is read *only* by the explicit `qstop` arms. Rebuilding both frontiers with every
`nothing`-head arm deleted changes the comparison by <2pt anywhere; factorized stays uniformly
3-11% worse. The stopping rule never touches that head.

**The per-problem decay is real.** Splitting test problems at the median learned `s` and measuring
the *ground-truth* hazard `p(n) = P(draw n+1 succeeds | first n failed)`:

| route | p(1)/p(0), low-s group | high-s group |
|---|---:|---:|
| scout | 0.000 | 1.549 |
| oss20 | 0.220 | 0.795 |
| oss120 | 0.417 | 1.083 |

Low-`s` problems genuinely decay faster. The model is learning something true. (Small n and a
partial confound with `theta` — low-`s` problems are also harder at n=0 — so this is directional,
not tight.) The factorized model is also better than the baseline on **every** per-route head, on
both ranking (AUC .831/.845/.785 vs .817/.827/.753) and **calibration** (ECE .087/.135/.122 vs
.172/.213/.231). It is a strictly better belief model by every measure we have.

**And it still cannot pay, for a structural reason.** Under the myopic buy rule, route `m` is worth
another draw while `theta*s/(s+n) >= c_m/R`, so the stopping depth is

```
n* = s * (theta*R/c_m - 1)
```

Two consequences, both fatal to the per-problem decay:

1. **Whether you buy a route at all depends on `theta`, never on `s`.** The sign of
   `(theta*R/c_m - 1)` has no `s` in it. At R=$0.05, **49% of problems never buy oss120 at any
   depth** — and `s` cannot flip a single one of those.
2. **Where `s` does move `n*`, it moves it only where depth is worthless.** Fraction of problems
   whose `floor(n*)` changes between the learned per-problem `s` and a single constant:

| R | scout | oss20 | oss120 |
|---:|---:|---:|---:|
| 0.05 | 98% | 50% | **0%** |
| 0.08 | 98% | 82% | **11%** |
| 0.12 | 99% | 86% | **21%** |

   Scout: mean `n*` is 26 vs 24 — but only 10 scout draws exist, so both mean "exhaust the route."
   oss20: a genuine 4.6-vs-3.0 change — on the route where **draws 2-10 are worth 0.00 accuracy
   points**. oss120, the only route where depth pays (1->10 is worth 6.16 points): mean `n*` is
   **0.07 vs 0.06**, both flooring to zero.

**The expensive route is priced so you get one shot regardless, so the decay never gets to act on
it.** All the money rides on "is oss120 worth even a single call on this problem," which is a
`theta` question. The decay is a `theta`-independent multiplier on depth, and depth is free on
scout, worthless on oss20, and clipped to zero on oss120.

This also reconciles the two decay results that looked contradictory:

| treatment | cost at matched accuracy vs no decay |
|---|---|
| no decay (s = inf) | — |
| **constant s = 2.0** | **+12% to +30% (positive 5/6 points)** |
| learned per-problem s | -2% to -13% (negative at every point) |

*Having* a decay is worth a lot, because it changes the shape of the belief with depth for
everything at once. *Tuning* the decay per problem is worth nothing, because per-problem shape only
matters on a route whose depth is already clipped. The response surface in `s` is steep at
`s = infinity` and flat everywhere near the optimum.

**Implication for the paper.** This is the fourth independent arrival at the same place: routing is
+3.0% of oracle headroom and stopping +63.5%; cross-route updating saturates after one failure; a
strictly better belief model yields a worse policy; and now the decay is shown to be structurally
unable to touch the decision that carries the money. The lever is **`theta` on the expensive route**
— predicting whether the top model can solve this problem at all — not the belief dynamics. Every
attempt to buy performance by improving the sequential belief has now failed, for a reason we can
finally state precisely.

### Would more draws (or hotter draws) give depth meaning? Measured: no (2026-09-01)

**Retraction 5.** The recorded claim that *"oss120 1->10 is worth 6.16 points and still climbing"*
is wrong. On the 171-problem test split:

| k | scout | oss20 | oss120 | union | marginal oss120 |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.07% | 38.01% | 57.89% | 61.99% | — |
| 2 | 30.99% | 47.95% | 65.50% | 67.25% | +7.60pt |
| 4 | 32.16% | 52.63% | 69.59% | 70.76% | +1.75pt |
| 6 | 32.16% | 54.39% | 71.93% | **73.10%** | +1.75pt |
| 7 | 32.16% | 54.39% | 71.93% | 73.10% | **0.00pt** |
| 8 | 32.16% | 55.56% | 71.93% | 73.10% | **0.00pt** |
| 10 | 32.16% | 55.56% | 71.93% | 73.10% | **0.00pt** |

oss120 **saturates completely at k=6**. Draws 7-10 add exactly zero. Same on the full 892 (draws 8
and 10 both add 0.00pt). RoR-style deep resampling has nothing to buy here.

**Temperature is not the cause.** We already collected a 10-draw scout arm at T=0.6 alongside T=0.2
(`lcb_multidraw_scout_1787547502`), 341 shared problems:

| | pass@1 | pass@10 | gain | mean distinct outcomes/problem |
|---|---:|---:|---:|---:|
| scout T=0.2 | 28.45% | 39.59% | +11.14pt | 1.185 |
| scout T=0.6 | 29.62% | 39.88% | **+10.26pt** | 1.199 |

Tripling the temperature changes the depth curve by **-0.9pt** and the outcome diversity by 0.014.
The saturation is a property of the models, not of the sampling.

**"Why can't we use all the expensive draws?" — we can, and we already beat doing so.**

```
best_of_10_oss120        71.93%   $0.16534
our learned policy       73.10%   $0.15019    <- more accurate, 9% cheaper
union ceiling of ALL collected draws          73.10%
```

`n* ≈ 0-1` at R=$0.05 is not a harness limitation. It is the policy correctly pricing that one
oss120 shot is all a correct answer is worth when a correct answer is valued at $0.05, i.e. under
two oss120 calls. Raise R and it *does* buy depth (mean `n*` = 3.08 at R=$0.25). Raise it further
and it buys draws measured to be worth 0.00pt.

**Consequence, and it reframes the paper.** Our best policy already sits *exactly at* the accuracy
ceiling of the entire collected draw set (73.10% = 73.10%). **There is no accuracy headroom left in
this pool — all remaining headroom is cost.** That is precisely what the oracle decomposition said
(stopping = 63.5% of headroom): the prize is reaching 73.10% for far less than $0.15, not reaching
higher. Every "collect more/deeper/hotter draws" lever is now measured and dead.

Minor gap found and closed: the tensors ingest only 4 of the 10 collected scout draws. Adding the
other six changes the union ceiling by **0.00pt** — **zero** test problems are uniquely solvable by
scout draws 5-10. Not worth rebuilding.

**The one lever left is pool composition, not pool depth.** Depth is exhausted; complementarity is
not. Gate 1 measured glm5+qmax at **36% realized complementarity, flat in draws**, against this
ladder's 7.4%. That is the only remaining direction that can raise the ceiling rather than just
lower the bill.

### Why an +11pt depth gain still leaves the decay worthless (2026-09-01)

Scout gains +7.02pt from 10 draws, oss20 gains **+17.54pt**. Those are large. They are also worth
exactly nothing to a router. Route-alone gain versus marginal contribution to the union with the
other routes at k=10, test split:

| route | alone k=1 | alone k=10 | alone gain | **marginal to union** | problems only its draws 2-10 solve |
|---|---:|---:|---:|---:|---:|
| scout | 28.07% | 35.09% | +7.02pt | **+0.00pt** | **0** |
| oss20 | 38.01% | 55.56% | **+17.54pt** | **+0.00pt** | **0** |
| oss120 | 57.89% | 71.93% | +14.04pt | **+8.19pt** | 14 |

Every problem that scout or oss20 solves only on draws 2-10 is *already solved by oss120*. Their
depth is entirely subsumed.

**The decay needs two conditions to earn money, and no route satisfies both:**

| route | (1) `n*` inside [1,10], so depth is a real decision | (2) depth adds something other routes don't |
|---|---|---|
| scout | **FAIL** — `n*` ≈ 26, always exhaust | **FAIL** — +0.00pt |
| oss20 | ok — `n*` ≈ 4.6, genuinely in range | **FAIL** — +0.00pt |
| oss120 | **FAIL** — `n*` ≈ 0.07, one shot or none | ok — +8.19pt |

A perfect diagonal. The only route whose depth is worth buying (oss120) is priced so you never get
to *choose* a depth; the only route where depth is a live decision (oss20) buys nothing the top
route hasn't already delivered. The decay is never in a position to act on anything that matters.

**This is the same mechanism as the complementarity collapse.** In a ladder, resampling the strong
model recovers what the weak models uniquely produced — that is why realized complementarity falls
19.3% -> 7.4% with draws, why cheap-route depth is subsumed, why routing is only +3.0% of oracle
headroom, and why the decay cannot pay. One fact explains all four. And it is a fact about *ladders*
specifically: the peer pool (glm5+qmax) holds 36% complementarity flat in draws, because neither
peer's successes are a subset of the other's.

**This is the paper's central mechanism, and it is stated in one sentence:** *in a cost-ordered
ladder, every lever except stopping is subsumed by resampling the top model.*

### Cheap-route depth: the cost argument, priced (2026-09-01)

The previous entry measured cheap-route depth by its contribution to the *accuracy* ceiling (+0.00pt)
and called it worthless. That is the wrong yardstick, since all remaining headroom is cost. Priced
properly, the cheap routes are worth a great deal — but their **depth** is not.

**Cheap routes carry large cost value.** Ten scout draws cost $0.00553 = **19% of one oss120 call**.
Oracle cheapest-solve cost per solved problem:

| route set | solves | mean cost per solved problem |
|---|---:|---:|
| oss120 only | 71.93% | $0.04157 |
| scout + oss20 + oss120 | 73.10% | **$0.01811** (2.3x cheaper) |

**But the value is exhausted by draw 2.** Grinding a cheap route to depth k, then escalating:

| k | scout->oss120 | vs k=0 | oss20->oss120 | vs k=0 |
|---:|---:|---:|---:|---:|
| 0 | $0.11334 | — | $0.11334 | — |
| 1 | $0.10363 | +8.6% | $0.10357 | +8.6% |
| **2** | **$0.10316** | **+9.0%** | **$0.10272** | **+9.4%** |
| 4 | $0.10358 | +8.6% | $0.10512 | +7.3% |
| 10 | $0.10474 | +7.6% | $0.11440 | **-0.9%** |

Both minimise at **k=2** and get worse monotonically after. oss20 at k=10 is *worse than never using
oss20 at all*, because ten oss20 draws cost 133% of one oss120 call. The reason is simple: extra
draws are paid on the ~65% of problems where the cheap route never succeeds, and draws 3+ convert
too rarely to cover that tax.

**The hard ceiling on any per-problem depth policy.** Give an oracle perfect foreknowledge of every
outcome and let it pick the depth *per problem*, against a single global constant:

| route | best global k | oracle per-problem | **headroom for any depth policy** |
|---|---:|---:|---:|
| scout | 2 — $0.10316 | $0.10115 | **1.9%** |
| oss20 | 2 — $0.10272 | $0.09678 | **5.8%** |

Compare stopping: **63.5%**. And the oracle's own depth choices are concentrated at k in {0,1} —
93% of problems for scout (111 at k=0, 48 at k=1), 83% for oss20.

**This upgrades the decay result from an empirical null to a proved cap.** The learned decay is not
failing because our belief model is weak — it is better than the baseline on every per-route head,
on both ranking and calibration. It fails because **the quantity it controls has almost no variance
worth predicting**: the cost-optimal cheap-route depth is 0 or 1 for ~90% of problems, and a
perfect per-problem depth oracle beats one global constant by 1.9-5.8%. A decay competing for 2-6%
cannot show up next to stopping competing for 63.5%.

**Refined statement of the mechanism.** Cheap routes matter enormously, but as a **routing** decision
(use them at all, one or two shots) rather than a **depth** decision. In a cost-ordered ladder the
levers rank: stopping (63.5%) >> cheap-route routing (2.3x on oracle cost, captured at k<=2) >>
per-problem depth (1.9-5.8%) > per-problem decay (0%, being a subset of depth).

### Correction: the "grind then escalate" family is restricted, and the policy is not

The depth pricing in the previous entry used a **restricted policy class** — "buy k draws of cheap
route m, then escalate to oss120" — and reported the oracle-per-problem-k headroom (1.9% scout,
5.8% oss20) as "the hard ceiling on any per-problem depth policy". That was an overclaim: the
family forbids interleaving, and the real policy interleaves freely.

The action set is `{abstain, scout, oss20, oss120}` at the root and **after every failed attempt**
(`--start-protocol scout_first` only pins the first action). `scout-scout-oss20-oss20-oss120` is
legal, and chains like it dominate. From `replay_tier1_v1/episode_traces.jsonl`,
`sequential_decay_value` at R=5.265 (the $0.15 / 73.10% operating region):

```
  285  oss120
   63  oss120-oss20-oss120-oss120-oss20-oss120-oss20
   51  oss120-oss120-oss20-oss120-oss20-oss120-oss120
   32  oss120-oss20-oss120-oss120-oss20-oss120-oss120
```

| R | uses oss120 | **steps back to a cheaper route after escalating** | mean chain length |
|---:|---:|---:|---:|
| 5.265 | 100% | **50.0%** | 9.79 |
| 0.656 | 100% | 51.9% | 9.87 |
| 0.0155 | 0% | **60.7%** | — |

Route switches per episode reach 21. The policy uses oss20 as cheap filler between oss120 calls.

**What this changes and what it does not.** The correct unrestricted bound was already computed and
already allows full interleaving — the oracle-routing arm picks, at every step, the cheapest route
whose next draw succeeds:

- **oracle routing: +3.0%** of headroom — this subsumes every escalation, interleaving and depth choice
- **oracle stopping: +63.5%**

So the decay null stands, bounded by the +3.0% unrestricted routing headroom. The grind-family
numbers are a mechanistic illustration of *why* depth is cheap to get right, not the bound. Report
them as illustration only, and never as a ceiling.

### Why "better resampling estimates" cannot pay, and quitting can (2026-09-01)

The natural objection: *if we better estimate when resampling beats escalating, shouldn't that save
a lot?* Measured, no — and the reason is that the resample-vs-escalate decision is almost never a
close call.

**The decision boundary is crossed by a cliff, not a drift.** The myopic rule resamples cheap route
m rather than escalating iff `p_m(n) > p_120 * c_m/c_120`. With `p_120 = 0.5789`:

| route | cost ratio | break-even `p_m` |
|---|---:|---:|
| scout | 53.8x | **1.08%** |
| oss20 | 7.5x | **7.72%** |

Empirical conditional hazard `P(draw n+1 succeeds | first n failed)`, test split:

| n | scout | | oss20 | |
|---:|---:|---|---:|---|
| 0 | 28.07% | RESAMPLE | 38.01% | RESAMPLE |
| 1 | 4.07% | RESAMPLE | 16.04% | RESAMPLE |
| 2 | **0.85%** | escalate | 7.87% | RESAMPLE |
| 3 | 0.85% | escalate | **1.22%** | escalate |

scout falls from 26x above its threshold to below it in two draws; oss20 drops 6.5x in a single
step. A belief only has to be right to within a factor of ~5 to get these calls right, and ours is
far better than that. The optimal rule is effectively a **constant**: resample the cheap route about
twice, then escalate. There is no per-problem variation left for a decay to predict.

**And the total prize is tiny.** Problems whose first cheap-route success arrives at draw >= 4:

| route | late successes | oss120 also solves them | **genuinely unique** | money at stake if predicted perfectly |
|---|---:|---:|---:|---:|
| scout | 6 | 6 | **0** | $0.00114/problem (0.76%) |
| oss20 | 6 | 6 | **0** | $0.00132/problem (0.88%) |

Deep cheap resampling uniquely rescues **0 of 171** problems. Perfect resample-vs-escalate decisions
at depth are worth **1.6%** of our $0.15019 operating cost, combined.

**Against quitting:**

| | accuracy | cost |
|---|---:|---:|
| our policy | 73.10% | $0.15019 |
| **oracle stopping** | 73.10% | **$0.02573** — same accuracy, **82.9% cheaper** |
| oracle routing | 70.99% | $0.06845 |

**1.6% versus 82.9% — a 50x ratio.** (Oracle stopping leaks future outcomes and is a diagnostic
bound, not a deployable policy; it locates the headroom, it is not an achievable target.)

**The clean statement.** The decay answers *"will the next draw of route m succeed?"* Quitting
requires *"will nothing on any route at any remaining depth succeed?"* Those are different
questions, and the second is dominated by `theta` on the expensive route — whether the top model can
solve this problem at all — which is fixed at n=0 by the problem text and has no decay in it. This
is exactly why the factorized model lost: it improved all three per-route heads (the resampling
question) and degraded the `nothing` head from 0.7335 to 0.7199 (the quitting question). It traded
accuracy on the one head that carries the money for accuracy on three that do not.

## The next step: 26.9% of problems eat 82% of the bill

Splitting realized spend at our $0.15019 / 73.10% operating point by whether the problem is solvable
by **anything** in the pool (any route, any depth):

| | episodes | mean spend | share of total bill |
|---|---:|---:|---:|
| solvable | 625 | $0.03701 | 18.0% |
| **impossible** | 230 | **$0.45772** | **82.0%** |

46 of 171 test problems (**26.90%**) are solvable by no route at any depth. The policy grinds every
route to exhaustion on them at **12.4x** the spend of a solvable problem. That 82.0% of the bill
matches the 82.9% oracle-stopping headroom almost exactly — **the entire headroom is this one
waste**. Oracle stopping reaches the same 73.10% at $0.02573 purely by refusing to start on them.

**We currently capture none of it.** At 73.10% the best real arm is the value rule at $0.15019; the
learned q-stop arm at the same accuracy costs **$0.17140 — worse**. Every stopping mechanism we have
tried is at or below the value rule.

### Why this target has never actually been trained

The `nothing` head is (a) per-reachable-state, (b) one of four heads sharing a loss with three
routing heads now shown to be irrelevant, (c) trained on mid-episode states. The quantity that
carries the money is different: **a problem-level binary label, "will any of the 24 available draws
succeed", evaluated at n=0 before any money is spent.** We have never trained a predictor for it.

### Proposed next experiment — pre-commitment solvability gate

1. **Train the target directly.** Single-head problem-level classifier for `solvable_by_pool`. Two
   variants: input-only (problem text) and **post-scout** (problem text + one scout draw and its
   test results). One scout draw costs $0.000553 = **0.4% of the bill** — if it can classify the
   26.9%, it is the cheapest possible probe.
2. **Gate first, then replay.** Report AUC before building any policy. Prior evidence is encouraging
   but not decisive: the abstention line got input-only 0.6009 -> post-scout **0.7629** (+0.1620,
   replicated 4/4 seeds) on the related "will oss120 succeed" target, and the jointly-trained
   `nothing` head reaches 0.7335. **Risk: if problem-level solvability tops out near 0.75 AUC, the
   gate will not convert, because at the 73.10% end no solvable problem may be abstained on.**
3. Only if the AUC clears, replay it as a pre-commitment abstain gate.

No collection required — the draws, labels and scout outputs all exist.

**This also rehabilitates "scout before you route" as "scout before you spend".** The probe's value
was never in ranking which route to use (routing is worth 3.0%); it is in deciding whether to spend
**anything at all**, which is worth 82.9%.

### Secondary queue

- **Seed variance**: 11.8% cost CV against an 18-28% headline. Half-data showed it is not
  data-limited, so the fix is more seeds or ensembling, not more problems. Cheap, no collection.
- **Peer pool**: gate 1 passed (36% complementarity, flat in draws) but `q35p` must be re-collected
  with a larger token cap and reasoning-field capture before the pool is usable.
- **TACO**: demoted. Justified only as a second dataset for external validity, not as a variance fix.
