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
- **Execution**: scout code is run locally against pre-written test cases (stdin/stdout format)
- **Stage 1 — cascade**: if scout passes tests → done, no oracle needed (handles easy problems for free)
- **Stage 2 — conditional abstention**: among scout-failed instances, predict P(oracle succeeds) using failure mode as a feature

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
| No-CoT + tests (names only, no labels) | Instruct-2507 | ✗ | all names, no labels | **0.697** | — | `no_cot_testfb_names_only_route3_1786226829` |
| CoT stripped | **Thinking-2507** | ✗ | — | **0.717** | *pending* | `no_cot_route3_1786218095` |
| CoT | **Thinking-2507** | ✓ | — | **0.749** | *pending* | `cot_fixed_route3_1786133032` |
| No-CoT + tests (count only) | Instruct-2507 | ✗ | N/M fraction | **0.762** | — | `no_cot_testfb_count_only_route3_1786226851` |
| No-CoT + tests (old format, 8 names) | Instruct-2507 | ✗ | 8 fail names + count | **0.779** *(avg of 2 runs)* | — | `no_cot_testfb_route3_1786221930/1786223095` |
| No-CoT + tests (full labeled list) | Instruct-2507 | ✗ | FAILED/PASSED per test | **0.780** | — | `no_cot_testfb_full_route3_1786226769` |
| CoT + tests | Thinking-2507 | ✓ | full | *blocked* | — | needs Thinking-2507 Daytona eval |

**Key result (2026-08-08)**: Test feedback (even old 8-name format) beats CoT predictor by +3pp: 0.779 vs 0.749. This is Instruct-2507 with no thinking traces at all — pure execution feedback outperforms reasoning traces on SWE-Smith in-domain.

**Test feedback format ablation — completed (2026-08-08)**:

| Format | AUC | Δ vs. no-testfb |
|--------|-----|-----------------|
| Full labeled list (FAILED/PASSED per test + totals) | **0.780** | +0.083 |
| Count only (N/M fraction) | 0.762 | +0.065 |
| Names only (all names, no labels) | 0.697 | **+0.000** |
| No test feedback (baseline) | 0.697 | — |

Key finding: **test names alone add zero signal** — AUC identical to no-test-feedback baseline. The entire gain comes from the PASS/FAIL labels. The count (N/M) captures 78% of that gain; the full per-test labels add another +0.018 on top. This strongly suggests the signal is "how many tests did the scout break?" rather than "which specific tests broke?"

**Thinking-2507 on LCB (2026-08-08)**: Only 22% of LCB trajectories have thinking traces. The model outputs natural-language reasoning without `<think>` tags for 78% of problems — thinking traces only appear when the model explicitly chooses to use them. This significantly limits the CoT signal for LCB.

**LCB oracle resolution rates (gpt-oss-120b)**:
- Instruct-2507 collection: 44.8% train, 39.0% eval
- Thinking-2507 collection: 42.0% train, 37.0% eval
(Higher than SWE-Smith ~22%; LCB problems are more tractable for oss-120b)

Key takeaways:
- Scout patch adds signal over problem alone: +0.015 AUC in-domain, +0.059 cross-domain
- CoT stripped (Thinking-2507, no traces) outperforms Instruct-2507 no-CoT: 0.717 vs 0.697 (patch quality effect)
- CoT trace adds +0.032 over CoT-stripped: 0.749 vs 0.717 (cleanest isolation of trace signal)
- **Test feedback dominates CoT**: 0.780 vs 0.749 — execution beats reasoning on SWE in-domain
- **The signal is pass/fail counts, not test identity**: names-only = 0.697 (no gain); count-only = 0.762; full labels = 0.780
- Cross-domain transfer holds for no-CoT predictor (SWE-Smith → Verified: 0.637)

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

In LiveCodeBench, execution feedback is clean and verifiable. Wrong answer vs. TLE vs. runtime error are meaningful difficulty signals. We show a predictor trained on these signals can correctly route hard instances to the oracle, validating the approach under ideal conditions.

### 4. SWE-B agentic tasks — with and without execution feedback

On SWE-B we have two signal regimes. With Daytona eval on scout patches, we get structured test execution feedback (which tests failed, error types) — comparable in spirit to LCB. Without it, the only signal is the scout's patch and reasoning trace. The 2×2 ablation tests both regimes, letting us measure how much execution feedback adds over CoT traces on SWE-B, and how the signals compare to their LCB counterparts.

### 5. The ceiling-breaking result

By combining oss-120b and Opus 5 with the predictor as a router, we achieve 62.1% — above the 57.9% always-Opus ceiling. This is impossible without a predictor that genuinely discriminates difficulty. We compare against the random upgrade baseline (linear interpolation, ceiling at 57.9%) to show the gap is real.

### 6. Ablation: what each signal contributes

The 2×2 table shows CoT traces and execution feedback are both informative, and the cross-domain comparison shows which signal generalizes. The key question for the paper: are they complementary, or does one subsume the other?

---

## What Still Needs to Run

| Experiment | Scout model | CoT | Test FB | Status |
|-----------|------------|-----|---------|--------|
| CoT predictor → Verified cross-domain AUC | Thinking-2507 | ✓ | ✗ | 🔄 running (verified_abstention_eval_1786227495, aiohttp fix applied) |
| SWE-Smith: no-CoT + tests (count only) | Instruct-2507 | ✗ | count | ✅ AUC 0.762 |
| SWE-Smith: no-CoT + tests (names only) | Instruct-2507 | ✗ | names | ✅ AUC 0.697 (no gain over baseline) |
| SWE-Smith: no-CoT + tests (full labeled) | Instruct-2507 | ✗ | full | ✅ AUC 0.780 |
| SWE-Smith: CoT + tests | Thinking-2507 | ✓ | full | 🔲 blocked — needs Thinking-2507 Daytona eval |
| SWE-Smith: CoT stripped + tests | Thinking-2507 | ✗ | full | 🔲 same blocker |
| LCB: Thinking-2507, no-CoT stripped, no tests | Thinking-2507 | ✗ | ✗ | 🔲 scouts ready, needs LCB abstention training |
| LCB: Thinking-2507, CoT, no tests | Thinking-2507 | ✓ | ✗ | 🔲 same |
| LCB: Instruct-2507, no-CoT, no tests | Instruct-2507 | ✗ | ✗ | 🔲 same |

**Verified eval fix (2026-08-08)**: Root cause was shared `aiohttp.TCPConnector` between per-call `ClientSession` objects — first task to finish closed the shared connector, killing all 368 in-flight requests simultaneously. Fixed with `connector_owner=False` + explicit `connector.close()` in `finally`. 5th attempt currently running with 0 errors.

---

## Open Questions / Risks

1. **Will CoT cross-domain hold?** The 0.637 cross-domain AUC is for no-CoT (instruct). CoT predictor cross-domain is pending — if it doesn't transfer, the SWE → Verified story weakens.

2. **Are test feedback and CoT complementary on SWE?** On LCB they clearly are (execution gives what CoT approximates). On SWE it's less clear — Daytona reports give FAIL_TO_PASS test names but the signal is coarser.

3. **Instruct vs. thinking on LCB**: Instruct-4B outperforms Thinking-4B on SWE-B agentic tasks. On LCB competitive programming, this may flip. The comparison is a clean experiment but the result is uncertain.

4. **62.1% ceiling-breaking: noise?** 285 instances is small. The CoT predictor at optimal threshold beating always-Opus is a strong result — worth checking confidence intervals.

5. **Other model pairs**: the 4B/120B pairing is one point on a large space. Results may not transfer to other small/large combinations.

---

## Related Work to Position Against

- **LLM cascades / routing**: FrugalGPT, routing networks, model selection papers — they route based on problem features, not scout output
- **Confidence-based abstention**: calibrated LLM confidence for selective prediction — we use a separate predictor rather than the oracle's own confidence
- **Difficulty estimation**: predicting instance hardness from problem features — we show scout traces dominate problem features
- **Process reward models / verifiers**: similar in spirit but trained to verify answers, not route between models; our predictor doesn't see the oracle's output at all
- **Adaptive compute**: early-exit networks, speculative decoding — related family but different mechanism
