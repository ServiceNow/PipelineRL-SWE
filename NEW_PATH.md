# NEW_PATH.md — a fresh contribution direction (2026-09-25)

Written after a full re-read of `PRIOR_ART.md` plus fresh arXiv sweeps (2026-09-25) across:
test-time compute allocation, LLM routing/cascades, optimal stopping, generated-test verifiers,
uncertainty-for-code, SWE-agent routing, and model co-failure structure. Companion to
`PRIOR_ART.md`, `EXPERIMENT_PLAN.md`, `THREADS_IN_PROGRESS.md`. Rule of this repo applies:
**check `PRIOR_ART.md` before writing any novelty claim**; IDs below were found via search and
read at abstract level — read the full paper before citing.

## 0. What the 2026-09-25 searches found that PRIOR_ART does not yet list

| work | what it does | why it matters here |
|---|---|---|
| **Co-failure ceiling** (2606.27288, Jun 2026) | β = rate all models are wrong on the same query caps routing/voting/MoA; pairwise ρ cannot identify β; a tetrachoric single-factor Gaussian copula **underprices the all-wrong tail ~2.5x** (code β=0.079). 67 models. | Measures model co-failure as a *ceiling*. Never makes decisions with dependence. Explicitly leaves proper dependence modeling open. |
| **Correlation-aware contextual bandits** (2607.09015) | bandit LLM routing with context-dependent inter-arm correlations + surrogate rewards | online bandit feedback; no resampling, no stopping, no per-query budget. Cite. |
| **Risa** (2608.22191) | MoE router traces steer exploration + select among SWE trajectories by agreement | verifier-free selection for SWE agents exists; uses router *traces*, not a probability model over routes |
| **SuperScout / Scouting** (2608.04804) | 7B scout explores repo, hidden states feed a router to 4 frontier fixers | honest: the handoff, not routing, carries the result |
| **COMPAS** (2608.04336) | difficulty-aware joint search over model × prompt × decoding, LCB + SWE-bench | crowds the "predict difficulty, then allocate" axis |
| **WISERouter** (2607.23765), **DRS** (2609.00662), **Canopy** (2609.30017) | budget-constrained contextual bandits; drift; multi-fidelity tree bandits (1.6x best-of-N on SWE-bench Verified) | the budgeted-bandit axis is busy; none model cross-route dependence of *outcomes* |
| **CPPO** (2605.27000) | RL-trained planner emits K diverse strategy tuples for pass@K | diversity via training; ours would be training-free of pool models |
| **CoVer** (2609.21208), **ExecCritic** (2609.09133) | co-trained coder+test-verifier (info-gain rewards); SWE test-verify-revise agents | the "generate tests" idea is CodeT-lineage-occupied; both train a verifier, neither budgets or chooses among verifiers |
| **Uncertainty self-correction study** (2608.14659) | token entropy / P(True) / semantic entropy on code: selection signals weak; uncertainty-driven self-correction *degrades* Pass@1 in 5/6 configs | independently replicates our "verifier-free selection is exhausted" finding |
| **Optimal Stopping vs Best-of-N** (2510.01394), **OptPO** (2512.02882) | UCB-Pandora stopping vs Best-of-N; SPRT stopping for voting | Pandora-for-LLM exists: single model, reward-model judge, no pool, no code |
| **Replay Gap** (2608.08239, COLM 2026) | replay-based evaluation of agentic routing mispredicts outcomes; fork-based harness released | evaluation-methodology citation for any SWE routing claim |
| **AgentRouter** (2609.22951) | 12M classifier routes each trajectory *step* to one of 4 tiers | step-level agentic routing taken |
| **Signed Rescue Routing** (2609.07786) | rank escalation by P(escalation rescues) − P(escalation replaces a correct answer) | predicts *signed incremental value* — same spirit as our value-of-continuation; single-escalation, no sampling |

---

## 1. IDEA A (primary): One Failure Is a Signal — learn cross-model failure dependence

### 1.1 The one-paragraph thesis

Conditional failure dependence between routes is (a) large and **problem-specific**, (b) **predictable
from one cheap prefill**, and (c) exactly the belief component that failure counting, fixed cascades,
self-consistency, and independent-Beta Bellman **structurally cannot represent**. Exploiting it makes
"resample or reroute" identifiable — the open problem RoR v3 states outright — and it is the one
component of the value function that counts, fixed cascades, self-consistency, and factorized-belief
DP all throw away.

### 1.2 Why now, and why it converges on us

* **RoR v3 states the gap verbatim:** "current evidence does not identify when to resample rather
  than reroute." Our §3b-lxxxviii measures the missing headroom: conditional on a failure, *switching*
  rungs beats *resampling* by 20–50pt — a dependence phenomenon by construction.
* **The co-failure paper (2606.27288) supplies outside motivation and a warning:** gains come from
  models failing on different questions; marginals + pairwise ρ do not identify the all-wrong tail;
  a single-factor copula underestimates it ~2.5x. They measure dependence and stop. We would *use* it.
* **Our own DP is honest about the gap:** the 8-draw Bellman is "exact for a factorized Beta-Bernoulli
  belief model, not the correlated true outcome process" (THREADS_IN_PROGRESS, Bellman continuation).
  It loses to the fixed cascade on BCB by 3.4pt — the cascade hard-codes the cross-route ordering the
  factorized belief cannot express.
* **Theory exists, practice does not:** Gergatsouli et al. solve Pandora's box with *known*
  correlations. "Pandora with learned, problem-conditioned dependence" is the empty cell
  [learned per-problem conditional dependence × sequential budgeted policy × code pools].

### 1.3 The three claims

1. **Measurement (a new observable):** on the 5-rung pools, estimate per problem the pairwise
   dependence Δ(x; m,m') = P(m' succeeds | m failed) − P(m' succeeds). Show it is large, varies by
   problem, is predictable from a 4B prefill (new prediction target: not "will m succeed" but "are
   m and m' the same failure"), and is near-zero on nested pools (Jaccard 0.90–0.98) — the old pool
   is a designed null.
2. **Decision:** replace the independent per-route Beta-Bernoulli belief in the Bellman lattice with
   a correlated belief: latent difficulty factor θ(x) + route-specific loadings (the "whether-not-which"
   latent extended to a covariance). After a failure on route m, beliefs over every route update.
   Implementation: ~a belief-update change to `replay_bellman_verification.py` + a dependence head
   (the 137M Jina encoder fine-tune running as eai job `3d8cd1f1` is directly repurposable: train it
   on "will m' succeed given m failed on x" states).
3. **Positioning:** with independent beliefs the resample-vs-reroute action is unidentified — we
   already hold the data (Bellman vs one-retry: +3.0pt LCB, −1.6pt BCB) — and with learned dependence
   it is identified. Direct answer to a problem the literature names as open.

### 1.4 The unifying figure

Phase diagram: value of dependence over (base rate q, pool diversity ρ). LCB (q≈0.89) → correlation
worth ~0; "hammer the cheapest rung" is near-optimal, which is *why* everything ties there. BCB
(q≈0.28) → correlation is where the value lives. One figure explains: counting winning on LCB, fixed
cascades tying on BCB, Bellman's dataset-dependent gains, and the verifier-regime absorption of the
cost head. It also tells a practitioner *when* to buy this machinery.

### 1.5 Method sketch (2–4 weeks, CPU-only, existing pools)

1. Estimate empirical Δ per problem pair on LCB + BCB pools (5 rungs, asymmetric depth already there).
2. Fit dependence heads on train/cal: probe features → (θ_m(x), dependence parameters). Model choices,
   in order: shared-latent + route loadings (IRT 2PL generalization); pairwise logits with shrinkage;
   copula on the (θ, ρ) plane. Select ONLY on the calibration frontier — never AUC (§2 of
   THREADS_IN_PROGRESS: four documented cases where predictor-metric selection hurt the policy).
3. Correlated Bellman: after an observed failure on m, update beliefs on all routes via the dependence
   model; rerun the exact 8-draw DP. Global budget via the standard dual sweep.
4. Pre-registered evaluation: primary pool **BCB** (q≈0.28, headroom exists), CodeContests
   out-of-sample, LCB as the negative-control pool. Matched realized spend, paired bootstrap CIs.
   Baselines: counts (RoR v1), independent-Beta Bellman, one-retry heuristic, enumerated best fixed
   cascade, prefix-value policy, CPPO-style planned diversity if it can be replayed offline.
5. **Kill criterion, stated in advance:** dependence-corrected Bellman must beat independent Bellman
   by ≥2pt on BCB at matched spend, AND show ~0 gain on the nested old pool. If not: report the null
   and move to §2.

### 1.6 Positioning deltas (checked 2026-09-25)

RoR v1/v3 (no dependence, no learned beliefs; we answer v3's stated open problem) · co-failure
ceiling 2606.27288 (measures a ceiling, never decides) · correlated bandits 2607.09015 (bandit
feedback, no resampling/stopping/budget) · CPPO (diversity via RL, trains pool-adjacent models) ·
Risa (router traces + agreement, SWE) · Smoothie / semantic-agreement (agreement for selection, not
budgeted routing) · Gergatsouli (known correlations, theory). Machinery conceded per PRIOR_ART §0b:
Altman, Weitzman/Gittins, Bouchard.

---

## 2. IDEA B (reformulated 2026-09-25 after user pushback): the test-writer is a routed model — priced verification with correlated false-accepts

**History of the idea:** v1 was "route among LLM judges." Killed by two observations from the
project's own data and the user's pushback: (1) LLM judges are ~random on strong-model code (our
measurements; 2608.14659 replicates the negative result), and direct code-judging is a dying
practice — the field's verifier for code is generated tests + execution (CodeT, CoVer, ExecCritic,
BeSpec). (2) Decomposing solution-writer from test-writer is already standard (ExecCritic trains
separate Test/Repair agents; CoVer co-trains one model as both), so the decomposition itself is not
a contribution.

### 2.1 The reformulated thesis

A generated test-suite is itself a **sample from a model**, with a price (write cost + run cost) and
an error profile — α(x) = P(suite passes a wrong solution), β(x) = P(suite rejects a correct one) —
that is problem-dependent and, critically, **plausibly correlated with the generator's errors**: if
both the solver and the test-writer misread the same ambiguous spec, the suite asserts the wrong
behavior and the wrong solution passes while the loop feels safe (the oracle-generation failure mode,
2410.21136: LLM oracles reproduce actual behavior rather than intended behavior). Nobody measures
this correlation or prices it.

The routing question therefore extends to verification: **which model writes the tests, at what
price, for this problem and this candidate** — with α/β learned per problem and per
(test-writer, generator) pair, and the verify-vs-draw-again decision priced as in Q2. The judge
(one more weak, differently-failing cell) becomes a baseline that documents the "tests beat judging"
result, not the centerpiece.

### 2.2 Why this is not already taken (the two anchors)

* **ExecCritic's own ablation is the existence proof** (2609.09133): with the Repair agent fixed,
  tests written by its base Qwen Test agent *lower* SWE-bench Verified resolution 61.2% → 57.3%
  (worse than no tests), while tests from GPT-5.6-sol raise it to 65.3%. A ~10-point swing from the
  test-writer choice alone, measured once, manually, and then left fixed. Neither ExecCritic (one
  fixed Test agent), CoVer (one co-trained coder+verifier), CodeT (one generator's tests), nor
  BeSpec learns which model should write the tests per problem, prices the choice, or models
  test-writer/generator error correlation.
* **Perfect-verifier gap is ours to fill:** our priced-verification replays (Q2, HANDOFF
  2026-09-25) assumed a *perfect* verifier, and the weak-verifier thread showed the ceiling
  collapses 84.8% → 62.1% at 11% false-accepts. Generated oracles are plausibly worse than 11% on
  some problem strata (2609.09315: fault detection near zero because LLM oracles miss faulty
  behavior). Nobody connects α(x) to the routing decision.

### 2.3 Why "which model writes the unit tests" is a routing dimension and not just a design choice

It carries the three properties that make verification a routing decision rather than a pipeline
stage: (1) a **price** (write + run cost, and on SWE a rationed execution capacity); (2) **learned,
problem-conditioned reliability** (α/β per pair); (3) **state-dependence** — a check is worth more
when the posterior is uncertain, so it cannot be pre-committed the way generation routing is. And
the selection criterion differs from generation routing: not "best test-writer" but "test-writer
whose errors are anti-correlated with the generator's" — the same dependence story as IDEA A.

### 2.4 Risks, restated after the reformulation

* **Dominance/collapse risk:** if α_j(x) is flat across models and problems — any cheap model's
  tests are equally mediocre, tests are near-free, so just run them — the dimension is real only
  where execution is expensive (SWE) or the correlation effect is large. This is the Phase 1
  go/no-go.
* **Judge cell is weak on strong code** (our measurements): fine — it is a comparison baseline
  showing tests beat judging (consistent with 2608.14659), not a contender.
* **Name collision:** "CoVeR"/"CoVer" are taken (2609.26086, 2609.21208); avoid.

### 2.5 Prior-art deltas for the reformulated idea (checked 2026-09-25)

CodeT / CoVer (2609.21208) / ExecCritic (2609.09133) / BeSpec (2607.02949): one test-writer each;
ExecCritic observes the test-writer matters but never routes or learns it. Oracle-generation study
(2410.21136): oracles reproduce actual behavior — the mechanism behind correlated false-accepts, cited
as the mechanism, not measured per-problem as a decision input. JuryProbe (2608.20607): judge-panel
escalation in factuality, no test-writers. Judge-entanglement audit (2604.07650): ensemble
reweighting of judge outputs, not verifier selection. Maestro Order (2606.23983): simulation-only
budget controller over verifiers. CoVeR (2609.26086): when-to-call-a-single-verifier. 2608.14659:
uncertainty signals fail, execution-based verification wins — supports "tests are the verifier."
**Empty cell:** learning per-problem, per-(test-writer, generator) α/β for generated test-suites and
pricing the choice into the sequential budgeted policy.

### 2.6 Near-neighbor sweep (2026-09-25, second adversarial pass — user asked "are you sure this isn't taken?")

**Verdict: the exact cell is still empty, but more crowded than first stated.** Nearest neighbors,
each taking one slice (v1 framing shown where the delta changed after reformulation):

| work | what it takes | what it leaves |
|---|---|---|
| **CoVeR** (2609.26086, Sep 2026) | gates **whether** to call a single LLM verifier per retrieval step (cheap coverage-margin gate; 62–68% of verifier calls cut). Name collision: "CoVeR" also 2609.21208's "CoVer". Cite; avoid the name. | one verifier, binary when-question, agentic retrieval. No verifier choice, no pool, no per-pair reliability. |
| **JuryProbe** (2608.20607, TMLR 08/2026) | detects correlated false negatives in cheap judge panels; routes flagged accepts to grounded verification | closest conceptual neighbor: cheap-vs-expensive verifier escalation. But: factuality, binary escalate/stand-down on panel consensus, no per-problem reliability model, no test-writers, no budget sweep |
| **Judge-entanglement audit** (2604.07650) | measures behavioral dependence across LLM judges; de-entangled verifier-ensemble reweighting (+3.5pp over majority voting) | post-hoc ensemble combination; not which verifier to call when, not priced, not code, not sequential |
| **Maestro Order** (2606.23983) | verifier ensemble with discrimination measured online + budget-aware controller by marginal reliability per cost | **Monte Carlo simulation of a parameterized solver/verifier model** — no learned per-problem reliability, no real models, no code tasks |
| **DISC** (2606.21724) | cross-model role allocation (verifier ≠ generator) cuts self-confirmation bias | static role assignment, unpriced, unlearned, not a decision problem |

Still-occupied-adjacent: CoVer (2609.21208, one trained verifier), ExecCritic (2609.09133, one fixed
Test agent — but its ablation proves the test-writer choice swings outcomes ~10pts), EAVer (2609.22223,
learned single-workflow verifier policy, factuality), budgeted verification (2606.15841, 2504.01005,
within one verifier), Speculative Uncertainty (2609.05274, learned cheap failure signal for coding
agents — verifier-free selection, not a verifier pool), Proof-Carrying Cognition (2609.09776, one
verifier's soundness under pressure), "don't read the log" (2609.28564, judge verdicts shift with
execution traces — further evidence verifier reliability is context-dependent).

**Empty cell, stated narrowly for the reformulated idea:** learning per-problem, per-(test-writer,
generator) α/β for *generated test-suites*, measuring their correlation with generator errors, and
pricing the write/run cost into a sequential budgeted generate-verify-submit policy on code.

**Honest caveat:** the crowd means the paper must be led by the learned per-pair measurement (smoke
test stats 2–4), and it must show the argmax test-writer genuinely varies by problem and price. If a
fixed writer dominates, fall back to the SWE-priced variant, where verifiers differ in *cost
structure* as well as accuracy (sandbox spin-up, CI minutes, ≤10 concurrent Daytona sandboxes make
verification rationed as well as priced — queueing-flavored, no published real price numbers) —
that variant is the natural escalation either way.

### 2.7 THE SMOKE TEST (2026-09-25): does the test-writer dimension have meat?

**Goal:** decide, for ~$3 of API spend and one day, whether "which model writes the tests" is a
routable dimension. Design decisions stated up front: tests are written **spec-only** (from the
problem statement, never seeing candidate code — this is what makes test-writer/generator correlation
arise through the problem's ambiguity, and it is the standard cheap CodeT setup); test-writers come
from the existing pool; ground truth is the stored real grader label (never proxy labels).

**Setup (LCB first, BCB after if LCB shows signal):**
1. Candidates: `pool_v2_tensors_5rung/draw_records.jsonl` — all stored draws per problem with real
   pass/fail labels (~341 test problems; cal split saved for fitting later).
2. Test-writers: Qwen3-4B-Instruct, gpt-oss-20b-low, gpt-oss-20b-md, dsv4f (+ gpt-oss-120b-md as the
   "expensive writer" cell, subsampled if needed). One suite per (problem, writer).
3. Execute every suite against every stored candidate → verdict matrix
   (problem × test-writer × candidate) → PASS/FAIL; truth = stored grader label. Local sandboxed
   execution (LCB problems are self-contained python + asserts; 1–10 CPU-s per run). No GPU, no
   Daytona, no concurrency cap.

**The four go/no-go statistics (report all four whatever the outcome):**
1. **Executability** — share of suites that run at all (LLM tests often fail to execute; report as
   measured). If <50% for all writers after one repair pass, the signal is buried in syntax noise.
2. **Discrimination** — per test-writer, AUC of the suite verdict separating correct from incorrect
   candidates, stratified by LCB difficulty labels (easy/medium/hard ship with the dataset). This is
   the direct test of "writer skill varies by problem type" vs "flat mediocrity."
3. **Crossover** — per-problem argmax test-writer; regret of "always the globally best writer" vs
   per-problem oracle at matched suite cost, problem-level bootstrap CIs. Bar: crossover on ≥20–30%
   of problems, or regret-of-fixed ≥3pp.
4. **The correlation effect (the novel measurement)** — per (test-writer j, generator m) pair:
   α(j|m) = P(suite passes | candidate wrong, m generated it). Headline: is α higher when j and m
   share a family, and on ambiguous problems — i.e., do self-tests rubber-stamp the generator's
   misreadings? If α(same-family) ≈ α(cross-family) everywhere, the anti-correlation story dies and
   only the price story survives.

**RESULT 2026-09-25 (smoke test run; full note at
`$R/testwriter_smoke_lcb/verdicts/analysis/SMOKE_RESULT.md`): there is meat, and the mechanism is
sharper than the framing.**
1. Executability 1.00 everywhere; loose-comparator sensitivity: 5/19,080 rows differ — numbers are
   not comparison artifacts.
2. GATING RESULT: a suite is only as good as the test-writer's own solve of the problem. Per-problem
   AUC on problems the writer's suite validates vs doesn't: 0.62-0.65 (all writers, when trusted)
   vs 0.31-0.53 (when not). Writer solve-rates span 30% (qwen4b) to 68% (dsv4f) — the writer
   difference is entirely in how often the suite is trustworthy, which is a problem-dependent,
   learnable quantity. This is the routing head.
3. Anti-signal: on unsolved problems, suites actively prefer wrong candidates (AUC ~0.31) — a naive
   CodeT pipeline is hurt by them; the trust gate is the product.
4. Crossover: per-problem argmax writer beats the best fixed writer by +9.5pp selection accuracy
   (bootstrap CI [6.1, 13.4]).
5. Correlated false-accepts: dsv4f suites pass 47% of dsv4f's own wrong solutions vs 23-34% for
   other writers' suites on the same candidates — self-family rubber-stamping is real for dsv4f;
   gpt-oss writers show the reverse (composition not yet controlled).
Kill criteria from below were NOT triggered. Next gates: top up missing suites (52 dsv4f +63
others), BCB replication, then alpha/beta heads + VERIFY WITH j in the priced replay.

**Kill criteria, stated in advance:** all writers' AUC ≈ 0.5 after executability repair → dimension
dead on code benchmarks; flat crossover with a dominant writer → pricing collapses to CodeT and only
the SWE-priced variant survives; α(same-family) ≈ α(cross-family) → the anti-correlation story dies,
the paper becomes test-economics-only.

**Costs:** ~4 writers × 341 problems ≈ 1.4k cheap calls (4B/20b-low ≈ free at market prices,
20b-md ~0.03$/M out, dsv4f cheap; cap oss-120b-md to a subsample). Suite execution: CPU minutes. No
GPU, no Daytona, no concurrency cap.

**Implementation (committed 2026-09-25, two jobs run in parallel):**
- **OpenRouter leg (CPU-only):** `launchers/abstention/launch_testwriter_smoke.sh` runs
  `pipelinerl/swe/scripts/livecodebench/generate_test_suites.py` for the 4 API writers
  (oss20lo, oss20md, dsv4f, oss120md).
- **qwen4b leg (GPU):** `launchers/abstention/launch_testwriter_smoke_qwen4b.sh` — qwen3-4b is not
  on OpenRouter (the pool serves it locally); the job spins a local vLLM server (the
  `lcb_corrected_temporal_*` pattern) and generates the qwen4b suites against it.
- `pipelinerl/swe/scripts/livecodebench/run_test_suites.py` — local execution of every suite case
  against every stored candidate (round-robin sample, cap 12 candidates/problem, 4 cases/suite,
  10 s/case; crash/timeout counts as case-FAIL, not as suite failure). Verdicts per writer under
  `$R/testwriter_smoke_lcb/verdicts/`, resumable.
- `pipelinerl/swe/scripts/livecodebench/analyze_test_writer_smoke.py` — the four stats +
  `smoke_report.md`/`smoke_stats.json` under `verdicts/analysis/`.

---

## 3. How the two ideas relate

They are the same missing-belief story applied to two different objects. IDEA A learns how *routes*
depend on each other (failure is contagion across generators); IDEA B learns how *verifier verdicts*
depend on the generator being checked (acceptance is not a fixed-quality coin). Both replace a
constant (independence across routes / a fixed verifier) with a learned, problem-conditioned
interaction — which is precisely the "whether-not-which" latent extended to a second structure. They
share the evaluation discipline (matched spend, paired CIs, calibration-only selection) and possibly
the same encoder backbone.

**Sequencing recommendation:** run the §2.7 smoke test first (one day, ~$3, decides whether the
user's favorite has legs); IDEA A's dependence estimation in parallel (CPU-only, pools exist,
answers a named open problem). If the smoke test passes its bars: fit α/β heads, extend
`replay_priced_verification.py` with `VERIFY WITH j` (writer choice) as an action, sweep v as before,
baselines = best fixed writer, always-cheapest-writer, oracle writer, never-verify. If it fails its
bars: IDEA A carries the program and IDEA B survives only as the SWE-priced variant.
### 2.8 CORRECTION (2026-09-25, Claude re-check of the smoke test) — read before building on §2.7

`pipelinerl/swe/scripts/livecodebench/check_test_writer_smoke.py`, same verdicts, 262 problems with
suites from all 5 writers and 12 common candidates. Selection = submit the candidate passing most
suite cases, ties broken uniformly (computed exactly).
1. **Every writer's suite is WORSE than no suite.** Random pick 83.5%; qwen4b 80.5% (-3.0, CI
   [-5.4,-0.7]); oss20lo/oss20md/dsv4f/oss120md 71-74% (-10 to -12). "qwen4b is the best fixed
   writer" = it is the least harmful: its suites fail everything, so ties fall back to random.
2. **The +9.5pt crossover was selection on noise.** Choosing the writer per problem on half the
   candidates and scoring on the other half: honest oracle 79.4% vs best fixed 81.5% (-2.1,
   [-3.4,-0.6]) vs random 83.6%.
3. **User's framing (X writes solution, Y writes tests vs X writes both), X's own candidates:**
   any-writer suites ≈ random for every X (differences within ±2pt; self vs best-other: oss20lo
   +0.0, oss20md +2.1 [+0.4,+3.8], dsv4f -0.1, oss120md -0.2).
Mechanism: a 4-case spec-only suite with the writer's own expected outputs rewards candidates that
share the writer's mistakes; on LCB (83.5% of candidates correct) any biased signal hurts.
Caveats: LCB only (high base rate — BCB ~50% is the fairer test); 4 cases per suite; this scores
against the writer's EXPECTED outputs — CodeT-style dual agreement (inputs from the writer, outputs
from candidate consensus) was not tested and needs raw candidate outputs, which the verdict files
do not store. As run, IDEA B's go/no-go is NO on LCB.

### 1.x IDEA A — first gate PASSED on BCB, not on LCB (2026-09-25, `livecodebench/idea_a_dependence_check.py`)
Held-out log-loss gain (nats×100 per draw) for predicting model m' from its probe prior PLUS one
observed draw of model m on the same problem (train fit, test eval):
- **BCB: large everywhere.** Off-diagonal (cross-model) 20-30, same-model 29-36; coefficient on the
  observed draw +2 to +4 logits. The probe's prior on BCB is weak (AUC ~0.72), so one draw of ANY model
  reveals much of the difficulty the probe missed, and a failure on m should lower beliefs on m'.
  An independent Beta-per-route belief (counts, current Bellman) cannot make that cross-update.
- **LCB: small (0.4-7.9).** The prior already knows the difficulty; little left to learn from a draw.
- Most of the BCB signal is SHARED difficulty (cross-model gains are ~70-90% of same-model gains), i.e.
  a one-factor latent-difficulty posterior should capture it; model-specific structure is secondary.
- Only usable when outcomes are observed (verifier / priced-verification regime).
Next gate (Codex's Bellman replay): one-factor correlated belief vs independent Bellman vs best fixed
cascade on BCB, matched spend; kill if < +2pt over independent Bellman.

## 3. SWE TEST-WRITER DIRECTION (2026-09-25) — plan and data inventory

**Why SWE and not LCB:** on LCB, writing a correct expected output requires solving, so test-writers
form the same difficulty ladder as solvers (§2.8). On SWE the verifier is a *reproduction test*
(fails before, passes after), which needs understanding the bug, not fixing it. SWT-bench (2406.12952,
NeurIPS'24) measured exactly this: instance-level success at test generation and at repair are
statistically independent (p = 0.80 / 0.73), and four test-generation methods are strongly
complementary (ideal ensemble 87 vs 51 best-single, +71%). A reproduction test also has a label-free
validity check (must FAIL on the unpatched repo), and checks are expensive (env build + suite in a
Daytona sandbox, org cap ~10 concurrent), so pricing is real.

**Novelty check (2026-09-25):** no paper routes the test-WRITER per instance. Neighbours: SWT-bench
(measures complementarity, doesn't exploit it); ExecCritic 2609.09133 (fixed writer; ablation: writer
identity swings Verified 57.3 vs 65.3); e-Otter++ 2508.06365 (heterogeneous prompting, selects among
tests); SCATE 2607.08983 (contextual bandit over testing actions for cost-effective coverage-driven
test generation, within one agent); Agentless/Otter (repro tests for patch selection, one writer).
Hard baseline to beat: ALL writers write tests (the 71%-complementarity ensemble) — our claim must be
"matches the ensemble at a fraction of the cost", which is why pricing is central.

**Candidate pool that already exists (real Daytona labels, same 369 SWE-bench Verified instances):**
| route | dir (`$R/opus_verified_daytona_eval_*`) | resolved |
|---|---|---|
| Qwen3-4B scout | 1788821285 | 91 (24.7%) |
| gpt-oss-20b | 1788821339 | 152 (41.2%) |
| Qwen3-30B | 1788821375 | 153 (41.5%) |
| gpt-oss-120b | 1788821418 | 193 (52.3%) |
| Gemini (route 4) | 1788821453 | 230 (62.3%) |
| Claude Opus | 1786735324 | 319 (86.4%) |
Patches: `predictions/predictions_opus_verified.jsonl`; labels: `...results.jsonl`. Route→model
mapping from `run_all_routes.sh` + the collect dir name (4b_scout_oss20_qwen30_oss120_gemini) — VERIFY
before use. (Runs 1788820478-0644 are the blown concurrent attempts: 0-2 resolved, ignore.)

**Pilot (go/no-go), in order:**
1. Pick ~100 of the 369 instances with MIXED patch outcomes across the 6 routes (only those can
   discriminate).
2. 3-4 writers (e.g. gpt-oss-20b, gpt-oss-120b, dsv4f, one Qwen) each write ONE reproduction test per
   instance (SWT-bench-style prompt: issue + retrieved files; output a test file). API cost small.
3. Validity gate: run each test on the unpatched repo in Daytona; keep only tests that FAIL (record
   the rate per writer — itself a finding). Daytona ≤10 concurrent TOTAL, one job at a time.
4. Run surviving tests against the 6 existing patches (+ gold patch as a sanity check: a valid test
   should pass on gold). Per (instance, writer): does the test separate resolved from unresolved patches?
5. Go/no-go statistics (same honesty rules as §2.8): per-writer patch-selection accuracy vs random vs
   perfect; writer crossover measured on HELD-OUT patches (choose writer on half the patches, score on
   the other half); does writer quality track instance difficulty (AUC vs mean patch resolve rate) or
   decouple (SWT-bench says decouple); self-family rubber-stamping (writer = patch author family).
   Kill if no writer beats random selection, or if per-instance writer choice doesn't beat the best
   fixed writer on held-out patches.
Reuse: SWT-bench harness (github.com/logic-star-ai/swt-bench) for prompts/eval logic; our Daytona
eval path (`launch_opus_verified_daytona_eval.sh`) for sandboxes.
