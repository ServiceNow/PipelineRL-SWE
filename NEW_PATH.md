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

### 1.y IDEA A — second gate: FAILS against the fixed cascade (2026-09-25, `replay_correlated_beliefs.py`)
Verifier regime, greedy P*V-c policy, learned cost head, calibration-matched budgets, paired CIs.
- Shared-coefficient beliefs: correlated beats independent on BCB (+3.4 [+1.4,+5.5] @0.05c, +3.6
  [+2.0,+5.4] @0.10c) — but the independent arm was misspecified (one own-failure weight for all routes;
  oss20lo's own failures are far less informative than a strong route's).
- Per-route beliefs (`--per-route`): the gain mostly vanishes — BCB +2.9 [+1.5,+4.5] @0.05c, +0.9/+0.8
  above; LCB correlated is WORSE at low budget (-3.0, -1.7, -2.2).
- Both lose clearly to the saved best fixed cascade on BCB (oss20lo x4-16, then dsv4f):
  59.9% @0.019 vs 41.5-42.1% @0.020; 66.5% @0.044 vs 58.9% @0.039; 69.6% @0.076 vs 65.9% @0.065.
Reading: failures on other routes ARE informative (diagnostic §1.x; learned weight on "other failures"
is as large as on "own failures"), but a greedy belief policy still loses to "hammer the cheap rung"
because its successive draws are nearly independent and the policy is myopic. Same conclusion Codex's
Bellman reached. Cross-route updating is a correct modelling fix, not a winning policy. PARKED.

### 3.1 SWE test-writer pilot RESULTS (2026-09-26; 100 Verified instances, 5 writers, 6 pool patches + gold)
`offline_router/analyze_swe_testwriter.py` on `$R/swe_testwriter_pilot/exec_clean.jsonl`. Selection is
label-free (a script counts only if it FAILS on the unpatched repo).
- Validity (fails on base): dsv4f 95%, oss120 88%, oss20 88%, devstral 85%, qcoder30 61%. Of valid
  scripts, pass gold: dsv4f 63%, oss120 56%, oss20 44%, devstral 42%, qcoder30 36%.
- Patch selection among applied pool patches: random 46.3%; dsv4f 61.0% (+14.7 [+10.2,+19.5]); oss120
  +10.1; devstral +9.4; oss20 +8.2; qcoder30 +3.4; **vote of all valid writers 63.4% (+17.1
  [+12.0,+22.6])**; perfect 100%. => Generated tests carry real signal on SWE (unlike LCB, §2.8).
- Difficulty coupling: Spearman rho of good-script rate vs #pool models resolving = +0.18..+0.22
  (qcoder30 +0.39). Weak, unlike LCB. Writer-writer correlation +0.34..+0.61 (partly complementary).
- Per-instance writer choice, HONEST (choose on 3 patches, score the other 3): 56.5% vs always-dsv4f
  57.1%, -0.6 [-2.4,+1.3]. **Choosing the writer does not pay; combining writers does.**
- Self-family false accepts: gpt-oss writers on gpt-oss patches 23% vs 17-20% other families — small.
- False-accept rates on wrong patches 12-25%: tests are imperfect in a way that matters.

### 3.2 PRE-REGISTERED de-risking conditions for "priced, imperfect, generated verification" (2026-09-26)
| # | condition | status |
|---|---|---|
| 1 | generated tests beat random selection by >= +10pt, false-accept < 30% | PASS (dsv4f +14.7, vote +17.1; FA 12-25%) |
| 2 | test quality not tied to difficulty (|rho| < 0.3) | PASS for 4/5 writers (+0.18..+0.22; qcoder30 +0.39) |
| 3 | a check (write + run) costs >= ~10% of an agent rollout | TODO: rollout tokens from collection parquet; Daytona seconds x price |
| 4 | test reliability predictable label-free, AUC >= 0.7 | TODO |
| 5 | offline SWE replay on real verdicts: selective policy >= 20% cheaper at matched accuracy than best of {fixed cascade + always check, never check, check final only, all-writer vote}; oracle-reliability first | TODO (decisive) |
| 6 | per-instance writer choice beats best fixed writer | FAIL (-0.6) -> drop writer routing; keep a fixed writer or the vote |
| 7 | novelty holds (targeted sweep: generated tests / budgeted checks inside SWE agent loops) | TODO |
Structural risk: only ONE patch per model per instance -> condition 5 runs as cross-model sequencing
(try A, check, try B...); within-model resampling needs new agent rollouts (cost TBD).

### 3.3 Condition 3 — what a check costs vs a rollout (2026-09-26)
Sandbox allocation (Daytona SDK + cgroup): 1 vCPU, 1 GiB RAM, 3 GiB disk (under the 5 GiB free tier).
List price $0.0504/vCPU-h + $0.0162/GiB-h = $0.0666/h = $1.85e-5/s (per-second billing).
Timing, 6 instances across 6 repos (`offline_router/time_swe_check.py`, `$R/swe_testwriter_pilot/timing.jsonl`):
running a repro script 0.4-2.4 s (warm ~0.4-1.5 s), applying a patch ~0.04 s, sandbox create 0.6-1.0 s
when the image is cached but 17-79 s cold (image pull), delete ~0.1 s.
Per-instance costs (cents; token costs at OpenRouter list prices; `analyze_writer_cost.py` for writers):
- running one check ~0.0015c (1 s); warm sandbox overhead ~0.002c; a cold start up to ~0.15c if billed.
- WRITING a test: oss20 0.030c, dsv4f 0.064c (mean 0.113), qcoder30 0.064c, oss120 0.188c, devstral 0.392c (medians).
- one single-shot patch (the pool's "rollout"): qwen4b 0.024c, oss20 0.036c, qwen30 0.16c, oss120 0.21c,
  gemini-3-flash 0.59c, Opus 5 10.1c.
Verdict: verification cost is dominated by WRITING the test, not running it (execution ~1-2% of the
cheapest rollout once warm). (write dsv4f + 1 run) / rollout = 180% vs oss20, 31% vs oss120, 11% vs
gemini, 0.65% vs Opus. PASS against cheap/mid generators, FAIL against Opus (checks are ~free there).
Caveats: our rollouts are single-shot calls; agentic rollouts cost far more tokens and sandbox time, which
would make checks relatively cheaper. A repro script (~1 s) is far cheaper than running a repo's full
test suite. Consequence for the method: the cost lever is WHICH MODEL WRITES the test (cheap-first
escalation matched the all-writer vote at 3.7x lower writing cost, 63.5% @0.21c vs 63.4% @0.78c).

---

## 4. TWO PARALLEL TRACKS (decided 2026-09-26)

We run a SAFER bet and a RISKIER bet at the same time.

### Track A (safer): per-query cost prediction for one-shot routing among reasoning models
**Claim.** The same cheap 4B prefill that prefill routers use to predict each candidate model's success
also predicts each candidate's per-query COST (output length), with one linear head per model. Replacing
the per-model median-length cost (arXiv 2603.20895's rule) with this estimate makes one-shot,
verifier-free routing cheaper at matched accuracy. Novelty sweep (PRIOR_ART.md §7): no paper predicts
OTHER models' per-query cost from a foreign model's prefill, and none measures its value vs the median.
**Established (LCB, test 341):** +4.4 / +9.6 / +4.3pt at 0.02 / 0.05 / 0.10c matched spend (CIs exclude
0); 1.3-1.6x cheaper at matched accuracy (hull, CI pending); within 1.1-1.2x of an oracle; replicates on
the gpt-oss-only ladder (+3.4 / +5.2 / +2.6); null on BCB (flat accuracy ladder -- the "when" story).
Chart: analysis/cost_head_gain.png. With a free verifier the gain is mostly absorbed (cost at matched
accuracy: analysis/priced_verification/cost_at_matched_accuracy.py) -- the claim is verifier-free.
**In flight:** CodeContests replication (700 reference-validated Codeforces problems, 5 routes x 2-4
draws, `$R/cc_pool/full`, prefill job cc_prefill); SWE-Smith replication (1432 problems, 4 routes, real
labels, in/out priced separately, `offline_router/swesmith_cost_head.py`, prefill job swesmith_prefill).
**To do:** bootstrap CI on the matched-accuracy ratio; "when does it pay" map over rung subsets x prices.
**Output:** 4-page workshop paper ("The Price Isn't Constant"); abstract drafted in chat 2026-09-26.
Caveat to state: the prefill is free only if the router already runs it for correctness.

### Track B (riskier): routing for verification -- who writes the tests, when to buy them, when to trust them
**Setting.** Candidates x tests pass matrix. Actions: draw a candidate from generator g; buy a test from
writer t (issue-only, or GUIDED by the current candidates' disagreement); running every test on every
candidate is automatic (~free: 1 vCPU sandbox ~$1.85e-5/s, ~1 s per check); submit; abstain. Verification
cost is dominated by WRITING the test (§3.3), so the lever is which (possibly cheap) model writes it.
**Baselines:** B0 no test (route once, submit); **B1 self-verification: one routing action, the same model
writes solution AND tests** (primary baseline); B2 fixed best writer; B3 all-writer vote; B4 label-free
cheap-first escalation.
**Settings:** S1 one-shot (route once for generator g, once for tester t, from the prefill; is the best
pair ever != (g,g)?); S2 sequential pass-matrix growth with supervised per-writer reliability (fit on the
train split) and a value-of-information per unit cost rule; S3 guided (candidate-conditioned) tests.
**Metrics:** cost at matched accuracy (headline), accuracy at matched spend, selective accuracy vs coverage.
**Evidence so far (SWE Verified pilot, 100 instances):** tests carry signal (dsv4f +14.7, vote +17.1 over a
random patch pick); weakly tied to difficulty (rho ~ +0.2); per-instance writer choice by ACCURACY does not
beat always-dsv4f (-0.6) but cheap-first escalation matches the vote at 3.7x lower writing cost; oracle
cheapest-sufficient writer 66.6% @0.086c vs dsv4f 61.0% @0.113c (headroom). Condition 3 passes vs
cheap/mid generators, fails vs Opus (checks ~free there).
**Phases:** P0 unified pass-matrix format (LCB + SWE pilot) 1/2 day; P1 B0-B4 + S1 1/2-1 day; P2 S2
1-1.5 days; P3 data in parallel (Verified to 369 + self-tests for all generators + 2-3 extra patches for
cheap generators; then SWE-Smith 1432 as the scale set); P4 S3 guided tests ~1 day.
**Kill points:** after P1, if no (g,t) pair beats B1 at matched cost, one-shot tester choice is dead and
S2/S3 must carry it; after P2, if S2 is not >=20% cheaper at matched accuracy than the best of B1-B4, we
publish the problem setting + testbed + baselines as the contribution.
**Dev vs evidence:** LCB is the development benchmark only (writers must solve to write expected outputs,
no label-free validity check, 84% base rate); SWE is the evidence.

Correction to §3.3: Verified's "qwen30" route is Qwen3-Coder-30B-A3B ($0.07/$0.28 per M), so its patch
costs ~0.08c, not 0.16c. No conclusion changes.

### 4.A.1 Track A updates (2026-09-26)
- **LCB, honest protocol** (`analysis/costhead_matched_accuracy_ci.py`, commit cecf8fd): operating points chosen
  on calibration, applied once to test, paired bootstrap. Cost ratio ours/paper at calibration targets
  60/65/70/75/80/85%: 0.51 [0.41,0.65] / 0.52 [0.41,0.64] / 0.51 [0.40,0.64] / 0.62 [0.52,0.73] /
  0.86 [0.77,0.95] / 0.96 [0.89,1.04]. Test accuracy of ours lands 1.4-2.3pt lower at the low targets (n.s.);
  reading the paper arm at our achieved accuracy gives ~0.57x / 0.66x / 0.83x. Headline: **~30-45% cheaper at
  60-75% accuracy, ~15% at 80%, none at 85%.**
- **SWE-Smith replication: NULL** (`offline_router/swesmith_cost_head.py`, 286 test problems, 4 routes, real
  labels, input/output priced separately). Matched spend: deltas -0.9..+0.6pt, all CIs include 0; matched
  accuracy ratio 0.92-1.08, CIs include 1. Why: (1) cost is dominated by KNOWN input tokens (~4.8k in vs ~1k
  out; patches, not long reasoning), so the median rule already prices most of it; (2) output length is
  weakly predictable (test R2 0.11-0.23 vs much higher on LCB reasoning outputs); (3) near-flat accuracy
  ladder (29/39/40/47%). Fits the "when" story: per-query cost matters when OUTPUT (reasoning) dominates the
  bill and varies with the query, and when the ladder is steep. State it as a boundary, not a failure.
- CodeContests: collection running (`$R/cc_pool/full`), prefill done (cc_prefill SUCCEEDED).
- **Mechanism (LCB, ~70% operating point; background agent, commit 32a34a6):** both arms use the same models
  in the same mix (~70% oss20lo / 30% dsv4f); they route 31% of problems differently. 48 problems move
  dsv4f -> oss20lo where dsv4f would reason long (0.405c -> 0.027c, accuracy 78% -> 20%); 56 move oss20lo ->
  dsv4f where dsv4f stays short (0.011c -> 0.071c, 58% -> 96%). dsv4f's per-problem cost spans ~63x: a
  per-model median misprices it in both directions. This is the paper's mechanism figure.
- **With abstention in both arms** (`analysis/costhead_matched_accuracy_ci_abstain.py`): still ~36-39%
  cheaper at 60-65% (n.s. accuracy gaps); at 70% ours lands 4.6pt lower on test (not like-for-like).
  Abstention helps the median arm more (it skips hopeless problems it would misprice) but the advantage
  survives at 60-75%.

### 4.A.2 Verifier regime REVISITED (2026-09-26): exact Bellman + learned cost beats the fixed cascade on LCB
`analysis/dist_bellman/compare.py` (Bellman replay `replay_bellman_verification.py --beliefs content,dist`,
horizon 8, free perfect verifier; fixed cascade + RoR-style counts from Codex's analysis; cost at matched
accuracy on each arm's calibration-selected test points; paired bootstrap).
LCB cost ratio vs fixed cascade at 70/75/80/85/90% accuracy:
- Bellman, content prior, LEARNED cost: 0.87 / 0.86 / **0.74 [0.63,0.94] / 0.71 [0.61,0.84] / 0.80 [0.69,0.89]**
- Bellman, distributional prior, learned cost: 0.82 / 0.80 / **0.70 [0.59,0.89] / 0.70 [0.58,0.82] / 0.78 [0.69,0.92]**
- Bellman, content prior, global cost: 0.95 / 1.00 / 0.86 / **0.80 [0.74,0.89] / 0.77 [0.72,0.84]**
- RoR-style counts: 1.19 / 1.05 / 0.94 / 0.95 / 0.99 (no gain over the cascade)
=> ~25-30% cheaper than the fixed cascade at 80-90% on LCB, CIs exclude 1. The earlier "cascade ~ optimal
with a verifier" came from ONE-STEP greedy policies; the exact finite-horizon DP plus the per-problem cost
head does beat it. The cost head is worth ~0.10-0.12 of the ratio here; distributional beliefs add only
~0.02-0.04 over the hyperbolic-decay prior.
BCB: Bellman arms are cheaper only at 55% (0.76-0.93) and worse from 65% up; they plateau below 68% (they
give up), and the distributional head (median pi0 8-18%) gives up even earlier (2.0x at 65%). Cascade wins there.
Distributional head items 1-4 (NEW `fit_entry_distribution.py`): entry beliefs calibrated on LCB (e.g. 0.690
vs 0.682 observed), beat a pooled Beta on held-out marginal LL on every route of both datasets; BCB ranking
weak (corr 0.18-0.36) and means 3-6pt low after the level fix.
To do: gpt-oss-only pool; horizon >8; priced checks (v>0); why BCB plateaus (give-up threshold).
- **Ablation: WHICH prefill signal pays in the verifier regime** (`--beliefs global` = no prefill, route base
  rates). LCB cost ratio vs fixed cascade at 80/85/90%: Bellman no-prefill 0.88 / 0.87 [0.80,0.95] / 0.87
  [0.81,0.91]; + cost head only 0.73 [0.60,0.86] / 0.75 [0.65,0.87] / 0.83 [0.69,0.92]; + success prior only
  0.86 / 0.80 / 0.77; both 0.74 / 0.71 / 0.80; both with dist prior 0.70 / 0.70 / 0.78.
  => (1) exact planning alone ~13% cheaper than the cascade at 85-90%; (2) the prefill COST head is the largest
  per-problem gain on top of planning; (3) prefill SUCCESS beliefs add little once cost is in. Track A's thesis
  holds WITH a verifier too -- but only under a planning policy (one-step greedy policies can't use cost).
  BCB: no-prefill Bellman is best at 55% (0.62); all Bellman arms give up before 68%.
- **CORRECTION / mechanism (`analysis/dist_bellman/why_planning.py`, LCB test, matched by cheapest point
  reaching each accuracy, everything selected on TEST, cascade over ~540 enumerated plans):**
  * Greedy loses because it is myopic: at high targets it opens with the strong route (95%: dsv4f first on
    100% of problems, 0.354c vs cascade 0.223c); at mid targets it abandons the cheap route after one failure
    (switches after 72% of failures).
  * Bellman with NO per-problem information is itself a fixed sequence (identical beliefs for every problem),
    and the best enumerated fixed plan matches or beats it (93%: 0.199c cascade vs 0.221c). The earlier
    "planning alone 13% cheaper" was a selection-protocol artifact (cascade: 6 plans chosen on train+cal;
    Bellman: 14 V values on cal) -- RETRACTED.
  * With per-problem info (cost head + prefill prior): 85% 0.121c vs 0.157c (-23%), 90% -2%, 93% -6%, but 95%
    +59% (over-resamples the cheap route ~3.4 draws/problem, gives up late: 40% of spend on never-solved
    problems; fixed decay pseudo=2.0 likely too optimistic).
  * The 0.70-0.80x table above and this test-selected table bias in opposite directions (few vs ~540 cascade
    plans). Truth likely a modest gain at 85-93% on LCB. TO DO: one protocol for all arms (select on cal,
    same enumerated cascade plan space, fine V grid); fit the post-failure decay on train; CodeContests replay.
- **DEFINITIVE (supersedes the 0.70-0.80x table and the test-selected table above).**
  `analysis/dist_bellman/clean_comparison.py`: ONE protocol for all arms -- operating points chosen on
  CALIBRATION (upper hull, two-point mix hitting the target), applied once to TEST; cascade chooses from all ~730
  enumerated plans; Bellman from a 60-point value grid; paired bootstrap. Free perfect verifier.
  LCB: planning without per-problem info ~= cascade (cost ratio 1.03 at 80-90%). Bellman + prefill (cost head +
  success prior) lands HIGHER ACCURACY at ~equal cost at the mid targets: 80%: +4.4pt [+1.7,+7.1] at 1.03x cost;
  85%: +2.8pt [+0.8,+4.9] at 0.99x; at >=90% it is equal or more expensive (93%: 1.29x, 95%: 1.44x).
  Cost head alone: 0.86x [0.77,0.97] at 80% (acc +1.3 n.s.), ~1.0 elsewhere.
  BCB: every Bellman arm costs as much or more than the cascade (1.03-1.58x), no accuracy gain.
  Decay fitted on train (k=0.25-0.5, faster than 2) does NOT help; it makes things costlier.
  => With a free verifier, the best fixed cascade is essentially as good as anything we built; the only real
  gain is +3-4pt accuracy at equal cost around 80-85% on LCB, using the prefill. The earlier "25-30% cheaper"
  came from comparing against a weak cascade candidate set (6 plans) -- RETRACTED.

### 4.B.1 Track B: one-shot + abstain, HELD-OUT on 368 Verified instances (2026-09-26)
`offline_router/analyze_oneshot_abstain.py --cv 5` (pair and rule chosen on 4/5 of instances, scored on 1/5;
open generators oss20/qwen30/oss120, testers oss20/qcoder30/dsv4f/oss120/devstral; all 368 instances,
not just the disagreement-selected pilot). Utility +1 correct / -lambda wrong / 0 abstain:
lambda 0: tests lose (-0.120); 0.5: +0.020 [-0.026,+0.062]; **1: +0.160 [+0.095,+0.226]** (gpt-oss-120b +
dsv4f test, strict, chosen in 4/5 folds); 2: +0.005 n.s.; 4: -0.011 n.s. The pilot's +0.09/+0.14/+0.10 at
lambda 0.5/1/2 was optimistic (selection on the same 100 disagreement instances). Real but NARROW: tests pay
only around lambda ~1. Self vs other tester (precision/coverage): gpt-oss-20b 65%/31% self vs 72%/18%
qcoder30; gpt-oss-120b 72%/43% self vs 76%/23% devstral -- other testers more precise but accept less.
Redraw verdicts (scripts x 4 redraw patches) queued after plan D labelling -> then the sequential setting.
- **SWE-Smith REASONING pool (plan D; 500 instances x 5 open reasoning routes, real Daytona labels):**
  pass rates oss20lo 23% / oss20md 26% / dsv4f 34% / oss120md 34% / oss120hi 36% (unconverted draws count as
  unresolved). `offline_router/swesmith_reason_cost_head.py` (275/75/150 split; in/out priced separately):
  matched spend ours - paper -1.1..-0.3pt (CIs mostly include 0); matched accuracy ratio 0.81-1.06, all CIs
  include 1. Cost head works for gpt-oss (log-output test R2 0.49-0.72) but weak for dsv4f (0.20); the success
  head is ~flat (C=1e-4 chosen) -> little to route on. NULL again on SWE, now with reasoning models; small test
  set (150). Track A's positive evidence remains LCB only (+ gpt-oss-only LCB ladder); CodeContests decides.
  Note: plan D's first labelling pass evaluated unconverted text (converter crashed on the training-stack import
  on a CPU node) -- relabelled after converting with convert_swesmith_patches_light.py.

### 4.B.2 Track B SEQUENTIAL result (2026-09-27; `offline_router/analyze_trackB_seq.py`, 368 Verified, open only)
Ladder oss20 x3 -> qwen30 x3 -> oss120 x1 (redraws labelled + all 5 writers' scripts run on them); 5-fold,
settings chosen on train folds as the cheapest reaching each target, scored held-out; paired bootstrap.
Ceiling (some open candidate correct) 66.6%.
- **Choosing a DIFFERENT tester beats self-verification on cost.** Cascade with dsv4f's tests vs the
  self-verification cascade: target 50%: 53.0% @0.221c vs 48.9% @0.378c (acc +4.1 [+0.8,+7.3], cost 0.58x
  [0.54,0.63]); target 55%: 54.3% @0.336c vs 54.6% @0.500c (acc -0.3 n.s., cost 0.67x [0.62,0.72]).
  qcoder30 tests also better than self at 50% (+3.5 [+0.3,+6.8] at 1.02x).
- **No-test routing is a strong baseline:** route once to gpt-oss-120b = 52.4% @0.180c. Tests only pay ABOVE
  that ceiling: dsv4f-tested cascade 54.3% @0.336c; reliability-weighted posterior policy is the only one
  reaching ~57% (57.3% @0.646c, +2.7 [+0.0,+5.7] vs self).
- So: vs self-verification (the natural baseline) a cheap cross-model tester is ~33-42% cheaper at matched
  accuracy; vs no tests, testing buys +2-5pt beyond the best single model at 1.9-3.6x its cost.
Caveats: draws that produced no patch are skipped for free (mild optimism for cascades); single SWE dataset;
writers' costs include only writing + ~1.5 s runs; posterior uses naive-Bayes independence across writers.
