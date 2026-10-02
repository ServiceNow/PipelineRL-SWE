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

### 4.A.3 CodeContests replication -- DEFINITIVE (2026-09-27; 700 problems, split 346/87/267)
Pass rates oss20lo 51.6 / oss20md 72.4 / dsv4f 84.1 / oss120md 79.2 / oss120hi 88.1%.
**Bug found and fixed in `activation_cost_preds.py`.** The last step (dollar-space linear recalibration on calibration)
had slope 1.2-1.7 with a negative intercept on CC, pushing the low tail to <= 0, which the clip turned into ONE token:
oss120hi was priced ~free on 68/700 problems, so every learned-cost policy bought it (that was the 2-3.5x
"learned-cost Bellman" anomaly and the 1.95x one-shot ratio at 60%). Fix: floor every prediction at the cheapest
per-problem mean observed on TRAIN for that route. LCB re-run with the fix: only 57 entries change (55 oss120md,
2 oss20lo), headline identical (ratio 0.51 / 0.51 / 0.51 / 0.62 / 0.86 / 0.97 at 60-85% targets;
analysis/costhead_matched_accuracy_ci_lcb_floored.json). Unfloored preds kept as cost_preds_unfloored.jsonl.
**One-shot (the Track A claim), after the fix** (analysis/costhead_matched_accuracy_ci_codecontests.json):
target 60/65/70/75/80% -> ratio 1.05 / 0.93 / 0.82 [0.69,0.99] / 0.88 [0.78,0.99] / 0.87 [0.78,0.96], but ours lands
1.4-2.5pt LOWER in accuracy. At EXACTLY matched accuracy (interpolating the paper-rule curve at ours' accuracy):
CC x1.04 / 0.97 / 0.95 / 0.93 / 0.94 -- a ~5% edge, not meaningful. Same view on LCB: x0.57 / 0.57 / 0.66 / 0.82.
=> **CodeContests does NOT replicate the LCB effect** (no harm, ~5% at most). Likely why: CC cost head dollar R2 on
calibration only 0.21-0.38 (LCB up to 0.79 with --rich), and dsv4f's per-problem cost spread is ~4x p90/p10 on CC vs
~63x range on LCB -- the rerouting mechanism (oss20lo <-> dsv4f) has little to exploit.
**Verifier regime (free perfect verifier), after the fix:** vs the fixed cascade at matched accuracy 45-70%:
Bellman content/global 0.90-1.04, content/learned 0.88-1.16, dist/global 0.88-1.37, global/learned 1.11-1.34,
RoR-style counts 1.03-1.14 -- no arm beats the cascade (all CIs include 1 or are above it). Matches LCB/BCB
(`clean_comparison`): planning ~= cascade.
**Track A scorecard:** LCB positive (and gpt-oss-only LCB ladder); BCB null (flat ladder); SWE-Smith null (x2);
CodeContests null. The claim is one dataset deep -- a workshop paper must frame it as "when does per-query cost
pay" (large per-problem cost spread on a route that is also accurate), with CC/BCB/SWE-Smith as the negative cells.

### 4.B.3 Per-instance test-writer choice, sequential (2026-09-28; 368 Verified, open only)
Question: does choosing the test-WRITER per instance beat always-dsv4f? Two steps.
1. **Label-oracle ceiling** (`offline_router/trackB_oracle_writer_ceiling.py`, analysis/trackB_oracle_writer_ceiling.json):
   per instance, the writer maximising correct - mu*cost chosen WITH labels, ladder length shared. Cost vs the best
   fixed writer at matched accuracy: 45% x0.70 [0.46,0.78], 50% x0.42 [0.32,0.74]; max accuracy 60.9% vs 54.9%.
   Oracle picks the cheapest writer that happens to work (oss20 48%, dsv4f 23%, qcoder30 15%). INFLATED: with 12-25%
   false accepts, label-picking among 5 noisy tests exploits luck (the pilot's split-half version gave -0.6pt). Not a kill.
2. **Label-free per-instance escalation, held out** (`analyze_trackB_seq.py --families escalate:*`, 5-fold, same protocol
   as 4.B.2; analysis/trackB_escalation.log). Families: `invalid` (buy writers cheapest-first until one's test is valid),
   `confirm` (second writer re-checks a pass), `rescue` (second writer re-checks a fail). Held-out, acc @ cents:
   | target | fixed dsv4f | escalate:invalid | escalate:rescue | escalate:confirm | self | route-once oss120 |
   | 50% | 53.0 @0.221 | 50.0 @0.232 | 52.2 @0.216 | 47.8 @0.315 | 48.9 @0.378 | 52.4 @0.180 |
   | 52% | 51.6 @0.283 | 50.3 @0.291 | 52.2 @0.280 | 50.5 @0.384 | 51.4 @0.437 | 52.4 @0.180 |
   | 55% | 54.3 @0.336 | 53.8 @0.337 | 52.2 @0.360 | 53.5 @0.444 | 54.6 @0.500 | 52.4 @0.180 |
   Train folds almost always pick an oss20 -> dsv4f order. **No escalation rule beats always-dsv4f** (rescue ties it at
   50-52%; invalid ties at 55%; confirm is worse). All tested policies still lose to route-once below ~52%.
**Verdict:** per-instance writer choice has no demonstrated value in the sequential setting. The writer claim stays
"a cheap cross-model writer (dsv4f) beats self-verification by 33-42% at matched accuracy". A learned writer router is
not worth building on this data (label-free rules already sit at the fixed writer; the oracle headroom is mostly luck).

### 4.A.4 Headroom x capture across 7 pools (2026-09-28; `analysis/cost_headroom/decompose.py --curve`)
HEADROOM = cost saved at matched accuracy by PERFECT per-problem cost vs the paper rule (input + median output);
LEARNED = the 4B-prefill head; test frontiers, same success head for every arm, paired bootstrap. Market in/out
prices for every pool (legacy blended prices give the same picture; analysis/cost_headroom/decompose_all.log).
| pool (market prices) | headroom [95% CI] | learned [95% CI] | head dollar R2 on the key routes |
| LCB 5 routes | 47.3% [39.7,52.9] | 34.4% [25.6,40.0] | 0.44-0.51 |
| LCB gpt-oss only | 30.8% [23.3,36.4] | 18.7% [12.4,23.6] | 0.44-0.56 |
| TACO (scout/oss20/oss120) | 36.1% [24.9,46.0] | 0.1% [-7.5,9.0] | 0.14-0.35 |
| CodeContests | 21.8% [13.5,29.1] | 0.7% [-9.5,8.2] | 0.22-0.27 |
| BCB 5 routes | 20.2% [9.9,29.8] | -4.3% [-26.5,15.5] | dsv4f 0.04, oss120hi 0.13 |
| SWE-Smith reasoning | 14.2% [-2.1,29.6] | 7.5% [-5.5,20.0] | 0.10-0.85 (n=150 test) |
| BCB gpt-oss only | 6.7% [-6.2,17.3] | 6.8% [-4.1,18.9] | |
**Findings.** (1) Headroom is large and general for reasoning-model pools (20-47% on 5 of 7, CIs excluding 0): output
length varies 10-90x per route across problems and 81-96% of its variance is between problems. CORRECTION of an
earlier claim: CodeContests does not lack variability (dsv4f p90/p10 22-26x, not ~4x -- that was the spread of the
predictions). (2) The bottleneck is PREDICTABILITY. A calibrated synthetic head with controlled R2 traces a convex
capture curve: ~0 gain below dollar R2 ~0.3, most of the headroom only above ~0.5-0.6. The 4B probe reaches ~0.5 only
on LCB. Real heads sit on the synthetic curve on LCB x2, CodeContests and BCB gpt-oss-only; TACO and BCB-5 fall below it
(unweighted mean R2 hides route heterogeneity). (3) So Track A's story: "the price isn't constant (20-47% headroom);
capturing it is a cost-PREDICTION problem, and a cheap probe clears the bar on 1 of 7 pools." Next: stronger cost
predictors (each route's OWN prefill -- TACO has oss20/oss120 activations; fine-tuned encoder) to move pools along
the curve; a non-reasoning contrast (RouterBench activations exist); a pre-registered math pool.

### 4.A.5 WHY is cost predictable on LCB and nowhere else? (2026-09-28; `analysis/cost_headroom/why_predictable.py`,
`metadata_cost_head.py`, decompose_metadata.json)
Target: log mean output tokens per problem, per route. Ruled out: (a) label noise -- the best achievable R2 given
draw noise is 0.92-0.99 in every pool, CC included; (b) runaways -- ~0 capped draws except TACO's two small routes
(9% of draws at the cap, 22-27% of problems straddling it: TACO's small routes are partly unpredictable by nature);
(c) platform mixture -- platform explains 0% of LCB length.
**The driver is a coarse, legible difficulty signal.** On LCB the easy/medium/hard tier alone explains 57-66% of log
length (balanced 265/312/315); the 4B probe recovers the tier with CV R2 0.72; holding platform x tier fixed the
head's R2 falls from 0.51-0.72 to 0.33-0.41 (dsv4f -0.08). A metadata-only cost head (tier + platform + statement
length, NO probe) already gets 24.7% of LCB's 46% headroom; the probe adds the rest (34.4%).
On CodeContests the Codeforces rating explains only 37-49% of log length, the probe reads the rating with R2 0.50, and
even the TRUE rating as a feature captures ~1% (meta+probe 4.2%, n.s.). TACO: tier explains 10-31%, probe reads it
at 0.35, metadata gives nothing.
**Threshold, not a slope.** The capture curve (4.A.4) is convex: gains start near log-R2 ~0.5 (rho ~0.7) on the routes
that matter. LCB's head sits at 0.51-0.72 and clears it; CC 0.18-0.43, TACO <= 0.29, BCB's key routes (dsv4f,
oss120hi) <= 0.12 do not -- and CC's true rating (0.37-0.49) falls just short too. Mechanism to state in the paper:
a routing decision flips only when the cost error is smaller than the utility margin between routes, so a weak
cost signal reroutes about as often wrongly as rightly until it crosses that margin.

### 4.A.6 Head-to-head with the literature's cost predictors (2026-09-28; `analysis/cost_headroom/baseline_cost_heads.py`,
analysis/cost_headroom/head_to_head.log). Same success head, same decomposition, market prices. Learned gain at matched
accuracy vs the paper rule [95% CI]; test log-output R2 in brackets after.
| pool | headroom | tuned 4B head | plain-ridge 4B probe | MixLLM-style (jina-code emb -> MLP+RF+kNN) | prompt-feature GBM | own-model prefill |
| LCB | 46-47% | 34.4% | 35.6% [27.1,41.9] (R2 .69-.76) | 14.1% [5.3,21.0] (.22-.30) | 12.3% [0.7,21.4] (.27-.35) | 34.0% (.64-.76) |
| CodeContests | 21.8% | 0.7% | 2.0% [-6.1,9.0] (.34-.47) | 0.8% (-.52..-.13) | -11.1% [-20.9,-1.3] | -- |
| BCB | 20.2% | -4.3% | -3.3% (.15-.71) | -4.5% | -10.1% | -- |
| TACO | 36.1% | 0.1% | 2.1% (.31-.47) | 8.3% [0.2,16.9] (.05-.09; likely noise) | -18.9% [-38.3,-5.4] | -- |
| SWE-Smith | 14.2% | 7.5% | 24.8% [8.2,36.0] (!) | void (adapter stored no statements) | void | -- |
Findings: (1) the 4B probe beats the literature's text predictors everywhere; on LCB they capture only 12-14% vs 34-36%.
(2) No pre-generation predictor captures CodeContests / BCB headroom; prompt-feature GBM actively hurts (-11 to -19%).
(3) Each model's OWN prefill is no better than the 4B scout's (LCB). (4) Our tuned calibration pipeline LOWERS R2 vs plain
RidgeCV (CC .18-.43 -> .34-.47, TACO -.25-.29 -> .31-.47) without changing capture on LCB -- simplify the head.
(5) CC's plain-ridge R2 .34-.47 still captures ~2%: consistent with the threshold (4.A.4). (6) SWE-Smith: learned gain >
"headroom" -- with ONE draw per problem the realised cost carries success information (failed patches run longer), so
perfect cost knowledge is not an upper bound there; small test set (150). Treat as a flag, not a result.

### 4.A.7 RouterBench contrast + partial-generation predictor on CodeContests (2026-09-28)
**RouterBench (non-reasoning; 11 chat models, 6k-query stratified sample, prices recovered from total_cost):**
headroom 10.5% [7.4,12.8] vs 20-47% for reasoning pools (output 17-42% of the bill, 24-58 tokens); the probe captures
8.5% (0.81) because cost is input-dominated. Prediction ("small headroom for non-reasoning models") held.
**Partial generation (CodeContests; `codecontests/collect_prefixes.py`, `analysis/cost_headroom/prefix_cost_head.py`):**
first 512 tokens per route (providers overran the cap: 527-1414 tokens mean). Test log-output R2: probe 0.18-0.43 ->
+ gpt-oss-20b-low's prefix 0.50-0.54 on EVERY route (dsv4f 0.18 -> 0.54, oss120hi 0.19 -> 0.50); own prefixes no better
(0.37-0.53) and 15x the price. But in one-shot routing it does NOT pay: gain vs paper rule 6.2% [-2.8,12.7] if the prefix
were free; -16.5% [-28.3,-9.0] with the prefix charged (0.008c/problem = up to +47% at the cheap end); -2.3% [-11.6,5.1]
with continuation credit (prefix free when oss20lo is the chosen route). Per accuracy: x0.87 at 51% -> x1.18 at 80%.
Even free, the prefix head is WORSE than the paper rule at the expensive end (x1.12 at 80%): the dollar-miscalibration
deficit (capture ~ R2 - 0.2 in 4.A.4's curves) -- fix calibration in dollars before revisiting prefixes.

### 4.A.8 Calibration, headline, robustness (2026-09-28)
- **Dollar calibration does not help** (`analysis/cost_headroom/dollar_calibrate.py`, isotonic on the calibration split):
  heads were already unbiased in dollars (0.8-1.1x); gains before -> after: LCB 34.4/35.6 -> 34.0/33.2, CC 0.7/2.0 ->
  -0.1/2.7, TACO 0.1/2.1 -> 3.2/3.1, BCB -4.3/-3.3 -> -5.2/-2.5, CC prefix 6.2 -> -1.2. The apparent "capture ~ R2 - 0.2"
  deficit was the log-R2 axis: against DOLLAR R2 the synthetic capture exceeds R2, and every real head sits ON the
  synthetic curve (LCB head 0.48 dollar R2 -> 34% vs synthetic ~34%; CC prefix -> 6.2% vs ~6.5%). The limit is
  prediction quality, not calibration.
- **Headline, simpler head (plain RidgeCV probe), market prices, honest calibration-chosen protocol, LCB test 341:**
  cost ratio 0.63 [0.52,0.77] at 70%, 0.67 [0.55,0.80] at 75%, 0.73 [0.63,0.83] at 80% with accuracy matched
  (+1.3/+1.5/-0.1, CIs incl 0); 0.94 at 85%. 60-65% targets land 3-4pt lower in accuracy -> not matched, excluded.
  Claim: 27-37% cheaper at 70-80% accuracy. (analysis/costhead_ci_lcb_market_cost_preds_probe.json)
- **TACO MixLLM-style 8.3% was noise:** 5 seeds give 1.3-6.2%, every CI includes 0.
- **Probe model** (`analysis/cost_headroom/probe_model_compare.py`; same plain ridge, only the prefill changes). Log-output
  R2 on the reasoning routes: Qwen3-4B-Thinking beats Instruct everywhere but modestly (LCB .69-.76 -> .73-.79, CC
  .34-.47 -> .40-.54, TACO oss20/oss120 .36/.47 -> .42/.54); Base is far worse (CC .14-.21); gpt-oss-20b's own prefill
  no better than Instruct (CC .34-.41), nor is gpt-oss-120b's (CC .37-.44, gain 0.6%). Routing gain Instruct -> Thinking: LCB 35.6 -> 34.6, CC 2.0 -> 6.3, TACO 2.1 ->
  -1.2 -- all within noise. A reasoning-model probe is slightly better at predicting length, not enough to move pools.

### 4.A.9 PRE-REGISTERED out-of-sample test: Omni-MATH-500 (rule committed 2026-09-28 02:45 ET, BEFORE any step-1 data)
Pool: Omni-MATH-500, the five LCB/CC routes, draws 4/3/3/2/2 (pilot n=40: acc 47/65/69/55/67%). Step 1 (cheap): gpt-oss-20b-low
x1 on all 500 + Qwen3-4B Instruct/Thinking prefills. Step 2 (~$17, pending sign-off): the full pool.
**Rule (fixed now):** from step 1 compute (a) the probe's 5-fold CV log-output R2 on gpt-oss-20b-low (the head's R2 on
the other routes was within ~0.1 of this one's on LCB/CC), (b) difficulty->length link (R2 of log length on the Omni
difficulty rating), (c) probe readability (CV R2 of the rating from the prefill). Predict, for the full pool, the
plain-ridge probe's gain vs the paper rule at matched accuracy (decompose.py, market prices):
  - (a) >= 0.50  -> GAIN: point estimate >= 10% and the 95% CI excludes 0.
  - (a) <= 0.35  -> NO GAIN: CI includes 0 (or the gain is negative).
  - in between   -> no directional call; report as a test of the capture curve only.
Also predicted regardless of (a): HEADROOM >= 15% (reasoning routes, 1.5k-18k output tokens, large length spread).
Whichever probe (Instruct/Thinking) scores higher on (a) is the one evaluated -- decided on step-1 data, stated before step 2.
**Step-1 result and the PREDICTION (committed 2026-09-28 02:50 ET, before step 2):** gpt-oss-20b-low on all 500: acc 0.46,
mean output 1542 tok, sd log length 0.90. (b) difficulty -> log length CV R2 0.51 (LCB tier 0.57-0.66; CC rating
0.37-0.49). (c) readability of the rating: Instruct 0.73, Thinking 0.75 (LCB tier 0.72; CC rating 0.50). (a) probe CV
log-output R2 on oss20lo: Instruct 0.66, Thinking 0.67 -> evaluated probe = Thinking.
**PREDICTION: GAIN -- the Thinking-probe plain-ridge head saves >= 10% vs the paper rule at matched accuracy, 95% CI
excluding 0; headroom >= 15%.** (Omni looks LCB-like on both rule inputs.)

### 4.B.4 Patch-aware test pilot: per-candidate writer selection is dead (2026-09-28; `offline_router/swe_patchaware_pilot.py`,
analysis/trackB_patchaware.log). 166 Verified instances (one correct + one wrong ladder patch each), writers oss20 /
dsv4f / qcoder30 each writing 2 independent PATCH-AWARE scripts per candidate (1992 scripts, $2.2 writing).
- Per writer (patch-aware): dsv4f valid 96%, beta 66.6%, alpha 17.5%, Lambda 3.81, 0.093c; oss20 94% / 37.7% / 15.4% / 2.45,
  0.033c; qcoder30 66% / 29.2% / 10.8% / 2.69, 0.055c (qcoder30: 184/664 replies had no code block, 166 from provider Novita;
  counted as no check). Draw-to-draw verdict agreement 81-85%.
- Pair selection: always dsv4f 74.5% @0.295c; oss20 61.1% @0.073c; qcoder30 59.2% @0.166c. Split-draw per-candidate
  writer ceiling (labels on one draw, scored on the other): max 69.0% @0.115c -> vs always-dsv4f -5.6pt [-8.3,-2.6] at
  equal cost and never reaches its accuracy; vs oss20 1.36x [1.20,1.52] cost at equal accuracy.
- **PRE-REGISTERED verdict: NO GO.** (The job's own printout said GO: the code tested "beats ANY writer" instead of
  the pre-registered "beats EVERY writer"; fixed, rerun -> NO GO.)
- Same-family rubber-stamping is ABSENT: false accepts oss20 on gpt-oss vs qwen patches 16.3% vs 14.4%; qcoder30 on qwen
  vs gpt-oss 10.6% vs 11.0%. Seeing the patch lowers dsv4f's false accepts ~20% vs issue-only (25.6 -> 20.3%, 18.8 -> 14.4%).
**Track B: parked.** The defensible content is a measurement (cheap cross-model writer beats self-verification; tests pay
only above route-once), fit for a section/appendix of the Track A paper, not a paper of its own.
**RESULT (2026-09-28 05:55 ET): PREDICTION CONFIRMED.** Full pool: 500 problems, valid draws 2000/1500/1473/1000/994 (14 dsv4f
draws still in flight, 0.2%; each affected problem keeps its other dsv4f draws), accuracy 46/62/75/63/70%, mean output
1.5k-17.7k tokens. Thinking-probe plain-ridge head: test log-output R2 0.72-0.81 (step-1 screen said 0.67). **Gain vs the
paper rule at matched accuracy 21.9% [8.9, 33.5]** (predicted >= 10%, CI excluding 0: YES). **Headroom 44.2% [31.8, 53.5]**
(predicted >= 15%: YES). (Instruct, not the pre-registered probe: 27.4% [15.0, 37.4].) analysis/cost_headroom/omni500.log.
The rule -- difficulty drives length AND the probe reads difficulty -> a cheap probe clears the capture threshold --
forecast a new dataset correctly before its data existed.

### 4.A.10 WHY is output length predictable? Variance decomposition (2026-09-28; `analysis/cost_headroom/why_decompose.py`)
Test problems, 5-fold CV R2 of log output length, probe = plain-ridge out-of-sample prediction. shared = length explained by
difficulty that the probe also captures; diff-only = difficulty the probe misses; probe-only = beyond difficulty.
| pool | probe R2 | empirical difficulty R2 | shared | diff-only | probe-only | labelled difficulty R2 (shared) |
| LCB | .73-.78 | .50-.58 | .47-.52 | .02-.06 | .21-.29 | .54-.65 (.53-.63) |
| Omni-MATH | .75-.83 | .34-.59 | .34-.53 | .00-.07 | .23-.49 | .53-.59 (.54-.58) |
| CodeContests | .42-.49 | .42-.49 | .30-.34 | .12-.17 | .09-.20 | .30-.47 (.29-.36) |
| TACO | .41-.52 | .17-.22 | .12-.18 | .03-.05 | .27-.33 | .13-.38 |
| BCB | .37-.71 | .08-.12 | .04-.09 | .01-.04 | .33-.63 | (no label) |
Reading: where it works (LCB, Omni) ~2/3 of the probe's explained length variance is SHARED with difficulty and the probe
misses almost none of the difficulty signal (<=.07); another .15-.29 is beyond difficulty. CodeContests: length is as
difficulty-driven as LCB, but the probe MISSES .12-.17 of it (it cannot read Codeforces difficulty well). TACO / BCB:
length is barely difficulty-driven (.08-.22); BCB's probe finds a lot beyond difficulty (.33-.63, library/boilerplate
"size"), but its key routes (dsv4f .37, oss120hi .49) and flat accuracy ladder (45-51%) keep capture at zero.
Mechanism check: within a problem-route, failed draws are NOT systematically longer than solved ones (x0.83-1.25; only
Omni dsv4f x1.89), so "hard is long" is a property of the PROBLEM (ICC .85-.95), not of failing: reasoning models spend
more tokens on problems they find hard even when they solve them.

### 4.A.11 Boosting the cost head: free ideas are null (2026-09-28; `analysis/cost_headroom/boost_cost_head.py`)
Frozen 4B Instruct probe, test log-output R2 (mean over routes): base ridge LCB .75, CC .46, TACO .46, BCB .58, Omni .75.
- Learning curve (25/50/75/100% of train): saturated by 50-75% everywhere (CC .24/.39/.47/.46; LCB .71/.74/.74/.75;
  Omni .68/.74/.73/.75; BCB .48/.54/.57/.58 mildly rising) -> NOT data-limited; more labels will not help much.
- Shared-factor (reduced-rank) multi-output head: identical to base (+-.005).
- Nearest-neighbour cost in probe space: worse (.27-.67); averaged with ridge: worse (-.02 to -.06).
- Pooling training data across datasets (route-matched, pool indicator): worse (CC .42, BCB .39, LCB .74, Omni .74).
Conclusion: the frozen prompt representation is the limit, not data or head. Combined with 4.A.6/4.A.8 (bigger / own /
Thinking / Base prefills, text predictors): prompt-only cost prediction on CC-like data is near its ceiling; closing the
difficulty gap (4.A.10: the probe misses .12-.17 on CC) needs information from generation (prefix R2 .50-.54 but pays for
itself only if nearly free) or a changed representation (fine-tuning).
- **#5 selective use of the learned cost** (`analysis/cost_headroom/selective_cost.py`; bootstrap-ensemble sigma, precision-
  weighted blend with the paper rule, tau chosen on calibration): test gain plain -> selective: CC 2.0 -> 2.8, TACO 1.0 ->
  1.2, BCB -1.5 -> -1.5, LCB 35.2 -> 35.2, Omni 24.3 -> 19.5. NULL. Ensemble sigma (0.04-0.27 log) is far below the actual
  error: the error is missing information (bias), not estimator variance, so the head cannot tell which predictions to
  distrust. Calibration-split gains are also unreliable at n = 87-136 (BCB calibration +27.5% vs test negative).
- **Abort-and-reroute on observed length: TAKEN** (agentic: SWE-Router 2607.00053, Fail-Fast Restart-Smart 2608.03222,
  TACIT-Switch 2608.27911, Doomed from the Start 2607.06503, EarlyEval 2609.02783). Dropped.
- Running: prompted probing (prefill with a difficulty question appended; CC + LCB, GPU); Codeforces-API ratings for the
  full CodeContests set (download of the remaining 35 train shards) -> train a rating reader on thousands of free labels.
- **Prompted probing** (difficulty question appended before reading the prefill): CC R2 .34-.47 -> .38-.48, gain 2.0 -> 2.2%;
  LCB 35.6 -> 34.4%. NULL.
- **#1 rating reader from free labels** (`analysis/cost_headroom/rating_reader.py`; 6362 CodeContests problems rated via the
  Codeforces API, outside the pool): readability of the rating on the pool 0.51 (pool-only CV) -> 0.46 (9x more labels):
  the frozen representation, not the label count, limits it. Stacking into the cost head: predicted rating 2.2%,
  even the TRUE rating only 4.0% [-4.7, 11.5] (log R2 .46-.52). NULL -- labelled difficulty explains too little of length;
  what drives length is the models' experienced difficulty (solve rate), which no label carries. Lowers the prior on
  fine-tuning to read difficulty; fine-tuning directly on length (~350 examples/pool) held.
**Boosting summary:** every prompt-side lever tried (bigger/own/Thinking/Base/prompted probes, text predictors, pooling,
low-rank, kNN, selective use, dollar calibration, difficulty labels at scale) leaves CodeContests at <= ~6%. The prompt-only
ceiling there is real; paper framing: diagnose, do not promise to fix.

### 4.A.12 Next pre-registered screens: AIME 1983-2024 (933) and OlympiadBench maths (674) -- rule committed BEFORE screen data
Identical rule to 4.A.9, no retuning: from gpt-oss-20b-low x1 + the Qwen3-4B Instruct prefill, (a) the probe's 5-fold CV
log-output R2 on oss20lo decides: >= 0.50 -> GAIN (>= 10%, CI excluding 0); <= 0.35 -> NO GAIN; else no call; headroom >= 15%
predicted for any reasoning pool. (b) difficulty -> length uses AIME's problem number (1-15); OlympiadBench has no graded label.
Full pools (the five gpt-oss/dsv4f routes, as Omni) only with sign-off; prefer one predicted GAIN and one predicted NO GAIN.
**AIME screen (2026-09-28 13:37 ET, before any full pool):** gpt-oss-20b-low on 931: acc 0.56, mean output 1896 tok, sd log
length 0.58 (Omni 0.90). (b) problem number -> log length CV R2 0.14; (c) readability of the problem number 0.42;
(a) probe CV log-output R2 on oss20lo **0.13** -> **PREDICTION: NO GAIN** (CI includes 0 or negative); headroom >= 15%.
**OlympiadBench screen:** gpt-oss-20b-low on 674: acc 0.63, mean output 1371 tok, sd log length 0.73; no graded difficulty
label; (a) probe CV log-output R2 **0.48** -> **no directional call** (between 0.35 and 0.50). Not a decisive test; AIME is.
**Omni-MATH, deployable protocol** (operating points chosen on calibration n=75, applied once to test n=150, Thinking probe,
market prices; analysis/costhead_ci_omni500_thinking.json): 60% target 0.70x [0.56,0.86] cost at acc -0.8 (n.s.);
65% 0.73x [0.58,0.88] at -0.7 (n.s.); 70% 0.91x [0.79,1.04] with acc +2.7 [+1.1,+4.5]; 75% 1.00x with acc +2.1 [+0.5,+3.9];
80-85% unreachable. => 27-30% cheaper at matched accuracy at 60-65%, same cost + 2-3pt accuracy at 70-75%.

### 4.A.13 Six more screens (2026-09-28; rule of 4.A.9 unchanged, committed before screen data)
ZebraLogic grid (1000; difficulty = houses x features), Knights & Knaves (700; 2-8 people), SuperGPQA (1000; easy/middle/hard),
MMLU-Pro (1000; by subject), BIG-Bench Extra Hard (1000; by task), APPS stdin/stdout (1000; intro/interview/competition).
Loaders + graders `math_pool/reasoning_datasets.py` (each grader: correct reference answers pass 100%, perturbed answers 0%).
Step 1 = gpt-oss-20b-low x1 + Qwen3-4B Instruct prefill; decision variable (a) as in 4.A.9. No full pool without sign-off.
**Screen results (2026-09-28 14:03 ET, before any full pool)** -- gpt-oss-20b-low x1 + Instruct prefill (analysis/cost_headroom/screen_six.log):
| dataset | n | oss20lo acc | mean out | sd log len | (b) label->length | (a) probe R2 | PREDICTION |
| Knights & Knaves | 700 | 0.85 | 1052 | 0.64 | 0.54 (people) | 0.61 | GAIN |
| SuperGPQA | 1000 | 0.29 | 309 | 0.79 | 0.26 (tier) | 0.58 | GAIN |
| MMLU-Pro | 1000 | 0.59 | 343 | 0.82 | -- | 0.65 | GAIN |
| BIG-Bench Extra Hard | 1000 | 0.22 | 617 | 1.33 | -- | 0.28 | NO GAIN |
| ZebraLogic | 995 | (INVALID) | 1951 | 1.17 | 0.40 (size) | 0.34 | (NO GAIN on length; UNGRADABLE) |
ZebraLogic: the public allenai/ZebraLogicBench test solutions are REDACTED ("___"); accuracy 0.00 is an artefact. My grader
check built "correct" answers from those placeholders and passed trivially -- graders must be checked on non-placeholder
references. Not a routing pool unless graded solutions are found. Caveats: K&K oss20lo already 85% (ladder may be flat at
the top); SuperGPQA / MMLU-Pro outputs are short at low effort (309-343 tok) -- headroom depends on the high-effort routes.
Running tally of decisive pre-registered calls: GAIN Omni (CONFIRMED), K&K, SuperGPQA, MMLU-Pro; NO GAIN AIME, BBEH. APPS pending.
| APPS (stdin/stdout) | 1000 | 0.59 | 798 | 0.87 | 0.34 (tier) | 0.61 | GAIN |
Updated tally of decisive pre-registered calls: GAIN Omni (CONFIRMED), K&K, SuperGPQA, MMLU-Pro, APPS; NO GAIN AIME, BBEH.

### 4.A.14 AGENTIC cost from public trajectories -- SWE-rebench July 2026 (2026-09-28; no API spend)
`ibragim-bad/swe_rebench_07_2026_trajectories`: 111 SWE tasks x 17 participants (13 standalone models in one scaffold + Claude
Code / Codex / Cursor / Junie) x 5 runs = 9435 runs with tokens, steps, resolved. Standalone runs priced from tokens at
OpenRouter list prices incl. cache-read rates (`analysis/cost_headroom/agentic_swe_rebench.py`); Claude Code / Junie use
their reported cost_usd; Codex / Cursor have no cost.
- Log-cost variance: participant 74%, task 15%, participant x task 6%, run-to-run 4% -> ACROSS models agentic cost is
  model-dominated (SWE-Router's "q-independent" assumption is roughly right in that sense).
- WITHIN a model: per-task cost p90/p10 2.7-9.0x, ICC across runs 0.69-0.92 -> per-task agentic cost is a stable, in-principle
  predictable property of the task.
- Headroom of per-task cost knowledge, one-shot routing, cross-fitted (success + cost from runs 0-2, scored on runs 3-4):
  all 13 standalone models (per-run price range ~190x): 8.5% [-5.0, 17.8] (n.s.); the 7 OPEN-WEIGHT models (range ~10x):
  **25.7% [6.3, 40.6]** -- as large as the reasoning pools.
**General statement: per-query cost knowledge pays when within-model cost variation is large RELATIVE TO the price gaps
between the routed models.** Holds across one-shot reasoning (20-47%), non-reasoning chat (RouterBench 10.5%), and agentic
SWE (open-weight 25.7% vs mixed 8.5%). Next (free): is agentic per-task cost PREDICTABLE from the task statement (4B prefill of
the 111 statements; leave-one-task-out) -- 111 tasks is small, treat as indicative.
- **RouterBench head-to-head (MixLLM's and 2509.09782's own benchmark)**: log-output R2 (mean over 11 models) probe 0.60,
  MixLLM-style 0.58, prompt-GBM 0.49; routing gain probe 8.5% [5.6,11.0], MixLLM-style 8.1% [5.3,10.3], GBM 7.0% [4.4,8.9]
  of a 10.5% headroom. On their benchmark we are at least as good, but low headroom makes every predictor converge.
  (analysis/cost_headroom/routerbench_h2h; baseline_cost_heads.py now reads a pool's prices.json.)
- **Agentic predictability (SWE-rebench, 111 tasks; `analysis/cost_headroom/agentic_predictability.py`)**: out-of-fold log-cost R2
  from the issue text (4B Instruct prefill, 10-fold ridge per model) is ~0 (-0.13 to +0.08, mean +0.01); routing with the
  predicted cost among the 7 open models 2.6% [-15.8, 15.1] vs the 25.7% headroom. NOT predictable here -- but heavily
  data-starved (~100 training tasks per fold; on CodeContests a quarter of the training data, ~87 problems, gave R2 .24).
  Agentic headroom is real; whether it is capturable needs >= several hundred tasks with repeated runs.

### 4.A.15 Non-circular difficulty (reply to "is 4.A.10 circular?") -- `analysis/cost_headroom/difficulty_noncircular.py`
4.A.10's "empirical difficulty" used solve rates from the SAME draws whose length it explained (partly circular; mitigated
only by fail/solve length ratio ~1). Cross-fitted versions, mean over routes, 5-fold CV R2:
| pool | link: other routes' solve rate -> length | link: even-draw solve rate -> odd-draw length | legibility (prefill -> that difficulty) | external label link / legibility |
| LCB | 0.56 | 0.55 | 0.46 | 0.61 / 0.72 |
| Omni | 0.44 | 0.42 | 0.47 | 0.56 / 0.73 |
| CodeContests | 0.43 | 0.36 | 0.23 / 0.21 | 0.43 / 0.51 |
| TACO | 0.08 | 0.12 | 0.32 | 0.22 / 0.35 |
| BCB | 0.10 | 0.10 | 0.09 | n/a |
Verdict survives: CodeContests = difficulty drives length (~0.4, like Omni) but experienced difficulty is half as legible
(0.21-0.23 vs 0.46-0.47); TACO / BCB = length not difficulty-driven (~0.1). Residual caveat: cross-fitting removes the
same-draw mechanical link but not a common cause of failure and length (operational difficulty = how often models fail);
external labels agree in direction. The 4.A.10 "probe misses .12-.17 of difficulty on CC" used the circular measure --
cite this table instead.
- **Agentic predictability at scale (nebius/SWE-agent-trajectories; SWE-agent + Llama-3.1-70B, 3387 tasks with >= 4 runs,
  ~20 runs each; cost proxy = cumulative context + 4x output chars; `analysis/cost_headroom/nebius_predictability.py`)**:
  single-run cost is mostly noise (ICC 0.29) but the per-task MEAN is reliable (split-half ceiling 0.91). Issue-text prefill
  -> per-task mean log cost: 5-fold CV R2 **+0.29**; learning curve 100 / 300 / 1000 / 2710 tasks -> +0.18 / +0.21 / +0.27 / +0.32
  (still rising). So agentic per-task cost IS moderately predictable from the issue and the SWE-rebench null (111 tasks)
  was largely data; but ~0.3 is below the ~0.5 capture threshold seen on the one-shot pools. One dominant model -> no
  routing headroom computable from this set.
- **Partial agent trajectory READ by the 4B** (issue + first 10 steps + "how many more steps / how costly will the rest be?";
  `analysis/cost_headroom/agentic_partial_probe.py`, 7215 SWE-rebench runs): own-run final-cost R2 **+0.28** (range .06-.74;
  hand-crafted trace features -0.01); CROSS-model -- the cheap scout's (MiMo-V2.5-Pro) first 10 steps predict the other
  models' per-task cost at R2 **+0.16** (range .01-.28; statement prefill +0.01, hand-crafted -0.54), with only 111 tasks.
  First agentic predictor with real cross-model cost signal; below the ~0.5 routing threshold, so no routing claim yet.
  Supports "explore cheaply, then price the candidates" (SWE-Router with cost in the decision -- a gap they assume away).
- **LCB prefix test** (gpt-oss-20b-low 512-token prefixes on 892 LCB problems, $0.06; same arms as CC): R2 on the test set
  rises (tuned probe .51-.72 -> probe+prefix .70-.86) but ROUTING GETS WORSE: gain 34.4% (probe) -> 26.1% [17.9,33.4] with a
  free prefix, 8.9% [-0.9,16.7] charged, 21.1% [12.5,28.7] with continuation credit; worse at every accuracy level. Why: 51%
  of prefixes FINISH inside 512 tokens -- R2 gains sit on those easy problems (e.g. oss120md .49 -> .69), which are routed
  cheaply anyway; on the hard, decision-relevant problems the prefix head is WORSE for the key routes (dsv4f .42 -> .33,
  oss120md .45 -> .35; plain probe -> probe+prefix). **R2 is the wrong yardstick for a routing cost predictor: what matters is
  accuracy on decision-relevant problems and routes.** Paper point; also explains why "better cost predictor" claims need
  not translate into routing gains.

### 4.A.16 WHY better cost prediction does not flip decisions: level vs relative cost (2026-09-28)
**4B READS problem + gpt-oss-20b-low prefix + cost question** (LCB, CC; plain ridge on the prefill): log-output R2 LCB .69-.76
-> .82-.87, CC .34-.47 -> .60-.75, and it improves the HARD problems on every route (CC .09-.22 -> .27-.45; LCB .27-.45 ->
.49-.59) -- unlike the hand-crafted prefix head. Yet routing: LCB 35.6% (probe) -> 32.2% free / 15.0% charged / 27.3%
continuation; CC 2.0% -> 2.9% / -19.9% / -6.5%. No gain even for free.
**Mechanism (analysis in chat, pairwise R2):** the prefix improves the SHARED per-problem length LEVEL (mean over routes:
CC .51 -> .72, LCB .77 -> .87) but not the between-route DIFFERENCES that decide a route (LCB dsv4f-oss20lo .59 -> .61,
dsv4f-oss120md .52 -> .53, dsv4f-oss120hi .32 -> .29; CC differences stay .0-.36). The shared level is essentially
difficulty, which the success head already carries. Routing needs per-query RELATIVE cost across models (model-specific
verbosity on this problem), not better per-query length. Explains the pools (LCB: prompt predicts differences .3-.6 ->
savings; CC: differences ~unpredictable by any reader -> none) and every "better R2, no savings" result (Thinking probe,
prefixes, prefix-reading). Reframes the capture curve: synthetic noise hit level and differences equally; real predictors
mostly get the level. Fine-tuned 137M reader: LCB problem-only R2 .52-.57 (< frozen 4B probe .69-.76); other 3 runs pending.
- **Fine-tuned 137M reader (full fine-tune, jina-code; `analysis/cost_headroom/finetune_cost_reader.py`)** is WORSE than the frozen
  4B in all four cells -- test log-output R2 LCB problem .52-.57 (4B .69-.76), LCB +prefix .71-.78 (4B .82-.87), CC problem
  ~0 (4B .34-.47), CC +prefix .43-.62 (4B .60-.75); routing (prefix free) LCB 26.3% / 21.8%, CC -2.3% / -3.4%. 350-450
  training problems cannot train a representation as good as the 4B's existing one.

### 4.A.17 Level vs relative cost -- the user's caveat holds; corrects part of 4.A.16 (2026-09-28; `level_vs_relative.py`)
Headroom (vs paper rule) when the router gets only one component of the TRUE log cost (level = mean over routes):
| pool | full | level only (true level + train-avg route ratios) | differences only (true differences + avg level) |
| LCB | 46.0% | 35.6% | 36.5% |   | Omni | 44.2% | 21.0% | 41.1% |   | CC | 21.8% | 12.7% | 19.3% |
| BCB | 20.2% | 9.8% | 15.2% |   | TACO | 36.1% | -1.9% | 30.0% |
=> The SHARED level ("long for everyone") is worth 10-36% by itself on 4/5 pools ("cheapest model that can plausibly
succeed"); 4.A.16's "level rarely changes the decision" was WRONG as stated. Real predictors: LCB probe level R2 .77 ->
33.3% (of 35.6), Omni .79 -> 18.6% (of 21.0) -- most of our savings come through the level; CC probe .51 -> 4.0%, CC
4B-reads-prefix .72 -> 2.0% (of 12.7). Synthetic calibrated LEVEL predictor on CC: R2 .50 / .72 / .90 -> 1.5 / 6.2 / 10.9%.
So (1) the level's value needs very high accuracy (steep convex curve), and (2) the real predictor at R2 .72 captures a
THIRD of what random noise at equal R2 does -> its errors concentrate where decisions happen (hypothesis: the long,
expensive tail). Differences carry comparable or more headroom (all of TACO's); real predictors capture them less
(LCB 22.8 of 36.5, Omni 14.2 of 41.1, CC ~0).
- **Joint 5-pool fine-tuned 137M reader** (~2150 training problems; `finetune_cost_reader_joint.py`): test log-output R2 LCB
  .51-.56 (single-pool FT .52-.57; frozen 4B .69-.76), CC .06-.14 (single ~0; frozen .34-.47), Omni .52-.57 (frozen ~.75),
  BCB .14-.48 (frozen .15-.71), TACO .04-.14 (frozen .31-.52). Sharing across pools helps CC only marginally; a trained
  small reader stays far below the frozen 4B everywhere. Closed.
- **Entropy scalars** (prefill prompt NLL / next-token entropy / max logprob + squares, stacked on the probe;
  `entropy_scalars.py`): routing LCB 35.6 -> 36.3%, Omni 27.4 -> 26.5, CC 2.0 -> 1.2, BCB -3.3 -> -2.8; between-route
  differences R2 unchanged (LCB .44 -> .44, CC .08 -> .02). NULL.
- **Per-model self-estimated budgets** (TALE-style; each route's model asked, at low effort, how many tokens it would need at
  its own effort; `math_pool/self_budget.py`; 100 LCB problems, ~$0.1): self-estimates rank problems (Spearman .52-.77) but
  worse than the probe (.88-.91); between-route DIFFERENCES Spearman **+0.12**; badly calibrated (gpt-oss-120b-high says
  ~350 tokens, writes ~2540). Models do not know their own relative verbosity. NULL; no full run.
**Summary of "predict the between-model differences":** prefix (hand-crafted / 4B-read), Thinking / Base / own-model
prefills, fine-tuned readers (single / joint), entropy scalars, self-estimates -- none moves the between-route differences
materially. The model-specific part of per-query cost looks unpredictable from anything available before generation.

### 4.A.18 Two constructive methods from the level-vs-relative analysis (2026-09-28; exploratory, offline)
**#1 Per-query token CAPS** (`analysis/cost_headroom/percall_caps.py`): router picks (model, max_tokens cap); caps from the probe's
predicted log-normal length distribution (quantiles), simulated exactly from stored draws (success iff solved AND length <= cap;
cost = input + min(length, cap)). Gain vs the paper rule at matched accuracy (test; this simulator's no-cap number differs from
decompose.py -- 23.0 vs 35.6 on LCB -- reconcile before quoting absolutes; arms are comparable within it):
| pool | runaway draws (> 4x problem median) | no cap | one global cap per model | PER-QUERY cap |
| TACO | 1.8% | 0.1% | 4.9% [-11.3, 22.6] | **14.3% [7.0, 26.9]** |
| LCB | 0.8% | 23.0% | 17.5% | 22.4% |   | Omni | 0.4% | 28.7% | 28.3% | 28.7% |   | CC | 0.7% | -4.8% | -1.2% | -4.8% |
=> caps help where runaways exist (TACO: rescues a pool where cost PREDICTION captured nothing), are neutral elsewhere; the
per-query cap beats a global per-model cap -> the predicted level sets tight caps on problems that should be short.
**#2 Onboarding a NEW model from k examples** (`onboard_new_model.py`): new model's cost = shared level (other routes' probe
predictions) + one offset from k labelled problems; its success head stays fully trained. Mean over held-out routes, gain vs
paper rule: LCB k=5 21.2% (full heads 23.0%) vs median-from-k 1.3% (oss120hi -43.6% at k=5); Omni k=5 26.5% (full 28.7%);
CC no gain to transfer (-4.5 vs -4.8). => ~92% of a fully trained cost head from 5 examples; limit = models with large
model-specific variation (dsv4f 15.0 vs 23.0 on LCB). Combine with IRT-style success onboarding for the full "add a model" story.
**CORRECTION to 4.A.17 / 4.A.18 (bug):** `percall_caps.py`, `onboard_new_model.py`, `level_error_location.py` read content_preds.jsonl
in FILE order; on LCB and CC that order differs from the tensor order (1/892 and 0/700 positions match; TACO, Omni match), so
success predictions were misaligned there. Fixed (keyed by problem_id); re-run:
- Caps simulator now agrees with decompose.py (no-cap LCB 36.0 vs 35.6, CC 2.4 vs 2.0). Caps: LCB 36.0 / global 35.2 / per-query
  36.2; Omni 28.7 / 28.5 / 28.7; CC 2.4 / -4.1 / 2.4; TACO (unaffected by the bug) per-query **8.0% [-4.2, 19.3]** after aligning the
  accuracy band with decompose.py (the earlier 14.3% [7.0, 26.9] used a 2-arm band). => caps: neutral where runaways are rare, a
  non-significant trend on TACO. NOT a claim yet; the test is agentic step caps.
- Onboarding (LCB) now STRONGER: k=5 34.5% (full heads 36.0%, 96%) vs median-from-k 12.3%; Omni unchanged 26.5 vs 28.7; CC 2.1 vs 2.0.
- CC level redundancy: success predictions DO explain 0.47 of the true level (the "~0" was the bug); the real predictor captures
  0.52 of the level's part beyond the success head vs 0.61 for synthetic noise at equal R2 -> partial redundancy explains part
  of why it cashes in less.
- **R2-Router read (2602.02823)**: jointly picks (LLM, per-query continuous output-length budget) and predicts quality vs budget;
  enforces the budget with length-constrained PROMPT INSTRUCTIONS; needs its own R2-Bench (behaviour across budgets). Our caps
  differed (hard max_tokens stop, simulated from ordinary uncapped runs, set by the cost predictor) -- moot, see below.
- **Agentic STEP caps backfire** (`analysis/cost_headroom/agentic_step_caps.py`; nebius SWE-agent + Llama-3.1-70B, 3387 tasks, 74k
  runs, exact simulation from per-step cumulative cost): per-task caps from the issue probe (R2 of mean log steps .32) cost
  **-22.3% [-25.5, -19.7]** MORE than one global step cap at matched resolve rate; caps from the TRUE per-task mean -81.5%.
  Mechanism: in agent runs long = mostly failing (loops); a global cap kills long runs everywhere, while per-task caps grant
  hard tasks more budget that their (still mostly failing) long runs waste -- the better the length estimate, the worse.
  The right cap is "how long a SUCCESSFUL run needs", which a global max_steps largely already captures.
**Caps direction CLOSED** (one-shot: neutral; agentic: per-task worse than global). Coverage-guarantee and live checks dropped.
- **#2.1 Full onboarding (cost + success) from k problems** (`analysis/cost_headroom/onboard_full.py`; success = logistic in the
  shared difficulty = mean logit of the other routes' success predictions, 2 parameters; cost = level + offset): mean over
  held-out routes, gain vs paper rule: LCB k=5 29.5% / k=50 32.1% (full heads 36.0%) vs naive median+base-rate from k -4.7% /
  7.3%; Omni k=5 24.0% (full 28.7%) vs naive ~15%. Big win for expensive routes (LCB oss120hi 35.9% vs naive -50.9% at k=5);
  WORSE than naive for the cheapest route (oss20lo LCB 25.3 vs 35.0, Omni 17.9 vs 30.2) -- its success does not track the
  shared difficulty well enough for a 2-parameter fit.
- **#2.2 Predicted onboarding loss:** loss concentrates in routes with a large MODEL-SPECIFIC share of cost variance
  (share .14-.22: losses 4-14 pt at k=50; share <= .07: 0-4 pt), exception LCB oss20md (share .04, loss 7.9 from the success
  side). => how much a model deviates from the shared difficulty predicts how cheaply it can be added.

### 4.A.19 MMLU-Pro full pool: PRE-REGISTERED GAIN CONFIRMED (2026-09-28 18:40 ET)
1000 problems (the screen's stratified sample), 5 routes, draws 4/3/3/2/2 (7 dsv4f draws of 14000 still hung, excluded), $~6.
Accuracy 60.5 / 68.0 / 83.1 / 75.0 / 77.5%; mean output 373 / 1289 / 4346 / 888 / 3527 tokens. Instruct probe (the screen's):
test log-output R2 .51 / .58 / .20 / .66 / .50. **Gain vs paper rule at matched accuracy 30.7% [16.3, 43.2]** (predicted >= 10%,
CI > 0: YES); **headroom 65.1% [55.2, 71.2]** (predicted >= 15%: YES; largest of any pool). Deployable protocol (calibration-chosen,
test once): cost ratio 0.55 / 0.39 / 0.52 / 0.58 at test accuracy 57-72% (accuracy diffs all n.s.), 0.91 [0.82, 0.99] at ~80%.
=> 42-61% cheaper at matched accuracy. Pre-registered record: GAIN Omni (confirmed), MMLU-Pro (confirmed); GAIN pending APPS, SuperGPQA,
K&K; NO GAIN pending AIME, BBEH.
- **#2.3 A genuinely NEW model family onboarded into the MMLU-Pro router** (`onboard_newfamily.py`; GLM-4.7-flash, Nemotron-3-super,
  MiniMax-M2.5 run once on MMLU-Pro test (300) + 50 train problems, ~$2.5 incl. a $0.3 token check; Nemotron hit 429s at 256
  concurrent -> retried at 16, 89% test coverage). New models: acc 73.2 / 82.8 / 76.9% at 0.22 / 0.18 / 0.34c per call -- all
  DOMINATED by the existing pool (dsv4f 83% for less), so adding them cannot lower cost. Gain vs the 5-route paper rule (5-route
  router with full heads: 30.9%): +3 onboarded from k = 5 / 10 / 20 / 50 -> 14.9 / 30.0 / 30.8 / 30.7%; +3 naive (median cost +
  base rate from k) -> -43.9 / -12.3 / -5.5 / +4.3%. => onboarding PROTECTS the router (prices dominated newcomers correctly from
  ~10 examples); naive onboarding wrecks it. A positive "new model lowers cost" test needs a newcomer that is actually cheaper or
  better than the pool.

### 4.A.20 Baselines for the win claim (2026-09-29; `simple_baselines.py`, decompose.py now saves paired bootstrap draws + single-best)
Gain vs the paper rule at matched accuracy (test), and OURS minus baseline as a PAIRED bootstrap difference:
| pool | ours | mean-per-model constant | cost FROM THE SUCCESS HEAD | ours - from-success [95% CI] | single best model |
| LCB | 35.6% | 1.3% | 34.1% | +1.5 [-1.3, +4.3] | -3.7% |
| Omni (pre-reg Thinking probe) | 21.9% | -1.4% | 21.0% | +0.8 [-7.2, +9.0] | -20.1% |
| MMLU-Pro (pre-reg Instruct) | 30.7% | 1.2% | 9.2% | **+21.5 [+8.5, +30.3]** | +5.4% |
| CC | 2.0% | -2.5% | 5.3% | -3.3 [-8.6, +2.4] | +3.2% |
(cost-from-success = per-model ridge of log output on the success head's logits (+squares); its log-output R2 LCB .67-.72,
Omni .68-.72, CC .37-.46 -- close to the dedicated probe -- but MMLU-Pro .14-.36 vs probe .20-.66.)
**Consequence for the claim:** the win vs the paper rule stands on all three pools, and routing beats the single best model.
But on LCB and Omni a DEDICATED cost probe adds nothing over cost inferred from the success head: the capturable value is
"harder problems cost more", which the success head already knows. Existing prefill routers could get most of it for free
from their own success predictions. The separate cost probe matters only where length is not just difficulty (MMLU-Pro,
+21.5 pt). Constant-vs-constant (mean vs median) is worthless. Pending: MixLLM-style / prompt-GBM on Omni + MMLU-Pro (GPU).
**Literature predictors on the pre-registered pools** (baseline_cost_heads.py, GPU embeddings; baselines_lit.json). Gain vs paper rule;
ours minus baseline, paired bootstrap:
| pool | ours | MixLLM-style | prompt-GBM | from-success |
| Omni | 21.9% | 12.5% (+9.4 [-4.4, +24.2]) | 6.6% (+15.2 [-1.4, +34.1]) | 21.0% (+0.8 n.s.) |
| MMLU-Pro | 30.7% | 4.0% (**+26.7 [+12.4, +40.2]**) | 9.8% (**+20.9 [+3.8, +33.3]**) | 9.2% (**+21.5 [+8.5, +30.3]**) |
(log-output R2: MixLLM-style Omni .04-.32 / MMLU-Pro ~.02; GBM .22-.29 / -.19..+.16; ours .70-.79 / .20-.66.)
**Where we win, definitively:** vs the paper rule on LCB, Omni, MMLU-Pro; vs the literature predictors significantly on LCB and
MMLU-Pro, directionally on Omni (+9 / +15 pt, CIs touch 0); vs cost-from-success only on MMLU-Pro (LCB, Omni: tie).

### 4.A.21 Remove a good model, then re-add it from k examples (2026-09-29; `onboard_full.py`, "without" arm)
Reference = the FULL 5-route pool's paper rule (fixed). "Without" = route made unavailable; onboard = cost (level + offset)
AND success (logistic in the shared difficulty) from k examples; naive = median cost + base rate from k; 20 draws of k.
| pool | full heads | WITHOUT dsv4f | re-added onboard k=5 / 10 / 50 | re-added naive k=5 / 10 / 50 |
| LCB | 36.0% | 4.1% | 23.2 / 25.0 / 28.8% | -16.3 / -19.3 / -22.0% |
| Omni | 28.7% | -25.0% | 18.9 / 19.1 / 23.5% | n/a / -14.9 / -11.6% |
| MMLU-Pro | 30.9% | -22.1% | 16.6 / 29.0 / 29.9% | -16.0 / -16.1 / -13.8% |
Other routes removed (LCB / Omni / MMLU-Pro): oss20lo 24.5 / 5.2 / 21.1; oss20md 30.0 / 35.2 / 35.7 (removing it HELPS on
Omni / MMLU-Pro -- its heads mislead the router); oss120md, oss120hi ~no change (redundant with each other).
=> Adding a genuinely useful model back from 5-10 examples recovers most of its value (LCB 4 -> 23-25 of 36; Omni -25 -> 19 of
29; MMLU-Pro -22 -> 29 of 31 at k=10); adding it naively is WORSE than leaving it out. dsv4f is the hardest route to onboard
(largest model-specific cost share, .21-.43), so this is a pessimistic case. See PAPER_OUTLINE.md for how this enters the paper.

### 4.A.22 Why a shared difficulty latent does not beat the fixed cascade (2026-09-29; `analysis/dist_bellman/shortcut_ceiling.py`)
LCB 5-rung pool, free perfect verifier, clean_comparison protocol (operating point chosen on calibration, applied once to
test; cascade = best of 728 enumerated plans). All non-cascade arms run ONE index policy (call the route with the lowest
c/q while V*q > c; stop at first success); only the per-problem information changes. Market prices; cost ratio vs cascade:
| information | 70% | 80% | 85% | 90% |
| ours (prefill q, cost head) | 1.23 | 1.67 | 1.52 | 1.88 |
| synthetic latent R2 .50 / .70 / .85 | 1.98 / 0.85 / 0.65 | 2.10 / 1.17 / 0.72 | 1.92 / 1.26 / 0.77 | 1.70 / 1.30 / 0.81 |
| ORACLE shared latent (true difficulty + true cost level) | 0.42 | 0.59 | 0.55 | 0.85 [0.66,1.06] |
| ORACLE per-route pass rates and costs | 0.40 | 0.28 | 0.24 | 0.24 |
| clairvoyant (knows which draw succeeds) | 0.13 | 0.08 | 0.08 | 0.07 |
Our prefill vs the true shared latent on test: difficulty R2 **0.42**, cost level R2 0.79.
Findings. (1) LATENT QUALITY is the bottleneck: a perfect shared latent is 40-45% cheaper than the cascade at 70-85%; the
break-even is around difficulty R2 ~.7-.8; ours reads .42. Convex, like the one-shot capture curve. (The index policy does
not update after failures, which handicaps weak information; with the Bellman policy ours ~= cascade, 0.99x / +2.8 pt at 85%,
4.A.2 DEFINITIVE -- same conclusion.) (2) NOT the price ladder: scaling oss120's price x0.25 or x4 moves the oracle-latent ratio
little (80%: 0.76 / 0.59 / 0.66). (3) "Shortcut to the right tier" is not where the money is: even oracle arms skip the
cheapest route on only 0-10% (latent) / 5-16% (per-route) of problems -- oss20lo costs 1/18-1/100 of the others and solves
53%, so trying it first is nearly always right. The savings come from stopping early on hopeless problems and not re-buying
draws that will fail (cascade: 52-84% of spend is on failed draws). (4) At >= 90% even the perfect SHARED latent only ties
the cascade; per-route information (0.24) is needed -- the model-specific part, which nothing predicts (4.A.17).
(5) The cascade's first cheap draw is itself a cheap noisy measurement of the latent -- the reason it is hard to beat.
Caveat: oracle quantities use the problem's own draws (leaky ceilings); synthetic arms add noise to that leaky latent.
Legacy prices give the same picture (analysis/dist_bellman/shortcut_lcb_legacy.json).
Follow-up (same day): with a perfect SHARED latent the route order by c/q equals the global order on 85% of test problems
(first call oss20lo on 87%, dsv4f 13%); with perfect PER-ROUTE info on only 47%. With a free verifier, the optimal policy
for known q orders routes by c/q, and a one-dimensional latent that shifts every route's logit barely changes that order.
So the optimal policy is ~ a fixed-order cascade plus a per-problem STOPPING rule, and the latent's 40% comes from the
stopping rule (how many draws, how far up the ladder, when to quit), not from reordering. Only 3% of test problems are
unsolved by every draw, so "hopeless" means "not worth the price at this V", not "unsolvable". Crossings exist (dsv4f
solves >= 1 draw on 43% of the problems where oss120hi solved none), and exploiting them needs model-specific information.
Abstention: all ONE-SHOT results (4.A.4-4.A.21, the outline's main claims) have NO abstain option -- every problem gets exactly
one call. The verifier-regime policies (Bellman, index, 4.A.2/4.A.22) may stop at any point, including before the first call.

### 4.A.23 One-shot routing WITH abstention (2026-09-29; `analysis/cost_headroom/abstain_oneshot.py`, abstain_oneshot.log)
No verifier, one submission; per problem answer with route m or abstain (cost 0). Score = accuracy - lam x error rate over
all problems (lam = 0: abstaining only saves money; lam > 0: a wrong answer is worse than none). Rule: answer iff
max_m V((1+lam)p_m - lam) - c_m > 0. decompose.py protocol (test frontiers over V, hulls through the origin so random
abstention is free for every arm; pairwise matched-score band where both arms have real points; paired bootstrap). Market
prices, the outline's cost heads (LCB probe, Omni Thinking, MMLU-Pro Instruct). Cost saved at matched score [95% CI]:
| | lam | LCB | Omni | MMLU-Pro |
| ours vs paper, no abstention (sanity: matches 35.6 / 21.9 / 30.7) | 0 | 37.0 | 22.0 | 30.2 |
| abstention vs none (ours+A vs ours) | 0 | 0.1 [-2.7, 2.9] | -7.0 [-15.4, 2.1] | 0.0 |
| ours+A vs paper+A | 0 | 23.8 [19.0, 28.1] | 20.0 [10.1, 27.4] | 13.5 [6.7, 21.2] |
| ours+A vs paper+A | 1 | 30.2 [21.3, 36.7] | 23.7 [3.8, 40.2] | 13.3 [-2.1, 29.6] |
| ours+A vs paper+A | 3 | 42.4 [31.5, 52.1] | 43.3 [27.6, 61.0] | 26.0 [-5.3, 49.5] |
| perfect success + cost, abstaining, vs ours+A (remaining headroom) | 0 / 1 / 3 | 37 / 59 / 66 | 45 / 63 / 73 | 59 / 81 / 85 |
Max achievable score, no abstention -> with (ours): lam=1 LCB .788 -> .792, Omni .484 -> .576, MMLU-Pro .642 -> .652;
lam=3 LCB .576 -> .655, Omni 0 -> .489, MMLU-Pro .284 -> .417.
Findings. (1) lam = 0: abstention buys nothing -- the cheapest route costs almost nothing and solves ~50%, so skipping a
problem saves ~0. (2) lam > 0: abstention is necessary (the always-answer routers cannot reach its scores), and per-query
cost pays MORE with abstention and a larger error penalty (LCB 24 -> 30 -> 42%, Omni 20 -> 24 -> 43%; MMLU-Pro n.s. at lam>0).
(3) The remaining headroom grows with lam (59-85% at lam >= 1): once wrong answers cost, knowing WHO will get it right
(the success head) is the bottleneck, not cost. Caveats: test-selected hulls (like decompose.py), not the deployable
protocol; lam is a chosen utility; Omni test n=150 (wide CIs); some lam=3 cells undefined (the always-answer arm never scores > 0).

### 4.A.24 Onboarding without deepseek: remove gpt-oss-120b, add it back (2026-09-29; `onboard_full.py`, DROP=dsv4f, HOLDGROUPS)
Pools with dsv4f removed entirely (4 routes: oss20lo/md, oss120md/hi); cost heads as 4.A.18 (LCB probe, Omni Instruct probe,
MMLU-Pro Instruct probe). The strongest remaining MODEL, gpt-oss-120b (BOTH efforts), is held out and re-added from the same k
labelled problems. Gain vs the 4-route paper rule at matched accuracy, and the frontier's MAX reachable test accuracy:
| pool | full heads | WITHOUT gpt-oss-120b | re-added onboard k=10 / 50 | re-added naive k=10 / 50 |
| LCB | 21.2% @ max 88.2 | max 67.0 (-21 pt) | 8.2% @ 86.7 / 12.2% @ 88.6 | -45.1% @ 86.0 / -44.4% @ 87.9 |
| Omni | 18.9% @ 69.4 | max 62.7 (-7 pt) | 10.5% @ 66.0 / 11.6% @ 66.1 | -7.8% @ 64.5 / -6.3% @ 64.9 |
| MMLU-Pro | 13.3% @ 73.4 | max 62.4 (-11 pt) | 15.7% @ 72.0 / 17.4% @ 72.6 | -20.2% @ 71.5 / -20.7% @ 71.8 |
=> Without gpt-oss-120b the router loses 7-21 pt of reachable accuracy. Re-adding it from 10 labelled problems restores most of
that accuracy while still saving 8-16% vs the paper rule; re-adding it naively restores similar accuracy but costs 8-45% MORE
than the paper rule. Onboarding recovers 40-100%+ of the full heads' saving (LCB 8.2/21.2 at k=10, 12.2 at k=50; MMLU-Pro
above full). Without dsv4f the pools are shallower (full-head gains 13-21% vs 29-36% with it).
**Protocol caveat (applies to 4.A.21 too):** a "without" arm's gain is measured only over the accuracies it can still reach,
so removing a strong model can LOOK better than the full pool (e.g. LCB without gpt-oss-120b 24.1% -- but only up to 67%).
Always read "without" together with max reachable accuracy.
**CORRECTION to 4.A.21:** max test accuracy full / without dsv4f / naive k=10 / onboard k=10: LCB 89.3 / 88.2 / 88.6 / 88.2;
Omni 74.3 / 69.4 / 73.9 / 73.6; MMLU-Pro 82.1 / 73.4 / 80.9 / 80.7. On LCB the band is ~unchanged, so "naive re-adding is worse
than leaving dsv4f out" (-19.3% vs 4.1%) holds there; on Omni and MMLU-Pro the "without" gains cover a truncated band, so that
sentence does NOT hold as stated. Correct general statement: naive re-adding restores accuracy at a 14-19% cost premium over
the paper rule; onboarding restores it while saving 16-29%.

### 4.A.25 No-verifier single-submission CASCADE with a learned judge vs the prefill router (2026-09-29; `judge_cascade.py`)
LCB, one submission, no test feedback; each tier call = one stored draw (3 orderings); market prices; cascades over <= 3 tiers
cheapest-first with per-tier judge thresholds (FrugalGPT-style); hybrid = router picks the entry tier (same 60-point V grid),
the judge escalates from there. 4B judge = frozen Qwen3-4B prefill probe on problem + code + "Is this solution correct?"
(40,070 attempts prefilled in the Sep 24 no-verifier line), charged at gpt-oss-20b input price: 0.00134c uncached, 0.00055c
cached (oss20lo call 0.0081c; ~16% / ~7%). Test AUC of the 4B judge: oss20lo .909 (within-problem .797), oss20md .914 (.820),
dsv4f .834 (.634), oss120md .844 (.567), oss120hi .857 (.661) vs the problem-only prefill prior .81-.82.
Cost saved vs the one-shot router at matched accuracy (lam = 0, decompose.py protocol, test hulls -- generous to the cascades,
~1000 plans vs 60 V values):
| paper rule | cascade[4B] | cascade[4B cached] | hybrid[4B] | cascade[PERFECT judge] |
| -64.4% [-89, -41] | -48.9% [-76, -29] | -47.1% [-74, -27] | +0.7% [0.0, 5.5] | -18.6% [-39, -1] |
=> The prefill router beats every single-submission cascade here, INCLUDING one with a perfect judge: a cascade pays the cheap
attempt on every problem, the prefill shortcut skips it (the "jump to the right tier" value, realised). The 4B judge adds
nothing on top of the router (hybrid +0.7%). (lam > 0 dropped as out of scope: a perfect judge would win 46-70% there, the 4B
judge loses badly -- its weakness is strong-model code.) Pending: the FrugalGPT-style per-tier fine-tuned 137M scorer
(finetune_judge_reader.py; first job OOM'd, relaunched 2026-09-29 07:00 UTC).

### 4.A.26 Targeted literature check (2026-09-29, overnight) + ZeroRouter-style baseline
Question: has anyone (a) priced each query from a prefill / shared difficulty latent, (b) onboarded models against it,
(c) measured when per-query cost pays? Read in full or in the relevant sections:
- **ZeroRouter (2601.06220, Yan et al., Jan 2026) -- CLOSEST PRIOR ART.** "Universal latent space" of query difficulty
  (multidimensional IRT, D=20) from a fine-tuned DistilBERT + 11 linguistic features. PER-QUERY output length = each model's
  mean output in the query's complexity bin (s_q = alpha_q^T b_q discretised into K bins, K unstated; Eq. 10); cost =
  fixed $/token x predicted tokens; latency = TTFT + tokens x TPOT. New models onboarded "zero-shot" from ~200 anchor queries
  (IRT ability fit + the same bin lookup for cost). Pool mostly non-reasoning (Qwen3-235B/32B, R1-Distill-70B, Mixtral,
  Phi-4, Llama-8B, ...); Open-LLM-leaderboard tasks (IFEval, BBH, MATH, GPQA, MuSR, MMLU-Pro). NO ablation of the per-query
  cost component vs a per-model average; NO test of fewer than 200 anchors; no analysis of output-length variation.
  => The FRAMING "one shared difficulty latent routes, prices and onboards" is theirs. Our paper must cite it as such.
- **Route-To-Reason (2505.19435, Pan et al.; WWW 2026)**: routes over (model x reasoning strategy) incl. reasoning models
  (R1, QwQ-32B); MLP on all-mpnet-base-v2 text embeddings predicts output tokens (~60% within 600 tokens for reasoning
  models); predicted length enters the score. No ablation vs average cost.
- **CARROT (2502.03261)**: per-query cost via kNN / RoBERTa on text embeddings; SPROUT dataset (15 models incl. o3-mini;
  GPQA, MuSR, MMLU-Pro, MATH, ...). Its comparison with the RouterBench router (constant per-model average cost) shows only
  "marginal improvements" -- consistent with our chat-pool headroom (RouterBench 10.5%); no cost-predictor accuracy reported.
- **GraphRouter (2410.03834, ICLR 2025)**: GNN edge prediction of both effect and cost per (query, LLM); new LLMs without
  retraining.
- **IRT-Router (2506.01048)**: cost is FIXED per model (output price mapped to [0,1]); BERT embeddings; cold start for new
  queries by kNN warm-up; new models weak (tested on one).
- **LLMs Encode Their Failures (2602.09924)**: prefill probes predict success (gpt-oss-20b low/med/high, Qwen-Math,
  R1-Distill-7B; MATH/GSM8K/AIME); routing cost = average output cost per model from train -- no per-query cost.
- **2603.20895 (prefill-activation router)**: median length per model (our reference rule).
- **RouterXBench (2602.11877)**: router evaluation assumes uniform cost per call.
- **AutoProbe (2510.02934)**: probes the GENERATOR's own hidden states for code correctness; not routing/cascades; degrades
  across models. (So our external 4B judge of other models' code is not AutoProbe's setting.)
Verdict: per-query cost prediction in routers EXISTS (MixLLM, CARROT, GraphRouter, Route-To-Reason, ZeroRouter), including a
difficulty-latent version with onboarding (ZeroRouter). What we did not find anywhere: (1) a measurement of what per-query cost
is worth (headroom vs a per-model constant at matched accuracy) and WHEN it pays (reasoning vs chat vs agents; the rule,
pre-registered); (2) cost read from an LLM prefill; (3) evidence that difficulty-only pricing is enough on some pools and not
others; (4) onboarding from 5-10 examples; (5) the router-vs-cascade shortcut result. The positioning must change from "nobody
prices per query from the latent" to "several routers price per query, none measured whether or when it helps; we do, and we
show where difficulty-only pricing (ZeroRouter's) fails".
**ZeroRouter-style pricing as a baseline** (`zerorouter_cost.py`: complexity = our shared difficulty = mean success logit,
K quantile bins on train, per-model mean train output per bin; uses OUR prefill latent, so it is an upper bound on their
DistilBERT version). Gain vs the paper rule; ours minus it, paired bootstrap:
| pool | ours | ZeroRouter-style K=5 | K=10 | cost-from-success (continuous) |
| LCB | 35.6% | 32.2% (+3.7 [+0.1, +7.4]) | 34.0% (+1.8 [-1.6, +5.1]) | 34.1% (+1.6 [-1.3, +4.3]) |
| Omni | 21.9% | 21.7% (-0.1 [-12.1, +10.7]) | 17.9% (+3.5 [-8.2, +14.9]) | 21.0% (+0.3 [-7.2, +9.0]) |
| MMLU-Pro | 30.7% | 10.9% (+19.6 [+7.8, +32.7]) | 13.3% (+19.1 [+7.6, +31.6]) | 9.2% (+19.4 [+8.5, +30.3]) |
=> Difficulty-bin pricing ties us where length is difficulty (LCB, Omni) and loses ~19 pt where it is not (MMLU-Pro) -- the
same pattern as cost-from-success. This is our sharpest differentiator from ZeroRouter.

### 4.A.27 FrugalGPT-style 137M judge: result (2026-09-29 03:30 ET; `finetune_judge_reader.py`, `judge_cascade.py`)
Per-tier full fine-tune of jina-code 137M (8k context, problem + code -> correct?), epoch chosen on calibration log-loss.
Test AUC 137M vs frozen 4B probe (within-problem in brackets): oss20lo .867 vs .909 (.751 vs .797); oss20md .876 vs .914
(.791 vs .820); dsv4f .795 vs .834 (.683 vs .634); oss120md .774 vs .844 (.471 vs .567); oss120hi .762 vs .857 (.419 vs .661).
=> The frozen 4B probe is the better judge on every tier overall, with zero gradient training (within-problem: better except dsv4f).
Cascade replay (lam = 0; cost saved vs the one-shot router at matched accuracy; 137M judge charged 0):
| cascade[137M] (FrugalGPT-style) | cascade[4B] | cascade[PERFECT judge] | hybrid[137M] | hybrid[4B] |
| -25.1% [-45.4, -5.2] | -48.9% [-75.9, -29.0] | -18.6% [-39.0, -1.1] | +2.1% [+0.4, +8.6] | +0.7% [0.0, +5.5] |
=> The prefill router beats the FrugalGPT-style cascade by 25% at matched accuracy, and every cascade including the
perfect-judge one. As an escalation add-on to the router, a judge buys at most ~2%. The 137M cascade beats the 4B cascade
despite the lower AUC (it is free; the thresholds landed better) -- not a paired comparison, CIs overlap; not a claim.
LCB only. C12 now: Solid on LCB.

### 4.A.28 What drives MMLU-Pro output length beyond difficulty? (2026-09-29; `mmlupro_length_driver.py` + inline checks)
Test R2 of log mean output tokens per route (oss20lo / oss20md / dsv4f / oss120md / oss120hi):
| features | R2 |
| success-head difficulty | .14 / .25 / .25 / .21 / .36 |
| TRUE difficulty (other routes' solve rate) | .02 / .19 / .27 / .12 / .33 (Omni mean .46, LCB .51) |
| subject (14) | .22 / .17 / .16 / .25 / .20 |
| difficulty + subject | .36 / .37 / .30 / .43 / .42 |
| + source (ori_mmlu / stemez / theoremQA / scibench) + n options + option length | .39 / .39 / .31 / .47 / .42 |
| 4B cost probe | .51 / .58 / .20 / .66 / .50 |
- Not a weak success head: even TRUE difficulty explains little of MMLU-Pro length (mean .19 vs .46-.51 on Omni/LCB). Caveat:
  solve rates are high (60-83%) and coarse, so "true difficulty" is itself a blunt measure here.
- Harder = longer both across and within subjects (corr -0.14..-0.54 either way): no Simpson reversal. My first hypothesis
  (computational vs recall subjects) is WRONG: law (solve .45) and engineering are the longest subjects, math is short.
- Subject closes about HALF the routing gap: difficulty only 9.2%, difficulty + subject 22.3%, probe 30.7% (probe minus
  difficulty+subject +9.7 [+0.3, +19.4]).
- The probe's remaining signal is real (corr .64 with the true residual after difficulty + subject) and item-level: questions
  it prices LONG are multi-quantity engineering calculations (pipes, heat transfer, mass transfer: many givens, unit
  conversions, multi-step formulas) and multi-part open-ended "explain X and distinguish it from Y"; questions it prices SHORT
  are fill-in-the-blank / single-concept recall and one-formula plug-ins. Surface proxies catch only part of it (digits in the
  question .27, question length .19).
=> Name for the paper: WORK REQUIRED (how many steps / quantities / parts the answer needs) vs DIFFICULTY (how likely the model
is to get it wrong). On LCB and Omni the two coincide (hard problems need more steps), so difficulty-only pricing suffices; on
MMLU-Pro they come apart (easy-but-laborious engineering calculations, short-but-failed recall), and the prefill reads work
required directly.

### 4.A.29 Per-query BUDGET ("spend at most X on this query") -- hard and soft caps (2026-09-29; `budget_cap.py`)
One call per query, market prices, exact replay from stored draws, same success head for every arm; budget X swept over 24
values from ~an oss20lo call to ~an oss120hi call. Arms differ only in the length model: median (paper rule: fits iff median
train length <= cap), constant (each model's TRAIN length distribution), ours (probe's per-query shift + the EMPIRICAL train
residual distribution), oracle (the problem's own draws). If nothing is predicted to fit, every arm calls the model with the
most room (a fix: without it the median rule made no call at tight budgets and looked absurdly bad).
**HARD cap** (max_tokens enforces X; an overrun is cut off and fails). Accuracy gain of ours over constant, paired 95% CI:
- LCB: +1.9 to +5.7 pt across 0.007-0.27c (all CIs > 0), e.g. +5.7 [+3.7, +7.8] at 0.20c (74.9 vs 69.3%); 0 at the extremes.
- Omni: +1.7 to +7.5 pt across 0.006-0.054c, e.g. +7.5 [+3.9, +11.2] at 0.054c; +1.5-2.7 above (CIs mostly > 0).
- MMLU-Pro: +2.4 to +5.5 pt across 0.01-0.046c, e.g. +5.5 [+2.2, +8.6] at 0.017c; ~0 above.
Budget the baseline needs for the same accuracy, relative to ours (geo-mean over the whole accuracy range; peak in the middle):
| | LCB | Omni | MMLU-Pro |
| constant distribution | 1.29x (peak 2.35x) | 1.33x (peak 2.86x) | 1.10x (peak 1.51x) |
| median rule | 1.44x (peak 2.9x) | 1.07x (peak 2.65x) | 1.07x (peak 1.95x) |
| (average-cost regime, for comparison: paper rule / ours at matched accuracy) | 1.55x | 1.28x | 1.44x |
- A LOG-NORMAL length model around the probe (instead of the empirical residuals) made MMLU-Pro WORSE than constant (-2 to -3.6
  pt at 0.08-0.21c): the tails matter under a hard cap. Use empirical residuals.
- Oracle length gives a further +2-4 pt (LCB), +1-6 (Omni), +4-11 (MMLU-Pro) in the middle band: prediction, not headroom, limits.
**SOFT cap** (not enforced; violation = realised cost > X; each arm's safety margin swept; accuracy at a MATCHED violation rate,
no CIs -- directional): at 10% violations, ours vs paper rule: LCB +3.2 / +5.3 / +4.3 pt at 0.03 / 0.08 / 0.20c with half
the mean overshoot at 0.03c (10% vs 20% of X); Omni +1.8 / +2.8 / +3.5 at 0.04 / 0.10 / 0.25c; MMLU-Pro +4.6 / +3.1 / -0.9 at
0.013 / 0.028 / 0.06c; at loose budgets all arms tie.
=> Verdict on "does a per-query cap make cost prediction matter MORE": LOCALLY yes -- in the middle band, where the strong
model's length straddles the cap, the constant rule needs up to 2.4-2.9x the budget and ours gains +5-7.5 pt accuracy at a
fixed budget. OVERALL no -- averaged across budgets the advantage (1.1-1.3x) is similar to or smaller than the average-cost
regime (1.3-1.55x), because at tight budgets every arm is forced to the cheapest model and at loose budgets every model fits.
Paper framing: "under a per-request budget (an SLA), per-query cost prediction buys +5-7 points of accuracy in the budget band
that matters".

### 4.A.30 How ZeroRouter works, and where we differ (2026-09-29; from the paper's method section)
**Their pipeline.** Stage 1 (defines the latent): fit a 20-D multidimensional 2PL IRT model, P(u solves i) =
sigma(alpha_i^T (theta_u - b_i)), to the correctness matrix of ~200 Open-LLM-Leaderboard models, hierarchical Bayesian priors,
SVI. Every training query gets alpha_i, b_i in R^20; every model an ability theta_u in R^20. Stage 2 (reads it from text):
fine-tune DistilBERT ([CLS], final layer; 40 epochs) + 11 linguistic features -> fusion trunk -> heads predicting b (residual
from the mean) and alpha (clustered expert heads). Inference: one DistilBERT pass -> alpha, b -> every model's P via the IRT
formula; complexity s = alpha^T b -> one of K bins -> per-model mean output length (lookup). New model: everything frozen, fit
theta_u (20 params) by BCE on ~200 D-optimal anchors; fill its length-per-bin table from those anchors.
**Differentiation (ours vs theirs):**
1. Cost from WORK, not difficulty: their length is a function of s = alpha^T b, a scalar of the success parameters -- locked to
   difficulty by construction. Where work != difficulty (MMLU-Pro, 4.A.28) difficulty-only pricing loses ~19 pt. Needs more
   pools (coding) to be a claim.
2. Population requirement: their latent is identified from a response matrix over ~200 models (40 numbers per query from its
   row of outcomes). A deployment pool of ~5 models cannot identify it; our probes need only the pool's own labels.
3. Onboarding cost: 20 ability parameters from ~200 anchors vs our 1 + 2 parameters from 5-10 examples. Head-to-head at
   matched k (5 / 10 / 50 / 200), giving them a low-D IRT our pool can support: TO DO (free, offline).
4. Reader: fine-tuned 66M DistilBERT vs a frozen 4B prefill. Our frozen 4B already beat a fine-tuned 137M on cost R2 and as a
   judge; their encoder not yet run on our pools.
5. Budgets: they predict a mean length per bin; per-request budgets need P(length <= cap | query), where the distribution's
   shape matters (4.A.29).
Their advantage to acknowledge: with a large model population, stage 1 gets a lot of free supervision for SUCCESS prediction.

### 4.A.31 Low-shot onboarding head-to-head vs ZeroRouter-style (2026-09-29; `onboard_vs_zerorouter.py`)
Hold out one route; all other routes keep identical full heads. ZeroRouter-style = 1-D 2PL IRT fitted on the other 4 routes'
TRAIN outcomes (their 20-D version is not identifiable from 4 models), stage 2 regresses (log a, b) on the same prefill
features we use (other routes' success logits + squares), new model = theta fitted on k anchors + per-bin (K=5) length
lookup; "zr-dopt" = their D-optimal anchors (deterministic, one draw). 20 draws of the k random anchors. Gain vs the full pool's
paper rule, MEAN over the 5 held-out routes (ours / zr / zr-dopt / naive):
| pool (full heads) | k=5 | k=10 | k=50 | k=200 |
| LCB (36.0) | 28.5 / 20.6 / 19.2 / 5.1 | 30.9 / 21.6 / 17.5 / 5.2 | 32.2 / 25.0 / 25.4 / 6.4 | 32.8 / 26.4 / 26.2 / 6.7 |
| Omni (28.7) | 23.1 / 21.3 / 27.6 / 6.3 | 23.6 / 24.3 / 28.0 / 13.7 | 23.9 / 26.1 / 28.2 / 16.9 | 23.6 / 26.3 / 26.4 / 18.3 |
| MMLU-Pro (30.9) | 26.9 / 21.0 / 26.6 / 3.5 | 28.3 / 19.2 / 24.9 / 4.7 | 30.3 / 23.6 / 23.5 / 8.7 | 29.9 / 24.8 / 24.2 / 9.6 |
On the model that matters most (dsv4f; without it: LCB 4.1, Omni -25.0, MMLU-Pro -22.1): LCB ours 23.0-30.8 vs zr -11.8..+6.5;
MMLU-Pro ours 23.8-31.2 vs zr 4.5-17.0; Omni ours 13.6-25.0 vs zr 21.3-35.9 (zr even beats our full heads there).
=> Ours wins clearly on LCB (+7-9 pt mean) and MMLU-Pro (+5-9 pt, ours > zr on 4-5 of 5 routes at every k); ZeroRouter-style wins
on Omni at k >= 10 (+1-3 pt; its D-optimal anchors +4-5). Consistent with where their cost design breaks: LCB has a large
model-specific cost share for dsv4f, MMLU-Pro has work != difficulty; Omni length IS difficulty, so difficulty bins are right.
No CIs (means over 20 anchor draws). CORRECTIONS to 4.A.30: point 2 ("their latent needs ~200 models") is too strong -- a 1-D
version fitted on only 4 models is competitive and wins on Omni; restate as "their 20-D latent needs a population; the version a
small pool supports is 1-D". Point 3 (onboarding) is a win on 2 of 3 pools, not across the board.

### 4.A.32 ZeroRouter fitted on our 5-model pools: dimension sweep + component swap (2026-09-29; `zr_dimsweep.py`)
Their stage 1 (D-dim 2PL IRT, MAP with Gaussian priors -- the paper uses SVI on the same model) fitted on the 5 pool models'
TRAIN outcomes; stage 2 = frozen 4B prefill activations -> PCA(256) -> ridge onto (log alpha, b) (a stronger reader than their
DistilBERT: generous to them); cost = their Eq. 8-10 (s = alpha^T b, K=10 bins, per-model mean train output per bin).
| pool | D | success log-loss zr (ours) | routing gain vs paper: zr / zr success + OUR cost / OUR success + zr cost (ours) | onboarding k=10 ours / zr; k=50 |
| LCB | 1 / 2 / 5 / 20 | .435 / .432 / .417 / .428 (.387) | 9.1 / -8.1 / 19.3 / 14.5; 32.4 / 27.3 / 35.7 / 35.1; 13.8 / 4.1 / 16.5 / 12.2 (36.0) | 30.8/18.5 30.3/9.3 30.3/19.2 29.5/12.7; k=50 ~32/18-26 |
| Omni | 1 / 2 / 5 / 20 | .502 / .495 / .483 / .499 (.449) | 20.1 / 6.9 / 12.3 / 7.3; 22.0 / 19.2 / 20.2 / 20.0; 14.1 / 10.1 / 11.8 / 11.3 (22.7) | 17.4/17.7 18.7/11.9 19.3/15.2 19.3/16.6; k=50 zr +0.1..2.0 |
| MMLU-Pro | 1 / 2 / 5 / 20 | .593 / .608 / .630 / .628 (.519) | 9.6 / 9.4 / 8.0 / 10.0; 33.6 / 31.2 / 32.4 / 33.9; 5.1 / 6.7 / 1.9 / 6.9 (30.9) | 28.2/19.6 28.8/14.7 28.0/18.5 29.6/19.0; k=50 ~30/20-23 |
Findings. (1) With 5 models, dimensionality buys nothing: no trend from D=1 to D=20 in success log-loss, routing or onboarding
(only noise, e.g. LCB routing 9 / -8 / 19 / 15%). The latent collapses as predicted. (2) COMPONENT SWAP -- the difference is the
COST side: their success model + our cost head ~= ours on every pool (LCB 27-36 vs 36.0, Omni 19-22 vs 22.7, MMLU-Pro 31-34 vs
30.9), while our success model + their bin-lookup cost collapses (LCB 4-17, Omni 10-14, MMLU-Pro 2-7). Their success predictions
are usable for routing (log-loss somewhat worse than our probes); their pricing is what fails -- even on Omni, where length
tracks difficulty, because s = alpha^T b from a fitted IRT is a noisier difficulty scalar than a direct read. (3) Onboarding: ours
ahead by 9-21 pt at k=10 on LCB and MMLU-Pro; Omni about even (zr +0-2 pt at k=50). Caveats: MAP not SVI; K unstated (10 used);
one seed, no CIs; stage 2 on our features, not their encoder. Supersedes the ZeroRouter-style stand-in (4.A.26 table used OUR
difficulty for the bins, which is kinder to them: 17.9-21.7% on Omni vs 10-14% here).

### 4.A.33 Hardened ZeroRouter reproduction: seeds, bin counts, paired CIs (2026-09-29; `zr_repro.py`, zr_repro.json)
Setup as 4.A.32 (their stage 1 at D = 1, 5 on the 5 pool models, stage 2 on our frozen 4B activations); 3 stage-1 seeds; K = 5 /
10 / 20 bins; paired bootstrap (500 test resamples, identical across arms) at K = 10; onboarding with 20 anchor draws per route.
Routing, ours minus ZeroRouter [95% CI], ranges over the 3 seeds (ours: LCB 36.0, Omni 22.7, MMLU-Pro 30.9):
| pool | D=1: ours - zr | D=1: cost side only (ours - our succ + zr cost) | D=5: ours - zr | D=5: cost side only |
| LCB | +26.9..+27.1 [~+17, ~+37] | +21.5..+21.9 [~+15, ~+29] | +16.7..+17.7 [~+9, ~+25] | +18.7..+19.9 [~+13, ~+27] |
| Omni | +2.3..+3.5 [~-11, ~+17] | +7.8..+8.5 [~-2, ~+19] | +11.3..+14.1 [~0, ~+25] | +10.5..+11.3 [~0, ~+21] |
| MMLU-Pro | +21.4 [+6, +39] | +24.9 [+12, +35] | +21.8..+25.1 [~+8, ~+38] | +26.7..+29.7 [~+13, ~+41] |
- Seeds barely matter (spread <= 3 pt). K matters on Omni: zr at D=1 ranges 13.8% (K=5) .. 16.6-22.7% (K=20), i.e. at its best bin
  count it ties us on Omni. On LCB zr stays at ~9-19% for every K; on MMLU-Pro 2-22% (best at K=5), always well below 31%.
- Their success model + our cost head ~= ours on every pool and setting (LCB 32-36, Omni 19-22, MMLU-Pro 32-34): the gap is the
  bin-lookup pricing.
Onboarding, ours minus zr (mean over held-out routes, 95% interval over anchor draws):
| pool | D=1 k=10 | D=1 k=50 | D=5 k=10 | D=5 k=50 |
| LCB | +10.4 [+4.9, +18.5] | +6.1 [+3.7, +8.4] | (see log) | |
| Omni | +0.7 [-5.4, +5.1] | -1.0 [-2.9, +0.2] | +2.6 [-13.2, +12.7] | -0.7 [-2.7, +0.6] |
| MMLU-Pro | +8.3 [+1.4, +16.0] | +6.7 [+3.6, +10.3] | +10.0 [+1.1, +19.4] | +7.3 [+4.4, +9.6] |
By route: ours wins big on the valuable model (dsv4f +9..+31 pt everywhere), loses on the cheapest (oss20lo -3..-19).
=> DEFINITIVE for the reproduction at pool size: ZeroRouter loses 17-27 pt (LCB) and 21-25 pt (MMLU-Pro) of cost savings to us, CIs
exclude 0, robust to seed and bin count; on Omni (length ~ difficulty) it ties at its best bin count. The whole gap is on the pricing
side. Onboarding: ours better on LCB and MMLU-Pro (CIs exclude 0), tie on Omni. Fairness to do: choose their K on calibration
(Omni favours K=20); their own encoder (zr_encoder.py job running).

### 4.A.34 ZeroRouter with its OWN encoder (fine-tuned DistilBERT + 11 linguistic features) (2026-09-29; `zr_encoder.py`, `zr_encoder_eval.py`)
Their stage 2 as in the paper (distilbert-base-uncased, 40 epochs, lr 3e-5, batch 32, fusion trunk + residual difficulty head;
features hand-computed; MSE onto the stage-1 targets; epoch chosen on held-out 10% of train), stage 1 on the 5 pool models, seed 0.
Held-out-train R2 of the latent (log alpha, b): DistilBERT / 4B reader -- LCB .26 / .40 (D=1), .22 / .22 (D=5); Omni .23 / .03, .12 /
.02; MMLU-Pro .03 / .00, .08 / .00. At pool size the fitted latent is barely readable from text by either reader (the targets are
noisy estimates from 5 models' outcomes).
Routing gain vs the paper rule, K = 5 / 10 / 20 (ours: LCB 36.0, Omni 22.7, MMLU-Pro 30.9):
| pool | D | zr[their DistilBERT] | zr[4B reader] | their success (DistilBERT) + our cost |
| LCB | 1 / 5 | 12.7 / 12.2 / 9.5; 3.4 / 11.3 / 7.6 | 9.8 / 9.1 / 9.0; 18.9 / 19.3 / 19.3 | 29.2; 26.9 |
| Omni | 1 / 5 | 13.0 / 9.1 / 12.0; 5.3 / 0.8 / -10.9 | 13.8 / 20.1 / 20.0; 7.6 / 12.3 / 19.2 | 24.9; 23.5 |
| MMLU-Pro | 1 / 5 | 8.4 / 7.1 / 8.2; 12.1 / 3.4 / 0.2 | 9.8 / 9.6 / 9.9; 21.6 / 8.0 / 2.3 | 33.3; 35.5 |
=> With its own encoder ZeroRouter is at best 13% (LCB), 13% (Omni), 12% (MMLU-Pro) -- equal to or below the 4B-reader version
(the Omni tie at K=20 in 4.A.33 disappears: 12.0%). Their success head + our cost head is again ~ours (27-36%). Conclusion unchanged
and stronger: at deployment-pool size the method's pricing is the failure, and its own reader does not rescue it.

### 4.A.35 ZeroRouter with a model POPULATION (preview, 53 leaderboard models; 2026-09-30; `zr_population.py`)
Their stage 1 fitted on TRAIN questions with N Open LLM Leaderboard models (their data source; per-question MMLU-Pro outcomes,
`leaderboard_mmlupro.py`, 196 models downloaded) + our 5 pool models; stage 2 = 4B reader. MMLU-Pro, ours 30.9%, log-loss .519:
| N | success log-loss | zr best-K | our success + zr cost | zr success + our cost |
| 0 | .59-.63 | 10-22% | 2-7% | 32-34% |
| 5-20 | .59-.64 | 7-19% | 2-12% | 30-34% |
| 50-53 | .60-.62 | 9-14% | 1-5% | 32-33% |
=> Population data does not rescue it: success prediction does not improve, pricing stays broken. Caveat: leaderboard models are
mostly small non-reasoning models (MMLU-Pro 11-59%, 5-shot loglikelihood), unlike our 60-83% reasoning pool -- their method needs
a population LIKE your deployment pool. Full 196-model run submitted (eai zr_pop_050201).
APPS full pool (one draw per route, ~$11) collecting: oss20md 72.8%, oss120md 79.1%, dsv4f 84.3%, oss120hi 87.4% (oss20lo 58%).

### 4.A.36 CORRECTION: ours vs ZeroRouter with ITS configuration tuned (2026-09-30; `zr_best_ci.py`, zr_best_ci.json)
4.A.33's "+21 to +25 pt on MMLU-Pro, CIs exclude 0" compared against ZeroRouter's DEFAULT configuration (K=10). With its
configuration chosen on CALIBRATION (D in {1,5} x 3 seeds x K in {5,10,20}; 4B reader) and applied once to test, paired bootstrap:
| pool | ours | ZeroRouter (calibration-chosen config) | ours - ZR [95% CI] | ZR best-on-test (upper bound) |
| LCB | 36.0 | 18.5 (D5, K10) | **+17.6 [+9.8, +24.9]** | 19.3 -> +16.7 [+9.1, +23.9] |
| Omni | 22.7 | 20.0 (D1, K10) | +3.0 [-10.7, +16.6] | 22.7 -> +0.7 |
| MMLU-Pro | 30.9 | 21.6 (D5, K5) | +9.3 [-2.6, +21.5] | same config |
=> The definitive ZeroRouter comparison: significant win on LCB only; MMLU-Pro +9 pt directional (CI touches 0); Omni tie. The
component-swap mechanism (their pricing is the weak part) is unchanged. MMLU-Pro test n = 300; a larger test set would decide it.
Headline figure: analysis/figures/fig0_headline_vs_zerorouter.png.
- 4.A.36 addendum (accuracy range behind "matched accuracy"): the 12 targets span, for BOTH arms identically, LCB 52-89%, Omni
  52-74%, MMLU-Pro 55-82% (test). Direct cost ratio ZeroRouter/ours at common accuracies: LCB x1.19 / 1.36 / 1.41 / 1.34 / 1.20 /
  1.06 at 54 / 60 / 67 / 74 / 80 / 87%; Omni x1.16 / 1.13 / 1.05 / 1.02 / 1.00 / 0.79 at 53 / 57 / 61 / 65 / 69 / 73%; MMLU-Pro
  x1.12 / 1.23 / 1.14 / 1.18 / 1.06 / 1.01 at 57 / 61 / 66 / 71 / 76 / 81%. Our edge is in the low-to-mid accuracy range and
  vanishes at the top (everyone calls the strongest model); on Omni ZeroRouter is cheaper at the very top.

### 4.A.37 Joint MLP heads versus linear rich-prefill heads (2026-09-30)
Authorized controlled comparison on LCB and Omni; `analysis/cost_headroom/mlp_heads.py`,
protocol and detailed results in `analysis/cost_headroom/MLP_HEADS.md`, compact raw results in
`mlp_heads_results.json`. Frozen identical mean+last features across eight stored layers,
same splits, labels and market prices. Width 64/128 shared-trunk MLPs, three seeds,
calibration-selected widths and epochs, all-seed ensembles. Four arms isolate success,
cost, and combined replacement. All 24 GPU runs plus CPU aggregation succeeded.

Direct additional cost saved versus linear at matched test accuracy [paired 95% CI]:
| replacement | LCB | Omni |
| --- | --- | --- |
| success only | -1.5% [-6.1, +2.1] | -0.03% [-7.0, +8.1] |
| cost only | -0.7% [-5.0, +2.7] | +5.5% [-2.4, +12.3] |
| both | -1.9% [-7.3, +2.4] | +7.1% [-1.6, +15.4] |

Shared bands: LCB 53.7–87.4%, Omni 52.6–73.1%. Reconstruction versus published
linear predictions changes routing cost by only .13% on LCB and 0% on Omni.
No significant averaged MLP improvement; keep linear heads primary. Omni cost
effect is positive across seeds, but uncertain. Calibration-selected deployment
rows show no consistent dominance (see raw results; achieved test accuracies differ).
This tests head architecture on our rich features, not the prefill-router paper's
PCA and layer-selection pipeline. No implication that every MLP is inferior.

### 4.A.38 Free ZeroRouter uncertainty and cross-fitting follow-up (2026-09-30)
User authorized the free analyses first. Protocol `analysis/cost_headroom/ZR_POWER_PROTOCOL.md`;
implementation `zr_power.py`, report/figure and raw artifacts in `zr_power_results/`.
All local CPU work; no new generations or API spending. Original per-arm fixed-split
savings reconstruct to numerical precision. Effects below are plug-in percentage-point
differences in savings versus median-output routing, not bootstrap means.

| pool | original fixed split [95% CI] | five-fold cross-fit [conditional 95% interval] |
| --- | --- | --- |
| LCB | +17.5 [+10.3, +24.6] | +18.4 [+14.1, +22.6] |
| Omni | +2.8 [-10.4, +16.7] | +13.8 [+6.5, +21.6] |
| MMLU-Pro | +9.3 [-2.0, +20.6] | +23.8 [+12.5, +33.0] |

Both methods refit per fold, with inner calibration only for configuration selection.
Intervals condition on fitted heads; overlapping training folds and omitted training
variance prevent algorithm-level significance claims. Cross-fitting uses larger training
sets and random splits (LCB original split is temporal). Only one partition (seed17).
Fold effects: LCB +19.9/+22.7/+11.7/+12.9/+26.3; Omni +25.6/+11.4/-4.2/+1.7/+30.2;
MMLU-Pro +1.8/+26.5/+43.2/+31.9/+13.1. Shared-three-arm accuracy-band sensitivity
preserves conclusions. Keep original fixed split primary; CV is supporting evidence.

Generation-only bootstrap SD is 1.72/3.72/2.69 pp (LCB/Omni/MMLU-Pro), versus
problem-only 3.72/6.82/6.00 pp. Generation noise is noticeable. Nested resampling is a
diagnostic, not an identified population variance decomposition; problem resampling
already contains noisy empirical means. Additional draws cannot be dismissed from
output ICC alone, and this analysis does not estimate their marginal value.

Exploratory equal-weight Stouffer combination on original fixed splits: one-sided
p=.00194; individual centered-bootstrap p=.001/.371/.057. Pooled directional evidence
does not establish three dataset-specific wins. Omni +/-5 pp equivalence fails:
original 90% CI [-8.1,+14.2], normal-approximation TOST p=.372. Cross-fit equivalence
also fails (90% [+7.4,+20.3]). Do not describe the original Omni result as equivalent.
If a separate MMLU-Pro win is needed, prioritize untouched additional problems; no
paid follow-up launched. Full-feature/reduced-row-space fit checked on Omni fold0,
route0: max probability delta .000155, max relative cost delta 3.7e-7, same ridge alpha.

### 4.A.39 Held-out MMLU-Pro and full Omni-MATH expansion (2026-10-01)
User requested collection jobs, cost estimate and an OpenRouter balance check.
Frozen plan and manifests: `analysis/cost_headroom/expansion_20261001/`;
preparer `prepare_expansion.py`, collector `collect_expansion.py`, launcher
`launchers/abstention/launch_math_expansion.sh`. New held-out evaluation:
2,000 MMLU-Pro and 1,000 full Omni-MATH questions, retaining old sample's
subject/difficulty proportions and excluding ID/normalized-text overlaps.
Existing train/calibration sets and configuration choices stay fixed for the
primary follow-up. Same five routes, original 4/3/3/2/2 valid draws, 64k cap.
42,000 calls. API estimate at provider price ceilings: MMLU $20.20, Omni
$45.62, total $65.81; 25% buffer total $82.26. Guards $30/$70. GPU feature
extraction/cluster compute are separate. Key has $150.03 remaining of $700
allowance at preparation; credentials never printed or committed. Full
Omni benchmark version is pinned; MMLU reuses existing cached version.

4.A.39 revision: user prioritized distinct new problems over repeated draws,
then requested a $60 total target to reserve funds for later collection.
Prepared replacement: 6,500 MMLU-Pro + 1,000 Omni, one draw per each of five
routes (37,500 calls). Price-ceiling estimates $27.65+$19.43=$47.08;
25% buffer $58.85; job guards $35+$25=$60. All new examples remain held-out.
Some small MMLU subjects reach capacity; recorded original-subject weights
preserve the primary comparison's intended mixture; also report unweighted.
The prior launch attempt was rejected by automatic approval review because
setup/estimate was not explicit permission for paid submission. No jobs have
launched. Revised plan supersedes the earlier multi-draw plan and guards.

4.A.39 launch: user explicitly approved the revised plan. Both jobs RUNNING
on 2026-10-01, snapshot 746a80a; MMLU job 73fe00f5-4310-411b-b509-775f0b60bd8b,
Omni job d1cd0673-3817-4a89-9544-3094edeb607f. Initial generation responses
confirmed, no initial API errors; collection incomplete. Exact status and
output root in expansion_20261001/launch_status.json.

### 4.A.40 Paper revision: shared-prefill contribution and four-page layout (2026-10-01)
User approved the editorial recommendations and requested the text update.
`paper_nowai/main.tex`, synchronized Markdown and compiled PDF now foreground
one frozen prefill supplying route-specific success and cost readouts. New
Figure 1 combines architecture with controlled pricing savings; both branches
have task labels (Success readouts / Cost readouts). Methods explains logistic
binomial likelihood versus ridge squared error in log mean length. Main table
has four cost estimators with their own gains/CIs instead of mixed difference
rows. ZeroRouter is a reimplementation; Omni is inconclusive, not equivalent.
ZeroRouter point differences are plug-in effects (+17.5/+2.8/+9.3), rather than
bootstrap means. Original fixed-split results remain primary; new collection
outcomes are not incorporated. Headroom, additional controls, prospective
screening and exploratory cross-fitting move to `supplement.tex/pdf`, with
explicit conditional-CI caveats. Main PDF is four pages including references;
supplement two pages. The MLP comparison remains outside this paper.

### 4.A.41 Paper prose revision (2026-10-01)
User requested less formulaic and deliberately catchy wording. Retitled the
paper "Predicting Reasoning-Model Costs from Shared Prefill Activations" and
revised the abstract, introduction, results headings, conclusion, and figure
captions into descriptive academic prose. Updated supplementary wording and
figure panel titles, regenerated Markdown and figures, and recompiled PDFs.
Methods, numeric results, comparisons, and uncertainty qualifications remain
unchanged. Main paper remains four pages including references; supplement two.

### 4.A.42 Dedicated CARROT-KNN-SBERT comparison (2026-10-01)
User requested setup of the pending CARROT comparison. Reference inspected at
somerstep/CARROT revision 3e6acff6aecf4cbcb8f31a118d04c799c2ea1655.
`carrot_compare.py` reproduces the upstream local SBERT variant (MiniLM-L12-v2,
cosine uniform kNN, separate multi-output success/raw-length regression,
training-only five-fold R2 selection of k). No API calls. Original LCB/Omni/
MMLU splits; all-valid-draw labels; cost-only, full-router, constant-cost and
pricing-swap contrasts with 1,000 paired problem bootstrap samples. Direct
cost-ratio effects over pair-specific common bands differ from existing
median-normalized difference summaries; do not interchange table metrics.
Protocol: CARROT_PROTOCOL.md. Launcher: launch_carrot_compare.sh. Isolated
sentence-transformers dependency; CPU inference disables unused DeepSpeed
integration to avoid Triton GPU initialization. LCB end-to-end five-bootstrap
smoke completed; its intervals are not reportable experiment results.
4.A.42 execution: all original-split comparisons completed locally with 1,000
paired resamples after three cluster submissions failed during startup without
usable logs. Results and execution IDs: `carrot_results/`. Runner now persists
startup logs and launcher supports local execution. Full-router direct savings
vs CARROT-KNN-SBERT: LCB 20.2% [11.0,26.9], Omni 19.9% [6.3,32.3], MMLU 5.2%
[-12.5,20.5]. With OUR success fixed, ours vs CARROT costs: 33.6% [26.4,40.3],
14.0% [1.7,24.2],14.4% [-0.3,26.6]. CARROT cost vs mean-length constant with
its success fixed: 1.7% [-0.8,5.0],9.7% [-1.8,21.2],26.3% [9.3,36.5].
The MMLU cost estimator is competitive; no universal superiority claim. These
are pair-specific direct cost ratios, not existing table's differences of
median-normalized savings. Expansion examples remain excluded; no API calls.

### 4.A.43 CARROT-style trained Jina-137M comparator (2026-10-01)
User requested our 137M encoder for the trained CARROT-family comparator.
`carrot_jina.py` fine-tunes separate success/cost encoders, BCE on valid-draw
success means and MSE on train-standardized raw output lengths; no log/smearing
postprocessing. Existing calibration selects task checkpoints. Six epochs,
seed42, lr2e-5, batch8, maxlen1024. This is an ADAPTED CARROT-style baseline,
not a RoBERTa reproduction; adaptations in CARROT_JINA_PROTOCOL.md. Reuses
six direct matched-accuracy contrasts and1,000 paired bootstrap draws. External
prediction integration reproduces all six completed LCB point estimates exactly.
Three single-GPU snapshot jobs; saved heads permit fresh expansion evaluation.
Original-split outputs only; new generation samples stay held out. No API calls.

### 4.A.44 Jev preliminary success pilot (2026-10-01)
User authorized a small Jev experiment following pricing estimate, then
explicitly requested success prediction with cost held constant. Frozen
100-per-dataset test sample, seed20261001, five correctness nouls per request;
300 total calls to pinned typesafe/jev-1.13. State includes only problem,
grading rule, model/effort descriptions and training-only aggregate accuracy
priors; no golds/generated answers/test labels. Same median TRAIN output
length pricing for all success arms; compare ours, Jev and constant base rates.
No new target generations or cost questions. $1 cumulative spend guard;
usage recorded and successful calls skipped on resume. Protocol and frozen
manifest in jev_pilot_20261001. Exploratory 100-problem subsets,1,000 paired
problem bootstrap; predictor metrics expected-draw Brier/logloss.
4.A.44 completion: 300/300 Jev requests succeeded, no retries/errors, recorded
spend $0.016132284. Resolved model typesafe/jev-1.13-20260917. Protocol/request
manifest was committed before the first call; prompt unchanged throughout.
Same median-length costs for both success predictors. Direct Jev savings vs
our heads: LCB+13.6%[-6.3,27.5],Omni+22.9%[-6.9,39.2],MMLU-15.0%[-44.9,2.3];
all three intervals include zero. Jev vs training-base-rate router on Omni
+31.6%[14.9,47.1]. Our heads have lower Brier/logloss point estimates on all
three pilot subsets; Jev underpredicts average correctness. Report and full
bootstrap samples in jev_pilot_20261001. All contrasts conditional/descriptive,
100 problems each, no prompt selection or calibration. Cost buckets untested.

### 4.A.45 Jev cost-bucket follow-up (2026-10-01)
User requested "try the cost?". Same frozen 100 test problems per dataset as
success pilot, 300 calls with five route-specific choice questions each.
Training-only quintile bins and arithmetic representative output lengths,
probability-weighted mean prediction; fixed OUR success heads for all cost
contrasts. Exact input costs, recorded route prices, all valid draw labels.
Compare Jev costs against median, mean and our prefill costs with 1,000 paired
problem bootstraps. $1 cumulative guard, no new target generations. Manifest
and protocol frozen before cost API calls; exploratory, no test calibration.
4.A.45 completion: 300/300 cost calls succeeded, zero errors, $0.042466284.
Resolved typesafe/jev-1.13-20260917. Jev vs median savings LCB+5.8%
[-11.0,16.2],Omni-0.8%[-7.6,5.8],MMLU-12.6%[-39.4,10.2]; no clear gains.
OUR cost head vs Jev, OUR success fixed: LCB37.4%[24.1,45.7],
Omni20.4%[5.5,31.8],MMLU18.1%[-7.8,36.9]. Exploratory conditional paired
intervals,100 problems each. Jev raw-token R² mostly near zero or negative;
ours stronger on LCB/Omni. MMLU Jev strongly overpredicts lengths. No prompt
tuning/calibration or paper edits. Manifest and derived predictions/results
versioned, raw response JSONL retained locally. This does not test fine-tuned
or calibrated Jev; current untrained prompted cost predictor is unpromising.

### 4.A.46 Jev success + our cost replay (2026-10-01)
User authorized hybrid replay, no new API calls. Same frozen100/pool test
subsets, stored Jev success probabilities, fixed our learned prefill costs.
Primary hybrid vs our full router direct cost savings: LCB-25.9%
[-49.7,-3.3],Omni+0.8%[-17.1,15.8],MMLU-9.2%[-34.6,15.3]. Hybrid worse
on LCB, others inconclusive; no established improvement. Previous median-cost
Jev success point advantages do not persist with learned cost. Cost-head
swap median->ours with Jev success fixed gives LCB15.1%[-3.7,30.1],
Omni-0.2%[-16.6,17.9],MMLU12.3%[-6.1,35.3], all inconclusive. Existing
our-success learned-cost and median-cost success contrasts reproduce earlier
pilot points exactly. 1,000 paired problem bootstrap, all valid draws,
conditional/descriptive pair-specific frontier bands; no tuning/calibration.
Code/protocol and results in jev_hybrid_20261001; raw success response hash
and derived probability vectors saved. Keep current full router, no paper edit.

### 4.A.47 Intern-Decision-4B success pilot (2026-10-01)
User requested closest open Jev counterpart and authorized trying Intern.
Same frozen300 Jev test questions, unchanged state and training-only priors.
Pin HF revision0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd, published inference
and defaulttemperature1.99241824; no test calibration. One local GPU forward
per problem scores5 routes. Primary our prefill costs fixed for Intern-vs-ours
and Intern-vs-Jev success comparisons. Secondary median-cost comparisons and
cost-head effect with Intern fixed. 1000 paired problem bootstrap, unchanged
stored valid generation draws, descriptive conditional frontiers. Zero API
calls; isolatedPython3.12 plus pinned upstream dependencies, no truncation,
public anonymous download avoids expired ambient HF OAuth. Frozenmanifest,
protocol and scripts in analysis/cost_headroom/intern_decision_20261001.
4.A.47 launched from snapshot1c5b8d58503d1fe4eb5cb6428f954750b9db8cc7,
job8d1272ac-7aec-4e38-a07a-a7ece98e85b4; verifiedQUEUED at submission.
One32GB GPU,8CPU,48GB RAM. Persistent outputroot recorded in launch.json.
Published pinned dependency releases checked available; actual inference
awaits scheduling. No success/result claim yet.
4.A.47 startup failure: job8d1272ac completed dependency setup but failed before
model download/inference because shared carrot_compare imports scikit-learn.
Added scikit-learn to isolated runtime dependencies; no predictions collected.
Relaunch with corrected snapshot, same frozen manifest and model protocol.
4.A.47 completion: replacement job9ddbdc06 SUCCEEDED;300/300 predictions and
1,000 paired bootstrap evaluations per contrast completed. Zero API spending.
Intern success vs OUR success with OUR costs fixed: LCB-25.8%[-45.0,+0.2],
Omni+1.1%[-16.3,+14.1],MMLU-10.1%[-34.9,+12.4]. None excludes zero.
Intern vs Jev same OUR costs: LCB+3.5%[-7.7,15.3],Omni-0.3%[-7.4,5.2],
MMLU-0.8%[-14.7,10.4], all inconclusive. Descriptive100/pool, conditional
paired CIs; no equivalence claim, calibration or fine-tuning performed.
Derived probability vectors, execution metadata, report and full bootstrap
results copied into versioned protocol folder; raw responses persist at
/mnt/llmd/results/exps/aristides/reason/intern_decision_20261001/responses.jsonl.

### 4.A.48 Intern success fine-tuning (2026-10-01)
User authorized fine-tuning; clarified both success comparison controls.
Primary OUR costs fixed for fine-tunedIntern-vs-OUR success heads; median-cost
control secondary, plus Brier/logloss and pilot fine-tuned-vs-untuned. Separate
LoRA adapters per originalpool, originaltrain441/275/550, calibration110/75/150,
full originaltest341/150/300. Expansion collections untouched. Published
five-noul decision format, BCE on candidate margin/fixedpublishedtemperature
against valid-draw means. Rank16 alpha32 dropout.05, languageattention/MLP
only, lr1e-4,5epochs,accum8,micro1,seed42; epoch0 is calibrationcandidate.
Choose minimum calibrationNLL, never test outcomes. Saveadapter/calprobs for
fresh expansion.1,000 pairedproblem bootstrap, allroutes/draws retained;
originaltestexploratory, conditionalone-seed intervals, no overhead pricing.
Zero API spending; three48GB-GPU snapshotjobs with isolatedPython3.12 and
pinnedupstreamruntime+peft0.21.2. Protocol and hashes frozen before launch.
4.A.48 submitted snapshotf910b80442233155737be09581738fe2d33a9ba6:
LCB3e84da1b-335b-4e87-a055-6cc66d2cf707(verifiedQUEUED),
Omni70d412b8-9968-44f6-a2f6-675dfb7c972c(verifiedQUEUED),
MMLUdf2ce134-988a-4bf8-ac13-1abfa4f9359d(verifiedQUEUING).
All await scheduling; no completed fine-tuning/result claim yet. Launchmetadata
versioned next to frozen protocol. No additional API calls authorized/needed.

### 4.A.49 Fixed operating-point check (2026-10-01)
Calibration selected V by mean validation utility of OUR success+cost router,
then froze V and routed once per untouched original-test problem. Compare learned
vs median costs holding OUR probabilities and V fixed; paired problem bootstrap
2,000 reps. Selected V hit grid ceiling 100 cents ($1)/correct on all pools.
Test: LCB cost savings+3.0%[0.4,6.0],accuracyΔ+0.01pp[-0.23,+0.23];
Omni+0.8%[-0.4,3.1],accuracyΔ+0.33pp[0,+1.0]; MMLU exactly0 at fixed V.
Historical test already explored, so not confirmatory. Fixed policy results saved
under fixed_policy_20261001; report selected grid boundary explicitly. Fresh
expansion evaluation supersedes as strongest evidence.

### 4.A.50 Expansion feature extraction preparation (2026-10-01)
Generation retry audit: Omni-MATH5,000/5,000 complete; MMLU-Pro32,499/32,500,
one valid response unresolved. Explicitly resumed both previously authorized
jobs; append-only retries remain active.
Answer-free raw-problem Qwen prefill inputs for all frozen task IDs prepared,
hashes in expansion_prefills_20261001 manifests. Submit separate Instruct
(MMLU-Pro) and Thinking (Omni) Qwen3-4B extraction jobs, matching original
feature route and eight-layer last/mean representation. Frozen readouts to
remain fitted on old train/calibration only. Fresh MMLU weighted/unweighted and
Omni difficulty-stratified estimates planned.

### 4.A.51 Fresh evaluation pipeline (2026-10-01)
Initial Qwen3-4B prefill files used a system prompt that differed from the
cached feature metadata (`expert competitive programmer` vs `helpful assistant`).
Invalidated those files and updated extraction to match the original prompt;
both no-cost prefill jobs are being rerun. Added tensor builder/readout runner to append
expansion outcomes/features while retaining original train/calibration IDs;
readout selection stays on original calibration only. Fresh results use one
route per problem at the frozen V with paired problem bootstrap, MMLU subject
weights/unweighted and Omni difficulty weights. Since the historical selected V
hit the grid ceiling, also report fixed-V sensitivity at $0.0001/$0.001/$0.01
per correct. Outcome-swept frontier remains secondary/descriptive. CPU launcher
is guarded on collection COMPLETE and prefill NPZ; launch when both prerequisites
finish.

### 4.A.52 Fresh expanded evaluation complete (2026-10-01)
Both collections, corrected-prompt prefills, and frozen-readout evaluations have
completed. New evaluation data: 6,500 MMLU-Pro problems and 1,000 Omni-MATH
problems; all original train/calibration readouts and calibration-selected
V=$1/correct were kept fixed. At this one operating point, learned costs reduce
mean generation spend by 13.0% (design-weighted problem-bootstrap CI
[11.1,14.9]) on MMLU-Pro and 9.2% (CI [4.9,13.9]) on Omni-MATH. Weighted accuracy
deltas are -0.29pp ([-0.58,-0.01]) and -0.20pp ([-0.50,0.00]); the unweighted
MMLU-Pro accuracy CI crosses zero. The selected V was at the upper edge of the
original grid. Secondary outcome-swept frontiers disagree by dataset: learned
costs have -33.4% savings on fresh MMLU-Pro and +14.2% on Omni-MATH. These
frontiers use evaluation outcomes to select mixtures over a lower common
accuracy range and are descriptive; report them separately from the fixed-policy
result. This is mixed evidence against a general frontier-wide claim, while
providing held-out support at the selected operating point on two datasets.
Results and protocol are summarized under
analysis/cost_headroom/expanded_eval_20261001/.

Intern-Decision LoRA jobs completed for LCB and Omni; MMLU-Pro fine-tuning is
still running (job df2ce134-988a-4bf8-ac13-1abfa4f9359d). These are exploratory
and not part of the paper results above.


### 4.A.53 Correction: expansion comparison invalidated (2026-10-01)
Section 4.A.52 and its fresh 13%/-33% results are withdrawn. The expansion encoder inputs omitted the original solving wrappers and MMLU answer options, and its cost fitter used calibration/target selection absent from the paper head. These are pipeline mismatches, not evidence of distribution shift. Generations remain valid and need no recollection. Corrected extraction uses full original prompt format, gates on 32 original-prompt anchor replays, and reconstructs archived train-only RidgeCV costs (max relative error 1.06e-6). Primary comparison: policies/mixes selected for the historical accuracy-target grid on original calibration; fixed on fresh problems; paired problem bootstrap for both achieved accuracy differences and spend. Include train-median and train-mean cost arms with the same success predictor. Fresh numerical claims removed from paper pending corrected results.

### 4.A.37 CI diagnosis, Intern-Decision check, and PROVIDER DRIFT on deepseek-v4-flash (2026-10-02; Claude)
Context: Codex expanded MMLU-Pro (+6,500) and Omni (+1,000) fresh problems (one draw per route) and built the NOWAI 4-pager
(paper_nowai/); see its commits 3b51d2f..35a8b63.
- **Intern-Decision-4B (fine-tuned) vs our prefill success readout, prediction quality on original test** (Intern test_predictions):
  log-loss ours/Intern LCB .387/.410 (ours better, borderline), Omni .449/.430 (n.s.), MMLU-Pro .519/.547 (ours better,
  Δ -.029 [-.050, -.007]). AUC on strong routes: Omni Intern .90/.91 vs ours .86/.88 (dsv4f, oss120hi) -> explains its Omni routing
  win; elsewhere no better. Not a general win; settle Omni on the fresh set (inference only, no API spend).
- **Net-utility metric** (`utility_metric_check.py`: U(V) = mean[V*correct - spend], same V for both arms, V range fixed on
  calibration): relative precision NOT better than savings-at-matched-accuracy (CI half-width / effect: LCB .25 vs .21, Omni .42
  vs .56, MMLU-Pro .64 vs .44). The only real lever for tight CIs is more test problems (fresh sets: ±4-9 pt).
- **Provider drift.** On fresh data the learned cost head overprices dsv4f (predicted/realized 1.37 MMLU-Pro, 1.23 Omni; dsv4f
  fresh MMLU-Pro log-length R2 -0.29 vs +0.20 on original test); all other routes within ±8%. Cause: OpenRouter served dsv4f mostly
  from OpenInference in the fresh run (64% MMLU-Pro, 72% Omni), a provider absent from the original pools, which writes about half the
  tokens at equal accuracy (MMLU-Pro median 465 vs StreamLake 969; acc .859 vs .857). Providers already differed 2-3x within the
  original pools (MMLU-Pro mean output Relace 9.9k vs GMICloud 2.8k) -> hidden label noise and a deployment drift risk.
- **One-offset recalibration from k fresh examples** (`drift_reoffset.py`; both arms get the same k: ours rescales dsv4f by the
  ratio of means, median recomputes the dsv4f median; k problems excluded from evaluation; 10 draws): dsv4f calibration 1.37 ->
  1.19 / 1.02 / 0.99 (k = 10 / 50 / 200) on MMLU-Pro, 1.23 -> 1.16 / 1.03 / 1.04 on Omni. Net-utility gain over median pricing
  (avg over the calibration V range): MMLU-Pro +6.4% -> +7.6 / +14.4 / +15.7%; Omni +20.8% -> +25.0 / +25.2 / +22.1% (point estimates,
  no CIs yet). Note: a MEAN-LOG offset overcorrects (calibration 0.43-0.51) because predictions are means of right-skewed lengths.
  Fixed-target policies (chosen with original costs) swing in savings vs accuracy after recalibration -- another reason the
  per-target table is hard to read.

### 4.A.38 Endpoints, not models: deepseek providers as "same model + offset" (2026-10-02; `provider_endpoints.py`)
Provider recorded per draw on LCB/Omni/MMLU-Pro (100%), CC/APPS (~99.8%), fresh sets (~83%; misses = failed calls); BCB tensors
lack it. Within-problem log-length variance explained by provider: gpt-oss routes 0.2-1.3%, dsv4f 18.8% (MMLU-Pro) / 21.5% (Omni).
dsv4f served by 14 providers (all pools, 22k calls): OpenInference 27.6%, GMICloud 15.1%, StreamLake 13.8%, Baidu 13.5%, ... TODAY's
OpenRouter output prices range $0.084-$1.60/M (we priced all calls at $0.094/M); today's list is inconsistent with the ~$14 actually
billed for the fresh run -> need the account's activity export for historical per-call cost. Accuracy differs by provider on code
(CodeContests OpenInference .66 vs .84-.92; APPS .72 vs .81-.90), not on MMLU-Pro.
Prediction test (shared dsv4f readouts, fitted with NO provider info; offsets on train problems; eval = original test + fresh):
- Length: providers that differ get large gains from ONE offset -- OpenInference x0.65 (MMLU-Pro log-length MSE 3.30 -> 2.26,
  pred/real mean 1.77 -> 1.16; Omni x0.63, 1.12 -> 0.73, 1.47 -> 0.93), GMICloud x0.65 (3.39 -> 2.31); providers near the pooled mean
  (Baidu, DigitalOcean, DeepInfra) unchanged.
- Onboarding a provider as a NEW endpoint from k draws: k=10 already ~ the full offset (OpenInference MMLU-Pro MSE 2.08; Omni 0.76)
  and far better than treating it as an unrelated model from the same k (scratch: 2.91 MMLU-Pro, 3.93 Omni; every group, every k).
- Success: provider offsets do not improve log-loss (pooled .33-.47 already; small-k logit shifts HURT: k=5 LL +.1-.6) ->
  success is ~provider-invariant here; shift only the length side, or shrink the success offset.
=> Supports the paper framing "route over endpoints (model x effort x provider) with shared problem-level structure + per-endpoint
offsets; new / drifted endpoints onboard from ~10 labels" (same mechanism as new-model onboarding 4.A.18/4.A.21 and drift fix 4.A.37).
Limit: provider ROUTING cannot be evaluated offline (~1 dsv4f draw per provider per problem); needs a pinned collection (paid).

### 4.A.39 Billed prices, provider-routing literature, provider pilot launched (2026-10-02)
- **Billed per-call cost exists** for the fresh collection (`usage_cost` in math_expand_20261001 rows). Effective billed $/M out
  (least squares): dsv4f OpenInference 0.12, StreamLake 0.09, Baidu 0.11, GMICloud/DeepInfra 0.18, DigitalOcean 0.20, others
  0.27-0.28; gpt-oss-120b 0.17-0.60 (most volume DeepInfra/CoreWeave 0.17, Crusoe 0.25, Mancer 0.30, BaseTen 0.50); gpt-oss-20b
  0.13-0.14. OpenInference's listing has since risen to $1.60/M (Oct 2) -> prices move within days. Our analysis assumed
  oss20 0.018/0.09, dsv4f 0.047/0.094, oss120 0.15/0.60: gpt-oss-120b is billed ~2-3x CHEAPER than assumed on average and
  gpt-oss-20b ~1.5x DEARER -> the real price ladder is flatter than the one we modelled. Re-price all fresh results with billed costs.
- **Literature**: provider variance measured (2605.02821); provider routing for a FIXED model (2609.37902, FACET: per-(provider x
  task) feasibility certification, realized costs, layered beneath a model router). Not found: joint per-query routing over
  (model x provider) endpoints with predicted per-query cost, or onboarding a new provider via shared readouts + offset.
- **Pilot launched** (eai provider_pilot_20261002_062906; `collect_provider_pilot.py`): dsv4f pinned (only=[P], no fallbacks) to
  StreamLake / GMICloud / DigitalOcean on 2,000 fresh MMLU-Pro + 1,000 APPS; est. ~$13, guard $18; smoke test OK.

### 4.A.40 APPS full pool: pre-registered GAIN NOT confirmed (2026-10-02; apps_tensors, apps.json)
1,000 problems, ONE draw per route (deviation: pre-registration assumed 4/3/3/2/2), split 618 train+cal / 382 test. Accuracy
oss20lo 59.1 / oss20md 75.3 / dsv4f 83.5 / oss120md 81.1 / oss120hi 86.6%. Instruct-probe log-length R2 .55-.70.
Headroom 31.9% [17.5, 42.0] (>= 15% predicted: CONFIRMED). Probe gain vs paper rule 14.3% [-2.3, 23.6]: point estimate >= 10% but
the CI includes 0 -> the pre-registered GAIN call FAILS its criterion (directional). Cost-from-success 13.3%, ZeroRouter-style bins
13.2%, prompt GBM -8.4% (probe - GBM +20.8 [+2.4, +41.2]), mean constant -0.9%. Pre-registration tally: GAIN Omni, MMLU-Pro
confirmed; APPS directional, not confirmed; AIME, BBEH (NO GAIN), K&K, SuperGPQA not run.

### 4.A.41 Provider-routing pilot results (2026-10-02; provider_pilot_20261002, $11.26; `provider_routing.py`)
dsv4f pinned to StreamLake / GMICloud / DigitalOcean on 1,995 fresh MMLU-Pro + 957 APPS problems (all routes present).
- Providers ARE "the same model + offset": correct-answer agreement .93-.96 (MMLU-Pro) / .90-.91 (APPS) vs .73-.78 if independent;
  log-length correlation across problems .83-.95. Per-query provider-specific signal is small.
- Providers differ in level: APPS accuracy GMICloud .898 / StreamLake .847 / DigitalOcean .828 / unpinned .794; billed cost per call
  1.0 / 5.0 / 2.3 / 3.2 m$; GMICloud writes x1.5 longer. MMLU-Pro: accuracy .847-.855 (equal), cost .32-.74 m$.
- Routing (shared prefill readouts + per-provider offsets fitted on held-out FIT problems; billed prices), cost saved vs unpinned at
  matched accuracy: APPS endpoints router (all 3 providers as routes) +20.0% [+10.7, +27.9] and max accuracy 89.5% (vs 85.3%
  unpinned); best fixed provider StreamLake +22.4% [+12.5, +30.7] (max acc 86.2%). MMLU-Pro: endpoints +2.3% [-4.9, +7.7]; best fixed
  StreamLake +6.0% [-1.5, +11.5]; DigitalOcean -23.7%.
=> Where providers differ (code), treating (model x provider) as endpoints with shared readouts + offsets gets the cheap
provider's savings AND the accurate provider's top accuracy; where they don't (MMLU-Pro), nothing to gain. Most of the value is
level (pin the right provider), not per-query provider choice.

### 4.A.42 Billed re-pricing of the fresh-set result (2026-10-02; `billed_reprice.py`)
Realized cost = billed usage_cost; predictions and calibration priced at effective billed $/M (gpt-oss-120b ~2.5x cheaper, gpt-oss-20b
~1.5x dearer than the list prices we assumed). Learned cost vs median-length pricing, fresh problems:
| | test-frontier savings (shared band) | calibration-selected policies, targets .65-.85 (savings / acc diff) |
| MMLU-Pro (6,500), assumed | +23.9% [+19.5, +27.4] | +30.0 / +48.1 / +32.8 / +30.1 / -5.0%; acc -0.45 / -1.91 / +0.97 / -1.52 / +0.28 |
| MMLU-Pro, BILLED | **+23.2% [+18.5, +26.9]** | +17.2 / +29.0 / +17.1 / +19.3 / +12.1%; acc +0.51 / +0.23 / +3.18 / +1.32 / -1.11 |
| Omni (1,000), assumed | +23.4% [+17.5, +29.6] | +24.9 / +16.0 / +3.3% (.65-.75) |
| Omni, BILLED | **+28.2% [+20.3, +33.2]** | +15.3 / +16.8 / +12.7%; acc +1.80 / -1.70 / -1.60 |
=> The headline survives real prices with tight CIs. Under billed prices every calibrated target saves money; accuracy differences
remain operating-point dependent. (Unweighted; Codex's tables use stratum weights.)

### 4.A.44 Literature: provider variance and routing over provider endpoints (2026-10-02, detailed search)
Measurement / audits (variance is well documented -- not our claim):
- 2605.02821 (Li et al., May 2026, AI Ping): 29 providers, DeepSeek/Qwen/Kimi/GLM/MiniMax; latency, throughput, context, protocol,
  errors, price; prices anchored near official, performance not; counterfactual provider routing (-37.8% cost Qwen3-32B); NOT
  OpenRouter, no accuracy or per-query length.
- 2604.21083 (Lin et al., IMC '26): gateways incl. OpenRouter; response length, reasoning tokens, accuracy, billing/token accounting,
  silent substitution/downgrade.
- Model-equality testing (Gao et al., ICLR 2025): 11/31 Llama endpoints serve a different distribution; audits of substitution:
  2504.04715, AgentProv 2609.00052, IRIS 2607.20860 (gateway routing dilution), rank-based uniformity test 2506.06975.
- Practice: Artificial Analysis per-provider gpt-oss-120b accuracy (GPQA x16, AIME25 x32): AIME25 93.3% to 36.7% across providers;
  Willison; 16x eval; OpenRouter "Exacto" (provider variance acknowledged); Kimi Vendor Verifier (Moonshot); LessWrong "not pinning
  your OpenRouter provider might invalidate your research" (CoT-legibility result overturned; up to 16.6 pt shifts);
  AMindToThink/openrouter_reliable_research_search prior-work list; Epoch "why benchmarking is hard".
- Our own code already notes a mechanism (collect_lcb_trajectories.py): some dsv4f endpoints (OpenInference, DigitalOcean) skip
  reasoning unless reasoning.enabled is sent.
Routing over providers:
- 2609.37902 (He et al., Sep 29 2026): provider selection for a FIXED model via a public multi-provider aggregator, 6 models (DeepSeek,
  Gemma, Llama, Mistral), GSM8K/MMLU/HumanEval; output length varies median 1.35x / max 3.04x across providers; price uncorrelated
  with accuracy (rho +0.05); routing per (provider x task) facet (cheapest provider within 5 pt of best, >= 90% availability),
  FACET online certification + drift detection; -50% median cost, -63.7% live. Stacked beneath a model router (RouteLLM).
- Self-hosted joint model/instance routing (RouteBalance 2606.17949, BOute 2602.10729): replicas of a model are identical; per-query
  length prediction used for load, not provider quality.
=> NOT found: per-query joint routing over (model x effort x provider) endpoints; predicted per-query cost that is
endpoint-specific; shared problem-level readouts + per-endpoint offsets for onboarding new / drifted providers from ~10-50 labels;
evidence that providers of one model are "same model + offset" per query (our agreement .90-.96, length corr .83-.95).

### 4.A.43 All cost baselines on the FRESH sets at BILLED prices (2026-10-02; `fresh_baselines.py`, fresh_baselines.json)
One protocol: heads/estimators fitted on original train only, realized cost = billed usage_cost, predictions priced at effective
billed $/M; test frontier on fresh problems; DIRECT comparison = cost saved by ours at matched accuracy over the band both reach;
paired problem bootstrap (300). Unweighted. ZeroRouter = full reproduction (stage-1 IRT on the 5 routes, 4B reader, their success
model AND their pricing), (D, K) chosen on original calibration.
| arm | Omni (1,000): saved vs median | ours saves vs it | MMLU-Pro (6,500): saved vs median | ours saves vs it |
| ours (frozen 4B cost readout) | +27.8% [+20.3, +33.1] | -- | +23.1% [+18.1, +26.8] | -- |
| median length (paper rule) | 0 | +27.8 [+20.3, +33.1] | 0 | +23.1 [+18.1, +26.8] |
| mean length | -0.7 | +28.2 [+22.4, +33.6] | +3.3 | +20.5 [+16.7, +23.6] |
| cost from success head | +25.8 | +3.2 [-1.7, +7.7] | +10.5 | **+14.1 [+11.4, +16.6]** |
| ZeroRouter-style bins (our difficulty) | +18.6 | **+11.6 [+5.2, +16.2]** | +11.9 | **+12.7 [+7.8, +16.1]** |
| prompt-feature GBM | +9.8 | **+20.3 [+11.8, +26.5]** | +5.7 | **+18.4 [+12.2, +21.7]** |
| ZeroRouter (full repro) | +11.9 | **+18.0 [+9.0, +24.7]** | +4.5 | **+19.3 [+15.5, +22.6]** |
=> On fresh data with real prices, ours beats every baseline with CIs excluding 0, except cost-from-success on Omni (+3.2, n.s.) --
exactly the pattern predicted by the mechanism (Omni length ~ difficulty; MMLU-Pro length = work, +14 pt). Pending on GPU (no API):
Intern-Decision success on fresh Omni/MMLU-Pro; MiniLM (CARROT kNN) and jina (MixLLM-style) embeddings for those baselines.
