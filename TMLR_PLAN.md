# TMLR plan (written 2026-10-08)

Draft: `paper_tmlr/main.tex` (TMLR style; 4-pager text ported; TODO boxes per section). Target ~12 pages body.
Results log: `NEW_PATH.md` (pinned results from 4.A.59 on). All numbers: deepseek-v4-flash pinned to StreamLake, billed prices,
train/calibration/test only (no "fresh/original" wording). Shadow root `/mnt/llmd/results/exps/aristides/reason_pinned`
(scripts honour `REASON_ROOT`, outputs tagged `RESULT_TAG=_pinned`). Every job runs on eai, never locally.

## Storyline
*Per-query cost in routing reasoning models: when it matters, what it is, and how cheaply a prefill router gets it.*
Framing rules: difficulty bins and cost-from-success are OUR ablations (never "estimators we lose to"); external baselines are
median (prefill-router rule), mean, prompt GBM (cite latencyrouting 2607.18253), MixLLM-style, CARROT-style kNN, ZeroRouter.

## Claims, evidence, and what is missing
| # | Claim | Have | Missing |
|---|---|---|---|
| C1 | Per-query cost matters for reasoning models, much less for non-reasoning | headroom 13-60% on 8 pools (4.A.60/61); RouterBench 10.5% (confounded); effort ladder p90/p10 grows with effort | **non-reasoning pool on the same problems** (paid, running); effort-ladder headroom; agentic SWE-rebench (price gaps dominate) |
| C2 | Per-query cost is mostly a shared, problem-level difficulty property | cost-from-success ties on 6/7 pools (4.A.61) | variance decomposition pinned; level vs differences pinned; where the saving comes from (running); layer-wise (running) |
| C3 | A prefill router gets it almost free | second family (4.A.48 pinned), old 10-example onboarding | cost-from-success label learning curve; onboarding pinned |
| C4 | Where length tracks work, not difficulty, a dedicated readout pays | MMLU-Pro +24 over cost-from-success; old subject-vs-work analysis (4.A.28) | MMLU-Pro length drivers pinned; worked examples; SuperGPQA / BBEH only if the free screen says length separates from difficulty |
| C5 | Two failure modes: little headroom (BCB, chat) vs illegible headroom (CC, agents) | BCB 12.6% headroom; CC 34.5% headroom nobody reads (4.A.61) | headroom x capture map + readable-share per pool; CC deep-dive; SWE-Smith one-shot point |
| C6 | Deployment | provider pinning/drift (4.A.55-59) | deployable table (pinned numbers exist); per-query budgets pinned; price ladders (running); effort vs model (running); FrugalGPT-style + perfect-judge cascades on all 3 test sets (GPU); live run with policies at the pinned price (~$0.5) |
| C7 | Simpler than the prefill router at no loss | MLP heads = linear on our features (4.A.37); PCA+whitening hurts 4-12 pt (4.A.47) | **faithful reimplementation of their success pipeline** (layer search by 5-fold CV, last vs mean, PCA 50-300, concatenated per-target features, SharedTrunkNet 10 seeds -> top-5 average) vs our linear readouts, same splits, same cost readout |
| -- | Pre-registration record (appendix) | GAIN Omni, MMLU-Pro confirmed; APPS directional (pinned 15.1%); AIME NO GAIN falsified | write-up |

## Paid collections
1. **Non-reasoning pool (approved 2026-10-08, ~$15).** Same problems as the reasoning pool: LCB 892, Omni-MATH-500 (275/75/150) + 1,000 Omni test,
   MMLU-Pro 1,000 + the 2,000-problem test subset (provider_pilot_20261002/mmlupro/problem_ids.json). One draw per route. Routes, each pinned
   to ONE provider (recorded per call), sampling from the model card: deepseek-v4-flash thinking OFF (same settings as the thinking route
   otherwise, pinned StreamLake), llama-3.1-8b-instruct, qwen3-30b-a3b-instruct-2507, llama-3.3-70b-instruct, qwen3-235b-a22b-2507 (instruct),
   kimi-k2-0905. Smoke test (20 problems per model) before the full run; spend guard per job; resumable. Exact config in NEW_PATH 4.A.64.
2. **SuperGPQA / BIG-Bench Extra Hard (NOT approved, ~$15-25 each).** Decided by a free screen on existing screen data (oss20lo x1 + Instruct
   prefill): buy a dataset only if length separates from difficulty there (dedicated readout beats cost-from-success on the screen route).

## Free analyses (eai CPU unless noted)
Running: layer-wise readouts (4.A.62), where-the-saving-comes-from + label learning curve + price ladders + effort vs model (4.A.63),
APPS/AIME MixLLM rows (GPU).
Queued:
- F1 variance decomposition pinned (between-problem share, cross-model length correlation, failed vs solved) — why_decompose.py
- F2 level vs differences pinned — level_vs_relative.py
- F3 cost-from-success learning curve (n = 10..all) vs dedicated readout
- F4 onboarding new routes from 5-10 examples, pinned — onboard_full.py / onboard_vs_zerorouter.py
- F5 MMLU-Pro length drivers pinned — mmlupro_length_driver.py
- F6 headroom x capture map + readable share of length variance, all pools
- F7 CodeContests deep-dive: why its difficulty is illegible from the prompt
- F8 prefill-router success pipeline reimplementation (C7)
- F9 per-query budgets pinned — budget_cap.py
- F10 SWE: agentic SWE-rebench re-analysis (public data); SWE-Smith one-shot headroom
- F11 effort-ladder headroom (effort-only pools), feeds C1
- F12 deployable-policies table from pinned outputs (deploy_matched / billed_reprice pinned logs)
- G1 (GPU) FrugalGPT-style / perfect-judge single-submission cascades on LCB, Omni, MMLU-Pro, pinned

## Writing
Rewrite intro/abstract around C1-C5; fill sections as results land; appendix: pool statistics, baseline details, protocol, pre-registration.

## Status (2026-10-08 evening)
Done (NEW_PATH): full pinned estimator suite on all pools (4.A.61); layer-wise (4.A.62); where-the-saving-comes-from, label curve for
the dedicated readout, price ladders, effort vs model (4.A.63); screen separation: SuperGPQA +.61 and BBEH +.40 separate like MMLU-Pro (4.A.65).
Running: non-reasoning pool (math done; Omni/MMLU test + LCB in flight; 2 LCB routes retrying after rate limits) (4.A.64);
F1 variance decomposition (also covers F7 CC), F2 level vs differences, F5 MMLU-Pro drivers, F4 onboarding, F9 budgets (4.A.66);
F3 label efficiency dedicated vs cost-from-success (4.A.67); F8 prefill-router success reimplementation (4.A.68); G1 perfect-judge
cascades on all three test sets, CPU (4.A.69; learned-judge arms need judge scores on the pinned draws: GPU, later if wanted);
APPS/AIME MixLLM rows (GPU).
Awaiting approval: SuperGPQA + BBEH full pools (~$13, guards $8 + $12); kimi-k2 non-reasoning route (~$19).
F10 SWE: agentic SWE-rebench results (4.A.14/4.A.15) are public-data and provider-independent, reuse; SWE-Smith one-shot from existing data.
F6 headroom x capture map: no new compute (suite + headroom outputs); build at the figure stage.
