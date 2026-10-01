# Held-out benchmark expansion — 2026-10-01

Fixed collection plan prepared before outcomes. Seed 20261001. No stopping on
significance and no tuning on the new evaluation outcomes.

| Pool | Existing problems | New held-out problems | Combined problems | Calls | Estimated API USD | With 25% buffer | Job spend guard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MMLU-Pro | 1,000 | 6,500 | 7,500 | 32,500 | 27.65 | 34.57 | 35 |
| Omni-MATH | 500 | 1,000 | 1,500 | 5,000 | 19.43 | 24.28 | 25 |
| Total | 1,500 | 7,500 | 9,000 | 37,500 | 47.08 | 58.85 | 60 |

All five routes retain the original settings and 64,000-token limit. Draws:
one draw per route (five calls per problem). Estimates use pooled
valid-draw prompt/completion token means from existing tensors, charging the
configured provider price ceilings; cache savings are not assumed. Price
ceilings (input/output USD per million): oss20 .04/.15; dsv4f .14/.28;
oss120 .15/.60. Cheaper eligible providers may reduce actual spending;
longer outputs, retries and distribution differences may increase it.
GPU prefill extraction and cluster CPU time are separate from API cost.

Samples target the existing sample's subject proportions (MMLU-Pro) and
rounded difficulty proportions (Omni). MMLU-Pro's larger sample exhausts
some small subjects, so capacity-aware redistribution changes sample counts;
`evaluation_stratum_weights` records the original proportions for a
pre-specified weighted comparison. Report the unweighted comparison too. All original IDs and normalized
problem-text overlaps are excluded. Further normalized text duplicates
within the source are also excluded. This does not establish semantic
independence of questions from shared sources.

- MMLU-Pro: cached full test split, 12,032 source rows. Exclude 1,058 rows
  overlapping existing IDs/text and 329 further duplicate texts; 10,645
  eligible rows before sampling. Reuse the cached source version and exact
  multiple-choice prompt formatter rather than silently changing revisions.
- Omni: full `KbsdJames/Omni-MATH`, revision
  `40ba231d8f16e29ecd40e6407e2c8640145a8f62`, 4,428 source rows. Exclude 503
  existing text overlaps and 18 further duplicate texts; 3,907 eligible.
  Original experiment is Omni-MATH-500; the expansion extends to the full
  benchmark, with matched rounded-difficulty proportions. It is not another
  sample drawn from the fixed 500-question benchmark.

`plan.json` records source provenance, exact hashes, strata and per-route
estimates. The two JSONL manifests include the complete prepared tasks;
collection does not reload or resample the source datasets.

## Evaluation protocol

All new examples are held-out evaluation, not training or calibration. Keep
existing train/calibration manifests, linear heads and calibration-selected
ZeroRouter configuration fixed for the primary expanded fixed-model
comparison. Collect correctness AND cost from each valid draw. Report the
fresh sample separately as well as the original-plus-new test set. Primary
sample sizes become 6,800 MMLU-Pro and 1,150 Omni evaluation problems when
combined with the original test sets, subject to collection validity.
This collects observations of frozen benchmark prompts; it does not imply
that the underlying LLMs never saw the public benchmarks during pretraining.

## Launch and resume

Commit and push first, then:

```bash
SUBMIT=1 bash launchers/abstention/launch_math_expansion.sh
```

Two CPU-only, resumable snapshot jobs; four CPUs/16 GB each, concurrency 48
per job. Output root defaults to
`/mnt/llmd/results/exps/aristides/reason/math_expand_20261001`.
Each dataset has append-only generation files, prepared problems,
`progress.json`, the frozen collection plan, and `COMPLETE.json` only after
all expected valid calls exist. A filesystem lock prevents duplicate
collectors. Complete calls are skipped on restart; failed calls are retried.

Observed spending uses response `usage.cost` when present, otherwise a
conservative calculation using token counts and provider price ceilings.
In-flight calls reserve the maximum configured token cost for four attempts.
Ambiguous network failures can incur unobserved charges, so the local spend
guard is not a guaranteed account-wide hard cap. The existing OpenRouter
key limit remains unchanged. A job stops explicitly if its guard or error
threshold prevents completion; it never silently reduces the planned sample.

The account check found $150.03 available under this key's $700 allowance at
preparation time. Account-wide credits are separate and may be shared; the
key allowance is the relevant restriction. The user requested a $60 total
collection target to preserve budget for later work. At the full $60 target,
roughly $90 of that last-checked key allowance would remain, assuming no
other consumption. No credential is saved here.

## Submission status

The original submission attempt was blocked by automatic approval review,
which requires explicit approval for paid API calls. Neither original nor
resized jobs has launched. The revised one-draw plan supersedes the original
4/3/3/2/2 plan and its $100 combined guards.
