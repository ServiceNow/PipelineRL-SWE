# Threads in progress

**Living document — overwrite in place, do not append.** Snapshot: 2026-09-07 18:05 UTC.
Companion to `PAPER_OUTLINE.md` (the paper as it currently stands) and `RESEARCH_LOG.md`
(append-only history). This file answers one question: *what is running, why, and what does it
decide?*

---

## 1. Jobs in flight

Ten collection jobs, all OpenRouter, all on the LiveCodeBench pool. Nothing else is running.

| job | what it collects | progress | rate | ETA |
|---|---|---|---|---|
| `peer_qmax` (test) | qwen3-max on the 171 test problems | 85/171 | 4.9/min | **~0.3h** |
| `peer_deepseek` (test) | deepseek-v4-flash, test | 28/171 | 1.6/min | ~1.5h |
| `peer_kimi` (test) | kimi-k2.5, test | 16/171 | 0.9/min | ~2.8h |
| `peer_minimax` (test) | minimax-m2.7, test | 15/171 | 0.9/min | ~3.0h |
| `peer_glm5` (test) | glm-5, test | 7/171 | 0.4/min | **~6.8h** |
| `peer_qmax` (cal) | qwen3-max on the 170 calibration problems | 10/170 | 3.9/min | ~0.7h |
| `peer_deepseek` (cal) | deepseek, calibration | 10/170 | 3.9/min | ~0.7h |
| `peer_glm5` (cal) | glm-5, calibration | 4/170 | 1.6/min | ~1.8h |
| `peer_kimi` (cal) | kimi, calibration | 4/170 | 1.6/min | ~1.8h |
| `peer_minimax` (cal) | minimax, calibration | 2/170 | 0.8/min | ~3.6h |

**Everything is gated on glm-5 (~7h).** The other four finish within ~3.5h. Partial analysis is
possible on qmax + deepseek alone, which are also the two models with clean labels historically.

**Why both splits.** The response curve must be *fitted* on calibration and *evaluated* on test.
The first launch collected only the test split, which would have repeated the very defect these
runs exist to remove (fit and score on the same 171 problems).

**Watch for:** empty-output rate on glm-5 / kimi / minimax. The earlier screen measured 42% / 10% /
18% empties from answers written to the `reasoning` channel. The collector now recovers those
(`answer_from_reasoning`), and **verifying that the rate has collapsed is the first check when
these land** — if it has not, these labels are unusable and the transfer table stays motivation-only.

---

### Thread D — SWE-bench Verified (launched; the out-of-sample test of the scope law)

**Five Daytona grading jobs**, one per route, on the existing 5-route Verified collection
(4B scout / gpt-oss-20b / Qwen3-30B / gpt-oss-120b / Gemini, 369 eval problems). The generations
already existed; only real labels were missing, and `route_successes` in the parquet is the proxy
field our own notes forbid. Each run takes ~30 min based on the Opus precedent.

**Correcting an earlier recommendation.** I advised cutting SWE-bench because the Daytona harness
"returned all-`error` in 6/10 runs". That was true of older runs, but the most recent Verified run
completed cleanly at **319/369 resolved (86.4%)** for Opus 5. The harness works; the advice was
wrong.

**Why this is the highest-value benchmark left.** §6.9k's scope law says the advantage scales with
the fraction of problems *nothing* in the pool solves, measured *within* TACO. SWE-bench Verified
with this weaker pool should sit at a high unsolvable fraction — Opus reaches 86.4% but the scout
is at 13.6% — which makes it an **out-of-sample test of the law on a different domain**
(repo-level software engineering, not competitive programming). It also tests whether the
difficulty latent generalises past competitive programming at all.

**Kill criterion:** if the advantage does *not* rise with the unsolvable fraction here, the scope
law is a TACO artifact and §6.9k should be demoted from a law to an observation.

**RESULT (2026-09-07 22:50): blocked, for a third distinct reason.** Grading ran; three jobs
returned **0.0%, 0.0% and 0.5% resolved**. The harness is not at fault and neither are the labels:
**the route outputs are not unified diffs.** Route 0 and 2 are prose ("I'll analyze the bug in..."),
routes 1 and 4 are SEARCH/REPLACE edit blocks, route 3 is markdown. The Daytona evaluator expects
an applicable `model_patch`, so it is scoring prose. Opus graded fine because that collection
emitted diffs.

**So SWE-bench Verified needs an edit-applier or a re-collection in diff format, not grading.**
An applier exists in `pipelinerl/swe/agents/repair_agent.py` and `scripts/repair_eval_utils.py`
and could be reused, but applying SEARCH/REPLACE blocks against a repo is error-prone and silent
failures here look exactly like model failures -- which is how this project got a 10% oracle rate
on Verified once before.

**Recommendation: do not sink more time into this before the paper is written.** SWE-bench has now
failed three separate ways (harness errors, proxy labels, output format). The core result stands on
two benchmarks with a mechanism and a scope law; Verified is a nice-to-have out-of-sample check,
not a load-bearing claim. If it is attempted later, the clean path is a fresh collection that
*prompts for unified diffs*, not post-hoc conversion.

**If resumed, after grading:** extract scout activations on the 369 Verified problems (one GPU
job), build tensors, fit heads, run the frontier.

## 2. What these jobs decide

### Thread A — cross-model transfer (the novelty claim)
Fit the difficulty latent from scout / gpt-oss-20b / gpt-oss-120b only; add each peer with a
2-parameter response curve on N labels. Preliminary (n=50 internal split, 3 of 5 peers on corrupt
labels): 25 labels beat a dedicated 40,960-feature probe at 4 of 5 models. These runs replace that
with a clean fit-on-calibration / score-on-test table.
**Decides:** whether §3b-ii can be a primary table. **Kill criterion:** if the clean AUCs collapse
toward the own-probe baseline, the label-efficiency claim goes and the paper leans on abstention.

### Thread B — pool structure (the most interesting open lead)
The earlier screen found the peer pool has **+8.0pt union over its best member with 32% of problems
contested**, against our cascade pool's **+0.88pt at ten draws** — roughly 9x the routing headroom.
That reframes every routing result we have: our pool is *nested* (each route dominates the last),
so routing headroom is ~0 and abstention is the whole story; a *complementary* pool should behave
differently.
**Decides:** whether "a router's value is set by pool structure, and the regime is measurable
before deploying" is a real claim. **Prediction to falsify:** peer-pool routing should beat the
+12-13% cost-efficiency our cascade pool gives. If it does not, the pool-structure story is wrong
and routing is simply weak everywhere.

### Thread C — predicting *contested* problems (the method proposal)
The 32% of problems where pool members disagree is exactly where routing pays. Predicting which
problems are contested is a per-problem task — the routing analogue of abstention — and the probe
already does the abstention version well. **This is the strongest candidate for a method
contribution rather than an observation**, and it is cheap once Thread A's labels exist.

---

## 3. Not running, deliberately

| dropped | why |
|---|---|
| further cost-head work | 37x better cost $R^2$ moved the policy **zero** (§6.9o); oracle bounds the payoff at +13pt mid-targets, nothing near the ceiling |
| 128k token cap | draw lengths are a power law — each doubling halves survivors and makes spend *more* tail-dominated (top decile 56%→62% of all spend) |
| LoRA baseline | our own baseline, trained on stale labels; dropping it costs only the "26-42x more expensive and ties" line, which §5.5's offline bound covers |
| SWE-bench Verified | Daytona harness returned all-`error` in 6/10 historical runs |
| TACO cost prediction | difficulty→cost is hump-shaped there (§6.9n); a monotone difficulty signal cannot express it, and the fix improves the estimate without improving the policy |

---

## 4. Known-stale, must be fixed before submission

1. **A true sentence-transformer baseline is missing.** IrtNet predicts difficulty from 768-d
   sentence embeddings; we only have TF-IDF (0.7642/0.7204) and statement length (0.7110/0.6248)
   against activations (0.8629/0.8814). No sentence-transformer is cached here.
2. **TACO's price-ratio sweep** was never regenerated; only LiveCodeBench's was, and that moved the
   threshold from ~6x to ~10x.
3. **Confidence intervals on the two headline floors** (+6.28% LCB, +4.37% TACO). Currently point
   estimates over 3-5 seeds; the strict-improvement claim is the most exposed without a
   problem-clustered bootstrap.
4. **§6.8 and §6.9's truncation table are superseded**, not refreshed — rewrite as "the ablation
   predicted the real re-collection to a tenth of a point," which is the stronger result.
