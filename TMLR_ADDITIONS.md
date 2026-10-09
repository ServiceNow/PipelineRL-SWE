# From the 4-pager to the TMLR paper: what was added (status 2026-10-09, 16:05 ET)

The 4-pager (NOWAI, `paper_nowai/`, submitted Oct 6) has: the main result on three test sets (ours vs median, mean, prompt GBM,
MixLLM-style, ZeroRouter; LCB / Omni-MATH / MMLU-Pro), the cost-from-success and difficulty-bin ablations, the representation x
readout grid, prefill size and families, two further routed families on MMLU-Pro, and endpoints as a side note.
Everything below is new in the TMLR work (`paper_tmlr/main.tex`; results log `NEW_PATH.md` 4.A.60-4.A.80). All numbers: deepseek-v4-flash
pinned to StreamLake, billed prices, train / calibration / test splits; "saving" = cost saved at matched accuracy vs median pricing.

## 1. Measurement: when per-query cost is worth predicting (C1)
- **Headroom and capture on nine pools, one protocol** (4.A.60/61/70). Perfect per-query cost knowledge saves 29-60% on seven pools;
  our readouts capture .47-.74 of it. LCB 45.3/.74, Omni 55.5/.63, MMLU-Pro 59.5/.49, AIME 29.3/.54, APPS 32.1/.47, BCB 12.6/.50,
  CC 34.5/.18, SuperGPQA 57.0/.48, BBEH 41.8/.63. New pools: AIME, APPS, BigCodeBench, CodeContests (pinned), SuperGPQA and
  BIG-Bench Extra Hard (collected for TMLR after a free screen predicted they would separate length from difficulty).
- **Reasoning vs non-reasoning routes on the same problems** (4.A.64/71). Five non-reasoning routes (deepseek thinking off,
  Llama-3.1-8B, Qwen3-30B-A3B-Instruct, Llama-3.3-70B, Qwen3-235B-A22B-Instruct). Headroom R / NR: LCB 45.3 / 28.9, Omni 55.5 / 40.7,
  MMLU-Pro 60.5 / 26.0; our saving R / NR: 33.7 / 19.7, 34.8 / 7.2, 30.6 / 1.3 (n.s.). Deepseek thinking on vs off: +3 / +15 / +22 points of
  accuracy for 11-33x the output.
  - Correction (4.A.71): non-reasoning length variation is two things. Llama-3.1-8B loops to the 16k cap on 9-12% of math (true
    degeneration, at its default sampling, one provider). Qwen3-30B-Instruct on LCB is NOT degenerate: told to output only code, it
    reasons in code comments until the cap. The largest non-reasoning headroom comes from a non-reasoning model that reasons anyway.
- **Price ladders** (4.A.63): with every route at the same per-token price, our readouts still save 13-36%; MMLU-Pro saves most at flat
  prices (length differs between questions, not between routes).
- **Effort vs model choice** (4.A.63): restricting to one model's efforts, or one effort per model, removes most of the saving.

## 2. Mechanism: what the cost readout captures (C2, C4)
- Length is mostly problem-level: prefill R2 .43-.83 by pool; the level shared across routes carries 87% of LCB's headroom (4.A.66).
- Success and cost are read from the same late, last-token layers; the two readouts are nearly one signal on LCB (corr -.90..-.97),
  partly on Omni, largely different on MMLU-Pro (4.A.62).
- Where length tracks work rather than difficulty, the dedicated readout pays over pricing from success: MMLU-Pro +24.1, BBEH +26.6,
  SuperGPQA +9.0 (4.A.66/70). MMLU-Pro: difficulty + subject explain R2 .34-.44 vs the prefill's .41-.66.
- Label efficiency of the dedicated readout vs cost-from-success (figure) (4.A.67).
- What changes in the routing: 31-66% of problems change route at mid-band; mostly deepseek <-> gpt-oss-20b-low (4.A.63).
- **Refined today (4.A.77/79): "cost is difficulty" is too strong.** The TRUE solve rate is a weak price even in-domain (LCB 21.5 vs 34.4
  for pricing from our success logits); the success readouts carry prefill length cues beyond difficulty. Across benchmarks, the
  same solve rate means very different lengths (see section 4). The C2 wording in section 5 still needs this revision.

## 3. When it fails (C5)
- **BigCodeBench**: little headroom (12.6%); no estimator significant.
- **CodeContests**: 34.5% headroom that no estimator reads (best captures ~1/4). ZeroRouter's 9.3% [1.1, 20.0] checked (4.A.80):
  over 27 configurations it averages 2.3% and beats ours in 30%; the calibration-chosen one sits near the top; its success model +
  our cost = ours. A favourable configuration, not an advantage.
- **Non-reasoning routes**: less headroom, mostly illegible (capture .18 Omni, .05 MMLU-Pro).
- **Capture map figure** with all nine pools and the three non-reasoning pools.

## 4. Generalisation: how cheaply a router gets it (new section `sec:general`; C3)
- **Few labels** (4.A.73): with 20 training problems ours keeps 57-75% of its full saving on four of five pools (LCB 24.7, Omni 22.8,
  MMLU-Pro 16.4, SuperGPQA 20.6); every external estimator saves < 11%. BBEH: text estimators catch up at >= 100 labels.
- **Onboarding a new model from k examples, vs ZeroRouter's onboarding (its headline use case)** (4.A.72/75). Whole models held out
  (no sibling effort left in the pool). k=10: ours 13.5-25.6% vs ZeroRouter 5.8-22.4%; ours - ZR +10.1 [1.4, 23.7] MMLU-Pro,
  +14.6 [3.0, 26.0] SuperGPQA, LCB +9.3 [3.0, 17.0] at k=50; Omni and BBEH ties. Gains concentrate on deepseek, the only model from a
  family not otherwise in the pool.
- **A whole router moved to a new benchmark, zero target labels** (4.A.74). Ours keeps 90-95% of its in-domain saving on LCB,
  MMLU-Pro, AIME (32.1 / 27.3 / 14.2); exceeds it on APPS; weaker on SuperGPQA 19.0, BBEH 7.3, Omni-500 8.1. The prefill router's own
  rule (median pricing) gets at most 2% on those four; ours beats every transferred external estimator by 7-24 points there.
- **Pooled training** (4.A.75): one readout trained on the target plus the other pools matches the per-benchmark readout within ~3 points
  on every pool. **Sources + 10 target labels** beat the 10 labels alone (MMLU-Pro 27.8 vs 8.1, AIME 13.1 vs 4.4, CC 7.4 vs -3.5).
- **Is ours genuinely more robust?** Mean change under cost-only transfer: ours -3.1, cost-from-success -12.9, ZeroRouter pricing -5.9;
  worst case -14.8 / -40.6 / -24.9. Ours never significantly worse; tied where difficulty transfers (APPS, CC, AIME); ours also
  degrades on BBEH / SuperGPQA. Difficulty-based pricing transfers UNEVENLY (holds on APPS / CC / AIME; collapses on LCB, SuperGPQA,
  BBEH, Omni-500), not uniformly badly: paper text corrected.
- **Why it transfers (4.A.77/79), each with a refuting outcome:**
  - Same solve rate, different benchmark: lengths differ by 1.13 log units on average (x3.1) over 470 matched cells; our readout trained
    on NEITHER benchmark predicts those gaps (corr .95, 90% of the squared gap explained); pricing from success 31%; difficulty-only 0
    by construction. **Strongest evidence.**
  - Solve-rate -> length slopes differ widely by benchmark (deepseek: LCB -4.3, AIME -4.1, BCB -0.5, SuperGPQA -0.8).
  - LCB without BigCodeBench (flat curve) in the sources: a one-feature difficulty map transfers almost fully (12.5 -> 32.5; in-domain
    34.7). BigCodeBench was what broke difficulty transfer to LCB.
  - A "work" oracle (realized cheap-route length) transfers far better than oracle difficulty (mean R2 drop .61 vs 2.35).
  - Our retention falls as the target looks less like its sources in prefill space (Spearman -.68, n 8).
  - Mixed: slope mismatch only weakly predicts difficulty-transfer error (+.32); leave-one-subject-out: difficulty holds within
    SuperGPQA, and ours fails leave-one-task-out on BBEH (6.8 vs 26.1).
  - Reading: difficulty is a benchmark-specific proxy for work (why a problem is hard differs by benchmark); the prefill readout reads
    cues of the work itself, which carry over when the sources contain the target's kinds of work.

## 5. Deployment
- **Perfect-judge cascades** (4.A.69): one-shot routing beats the best possible cascade by 29.7% (LCB), 31.9% (Omni), 34.2 (BBEH),
  34.4 (SuperGPQA); tie on MMLU-Pro.
- **Per-query budgets at billed prices** (4.A.74): LCB constant rule needs 1.26x our budget, median rule 1.34x.

## 6. Controls and simplicity (C7)
- **Faithful reimplementation of the prefill router's success pipeline** (4.A.68/70): ties our linear readouts on all five sets.
- **Fit timing** (4.A.74): ours ~3 min on 16 CPU threads vs ~7.4 min for their pipeline (48 configs x 5 folds per route + 10 nets);
  no extra encoder pass. ZeroRouter / MixLLM / GBM fit faster but need their own encoder and route worse.

## 7. Software engineering
- SWE-Smith one-shot (500, unpinned): headroom 14.2 n.s., ours 7.5 n.s. SWE-rebench agents: cost variance 74% model / 15% task;
  open-weight headroom 25.7 [6.3, 40.6]; issue-text cost R2 .29 on 3,387 tasks (4.A.14/15).

## 8. In progress (launched today; NEW_PATH 4.A.76 / 4.A.78)
| What | Why | Status (16:05 ET) | Cost |
|---|---|---|---|
| Qwen3-32B (SiliconFlow), GLM-4.7-flash (Novita), Nemotron-3-super-120B (DekaLLM) on LCB 892, Omni 1,350; Nemotron on MMLU-Pro 2,700 | cross-family onboarding beyond one model; transfer with unseen routes; "narrow pool" critique | Omni 32-127 / 1,350, LCB 67-153 / 892, MMLU-Pro 157 / 2,700; long reasoning outputs: roughly 10-30 h at concurrency 12 | ~$35 projected (guards $88) |
| SWE-Smith v2: 1,432 instances, deepseek pinned, Daytona labels | pinned SWE numbers, 3x test set | generation: oss20lo done, oss20md 1,234, dsv4f 620; oss120 pair pending; then labelling (~hours) | ~$5 (guards $10) |
| Llama-3.1-8B loop check: 300 loop + 150 control problems x {redraw, Novita, repetition penalty 1.1} | model vs provider vs sampling for the non-reasoning loops | collecting (521 rows) | < $1 |
| Qwen3-32B note | SiliconFlow stops at 24,575 tokens; ~half the Omni calls hit it (as in the MMLU-Pro rows) | route property | |

## 9. Draft status and what is left
- Written into the draft: abstract / intro around the three questions; non-reasoning section + table; new generalisation section with
  three tables; capture map with non-reasoning pools; billed budgets; fit timing; corrected transfer claims. Body ~13.5 pages
  (target ~12): needs a trim.
- To write: the C2 revision and the transfer mechanism (4.A.77/79) into the paper; new-family results; SWE-Smith v2; deployable
  policies table; endpoints; appendix (pool statistics, reimplemented baselines, protocol); nebius citation; final number pass.
- Open decisions: pre-registration section in or out; non-reasoning placement (currently in section 4); Table 1 scope (currently 3 sets;
  all 9 in the pools table).
