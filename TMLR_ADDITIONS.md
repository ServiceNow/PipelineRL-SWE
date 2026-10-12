# From the 4-pager to the TMLR paper: what was added (status 2026-10-12, 02:45 ET)

The 4-pager (NOWAI, `paper_nowai/`, submitted Oct 6) has: the main result on three test sets (ours vs median, mean, prompt GBM,
MixLLM-style, ZeroRouter; LCB / Omni-MATH / MMLU-Pro), the cost-from-success and difficulty-bin ablations, the representation x
readout grid, prefill size and families, two further routed families on MMLU-Pro, and endpoints as a side note.
Everything below is new in the TMLR work (`paper_tmlr/main.tex`; results log `NEW_PATH.md` 4.A.60-4.A.85). All numbers: deepseek-v4-flash
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
- **Refined (4.A.77/79/81; now in the paper, sec:mechanism "The success readouts carry more than difficulty"): "cost is difficulty" is too strong.** The TRUE solve rate is a weak price even in-domain (LCB 21.5 vs 34.4
  for pricing from our success logits); the success readouts carry prefill length cues beyond difficulty. Across benchmarks, the
  same solve rate means very different lengths (see section 4). Residual beyond true solve rates matches realized residual at .56-.92 on six pools and predicts five held-out
  non-reasoning models' length (MMLU-Pro partial r .40-.71) but not strong models' success (|r| <= .05).

## 3. When it fails (C5)
- **BigCodeBench**: little headroom (12.6%); no estimator significant.
- **CodeContests**: 34.5% headroom that no estimator reads (best captures ~1/4). ZeroRouter's 9.3% [1.1, 20.0] checked (4.A.80):
  over 27 configurations it averages 2.3% and beats ours in 30%; the calibration-chosen one sits near the top; its success model +
  our cost = ours. A favourable configuration, not an advantage.
- **Non-reasoning routes**: less headroom, mostly illegible (capture .18 Omni, .05 MMLU-Pro).
- **Capture map figure** with all nine pools and the three non-reasoning pools.

## 4. Generalisation: how cheaply a router gets it (new section `sec:general`; C3)
- **Few labels** (4.A.73/84): with 20 training problems ours keeps 51-75% of its full saving on all five pools (LCB 24.7, Omni 22.8,
  MMLU-Pro 16.4, SuperGPQA 20.5, BBEH 14.7); every external estimator saves < 11%. BBEH: text estimators catch up at >= 100 labels.
- **Onboarding a new model from k examples, vs ZeroRouter's onboarding (its headline use case)** (4.A.72/75). Whole models held out
  (no sibling effort left in the pool). k=10: ours 17.1-25.6% vs ZeroRouter 4.4-22.4%; ours - ZR +10.1 [1.4, 23.7] MMLU-Pro,
  +15.7 [0.4, 31.5] SuperGPQA, LCB +9.3 [3.0, 17.0] at k=50; Omni and BBEH ties. Gains concentrate on deepseek, the only model from a
  family not otherwise in the pool.
- **A whole router moved to a new benchmark, zero target labels** (4.A.74). Ours keeps 90-95% of its in-domain saving on LCB,
  MMLU-Pro, AIME (32.1 / 27.3 / 14.1); exceeds it on APPS; weaker on SuperGPQA 19.3, BBEH 8.6, Omni-500 7.8 (still ahead of
  every transferred external, which save <= 4.0 there). The prefill router's own
  rule (median pricing) gets at most 2% on those four; ours beats every transferred external estimator by 7-24 points there.
- **Pooled training** (4.A.75): one readout trained on the target plus the other pools matches the per-benchmark readout within ~3 points
  on every pool (within 4.4, n.s.). **Sources + 10 target labels** beat the 10 labels alone (MMLU-Pro 27.9 vs 8.1, AIME 13.1 vs 4.4, CC 7.4 vs -3.5).
- **Is ours genuinely more robust?** (4.A.82/84; in the paper as "Which estimators lose least") Ours' drop under cost-only transfer is
  significantly smaller than cost-from-success's on LCB (+37.1), SuperGPQA (+31.8) and BBEH (+13.4), and than ZeroRouter pricing's on LCB
  (+21.0) and MMLU-Pro (+4.0); never significantly larger (the earlier SuperGPQA vs ZR -17.4 was pre-refill data; now +3.2 n.s.). Ours
  also degrades on BBEH / SuperGPQA (12.5 / 9.8). Difficulty-based pricing transfers UNEVENLY (holds on APPS / CC / AIME; collapses on LCB, SuperGPQA,
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
  - In the paper (end of "Why the cost readout transfers"): matched gaps and the BigCodeBench curve.
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
- **SWE-Smith one-shot v2 (4.A.83/84; 1,432 instances, deepseek pinned, Daytona labels, test 389): positive.** Headroom 25.2 [15.0, 31.1];
  ours 14.5 [5.8, 19.8] (capture .58); beats mean +7.8, cost-from-success +9.2, MixLLM-style +6.8 [2.5, 10.6], prompt GBM +18.2; ties
  bins / ZeroRouter. Success AUC .80-.83. (Replaces the unpinned 500-instance n.s. result.) SWE-rebench agents: cost variance 74% model / 15% task;
  open-weight headroom 25.7 [6.3, 40.6]; issue-text cost R2 .29 on 3,387 tasks (4.A.14/15).

## 8. Three further model families (4.A.76/85; collected Oct 9-12, ~$66 incl. SWE-Smith v2)
- Qwen3-32B (SiliconFlow; 24,575-token stop), GLM-4.7-Flash (Novita), Nemotron-3-Super-120B (DekaLLM; Omni 27% invalid after a refill
  that was stopped because the provider kept erroring) on LCB, Omni-MATH, MMLU-Pro (2,700 subset). New readouts AUC .72-.87, R2 .23-.81.
- 8 routes: ours 34.8 / 33.0 / 28.6; beats bins on Omni (+8.7) and MMLU-Pro (+21.3) and cost-from-success on MMLU-Pro (+10.4); ties on LCB.
- Cheap pool (2 gpt-oss-20b + 3 new): only Nemotron is needed. Onboarded from 10 problems: ours 26.0 (LCB) / 26.0 (MMLU-Pro) vs
  ZeroRouter -30.2 / 10.8; k=50 ours - ZR +34.9 [18.0, 61.3] / +18.6 [4.7, 35.6]; Omni n.s. Naive onboarding loses money in 7/9
  model-pool pairs, ZeroRouter 2/9, ours 0/9.
- New readouts trained on Omni-MATH work on MMLU-Pro (28.5 vs 28.6).
- Llama-3.1-8B loop check (4.A.78): loops are a model property; repetition penalty removes most at some accuracy cost.

## 9. Draft status and what is left
- Written into the draft: abstract / intro around the three questions; non-reasoning section + table; generalisation section with
  three tables, robustness paragraph, new families; C2 revision and transfer mechanism; SWE-Smith v2 (positive); capture map with
  non-reasoning pools; billed budgets; fit timing. Body ~14 pages (refs start p.15; target ~12): needs a trim.
- To write: deployable policies table; endpoints; appendix (pool statistics, reimplemented baselines, protocol); nebius citation;
  final number pass.
- Open decisions: pre-registration section in or out; non-reasoning placement (currently in section 4); Table 1 scope (currently 3 sets;
  all 9 in the pools table).
