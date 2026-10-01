# CARROT-KNN-SBERT comparison results

Completed locally on the original frozen splits with 1,000 paired problem bootstrap resamples. No API calls. See [protocol](../CARROT_PROTOCOL.md).

CARROT variant: the upstream local all-MiniLM-L12-v2 encoder and cosine uniform kNN regressors. This does not evaluate its OpenAI-embedding or fine-tuned RoBERTa variants.

Cells report **direct cost savings (%) of the left router relative to the right router at matched accuracy**, with conditional 95% bootstrap intervals. Each pair has its own shared accuracy band. These are not percentage-point differences between savings relative to median pricing.

| Comparison (left vs. right) | LCB | Omni | MMLU-Pro |
| --- | ---: | ---: | ---: |
| Our cost vs. CARROT cost; our success fixed | 33.6 [26.4, 40.3] | 14.0 [1.7, 24.2] | 14.4 [-0.3, 26.6] |
| Our full router vs. CARROT full router | 20.2 [11.0, 26.9] | 19.9 [6.3, 32.3] | 5.2 [-12.5, 20.5] |
| CARROT cost vs. mean-length constant; CARROT success fixed | 1.7 [-0.8, 5.0] | 9.7 [-1.8, 21.2] | 26.3 [9.3, 36.5] |
| Our cost vs. CARROT cost; CARROT success fixed | 15.5 [5.9, 22.8] | 10.7 [-0.2, 22.1] | 8.4 [-6.9, 21.2] |

The full-router advantage is positive with intervals excluding zero on LCB and Omni; MMLU-Pro is inconclusive. CARROT cost prediction improves over its own mean-length constant on MMLU-Pro, while the corresponding LCB and Omni intervals include zero.

Neighbor counts (success / output tokens): LCB 64 / 256; Omni 16 / 16; MMLU-Pro 64 / 32. Every contrast retained all 1,000 bootstrap samples.

Intervals condition on fitted predictors and omit training and configuration-selection uncertainty. Frontiers permit mixtures selected using evaluation outcomes. Common encoder overhead is excluded. New expansion problems are not included.

Artifacts (embeddings and predictions): `/mnt/llmd/results/exps/aristides/reason/carrot_compare_20261001/`. The JSON snapshots here preserve complete bootstrap samples, bands, k-selection scores, and diagnostics.

Execution: initial cluster jobs failed during startup without usable logs; verified local execution completed all three experiments. The runner now writes startup diagnostics and supports local execution.
