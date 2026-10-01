# Fresh expanded fixed-policy evaluation (2026-10-01)

Readouts are fitted using the original training problems, with hyperparameters and calibration corrections selected on the original calibration problems. The calibration-selected value of correctness is held fixed. No expansion labels were used to fit or select models. The readouts are reconstructed from the original splits rather than loaded as serialized coefficient files; their archived predictions on the original examples are not byte-identical, so exact estimator-version alignment remains a caveat. On each new problem the frozen policy chose one route; paired bootstrap resampling clustered at the problem level. MMLU-Pro is reported both unweighted and with the predeclared subject-stratum weights; Omni-MATH uses the predeclared difficulty-stratum weights. Full counts, estimates, sensitivity points, and secondary frontiers are in `results.json`.

| Dataset | Fresh problems | Frozen V ($/correct) | Cost savings vs. median (95% CI) | Accuracy delta (percentage points, 95% CI) |
|---|---:|---:|---:|---:|
| MMLU-Pro, subject weighted | 6,500 | 1.00 | 13.0% [11.1, 14.9] | -0.29 [-0.58, -0.01] |
| MMLU-Pro, unweighted | 6,500 | 1.00 | 13.3% [11.3, 15.3] | -0.26 [-0.54, 0.02] |
| Omni-MATH, difficulty weighted | 1,000 | 1.00 | 9.2% [4.9, 13.9] | -0.20 [-0.50, 0.00] |

The original calibration-selected V was at the upper edge of its search grid. Therefore these results support the policy at that selected operating point; they do not establish a broad operating-range benefit. Secondary test-outcome-swept frontiers yield -33.4% savings for learned costs on MMLU-Pro and +14.2% on Omni-MATH, over a different and lower shared accuracy band. They allow mixtures selected using evaluation outcomes and are descriptive. Keep this distinction explicit in the paper.

Jobs: MMLU-Pro readout `10e154f3-2a26-4303-970f-76fdf006c364` and Omni-MATH readout `d1d97566-f4ef-44df-8bb4-d5654641b161` succeeded. Collection, corrected-prompt feature inference, tensor construction, and logs remain in `/mnt/llmd/results/exps/aristides/reason/expanded_eval_20261001/`.
