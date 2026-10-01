# Intern-Decision-4B success pilot

User-authorized local open-weight comparator. Frozen same 300 problems as Jev (100 LCB, 100 Omni, 100 MMLU-Pro). Exact original Jev state, training aggregate accuracy priors, grading rules, route descriptions and five noul success questions; omit hosted model selector. No gold answers or generated answers in input. No test calibration or prompt changes.

Checkpoint: internlm/Intern-Decision-4B, revision 0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd. Use its published inference.py and DecisionEngine, default temperature 1.99241824. Single causal forward over decision placeholders, candidate-symbol softmax, published calibration; no generated explanations. All requests checked against default 8192-token limit, no truncation. Public download uses token=False and disabled implicit token to avoid expired ambient OAuth. Separate Python3.12 environment (upstream inference syntax requires3.12), upstream pinned torch2.9.1/torchvision0.24.1/transformers5.14.1. BF16 when supported else FP16, SDPA.

Primary: Intern success + our prefill costs vs our success + the same costs. Also compare Intern vs saved Jev with our costs fixed, both success alternatives under median costs, and learned-vs-median costs with Intern success fixed. Actual stored valid-draw outcome/cost means. Direct matched-accuracy savings over pair-specific shared convex-hull frontier bands, 1000 paired problem bootstraps seed0; all routes/draws clustered by problem. Expected-draw Brier/logloss point estimates. Exploratory on previously examined subsets, conditional on existing predictors; no claim of a separately validated deployment policy. Predictor overhead excluded from generation spending.

No OpenRouter calls, no target-model generations, no API key needed. One GPU job (32GB requested), persistent startup logs and resumable response file. Output /mnt/llmd/results/exps/aristides/reason/intern_decision_20261001. GPU scheduling/runtime depend on cluster availability. Code and protocol committed/pushed before snapshot launch.

Sources: https://huggingface.co/internlm/Intern-Decision-4B and its pinned inference.py / requirements.txt. No paper edits.
