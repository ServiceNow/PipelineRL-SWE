# Archive

Scripts from earlier experiment phases. Kept for reference; not actively maintained.

| Script | Era | What it was doing |
|---|---|---|
| `calibrate_q_bare_direct.py` | Q-policy | Calibrate Q success probabilities from bare-state predictions, no retraining |
| `analyze_q_budget_frontier.py` | Q-policy | Budget-cap frontiers for Q-policy runs |
| `analyze_q_spend_quality.py` | Q-policy | Whether Q router spends budget on better instances than random |
| `diagnose_cost_oracle_gap.py` | Q-policy | How much Q routing loses to imperfect per-instance cost estimates |
| `no_retrain_out_rescore.py` | Q-policy | OUT/abstention rescoring using Q predictions without retraining |
| `analyze_oss20_vs_oss120_bce.py` | 2-route BCE | Comparative analysis of a 2-route (OSS-20B vs OSS-120B) reward-BCE predictor |
| `abstention_sample_efficiency.py` | Multirollout | Abstention AUC as a function of real-label training set size (N=50–1000) |
| `abstention_plus_bon.py` | Multirollout | Combined abstention + best-of-N on 150-task multirollout eval set |
| `compare_state_policy_to_direct_random.py` | State policy | State-policy utility vs direct-random budget baseline |
| `check_anthropic_key.py` | Utility | One-shot check that the Anthropic API key works |
