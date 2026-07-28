# Analysis Scripts

Scripts for the post-scout routing / cascade abstention paper.
All scripts run with `/home/toolkit/.conda/envs/pipeline-rl/bin/python`.

## Common data paths

| Variable | Path |
|---|---|
| Eval parquet (217 instances) | `/mnt/llmd/results/.../offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/eval/` |
| PS predictions | `...offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_.../eval_predictions.jsonl` |
| IO predictions | `...input_only_10epoch_.../eval_predictions.jsonl` |
| Claude eval results | `logs/run_evaluation/claude_sonnet_eval286_daytona/report.json` |

## Active scripts

### Core routing / abstention Pareto

| Script | What it does | Key output |
|---|---|---|
| `routing_pareto.py` | Routing Pareto frontier sweeping λ (cost weight). PS vs IO vs oracle. | `routing_pareto.png` |
| `routing_abstention_unified.py` | 2D sweep over (λ, abstention rate α). Shows joint routing + abstention frontier. | `unified_pareto.png`, `utility_vs_abstention.png` |
| `abstention_baselines.py` | Multi-signal abstention comparison: PS, IO, prompt-length heuristic. Includes AUC table. | `abstention_baselines.png` |

### Claude escalation experiments

| Script | What it does | Key output |
|---|---|---|
| `abstention_claude_escalation.py` | What if abstained tasks go to Claude instead of being dropped? PS and IO escalation vs random mix. Derives exact K=3.18× from API pricing. | `escalation_claude_pareto.png`, `escalation_claude_quality.png` |
| `routing_headroom_vs_gap.py` | 5-model analysis: capability gap vs mixed-outcome fraction vs correlation. Shows routing headroom scales with capability gap. | `routing_headroom_vs_gap.png` |

### Alternative routing signals

| Script | What it does | Key output |
|---|---|---|
| `entropy_abstention.py` | **Entropy baseline.** Uses teacher-forcing to compute exact per-token entropy of saved 4B outputs (no re-generation needed). Compares entropy signal vs PS/IO as abstention criterion. | `entropy_logprobs.jsonl`, `entropy_abstention.png` |
| `knn_retrieval_abstention.py` | Exp B: kNN retrieval abstention — embed tasks with Qwen3-8B (base and LoRA), score by mean success rate of k nearest training neighbours. | `knn_retrieval_results.json` |
| `difficulty_band_routing.py` | Exp E: 3-zone difficulty-band routing (easy/medium/hard). Tests whether a discrete zone policy beats continuous λ sweep. | `difficulty_band_pareto.png`, `difficulty_band_detail.png` |
| `routing_headroom_vs_gap.py` | 5-model analysis: capability gap × (1 − correlation) = routing headroom. | `routing_headroom_vs_gap.png` |

### Diversity sweep (see also `pipelinerl/swe/scripts/openrouter_sweep/`)

| Script | What it does | Key output |
|---|---|---|
| `openrouter_sweep/analyze_openrouter_sweep.py` | Load Daytona results for 15 OpenRouter models, compute phi-correlation + mixed-outcome fractions. | `phi_correlation_matrix.png`, `routing_headroom_matrix.png` |

### Paper figures

| Script | What it does |
|---|---|
| `generate_paper_figures.py` | Assembles publication-quality figures for Overleaf. Reads from `overleaf/figures/`. |

## Archive

`archive/` contains scripts from earlier experiment phases (Q-policy era, multirollout era,
2-route BCE analysis). They still run but the problems they address have been superseded.
See `archive/README.md`.
