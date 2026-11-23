# Provenance map (working notes for the code move)

Source: PipelineRL-SWE branch `offline_router_training`, paper at commit 2840cae.
Paths below are relative to `analysis/cost_headroom/` in that repo unless stated.
Stored results live under `/mnt/llmd/results/exps/aristides/reason_pinned` (data) and
`/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs` (per-job outputs).

## Tables and figures -> summary file -> aggregator -> job script

| Paper item | Summary file | Aggregator | Job script |
|---|---|---|---|
| tab:allest, tab:main, tab:pools | draws_summary_pinned.json (`draws`) | draws_aggregate.py | tmlr_strengthen.py (draws) |
| tab:allest (Route-To-Reason, SCX rows) | extra_estimators_draws_summary_pinned.json | extra_estimators_draws_aggregate.py | extra_estimators_draws.py |
| tab:labels, fig:labels | labels_samen_summary_pinned.json (+ draws_summary for the last row) | labels_samen_aggregate.py | cost_generalization.py (SAME_N_MEDIAN=1) |
| tab:onboard | draws_summary_pinned.json (`onboard`) | draws_aggregate.py | tmlr_strengthen.py |
| tab:subsets | drop_summary_pinned.json | drop_aggregate.py | tmlr_strengthen.py (drop) |
| tab:textemb | textemb_summary_pinned.json | textemb_aggregate.py | tmlr_strengthen.py (textemb) |
| tab:transfer, tab:nonreason, tab:deploy | item6_summary_pinned.json (`transfer`, `nonreason`, `deploy`) | item6_aggregate.py | cost_generalization.py (fulltransfer), nonreason_compare.py, newfamily_eval.py |
| tab:nonreason (estimator columns) | nonreason_estimators_summary_pinned.json | prefill_nonreason_aggregate.py | nonreason_estimators.py |
| tab:mixed | mixed2_summary_pinned.json + mixllm_mixed_summary_pinned.json | **mixed2: writer not saved** (from mixed_pool.py outputs); mixllm_mixed_aggregate.py | mixed_pool.py, mixllm_mixed.py |
| tab:scale | prefill_scale_summary_pinned.json | prefill_scale_aggregate.py | prefill_draws.py |
| tab:size | prefill_draws_summary_pinned.json | prefill_nonreason_aggregate.py | prefill_draws.py |
| tab:bracket | best_estimators_summary_pinned.json | best_aggregate.py | best_estimators.py |
| tab:routes | route_alone_pinned.json | **writer not saved** | n/a |
| tab:resplit | reason_pinned_logs/tmlr_strengthen_resplit_*_pinned.json | **no aggregator saved** | tmlr_strengthen.py (resplit) |
| tab:poolstats | none (recomputed from tensors in the Oct 18 audit) | **no script saved** | n/a |
| fig:hero | route_shares, route_snapshot, draws_summary, per_accuracy_savings (_pinned) | route_shares.py, route_snapshot.py, per_accuracy_savings.py | paper_tmlr/make_hero_fig.py |
| fig:overview | paper_nowai/data/fresh_billed_curves_pinned.json, frontier_curves, zr_faithful_draws_summary, draws_summary, single_routes (_pinned) | frontier_curves.py, zr_faithful_draws_aggregate.py, single_routes.py | paper_tmlr/make_headline_fig.py |
| fig:shares | route_shares_pinned.json | route_shares.py | paper_tmlr/make_shares_fig.py |

Matching method: each table row's numbers were matched against stored result fields across datasets,
in order (exact to the printed decimal). Rows that mix quantities were matched by coverage of a
single small file. A few numbers per table are derived (ratios, means) and appear in no file.

## Code closure

35 modules in `analysis/cost_headroom/` (about 3,500 lines), imports only numpy, scipy, scikit-learn,
torch, transformers, sentence-transformers and the standard library:

baseline_cost_heads best_aggregate best_estimators billed carrot_compare cost_generalization decompose
draws_aggregate drop_aggregate extra_estimators_draws extra_estimators_draws_aggregate frontier_curves
item6_aggregate labels_samen_aggregate mixed_aggregate mixed_pool mixllm_mixed mixllm_mixed_aggregate
newfamily_eval nonreason_compare nonreason_estimators per_accuracy_savings prefill_draws
prefill_nonreason_aggregate prefill_scale_aggregate route_shares route_snapshot single_routes
textemb_aggregate tmlr_free_analyses tmlr_strengthen zr_dimsweep zr_encoder zr_faithful_draws
zr_faithful_draws_aggregate

Upstream, also standalone (no PipelineRL imports):
- `pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase extract` (scout activations, GPU)
- `pipelinerl/swe/scripts/livecodebench/activation_content_preds.py` (success probes on the paper split)

Data-table builders (raw collections -> tensors.npz), not yet traced in detail:
`build_pinned_root.py`, `build_math_tensors.py`, `pipelinerl/swe/scripts/livecodebench/build_pool_v2_tensors.py`,
`pipelinerl/swe/scripts/bigcodebench/build_bcb_tensors.py`.

## Data the code reads (per dataset, under the pinned root)

- `<pool>_tensors/tensors.npz`: per-call numbers (outcome, validity, prompt and completion tokens)
- `<pool>_tensors/split_manifest.json`: train / calibration / test problem IDs
- scout activation file (`instruct.npz` / `scout.npz` / `scout_prefill.npz` / `prefill_combined.npz`, location varies by dataset)
- `content_preds.jsonl` / `success_preds.jsonl`: stored success predictions (regenerable)
- `emb_jina_code.npy`, `qwen3emb4b.npz`, `text_embeddings.npz`: text embeddings for baselines (regenerable from text)
- `problems.jsonl`: **problem text**, read by the loader for the prompt-feature GBM. Not releasable as is.
- Omni-MATH and MMLU-Pro: `expanded_eval_20261001/<ds>/` and billed per-call rows in `math_expand_20261001/<ds>/*_d0.jsonl`

## Open issues

1. Three paper items have no saved code: tab:routes (route_alone_pinned.json), the mixed2 part of tab:mixed, tab:poolstats; plus no aggregator for tab:resplit.
2. The loader reads problem text. Either fetch text through a benchmark loader, or ship the computed prompt features and embeddings instead.
3. Data is spread over many folders on /mnt; it needs one export step into a clean `data/<dataset>/` layout.
4. Reruns since Oct 19 differ slightly from the Oct 13 draws behind the tables (dataset means by up to .27 points; NEW_PATH 4.A.134). The released code may not reproduce every table digit for digit.
