"""
Main pipeline orchestration for SWE rollouts.
Coordinates the execution of all stages and produces the final RolloutResult.
Now runs policy + optional expert model (for reward regression) without A2A/self-eval layers.
"""

import asyncio
import logging
from pathlib import Path
import time

from omegaconf import DictConfig
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.rollouts import RolloutResult
from pipelinerl.config_utils import get_performance_value_dim
from pipelinerl.swe.metrics import UnifiedMetrics
from pipelinerl.swe.utils.file_context_enricher import FileContextEnricher
from pipelinerl.swe.utils.localization_utils import parse_patch_for_gold_files
from .stages import run_localization, run_file_selection, run_repair

logger = logging.getLogger(__name__)


def _resolve_repo_base_path(cfg: DictConfig, dataset: str) -> str:
    swe_cfg = cfg.swe or {}
    repo_path_by_dataset = swe_cfg.get("repo_path_by_dataset")
    if repo_path_by_dataset:
        mapped = repo_path_by_dataset.get(dataset)
        if mapped:
            return mapped
    # Backward-compatible defaults.
    if dataset == "swegym":
        return swe_cfg.get("repo_path_train", "/mnt/llmd/data/swegym/repos")
    if dataset == "swebench_lite":
        return swe_cfg.get("repo_path_test", "/mnt/llmd/data/swebench_lite/repos")
    if swe_cfg.get("repo_path_default"):
        return swe_cfg.get("repo_path_default")
    if swe_cfg.get("repo_path_train"):
        return swe_cfg.get("repo_path_train")
    if swe_cfg.get("repo_path_test"):
        return swe_cfg.get("repo_path_test")
    raise ValueError(
        f"No repo path configured for dataset='{dataset}'. "
        "Set swe.repo_path_by_dataset[dataset] or swe.repo_path_default."
    )


async def _run_expert_stage_with_timing(
    stage_name: str,
    problem: dict,
    expert_llms: list[TrainableLLM],
    runner,
) -> tuple[list[tuple[int, TrainableLLM, dict]], dict]:
    """Run expert calls serially and log how much wall time they add."""
    expert_results: list[tuple[int, TrainableLLM, dict]] = []
    timings: list[dict] = []
    total_start = time.perf_counter()

    for expert_idx, expert_llm in enumerate(expert_llms):
        call_start = time.perf_counter()
        result = await runner(expert_llm)
        wall_time_s = time.perf_counter() - call_start
        timings.append(
            {
                "expert_rank": expert_idx,
                "model_name": getattr(expert_llm, "model_name", None),
                "wall_time_s": wall_time_s,
                "reported_latency_s": result.get("latency", 0.0),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
            }
        )
        expert_results.append((expert_idx, expert_llm, result))

    total_wait_s = time.perf_counter() - total_start
    if timings:
        timing_summary = ", ".join(
            f"rank={entry['expert_rank']} model={entry['model_name']} wall={entry['wall_time_s']:.2f}s "
            f"reported={entry['reported_latency_s']:.2f}s tok={entry['prompt_tokens']}/{entry['output_tokens']}"
            for entry in timings
        )
        problem_id = problem.get("id") or problem.get("problem_id") or problem.get("instance_id")
        logger.info(
            "%s expert timing: problem=%s total_wait=%.2fs [%s]",
            stage_name,
            problem_id,
            total_wait_s,
            timing_summary,
        )

    return expert_results, {"total_wait_s": total_wait_s, "per_expert": timings}


async def _run_repair_bundle_with_timing(
    cfg: DictConfig,
    llm: TrainableLLM,
    expert_llms: list[TrainableLLM],
    problem: dict,
    file_contents: dict,
    session,
) -> tuple[dict, list[tuple[int, TrainableLLM, dict]], dict]:
    """
    Run the policy repair plus all expert repairs concurrently.

    This removes the avoidable policy->expert and expert->expert serialization
    while preserving deterministic expert ordering in the returned results.
    """

    async def _timed_policy_call() -> dict:
        start = time.perf_counter()
        result = await run_repair(cfg, llm, problem, file_contents, session)
        return {
            "kind": "policy",
            "result": result,
            "wall_time_s": time.perf_counter() - start,
        }

    async def _timed_expert_call(expert_idx: int, expert_llm: TrainableLLM) -> dict:
        start = time.perf_counter()
        result = await run_repair(cfg, expert_llm, problem, file_contents, session, collect_training_text=False)
        return {
            "kind": "expert",
            "expert_idx": expert_idx,
            "expert_llm": expert_llm,
            "result": result,
            "wall_time_s": time.perf_counter() - start,
        }

    total_start = time.perf_counter()
    tasks = [_timed_policy_call(), *[_timed_expert_call(idx, expert_llm) for idx, expert_llm in enumerate(expert_llms)]]
    completed = await asyncio.gather(*tasks)
    total_wait_s = time.perf_counter() - total_start

    policy_entry = next(entry for entry in completed if entry["kind"] == "policy")
    expert_entries = [entry for entry in completed if entry["kind"] == "expert"]
    expert_entries.sort(key=lambda entry: int(entry["expert_idx"]))

    expert_results = [
        (entry["expert_idx"], entry["expert_llm"], entry["result"])
        for entry in expert_entries
    ]

    timing = {
        "total_wait_s": total_wait_s,
        "policy_wall_time_s": policy_entry["wall_time_s"],
        "policy_reported_latency_s": policy_entry["result"].get("latency", 0.0),
        "per_expert": [
            {
                "expert_rank": entry["expert_idx"],
                "model_name": getattr(entry["expert_llm"], "model_name", None),
                "wall_time_s": entry["wall_time_s"],
                "reported_latency_s": entry["result"].get("latency", 0.0),
                "prompt_tokens": entry["result"].get("prompt_tokens", 0),
                "output_tokens": entry["result"].get("output_tokens", 0),
            }
            for entry in expert_entries
        ],
    }

    if expert_entries:
        problem_id = problem.get("id") or problem.get("problem_id") or problem.get("instance_id")
        expert_summary = ", ".join(
            f"rank={entry['expert_idx']} model={getattr(entry['expert_llm'], 'model_name', None)} "
            f"wall={entry['wall_time_s']:.2f}s reported={entry['result'].get('latency', 0.0):.2f}s "
            f"tok={entry['result'].get('prompt_tokens', 0)}/{entry['result'].get('output_tokens', 0)}"
            for entry in expert_entries
        )
        logger.info(
            "Repair parallel timing: problem=%s total_wait=%.2fs policy_wall=%.2fs policy_reported=%.2fs [%s]",
            problem_id,
            total_wait_s,
            policy_entry["wall_time_s"],
            policy_entry["result"].get("latency", 0.0),
            expert_summary,
        )

    return policy_entry["result"], expert_results, timing


async def generate_unified_swe_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session,
    expert_llms: list[TrainableLLM] | None = None,
):
    """
    Generate complete SWE pipeline rollout using the streamlined architecture.
    Runs each stage once with the policy model; if an expert model is available, also runs
    the expert to collect an expert reward for auxiliary regression.
    """
    # Expect expert LLMs to be constructed by the caller (actor loop).
    if expert_llms is None:
        expert_llms = []
    if expert_llms:
        performance_value_dim = 1 + len(expert_llms)
    else:
        performance_value_dim = get_performance_value_dim(cfg)
    training_texts = []
    metrics = UnifiedMetrics()
    
    total_latency = 0.0
    all_prompt_tokens = []
    all_output_tokens = []
    
    enricher = FileContextEnricher()
    dataset = problem['dataset']
    
    base_repo_path = _resolve_repo_base_path(cfg, dataset)

    trace_cfg = cfg.swe.get("router_trace", {}) or {}
    trace_enabled = bool(trace_cfg.get("enabled", False))
    trace_include_outputs = bool(trace_cfg.get("include_outputs", True))
    trace_include_policy_token_ids = bool(trace_cfg.get("include_policy_token_ids", False))
    trace_include_file_contents = bool(trace_cfg.get("include_file_contents", False))
    trace_include_prompt_text = bool(
        trace_cfg.get("include_prompt_text", False) or trace_cfg.get("include_expert_prompt_text", False)
    )
    trace_include_output_text = bool(
        trace_cfg.get("include_output_text", False) or trace_cfg.get("include_expert_output_text", False)
    )
    trace_include_prompt_messages = bool(trace_cfg.get("include_prompt_messages", False))
    router_trace = None
    
    try:
        # Stage 1: Localization
        top_files = []
        if cfg.swe.get('enable_localization', False) or cfg.swe.get('run_localization', False):
            logger.info("Running localization stage")
            loc_result = await run_localization(cfg, llm, problem, session)
            if cfg.swe.get('enable_localization', False) and loc_result['training_text']:
                # Attach expert reward if available
                if expert_llms:
                    expert_loc_results, _ = await _run_expert_stage_with_timing(
                        "Localization",
                        problem,
                        expert_llms,
                        lambda expert_llm: run_localization(
                            cfg, expert_llm, problem, session, collect_training_text=False
                        ),
                    )
                    expert_rewards = [result.get('reward', 0.0) for _, _, result in expert_loc_results]
                    performance_targets = [loc_result.get('reward', 0.0), *expert_rewards]
                    if len(performance_targets) != performance_value_dim:
                        logger.warning(
                            "performance_value_dim=%d but got %d targets for localization",
                            performance_value_dim,
                            len(performance_targets),
                        )
                    loc_result['training_text'].performance_targets = performance_targets
                training_texts.append(loc_result['training_text'])
            
            # Update metrics (these are always final/post-enhancement metrics)
            loc_metrics = loc_result['metrics']
            metrics.localization_mrr = loc_metrics.get('mrr', 0.0)
            metrics.localization_ndcg = loc_metrics.get('ndcg', 0.0)
            metrics.localization_num_queries = loc_metrics.get('num_queries', 0)
            metrics.localization_format_penalty = loc_metrics.get('format_penalty', 0.0)
            metrics.localization_recall = loc_metrics.get('localization_recall', 0.0)
            
            total_latency += loc_result['latency']
            all_prompt_tokens.append(loc_result['prompt_tokens'])
            all_output_tokens.append(loc_result['output_tokens'])
            
            top_files = loc_result['top_files']
            
            if not top_files:
                logger.warning("Localization failed, using oracle files")
                top_files = parse_patch_for_gold_files(problem.get("patch", ""))[:10]
        else:
            logger.info("Skipping localization, using oracle files")
            top_files = parse_patch_for_gold_files(problem.get("patch", ""))[:10]
        
        # Stage 2: File Selection
        files_for_repair = []
        enriched_context = {}
        
        if cfg.swe.get('enable_file_selection', False) or cfg.swe.get('run_file_selection', False):
            logger.info("Running file selection stage")
            
            if top_files:
                repo_path = Path(base_repo_path) / Path(problem.get('repo', '').replace("/", "_"))
                base_commit = problem.get('base_commit', '')
                
                enriched_context = enricher.enrich_files_on_demand(top_files, repo_path, base_commit)
                
                if enriched_context:
                    logger.info("Using pure stage mode for file selection")
                    sel_result = await run_file_selection(cfg, llm, problem, enriched_context, session)
                    if cfg.swe.get('enable_file_selection', False) and sel_result['training_text']:
                        if expert_llms:
                            expert_sel_results, _ = await _run_expert_stage_with_timing(
                                "Selection",
                                problem,
                                expert_llms,
                                lambda expert_llm: run_file_selection(
                                    cfg, expert_llm, problem, enriched_context, session, collect_training_text=False
                                ),
                            )
                            expert_rewards = [result.get('reward', 0.0) for _, _, result in expert_sel_results]
                            performance_targets = [sel_result.get('reward', 0.0), *expert_rewards]
                            if len(performance_targets) != performance_value_dim:
                                logger.warning(
                                    "performance_value_dim=%d but got %d targets for selection",
                                    performance_value_dim,
                                    len(performance_targets),
                                )
                            sel_result['training_text'].performance_targets = performance_targets
                        training_texts.append(sel_result['training_text'])
                    
                    files_for_repair = sel_result.get('files_for_repair', [])
                    
                    # Update metrics (always final/post-enhancement)
                    sel_metrics = sel_result['metrics']
                    metrics.selection_precision = sel_metrics.get('selection_precision', 0.0)
                    metrics.selection_recall = sel_metrics.get('selection_recall', 0.0)
                    metrics.selection_f1 = sel_metrics.get('selection_f1', 0.0)
                    metrics.selection_format_penalty = sel_metrics.get('format_penalty', 0.0)

                    total_latency += sel_result['latency']
                    all_prompt_tokens.append(sel_result['prompt_tokens'])
                    all_output_tokens.append(sel_result['output_tokens'])
                else:
                    logger.warning("File enrichment failed, using top 3 files")
                    files_for_repair = top_files[:3]
            else:
                logger.warning("No top files for selection")
                files_for_repair = []
        else:
            logger.info("Skipping file selection, using top 3 files")
            files_for_repair = top_files[:3]
        
        if not files_for_repair:
            logger.warning("No files for repair, using all oracle files")
            files_for_repair = parse_patch_for_gold_files(problem.get("patch", ""))
        
        # Stage 3: Repair
        if (cfg.swe.get('enable_repair', False) or cfg.swe.get('run_repair', False)):
            logger.info("Running repair stage")
            
            if enriched_context:
                file_contents = {
                    filepath: enriched_context[filepath]['content'] 
                    for filepath in files_for_repair 
                    if filepath in enriched_context
                }
            else:
                original_contents = problem.get('file_contents', {})
                file_contents = {
                    filepath: original_contents[filepath]
                    for filepath in files_for_repair
                    if filepath in original_contents
                }

            if file_contents:
                logger.info("Using pure stage mode for repair")
                rep_result, expert_rep_results, expert_rep_timing = await _run_repair_bundle_with_timing(
                    cfg,
                    llm,
                    expert_llms,
                    problem,
                    file_contents,
                    session,
                )
                if cfg.swe.get('enable_repair', False) and rep_result['training_text']:
                    if expert_rep_results:
                        expert_rewards = [result.get('reward', 0.0) for _, _, result in expert_rep_results]
                        performance_targets = [rep_result.get('reward', 0.0), *expert_rewards]
                        if len(performance_targets) != performance_value_dim:
                            logger.warning(
                                "performance_value_dim=%d but got %d targets for repair",
                                performance_value_dim,
                                len(performance_targets),
                            )
                        rep_result['training_text'].performance_targets = performance_targets
                    training_texts.append(rep_result['training_text'])

                if trace_enabled:
                    policy_training_text = rep_result.get("training_text")
                    policy_trace = {
                        "model_name": getattr(llm, "model_name", None),
                        "base_url": getattr(llm, "base_url", None),
                        "reward": rep_result.get("reward", 0.0),
                        "success": rep_result.get("success", False),
                        "format_error": bool((rep_result.get("metrics") or {}).get("format_error", False)),
                        "semantic_failure": bool((rep_result.get("metrics") or {}).get("semantic_failure", False)),
                        "failure_type": (rep_result.get("metrics") or {}).get("failure_type"),
                        "no_edits": bool((rep_result.get("metrics") or {}).get("no_edits", False)),
                        "latency": rep_result.get("latency", 0.0),
                        "prompt_tokens": rep_result.get("prompt_tokens", 0),
                        "output_tokens": rep_result.get("output_tokens", 0),
                    }
                    if trace_include_outputs:
                        policy_trace["repair_output"] = rep_result.get("repair_output", "")
                    if trace_include_output_text:
                        policy_trace["output_text"] = rep_result.get("output_text")
                    if policy_training_text is not None:
                        policy_trace["prompt_text"] = policy_training_text.prompt_text
                        policy_trace["output_text"] = policy_training_text.output_text
                        if trace_include_policy_token_ids:
                            policy_trace["input_ids"] = policy_training_text.input_ids
                            policy_trace["labels"] = policy_training_text.labels
                    elif trace_include_prompt_text:
                        policy_trace["prompt_text"] = rep_result.get("prompt_text")
                    if trace_include_prompt_messages:
                        policy_trace["prompt_messages"] = rep_result.get("prompt_messages")

                    experts_trace = []
                    for expert_idx, expert_llm, expert_result in expert_rep_results:
                        expert_trace = {
                            "expert_rank": expert_idx,
                            "model_name": getattr(expert_llm, "model_name", None),
                            "base_url": getattr(expert_llm, "base_url", None),
                            "reward": expert_result.get("reward", 0.0),
                            "success": expert_result.get("success", False),
                            "format_error": bool((expert_result.get("metrics") or {}).get("format_error", False)),
                            "semantic_failure": bool((expert_result.get("metrics") or {}).get("semantic_failure", False)),
                            "failure_type": (expert_result.get("metrics") or {}).get("failure_type"),
                            "no_edits": bool((expert_result.get("metrics") or {}).get("no_edits", False)),
                            "latency": expert_result.get("latency", 0.0),
                            "wall_time_s": expert_rep_timing["per_expert"][expert_idx]["wall_time_s"]
                            if expert_idx < len(expert_rep_timing["per_expert"])
                            else None,
                            "prompt_tokens": expert_result.get("prompt_tokens", 0),
                            "output_tokens": expert_result.get("output_tokens", 0),
                        }
                        if trace_include_outputs:
                            expert_trace["repair_output"] = expert_result.get("repair_output", "")
                        if trace_include_prompt_text:
                            expert_trace["prompt_text"] = expert_result.get("prompt_text")
                        if trace_include_output_text:
                            expert_trace["output_text"] = expert_result.get("output_text")
                        if trace_include_prompt_messages:
                            expert_trace["prompt_messages"] = expert_result.get("prompt_messages")
                        experts_trace.append(expert_trace)

                    router_trace = {
                        "schema_version": 1,
                        "generated_at_unix": time.time(),
                        "dataset": problem.get("dataset"),
                        "problem_id": problem.get("id") or problem.get("problem_id") or problem.get("instance_id"),
                        "id": problem.get("id"),
                        "instance_id": problem.get("instance_id"),
                        "repo": problem.get("repo"),
                        "base_commit": problem.get("base_commit"),
                        "files_for_repair": files_for_repair,
                        "performance_targets": [
                            rep_result.get("reward", 0.0),
                            *[expert_result.get("reward", 0.0) for _, _, expert_result in expert_rep_results],
                        ],
                        "expert_timing": expert_rep_timing,
                        "policy_wall_time_s": expert_rep_timing.get("policy_wall_time_s"),
                        "policy": policy_trace,
                        "experts": experts_trace,
                    }
                    if trace_include_file_contents:
                        router_trace["file_contents"] = file_contents

                # Update metrics (always final/post-enhancement)
                rep_metrics = rep_result['metrics']
                metrics.repair_reward = rep_metrics.get('reward')
                metrics.repair_success = rep_metrics.get('success')
                metrics.repair_format_error = rep_metrics.get('format_error')

                total_latency += rep_result['latency']
                all_prompt_tokens.append(rep_result['prompt_tokens'])
                all_output_tokens.append(rep_result['output_tokens'])
            else:
                logger.error("No file contents available for repair")
                metrics.repair_reward = 0
                metrics.repair_format_error = True
        
        metrics.compute_derived_metrics()
        
        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=total_latency,
            router_trace=router_trace,
            dataset_name=problem.get("dataset"),
            prompt_tokens=all_prompt_tokens,
            output_tokens=all_output_tokens,
        )
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        
        failed_metrics = UnifiedMetrics()
        failed_metrics.reward = 0.0
        failed_metrics.success = False
        failed_metrics.no_error = False
        failed_metrics.file_pipeline_success = False
        failed_metrics.total_pipeline_success = False
        
        return RolloutResult(
            training_texts=training_texts,
            metrics=failed_metrics,
            latency=total_latency,
            router_trace=router_trace,
            dataset_name=problem.get("dataset"),
            prompt_tokens=all_prompt_tokens,
            output_tokens=all_output_tokens,
        )
