"""
Main pipeline orchestration for SWE rollouts.
Coordinates the execution of all stages and produces the final RolloutResult.
Now runs policy + optional expert model (for reward regression) without A2A/self-eval layers.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.rollouts import RolloutResult
from pipelinerl.config_utils import get_performance_value_dim
from pipelinerl.swe.metrics import UnifiedMetrics
from pipelinerl.swe.utils.file_context_enricher import FileContextEnricher
from pipelinerl.swe.utils.localization_utils import parse_patch_for_gold_files
from .stages import run_localization, run_file_selection, run_repair

logger = logging.getLogger(__name__)


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
    
    if dataset == 'swegym':
        base_repo_path = cfg.swe.get('repo_path_train', '/mnt/llmd/data/swegym/repos')
    elif dataset == 'swebench_lite':
        base_repo_path = cfg.swe.get('repo_path_test', '/mnt/llmd/data/swebench_lite/repos')
    else:
        base_repo_path = '/tmp'
    
    try:
        # Stage 1: Localization
        top_files = []
        if cfg.swe.get('enable_localization', False) or cfg.swe.get('run_localization', False):
            logger.info("Running localization stage")
            loc_result = await run_localization(cfg, llm, problem, session)
            if cfg.swe.get('enable_localization', False) and loc_result['training_text']:
                # Attach expert reward if available
                if expert_llms:
                    expert_rewards = []
                    for expert_llm in expert_llms:
                        expert_loc_result = await run_localization(
                            cfg, expert_llm, problem, session, collect_training_text=False
                        )
                        expert_rewards.append(expert_loc_result.get('reward', 0.0))
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
                            expert_rewards = []
                            for expert_llm in expert_llms:
                                expert_sel_result = await run_file_selection(
                                    cfg, expert_llm, problem, enriched_context, session, collect_training_text=False
                                )
                                expert_rewards.append(expert_sel_result.get('reward', 0.0))
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
                rep_result = await run_repair(cfg, llm, problem, file_contents, session)
                if cfg.swe.get('enable_repair', False) and rep_result['training_text']:
                    if expert_llms:
                        expert_rewards = []
                        for expert_llm in expert_llms:
                            expert_rep_result = await run_repair(
                                cfg, expert_llm, problem, file_contents, session, collect_training_text=False
                            )
                            expert_rewards.append(expert_rep_result.get('reward', 0.0))
                        performance_targets = [rep_result.get('reward', 0.0), *expert_rewards]
                        if len(performance_targets) != performance_value_dim:
                            logger.warning(
                                "performance_value_dim=%d but got %d targets for repair",
                                performance_value_dim,
                                len(performance_targets),
                            )
                        rep_result['training_text'].performance_targets = performance_targets
                    training_texts.append(rep_result['training_text'])

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
            dataset_name=problem.get("dataset"),
            prompt_tokens=all_prompt_tokens,
            output_tokens=all_output_tokens,
        )
