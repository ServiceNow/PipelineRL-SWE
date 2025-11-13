"""
Main pipeline orchestration for SWE rollouts.
Coordinates the execution of all stages and produces the final RolloutResult.
Uses the layered architecture to select appropriate execution mode.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.rollouts import RolloutResult
from pipelinerl.swe.metrics import UnifiedMetrics
from pipelinerl.swe.utils.file_context_enricher import FileContextEnricher
from pipelinerl.swe.utils.localization_utils import parse_patch_for_gold_files
from .stages import run_localization, run_file_selection, run_repair
from .self_evaluation import run_localization_with_self_eval, run_file_selection_with_self_eval, run_repair_with_self_eval

logger = logging.getLogger(__name__)


async def generate_unified_swe_rollout(cfg: DictConfig, llm: TrainableLLM, problem: dict, session):
    """
    Generate complete SWE pipeline rollout using the layered architecture.
    Selects appropriate execution mode based on configuration.
    
    Args:
        cfg: Configuration
        llm: Main language model
        problem: Problem dictionary
        session: HTTP session
    """
    training_texts = []
    metrics = UnifiedMetrics()
    
    # Set abstention threshold from config (for statistics only)
    abstention_threshold = getattr(cfg.swe, 'abstention_threshold', None)
    if abstention_threshold is not None:
        metrics.set_abstention_threshold(abstention_threshold)
    
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
    
    def should_use_self_eval_for_stage(stage_name):
        return cfg.swe.get(f'enable_{stage_name}_self_eval', False)
    
    try:
        # Stage 1: Localization
        top_files = []
        if cfg.swe.get('enable_localization', False) or cfg.swe.get('run_localization', False):
            logger.info("Running localization stage")
            
            if should_use_self_eval_for_stage('localization'):
                logger.info("Using self-eval mode for localization")
                loc_result = await run_localization_with_self_eval(cfg, llm, problem, session)
                if loc_result['training_text']:
                    training_texts.append(loc_result['training_text'])
                if loc_result.get('self_eval_result') and loc_result['self_eval_result'].get('training_text'):
                    training_texts.append(loc_result['self_eval_result']['training_text'])
            else:
                logger.info("Using pure stage mode for localization")
                loc_result = await run_localization(cfg, llm, problem, session)
                if cfg.swe.get('enable_localization', False) and loc_result['training_text']:
                    training_texts.append(loc_result['training_text'])
            
            # Update metrics (these are always final/post-enhancement metrics)
            loc_metrics = loc_result['metrics']
            metrics.localization_mrr = loc_metrics.get('mrr', 0.0)
            metrics.localization_ndcg = loc_metrics.get('ndcg', 0.0)
            metrics.localization_num_queries = loc_metrics.get('num_queries', 0)
            metrics.localization_format_penalty = loc_metrics.get('format_penalty', 0.0)
            metrics.localization_recall = loc_metrics.get('localization_recall', 0.0)

            metrics.localization_prompt_tokens = loc_result.get('prompt_tokens', 0)
            metrics.localization_output_tokens = loc_result.get('output_tokens', 0)
            
            # Add self-eval metrics if available
            if loc_result.get('self_eval_result'):
                se_metrics = loc_result['self_eval_result']['metrics']
                metrics.localization_self_eval_predicted_score = se_metrics.get('predicted_score', 0.0)
                metrics.localization_self_eval_prediction_error = se_metrics.get('prediction_error', 1.0)
                metrics.localization_self_eval_parsing_error = se_metrics.get('parsing_error', True)
                metrics.localization_self_eval_prompt_tokens = loc_result['self_eval_result'].get('prompt_tokens', 0)
                metrics.localization_self_eval_output_tokens = loc_result['self_eval_result'].get('output_tokens', 0)
            
            total_latency += loc_result['latency']
            all_prompt_tokens.append(loc_result.get('prompt_tokens', 0))
            all_output_tokens.append(loc_result.get('output_tokens', 0))
            
            # Add self-eval tokens if available
            if loc_result.get('self_eval_result'):
                total_latency += loc_result['self_eval_result']['latency']
                all_prompt_tokens.append(loc_result['self_eval_result'].get('prompt_tokens', 0))
                all_output_tokens.append(loc_result['self_eval_result'].get('output_tokens', 0))
            
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
                    if should_use_self_eval_for_stage('file_selection'):
                        logger.info("Using self-eval mode for file selection")
                        sel_result = await run_file_selection_with_self_eval(cfg, llm, problem, enriched_context, session)
                        if sel_result['training_text']:
                            training_texts.append(sel_result['training_text'])
                        if sel_result.get('self_eval_result') and sel_result['self_eval_result'].get('training_text'):
                            training_texts.append(sel_result['self_eval_result']['training_text'])
                    else:
                        logger.info("Using pure stage mode for file selection")
                        sel_result = await run_file_selection(cfg, llm, problem, enriched_context, session)
                        if cfg.swe.get('enable_file_selection', False) and sel_result['training_text']:
                            training_texts.append(sel_result['training_text'])
                    
                    files_for_repair = sel_result.get('files_for_repair', [])
                    
                    # Update metrics (always final/post-enhancement)
                    sel_metrics = sel_result['metrics']
                    metrics.selection_precision = sel_metrics.get('selection_precision', 0.0)
                    metrics.selection_recall = sel_metrics.get('selection_recall', 0.0)
                    metrics.selection_f1 = sel_metrics.get('selection_f1', 0.0)
                    metrics.selection_format_penalty = sel_metrics.get('format_penalty', 0.0)

                    metrics.selection_prompt_tokens = sel_result.get('prompt_tokens', 0)
                    metrics.selection_output_tokens = sel_result.get('output_tokens', 0)
                    
                    # Add self-eval metrics if available
                    if sel_result.get('self_eval_result'):
                        se_metrics = sel_result['self_eval_result']['metrics']
                        metrics.selection_self_eval_predicted_score = se_metrics.get('predicted_score', 0.0)
                        metrics.selection_self_eval_prediction_error = se_metrics.get('prediction_error', 1.0)
                        metrics.selection_self_eval_parsing_error = se_metrics.get('parsing_error', True)
                        metrics.selection_self_eval_prompt_tokens = sel_result['self_eval_result'].get('prompt_tokens', 0)
                        metrics.selection_self_eval_output_tokens = sel_result['self_eval_result'].get('output_tokens', 0)
                    
                    total_latency += sel_result['latency']
                    all_prompt_tokens.append(sel_result.get('prompt_tokens', 0))
                    all_output_tokens.append(sel_result.get('output_tokens', 0))
                    
                    # Add self-eval tokens if available
                    if sel_result.get('self_eval_result'):
                        total_latency += sel_result['self_eval_result']['latency']
                        all_prompt_tokens.append(sel_result['self_eval_result'].get('prompt_tokens', 0))
                        all_output_tokens.append(sel_result['self_eval_result'].get('output_tokens', 0))
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
                if should_use_self_eval_for_stage('repair'):
                    logger.info("Using self-eval mode for repair")
                    rep_result = await run_repair_with_self_eval(cfg, llm, problem, file_contents, session)
                    if rep_result['training_text']:
                        training_texts.append(rep_result['training_text'])
                    if rep_result.get('self_eval_result') and rep_result['self_eval_result'].get('training_text'):
                        training_texts.append(rep_result['self_eval_result']['training_text'])
                else:
                    logger.info("Using pure stage mode for repair")
                    rep_result = await run_repair(cfg, llm, problem, file_contents, session)
                    if cfg.swe.get('enable_repair', False) and rep_result['training_text']:
                        training_texts.append(rep_result['training_text'])
                
                # Update metrics (always final/post-enhancement)
                rep_metrics = rep_result['metrics']
                metrics.repair_reward = rep_metrics.get('reward')
                metrics.repair_success = rep_metrics.get('success')
                metrics.repair_format_error = rep_metrics.get('format_error')

                metrics.repair_prompt_tokens = rep_result.get('prompt_tokens', 0)
                metrics.repair_output_tokens = rep_result.get('output_tokens', 0)
                
                # Add self-eval metrics if available
                if rep_result.get('self_eval_result'):
                    se_metrics = rep_result['self_eval_result']['metrics']
                    metrics.repair_self_eval_predicted_score = se_metrics.get('predicted_score', 0.0)
                    metrics.repair_self_eval_prediction_error = se_metrics.get('prediction_error', 1.0)
                    metrics.repair_self_eval_parsing_error = se_metrics.get('parsing_error', True)
                    metrics.repair_self_eval_prompt_tokens = rep_result['self_eval_result'].get('prompt_tokens', 0)
                    metrics.repair_self_eval_output_tokens = rep_result['self_eval_result'].get('output_tokens', 0)
                
                total_latency += rep_result['latency']
                all_prompt_tokens.append(rep_result.get('prompt_tokens', 0))
                all_output_tokens.append(rep_result.get('output_tokens', 0))
                
                # Add self-eval tokens if available
                if rep_result.get('self_eval_result'):
                    total_latency += rep_result['self_eval_result']['latency']
                    all_prompt_tokens.append(rep_result['self_eval_result'].get('prompt_tokens', 0))
                    all_output_tokens.append(rep_result['self_eval_result'].get('output_tokens', 0))
            else:
                logger.error("No file contents available for repair")
                metrics.repair_reward = 0
                metrics.repair_format_error = True
        
        metrics.total_prompt_tokens = sum(all_prompt_tokens)
        metrics.total_output_tokens = sum(all_output_tokens)
        metrics.compute_derived_metrics()
        
        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=total_latency,
            dataset_name=problem.get("dataset"),
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
        )
