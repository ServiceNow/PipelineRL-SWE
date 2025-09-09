"""
Complete SWE pipeline rollout supporting localization, file selection, repair, and self-evaluation stages.
"""

import json
import time
import math
import logging
from typing import Dict, List
from pathlib import Path

import aiohttp
from omegaconf import DictConfig
from tapeagents.orchestrator import async_execute_agent
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from pipelinerl.rollouts import RolloutResult
from pipelinerl.async_llm import make_training_text
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.core import LLMCall, Action, Observation
from tapeagents.environment import AsyncEnvironment
from tapeagents.agent import TapeType

from pipelinerl.swe.agents.localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from pipelinerl.swe.agents.file_selection_agent import FileSelectionAgent, FileSelectionTask, FileSelectionTape, FileSelectionResponse
from pipelinerl.swe.agents.repair_agent import RepairAgent, RepairTask, RepairTape, SearchReplaceResponse
from pipelinerl.swe.agents.self_eval_agent import SelfEvalAgent, SelfEvalTask, SelfEvalTape, SelfEvalResponse
from pipelinerl.swe.utils.file_context_enricher import FileContextEnricher
from pipelinerl.swe.utils.bm25_searcher import BM25Searcher
from pipelinerl.swe.utils.repair_utils import calculate_precise_reward
from pipelinerl.swe.utils.localization_utils import parse_patch_for_gold_files, calculate_multi_query_mrr
from pipelinerl.swe.metrics import UnifiedMetrics

logger = logging.getLogger(__name__)


class EmptyAsyncEnvironment(AsyncEnvironment):
    """No-op environment for agent execution."""
    
    async def ainitialize(self):
        pass

    async def areact(self, tape: TapeType) -> TapeType:
        return tape

    async def astep(self, action: Action):
        pass

    async def areset(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def react(self, tape: TapeType) -> TapeType:
        return tape


async def execute_agent_with_retry(agent, tape, session):
    """Execute agent with retry logic for robust LLM call extraction."""
    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(5)):
            with attempt:
                new_tape = await async_execute_agent(agent, tape, EmptyAsyncEnvironment(), session, max_loops=1)
                
                llm_call = None
                for step in new_tape.steps:
                    if (hasattr(step, 'metadata') and 
                        step.metadata and 
                        hasattr(step.metadata, 'other') and
                        "llm_call" in step.metadata.other and
                        step.metadata.other["llm_call"] is not None):
                        llm_call = step.metadata.other["llm_call"]
                        break
                
                if llm_call is None:
                    raise ValueError("No LLM call found in tape")
                
                return new_tape, llm_call
                
    except RetryError:
        raise ValueError("Failed to extract LLM call after 5 attempts")


async def run_localization(cfg, llm, problem, session):
    """Execute localization stage: problem statement -> top 10 files."""
    agent = LocalizationAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'max_prompt_length', 8000)
    )
    
    try:
        file_stats = json.loads(problem['all_file_stats'])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse file stats")
        return {
            'training_text': None, 'top_files': [], 'metrics': {"error": "parse_failure"},
            'latency': 0.0, 'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }
    
    task = LocalizationTask(
        problem_statement=problem["problem_statement"],
        file_stats=file_stats
    )
    tape = LocalizationTape(steps=[task], context=None)
    
    start_time = time.time()
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - start_time
        
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        queries = None
        num_queries = 0
        format_penalty = 0.0
        
        for step in new_tape.steps:
            if isinstance(step, LocalizationQuery):
                queries = step.queries
                num_queries = step.num_queries
                format_penalty = step.format_penalty
                break
        
        top_files = []
        reward = 0.0
        metrics_dict = {}
        
        if queries and file_stats:
            searcher = BM25Searcher(file_stats)
            budget_per_query = max(1, 10 // num_queries)
            
            all_results = []
            for query in queries:
                results = searcher.search(query, top_k=budget_per_query)
                all_results.append(results)
            
            file_scores = {}
            for query_results in all_results:
                for filepath, score in query_results:
                    if filepath not in file_scores or score > file_scores[filepath]:
                        file_scores[filepath] = score
            
            sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
            top_files = [filepath for filepath, _ in sorted_files[:10]]
            
            gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
            reward, reward_metadata = calculate_multi_query_mrr(gold_files, all_results)
            
            # Inject oracle files for downstream stages
            oracle_not_found = [f for f in gold_files if f not in set(top_files) and f in file_stats]
            if oracle_not_found:
                remaining_slots = max(0, 10 - len(oracle_not_found))
                non_oracle_files = [f for f in top_files if f not in set(gold_files)]
                top_files = oracle_not_found + non_oracle_files[:remaining_slots]
                logger.info(f"Injected {len(oracle_not_found)} oracle files")
            
            found_gold = len(set(top_files) & set(gold_files))
            localization_recall = found_gold / len(gold_files) if gold_files else 1.0
            
            reward = max(0.0, reward - format_penalty)
            metrics_dict = {
                **reward_metadata,
                "mrr": reward,
                "localization_recall": localization_recall,
                "format_penalty": format_penalty
            }
        else:
            metrics_dict = {"error": "no_valid_queries"}
        
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

        training_text = make_training_text(llm, llm_call)
        training_text.reward = reward if (reward is not None and not math.isnan(reward)) else 0.0
        
        return {
            'training_text': training_text,
            'top_files': top_files,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': reward > 0
        }
        
    except Exception as e:
        logger.error(f"Localization error: {e}")
        return {
            'training_text': None, 'top_files': [], 'metrics': {"error": str(e)},
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }


async def run_file_selection(cfg, llm, problem, enriched_context, session):
    """Execute file selection stage: candidate files -> 1-3 selected files."""
    agent = FileSelectionAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'selection_max_prompt_length', 16000)
    )
    
    task = FileSelectionTask(
        problem_statement=problem["problem_statement"],
        candidate_files=enriched_context
    )
    tape = FileSelectionTape(steps=[task], context=None)
    
    start_time = time.time()
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - start_time
        
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        selected_files = []
        format_penalty = 0.0
        
        for step in new_tape.steps:
            if isinstance(step, FileSelectionResponse):
                selected_files = step.selected_files
                format_penalty = step.format_penalty
                break
        
        gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
        
        if not selected_files:
            reward = 0
            metrics_dict = {"error": "no_files_selected"}
        else:
            if gold_files:
                relevant_selected = set(selected_files) & set(gold_files)
                precision = len(relevant_selected) / len(selected_files)
                recall = len(relevant_selected) / len(gold_files)
                reward = recall
                metrics_dict = {
                    "selection_precision": precision,
                    "selection_recall": recall,
                    "gold_files": gold_files,
                    "selected_files": selected_files,
                }
            else:
                reward = 0.5
                metrics_dict = {"warning": "no_gold_files"}
        
        reward = max(0.0, reward - format_penalty)
        metrics_dict["format_penalty"] = format_penalty
        
        # Use oracle files for repair if selection differs
        if set(selected_files) != set(gold_files):
            files_for_repair = gold_files
            logger.info(f"Using oracle files for repair: {gold_files}")
        else:
            files_for_repair = selected_files
        
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

        training_text = make_training_text(llm, llm_call)
        training_text.reward = reward if (reward is not None and not math.isnan(reward)) else 0.0
        
        return {
            'training_text': training_text,
            'selected_files': selected_files,
            'files_for_repair': files_for_repair,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': reward > 0.5
        }
        
    except Exception as e:
        logger.error(f"File selection error: {e}")
        return {
            'training_text': None, 'selected_files': [], 'files_for_repair': [],
            'metrics': {"error": str(e)}, 'latency': time.time() - start_time,
            'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }


async def run_repair(cfg, llm, problem, file_contents, session):
    """Execute repair stage: file contents -> search/replace edits."""
    agent = RepairAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'repair_max_prompt_length', 16000)
    )
    
    task = RepairTask(
        problem_statement=problem["problem_statement"],
        file_contents=file_contents
    )
    tape = RepairTape(steps=[task], context=None)
    
    start_time = time.time()
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - start_time
        
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        edits = []
        for step in new_tape.steps:
            if isinstance(step, SearchReplaceResponse):
                edits = step.edits
                break
        
        gold_patch = problem["patch"]
        
        if edits:
            reward, reward_metadata = calculate_precise_reward(file_contents, gold_patch, edits)
        else:
            reward = 0
            reward_metadata = {"format_error": True}
        
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
        
        training_text = make_training_text(llm, llm_call)
        training_text.reward = reward if (reward is not None and not math.isnan(reward)) else 0.0
        
        success_threshold = getattr(cfg.actor, 'success_threshold', 0.8)
        success = reward > success_threshold
        
        metrics_dict = {
            "reward": reward,
            "success": success,
            "no_edits": len(edits) == 0,
            "format_error": "format_error" in reward_metadata,
            **reward_metadata
        }
        
        return {
            'training_text': training_text,
            'repair_edits': edits,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': success
        }
        
    except Exception as e:
        logger.error(f"Repair error: {e}")
        return {
            'training_text': None, 'repair_edits': [], 'metrics': {"error": str(e)},
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }


async def run_self_eval(cfg, llm, problem, file_contents, repair_edits, true_reward, session):
    """Execute self-evaluation stage: repair edits -> predicted quality score."""
    agent = SelfEvalAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'self_eval_max_prompt_length', 20000)
    )
    
    task = SelfEvalTask(
        problem_statement=problem["problem_statement"],
        file_contents=file_contents,
        repair_edits=repair_edits
    )
    tape = SelfEvalTape(steps=[task], context=None)
    
    start_time = time.time()
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - start_time
        
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        analysis = ""
        predicted_score = 0.0
        parsing_error = False
        
        for step in new_tape.steps:
            if isinstance(step, SelfEvalResponse):
                analysis = step.analysis
                predicted_score = step.predicted_score
                parsing_error = step.parsing_error
                break
        
        prediction_error = abs(predicted_score - true_reward)
        reward = max(0.0, 1.0 - prediction_error)
        
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
        
        training_text = make_training_text(llm, llm_call)
        training_text.reward = reward if (reward is not None and not math.isnan(reward)) else 0.0
        
        threshold = getattr(cfg.actor, 'self_eval_threshold', 0.2)
        success = prediction_error <= threshold
        
        metrics_dict = {
            "predicted_score": predicted_score,
            "true_reward": true_reward,
            "prediction_error": prediction_error,
            "reward": reward,
            "success": success,
            "parsing_error": parsing_error,
            "analysis_length": len(analysis.split()) if analysis else 0
        }
        
        return {
            'training_text': training_text,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': success
        }
        
    except Exception as e:
        logger.error(f"Self-eval error: {e}")
        return {
            'training_text': None, 'metrics': {"error": str(e)},
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }


def get_oracle_files(patch, max_files=10):
    """Extract oracle files from patch, limited to max_files."""
    oracle_files = parse_patch_for_gold_files(patch)
    return oracle_files[:max_files]


def log_variance_info(training_texts, metrics, group_id=""):
    """Log rollout variance information for debugging."""
    if not training_texts:
        logger.warning(f"Group {group_id}: No training texts")
        return
    
    rewards = [tt.reward for tt in training_texts]
    unique_rewards = set(rewards)
    
    logger.info(f"Group {group_id}: {len(training_texts)} texts, {len(unique_rewards)} unique rewards")
    
    if len(unique_rewards) == 1:
        logger.warning(f"All rewards identical: {list(unique_rewards)[0]}")
        
        stage_info = []
        if hasattr(metrics, 'localization_mrr') and metrics.localization_mrr is not None:
            stage_info.append(f"loc={metrics.localization_mrr:.3f}")
        if hasattr(metrics, 'selection_recall') and metrics.selection_recall is not None:
            stage_info.append(f"sel={metrics.selection_recall:.3f}")
        if hasattr(metrics, 'repair_reward') and metrics.repair_reward is not None:
            stage_info.append(f"rep={metrics.repair_reward:.3f}")
        if hasattr(metrics, 'self_eval_predicted_score') and metrics.self_eval_predicted_score is not None:
            stage_info.append(f"eval={metrics.self_eval_predicted_score:.3f}")
        
        logger.warning(f"Stage metrics: {', '.join(stage_info)}")


async def generate_unified_swe_rollout(cfg, llm, problem, session):
    """
    Generate complete SWE pipeline rollout with all configurable stages.
    
    Args:
        cfg: Configuration with stage enable flags
        llm: Trainable LLM
        problem: Problem dict with statement, patch, file_stats, etc.
        session: HTTP session for async requests
    
    Returns:
        RolloutResult with training texts and unified metrics
    """
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
                training_texts.append(loc_result['training_text'])
            
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
                top_files = get_oracle_files(problem.get("patch", ""), max_files=10)
        else:
            logger.info("Skipping localization, using oracle files")
            top_files = get_oracle_files(problem.get("patch", ""), max_files=10)
        
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
                    sel_result = await run_file_selection(cfg, llm, problem, enriched_context, session)
                    
                    if cfg.swe.get('enable_file_selection', False) and sel_result['training_text']:
                        training_texts.append(sel_result['training_text'])
                    
                    files_for_repair = sel_result.get('files_for_repair', [])
                    
                    sel_metrics = sel_result['metrics']
                    metrics.selection_precision = sel_metrics.get('selection_precision', 0.0)
                    metrics.selection_recall = sel_metrics.get('selection_recall', 0.0)
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
            files_for_repair = get_oracle_files(problem.get("patch", ""))
        
        # Stage 3: Repair
        repair_edits = []
        repair_reward = 0.0
        file_contents = {}
        
        if (cfg.swe.get('enable_repair', False) or 
            cfg.swe.get('run_repair', False) or 
            cfg.swe.get('enable_self_eval', False)):
            
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
                rep_result = await run_repair(cfg, llm, problem, file_contents, session)
                
                if cfg.swe.get('enable_repair', False) and rep_result['training_text']:
                    training_texts.append(rep_result['training_text'])
                
                repair_edits = rep_result.get('repair_edits', [])
                
                rep_metrics = rep_result['metrics']
                metrics.repair_reward = rep_metrics.get('reward')
                repair_reward = metrics.repair_reward or 0.0
                metrics.repair_success = rep_metrics.get('success')
                metrics.repair_format_error = rep_metrics.get('format_error')
                
                total_latency += rep_result['latency']
                all_prompt_tokens.append(rep_result['prompt_tokens'])
                all_output_tokens.append(rep_result['output_tokens'])
            else:
                logger.error("No file contents available for repair")
                metrics.repair_reward = 0
                metrics.repair_format_error = True
        
        # Stage 4: Self-Evaluation
        if cfg.swe.get('enable_self_eval', False) and repair_edits and file_contents:
            logger.info("Running self-evaluation stage")
            
            eval_result = await run_self_eval(cfg, llm, problem, file_contents, repair_edits, repair_reward, session)
            
            if eval_result['training_text']:
                training_texts.append(eval_result['training_text'])
            
            eval_metrics = eval_result['metrics']
            metrics.self_eval_predicted_score = eval_metrics.get('predicted_score')
            metrics.self_eval_prediction_error = eval_metrics.get('prediction_error')
            metrics.self_eval_reward = eval_metrics.get('reward')
            metrics.self_eval_success = eval_metrics.get('success')
            metrics.self_eval_parsing_error = eval_metrics.get('parsing_error')
            
            total_latency += eval_result['latency']
            all_prompt_tokens.append(eval_result['prompt_tokens'])
            all_output_tokens.append(eval_result['output_tokens'])
        
        metrics.compute_derived_metrics()
        log_variance_info(training_texts, metrics, group_id=problem.get('id', 'unknown'))
        
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