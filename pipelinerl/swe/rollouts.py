"""
Unified rollout function for the SWE pipeline that can handle localization,
file selection, and repair stages with configurable training.
"""

import json
import time
import logging
from typing import Dict, List, Set
from pathlib import Path

import aiohttp
from omegaconf import DictConfig
from tapeagents.orchestrator import async_execute_agent
from tenacity import AsyncRetrying, RetryError, stop_after_attempt

from pipelinerl.rollouts import RolloutResult
from pipelinerl.async_llm import make_training_text
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.core import LLMCall

from agents.localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from agents.file_selection_agent import FileSelectionAgent, FileSelectionTask, FileSelectionTape, FileSelectionResponse
from utils.file_context_enricher import FileContextEnricher
from agents.repair_agent import RepairAgent, RepairTask, RepairTape, SearchReplaceResponse
from utils.bm25_searcher import BM25Searcher
from utils.repair_utils import calculate_precise_reward
from metrics import UnifiedMetrics

logger = logging.getLogger(__name__)


async def execute_agent_with_retry(agent, tape, session):
    """Execute agent with retry logic to handle cases where llm_call is None."""
    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(5)):
            with attempt:
                new_tape = await async_execute_agent(agent, tape, None, session)
                
                # Extract LLM call and validate it exists
                llm_call = None
                for step in new_tape.steps:
                    if (
                        hasattr(step, 'metadata') and 
                        step.metadata and 
                        hasattr(step.metadata, 'other') and
                        "llm_call" in step.metadata.other and
                        step.metadata.other["llm_call"] is not None
                    ):
                        llm_call = step.metadata.other["llm_call"]
                        break
                
                if llm_call is None:
                    raise ValueError("No LLM call found in the generated tape - retrying")
                
                return new_tape, llm_call
                
    except RetryError:
        raise ValueError("No LLM call found in the generated tape after 5 retry attempts")


async def run_localization_stage(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession
) -> Dict:
    """
    Run the localization stage: generate queries and search for top files.
    
    Args:
        cfg: Configuration
        llm: Trainable LLM  
        problem: Problem dictionary with 'all_file_stats' and 'problem_statement'
        session: HTTP session
        
    Returns:
        Dictionary with training_text, metrics, top_files, etc.
    """
    # Create localization agent
    agent = LocalizationAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'max_prompt_length', 8000)
    )
    
    # Create the localization task
    try:
        file_stats = json.loads(problem['all_file_stats'])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse all_file_stats")
        return {
            'training_text': None,
            'top_files': [],
            'metrics': {"error": "Failed to parse file stats"},
            'latency': 0.0,
            'prompt_tokens': 0,
            'output_tokens': 0,
            'success': False
        }
    
    task_step = LocalizationTask(
        problem_statement=problem["problem_statement"],
        file_stats=file_stats
    )
    
    tape = LocalizationTape(steps=[task_step], context=None)
    
    # Generate response using the agent
    time_start = time.time()
    
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - time_start
        
        # Extract the queries and format penalty from the response
        queries = None
        num_queries = 0
        format_penalty = 0.0
        
        for step in new_tape.steps:
            if isinstance(step, LocalizationQuery):
                queries = step.queries
                num_queries = step.num_queries
                format_penalty = step.format_penalty
                break
        
        # Convert to LLMCall object if it's a dict
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        # Perform BM25 search if we have valid queries
        top_files = []
        reward = 0.0
        metrics_dict = {}
        
        if queries and file_stats:
            # Create BM25 searcher and perform search
            searcher = BM25Searcher(file_stats)
            budget_per_query = 10 // num_queries
            
            all_query_results = []
            for query in queries:
                results = searcher.search(query, top_k=budget_per_query)
                all_query_results.append(results)
            
            # Extract top files (combine and deduplicate)
            file_scores = {}
            for query_results in all_query_results:
                for filepath, score in query_results:
                    if filepath not in file_scores or score > file_scores[filepath]:
                        file_scores[filepath] = score
            
            # Sort and take top 10
            sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
            top_files = [filepath for filepath, _ in sorted_files[:10]]
            
            # Calculate reward (MRR) using gold files
            from .localization_rollouts import parse_patch_for_gold_files, calculate_multi_query_mrr
            gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
            reward, reward_metadata = calculate_multi_query_mrr(gold_files, all_query_results)
            
            # Apply format penalty
            if format_penalty > 0:
                reward = reward - format_penalty
            
            metrics_dict = reward_metadata
            metrics_dict["mrr"] = reward
            metrics_dict["format_penalty"] = format_penalty
            
        else:
            reward = -1.0
            metrics_dict = {"error": "No valid queries generated"}
        
        # Apply discount factor if configured
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

        # Create training text
        training_text = make_training_text(llm, llm_call)
        
        if llm_call.logprobs:
            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            
            from pipelinerl.finetune.data import MASKED_TOKEN_ID
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
            
            training_text.input_ids = input_ids
            training_text.labels = labels
            training_text.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
        
        training_text.reward = reward
        training_text.group_id = new_tape.metadata.parent_id if new_tape.metadata else None
        
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
        logger.error(f"Error in localization stage: {e}")
        latency = time.time() - time_start
        
        return {
            'training_text': None,
            'top_files': [],
            'metrics': {"error": str(e)},
            'latency': latency,
            'prompt_tokens': 0,
            'output_tokens': 0,
            'success': False
        }


async def run_file_selection_stage(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    enriched_context: Dict[str, Dict],
    session: aiohttp.ClientSession
) -> Dict:
    """
    Run the file selection stage: select 1-3 files from candidates.
    
    Args:
        cfg: Configuration
        llm: Trainable LLM
        problem: Problem dictionary
        enriched_context: Dictionary of file paths to enriched context
        session: HTTP session for async requests
    
    Returns:
        Dictionary with training_text, metrics, selected_files, etc.
    """
    # Create file selection agent
    agent = FileSelectionAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'selection_max_prompt_length', 16000)
    )
    
    # Create the file selection task
    task_step = FileSelectionTask(
        problem_statement=problem["problem_statement"],
        candidate_files=enriched_context
    )
    
    tape = FileSelectionTape(steps=[task_step], context=None)
    
    # Generate response using the agent
    time_start = time.time()
    
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - time_start
        
        # Extract the selected files and format penalty from the response
        selected_files = []
        num_selected = 0
        format_penalty = 0.0
        
        for step in new_tape.steps:
            if isinstance(step, FileSelectionResponse):
                selected_files = step.selected_files
                num_selected = step.num_selected
                format_penalty = step.format_penalty
                break
        
        # Convert to LLMCall object if it's a dict
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        # Calculate reward based on selection accuracy
        from .localization_rollouts import parse_patch_for_gold_files
        gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
        
        if not selected_files:
            reward = -1.0
            metrics_dict = {"error": "No files selected"}
        else:
            # Calculate selection metrics
            if gold_files:
                relevant_selected = set(selected_files) & set(gold_files)
                selection_accuracy = len(relevant_selected) / len(gold_files)
                selection_precision = len(relevant_selected) / len(selected_files) if selected_files else 0
                selection_recall = len(relevant_selected) / len(gold_files) if gold_files else 0
                
                if selection_precision + selection_recall > 0:
                    selection_f1 = 2 * (selection_precision * selection_recall) / (selection_precision + selection_recall)
                else:
                    selection_f1 = 0.0
                
                reward = selection_accuracy
                metrics_dict = {
                    "selection_accuracy": selection_accuracy,
                    "selection_precision": selection_precision,
                    "selection_recall": selection_recall,
                    "selection_f1": selection_f1,
                    "gold_files": gold_files,
                    "selected_files": selected_files,
                }
            else:
                reward = 0.5  # Neutral reward when no gold files
                metrics_dict = {"warning": "No gold files for comparison"}
        
        # Apply format penalty
        if format_penalty > 0:
            reward = reward - format_penalty
            metrics_dict["format_penalty"] = format_penalty
        
        # Apply discount factor if configured
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

        # Create training text
        training_text = make_training_text(llm, llm_call)
        
        if llm_call.logprobs:
            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            
            from pipelinerl.finetune.data import MASKED_TOKEN_ID
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
            
            training_text.input_ids = input_ids
            training_text.labels = labels
            training_text.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
        
        training_text.reward = reward
        training_text.group_id = new_tape.metadata.parent_id if new_tape.metadata else None
        
        return {
            'training_text': training_text,
            'selected_files': selected_files,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': reward > 0.5
        }
        
    except Exception as e:
        logger.error(f"Error in file selection stage: {e}")
        latency = time.time() - time_start
        
        return {
            'training_text': None,
            'selected_files': [],
            'metrics': {"error": str(e)},
            'latency': latency,
            'prompt_tokens': 0,
            'output_tokens': 0,
            'success': False
        }


async def run_repair_stage(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    selected_file_contents: Dict[str, str],
    session: aiohttp.ClientSession
) -> Dict:
    """
    Run the repair stage: generate search/replace edits.
    
    Args:
        cfg: Configuration
        llm: Trainable LLM
        problem: Problem dictionary
        selected_file_contents: File contents for repair
        session: HTTP session
        
    Returns:
        Dictionary with training_text, metrics, etc.
    """
    # Create repair agent
    agent = RepairAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'repair_max_prompt_length', 16000)
    )
    
    # Create the task step
    task_step = RepairTask(
        problem_statement=problem["problem_statement"],
        file_contents=selected_file_contents
    )

    tape = RepairTape(steps=[task_step], context=None)
    
    # Generate response using the agent
    time_start = time.time()
    
    try:
        new_tape = await async_execute_agent(agent, tape, None, session)
        latency = time.time() - time_start
        
        # Extract edits from the response step
        predicted_edits = []
        llm_call = None
        
        for step in new_tape.steps:
            if isinstance(step, SearchReplaceResponse):
                predicted_edits = step.edits
            # Get the LLM call for training data
            if (
                hasattr(step, 'metadata') and 
                step.metadata and 
                hasattr(step.metadata, 'other') and
                "llm_call" in step.metadata.other and
                step.metadata.other["llm_call"] is not None
            ):
                llm_call = step.metadata.other["llm_call"]
        
        if llm_call is None:
            raise ValueError("No LLM call found in the generated tape")
        
        # Convert to LLMCall object if it's a dict
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        # Calculate reward using precise file-by-file analysis
        gold_patch = problem["patch"]
        
        if predicted_edits:
            reward, reward_metadata = calculate_precise_reward(
                selected_file_contents, gold_patch, predicted_edits
            )
        else:
            reward = -1.0
            reward_metadata = {"format_error": "No edits found"}
        
        # Apply discount factor if configured
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
        
        # Create training text
        training_text = make_training_text(llm, llm_call)
        
        # Set up input_ids and labels for training
        input_ids = [lp.token_id for lp in llm_call.logprobs]
        labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
        
        from pipelinerl.finetune.data import MASKED_TOKEN_ID
        labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
        
        training_text.input_ids = input_ids
        training_text.labels = labels
        training_text.reward = reward
        training_text.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
        training_text.group_id = new_tape.metadata.parent_id if new_tape.metadata else None
        
        # Calculate success metric
        success_threshold = getattr(cfg.actor, 'success_threshold', 0.8)
        success = reward > success_threshold
        
        # Prepare metrics
        metrics_dict = {
            "reward": reward,
            "success": success,
            "no_edits": len(predicted_edits) == 0,
            "format_error": "format_error" in reward_metadata or "error" in reward_metadata,
            **reward_metadata
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
        logger.error(f"Error in repair stage: {e}")
        latency = time.time() - time_start
        
        return {
            'training_text': None,
            'metrics': {"error": str(e)},
            'latency': latency,
            'prompt_tokens': 0,
            'output_tokens': 0,
            'success': False
        }


def get_oracle_files_from_patch(patch: str, max_files: int = 10) -> List[str]:
    """Get oracle files from patch, limited to max_files."""
    from .localization_rollouts import parse_patch_for_gold_files
    oracle_files = parse_patch_for_gold_files(patch)
    return oracle_files[:max_files]


async def unified_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession
) -> RolloutResult:
    """
    Generate a unified rollout with configurable pipeline stages.
    
    Args:
        cfg: Configuration with training flags
        llm: Trainable LLM
        problem: Problem dictionary containing:
            - problem_statement: The issue description
            - patch: Gold patch to extract modified files from
            - all_file_stats: JSON string of file statistics for BM25
            - file_contents: Dict mapping file paths to content (for repair-only mode)
            - repo_path: Path to the repository (for file selection)
            - base_commit: Git commit hash (for file selection)
            - Additional metadata
        session: HTTP session for async requests
    
    Returns:
        RolloutResult with training texts and unified metrics
    """
    training_texts = []
    metrics = UnifiedMetrics()
    total_latency = 0.0
    all_prompt_tokens = []
    all_output_tokens = []
    
    # Initialize file context enricher
    enricher = FileContextEnricher()
    
    try:
        # STEP 1: Coarse Localization (repo -> top 10 files)
        top_files = []
        if cfg.training.get('enable_localization', False) or cfg.training.get('run_localization', False):
            logger.info("Running localization stage...")
            loc_result = await run_localization_stage(cfg, llm, problem, session)
            
            if cfg.training.get('enable_localization', False) and loc_result['training_text']:
                training_texts.append(loc_result['training_text'])
            
            # Copy localization metrics
            loc_metrics = loc_result['metrics']
            metrics.localization_mrr = loc_metrics.get('mrr')
            metrics.localization_num_queries = loc_metrics.get('num_queries')
            metrics.localization_format_penalty = loc_metrics.get('format_penalty', 0.0)
            metrics.localization_format_violation = loc_metrics.get('format_penalty', 0.0) > 0
            
            total_latency += loc_result['latency']
            all_prompt_tokens.append(loc_result['prompt_tokens'])
            all_output_tokens.append(loc_result['output_tokens'])
            
            top_files = loc_result['top_files']
            
            if not top_files:  # Fallback if localization fails
                logger.warning("Localization failed, using oracle files")
                top_files = get_oracle_files_from_patch(problem.get("patch", ""), max_files=10)
        else:
            logger.info("Skipping localization, using oracle files")
            top_files = get_oracle_files_from_patch(problem.get("patch", ""), max_files=10)
        
        # STEP 2: Fine Selection (top 10 -> 1-3 files)
        selected_files = []
        enriched_context = {}
        
        if cfg.training.get('enable_file_selection', False) or cfg.training.get('run_file_selection', False):
            logger.info("Running file selection stage...")
            
            if top_files:
                # Enrich the top files with detailed context
                repo_path = Path(problem.get('repo_path', ''))
                base_commit = problem.get('base_commit', '')
                
                enriched_context = enricher.enrich_files_on_demand(
                    top_files, repo_path, base_commit
                )
                
                if enriched_context:
                    selection_result = await run_file_selection_stage(
                        cfg, llm, problem, enriched_context, session
                    )
                    
                    if cfg.training.get('enable_file_selection', False) and selection_result['training_text']:
                        training_texts.append(selection_result['training_text'])
                    
                    # Extract selection metrics
                    selected_files = selection_result['selected_files']
                    sel_metrics = selection_result['metrics']
                    
                    metrics.selection_num_candidates = len(enriched_context)
                    metrics.selection_num_selected = len(selected_files)
                    metrics.selection_accuracy = sel_metrics.get('selection_accuracy')
                    metrics.selection_precision = sel_metrics.get('selection_precision')
                    metrics.selection_recall = sel_metrics.get('selection_recall')
                    metrics.selection_f1_score = sel_metrics.get('selection_f1')
                    metrics.selection_format_penalty = sel_metrics.get('format_penalty', 0.0)
                    metrics.selection_format_violation = sel_metrics.get('format_penalty', 0.0) > 0
                    
                    total_latency += selection_result['latency']
                    all_prompt_tokens.append(selection_result['prompt_tokens'])
                    all_output_tokens.append(selection_result['output_tokens'])
                else:
                    logger.warning("Failed to enrich file context, using top 3 files")
                    selected_files = top_files[:3]
            else:
                logger.warning("No top files available for selection")
                selected_files = []
        else:
            logger.info("Skipping file selection, using top 3 files")
            selected_files = top_files[:3]
        
        # Ensure we have files for repair
        if not selected_files:
            logger.warning("No files selected, using all oracle files for repair")
            selected_files = get_oracle_files_from_patch(problem.get("patch", ""))
        
        # STEP 3: Repair (1-3 files -> patch)
        if cfg.training.get('enable_repair', False) or cfg.training.get('run_repair', False):
            logger.info("Running repair stage...")
            
            # Prepare file contents for repair
            if enriched_context:
                # Use enriched context if available
                selected_file_contents = {
                    filepath: enriched_context[filepath]['content'] 
                    for filepath in selected_files 
                    if filepath in enriched_context
                }
            else:
                # Fallback to original file_contents from problem
                original_contents = problem.get('file_contents', {})
                selected_file_contents = {
                    filepath: original_contents[filepath]
                    for filepath in selected_files
                    if filepath in original_contents
                }
            
            if selected_file_contents:
                repair_result = await run_repair_stage(cfg, llm, problem, selected_file_contents, session)
                
                if cfg.training.get('enable_repair', False) and repair_result['training_text']:
                    training_texts.append(repair_result['training_text'])
                
                # Copy repair metrics
                rep_metrics = repair_result['metrics']
                metrics.repair_reward = rep_metrics.get('reward')
                metrics.repair_success = rep_metrics.get('success')
                metrics.repair_no_edits = rep_metrics.get('no_edits')
                metrics.repair_format_error = rep_metrics.get('format_error')
                
                total_latency += repair_result['latency']
                all_prompt_tokens.append(repair_result['prompt_tokens'])
                all_output_tokens.append(repair_result['output_tokens'])
            else:
                logger.error("No file contents available for repair")
                metrics.repair_reward = -1.0
                metrics.repair_format_error = True
        
        # Compute derived metrics
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
        logger.error(f"Error in unified rollout: {e}")
        
        # Return failed rollout
        metrics = UnifiedMetrics()
        metrics.reward = 0.0
        metrics.success = False
        metrics.no_error = False
        metrics.pipeline_success = False
        
        return RolloutResult(
            training_texts=training_texts,  # Return any partial training texts
            metrics=metrics,
            latency=total_latency,
            dataset_name=problem.get("dataset"),
            prompt_tokens=all_prompt_tokens,
            output_tokens=all_output_tokens,
        )