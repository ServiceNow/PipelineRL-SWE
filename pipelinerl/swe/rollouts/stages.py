"""
Pure stage functions for SWE rollouts.
Contains the core stage execution logic with optional expert feedback integration.
"""

import json
import math
import time
import logging
from typing import Dict, Optional

from omegaconf import DictConfig
from tapeagents.core import LLMCall
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import make_training_text
from pipelinerl.swe.agents.localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from pipelinerl.swe.agents.file_selection_agent import FileSelectionAgent, FileSelectionTask, FileSelectionTape, FileSelectionResponse
from pipelinerl.swe.agents.repair_agent import RepairAgent, RepairTask, RepairTape, SearchReplaceResponse
from pipelinerl.swe.utils.bm25_searcher import BM25Searcher
from pipelinerl.swe.utils.localization_utils import parse_patch_for_gold_files, calculate_multi_query_mrr
from pipelinerl.swe.utils.repair_utils import calculate_precise_reward
from .base import execute_agent_with_retry

logger = logging.getLogger(__name__)


async def run_localization(cfg: DictConfig, llm: TrainableLLM, problem: Dict, session, expert_feedback=None):
    """Run core localization stage with optional expert feedback."""
    agent = LocalizationAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'max_prompt_length', 8000)
    )
    
    try:
        file_stats = json.loads(problem['all_file_stats'])
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse file stats")
        return {
            'training_text': None, 'top_files': [], 'queries': [], 'metrics': {"error": "parse_failure"},
            'latency': 0.0, 'prompt_tokens': 0, 'output_tokens': 0, 'success': False
        }
    
    task = LocalizationTask(
        problem_statement=problem["problem_statement"],
        file_stats=file_stats
    )
    
    # Create tape with optional expert feedback
    steps = [expert_feedback, task] if expert_feedback else [task]
    tape = LocalizationTape(steps=steps, context=None)
    
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
            'queries': queries if queries else [],  # Add queries to result
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'success': reward > 0
        }
        
    except Exception as e:
        logger.error(f"Localization error: {e}")
        return {
            'training_text': None, 'top_files': [], 'queries': [], 'metrics': {"error": str(e)},
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 
            'success': False
        }


async def run_file_selection(cfg: DictConfig, llm: TrainableLLM, problem: Dict, enriched_context: Dict, session, expert_feedback=None):
    """Run core file selection stage with optional expert feedback."""
    agent = FileSelectionAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'selection_max_prompt_length', 16000)
    )
    
    task = FileSelectionTask(
        problem_statement=problem["problem_statement"],
        candidate_files=enriched_context
    )
    
    # Create tape with optional expert feedback
    steps = [expert_feedback, task] if expert_feedback else [task]
    tape = FileSelectionTape(steps=steps, context=None)
    
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
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                reward = f1
                metrics_dict = {
                    "selection_precision": precision,
                    "selection_recall": recall,
                    "selection_f1": f1,
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


async def run_repair(cfg: DictConfig, llm: TrainableLLM, problem: Dict, file_contents: Dict, session, expert_feedback=None):
    """Run core repair stage with optional expert feedback."""
    agent = RepairAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'repair_max_prompt_length', 16000)
    )
    
    task = RepairTask(
        problem_statement=problem["problem_statement"],
        file_contents=file_contents
    )
    
    # Create tape with optional expert feedback
    steps = [expert_feedback, task] if expert_feedback else [task]
    tape = RepairTape(steps=steps, context=None)
    
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
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 
            'success': False
        }