import json
import time
import logging
from typing import Dict, List, Tuple

import aiohttp
from omegaconf import DictConfig
from tapeagents.orchestrator import async_execute_agent

from pipelinerl.rollouts import RolloutResult
from pipelinerl.async_llm import make_training_text
from tapeagents.llms.trainable import TrainableLLM
from .localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from .bm25_searcher import BM25Searcher

logger = logging.getLogger(__name__)


def calculate_mrr_reward(gold_files: List[str], ranked_files: List[Tuple[str, float]]) -> Tuple[float, Dict]:
    """
    Calculate Mean Reciprocal Rank (MRR) reward for localization.
    
    Args:
        gold_files: List of gold file paths that should be found
        ranked_files: List of (filepath, score) tuples from BM25 search
        
    Returns:
        Tuple of (mrr_reward, metadata_dict)
    """
    if not gold_files:
        return 0.0, {"error": "No gold files provided"}
        
    if not ranked_files:
        return 0.0, {"gold_files": gold_files, "found_ranks": []}
        
    # Extract just the file paths from ranked results
    ranked_paths = [filepath for filepath, _ in ranked_files]
    
    # Find ranks of gold files (1-indexed)
    gold_ranks = []
    found_files = []
    
    for gold_file in gold_files:
        try:
            rank = ranked_paths.index(gold_file) + 1  # 1-indexed
            gold_ranks.append(rank)
            found_files.append(gold_file)
        except ValueError:
            # Gold file not found in results
            gold_ranks.append(0)  # 0 means not found
            
    # Calculate reciprocal ranks (1/rank for found files, 0 for not found)
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in gold_ranks]
    
    # Average MRR across all gold files
    mrr = sum(reciprocal_ranks) / len(gold_files)
    
    metadata = {
        "gold_files": gold_files,
        "found_files": found_files,
        "gold_ranks": gold_ranks,
        "reciprocal_ranks": reciprocal_ranks,
        "num_gold_files": len(gold_files),
        "num_found": len(found_files),
        "best_rank": min([r for r in gold_ranks if r > 0]) if any(r > 0 for r in gold_ranks) else 0,
        "worst_rank": max(gold_ranks),
    }
    
    return mrr, metadata


def parse_patch_for_gold_files(patch_string: str) -> List[str]:
    """
    Extract gold file paths from a git patch string.
    
    Args:
        patch_string: Git patch in unified diff format
        
    Returns:
        List of file paths that were modified
    """
    import re
    
    if not patch_string:
        return []
        
    # Find lines starting with '--- a/' and capture the path following 'a/'
    gold_filepaths = re.findall(r'^--- a/(.+)$', patch_string, re.MULTILINE)
    return gold_filepaths


async def generate_localization_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession
) -> RolloutResult:
    """
    Generate a single localization rollout.
    
    Args:
        cfg: Configuration
        llm: Trainable LLM
        problem: Problem dictionary containing:
            - problem_statement: The issue description
            - patch: Gold patch to extract modified files from
            - all_file_stats: JSON string of file statistics for BM25
            - Additional metadata
        session: HTTP session for async requests
    
    Returns:
        RolloutResult with training text and MRR metrics
    """
    # Create localization agent
    agent = LocalizationAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, 'max_prompt_length', 8000)
    )
    
    # Create the localization task
    task_step = LocalizationTask(
        problem_statement=problem["problem_statement"],
        file_stats=json.loads(problem['all_file_stats'])
    )
    
    # Create initial tape with just the task
    tape = LocalizationTape(steps=[task_step], context=None)
    
    # Generate response using the agent
    time_start = time.time()
    
    try:
        # Run the agent to get the localization query
        new_tape = await async_execute_agent(
            agent,
            tape,
            None,  # no environment needed
            session,
        )
        
        latency = time.time() - time_start
        
        # Extract the query and LLM call from the response
        query = None
        llm_call = None
        
        for step in new_tape.steps:
            if isinstance(step, LocalizationQuery):
                query = step.query
            # Get the LLM call for training data
            if (
                hasattr(step, 'metadata') and 
                step.metadata and 
                hasattr(step.metadata, 'other') and
                "llm_call" in step.metadata.other and
                step.metadata.other["llm_call"] is not None
            ):
                llm_call = step.metadata.other["llm_call"]
        
        # Critical failure: No LLM call means we can't generate training data
        if llm_call is None:
            raise ValueError("No LLM call found in the generated tape")
            
        # Convert to LLMCall object if it's a dict
        from tapeagents.core import LLMCall
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        # Determine reward based on whether we have a valid query
        if query is None:
            logger.warning("No localization query found - treating as format violation")
            reward = -0.1  # Negative reward for format violation
            reward_metadata = {"error": "No query found - format violation"}
            search_results = []
            format_violation = True
        else:
            # Normal case: we have a query, proceed with BM25 search
            try:
                all_file_stats = json.loads(problem.get("all_file_stats", "{}"))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse all_file_stats, using empty dict")
                all_file_stats = {}
                
            if not all_file_stats:
                # No files to search - give zero reward
                reward = 0.0
                reward_metadata = {"error": "No file statistics available"}
                search_results = []
                logger.error("NO ALL FILE STATS.")
            else:
                # Create BM25 searcher and perform search
                searcher = BM25Searcher(all_file_stats)
                search_results = searcher.search(query, top_k=100)
                
                # Extract gold files from patch and calculate MRR reward
                gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
                reward, reward_metadata = calculate_mrr_reward(gold_files, search_results)
            
            format_violation = False
            
        # Apply discount factor if configured
        if hasattr(cfg.actor, 'discount_factor'):
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens

        # Create training text using pipelinerl function
        training_text = make_training_text(llm, llm_call)
        
        # Set up additional training data fields if available
        if llm_call.logprobs:
            input_ids = [lp.token_id for lp in llm_call.logprobs]
            labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
            
            from pipelinerl.finetune.data import MASKED_TOKEN_ID
            labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
            
            training_text.input_ids = input_ids
            training_text.labels = labels
            training_text.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
        else:
            logger.warning("No logprobs available for training text")
            
        # Check if the generation finished properly
        finished = 1 if (llm_call.logprobs and 
                        llm_call.logprobs[-1].token_id == llm.tokenizer.eos_token_id) else 0
        
        training_text.reward = reward
        training_text.group_id = new_tape.metadata.parent_id if new_tape.metadata else None
        
        # Calculate success metric (any gold file found in top 10)
        top_10_files = [fp for fp, _ in search_results[:10]] if search_results else []
        gold_files = reward_metadata.get("gold_files", [])
        #success = any(gf in top_10_files for gf in gold_files) if not format_violation else False
        success = sum(1 for gf in gold_files if gf in top_10_files) / len(gold_files)
        success = success == 1 # if all gold files are found in top 10
        
        # Prepare metrics
        metrics = {
            "reward": reward,
            "mrr": reward if not format_violation else -1,
            "success": success,
            "query_length": len(query.split()) if query else 0,
            "no_answer": query is None or (query and query.strip() == ""),
            "no_error": True,
            "overflow": 0 if finished else 1,
            "num_search_results": len(search_results),
            "prompt_tokens": llm_call.prompt_length_tokens,
            "output_tokens": llm_call.output_length_tokens,
            "format_violation": format_violation,
        }
        
        # Add reward metadata to metrics
        metrics.update({f"localization_{k}": v for k, v in reward_metadata.items() 
                       if isinstance(v, (int, float, bool))})
        
        # Return training result
        training_texts = [training_text]
        if hasattr(training_text, 'model_dump'):
            try:
                return RolloutResult(
                    training_texts=training_texts,
                    metrics=metrics,
                    latency=latency,
                    dataset_name=problem.get("dataset"),
                    prompt_tokens=[llm_call.prompt_length_tokens],
                    output_tokens=[llm_call.output_length_tokens],
                )
            except Exception:
                training_texts = [training_text.model_dump()]
        
        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=latency,
            dataset_name=problem.get("dataset"),
            prompt_tokens=[llm_call.prompt_length_tokens],
            output_tokens=[llm_call.output_length_tokens],
        )
        
    except Exception as e:
        logger.error(f"Error in localization rollout: {e}")
        latency = time.time() - time_start
        
        # Return failed rollout (this catches the "No LLM call" error and other system errors)
        return RolloutResult(
            training_texts=[],
            metrics={
                "reward": 0.0,
                "success": False,
                "no_error": False,
                "error": str(e),
                "prompt_tokens": 0,
                "output_tokens": 0,
                "no_answer": True,
            },
            latency=latency,
            dataset_name=problem.get("dataset"),
            prompt_tokens=[0],
            output_tokens=[0],
        )