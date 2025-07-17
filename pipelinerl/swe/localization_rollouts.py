import json
import time
import logging
from typing import Dict, List, Tuple

import aiohttp
from omegaconf import DictConfig
from tapeagents.orchestrator import async_execute_agent

from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.async_llm import make_training_text
from tapeagents.llms.trainable import TrainableLLM
from .localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from .bm25_searcher import BM25Searcher

logger = logging.getLogger(__name__)


class LocalizationMetrics(BaseMetrics):
    mrr: float
    num_queries: int
    total_query_length: int
    avg_query_length: float
    num_search_results: int
    prompt_tokens: int
    output_tokens: int
    format_violation: bool
    overflow: int
    localization_num_gold_files: int = 0
    localization_num_found: int = 0
    localization_best_rank: int = 0
    localization_worst_rank: int = 0
    localization_num_queries: int = 0


def calculate_multi_query_mrr(gold_files: List[str], query_results: List[List[Tuple[str, float]]]) -> Tuple[float, Dict]:
    """
    Calculate Mean Reciprocal Rank (MRR) using best rank across multiple queries.
    
    Args:
        gold_files: List of gold file paths that should be found
        query_results: List of query results, where each is a list of (filepath, score) tuples
        
    Returns:
        Tuple of (mrr_reward, metadata_dict)
    """
    if not gold_files:
        return 0.0, {"error": "No gold files provided"}
        
    if not query_results or not any(query_results):
        return 0.0, {"gold_files": gold_files, "best_ranks": {}}
    
    # Find best rank for each gold file across all queries
    best_ranks = {}
    found_files = []
    
    for gold_file in gold_files:
        best_rank = float('inf')
        
        for query_idx, ranked_files in enumerate(query_results):
            if not ranked_files:
                continue
                
            ranked_paths = [filepath for filepath, _ in ranked_files]
            try:
                rank = ranked_paths.index(gold_file) + 1  # 1-indexed
                best_rank = min(best_rank, rank)
            except ValueError:
                continue  # File not found in this query
        
        if best_rank != float('inf'):
            best_ranks[gold_file] = best_rank
            found_files.append(gold_file)
        else:
            best_ranks[gold_file] = 0  # Not found in any query
    
    # Calculate MRR from best ranks
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in best_ranks.values()]
    mrr = sum(reciprocal_ranks) / len(gold_files)
    
    metadata = {
        "gold_files": gold_files,
        "found_files": found_files,
        "best_ranks": best_ranks,
        "reciprocal_ranks": reciprocal_ranks,
        "num_gold_files": len(gold_files),
        "num_found": len(found_files),
        "best_rank": min([r for r in best_ranks.values() if r > 0]) if any(r > 0 for r in best_ranks.values()) else 0,
        "worst_rank": max([r for r in best_ranks.values() if r > 0]) if any(r > 0 for r in best_ranks.values()) else 0,
        "num_queries": len(query_results),
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
    Generate a single localization rollout with multi-query support.
    
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
        
        # Extract the queries and LLM call from the response
        queries = None
        num_queries = 0
        llm_call = None
        
        for step in new_tape.steps:
            if isinstance(step, LocalizationQuery):
                queries = step.queries
                num_queries = step.num_queries
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
        
        # Determine reward based on whether we have valid queries
        if queries is None or not queries:
            logger.warning("No localization queries found - treating as format violation")
            reward = -0.1  # Negative reward for format violation
            reward_metadata = {"error": "No queries found - format violation"}
            all_query_results = []
            format_violation = True
        else:
            # Normal case: we have queries, proceed with BM25 search
            try:
                all_file_stats = json.loads(problem.get("all_file_stats", "{}"))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Failed to parse all_file_stats, using empty dict")
                all_file_stats = {}
                
            if not all_file_stats:
                # No files to search - give zero reward
                reward = 0.0
                reward_metadata = {"error": "No file statistics available"}
                all_query_results = []
                logger.error("NO ALL FILE STATS.")
            else:
                # Create BM25 searcher and perform search for each query
                searcher = BM25Searcher(all_file_stats)
                
                # Split budget among queries
                budget_per_query = 100 // num_queries
                
                all_query_results = []
                for query in queries:
                    results = searcher.search(query, top_k=budget_per_query)
                    all_query_results.append(results)
                
                # Extract gold files from patch and calculate multi-query MRR reward
                gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
                reward, reward_metadata = calculate_multi_query_mrr(gold_files, all_query_results)
                
                # Apply small penalty for using multiple queries
                if num_queries > 1:
                    query_penalty = 0.01 * (num_queries - 1)
                    reward = max(0.0, reward - query_penalty)  # Don't go negative from penalty alone
                    reward_metadata["query_penalty"] = query_penalty
            
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
        
        # Calculate success metric (all gold files found in top 10 across all queries)
        top_10_files = set()
        for query_results in all_query_results:
            top_10_files.update([fp for fp, _ in query_results[:10]])
        
        gold_files = reward_metadata.get("gold_files", [])
        if gold_files:
            success = sum(1 for gf in gold_files if gf in top_10_files) / len(gold_files)
            success = success == 1  # True if all gold files are found in top 10
        else:
            success = False
        
        # Prepare metrics using the LocalizationMetrics class
        metrics = LocalizationMetrics(
            reward=reward,
            success=success,
            no_error=True,
            no_answer=queries is None or (queries and all(q.strip() == "" for q in queries)),
            mrr=reward if not format_violation else -1,
            num_queries=num_queries,
            total_query_length=sum(len(q.split()) for q in queries) if queries else 0,
            avg_query_length=sum(len(q.split()) for q in queries) / len(queries) if queries else 0,
            overflow=0 if finished else 1,
            num_search_results=sum(len(qr) for qr in all_query_results),
            prompt_tokens=llm_call.prompt_length_tokens,
            output_tokens=llm_call.output_length_tokens,
            format_violation=format_violation,
            localization_num_gold_files=reward_metadata.get("num_gold_files", 0),
            localization_num_found=reward_metadata.get("num_found", 0),
            localization_best_rank=reward_metadata.get("best_rank", 0),
            localization_worst_rank=reward_metadata.get("worst_rank", 0),
            localization_num_queries=reward_metadata.get("num_queries", 0),
        )
        
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
        
        # Return failed rollout with proper metrics class
        metrics = LocalizationMetrics(
            reward=0.0,
            success=False,
            no_error=False,
            no_answer=True,
            mrr=0.0,
            num_queries=0,
            total_query_length=0,
            avg_query_length=0.0,
            overflow=0,
            num_search_results=0,
            prompt_tokens=0,
            output_tokens=0,
            format_violation=False,
        )
        
        return RolloutResult(
            training_texts=[],
            metrics=metrics,
            latency=latency,
            dataset_name=problem.get("dataset"),
            prompt_tokens=[0],
            output_tokens=[0],
        )