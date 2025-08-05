import json
import time
import logging
import math
from typing import Dict, List, Tuple

import aiohttp
from omegaconf import DictConfig
from tapeagents.orchestrator import async_execute_agent
from tenacity import retry, stop_after_attempt, AsyncRetrying, RetryError

from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.async_llm import make_training_text
from tapeagents.llms.trainable import TrainableLLM
from pipelinerl.swe.agents.localization_agent import LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
from pipelinerl.swe.utils.bm25_searcher import BM25Searcher

logger = logging.getLogger(__name__)

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


def calculate_recall_precision(gold_files: List[str], query_results: List[List[Tuple[str, float]]], k: int = 10) -> Tuple[float, float, float]:
    """
    Calculate recall, precision, and F1 score for the search results.
    
    Args:
        gold_files: List of gold file paths that should be found
        query_results: List of query results, where each is a list of (filepath, score) tuples
        k: Cutoff for precision calculation (default: 10, considers top-k results)
        
    Returns:
        Tuple of (recall, precision, f1_score)
    """
    if not gold_files:
        return 0.0, 0.0, 0.0
        
    if not query_results or not any(query_results):
        return 0.0, 0.0, 0.0
    
    # Combine all query results and deduplicate
    all_results = []
    for query_result in query_results:
        all_results.extend(query_result)
    
    # Deduplicate, keeping best score per file
    file_scores = {}
    for filepath, score in all_results:
        if filepath not in file_scores or score > file_scores[filepath]:
            file_scores[filepath] = score
    
    # Sort by score and take top-k for precision calculation
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    retrieved_files = [filepath for filepath, _ in sorted_files]
    
    # Calculate recall: fraction of relevant items that were retrieved
    relevant_retrieved = set(gold_files) & set(retrieved_files)
    recall = len(relevant_retrieved) / len(gold_files) if gold_files else 0.0
    
    # Calculate precision: fraction of retrieved items that are relevant
    precision = len(relevant_retrieved) / len(retrieved_files) if retrieved_files else 0.0
    
    # Calculate F1 score: harmonic mean of precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return recall, precision, f1_score


def calculate_ndcg_at_k(gold_files: List[str], query_results: List[List[Tuple[str, float]]], k: int = 10) -> float:
    """
    Calculate NDCG@k by concatenating and deduplicating results from multiple queries.
    
    Args:
        gold_files: List of gold file paths that should be found
        query_results: List of query results, where each is a list of (filepath, score) tuples  
        k: Cutoff for NDCG calculation (default: 10)
        
    Returns:
        NDCG@k score
    """
    if not gold_files:
        return 0.0
        
    if not query_results or not any(query_results):
        return 0.0
    
    # Combine all query results
    all_results = []
    for query_result in query_results:
        all_results.extend(query_result)
    
    # Deduplicate, keeping best score per file
    file_scores = {}
    for filepath, score in all_results:
        if filepath not in file_scores or score > file_scores[filepath]:
            file_scores[filepath] = score
    
    # Sort by score and take top-k
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Calculate DCG@k
    dcg = 0
    for i, (filepath, _) in enumerate(sorted_files):
        relevance = 1 if filepath in gold_files else 0
        dcg += relevance / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate ideal DCG@k (all relevant docs at top)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold_files), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


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