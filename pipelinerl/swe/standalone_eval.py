#!/usr/bin/env python3
"""
Standalone SWE Pipeline Evaluation Script

Evaluates localization, file selection, and repair agents on SWE dataset
using TapeAgents/LiteLLM with OpenRouter or other providers.

Usage:
    python evaluate_swe.py --dataset_path /path/to/dataset --model gpt-4o-mini \
                          --api_key your_key --base_url https://openrouter.ai/api/v1 \
                          --num_examples 100

Dependencies:
    pip install datasets tapeagents litellm tenacity tqdm
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from datasets import load_from_disk
from tapeagents.llms import LiteLLM
from tapeagents.orchestrator import main_loop
from tqdm import tqdm

# Import SWE agents and utilities
from pipelinerl.swe.agents.localization_agent import (
    LocalizationAgent, LocalizationTask, LocalizationTape, LocalizationQuery
)
from pipelinerl.swe.agents.file_selection_agent import (
    FileSelectionAgent, FileSelectionTask, FileSelectionTape, FileSelectionResponse
)
from pipelinerl.swe.agents.repair_agent import (
    RepairAgent, RepairTask, RepairTape, SearchReplaceResponse
)
from pipelinerl.swe.utils.bm25_searcher import BM25Searcher
from pipelinerl.swe.utils.file_context_enricher import FileContextEnricher
from pipelinerl.swe.utils.localization_utils import (
    parse_patch_for_gold_files, calculate_multi_query_mrr, calculate_recall_precision
)
from pipelinerl.swe.utils.repair_utils import calculate_precise_reward

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_requests: int = 20, time_window: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # If we're at the limit, wait until we can make another request
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request falls outside the window
            wait_time = self.time_window - (now - self.requests[0]) + 0.1  # Small buffer
            logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            # Remove the old request
            self.requests = self.requests[1:]
        
        # Record this request
        self.requests.append(now)


class RunningStats:
    """Track running statistics for tqdm display"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_examples = 0
        self.successful_examples = 0
        
        # Localization stats
        self.loc_successes = 0
        self.loc_mrr_sum = 0.0
        self.loc_recall_sum = 0.0
        
        # File selection stats
        self.sel_successes = 0
        self.sel_precision_sum = 0.0
        self.sel_recall_sum = 0.0
        self.sel_f1_sum = 0.0
        
        # Repair stats
        self.repair_successes = 0
        self.repair_reward_sum = 0.0
        
        # Timing
        self.total_time = 0.0
    
    def update(self, result: Dict):
        """Update stats with a new result"""
        self.total_examples += 1
        
        if result.get('overall_success', False):
            self.successful_examples += 1
        
        # Localization
        if result['localization'].get('success', False):
            self.loc_successes += 1
            self.loc_mrr_sum += result['localization']['metrics'].get('mrr', 0)
            self.loc_recall_sum += result['localization']['metrics'].get('recall', 0)
        
        # File selection
        if result['file_selection'].get('success', False):
            self.sel_successes += 1
            self.sel_precision_sum += result['file_selection']['metrics'].get('precision', 0)
            self.sel_recall_sum += result['file_selection']['metrics'].get('recall', 0)
            self.sel_f1_sum += result['file_selection']['metrics'].get('f1', 0)
        
        # Repair
        if result['repair'].get('success', False):
            self.repair_successes += 1
            self.repair_reward_sum += result['repair']['metrics'].get('reward', 0)
        
        # Timing
        self.total_time += result.get('total_time', 0)
    
    def get_averages(self) -> Dict:
        """Get current running averages"""
        if self.total_examples == 0:
            return {}
        
        return {
            'overall_success_rate': self.successful_examples / self.total_examples,
            'loc_success_rate': self.loc_successes / self.total_examples,
            'loc_avg_mrr': self.loc_mrr_sum / max(1, self.loc_successes),
            'loc_avg_recall': self.loc_recall_sum / max(1, self.loc_successes),
            'sel_success_rate': self.sel_successes / self.total_examples,
            'sel_avg_precision': self.sel_precision_sum / max(1, self.sel_successes),
            'sel_avg_recall': self.sel_recall_sum / max(1, self.sel_successes),
            'sel_avg_f1': self.sel_f1_sum / max(1, self.sel_successes),
            'repair_success_rate': self.repair_successes / self.total_examples,
            'repair_avg_reward': self.repair_reward_sum / max(1, self.repair_successes),
            'avg_time_per_example': self.total_time / self.total_examples
        }
    
    def format_for_tqdm(self) -> str:
        """Format current stats for tqdm postfix"""
        avgs = self.get_averages()
        if not avgs:
            return ""
        
        return (f"Overall: {avgs['overall_success_rate']:.1%} | "
                f"Loc: MRR={avgs['loc_avg_mrr']:.3f} | "
                f"Sel: F1={avgs['sel_avg_f1']:.3f} | "
                f"Repair: R={avgs['repair_avg_reward']:.3f}")


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    # Overall stats
    total_examples: int = 0
    successful_examples: int = 0
    
    # Localization metrics
    localization_mrr: float = 0.0
    localization_recall: float = 0.0
    localization_success_rate: float = 0.0
    
    # File selection metrics  
    selection_precision: float = 0.0
    selection_recall: float = 0.0
    selection_f1: float = 0.0
    selection_success_rate: float = 0.0
    
    # Repair metrics
    repair_reward: float = 0.0
    repair_success_rate: float = 0.0
    
    # Timing
    avg_time_per_example: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EmptyEnvironment:
    """No-op environment for agent execution"""
    
    def react(self, tape):
        return tape


def run_localization_stage(
    agent: LocalizationAgent, 
    problem: Dict, 
    rate_limiter: RateLimiter
) -> Dict:
    """Run localization stage and return results"""
    
    try:
        file_stats = json.loads(problem.get('all_file_stats', '{}'))
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse file stats for localization")
        return {
            'queries': [],
            'top_files': [],
            'metrics': {'error': 'parse_failure'},
            'success': False
        }
    
    if not file_stats:
        return {
            'queries': [],
            'top_files': [],
            'metrics': {'error': 'no_file_stats'},
            'success': False
        }
    
    # Create task and tape
    task = LocalizationTask(
        problem_statement=problem["problem_statement"],
        file_stats=file_stats
    )
    tape = LocalizationTape(steps=[task], context=None)
    
    try:
        # Rate limit before API call
        rate_limiter.acquire()
        
        # Run agent
        for event in main_loop(agent, tape, EmptyEnvironment(), max_loops=1):
            if event.agent_event and event.agent_event.final_tape:
                tape = event.agent_event.final_tape
                break
        
        # Extract results
        queries = []
        for step in tape.steps:
            if isinstance(step, LocalizationQuery):
                queries = step.queries
                break
        
        if not queries:
            return {
                'queries': [],
                'top_files': [],
                'metrics': {'error': 'no_queries'},
                'success': False
            }
        
        # Run BM25 search
        searcher = BM25Searcher(file_stats)
        budget_per_query = max(1, 10 // len(queries))
        
        all_results = []
        for query in queries:
            results = searcher.search(query, top_k=budget_per_query)
            all_results.append(results)
        
        # Combine results
        file_scores = {}
        for query_results in all_results:
            for filepath, score in query_results:
                if filepath not in file_scores or score > file_scores[filepath]:
                    file_scores[filepath] = score
        
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        top_files = [filepath for filepath, _ in sorted_files[:10]]
        
        # Calculate metrics
        gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
        mrr, metadata = calculate_multi_query_mrr(gold_files, all_results)
        recall, precision, f1 = calculate_recall_precision(gold_files, all_results, k=10)
        
        return {
            'queries': queries,
            'top_files': top_files,
            'metrics': {
                'mrr': mrr,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                **metadata
            },
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Localization error: {e}")
        return {
            'queries': [],
            'top_files': [],
            'metrics': {'error': str(e)},
            'success': False
        }


def run_file_selection_stage(
    agent: FileSelectionAgent,
    problem: Dict,
    top_files: List[str],
    enricher: FileContextEnricher,
    repo_base_path: str,
    rate_limiter: RateLimiter
) -> Dict:
    """Run file selection stage and return results"""
    
    if not top_files:
        return {
            'selected_files': [],
            'files_for_repair': [],
            'metrics': {'error': 'no_input_files'},
            'success': False
        }
    
    try:
        # Get repo info for real enrichment
        dataset = problem.get('dataset', '')
        repo_name = problem.get('repo', '')
        base_commit = problem.get('base_commit', '')
        
        if dataset and repo_name and base_commit:
            # Construct repo path (same logic as in rollouts.py)
            if dataset == 'swegym':
                repo_path = Path(repo_base_path) / "swegym/repos" / Path(repo_name.replace("/", "_"))
            elif dataset == 'swebench_lite':
                repo_path = Path(repo_base_path) / "swebench_lite/repos" / Path(repo_name.replace("/", "_"))
            else:
                repo_path = Path(repo_base_path) / Path(repo_name.replace("/", "_"))
            
            # Do real enrichment with actual file contents
            enriched_context = enricher.enrich_files_on_demand(top_files, repo_path, base_commit)
        else:
            logger.warning("Missing repo info for enrichment, using dummy context")
            # Fallback to dummy enrichment
            enriched_context = {}
            for filepath in top_files:
                enriched_context[filepath] = {
                    'summary': f"Source file: {filepath}",
                    'functions': [],
                    'classes': [],
                    'imports': [],
                    'content': ''
                }
        
        # Create task and tape
        task = FileSelectionTask(
            problem_statement=problem["problem_statement"],
            candidate_files=enriched_context
        )
        tape = FileSelectionTape(steps=[task], context=None)
        
        # Rate limit before API call
        rate_limiter.acquire()
        
        # Run agent
        for event in main_loop(agent, tape, EmptyEnvironment(), max_loops=1):
            if event.agent_event and event.agent_event.final_tape:
                tape = event.agent_event.final_tape
                break
        
        # Extract results
        selected_files = []
        for step in tape.steps:
            if isinstance(step, FileSelectionResponse):
                selected_files = step.selected_files
                break
        
        if not selected_files:
            return {
                'selected_files': [],
                'files_for_repair': [],
                'metrics': {'error': 'no_files_selected'},
                'success': False
            }
        
        # Calculate metrics
        gold_files = parse_patch_for_gold_files(problem.get("patch", ""))
        
        if gold_files:
            relevant_selected = set(selected_files) & set(gold_files)
            precision = len(relevant_selected) / len(selected_files) if selected_files else 0
            recall = len(relevant_selected) / len(gold_files) if gold_files else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
        
        # Oracle injection: if selection differs from gold files, use gold files for downstream
        files_for_repair = gold_files if set(selected_files) != set(gold_files) else selected_files
        
        return {
            'selected_files': selected_files,
            'files_for_repair': files_for_repair,  # This is what goes to repair stage
            'enriched_context': enriched_context,  # Pass enriched context to repair
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_selected': len(selected_files),
                'num_gold': len(gold_files),
                'oracle_injected': files_for_repair != selected_files
            },
            'success': True
        }
        
    except Exception as e:
        logger.error(f"File selection error: {e}")
        return {
            'selected_files': [],
            'files_for_repair': [],
            'enriched_context': {},
            'metrics': {'error': str(e)},
            'success': False
        }


def run_repair_stage(
    agent: RepairAgent,
    problem: Dict,
    files_for_repair: List[str],
    enriched_context: Dict,
    rate_limiter: RateLimiter
) -> Dict:
    """Run repair stage and return results"""
    
    if not files_for_repair:
        return {
            'edits': [],
            'metrics': {'error': 'no_input_files'},
            'success': False
        }
    
    try:
        # Get file contents - prefer enriched context if available, fallback to preprocessed data
        file_contents = {}
        
        # First try to get content from enriched context (real files)
        for filepath in files_for_repair:
            if filepath in enriched_context and 'content' in enriched_context[filepath]:
                content = enriched_context[filepath]['content']
                if content:  # Only use if not empty
                    file_contents[filepath] = content
        
        # Fallback to preprocessed gold file contents if needed
        if not file_contents:
            try:
                all_file_contents = json.loads(problem.get('gold_file_contents', '{}'))
            except (json.JSONDecodeError, TypeError):
                all_file_contents = problem.get('file_contents', {})
            
            for filepath in files_for_repair:
                if filepath in all_file_contents:
                    file_contents[filepath] = all_file_contents[filepath]
        
        if not file_contents:
            return {
                'edits': [],
                'metrics': {'error': 'no_file_contents'},
                'success': False
            }
        
        # Create task and tape
        task = RepairTask(
            problem_statement=problem["problem_statement"],
            file_contents=file_contents
        )
        tape = RepairTape(steps=[task], context=None)
        
        # Rate limit before API call
        rate_limiter.acquire()
        
        # Run agent
        for event in main_loop(agent, tape, EmptyEnvironment(), max_loops=1):
            if event.agent_event and event.agent_event.final_tape:
                tape = event.agent_event.final_tape
                break
        
        # Extract results
        edits = []
        for step in tape.steps:
            if isinstance(step, SearchReplaceResponse):
                edits = step.edits
                break
        
        # Calculate reward
        gold_patch = problem.get("patch", "")
        
        if edits and gold_patch:
            reward, metadata = calculate_precise_reward(file_contents, gold_patch, edits)
        else:
            reward = 0
            metadata = {'format_error': True if not edits else False}
        
        return {
            'edits': edits,
            'metrics': {
                'reward': reward,
                'num_edits': len(edits),
                **metadata
            },
            'success': reward > 0.5  # Success threshold
        }
        
    except Exception as e:
        logger.error(f"Repair error: {e}")
        return {
            'edits': [],
            'metrics': {'error': str(e)},
            'success': False
        }


def evaluate_example(
    example: Dict,
    agents: Dict,
    enricher: FileContextEnricher,
    repo_base_path: str,
    rate_limiter: RateLimiter,
    skip_repair: bool = False
) -> Dict:
    """Evaluate a single example through the full pipeline"""
    
    start_time = time.time()
    results = {
        'localization': {},
        'file_selection': {},
        'repair': {},
        'overall_success': False,
        'total_time': 0
    }
    
    try:
        # Stage 1: Localization with oracle injection
        loc_result = run_localization_stage(agents['localization'], example, rate_limiter)
        results['localization'] = loc_result
        
        top_files = loc_result.get('top_files', [])
        
        # Oracle injection for localization: inject missing gold files
        gold_files = parse_patch_for_gold_files(example.get("patch", ""))
        if gold_files:
            oracle_not_found = [f for f in gold_files if f not in set(top_files)]
            if oracle_not_found:
                # Add oracle files and remove some non-oracle files to maintain budget
                remaining_slots = max(0, 10 - len(oracle_not_found))
                non_oracle_files = [f for f in top_files if f not in set(gold_files)]
                top_files = oracle_not_found + non_oracle_files[:remaining_slots]
        
        if not top_files:
            # Complete fallback to gold files
            top_files = gold_files[:10]
        
        # Stage 2: File Selection
        sel_result = run_file_selection_stage(
            agents['file_selection'], example, top_files, enricher, repo_base_path, rate_limiter
        )
        results['file_selection'] = sel_result
        
        # Get files for repair (includes oracle injection logic)
        files_for_repair = sel_result.get('files_for_repair', [])
        enriched_context = sel_result.get('enriched_context', {})
        
        # Stage 3: Repair (optional)
        if not skip_repair:
            repair_result = run_repair_stage(
                agents['repair'], example, files_for_repair, enriched_context, rate_limiter
            )
            results['repair'] = repair_result
            
            # Overall success: all stages successful
            results['overall_success'] = (
                loc_result.get('success', False) and
                sel_result.get('success', False) and
                repair_result.get('success', False)
            )
        else:
            # Skip repair stage
            results['repair'] = {
                'edits': [],
                'metrics': {'skipped': True},
                'success': True  # Consider skipped as successful for overall calculation
            }
            
            # Overall success: just localization and file selection
            results['overall_success'] = (
                loc_result.get('success', False) and
                sel_result.get('success', False)
            )
        
        results['total_time'] = time.time() - start_time
        
    except Exception as e:
        logger.error(f"Error evaluating example: {e}")
        results['error'] = str(e)
        results['total_time'] = time.time() - start_time
    
    return results


def run_evaluation(
    dataset_path: str,
    model_config: Dict,
    repo_base_path: str = "/mnt/llmd/data",
    max_requests_per_minute: int = 20,
    num_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    skip_repair: bool = False
) -> EvaluationResults:
    """Run evaluation on the dataset"""
    
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    logger.info(f"Evaluating {len(dataset)} examples")
    
    # Initialize LLM
    llm = LiteLLM(**model_config)
    
    # Initialize agents
    agents = {}
    agents['localization'] = LocalizationAgent.create(llm=llm, max_prompt_length=8000)
    agents['file_selection'] = FileSelectionAgent.create(llm=llm, max_prompt_length=16000)
    
    if not skip_repair:
        agents['repair'] = RepairAgent.create(llm=llm, max_prompt_length=16000)
    
    # Initialize enricher and rate limiter
    enricher = FileContextEnricher()
    rate_limiter = RateLimiter(max_requests=max_requests_per_minute, time_window=60)
    
    # Initialize running stats for tqdm
    running_stats = RunningStats()
    
    # Collect results
    all_results = []
    
    # Create progress bar with running stats
    pbar = tqdm(total=len(dataset), desc="Evaluating SWE Pipeline")
    
    for i, example in enumerate(dataset):
        example_id = example.get('id', f'example_{i}')
        pbar.set_description(f"Evaluating {example_id}")
        
        result = evaluate_example(
            example, agents, enricher, repo_base_path, rate_limiter, skip_repair
        )
        all_results.append(result)
        
        # Update running stats and progress bar
        running_stats.update(result)
        pbar.set_postfix_str(running_stats.format_for_tqdm())
        pbar.update(1)
    
    pbar.close()
    
    # Calculate aggregate metrics
    total_examples = len(all_results)
    successful_examples = sum(1 for r in all_results if r.get('overall_success', False))
    
    # Localization metrics
    loc_mrrs = [r['localization']['metrics'].get('mrr', 0) for r in all_results 
                if r['localization'].get('success', False)]
    loc_recalls = [r['localization']['metrics'].get('recall', 0) for r in all_results 
                   if r['localization'].get('success', False)]
    
    # File selection metrics
    sel_precisions = [r['file_selection']['metrics'].get('precision', 0) for r in all_results 
                      if r['file_selection'].get('success', False)]
    sel_recalls = [r['file_selection']['metrics'].get('recall', 0) for r in all_results 
                   if r['file_selection'].get('success', False)]
    sel_f1s = [r['file_selection']['metrics'].get('f1', 0) for r in all_results 
               if r['file_selection'].get('success', False)]
    
    # Repair metrics (only if repair was not skipped)
    repair_rewards = [r['repair']['metrics'].get('reward', 0) for r in all_results 
                      if r['repair'].get('success', False) and not r['repair']['metrics'].get('skipped', False)]
    
    # Timing
    times = [r.get('total_time', 0) for r in all_results]
    
    results = EvaluationResults(
        total_examples=total_examples,
        successful_examples=successful_examples,
        
        localization_mrr=sum(loc_mrrs) / len(loc_mrrs) if loc_mrrs else 0,
        localization_recall=sum(loc_recalls) / len(loc_recalls) if loc_recalls else 0,
        localization_success_rate=len(loc_mrrs) / total_examples,
        
        selection_precision=sum(sel_precisions) / len(sel_precisions) if sel_precisions else 0,
        selection_recall=sum(sel_recalls) / len(sel_recalls) if sel_recalls else 0,
        selection_f1=sum(sel_f1s) / len(sel_f1s) if sel_f1s else 0,
        selection_success_rate=len(sel_precisions) / total_examples,
        
        repair_reward=sum(repair_rewards) / len(repair_rewards) if repair_rewards else 0,
        repair_success_rate=len(repair_rewards) / total_examples,
        
        avg_time_per_example=sum(times) / len(times) if times else 0
    )
    
    # Save detailed results if requested
    if output_file:
        detailed_output = {
            'summary': results.to_dict(),
            'per_example_results': all_results
        }
        with open(output_file, 'w') as f:
            json.dump(detailed_output, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWE Pipeline")
    parser.add_argument("--dataset_path", required=True, help="Path to preprocessed SWE dataset")
    parser.add_argument("--repo_base_path", default="/mnt/llmd/data", help="Base path for repositories")
    parser.add_argument("--max_requests_per_minute", type=int, default=20, help="Rate limit for API calls")
    parser.add_argument("--skip_repair", action="store_true", help="Skip the repair stage (only run localization and file selection)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--api_key", help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", help="Base URL for API (e.g., OpenRouter)")
    parser.add_argument("--num_examples", type=int, help="Number of examples to evaluate")
    parser.add_argument("--output_file", help="Path to save detailed results")
    parser.add_argument("--temperature", type=float, default=0.6, help="Model temperature")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens")
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset_path).exists():
        raise ValueError(f"Dataset path not found: {args.dataset_path}")
    
    # Model configuration
    model_config = {
        "model_name": args.model,
        "parameters": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        }
    }
    
    if args.api_key:
        model_config["api_key"] = args.api_key
    if args.base_url:
        model_config["base_url"] = args.base_url
    
    # Run evaluation
    logger.info("Starting SWE Pipeline Evaluation")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset_path}")
    
    results = run_evaluation(
        args.dataset_path,
        model_config,
        args.repo_base_path,
        args.max_requests_per_minute,
        args.num_examples,
        args.output_file,
        args.skip_repair
    )
    
    # Print results
    print("\n" + "="*60)
    print("SWE PIPELINE EVALUATION RESULTS")
    print("="*60)
    print(f"Total Examples: {results.total_examples}")
    print(f"Overall Success Rate: {results.successful_examples/results.total_examples:.1%}")
    print(f"Average Time per Example: {results.avg_time_per_example:.1f}s")
    print()
    
    print("LOCALIZATION STAGE:")
    print(f"  Success Rate: {results.localization_success_rate:.1%}")
    print(f"  Mean Reciprocal Rank: {results.localization_mrr:.3f}")
    print(f"  Recall@10: {results.localization_recall:.3f}")
    print()
    
    print("FILE SELECTION STAGE:")
    print(f"  Success Rate: {results.selection_success_rate:.1%}")
    print(f"  Precision: {results.selection_precision:.3f}")
    print(f"  Recall: {results.selection_recall:.3f}")
    print(f"  F1 Score: {results.selection_f1:.3f}")
    print()
    
    print("REPAIR STAGE:")
    print(f"  Success Rate: {results.repair_success_rate:.1%}")
    print(f"  Average Reward: {results.repair_reward:.3f}")
    print()


if __name__ == "__main__":
    main()