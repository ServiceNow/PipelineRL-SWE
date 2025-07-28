#!/usr/bin/env python3
"""
Script to analyze context usage for different localization approaches.
Helps determine what fits in the 20K token budget.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datasets import load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def create_system_prompt() -> str:
    """Create the system prompt for localization."""
    return (
        "You are an expert software engineer tasked with finding relevant files in a codebase "
        "to fix a given issue. Your job is to generate a search query that will help locate "
        "the most relevant files using keyword search.\n\n"
        "Generate a search query containing keywords, function names, class names, or other "
        "terms that would help find files related to the issue. Focus on technical terms "
        "that are likely to appear in the relevant source code.\n\n"
        "Output only the search query, nothing else."
    )


def create_user_prompt_basic(problem_statement: str) -> str:
    """Create basic user prompt with just problem statement."""
    return (
        f"Generate a search query to find files relevant to this issue:\n\n"
        f"{problem_statement}\n\n"
        f"Search query:"
    )


def create_user_prompt_with_files(problem_statement: str, file_list: List[str]) -> str:
    """Create user prompt with problem statement + file list."""
    files_text = "\n".join(file_list)
    return (
        f"Repository files:\n{files_text}\n\n"
        f"Generate a search query to find files relevant to this issue:\n\n"
        f"{problem_statement}\n\n"
        f"Search query:"
    )


def create_user_prompt_with_file_terms(problem_statement: str, file_stats: Dict[str, Dict]) -> str:
    """Create user prompt with problem statement + file list + top terms."""
    file_info_lines = []
    
    for filepath, stats in file_stats.items():
        term_counts = stats.get('term_counts', {})
        if not term_counts:
            file_info_lines.append(f"{filepath}")
        else:
            # Get top 5 terms by frequency
            top_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            terms_str = ",".join([term for term, _ in top_terms])
            file_info_lines.append(f"{filepath}: {terms_str}")
    
    files_text = "\n".join(file_info_lines)
    return (
        f"Repository files and common terms:\n{files_text}\n\n"
        f"Generate a search query to find files relevant to this issue:\n\n"
        f"{problem_statement}\n\n"
        f"Search query:"
    )


def filter_source_files(file_stats: Dict[str, Dict]) -> Dict[str, Dict]:
    """Filter to only include likely source files."""
    source_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs'}
    skip_dirs = {'test', 'tests', '__pycache__', '.git', 'node_modules', 'build', 'dist', 'docs', 'doc'}
    
    filtered = {}
    for filepath, stats in file_stats.items():
        # Check extension
        if not any(filepath.endswith(ext) for ext in source_extensions):
            continue
            
        # Check if in skip directory
        path_parts = filepath.split('/')
        if any(part.lower() in skip_dirs for part in path_parts):
            continue
            
        filtered[filepath] = stats
    
    return filtered


def analyze_context_usage(dataset_path: str, model_name: str = "Qwen/Qwen2.5-7B", num_samples: int = 10):
    """Analyze context usage for different prompting strategies."""
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Take a sample of examples
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    results = {
        'basic': [],
        'with_files': [],
        'with_filtered_files': [],
        'with_file_terms': [],
        'with_filtered_file_terms': []
    }
    
    system_prompt = create_system_prompt()
    system_tokens = count_tokens(system_prompt, tokenizer)
    
    logger.info(f"System prompt tokens: {system_tokens}")
    
    for i, example in enumerate(samples):
        problem_statement = example['problem_statement']
        
        try:
            all_file_stats = json.loads(example.get('all_file_stats', '{}'))
        except:
            logger.warning(f"Could not parse all_file_stats for example {i}")
            continue
            
        if not all_file_stats:
            logger.warning(f"Empty file stats for example {i}")
            continue
            
        file_list = list(all_file_stats.keys())
        filtered_file_stats = filter_source_files(all_file_stats)
        filtered_file_list = list(filtered_file_stats.keys())
        
        # Strategy 1: Basic (problem statement only)
        basic_prompt = create_user_prompt_basic(problem_statement)
        basic_tokens = system_tokens + count_tokens(basic_prompt, tokenizer)
        results['basic'].append(basic_tokens)
        
        # Strategy 2: With all files
        files_prompt = create_user_prompt_with_files(problem_statement, file_list)
        files_tokens = system_tokens + count_tokens(files_prompt, tokenizer)
        results['with_files'].append(files_tokens)
        
        # Strategy 3: With filtered files only
        filtered_prompt = create_user_prompt_with_files(problem_statement, filtered_file_list)
        filtered_tokens = system_tokens + count_tokens(filtered_prompt, tokenizer)
        results['with_filtered_files'].append(filtered_tokens)
        
        # Strategy 4: With all files + terms
        terms_prompt = create_user_prompt_with_file_terms(problem_statement, all_file_stats)
        terms_tokens = system_tokens + count_tokens(terms_prompt, tokenizer)
        results['with_file_terms'].append(terms_tokens)
        
        # Strategy 5: With filtered files + terms
        filtered_terms_prompt = create_user_prompt_with_file_terms(problem_statement, filtered_file_stats)
        filtered_terms_tokens = system_tokens + count_tokens(filtered_terms_prompt, tokenizer)
        results['with_filtered_file_terms'].append(filtered_terms_tokens)
        
        logger.info(f"Example {i+1}: {len(file_list)} total files, {len(filtered_file_list)} filtered files")
    
    # Print results
    print("\n" + "="*60)
    print("CONTEXT USAGE ANALYSIS")
    print("="*60)
    print(f"Budget: 20,000 tokens")
    print(f"Analyzed {len(samples)} examples")
    print()
    
    for strategy, token_counts in results.items():
        if not token_counts:
            continue
            
        avg_tokens = sum(token_counts) / len(token_counts)
        max_tokens = max(token_counts)
        min_tokens = min(token_counts)
        over_budget = sum(1 for t in token_counts if t > 20000)
        
        print(f"{strategy.upper().replace('_', ' ')}:")
        print(f"  Average: {avg_tokens:,.0f} tokens")
        print(f"  Range: {min_tokens:,.0f} - {max_tokens:,.0f} tokens")
        print(f"  Over budget: {over_budget}/{len(token_counts)} examples")
        print(f"  Fits budget: {'✓' if max_tokens <= 20000 else '✗'}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_context_usage.py <dataset_path>")
        print("Example: python analyze_context_usage.py /path/to/swegym/dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    analyze_context_usage(dataset_path)