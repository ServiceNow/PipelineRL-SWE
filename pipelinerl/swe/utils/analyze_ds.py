#!/usr/bin/env python3
"""
Quick script to analyze SWE-Gym dataset statistics.
Focuses on number of files edited per issue and other basic stats.
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from datasets import load_from_disk
from pathlib import Path

def parse_patch(patch_string: str) -> list[str]:
    """Extract file paths from git patch."""
    if not patch_string:
        return []
    return re.findall(r'^--- a/(.+)$', patch_string, re.MULTILINE)

def analyze_dataset(dataset_path: str):
    """Analyze the preprocessed SWE-Gym dataset."""
    
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} examples")
    
    # Basic stats
    print("\n=== BASIC DATASET STATS ===")
    
    # Repository distribution
    repo_counts = Counter(dataset['repo'])
    print(f"Unique repositories: {len(repo_counts)}")
    print(f"Top 5 repos by examples:")
    for repo, count in repo_counts.most_common(5):
        print(f"  {repo}: {count} examples")
    
    # Files edited per issue
    files_edited_counts = []
    single_file_issues = 0
    multi_file_issues = 0
    no_files_issues = 0
    
    # File extension analysis
    extension_counts = Counter()
    
    print("\nAnalyzing patches...")
    for example in dataset:
        patch = example.get('patch', '')
        gold_files = parse_patch(patch)
        
        num_files = len(gold_files)
        files_edited_counts.append(num_files)
        
        if num_files == 0:
            no_files_issues += 1
        elif num_files == 1:
            single_file_issues += 1
        else:
            multi_file_issues += 1
            
        # Analyze file extensions
        for filepath in gold_files:
            ext = Path(filepath).suffix.lower()
            if ext:
                extension_counts[ext] += 1
            else:
                extension_counts['<no_ext>'] += 1
    
    print("\n=== FILES EDITED PER ISSUE ===")
    print(f"Issues with 0 files: {no_files_issues} ({no_files_issues/len(dataset)*100:.1f}%)")
    print(f"Single-file issues: {single_file_issues} ({single_file_issues/len(dataset)*100:.1f}%)")
    print(f"Multi-file issues: {multi_file_issues} ({multi_file_issues/len(dataset)*100:.1f}%)")
    
    if files_edited_counts:
        print(f"\nFiles edited statistics:")
        print(f"  Mean: {np.mean(files_edited_counts):.2f}")
        print(f"  Median: {np.median(files_edited_counts):.1f}")
        print(f"  Max: {max(files_edited_counts)}")
        print(f"  95th percentile: {np.percentile(files_edited_counts, 95):.1f}")
        
        # Distribution breakdown
        file_count_dist = Counter(files_edited_counts)
        print(f"\nDistribution of files edited:")
        for num_files in sorted(file_count_dist.keys())[:10]:  # Show first 10
            count = file_count_dist[num_files]
            pct = count / len(dataset) * 100
            print(f"  {num_files} files: {count} issues ({pct:.1f}%)")
        
        if max(files_edited_counts) > 10:
            total_10_plus = sum(count for files, count in file_count_dist.items() if files > 10)
            print(f"  >10 files: {total_10_plus} issues ({total_10_plus/len(dataset)*100:.1f}%)")
    
    print("\n=== FILE EXTENSIONS ===")
    print("Top 10 file extensions:")
    for ext, count in extension_counts.most_common(10):
        print(f"  {ext}: {count} files")
    
    # Repository size analysis (if file stats available)
    print("\n=== REPOSITORY SIZES ===")
    repo_sizes = []
    examples_with_stats = 0
    
    for example in dataset:
        try:
            file_stats = json.loads(example.get('all_file_stats', '{}'))
            if file_stats:
                repo_sizes.append(len(file_stats))
                examples_with_stats += 1
        except:
            continue
    
    print(f"Examples with file stats: {examples_with_stats}/{len(dataset)}")
    
    if repo_sizes:
        print(f"Repository sizes (# of source files):")
        print(f"  Mean: {np.mean(repo_sizes):.1f}")
        print(f"  Median: {np.median(repo_sizes):.1f}")
        print(f"  Min: {min(repo_sizes)}")
        print(f"  Max: {max(repo_sizes)}")
        print(f"  95th percentile: {np.percentile(repo_sizes, 95):.1f}")
    
    # Gold file coverage analysis
    print("\n=== GOLD FILE COVERAGE ===")
    total_gold_files = 0
    covered_gold_files = 0
    
    for example in dataset:
        try:
            patch = example.get('patch', '')
            gold_files = parse_patch(patch)
            file_stats = json.loads(example.get('all_file_stats', '{}'))
            
            total_gold_files += len(gold_files)
            for gold_file in gold_files:
                if gold_file in file_stats:
                    covered_gold_files += 1
        except:
            continue
    
    if total_gold_files > 0:
        coverage_rate = covered_gold_files / total_gold_files * 100
        print(f"Gold file coverage: {covered_gold_files}/{total_gold_files} ({coverage_rate:.1f}%)")
    
    # Create histograms
    create_histograms(files_edited_counts, repo_sizes)
    
    return {
        'total_examples': len(dataset),
        'single_file_pct': single_file_issues / len(dataset) * 100,
        'multi_file_pct': multi_file_issues / len(dataset) * 100,
        'files_edited_stats': {
            'mean': np.mean(files_edited_counts) if files_edited_counts else 0,
            'median': np.median(files_edited_counts) if files_edited_counts else 0,
            'max': max(files_edited_counts) if files_edited_counts else 0,
        },
        'repo_count': len(repo_counts),
        'coverage_rate': covered_gold_files / total_gold_files * 100 if total_gold_files > 0 else 0
    }

def create_histograms(files_edited_counts, repo_sizes):
    """Create histogram visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Files edited histogram
    if files_edited_counts:
        axes[0, 0].hist(files_edited_counts, bins=range(0, min(21, max(files_edited_counts)+2)), 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Number of Files Edited')
        axes[0, 0].set_ylabel('Number of Issues')
        axes[0, 0].set_title('Distribution of Files Edited per Issue')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Files edited (log scale)
    if files_edited_counts:
        axes[0, 1].hist(files_edited_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Files Edited')
        axes[0, 1].set_ylabel('Number of Issues (log scale)')
        axes[0, 1].set_title('Files Edited Distribution (Log Scale)')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Repository sizes
    if repo_sizes:
        axes[1, 0].hist(repo_sizes, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Repository Size (# source files)')
        axes[1, 0].set_ylabel('Number of Examples')
        axes[1, 0].set_title('Repository Size Distribution')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Repository sizes (log scale)
    if repo_sizes:
        axes[1, 1].hist(repo_sizes, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Repository Size (# source files)')
        axes[1, 1].set_ylabel('Number of Examples (log scale)')
        axes[1, 1].set_title('Repository Size Distribution (Log Scale)')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('swe_dataset_stats.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nHistograms saved as 'swe_dataset_stats.png'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python swe_dataset_stats.py <path_to_processed_dataset>")
        print("Example: python swe_dataset_stats.py ./processed_swe_dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        sys.exit(1)
    
    try:
        stats = analyze_dataset(dataset_path)
        print(f"\n=== SUMMARY ===")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Single-file issues: {stats['single_file_pct']:.1f}%")
        print(f"Multi-file issues: {stats['multi_file_pct']:.1f}%")
        print(f"Average files per issue: {stats['files_edited_stats']['mean']:.2f}")
        print(f"Gold file coverage: {stats['coverage_rate']:.1f}%")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)