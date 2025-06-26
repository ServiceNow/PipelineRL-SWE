import json
import logging
from datasets import load_from_disk
from omegaconf import DictConfig
from typing import List, Dict
from hydra.utils import instantiate
from pathlib import Path

logger = logging.getLogger(__name__)

def load_swegym_dataset(dataset_names, dataset_path) -> list[dict]:
    """
    Load the preprocessed SWE-Gym dataset from filesystem path in config.
    
    Args:
        
    Returns:
        List of dictionaries containing SWE repair tasks
    """
    try:
        if not dataset_path:
            logger.error("SWE-Gym dataset path not found in config or environment")
            return []
            
        logger.info(f"Loading SWE-Gym dataset from {dataset_path}")
        
        # Load dataset directly from disk
        dataset = load_from_disk(dataset_path)
        logger.info(f"SWE-Gym dataset loaded with {len(dataset)} examples")
        
        # Process the dataset into the expected format
        samples = []
        for item in dataset:
            try:
                file_contents = json.loads(item.get("gold_file_contents", "{}"))
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse 'gold_file_contents' for an item. Skipping.")
                continue
                
            # Skip items with no file contents
            if not file_contents:
                continue
            
            # Format compatible with convert_swe_problems_to_tapes
            samples.append({
                "id": item.get("issue_id", "") or item.get("id", ""),
                "dataset": "swegym",
                "repo": item.get("repo", ""),
                "base_commit": item.get("base_commit", ""),
                "problem_statement": item.get("problem_statement"),
                "patch": item.get("patch"),
                "file_contents": file_contents,  # Pre-loaded file contents
                "all_file_stats": item['all_file_stats']
            })
            
        logger.info(f"Processed {len(samples)} valid SWE-Gym samples")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading SWE-Gym dataset: {e}", exc_info=True)
        return []

def process_swebench_lite(dataset_names, dataset_path):
    """
    Process the SWE-Bench Lite dataset.
    
    Args:
        dataset: The loaded SWE-Bench Lite dataset.
        
    Yields:
        Processed samples in the expected format.
    """
        
    if not dataset_path:
        logger.error("SWE-bench dataset path not found in config or environment")
        return []
        
    logger.info(f"Loading SWE-bench dataset from {dataset_path}")
    
    # Load dataset directly from disk
    dataset = load_from_disk(dataset_path)
    logger.info(f"SWE-Gym dataset loaded with {len(dataset)} examples")
    samples = []
    for item in dataset:
        file_contents = json.loads(item["gold_file_contents"])
        samples.append({
            "id": item["instance_id"],
            "dataset": "swebench_lite",
            "repo": item["repo"],
            "base_commit": item["base_commit"],
            "problem_statement": item["problem_statement"],
            "patch": item["patch"],
            "file_contents": file_contents,
            "all_file_stats": item["all_file_stats"]
        })
    return samples