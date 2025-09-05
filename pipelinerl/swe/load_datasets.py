import json
import logging
import random
from datasets import load_from_disk
from typing import List, Dict

logger = logging.getLogger(__name__)

def load_local_swe_dataset(dataset_names: List[str], dataset_path: str, shuffle: bool = True, seed: int = 42) -> List[Dict]:
    """
    Load preprocessed SWE datasets from filesystem path.
    Supports both SWE-Gym and SWE-Bench Lite datasets.
    
    Args:
        dataset_names: List of dataset names (used to determine dataset type)
        dataset_path: Path to the dataset on disk
        shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducible shuffling
        
    Returns:
        List of dictionaries containing SWE repair tasks
    """
    try:
        if not dataset_path:
            logger.error("Dataset path not found in config or environment")
            return []
            
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset directly from disk
        dataset = load_from_disk(dataset_path)
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        
        # Determine dataset type
        is_swebench = any("swebench" in name.lower() for name in dataset_names)
        dataset_type = "swebench_lite" if is_swebench else "swegym"
        
        # Process the dataset into the expected format
        samples = []
        for item in dataset:
            try:
                # Parse file contents with error handling
                file_contents_raw = item.get("gold_file_contents", "{}")
                try:
                    file_contents = json.loads(file_contents_raw)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Failed to parse 'gold_file_contents' for an item. Skipping.")
                    continue
                    
                # Skip items with no file contents
                if not file_contents:
                    continue
                
                # Extract ID based on dataset type
                if is_swebench:
                    item_id = item.get("instance_id", "")
                else:
                    item_id = item.get("issue_id", "") or item.get("id", "")
                
                # Format compatible with convert_swe_problems_to_tapes
                samples.append({
                    "id": item_id,
                    "dataset": dataset_type,
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "problem_statement": item.get("problem_statement"),
                    "patch": item.get("patch"),
                    "file_contents": file_contents,
                    "all_file_stats": item.get("all_file_stats", {})
                })
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        # Shuffle the samples if requested
        if shuffle:
            random.seed(seed)
            random.shuffle(samples)
            logger.info(f"Shuffled {len(samples)} samples with seed {seed}")
            
        logger.info(f"Processed {len(samples)} valid {dataset_type} samples")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return []