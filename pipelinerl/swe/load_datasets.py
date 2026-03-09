import json
import logging
import random
from datasets import load_from_disk
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def _parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    return {}


def _resolve_dataset_label(item: Dict[str, Any], dataset_names: List[str], dataset_label: str | None) -> str:
    if dataset_label:
        return dataset_label
    row_dataset = item.get("dataset")
    if isinstance(row_dataset, str) and row_dataset:
        return row_dataset
    if dataset_names:
        return str(dataset_names[0])
    return "swe"


def load_local_swe_dataset(
    dataset_names: List[str],
    dataset_path: str,
    shuffle: bool = True,
    seed: int = 42,
    dataset_label: str | None = None,
) -> List[Dict]:
    """
    Load preprocessed SWE datasets from filesystem path.
    
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
        
        # Process the dataset into the expected format
        samples = []
        for item in dataset:
            try:
                # Parse file contents with error handling
                file_contents = _parse_file_contents(item.get("gold_file_contents", "{}"))
                    
                # Skip items with no file contents
                if not file_contents:
                    continue
                
                item_id = item.get("issue_id", "") or item.get("instance_id", "") or item.get("id", "")
                resolved_dataset = _resolve_dataset_label(item, dataset_names, dataset_label)
                all_file_stats = item.get("all_file_stats", "{}")
                if isinstance(all_file_stats, dict):
                    all_file_stats = json.dumps(all_file_stats)
                elif not isinstance(all_file_stats, str):
                    all_file_stats = "{}"
                
                # Format compatible with convert_swe_problems_to_tapes
                samples.append({
                    "id": item_id,
                    "dataset": resolved_dataset,
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "problem_statement": item.get("problem_statement"),
                    "patch": item.get("patch"),
                    "file_contents": file_contents,
                    "all_file_stats": all_file_stats,
                })
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
        
        # Shuffle the samples if requested
        if shuffle:
            random.seed(seed)
            random.shuffle(samples)
            logger.info(f"Shuffled {len(samples)} samples with seed {seed}")
            
        logger.info(f"Processed {len(samples)} valid samples")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return []
