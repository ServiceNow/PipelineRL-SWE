#!/usr/bin/env python3
"""
Script to convert SWE repair tapes to evaluation format.
Transforms tapes from JSONL format to evaluation predictions.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import DictConfig

from ..actor_processing import apply_edits_to_files, generate_unified_diff, get_normalized_patch

logger = logging.getLogger(__name__)


def extract_edits_from_tape(tape_data: Dict) -> List[Dict]:
    """Extract search/replace edits from a tape."""
    edits = []
    
    for step in tape_data.get("steps", []):
        if step.get("kind") == "search_replace_response":
            edits = step.get("edits", [])
            break
    
    return edits


def extract_file_contents_from_tape(tape_data: Dict) -> Dict[str, str]:
    """Extract file contents from the SWETask step in a tape."""
    for step in tape_data.get("steps", []):
        if step.get("kind") == "swe_task":
            return step.get("file_contents", {})
    
    return {}


def extract_instance_id_from_tape(tape_data: Dict) -> str:
    """Extract instance_id from tape step metadata."""
    for step in tape_data.get("steps", []):
        step_id = step.get("metadata", {}).get("other", {}).get("id", "")
        if step_id:
            return step_id
    return ""


def generate_prediction_from_tape(tape_data: Dict) -> str:
    """
    Generate a unified diff prediction from a tape by applying edits to file contents.
    
    Args:
        tape_data: Dictionary containing the tape data
        
    Returns:
        Unified diff string in git format, or empty string if processing fails
    """
    try:
        # Extract file contents and edits
        file_contents = extract_file_contents_from_tape(tape_data)
        edits = extract_edits_from_tape(tape_data)
        
        if not file_contents or not edits:
            logger.warning("No file contents or edits found in tape")
            return ""
        
        # Apply edits to get modified file contents
        modified_contents = apply_edits_to_files(file_contents, edits, silent=True)
        
        # Generate unified diff for each modified file
        patch_dict = get_normalized_patch(file_contents, modified_contents)
        
        # Combine all file patches into a single diff string
        if not patch_dict:
            return ""
        
        # Format as a complete git-style unified diff
        diff_parts = []
        for file_path, patch in patch_dict.items():
            # Add git diff headers
            diff_parts.append(f"diff --git a/{file_path} b/{file_path}")
            diff_parts.append(f"index 0000000..1111111 100644")  # Placeholder git blob hashes
            diff_parts.append(f"--- a/{file_path}")
            diff_parts.append(f"+++ b/{file_path}")
            diff_parts.append(patch)
        
        return "\n".join(diff_parts)
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return ""


def process_tapes_file(tape_file: str, model: str, output_file: str) -> None:
    """
    Process a JSONL file of tapes and convert to evaluation format.
    Handles duplicate instance_ids by randomly selecting one tape per instance.
    
    Args:
        tape_file: Path to input JSONL file containing tapes
        model: Model name to use in output
        output_file: Path to output JSONL file
    """
    tape_path = Path(tape_file)
    output_path = Path(output_file)
    
    if not tape_path.exists():
        raise FileNotFoundError(f"Tape file not found: {tape_file}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # First pass: collect all tapes grouped by instance_id
    tapes_by_instance = {}
    error_count = 0
    
    with open(tape_path, 'r') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse tape data
                tape_data = json.loads(line.strip())
                
                # Extract instance_id
                instance_id = extract_instance_id_from_tape(tape_data)
                
                if not instance_id:
                    logger.warning(f"No instance_id found in tape at line {line_num}")
                    error_count += 1
                    continue
                
                # Group tapes by instance_id
                if instance_id not in tapes_by_instance:
                    tapes_by_instance[instance_id] = []
                tapes_by_instance[instance_id].append(tape_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Error processing tape at line {line_num}: {e}")
                error_count += 1
    
    # Second pass: randomly select one tape per instance and generate predictions
    processed_count = 0
    duplicate_count = 0
    
    with open(output_path, 'w') as outfile:
        for instance_id, tapes in tapes_by_instance.items():
            if len(tapes) > 1:
                duplicate_count += len(tapes) - 1
                logger.info(f"Found {len(tapes)} tapes for instance {instance_id}, randomly selecting one")
            
            # Randomly select one tape for this instance
            selected_tape = random.choice(tapes)
            
            try:
                # Generate prediction
                prediction = generate_prediction_from_tape(selected_tape)
                
                # Create output record
                output_record = {
                    "instance_id": instance_id,
                    "model_name_or_path": model,
                    "model_patch": prediction
                }
                
                # Write to output file
                outfile.write(json.dumps(output_record) + '\n')
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error generating prediction for instance {instance_id}: {e}")
                error_count += 1
    
    logger.info(f"Processed {processed_count} unique instances successfully")
    if duplicate_count > 0:
        logger.info(f"Skipped {duplicate_count} duplicate tapes")
    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")


@hydra.main(version_base=None, config_path="../../conf/swe", config_name="swebench_eval")
def main(cfg: DictConfig) -> None:
    """Main function to process tapes using Hydra configuration."""
    
    # Validate required config fields
    required_fields = ["tape_file", "model", "output_file"]
    for field in required_fields:
        if field not in cfg:
            raise ValueError(f"Missing required config field: {field}")
    
    logger.info(f"Processing tapes from: {cfg.tape_file}")
    logger.info(f"Model: {cfg.model}")
    logger.info(f"Output file: {cfg.output_file}")
    
    process_tapes_file(cfg.tape_file, cfg.model, cfg.output_file)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()