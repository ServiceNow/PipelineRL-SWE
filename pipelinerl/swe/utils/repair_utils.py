import time
import logging
import difflib
from typing import Dict, List, Tuple, TypedDict
import aiohttp
from omegaconf import DictConfig
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from pipelinerl.rollouts import RolloutResult
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM
from pipelinerl.async_llm import llm_async_generate, make_training_text

from tapeagents.orchestrator import async_execute_agent

logger = logging.getLogger(__name__)

class FormatError(Exception):
    pass

class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float

def _leading_ws(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _strip_trailing_blank_lines(lines: List[str]) -> List[str]:
    result = list(lines)
    while result and not result[-1].strip():
        result.pop()
    while result and not result[0].strip():
        result.pop(0)
    return result


def _find_unique_normalized_block(content: str, search_text: str) -> Tuple[int, int, str] | None:
    """Find a unique linewise match while ignoring indentation/trailing spaces.

    This is deliberately conservative. It only returns a match if the stripped
    SEARCH lines match exactly once in the target file.
    """
    search_lines = _strip_trailing_blank_lines(search_text.splitlines())
    if not search_lines:
        return None
    stripped_search = [line.strip() for line in search_lines]
    content_lines = content.splitlines(keepends=True)
    logical_lines = [line.rstrip("\r\n") for line in content_lines]
    candidates: List[Tuple[int, int, str]] = []
    n = len(stripped_search)
    for start in range(0, len(logical_lines) - n + 1):
        window = logical_lines[start:start + n]
        if [line.strip() for line in window] != stripped_search:
            continue
        start_char = sum(len(line) for line in content_lines[:start])
        end_char = sum(len(line) for line in content_lines[:start + n])
        candidates.append((start_char, end_char, _leading_ws(logical_lines[start])))
        if len(candidates) > 1:
            return None
    return candidates[0] if len(candidates) == 1 else None


def _reindent_replacement(replace_text: str, target_indent: str) -> str:
    replace_lines = replace_text.splitlines()
    if not replace_lines:
        return replace_text
    nonempty = [line for line in replace_lines if line.strip()]
    if not nonempty:
        return replace_text
    # If the model emitted an unindented block for an indented target, shift the
    # whole replacement under the target indentation. Preserve relative spacing.
    min_indent = min(len(_leading_ws(line)) for line in nonempty)
    adjusted = []
    for line in replace_lines:
        if line.strip():
            adjusted.append(target_indent + line[min_indent:])
        else:
            adjusted.append("")
    return "\n".join(adjusted)

def generate_unified_diff(
    old_code: str,
    new_code: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code strings."""
    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
        n=n_context,
    )
    try:
        next(diff)
        next(diff)
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""

def apply_edits_to_files(
    file_contents: Dict[str, str],
    edits: List[Dict],
    silent: bool = False
) -> Dict[str, str]:
    """
    Apply a list of edits to file contents and return the modified file contents.
    
    Args:
        file_contents: Dictionary mapping file paths to their original content
        edits: List of edit dictionaries with 'file_path', 'search', and 'replace' keys
        silent: Whether to suppress format errors (for internal use)
        
    Returns:
        Dictionary mapping file paths to their modified content
        
    Raises:
        FormatError: If search text not found or search equals replace
    """
    new_content_dict = {}
    
    # Start with original file contents
    for path, content in file_contents.items():
        new_content_dict[path] = content
    
    # Apply each edit
    for edit in edits:
        file_path = edit.get('file_path', '')
        search_text = edit.get('search', '')
        replace_text = edit.get('replace', '')
        
        # Check for identical search and replace
        if not silent and search_text == replace_text:
            raise FormatError("Search and replace blocks are identical")
        
        if file_path not in new_content_dict:
            if not silent:
                raise FormatError(f"File {file_path} not found in file_contents")
            else:
                logger.warning(f"File {file_path} not found in file_contents")
                continue
        
        current_content = new_content_dict[file_path]
        if search_text in current_content:
            new_content_dict[file_path] = current_content.replace(search_text, replace_text, 1)
            continue

        normalized_match = _find_unique_normalized_block(current_content, search_text)
        if normalized_match is not None:
            start_char, end_char, target_indent = normalized_match
            adjusted_replace = _reindent_replacement(replace_text, target_indent)
            # Preserve the line ending after the matched block when replacing a
            # linewise span produced by splitlines(keepends=True).
            matched_text = current_content[start_char:end_char]
            if matched_text.endswith(("\n", "\r")) and adjusted_replace and not adjusted_replace.endswith("\n"):
                adjusted_replace += "\n"
            new_content_dict[file_path] = current_content[:start_char] + adjusted_replace + current_content[end_char:]
            continue

        if not silent:
            raise FormatError(f"Search text not found in {file_path}: {search_text}")
        else:
            logger.warning(f"Search text not found in {file_path}")
            continue
    
    return new_content_dict

def get_normalized_patch(
    code_context: Dict[str, str],
    new_content_dict: Dict[str, str],
) -> Dict[str, str]:
    """
    Generate the normalized patch for each file based on code context and new content.
    """
    patch_dict = {}
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        patch = generate_unified_diff(old_content, new_content)
        # Only add the patch if it's not empty
        if patch:
            patch_dict[path] = patch
    return patch_dict

def get_filelevel_diff(patch_text: str) -> Dict[str, str]:
    """
    Convert a unified diff text into a dictionary of file patches.
    Only handles text file modifications.
    """
    try:
        patch = PatchSet(patch_text)
    except UnidiffParseError:
        return {}
    except Exception as e:
        logger.warning(f"Unexpected unidiff parsing error: {str(e)}")
        return {}
    
    result = {}
    for patchfile in patch:
        path = patchfile.path
        body = "\n".join(str(hunk).strip() for hunk in patchfile)
        result[path] = body.strip()
    return result

def compute_change_similarities(
    pred_patch: Dict[str, str],
    oracle_patch: Dict[str, str],
) -> List[ChangeSimilarity]:
    """Compute similarity between predicted and oracle patches for each file."""
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = []
    
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        
        if oracle_change == "" or pred_change == "":
            # Empty changes should be penalized
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change,
                oracle_change,
                autojunk=False,
            ).ratio()
        
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )
    return similarities

def calculate_precise_reward(
    file_contents: Dict[str, str],
    oracle_patch_text: str,
    predicted_edits: List[Dict]
) -> Tuple[float, Dict]:
    """
    Calculate reward using precise file-by-file patch analysis.
    
    Args:
        file_contents: Original file contents
        oracle_patch_text: Gold patch in unified diff format
        predicted_edits: List of predicted edits
        
    Returns:
        Tuple of (reward_score, metadata_dict)
    """
    try:
        # Check for empty edits first
        if len(predicted_edits) == 0:
            raise FormatError("No valid search blocks found")
        
        # Get oracle patch as file-level diffs
        oracle_patch = get_filelevel_diff(oracle_patch_text)
        
        # Apply predicted edits to get new file contents (this can raise FormatError)
        pred_new_content = apply_edits_to_files(file_contents, predicted_edits)
        
        # Generate predicted patch as file-level diffs
        pred_patch = get_normalized_patch(file_contents, pred_new_content)
        
        # Calculate similarities for each file
        similarities = compute_change_similarities(pred_patch, oracle_patch)
        
        # Handle edge case where both patches are empty
        if len(similarities) == 0:
            assert len(oracle_patch) == 0 and len(pred_patch) == 0
            return 1.0, {"similarities": []}
        
        # Calculate average similarity across all files
        reward = sum(sim["similarity"] for sim in similarities) / len(similarities)
        
        return reward, {
            "similarities": similarities,
            "num_files_changed": len(similarities),
            "oracle_files": list(oracle_patch.keys()),
            "predicted_files": list(pred_patch.keys())
        }
        
    except FormatError as e:
        # Format errors get 0 reward
        logger.warning(f"Format error calculating precise reward: {str(e)}")
        return 0, {"format_error": True, "error_message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error calculating precise reward: {str(e)}")
        return 0, {"error": str(e)}