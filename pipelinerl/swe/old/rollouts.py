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
        if search_text not in current_content:
            if not silent:
                raise FormatError(f"Search text not found in {file_path}: {search_text}")
            else:
                logger.warning(f"Search text not found in {file_path}")
                continue
        
        new_content_dict[file_path] = current_content.replace(search_text, replace_text, 1)
    
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
        # Format errors get -1.0 reward
        return -1.0, {"format_error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error calculating precise reward: {str(e)}")
        return -1.0, {"error": str(e)}

async def generate_swe_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession
) -> RolloutResult:
    """
    Generate a single SWE repair rollout with reward calculation using SWERepairAgent.
    
    Args:
        cfg: Configuration
        llm: Trainable LLM
        problem: SWE problem dictionary containing:
            - problem_statement: The issue description
            - file_contents: Dict mapping file paths to content
            - patch: Gold patch in unified diff format
            - Additional metadata (repo, instance_id, etc.)
        session: HTTP session for async requests
        agent: SWERepairAgent instance for handling the task
    
    Returns:
        RolloutResult with training text and metrics
    """
    from .swe_repair_agent import SWETask, SWERepairTape, SearchReplaceResponse, SWERepairAgent
    
    # Create agent
    agent = SWERepairAgent.create(
        llm=llm,
        max_prompt_length=cfg.agent.max_prompt_length
    )
    
    # Create the task step
    task_step = SWETask(
        problem_statement=problem["problem_statement"],
        file_contents=problem["file_contents"]
    )
    
    # Create initial tape with just the task
    tape = SWERepairTape(steps=[task_step], context=None)
    
    # Generate response using the agent
    time_start = time.time()
    
    # Run the agent to get the response
    new_tape = await async_execute_agent(
        agent,
        tape,
        None, # no env
        session,
    )
    
    latency = time.time() - time_start
    
    # Extract edits from the response step
    predicted_edits = []
    llm_call = None
    
    for step in new_tape.steps:
        if isinstance(step, SearchReplaceResponse):
            predicted_edits = step.edits
        # Get the LLM call for training data
        if (
            hasattr(step, 'metadata') and 
            step.metadata and 
            hasattr(step.metadata, 'other') and
            "llm_call" in step.metadata.other and
            step.metadata.other["llm_call"] is not None
        ):
            llm_call = step.metadata.other["llm_call"]
    
    if llm_call is None:
        raise ValueError("No LLM call found in the generated tape")
        print("Predicted edits:", predicted_edits)
    
    # Convert to LLMCall object if it's a dict
    from tapeagents.core import LLMCall
    if isinstance(llm_call, dict):
        llm_call = LLMCall(**llm_call)
    
    # Calculate reward using precise file-by-file analysis
    file_contents = problem["file_contents"]
    gold_patch = problem["patch"]
    
    if predicted_edits:
        reward, reward_metadata = calculate_precise_reward(
            file_contents, gold_patch, predicted_edits
        )
    else:
        reward = -1.0
        reward_metadata = {"format_error": "No edits found"}
    
    # Create training text from the LLM call
    trace = agent.llm.make_training_text(llm_call.prompt, llm_call.output)
    
    # Set up input_ids and labels for training
    input_ids = [lp.token_id for lp in llm_call.logprobs]
    labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
    
    # MASKED_TOKEN_ID is -100 and is the default "ignore_index" in nn.CrossEntropyLoss
    from pipelinerl.finetune.data import MASKED_TOKEN_ID
    labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
    
    trace.input_ids = input_ids
    trace.labels = labels
    
    # Check if the generation is finished (ended with EOS token)
    finished = 1 if input_ids[-1] == agent.llm.tokenizer.eos_token_id else 0
    
    # Apply discount factor if configured
    if hasattr(cfg.actor, 'discount_factor'):
        reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
    
    trace.reward = reward
    trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
    trace.group_id = new_tape.metadata.parent_id if new_tape.metadata else None
    
    # Calculate success metric (reward > threshold)
    success_threshold = getattr(cfg.actor, 'success_threshold', 0.8)
    success = reward > success_threshold
    
    # Prepare metrics
    metrics = {
        "reward": reward,
        "success": success,
        "no_edits": len(predicted_edits) == 0,
        "no_answer": len(predicted_edits) == 0, # we make no_answer a proxy for no edits
        "overflow": 0 if finished else 1,
        "num_files_changed": reward_metadata.get("num_files_changed", 0),
        "format_error": "format_error" in reward_metadata or "error" in reward_metadata,
        'no_error': not reward_metadata.get("format_error", False) and not reward_metadata.get("error", False),
        "steps": len(new_tape.steps),
        "prompt_tokens": llm_call.prompt_length_tokens,
        "output_tokens": llm_call.output_length_tokens,
    }
    
    # Add detailed reward metadata if available
    if "similarities" in reward_metadata:
        metrics["avg_similarity"] = reward
        metrics["oracle_files"] = len(reward_metadata.get("oracle_files", []))
        metrics["predicted_files"] = len(reward_metadata.get("predicted_files", []))
    
    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        prompt_tokens=[llm_call.prompt_length_tokens],
        output_tokens=[llm_call.output_length_tokens],
    )