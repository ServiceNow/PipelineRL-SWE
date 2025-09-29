"""
Self-evaluation layer for SWE rollouts.
Wraps pure stage functions with optional self-evaluation capabilities.
"""

import math
import time
import logging
from typing import Dict, Optional
import json

from omegaconf import DictConfig
from tapeagents.core import LLMCall
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import make_training_text
from pipelinerl.swe.agents.generic_self_eval_agent import GenericSelfEvalAgent, GenericSelfEvalTask, GenericSelfEvalTape, GenericSelfEvalResponse
from pipelinerl.swe.agents.localization_agent import LocalizationQuery
from pipelinerl.swe.agents.file_selection_agent import FileSelectionResponse
from pipelinerl.swe.agents.repair_agent import SearchReplaceResponse
from .base import execute_agent_with_retry
from .stages import run_localization, run_file_selection, run_repair

logger = logging.getLogger(__name__)


async def run_generic_self_eval(
    cfg: DictConfig, 
    llm: TrainableLLM, 
    stage_name: str, 
    problem_statement: str, 
    stage_input: str, 
    stage_output: str, 
    true_reward: float, 
    session
) -> Dict:
    """
    Execute generic self-evaluation for any stage.
    
    Args:
        cfg: Configuration
        llm: Language model
        stage_name: Name of the stage being evaluated
        problem_statement: Original problem statement
        stage_input: Input that was provided to the stage
        stage_output: Output produced by the stage
        true_reward: Actual reward/performance of the stage
        session: HTTP session
        
    Returns:
        Dictionary with training_text, metrics, latency, etc.
    """
    agent = GenericSelfEvalAgent.create(
        llm=llm,
        max_prompt_length=getattr(cfg.agent, f'{stage_name}_self_eval_max_prompt_length', 20000)
    )
    
    task = GenericSelfEvalTask(
        stage_name=stage_name,
        problem_statement=problem_statement,
        stage_input=stage_input,
        stage_output=stage_output
    )
    tape = GenericSelfEvalTape(steps=[task], context=None)
    
    start_time = time.time()
    try:
        new_tape, llm_call = await execute_agent_with_retry(agent, tape, session)
        latency = time.time() - start_time
        
        if isinstance(llm_call, dict):
            llm_call = LLMCall(**llm_call)
        
        analysis = ""
        predicted_score = 0.0
        parsing_error = False
        
        for step in new_tape.steps:
            if isinstance(step, GenericSelfEvalResponse):
                step.stage_name = stage_name  # Ensure stage name is set
                analysis = step.analysis
                predicted_score = step.predicted_score
                parsing_error = step.parsing_error
                break
        
        prediction_error = abs(predicted_score - true_reward)
        
        if hasattr(cfg.actor, 'discount_factor'):
            # Use prediction error to compute reward for self-eval training
            reward = max(0.0, 1.0 - prediction_error)
            reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens
        else:
            reward = max(0.0, 1.0 - prediction_error)

        training_text = make_training_text(llm, llm_call)
        training_text.reward = reward if (reward is not None and not math.isnan(reward)) else 0.0
        
        metrics_dict = {
            "predicted_score": predicted_score,
            "true_reward": true_reward,
            "prediction_error": prediction_error,
            "parsing_error": parsing_error,
            "analysis_length": len(analysis.split()) if analysis else 0
        }
        
        return {
            'training_text': training_text,
            'metrics': metrics_dict,
            'latency': latency,
            'prompt_tokens': llm_call.prompt_length_tokens,
            'output_tokens': llm_call.output_length_tokens,
            'predicted_score': predicted_score
        }
        
    except Exception as e:
        logger.error(f"Self-eval error for {stage_name}: {e}")
        return {
            'training_text': None, 'metrics': {"error": str(e)},
            'latency': time.time() - start_time, 'prompt_tokens': 0, 'output_tokens': 0, 
            'success': False, 'predicted_score': 0.0
        }


def format_stage_output(stage_name: str, stage_data: Dict) -> str:
    """Format stage output for self-evaluation input."""
    if stage_name == "localization":
        queries = stage_data.get('queries', [])
        if queries:
            return "\n".join([f"Query {i+1}: {q}" for i, q in enumerate(queries)])
        return "No queries generated"
    
    elif stage_name == "file_selection" or stage_name == "selection":
        files = stage_data.get('selected_files', [])
        if files:
            return "\n".join([f"Selected: {f}" for f in files])
        return "No files selected"
    
    elif stage_name == "repair":
        edits = stage_data.get('edits', [])
        if edits:
            output_lines = []
            for i, edit in enumerate(edits):
                output_lines.append(f"Edit {i+1} - {edit.get('file_path', 'unknown')}:")
                output_lines.append(f"SEARCH:\n{edit.get('search', '')}")
                output_lines.append(f"REPLACE:\n{edit.get('replace', '')}")
                output_lines.append("")
            return "\n".join(output_lines)
        return "No edits generated"
    
    return str(stage_data)


async def run_localization_with_self_eval(cfg: DictConfig, llm: TrainableLLM, problem: Dict, session, expert_feedback=None):
    """Run localization stage with optional self-evaluation."""
    # Call the pure stage function
    result = await run_localization(cfg, llm, problem, session, expert_feedback)
    
    # Add self-evaluation if enabled
    if cfg.swe.get('enable_localization_self_eval', False):
        try:
            file_stats = json.loads(problem['all_file_stats'])
            stage_input = f"Repository has {len(file_stats)} files"
        except:
            stage_input = "Repository context"
        
        # Extract queries from result (now returned by run_localization)
        queries = result.get('queries', [])
        
        stage_output = format_stage_output("localization", {"queries": queries})
        reward = result['metrics'].get('mrr', 0.0)
        
        self_eval_result = await run_generic_self_eval(
            cfg, llm, "localization", problem["problem_statement"],
            stage_input, stage_output, reward, session
        )
        result['self_eval_result'] = self_eval_result
    
    return result


async def run_file_selection_with_self_eval(cfg: DictConfig, llm: TrainableLLM, problem: Dict, enriched_context: Dict, session, expert_feedback=None):
    """Run file selection stage with optional self-evaluation."""
    # Call the pure stage function
    result = await run_file_selection(cfg, llm, problem, enriched_context, session, expert_feedback)
    
    # Add self-evaluation if enabled
    if cfg.swe.get('enable_file_selection_self_eval', False):
        stage_input = "\n".join([f"{fp}: {ctx.get('summary', 'No summary')[:100]}..." 
                               for fp, ctx in enriched_context.items()])
        
        # Extract selected_files from result - this is available!
        selected_files = result.get('selected_files', [])
        
        stage_output = format_stage_output("file_selection", {"selected_files": selected_files})
        reward = result['metrics'].get('selection_f1', 0.0)
        
        self_eval_result = await run_generic_self_eval(
            cfg, llm, "file_selection", problem["problem_statement"],
            stage_input, stage_output, reward, session
        )
        result['self_eval_result'] = self_eval_result
    
    return result


async def run_repair_with_self_eval(cfg: DictConfig, llm: TrainableLLM, problem: Dict, file_contents: Dict, session, expert_feedback=None):
    """Run repair stage with optional self-evaluation."""
    # Call the pure stage function
    result = await run_repair(cfg, llm, problem, file_contents, session, expert_feedback)
    
    # Add self-evaluation if enabled
    if cfg.swe.get('enable_repair_self_eval', False):
        stage_input = "\n".join([f"**{fp}**\n{content[:500]}..." 
                               for fp, content in file_contents.items()])
        
        # Extract edits from result - this is available!
        repair_edits = result.get('repair_edits', [])
        
        stage_output = format_stage_output("repair", {"edits": repair_edits})
        reward = result['metrics'].get('reward', 0.0)
        
        self_eval_result = await run_generic_self_eval(
            cfg, llm, "repair", problem["problem_statement"],
            stage_input, stage_output, reward, session
        )
        result['self_eval_result'] = self_eval_result
    
    return result