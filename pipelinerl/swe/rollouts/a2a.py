"""
Agent-to-Agent (A2A) layer for SWE rollouts.
Wraps self-evaluation functions with A2A capabilities when confidence is low.
"""

import json
import time
import logging
from typing import Dict, List

from omegaconf import DictConfig
from tapeagents.core import LLMCall, Observation
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import make_training_text
from pipelinerl.swe.agents.query_generator_agent import QueryGeneratorAgent, QueryGenerationTask, QueryGenerationTape, QueryGenerationResponse
from pipelinerl.swe.agents.expert_llm_advice_agent import ExpertLLMAdviceAgent, ExpertAdviceTask, ExpertAdviceTape, ExpertAdviceResponse
from pipelinerl.swe.types import ExpertModelAdvice
from .base import execute_agent_with_retry
from .self_evaluation import run_localization_with_self_eval, run_file_selection_with_self_eval, run_repair_with_self_eval, format_stage_output

logger = logging.getLogger(__name__)


async def run_a2a(
    cfg: DictConfig, 
    stage_name: str, 
    problem_statement: str, 
    stage_input: str, 
    stage_output: str, 
    self_eval_score: float, 
    self_eval_analysis: str,
    llm: TrainableLLM,
    expert_llm: TrainableLLM,
    session
) -> Dict:
    """
    Run agent-to-agent consultation: query generation -> expert advice.
    
    Returns:
        Dictionary with query_training_text, expert_advice, and token usage metadata
    """
    start_time = time.time()
    
    try:
        # Step 1: Generate query for expert model
        query_agent = QueryGeneratorAgent.create(llm=llm)
        query_task = QueryGenerationTask(
            stage_name=stage_name,
            problem_statement=problem_statement,
            stage_input=stage_input,
            stage_output=stage_output,
            self_eval_score=self_eval_score,
            self_eval_analysis=self_eval_analysis
        )
        query_tape = QueryGenerationTape(steps=[query_task], context=None)
        
        query_tape_result, query_llm_call = await execute_agent_with_retry(query_agent, query_tape, session)
        
        if isinstance(query_llm_call, dict):
            query_llm_call = LLMCall(**query_llm_call)
        
        generated_query = ""
        for step in query_tape_result.steps:
            if isinstance(step, QueryGenerationResponse):
                generated_query = step.generated_query
                break
        
        if not generated_query:
            logger.warning("No query generated, skipping A2A")
            # Still create training data with penalty to prevent reward hacking
            query_training_text = make_training_text(llm, query_llm_call)
            query_training_text.reward = -1.0  # Penalty for malformed query
            return {
                'query_training_text': query_training_text,
                'expert_advice': None,
                'latency': time.time() - start_time,
                'success': False
            }
        
        # Step 2: Get expert advice
        expert_agent = ExpertLLMAdviceAgent.create(llm=expert_llm)
        expert_task = ExpertAdviceTask(query=generated_query)
        expert_tape = ExpertAdviceTape(steps=[expert_task], context=None)
        
        expert_tape_result, expert_llm_call = await execute_agent_with_retry(expert_agent, expert_tape, session)
        
        if isinstance(expert_llm_call, dict):
            expert_llm_call = LLMCall(**expert_llm_call)
        
        expert_advice = ""
        for step in expert_tape_result.steps:
            if isinstance(step, ExpertAdviceResponse):
                expert_advice = step.advice
                break
        
        if not expert_advice:
            logger.warning("No expert advice received")
            expert_advice = "No specific guidance provided."
        
        # Create training text for query generation (will get reward from enhanced stage later)
        query_training_text = make_training_text(llm, query_llm_call)
        # Reward will be set later based on enhanced stage performance
        
        return {
            'query_training_text': query_training_text,
            'expert_advice': ExpertModelAdvice(
                original_query=generated_query,
                advice=expert_advice,
                stage_name=stage_name
            ),
            'latency': time.time() - start_time,
            'query_prompt_tokens': query_llm_call.prompt_length_tokens,
            'query_output_tokens': query_llm_call.output_length_tokens,
            'expert_prompt_tokens': expert_llm_call.prompt_length_tokens,
            'expert_output_tokens': expert_llm_call.output_length_tokens,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error in A2A for {stage_name}: {e}")
        return {
            'query_training_text': None,
            'expert_advice': None,
            'latency': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


async def run_localization_a2a(cfg: DictConfig, llm: TrainableLLM, expert_llm: TrainableLLM, problem: Dict, session):
    """Run localization with A2A when confidence is low."""
    # First run with self-eval
    initial_result = await run_localization_with_self_eval(cfg, llm, problem, session)
    
    training_texts = []
    if initial_result['training_text']:
        training_texts.append(initial_result['training_text'])
    
    # Add self-eval training text if available
    if initial_result.get('self_eval_result') and initial_result['self_eval_result'].get('training_text'):
        training_texts.append(initial_result['self_eval_result']['training_text'])
    
    # Check if we should trigger A2A
    threshold = cfg.swe.get('localization_a2a_trigger_threshold', 0.5)
    self_eval_score = 0.0
    self_eval_analysis = ""
    
    if initial_result.get('self_eval_result'):
        self_eval_score = initial_result['self_eval_result'].get('predicted_score', 0.0)
        self_eval_analysis = initial_result['self_eval_result']['metrics'].get('analysis', '')
    
    if self_eval_score < threshold:
        logger.info(f"Localization confidence {self_eval_score:.2f} below threshold {threshold}, triggering A2A")
        
        # Prepare stage input and output for A2A
        try:
            file_stats = json.loads(problem.get('all_file_stats', '{}'))
            stage_input = f"Repository context with {len(file_stats)} files"
        except:
            stage_input = "Repository context"
        
        # Extract queries from initial result
        stage_output = format_stage_output("localization", {"queries": initial_result.get('queries', [])})
        
        # Run A2A
        a2a_result = await run_a2a(
            cfg, "localization", problem["problem_statement"],
            stage_input, stage_output, self_eval_score, self_eval_analysis,
            llm, expert_llm, session
        )
        
        if a2a_result['success'] and a2a_result['expert_advice']:
            # Add query training text
            if a2a_result['query_training_text']:
                training_texts.append(a2a_result['query_training_text'])

            # Re-run localization with expert advice
            enhanced_result = await run_localization_with_self_eval(
                cfg, llm, problem, session, a2a_result['expert_advice']
            )

            if enhanced_result['training_text']:
                # Set query reward based on enhanced performance
                if a2a_result['query_training_text']:
                    improvement = enhanced_result['metrics']['reward'] - initial_result['metrics']['reward']
                    a2a_result['query_training_text'].reward = improvement
                training_texts.append(enhanced_result['training_text'])

            # Add enhanced result's self-eval training text if available
            if enhanced_result.get('self_eval_result') and enhanced_result['self_eval_result'].get('training_text'):
                training_texts.append(enhanced_result['self_eval_result']['training_text'])

            # Add A2A metadata at top level (not in metrics dict)
            enhanced_result['training_texts'] = training_texts
            enhanced_result['a2a_enhanced'] = True
            enhanced_result['initial_self_eval_score'] = self_eval_score

            # Store initial metrics for comparison
            enhanced_result['initial_mrr'] = initial_result['metrics'].get('mrr', 0.0)
            enhanced_result['initial_recall'] = initial_result['metrics'].get('localization_recall', 0.0)

            # Store token usage
            enhanced_result['a2a_query_prompt_tokens'] = a2a_result.get('query_prompt_tokens', 0)
            enhanced_result['a2a_query_output_tokens'] = a2a_result.get('query_output_tokens', 0)
            enhanced_result['a2a_expert_prompt_tokens'] = a2a_result.get('expert_prompt_tokens', 0)
            enhanced_result['a2a_expert_output_tokens'] = a2a_result.get('expert_output_tokens', 0)

            return enhanced_result
        else:
            # A2A failed (e.g., malformed query) - still need to include failed query for training
            if a2a_result.get('query_training_text'):
                training_texts.append(a2a_result['query_training_text'])
    
    # Return original result if no A2A triggered
    initial_result['training_texts'] = training_texts
    initial_result['a2a_enhanced'] = False
    return initial_result



async def run_file_selection_a2a(cfg: DictConfig, llm: TrainableLLM, expert_llm: TrainableLLM, problem: Dict, enriched_context: Dict, session):
    """Run file selection with A2A when confidence is low."""
    # First run with self-eval
    initial_result = await run_file_selection_with_self_eval(cfg, llm, problem, enriched_context, session)
    
    training_texts = []
    if initial_result['training_text']:
        training_texts.append(initial_result['training_text'])
    
    # Add self-eval training text if available
    if initial_result.get('self_eval_result') and initial_result['self_eval_result'].get('training_text'):
        training_texts.append(initial_result['self_eval_result']['training_text'])
    
    # Check if we should trigger A2A
    threshold = cfg.swe.get('file_selection_a2a_trigger_threshold', 0.5)
    self_eval_score = 0.0
    self_eval_analysis = ""
    
    if initial_result.get('self_eval_result'):
        self_eval_score = initial_result['self_eval_result'].get('predicted_score', 0.0)
        self_eval_analysis = initial_result['self_eval_result']['metrics'].get('analysis', '')
    
    if self_eval_score < threshold:
        logger.info(f"File selection confidence {self_eval_score:.2f} below threshold {threshold}, triggering A2A")
        
        # Prepare stage input and output for A2A
        stage_input = "\n".join([f"{fp}: {ctx.get('summary', 'No summary')}..." 
                               for fp, ctx in enriched_context.items()])
        stage_output = format_stage_output("file_selection", {"selected_files": initial_result.get('selected_files', [])})
        
        # Run A2A
        a2a_result = await run_a2a(
            cfg, "file_selection", problem["problem_statement"],
            stage_input, stage_output, self_eval_score, self_eval_analysis,
            llm, expert_llm, session
        )
        
        if a2a_result['success'] and a2a_result['expert_advice']:
            # Add query training text
            if a2a_result['query_training_text']:
                training_texts.append(a2a_result['query_training_text'])

            # Re-run file selection with expert advice
            enhanced_result = await run_file_selection_with_self_eval(
                cfg, llm, problem, enriched_context, session, a2a_result['expert_advice']
            )

            if enhanced_result['training_text']:
                # Set query reward based on enhanced performance
                if a2a_result['query_training_text']:
                    improvement = enhanced_result['metrics']['reward'] - initial_result['metrics']['reward']
                    a2a_result['query_training_text'].reward = improvement
                training_texts.append(enhanced_result['training_text'])

            # Add enhanced result's self-eval training text if available
            if enhanced_result.get('self_eval_result') and enhanced_result['self_eval_result'].get('training_text'):
                training_texts.append(enhanced_result['self_eval_result']['training_text'])

            # Add A2A metadata at top level
            enhanced_result['training_texts'] = training_texts
            enhanced_result['a2a_enhanced'] = True
            enhanced_result['initial_self_eval_score'] = self_eval_score

            # Store initial metrics for comparison
            enhanced_result['initial_precision'] = initial_result['metrics'].get('selection_precision', 0.0)
            enhanced_result['initial_recall'] = initial_result['metrics'].get('selection_recall', 0.0)
            enhanced_result['initial_f1'] = initial_result['metrics'].get('selection_f1', 0.0)

            # Store token usage
            enhanced_result['a2a_query_prompt_tokens'] = a2a_result.get('query_prompt_tokens', 0)
            enhanced_result['a2a_query_output_tokens'] = a2a_result.get('query_output_tokens', 0)
            enhanced_result['a2a_expert_prompt_tokens'] = a2a_result.get('expert_prompt_tokens', 0)
            enhanced_result['a2a_expert_output_tokens'] = a2a_result.get('expert_output_tokens', 0)

            return enhanced_result
        else:
            # A2A failed (e.g., malformed query) - still need to include failed query for training
            if a2a_result.get('query_training_text'):
                training_texts.append(a2a_result['query_training_text'])
    
    # Return original result if no A2A triggered
    initial_result['training_texts'] = training_texts
    initial_result['a2a_enhanced'] = False
    return initial_result


async def run_repair_a2a(cfg: DictConfig, llm: TrainableLLM, expert_llm: TrainableLLM, problem: Dict, file_contents: Dict, session):
    """Run repair with A2A when confidence is low."""
    # First run with self-eval
    initial_result = await run_repair_with_self_eval(cfg, llm, problem, file_contents, session)
    
    training_texts = []
    if initial_result['training_text']:
        training_texts.append(initial_result['training_text'])
    
    # Add self-eval training text if available
    if initial_result.get('self_eval_result') and initial_result['self_eval_result'].get('training_text'):
        training_texts.append(initial_result['self_eval_result']['training_text'])
    
    # Check if we should trigger A2A
    threshold = cfg.swe.get('repair_a2a_trigger_threshold', 0.5)
    self_eval_score = 0.0
    self_eval_analysis = ""
    
    if initial_result.get('self_eval_result'):
        self_eval_score = initial_result['self_eval_result'].get('predicted_score', 0.0)
        self_eval_analysis = initial_result['self_eval_result']['metrics'].get('analysis', '')
    
    if self_eval_score < threshold:
        logger.info(f"Repair confidence {self_eval_score:.2f} below threshold {threshold}, triggering A2A")
        
        # Prepare stage input and output for A2A
        stage_input = "\n".join([f"**{fp}**\n{content}..." 
                               for fp, content in file_contents.items()])
        stage_output = format_stage_output("repair", {"edits": initial_result.get('repair_edits', [])})
        
        # Run A2A
        a2a_result = await run_a2a(
            cfg, "repair", problem["problem_statement"],
            stage_input, stage_output, self_eval_score, self_eval_analysis,
            llm, expert_llm, session
        )
        
        if a2a_result['success'] and a2a_result['expert_advice']:
            # Add query training text
            if a2a_result['query_training_text']:
                training_texts.append(a2a_result['query_training_text'])

            # Re-run repair with expert advice
            enhanced_result = await run_repair_with_self_eval(
                cfg, llm, problem, file_contents, session, a2a_result['expert_advice']
            )

            if enhanced_result['training_text']:
                # Set query reward based on enhanced performance
                if a2a_result['query_training_text']:
                    improvement = enhanced_result['metrics']['reward'] - initial_result['metrics']['reward']
                    a2a_result['query_training_text'].reward = improvement
                training_texts.append(enhanced_result['training_text'])

            # Add enhanced result's self-eval training text if available
            if enhanced_result.get('self_eval_result') and enhanced_result['self_eval_result'].get('training_text'):
                training_texts.append(enhanced_result['self_eval_result']['training_text'])

            # Add A2A metadata at top level
            enhanced_result['training_texts'] = training_texts
            enhanced_result['a2a_enhanced'] = True
            enhanced_result['initial_self_eval_score'] = self_eval_score

            # Store initial metrics for comparison
            enhanced_result['initial_reward'] = initial_result['metrics'].get('reward', 0.0)
            enhanced_result['initial_success'] = initial_result['metrics'].get('success', False)

            # Store token usage
            enhanced_result['a2a_query_prompt_tokens'] = a2a_result.get('query_prompt_tokens', 0)
            enhanced_result['a2a_query_output_tokens'] = a2a_result.get('query_output_tokens', 0)
            enhanced_result['a2a_expert_prompt_tokens'] = a2a_result.get('expert_prompt_tokens', 0)
            enhanced_result['a2a_expert_output_tokens'] = a2a_result.get('expert_output_tokens', 0)

            return enhanced_result
        else:
            # A2A failed (e.g., malformed query) - still need to include failed query for training
            if a2a_result.get('query_training_text'):
                training_texts.append(a2a_result['query_training_text'])
    
    # Return original result if no A2A triggered
    initial_result['training_texts'] = training_texts
    initial_result['a2a_enhanced'] = False
    return initial_result
