import logging
from typing import Annotated, Any, Generator, Literal, TypeAlias, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    Prompt,
    Step,
    Tape,
    Action,
)
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

logger = logging.getLogger(__name__)


class ExpertAdviceTask(Observation):
    """Task for getting advice from a stronger expert model."""
    kind: Literal["expert_advice_task"] = "expert_advice_task"
    query: str = Field(description="Query formulated by the smaller model")
    
    def llm_view(self, indent: int | None = 2) -> str:
        return (
            f"You are an expert AI assistant. A smaller model is seeking your guidance "
            f"on improving their work. Please provide helpful, specific, and actionable advice.\n\n"
            f"=== QUERY FROM SMALLER MODEL ===\n"
            f"{self.query}\n\n"
            f"Please provide clear, specific guidance that will help improve their output. "
            f"Focus on actionable recommendations rather than just evaluation."
        )


class ExpertAdviceResponse(Action):
    """Response containing advice from the expert model."""
    kind: Literal["expert_advice_response"] = "expert_advice_response"
    advice: str = Field(description="Advice provided by the expert model")


ExpertAdviceStep: TypeAlias = Annotated[
    ExpertAdviceResponse,
    Field(discriminator="kind"),
]

ExpertAdviceTape = Tape[
    None,
    Union[
        ExpertAdviceTask,
        ExpertAdviceResponse,
        LLMOutputParsingFailureAction,
    ],
]


class ExpertAdviceNode(StandardNode):
    """Node that provides expert advice."""
    
    max_prompt_length: int = 20000  # Larger for comprehensive advice
    
    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        """Parse the LLM completion to extract the advice."""
        try:
            # The entire completion is the advice
            advice = completion.strip()
            
            if not advice:
                yield LLMOutputParsingFailureAction(
                    error="Empty advice generated", 
                    llm_output=completion
                )
                return
            
            yield ExpertAdviceResponse(advice=advice)
            
        except Exception as e:
            logger.info(f"Failed to parse expert advice: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse expert advice: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        task = tape.steps[0]
        assert isinstance(task, ExpertAdviceTask), f"Expected ExpertAdviceTask, got {task.__class__.__name__}"
        
        system_message = {
            "role": "system",
            "content": (
                "You are an expert software engineer and AI assistant providing guidance to improve "
                "software engineering outputs. Provide specific, actionable advice."
            )
        }
        
        # The query now contains both context and question
        user_message = {
            "role": "user",
            "content": task.query  # Just use the query directly
        }
        
        messages = [system_message, user_message]
        
        # Apply token limit
        prompt_token_ids = None
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'tokenizer') and agent.llm.tokenizer:
            prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
                messages, add_special_tokens=True, add_generation_prompt=True
            )
            prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            
        return Prompt(messages=messages, token_ids=prompt_token_ids)


class ExpertLLMAdviceAgent(Agent):
    """Agent that provides expert advice using a stronger model."""
    
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 20000):
        """Create an ExpertLLMAdviceAgent."""
        # Handle the llm parameter correctly for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                ExpertAdviceNode(
                    name="expert_advice",
                    agent_step_cls=ExpertAdviceStep,
                    system_prompt=system_prompt if system_prompt else "",
                    max_prompt_length=max_prompt_length,
                ),
            ],
            max_iterations=1,  # Single step agent
        )
        agent.store_llm_calls = True
        if llm:
            agent.llm.load_tokenizer()
        return agent