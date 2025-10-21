import logging
from typing import Annotated, Any, Generator, Literal, TypeAlias, Union, Optional, Dict

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


class QueryGenerationTask(Observation):
    """Task for generating a query to send to a stronger model for guidance."""
    kind: Literal["query_generation_task"] = "query_generation_task"
    stage_name: str = Field(description="Name of the stage seeking guidance")
    problem_statement: str = Field(description="Original problem statement")
    stage_input: str = Field(description="Input provided to the stage")
    stage_output: str = Field(description="Output produced by the stage")
    self_eval_score: float = Field(description="Self-evaluation score that triggered this query")
    self_eval_analysis: str = Field(default="", description="Self-evaluation analysis")
    
    def llm_view(self, indent: int | None = 2) -> str:
        return (
            f"You need to generate a query to send to a stronger model for guidance on improving "
            f"the {self.stage_name} stage output.\n\n"
            f"=== AVAILABLE INFORMATION ===\n"
            f"Problem: {self.problem_statement}\n\n"
            f"Stage Input:\n{self.stage_input}\n\n"
            f"Your {self.stage_name.upper()} Output:\n{self.stage_output}\n\n"
            f"Self-Evaluation Score: {self.self_eval_score:.2f}/1.0\n"
            f"Analysis: {self.self_eval_analysis}\n\n"
            f"Generate a query with two parts:\n"
            f"1. <context> - Copy any information the expert needs to understand your situation\n"
            f"2. <question> - Your specific question\n\n"
            f"The expert will ONLY see what you put in these tags."
        )


class QueryGenerationResponse(Action):
    """Response containing the generated query for the stronger model."""
    kind: Literal["query_generation_response"] = "query_generation_response"
    generated_query: str = Field(description="Query to send to the stronger model")
    reasoning: str = Field(default="", description="Reasoning behind the query formulation")


QueryGenerationStep: TypeAlias = Annotated[
    QueryGenerationResponse,
    Field(discriminator="kind"),
]

QueryGenerationTape = Tape[
    None,
    Union[
        QueryGenerationTask,
        QueryGenerationResponse,
        LLMOutputParsingFailureAction,
    ],
]


class QueryGenerationNode(StandardNode):
    """Node that generates queries for stronger models."""
    
    max_prompt_length: int = 16000
    
    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        """Parse the LLM completion to extract the generated query."""
        try:
            # Extract context
            context = ""
            if "<context>" in completion and "</context>" in completion:
                ctx_start = completion.find("<context>") + 9
                ctx_end = completion.find("</context>")
                context = completion[ctx_start:ctx_end].strip()
            
            # Extract question
            question = ""
            if "<question>" in completion and "</question>" in completion:
                q_start = completion.find("<question>") + 10
                q_end = completion.find("</question>")
                question = completion[q_start:q_end].strip()
            
            # Basic non-empty check
            if not context or not question:
                yield LLMOutputParsingFailureAction(
                    error="Missing or empty <context> or <question> tags", 
                    llm_output=completion
                )
                return
            
            # Combine into final query
            generated_query = f"{context}\n\n{question}"
            
            # Extract reasoning (everything before tags)
            reasoning = ""
            if "<context>" in completion:
                reasoning = completion[:completion.find("<context>")].strip()
            
            yield QueryGenerationResponse(
                generated_query=generated_query,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.info(f"Failed to parse query generation: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse query generation: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt for the model to generate a query."""
        task = tape.steps[0]
        assert isinstance(task, QueryGenerationTask), f"Expected QueryGenerationTask, got {task.__class__.__name__}"
        
        system_message = {
            "role": "system",
            "content": (
                "You are an expert at formulating queries to get helpful guidance from stronger models.\n\n"
                "Format your response as:\n"
                "[Your reasoning about what to include]\n\n"
                "<context>\n"
                "[Copy any information the expert needs - problem details, your output, etc.]\n"
                "</context>\n\n"
                "<question>\n"
                "[Your specific question]\n"
                "</question>"
            )
        }
        
        user_message = {
            "role": "user",
            "content": task.llm_view()
        }
        
        messages = [system_message, user_message]
        
        # Apply token limit if we have a tokenizer
        prompt_token_ids = None
        if hasattr(agent, 'llm') and hasattr(agent.llm, 'tokenizer') and agent.llm.tokenizer:
            prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
                messages, add_special_tokens=True, add_generation_prompt=True
            )
            prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            
        return Prompt(messages=messages, token_ids=prompt_token_ids)


class QueryGeneratorAgent(Agent):
    """Agent for generating queries to send to stronger models."""
    
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 16000):
        """Create a QueryGeneratorAgent."""
        # Handle the llm parameter correctly for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                QueryGenerationNode(
                    name="query_generation",
                    agent_step_cls=QueryGenerationStep,
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