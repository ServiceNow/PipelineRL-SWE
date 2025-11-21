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
    Thought,
    FinalStep,
)
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

logger = logging.getLogger(__name__)


class GenericSelfEvalTask(Observation):
    kind: Literal["generic_self_eval_task"] = "generic_self_eval_task"
    stage_name: str = Field(description="Name of the stage being evaluated (e.g., 'localization', 'file_selection', 'repair')")
    problem_statement: str = Field(description="Original problem statement")
    stage_input: str = Field(description="Input provided to the stage")
    stage_output: str = Field(description="Output produced by the stage")
    context: Optional[Dict] = Field(default=None, description="Additional context for evaluation")
    
    # Stage-specific evaluation templates
    templates: Dict[str, str] = Field(
        default_factory=lambda: {
            "localization": (
                "Evaluate the localization stage outputs.\n"
                "- Predict how well the queries will find relevant files.\n"
                "- Score: 0.0 (ineffective/harmful) to 1.0 (perfect).\n\n"
                "=== PROBLEM STATEMENT ===\n"
                "{problem_statement}\n\n"
                "=== REPOSITORY CONTEXT ===\n"
                "{stage_input}\n\n"
                "=== GENERATED SEARCH QUERIES ===\n"
                "{stage_output}\n"
            ),
            "file_selection": (
                "Evaluate the file selection stage outputs.\n"
                "- Predict how well the selected files will help solve the problem.\n"
                "- Score: 0.0 (wrong files) to 1.0 (perfect selection).\n\n"
                "=== PROBLEM STATEMENT ===\n"
                "{problem_statement}\n\n"
                "=== CANDIDATE FILES ===\n"
                "{stage_input}\n\n"
                "=== SELECTED FILES ===\n"
                "{stage_output}\n"
            ),
            "repair": (
                "Evaluate the code repair stage outputs.\n"
                "- Predict how well the proposed edits solve the problem.\n"
                "- Score: 0.0 (wrong/harmful) to 1.0 (clearly fixes the issue).\n\n"
                "=== PROBLEM STATEMENT ===\n"
                "{problem_statement}\n\n"
                "=== CODE FILES ===\n"
                "{stage_input}\n\n"
                "=== PROPOSED EDITS ===\n"
                "{stage_output}\n"
            )
        }
    )

    def llm_view(self, indent: int | None = 2) -> str:
        template = self.templates.get(self.stage_name, self.templates["repair"])
        
        formatted_template = template.format(
            problem_statement=self.problem_statement,
            stage_input=self.stage_input,
            stage_output=self.stage_output
        )
        
        return (
            f"{formatted_template}\n\n"
            "Return exactly one score between 0.0 and 1.0.\n"
            "FORMAT:\n"
            "<analysis>your reasoning</analysis>\n"
            "<score>0.0-1.0</score>"
        )


class GenericSelfEvalResponse(Thought):
    """Response containing self-evaluation analysis and predicted score."""
    kind: Literal["generic_self_eval_response"] = "generic_self_eval_response"
    stage_name: str = Field(description="Name of the stage that was evaluated")
    response: str = Field(description="complete response with analysis and score")
    analysis: str = Field(default="", description="detailed analysis of the stage output")
    predicted_score: float = Field(default=0.0, description="predicted quality score from 0.0 to 1.0")
    parsing_error: bool = Field(default=False, description="whether there was an error parsing the response")


GenericSelfEvalStep: TypeAlias = Annotated[
    GenericSelfEvalResponse,
    Field(discriminator="kind"),
]

GenericSelfEvalTape = Tape[
    None,
    Union[
        GenericSelfEvalTask,
        GenericSelfEvalResponse,
        LLMOutputParsingFailureAction,
    ],
]


class GenericSelfEvalNode(StandardNode):
    max_prompt_length: int = 20000  # Large enough for various stage inputs

    def _extract_analysis_and_score(self, response_text: str) -> tuple[str, float, bool]:
        """Extract analysis and score from the response text."""
        analysis = ""
        predicted_score = 0.0
        parsing_error = False
        
        try:
            # Extract analysis section
            analysis_start = response_text.find("<analysis>")
            analysis_end = response_text.find("</analysis>")
            
            if analysis_start != -1 and analysis_end != -1:
                analysis = response_text[analysis_start + 10:analysis_end].strip()
            else:
                # Fallback: use everything before score as analysis
                score_start = response_text.find("<score>")
                if score_start != -1:
                    analysis = response_text[:score_start].strip()
                else:
                    analysis = response_text.strip()
                parsing_error = True
                logger.warning("Could not find proper <analysis> tags")
            
            # Extract score section
            score_start = response_text.find("<score>")
            score_end = response_text.find("</score>")
            
            if score_start != -1 and score_end != -1:
                score_text = response_text[score_start + 7:score_end].strip()
                try:
                    predicted_score = float(score_text)
                    # Clamp to valid range
                    predicted_score = max(0.0, min(1.0, predicted_score))
                except ValueError:
                    logger.warning(f"Could not parse score as float: '{score_text}'")
                    parsing_error = True
                    predicted_score = 0.0
            else:
                logger.warning("Could not find <score> tags in response")
                parsing_error = True
                predicted_score = 0.0
                
        except Exception as e:
            logger.warning(f"Error parsing self-eval response: {e}")
            parsing_error = True
            analysis = response_text
            predicted_score = 0.0
        
        return analysis, predicted_score, parsing_error

    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        try:
            analysis, predicted_score, parsing_error = self._extract_analysis_and_score(completion)
            
            # Get stage name from the task
            stage_name = "unknown"
            # This will be set by the agent when creating the response
            
            step = GenericSelfEvalResponse(
                stage_name=stage_name,
                response=completion,
                analysis=analysis,
                predicted_score=predicted_score,
                parsing_error=parsing_error
            )
            yield step
            
        except Exception as e:
            logger.info(f"Failed to parse generic self-eval output: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse generic self-eval output: {completion}\n\nError: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        task = tape.steps[0]
        assert isinstance(task, GenericSelfEvalTask), f"Expected GenericSelfEvalTask, got {task.__class__.__name__}"
        
        system_message = {
            "role": "system",
            "content": f"You are an expert evaluator that predicts the quality of {task.stage_name} stage outputs on a 0.0-1.0 scale."
        }
        
        user_message = {
            "role": "user",
            "content": task.llm_view()
        }
        
        messages = [system_message, user_message]
        
        prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
            messages, add_special_tokens=True, add_generation_prompt=True
        )
        prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
        return Prompt(messages=messages, token_ids=prompt_token_ids)


class GenericSelfEvalAgent(Agent):
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 20000):
        # Handle the llm parameter for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                GenericSelfEvalNode(
                    name="generic_self_eval",
                    agent_step_cls=GenericSelfEvalStep,
                    system_prompt=system_prompt if system_prompt else "",
                    max_prompt_length=max_prompt_length,
                ),
            ],
            max_iterations=1,
        )
        agent.store_llm_calls = True
        if llm:
            try:
                agent.llm.load_tokenizer()
            except AttributeError as e:
                logger.error(f"Failed to load tokenizer for LLM: {e}")
        return agent
