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
                "You are evaluating a localization stage that generates search queries to find relevant files for a bug fix.\n\n"
                "TASK: Predict how well the generated search queries will find the relevant files.\n"
                "Scale: 0.0 (completely ineffective queries) to 1.0 (perfect queries that will find all relevant files)\n\n"
                "EVALUATION CRITERIA:\n"
                "- Relevance: Do the queries target the right concepts and terminology?\n"
                "- Coverage: Do the queries cover different aspects of the problem?\n"
                "- Specificity: Are the queries specific enough to avoid noise?\n"
                "- Completeness: Are there important search terms missing?\n\n"
                "=== PROBLEM STATEMENT ===\n"
                "{problem_statement}\n\n"
                "=== REPOSITORY CONTEXT ===\n"
                "{stage_input}\n\n"
                "=== GENERATED SEARCH QUERIES ===\n"
                "{stage_output}\n"
            ),
            "file_selection": (
                "You are evaluating a file selection stage that chooses relevant files from candidates for bug fixing.\n\n"
                "TASK: Predict how well the selected files will help solve the given problem.\n"
                "Scale: 0.0 (completely wrong files) to 1.0 (perfect file selection)\n\n"
                "EVALUATION CRITERIA:\n"
                "- Relevance: Are the selected files likely to contain the bug or need modification?\n"
                "- Completeness: Are all necessary files included?\n"
                "- Efficiency: Are unnecessary files excluded?\n"
                "- Coverage: Do the selected files cover all aspects of the problem?\n\n"
                "=== PROBLEM STATEMENT ===\n"
                "{problem_statement}\n\n"
                "=== CANDIDATE FILES ===\n"
                "{stage_input}\n\n"
                "=== SELECTED FILES ===\n"
                "{stage_output}\n"
            ),
            "repair": (
                "You are evaluating a code repair stage that generates fixes for a given problem.\n\n"
                "TASK: Predict how well the proposed edits solve the given problem.\n"
                "Scale: 0.0 (completely wrong/harmful) to 1.0 (perfect solution)\n\n"
                "EVALUATION CRITERIA:\n"
                "- Correctness: Do the edits fix the described issue?\n"
                "- Completeness: Are all necessary changes included?\n"
                "- Safety: Do the edits avoid introducing new bugs?\n"
                "- Quality: Is the code well-written and maintainable?\n\n"
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
            f"SCORING GUIDELINES:\n"
            f"• 0.0-0.2: Completely incorrect, harmful, or off-target\n"
            f"• 0.3-0.4: Partially addresses the issue but has major problems\n"
            f"• 0.5-0.6: Good attempt but missing key elements or has notable issues\n"
            f"• 0.7-0.8: Solid approach with minor room for improvement\n"
            f"• 0.9-1.0: Excellent or perfect solution\n\n"
            f"FORMAT YOUR RESPONSE:\n"
            f"<analysis>\n"
            f"[Step-by-step evaluation explaining your reasoning]\n"
            f"</analysis>\n\n"
            f"<score>\n"
            f"[Single number from 0.0 to 1.0]\n"
            f"</score>"
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
            agent.llm.load_tokenizer()
        return agent