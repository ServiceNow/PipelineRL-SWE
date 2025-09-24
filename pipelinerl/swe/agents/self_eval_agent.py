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
    Thought,
    FinalStep,
)
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode

logger = logging.getLogger(__name__)


class SelfEvalTask(Observation):
    kind: Literal["self_eval_task"] = "self_eval_task"
    problem_statement: str
    file_contents: dict[str, str]  # map of file path to content
    repair_edits: list[dict]  # The parsed search/replace edits
    template: str = Field(
        default=(
            "You are an expert code reviewer evaluating a proposed bug fix.\n\n"
            "TASK: Predict how well the proposed edits solve the given problem.\n"
            "Scale: 0.0 (completely wrong/harmful) to 1.0 (perfect solution)\n\n"
            "EVALUATION CRITERIA:\n"
            "- Correctness: Do the edits fix the described issue?\n"
            "- Completeness: Are all necessary changes included?\n"
            "- Safety: Do the edits avoid introducing new bugs?\n"
            "- Quality: Is the code well-written and maintainable?\n\n"
            "SCORING GUIDELINES:\n"
            "• 0.0-0.2: Incorrect, harmful, or completely off-target\n"
            "• 0.3-0.4: Partially addresses the issue but has major problems\n"
            "• 0.5-0.6: Good attempt but missing key fixes or has notable issues\n"
            "• 0.7-0.8: Solid solution with minor room for improvement\n"
            "• 0.9-1.0: Excellent or perfect solution\n\n"
            "FORMAT YOUR RESPONSE:\n"
            "<analysis>\n"
            "[Step-by-step evaluation explaining your reasoning]\n"
            "</analysis>\n\n"
            "<score>\n"
            "[Single number from 0.0 to 1.0]\n"
            "</score>\n\n"
            "=== PROBLEM STATEMENT ===\n"
            "{problem_statement}\n\n"
            "=== CODE FILES ===\n"
            "{file_contents}\n\n"
            "=== PROPOSED EDITS ===\n"
            "{formatted_edits}\n"
        )
    )

    def llm_view(self, indent: int | None = 2) -> str:
        # Format file contents cleanly
        formatted_contents = ""
        for file_path, content in self.file_contents.items():
            formatted_contents += f"**{file_path}**\n```\n{content}\n```\n\n"

        # Format edits in a clear, readable way
        if self.repair_edits:
            formatted_edits = ""
            for i, edit in enumerate(self.repair_edits, 1):
                file_path = edit.get('file_path', 'Unknown')
                search_text = edit.get('search', '')
                replace_text = edit.get('replace', '')
                
                formatted_edits += f"**Edit {i}: {file_path}**\n"
                formatted_edits += "```\n"
                formatted_edits += "SEARCH:\n"
                formatted_edits += f"{search_text}\n"
                formatted_edits += "---\n"
                formatted_edits += "REPLACE:\n" 
                formatted_edits += f"{replace_text}\n"
                formatted_edits += "```\n\n"
        else:
            formatted_edits = "No edits provided.\n"

        return self.template.format(
            problem_statement=self.problem_statement,
            file_contents=formatted_contents,
            formatted_edits=formatted_edits
        )


class SelfEvalResponse(Thought):
    """Response containing self-evaluation analysis and predicted score."""
    kind: Literal["self_eval_response"] = "self_eval_response"
    response: str = Field(description="complete response with analysis and score")
    analysis: str = Field(default="", description="detailed analysis of the repair")
    predicted_score: float = Field(default=0.0, description="predicted quality score from 0.0 to 1.0")
    parsing_error: bool = Field(default=False, description="whether there was an error parsing the response")


SelfEvalStep: TypeAlias = Annotated[
    SelfEvalResponse,
    Field(discriminator="kind"),
]

SelfEvalTape = Tape[
    None,
    Union[
        SelfEvalTask,
        SelfEvalResponse,
        LLMOutputParsingFailureAction,
    ],
]


class SelfEvalNode(StandardNode):
    max_prompt_length: int = 20000  # Large enough for code files + edits

    def _extract_analysis_and_score(self, response_text: str) -> tuple[str, float, bool]:
        """Extract analysis and score from the response text.
        
        Returns:
            Tuple of (analysis, predicted_score, parsing_error)
        """
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
            
            step = SelfEvalResponse(
                response=completion,
                analysis=analysis,
                predicted_score=predicted_score,
                parsing_error=parsing_error
            )
            yield step
            
        except Exception as e:
            logger.info(f"Failed to parse self-eval output: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse self-eval output: {completion}\n\nError: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        task = tape.steps[0]
        assert isinstance(task, SelfEvalTask), f"Expected SelfEvalTask, got {task.__class__.__name__}"
        
        system_message = {
            "role": "system",
            "content": "You are an expert code reviewer that evaluates proposed bug fixes and predicts their quality on a 0.0-1.0 scale."
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


class SelfEvalAgent(Agent):
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 20000):
        # Handle the llm parameter for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                SelfEvalNode(
                    name="self_eval",
                    agent_step_cls=SelfEvalStep,
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