import logging
import re
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
            f"Problem: {self.problem_statement}\n\n"
            f"Stage: {self.stage_name}\n"
            f"Stage Input:\n{self.stage_input}\n\n"
            f"Your Output:\n{self.stage_output}\n\n"
            f"Self-Evaluation Score: {self.self_eval_score:.2f}/1.0\n"
            f"Analysis: {self.self_eval_analysis}"
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
    _code_block_pattern = re.compile(
        r'<code\s+path="([^"\n]+)">\s*(.*?)\s*</code>',
        re.DOTALL | re.IGNORECASE,
    )
    
    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        """Parse the LLM completion to extract the generated query."""
        try:
            def extract_required_tag(tag: str) -> str:
                pattern = re.compile(
                    rf"<{tag}>\s*(.*?)\s*</{tag}>",
                    re.DOTALL | re.IGNORECASE,
                )
                match = pattern.search(completion)
                if not match:
                    raise ValueError(f"Missing or empty <{tag}> tag")
                value = match.group(1)
                if not value or not value.strip():
                    raise ValueError(f"Missing or empty <{tag}> tag")
                return value

            def format_tag(tag: str, value: str) -> str:
                cleaned = value.strip("\n")
                return f"<{tag}>\n{cleaned}\n</{tag}>"

            # Extract required sections; fail fast if any tag is missing/empty
            context_content = extract_required_tag("context")
            question_content = extract_required_tag("question")

            code_matches = list(self._code_block_pattern.finditer(completion))
            if not code_matches:
                raise ValueError("At least one <code path=\"...\"> block is required")

            formatted_code_blocks: list[str] = []
            invalid_lines: list[tuple[str, str]] = []
            stage_input = getattr(self, "_current_stage_input", "") or ""
            stage_lines = {
                line.rstrip("\r")
                for line in stage_input.splitlines()
            } if stage_input else set()

            for match in code_matches:
                file_path = match.group(1).strip()
                if not file_path:
                    raise ValueError("Each <code> block must include a non-empty path attribute")

                block_content = match.group(2)
                if not block_content or not block_content.strip():
                    raise ValueError(f"<code path=\"{file_path}\"> block is empty")

                # Validate that every non-empty line exists somewhere in the stage input.
                if stage_lines:
                    for raw_line in block_content.splitlines():
                        normalized_line = raw_line.rstrip("\r")
                        if not normalized_line.strip():
                            continue
                        if normalized_line not in stage_lines:
                            invalid_lines.append((file_path, normalized_line))

                cleaned_block = block_content.strip("\n")
                formatted_code_blocks.append(
                    f'<code path="{file_path}">\n{cleaned_block}\n</code>'
                )

            if invalid_lines:
                preview = "\n".join(f"{path}: {line}" for path, line in invalid_lines[:5])
                raise ValueError(
                    "Code lines must match the stage input exactly (no paraphrasing or ellipses). "
                    f"Examples:\n{preview}"
                )

            context_tag = format_tag("context", context_content)
            question_tag = format_tag("question", question_content)
            codes_tag = "\n\n".join(formatted_code_blocks)

            parts = [context_tag, codes_tag, question_tag]
            generated_query = "\n\n".join(part for part in parts if part)

            # Extract reasoning (everything before tags)
            reasoning = ""
            first_tag_pos = min(
                completion.find(tag) for tag in ["<context>", "<code>", "<question>"]
                if tag in completion
            )
            reasoning = completion[:first_tag_pos].strip() if first_tag_pos > 0 else ""
            
            yield QueryGenerationResponse(
                generated_query=generated_query,
                reasoning=reasoning
            )
            
        except ValueError as e:
            yield LLMOutputParsingFailureAction(
                error=str(e),
                llm_output=completion
            )
            return
        except Exception as e:
            logger.info(f"Failed to parse query generation: {completion}\n\nError: {e}")
            logger.debug("Stage input used for validation:\n%s", getattr(self, "_current_stage_input", ""))
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
                "You formulate queries for a stronger model that will help improve your work.\n\n"
                "CRITICAL: The stronger model is ISOLATED. It can ONLY see what you put in the tags below. "
                "It has NO access to the problem statement, your code, or any other context unless you explicitly copy it.\n\n"
                "Use these tags:\n"
                "<context>Background information the expert needs</context>\n"
                "<code path=\"relative/file.py\">Copy-pasted code (use one block per file or snippet)</code>\n"
                "<question>Your specific question</question>\n\n"
                "Rules:\n"
                "- If your question references code, YOU MUST paste it into one or more <code path=\"...\"> blocks.\n"
                "- Use the exact file path in the path attribute so the expert knows where the code lives.\n"
                "- You may include excerpts, but every line must be copied verbatim from the code you saw (no summaries, ellipses, or renaming).\n"
                "- If your question needs problem context, YOU MUST copy it into <context> tags.\n"
                "- Don't reference 'the code above' or 'my output' - the expert can't see those\n"
                "- Make your query self-contained"
            )
        }
        
        user_message = {
            "role": "user",
            "content": task.llm_view()
        }

        # Store the current task input so we can validate copied code lines.
        self._current_stage_input = task.stage_input
        
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
