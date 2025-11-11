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

    # ----------------------------
    # Few-shot input+output pairs
    # ----------------------------
    def _fewshot_pairs(self) -> list[dict]:
        """
        Repair-stage paired demonstrations with full I/O:
        - USER turns mirror QueryGenerationTask.llm_view() (Problem, Stage=repair, Stage Input code, Your Output, Self-Eval, Analysis)
        - ASSISTANT turns emit <context> + <code> (verbatim lines) + <question>
        - Code lines in <code> appear exactly in the corresponding USER Stage Input.
        """
        pairs: list[dict] = []

        # Pair A — Pagination off-by-one
        user_a = (
            "Problem: JSON pagination returns 49 items when total is an exact multiple of page_size. "
            "Client API expects exactly page_size items per full page.\n\n"
            "Stage: repair\n"
            "Stage Input:\n"
            "### paginator.py\n"
            "```\n"
            "def page(items, start, page_size):\n"
            "    \"\"\"Return a slice of items for one page.\n"
            "    start: index of first item on the page (0-based)\n"
            "    page_size: items per page\n"
            "    \"\"\"\n"
            "    if page_size <= 0:\n"
            "        return []\n"
            "    end = start + page_size - 1  # suspect off-by-one\n"
            "    return items[start:end]\n"
            "```\n\n"
            "Your Output:\n"
            "Produced no edit. Summarized: \"might be inclusive end confusion\".\n\n"
            "Self-Evaluation Score: 0.34/1.0\n"
            "Analysis: I think end should be start+page_size (exclusive) but I'm not 100%."
        )
        asst_a = (
            "<context>\n"
            "Python slices are half-open: items[a:b] includes a..b-1. Using `- 1` when computing `end` then passing to an exclusive end drops the last element on full pages.\n"
            "</context>\n"
            "<code path=\"paginator.py\">\n"
            "end = start + page_size - 1\n"
            "return items[start:end]\n"
            "</code>\n"
            "<question>\n"
            "Confirm contract is exactly `page_size` items for full pages. If so, should I:\n"
            "1) change `end` to `start + page_size` and keep the half-open slice, and\n"
            "2) add a brief comment on half-open slicing?\n"
            "Any edge cases to honor (start < 0, start >= len, empty list)?\n"
            "</question>"
        )
        pairs += [{"role": "user", "content": user_a}, {"role": "assistant", "content": asst_a}]

        # Pair B — Unknown 'strict' + normalization
        user_b = (
            "Problem: ValueError: unknown key 'strict' during config parsing. The flag is later read in validation.\n\n"
            "Stage: repair\n"
            "Stage Input:\n"
            "### config_schema.py\n"
            "```\n"
            "ALLOWED = {\n"
            "    'timeout', 'retries', 'endpoint'\n"
            "}\n"
            "```\n"
            "### parser.py\n"
            "```\n"
            "def load_config(d):\n"
            "    # Validate keys\n"
            "    for k in d:\n"
            "        if k not in ALLOWED:\n"
            "            raise ValueError(f\"unknown key '{k}'\")\n"
            "    # Later logic expects 'strict' to be a boolean\n"
            "    if d.get('strict'):\n"
            "        validate_strict(d)\n"
            "    return d\n"
            "```\n\n"
            "Your Output:\n"
            "Proposed: add 'strict' to ALLOWED; uncertain where to coerce to bool (parser vs. downstream).\n\n"
            "Self-Evaluation Score: 0.39/1.0\n"
            "Analysis: Leaning toward normalization in load_config, but want confirmation about accepted string/int forms."
        )
        asst_b = (
            "<context>\n"
            "'strict' is referenced later as a boolean but currently rejected by allowlist. The right boundary is input loading.\n"
            "</context>\n"
            "<code path=\"config_schema.py\">\n"
            "ALLOWED = {\n"
            "    'timeout', 'retries', 'endpoint'\n"
            "}\n"
            "</code>\n"
            "<code path=\"parser.py\">\n"
            "if k not in ALLOWED:\n"
            "    raise ValueError(f\"unknown key '{k}'\")\n"
            "if d.get('strict'):\n"
            "    validate_strict(d)\n"
            "</code>\n"
            "<question>\n"
            "Should I (a) add 'strict' to ALLOWED and (b) canonicalize 'strict' to a boolean inside load_config, accepting common\n"
            "truthy/falsey strings ('true'/'false', '1'/'0', 'yes'/'no') and numerics? Default when absent?\n"
            "</question>"
        )
        pairs += [{"role": "user", "content": user_b}, {"role": "assistant", "content": asst_b}]

        # Pair C — Backoff ms vs seconds
        user_c = (
            "Problem: Retry delays are ~1000× too long in production; likely milliseconds passed to time.sleep (seconds).\n\n"
            "Stage: repair\n"
            "Stage Input:\n"
            "### backoff.py\n"
            "```\n"
            "def backoff(attempt):\n"
            "    \"\"\"Exponential backoff duration in milliseconds.\"\"\"\n"
            "    return min(32000, (2 ** attempt) * 100)\n"
            "```\n"
            "### client.py\n"
            "```\n"
            "import time\n"
            "from .backoff import backoff\n"
            "\n"
            "def call(max_attempts=5):\n"
            "    for attempt in range(max_attempts):\n"
            "        try:\n"
            "            return do_request()\n"
            "        except TransientError:\n"
            "            time.sleep(backoff(attempt))  # expects seconds\n"
            "```\n\n"
            "Your Output:\n"
            "No edit. Considered dividing by 1000 at the call site vs. changing backoff() to seconds.\n\n"
            "Self-Evaluation Score: 0.42/1.0\n"
            "Analysis: Unsure about other callers of backoff()."
        )
        asst_c = (
            "<context>\n"
            "Producer returns milliseconds; consumer sleeps in seconds. Safer to convert at call site unless all callers can be migrated.\n"
            "</context>\n"
            "<code path=\"client.py\">\n"
            "time.sleep(backoff(attempt))\n"
            "</code>\n"
            "<question>\n"
            "Prefer patching the call to `time.sleep(backoff(attempt) / 1000.0)` to avoid changing backoff()'s contract? "
            "If you want backoff() to return seconds universally, confirm there are no ms-dependent callers.\n"
            "</question>"
        )
        pairs += [{"role": "user", "content": user_c}, {"role": "assistant", "content": asst_c}]

        return pairs
    
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
                "You craft self-contained queries for an isolated expert (REPAIR stage). "
                "The expert sees ONLY what you place inside tags:\n"
                "<context>background</context>\n"
                "<code path=\"relative/file.py\">verbatim snippet(s)</code>\n"
                "<question>precise question</question>\n\n"
                "Rules:\n"
                "- Paste any referenced code verbatim in <code> with exact file paths.\n"
                "- No paraphrasing/ellipses/renames in code lines.\n"
                "- Add <context> only if essential.\n"
                "- Make the query compact yet fully self-contained."
            )
        }
        
        # Keep the exact-line validation source
        self._current_stage_input = task.stage_input
        
        user_message = {
            "role": "user",
            "content": task.llm_view()
        }

        # Prepend large repair-focused I/O demos
        messages = [system_message, *self._fewshot_pairs(), user_message]
        
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
