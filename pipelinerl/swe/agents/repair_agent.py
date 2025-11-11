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

from pipelinerl.swe.types import ExpertModelAdvice

logger = logging.getLogger(__name__)


class RepairTask(Observation):
    kind: Literal["repair_task"] = "repair_task"
    problem_statement: str
    file_contents: dict[str, str]  # map of file path to content
    template: str = Field(
        default=(
            "Analyze the following code to find and fix bugs. Use this format:\n\n"
            "<think>\n"
            "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
            "</think>\n\n"
            "<solution>\n"
            "[Your SEARCH/REPLACE edits using this format:]\n\n"
            "### filename.py\n"
            "<<<<<<< SEARCH\n"
            "[exact code to find]\n"
            "=======\n"
            "[replacement code]\n"
            ">>>>>>> REPLACE\n"
            "</solution>\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- Every SEARCH/REPLACE edit must use the exact format above\n"
            "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
            "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
            "- Wrap each SEARCH/REPLACE edit in a code block\n"
            "- Use separate code blocks for multiple edits\n\n"
            "Example:\n"
            "```python\n"
            "### mathweb/flask/app.py\n"
            "<<<<<<< SEARCH\n"
            "from flask import Flask\n"
            "=======\n"
            "import math\n"
            "from flask import Flask\n"
            ">>>>>>> REPLACE\n"
            "```\n\n"
            "Here is the issue:\n"
            "--- BEGIN ISSUE ---\n"
            "{problem_statement}\n"
            "--- END ISSUE ---\n\n"
            "Below are the code files that may contain bugs:\n"
            "{file_contents}"
        )
    )

    def llm_view(self, indent: int | None = 2) -> str:
        # Format the file contents for display
        formatted_contents = ""
        for file_path, content in self.file_contents.items():
            formatted_contents += f"### {file_path}\n```\n{content}\n```\n\n"

        processed_template = self.template.format(
            problem_statement=self.problem_statement,
            file_contents=formatted_contents
        )
        return processed_template


class SearchReplaceResponse(Thought):
    """
    Response containing search/replace edits produced by the agent.
    """
    kind: Literal["search_replace_response"] = "search_replace_response"
    response: str = Field(description="complete response with search/replace edits")
    edits: list[dict] = Field(default_factory=list, description="parsed search/replace edits")


RepairStep: TypeAlias = Annotated[
    SearchReplaceResponse,
    Field(discriminator="kind"),
]

RepairTape = Tape[
    None,
    Union[
        RepairTask,
        SearchReplaceResponse,
        LLMOutputParsingFailureAction,
        ExpertModelAdvice,
    ],
]


class RepairNode(StandardNode):
    max_prompt_length: int = 16000  # Increased for code tasks which tend to be longer

    # --------------------------------
    # Few-shot pairs when advice exists
    # --------------------------------
    def _advice_present_fewshots(self) -> list[dict]:
        """
        Used ONLY when ExpertModelAdvice is present.
        Large repair demos:
        - USER: expert guidance block + your repair task template + full code context
        - ASSISTANT: detailed <think> integrating the advice + strict SEARCH/REPLACE edits
        """
        demos: list[dict] = []

        # 1) Pagination off-by-one
        user1 = (
            "=== EXPERT GUIDANCE FOR REPAIR ===\n"
            "You previously asked for guidance: Pagination off-by-one; should end be start+page_size?\n\n"
            "Expert advice received:\n"
            "- Compute end = start + page_size; Python slice end is exclusive.\n"
            "- Keep items[start:end].\n"
            "- Re-check conceptual edges.\n\n"
            "Please incorporate this expert guidance to improve your repair output.\n"
            "=== END EXPERT GUIDANCE ===\n\n"
            "Analyze the following code to find and fix bugs. Use this format:\n\n"
            "<think>\n"
            "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
            "</think>\n\n"
            "<solution>\n"
            "[Your SEARCH/REPLACE edits using this format:]\n\n"
            "### filename.py\n"
            "<<<<<<< SEARCH\n"
            "[exact code to find]\n"
            "=======\n"
            "[replacement code]\n"
            ">>>>>>> REPLACE\n"
            "</solution>\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- Every SEARCH/REPLACE edit must use the exact format above\n"
            "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
            "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
            "- Wrap each SEARCH/REPLACE edit in a code block\n"
            "- Use separate code blocks for multiple edits\n\n"
            "Here is the issue:\n"
            "--- BEGIN ISSUE ---\n"
            "Off-by-one: full pages return 49 instead of 50.\n"
            "--- END ISSUE ---\n\n"
            "Below are the code files that may contain bugs:\n"
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
            "```\n"
        )
        asst1 = (
            "<think>\n"
            "Given the advice, the intended contract matches half-open slicing. The current code subtracts 1 from end, then uses an exclusive end, "
            "losing the last element on exact multiples. Minimal edit: compute end = start + page_size and keep the slice. "
            "No other callers are shown; this change is local and safe. Edge cases: when page_size<=0 we already return [].\n"
            "</think>\n\n"
            "<solution>\n"
            "```python\n"
            "### paginator.py\n"
            "<<<<<<< SEARCH\n"
            "    end = start + page_size - 1  # suspect off-by-one\n"
            "    return items[start:end]\n"
            "=======\n"
            "    end = start + page_size  # half-open slice end is exclusive\n"
            "    return items[start:end]\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "</solution>"
        )
        demos += [{"role": "user", "content": user1}, {"role": "assistant", "content": asst1}]

        # 2) Unknown 'strict' + normalization
        user2 = (
            "=== EXPERT GUIDANCE FOR REPAIR ===\n"
            "You previously asked for guidance: 'strict' rejected by ALLOWED; normalize to bool at load time.\n\n"
            "Expert advice received:\n"
            "- Add 'strict' to ALLOWED.\n"
            "- Normalize 'strict' to boolean inside load_config.\n"
            "- Accept common truthy/falsey forms.\n\n"
            "Please incorporate this expert guidance to improve your repair output.\n"
            "=== END EXPERT GUIDANCE ===\n\n"
            "Analyze the following code to find and fix bugs. Use this format:\n\n"
            "<think>\n"
            "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
            "</think>\n\n"
            "<solution>\n"
            "[Your SEARCH/REPLACE edits using this format:]\n\n"
            "### filename.py\n"
            "<<<<<<< SEARCH\n"
            "[exact code to find]\n"
            "=======\n"
            "[replacement code]\n"
            ">>>>>>> REPLACE\n"
            "</solution>\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- Every SEARCH/REPLACE edit must use the exact format above\n"
            "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
            "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
            "- Wrap each SEARCH/REPLACE edit in a code block\n"
            "- Use separate code blocks for multiple edits\n\n"
            "Here is the issue:\n"
            "--- BEGIN ISSUE ---\n"
            "Config parser raises ValueError for unknown key 'strict', but downstream reads it as a boolean feature flag.\n"
            "--- END ISSUE ---\n\n"
            "Below are the code files that may contain bugs:\n"
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
            "```\n"
        )
        asst2 = (
            "<think>\n"
            "We must align schema and usage. Add 'strict' to the allowlist, and normalize at the input boundary so downstream code sees a bool. "
            "Normalization should accept common truthy/falsey strings and numerics. Keeping behavior local to load_config avoids ripple effects.\n"
            "</think>\n\n"
            "<solution>\n"
            "```python\n"
            "### config_schema.py\n"
            "<<<<<<< SEARCH\n"
            "ALLOWED = {\n"
            "    'timeout', 'retries', 'endpoint'\n"
            "}\n"
            "=======\n"
            "ALLOWED = {\n"
            "    'timeout', 'retries', 'endpoint', 'strict'\n"
            "}\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "```python\n"
            "### parser.py\n"
            "<<<<<<< SEARCH\n"
            "def load_config(d):\n"
            "    # Validate keys\n"
            "    for k in d:\n"
            "        if k not in ALLOWED:\n"
            "            raise ValueError(f\"unknown key '{k}'\")\n"
            "    # Later logic expects 'strict' to be a boolean\n"
            "    if d.get('strict'):\n"
            "        validate_strict(d)\n"
            "    return d\n"
            "=======\n"
            "def load_config(d):\n"
            "    # Validate keys\n"
            "    for k in d:\n"
            "        if k not in ALLOWED:\n"
            "            raise ValueError(f\"unknown key '{k}'\")\n"
            "    # Normalize 'strict' to bool (accepting common forms)\n"
            "    if 'strict' in d:\n"
            "        v = d['strict']\n"
            "        if isinstance(v, str):\n"
            "            lv = v.strip().lower()\n"
            "            if lv in ('true', '1', 'yes', 'on'):\n"
            "                d['strict'] = True\n"
            "            elif lv in ('false', '0', 'no', 'off'):\n"
            "                d['strict'] = False\n"
            "        elif isinstance(v, (int, float)):\n"
            "            d['strict'] = bool(v)\n"
            "        elif isinstance(v, bool):\n"
            "            pass  # already boolean\n"
            "    # Later logic expects 'strict' to be a boolean\n"
            "    if d.get('strict'):\n"
            "        validate_strict(d)\n"
            "    return d\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "</solution>"
        )
        demos += [{"role": "user", "content": user2}, {"role": "assistant", "content": asst2}]

        # 3) Backoff units mismatch (convert at call site)
        user3 = (
            "=== EXPERT GUIDANCE FOR REPAIR ===\n"
            "You previously asked for guidance: backoff() returns ms but time.sleep expects seconds.\n\n"
            "Expert advice received:\n"
            "- Prefer converting at the call site: divide by 1000.0.\n"
            "- Consider migrating backoff() later only if all callers updated coherently.\n\n"
            "Please incorporate this expert guidance to improve your repair output.\n"
            "=== END EXPERT GUIDANCE ===\n\n"
            "Analyze the following code to find and fix bugs. Use this format:\n"
            "<think>\n"
            "[Your analysis process - be as detailed as you want until you're confident in your solution]\n"
            "</think>\n\n"
            "<solution>\n"
            "[Your SEARCH/REPLACE edits using this format:]\n\n"
            "### filename.py\n"
            "<<<<<<< SEARCH\n"
            "[exact code to find]\n"
            "=======\n"
            "[replacement code]\n"
            ">>>>>>> REPLACE\n"
            "</solution>\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "- Every SEARCH/REPLACE edit must use the exact format above\n"
            "- The SEARCH block must contain a contiguous chunk of lines that exist in the source code\n"
            "- PROPER INDENTATION IS CRITICAL - if you want to add '    print(x)', you must include all those spaces\n"
            "- Wrap each SEARCH/REPLACE edit in a code block\n"
            "- Use separate code blocks for multiple edits\n\n"
            "Here is the issue:\n"
            "--- BEGIN ISSUE ---\n"
            "Delays are ~1000× longer than configured.\n"
            "--- END ISSUE ---\n\n"
            "Below are the code files that may contain bugs:\n"
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
            "```\n"
        )
        asst3 = (
            "<think>\n"
            "Advice indicates we should preserve backoff()'s current ms contract and adjust only the consumer. "
            "Therefore patch client.py to divide the value by 1000.0 when sleeping. This is minimal and avoids breaking other callers.\n"
            "</think>\n\n"
            "<solution>\n"
            "```python\n"
            "### client.py\n"
            "<<<<<<< SEARCH\n"
            "            time.sleep(backoff(attempt))  # expects seconds\n"
            "=======\n"
            "            time.sleep(backoff(attempt) / 1000.0)  # convert ms→s\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "</solution>"
        )
        demos += [{"role": "user", "content": user3}, {"role": "assistant", "content": asst3}]

        return demos

    def _extract_search_replace_edits(self, solution_text: str) -> list[dict]:
        """Extract search/replace edits from the solution text.
        
        Args:
            solution_text: The solution text containing search/replace edits
            
        Returns:
            A list of dictionaries, each containing 'file_path', 'search', and 'replace'
        """
        edits = []
        
        # Split the solution text by code blocks
        code_blocks = []
        in_code_block = False
        current_block = []
        
        for line in solution_text.split('\n'):
            if line.strip().startswith('```'):
                if in_code_block:
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        
        # Process each code block
        for block in code_blocks:
            try:
                lines = block.split('\n')
                
                # Extract the file path (first line should start with "###")
                file_path = None
                start_index = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('###'):
                        file_path = line.strip()[3:].strip()
                        start_index = i + 1
                        break
                
                if not file_path:
                    continue
                
                # Find the search and replace parts
                search_start = None
                search_end = None
                replace_start = None
                replace_end = None
                
                for i, line in enumerate(lines[start_index:], start=start_index):
                    if "<<<<<<< SEARCH" in line:
                        search_start = i + 1
                    elif "=======" in line and search_start is not None:
                        search_end = i
                        replace_start = i + 1
                    elif ">>>>>>> REPLACE" in line and replace_start is not None:
                        replace_end = i
                        break
                
                if None in [search_start, search_end, replace_start, replace_end]:
                    continue
                
                # Extract the search and replace parts
                search_text = '\n'.join(lines[search_start:search_end])
                replace_text = '\n'.join(lines[replace_start:replace_end])
                
                edits.append({
                    'file_path': file_path,
                    'search': search_text,
                    'replace': replace_text
                })
                
            except Exception as e:
                logger.warning(f"Failed to parse code block: {e}")
        
        return edits

    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        try:
            # Extract and parse search/replace edits
            edits = self._extract_search_replace_edits(completion)
            
            if not edits:
                logger.warning("No valid search/replace edits found in the response")
            
            step = SearchReplaceResponse(response=completion, edits=edits)
            yield step
            
        except Exception as e:
            logger.info(f"Failed to parse agent output: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse agent output: {completion}\n\nError: {e}", 
                llm_output=completion
            )
            return

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt for the model to perform code repair."""
        # Extract expert feedback and task from tape
        expert_feedback = None
        task = None
        
        for step in tape.steps:
            if hasattr(step, 'kind') and step.kind == "expert_model_advice":
                expert_feedback = step
            elif isinstance(step, RepairTask):
                task = step
        
        assert task is not None, f"No RepairTask found in tape steps: {[type(s).__name__ for s in tape.steps]}"
        
        # Simplified system message (instructions live in the user content/template)
        system_message = {
            "role": "system",
            "content": "You are a helpful coding assistant that analyzes code and fixes bugs."
        }
        
        # All task-specific instructions now in the user message
        user_content = task.llm_view()
        
        # Add expert feedback if present
        if expert_feedback:
            user_content = expert_feedback.llm_view() + "\n\n" + user_content
        
        user_message = {
            "role": "user",
            "content": user_content
        }

        # If we have advice, prepend the large repair demos
        if expert_feedback:
            messages = [system_message, *self._advice_present_fewshots(), user_message]
        else:
            messages = [system_message, user_message]
        
        prompt_token_ids = agent.llm.tokenizer.apply_chat_template(
            messages, add_special_tokens=True, add_generation_prompt=True
        )
        prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
        return Prompt(messages=messages, token_ids=prompt_token_ids)


#### Agent and Environment ####
class RepairAgent(Agent):
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 16000):
        # Handle the llm parameter correctly for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                RepairNode(
                    name="repair",
                    agent_step_cls=RepairStep,
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
