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

        # dump the formatted template into a file
        #with open("swe_task.txt", "w") as f:
        #    f.write(processed_template)
        
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
        
        # Simplified system message following Option 1
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