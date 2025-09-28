import ast
import logging
import re
from typing import Annotated, Any, Generator, Literal, TypeAlias, Union, List, Dict, Optional

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


class FileSelectionTask(Observation):
    """Task step containing the problem statement and candidate files for selection."""
    kind: Literal["file_selection_task"] = "file_selection_task"
    problem_statement: str = Field(description="The issue description to select relevant files for")
    candidate_files: Dict[str, Dict] = Field(description="Top-k files with enriched context")
    
    def llm_view(self, indent: int | None = 2) -> str:
        files_text = f"Issue to analyze: {self.problem_statement}\n\n"
        files_text += "Candidate files to choose from:\n\n"
        
        for i, (filepath, context) in enumerate(self.candidate_files.items(), 1):
            files_text += f"{i}. **{filepath}**\n"
            
            # Add summary
            summary = context.get('summary', '')
            if summary:
                files_text += f"   Summary: {summary}\n"
            
            # Add function signatures
            functions = context.get('functions', [])
            if functions:
                func_list = ', '.join(functions[:5])  # Limit to 5 functions
                if len(functions) > 5:
                    func_list += f" (and {len(functions) - 5} more)"
                files_text += f"   Functions: {func_list}\n"
            
            # Add class definitions
            classes = context.get('classes', [])
            if classes:
                class_list = ', '.join(classes[:3])  # Limit to 3 classes
                if len(classes) > 3:
                    class_list += f" (and {len(classes) - 3} more)"
                files_text += f"   Classes: {class_list}\n"
            
            # Add key imports
            imports = context.get('imports', [])
            if imports:
                import_list = ', '.join(imports[:3])  # Limit to 3 imports
                if len(imports) > 3:
                    import_list += f" (and {len(imports) - 3} more)"
                files_text += f"   Key imports: {import_list}\n"
            
            files_text += "\n"
        
        return files_text


class FileSelectionResponse(Action):
    """Action containing selected files and reasoning."""
    kind: Literal["file_selection_response"] = "file_selection_response"
    selected_files: List[str] = Field(description="List of selected file paths")
    reasoning: str = Field(default="", description="The reasoning process used for selection")
    format_penalty: float = Field(default=0.0, description="Penalty for extra/garbage content in output")
    garbage_content: str = Field(default="", description="The garbage content that was found")


FileSelectionStep: TypeAlias = Annotated[
    FileSelectionResponse,
    Field(discriminator="kind"),
]

FileSelectionTape = Tape[
    None,
    Union[
        FileSelectionTask,
        FileSelectionResponse,
        LLMOutputParsingFailureAction,
    ],
]


class FileSelectionNode(StandardNode):
    """Node that selects relevant files from candidates."""
    
    max_prompt_length: int = 16000  # Larger for file content analysis
    
    def parse_completion(self, completion: str) -> Generator[Step, None, None]:
        """Parse the LLM completion to extract selected files."""
        try:
            # Find all <file> tags
            file_pattern = r'<file>(.*?)</file>'
            file_matches = re.findall(file_pattern, completion, re.DOTALL)
            
            if not file_matches:
                yield LLMOutputParsingFailureAction(
                    error="No <file> tags found in output", 
                    llm_output=completion
                )
                return
            
            # Extract and validate file paths
            selected_files = []
            for filepath in file_matches:
                filepath = filepath.strip()
                if not filepath:
                    yield LLMOutputParsingFailureAction(
                        error="Empty filepath found in <file> tags", 
                        llm_output=completion
                    )
                    return
                selected_files.append(filepath)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_files = []
            for f in selected_files:
                if f not in seen:
                    seen.add(f)
                    unique_files.append(f)
            selected_files = unique_files
            
            # Extract reasoning (everything before the first <file> tag)
            reasoning = ""
            if '<file>' in completion:
                reasoning = completion.split('<file>')[0].strip()
                # Remove thinking tags if present
                if reasoning.startswith('<thinking>') and '</thinking>' in reasoning:
                    thinking_match = re.search(r'<thinking>(.*?)</thinking>', reasoning, re.DOTALL)
                    if thinking_match:
                        reasoning = thinking_match.group(1).strip()
            
            # Check for garbage content after the last </file> tag
            format_penalty = 0.0
            garbage_content = ""
            
            last_file_matches = list(re.finditer(r'</file>', completion))
            if last_file_matches:
                last_tag_pos = last_file_matches[-1].end()
                content_after = completion[last_tag_pos:].strip()
                
                if content_after:
                    garbage_content = content_after
                    format_penalty = 0.5  # Apply -0.5 penalty for garbage content
                    logger.info(f"Garbage content detected after last tag: {repr(content_after[:100])}")
            
            yield FileSelectionResponse(
                selected_files=selected_files,
                reasoning=reasoning,
                format_penalty=format_penalty,
                garbage_content=garbage_content
            )
            
        except Exception as e:
            logger.info(f"Failed to parse file selection response: {completion}\n\nError: {e}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse file selection response: {e}", 
                llm_output=completion
            )

    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        """Create a prompt for the model to select relevant files."""
        # Extract expert feedback and task from tape
        expert_feedback = None
        task = None
        
        for step in tape.steps:
            if hasattr(step, 'kind') and step.kind == "expert_model_advice":
                expert_feedback = step
            elif isinstance(step, FileSelectionTask):
                task = step
        
        assert task is not None, f"No FileSelectionTask found in tape steps: {[type(s).__name__ for s in tape.steps]}"
        
        system_content = (
            "You are an expert software engineer tasked with selecting the most relevant files "
            "for fixing a given issue. You will be shown candidate files that were identified "
            "through initial search, along with their key components (functions, classes, imports).\n\n"
            "Your goal is to select the files that are most likely to contain the bug or need "
            "modification to fix the issue. Select as many files as you think are necessary - "
            "this could be 1 file for simple issues or several files for complex issues that "
            "span multiple components.\n\n"
            "Consider:\n"
            "- Which files contain the specific functionality mentioned in the issue\n"
            "- Which files are most likely to contain the root cause of the problem\n"
            "- Which files would need to be modified to implement the fix\n"
            "- Dependencies and relationships between files\n\n"
            "Selection guidelines:\n"
            "- Include files that directly implement the problematic functionality\n"
            "- Include related files that might need coordinated changes\n"
            "- Don't include files that are clearly unrelated to the issue\n"
            "- You are selecting from among the provided candidate files ONLY.\n"
            "- Prioritize quality over quantity - select only truly relevant files\n\n"
            "IMPORTANT: Your response must end immediately after the last </file> tag. "
            "Any additional content after the final file tag will result in a penalty.\n\n"
            
            "Format your response as:\n"
            "<thinking>[Your analysis of each candidate file and reasoning for selection]</thinking>\n\n"
            "<file>full/path/to/first/selected/file.py</file>\n"
            "<file>full/path/to/second/selected/file.py</file>\n"
            "<file>full/path/to/additional/files/as/needed.py</file>"
        )
        
        user_content = task.llm_view()
        
        # Add expert feedback if present
        if expert_feedback:
            user_content = expert_feedback.llm_view() + "\n\n" + user_content
        
        system_message = {
            "role": "system",
            "content": system_content
        }
        
        user_message = {
            "role": "user", 
            "content": user_content
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


class FileSelectionAgent(Agent):
    """Agent for selecting relevant files from candidates."""
    
    @classmethod
    def create(cls, system_prompt: str = None, llm: LLM = None, max_prompt_length: int = 16000):
        """Create a FileSelectionAgent.
        
        Args:
            system_prompt: Optional system prompt override
            llm: The LLM to use
            max_prompt_length: Maximum prompt length in tokens
        """
        # Handle the llm parameter correctly for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                FileSelectionNode(
                    name="file_selection",
                    agent_step_cls=FileSelectionStep,
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