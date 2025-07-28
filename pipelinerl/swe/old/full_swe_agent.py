import logging
import os
import re
from pathlib import Path
from typing import Generator, List, Literal, Optional, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    LLMOutputParsingFailureAction,
    Observation,
    Prompt,
    AgentStep,
    Tape,
    Thought,
    Action,
    SetNextNode,
    FinalStep
)
from tapeagents.llms import LLM
from tapeagents.nodes import StandardNode
from tapeagents.environment import Environment

logger = logging.getLogger(__name__)

def extract_file_paths(content: str) -> list:
    """Extracts file paths from numbered lists in LLM-generated content."""
    file_paths = []
    
    # Pattern to identify numbered list items
    list_pattern = re.compile(r'^\s*\d+\.\s+(.*?)$', re.MULTILINE)
    
    # Pattern to extract likely file paths
    # Matches continuous sequences that look like file paths, with or without backticks
    file_path_pattern = re.compile(r'`?([a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]+)*(?:\.[a-zA-Z0-9]+)?)`?')
    
    # Find all numbered list items
    list_items = list_pattern.findall(content)
    
    for item in list_items:
        # Get the first match that looks like a file path
        match = file_path_pattern.search(item)
        if match:
            file_path = match.group(1)
            # Skip if it looks like a sentence rather than a file path
            if len(file_path.split()) > 1 or len(file_path) > 255:
                continue
            file_paths.append(file_path)
    
    return list(set(file_paths))  # Remove duplicates

class SWEIssue(Observation):
    """Step representing an issue to be solved."""
    kind: Literal["issue_problem"] = "issue_problem"
    problem_statement: str = Field(
        description="The description of the issue to be fixed."
    )
    repo: str = Field(
        description="The name of the repository containing the code."
    )

    def llm_view(self, indent: int | None = 2) -> str:
        return f"Issue: {self.problem_statement}\nRepository: {self.repo}"


class FileListObservation(Observation):
    """Observation containing the list of files in a repository."""
    kind: Literal["file_list_observation"] = "file_list_observation"
    repo_path: str = Field(description="Path to the repository")
    file_list: List[str] = Field(description="List of file paths in the repository")

    def llm_view(self, indent: int | None = 2) -> str:
        # Format the file list for LLM consumption
        formatted_list = "\n".join(self.file_list)
        return f"Repository files for {self.repo_path}:\n\n{formatted_list}"


class FileLocalizationThought(Thought):
    """Thought produced during file localization."""
    kind: Literal["file_localization_thought"] = "file_localization_thought"
    reasoning: str = Field(description="Reasoning about which files need to be examined")
    

class FileLocalizationAction(Action):
    """Action to request specific files to be examined."""
    kind: Literal["file_localization_action"] = "file_localization_action"
    file_paths: List[str] = Field(
        description="List of file paths that need to be examined to fix the issue."
    )


class FileContentObservation(Observation):
    """Observation containing file contents retrieved from the filesystem."""
    kind: Literal["file_content_observation"] = "file_content_observation"
    files: dict = Field(
        description="Dictionary mapping file paths to their contents."
    )

    def llm_view(self, indent: int | None = 2) -> str:
        result = []
        for path, content in self.files.items():
            result.append(f"--- BEGIN FILE: {path} ---\n```\n{content}\n```\n--- END FILE ---")
        return "\n\n".join(result)


class RepairThought(Thought):
    """Thought produced during the repair reasoning process."""
    kind: Literal["repair_thought"] = "repair_thought"
    reasoning: str = Field(description="Chain of thoughts for repairing the issue")


class SearchReplaceEditAction(Action):
    """Action representing search/replace edits to fix the issue."""
    kind: Literal["search_replace_edit_action"] = "search_replace_edit_action"
    edits: List[dict] = Field(
        description="List of search/replace edits. Each edit contains 'file_path', 'search', and 'replace'."
    )


def get_file_list(repo_path: Path, max_depth: int = 4) -> List[str]:
    """Generate a list of file paths in the repository.
    
    Args:
        repo_path: Path to the repository
        max_depth: Maximum depth to traverse
        
    Returns:
        A list of file paths without the top-level repository directory
    """
    if not os.path.exists(repo_path):
        raise Exception(f"Repository path not found: {repo_path}")
    
    result = []
    excluded_dirs = ['.git', 'node_modules', 'venv', '__pycache__']
    repo_name = os.path.basename(repo_path)
    
    def _generate_paths(current_path, depth=0, relative_path=""):
        if depth > max_depth:
            # Strip repo name from the path
            stripped_path = relative_path.replace(f"{repo_name}/", "", 1)
            result.append(f"{stripped_path}/...")
            return
        
        try:
            entries = sorted([entry for entry in os.listdir(current_path) if not entry.startswith('.')])
            # Filter out common directories that are likely not relevant
            entries = [entry for entry in entries if entry not in excluded_dirs]
            
            for entry in entries:
                entry_path = os.path.join(current_path, entry)
                entry_relative_path = os.path.join(relative_path, entry) if relative_path else entry
                
                if os.path.isdir(entry_path):
                    # Only recursively process subdirectories without adding them to the result
                    if depth < max_depth:
                        _generate_paths(entry_path, depth + 1, entry_relative_path)
                else:
                    # For files, strip the repo name from the path
                    stripped_path = entry_relative_path.replace(f"{repo_name}/", "", 1)
                    result.append(stripped_path)
        except PermissionError:
            # Strip repo name from the permission denied error
            stripped_path = relative_path.replace(f"{repo_name}/", "", 1)
            result.append(f"{stripped_path}/ [Permission denied]")
    
    # Generate paths starting from the repository
    _generate_paths(repo_path, 0, repo_name)
    
    return result


class FileListRequestAction(Action):
    """Action explicitly requesting a file list from the environment."""
    kind: Literal["file_list_request_action"] = "file_list_request_action"


class FileListRequestNode(StandardNode):
    """Initial node that requests a file list observation by allowing the environment to react."""
    
    def make_prompt(self, agent, tape: Tape) -> Prompt:
        """Create a minimal prompt as this node doesn't need to query the LLM."""
        return Prompt(messages=[{"role": "system", "content": "Requesting repository file list"}])
    
    def generate_steps(self, agent, tape, llm_stream) -> Generator[AgentStep, None, None]:
        """First yield an explicit request action, then transition to the next node."""
        yield FileListRequestAction()


class FileLocalizationNode(StandardNode):
    """Node for localizing the files that need to be examined to fix the issue."""
    
    def make_prompt(self, agent, tape: Tape) -> Prompt:
        """Create a prompt to identify files that need to be examined."""
        view = agent.compute_view(tape)
        
        # Find the issue problem step and file list observation
        issue_step = None
        file_list_observation = None
        
        for step in tape.steps:
            if isinstance(step, SWEIssue):
                issue_step = step
            elif isinstance(step, FileListObservation):
                file_list_observation = step
        
        if not issue_step:
            raise ValueError("No SWEIssue found in the tape")
        if not file_list_observation:
            raise ValueError("No FileListObservation found in the tape")
        
        system_message = {
            "role": "system",
            "content": (
                "You are an expert software engineer tasked with fixing issues in a repository. "
                "Your first step is to identify which files need to be examined and potentially modified."
            )
        }
        
        user_message = {
            "role": "user",
            "content": (
                f"I need to fix an issue in a repository. Here's the issue description:\n\n"
                f"--- BEGIN ISSUE ---\n{issue_step.problem_statement}\n--- END ISSUE ---\n\n"
                f"Here's the repository file structure:\n\n{file_list_observation.llm_view()}\n\n"
                f"Based on the issue description, which files do you think need to be examined "
                f"and potentially modified to fix this issue? Please provide a list of file paths, "
                f"one per line, in numbered format (e.g. \"1. `filepath1.file`\n2. `filepath2.file`\", etc), with a brief explanation of why each file might be relevant."
            )
        }
        
        return Prompt(messages=[system_message, user_message])
    
    def generate_steps(self, agent, tape, llm_stream) -> Generator[AgentStep, None, None]:
        """Generate steps based on the LLM's output."""
        output = llm_stream.get_output()
        content = output.content
        
        if not content:
            yield LLMOutputParsingFailureAction(
                error="Empty response from LLM", 
                llm_output=""
            )
            return
        
        try:
            # First yield a thought containing the reasoning
            yield FileLocalizationThought(reasoning=content)
            
            # Extract file paths from the response
            # This is a simple heuristic: find lines that look like file paths
            file_paths = extract_file_paths(content)

            # If we found file paths, yield an action to request them
            if file_paths:
                print("file paths:", file_paths)
                action = FileLocalizationAction(file_paths=file_paths)
                print(f"🟢 Type of action: {type(action)}")
                print(f"🟢 Is subclass of AgentStep? {issubclass(type(action), AgentStep)}")
                yield action
            else:
                yield LLMOutputParsingFailureAction(
                    error="No file paths found in the response", 
                    llm_output=content
                )
        except Exception as e:
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse file localization response: {str(e)}", 
                llm_output=content
            )


class RepairNode(StandardNode):
    """Node for generating repairs based on file contents."""
    
    def make_prompt(self, agent, tape: Tape) -> Prompt:
        """Create a prompt to generate search/replace edits."""
        view = agent.compute_view(tape)
        
        # Find the issue problem step and file content observation
        issue_step = None
        file_content_observation = None
        
        for step in tape.steps:
            if isinstance(step, SWEIssue):
                issue_step = step
            elif isinstance(step, FileContentObservation):
                file_content_observation = step
        
        if not issue_step:
            raise ValueError("No SWEIssue found in the tape")
        if not file_content_observation:
            raise ValueError("No FileContentObservation found in the tape")
        
        # Create a combined representation of all file contents
        content = file_content_observation.llm_view()
        
        system_message = {
            "role": "system",
            "content": (
                "A user will ask you to solve a task. You should first draft your thinking process (inner "
                "monologue). Then, generate the solution. "
                "Your response format must follow the template below:\n"
                "<think>\n"
                "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as "
                "casual and as long as you want until you are confident to generate a correct solution.\n"
                "</think>\n"
                "<solution>\n"
                "Final solution presented to the user.\n"
                "</solution>"
            )
        }
        
        user_message = {
            "role": "user",
            "content": (
                f"We are currently solving the following issue within our repository. Here is the issue "
                f"text:\n"
                f"--- BEGIN ISSUE ---\n{issue_step.problem_statement}\n--- END ISSUE ---\n\n"
                f"Below are some code segments, each from a relevant file. One or more of these files may "
                f"contain bugs.\n{content}\n\n"
                f"Please first localize the bug based on the issue statement, and then generate *SEARCH/"
                f"REPLACE* edits to fix the issue.\n"
                f"Every *SEARCH/REPLACE* edit must use this format:\n"
                f"1. The file path\n"
                f"2. The start of search block: <<<<<<< SEARCH\n"
                f"3. A contiguous chunk of lines to search for in the existing source code\n"
                f"4. The dividing line: =======\n"
                f"5. The lines to replace into the source code\n"
                f"6. The end of the replace block: >>>>>>> REPLACE\n"
                f"Here is an example:\n"
                f"```python\n"
                f"### mathweb/flask/app.py\n"
                f"<<<<<<< SEARCH\n"
                f"from flask import Flask\n"
                f"=======\n"
                f"import math\n"
                f"from flask import Flask\n"
                f">>>>>>> REPLACE\n"
                f"```\n"
                f"Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like "
                f"to add the line ' print(x)', you must fully write that out, with all those "
                f"spaces before the code!\n"
                f"Wrap each *SEARCH/REPLACE* edit in a code block as shown in the example above. If you "
                f"have multiple *SEARCH/REPLACE* edits, use a separate code block for each one."
            )
        }
        
        return Prompt(messages=[system_message, user_message])
    
    # In RepairNode.generate_steps method:
    def generate_steps(self, agent, tape, llm_stream) -> Generator[AgentStep, None, None]:
        """Generate steps based on the LLM's output."""
        output = llm_stream.get_output()
        content = output.content
        
        if not content:
            yield LLMOutputParsingFailureAction(
                error="Empty response from LLM", 
                llm_output=""
            )
            # Add this line to tell the agent where to go next
            yield FinalStep(reason="error no llm output")
            return
        
        try:
            # Extract the thinking part and solution part more robustly
            think_parts = content.split("<think>")
            if len(think_parts) > 1:
                think_match = think_parts[1].split("</think>")[0].strip()
            else:
                think_match = ""
            
            if not think_match:
                error_msg = "Could not extract thinking part"
                print(f"Error: {error_msg}")
                yield LLMOutputParsingFailureAction(
                    error=error_msg, 
                    llm_output=content
                )
                # Add this line
                yield FinalStep(reason="error, no thinking step")
                return
            
            # Yield the thinking part as a repair thought
            if think_match:
                yield RepairThought(reasoning=think_match)
        
            edits = self._extract_search_replace_edits(content)
            if edits:
                print(f"Found {len(edits)} edits")
                yield SearchReplaceEditAction(edits=edits)
                # Add this to specify the next node after successful parsing
                yield FinalStep(reason="done")  # Or another appropriate node
            else:
                error_msg = "No valid search/replace edits found in the solution"
                print(f"Error: {error_msg}")
                yield LLMOutputParsingFailureAction(
                    error=error_msg, 
                    llm_output=content
                )
                # Add this line
                yield FinalStep(reason="error")
            
        except Exception as e:
            print(f"Exception in RepairNode.generate_steps: {str(e)}")
            yield LLMOutputParsingFailureAction(
                error=f"Failed to parse repair response: {str(e)}", 
                llm_output=content
            )
            # Add this line
            yield FinalStep(reason="error")
    
    def _extract_search_replace_edits(self, solution_text: str) -> List[dict]:
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


# Define the tape type for SWE tasks
SWETape = Tape[
    None,
    Union[
        SWEIssue,
        FileListObservation,
        FileListRequestAction,
        FileLocalizationThought,
        FileLocalizationAction,
        FileContentObservation,
        RepairThought,
        SearchReplaceEditAction,
        LLMOutputParsingFailureAction,
        FinalStep
    ],
]


class SWEAgent(Agent):
    """Agent for solving software engineering tasks."""
    
    @classmethod
    def create(cls, llm=None, max_iterations: int = 10):
        """Create a new SWEAgent.
        
        Args:
            llm: The LLM to use for generating responses. Can be a single LLM or a dictionary of LLMs.
            max_iterations: Maximum number of iterations to run
            
        Returns:
            A new SWEAgent
        """
        # Handle the llm parameter correctly for the Agent base class
        llms = llm
        if llm is not None and not isinstance(llm, dict):
            llms = {"default": llm}
            
        agent = super().create(
            llms=llms,
            nodes=[
                FileListRequestNode(name="file_list_request"),
                FileLocalizationNode(name="file_localization"),
                RepairNode(name="repair"),
            ],
            max_iterations=max_iterations,
        )
        return agent


# Environment for SWE tasks
class SWEEnvironment(Environment):
    """Environment for SWE tasks."""
    
    def __init__(self):
        pass
    
    def react(self, tape: SWETape) -> SWETape:
        """React to the agent's actions.
        
        Args:
            tape: The current tape
            
        Returns:
            Updated tape with observations
        """
        new_tape = tape.model_copy(deep=True)
        
        # Check for any FileListRequestAction in the tape
        for step in tape.steps:
            if isinstance(step, FileListRequestAction):
                # Find the SWEIssue to get the repo path
                repo = None
                for search_step in tape.steps:
                    if isinstance(search_step, SWEIssue):
                        repo = search_step.repo
                        break
                
                if not repo:
                    print("No repository found in the tape")
                    raise ValueError("No repository found in the tape")
                
                # Only generate the file list if we haven't already
                file_list_exists = any(isinstance(s, FileListObservation) for s in tape.steps)
                if not file_list_exists:
                    repo_path = Path(repo)
                    file_list = get_file_list(repo_path)
                    
                    new_tape.steps.append(FileListObservation(repo_path=str(repo_path), file_list=file_list))
                    print(f"Added FileListObservation with {len(file_list)} files to tape")
                
                break  # We've handled the file list request
        
        # Check for any FileLocalizationAction in the tape
        for step in tape.steps:
            if isinstance(step, FileLocalizationAction):
                # Find the SWEIssue to get the repo path
                repo = None
                for search_step in tape.steps:
                    if isinstance(search_step, SWEIssue):
                        repo = search_step.repo
                        break
                
                if not repo:
                    print("No repository found in the tape")
                    raise ValueError("No repository found in the tape")
                
                # Only process this action if we haven't already done so
                # Check if we've already processed this specific action's file paths
                file_contents_exist = False
                for search_step in tape.steps:
                    if isinstance(search_step, FileContentObservation):
                        # We've already processed some file content
                        # This is a simplification - in a real system, you might want to check
                        # if the specific file paths from this action have been processed
                        file_contents_exist = True
                        break
                
                if not file_contents_exist:
                    # Load file contents
                    files = {}
                    repo_path = Path(repo)
                    print(f"Looking for files in repository: {repo_path}")
                    
                    for file_path in step.file_paths:
                        full_path = repo_path / file_path
                        print(f"Checking file: {full_path}, exists: {os.path.exists(full_path)}")
                        if os.path.exists(full_path):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                files[file_path] = f.read()
                                print(f"Successfully read file: {file_path}")
                        else:
                            print(f"File not found: {full_path}")
                            logger.warning(f"File not found: {full_path}")
                    
                    if not files:
                        print("No files were found!")
                    
                    new_tape.steps.append(FileContentObservation(files=files))
                    print(f"Added FileContentObservation with {len(files)} files to tape")
                
                break  # We've handled the file localization action
        
        return new_tape