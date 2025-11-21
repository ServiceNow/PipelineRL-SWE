import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)

REPAIR_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You will see a bug report and the relevant files. "
    "Produce SEARCH/REPLACE patches using the exact format requested."
)

REPAIR_TEMPLATE = (
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

SELF_EVAL_SYSTEM_PROMPT = (
    "You are an expert evaluator that predicts the quality of repair-stage outputs on a 0.0-1.0 scale."
)

SELF_EVAL_TEMPLATE = (
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
    "{stage_output}\n\n"
    "SCORING GUIDELINES:\n"
    "• 0.0-0.2: Completely incorrect, harmful, or off-target\n"
    "• 0.3-0.4: Partially addresses the issue but has major problems\n"
    "• 0.5-0.6: Good attempt but missing key elements or has notable issues\n"
    "• 0.7-0.8: Solid approach with minor room for improvement\n"
    "• 0.9-1.0: Excellent or perfect solution\n\n"
    "FORMAT YOUR RESPONSE:\n"
    "<analysis>\n"
    "[Step-by-step evaluation explaining your reasoning]\n"
    "</analysis>\n\n"
    "<score>\n"
    "[Single number from 0.0 to 1.0]\n"
    "</score>"
)


def _format_file_context(file_contents: Dict[str, str]) -> str:
    formatted = []
    for path, content in file_contents.items():
        formatted.append(f"### {path}\n```\n{content}\n```\n")
    return "\n".join(formatted)


def build_repair_messages(problem_statement: str, file_contents: Dict[str, str]) -> Tuple[List[Dict[str, str]], str]:
    file_context = _format_file_context(file_contents)
    user_content = REPAIR_TEMPLATE.format(
        problem_statement=problem_statement,
        file_contents=file_context,
    )
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, file_context


def build_self_eval_messages(problem_statement: str, stage_input: str, stage_output: str) -> List[Dict[str, str]]:
    user_content = SELF_EVAL_TEMPLATE.format(
        problem_statement=problem_statement,
        stage_input=stage_input,
        stage_output=stage_output,
    )
    return [
        {"role": "system", "content": SELF_EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_search_replace_edits(solution_text: str) -> List[Dict[str, str]]:
    edits: List[Dict[str, str]] = []
    code_blocks: List[str] = []
    in_block = False
    current: List[str] = []

    for line in solution_text.split("\n"):
        if line.strip().startswith("```"):
            if in_block:
                code_blocks.append("\n".join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)

    for block in code_blocks:
        try:
            lines = block.split("\n")
            file_path = None
            start_index = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("###"):
                    file_path = line.strip()[3:].strip()
                    start_index = i + 1
                    break
            if not file_path:
                continue

            search_start = search_end = replace_start = replace_end = None
            for i, line in enumerate(lines[start_index:], start=start_index):
                if "<<<<<<< SEARCH" in line:
                    search_start = i + 1
                elif "=======" in line and search_start is not None:
                    search_end = i
                    replace_start = i + 1
                elif ">>>>>>> REPLACE" in line and replace_start is not None:
                    replace_end = i
                    break

            if None in (search_start, search_end, replace_start, replace_end):
                continue

            search_text = "\n".join(lines[search_start:search_end])
            replace_text = "\n".join(lines[replace_start:replace_end])
            edits.append({
                "file_path": file_path,
                "search": search_text,
                "replace": replace_text,
            })
        except Exception:
            continue
    return edits


def parse_self_eval_response(response_text: str) -> Tuple[str, float, bool]:
    analysis = ""
    predicted_score = 0.0
    parsing_error = False
    try:
        analysis_start = response_text.find("<analysis>")
        analysis_end = response_text.find("</analysis>")
        if analysis_start != -1 and analysis_end != -1:
            analysis = response_text[analysis_start + 10:analysis_end].strip()
        else:
            parsing_error = True
            score_start = response_text.find("<score>")
            analysis = response_text[:score_start if score_start != -1 else None].strip()

        score_start = response_text.find("<score>")
        score_end = response_text.find("</score>")
        if score_start != -1 and score_end != -1:
            score_text = response_text[score_start + 7:score_end].strip()
            predicted_score = float(score_text)
            predicted_score = max(0.0, min(1.0, predicted_score))
        else:
            parsing_error = True
    except Exception:
        parsing_error = True
        analysis = response_text
        predicted_score = 0.0
    return analysis, predicted_score, parsing_error


async def chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    messages: List[Dict[str, str]],
    parameters: Dict[str, Any] | DictConfig,
    api_key: str | None = None,
) -> Tuple[str, Dict[str, Any], float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        key_preview = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) >= 9 else api_key
        print("chat_completion using api key %s", key_preview)
    if isinstance(parameters, DictConfig):
        parameters = OmegaConf.to_container(parameters, resolve=True) or {}
    payload = {"model": model_name, "messages": messages} | (parameters or {})
    start = time.time()
    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        data = await response.json()
    latency = time.time() - start
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage, latency
