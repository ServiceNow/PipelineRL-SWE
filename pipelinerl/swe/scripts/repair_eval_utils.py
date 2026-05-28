import json
import logging
import re
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

SELF_EVAL_SYSTEM_PROMPT = "You are an expert evaluator. Return one score between 0.0 (failed/harmful) and 1.0 (clearly fixes the issue)."

SELF_EVAL_TEMPLATE = (
    "Evaluate the proposed repair.\n"
    "- Focus on correctness, completeness, and avoiding new bugs.\n"
    "- Return exactly one score in <score> tags.\n\n"
    "=== PROBLEM STATEMENT ===\n"
    "{problem_statement}\n\n"
    "=== CODE FILES ===\n"
    "{stage_input}\n\n"
    "=== PROPOSED EDITS ===\n"
    "{stage_output}\n\n"
    "FORMAT:\n"
    "<analysis>your reasoning</analysis>\n"
    "<score>0.0-1.0</score>"
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

    def _clean_path(raw: str) -> str:
        value = raw.strip().strip("` ").strip()
        if value.startswith("###"):
            value = value[3:].strip()
        if value.startswith("[") and "]" in value:
            value = value[1:value.index("]")].strip()
        if value.lower().startswith(("file:", "filename:")):
            value = value.split(":", 1)[1].strip()
        return value.strip().strip("` ")

    def _looks_like_file_path(raw: str) -> bool:
        path = _clean_path(raw)
        if not path or path in {"filename.py", "file.py", "path/to/file.py"}:
            return False
        if path.startswith("<") or path.endswith(">"):
            return False
        if "`" in path or "\t" in path:
            return False
        if any(ch in path for ch in ("*", "|", "{", "}")):
            return False
        lowered = path.lower()
        bad_headings = {
            "analysis", "analysis:", "analysis of the issue", "root cause", "solution",
            "fix", "the fix", "implementation", "patch", "proposed solution",
        }
        if lowered in bad_headings or lowered.startswith(("analysis ", "root cause", "solution ")):
            return False
        if " " in path:
            return False
        if path.endswith(":"):
            path = path[:-1]
        # SWE file paths nearly always have either a directory separator or a source-file suffix.
        suffixes = (
            ".py", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".js",
            ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".php", ".scala", ".sh",
            ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".rst", ".md", ".txt",
        )
        return "/" in path or "\\" in path or path.endswith(suffixes)

    def _path_from_preceding_lines(lines: List[str], marker_index: int) -> str | None:
        # Prefer the nearest plausible path header. This avoids grabbing headings such as
        # "### Analysis" when a later "### pkg/module.py" header precedes the edit.
        for j in range(marker_index - 1, max(-1, marker_index - 25), -1):
            stripped = lines[j].strip()
            if not stripped:
                continue
            candidates: list[str] = []
            if stripped.startswith("###"):
                candidates.append(stripped)
            if stripped.startswith("[") and "]" in stripped:
                candidates.append(stripped)
            if stripped.lower().startswith(("file:", "filename:")):
                candidates.append(stripped)
            # Some models emit a bare path on its own line immediately before the marker.
            bare = stripped.rstrip(":")
            if re.fullmatch(r"[A-Za-z0-9_./\\-]+", bare):
                candidates.append(bare)
            for candidate in candidates:
                if _looks_like_file_path(candidate):
                    return _clean_path(candidate).rstrip(":")
        return None

    def _append_edit(file_path: str | None, search_lines: List[str], replace_lines: List[str]) -> None:
        if not file_path:
            return
        search_text = "\n".join(search_lines).strip("\n")
        replace_text = "\n".join(replace_lines).strip("\n")
        if not search_text and not replace_text:
            return
        edits.append({
            "file_path": file_path,
            "search": search_text,
            "replace": replace_text,
        })

    def _extract_from_lines(lines: List[str]) -> None:
        i = 0
        while i < len(lines):
            if "<<<<<<< SEARCH" not in lines[i]:
                i += 1
                continue
            file_path = _path_from_preceding_lines(lines, i)
            search_start = i + 1
            sep = end = None
            j = search_start
            while j < len(lines):
                stripped = lines[j].strip()
                if sep is None and stripped == "=======":
                    sep = j
                elif sep is not None and ">>>>>>> REPLACE" in stripped:
                    end = j
                    break
                elif sep is None and "<<<<<<< SEARCH" in lines[j]:
                    # Malformed nested block; restart from the newer marker.
                    break
                j += 1
            if sep is not None and end is not None:
                _append_edit(file_path, lines[search_start:sep], lines[sep + 1:end])
                i = end + 1
            else:
                i += 1

    # Parse both fenced code blocks and the whole response. Headers are often outside
    # fences, while some models emit unfenced SEARCH/REPLACE blocks.
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

    for block in [solution_text, *code_blocks]:
        _extract_from_lines(block.split("\n"))

    seen: set[tuple[str, str, str]] = set()
    deduped: List[Dict[str, str]] = []
    for edit in edits:
        key = (edit["file_path"], edit["search"], edit["replace"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edit)
    return deduped


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
    extra_headers: Dict[str, str] | None = None,
    debug_dump_dir: str | None = None,
    debug_metadata: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any], float]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    if isinstance(parameters, DictConfig):
        parameters = OmegaConf.to_container(parameters, resolve=True) or {}
    payload = {"model": model_name, "messages": messages} | (parameters or {})
    start = time.time()
    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        data = await response.json()
    latency = time.time() - start
    text = data["choices"][0]["message"]["content"]
    if text is None and debug_dump_dir:
        dump_dir = Path(debug_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_payload = {
            "request": payload,
            "response": data,
            "latency_s": latency,
            "metadata": debug_metadata or {},
        }
        dump_name = f"{int(time.time() * 1000)}_{(debug_metadata or {}).get('problem_id', 'unknown')}.json"
        dump_path = dump_dir / dump_name
        dump_path.write_text(json.dumps(dump_payload, indent=2))
        logger.warning("Wrote null-content response dump to %s", dump_path)
    usage = data.get("usage", {})
    return text, usage, latency
