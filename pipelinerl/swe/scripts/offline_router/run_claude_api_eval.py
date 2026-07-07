#!/usr/bin/env python3
"""Run Claude API inference on the SWE-Smith 286-task eval set.

Outputs (in --output-dir):
  outputs.jsonl         – raw outputs (compatible with materialize_real_label_router_dataset)
  predictions.jsonl     – SWE-bench-style predictions (instance_id + unified diff patch)
  run_summary.json      – token counts, cost estimates, per-task stats

Resume: already-written instance_ids in outputs.jsonl are skipped automatically.

Example:
  ANTHROPIC_API_KEY=sk-ant-... python -m pipelinerl.swe.scripts.offline_router.run_claude_api_eval \\
    --eval-parquet-dir /mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/eval \\
    --output-dir /mnt/llmd/results/exps/aristides/reason/claude_swe_smith_eval286_sonnet_<timestamp> \\
    --model claude-sonnet-4-6

Pricing reference (as of 2025):
  claude-haiku-4-5-20251001  $0.80/$4.00  per 1M in/out tokens  (~$0.50 for 286 tasks)
  claude-sonnet-4-6           $3.00/$15.00 per 1M in/out tokens  (~$8-12 for 286 tasks)
  claude-opus-4-7             $15.00/$75.00 per 1M in/out tokens (~$50+ for 286 tasks)
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (input, output) in USD
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}


# ── Prompt parsing ─────────────────────────────────────────────────────────────

def _extract_chat_section(prompt_text: str, role: str) -> str:
    marker = f"<|im_start|>{role}\n"
    start = prompt_text.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = prompt_text.find("<|im_end|>", start)
    return prompt_text[start:end].rstrip() if end != -1 else prompt_text[start:].rstrip()


def _parse_file_contents_from_user_message(user_content: str) -> dict[str, str]:
    """Extract {filename: content} from the embedded file-context section of the user message."""
    # The file context starts after "Below are the code files that may contain bugs:"
    marker = "Below are the code files that may contain bugs:"
    ctx_start = user_content.find(marker)
    if ctx_start == -1:
        return {}
    file_section = user_content[ctx_start + len(marker):]

    file_contents: dict[str, str] = {}
    # Matches: ### filename\n```[optional lang]\n[content]\n```
    pattern = re.compile(r"^### (.+?)\n```(?:\w+)?\n(.*?)\n```", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(file_section):
        filename = match.group(1).strip()
        content = match.group(2)
        file_contents[filename] = content
    return file_contents


# ── Patch generation ───────────────────────────────────────────────────────────

def _extract_search_replace_edits(output_text: str) -> list[dict[str, str]]:
    edits: list[dict[str, str]] = []

    def _extract_from_block(block: str) -> None:
        lines = block.split("\n")
        file_path = None
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("###"):
                file_path = line.strip()[3:].strip()
                start_idx = i + 1
                break
        if not file_path:
            return
        search_start = search_end = replace_start = replace_end = None
        for i, line in enumerate(lines[start_idx:], start=start_idx):
            if "<<<<<<< SEARCH" in line:
                search_start = i + 1
            elif "=======" in line and search_start is not None and search_end is None:
                search_end = i
                replace_start = i + 1
            elif ">>>>>>> REPLACE" in line and replace_start is not None:
                replace_end = i
                break
        if None in (search_start, search_end, replace_start, replace_end):
            return
        edits.append({
            "file_path": file_path,
            "search": "\n".join(lines[search_start:search_end]),
            "replace": "\n".join(lines[replace_start:replace_end]),
        })

    # Try fenced code blocks first.
    code_blocks: list[str] = []
    in_block = False
    current: list[str] = []
    for line in output_text.split("\n"):
        if line.strip().startswith("```"):
            if in_block:
                code_blocks.append("\n".join(current))
                current = []
            in_block = not in_block
        elif in_block:
            current.append(line)
    for block in code_blocks:
        _extract_from_block(block)
    # Fallback: try the whole output if no code blocks matched.
    if not edits and "<<<<<<< SEARCH" in output_text and ">>>>>>> REPLACE" in output_text:
        _extract_from_block(output_text)
    return edits


def _apply_edits(file_contents: dict[str, str], edits: list[dict[str, str]]) -> dict[str, str] | None:
    """Return updated file_contents, or None if any edit fails to apply."""
    updated = dict(file_contents)
    for edit in edits:
        fp = edit["file_path"]
        search = edit["search"]
        replace = edit["replace"]
        if fp not in updated:
            logger.warning("Edit targets unknown file %r", fp)
            return None
        if search not in updated[fp]:
            logger.warning("SEARCH block not found in %r", fp)
            return None
        if search == replace:
            continue
        updated[fp] = updated[fp].replace(search, replace, 1)
    return updated


def _build_unified_patch(old_contents: dict[str, str], new_contents: dict[str, str]) -> str:
    parts: list[str] = []
    for path, new_code in new_contents.items():
        old_code = old_contents.get(path, "")
        if old_code == new_code:
            continue
        diff_lines = list(difflib.unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))
        if diff_lines:
            parts.append("".join(diff_lines))
    return "\n".join(parts)


def _output_text_to_patch(output_text: str, file_contents: dict[str, str]) -> tuple[str, str]:
    """Return (patch_text, failure_type). patch_text is empty on failure."""
    if not file_contents:
        return "", "missing_file_contents"
    edits = _extract_search_replace_edits(output_text)
    if not edits:
        return "", "format_error"
    new_contents = _apply_edits(file_contents, edits)
    if new_contents is None:
        return "", "apply_error"
    patch = _build_unified_patch(file_contents, new_contents)
    if not patch:
        return "", "empty_patch"
    return patch, "none"


# ── API call ───────────────────────────────────────────────────────────────────

async def _call_claude(
    client: anthropic.AsyncAnthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple[str, int, int, float, str | None]:
    """Return (output_text, prompt_tokens, completion_tokens, latency_s, error_str)."""
    async with semaphore:
        t0 = time.time()
        try:
            msg = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            latency = time.time() - t0
            text = msg.content[0].text if msg.content else ""
            return text, msg.usage.input_tokens, msg.usage.output_tokens, latency, None
        except Exception as exc:  # pylint: disable=broad-except
            return "", 0, 0, time.time() - t0, str(exc)


# ── Main ───────────────────────────────────────────────────────────────────────

def _load_eval_rows(parquet_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(parquet_dir.glob("*.parquet")):
        rows.extend(pd.read_parquet(f).to_dict(orient="records"))
    return rows


def _load_done_ids(outputs_path: Path) -> set[str]:
    done: set[str] = set()
    if not outputs_path.exists():
        return done
    with outputs_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                pid = row.get("instance_id") or row.get("problem_id")
                if pid:
                    done.add(str(pid))
            except json.JSONDecodeError:
                pass
    return done


def _estimate_cost(total_input: int, total_output: int, model: str) -> float:
    in_cost, out_cost = MODEL_COSTS.get(model, (3.00, 15.00))
    return (total_input / 1_000_000) * in_cost + (total_output / 1_000_000) * out_cost


async def _run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = output_dir / "outputs.jsonl"
    predictions_path = output_dir / "predictions.jsonl"

    rows = _load_eval_rows(Path(args.eval_parquet_dir))
    done_ids = _load_done_ids(outputs_path)
    pending = [r for r in rows if str(r.get("problem_id", "")) not in done_ids]
    logger.info("Total tasks: %d  |  already done: %d  |  pending: %d",
                len(rows), len(done_ids), len(pending))

    if not pending:
        logger.info("Nothing to do.")
        return

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Provide --api-key or set ANTHROPIC_API_KEY")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    total_input_tokens = 0
    total_output_tokens = 0
    n_done = 0
    n_errors = 0
    n_format_fail = 0

    async def _process_row(row: dict[str, Any]) -> None:
        nonlocal total_input_tokens, total_output_tokens, n_done, n_errors, n_format_fail

        problem_id = str(row.get("problem_id", ""))
        prompt_text = str(row.get("prompt_text", ""))
        system_content = _extract_chat_section(prompt_text, "system")
        user_content = _extract_chat_section(prompt_text, "user")

        output_text, prompt_tokens, completion_tokens, latency, error = await _call_claude(
            client=client,
            model=args.model,
            system=system_content,
            user=user_content,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            semaphore=semaphore,
        )

        total_input_tokens += prompt_tokens
        total_output_tokens += completion_tokens

        if error:
            n_errors += 1
            logger.warning("API error for %s: %s", problem_id, error)
            failure_type = "request_error"
            patch = ""
        else:
            file_contents = _parse_file_contents_from_user_message(user_content)
            patch, failure_type = _output_text_to_patch(output_text, file_contents)
            if failure_type != "none":
                n_format_fail += 1

        output_row = {
            "instance_id": problem_id,
            "problem_id": problem_id,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "latency_s": round(latency, 3),
            "failure_type": failure_type,
            "request_error": error,
        }
        pred_row = {
            "instance_id": problem_id,
            "patch": patch,
            "model_patch": patch,
            "model_name_or_path": args.model,
        }

        with outputs_path.open("a") as fh:
            fh.write(json.dumps(output_row) + "\n")
        with predictions_path.open("a") as fh:
            fh.write(json.dumps(pred_row) + "\n")

        n_done += 1
        if n_done % args.log_every == 0:
            cost = _estimate_cost(total_input_tokens, total_output_tokens, args.model)
            logger.info(
                "[%d/%d] done=%d errors=%d fmt_fail=%d  tokens_in=%d out=%d  est_cost=$%.2f",
                n_done, len(pending), n_done, n_errors, n_format_fail,
                total_input_tokens, total_output_tokens, cost,
            )

    tasks = [_process_row(r) for r in pending]
    await asyncio.gather(*tasks)

    cost = _estimate_cost(total_input_tokens, total_output_tokens, args.model)
    summary = {
        "model": args.model,
        "n_tasks": len(rows),
        "n_completed": n_done,
        "n_errors": n_errors,
        "n_format_fail": n_format_fail,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(cost, 4),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Done. %d tasks, %d errors, %d format failures. Est. cost: $%.2f",
                n_done, n_errors, n_format_fail, cost)
    logger.info("Outputs:     %s", outputs_path)
    logger.info("Predictions: %s", predictions_path)
    logger.info("Summary:     %s", summary_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-parquet-dir", required=True,
                        help="Dir containing eval-*.parquet shards (the 286-task eval set)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write outputs.jsonl, predictions.jsonl, run_summary.json")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model ID (default: claude-sonnet-4-6)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (falls back to ANTHROPIC_API_KEY env var)")
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="Max output tokens per task (default: 16000)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max-concurrent", type=int, default=10,
                        help="Max concurrent API requests (default: 10)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log progress every N completed tasks (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run(_parse_args()))
