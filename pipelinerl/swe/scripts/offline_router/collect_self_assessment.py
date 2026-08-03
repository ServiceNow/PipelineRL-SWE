#!/usr/bin/env python3
"""
Collect strong-model self-assessment predictions for abstention baseline.

Prompts a strong model (via OpenRouter) with the problem statement and asks
whether it thinks it can fix the bug, recording Y/N and token counts.

This is the "oracle-cost" abstention baseline: uses the same strong model
that will eventually solve the task, but only generates 1 output token
instead of a full patch. Still pays full prompt-token cost.

Output JSONL: {problem_id, p_yes, prompt_tokens, completion_tokens, model}
Compatible with analyze_cot_verifier_abstention.py --cot-embedding-preds-path.

Usage:
  OPENROUTER_API_KEY=... python collect_self_assessment.py \\
    --dataset-path /mnt/llmd/data/swe_smith_bugged_context/ds_train \\
    --instance-ids-file /mnt/.../eval_instance_ids.json \\
    --output-path /mnt/.../self_assessment_eval.jsonl \\
    --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import aiohttp
from datasets import load_from_disk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SELF_ASSESSMENT_SYSTEM = (
    "Output a single character: Y or N. Do not think, reason, or explain. "
    "Y = you could fix the bug. N = you could not."
)

SELF_ASSESSMENT_TEMPLATE = (
    "Bug report:\n"
    "--- BEGIN ISSUE ---\n"
    "{problem_statement}\n"
    "--- END ISSUE ---\n\n"
    "Relevant source files:\n"
    "{file_contents}\n\n"
    "Can you fix this bug? Y or N."
)


def _format_file_contents(fc: dict[str, str]) -> str:
    parts = []
    for path, content in fc.items():
        parts.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(parts)


def _parse_file_contents(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def load_instances(dataset_path: str, instance_ids: set[str]) -> list[dict[str, Any]]:
    ds = load_from_disk(dataset_path)
    items = []
    for row in ds:
        pid = str(row.get("instance_id") or row.get("issue_id") or row.get("id") or "").strip()
        if not pid or pid not in instance_ids:
            continue
        fc = _parse_file_contents(row.get("gold_file_contents", "{}"))
        stmt = str(row.get("problem_statement") or "").strip()
        if not fc or not stmt:
            continue
        items.append({"problem_id": pid, "problem_statement": stmt, "file_contents": fc})
    logger.info("Loaded %d instances from dataset", len(items))
    return items


def _read_api_key(args: argparse.Namespace) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and args.api_key_file:
        key = Path(args.api_key_file).read_text().strip()
    if not key:
        env_file = Path(__file__).parents[4] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY. Set env var, pass --api-key-file, or add to .env")
    return key


def _extract_p_yes_from_logprobs(choice: dict[str, Any]) -> float | None:
    """Extract P(Y) from top_logprobs using log-sum-exp over Y/N variants."""
    import math
    logprobs_obj = choice.get("logprobs") or {}
    content = logprobs_obj.get("content") or []
    if not content:
        return None
    top = content[0].get("top_logprobs") or []
    if not top:
        return None
    yes_tokens = {"Y", "y", "Yes", "yes", "YES"}
    no_tokens  = {"N", "n", "No", "no", "NO"}
    log_yes = [entry["logprob"] for entry in top if entry.get("token", "").strip() in yes_tokens]
    log_no  = [entry["logprob"] for entry in top if entry.get("token", "").strip() in no_tokens]
    if not log_yes and not log_no:
        return None
    # log-sum-exp within each class, then softmax between classes
    def logsumexp(vals: list[float]) -> float:
        m = max(vals)
        return m + math.log(sum(math.exp(v - m) for v in vals))
    lse_yes = logsumexp(log_yes) if log_yes else -1e9
    lse_no  = logsumexp(log_no)  if log_no  else -1e9
    m = max(lse_yes, lse_no)
    p_yes = math.exp(lse_yes - m) / (math.exp(lse_yes - m) + math.exp(lse_no - m))
    return p_yes


async def _assess_one(
    session: aiohttp.ClientSession,
    instance: dict[str, Any],
    model: str,
    api_key: str,
    sem: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with sem:
        pid = instance["problem_id"]
        file_contents_str = _format_file_contents(instance["file_contents"])
        user_content = SELF_ASSESSMENT_TEMPLATE.format(
            problem_statement=instance["problem_statement"],
            file_contents=file_contents_str,
        )
        messages = [
            {"role": "system", "content": SELF_ASSESSMENT_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 500,  # reasoning model needs ~100 tokens to think then output Y/N
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
        }
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "PipelineRL-SWE self-assessment",
        }
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("problem=%s status=%d body=%s", pid, resp.status, body[:200])
                    return None
                data = await resp.json()
        except Exception as exc:
            logger.warning("problem=%s error=%s", pid, exc)
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        choice = choices[0]
        text = ((choice.get("message") or {}).get("content") or "").strip().upper()

        # Extract soft P(Y) from logprobs if available
        p_yes = _extract_p_yes_from_logprobs(choice)
        if p_yes is None:
            # Fallback to hard 0/1 if logprobs not returned
            p_yes = 1.0 if text.startswith("Y") else 0.0

        usage = data.get("usage", {})
        reasoning_tokens = int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        )
        return {
            "problem_id": pid,
            "p_yes": p_yes,
            "answer": text[:4],
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "reasoning_tokens": reasoning_tokens,
            "model": model,
        }


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                pid = rec.get("problem_id")
                if pid:
                    done.add(str(pid))
            except json.JSONDecodeError:
                pass
    return done


async def run(args: argparse.Namespace) -> None:
    api_key = _read_api_key(args)

    instance_ids: set[str] = set()
    if args.instance_ids_file:
        with open(args.instance_ids_file) as f:
            instance_ids = set(json.load(f))
    else:
        raise ValueError("--instance-ids-file is required")

    instances = load_instances(args.dataset_path, instance_ids)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(out_path)
    remaining = [inst for inst in instances if inst["problem_id"] not in done_ids]
    logger.info("done=%d  remaining=%d", len(done_ids), len(remaining))
    if not remaining:
        logger.info("All instances already assessed.")
        return

    sem = asyncio.Semaphore(int(args.concurrency))
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=int(args.concurrency) + 4)

    n_yes = n_no = n_err = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [_assess_one(session, inst, args.model, api_key, sem) for inst in remaining]
        with out_path.open("a") as fh:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is None:
                    n_err += 1
                    continue
                if result["p_yes"] > 0.5:
                    n_yes += 1
                else:
                    n_no += 1
                fh.write(json.dumps(result) + "\n")
                fh.flush()

    total = n_yes + n_no
    logger.info(
        "Done. Y=%d (%.1f%%)  N=%d (%.1f%%)  errors=%d",
        n_yes, 100 * n_yes / total if total else 0,
        n_no, 100 * n_no / total if total else 0,
        n_err,
    )
    logger.info("Output: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strong-model self-assessment abstention baseline")
    parser.add_argument("--dataset-path", default="/mnt/llmd/data/swe_smith_bugged_context/ds_train")
    parser.add_argument("--instance-ids-file", required=True,
                        help="JSON file with list of instance IDs to assess")
    parser.add_argument("--output-path", required=True,
                        help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-4o",
                        help="OpenRouter model to use for self-assessment")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--api-key-file", default="",
                        help="File containing OpenRouter API key (fallback to env/. env)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
