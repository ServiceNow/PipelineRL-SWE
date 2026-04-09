import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import hydra
from hydra.utils import get_method
from omegaconf import DictConfig
from tqdm import tqdm

from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    chat_completion,
    extract_search_replace_edits,
)
from pipelinerl.swe.utils.repair_utils import FormatError, calculate_precise_reward

logger = logging.getLogger(__name__)


def get_problem_id(problem: Dict[str, Any]) -> str:
    for key in ("problem_id", "issue_id", "instance_id", "id"):
        value = problem.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError("Problem is missing an identifier (problem_id/issue_id/instance_id/id)")


async def _evaluate_problem(
    cfg: DictConfig,
    eval_cfg: DictConfig,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession,
) -> Dict[str, Any]:
    file_contents = problem.get("file_contents") or {}
    if not file_contents:
        raise ValueError("Problem missing file_contents")

    repair_messages, _ = build_repair_messages(problem["problem_statement"], file_contents)
    repair_text, usage, latency = await chat_completion(
        session,
        eval_cfg.base_url,
        eval_cfg.model_name,
        repair_messages,
        eval_cfg.get("parameters", {}),
        eval_cfg.get("api_key"),
    )
    if repair_text is None:
        raise ValueError("Model returned empty content")

    edits = extract_search_replace_edits(repair_text)
    try:
        reward, reward_metadata = calculate_precise_reward(file_contents, problem.get("patch", ""), edits)
    except FormatError as exc:
        reward = 0.0
        reward_metadata = {"format_error": True, "error": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except
        reward = 0.0
        reward_metadata = {"error": str(exc)}

    success_threshold = eval_cfg.get("success_threshold")
    if success_threshold is None:
        success_threshold = cfg.actor.get("success_threshold")
    if success_threshold is None:
        success_threshold = 0.8
    success = bool(reward is not None and reward > success_threshold)

    return {
        "problem_id": get_problem_id(problem),
        "dataset": problem.get("dataset"),
        "repo": problem.get("repo"),
        "source": eval_cfg.get("source_label", "expert_eval"),
        "repair_output": repair_text,
        "repair_reward": reward or 0.0,
        "repair_success": success,
        "repair_metrics": reward_metadata,
        "repair_prompt_tokens": usage.get("prompt_tokens", 0),
        "repair_output_tokens": usage.get("completion_tokens", 0),
        "repair_latency": latency,
        "repair_edits": edits,
    }


async def _evaluate(cfg: DictConfig) -> None:
    dataset_loader = get_method(cfg.dataset_loader)
    dataset_loader_params = cfg.get("dataset_loader_params", {}) or {}

    test_params = dict(dataset_loader_params)
    if "test_dataset_path" in test_params:
        test_params["dataset_path"] = test_params.pop("test_dataset_path")
    if "test_max_samples" in test_params:
        test_params["max_samples"] = test_params.pop("test_max_samples")

    dataset_names: List[str] = cfg.get("test_dataset_names", [])
    dataset: List[Dict[str, Any]] = dataset_loader(dataset_names, **test_params)
    logger.info("Loaded %d evaluation problems", len(dataset))

    expert_cfg = cfg.expert_eval

    ids_path = expert_cfg.get("problem_ids_path")
    if ids_path:
        include_file = Path(ids_path)
        if not include_file.exists():
            raise FileNotFoundError(f"problem_ids_path not found: {include_file}")
        include_ids = {line.strip() for line in include_file.open() if line.strip()}
        before = len(dataset)
        dataset = [p for p in dataset if get_problem_id(p) in include_ids]
        logger.info(
            "Filtered dataset to %d problems based on ids in %s (was %d)",
            len(dataset),
            ids_path,
            before,
        )

    subsample = expert_cfg.get("subsample")
    if subsample:
        rng = random.Random(cfg.get("seed", 42))
        size = min(int(subsample), len(dataset))
        dataset = rng.sample(dataset, size)
        logger.info(
            "Randomly selected %d problems for this expert run (subsample=%s)",
            len(dataset),
            subsample,
        )

    if not expert_cfg.get("base_url"):
        raise ValueError("expert_eval.base_url must be set (point to the expert vLLM server)")
    if not expert_cfg.get("model_name"):
        raise ValueError("expert_eval.model_name must be provided")

    output_path = Path(expert_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit = expert_cfg.get("limit")
    connector = aiohttp.TCPConnector(limit=expert_cfg.get("connector_limit", 128))
    timeout = aiohttp.ClientTimeout(total=expert_cfg.get("request_timeout", 600))

    processed = 0
    skipped = 0
    total = min(len(dataset), limit) if limit else len(dataset)
    progress = tqdm(total=total, desc="Expert repair", unit="problem")
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with output_path.open("w") as sink:
            for problem in dataset:
                if limit is not None and processed >= limit:
                    break
                try:
                    record = await _evaluate_problem(cfg, expert_cfg, problem, session)
                    sink.write(json.dumps(record) + "\n")
                    processed += 1
                    progress.update(1)
                    if processed % 10 == 0:
                        logger.info("Processed %d problems", processed)
                except Exception as exc:  # pylint: disable=broad-except
                    skipped += 1
                    logger.exception("Failed to evaluate problem %s: %s", get_problem_id(problem), exc)
    progress.close()

    logger.info("Expert evaluation complete. Wrote %d records to %s (skipped %d).", processed, output_path, skipped)


@hydra.main(config_path="../../../conf", config_name="swe", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entrypoint
    """Entry point for running the expert repair evaluation."""
    asyncio.run(_evaluate(cfg))


if __name__ == "__main__":  # pragma: no cover
    main()
