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
from tapeagents.llms.trainable import TrainableLLM
from tqdm import tqdm

from pipelinerl.swe.rollouts.stages import run_repair
from pipelinerl.swe.rollouts.utils import annotate_training_text, get_problem_id
from pipelinerl.tokenizers import configure_devstral_tokenizer, is_devstral_model_name

logger = logging.getLogger(__name__)


async def _run_single_problem(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession,
) -> Dict[str, Any]:
    """Run the repair stage on a single problem using the expert model."""
    file_contents = problem.get("file_contents") or {}
    if not file_contents:
        raise ValueError("Problem is missing file contents")

    result = await run_repair(cfg, llm, problem, file_contents, session)

    record: Dict[str, Any] = {
        "problem_id": get_problem_id(problem),
        "dataset": problem.get("dataset"),
        "repo": problem.get("repo"),
        "success": result.get("success", False),
        "metrics": result.get("metrics", {}),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
        "latency": result.get("latency", 0.0),
        "training_texts": [],
    }

    training_text = result.get("training_text")
    if training_text is not None:
        # Override metadata source so downstream analysis can distinguish these samples.
        annotate_training_text(
            training_text,
            stage="repair",
            problem=problem,
            llm=llm,
            source="expert",
            extra={
                "metrics": result.get("metrics", {}),
                "success": result.get("success", False),
            },
        )
        training_text.metadata["source"] = "expert"
        training_text.metadata["model_name"] = llm.model_name
        record["training_texts"].append(training_text.model_dump())

    return record


async def _evaluate(cfg: DictConfig) -> None:
    dataset_loader = get_method(cfg.dataset_loader)
    dataset_loader_params = cfg.get("dataset_loader_params", {}) or {}

    test_params = dict(dataset_loader_params)
    if "test_dataset_path" in test_params:
        test_params["dataset_path"] = test_params.pop("test_dataset_path")

    dataset_names: List[str] = cfg.get("test_dataset_names", [])
    dataset: List[Dict[str, Any]] = dataset_loader(dataset_names, **test_params)
    logger.info("Loaded %d evaluation problems", len(dataset))

    subsample = expert_cfg.get("subsample")
    if subsample:
        rng = random.Random(cfg.get("seed", 42))
        size = min(int(subsample), len(dataset))
        dataset = rng.sample(dataset, size)
        logger.info("Randomly selected %d problems for this expert run (subsample=%s)", len(dataset), subsample)

    expert_cfg = cfg.expert_eval
    if not expert_cfg.get("base_url"):
        raise ValueError("expert_eval.base_url must be set (point to the expert vLLM server)")
    if not expert_cfg.get("model_name"):
        raise ValueError("expert_eval.model_name must be provided")

    llm = TrainableLLM(
        base_url=expert_cfg.base_url,
        model_name=expert_cfg.model_name,
        tokenizer_name=expert_cfg.get("tokenizer_name", expert_cfg.model_name),
        parameters=expert_cfg.get("parameters", {}),
        use_cache=False,
        collect_logprobs=expert_cfg.get("collect_logprobs", False),
        observe_llm_calls=False,
        api_token=expert_cfg.get("api_token") or "",
    )
    if is_devstral_model_name(llm.model_name):
        configure_devstral_tokenizer(llm)
        existing_params = llm.parameters or {}
        llm.parameters = {**existing_params, "skip_special_tokens": True}

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
                    record = await _run_single_problem(cfg, llm, problem, session)
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
