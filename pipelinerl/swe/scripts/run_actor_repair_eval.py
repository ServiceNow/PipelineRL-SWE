import asyncio
import json
import logging
import random
from contextlib import nullcontext, asynccontextmanager
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp
import hydra
from hydra.utils import get_method
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer
from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
from tqdm import tqdm

from pipelinerl.swe.rollouts.utils import get_problem_id
from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    build_self_eval_messages,
    chat_completion,
    extract_search_replace_edits,
    parse_self_eval_response,
)
from pipelinerl.swe.utils.repair_utils import FormatError, calculate_precise_reward

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _noop_async_cm():
    yield None


def _encode_chat_for_value(tokenizer, messages: List[Dict[str, str]], response: str):
    """Tokenize chat prompt and prompt+response using chat template if available."""
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        full_messages = list(messages) + [{"role": "assistant", "content": response}]
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            return_tensors="pt",
        )
    except Exception:  # pylint: disable=broad-except
        prompt_text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
        full_text = prompt_text + f"assistant: {response}"
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        full_ids = tokenizer(full_text, return_tensors="pt").input_ids
    return prompt_ids, full_ids


def _compute_value_scores(tokenizer, model, device, messages, response) -> tuple[float | None, float | None]:
    """Return (mean_value, last_value) for the response tokens."""
    if not messages or response is None:
        return None, None
    try:
        prompt_ids, full_ids = _encode_chat_for_value(tokenizer, messages, response)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Value scoring failed to build inputs: %s", exc)
        return None, None

    prompt_len = prompt_ids.shape[1]
    input_ids = full_ids.to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    values = outputs.value.squeeze(0)
    if values.shape[0] <= prompt_len:
        return None, None
    response_values = values[prompt_len:]
    mean_val = response_values.mean().item()
    last_val = response_values[-1].item()
    return mean_val, last_val


def _local_chat_completion(model, tokenizer, device, messages, parameters):
    """Generate locally with the value-head model (policy logits)."""
    start = time.time()
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = torch.ones_like(prompt, device=device)
    gen_kwargs = {
        "max_new_tokens": parameters.get("max_tokens", 2000),
        "temperature": parameters.get("temperature", 1.0),
        "top_p": parameters.get("top_p"),
        "top_k": parameters.get("top_k"),
        "do_sample": True,
        "attention_mask": attention_mask,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    # drop None entries
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    with torch.no_grad():
        output_ids = model.generate(prompt, **gen_kwargs)
    completion_ids = output_ids[0, prompt.shape[1]:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    latency = time.time() - start
    usage = {
        "prompt_tokens": int(prompt.numel()),
        "completion_tokens": int(completion_ids.numel()),
    }
    return text, usage, latency


async def _evaluate_problem(
    cfg: DictConfig,
    eval_cfg: DictConfig,
    problem: Dict[str, Any],
    session: aiohttp.ClientSession | None,
    tokenizer,
    value_model,
    device,
) -> Dict[str, Any]:
    file_contents = problem.get("file_contents") or {}
    if not file_contents:
        raise ValueError("Problem missing file_contents")

    repair_messages, stage_input = build_repair_messages(problem["problem_statement"], file_contents)
    if eval_cfg.get("base_url"):
        if session is None:
            raise ValueError("HTTP generation requested but no session provided")
        repair_text, usage, latency = await chat_completion(
            session,
            eval_cfg.base_url,
            eval_cfg.model_name,
            repair_messages,
            eval_cfg.get("parameters", {}),
            eval_cfg.get("api_key"),
        )
    else:
        repair_text, usage, latency = _local_chat_completion(
            value_model, tokenizer, device, repair_messages, eval_cfg.get("parameters", {})
        )

    edits = extract_search_replace_edits(repair_text)
    try:
        reward, reward_metadata = calculate_precise_reward(file_contents, problem.get("patch", ""), edits)
    except FormatError as exc:
        reward = 0.0
        reward_metadata = {"format_error": True, "error": str(exc)}
    except Exception as exc:  # pylint: disable=broad-except
        reward = 0.0
        reward_metadata = {"error": str(exc)}

    success_threshold = cfg.actor.get("success_threshold", 0.8)
    success = bool(reward and reward > success_threshold)

    self_eval_analysis = ""
    self_eval_score = 1.0
    self_eval_output = ""
    self_eval_usage: Dict[str, int] = {}
    self_eval_latency = 0.0
    self_eval_parsing_error = False

    if eval_cfg.get("run_self_eval", True):
        self_eval_messages = build_self_eval_messages(
            problem["problem_statement"], stage_input, repair_text
        )
        if eval_cfg.get("base_url"):
            if session is None:
                raise ValueError("HTTP generation requested but no session provided")
            self_eval_output, self_eval_usage, self_eval_latency = await chat_completion(
                session,
                eval_cfg.base_url,
                eval_cfg.model_name,
                self_eval_messages,
                eval_cfg.get("parameters", {}),
                eval_cfg.get("api_key"),
            )
        else:
            self_eval_output, self_eval_usage, self_eval_latency = _local_chat_completion(
                value_model, tokenizer, device, self_eval_messages, eval_cfg.get("parameters", {})
            )
        self_eval_analysis, self_eval_score, self_eval_parsing_error = parse_self_eval_response(
            self_eval_output
        )

    mean_value, last_value = _compute_value_scores(tokenizer, value_model, device, repair_messages, repair_text)

    record: Dict[str, Any] = {
        "problem_id": get_problem_id(problem),
        "dataset": problem.get("dataset"),
        "repo": problem.get("repo"),
        "source": eval_cfg.get("source_label", "actor_eval"),
        "problem_statement": problem.get("problem_statement"),
        "stage_input": stage_input,
        "repair_prompt": repair_messages,
        "repair_output": repair_text,
        "repair_reward": reward or 0.0,
        "repair_success": success,
        "repair_metrics": reward_metadata,
        "repair_prompt_tokens": usage.get("prompt_tokens", 0),
        "repair_output_tokens": usage.get("completion_tokens", 0),
        "repair_latency": latency,
        "repair_edits": edits,
        "self_eval_output": self_eval_output,
        "self_eval_analysis": self_eval_analysis,
        "self_eval_score": self_eval_score,
        "self_eval_prompt_tokens": self_eval_usage.get("prompt_tokens", 0),
        "self_eval_output_tokens": self_eval_usage.get("completion_tokens", 0),
        "self_eval_latency": self_eval_latency,
        "self_eval_parsing_error": self_eval_parsing_error,
        "self_eval_prompt": self_eval_messages if eval_cfg.get("run_self_eval", True) else None,
        "value_score_mean": mean_value,
        "value_score_last": last_value,
    }

    return record


async def _evaluate(cfg: DictConfig) -> None:
    dataset_loader = get_method(cfg.dataset_loader)
    loader_params = cfg.get("dataset_loader_params", {}) or {}
    test_params = dict(loader_params)
    if "test_dataset_path" in test_params:
        test_params["dataset_path"] = test_params.pop("test_dataset_path")

    dataset_names: List[str] = cfg.get("test_dataset_names", [])
    dataset: List[Dict[str, Any]] = dataset_loader(dataset_names, **test_params)
    logger.info("Loaded %d evaluation problems", len(dataset))

    eval_cfg = cfg.small_eval
    if not eval_cfg.get("model_name"):
        raise ValueError("small_eval.model_name must be provided")

    subsample = eval_cfg.get("subsample")
    if subsample:
        rng = random.Random(cfg.get("seed", 42))
        size = min(int(subsample), len(dataset))
        dataset = rng.sample(dataset, size)
        logger.info("Subsampled to %d problems for evaluation", len(dataset))

    ids_path = eval_cfg.get("problem_ids_path")
    if ids_path:
        path = Path(ids_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for problem in dataset:
                pid = get_problem_id(problem)
                if pid:
                    handle.write(f"{pid}\n")
        logger.info("Saved %d problem ids to %s", len(dataset), path)

    output_path = Path(eval_cfg.get("output_path", cfg.output_dir + "/actor_eval.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    value_model_path = eval_cfg.get("value_model_path", cfg.get("model_path"))
    if not value_model_path:
        raise ValueError("value_model_path must be provided for value-head scoring")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(value_model_path, use_fast=True)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(value_model_path)
    if device.type == "cuda":
        value_model = value_model.to(device=device, dtype=torch.bfloat16)
    else:
        value_model = value_model.to(device)
    value_model.eval()

    connector = aiohttp.TCPConnector(limit=eval_cfg.get("connector_limit", 32)) if eval_cfg.get("base_url") else None
    timeout = aiohttp.ClientTimeout(total=eval_cfg.get("request_timeout", 600)) if eval_cfg.get("base_url") else None

    progress = tqdm(total=len(dataset), desc="Actor repair eval", unit="problem")
    skipped = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) if eval_cfg.get("base_url") else _noop_async_cm() as session:  # type: ignore
        with output_path.open("w") as sink:
            for problem in dataset:
                try:
                    record = await _evaluate_problem(cfg, eval_cfg, problem, session, tokenizer, value_model, device)
                except Exception as exc:  # pylint: disable=broad-except
                    skipped += 1
                    logger.exception("Failed to evaluate problem %s: %s", get_problem_id(problem), exc)
                    progress.update(1)
                    continue

                sink.write(json.dumps(record) + "\n")
                progress.update(1)

    progress.close()
    logger.info(
        "Completed actor repair eval. Wrote %s (%d problems, skipped %d)",
        output_path,
        len(dataset) - skipped,
        skipped,
    )


async def _evaluate_reuse(cfg: DictConfig) -> None:
    """
    Re-run only the self-evaluation step on an existing set of repair trajectories.
    Expects JSONL input with problem_statement, stage_input, and repair_output fields.
    """
    eval_cfg = cfg.small_eval
    reuse_path = eval_cfg.get("reuse_repair_path")
    if not reuse_path:
        raise ValueError("reuse_repair_path must be provided for reuse mode")
    reuse_path = Path(reuse_path)
    if not reuse_path.exists():
        raise FileNotFoundError(f"reuse_repair_path not found: {reuse_path}")

    with reuse_path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    logger.info("Loaded %d trajectories from %s", len(records), reuse_path)

    subsample = eval_cfg.get("subsample")
    if subsample:
        rng = random.Random(cfg.get("seed", 42))
        size = min(int(subsample), len(records))
        records = rng.sample(records, size)
        logger.info("Subsampled to %d records for self-eval reuse", len(records))

    output_path = Path(eval_cfg.get("output_path", cfg.output_dir + "/actor_eval_reuse.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    value_model_path = eval_cfg.get("value_model_path", cfg.get("model_path"))
    if not value_model_path:
        raise ValueError("value_model_path must be provided for value-head scoring")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(value_model_path, use_fast=True)
    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(value_model_path)
    if device.type == "cuda":
        value_model = value_model.to(device=device, dtype=torch.bfloat16)
    else:
        value_model = value_model.to(device)
    value_model.eval()

    connector = aiohttp.TCPConnector(limit=eval_cfg.get("connector_limit", 32)) if eval_cfg.get("base_url") else None
    timeout = aiohttp.ClientTimeout(total=eval_cfg.get("request_timeout", 600)) if eval_cfg.get("base_url") else None

    progress = tqdm(total=len(records), desc="Actor self-eval reuse", unit="problem")
    skipped = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) if eval_cfg.get("base_url") else _noop_async_cm() as session:  # type: ignore
        with output_path.open("w") as sink:
            for record in records:
                try:
                    problem_statement = record.get("problem_statement")
                    stage_input = record.get("stage_input")
                    repair_text = record.get("repair_output") or record.get("repair_text")
                    if not (problem_statement and stage_input is not None and repair_text):
                        raise ValueError("Record missing problem_statement, stage_input, or repair_output")

                    self_eval_messages = build_self_eval_messages(problem_statement, stage_input, repair_text)
                    if eval_cfg.get("base_url"):
                        if session is None:
                            raise ValueError("HTTP generation requested but no session provided")
                        self_eval_output, self_eval_usage, self_eval_latency = await chat_completion(
                            session,
                            eval_cfg.base_url,
                            eval_cfg.model_name,
                            self_eval_messages,
                            eval_cfg.get("parameters", {}),
                            eval_cfg.get("api_key"),
                        )
                    else:
                        self_eval_output, self_eval_usage, self_eval_latency = _local_chat_completion(
                            value_model, tokenizer, device, self_eval_messages, eval_cfg.get("parameters", {})
                        )
                    self_eval_analysis, self_eval_score, self_eval_parsing_error = parse_self_eval_response(
                        self_eval_output
                    )

                    mean_value, last_value = _compute_value_scores(
                        tokenizer, value_model, device, self_eval_messages, repair_text
                    )

                    record.update(
                        {
                            "source": eval_cfg.get("source_label", "actor_eval_reuse"),
                            "self_eval_output": self_eval_output,
                            "self_eval_analysis": self_eval_analysis,
                            "self_eval_score": self_eval_score,
                            "self_eval_prompt_tokens": self_eval_usage.get("prompt_tokens", 0),
                            "self_eval_output_tokens": self_eval_usage.get("completion_tokens", 0),
                            "self_eval_latency": self_eval_latency,
                            "self_eval_parsing_error": self_eval_parsing_error,
                            "self_eval_prompt": self_eval_messages,
                            "value_score_mean": mean_value,
                            "value_score_last": last_value,
                        }
                    )
                    sink.write(json.dumps(record) + "\n")
                except Exception as exc:  # pylint: disable=broad-except
                    skipped += 1
                    logger.exception("Failed to self-eval record %s: %s", record.get("problem_id"), exc)
                finally:
                    progress.update(1)

    progress.close()
    logger.info(
        "Completed self-eval reuse. Wrote %s (%d records, skipped %d)",
        output_path,
        len(records) - skipped,
        skipped,
    )


@hydra.main(config_path="../../../conf", config_name="swe", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    if cfg.small_eval.get("reuse_repair_path"):
        asyncio.run(_evaluate_reuse(cfg))
    else:
        asyncio.run(_evaluate(cfg))


if __name__ == "__main__":  # pragma: no cover
    main()
