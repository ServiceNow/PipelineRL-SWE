#!/usr/bin/env python
"""Collect proxy SWE repair outputs for a sequence of candidate solver models.

This is intentionally separate from the router dataset collector: here each
candidate is served locally with vLLM, evaluated on the same SWE-Bench subset,
and written as an independent SWE-Bench predictions package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

from pipelinerl.swe.load_datasets import load_local_swe_dataset
from pipelinerl.swe.scripts.offline_router.common import problem_id_from_item, sanitize_for_json
from pipelinerl.swe.scripts.repair_eval_utils import (
    build_repair_messages,
    chat_completion,
    extract_search_replace_edits,
)
from pipelinerl.swe.utils.repair_utils import (
    apply_edits_to_files,
    calculate_precise_reward,
    get_normalized_patch,
)

logger = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "Qwen/Qwen3-Coder-Next",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "model"


def _parse_model_list(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(DEFAULT_MODELS)
    models: list[str] = []
    for raw in raw_values:
        for piece in re.split(r"[\s,]+", raw):
            piece = piece.strip()
            if piece:
                models.append(piece)
    if not models:
        raise ValueError("No models were parsed from --models.")
    return models


def _load_instance_ids(path: str | None) -> list[str] | None:
    if not path:
        return None
    values = [line.strip() for line in Path(path).open() if line.strip()]
    if not values:
        raise ValueError(f"No instance ids found in {path}")
    return values


def _load_dataset_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_local_swe_dataset(
        dataset_names=[args.dataset_name] if args.dataset_name else [],
        dataset_path=args.dataset_path,
        shuffle=False,
        seed=args.seed,
        dataset_label=args.dataset_name,
        max_samples=None,
    )
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = problem_id_from_item(row)
        if row_id and row_id not in rows_by_id:
            rows_by_id[row_id] = row

    instance_ids = _load_instance_ids(args.instance_ids_path)
    if instance_ids is not None:
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for instance_id in instance_ids:
            row = rows_by_id.get(instance_id)
            if row is None:
                missing.append(instance_id)
                continue
            selected.append(row)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"{len(missing)} instance ids from {args.instance_ids_path} were missing in "
                f"{args.dataset_path}; first missing: {preview}"
            )
        rows = selected
    else:
        rows = sorted(rows, key=problem_id_from_item)

    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No dataset rows selected for collection.")
    return rows


def _healthcheck(base_url: str, timeout_s: float = 2.0) -> bool:
    request = urllib.request.Request(base_url.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _tail_file(path: Path, n_lines: int = 60) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def _kwargs_to_vllm_cli(args: argparse.Namespace) -> list[str]:
    cli = [
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
    ]
    if args.max_model_len and args.max_model_len > 0:
        cli.extend(["--max-model-len", str(args.max_model_len)])
    if args.trust_remote_code:
        cli.append("--trust-remote-code")
    for raw in args.vllm_extra_arg:
        raw = raw.strip()
        if not raw:
            continue
        cli.extend(raw.split())
    return cli


def _launch_vllm_server(
    *,
    args: argparse.Namespace,
    model_name: str,
    served_model_name: str,
    model_dir: Path,
) -> tuple[subprocess.Popen, Path, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "vllm_server.log"
    log_handle = log_path.open("a")
    base_url = f"http://127.0.0.1:{args.port}"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
    ]
    cmd.extend(_kwargs_to_vllm_cli(args))
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/transformers_cache")
    env.setdefault("HF_DATASETS_CACHE", "/transformers_cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger.info("Launching vLLM for %s at %s", model_name, base_url)
    logger.info("vLLM log: %s", log_path)
    logger.info("vLLM command: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=log_handle, env=env)
    setattr(proc, "_log_handle", log_handle)
    return proc, log_path, base_url


def _terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
    finally:
        handle = getattr(proc, "_log_handle", None)
        if handle is not None:
            handle.close()


def _wait_for_vllm(
    proc: subprocess.Popen,
    base_url: str,
    log_path: Path,
    model_name: str,
    timeout_s: float,
    poll_s: float,
) -> None:
    start = time.time()
    with tqdm(
        total=int(timeout_s),
        desc=f"wait {model_name}",
        unit="s",
        dynamic_ncols=True,
    ) as progress:
        last_elapsed = 0
        while True:
            elapsed = int(time.time() - start)
            if elapsed > last_elapsed:
                progress.update(elapsed - last_elapsed)
                last_elapsed = elapsed
            if _healthcheck(base_url):
                logger.info("%s is healthy at %s", model_name, base_url)
                return
            exit_code = proc.poll()
            if exit_code is not None:
                tail = _tail_file(log_path)
                raise RuntimeError(
                    f"vLLM for {model_name} exited with code {exit_code} before healthcheck passed.\n"
                    f"Last server log lines:\n{tail}"
                )
            if time.time() - start > timeout_s:
                tail = _tail_file(log_path)
                raise TimeoutError(
                    f"Timed out waiting for vLLM for {model_name} at {base_url}.\n"
                    f"Last server log lines:\n{tail}"
                )
            time.sleep(poll_s)


def _build_git_patch(file_contents: dict[str, str], repair_output: str) -> tuple[str, int, str | None]:
    edits = extract_search_replace_edits(repair_output)
    if not edits:
        return "", 0, "no_edits"
    modified_contents = apply_edits_to_files(file_contents, edits, silent=False)
    patch_dict = get_normalized_patch(file_contents, modified_contents)
    if not patch_dict:
        return "", len(edits), "empty_patch"

    diff_parts: list[str] = []
    for file_path, patch in patch_dict.items():
        diff_parts.append(f"diff --git a/{file_path} b/{file_path}")
        diff_parts.append("index 0000000..1111111 100644")
        diff_parts.append(f"--- a/{file_path}")
        diff_parts.append(f"+++ b/{file_path}")
        diff_parts.append(patch)
    text = "\n".join(diff_parts)
    if text and not text.endswith("\n"):
        text += "\n"
    return text, len(edits), None


def _derive_failure_type(reward_metadata: dict[str, Any], request_error: str | None, patch_status: str | None) -> str:
    if request_error:
        return "request_error"
    if reward_metadata.get("format_error") or patch_status in {"no_edits", "reconstruction_error"}:
        return "format"
    if patch_status == "empty_patch":
        return "empty_patch"
    return "none"


async def _collect_one(
    *,
    session: aiohttp.ClientSession,
    base_url: str,
    served_model_name: str,
    model_name: str,
    problem: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    problem_id = problem_id_from_item(problem)
    file_contents = problem.get("file_contents") or {}
    problem_statement = problem.get("problem_statement")
    if not isinstance(file_contents, dict) or not file_contents:
        raise ValueError(f"{problem_id} missing file_contents")
    if not isinstance(problem_statement, str) or not problem_statement.strip():
        raise ValueError(f"{problem_id} missing problem_statement")

    repair_messages, _stage_input = build_repair_messages(problem_statement, file_contents)
    request_error: str | None = None
    repair_text = ""
    usage: dict[str, Any] = {}
    latency_s = 0.0
    start = time.time()
    try:
        generation_parameters = {
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.max_tokens and args.max_tokens > 0:
            generation_parameters["max_tokens"] = args.max_tokens
        repair_text, usage, latency_s = await chat_completion(
            session=session,
            base_url=base_url,
            model_name=served_model_name,
            messages=repair_messages,
            parameters=generation_parameters,
        )
        if repair_text is None:
            repair_text = ""
    except Exception as exc:  # pylint: disable=broad-except
        latency_s = time.time() - start
        request_error = repr(exc)

    reward_metadata: dict[str, Any]
    if request_error:
        reward = 0.0
        reward_metadata = {"request_error": True, "error": request_error}
    else:
        edits = extract_search_replace_edits(repair_text)
        reward, reward_metadata = calculate_precise_reward(
            file_contents,
            str(problem.get("patch") or ""),
            edits,
        )

    try:
        model_patch, n_edits, patch_status = _build_git_patch(file_contents, repair_text)
    except Exception as exc:  # pylint: disable=broad-except
        model_patch = ""
        n_edits = 0
        patch_status = "reconstruction_error"
        if "patch_reconstruction_error" not in reward_metadata:
            reward_metadata = dict(reward_metadata)
            reward_metadata["patch_reconstruction_error"] = repr(exc)

    return {
        "instance_id": problem_id,
        "dataset": str(problem.get("dataset") or args.dataset_name or ""),
        "repo": str(problem.get("repo") or ""),
        "base_commit": str(problem.get("base_commit") or ""),
        "model_name": model_name,
        "served_model_name": served_model_name,
        "output_text": repair_text,
        "proxy_reward": float(reward or 0.0),
        "proxy_success": bool(reward and reward > args.success_threshold),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "latency_s": float(latency_s),
        "request_error": request_error,
        "failure_type": _derive_failure_type(reward_metadata, request_error, patch_status),
        "reward_metadata": sanitize_for_json(reward_metadata),
        "model_patch": model_patch,
        "n_edits": int(n_edits),
        "patch_status": patch_status,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _read_existing_rows(path: Path) -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL row in %s", path)
                continue
            instance_id = str(row.get("instance_id") or "")
            if instance_id:
                rows[instance_id] = row
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _model_summary(
    *,
    model_name: str,
    model_slug: str,
    rows_by_id: OrderedDict[str, dict[str, Any]],
    n_selected_instances: int,
    status: str,
) -> dict[str, Any]:
    rows = list(rows_by_id.values())
    n = len(rows)
    rewards = [float(row.get("proxy_reward") or 0.0) for row in rows]
    output_tokens = [int(row.get("output_tokens") or 0) for row in rows]
    prompt_tokens = [int(row.get("prompt_tokens") or 0) for row in rows]
    return {
        "status": status,
        "model_name": model_name,
        "model_slug": model_slug,
        "n_selected_instances": n_selected_instances,
        "n_completed": n,
        "n_remaining": max(n_selected_instances - n, 0),
        "n_request_errors": sum(1 for row in rows if row.get("request_error")),
        "n_empty_patches": sum(1 for row in rows if not row.get("model_patch")),
        "n_format_failures": sum(1 for row in rows if row.get("failure_type") == "format"),
        "mean_proxy_reward": (sum(rewards) / n) if n else 0.0,
        "proxy_success_rate": (sum(1 for row in rows if row.get("proxy_success")) / n) if n else 0.0,
        "mean_prompt_tokens": (sum(prompt_tokens) / n) if n else 0.0,
        "mean_output_tokens": (sum(output_tokens) / n) if n else 0.0,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_output_tokens": sum(output_tokens),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_predictions_package(
    *,
    model_dir: Path,
    model_name: str,
    rows_by_id: OrderedDict[str, dict[str, Any]],
    instance_order: list[str],
) -> None:
    ordered_rows = [rows_by_id[instance_id] for instance_id in instance_order if instance_id in rows_by_id]
    predictions_jsonl = [
        {
            "instance_id": row["instance_id"],
            "model_patch": row.get("model_patch") or "",
            "model_name_or_path": model_name,
        }
        for row in ordered_rows
    ]
    predictions_by_instance = {
        row["instance_id"]: {
            "model_patch": row.get("model_patch") or "",
            "model_name_or_path": model_name,
        }
        for row in ordered_rows
    }
    _write_json(model_dir / "predictions.json", predictions_by_instance)
    _write_json(model_dir / "predictions_by_instance.json", predictions_by_instance)
    _write_json(model_dir / "predictions_list.json", predictions_jsonl)
    with (model_dir / "predictions.jsonl").open("w") as handle:
        for row in predictions_jsonl:
            handle.write(json.dumps(row) + "\n")


def _write_root_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    model_summaries: dict[str, dict[str, Any]],
    models: list[str],
    instance_order: list[str],
) -> None:
    manifest = {
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "instance_ids_path": args.instance_ids_path,
        "n_instances": len(instance_order),
        "models": models,
        "model_summaries": model_summaries,
        "collection": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "success_threshold": args.success_threshold,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_model_len_explicitly_disabled": bool(args.max_model_len is not None and args.max_model_len <= 0),
            "max_tokens_explicitly_disabled": bool(args.max_tokens is not None and args.max_tokens <= 0),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(output_dir / "manifest.json", manifest)


async def _collect_model(
    *,
    args: argparse.Namespace,
    model_name: str,
    model_slug: str,
    model_dir: Path,
    rows: list[dict[str, Any]],
    base_url: str,
    served_model_name: str,
) -> OrderedDict[str, dict[str, Any]]:
    outputs_path = model_dir / "outputs.jsonl"
    existing_rows = _read_existing_rows(outputs_path)
    done_ids = {
        instance_id
        for instance_id, row in existing_rows.items()
        if not args.retry_request_errors or not row.get("request_error")
    }
    rows_to_run = [row for row in rows if problem_id_from_item(row) not in done_ids]
    logger.info(
        "%s: %d/%d rows already present; %d rows to collect",
        model_name,
        len(done_ids),
        len(rows),
        len(rows_to_run),
    )

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.connector_limit)
    lock = asyncio.Lock()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for row in rows_to_run:
        queue.put_nowait(row)

    output_handle = outputs_path.open("a", buffering=1)
    pbar = tqdm(total=len(rows_to_run), desc=model_slug, unit="ex", dynamic_ncols=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker(worker_idx: int) -> None:
            while True:
                try:
                    problem = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                instance_id = problem_id_from_item(problem)
                try:
                    result = await _collect_one(
                        session=session,
                        base_url=base_url,
                        served_model_name=served_model_name,
                        model_name=model_name,
                        problem=problem,
                        args=args,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    result = {
                        "instance_id": instance_id,
                        "dataset": str(problem.get("dataset") or args.dataset_name or ""),
                        "repo": str(problem.get("repo") or ""),
                        "base_commit": str(problem.get("base_commit") or ""),
                        "model_name": model_name,
                        "served_model_name": served_model_name,
                        "output_text": "",
                        "proxy_reward": 0.0,
                        "proxy_success": False,
                        "prompt_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency_s": 0.0,
                        "request_error": repr(exc),
                        "failure_type": "request_error",
                        "reward_metadata": {"request_error": True, "error": repr(exc), "worker_idx": worker_idx},
                        "model_patch": "",
                        "n_edits": 0,
                        "patch_status": "request_error",
                        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                async with lock:
                    existing_rows[instance_id] = result
                    output_handle.write(json.dumps(sanitize_for_json(result)) + "\n")
                    pbar.update(1)
                queue.task_done()

        n_workers = max(1, min(args.max_concurrent_problems, len(rows_to_run) or 1))
        await asyncio.gather(*(worker(idx) for idx in range(n_workers)))

    pbar.close()
    output_handle.close()
    return existing_rows


def _run_one_model(
    *,
    args: argparse.Namespace,
    model_name: str,
    rows: list[dict[str, Any]],
    instance_order: list[str],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    model_slug = _slugify(model_name)
    model_dir = output_dir / "models" / model_slug
    summary_path = model_dir / "summary.json"

    if args.skip_existing_complete and summary_path.exists() and not args.force:
        try:
            summary = json.loads(summary_path.read_text())
            if summary.get("status") == "complete":
                logger.info("Skipping %s because %s is already complete", model_name, summary_path)
                return summary
        except json.JSONDecodeError:
            pass

    served_model_name = args.served_model_prefix + model_slug
    proc: subprocess.Popen | None = None
    try:
        proc, log_path, base_url = _launch_vllm_server(
            args=args,
            model_name=model_name,
            served_model_name=served_model_name,
            model_dir=model_dir,
        )
        _wait_for_vllm(
            proc,
            base_url=base_url,
            log_path=log_path,
            model_name=model_name,
            timeout_s=args.healthcheck_timeout,
            poll_s=args.healthcheck_poll,
        )
        rows_by_id = asyncio.run(
            _collect_model(
                args=args,
                model_name=model_name,
                model_slug=model_slug,
                model_dir=model_dir,
                rows=rows,
                base_url=base_url,
                served_model_name=served_model_name,
            )
        )
        _write_predictions_package(
            model_dir=model_dir,
            model_name=model_name,
            rows_by_id=rows_by_id,
            instance_order=instance_order,
        )
        status = "complete" if len(rows_by_id) >= len(rows) else "incomplete"
        summary = _model_summary(
            model_name=model_name,
            model_slug=model_slug,
            rows_by_id=rows_by_id,
            n_selected_instances=len(rows),
            status=status,
        )
        _write_json(summary_path, summary)
        logger.info(
            "%s summary: n=%d mean_proxy=%.4f success=%.4f mean_out_tok=%.1f",
            model_name,
            summary["n_completed"],
            summary["mean_proxy_reward"],
            summary["proxy_success_rate"],
            summary["mean_output_tokens"],
        )
        return summary
    finally:
        _terminate_process(proc)
        if args.sleep_after_model > 0:
            time.sleep(args.sleep_after_model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially serve candidate SWE solver models, collect proxy rewards, and emit SWE-Bench predictions."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", default="/mnt/llmd/data/swebench_verified/all_16k/ds")
    parser.add_argument("--dataset-name", default="swe_bench_verified")
    parser.add_argument("--instance-ids-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Whitespace/comma separated model ids. Defaults to the discovery set.")
    parser.add_argument("--served-model-prefix", default="candidate_")
    parser.add_argument("--port", type=int, default=8390)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-extra-arg", action="append", default=[])
    parser.add_argument("--healthcheck-timeout", type=float, default=5400)
    parser.add_argument("--healthcheck-poll", type=float, default=10)
    parser.add_argument("--max-concurrent-problems", type=int, default=8)
    parser.add_argument("--connector-limit", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-after-model", type=float, default=30)
    parser.add_argument("--retry-request-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-complete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-model-failure", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    models = _parse_model_list(args.models)
    rows = _load_dataset_rows(args)
    instance_order = [problem_id_from_item(row) for row in rows]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "instance_ids.txt").write_text("\n".join(instance_order) + "\n")

    logger.info("Selected %d instances from %s", len(rows), args.dataset_path)
    logger.info("Model collection order: %s", ", ".join(models))

    if args.dry_run:
        logger.info("Dry run only; not launching vLLM.")
        _write_root_manifest(
            output_dir=output_dir,
            args=args,
            model_summaries={},
            models=models,
            instance_order=instance_order,
        )
        return

    model_summaries: dict[str, dict[str, Any]] = {}
    for model_name in models:
        try:
            summary = _run_one_model(
                args=args,
                model_name=model_name,
                rows=rows,
                instance_order=instance_order,
            )
        except Exception as exc:  # pylint: disable=broad-except
            model_slug = _slugify(model_name)
            summary = {
                "status": "failed",
                "model_name": model_name,
                "model_slug": model_slug,
                "n_selected_instances": len(rows),
                "n_completed": 0,
                "n_remaining": len(rows),
                "error": repr(exc),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            _write_json(output_dir / "models" / model_slug / "summary.json", summary)
            logger.exception("Collection failed for %s", model_name)
            if args.stop_on_model_failure:
                raise
        model_summaries[_slugify(model_name)] = summary
        _write_root_manifest(
            output_dir=output_dir,
            args=args,
            model_summaries=model_summaries,
            models=models,
            instance_order=instance_order,
        )

    logger.info("Completed model discovery collection into %s", output_dir)


if __name__ == "__main__":
    main()
