#!/usr/bin/env python
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from pipelinerl.swe.scripts.offline_router.collect_router_dataset import run_collection

logger = logging.getLogger(__name__)


@dataclass
class ServerSpec:
    name: str
    model_path: str
    served_model_name: str
    port: int
    conda_env: str | None
    gpus: list[int]
    kwargs: dict[str, Any]


def _visible_gpu_ids() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
        return list(range(len(entries)))
    return list(range(torch.cuda.device_count()))


def _kwargs_to_cli(kwargs: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in kwargs.items():
        args.append(f"--{key}")
        if value not in (None, ""):
            args.append(str(value))
    return args


def _healthcheck(base_url: str, timeout_s: float = 2.0) -> bool:
    request = urllib.request.Request(base_url.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _wait_for_health(base_url: str, name: str, timeout_s: float, poll_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _healthcheck(base_url):
            logger.info("%s is healthy at %s", name, base_url)
            return
        time.sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for {name} at {base_url}")


def _launch_server(spec: ServerSpec, log_dir: Path) -> subprocess.Popen:
    cmd: list[str]
    if spec.conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", spec.conda_env, "python"]
    else:
        cmd = ["python"]
    cmd += [
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        spec.model_path,
        "--served-model-name",
        spec.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        "--seed",
        "42",
    ]
    cmd.extend(_kwargs_to_cli(spec.kwargs))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in spec.gpus)
    safe_name = spec.name.replace("/", "_").replace(":", "_").replace(" ", "_")
    log_path = log_dir / f"{safe_name}.log"
    handle = log_path.open("a")
    logger.info("Launching %s on GPUs %s: %s", spec.name, spec.gpus, " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=handle)
    setattr(proc, "_offline_router_log_handle", handle)
    return proc


def _terminate_process(proc: subprocess.Popen) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    finally:
        handle = getattr(proc, "_offline_router_log_handle", None)
        if handle is not None:
            handle.close()


def _reserve_gpus(available: list[int], needed: int) -> list[int]:
    if len(available) < needed:
        raise ValueError(f"Need {needed} GPUs but only {len(available)} visible GPUs remain")
    assigned = available[:needed]
    del available[:needed]
    return assigned


def _build_primary_spec(cfg: DictConfig, available: list[int]) -> tuple[ServerSpec, str]:
    primary_cfg = cfg.offline_router.primary_model
    kwargs = dict(OmegaConf.to_container(primary_cfg.get("vllm_kwargs", {}), resolve=True) or {})
    tp = int(kwargs.get("tensor-parallel-size", 1))
    pp = int(kwargs.get("pipeline-parallel-size", 1))
    gpus = _reserve_gpus(available, max(tp * pp, 1))
    port = int(primary_cfg.port)
    spec = ServerSpec(
        name=str(primary_cfg.get("label") or "primary_model"),
        model_path=str(primary_cfg.model_path),
        served_model_name=str(primary_cfg.get("served_model_name") or primary_cfg.get("model_name") or primary_cfg.model_path),
        port=port,
        conda_env=primary_cfg.get("conda_env"),
        gpus=gpus,
        kwargs=kwargs,
    )
    return spec, f"http://127.0.0.1:{port}"


def _build_expert_specs(cfg: DictConfig, available: list[int]) -> tuple[list[ServerSpec], list[str]]:
    expert_cfgs = sorted(list(cfg.offline_router.get("experts", [])), key=lambda item: int(item.get("expert_rank", 0)))

    specs: list[ServerSpec] = []
    base_urls: list[str] = []
    for idx, expert_cfg in enumerate(expert_cfgs):
        kwargs = dict(OmegaConf.to_container(expert_cfg.get("vllm_kwargs", {}), resolve=True) or {})
        tp = int(kwargs.get("tensor-parallel-size", 1))
        pp = int(kwargs.get("pipeline-parallel-size", 1))
        gpus = _reserve_gpus(available, max(tp * pp, 1))
        port = int(expert_cfg.get("port", 8280 + idx))
        specs.append(
            ServerSpec(
                name=str(expert_cfg.get("label") or f"expert_{idx}"),
                model_path=str(expert_cfg.model_path),
                served_model_name=str(
                    expert_cfg.get("served_model_name") or expert_cfg.get("model_name") or expert_cfg.model_path
                ),
                port=port,
                conda_env=expert_cfg.get("conda_env"),
                gpus=gpus,
                kwargs=kwargs,
            )
        )
        base_urls.append(f"http://127.0.0.1:{port}")
    return specs, base_urls


@hydra.main(config_path="../../../../conf", config_name="offline_router_collect", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(cfg.output_dir)
    log_dir = output_dir / "server_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    available_gpus = _visible_gpu_ids()
    if not available_gpus:
        raise ValueError("No visible GPUs available for offline router collection job")

    primary_spec, primary_base_url = _build_primary_spec(cfg, available_gpus)
    expert_specs, expert_base_urls = _build_expert_specs(cfg, available_gpus)

    procs: list[subprocess.Popen] = []
    try:
        procs.append(_launch_server(primary_spec, log_dir))
        for spec in expert_specs:
            procs.append(_launch_server(spec, log_dir))

        timeout_s = float(cfg.offline_router.launch.get("healthcheck_timeout_s", 1800))
        poll_s = float(cfg.offline_router.launch.get("healthcheck_poll_s", 10))
        _wait_for_health(primary_base_url, str(cfg.offline_router.primary_model.get("label") or "primary_model"), timeout_s, poll_s)
        for idx, base_url in enumerate(expert_base_urls):
            label = str(cfg.offline_router.experts[idx].get("label") or f"expert_{idx}")
            _wait_for_health(base_url, label, timeout_s, poll_s)

        cfg.offline_router.primary_model.base_url = primary_base_url
        cfg.offline_router.primary_model.model_name = str(cfg.offline_router.primary_model.served_model_name)
        cfg.offline_router.primary_model.tokenizer_name = str(
            cfg.offline_router.primary_model.get("tokenizer_name") or cfg.offline_router.primary_model.model_path
        )
        cfg.offline_router.expert_base_urls = expert_base_urls
        run_collection(cfg)
    finally:
        for proc in reversed(procs):
            _terminate_process(proc)


if __name__ == "__main__":
    main()
