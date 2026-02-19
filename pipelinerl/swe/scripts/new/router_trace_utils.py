import glob
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def discover_jsonl_files(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_model_version(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def iter_jsonl_dicts(paths: Iterable[Path]):
    for path in paths:
        with path.open() as handle:
            for lineno, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON in %s:%d", path, lineno)
                    continue
                if isinstance(item, dict):
                    yield item
                else:
                    logger.warning("Skipping non-dict JSON entry in %s:%d", path, lineno)


def _problem_key(trace: dict[str, Any]) -> str:
    dataset = trace.get("dataset") or ""
    problem_id = trace.get("problem_id") or trace.get("instance_id") or ""
    return f"{dataset}::{problem_id}"


def _trace_sort_key(trace: dict[str, Any]) -> tuple[int, float, str]:
    version = _as_model_version(trace.get("model_version"))
    ts = _as_float(trace.get("generated_at_unix"), 0.0)
    group_id = str(trace.get("group_id") or "")
    return (version, ts, group_id)


def select_latest_model_version(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    versions = [_as_model_version(t.get("model_version")) for t in traces if t.get("model_version") is not None]
    if not versions:
        return traces
    latest_version = max(versions)
    return [t for t in traces if _as_model_version(t.get("model_version")) == latest_version]


def dedupe_latest_by_problem(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for trace in traces:
        key = _problem_key(trace)
        previous = latest.get(key)
        if previous is None or _trace_sort_key(trace) >= _trace_sort_key(previous):
            latest[key] = trace
    return list(latest.values())


def load_router_traces(
    input_globs: list[str],
    split: str | None,
    latest_model_only: bool,
    dedupe_by_problem: bool,
) -> list[dict[str, Any]]:
    files = discover_jsonl_files(input_globs)
    if not files:
        raise FileNotFoundError(f"No JSONL files found for patterns: {input_globs}")

    traces: list[dict[str, Any]] = []
    for trace in iter_jsonl_dicts(files):
        if split is not None and trace.get("split") != split:
            continue
        traces.append(trace)

    if latest_model_only:
        traces = select_latest_model_version(traces)
    if dedupe_by_problem:
        traces = dedupe_latest_by_problem(traces)

    return traces


def extract_reward_vector(trace: dict[str, Any]) -> list[float]:
    targets = trace.get("performance_targets")
    if isinstance(targets, list) and targets:
        values = [_as_float(v, 0.0) for v in targets]
        if values:
            return values

    policy = trace.get("policy") or {}
    experts = trace.get("experts") or []
    values = [_as_float(policy.get("reward"), 0.0)]
    for expert in experts:
        values.append(_as_float((expert or {}).get("reward"), 0.0))
    return values


def extract_token_vector(trace: dict[str, Any]) -> list[int]:
    policy = trace.get("policy") or {}
    experts = trace.get("experts") or []

    policy_tokens = _as_int(policy.get("prompt_tokens"), 0) + _as_int(policy.get("output_tokens"), 0)
    values = [policy_tokens]
    for expert in experts:
        expert = expert or {}
        values.append(_as_int(expert.get("prompt_tokens"), 0) + _as_int(expert.get("output_tokens"), 0))
    return values


def extract_route_labels(traces: list[dict[str, Any]], n_experts: int) -> list[str]:
    labels = ["policy"]
    for expert_idx in range(n_experts):
        label = None
        for trace in traces:
            experts = trace.get("experts") or []
            if expert_idx < len(experts):
                model_name = (experts[expert_idx] or {}).get("model_name")
                if model_name:
                    label = str(model_name)
                    break
        if label:
            labels.append(f"expert_{expert_idx}:{label}")
        else:
            labels.append(f"expert_{expert_idx}")
    return labels


def extract_score_vector(trace: dict[str, Any], score_key: str, expected_dim: int) -> list[float] | None:
    value = trace.get(score_key)
    if value is None:
        value = (trace.get("policy") or {}).get(score_key)
    if not isinstance(value, list):
        return None

    scores = [_as_float(v, 0.0) for v in value]
    if len(scores) < expected_dim:
        return None
    if len(scores) > expected_dim:
        scores = scores[:expected_dim]
    return scores
