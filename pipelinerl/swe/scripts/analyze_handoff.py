import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import glob
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def _iter_training_texts(actor_files: Iterable[Path]):
    for path in actor_files:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items = json.loads(line)
                    if isinstance(items, list):
                        for obj in items:
                            yield obj
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", path)


def _collect_latest_actor_records(actor_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    max_version = None
    for entry in _iter_training_texts(actor_files):
        version = entry.get("metadata", {}).get("model_version")
        if version is not None:
            max_version = version if max_version is None else max(max_version, version)

    if max_version is None:
        raise ValueError("Unable to find any model_version in actor traces")

    records: Dict[str, Dict[str, Any]] = {}
    for entry in _iter_training_texts(actor_files):
        meta = entry.get("metadata", {})
        if meta.get("model_version") != max_version:
            continue
        stage = meta.get("stage")
        if stage not in {"repair", "repair_self_eval"}:
            continue
        problem_id = meta.get("problem_id")
        if not problem_id:
            continue
        records.setdefault(problem_id, {"dataset": meta.get("dataset"), "model_version": max_version})
        records[problem_id][stage] = entry

    return records


def _load_expert_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Expert JSONL not found: {path}")

    expert: Dict[str, Dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            problem_id = record.get("problem_id")
            if not problem_id:
                continue
            expert[problem_id] = record
    return expert


def _frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    # Avoid floating precision drift by iterating in integers
    count = int(round((stop - start) / step)) + 1
    for i in range(count):
        values.append(round(start + i * step, 6))
    return values


def _to_tokens(entry: Dict[str, Any]) -> int:
    return int(entry.get("prompt_tokens", 0) + entry.get("output_tokens", 0))


def _problem_success(entry: Dict[str, Any]) -> bool:
    meta = entry.get("metadata", {})
    if "success" in meta:
        return bool(meta["success"])
    metrics = meta.get("metrics") or {}
    if "success" in metrics:
        return bool(metrics["success"])
    return bool(entry.get("reward", 0) > 0)


def _predicted_score(entry: Dict[str, Any]) -> float:
    meta = entry.get("metadata", {})
    metrics = meta.get("metrics") or {}
    if "predicted_score" in meta:
        return float(meta["predicted_score"])
    if "predicted_score" in metrics:
        return float(metrics["predicted_score"])
    return 1.0


def _merge_records(actor_records: Dict[str, Dict[str, Any]], expert_records: Dict[str, Dict[str, Any]]):
    merged = {}
    for problem_id, data in actor_records.items():
        if "repair" not in data:
            continue
        if problem_id not in expert_records:
            continue
        merged[problem_id] = {
            "repair": data["repair"],
            "repair_self_eval": data.get("repair_self_eval"),
            "expert": expert_records[problem_id],
        }
    return merged


def compute_handoff_curve(records: Dict[str, Dict[str, Any]], thresholds: List[float]) -> List[Dict[str, Any]]:
    total_problems = len(records)
    if total_problems == 0:
        raise ValueError("No overlapping problems between actor and expert results")

    base_small_tokens = 0
    per_problem_small_tokens: Dict[str, int] = {}
    for pid, data in records.items():
        repair_tokens = _to_tokens(data["repair"])
        self_eval_tokens = 0
        if data.get("repair_self_eval"):
            self_eval_tokens = _to_tokens(data["repair_self_eval"])
        tokens = repair_tokens + self_eval_tokens
        per_problem_small_tokens[pid] = tokens
        base_small_tokens += tokens

    results = []
    for threshold in thresholds:
        total_tokens = 0
        successes = 0
        handoffs = 0

        for pid, data in records.items():
            small_success = _problem_success(data["repair"])
            predicted = _predicted_score(data.get("repair_self_eval") or {})
            small_tokens = per_problem_small_tokens[pid]

            use_expert = predicted < threshold
            if use_expert:
                expert_entry = data["expert"]
                total_tokens += small_tokens + expert_entry.get("prompt_tokens", 0) + expert_entry.get("output_tokens", 0)
                successes += 1 if expert_entry.get("success") else 0
                handoffs += 1
            else:
                total_tokens += small_tokens
                successes += 1 if small_success else 0

        results.append(
            {
                "threshold": threshold,
                "success_rate": successes / total_problems,
                "avg_tokens": total_tokens / total_problems,
                "handoff_fraction": handoffs / total_problems,
                "handed_off": handoffs,
                "total_problems": total_problems,
            }
        )
    return results


@hydra.main(config_path="../../conf", config_name="swe", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    """Analyze the cost/accuracy Pareto frontier for the repair-stage handoff."""
    analysis_cfg = cfg.handoff_analysis
    actor_pattern = analysis_cfg.get("actor_glob")
    if not actor_pattern:
        raise ValueError("handoff_analysis.actor_glob must be provided (glob to actor JSONL shards)")
    actor_files = [Path(p) for p in glob.glob(actor_pattern, recursive=True) if Path(p).is_file()]
    if not actor_files:
        raise ValueError(f"No actor JSONL files found for pattern: {actor_pattern}")
    actor_files.sort()
    logger.info("Reading actor traces from %d files", len(actor_files))

    actor_records = _collect_latest_actor_records(actor_files)
    if not actor_records:
        raise ValueError("No repair entries found in actor traces for the latest model version")
    sample_version = next(iter(actor_records.values())).get("model_version")
    logger.info("Found %d problems in latest model_version=%s", len(actor_records), sample_version)

    expert_path = Path(analysis_cfg.get("expert_jsonl", cfg.expert_eval.output_path))
    expert_records = _load_expert_records(expert_path)
    logger.info("Loaded %d expert records from %s", len(expert_records), expert_path)

    merged = _merge_records(actor_records, expert_records)
    logger.info("Overlap contains %d problems", len(merged))

    thresholds = analysis_cfg.get("thresholds")
    if thresholds:
        threshold_values = [float(t) for t in thresholds]
    else:
        threshold_values = _frange(
            float(analysis_cfg.get("threshold_start", 0.0)),
            float(analysis_cfg.get("threshold_stop", 1.0)),
            float(analysis_cfg.get("threshold_step", 0.05)),
        )

    curve = compute_handoff_curve(merged, threshold_values)

    output_path = Path(analysis_cfg.get("output_path", cfg.output_dir + "/handoff_analysis.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(curve, handle, indent=2)
    logger.info("Wrote analysis with %d thresholds to %s", len(curve), output_path)

    if curve:
        best = max(curve, key=lambda x: x["success_rate"])
        logger.info("Best success %.2f%% at threshold %.2f with avg tokens %.1f", best["success_rate"] * 100, best["threshold"], best["avg_tokens"])


if __name__ == "__main__":  # pragma: no cover
    main()
