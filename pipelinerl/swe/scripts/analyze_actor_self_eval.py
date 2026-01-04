import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

import glob
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

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


def _count_lines(paths: List[Path]) -> int:
    total = 0
    for path in paths:
        with path.open() as handle:
            for _ in handle:
                total += 1
    return total


def _detect_actor_format(actor_files: List[Path]) -> Literal["legacy", "direct"]:
    for path in actor_files:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, list):
                    return "legacy"
                if isinstance(obj, dict):
                    return "direct"
    raise ValueError("Actor log files are empty or malformed")


def _collect_legacy_actor_records(actor_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    total_lines = _count_lines(actor_files)
    pbar = tqdm(total=total_lines, desc="Scanning actor logs", unit="line", dynamic_ncols=True)
    max_version = None
    versions_with_repair: set[int] = set()
    for entry in _iter_training_texts(actor_files):
        pbar.update(1)
        version = entry.get("metadata", {}).get("model_version")
        if version is not None:
            max_version = version if max_version is None else max(max_version, version)
            stage = entry.get("metadata", {}).get("stage")
            if stage in {"repair", "repair_self_eval"}:
                versions_with_repair.add(version)

    if not versions_with_repair:
        pbar.close()
        raise ValueError("Unable to find any repair entries in actor traces.")

    target_version = max(versions_with_repair)

    records: Dict[str, Dict[str, Any]] = {}
    pbar.reset(total=total_lines)
    pbar.set_description("Collecting latest-model entries")
    for entry in _iter_training_texts(actor_files):
        pbar.update(1)
        meta = entry.get("metadata", {})
        if meta.get("model_version") != target_version:
            continue
        stage = meta.get("stage")
        if stage not in {"repair", "repair_self_eval"}:
            continue
        problem_id = meta.get("problem_id")
        if not problem_id:
            continue
        records.setdefault(problem_id, {"dataset": meta.get("dataset"), "model_version": target_version})
        records[problem_id][stage] = entry

    pbar.close()
    return records


def _collect_direct_actor_records(actor_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    total_lines = _count_lines(actor_files)
    pbar = tqdm(total=total_lines, desc="Scanning actor logs", unit="line", dynamic_ncols=True)
    records: Dict[str, Dict[str, Any]] = {}
    for path in actor_files:
        with path.open() as handle:
            for line in handle:
                pbar.update(1)
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed line in %s", path)
                    continue
                if not isinstance(entry, dict):
                    continue
                problem_id = entry.get("problem_id")
                if not problem_id:
                    continue
                record: Dict[str, Any] = {
                    "repair": {
                        "prompt_tokens": entry.get("repair_prompt_tokens", 0),
                        "output_tokens": entry.get("repair_output_tokens", 0),
                        "reward": entry.get("repair_reward", 0.0),
                        "success": entry.get("repair_success"),
                        "metadata": {
                            "success": entry.get("repair_success"),
                            "metrics": entry.get("repair_metrics", {}),
                            "predicted_score": entry.get("self_eval_score"),
                            "source": entry.get("source"),
                        },
                    },
                    "dataset": entry.get("dataset"),
                }
                if "self_eval_score" in entry or entry.get("self_eval_output"):
                    record["repair_self_eval"] = {
                        "prompt_tokens": entry.get("self_eval_prompt_tokens", 0),
                        "output_tokens": entry.get("self_eval_output_tokens", 0),
                        "metadata": {
                            "predicted_score": entry.get("self_eval_score"),
                            "parsing_error": entry.get("self_eval_parsing_error"),
                        },
                    }
                records[problem_id] = record
    pbar.close()
    return records


def _collect_latest_actor_records(actor_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    actor_format = _detect_actor_format(actor_files)
    if actor_format == "legacy":
        return _collect_legacy_actor_records(actor_files)
    return _collect_direct_actor_records(actor_files)


def _entry_reward(entry: Dict[str, Any]) -> float:
    if not entry:
        return 0.0
    if "reward" in entry and entry["reward"] is not None:
        return float(entry["reward"])
    meta = entry.get("metadata") or {}
    if "reward" in meta and meta["reward"] is not None:
        return float(meta["reward"])
    metrics = meta.get("metrics") or {}
    if "reward" in metrics and metrics["reward"] is not None:
        return float(metrics["reward"])
    for key in ("repair_reward", "score"):
        if key in entry and entry[key] is not None:
            return float(entry[key])
    return 0.0


def _predicted_score(entry: Dict[str, Any]) -> float:
    meta = entry.get("metadata", {})
    metrics = meta.get("metrics") or {}
    if "predicted_score" in meta:
        return float(meta["predicted_score"])
    if "predicted_score" in metrics:
        return float(metrics["predicted_score"])
    return 1.0


def _pearson(x: List[float], y: List[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _rank(values: List[float]) -> List[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x: List[float], y: List[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = _rank(x)
    ry = _rank(y)
    return _pearson(rx, ry)


@hydra.main(config_path="../../../conf", config_name="swe", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    """Analyze self-eval signal using actor logs only (no expert data required)."""
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
    if sample_version is not None:
        logger.info("Found %d problems in latest model_version=%s", len(actor_records), sample_version)
    else:
        logger.info("Found %d problems from actor logs", len(actor_records))

    # Collect paired scores and rewards
    scores: List[float] = []
    rewards: List[float] = []
    for data in actor_records.values():
        if data.get("repair_self_eval"):
            score = _predicted_score(data["repair_self_eval"])
            scores.append(float(score))
            rewards.append(_entry_reward(data["repair"]))

    if not scores:
        raise ValueError("No self-eval scores found in actor logs.")

    pearson = _pearson(scores, rewards)
    spearman = _spearman(scores, rewards)
    if pearson is not None:
        logger.info("Self-eval vs reward Pearson correlation: %.4f", pearson)
    else:
        logger.info("Pearson correlation unavailable (insufficient data or zero variance).")
    if spearman is not None:
        logger.info("Self-eval vs reward Spearman correlation: %.4f", spearman)
    else:
        logger.info("Spearman correlation unavailable (insufficient data or zero variance).")

    output_path = Path(analysis_cfg.get("output_path", cfg.output_dir + "/actor_self_eval_analysis.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Histogram of self-eval scores
    hist_path = analysis_cfg.get("histogram_path")
    hist_path = Path(hist_path) if hist_path else output_path.with_name(output_path.stem + "_self_eval_hist.png")
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
    plt.xlabel("Self-eval score")
    plt.ylabel("Count")
    plt.title("Self-eval score distribution (actor)")
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()
    logger.info("Saved self-eval histogram to %s", hist_path)

    # Histogram of actor rewards
    actor_rewards = [_entry_reward(val["repair"]) for val in actor_records.values()]
    actor_hist_path = analysis_cfg.get("actor_histogram_path")
    actor_hist_path = Path(actor_hist_path) if actor_hist_path else output_path.with_name(output_path.stem + "_actor_reward_hist.png")
    plt.figure(figsize=(6, 4))
    plt.hist(actor_rewards, bins=20, range=(0, 1), color="seagreen", edgecolor="white")
    plt.xlabel("Actor reward")
    plt.ylabel("Count")
    plt.title("Actor reward distribution")
    plt.tight_layout()
    plt.savefig(actor_hist_path)
    plt.close()
    logger.info("Saved actor reward histogram to %s", actor_hist_path)

    # Scatter: self-eval score vs repair reward
    scatter_path = analysis_cfg.get("scatter_path")
    scatter_path = Path(scatter_path) if scatter_path else output_path.with_name(output_path.stem + "_self_eval_vs_reward.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(scores, rewards, alpha=0.6, color="slateblue", edgecolor="white", linewidth=0.5)
    plt.xlabel("Self-eval score")
    plt.ylabel("Repair reward")
    plt.title("Self-eval score vs repair reward (actor)")
    if pearson is not None:
        plt.annotate(f"Pearson: {pearson:.3f}", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, ha="left", va="top")
    if spearman is not None:
        plt.annotate(f"Spearman: {spearman:.3f}", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=9, ha="left", va="top")
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    logger.info("Saved self-eval vs reward scatter to %s", scatter_path)

    stats_path = analysis_cfg.get("stats_path")
    stats_path = Path(stats_path) if stats_path else output_path.with_name(output_path.stem + "_stats.json")
    stats_payload = {
        "num_problems": len(actor_records),
        "num_self_eval_scores": len(scores),
        "avg_self_eval_score": sum(scores) / len(scores) if scores else None,
        "avg_reward": sum(rewards) / len(rewards) if rewards else None,
        "pearson_self_eval_reward": pearson,
        "spearman_self_eval_reward": spearman,
    }
    with stats_path.open("w") as handle:
        json.dump(stats_payload, handle, indent=2)
    logger.info("Saved stats to %s", stats_path)


if __name__ == "__main__":  # pragma: no cover
    main()
