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
    # Reuse the same line count to drive the progress bar for pass 2
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
                repair_entry = {
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
                }
                record: Dict[str, Any] = {"repair": repair_entry, "dataset": entry.get("dataset")}
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
            expert_entry = {
                "prompt_tokens": record.get("repair_prompt_tokens", record.get("prompt_tokens", 0)),
                "output_tokens": record.get("repair_output_tokens", record.get("output_tokens", 0)),
                "reward": record.get("repair_reward", record.get("reward", 0.0)),
                "success": record.get("repair_success", record.get("success")),
                "metadata": {
                    "success": record.get("repair_success", record.get("success")),
                    "metrics": record.get("repair_metrics", record.get("metrics", {})),
                },
            }
            expert[problem_id] = expert_entry
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


def compute_handoff_curve(
    records: Dict[str, Dict[str, Any]],
    thresholds: List[float],
    small_token_cost: float,
    expert_token_cost: float,
) -> List[Dict[str, Any]]:
    total_problems = len(records)
    if total_problems == 0:
        raise ValueError("No overlapping problems between actor and expert results")

    per_problem_small_tokens: Dict[str, int] = {}
    for pid, data in records.items():
        repair_tokens = _to_tokens(data["repair"])
        self_eval_tokens = 0
        if data.get("repair_self_eval"):
            self_eval_tokens = _to_tokens(data["repair_self_eval"])
        per_problem_small_tokens[pid] = repair_tokens + self_eval_tokens

    results = []
    for threshold in thresholds:
        total_tokens = 0.0
        total_reward = 0.0
        total_cost = 0.0
        handoffs = 0

        for pid, data in records.items():
            predicted = _predicted_score(data.get("repair_self_eval") or {})
            small_tokens = per_problem_small_tokens[pid]
            small_reward = _entry_reward(data["repair"])
            small_cost = (small_tokens / 1000.0) * small_token_cost

            use_expert = predicted < threshold
            if use_expert:
                expert_entry = data["expert"]
                expert_tokens = _to_tokens(expert_entry)
                total_tokens += small_tokens + expert_tokens
                total_reward += _entry_reward(expert_entry)
                expert_cost = (expert_tokens / 1000.0) * expert_token_cost
                total_cost += small_cost + expert_cost
                handoffs += 1
            else:
                total_tokens += small_tokens
                total_reward += small_reward
                total_cost += small_cost

        results.append(
            {
                "threshold": threshold,
                "avg_reward": total_reward / total_problems,
                "avg_tokens": total_tokens / total_problems,
                "avg_cost": total_cost / total_problems,
                "handoff_fraction": handoffs / total_problems,
                "handed_off": handoffs,
                "total_problems": total_problems,
            }
        )
    return results


def _predicted_scores(records: Dict[str, Dict[str, Any]]) -> List[float]:
    scores: List[float] = []
    for data in records.values():
        if data.get("repair_self_eval"):
            score = _predicted_score(data["repair_self_eval"])
            if score is not None:
                scores.append(float(score))
    return scores


def _rewards(records: Dict[str, Dict[str, Any]]) -> List[float]:
    rewards: List[float] = []
    for data in records.values():
        rewards.append(_entry_reward(data["repair"]))
    return rewards


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
    """Return average ranks for values with ties."""
    indexed = list(enumerate(values))
    indexed.sort(key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ranks are 1-based
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
    if sample_version is not None:
        logger.info("Found %d problems in latest model_version=%s", len(actor_records), sample_version)
    else:
        logger.info("Found %d problems from actor logs", len(actor_records))

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

    small_cost = float(analysis_cfg.get("small_token_cost_per_1k", 0.0))
    expert_cost = float(analysis_cfg.get("expert_token_cost_per_1k", 0.0))

    curve = compute_handoff_curve(merged, threshold_values, small_cost, expert_cost)

    output_path = Path(analysis_cfg.get("output_path", cfg.output_dir + "/handoff_analysis.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(curve, handle, indent=2)
    logger.info("Wrote analysis with %d thresholds to %s", len(curve), output_path)

    # Additional summaries: self-eval histogram, correlation, handoff table
    scores_for_hist: List[float] = []
    rewards_for_corr: List[float] = []
    for data in merged.values():
        if data.get("repair_self_eval"):
            score = _predicted_score(data["repair_self_eval"])
            if score is not None:
                scores_for_hist.append(float(score))
                rewards_for_corr.append(_entry_reward(data["repair"]))

    hist_path = analysis_cfg.get("histogram_path")
    hist_path = Path(hist_path) if hist_path else output_path.with_name(output_path.stem + "_self_eval_hist.png")
    if scores_for_hist:
        plt.figure(figsize=(6, 4))
        plt.hist(scores_for_hist, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
        plt.xlabel("Self-eval score")
        plt.ylabel("Count")
        plt.title("Self-eval score distribution")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        logger.info("Saved self-eval histogram to %s", hist_path)
    else:
        logger.info("No self-eval scores found; skipping histogram.")

    corr = _pearson(scores_for_hist, rewards_for_corr) if scores_for_hist else None
    if corr is not None:
        logger.info("Self-eval vs reward Pearson correlation: %.4f", corr)
    else:
        logger.info("Correlation unavailable (insufficient data or zero variance).")

    spearman = _spearman(scores_for_hist, rewards_for_corr) if scores_for_hist else None
    if spearman is not None:
        logger.info("Self-eval vs reward Spearman correlation: %.4f", spearman)
    else:
        logger.info("Spearman correlation unavailable (insufficient data or zero variance).")

    # Scatter: self-eval score vs repair reward
    scatter_path = analysis_cfg.get("scatter_path")
    scatter_path = Path(scatter_path) if scatter_path else output_path.with_name(output_path.stem + "_self_eval_vs_reward.png")
    if scores_for_hist and rewards_for_corr and len(scores_for_hist) == len(rewards_for_corr):
        plt.figure(figsize=(6, 4))
        plt.scatter(scores_for_hist, rewards_for_corr, alpha=0.6, color="slateblue", edgecolor="white", linewidth=0.5)
        plt.xlabel("Self-eval score")
        plt.ylabel("Repair reward")
        plt.title("Self-eval score vs repair reward")
        if corr is not None:
            plt.annotate(f"Pearson: {corr:.3f}", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, ha="left", va="top")
        if spearman is not None:
            plt.annotate(f"Spearman: {spearman:.3f}", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=9, ha="left", va="top")
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        logger.info("Saved self-eval vs reward scatter to %s", scatter_path)
    else:
        logger.info("Skipping scatter plot (missing scores/rewards).")

    stats_path = analysis_cfg.get("stats_path")
    stats_path = Path(stats_path) if stats_path else output_path.with_name(output_path.stem + "_stats.json")
    stats_payload = {
        "num_problems": len(merged),
        "num_self_eval_scores": len(scores_for_hist),
        "avg_self_eval_score": sum(scores_for_hist) / len(scores_for_hist) if scores_for_hist else None,
        "avg_reward": sum(rewards_for_corr) / len(rewards_for_corr) if rewards_for_corr else None,
        "pearson_self_eval_reward": corr,
        "spearman_self_eval_reward": spearman,
    }
    with stats_path.open("w") as handle:
        json.dump(stats_payload, handle, indent=2)
    logger.info("Saved stats to %s", stats_path)

    # Reward histograms for actor vs expert
    actor_rewards = _rewards(merged)
    expert_rewards = [_entry_reward(val["expert"]) for val in merged.values()]

    actor_hist_path = analysis_cfg.get("actor_histogram_path")
    actor_hist_path = Path(actor_hist_path) if actor_hist_path else output_path.with_name(output_path.stem + "_actor_reward_hist.png")
    if actor_rewards:
        plt.figure(figsize=(6, 4))
        plt.hist(actor_rewards, bins=20, range=(0, 1), color="seagreen", edgecolor="white")
        plt.xlabel("Actor reward")
        plt.ylabel("Count")
        plt.title("Actor reward distribution")
        plt.tight_layout()
        plt.savefig(actor_hist_path)
        plt.close()
        logger.info("Saved actor reward histogram to %s", actor_hist_path)

    expert_hist_path = analysis_cfg.get("expert_histogram_path")
    expert_hist_path = Path(expert_hist_path) if expert_hist_path else output_path.with_name(output_path.stem + "_expert_reward_hist.png")
    if expert_rewards:
        plt.figure(figsize=(6, 4))
        plt.hist(expert_rewards, bins=20, range=(0, 1), color="darkorange", edgecolor="white")
        plt.xlabel("Expert reward")
        plt.ylabel("Count")
        plt.title("Expert reward distribution")
        plt.tight_layout()
        plt.savefig(expert_hist_path)
        plt.close()
        logger.info("Saved expert reward histogram to %s", expert_hist_path)

    table_path = analysis_cfg.get("handoff_table_path")
    table_path = Path(table_path) if table_path else output_path.with_name(output_path.stem + "_handoff_table.csv")
    with table_path.open("w") as handle:
        handle.write("threshold,handed_off,handed_off_pct,avg_reward,avg_cost\n")
        for pt in curve:
            handle.write(
                f"{pt['threshold']},{pt['handed_off']},{pt['handoff_fraction']*100:.2f},{pt['avg_reward']:.4f},{pt.get('avg_cost', 0):.4f}\n"
            )
    logger.info("Saved handoff table to %s", table_path)

    if curve:
        best = max(curve, key=lambda x: x["avg_reward"])
        logger.info(
            "Best avg reward %.3f at threshold %.2f with avg tokens %.1f",
            best["avg_reward"],
            best["threshold"],
            best["avg_tokens"],
        )

        plot_path = analysis_cfg.get("plot_path")
        if not plot_path:
            plot_path = output_path.with_suffix(".png")
        else:
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 4))
        thresholds = [pt["threshold"] for pt in curve]
        rewards = [pt["avg_reward"] for pt in curve]
        costs = [pt["avg_cost"] for pt in curve]
        plt.scatter(thresholds, rewards, c=costs, cmap="viridis", s=40)
        offsets = [8, -6, 14, -12, 0]
        for idx, pt in enumerate(curve):
            offset = offsets[idx % len(offsets)]
            plt.annotate(
                f"${pt['avg_cost']:.2f}",
                (pt["threshold"], pt["avg_reward"]),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=7,
            )
        plt.xlabel("Self-eval threshold")
        plt.ylabel("Avg reward (string similarity)")
        plt.title("Actor/Expert Handoff Pareto Curve")
        cbar = plt.colorbar()
        cbar.set_label("Avg cost (USD)")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info("Saved Pareto plot to %s", plot_path)


if __name__ == "__main__":  # pragma: no cover
    main()
