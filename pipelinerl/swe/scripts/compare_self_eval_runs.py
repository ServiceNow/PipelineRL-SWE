import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
                continue
            if isinstance(rec, list):
                records.extend(rec)
            elif isinstance(rec, dict):
                records.append(rec)
    return records


def _to_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        pid = rec.get("problem_id")
        if not pid:
            continue
        out[pid] = rec
    return out


def _entry_reward(entry: Dict[str, Any]) -> float:
    if not entry:
        return 0.0
    for key in ("repair_reward", "reward", "success"):
        if key in entry and entry[key] is not None:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                continue
    metrics = entry.get("repair_metrics") or entry.get("metrics") or {}
    if "reward" in metrics and metrics["reward"] is not None:
        return float(metrics["reward"])
    return 0.0


def _predicted_score(entry: Dict[str, Any]) -> float | None:
    for key in ("self_eval_score", "predicted_score"):
        if key in entry and entry[key] is not None:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                continue
    meta = entry.get("metadata") or {}
    metrics = meta.get("metrics") or {}
    for key in ("predicted_score", "self_eval_score"):
        if key in meta and meta[key] is not None:
            return float(meta[key])
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


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
        avg_rank = (i + j) / 2 + 1  # 1-based
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


def _paired_scores_and_rewards(
    initial: Dict[str, Dict[str, Any]],
    trained: Dict[str, Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    init_scores: List[float] = []
    trained_scores: List[float] = []
    rewards: List[float] = []
    deltas: List[float] = []
    for pid, init_rec in initial.items():
        trained_rec = trained.get(pid)
        if not trained_rec:
            continue
        init_score = _predicted_score(init_rec)
        trained_score = _predicted_score(trained_rec)
        if init_score is None or trained_score is None:
            continue
        init_scores.append(float(init_score))
        trained_scores.append(float(trained_score))
        rewards.append(_entry_reward(trained_rec) or _entry_reward(init_rec))
        deltas.append(float(trained_score) - float(init_score))
    return init_scores, trained_scores, rewards, deltas


@hydra.main(config_path="../../../conf", config_name="swe", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    """
    Compare self-eval outputs between two runs (e.g., initial vs trained weights) on identical problems.
    Expects two JSONL files with actor-eval style entries keyed by problem_id.
    """
    analysis_cfg = cfg.handoff_analysis
    initial_path = Path(analysis_cfg.get("actor_glob"))
    trained_path = Path(analysis_cfg.get("expert_jsonl", cfg.expert_eval.output_path))
    if not initial_path.exists() or not trained_path.exists():
        raise FileNotFoundError("Provide initial JSONL via handoff_analysis.actor_glob and trained JSONL via handoff_analysis.expert_jsonl.")

    logger.info("Loading initial run from %s", initial_path)
    init_records = _to_map(_load_jsonl(initial_path))
    logger.info("Loading trained run from %s", trained_path)
    trained_records = _to_map(_load_jsonl(trained_path))

    init_scores, trained_scores, rewards, deltas = _paired_scores_and_rewards(init_records, trained_records)
    if not init_scores:
        raise ValueError("No overlapping problem_ids with self-eval scores between the two runs.")

    logger.info("Paired problems: %d", len(init_scores))

    init_pearson = _pearson(init_scores, rewards)
    init_spearman = _spearman(init_scores, rewards)
    trained_pearson = _pearson(trained_scores, rewards)
    trained_spearman = _spearman(trained_scores, rewards)

    logger.info("Initial Pearson: %s", f"{init_pearson:.4f}" if init_pearson is not None else "n/a")
    logger.info("Initial Spearman: %s", f"{init_spearman:.4f}" if init_spearman is not None else "n/a")
    logger.info("Trained Pearson: %s", f"{trained_pearson:.4f}" if trained_pearson is not None else "n/a")
    logger.info("Trained Spearman: %s", f"{trained_spearman:.4f}" if trained_spearman is not None else "n/a")

    output_path = Path(analysis_cfg.get("output_path", cfg.output_dir + "/self_eval_compare.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scatter trained vs initial
    scatter_path = analysis_cfg.get("scatter_path")
    scatter_path = Path(scatter_path) if scatter_path else output_path.with_name(output_path.stem + "_trained_vs_initial.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(init_scores, trained_scores, alpha=0.6, color="slateblue", edgecolor="white", linewidth=0.5)
    plt.xlabel("Initial self-eval score")
    plt.ylabel("Trained self-eval score")
    plt.title("Self-eval: trained vs initial")
    plt.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    logger.info("Saved trained vs initial scatter to %s", scatter_path)

    # Histogram of deltas
    delta_path = output_path.with_name(output_path.stem + "_delta_hist.png")
    plt.figure(figsize=(6, 4))
    plt.hist(deltas, bins=20, color="orchid", edgecolor="white")
    plt.xlabel("Trained - Initial self-eval score")
    plt.ylabel("Count")
    plt.title("Self-eval score deltas")
    plt.tight_layout()
    plt.savefig(delta_path)
    plt.close()
    logger.info("Saved delta histogram to %s", delta_path)

    # Stats payload
    init_mean = sum(init_scores) / len(init_scores)
    trained_mean = sum(trained_scores) / len(trained_scores)
    delta_mean = sum(deltas) / len(deltas)

    init_var = sum((x - init_mean) ** 2 for x in init_scores) / len(init_scores)
    trained_var = sum((x - trained_mean) ** 2 for x in trained_scores) / len(trained_scores)
    delta_var = sum((d - delta_mean) ** 2 for d in deltas) / len(deltas)

    stats = {
        "num_pairs": len(init_scores),
        "init_mean": init_mean,
        "init_std": math.sqrt(init_var),
        "init_var": init_var,
        "trained_mean": trained_mean,
        "trained_std": math.sqrt(trained_var),
        "trained_var": trained_var,
        "delta_mean": delta_mean,
        "delta_std": math.sqrt(delta_var),
        "delta_var": delta_var,
        "init_pearson": init_pearson,
        "init_spearman": init_spearman,
        "trained_pearson": trained_pearson,
        "trained_spearman": trained_spearman,
    }
    with output_path.open("w") as handle:
        json.dump(stats, handle, indent=2)
    logger.info("Saved stats to %s", output_path)
    logger.info(
        "Init variance: %.6f, Trained variance: %.6f, Delta variance: %.6f",
        init_var,
        trained_var,
        delta_var,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
