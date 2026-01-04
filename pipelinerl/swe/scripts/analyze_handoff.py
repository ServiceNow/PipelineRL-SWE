#!/usr/bin/env python
import argparse
import glob
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal

try:
    import matplotlib.pyplot as plt  # type: ignore
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
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
                if "value_score_mean" in entry or "value_score_last" in entry:
                    record["value_score_mean"] = entry.get("value_score_mean")
                    record["value_score_last"] = entry.get("value_score_last")
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


def _merge_records(actor_records: Dict[str, Dict[str, Any]], expert_records: Dict[str, Dict[str, Any]]):
    merged = {}
    for problem_id, data in actor_records.items():
        if "repair" not in data:
            continue
        if problem_id not in expert_records:
            continue
        merged_entry: Dict[str, Any] = {
            "repair": data["repair"],
            "repair_self_eval": data.get("repair_self_eval"),
            "expert": expert_records[problem_id],
        }
        for key in ("value_score_mean", "value_score_last"):
            if key in data:
                merged_entry[key] = data[key]
        merged[problem_id] = merged_entry
    return merged


def compute_handoff_curve(
    records: Dict[str, Dict[str, Any]],
    thresholds: List[float],
    small_token_cost: float,
    expert_token_cost: float,
    score_key: str,
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
            predicted = data.get(score_key)
            if predicted is None:
                continue
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


def compute_topk_curve(
    records: Dict[str, Dict[str, Any]],
    small_token_cost: float,
    expert_token_cost: float,
    score_key: str,
) -> List[Dict[str, Any]]:
    """Sweep k lowest scores → handoff; return metrics per k."""
    scored = [(pid, data.get(score_key)) for pid, data in records.items() if data.get(score_key) is not None]
    scored = sorted(scored, key=lambda x: x[1])
    total = len(scored)
    if total == 0:
        raise ValueError(f"No {score_key} found in actor records; cannot compute top-k curve")

    # Precompute per-problem stats
    per_problem = {}
    for pid, score in scored:
        data = records[pid]
        repair_tokens = _to_tokens(data["repair"]) + (_to_tokens(data["repair_self_eval"]) if data.get("repair_self_eval") else 0)
        expert_tokens = _to_tokens(data["expert"])
        per_problem[pid] = {
            "score": score,
            "repair_reward": _entry_reward(data["repair"]),
            "expert_reward": _entry_reward(data["expert"]),
            "repair_tokens": repair_tokens,
            "expert_tokens": expert_tokens,
        }

    results = []
    for k in range(0, total + 1):
        handoff_pids = set(pid for pid, _ in scored[:k])
        total_tokens = 0.0
        total_reward = 0.0
        total_cost = 0.0
        for pid, _ in scored:
            info = per_problem[pid]
            small_tokens = info["repair_tokens"]
            small_reward = info["repair_reward"]
            small_cost = (small_tokens / 1000.0) * small_token_cost
            if pid in handoff_pids:
                total_tokens += small_tokens + info["expert_tokens"]
                total_reward += info["expert_reward"]
                total_cost += small_cost + (info["expert_tokens"] / 1000.0) * expert_token_cost
            else:
                total_tokens += small_tokens
                total_reward += small_reward
                total_cost += small_cost
        results.append(
            {
                "k": k,
                "fraction": k / total,
                "tau": scored[k - 1][1] if k > 0 else None,
                "avg_reward": total_reward / total,
                "avg_tokens": total_tokens / total,
                "avg_cost": total_cost / total,
            }
        )
    return results


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


def _bin_stats(scores: List[float], rewards: List[float], bins: int = 10):
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        hi = lo + 1e-6
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    rows = []
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        idxs = [j for j, s in enumerate(scores) if (s >= left and (s < right or (i == bins - 1 and s <= right)))]
        if not idxs:
            rows.append(
                {"bin": i, "left": left, "right": right, "count": 0, "score_mean": None, "success_rate": None,
                 "p_r_eq_0": None, "reward_pos_mean": None, "reward_mean": None}
            )
            continue
        bin_scores = [scores[j] for j in idxs]
        bin_rewards = [rewards[j] for j in idxs]
        successes = [r > 0 for r in bin_rewards]
        reward_pos = [r for r in bin_rewards if r > 0]
        rows.append(
            {
                "bin": i,
                "left": left,
                "right": right,
                "count": len(idxs),
                "score_mean": sum(bin_scores) / len(bin_scores),
                "success_rate": sum(successes) / len(successes),
                "p_r_eq_0": (len(bin_rewards) - len(reward_pos)) / len(bin_rewards),
                "reward_pos_mean": sum(reward_pos) / len(reward_pos) if reward_pos else None,
                "reward_mean": sum(bin_rewards) / len(bin_rewards),
            }
        )
    return rows


def _cdf(scores: List[float]):
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    return [{"score": s, "cdf": (i + 1) / n} for i, s in enumerate(sorted_scores)]


def _lift_curve(scores: List[float], rewards: List[float]):
    paired = sorted(zip(scores, rewards), key=lambda x: x[0])
    cum = []
    total = 0.0
    for i, (_, r) in enumerate(paired, 1):
        total += r
        cum.append({"fraction": i / len(paired), "cum_reward_mean": total / i})
    return cum


def _roc_pr(scores: List[float], rewards: List[float]):
    # positives are rewards > 0
    paired = sorted(zip(scores, rewards), key=lambda x: x[0], reverse=True)
    P = sum(1 for _, r in paired if r > 0)
    N = len(paired) - P
    if P == 0 or N == 0:
        return None, None, None, None, None
    tp = fp = 0
    roc_points = []
    pr_points = []
    prev_score = None
    for score, r in paired:
        if score != prev_score:
            roc_points.append((fp / N, tp / P))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / P
            pr_points.append((recall, precision))
            prev_score = score
        if r > 0:
            tp += 1
        else:
            fp += 1
    roc_points.append((fp / N, tp / P))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / P
    pr_points.append((recall, precision))
    # AUROC via trapezoid on FPR-TPR
    roc_points_sorted = sorted(roc_points, key=lambda p: p[0])
    auroc = 0.0
    for (x0, y0), (x1, y1) in zip(roc_points_sorted[:-1], roc_points_sorted[1:]):
        auroc += (x1 - x0) * (y0 + y1) / 2
    pr_points_sorted = sorted(pr_points, key=lambda p: p[0])
    auprc = 0.0
    for (r0, p0), (r1, p1) in zip(pr_points_sorted[:-1], pr_points_sorted[1:]):
        auprc += (r1 - r0) * ((p0 + p1) / 2)
    return auroc, auprc, roc_points_sorted, pr_points_sorted


def run_analysis(
    actor_glob: str,
    expert_jsonl: str,
    output_path: str,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
    small_token_cost_per_1k: float,
    expert_token_cost_per_1k: float,
):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    actor_files = [Path(p) for p in glob.glob(actor_glob, recursive=True) if Path(p).is_file()]
    if not actor_files:
        raise ValueError(f"No actor JSONL files found for pattern: {actor_glob}")
    actor_files.sort()
    logger.info("Reading actor traces from %d files", len(actor_files))

    actor_records = _collect_latest_actor_records(actor_files)
    if not actor_records:
        raise ValueError("No repair entries found in actor traces for the latest model version")

    expert_path = Path(expert_jsonl)
    expert_records = _load_expert_records(expert_path)
    logger.info("Loaded %d expert records from %s", len(expert_records), expert_path)

    merged = _merge_records(actor_records, expert_records)
    logger.info("Overlap contains %d problems", len(merged))

    threshold_values = _frange(threshold_start, threshold_stop, threshold_step)

    variants = [("value_score_mean", "mean"), ("value_score_last", "last")]
    base_output_path = Path(output_path)
    small_cost = float(small_token_cost_per_1k)
    expert_cost = float(expert_token_cost_per_1k)

    for score_key, label in variants:
        if not any(data.get(score_key) is not None for data in merged.values()):
            raise ValueError(f"No {score_key} found in actor records; cannot run {label} handoff analysis")

        curve = compute_handoff_curve(merged, threshold_values, small_cost, expert_cost, score_key)
        topk_curve = compute_topk_curve(merged, small_cost, expert_cost, score_key)

        out_path = base_output_path if label == "mean" else base_output_path.with_name(
            base_output_path.stem + f"_{label}" + base_output_path.suffix
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(curve, handle, indent=2)
        logger.info("Wrote %s analysis with %d thresholds to %s", label, len(curve), out_path)

        topk_path = out_path.with_name(out_path.stem + f"_{label}_topk.json")
        with topk_path.open("w") as handle:
            json.dump(topk_curve, handle, indent=2)
        logger.info("Wrote %s top-k analysis to %s", label, topk_path)

        scores_for_hist: List[float] = []
        rewards_for_corr: List[float] = []
        for data in merged.values():
            score = data.get(score_key)
            if score is not None:
                scores_for_hist.append(float(score))
                rewards_for_corr.append(_entry_reward(data["repair"]))

        hist_path = out_path.with_name(out_path.stem + f"_{label}_hist.png")
        if MATPLOTLIB_AVAILABLE and scores_for_hist:
            plt.figure(figsize=(6, 4))
            plt.hist(scores_for_hist, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
            plt.xlabel(f"Value score ({label})")
            plt.ylabel("Count")
            plt.title(f"Value score ({label}) distribution")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()
            logger.info("Saved value histogram to %s", hist_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping histogram for %s", label)
        else:
            logger.info("No value scores found; skipping histogram for %s.", label)

        corr = _pearson(scores_for_hist, rewards_for_corr) if scores_for_hist else None
        spearman = _spearman(scores_for_hist, rewards_for_corr) if scores_for_hist else None

        scatter_path = out_path.with_name(out_path.stem + f"_{label}_vs_reward.png")
        if MATPLOTLIB_AVAILABLE and scores_for_hist and rewards_for_corr and len(scores_for_hist) == len(rewards_for_corr):
            plt.figure(figsize=(6, 4))
            plt.scatter(scores_for_hist, rewards_for_corr, alpha=0.6, color="slateblue", edgecolor="white", linewidth=0.5)
            plt.xlabel(f"Value score ({label})")
            plt.ylabel("Repair reward")
            plt.title(f"Value score ({label}) vs repair reward")
            if corr is not None:
                plt.annotate(f"Pearson: {corr:.3f}", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, ha="left", va="top")
            if spearman is not None:
                plt.annotate(f"Spearman: {spearman:.3f}", xy=(0.02, 0.88), xycoords="axes fraction", fontsize=9, ha="left", va="top")
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()
            logger.info("Saved value vs reward scatter to %s", scatter_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping scatter for %s", label)
        else:
            logger.info("Skipping scatter plot for %s (missing scores/rewards).", label)

        stats_path = out_path.with_name(out_path.stem + f"_{label}_stats.json")
        stats_payload = {
            "num_problems": len(merged),
            f"num_value_scores_{label}": len(scores_for_hist),
            f"avg_value_score_{label}": sum(scores_for_hist) / len(scores_for_hist) if scores_for_hist else None,
            "avg_reward": sum(rewards_for_corr) / len(rewards_for_corr) if rewards_for_corr else None,
            f"pearson_value_reward_{label}": corr,
            f"spearman_value_reward_{label}": spearman,
            f"auroc_{label}": None,
            f"auprc_{label}": None,
        }
        with stats_path.open("w") as handle:
            json.dump(stats_payload, handle, indent=2)
        logger.info("Saved stats to %s", stats_path)

        # Diagnostics
        bin_rows = _bin_stats(scores_for_hist, rewards_for_corr, bins=10)
        reliability_path = out_path.with_name(out_path.stem + f"_{label}_reliability.csv")
        with reliability_path.open("w") as fh:
            fh.write("bin,left,right,count,score_mean,success_rate,p_r_eq_0,reward_pos_mean,reward_mean\n")
            for row in bin_rows:
                fh.write(
                    f"{row['bin']},{row['left']},{row['right']},{row['count']},{row['score_mean']},"
                    f"{row['success_rate']},{row['p_r_eq_0']},{row['reward_pos_mean']},{row['reward_mean']}\n"
                )
        logger.info("Saved reliability/conditional reward table to %s", reliability_path)

        cdf_rows = _cdf(scores_for_hist)
        cdf_path = out_path.with_name(out_path.stem + f"_{label}_cdf.csv")
        with cdf_path.open("w") as fh:
            fh.write("score,cdf\n")
            for row in cdf_rows:
                fh.write(f"{row['score']},{row['cdf']}\n")
        logger.info("Saved CDF data to %s", cdf_path)

        lift_rows = _lift_curve(scores_for_hist, rewards_for_corr)
        lift_path = out_path.with_name(out_path.stem + f"_{label}_lift.csv")
        with lift_path.open("w") as fh:
            fh.write("fraction,cum_reward_mean\n")
            for row in lift_rows:
                fh.write(f"{row['fraction']},{row['cum_reward_mean']}\n")
        logger.info("Saved lift curve data to %s", lift_path)

        auroc, auprc, roc_points, pr_points = _roc_pr(scores_for_hist, rewards_for_corr)
        stats_payload[f"auroc_{label}"] = auroc
        stats_payload[f"auprc_{label}"] = auprc
        if roc_points and pr_points:
            roc_path = out_path.with_name(out_path.stem + f"_{label}_roc.csv")
            with roc_path.open("w") as fh:
                fh.write("fpr,tpr\n")
                for fpr, tpr in roc_points:
                    fh.write(f"{fpr},{tpr}\n")
            pr_path = out_path.with_name(out_path.stem + f"_{label}_pr.csv")
            with pr_path.open("w") as fh:
                fh.write("recall,precision\n")
                for rec, prec in pr_points:
                    fh.write(f"{rec},{prec}\n")
            logger.info("Saved ROC/PR data to %s and %s", roc_path, pr_path)

        actor_rewards = [_entry_reward(val["repair"]) for val in merged.values()]
        expert_rewards = [_entry_reward(val["expert"]) for val in merged.values()]

        actor_hist_path = out_path.with_name(out_path.stem + f"_{label}_actor_reward_hist.png")
        if MATPLOTLIB_AVAILABLE and actor_rewards:
            plt.figure(figsize=(6, 4))
            plt.hist(actor_rewards, bins=20, range=(0, 1), color="seagreen", edgecolor="white")
            plt.xlabel("Actor reward")
            plt.ylabel("Count")
            plt.title("Actor reward distribution")
            plt.tight_layout()
            plt.savefig(actor_hist_path)
            plt.close()
            logger.info("Saved actor reward histogram to %s", actor_hist_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping actor reward histogram for %s", label)

        expert_hist_path = out_path.with_name(out_path.stem + f"_{label}_expert_reward_hist.png")
        if MATPLOTLIB_AVAILABLE and expert_rewards:
            plt.figure(figsize=(6, 4))
            plt.hist(expert_rewards, bins=20, range=(0, 1), color="darkorange", edgecolor="white")
            plt.xlabel("Expert reward")
            plt.ylabel("Count")
            plt.title("Expert reward distribution")
            plt.tight_layout()
            plt.savefig(expert_hist_path)
            plt.close()
            logger.info("Saved expert reward histogram to %s", expert_hist_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping expert reward histogram for %s", label)

        table_path = out_path.with_name(out_path.stem + f"_{label}_handoff_table.csv")
        with table_path.open("w") as handle:
            handle.write("threshold,handed_off,handed_off_pct,avg_reward,avg_cost\n")
            for pt in curve:
                handle.write(
                    f"{pt['threshold']},{pt['handed_off']},{pt['handoff_fraction']*100:.2f},{pt['avg_reward']:.4f},{pt.get('avg_cost', 0):.4f}\n"
                )
        logger.info("Saved handoff table to %s", table_path)

        if curve and MATPLOTLIB_AVAILABLE:
            best = max(curve, key=lambda x: x["avg_reward"])
            logger.info(
                "[%s] Best avg reward %.3f at threshold %.2f with avg tokens %.1f",
                label,
                best["avg_reward"],
                best["threshold"],
                best["avg_tokens"],
            )
            plot_path = out_path.with_name(out_path.stem + f"_{label}.png")
            plt.figure(figsize=(6, 4))
            thresholds_pts = [pt["threshold"] for pt in curve]
            rewards = [pt["avg_reward"] for pt in curve]
            costs = [pt["avg_cost"] for pt in curve]
            plt.scatter(thresholds_pts, rewards, c=costs, cmap="viridis", s=40)
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
            plt.xlabel("Value threshold")
            plt.ylabel("Avg reward (string similarity)")
            plt.title(f"Actor/Expert Handoff Pareto Curve ({label})")
            cbar = plt.colorbar()
            cbar.set_label("Avg cost (USD)")
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info("Saved Pareto plot to %s", plot_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping Pareto plot for %s", label)

        if topk_curve and MATPLOTLIB_AVAILABLE:
            plot_path = out_path.with_name(out_path.stem + f"_{label}_topk.png")
            plt.figure(figsize=(6, 4))
            ks = [pt["k"] for pt in topk_curve]
            rewards = [pt["avg_reward"] for pt in topk_curve]
            costs = [pt["avg_cost"] for pt in topk_curve]
            plt.scatter(costs, rewards, c=ks, cmap="plasma", s=40)
            for pt in topk_curve:
                if pt["tau"] is not None:
                    plt.annotate(f"τ={pt['tau']:.3f}", (pt["avg_cost"], pt["avg_reward"]), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
            plt.xlabel("Avg cost (USD)")
            plt.ylabel("Avg reward (string similarity)")
            plt.title(f"Top-k handoff curve ({label})")
            cbar = plt.colorbar()
            cbar.set_label("k")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info("Saved top-k Pareto plot to %s", plot_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping top-k plot for %s", label)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Analyze handoff using value-head scores.")
    parser.add_argument("--actor_glob", required=True, help="Glob to actor JSONL shards")
    parser.add_argument("--expert_jsonl", required=True, help="Path to expert JSONL")
    parser.add_argument("--output_path", required=True, help="Base output path for analysis JSON")
    parser.add_argument("--threshold_start", type=float, default=0.0)
    parser.add_argument("--threshold_stop", type=float, default=1.0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--small_token_cost_per_1k", type=float, default=0.0)
    parser.add_argument("--expert_token_cost_per_1k", type=float, default=0.0)
    args = parser.parse_args()

    run_analysis(
        actor_glob=args.actor_glob,
        expert_jsonl=args.expert_jsonl,
        output_path=args.output_path,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
        small_token_cost_per_1k=args.small_token_cost_per_1k,
        expert_token_cost_per_1k=args.expert_token_cost_per_1k,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
