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

from pipelinerl.swe.handoff_eval import HandoffRecord, compute_handoff_curve, summarize_handoff_curve

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
        if stage in {"repair", "repair_self_eval"}:
            if "performance_value_head_prompt_last_all" in entry:
                records[problem_id]["performance_value_head_prompt_last_all"] = entry.get(
                    "performance_value_head_prompt_last_all"
                )

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
                if "performance_value_head_prompt_last_all" in entry:
                    record["performance_value_head_prompt_last_all"] = entry.get(
                        "performance_value_head_prompt_last_all"
                    )
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


def _merge_records(
    actor_records: Dict[str, Dict[str, Any]],
    expert_records_list: List[Dict[str, Dict[str, Any]]],
):
    merged = {}
    for problem_id, data in actor_records.items():
        if "repair" not in data:
            continue
        expert_entries: List[Dict[str, Any]] = []
        missing = False
        for expert_records in expert_records_list:
            if problem_id not in expert_records:
                missing = True
                break
            expert_entries.append(expert_records[problem_id])
        if missing:
            continue
        merged_entry: Dict[str, Any] = {
            "repair": data["repair"],
            "repair_self_eval": data.get("repair_self_eval"),
            "experts": expert_entries,
        }
        if "performance_value_head_prompt_last_all" in data:
            merged_entry["performance_value_head_prompt_last_all"] = data["performance_value_head_prompt_last_all"]
        merged[problem_id] = merged_entry
    return merged


def _extract_expert_score(
    data: Dict[str, Any],
    score_list_key: str,
) -> float | None:
    value = data.get(score_list_key)
    if isinstance(value, list):
        expert_values = [float(v) for v in value[1:]]
        return max(expert_values) if expert_values else None
    return None


def _best_expert_index(score_list: List[float]) -> int | None:
    if len(score_list) < 2:
        return None
    best_offset = max(range(1, len(score_list)), key=lambda i: float(score_list[i]))
    return best_offset - 1


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


def _routing_summary(
    merged: Dict[str, Dict[str, Any]],
    score_list_key: str,
    threshold: float,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {"policy": 0, "abstain": 0}
    total = 0
    for data in merged.values():
        score_list = data.get(score_list_key)
        if not isinstance(score_list, list) or not score_list:
            continue
        total += 1
        max_score = max(float(v) for v in score_list)
        if max_score < threshold:
            counts["abstain"] += 1
            continue
        best_idx = max(range(len(score_list)), key=lambda i: float(score_list[i]))
        if best_idx == 0:
            counts["policy"] += 1
        else:
            key = f"expert_{best_idx}"
            counts[key] = counts.get(key, 0) + 1
    if total == 0:
        return {
            "threshold": threshold,
            "total": 0,
            "counts": counts,
            "percentages": {},
        }
    percentages = {k: (v / total) * 100.0 for k, v in counts.items()}
    return {
        "threshold": threshold,
        "total": total,
        "counts": counts,
        "percentages": percentages,
    }


def run_analysis(
    actor_glob: str,
    expert_jsonls: List[str],
    output_path: str,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
    score_list_key: str,
    handoff_margin: float,
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

    expert_records_list = []
    for expert_jsonl in expert_jsonls:
        expert_path = Path(expert_jsonl)
        expert_records = _load_expert_records(expert_path)
        logger.info("Loaded %d expert records from %s", len(expert_records), expert_path)
        expert_records_list.append(expert_records)

    merged = _merge_records(actor_records, expert_records_list)
    logger.info("Overlap contains %d problems", len(merged))

    threshold_values = _frange(threshold_start, threshold_stop, threshold_step)

    variants = [("performance_value_head_prompt_last_all", "prompt_last")]
    base_output_path = Path(output_path)

    for score_key, label in variants:
        if not any(data.get(score_key) is not None for data in merged.values()):
            raise ValueError(f"No {score_key} found in actor records; cannot run {label} handoff analysis")
        if not any(_extract_expert_score(data, score_list_key) is not None for data in merged.values()):
            raise ValueError(
                f"No expert scores found in {score_list_key}; cannot run {label} handoff analysis"
            )

        records: List[HandoffRecord] = []
        for data in merged.values():
            score_list = data.get(score_key)
            if not isinstance(score_list, list) or not score_list:
                continue
            policy_score = float(score_list[0])
            expert_score = _extract_expert_score(data, score_list_key)
            best_expert_index = _best_expert_index(score_list)
            if expert_score is None or best_expert_index is None:
                continue
            experts = data.get("experts") or []
            if best_expert_index >= len(experts):
                continue
            expert_reward = _entry_reward(experts[best_expert_index])
            records.append(
                HandoffRecord(
                    policy_score=policy_score,
                    expert_score=float(expert_score),
                    policy_reward=_entry_reward(data["repair"]),
                    expert_reward=expert_reward,
                )
            )

        curve = compute_handoff_curve(records, threshold_values, handoff_margin)
        summary = summarize_handoff_curve(curve)

        out_path = base_output_path if label == "mean" else base_output_path.with_name(
            base_output_path.stem + f"_{label}" + base_output_path.suffix
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(curve, handle, indent=2)
        logger.info("Wrote %s analysis with %d thresholds to %s", label, len(curve), out_path)

        scores_for_hist: List[float] = []
        rewards_for_corr: List[float] = []
        for data in merged.values():
            score_list = data.get(score_key)
            if not isinstance(score_list, list) or not score_list:
                continue
            policy_score = float(score_list[0])
            expert_score = _extract_expert_score(data, score_list_key)
            if expert_score is None:
                continue
            routing_score = max(policy_score, expert_score)
            scores_for_hist.append(routing_score)
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
            "handoff_margin": handoff_margin,
            "score_list_key": score_list_key,
        }
        with stats_path.open("w") as handle:
            json.dump(stats_payload, handle, indent=2)
        logger.info("Saved stats to %s", stats_path)

        routing_dir = out_path.with_name(out_path.stem + f"_{label}_routing")
        routing_dir.mkdir(parents=True, exist_ok=True)
        if not threshold_values:
            logger.warning("No thresholds provided; skipping routing pie summaries")
        else:
            for threshold in threshold_values:
                routing = _routing_summary(merged, score_list_key, float(threshold))
                routing_path = routing_dir / f"routing_{threshold:.2f}.csv"
                with routing_path.open("w") as fh:
                    fh.write("label,count,percentage\n")
                    for key, count in sorted(routing["counts"].items()):
                        pct = routing["percentages"].get(key, 0.0)
                        fh.write(f"{key},{count},{pct}\n")
                if MATPLOTLIB_AVAILABLE and routing["total"] > 0:
                    pie_path = routing_dir / f"routing_{threshold:.2f}.png"
                    labels = []
                    values = []
                    for key, count in sorted(routing["counts"].items()):
                        if count <= 0:
                            continue
                        labels.append(key)
                        values.append(count)
                    if values:
                        plt.figure(figsize=(5, 5))
                        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
                        plt.title(f"Routing share @ threshold {routing['threshold']:.2f}")
                        plt.tight_layout()
                        plt.savefig(pie_path)
                        plt.close()
                elif not MATPLOTLIB_AVAILABLE:
                    logger.warning("matplotlib not available; skipping routing pies")
                    break

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
        expert_rewards = []
        for val in merged.values():
            score_list = val.get(score_key)
            if not isinstance(score_list, list) or not score_list:
                continue
            best_expert_index = _best_expert_index(score_list)
            if best_expert_index is None:
                continue
            experts = val.get("experts") or []
            if best_expert_index >= len(experts):
                continue
            expert_rewards.append(_entry_reward(experts[best_expert_index]))

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

        if curve and MATPLOTLIB_AVAILABLE:
            best = summary
            if best.get("best_avg_reward") is not None:
                logger.info(
                    "[%s] Best avg reward %.3f at threshold %.2f",
                    label,
                    best["best_avg_reward"],
                    best["best_threshold"],
                )
            plot_path = out_path.with_name(out_path.stem + f"_{label}.png")
            plt.figure(figsize=(6, 4))
            thresholds_pts = [pt["threshold"] for pt in curve]
            rewards = [pt["avg_reward"] if pt["avg_reward"] is not None else 0.0 for pt in curve]
            plt.plot(thresholds_pts, rewards, marker="o", linestyle="-", color="steelblue")
            plt.xlabel("Quality threshold")
            plt.ylabel("Avg reward (string similarity)")
            plt.title(f"Actor/Expert Handoff Curve ({label})")
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            logger.info("Saved handoff plot to %s", plot_path)
        elif not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available; skipping handoff plot for %s", label)


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Analyze handoff using policy/expert value scores.")
    parser.add_argument("--actor_glob", required=True, help="Glob to actor JSONL shards")
    parser.add_argument(
        "--expert_jsonls",
        required=True,
        help="Comma-separated list of expert JSONL paths (one per expert, in order)",
    )
    parser.add_argument("--output_path", required=True, help="Base output path for analysis JSON")
    parser.add_argument("--threshold_start", type=float, default=0.0)
    parser.add_argument("--threshold_stop", type=float, default=1.0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--score_list_key", default="performance_value_head_prompt_last_all")
    parser.add_argument("--handoff_margin", type=float, default=0.0)
    args = parser.parse_args()

    expert_jsonls = [p.strip() for p in args.expert_jsonls.split(",") if p.strip()]
    run_analysis(
        actor_glob=args.actor_glob,
        expert_jsonls=expert_jsonls,
        output_path=args.output_path,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
        score_list_key=args.score_list_key,
        handoff_margin=args.handoff_margin,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
