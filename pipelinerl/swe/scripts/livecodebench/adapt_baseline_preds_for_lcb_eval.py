#!/usr/bin/env python3
"""
Adapt train_qwen_embedding_router_baseline.py eval_predictions.jsonl (rows with
pred_rewards) into the format evaluate_lcb_full_router.py expects
(p_successes + per-route tokens/public successes from router_eval.jsonl).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as in_f:
        for line in in_f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = str(row.get("problem_id") or "")
            if pid:
                out[pid] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--router-eval", required=True, help="router_eval.jsonl from materialize")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    preds = _read_jsonl(Path(args.baseline_predictions))
    source = _read_jsonl(Path(args.router_eval))

    written = 0
    missing = 0
    with open(args.output, "w") as out_f:
        for pid, src in source.items():
            pred = preds.get(pid)
            if pred is None:
                missing += 1
                continue
            out_f.write(json.dumps({
                "problem_id": pid,
                "route_labels": list(src["route_labels"]),
                "p_successes": [float(v) for v in pred["pred_rewards"]],
                "route_successes": list(src["route_successes"]),
                "route_public_successes": list(src["route_public_successes"]),
                "route_prompt_tokens": list(src["route_prompt_tokens"]),
                "route_completion_tokens": list(src["route_completion_tokens"]),
            }) + "\n")
            written += 1

    print(f"adapted {written} rows ({missing} predictions missing)")


if __name__ == "__main__":
    main()
