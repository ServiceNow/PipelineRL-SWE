#!/usr/bin/env python3
"""Emit TACO problems in the LiveCodeBench record shape.

LiveCodeBench is exhausted at 1055 problems (the lite repo has not changed since
2025-06-05) and our fold analysis shows routing savings scaling with training-set
size, so the binding constraint is data. TACO carries 26,443 competitive-programming
problems with hidden tests.

TACO predates our models, so it is contaminated for *evaluation*. It is not
obviously contaminated for *router training*: the router predicts which model will
succeed, not the answer, so memorisation only breaks it if the relative difficulty
ordering across routes is distorted rather than uniformly shifted. That is the
hypothesis this data is collected to test -- train the router on TACO, evaluate on
untouched LiveCodeBench.

No conversion of the test payload is needed. TACO's `input_output` is already the
`{"inputs", "outputs", "fn_name"}` JSON that LCB normalises to internally, so the
existing generation, extraction and grading path consumes these rows unchanged.

TACO ships one test suite with no public/private split, and none is manufactured
here. LiveCodeBench's public tests are meaningful -- they are the examples shown in
the problem statement -- whereas any prefix of TACO's tests would be arbitrary. It
would also be inert: the MDP pipeline reads `final_outcome` (full grading) for
outcomes and builds the router's feedback string from the full result codes, so
`weak_verifier_outcome` is written into the tensors and never read.

The consequence is that TACO cannot support the weak-verifier ablation. That is
correct rather than a limitation: that ablation needs a verifier that is weak for a
real reason, which only LiveCodeBench has here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DIFFICULTY_MAP = {"EASY": "easy", "MEDIUM": "medium", "MEDIUM_HARD": "medium",
                  "HARD": "hard", "VERY_HARD": "hard", "UNKNOWN_DIFFICULTY": "unknown"}


def _tests(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """The one suite TACO ships, truncated to matched input/output pairs."""
    inputs = list(payload.get("inputs") or [])
    outputs = list(payload.get("outputs") or [])
    pairs = min(len(inputs), len(outputs))
    return (
        {"inputs": inputs[:pairs], "outputs": outputs[:pairs], "fn_name": payload.get("fn_name")},
        pairs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSONL of LCB-shaped records")
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--max-problems", type=int, default=0, help="0 keeps everything")
    parser.add_argument("--min-tests", type=int, default=2,
                        help="Drop problems with fewer usable test pairs than this")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    shards = (
        ["ALL/test-00000-of-00001.parquet"] if args.split == "test"
        else [f"ALL/train-0000{i}-of-00009.parquet" for i in range(9)]
    )
    frames = [
        pd.read_parquet(hf_hub_download("BAAI/TACO", shard, repo_type="dataset"))
        for shard in shards
    ]
    frame = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    skipped_no_tests = skipped_unparsable = 0
    for index, record in frame.iterrows():
        raw = record.get("input_output")
        if not raw:
            skipped_no_tests += 1
            continue
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            skipped_unparsable += 1
            continue
        suite, n_tests = _tests(payload)
        if n_tests < args.min_tests:
            skipped_no_tests += 1
            continue
        source = str(record.get("source") or "taco")
        rows.append({
            "question_content": str(record.get("question") or ""),
            "question_title": str(record.get("name") or "") or f"taco_{index}",
            "question_id": str(index),
            "platform": f"taco_{source}",
            "difficulty": DIFFICULTY_MAP.get(str(record.get("difficulty")), "unknown"),
            "contest_date": str(record.get("date") or "")[:10] or "1970-01-01",
            "starter_code": str(record.get("starter_code") or ""),
            "metadata": {"func_name": payload.get("fn_name"), "taco_source": source},
            # Both fields carry the same suite: the shape is required downstream, the
            # distinction is not real here and is not invented.
            "_public_evaluation_sample": {"input_output": json.dumps(suite)},
            "_evaluation_sample": {"input_output": json.dumps(suite)},
            "_n_public_tests": n_tests,
            "_n_private_tests": 0,
            "_taco_split": args.split,
        })

    if args.max_problems and len(rows) > args.max_problems:
        rows = random.Random(args.seed).sample(rows, args.max_problems)

    from collections import Counter
    stats = {
        "split": args.split,
        "rows_in_source": int(len(frame)),
        "emitted": len(rows),
        "skipped_no_or_too_few_tests": skipped_no_tests,
        "skipped_unparsable_input_output": skipped_unparsable,
        "difficulty": dict(Counter(r["difficulty"] for r in rows)),
        "source": dict(Counter(r["platform"] for r in rows).most_common(10)),
        "function_call_problems": sum(
            1 for r in rows if (r["metadata"] or {}).get("func_name")
        ),
        "stdin_problems": sum(
            1 for r in rows if not (r["metadata"] or {}).get("func_name")
        ),
        "date_range": [
            min((r["contest_date"] for r in rows), default=None),
            max((r["contest_date"] for r in rows), default=None),
        ],
        "median_tests": (
            sorted(r["_n_public_tests"] for r in rows)[len(rows) // 2] if rows else 0
        ),
        "public_private_split": "none (TACO ships one suite; no split is manufactured)",
    }
    print(json.dumps(stats, indent=2))
    if args.stats_only:
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (output.parent / f"{output.stem}_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {len(rows)} problems -> {output}")


if __name__ == "__main__":
    main()
