"""Sample a deterministic subset from an existing instance-id pool."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.open() if line.strip()]
    if not ids:
        raise ValueError(f"No ids found in {path}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sort-output", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pool_path = Path(args.pool_path)
    output_path = Path(args.output_path)
    ids = _read_ids(pool_path)
    if args.n <= 0:
        raise ValueError("--n must be positive")
    if args.n > len(ids):
        raise ValueError(f"Requested {args.n} ids from pool of {len(ids)} ids: {pool_path}")

    rng = random.Random(args.seed)
    sampled = rng.sample(ids, args.n)
    if args.sort_output:
        sampled = sorted(sampled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for instance_id in sampled:
            handle.write(instance_id + "\n")

    print(f"Wrote {len(sampled)} ids to {output_path} from {pool_path}")


if __name__ == "__main__":
    main()
