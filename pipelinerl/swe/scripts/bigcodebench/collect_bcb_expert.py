#!/usr/bin/env python3
"""Collect draws for one route on BigCodeBench, graded by our own validated harness.

Mirrors collect_lcb_expert.py so the two pools produce the SAME row schema and the downstream
tensor/probe/replay code is shared: one row per (task, draw) with problem_id, resolved, token
counts, finish_reason and provider. It inherits every collection fix from the LCB path --
provider.ignore for the harmony tool-call endpoints, reasoning.enabled for hybrid models,
require_parameters, the tool_calls retry -- because those were serving artifacts, not LCB ones.

Only tasks whose own reference solution passed repeatedly in this environment are collected
(bcb_keep.json from validate_bcb.py); everything else would measure our sandbox.

New here, and a gap in the LCB rows: `_max_tokens` is recorded, so a pool is self-describing
about the cap that produced it -- the cap turned out to move a rung by 8.7 points.
"""
from __future__ import annotations
import argparse, asyncio, json, logging, random, sys
from pathlib import Path
from typing import Any

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (  # noqa: E402
    openrouter_call, extract_code,
)
from pipelinerl.swe.scripts.bigcodebench.bcb_grader import grade  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BCB_SYSTEM = (
    "You are an expert Python programmer. Write a complete, correct solution. "
    "Output only Python code with no explanation: include every import the code needs and "
    "define the required function exactly as named in the prompt."
)


def _read_latest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[str(r["problem_id"])] = r
    return rows


def _is_complete(row: dict) -> bool:
    finish = row.get("finish_reason")
    has_text = bool(str(row.get("full_output") or "").strip())
    return isinstance(row.get("resolved"), bool) and (has_text or finish == "length")


async def collect(tasks: list[dict], out_path: Path, a) -> None:
    done = {k: v for k, v in _read_latest(out_path).items() if _is_complete(v)}
    todo = [t for t in tasks if t["task_id"] not in done]
    logger.info("%s: %d/%d reusable, %d to collect", out_path.name, len(done), len(tasks), len(todo))
    if not todo:
        return
    key = Path(a.api_key_file).read_text().strip()
    sem = asyncio.Semaphore(a.concurrency)
    grade_sem = asyncio.Semaphore(a.grade_concurrency)
    results = list(done.values())

    async with aiohttp.ClientSession() as session:
        async def one(t: dict) -> dict:
            try:
                out = await openrouter_call(
                    session, a.model, BCB_SYSTEM, t["instruct_prompt"], key,
                    base_url=a.base_url, max_tokens=a.max_tokens, temperature=a.temperature,
                    title=a.title, semaphore=sem, gen_timeout=a.gen_timeout,
                    top_p=a.top_p, reasoning_effort=a.reasoning_effort or None,
                    reasoning_enabled=a.reasoning_enabled, require_parameters=a.require_parameters,
                    ignore_providers=[x for x in a.ignore_providers.split(",") if x.strip()] or None,
                    logprobs=a.logprobs,
                )
            except Exception as e:                      # network/provider error, not a model failure
                return {"problem_id": t["task_id"], "route_label": a.route_label, "model": a.model,
                        "full_output": "", "code": "", "resolved": False,
                        "finish_reason": "error", "error": f"{type(e).__name__}: {e}"[:300],
                        "prompt_tokens": 0, "completion_tokens": 0, "_max_tokens": a.max_tokens}
            code = extract_code(out["full_output"]) or out["full_output"]
            async with grade_sem:
                g = await asyncio.to_thread(grade, code, t["test"], a.grade_timeout)
            return {"problem_id": t["task_id"], "route_label": a.route_label, "model": a.model,
                    "full_output": out["full_output"], "code": code,
                    "thinking_text": out.get("thinking_text", ""),
                    "thinking_chars": len(out.get("thinking_text") or ""),
                    "prompt_tokens": out.get("prompt_tokens", 0),
                    "completion_tokens": out.get("completion_tokens", 0),
                    "finish_reason": out.get("finish_reason"), "provider": out.get("provider"),
                    "latency_s": out.get("latency_s"), "resolved": bool(g.ok),
                    "tests_ran": g.ran, "grade_status": g.status, "grade_detail": g.detail,
                    "_generation_temperature": a.temperature, "_top_p": a.top_p,
                    "_reasoning_effort": a.reasoning_effort, "_reasoning_enabled": a.reasoning_enabled,
                    "_max_tokens": a.max_tokens, "_bcb_version": "v0.1.4",
                    "logprobs": out.get("logprobs")}

        chunk = 50
        for i in range(0, len(todo), chunk):
            results += await asyncio.gather(*[one(t) for t in todo[i:i + chunk]])
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            n_ok = sum(bool(r.get("resolved")) for r in results)
            logger.info("%s: %d/%d collected, pass@1 %.1f%%", a.route_label, len(results),
                        len(tasks), 100 * n_ok / max(1, len(results)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keep-file", required=True, help="bcb_keep.json from validate_bcb.py")
    p.add_argument("--tasks-file", default="", help=(
        "materialised tasks jsonl. Compute nodes cannot reach the HF Hub -- load_dataset dies with "
        "DatasetNotFoundError there while working fine on the login node -- so the tasks are "
        "staged on /mnt/llmd and read from disk. Falls back to load_dataset when empty."))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--route-label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", required=True)
    p.add_argument("--base-url", default="https://openrouter.ai/api")
    p.add_argument("--splits", default="train,eval")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--eval-frac", type=float, default=0.382, help="matches LCB's 341/892")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--output-suffix", default="")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=110000)
    p.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"])
    p.add_argument("--reasoning-enabled", action="store_true")
    p.add_argument("--require-parameters", action="store_true")
    p.add_argument("--ignore-providers", default="Parasail,AkashML")
    p.add_argument("--logprobs", action="store_true", help=(
        "Record answer-token logprobs (top-5) for confidence-based selection. Restricts routing "
        "to endpoints that return them, so these draws are a separate pool, not a top-up."))
    p.add_argument("--problem-ids-file", default="", help=(
        "JSON list of task_ids; collect only these (applied after the train/eval split)."))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grade-concurrency", type=int, default=4)
    p.add_argument("--grade-timeout", type=int, default=60)
    p.add_argument("--gen-timeout", type=int, default=3600)
    p.add_argument("--title", default="PipelineRL-BCB-routing")
    a = p.parse_args()

    keep = set(json.loads(Path(a.keep_file).read_text()))
    if a.tasks_file:
        tasks = [json.loads(l) for l in open(a.tasks_file) if l.strip()]
        tasks = [t for t in tasks if t["task_id"] in keep]
    else:
        from datasets import load_dataset
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
        tasks = [dict(r) for r in ds if r["task_id"] in keep]
    rng = random.Random(a.split_seed)
    order = sorted(tasks, key=lambda t: t["task_id"])
    rng.shuffle(order)
    n_eval = int(round(len(order) * a.eval_frac))
    split_of = {"eval": order[:n_eval], "train": order[n_eval:]}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    out.joinpath("bcb_split_manifest.json").write_text(json.dumps(
        {k: [t["task_id"] for t in v] for k, v in split_of.items()}, indent=1))
    for split in [s for s in a.splits.split(",") if s.strip()]:
        rows = split_of[split]
        if a.problem_ids_file:
            wanted = set(json.loads(Path(a.problem_ids_file).read_text()))
            rows = [t for t in rows if t["task_id"] in wanted]
        if a.max_problems:
            rows = rows[: a.max_problems]
        logger.info("%s split: %d tasks", split, len(rows))
        asyncio.run(collect(rows, out / f"{a.route_label}_{split}{a.output_suffix}.jsonl", a))


if __name__ == "__main__":
    main()
