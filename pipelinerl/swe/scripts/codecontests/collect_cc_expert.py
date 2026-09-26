#!/usr/bin/env python3
"""Collect one draw per CodeContests problem for one route (model + settings), graded locally.

Output rows match the BigCodeBench collector, so `bigcodebench/build_bcb_tensors.py` builds the
tensor bundle unchanged: <out>/<route>_<train|eval>_d<k>.jsonl plus bcb_split_manifest.json.
Serving hygiene is inherited from `openrouter_call` (tool_calls leak providers ignored, empty-answer
retries, reasoning channel fallback).
"""
from __future__ import annotations
import argparse, asyncio, json, logging, random
from pathlib import Path
import aiohttp
from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import openrouter_call, extract_code
from pipelinerl.swe.scripts.codecontests.cc_grader import grade

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SYSTEM = ("You are an expert competitive programmer. Solve the problem and write a complete, correct "
          "Python 3 solution that reads from standard input and writes to standard output. "
          "Output only Python code with no explanation.")


async def collect(tasks, out_path: Path, a):
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            if r.get("finish_reason") != "error":
                done[r["problem_id"]] = r
    todo = [t for t in tasks if t["problem_id"] not in done]
    log.info("%s: %d/%d reusable, %d to collect", out_path.name, len(done), len(tasks), len(todo))
    if not todo:
        return
    key = Path(a.api_key_file).read_text().strip()
    sem = asyncio.Semaphore(a.concurrency); gsem = asyncio.Semaphore(a.grade_concurrency)
    results = list(done.values())
    async with aiohttp.ClientSession() as session:
        async def one(t):
            try:
                out = await openrouter_call(
                    session, a.model, SYSTEM, t["prompt"], key, max_tokens=a.max_tokens,
                    temperature=a.temperature, top_p=a.top_p, title="PipelineRL-CC-routing",
                    semaphore=sem, gen_timeout=a.gen_timeout,
                    reasoning_effort=a.reasoning_effort or None, reasoning_enabled=a.reasoning_enabled,
                    require_parameters=a.require_parameters,
                    ignore_providers=[x for x in a.ignore_providers.split(",") if x.strip()] or None)
            except Exception as e:
                return {"problem_id": t["problem_id"], "route_label": a.route_label, "model": a.model,
                        "full_output": "", "code": "", "resolved": False, "finish_reason": "error",
                        "error": f"{type(e).__name__}: {e}"[:300], "prompt_tokens": 0, "completion_tokens": 0}
            code = extract_code(out["full_output"])
            async with gsem:
                g = await asyncio.to_thread(grade, code, [tuple(x) for x in t["tests"]], t["timeout_s"])
            return {"problem_id": t["problem_id"], "route_label": a.route_label, "model": a.model,
                    "full_output": out["full_output"], "code": code,
                    "thinking_chars": len(out.get("thinking_text") or ""),
                    "prompt_tokens": out.get("prompt_tokens", 0), "completion_tokens": out.get("completion_tokens", 0),
                    "finish_reason": out.get("finish_reason"), "provider": out.get("provider"),
                    "resolved": bool(g.ok), "grade_status": g.status, "tests_passed": g.n_passed, "n_tests": g.n_tests,
                    "_temperature": a.temperature, "_top_p": a.top_p, "_reasoning_effort": a.reasoning_effort,
                    "_reasoning_enabled": a.reasoning_enabled, "_max_tokens": a.max_tokens}
        for i in range(0, len(todo), 50):
            results += await asyncio.gather(*[one(t) for t in todo[i:i + 50]])
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            n_ok = sum(bool(r.get("resolved")) for r in results)
            log.info("%s: %d/%d collected, pass@1 %.1f%%", a.route_label, len(results), len(tasks),
                     100 * n_ok / max(1, len(results)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--route-label", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    p.add_argument("--output-suffix", default="")
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--eval-frac", type=float, default=0.382, help="matches LCB's 341/892")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=117964)
    p.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"])
    p.add_argument("--reasoning-enabled", action="store_true")
    p.add_argument("--require-parameters", action="store_true")
    p.add_argument("--ignore-providers", default="Parasail,AkashML")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--grade-concurrency", type=int, default=6)
    p.add_argument("--gen-timeout", type=int, default=3600)
    a = p.parse_args()
    tasks = [json.loads(l) for l in open(a.tasks_file)]
    order = sorted(tasks, key=lambda t: t["problem_id"]); random.Random(a.split_seed).shuffle(order)
    n_eval = int(round(len(order) * a.eval_frac))
    split = {"eval": order[:n_eval], "train": order[n_eval:]}
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "bcb_split_manifest.json").write_text(json.dumps({k: [t["problem_id"] for t in v] for k, v in split.items()}, indent=1))
    for s in ("train", "eval"):
        rows = split[s][: a.max_problems] if a.max_problems else split[s]
        asyncio.run(collect(rows, out / f"{a.route_label}_{s}{a.output_suffix}.jsonl", a))


if __name__ == "__main__":
    main()
