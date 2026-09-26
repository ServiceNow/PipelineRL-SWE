#!/usr/bin/env python3
"""Extra patch draws from OPEN models on SWE-bench Verified (Track B P3: redraws for the sequential setting).

Reuses the Verified 5-route collection exactly: the stored prompt (system + user, with the relevant
files embedded), the same sampling temperature, and `route_outputs_to_patches.to_patch` to turn the
model's SEARCH/REPLACE answer into an applicable diff -- the pipeline that produced the existing labels.
Writes <out>/predictions_<label>_d<k>.jsonl ({instance_id, model_patch, model, tokens}) for
`run_swebench_eval_daytona.py`. Draw 0 is the existing collection; new draws start at --first-draw.
"""
from __future__ import annotations
import argparse, asyncio, glob, json, re
from pathlib import Path
import aiohttp, pandas as pd
from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import openrouter_call
from pipelinerl.swe.scripts.offline_router.route_outputs_to_patches import files_from_prompt, to_patch

MODELS = {  # label: (OpenRouter id, reasoning effort) -- open models only (user decision 2026-09-26)
    "oss20": ("openai/gpt-oss-20b", "medium"),
    "qwen30": ("qwen/qwen3-coder-30b-a3b-instruct", None),
    "oss120": ("openai/gpt-oss-120b", "medium"),
}
COLL = ("/mnt/llmd/results/exps/aristides/reason/offline_router_swe_bench_train_all_16k_verified_eval_collect_"
        "5route_4b_scout_oss20_qwen30_oss120_gemini/collect/eval")


def split_chat(prompt_text: str) -> tuple[str, str]:
    seg = dict(re.findall(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", prompt_text, re.S))
    return seg.get("system", ""), seg.get("user", prompt_text)


async def run(a):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{COLL}/*.parquet"))]).drop_duplicates("problem_id")
    ids = set(json.loads(Path(a.instances_file).read_text()))
    df = df[df.problem_id.isin(ids)]
    key = Path(a.api_key_file).read_text().strip()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as session:
        for label in a.models.split(","):
            model, effort = MODELS[label]
            for k in range(a.first_draw, a.first_draw + a.draws):
                path = out / f"predictions_{label}_d{k}.jsonl"
                done = {json.loads(l)["instance_id"] for l in open(path)} if path.exists() else set()
                rows = [r for _, r in df.iterrows() if r.problem_id not in done]

                async def one(r):
                    system, user = split_chat(r.prompt_text)
                    try:
                        o = await openrouter_call(session, model, system, user, key, max_tokens=a.max_tokens,
                                                  temperature=a.temperature, semaphore=sem, gen_timeout=1800,
                                                  reasoning_effort=effort, title="PipelineRL-SWE-redraw")
                        patch, why = to_patch(o["full_output"], files_from_prompt(r.prompt_text))
                        return {"instance_id": r.problem_id, "model_patch": patch, "model": f"{label}_d{k}",
                                "why_empty": why, "prompt_tokens": o.get("prompt_tokens", 0),
                                "completion_tokens": o.get("completion_tokens", 0), "provider": o.get("provider")}
                    except Exception as e:
                        return {"instance_id": r.problem_id, "model_patch": "", "model": f"{label}_d{k}",
                                "why_empty": f"error {type(e).__name__}", "prompt_tokens": 0, "completion_tokens": 0}

                res = await asyncio.gather(*[one(r) for r in rows])
                with open(path, "a") as f:
                    for x in res:
                        f.write(json.dumps(x) + "\n")
                n = sum(bool(x["model_patch"]) for x in res)
                print(f"{label} d{k}: {n}/{len(res)} produced a patch", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="oss20,qwen30")
    ap.add_argument("--first-draw", type=int, default=1)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7, help="the 5-route collection's setting")
    ap.add_argument("--max-tokens", type=int, default=15000, help="the 5-route collection's setting")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
