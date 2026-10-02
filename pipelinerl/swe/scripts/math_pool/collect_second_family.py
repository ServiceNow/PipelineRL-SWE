#!/usr/bin/env python3
"""Second model family (NEW_PATH 4.A.48): new routes on the ORIGINAL train+calibration problems (to fit their readouts) and the
fresh evaluation problems already used for the provider pilot (MMLU-Pro: those 2,000; Omni-MATH: all 1,000 fresh). One draw per
route; same prompts and graders as collect_math_pool.py / collect_expansion.py; rows record provider, generation_id and BILLED
usage_cost; one shared spend guard. Output: <out>/<dataset>/<route>_d0.jsonl (appended; resumable).
Routes (OpenRouter ids checked 2026-10-02):
  qw32     qwen/qwen3-32b, reasoning enabled           (Qwen3 family)
  glm47f   z-ai/glm-4.7-flash, reasoning enabled       (GLM family; settings as collect_math_pool.ROUTES)
Usage: python collect_second_family.py --out DIR --routes qw32,glm47f --budget-usd 25 [--pilot 20]
"""
from __future__ import annotations
import argparse, asyncio, json, sys
from pathlib import Path
import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
import collect_math_pool as cmp  # noqa: E402

cmp.ROUTES.setdefault("qw32", ("qwen/qwen3-32b", {"reasoning": {"enabled": True}}, 0.6, 0.95))
REPO = Path(__file__).resolve().parents[4]
R = Path("/mnt/llmd/results/exps/aristides/reason")


def tasks_for(ds):
    pool = {"mmlupro": "mmlupro_tensors", "omni500": "omni500_tensors"}[ds]
    sp = json.loads((R / pool / "split_manifest.json").read_text())
    keep = set(map(str, sp["train_problem_ids"])) | set(map(str, sp["calibration_problem_ids"]))
    orig = [t for t in cmp.load(ds) if t["problem_id"] in keep]
    if ds == "mmlupro":
        ids = set(json.loads((R / "provider_pilot_20261002" / "mmlupro" / "problem_ids.json").read_text()))
        fresh = [json.loads(l) for l in (REPO / "analysis/cost_headroom/expansion_20261001/mmlupro_tasks.jsonl").read_text().splitlines()]
        fresh = [t for t in fresh if t["problem_id"] in ids]
    else:
        fresh = [json.loads(l) for l in (REPO / "analysis/cost_headroom/expansion_20261001/omni500_tasks.jsonl").read_text().splitlines()]
    return orig + fresh


async def run(a):
    key = Path(a.api_key_file).read_text().strip(); spent = {"usd": 0.0, "stop": False}; jobs = []
    for ds in a.datasets.split(","):
        tasks = tasks_for(ds)
        if a.pilot:
            tasks = tasks[:a.pilot // 2] + tasks[-(a.pilot - a.pilot // 2):]
        od = Path(a.out) / ds; od.mkdir(parents=True, exist_ok=True)
        for route in a.routes.split(","):
            path = od / f"{route}_d0.jsonl"; done = set()
            if path.exists():
                for l in path.read_text().splitlines():
                    try: r = json.loads(l)
                    except json.JSONDecodeError: continue
                    spent["usd"] += float(r.get("usage_cost") or 0)
                    if r.get("finish_reason") != "error": done.add(r["problem_id"])
            jobs.append((ds, route, path, [t for t in tasks if t["problem_id"] not in done]))
    print(json.dumps({"event": "start", "already_spent": spent["usd"], "todo": {f"{d}/{r}": len(t) for d, r, _, t in jobs}}), flush=True)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one_job(ds, route, path, todo):
            sem = asyncio.Semaphore(a.concurrency); fh = open(path, "a"); n = {"ok": 0, "err": 0}

            async def one(t):
                if spent["stop"]:
                    return
                r = await cmp.call(session, key, route, t.get("prompt") or cmp.PROMPT.format(problem=t["problem"]), sem, a.max_tokens)
                r.update(problem_id=t["problem_id"], dataset=ds, route_label=route, model=cmp.ROUTES[route][0], draw=0,
                         difficulty=t.get("difficulty"), resolved=cmp._grade_task(t, r) if r["finish_reason"] != "error" else False)
                spent["usd"] += float(r.get("usage_cost") or 0); spent["stop"] = spent["usd"] > a.budget_usd
                n["err" if r["finish_reason"] == "error" else "ok"] += 1
                fh.write(json.dumps(r) + "\n"); fh.flush()
            await asyncio.gather(*[one(t) for t in todo]); fh.close()
            print(json.dumps({"event": "done", "dataset": ds, "route": route, **n, "spent_total": round(spent["usd"], 4)}), flush=True)
        await asyncio.gather(*[one_job(*j) for j in jobs])
    print(json.dumps({"event": "finished", "spent_usd": spent["usd"], "budget_stop": spent["stop"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--routes", default="qw32,glm47f"); ap.add_argument("--datasets", default="mmlupro,omni500")
    ap.add_argument("--budget-usd", type=float, default=25.0); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64000); ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
