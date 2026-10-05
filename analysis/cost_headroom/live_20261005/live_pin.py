"""Live run (NEW_PATH 4.A.56), pinned variant: re-issue every live deepseek-v4-flash call with the provider PINNED to StreamLake (the
provider chosen on FIT problems in 4.A.49 / provider_pinning_check.py), as the paper recommends. Routing decisions are unchanged (frozen
before any call); only the dsv4f endpoint changes. Same generation settings as the pinned provider pilot (collect_provider_pilot.call).
Output: live_run_20261005/calls_dsv4f_pinned_StreamLake.jsonl (resumable). Usage: python live_pin.py [--budget-usd 2]
"""
import argparse, asyncio, json, sys
from pathlib import Path
import aiohttp
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pipelinerl/swe/scripts/math_pool"))
from collect_provider_pilot import call
from collect_math_pool import _grade_task
OUT = Path("/mnt/llmd/results/exps/aristides/reason/live_run_20261005"); PROV = "StreamLake"


async def main(a):
    D = json.load(open(OUT / "decisions.json")); T = {json.loads(l)["problem_id"]: json.loads(l) for l in open(OUT / "tasks.jsonl")}
    pids = sorted({p for d in D["decisions"].values() for arm in ("ours", "median") for p, s in zip(D["problem_ids"], d[arm]) if s == "dsv4f"})
    path = OUT / f"calls_dsv4f_pinned_{PROV}.jsonl"; done = set(); spent = {"usd": 0.0, "stop": False}
    if path.exists():
        for l in open(path):
            r = json.loads(l); spent["usd"] += float(r.get("usage_cost") or 0)
            if r.get("finish_reason") != "error": done.add(r["problem_id"])
    todo = [p for p in pids if p not in done]; print(f"{len(pids)} dsv4f live problems, {len(todo)} to call pinned to {PROV}", flush=True)
    key = Path("/home/toolkit/.secrets/openrouter_api_key").read_text().strip(); sem = asyncio.Semaphore(64); fh = open(path, "a")
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(pid):
            if spent["stop"]:
                return
            r = await call(session, key, "mmlupro", T[pid], PROV, sem)
            r.update(problem_id=pid, route_label="dsv4f", pinned_provider=PROV,
                     resolved=_grade_task(T[pid], r) if r["finish_reason"] != "error" else False)
            r.pop("reasoning", None); fh.write(json.dumps(r) + "\n"); fh.flush()
            spent["usd"] += float(r.get("usage_cost") or 0)
            if spent["usd"] > a.budget_usd:
                spent["stop"] = True
        await asyncio.gather(*[one(p) for p in todo])
    fh.close(); print(f"spent ${spent['usd']:.3f} (budget stop {spent['stop']})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--budget-usd", type=float, default=2.0); asyncio.run(main(ap.parse_args()))
