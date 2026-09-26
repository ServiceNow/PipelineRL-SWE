#!/usr/bin/env python3
"""Condition 3 (NEW_PATH 3.2): what does ONE verification check cost on SWE, in sandbox time?

For a few instances: create the Daytona sandbox (SWE-bench eval image), read its resources, run a
writer's reproduction script on the unpatched repo, apply a pool patch, run the script again. Every
step is wall-clock timed. Cost = seconds x (vCPU x $/vCPU-s + GiB x $/GiB-s) at Daytona list prices.
"""
from __future__ import annotations
import argparse, asyncio, json, os, time
from pathlib import Path
from daytona import AsyncDaytona, CreateSandboxFromImageParams
from pipelinerl.swe.scripts.offline_router.swe_testwriter_execute import ACTIVATE, APPLY, image, sh


async def one(daytona, sem, iid, script, patch, timeout):
    t = {"instance_id": iid}
    async with sem:
        sb = None
        try:
            t0 = time.monotonic()
            sb = await daytona.create(CreateSandboxFromImageParams(image=image(iid), auto_delete_interval=15), timeout=300)
            t["create_s"] = time.monotonic() - t0
            _, res = await sh(sb, "nproc; free -m | awk '/Mem/{print $2}'", 30)
            t["nproc"], t["mem_mb"] = [int(x) for x in res.split()[:2]]
            await sb.fs.upload_file(script.encode(), "/tmp/repro.py")
            for tag in ("run_base_s", "run_base_warm_s"):          # second run: warm caches
                t0 = time.monotonic()
                await sh(sb, f"bash -c '{ACTIVATE} && cd /testbed && timeout {timeout} python /tmp/repro.py' > /dev/null 2>&1; echo $?", timeout + 30)
                t[tag] = time.monotonic() - t0
            await sb.fs.upload_file((patch if patch.endswith("\n") else patch + "\n").encode(), "/tmp/patch.diff")
            t0 = time.monotonic()
            for cmd in APPLY:
                c, _ = await sh(sb, f"cd /testbed && {cmd} /tmp/patch.diff > /dev/null 2>&1", 60)
                if c == 0:
                    break
            t["apply_s"] = time.monotonic() - t0
            t0 = time.monotonic()
            await sh(sb, f"bash -c '{ACTIVATE} && cd /testbed && timeout {timeout} python /tmp/repro.py' > /dev/null 2>&1; echo $?", timeout + 30)
            t["run_patched_s"] = time.monotonic() - t0
        except Exception as e:
            t["error"] = f"{type(e).__name__}: {e}"[:200]
        finally:
            if sb is not None:
                t0 = time.monotonic()
                try:
                    await sb.delete()
                except Exception:
                    pass
                t["delete_s"] = time.monotonic() - t0
    return t


async def main_async(a):
    ids = json.loads(Path(a.instances_file).read_text())[: a.n]
    scripts = {json.loads(l)["instance_id"]: json.loads(l)["script"] for l in open(a.scripts_file)}
    runs = json.loads(Path(a.patch_runs).read_text())
    R = Path(a.results_root)
    pool = {json.loads(l)["instance_id"]: json.loads(l).get("model_patch", "")
            for l in open(R / f"opus_verified_daytona_eval_{runs[a.patch_route]}/predictions/predictions_opus_verified.jsonl")}
    sem = asyncio.Semaphore(a.concurrency)
    async with AsyncDaytona() as d:
        rows = await asyncio.gather(*[one(d, sem, i, scripts[i], pool.get(i, ""), a.timeout) for i in ids if scripts.get(i)])
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--scripts-file", required=True)
    ap.add_argument("--patch-runs", required=True)
    ap.add_argument("--patch-route", default="oss120")
    ap.add_argument("--results-root", default="/mnt/llmd/results/exps/aristides/reason")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    if not os.environ.get("DAYTONA_API_KEY"):
        for env in ("/home/toolkit/PipelineRL-SWE/.env", "/home/toolkit/.env"):
            if Path(env).exists():
                for line in open(env):
                    if line.startswith("DAYTONA_API_KEY="):
                        os.environ["DAYTONA_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
