#!/usr/bin/env python3
"""SWE test-writer pilot, step 2: run every writer's reproduction script against every patch, in Daytona.

One sandbox per instance (the SWE-bench eval image = the buggy base commit). Inside it:
  1. each writer's script on the UNPATCHED repo  -> must exit non-zero to be a valid reproduction
  2. for each candidate patch (6 pool routes) and the GOLD patch: reset the repo, apply the patch,
     run every script -> exit 0 = "this script says the patch fixes the issue"
A valid script should also pass on gold; that is recorded, never used for selection.

Daytona's org cap is ~10 concurrent sandboxes ACROSS ALL JOBS (a blown cap silently scores
everything 0), so keep --concurrency <= 8 and run one such job at a time.
Output: <out>/exec.jsonl, one row per instance: {instance_id, base: {w: code}, patches: {p: {w: code}},
apply: {p: ok}}. Exit code 124 = timeout, None = script missing.
"""
from __future__ import annotations
import argparse, asyncio, json, os
from pathlib import Path
from daytona import AsyncDaytona, CreateSandboxFromImageParams

ACTIVATE = ("source /opt/miniconda3/bin/activate testbed 2>/dev/null || "
            "source /opt/miniconda3/bin/activate && conda activate testbed 2>/dev/null || true")
APPLY = ["git apply --verbose", "git apply --verbose --reject", "patch --batch --fuzz=5 -p1 -i"]


def image(iid: str) -> str:
    return f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid.lower()}:latest"


async def sh(sb, cmd, timeout):
    r = await sb.process.exec(cmd, timeout=timeout)
    return r.exit_code, (r.result or "")


async def reset(sb):
    # tracked files back to base; delete only untracked paths the patch created (not build artefacts
    # that existed at base, which a blanket `git clean` could remove)
    await sh(sb, "cd /testbed && git checkout -- . && "
                 "comm -13 /tmp/base_untracked.txt <(git ls-files --others --exclude-standard | sort) "
                 "| xargs -r rm -rf", 60)


async def run_scripts(sb, writers, timeout):
    codes = {}
    for w in writers:
        # the command's own exit status is echo's (always 0); the script's status is what echo PRINTS
        _, printed = await sh(sb, f"bash -c '{ACTIVATE} && cd /testbed && timeout {timeout} python /tmp/repro_{w}.py' "
                                  f"> /tmp/out_{w}.txt 2>&1; echo __RC__$?", timeout + 30)
        rc = printed.rsplit("__RC__", 1)[-1].strip().split()[0] if "__RC__" in printed else ""
        codes[w] = int(rc) if rc.lstrip("-").isdigit() else None
    return codes


async def one(daytona, sem, iid, scripts, patches, timeout):
    writers = [w for w, s in scripts.items() if s]
    row = {"instance_id": iid, "base": {}, "patches": {}, "apply": {}, "missing": [w for w, s in scripts.items() if not s]}
    async with sem:
        sb = None
        try:
            sb = await daytona.create(CreateSandboxFromImageParams(image=image(iid), auto_delete_interval=int(os.environ.get("DAYTONA_TTL", "15"))), timeout=180)
            for w in writers:
                await sb.fs.upload_file(scripts[w].encode(), f"/tmp/repro_{w}.py")
            await sh(sb, "cd /testbed && git ls-files --others --exclude-standard | sort > /tmp/base_untracked.txt", 60)
            row["base"] = await run_scripts(sb, writers, timeout)
            for p, diff in patches.items():
                await reset(sb)
                if not diff.strip():
                    row["apply"][p] = False; continue
                await sb.fs.upload_file((diff if diff.endswith("\n") else diff + "\n").encode(), "/tmp/patch.diff")
                ok = False
                for cmd in APPLY:
                    c, _ = await sh(sb, f"cd /testbed && {cmd} /tmp/patch.diff > /dev/null 2>&1", 60)
                    if c == 0:
                        ok = True; break
                row["apply"][p] = ok
                if ok:
                    row["patches"][p] = await run_scripts(sb, writers, timeout)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if sb is not None:
                try:
                    await sb.delete()
                except Exception:
                    pass
    return row


async def main_async(a):
    from datasets import load_dataset
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    ids = json.loads(Path(a.instances_file).read_text())
    runs = json.loads(Path(a.patch_runs).read_text())
    R = Path(a.results_root)
    pool = {k: {json.loads(l)["instance_id"]: json.loads(l).get("model_patch", "")
                for l in open(R / f"opus_verified_daytona_eval_{r}/predictions/predictions_opus_verified.jsonl")}
            for k, r in runs.items()}
    scripts = {}
    for f in Path(a.scripts_dir).glob("scripts_*.jsonl"):
        for l in open(f):
            r = json.loads(l); scripts.setdefault(r["instance_id"], {})[r["writer"]] = r["script"]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["instance_id"] for l in open(out) if not json.loads(l).get("error")} if out.exists() else set()
    todo = [i for i in ids if i not in done and i in scripts]
    print(f"{len(todo)} instances to execute ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with AsyncDaytona() as daytona:
        tasks = [one(daytona, sem, i, scripts[i], {**{k: pool[k].get(i, "") for k in runs}, "gold": ds[i]["patch"]}, a.timeout)
                 for i in todo]
        with open(out, "a") as f:
            for n, t in enumerate(asyncio.as_completed(tasks), 1):
                r = await t
                f.write(json.dumps(r) + "\n"); f.flush()
                print(f"[{n}/{len(todo)}] {r['instance_id']} base={r['base']} err={r.get('error','')[:80]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--scripts-dir", required=True)
    ap.add_argument("--patch-runs", required=True, help="JSON {route_label: opus_verified_daytona_eval run id}")
    ap.add_argument("--results-root", default="/mnt/llmd/results/exps/aristides/reason")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
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
